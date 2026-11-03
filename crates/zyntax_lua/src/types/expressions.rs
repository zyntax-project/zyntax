//! Expressions support.

use super::*;

impl<'a> Typer<'a> {
    pub(super) fn name_ty(&self, token: &full_moon::tokenizer::TokenReference) -> Ty {
        match self.scopes.binding(token) {
            Some(b) => self.binding_ty(b),
            None => Ty::Any,
        }
    }

    pub fn binding_ty(&self, binding: &Binding) -> Ty {
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => self.known.var(*v),
            Binding::Global(name) => self.global_ty(name),
            // A free name in the scope of a local `_ENV`: its field.
            Binding::Field(env, name, _) => self.field_ty(self.known.var(*env), name),
        }
    }

    pub fn global_ty(&self, name: &str) -> Ty {
        // An entry of the globals table can be anything.
        if self.scopes.dynamic_globals {
            return Ty::Any;
        }
        if builtin_named(self.scopes, name).is_some() {
            return Ty::Any;
        }
        if !self.scopes.global_writes.contains_key(name) {
            return match name {
                "_VERSION" => Ty::Str,
                _ if preset_global(name) => Ty::Any,
                _ => Ty::Nil,
            };
        }
        let assigned = self.known.global(name);
        if self.scopes.globals_initialized.contains(name) {
            assigned
        } else if preset_global(name) {
            // Read before the program assigns it: the library's value.
            Ty::Any
        } else {
            assigned.join(Ty::Nil)
        }
    }

    /// The function a callee expression is, when its type says.
    pub fn known_callee(&self, prefix: &Prefix) -> Option<FuncId> {
        match self.prefix_ty(prefix) {
            Ty::Func(f) => Some(f),
            _ => None,
        }
    }

    /// The builtin a callee names: `print`, or `string.format` where
    /// `string` is the library's table.
    pub fn builtin_callee(
        &self,
        prefix: &Prefix,
        suffixes: &[&Suffix],
    ) -> Option<&'static Builtin> {
        let Prefix::Name(token) = prefix else {
            return None;
        };
        let Some(Binding::Global(name)) = self.scopes.binding(token) else {
            return None;
        };
        match suffixes {
            [Suffix::Call(ast::Call::AnonymousCall(_))] => builtin_named(self.scopes, name),
            [
                Suffix::Index(ast::Index::Dot { name: member, .. }),
                Suffix::Call(ast::Call::AnonymousCall(_)),
            ] => builtin_member(self.scopes, name, &ident(member)),
            _ => None,
        }
    }

    /// Whether `e` is never nil, by its form or by its type.
    pub fn non_nil(&self, e: &Expression) -> bool {
        non_nil(self.scopes, &self.known.non_nil_vars, e) || {
            let ty = self.ty_of(e);
            ty != Ty::Unknown && !ty.may_be_nil()
        }
    }

    /// The captures a generic `for` over `string.gmatch(s, p)` binds,
    /// each whether it is a position, when `s` is a string and `p` a
    /// literal pattern whose captures are known: such a loop walks the
    /// matches in place. A pattern without captures binds the match.
    pub fn gmatch_walk(&self, exprs: &[&Expression]) -> Option<Vec<bool>> {
        let [Expression::FunctionCall(c)] = exprs else {
            return None;
        };
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        let b = self.builtin_callee(c.prefix(), &suffixes)?;
        if b.lib != "string" || b.name != "gmatch" {
            return None;
        }
        let Some(Suffix::Call(ast::Call::AnonymousCall(ast::FunctionArgs::Parentheses {
            arguments,
            ..
        }))) = suffixes.last()
        else {
            return None;
        };
        let args: Vec<&Expression> = arguments.iter().collect();
        let [subject, Expression::String(pattern)] = args.as_slice() else {
            return None;
        };
        if self.ty_of(subject) != Ty::Str {
            return None;
        }
        let bytes = crate::lower::string_bytes(pattern).ok()?;
        let kinds = crate::pattern::capture_kinds(&bytes)?;
        Some(if kinds.is_empty() { vec![false] } else { kinds })
    }

    /// What a call returns. `suffixes` are the callee's suffixes,
    /// ending in the call being typed.
    pub fn call_returns(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Returns {
        let Some(Suffix::Call(last)) = suffixes.last() else {
            return Returns::Dynamic;
        };
        if let Some(b) = self.builtin_callee(prefix, suffixes) {
            if let ast::Call::AnonymousCall(args) = last
                && let Some(t) = self.math_call_ty(b, args)
            {
                return Returns::Fixed(vec![t]);
            }
            if let ast::Call::AnonymousCall(args) = last
                && let Some(r) = self.metatable_call_returns(b, args)
            {
                return r;
            }
            if b.lib.is_empty()
                && b.name == "rawget"
                && let ast::Call::AnonymousCall(ast::FunctionArgs::Parentheses {
                    arguments, ..
                }) = last
                && let Some(t) = self.rawget_ty(&arguments.iter().collect::<Vec<_>>())
            {
                return Returns::Fixed(vec![t]);
            }
            return match b.ret {
                Ret::Multi => Returns::Dynamic,
                r => Returns::Fixed(vec![ret_ty(r)]),
            };
        }
        // The receiver of the last call is the prefix through every
        // suffix before it.
        let receiver = self.suffixed_ty(prefix, &suffixes[..suffixes.len() - 1]);
        match last {
            ast::Call::MethodCall(m) => self.method_returns(receiver, &ident(m.name())),
            ast::Call::AnonymousCall(_) => match (receiver, self.field_callee(prefix, suffixes)) {
                (Ty::Func(_) | Ty::Unknown, _) | (_, None) => self.callee_returns(receiver),
                (_, Some((k, name))) => self.method_returns(Ty::Shape(k), &name),
            },
            _ => Returns::Dynamic,
        }
    }

    /// For a call `obj.name(...)`, the last of `suffixes`, on a shaped
    /// receiver: the receiver's shape and the name. Its callee is one
    /// of the functions `method_targets` gives, the receiver not
    /// passed.
    pub fn field_callee(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Option<(ShapeId, String)> {
        let n = suffixes.len();
        if n < 2 || !matches!(suffixes[n - 1], Suffix::Call(ast::Call::AnonymousCall(_))) {
            return None;
        }
        let Suffix::Index(index) = suffixes[n - 2] else {
            return None;
        };
        let name = constant_key(index)?;
        if self.global_member(prefix, &suffixes[..n - 1]).is_some() {
            return None;
        }
        match self.suffixed_ty(prefix, &suffixes[..n - 2]) {
            Ty::Shape(k) if (k.0 as usize) < self.known.shapes.len() => Some((k, name)),
            _ => None,
        }
    }

    /// What a call through a value of this type returns.
    pub(super) fn callee_returns(&self, callee: Ty) -> Returns {
        match callee {
            // A function no round has settled yet returns nothing
            // known: the value stays undecided until it has.
            Ty::Func(f) => self
                .known
                .sig(f)
                .map(|s| s.returns.clone())
                .unwrap_or(Returns::Unknown),
            Ty::Unknown => Returns::Unknown,
            _ => Returns::Dynamic,
        }
    }

    /// What `receiver:name(...)` returns: the joined results of the
    /// functions the receiver's classes resolve it to, when every one
    /// does; a string method's declared result.
    pub fn method_returns(&self, receiver: Ty, name: &str) -> Returns {
        match receiver {
            Ty::Str => match builtin_member(self.scopes, "string", name) {
                Some(b) => match b.ret {
                    Ret::Multi => Returns::Dynamic,
                    r => Returns::Fixed(vec![ret_ty(r)]),
                },
                None => Returns::Dynamic,
            },
            Ty::Shape(k) => match self.method_targets(k, name) {
                Some(targets) => {
                    let mut out = Returns::Unknown;
                    for target in targets {
                        let r = self.callee_returns(target);
                        out = match (out, r) {
                            (Returns::Unknown, r) | (r, Returns::Unknown) => r,
                            (Returns::Fixed(x), Returns::Fixed(y)) if x.len() == y.len() => {
                                Returns::Fixed(
                                    x.into_iter().zip(y).map(|(a, b)| a.join(b)).collect(),
                                )
                            }
                            _ => Returns::Dynamic,
                        };
                    }
                    out
                }
                None => Returns::Dynamic,
            },
            Ty::Unknown => Returns::Unknown,
            _ => Returns::Dynamic,
        }
    }

    /// The functions `receiver:name` may be for a receiver of shape
    /// `k`: one per class the shape's tables get, when the field is
    /// not one of the shape's own, every class resolves it, and no
    /// other metatable can reach the tables. Each is a function or,
    /// not settled yet, unknown.
    pub fn method_targets(&self, k: ShapeId, name: &str) -> Option<Vec<Ty>> {
        let info = self.known.shapes.get(k.0 as usize)?;
        let mut out = Vec::new();
        // The table's own field first; one that is always there is
        // all there is.
        match info.field(name) {
            Some((_, ty @ (Ty::Func(_) | Ty::Unknown))) => {
                out.push(ty);
                if info.always_present(name) {
                    return Some(out);
                }
            }
            Some(_) => return None,
            None if self.known.may_hold_unnamed(k, name) => return None,
            None => {}
        }
        if info.unknown_meta {
            return None;
        }
        if info.classes.is_empty() {
            return if out.is_empty() { None } else { Some(out) };
        }
        for class in &info.classes {
            out.extend(self.known.resolve_method(*class, name)?);
        }
        out.sort_by_key(|t| match t {
            Ty::Func(f) => f.0 as i64,
            _ => -1,
        });
        out.dedup();
        Some(out)
    }

    /// What `rawget(t, k)` finds, for these arguments, when `t` is a
    /// shaped table and the types know what it holds under `k`.
    pub fn rawget_ty(&self, args: &[&Expression]) -> Option<Ty> {
        let [t, key, ..] = args else {
            return None;
        };
        let Ty::Shape(k) = self.ty_of(t) else {
            return None;
        };
        let info = self.known.shapes.get(k.0 as usize)?;
        let ty = match crate::scope::literal_string(key) {
            Some(name) => match info.field(&name) {
                Some((_, ty)) if info.always_present(&name) => ty,
                Some((_, ty)) => join_read(ty, Ty::Nil),
                None if info.unknown_meta => Ty::Any,
                None => self.known.unnamed_ty(k, &name),
            },
            None => self.known.raw_keyed_ty(k, self.ty_of(key)),
        };
        (ty != Ty::Any).then_some(ty)
    }

    /// `setmetatable(t, m)` returns `t`; `getmetatable(t)` is anything.
    pub(super) fn metatable_call_returns(
        &self,
        b: &Builtin,
        args: &ast::FunctionArgs,
    ) -> Option<Returns> {
        if !b.lib.is_empty() || b.name != "setmetatable" {
            return None;
        }
        let ast::FunctionArgs::Parentheses { arguments, .. } = args else {
            return None;
        };
        let first = arguments.iter().next()?;
        match self.ty_of(first) {
            t @ Ty::Shape(_) => Some(Returns::Fixed(vec![t])),
            Ty::Unknown => Some(Returns::Unknown),
            _ => None,
        }
    }

    /// What `receiver.name` yields.
    pub fn field_ty(&self, receiver: Ty, name: &str) -> Ty {
        match receiver {
            Ty::Shape(k) => match self.known.shapes.get(k.0 as usize) {
                Some(_) => self.known.field_read_ty(k, name),
                None => Ty::Unknown,
            },
            Ty::Unknown => Ty::Unknown,
            _ => Ty::Any,
        }
    }

    /// What a `math` call's arguments make its result, when they do.
    /// The arguments are each expression's one value, the last one's
    /// several; a last argument of no fixed count decides nothing.
    pub(super) fn math_call_ty(&self, b: &Builtin, args: &ast::FunctionArgs) -> Option<Ty> {
        if b.lib != "math" {
            return None;
        }
        let ast::FunctionArgs::Parentheses { arguments, .. } = args else {
            return None;
        };
        let exprs: Vec<&Expression> = arguments.iter().collect();
        let mut tys = Vec::with_capacity(exprs.len());
        for (i, e) in exprs.iter().enumerate() {
            if i + 1 == exprs.len() {
                match multi_returns(self, e) {
                    Some(Returns::Fixed(more)) => tys.extend(more),
                    Some(Returns::Dynamic) => return None,
                    Some(Returns::Unknown) => tys.push(Ty::Unknown),
                    None => tys.push(self.ty_of(e)),
                }
            } else {
                tys.push(self.ty_of(e));
            }
        }
        math_result(b, &tys)
    }

    pub(super) fn prefix_ty(&self, p: &Prefix) -> Ty {
        match p {
            Prefix::Name(token) => self.name_ty(token),
            Prefix::Expression(e) => self.ty_of(e),
            _ => Ty::Any,
        }
    }

    /// The global `_G.name` names, when `_G` is the library's table.
    pub fn global_member(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Option<String> {
        let Prefix::Name(token) = prefix else {
            return None;
        };
        let Some(Binding::Global(g)) = self.scopes.binding(token) else {
            return None;
        };
        if !Scopes::is_globals_name(g) || self.scopes.global_writes.contains_key(g) {
            return None;
        }
        // Once `_ENV` is assigned, `_G` is a global like any other,
        // found in whatever the environment is.
        if g == "_G" && self.scopes.global_writes.contains_key("_ENV") {
            return None;
        }
        match suffixes {
            [Suffix::Index(ast::Index::Dot { name, .. })] => Some(ident(name)),
            [Suffix::Index(ast::Index::Brackets { expression, .. })] => {
                crate::scope::literal_string(expression)
            }
            _ => None,
        }
    }

    /// The type of a prefix followed by some of its suffixes.
    pub fn suffixed_ty(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Ty {
        if suffixes.is_empty() {
            return self.prefix_ty(prefix);
        }
        if let Some(name) = self.global_member(prefix, suffixes) {
            return self.global_ty(&name);
        }
        match suffixes.last().unwrap() {
            Suffix::Call(_) => self.call_returns(prefix, suffixes).first(),
            Suffix::Index(index) => {
                let receiver = self.suffixed_ty(prefix, &suffixes[..suffixes.len() - 1]);
                match constant_key(index) {
                    Some(name) => self.field_ty(receiver, &name),
                    None => match (receiver, index) {
                        (Ty::Shape(k), ast::Index::Brackets { expression, .. }) => {
                            self.known.element_read_ty(k, self.ty_of(expression))
                        }
                        (Ty::Unknown, _) => Ty::Unknown,
                        _ => Ty::Any,
                    },
                }
            }
            _ => Ty::Any,
        }
    }

    /// The type of an expression in single-value position.
    pub fn ty_of(&self, e: &Expression) -> Ty {
        match e {
            Expression::Number(t) => numeral_ty(&t.token().to_string()),
            Expression::String(_) => Ty::Str,
            Expression::Symbol(t) => match t.token().to_string().trim() {
                "true" | "false" => Ty::Bool,
                "nil" => Ty::Nil,
                _ => Ty::Any,
            },
            Expression::Parentheses { expression, .. } => self.ty_of(expression),
            Expression::Function(f) => Ty::Func(self.scopes.function_of(f.body())),
            Expression::TableConstructor(t) => match constructor_shape(self.known, t) {
                Some(k) => Ty::Shape(k),
                None => Ty::Table,
            },
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                self.suffixed_ty(c.prefix(), &suffixes)
            }
            Expression::Var(v) => match v {
                Var::Name(token) => self.name_ty(token),
                Var::Expression(v) => {
                    let suffixes: Vec<&Suffix> = v.suffixes().collect();
                    self.suffixed_ty(v.prefix(), &suffixes)
                }
                _ => Ty::Any,
            },
            Expression::UnaryOperator { unop, expression } => {
                let t = self.ty_of(expression);
                if t == Ty::Unknown && !matches!(unop, UnOp::Not(_)) {
                    return Ty::Unknown;
                }
                match unop {
                    UnOp::Not(_) => Ty::Bool,
                    UnOp::Minus(_) => match t {
                        Ty::Int | Ty::Float | Ty::Number => t,
                        Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil | Ty::Nil | Ty::Bool => {
                            Ty::Number
                        }
                        Ty::Shape(_) => self.metamethod_result("__unm", t, t).unwrap_or(Ty::Any),
                        _ => Ty::Any,
                    },
                    UnOp::Hash(_) => match t {
                        Ty::Str => Ty::Int,
                        Ty::Table | Ty::Shape(_) if !self.scopes.len_meta => Ty::Int,
                        _ => Ty::Any,
                    },
                    UnOp::Tilde(_) => match t {
                        Ty::Int => Ty::Int,
                        _ => Ty::Any,
                    },
                    _ => Ty::Any,
                }
            }
            Expression::BinaryOperator { lhs, binop, rhs } => {
                let a = self.ty_of(lhs);
                let b = self.ty_of(rhs);
                if let Some(event) = arith_event(binop)
                    && let Some(ty) = self.metamethod_result(event, a, b)
                {
                    return ty;
                }
                binary_ty(binop, a, b)
            }
            _ => Ty::Any,
        }
    }

    /// What arithmetic on a table operand yields: the joined first
    /// results of the handlers the types know, the left operand's
    /// taking precedence as the runtime's do. None when the runtime
    /// decides.
    pub fn metamethod_result(&self, event: &str, a: Ty, b: Ty) -> Option<Ty> {
        let sides = self.metamethod_sides(event, a, b)?;
        let mut out: Option<Ty> = None;
        for (_, _, f) in sides {
            let r = self.callee_returns(Ty::Func(f)).first();
            out = Some(match out {
                None => r,
                Some(so_far) => join_read(so_far, r),
            });
        }
        Some(out.unwrap_or(Ty::Unknown))
    }

    /// The handlers arithmetic on `a` and `b` may call, in the order
    /// the runtime tries them: the left operand's classes' when it is a
    /// shaped table, then the right's when the left cannot carry a
    /// handler of its own. Each with the operand's side and class.
    /// None when an operand the types do not follow may answer first,
    /// or when neither side is a shaped table with handlers.
    pub fn metamethod_sides(
        &self,
        event: &str,
        a: Ty,
        b: Ty,
    ) -> Option<Vec<(bool, ShapeId, FuncId)>> {
        let mut out = Vec::new();
        let mut left_settled = false;
        if let Ty::Shape(k) = a {
            match self.known.metamethod_targets(k, event) {
                MetaTargets::Funcs(funcs) => {
                    out.extend(funcs.into_iter().map(|(c, f)| (true, c, f)));
                    left_settled = true;
                }
                MetaTargets::Unsettled => return Some(Vec::new()),
                MetaTargets::None => {}
            }
        }
        if let Ty::Shape(k) = b
            && (left_settled || !may_handle_arith(a))
        {
            match self.known.metamethod_targets(k, event) {
                MetaTargets::Funcs(funcs) => {
                    out.extend(funcs.into_iter().map(|(c, f)| (false, c, f)));
                }
                MetaTargets::Unsettled => return Some(Vec::new()),
                MetaTargets::None => {}
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}
