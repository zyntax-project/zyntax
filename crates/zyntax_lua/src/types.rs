//! Static types for a dynamically typed program.
//!
//! A variable's type is the join of everything assigned to it, over the
//! whole chunk; a function's parameter types are the join of what its
//! direct calls pass; its result is the join of what it returns. The
//! join of two different types is the dynamic value, so a variable
//! holding an integer everywhere is an `i64` and one holding an
//! integer here and a string there is boxed. Rounds repeat until
//! nothing changes: a recursive function's result depends on itself.

use std::collections::HashMap;

use full_moon::ast::{self, BinOp, Block, Expression, Prefix, Stmt, Suffix, UnOp, Var};

use crate::library::stdlib::{BUILTINS, Builtin, Ret};
use crate::scope::{Binding, CHUNK, FuncId, Scopes, VarId};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Ty {
    Nil,
    Bool,
    Int,
    Float,
    /// An integer or a float, whichever it holds: the join of the two,
    /// carried unboxed with a tag.
    Number,
    Str,
    /// A table, held by pointer; never nil.
    Table,
    /// A dynamic value: a boxed `Any`.
    Any,
    /// Nothing known yet: the bottom of the join.
    #[default]
    Unknown,
}

impl Ty {
    pub fn join(self, other: Ty) -> Ty {
        match (self, other) {
            (Ty::Unknown, t) | (t, Ty::Unknown) => t,
            (a, b) if a == b => a,
            (a, b) if a.is_number() && b.is_number() => Ty::Number,
            _ => Ty::Any,
        }
    }

    /// An integer, a float, or one or the other: arithmetic on it
    /// needs no box.
    pub fn is_number(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Number)
    }

    /// Whether a value of this type is always true in a condition.
    pub fn always_truthy(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Number | Ty::Str | Ty::Table)
    }

    /// What is known once inference has settled: a variable nothing
    /// assigned holds nil.
    pub fn settled(self) -> Ty {
        match self {
            Ty::Unknown => Ty::Nil,
            t => t,
        }
    }
}

/// What a function returns: a fixed number of values, each typed, or
/// a dynamic value that may hold several.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Returns {
    Fixed(Vec<Ty>),
    Dynamic,
}

impl Returns {
    /// Two returns of one arity join value by value; of different
    /// arities, the count itself varies, which a caller can observe,
    /// so the result is dynamic.
    fn join(self, other: Returns) -> Returns {
        match (self, other) {
            (Returns::Fixed(a), Returns::Fixed(b)) if a.len() == b.len() => {
                Returns::Fixed(a.into_iter().zip(b).map(|(x, y)| x.join(y)).collect())
            }
            _ => Returns::Dynamic,
        }
    }

    /// The type of the call in single-value position.
    pub fn first(&self) -> Ty {
        match self {
            Returns::Fixed(v) => v.first().copied().unwrap_or(Ty::Nil),
            Returns::Dynamic => Ty::Any,
        }
    }

    pub fn settled(self) -> Returns {
        match self {
            Returns::Fixed(v) => Returns::Fixed(v.into_iter().map(Ty::settled).collect()),
            Returns::Dynamic => Returns::Dynamic,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sig {
    pub params: Vec<Ty>,
    pub returns: Returns,
}

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Inferred {
    pub funcs: HashMap<FuncId, Sig>,
    pub vars: HashMap<VarId, Ty>,
    pub globals: HashMap<String, Ty>,
}

impl Inferred {
    pub fn var(&self, v: VarId) -> Ty {
        self.vars.get(&v).copied().unwrap_or(Ty::Unknown)
    }
    pub fn global(&self, name: &str) -> Ty {
        self.globals.get(name).copied().unwrap_or(Ty::Unknown)
    }
    pub fn sig(&self, f: FuncId) -> Option<&Sig> {
        self.funcs.get(&f)
    }
}

/// The builtin a global name is, when the program never assigns it.
pub fn builtin_named(scopes: &Scopes, name: &str) -> Option<&'static Builtin> {
    if scopes.global_writes.contains_key(name) {
        return None;
    }
    BUILTINS.iter().find(|b| b.lib.is_empty() && b.name == name)
}

/// The library function `lib.name` is, when `lib` is the untouched
/// global of that library.
pub fn builtin_member(scopes: &Scopes, lib: &str, name: &str) -> Option<&'static Builtin> {
    if scopes.global_writes.contains_key(lib) {
        return None;
    }
    BUILTINS.iter().find(|b| b.lib == lib && b.name == name)
}

/// The result of a `math` function whose arguments' types decide it:
/// `math.abs` of an integer is an integer, `math.max` of floats a
/// float. `None` when they do not, and the function's declared result
/// stands; unknown when an argument is not known yet.
pub fn math_result(b: &Builtin, args: &[Ty]) -> Option<Ty> {
    if b.lib != "math" {
        return None;
    }
    let numbers = |args: &[Ty]| -> Option<Ty> {
        // All integers, or all floats.
        let first = *args.first()?;
        if !first.is_number() || args.iter().any(|t| *t != first) {
            return None;
        }
        Some(first)
    };
    let result = match (b.name, args) {
        ("abs" | "max" | "min", args) => numbers(args),
        ("floor" | "ceil", [Ty::Int]) => Some(Ty::Int),
        ("fmod", [Ty::Int, Ty::Int]) => Some(Ty::Int),
        ("fmod", [Ty::Int | Ty::Float, Ty::Int | Ty::Float]) => Some(Ty::Float),
        ("floor" | "ceil" | "fmod", _) => None,
        _ => return None,
    };
    if args.contains(&Ty::Unknown) {
        return Some(Ty::Unknown);
    }
    result
}

pub fn ret_ty(r: Ret) -> Ty {
    match r {
        Ret::Unit => Ty::Nil,
        Ret::Bool => Ty::Bool,
        Ret::Int => Ty::Int,
        Ret::Float => Ty::Float,
        Ret::Str => Ty::Str,
        Ret::Any | Ret::Multi => Ty::Any,
        Ret::Table => Ty::Table,
    }
}

/// What a numeral literal is: an integer, or a float when it has a
/// fraction, an exponent, or does not fit.
pub fn numeral_ty(text: &str) -> Ty {
    match crate::host::parse_numeral(text) {
        crate::host::Numeral::Int(_) => Ty::Int,
        _ => Ty::Float,
    }
}

/// Types expressions against what the last round settled.
pub struct Typer<'a> {
    pub scopes: &'a Scopes,
    pub known: &'a Inferred,
}

impl<'a> Typer<'a> {
    fn name_ty(&self, token: &full_moon::tokenizer::TokenReference) -> Ty {
        match self.scopes.binding(token) {
            Some(Binding::Local(v)) | Some(Binding::Upvalue(v)) => {
                if self.scopes.known_local_function(*v).is_some() {
                    return Ty::Any;
                }
                self.known.var(*v)
            }
            Some(Binding::Global(name)) => self.global_ty(name),
            Some(Binding::Field(..)) | None => Ty::Any,
        }
    }

    pub fn global_ty(&self, name: &str) -> Ty {
        // An entry of the globals table can be anything.
        if self.scopes.dynamic_globals {
            return Ty::Any;
        }
        if self.scopes.known_global_function(name).is_some() {
            return Ty::Any;
        }
        if builtin_named(self.scopes, name).is_some() {
            return Ty::Any;
        }
        if !self.scopes.global_writes.contains_key(name) {
            return Ty::Nil;
        }
        let assigned = self.known.global(name);
        if self.scopes.globals_initialized.contains(name) {
            assigned
        } else {
            assigned.join(Ty::Nil)
        }
    }

    /// The function a callee expression names, if it is a known one.
    pub fn known_callee(&self, prefix: &Prefix) -> Option<FuncId> {
        match prefix {
            Prefix::Name(token) => self.scopes.known_function(self.scopes.binding(token)?),
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

    /// What a call returns. `suffixes` are the callee's suffixes,
    /// ending in the call being typed.
    pub fn call_returns(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Returns {
        let Some(Suffix::Call(last)) = suffixes.last() else {
            return Returns::Dynamic;
        };
        if let ast::Call::MethodCall(m) = last {
            // A string method with a known result.
            if suffixes.len() == 1
                && self.prefix_ty(prefix) == Ty::Str
                && let Some(b) = builtin_member(self.scopes, "string", &ident(m.name()))
            {
                return match b.ret {
                    Ret::Multi => Returns::Dynamic,
                    r => Returns::Fixed(vec![ret_ty(r)]),
                };
            }
            return Returns::Dynamic;
        }
        if suffixes.len() == 1
            && let Some(f) = self.known_callee(prefix)
        {
            // A function no round has settled yet returns nothing
            // known: the value stays undecided until it has.
            return self
                .known
                .sig(f)
                .map(|s| s.returns.clone())
                .unwrap_or(Returns::Fixed(vec![Ty::Unknown]));
        }
        if let Some(b) = self.builtin_callee(prefix, suffixes) {
            if let ast::Call::AnonymousCall(args) = last
                && let Some(t) = self.math_call_ty(b, args)
            {
                return Returns::Fixed(vec![t]);
            }
            return match b.ret {
                Ret::Multi => Returns::Dynamic,
                r => Returns::Fixed(vec![ret_ty(r)]),
            };
        }
        Returns::Dynamic
    }

    /// What a `math` call's arguments make its result, when they do.
    /// The arguments are each expression's one value, the last one's
    /// several; a last argument of no fixed count decides nothing.
    fn math_call_ty(&self, b: &Builtin, args: &ast::FunctionArgs) -> Option<Ty> {
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
                    None => tys.push(self.ty_of(e)),
                }
            } else {
                tys.push(self.ty_of(e));
            }
        }
        math_result(b, &tys)
    }

    fn prefix_ty(&self, p: &Prefix) -> Ty {
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
    fn suffixed_ty(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Ty {
        if suffixes.is_empty() {
            return self.prefix_ty(prefix);
        }
        if let Some(name) = self.global_member(prefix, suffixes) {
            return self.global_ty(&name);
        }
        match suffixes.last().unwrap() {
            Suffix::Call(_) => self.call_returns(prefix, suffixes).first(),
            Suffix::Index(_) => Ty::Any,
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
            Expression::Function(_) => Ty::Any,
            Expression::TableConstructor(_) => Ty::Table,
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
                        _ => Ty::Any,
                    },
                    UnOp::Hash(_) => match t {
                        Ty::Str => Ty::Int,
                        Ty::Table if !self.scopes.len_meta => Ty::Int,
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
                binary_ty(binop, a, b)
            }
            _ => Ty::Any,
        }
    }
}

/// The identifier a name token holds.
pub fn ident(token: &full_moon::tokenizer::TokenReference) -> String {
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::Identifier { identifier } => identifier.to_string(),
        _ => token.token().to_string(),
    }
}

/// The type of `a and b` or `a or b`.
pub fn logical_ty(is_and: bool, a: Ty, b: Ty) -> Ty {
    if a == Ty::Unknown || b == Ty::Unknown {
        return Ty::Unknown;
    }
    if is_and {
        if a.always_truthy() {
            b
        } else if a == Ty::Nil {
            Ty::Nil
        } else if a == Ty::Bool && b == Ty::Bool {
            Ty::Bool
        } else {
            a.join(b)
        }
    } else if a.always_truthy() {
        a
    } else if a == Ty::Nil {
        b
    } else if a == Ty::Bool && b == Ty::Bool {
        Ty::Bool
    } else {
        a.join(b)
    }
}

pub fn binary_ty(op: &BinOp, a: Ty, b: Ty) -> Ty {
    // Nothing known on one side settles nothing yet, except that a
    // comparison is a boolean whatever its operands.
    let comparison = matches!(
        op,
        BinOp::TwoEqual(_)
            | BinOp::TildeEqual(_)
            | BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_)
    );
    if (a == Ty::Unknown || b == Ty::Unknown) && !comparison {
        return Ty::Unknown;
    }
    match op {
        BinOp::Plus(_)
        | BinOp::Minus(_)
        | BinOp::Star(_)
        | BinOp::Percent(_)
        | BinOp::DoubleSlash(_) => {
            // A float operand makes a float; two integers an integer;
            // a number whose kind is not known keeps it open.
            if !(a.is_number() && b.is_number()) {
                Ty::Any
            } else if a == Ty::Int && b == Ty::Int {
                Ty::Int
            } else if a == Ty::Float || b == Ty::Float {
                Ty::Float
            } else {
                Ty::Number
            }
        }
        BinOp::Slash(_) | BinOp::Caret(_) => {
            if a.is_number() && b.is_number() {
                Ty::Float
            } else {
                Ty::Any
            }
        }
        BinOp::Ampersand(_)
        | BinOp::Pipe(_)
        | BinOp::Tilde(_)
        | BinOp::DoubleLessThan(_)
        | BinOp::DoubleGreaterThan(_) => {
            if a.is_number() && b.is_number() {
                Ty::Int
            } else {
                Ty::Any
            }
        }
        BinOp::TwoDots(_) => {
            let text = |t: Ty| t == Ty::Str || t.is_number();
            if text(a) && text(b) { Ty::Str } else { Ty::Any }
        }
        BinOp::TwoEqual(_)
        | BinOp::TildeEqual(_)
        | BinOp::LessThan(_)
        | BinOp::LessThanEqual(_)
        | BinOp::GreaterThan(_)
        | BinOp::GreaterThanEqual(_) => Ty::Bool,
        BinOp::And(_) => logical_ty(true, a, b),
        BinOp::Or(_) => logical_ty(false, a, b),
        _ => Ty::Any,
    }
}

/// Whether the last statement of a block can be run past, so the
/// function returns nothing there.
pub fn falls_through(block: &Block) -> bool {
    match block.last_stmt() {
        Some(ast::LastStmt::Return(_)) => false,
        Some(ast::LastStmt::Break(_)) => false,
        _ => match block.stmts().last() {
            Some(Stmt::If(i)) => {
                let Some(els) = i.else_block() else {
                    return true;
                };
                falls_through(i.block())
                    || i.else_if()
                        .is_some_and(|e| e.iter().any(|e| falls_through(e.block())))
                    || falls_through(els)
            }
            Some(Stmt::Do(d)) => falls_through(d.block()),
            _ => true,
        },
    }
}

/// One round of inference over the chunk, from what the last round
/// settled.
struct Round<'a> {
    scopes: &'a Scopes,
    known: &'a Inferred,
    out: Inferred,
    /// The function being walked.
    func: FuncId,
    /// What the function being walked returns, joined so far.
    returns: Option<Returns>,
}

impl<'a> Round<'a> {
    fn typer(&self) -> Typer<'_> {
        Typer {
            scopes: self.scopes,
            known: self.known,
        }
    }

    fn assign_var(&mut self, v: VarId, ty: Ty) {
        let joined = self.out.var(v).join(ty);
        self.out.vars.insert(v, joined);
    }

    fn assign_binding(&mut self, binding: &Binding, ty: Ty) {
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => self.assign_var(*v, ty),
            Binding::Global(name) => {
                let joined = self.out.global(name).join(ty);
                self.out.globals.insert(name.clone(), joined);
            }
            Binding::Field(..) => {}
        }
    }

    /// The types of `n` targets assigned from `exprs`, Lua's way: the
    /// last expression's several values fill the rest, missing ones
    /// are nil.
    fn assigned_types(&self, exprs: &[&Expression], n: usize) -> Vec<Ty> {
        let typer = self.typer();
        let mut out = Vec::with_capacity(n);
        for (i, e) in exprs.iter().enumerate() {
            if out.len() >= n {
                break;
            }
            let last = i == exprs.len() - 1;
            if last && out.len() + 1 < n {
                // The last expression supplies the rest.
                match multi_returns(&typer, e) {
                    Some(Returns::Fixed(types)) => {
                        for k in 0..(n - out.len()) {
                            out.push(types.get(k).copied().unwrap_or(Ty::Nil));
                        }
                    }
                    Some(Returns::Dynamic) => {
                        while out.len() < n {
                            out.push(Ty::Any);
                        }
                    }
                    None => {
                        out.push(typer.ty_of(e));
                    }
                }
            } else {
                out.push(typer.ty_of(e));
            }
        }
        while out.len() < n {
            out.push(Ty::Nil);
        }
        out
    }

    fn block(&mut self, block: &Block) {
        for stmt in block.stmts() {
            self.stmt(stmt);
        }
        if let Some(ast::LastStmt::Return(r)) = block.last_stmt() {
            let exprs: Vec<&Expression> = r.returns().iter().collect();
            for e in &exprs {
                self.expr(e);
            }
            let typer = self.typer();
            let returns = match exprs.last() {
                None => Returns::Fixed(Vec::new()),
                Some(last) => {
                    let mut types: Vec<Ty> = exprs[..exprs.len() - 1]
                        .iter()
                        .map(|e| typer.ty_of(e))
                        .collect();
                    match multi_returns(&typer, last) {
                        Some(Returns::Fixed(rest)) => {
                            types.extend(rest);
                            Returns::Fixed(types)
                        }
                        Some(Returns::Dynamic) => Returns::Dynamic,
                        None => {
                            types.push(typer.ty_of(last));
                            Returns::Fixed(types)
                        }
                    }
                }
            };
            self.returns = Some(match self.returns.take() {
                Some(r) => r.join(returns),
                None => returns,
            });
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assignment(a) => {
                let exprs: Vec<&Expression> = a.expressions().iter().collect();
                for e in &exprs {
                    self.expr(e);
                }
                let targets: Vec<&Var> = a.variables().iter().collect();
                let types = self.assigned_types(&exprs, targets.len());
                for (target, ty) in targets.iter().zip(types) {
                    match target {
                        Var::Name(token) => {
                            if let Some(b) = self.scopes.binding(token).cloned() {
                                self.assign_binding(&b, ty);
                            }
                        }
                        Var::Expression(v) => {
                            let suffixes: Vec<&Suffix> = v.suffixes().collect();
                            if let Some(name) = self.typer().global_member(v.prefix(), &suffixes) {
                                self.assign_binding(&Binding::Global(name), ty);
                                continue;
                            }
                            self.prefix(v.prefix());
                            for s in v.suffixes() {
                                self.suffix(s);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Stmt::Do(d) => self.block(d.block()),
            Stmt::FunctionCall(c) => self.call(c),
            Stmt::FunctionDeclaration(f) => {
                let names: Vec<_> = f.name().names().iter().collect();
                if names.len() == 1
                    && f.name().method_name().is_none()
                    && let Some(b) = self.scopes.binding(names[0]).cloned()
                    && self.scopes.known_function(&b).is_none()
                {
                    self.assign_binding(&b, Ty::Any);
                }
                self.function(f.body());
            }
            Stmt::GenericFor(f) => {
                let exprs: Vec<&Expression> = f.expressions().iter().collect();
                for e in &exprs {
                    self.expr(e);
                }
                let names: Vec<VarId> = f.names().iter().map(|n| self.scopes.declared(n)).collect();
                // `ipairs` gives an integer key; anything else, dynamic
                // values.
                let ipairs = exprs.len() == 1
                    && matches!(exprs[0], Expression::FunctionCall(c)
                        if self.typer().builtin_callee(c.prefix(), &c.suffixes().collect::<Vec<_>>())
                            .is_some_and(|b| b.name == "ipairs" && b.lib.is_empty()));
                for (i, v) in names.iter().enumerate() {
                    let ty = if ipairs && i == 0 { Ty::Int } else { Ty::Any };
                    self.assign_var(*v, ty);
                }
                self.block(f.block());
            }
            Stmt::If(i) => {
                self.expr(i.condition());
                self.block(i.block());
                if let Some(elseifs) = i.else_if() {
                    for e in elseifs {
                        self.expr(e.condition());
                        self.block(e.block());
                    }
                }
                if let Some(b) = i.else_block() {
                    self.block(b);
                }
            }
            Stmt::LocalAssignment(l) => {
                let exprs: Vec<&Expression> = l.expressions().iter().collect();
                for e in &exprs {
                    self.expr(e);
                }
                let names: Vec<VarId> = l.names().iter().map(|n| self.scopes.declared(n)).collect();
                let types = if exprs.is_empty() {
                    vec![Ty::Nil; names.len()]
                } else {
                    self.assigned_types(&exprs, names.len())
                };
                for (v, ty) in names.iter().zip(types) {
                    self.assign_var(*v, ty);
                }
            }
            Stmt::LocalFunction(f) => {
                let v = self.scopes.declared(f.name());
                self.assign_var(v, Ty::Any);
                self.function(f.body());
            }
            Stmt::NumericFor(f) => {
                self.expr(f.start());
                self.expr(f.end());
                if let Some(s) = f.step() {
                    self.expr(s);
                }
                let typer = self.typer();
                let start = typer.ty_of(f.start());
                let step = f.step().map(|s| typer.ty_of(s)).unwrap_or(Ty::Int);
                let ty = if start == Ty::Float || step == Ty::Float {
                    Ty::Float
                } else {
                    Ty::Int
                };
                let v = self.scopes.declared(f.index_variable());
                self.assign_var(v, ty);
                self.block(f.block());
            }
            Stmt::Repeat(r) => {
                self.block(r.block());
                self.expr(r.until());
            }
            Stmt::While(w) => {
                self.expr(w.condition());
                self.block(w.block());
            }
            _ => {}
        }
    }

    /// A nested function: walked as its own, its signature recorded.
    fn function(&mut self, body: &ast::FunctionBody) {
        let id = self.scopes.function_of(body);
        let info = self.scopes.func(id);
        let outer_func = self.func;
        let outer_returns = self.returns.take();
        self.func = id;
        // Parameters: what direct calls pass, unless the function is
        // called through values too, when anything may arrive.
        let param_tys: Vec<Ty> = if info.escapes {
            vec![Ty::Any; info.params.len()]
        } else {
            info.params
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    self.known
                        .sig(id)
                        .and_then(|s| s.params.get(i).copied())
                        .unwrap_or(Ty::Unknown)
                })
                .collect()
        };
        for (v, ty) in info.params.iter().zip(&param_tys) {
            self.assign_var(*v, *ty);
        }
        self.block(body.block());
        let mut returns = self.returns.take().unwrap_or(Returns::Fixed(Vec::new()));
        if crate::types::falls_through(body.block()) {
            returns = returns.join(Returns::Fixed(Vec::new()));
        }
        // The parameters stay what this round's calls joined into them;
        // only the result is settled here.
        let n = info.params.len();
        let entry = self.out.funcs.entry(id).or_insert_with(|| Sig {
            params: vec![Ty::Unknown; n],
            returns: Returns::Fixed(Vec::new()),
        });
        entry.returns = returns;
        if info.escapes {
            entry.params = vec![Ty::Any; n];
        }
        self.func = outer_func;
        self.returns = outer_returns;
    }

    fn call(&mut self, c: &ast::FunctionCall) {
        self.prefix(c.prefix());
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        for s in &suffixes {
            self.suffix(s);
        }
        // A direct call to a known function records what it passes.
        if suffixes.len() == 1
            && let (Some(f), Some(Suffix::Call(ast::Call::AnonymousCall(args)))) =
                (self.typer().known_callee(c.prefix()), suffixes.first())
        {
            {
                let exprs: Vec<&Expression> = match args {
                    ast::FunctionArgs::Parentheses { arguments, .. } => arguments.iter().collect(),
                    _ => Vec::new(),
                };
                let n = self.scopes.func(f).params.len();
                let mut types = match args {
                    ast::FunctionArgs::String(_) => vec![Ty::Str],
                    ast::FunctionArgs::TableConstructor(_) => vec![Ty::Table],
                    _ => self.assigned_types(&exprs, n.max(exprs.len())),
                };
                types.resize(n, Ty::Nil);
                let sig = self.out.funcs.entry(f).or_insert_with(|| Sig {
                    params: vec![Ty::Unknown; n],
                    returns: Returns::Fixed(Vec::new()),
                });
                for (p, t) in sig.params.iter_mut().zip(types) {
                    *p = p.join(t);
                }
            }
        }
    }

    fn prefix(&mut self, p: &Prefix) {
        if let Prefix::Expression(e) = p {
            self.expr(e);
        }
    }

    fn suffix(&mut self, s: &Suffix) {
        match s {
            Suffix::Call(c) => match c {
                ast::Call::AnonymousCall(args) => self.args(args),
                ast::Call::MethodCall(m) => self.args(m.args()),
                _ => {}
            },
            Suffix::Index(ast::Index::Brackets { expression, .. }) => self.expr(expression),
            _ => {}
        }
    }

    fn args(&mut self, args: &ast::FunctionArgs) {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => {
                for a in arguments {
                    self.expr(a);
                }
            }
            ast::FunctionArgs::TableConstructor(t) => self.table(t),
            _ => {}
        }
    }

    fn table(&mut self, t: &ast::TableConstructor) {
        for field in t.fields() {
            match field {
                ast::Field::ExpressionKey { key, value, .. } => {
                    self.expr(key);
                    self.expr(value);
                }
                ast::Field::NameKey { value, .. } => self.expr(value),
                ast::Field::NoKey(e) => self.expr(e),
                _ => {}
            }
        }
    }

    /// Walk an expression for the functions and calls inside it.
    fn expr(&mut self, e: &Expression) {
        match e {
            Expression::BinaryOperator { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expression::Parentheses { expression, .. } => self.expr(expression),
            Expression::UnaryOperator { expression, .. } => self.expr(expression),
            Expression::Function(f) => self.function(f.body()),
            Expression::FunctionCall(c) => self.call(c),
            Expression::TableConstructor(t) => self.table(t),
            Expression::Var(Var::Expression(v)) => {
                self.prefix(v.prefix());
                for s in v.suffixes() {
                    self.suffix(s);
                }
            }
            _ => {}
        }
    }
}

/// What an expression in last position of a list supplies when it may
/// supply several values: a call's returns, or every vararg. `None`
/// for an expression of one value.
pub fn multi_returns(typer: &Typer<'_>, e: &Expression) -> Option<Returns> {
    match e {
        Expression::FunctionCall(c) => {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            Some(typer.call_returns(c.prefix(), &suffixes))
        }
        Expression::Var(Var::Expression(v)) => {
            let suffixes: Vec<&Suffix> = v.suffixes().collect();
            if matches!(suffixes.last(), Some(Suffix::Call(_))) {
                Some(typer.call_returns(v.prefix(), &suffixes))
            } else {
                None
            }
        }
        Expression::Symbol(t) if t.token().to_string().trim() == "..." => Some(Returns::Dynamic),
        _ => None,
    }
}

/// Infer the chunk: rounds until the types settle, then everything
/// unknown is nil.
pub fn infer(scopes: &Scopes, ast: &ast::Ast) -> Inferred {
    let mut known = Inferred::default();
    for _ in 0..16 {
        let mut round = Round {
            scopes,
            known: &known,
            out: Inferred::default(),
            func: CHUNK,
            returns: None,
        };
        // The signatures from the last round carry over so a recursive
        // call in this one sees them; params are rejoined from calls.
        for (f, sig) in &known.funcs {
            round.out.funcs.insert(
                *f,
                Sig {
                    params: vec![Ty::Unknown; sig.params.len()],
                    returns: sig.returns.clone(),
                },
            );
        }
        round.block(ast.nodes());
        // Functions declared but whose calls this round recorded
        // fewer parameters than declared.
        let out = round.out;
        if out == known {
            break;
        }
        known = out;
    }
    for ty in known.vars.values_mut() {
        *ty = ty.settled();
    }
    for ty in known.globals.values_mut() {
        *ty = ty.settled();
    }
    for sig in known.funcs.values_mut() {
        for p in &mut sig.params {
            *p = p.settled();
        }
        sig.returns = sig.returns.clone().settled();
    }
    if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
        let mut funcs: Vec<(&FuncId, &Sig)> = known.funcs.iter().collect();
        funcs.sort_by_key(|(f, _)| **f);
        for (f, sig) in funcs {
            let info = scopes.func(*f);
            eprintln!(
                "[types] {}@{} ({:?}) -> {:?}",
                info.name, info.line, sig.params, sig.returns
            );
        }
        let mut vars: Vec<(&VarId, &Ty)> = known.vars.iter().collect();
        vars.sort_by_key(|(v, _)| **v);
        for (v, ty) in vars {
            let info = scopes.var(*v);
            eprintln!("[types] local {} #{}: {ty:?}", info.name, v.0);
        }
        let mut globals: Vec<(&String, &Ty)> = known.globals.iter().collect();
        globals.sort_by(|a, b| a.0.cmp(b.0));
        for (name, ty) in globals {
            eprintln!("[types] global {name}: {ty:?}");
        }
    }
    known
}
