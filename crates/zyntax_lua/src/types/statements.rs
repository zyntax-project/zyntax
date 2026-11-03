//! Statements support.

use super::*;

impl<'a> Round<'a> {
    /// `escape`, and whether it made the function or shape escape.
    pub(super) fn escape_new(&mut self, ty: Ty) -> bool {
        let new = match ty {
            Ty::Func(f) => !self.out.escaping.contains(&f),
            Ty::Shape(k) => !self.out.shapes[k.0 as usize].escapes,
            _ => false,
        };
        self.escape(ty);
        new
    }

    /// Whether the function's parameters take anything: decided by the
    /// escapes found at the last settled typing, not by this round's,
    /// so that a type an early round got wrong cannot fix a function's
    /// parameters for good.
    pub(super) fn is_escaping(&self, f: FuncId) -> bool {
        self.scopes.func(f).escapes || self.known.escaping.contains(&f)
    }

    /// `value` joined into what `slot` holds: a function joined with
    /// anything but itself is lost to the types, and escapes.
    pub(super) fn join_into(&mut self, slot: Ty, value: Ty) -> Ty {
        let joined = slot.join(value);
        // A side the join did not keep as it was is lost to the types.
        if joined != slot {
            self.escape(slot);
        }
        if joined != value {
            self.escape(value);
        }
        joined
    }

    pub(super) fn join_returns(&mut self, a: Returns, b: Returns) -> Returns {
        match (a, b) {
            (Returns::Unknown, r) | (r, Returns::Unknown) => r,
            (Returns::Fixed(x), Returns::Fixed(y)) if x.len() == y.len() => Returns::Fixed(
                x.into_iter()
                    .zip(y)
                    .map(|(p, q)| self.join_into(p, q))
                    .collect(),
            ),
            (a, b) => {
                self.escape_returns(&a);
                self.escape_returns(&b);
                Returns::Dynamic
            }
        }
    }

    /// A call that may reach any of `targets` yields the join of what
    /// they return, as the call's type says. A result the join does not
    /// keep as it was is lost to the types, and escapes: the tables it
    /// names reach code that does not know their shape.
    pub(super) fn join_target_returns(&mut self, targets: &[Ty]) {
        let mut out = Returns::Unknown;
        for target in targets {
            if let Ty::Func(f) = target
                && let Some(sig) = self.known.sig(*f)
            {
                let returns = sig.returns.clone();
                out = self.join_returns(out, returns);
            }
        }
    }

    pub(super) fn escape_returns(&mut self, r: &Returns) {
        if let Returns::Fixed(types) = r {
            for t in types {
                self.escape(*t);
            }
        }
    }

    pub(super) fn assign_var(&mut self, v: VarId, ty: Ty) {
        let joined = self.join_into(self.out.var(v), ty);
        self.out.vars.insert(v, joined);
    }

    pub(super) fn assign_binding(&mut self, binding: &Binding, ty: Ty) {
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => self.assign_var(*v, ty),
            Binding::Global(name) => {
                // In a program that reaches its globals through a
                // table, a global is an entry of it.
                if self.scopes.dynamic_globals {
                    self.escape(ty);
                }
                let joined = self.join_into(self.out.global(name), ty);
                self.out.globals.insert(name.clone(), joined);
            }
            // A free name in the scope of a local `_ENV`: a store into
            // its field.
            Binding::Field(env, name, _) => {
                let receiver = self.known.var(*env);
                self.store_through(receiver, Some(name.clone()), Ty::Str, ty, false);
            }
        }
    }

    /// An expression whose value goes where the types do not follow
    /// it: a function there escapes.
    pub(super) fn value(&mut self, e: &Expression) {
        self.expr(e);
        let ty = self.typer().ty_of(e);
        self.escape(ty);
    }

    /// The types of `n` targets assigned from `exprs`, Lua's way: the
    /// last expression's several values fill the rest, missing ones
    /// are nil.
    pub(super) fn assigned_types(&self, exprs: &[&Expression], n: usize) -> Vec<Ty> {
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
                    Some(Returns::Unknown) => {
                        while out.len() < n {
                            out.push(Ty::Unknown);
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

    pub(super) fn block(&mut self, block: &Block) {
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
                        // A tail not typed yet: this return decides
                        // nothing this round.
                        Some(Returns::Unknown) => Returns::Unknown,
                        None => {
                            types.push(typer.ty_of(last));
                            Returns::Fixed(types)
                        }
                    }
                }
            };
            self.returns = Some(match self.returns.take() {
                Some(r) => self.join_returns(r, returns),
                None => returns,
            });
        }
    }

    pub(super) fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assignment(a) => {
                let exprs: Vec<&Expression> = a.expressions().iter().collect();
                let targets: Vec<&Var> = a.variables().iter().collect();
                // Stored into a variable or a shape's field, a value
                // keeps its type; into a table, it is on its own.
                let kept = targets.iter().all(|t| match t {
                    Var::Name(_) => true,
                    Var::Expression(v) => {
                        let suffixes: Vec<&Suffix> = v.suffixes().collect();
                        if self.typer().global_member(v.prefix(), &suffixes).is_some() {
                            return true;
                        }
                        let receiver =
                            self.typer().suffixed_ty(v.prefix(), &suffixes[..suffixes.len() - 1]);
                        // Under a key not known at compile time, into a
                        // table no handler may take it from.
                        let keyed = match receiver {
                            Ty::Shape(k) => self.known.plain(k),
                            _ => receiver == Ty::Unknown,
                        };
                        matches!(receiver, Ty::Shape(_) | Ty::Unknown)
                            && matches!(suffixes.last(), Some(Suffix::Index(i)) if keyed || constant_key(i).is_some())
                    }
                    _ => false,
                });
                for e in &exprs {
                    if kept {
                        self.expr(e);
                    } else {
                        self.value(e);
                    }
                }
                let types = self.assigned_types(&exprs, targets.len());
                for (i, (target, ty)) in targets.iter().zip(types).enumerate() {
                    // A target takes the first value of its own
                    // expression, or one the last expression adds.
                    let non_nil = exprs.get(i).is_some_and(|e| self.typer().non_nil(e));
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
                            self.chain(v.prefix(), &suffixes[..suffixes.len() - 1]);
                            let receiver = self
                                .typer()
                                .suffixed_ty(v.prefix(), &suffixes[..suffixes.len() - 1]);
                            if let Some(Suffix::Index(index)) = suffixes.last() {
                                let key = constant_key(index);
                                let key_ty = match index {
                                    ast::Index::Brackets { expression, .. } => {
                                        self.value(expression);
                                        self.typer().ty_of(expression)
                                    }
                                    _ => Ty::Str,
                                };
                                self.store_through(receiver, key, key_ty, ty, non_nil);
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
                let id = self.scopes.function_of(f.body());
                let method = f.name().method_name();
                if names.len() == 1
                    && method.is_none()
                    && let Some(b) = self.scopes.binding(names[0]).cloned()
                {
                    self.assign_binding(&b, Ty::Func(id));
                } else {
                    // `function a.b.c()` or `function a.b:c()`: stored in
                    // the field `c` of `a.b`, whatever that is.
                    let last = match method {
                        Some(m) => ident(m),
                        None => ident(names[names.len() - 1]),
                    };
                    let path_len = if method.is_some() {
                        names.len()
                    } else {
                        names.len() - 1
                    };
                    let mut receiver = match self.scopes.binding(names[0]) {
                        Some(b) => self.typer().binding_ty(b),
                        None => Ty::Any,
                    };
                    for name in &names[1..path_len] {
                        receiver = self.typer().field_ty(receiver, &ident(name));
                    }
                    self.store_through(receiver, Some(last), Ty::Str, Ty::Func(id), true);
                }
                self.function(f.body());
            }
            Stmt::GenericFor(f) => {
                let exprs: Vec<&Expression> = f.expressions().iter().collect();
                // The iterator is called by the loop, through its value.
                let outer = self.loop_iterator.take();
                if let [Expression::FunctionCall(c)] = exprs.as_slice()
                    && let Prefix::Name(token) = c.prefix()
                {
                    self.loop_iterator = Some(crate::scope::pos_of(token));
                }
                for e in &exprs {
                    self.value(e);
                }
                self.loop_iterator = outer;
                let names: Vec<VarId> = f.names().iter().map(|n| self.scopes.declared(n)).collect();
                // A `gmatch` walked in place binds each capture: a string,
                // or a position for `()`; nil past the last.
                if let Some(captures) = self.typer().gmatch_walk(&exprs) {
                    for (i, v) in names.iter().enumerate() {
                        let ty = match captures.get(i) {
                            Some(true) => Ty::Int,
                            Some(false) => Ty::Str,
                            None => Ty::Nil,
                        };
                        self.assign_var(*v, ty);
                    }
                    self.loop_block(f.block());
                    return;
                }
                // `ipairs` gives an integer key and each element; a
                // typed `pairs` any key and each value its shape holds;
                // anything else, dynamic values.
                let walk = match exprs.as_slice() {
                    [Expression::FunctionCall(c)] => {
                        let suffixes: Vec<&Suffix> = c.suffixes().collect();
                        match (
                            self.typer().builtin_callee(c.prefix(), &suffixes),
                            suffixes.last(),
                        ) {
                            (
                                Some(b),
                                Some(Suffix::Call(ast::Call::AnonymousCall(
                                    ast::FunctionArgs::Parentheses { arguments, .. },
                                ))),
                            ) if borrows(b) == Borrows::Walk => {
                                let table = arguments
                                    .iter()
                                    .next()
                                    .map(|a| self.typer().ty_of(a))
                                    .unwrap_or(Ty::Nil);
                                Some((b.name, table))
                            }
                            (Some(b), _) if borrows(b) == Borrows::Walk => Some((b.name, Ty::Any)),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                let (key, value) = match walk {
                    Some(("ipairs", table)) => (Ty::Int, self.known.ipairs_value_ty(table)),
                    Some((_, table)) => (Ty::Any, self.known.pairs_value_ty(table)),
                    None => (Ty::Any, Ty::Any),
                };
                for (i, v) in names.iter().enumerate() {
                    let ty = match i {
                        0 => key,
                        1 => value,
                        _ => Ty::Any,
                    };
                    self.assign_var(*v, ty);
                }
                self.loop_block(f.block());
            }
            Stmt::If(i) => {
                self.condition(i.condition());
                self.block(i.block());
                if let Some(elseifs) = i.else_if() {
                    for e in elseifs {
                        self.condition(e.condition());
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
                let id = self.scopes.function_of(f.body());
                self.assign_var(v, Ty::Func(id));
                self.function(f.body());
            }
            Stmt::NumericFor(f) => {
                self.value(f.start());
                self.value(f.end());
                if let Some(s) = f.step() {
                    self.value(s);
                }
                let typer = self.typer();
                let start = typer.ty_of(f.start());
                let step = f.step().map(|s| typer.ty_of(s)).unwrap_or(Ty::Int);
                let ty = numeric_for_ty(start, step);
                let v = self.scopes.declared(f.index_variable());
                self.assign_var(v, ty);
                self.loop_block(f.block());
            }
            Stmt::Repeat(r) => {
                self.loops += 1;
                self.block(r.block());
                self.condition(r.until());
                self.loops -= 1;
            }
            Stmt::While(w) => {
                self.loops += 1;
                self.condition(w.condition());
                self.block(w.block());
                self.loops -= 1;
            }
            Stmt::Goto(_) | Stmt::Label(_) if self.func == CHUNK => self.chunk_jumps = true,
            _ => {}
        }
    }

    /// An expression whose value is only tested for truth, which Lua
    /// takes raw: what `not`, `and` and `or` combine goes nowhere.
    pub(super) fn condition(&mut self, e: &Expression) {
        match e {
            Expression::Parentheses { expression, .. }
            | Expression::UnaryOperator {
                unop: UnOp::Not(_),
                expression,
            } => self.condition(expression),
            Expression::BinaryOperator {
                lhs,
                binop: BinOp::And(_) | BinOp::Or(_),
                rhs,
            } => {
                self.condition(lhs);
                self.condition(rhs);
            }
            _ => self.expr(e),
        }
    }

    /// A loop's body, which may run any number of times.
    pub(super) fn loop_block(&mut self, block: &Block) {
        self.loops += 1;
        self.block(block);
        self.loops -= 1;
    }

    /// A nested function: walked as its own, its signature recorded.
    pub(super) fn function(&mut self, body: &ast::FunctionBody) {
        let id = self.scopes.function_of(body);
        let info = self.scopes.func(id);
        let outer_func = self.func;
        let outer_returns = self.returns.take();
        self.func = id;
        // Parameters: what direct calls pass, unless the function is
        // called through values too, when anything may arrive.
        let escapes = self.is_escaping(id);
        let param_tys: Vec<Ty> = if escapes {
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
            returns = self.join_returns(returns, Returns::Fixed(Vec::new()));
        }
        // What an escaping function returns leaves through its record,
        // as dynamic values.
        if escapes {
            self.escape_returns(&returns);
        }
        // The parameters stay what this round's calls joined into them;
        // only the result is settled here.
        let n = info.params.len();
        let entry = self.out.funcs.entry(id).or_insert_with(|| Sig {
            params: vec![Ty::Unknown; n],
            returns: Returns::Fixed(Vec::new()),
        });
        entry.returns = returns;
        if escapes {
            entry.params = vec![Ty::Any; n];
        }
        self.func = outer_func;
        self.returns = outer_returns;
    }
}
