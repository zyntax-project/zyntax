//! Fresh constructor results may share a layout without sharing a metatable.

use super::*;
use full_moon::visitors::{Visit, Visitor};

impl Lowerer<'_, '_> {
    /// Every normal return preserves the unchanged first argument.
    pub(super) fn returns_first_param(&self, id: FuncId, body: &ast::FunctionBody) -> bool {
        let Some(&param) = self.scopes().func(id).params.first() else {
            return false;
        };
        if !self.fixed_receiver(param) || types::falls_through(body.block()) {
            return false;
        }
        struct Returns<'a> {
            scopes: &'a Scopes,
            param: VarId,
            seen: bool,
            same: bool,
        }
        impl Visitor for Returns<'_> {
            fn visit_return(&mut self, r: &ast::Return) {
                self.seen = true;
                self.same &= r
                    .returns()
                    .iter()
                    .next()
                    .is_some_and(|e| local(self.scopes, unparen(e)) == Some(self.param));
            }
        }
        let mut returns = Returns {
            scopes: self.scopes(),
            param,
            seen: false,
            same: true,
        };
        body.block().visit(&mut returns);
        returns.seen && returns.same
    }

    /// A colon call supplies the receiver as its first argument. Its
    /// identity survives only when every resolved target returns it.
    pub(super) fn returned_receiver(
        &self,
        ty: Ty,
        name: &str,
        exact: Option<ShapeId>,
    ) -> Option<ShapeId> {
        let class = exact?;
        let Ty::Shape(k) = ty else {
            return None;
        };
        let folded = self.folded_class_lookups(k, name, exact)?;
        let returns_first = self.m.returns_first.borrow();
        let same = |ty| matches!(ty, Ty::Func(f) if returns_first.contains(&f));
        let preserved = match folded {
            Folded::Own(ty) => same(ty),
            Folded::Classes(classes) => {
                !classes.is_empty()
                    && classes
                        .iter()
                        .all(|(_, end)| end.is_some_and(|(_, ty)| same(ty)))
            }
        };
        preserved.then_some(class)
    }

    pub(super) fn fixed_receiver(&self, v: VarId) -> bool {
        let info = self.scopes().var(v);
        !info.assigned
            && !info.captured
            && info.attribute.is_none()
            && !self.scopes().debug_setlocal
            && !self.scopes().debug_rebinds
    }

    /// Resolve a constructor candidate only when its callee is present here.
    pub(super) fn fresh_callee(&self, e: &Expression) -> Option<FuncId> {
        let Expression::FunctionCall(c) = unparen(e) else {
            return None;
        };
        let suffixes: Vec<_> = c.suffixes().collect();
        let f = match suffixes.as_slice() {
            [Suffix::Call(ast::Call::AnonymousCall(_))] => self.typer().known_callee(c.prefix())?,
            [
                Suffix::Index(ast::Index::Dot { name, .. }),
                Suffix::Call(ast::Call::AnonymousCall(_)),
            ] => {
                let Ty::Shape(k) = self.typer().suffixed_ty(c.prefix(), &[]) else {
                    return None;
                };
                let Folded::Own(Ty::Func(f)) = self.folded_lookups(k, &ident(name))? else {
                    return None;
                };
                f
            }
            _ => return None,
        };
        Some(f)
    }

    fn fresh_call(&self, e: &Expression) -> Option<ShapeId> {
        let f = self.fresh_callee(e)?;
        self.m.fresh_classes.borrow().get(&f).copied()
    }

    fn set_class<'e>(&self, c: &'e ast::FunctionCall) -> Option<(&'e Expression, ShapeId)> {
        let suffixes: Vec<_> = c.suffixes().collect();
        let b = self.typer().builtin_callee(c.prefix(), &suffixes)?;
        if !b.lib.is_empty() || b.name != "setmetatable" {
            return None;
        }
        let [
            Suffix::Call(ast::Call::AnonymousCall(ast::FunctionArgs::Parentheses {
                arguments,
                ..
            })),
        ] = suffixes.as_slice()
        else {
            return None;
        };
        let mut args = arguments.iter();
        let table = args.next()?;
        let class = args.next()?;
        if args.next().is_some() || !self.typer().non_nil(class) {
            return None;
        }
        let Ty::Shape(k) = self.typer().ty_of(class) else {
            return None;
        };
        Some((table, k))
    }

    /// Recognize straight-line constructors whose fresh result is not
    /// exposed before return. Other bodies keep ordinary dispatch.
    pub(super) fn fresh_body(&self, body: &ast::FunctionBody) -> Option<ShapeId> {
        let block = body.block();
        let Some(ast::LastStmt::Return(r)) = block.last_stmt() else {
            return None;
        };
        let mut values = r.returns().iter();
        let value = unparen(values.next()?);
        if values.next().is_some() {
            return None;
        }
        if let Expression::FunctionCall(c) = value {
            let (table, class) = self.set_class(c)?;
            return (block.stmts().next().is_none()
                && matches!(unparen(table), Expression::TableConstructor(_)))
            .then_some(class);
        }
        let v = local(self.scopes(), value)?;
        if !self.fixed_receiver(v) {
            return None;
        }
        let mut initialized = false;
        let mut class = None;
        for stmt in block.stmts() {
            match stmt {
                Stmt::LocalAssignment(l) if !initialized => {
                    let mut names = l.names().iter();
                    if self.scopes().declared(names.next()?) != v || names.next().is_some() {
                        return None;
                    }
                    let mut values = l.expressions().iter();
                    let init = unparen(values.next()?);
                    if values.next().is_some()
                        || !(matches!(init, Expression::TableConstructor(_))
                            || self.fresh_call(init).is_some())
                    {
                        return None;
                    }
                    initialized = true;
                }
                Stmt::FunctionCall(c) if initialized => {
                    let (table, k) = self.set_class(c)?;
                    if local(self.scopes(), unparen(table)) != Some(v) {
                        return None;
                    }
                    class = Some(k);
                }
                Stmt::Assignment(a) if initialized => {
                    for target in a.variables() {
                        let Var::Expression(e) = target else {
                            return None;
                        };
                        let Prefix::Name(t) = e.prefix() else {
                            return None;
                        };
                        if !matches!(self.scopes().binding(t), Some(Binding::Local(id)) if *id == v)
                            || !matches!(
                                e.suffixes().collect::<Vec<_>>().as_slice(),
                                [Suffix::Index(ast::Index::Dot { .. })]
                            )
                        {
                            return None;
                        }
                    }
                    for value in a.expressions() {
                        if mentions(self.scopes(), v, value) {
                            return None;
                        }
                    }
                    // A setter could publish the receiver before return.
                    let Ty::Shape(k) = self.m.inferred.var(v) else {
                        return None;
                    };
                    let info = self.m.inferred.shape(k);
                    if info.unknown_meta
                        || info
                            .classes
                            .iter()
                            .any(|c| self.m.inferred.class_may_hold(*c, "__newindex"))
                    {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        class
    }

    pub(super) fn note_fresh_locals(&mut self, l: &ast::LocalAssignment) {
        let scopes = self.scopes();
        let inferred = self.m.inferred;
        if scopes.unseen_metatables
            || scopes.dynamic_code
            || inferred.blind_setmetatable
            || !scopes.gotos.is_empty()
        {
            return;
        }
        for (name, init) in l.names().iter().zip(l.expressions()) {
            let v = scopes.declared(name);
            if !self.fixed_receiver(v) {
                continue;
            }
            let Some(class) = self.fresh_call(init) else {
                continue;
            };
            self.m.fresh_locals.borrow_mut().insert(v);
            let Ty::Shape(k) = inferred.var(v) else {
                continue;
            };
            let info = inferred.shape(k);
            if info.unknown_meta || info.escapes {
                continue;
            }
            let fresh = self.m.fresh_locals.borrow();
            if info.meta_receivers.iter().all(|site| match site {
                types::MetaReceiver::Built => true,
                types::MetaReceiver::Local(other) => *other != v && fresh.contains(other),
                types::MetaReceiver::Other => false,
            }) {
                self.exact_classes.insert(v, class);
                if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
                    eprintln!("[types] exact {}#{}: {:?}", scopes.var(v).name, v.0, class);
                }
            }
        }
    }

    pub(super) fn exact_prefix(&self, prefix: &Prefix) -> Option<ShapeId> {
        let Prefix::Name(t) = prefix else {
            return None;
        };
        let Some(Binding::Local(v)) = self.scopes().binding(t) else {
            return None;
        };
        self.exact_classes.get(v).copied()
    }
}

fn unparen(mut e: &Expression) -> &Expression {
    while let Expression::Parentheses { expression, .. } = e {
        e = expression;
    }
    e
}

fn local(scopes: &Scopes, e: &Expression) -> Option<VarId> {
    match e {
        Expression::Var(Var::Name(t)) => match scopes.binding(t) {
            Some(Binding::Local(v)) => Some(*v),
            _ => None,
        },
        _ => None,
    }
}

fn mentions(scopes: &Scopes, var: VarId, e: &Expression) -> bool {
    struct Names<'a> {
        scopes: &'a Scopes,
        var: VarId,
        found: bool,
    }
    impl Visitor for Names<'_> {
        fn visit_token_reference(&mut self, t: &TokenReference) {
            self.found |= matches!(self.scopes.binding(t), Some(Binding::Local(v) | Binding::Upvalue(v)) if *v == self.var);
        }
    }
    let mut names = Names {
        scopes,
        var,
        found: false,
    };
    e.visit(&mut names);
    names.found
}
