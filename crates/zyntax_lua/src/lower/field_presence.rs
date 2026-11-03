//! Constructor fields carried into entries whose callers all initialize them.

use super::*;

impl Lowerer<'_, '_> {
    fn tracks_fields(&self) -> bool {
        let scopes = self.scopes();
        !scopes.debug
            && !scopes.dynamic_code
            && !scopes.unseen_metatables
            && !scopes.debug_setlocal
            && !scopes.debug_rebinds
            && scopes.gotos.is_empty()
            && !self.m.inferred.blind_setmetatable
    }

    /// Called only after the straight-line fresh-result proof succeeds.
    pub(super) fn constructor_fields(
        &self,
        body: &ast::FunctionBody,
    ) -> Option<Rc<BTreeSet<String>>> {
        if !self.tracks_fields() {
            return None;
        }
        let Some(ast::LastStmt::Return(r)) = body.block().last_stmt() else {
            return None;
        };
        let result = r.returns().iter().next()?;
        let Ty::Shape(k) = self.typer().ty_of(result) else {
            return None;
        };
        if self.m.inferred.may_be_weak(k) {
            return None;
        }
        let info = self.m.inferred.shape(k);
        let mut fields: BTreeSet<String> = info
            .fields
            .iter()
            .filter(|(name, ty)| !ty.may_be_nil() && self.m.inferred.born_present(k, name))
            .map(|(name, _)| name.clone())
            .collect();
        for stmt in body.block().stmts() {
            match stmt {
                Stmt::LocalAssignment(l) => {
                    if let Some(init) = l.expressions().iter().next()
                        && let Some(f) = self.fresh_callee(init)
                        && let Some(inherited) = self.m.fresh_fields.borrow().get(&f)
                    {
                        fields.extend(inherited.iter().cloned());
                    }
                }
                Stmt::Assignment(a) => {
                    // The freshness proof restricts every target to a field
                    // of the returned local. Unpaired results remain unknown.
                    for (target, value) in a.variables().iter().zip(a.expressions()) {
                        let Var::Expression(e) = target else {
                            continue;
                        };
                        if let Some(Suffix::Index(ast::Index::Dot { name, .. })) =
                            e.suffixes().next()
                            && self.typer().non_nil(value)
                        {
                            fields.insert(ident(name));
                        }
                    }
                }
                _ => {}
            }
        }
        // These facts survive argument effects, aliases and later method calls.
        fields.retain(|name| self.m.inferred.keeps(k, name));
        (!fields.is_empty()).then(|| Rc::new(fields))
    }

    pub(super) fn note_initialized_locals(&mut self, l: &ast::LocalAssignment) {
        if !self.tracks_fields() {
            return;
        }
        for (name, init) in l.names().iter().zip(l.expressions()) {
            let v = self.scopes().declared(name);
            if !self.fixed_receiver(v) {
                continue;
            }
            let Some(f) = self.fresh_callee(init) else {
                continue;
            };
            let fields = self.m.fresh_fields.borrow().get(&f).cloned();
            if let Some(fields) = fields
                && let Storage::Local(name, _) | Storage::Module(name, _) = self.storage_of(v)
            {
                self.field_values.insert(name, fields);
            }
        }
    }

    pub(super) fn value_fields(&self, v: &Val) -> Option<Rc<BTreeSet<String>>> {
        let TypedExpression::Variable(name) = &v.node.node else {
            return None;
        };
        self.field_values.get(name).cloned()
    }

    /// Unknown, missing and dynamically expanded arguments empty the intersection.
    /// Record-code calls are excluded by the escape check when seeding an entry.
    pub(super) fn note_argument_fields(&self, f: FuncId, first: Option<&Val>) {
        if self.m.raising.is_some()
            || !self.tracks_fields()
            || !matches!(
                self.m.inferred.sig(f).and_then(|sig| sig.params.first()),
                Some(Ty::Shape(_))
            )
        {
            return;
        }
        let fields = first.and_then(|v| self.value_fields(v));
        self.m
            .argument_fields
            .borrow_mut()
            .entry(f)
            .and_modify(|common| {
                if let (Some(common), Some(fields)) = (common.as_mut(), fields.as_ref()) {
                    if !Rc::ptr_eq(common, fields) {
                        Rc::make_mut(common).retain(|name| fields.contains(name));
                    }
                } else {
                    *common = None;
                }
            })
            .or_insert(fields);
    }

    pub(super) fn seed_argument_fields(&mut self, f: FuncId) {
        let info = self.scopes().func(f);
        if self.m.raising.is_none()
            || !self.tracks_fields()
            || info.escapes
            || self.m.inferred.escaping.contains(&f)
        {
            return;
        }
        let Some(&v) = info.params.first() else {
            return;
        };
        if !self.fixed_receiver(v) {
            return;
        }
        let fields = self.m.argument_fields.borrow().get(&f).cloned().flatten();
        if let Some(fields) = fields
            && let Storage::Local(name, _) = self.storage_of(v)
        {
            self.field_values.insert(name, fields);
        }
    }
}
