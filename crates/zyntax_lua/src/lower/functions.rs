//! Functions for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// Lower a function's body into its typed entry, and its record
    /// code if it is used as a value. Nested functions that capture
    /// something take their record as `env`.
    pub(super) fn lower_function(
        &mut self,
        id: FuncId,
        body: &ast::FunctionBody,
        _is_method: bool,
    ) -> Result<()> {
        let span = span_of(body);
        let fresh = self.fresh_body(body);
        let fields = fresh.and_then(|_| self.constructor_fields(body));
        let returns_first = self.returns_first_param(id, body);
        let info = self.scopes().func(id).clone();
        let sig = self.m.sig(id);
        let has_env = self.m.takes_env(id);
        let mut child = Lowerer::new(self.m, id);
        child.frames = self.m.debug.is_some();
        child.live = info.params.iter().map(|v| Live::Var(*v)).collect();
        // The function runs after it is made, when everything there by
        // then still is.
        child.present = self.present.clone();
        let mut params = Vec::new();
        let mut statements = Vec::new();
        if has_env {
            params.push(parameter(intern("env"), self.m.anys(), span));
            statements.extend(child.env_prologue(span));
        }
        for (i, v) in info.params.iter().enumerate() {
            let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
            let local_ty = child.var_ty(*v);
            let vinfo = self.scopes().var(*v).clone();
            let symbol = child.local_symbol(*v);
            let arg = intern(&format!("{}$arg", vinfo.name));
            params.push(parameter(arg, child.ir(param_ty), span));
            // The parameter as the body's variable: its own type, or a cell.
            let value = Val {
                node: var(arg, child.ir(param_ty), span),
                ty: param_ty,
            };
            if vinfo.needs_cell() {
                let cell_symbol = intern(&format!("{}$cell{}", vinfo.name, v.0));
                child
                    .storage
                    .insert(*v, Storage::Cell(cell_symbol, local_ty));
                statements.push(child.declare_var(*v, value, span));
            } else {
                child.storage.insert(*v, Storage::Local(symbol, local_ty));
                let value = child.coerce(value, local_ty);
                statements.push(let_(symbol, child.ir(local_ty), value, span));
            }
        }
        child.seed_argument_fields(id);
        if info.is_vararg {
            let name = intern("$varargs");
            params.push(parameter(name, self.m.anys(), span));
            child.varargs = Some(name);
        }
        // The function counts itself on the stack, so an unbounded
        // recursion is an error to catch; every return uncounts it. One
        // whose depth is bounded by the text keeps no count.
        child.counts_depth = !self.m.is_bounded(id);
        child.end_span = Some(span_of(body.end_token()));
        let body_statements = child.block(body.block())?;
        if child.entry_line {
            statements.push(entry_line_save(span));
        }
        if child.frames {
            statements.push(child.debug_enter(id, span));
        }
        if child.counts_depth {
            statements.push(child.stack_check(span));
        }
        statements.extend(body_statements);
        // Falling off the end returns nothing, from the line of `end`.
        let last_line = self.m.line_of(span_of(body.end_token()));
        if types::falls_through(body.block()) {
            if child.frames {
                statements.extend(child.debug_line(last_line, span));
            }
            child.return_stmt(&[], span, &mut statements)?;
        }
        if child.frames {
            let line = self
                .m
                .source_line(span_of(body.parameters_parentheses().tokens().0));
            let last_line = self.m.source_line(span_of(body.end_token()));
            child.debug_function_records(id, line, last_line);
        }
        self.m.strip_line_restores(&mut statements);
        let entry = self.m.entry_name(id);
        let function = typed_function(
            &entry,
            params,
            self.m.return_ir(&sig.returns),
            statements,
            span,
        );
        let dynamic = self.m.calls_dynamically(
            function
                .body
                .as_ref()
                .map_or(&[][..], |b| b.statements.as_slice()),
        );
        self.m.facts.borrow_mut().insert(
            id,
            RaiseFact {
                own: child.raised,
                checks_depth: child.counts_depth,
                dynamic,
                callees: child.raise_callees.clone(),
            },
        );
        self.m.functions.borrow_mut().push(function);
        if let Some(class) = fresh {
            self.m.fresh_classes.borrow_mut().insert(id, class);
        }
        if let Some(fields) = fields {
            self.m.fresh_fields.borrow_mut().insert(id, fields);
        }
        if returns_first {
            self.m.returns_first.borrow_mut().insert(id);
        }
        // The record code, whether or not anything takes the function
        // as a value: unreached, it costs nothing.
        self.record_code(id, &sig, span);
        Ok(())
    }

    /// The record code of a function: the shape every function value
    /// has, calling the typed entry.
    pub(super) fn record_code(&mut self, id: FuncId, sig: &types::Sig, span: Span) {
        let info = self.scopes().func(id).clone();
        let mut params = vec![parameter(intern("env"), self.m.anys(), span)];
        let mut args: Vec<Node> = Vec::new();
        if self.m.takes_env(id) {
            args.push(var(intern("env"), self.m.anys(), span));
        }
        let mut statements = Vec::new();
        let n = info.params.len();
        let mut child = Lowerer::new(self.m, id);
        if info.is_vararg {
            // Variadic: one packed list, split into the parameters and
            // the rest.
            params.push(parameter(intern("packed"), Type::Any, span));
            let all = intern("$all");
            statements.push(let_(
                all,
                self.m.anys(),
                call(
                    "zl_values",
                    vec![var(intern("packed"), Type::Any, span)],
                    self.m.anys(),
                    span,
                ),
                span,
            ));
            for i in 0..n {
                let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
                let value = call(
                    "zl_value_at",
                    vec![var(all, self.m.anys(), span), int_lit(i as i64 + 1, span)],
                    Type::Any,
                    span,
                );
                args.push(child.coerce(
                    Val {
                        node: value,
                        ty: Ty::Any,
                    },
                    param_ty,
                ));
            }
            args.push(call(
                "zl_slice",
                vec![
                    var(all, self.m.anys(), span),
                    int_lit(n as i64, span),
                    list_len(var(all, self.m.anys(), span), span),
                ],
                self.m.anys(),
                span,
            ));
        } else {
            for i in 0..n {
                let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
                let arg = intern(&format!("a{i}"));
                params.push(parameter(arg, Type::Any, span));
                args.push(child.coerce(
                    Val {
                        node: var(arg, Type::Any, span),
                        ty: Ty::Any,
                    },
                    param_ty,
                ));
            }
        }
        let value = call(
            &self.m.entry_name(id),
            args,
            self.m.return_ir(&sig.returns),
            span,
        );
        // The result as one dynamic value.
        let result = match &sig.returns {
            Returns::Fixed(types) if types.is_empty() => block_value(
                vec![expr_stmt(value)],
                call("zl_none", vec![], Type::Any, span),
                span,
            ),
            Returns::Fixed(types) if types.len() == 1 => child.coerce(
                Val {
                    node: value,
                    ty: types[0].settled(),
                },
                Ty::Any,
            ),
            Returns::Fixed(_) => call("zb_box_tuple", vec![value], Type::Any, span),
            Returns::Dynamic | Returns::Unknown => value,
        };
        statements.push(ret(Some(result), span));
        let function = typed_function(&self.m.code_name(id), params, Type::Any, statements, span);
        self.m.functions.borrow_mut().push(function);
    }
}
