//! `switch`, `do`/`while` and labeled statements. A `switch` dispatches with gotos to labels
//! laid out in source order, so clauses fall through as in JavaScript. A `break` or `continue`
//! that targets anything other than the innermost loop is a goto: past its target, or to the end
//! of the target loop's body.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum JumpKind {
    Loop,
    Switch,
    Block,
}

/// A statement that `break`, or for a loop `continue`, can leave.
pub(super) struct Jump<'ast> {
    pub kind: JumpKind,
    pub label: Option<&'ast str>,
    /// Placed after the statement: the target of a goto that leaves it.
    pub exit: InternedString,
    pub exit_used: bool,
    /// Placed at the end of a loop body: the target of a labeled `continue`.
    pub next: InternedString,
    pub next_used: bool,
    /// The flow at each jump to `exit`, merged into the flow after the statement.
    pub exits: Vec<FlowState>,
}

/// Names declared by `let`, `const` or `class` directly in a clause; they share the switch's
/// block scope.
fn clause_declarations<'ast>(case: &js::SwitchCase<'ast>) -> SmallVec<[&'ast str; 2]> {
    let mut names = SmallVec::new();
    for statement in &case.consequent {
        match statement {
            js::Statement::VariableDeclaration(d) if d.kind.as_str() != "var" => {
                for declarator in &d.declarations {
                    if let js::BindingPattern::BindingIdentifier(id) = &declarator.id {
                        names.push(id.name.as_str());
                    }
                }
            }
            js::Statement::ClassDeclaration(c) => {
                names.extend(c.id.as_ref().map(|id| id.name.as_str()));
            }
            _ => {}
        }
    }
    names
}

/// Whether statements can leave an enclosing `switch` or labeled statement other than by
/// returning: a `break` outside nested loops and switches, a `continue` outside nested loops,
/// or any labeled jump.
fn leaves(statements: &[js::Statement<'_>]) -> bool {
    struct Leaves {
        loops: usize,
        switches: usize,
        found: bool,
    }
    impl<'a> Visit<'a> for Leaves {
        fn visit_break_statement(&mut self, it: &js::BreakStatement<'a>) {
            self.found |= it.label.is_some() || (self.loops == 0 && self.switches == 0);
        }
        fn visit_continue_statement(&mut self, it: &js::ContinueStatement<'a>) {
            self.found |= it.label.is_some() || self.loops == 0;
        }
        fn visit_while_statement(&mut self, it: &js::WhileStatement<'a>) {
            self.loops += 1;
            walk::walk_while_statement(self, it);
            self.loops -= 1;
        }
        fn visit_do_while_statement(&mut self, it: &js::DoWhileStatement<'a>) {
            self.loops += 1;
            walk::walk_do_while_statement(self, it);
            self.loops -= 1;
        }
        fn visit_for_statement(&mut self, it: &js::ForStatement<'a>) {
            self.loops += 1;
            walk::walk_for_statement(self, it);
            self.loops -= 1;
        }
        fn visit_for_of_statement(&mut self, it: &js::ForOfStatement<'a>) {
            self.loops += 1;
            walk::walk_for_of_statement(self, it);
            self.loops -= 1;
        }
        fn visit_for_in_statement(&mut self, it: &js::ForInStatement<'a>) {
            self.loops += 1;
            walk::walk_for_in_statement(self, it);
            self.loops -= 1;
        }
        fn visit_switch_statement(&mut self, it: &js::SwitchStatement<'a>) {
            self.switches += 1;
            walk::walk_switch_statement(self, it);
            self.switches -= 1;
        }
        fn visit_function(&mut self, _: &js::Function<'a>, _: oxc_semantic::ScopeFlags) {}
        fn visit_arrow_function_expression(&mut self, _: &js::ArrowFunctionExpression<'a>) {}
    }
    let mut leaves = Leaves {
        loops: 0,
        switches: 0,
        found: false,
    };
    for statement in statements {
        leaves.visit_statement(statement);
    }
    leaves.found
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// Opens a breakable statement; a loop or block takes the label that names it, if any.
    pub(super) fn open_jump(&mut self, kind: JumpKind) {
        let label = match kind {
            JumpKind::Switch => None,
            JumpKind::Loop | JumpKind::Block => self.pending_label.take(),
        };
        let exit = self.fresh("exit");
        let next = self.fresh("next");
        self.jumps.push(Jump {
            kind,
            label,
            exit,
            exit_used: false,
            next,
            next_used: false,
            exits: vec![],
        });
    }

    /// Closes a breakable statement: places the labels its jumps used, and merges the flow at
    /// each jump to its exit into the flow after it.
    pub(super) fn close_jump(
        &mut self,
        body: Option<&mut TypedBlock>,
        after: &mut Vec<Stmt>,
        at: oxc_span::Span,
    ) {
        let jump = self.jumps.pop().unwrap();
        if jump.next_used
            && let Some(body) = body
        {
            // A nested block keeps a `return` in the body from ending the list the label is in.
            let statements = std::mem::take(&mut body.statements);
            body.statements = vec![
                stmt(
                    TypedStatement::Block(TypedBlock {
                        statements,
                        span: body.span,
                    }),
                    at,
                ),
                stmt(TypedStatement::Label(jump.next), at),
            ];
        }
        if jump.exit_used {
            after.push(stmt(TypedStatement::Label(jump.exit), at));
        }
        for exit in &jump.exits {
            self.merge_flow(exit);
        }
    }

    /// `break` and `break label`.
    pub(super) fn break_statement(
        &mut self,
        label: Option<&js::LabelIdentifier<'ast>>,
        at: oxc_span::Span,
    ) -> Result<Vec<Stmt>> {
        let target = match label {
            Some(label) => self
                .jumps
                .iter()
                .rposition(|j| j.label == Some(label.name.as_str())),
            None => self.jumps.iter().rposition(|j| j.kind != JumpKind::Block),
        };
        let Some(target) = target else {
            return self.error(ErrorKind::Syntax, "break outside a loop or switch", at);
        };
        let innermost = self.jumps.iter().rposition(|j| j.kind == JumpKind::Loop);
        if self.jumps[target].kind == JumpKind::Loop && innermost == Some(target) {
            return Ok(vec![stmt(TypedStatement::Break(None), at)]);
        }
        let flow = self.flow.clone();
        let jump = &mut self.jumps[target];
        jump.exit_used = true;
        jump.exits.push(flow);
        Ok(vec![stmt(TypedStatement::Goto(jump.exit), at)])
    }

    /// `continue` and `continue label`.
    pub(super) fn continue_statement(
        &mut self,
        label: Option<&js::LabelIdentifier<'ast>>,
        at: oxc_span::Span,
    ) -> Result<Vec<Stmt>> {
        let target = match label {
            Some(label) => self
                .jumps
                .iter()
                .rposition(|j| j.label == Some(label.name.as_str()) && j.kind == JumpKind::Loop),
            None => self.jumps.iter().rposition(|j| j.kind == JumpKind::Loop),
        };
        let Some(target) = target else {
            return self.error(ErrorKind::Syntax, "continue outside a loop", at);
        };
        let innermost = self.jumps.iter().rposition(|j| j.kind == JumpKind::Loop);
        if innermost == Some(target) {
            return Ok(vec![stmt(TypedStatement::Continue, at)]);
        }
        let jump = &mut self.jumps[target];
        jump.next_used = true;
        Ok(vec![stmt(TypedStatement::Goto(jump.next), at)])
    }

    /// `label: statement`. A loop takes the label; any other statement becomes a block that
    /// `break label` leaves.
    pub(super) fn labeled(&mut self, l: &js::LabeledStatement<'ast>) -> Result<Vec<Stmt>> {
        let label = l.label.name.as_str();
        if self.jumps.iter().any(|j| j.label == Some(label)) {
            return self.error(ErrorKind::Syntax, "duplicate label", l.label.span);
        }
        self.pending_label = Some(label);
        if matches!(
            l.body,
            js::Statement::WhileStatement(_)
                | js::Statement::DoWhileStatement(_)
                | js::Statement::ForStatement(_)
                | js::Statement::ForOfStatement(_)
                | js::Statement::ForInStatement(_)
        ) {
            return self.statement(&l.body);
        }
        self.open_jump(JumpKind::Block);
        let statements = self.statement(&l.body)?;
        let mut out = vec![stmt(
            TypedStatement::Block(TypedBlock {
                statements,
                span: span(l.span),
            }),
            l.span,
        )];
        self.close_jump(None, &mut out, l.span);
        Ok(out)
    }

    /// `do body while (test)`: a `while (first || test)` whose body clears `first`, so a
    /// `continue` still evaluates the test.
    pub(super) fn do_while(&mut self, d: &js::DoWhileStatement<'ast>) -> Result<Vec<Stmt>> {
        let at = d.span;
        let first = self.fresh("first");
        let flag = |value| {
            node(
                TypedExpression::Literal(TypedLiteral::Bool(value)),
                boolean(),
                at,
            )
        };
        let mut out = vec![stmt(
            TypedStatement::Let(TypedLet {
                name: first,
                ty: boolean(),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(flag(true))),
                span: span(at),
            }),
            at,
        )];
        // Later iterations start after the test, with whatever the body wrote.
        self.clear_written_flow(at);
        self.open_jump(JumpKind::Loop);
        self.loop_depth += 1;
        let body = self.block(&d.body);
        self.loop_depth -= 1;
        let mut body = body?;
        body.statements.insert(
            0,
            expression(binary(
                BinaryOp::Assign,
                var(first, boolean(), at),
                flag(false),
                unit(),
                at,
            )),
        );
        let test = self.condition(&d.test)?;
        let mut after = vec![];
        self.close_jump(Some(&mut body), &mut after, at);
        self.clear_written_flow(at);
        if !Self::may_break(std::slice::from_ref(&d.body)) {
            self.narrow(&d.test, false);
        }
        out.push(stmt(
            TypedStatement::While(TypedWhile {
                condition: Box::new(binary(
                    BinaryOp::Or,
                    var(first, boolean(), at),
                    test,
                    boolean(),
                    at,
                )),
                body,
                span: span(at),
            }),
            at,
        ));
        out.extend(after);
        Ok(out)
    }

    pub(super) fn switch(&mut self, s: &js::SwitchStatement<'ast>) -> Result<Vec<Stmt>> {
        let at = s.span;
        // A later clause naming a binding declared in an earlier one would read it in its
        // temporal dead zone when entered directly.
        for (index, case) in s.cases.iter().enumerate() {
            let declared = clause_declarations(case);
            if declared.is_empty() {
                continue;
            }
            struct Mentions<'n, 'a> {
                names: &'n [&'a str],
                found: bool,
            }
            impl<'a> Visit<'a> for Mentions<'_, 'a> {
                fn visit_identifier_reference(&mut self, id: &js::IdentifierReference<'a>) {
                    self.found |= self.names.contains(&id.name.as_str());
                }
            }
            let mut mentions = Mentions {
                names: &declared,
                found: false,
            };
            for later in &s.cases[index + 1..] {
                for statement in &later.consequent {
                    mentions.visit_statement(statement);
                }
            }
            if mentions.found {
                return self.unsupported(
                    "switch clauses sharing block-scoped declarations",
                    case.span,
                );
            }
        }
        let discriminant = self.expr(&s.discriminant)?;
        let ty = discriminant.ty.clone();
        if !(is_scalar(&ty) || ty == string()) {
            return self.unsupported("switch over this type", s.discriminant.span());
        }
        let value = self.fresh("switch");
        let mut out = vec![stmt(
            TypedStatement::Let(TypedLet {
                name: value,
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(discriminant)),
                span: span(at),
            }),
            at,
        )];
        let labels: SmallVec<[InternedString; 8]> =
            s.cases.iter().map(|_| self.fresh("case")).collect();
        // Tests run in order until one matches, before any clause.
        let mut covered: Option<enums::Members> = None;
        for (case, label) in s.cases.iter().zip(&labels) {
            let Some(test) = &case.test else {
                continue;
            };
            self.check_comparison(&s.discriminant, test, test.span())?;
            let tested = self.expr(test)?;
            if tested.ty != ty {
                return self.error(
                    ErrorKind::Type,
                    format!(
                        "type `{}` is not comparable to type `{}`",
                        self.type_description(&tested.ty),
                        self.type_description(&ty)
                    ),
                    test.span(),
                );
            }
            let tested_members = match (self.static_value(test), self.static_value(&s.discriminant))
            {
                (
                    enums::Static::Enum { view, members },
                    enums::Static::Enum { view: other, .. },
                ) if view == other => Some(members),
                // A literal case covers the member of a literal type with its value.
                (literal, enums::Static::Enum { view, .. }) => self
                    .enum_of(Some(view))
                    .filter(|enumeration| enumeration.literal)
                    .and_then(|enumeration| match literal {
                        enums::Static::Text(text) => {
                            Some(enumeration.valued(enums::Value::Text(text)))
                        }
                        enums::Static::Number(n) => {
                            Some(enumeration.valued(enums::Value::Number(n)))
                        }
                        _ => None,
                    }),
                _ => None,
            };
            if let Some(members) = tested_members {
                covered = Some(match covered {
                    Some(earlier) => earlier.union(&members),
                    None => members,
                });
            }
            let read = var(value, ty.clone(), test.span());
            let equal = if ty == string() {
                self.call("$String$equals", vec![read, tested], boolean(), test.span())
            } else {
                binary(BinaryOp::Eq, read, tested, boolean(), test.span())
            };
            out.push(stmt(
                TypedStatement::If(TypedIf {
                    condition: Box::new(equal),
                    then_block: TypedBlock {
                        statements: vec![stmt(TypedStatement::Goto(*label), test.span())],
                        span: span(test.span()),
                    },
                    else_block: None,
                    span: span(test.span()),
                }),
                test.span(),
            ));
        }
        let default = s.cases.iter().position(|c| c.test.is_none());
        // Every member the discriminant may hold has a clause: the unmatched path is unreachable.
        let exhaustive = default.is_none()
            && covered.is_some_and(|covered| {
                matches!(
                    self.static_value(&s.discriminant),
                    enums::Static::Enum { members, .. } if members.without(&covered).is_empty()
                )
            });
        if exhaustive {
            self.exhaustive_switches.insert(at.start);
        }
        let dispatched = self.flow.clone();
        self.open_jump(JumpKind::Switch);
        let exit = self.jumps.last().unwrap().exit;
        if exhaustive {
            // Only a numeric value outside the enum's members gets here, which TypeScript
            // treats as impossible and JavaScript would run past.
            out.push(expression(self.call("$TS$switch_trap", vec![], unit(), at)));
        }
        out.push(stmt(
            TypedStatement::Goto(default.map_or(exit, |index| labels[index])),
            at,
        ));
        if default.is_none() && !exhaustive {
            self.narrow_cases(&s.discriminant, &s.cases);
            let unmatched = std::mem::replace(&mut self.flow, dispatched.clone());
            let jump = self.jumps.last_mut().unwrap();
            jump.exit_used = true;
            jump.exits.push(unmatched);
        }
        self.scopes.push(FxHashMap::default());
        let mut falling: Option<FlowState> = None;
        for (case, label) in s.cases.iter().zip(&labels) {
            self.flow = dispatched.clone();
            match &case.test {
                Some(test) => self.narrow_case(&s.discriminant, test, true),
                None => self.narrow_cases(&s.discriminant, &s.cases),
            }
            if let Some(previous) = falling.take() {
                self.merge_flow(&previous);
            }
            out.push(stmt(TypedStatement::Label(*label), case.span));
            self.reserve(&case.consequent)?;
            let mut clause = vec![];
            for statement in &case.consequent {
                clause.extend(self.statement(statement)?);
            }
            // A nested block keeps a `return` in the clause from ending the list of labels.
            out.push(stmt(
                TypedStatement::Block(TypedBlock {
                    statements: clause,
                    span: span(case.span),
                }),
                case.span,
            ));
            falling = (!self.returns(&case.consequent) && !leaves(&case.consequent))
                .then(|| self.flow.clone());
        }
        self.pop_scope();
        // Control after the switch comes from its exits and from the end of the last clause.
        let jump = self.jumps.last_mut().unwrap();
        let reached: Vec<FlowState> = std::mem::take(&mut jump.exits)
            .into_iter()
            .chain(falling)
            .collect();
        jump.exit_used = true;
        match reached.split_first() {
            Some((first, rest)) => {
                self.flow = first.clone();
                for flow in rest {
                    self.merge_flow(flow);
                }
            }
            None => self.flow = dispatched,
        }
        self.close_jump(None, &mut out, at);
        Ok(out)
    }

    /// Narrows the discriminant inside a clause, or for `false`, past a clause that did not match.
    fn narrow_case(
        &mut self,
        discriminant: &js::Expression<'ast>,
        test: &js::Expression<'ast>,
        equality: bool,
    ) {
        if self.has_binding_write(test.span()) {
            return;
        }
        self.narrow_equal(discriminant, test, equality);
    }

    /// Past every case test: what the default clause and the unmatched path see.
    fn narrow_cases(
        &mut self,
        discriminant: &js::Expression<'ast>,
        cases: &[js::SwitchCase<'ast>],
    ) {
        for case in cases {
            if let Some(test) = &case.test {
                self.narrow_case(discriminant, test, false);
            }
        }
    }

    /// Whether statements reach their end only through a `return`: also a `switch` whose
    /// clauses all return and which no jump leaves, and a `do` body that returns.
    pub(super) fn switch_returns(&self, s: &js::SwitchStatement<'ast>) -> bool {
        let complete = s.cases.iter().any(|c| c.test.is_none())
            || self.exhaustive_switches.contains(&s.span.start);
        complete
            && s.cases.last().is_some_and(|c| self.returns(&c.consequent))
            && !s.cases.iter().any(|c| leaves(&c.consequent))
    }
}
