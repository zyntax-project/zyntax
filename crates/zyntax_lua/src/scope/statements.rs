//! Statements support.

use super::*;

impl Walker {
    pub(super) fn block(&mut self, block: &Block) {
        self.frame().blocks.push(HashMap::new());
        self.stmts(block, false);
        self.frame().blocks.pop();
    }

    pub(super) fn stmts(&mut self, block: &Block, _top: bool) {
        // The block's labels, numbered before its statements are walked.
        let mut labels = HashMap::new();
        let depth = self.frame().labels.len() + 1;
        for stmt in block.stmts() {
            if let Stmt::Label(l) = stmt {
                let id = self.out.next_label;
                self.out.next_label += 1;
                self.out.labels.insert(pos_of(l.name()), id);
                self.out.label_depths.insert(id, depth);
                labels.insert(name_of(l.name()), id);
            }
        }
        self.frame().labels.push(labels);
        for stmt in block.stmts() {
            self.stmt(stmt);
        }
        self.frame().labels.pop();
        if let Some(last) = block.last_stmt() {
            match last {
                ast::LastStmt::Return(r) => {
                    for e in r.returns() {
                        self.expr(e);
                    }
                }
                ast::LastStmt::Break(_) => {}
                _ => {}
            }
        }
    }

    pub(super) fn stmt(&mut self, stmt: &Stmt) {
        let notes = crate::annotation::before(stmt);
        self.stmt_of(stmt);
        if !notes.is_empty() {
            self.register(stmt, notes);
        }
    }

    /// Register the annotations before `stmt` with what it declares,
    /// once it is resolved.
    pub(super) fn register(&mut self, stmt: &Stmt, notes: Vec<((usize, usize), Annotation)>) {
        // What the statement declares: a function, and the variable or
        // global, or the field of one, it binds.
        let (function, holder, field) = match stmt {
            Stmt::FunctionDeclaration(f) => (
                self.out.func_at.get(&body_pos(f.body())).copied(),
                None,
                None,
            ),
            Stmt::LocalFunction(f) => (
                self.out.func_at.get(&body_pos(f.body())).copied(),
                None,
                None,
            ),
            Stmt::LocalAssignment(l) => {
                let holder = l
                    .names()
                    .iter()
                    .next()
                    .and_then(|n| self.out.decls.get(&pos_of(n)))
                    .map(|&v| Holder::Var(v));
                let function = l
                    .expressions()
                    .iter()
                    .next()
                    .and_then(|e| self.function_of(e));
                (function, holder, None)
            }
            Stmt::Assignment(a) => {
                let function = a
                    .expressions()
                    .iter()
                    .next()
                    .and_then(|e| self.function_of(e));
                match a.variables().iter().next() {
                    Some(Var::Name(token)) => (function, self.holder(token), None),
                    Some(Var::Expression(v)) => {
                        let suffixes: Vec<&Suffix> = v.suffixes().collect();
                        match (v.prefix(), suffixes.as_slice()) {
                            (
                                Prefix::Name(token),
                                [Suffix::Index(ast::Index::Dot { name, .. })],
                            ) => (function, self.holder(token), Some(name_of(name))),
                            _ => (function, None, None),
                        }
                    }
                    _ => (function, None, None),
                }
            }
            _ => (None, None, None),
        };
        let mut signature = Signature::default();
        let mut class: Option<Class> = None;
        for (span, note) in notes {
            match note {
                Annotation::Param(name, ty) => match function {
                    Some(id) if self.names_param(id, &name) => signature.params.push((name, ty)),
                    Some(_) => self.out.declared.problems.push((
                        span,
                        format!("@param `{name}` names no parameter of the function"),
                    )),
                    None => self
                        .out
                        .declared
                        .problems
                        .push((span, "@param annotates no function".to_owned())),
                },
                Annotation::Return(types) => signature.returns.extend(types),
                Annotation::Class(name) => match (&holder, &field) {
                    (Some(_), None) => {
                        class = Some(Class {
                            name,
                            fields: Vec::new(),
                        })
                    }
                    _ => self
                        .out
                        .declared
                        .problems
                        .push((span, "@class annotates no variable or global".to_owned())),
                },
                Annotation::Field(name, ty) => match &mut class {
                    Some(c) => c.fields.push((name, ty)),
                    None => self
                        .out
                        .declared
                        .problems
                        .push((span, "@field comes before any @class".to_owned())),
                },
                Annotation::Type(ty) => {
                    if let (Some(h), Some(f)) = (&holder, &field) {
                        self.out.declared.fields.insert((h.clone(), f.clone()), ty);
                    }
                }
            }
        }
        if let Some(id) = function
            && (!signature.params.is_empty() || !signature.returns.is_empty())
        {
            self.out.declared.funcs.insert(id, signature);
        }
        if let (Some(c), Some(h)) = (class, holder) {
            self.out.declared.classes.insert(h, c);
        }
    }

    /// The function an expression is, when it is a function literal.
    pub(super) fn function_of(&self, e: &Expression) -> Option<FuncId> {
        match e {
            Expression::Function(f) => self.out.func_at.get(&body_pos(f.body())).copied(),
            Expression::Parentheses { expression, .. } => self.function_of(expression),
            _ => None,
        }
    }

    /// The variable or global a name resolved to.
    pub(super) fn holder(&self, token: &TokenReference) -> Option<Holder> {
        match self.out.names.get(&pos_of(token))? {
            Binding::Local(v) | Binding::Upvalue(v) => Some(Holder::Var(*v)),
            Binding::Global(name) => Some(Holder::Global(name.clone())),
            Binding::Field(..) => None,
        }
    }

    /// Whether function `id` has a parameter `name`, `...` for its
    /// variable arguments.
    pub(super) fn names_param(&self, id: FuncId, name: &str) -> bool {
        let info = self.out.func(id);
        if name == "..." {
            return info.is_vararg;
        }
        info.params.iter().any(|&v| self.out.var(v).name == name)
    }

    pub(super) fn stmt_of(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assignment(a) => {
                // The targets resolve before the values, as the
                // reference's parser reads them, so upvalues are
                // numbered in its order; a global counts as written
                // only after the values that may read it.
                let mut writes = Vec::new();
                for target in a.variables() {
                    match target {
                        Var::Name(token) => writes.extend(self.assign_name(token)),
                        Var::Expression(v) => match global_table_member(self, v) {
                            Some(name) => {
                                if let Prefix::Name(token) = v.prefix() {
                                    self.out
                                        .names
                                        .insert(pos_of(token), Binding::Global(name_of(token)));
                                }
                                writes.push(name);
                            }
                            None => self.var_expression(v),
                        },
                        _ => {}
                    }
                }
                for e in a.expressions() {
                    self.expr(e);
                }
                for name in writes {
                    self.global_write(name);
                }
            }
            Stmt::Do(d) => self.block(d.block()),
            Stmt::FunctionCall(c) => self.call(c),
            Stmt::FunctionDeclaration(f) => {
                let names: Vec<&TokenReference> = f.name().names().iter().collect();
                let is_method = f.name().method_name().is_some();
                let mut fname = names
                    .iter()
                    .map(|n| name_of(n))
                    .collect::<Vec<_>>()
                    .join(".");
                if let Some(m) = f.name().method_name() {
                    fname = format!("{fname}:{}", name_of(m));
                }
                let single = names.len() == 1 && !is_method;
                if single {
                    // `function name()` assigns the name; at the chunk's
                    // outermost block it declares a top-level function.
                    let token = names[0];
                    let binding = self.lookup(&name_of(token));
                    let top = self.current() == CHUNK && self.frame().blocks.len() == 1;
                    match &binding {
                        Binding::Global(name) => self.global_write(name.clone()),
                        Binding::Local(v) | Binding::Upvalue(v) => {
                            self.out.vars[v.0 as usize].assigned = true;
                        }
                        Binding::Field(..) => {}
                    }
                    self.out.names.insert(pos_of(token), binding.clone());
                    let id = self.function(f.body(), is_method, fname.clone());
                    match &binding {
                        Binding::Global(name)
                            if top
                                && !self.out.global_functions.contains_key(name)
                                && self.out.func(id).captures.is_empty() =>
                        {
                            self.out.global_functions.insert(name.clone(), id);
                            self.out.funcs[id.0 as usize].top_level = true;
                        }
                        // Stored in an environment, the function is a
                        // value: nothing knows its callers. Assigned to
                        // a variable, the types follow it.
                        Binding::Field(..) => {
                            self.out.funcs[id.0 as usize].escapes = true;
                        }
                        Binding::Local(_) | Binding::Upvalue(_) | Binding::Global(_) => {}
                    }
                } else {
                    // `function a.b.c()`: `a` is read, the rest indexed;
                    // the types decide whether the table it is stored in
                    // is known.
                    self.use_name(names[0]);
                    self.function(f.body(), is_method, fname.clone());
                }
            }
            Stmt::GenericFor(f) => {
                for e in f.expressions() {
                    self.expr(e);
                }
                self.frame().blocks.push(HashMap::new());
                for name in f.names() {
                    self.declare(name, None);
                }
                self.block(f.block());
                self.frame().blocks.pop();
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
                // `local d = require "debug"`: `d` is the library, and
                // its calls by name are no use of it as a value.
                let alias = l.names().len() == 1
                    && l.expressions().len() == 1
                    && l.expressions()
                        .iter()
                        .next()
                        .is_some_and(|e| self.requires_debug(e));
                self.requiring_debug = alias;
                for e in l.expressions() {
                    self.expr(e);
                }
                self.requiring_debug = false;
                let attributes: Vec<Option<String>> = l
                    .attributes()
                    .map(|a| a.map(|a| name_of(a.name())))
                    .collect();
                let exprs: Vec<&Expression> = l.expressions().iter().collect();
                let names = l.names().len();
                for (i, name) in l.names().iter().enumerate() {
                    let attribute = attributes.get(i).cloned().flatten();
                    // Only the last of as many names as values can be
                    // folded, as in the reference compiler.
                    let folded = attribute.as_deref() == Some("const")
                        && i + 1 == names
                        && exprs.len() == names
                        && self.is_constant(exprs[i]);
                    let var = self.declare(name, attribute);
                    self.out.vars[var.0 as usize].folded = folded;
                    if folded && let Some((a, b)) = exprs[i].range() {
                        self.out.const_inits.insert(var, (a.bytes(), b.bytes()));
                    }
                    let init = exprs.get(i).map_or(Init::Other, |e| self.init_of(e));
                    self.out.vars[var.0 as usize].init = init;
                    if alias {
                        self.debug_aliases.insert(var);
                    }
                }
            }
            Stmt::LocalFunction(f) => {
                let var = self.declare(f.name(), None);
                self.out.vars[var.0 as usize].is_function = true;
                let id = self.function(f.body(), false, name_of(f.name()));
                self.out.vars[var.0 as usize].init = Init::Function(id);
                self.out.local_functions.insert(var, id);
                // A top-level function captures nothing but module
                // variables.
                let top = self.current() == CHUNK
                    && self.frame().blocks.len() == 1
                    && self.out.func(id).captures.is_empty();
                self.out.funcs[id.0 as usize].top_level = top;
            }
            Stmt::NumericFor(f) => {
                self.expr(f.start());
                self.expr(f.end());
                if let Some(step) = f.step() {
                    self.expr(step);
                }
                self.frame().blocks.push(HashMap::new());
                self.declare(f.index_variable(), None);
                self.block(f.block());
                self.frame().blocks.pop();
            }
            Stmt::Repeat(r) => {
                // The condition sees the body's locals.
                self.frame().blocks.push(HashMap::new());
                self.stmts(r.block(), false);
                self.expr(r.until());
                self.frame().blocks.pop();
            }
            Stmt::While(w) => {
                self.expr(w.condition());
                self.block(w.block());
            }
            Stmt::Goto(g) => {
                let name = name_of(g.label_name());
                let target = self
                    .frame()
                    .labels
                    .iter()
                    .rev()
                    .find_map(|labels| labels.get(&name).copied());
                if let Some(id) = target {
                    self.out.gotos.insert(pos_of(g.goto_token()), id);
                }
            }
            Stmt::Label(_) => {}
            _ => {}
        }
    }
}
