//! Expressions support.

use super::*;

impl Walker {
    /// A function body: a new frame with its parameters declared.
    pub(super) fn function(
        &mut self,
        body: &FunctionBody,
        is_method: bool,
        name: String,
    ) -> FuncId {
        let id = FuncId(self.out.funcs.len() as u32);
        let pos = body_pos(body);
        let line = body
            .parameters_parentheses()
            .tokens()
            .0
            .token()
            .start_position()
            .line();
        let end = body.end_token().token().end_position().bytes();
        self.out.funcs.push(FuncInfo {
            params: Vec::new(),
            is_vararg: false,
            captures: Vec::new(),
            upvalues: Vec::new(),
            line,
            body: (pos, end),
            method: is_method,
            name,
            escapes: false,
            top_level: false,
            returns_function: false,
        });
        self.out.func_at.insert(pos, id);
        self.frames.push(Frame {
            id,
            blocks: vec![HashMap::new()],
            labels: Vec::new(),
        });
        let mut params = Vec::new();
        if is_method {
            let name = "self".to_string();
            let vid = VarId(self.out.vars.len() as u32);
            self.out.vars.push(VarInfo {
                name: name.clone(),
                func: id,
                captured: false,
                assigned: false,
                outermost: false,
                is_function: false,
                attribute: None,
                folded: false,
                self_captured: false,
                init: Init::Other,
            });
            self.frame().blocks.last_mut().unwrap().insert(name, vid);
            params.push(vid);
        }
        for p in body.parameters() {
            match p {
                ast::Parameter::Name(token) => {
                    params.push(self.declare(token, None));
                }
                ast::Parameter::Ellipsis(_) => {
                    self.out.funcs[id.0 as usize].is_vararg = true;
                }
                _ => {}
            }
        }
        self.out.funcs[id.0 as usize].params = params;
        self.stmts(body.block(), false);
        self.frames.pop();
        self.out.funcs[id.0 as usize].returns_function =
            !crate::types::falls_through(body.block()) && returns_functions(body.block());
        id
    }

    /// Whether `e` is a value Lua computes where it compiles it: a nil,
    /// boolean, number or string literal, a folded `<const>` local, or
    /// unary `-` and binary `+`, `-`, `*` over number literals.
    pub(super) fn is_constant(&self, e: &Expression) -> bool {
        fn numeric(e: &Expression) -> bool {
            match e {
                Expression::Number(_) => true,
                Expression::UnaryOperator {
                    unop: ast::UnOp::Minus(_),
                    expression,
                } => numeric(expression),
                Expression::BinaryOperator { lhs, binop, rhs } => {
                    matches!(
                        binop,
                        ast::BinOp::Plus(_) | ast::BinOp::Minus(_) | ast::BinOp::Star(_)
                    ) && numeric(lhs)
                        && numeric(rhs)
                }
                _ => false,
            }
        }
        match e {
            Expression::String(_) => true,
            Expression::Symbol(t) => {
                matches!(t.token().to_string().trim(), "nil" | "true" | "false")
            }
            Expression::Var(Var::Name(t)) => self.folded(t),
            _ => numeric(e),
        }
    }

    /// Whether `token` names a folded `<const>` local here.
    pub(super) fn folded(&self, token: &TokenReference) -> bool {
        let name = name_of(token);
        for frame in self.frames.iter().rev() {
            for block in frame.blocks.iter().rev() {
                if let Some(&id) = block.get(&name) {
                    return self.out.vars[id.0 as usize].folded;
                }
            }
        }
        false
    }

    /// What a declaration's expression tells of the value, once the
    /// expression has been walked.
    pub(super) fn init_of(&self, e: &Expression) -> Init {
        match e {
            Expression::Parentheses { expression, .. } => self.init_of(expression),
            Expression::Function(f) => Init::Function(self.out.func_at[&body_pos(f.body())]),
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                match (c.prefix(), suffixes.as_slice()) {
                    (Prefix::Name(callee), [Suffix::Call(ast::Call::AnonymousCall(_))]) => {
                        Init::Call(pos_of(callee))
                    }
                    _ => Init::Other,
                }
            }
            _ => Init::Other,
        }
    }

    pub(super) fn var_expression(&mut self, v: &ast::VarExpression) {
        if let Some(name) = global_table_member(self, v) {
            // `_G.name` is the global `name`; the table's own name is
            // bound without being a value, under its own name so that
            // a program which replaces `_ENV` reads it from there.
            if let Prefix::Name(token) = v.prefix() {
                self.out
                    .names
                    .insert(pos_of(token), Binding::Global(name_of(token)));
            }
            self.global_value(&name);
            if name == "debug" {
                self.debug_value();
            }
            self.out.mentioned.insert(name.clone());
            self.out.globals.insert(name);
            return;
        }
        let suffixes: Vec<&Suffix> = v.suffixes().collect();
        self.note_require(v.prefix(), &suffixes);
        self.note_metatable_callee(v.prefix(), &suffixes);
        self.note_debug_call(v.prefix(), &suffixes);
        self.note_table_index(v.prefix(), &suffixes);
        self.prefix(v.prefix());
        for s in v.suffixes() {
            self.suffix(s);
        }
    }

    /// `require "name"` heading a call or an index chain: a file the
    /// program is made of, whatever is done with what it returns.
    pub(super) fn note_require(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        if let (Prefix::Name(callee), [Suffix::Call(ast::Call::AnonymousCall(args)), ..]) =
            (prefix, suffixes)
            && name_of(callee) == "require"
            && !self.shadowed("require")
        {
            self.require_calls.insert(pos_of(callee));
            let name = required_name(args);
            if name.as_deref() == Some("debug") {
                if self.requiring_debug {
                    self.out.debug = true;
                } else {
                    self.debug_value();
                }
            }
            match name {
                Some(name) => {
                    if !self.out.requires.contains(&name) {
                        self.out.requires.push(name);
                    }
                }
                // A name only known when it runs may be "debug".
                None => {
                    self.out.dynamic_code = true;
                    self.debug_value();
                }
            }
        }
    }

    pub(super) fn call(&mut self, c: &ast::FunctionCall) {
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        self.note_require(c.prefix(), &suffixes);
        // `rawget(_G, "name")` and `rawset(_G, "name", v)` name the
        // global; `_G` there is not the table as a value.
        if let (Prefix::Name(callee), [Suffix::Call(ast::Call::AnonymousCall(args))]) =
            (c.prefix(), suffixes.as_slice())
            && let ast::FunctionArgs::Parentheses { arguments, .. } = args
            && let name = name_of(callee)
            && (name == "rawget" || name == "rawset")
            && !self.shadowed(&name)
            && let arguments = arguments.iter().collect::<Vec<&Expression>>()
            && let Some(Expression::Var(Var::Name(g))) = arguments.first()
            && Scopes::is_globals_name(&name_of(g))
            && !self.shadowed(&name_of(g))
            && let Some(member) = arguments.get(1).and_then(|e| literal_string(e))
        {
            self.use_name(callee);
            let binding = self.lookup(&name_of(g));
            self.out.names.insert(pos_of(g), binding);
            if name == "rawget" {
                self.global_value(&member);
            }
            self.out.mentioned.insert(member.clone());
            self.out.globals.insert(member.clone());
            if name == "rawset" {
                self.global_write(member);
            }
            for a in arguments.iter().skip(1) {
                self.expr(a);
            }
            return;
        }
        self.note_metatable_callee(c.prefix(), &suffixes);
        self.note_debug_call(c.prefix(), &suffixes);
        self.note_table_index(c.prefix(), &suffixes);
        self.prefix(c.prefix());
        for s in suffixes {
            self.suffix(s);
        }
    }

    pub(super) fn prefix(&mut self, p: &Prefix) {
        match p {
            Prefix::Name(token) => {
                self.use_name(token);
            }
            Prefix::Expression(e) => self.expr(e),
            _ => {}
        }
    }

    pub(super) fn suffix(&mut self, s: &Suffix) {
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

    pub(super) fn args(&mut self, args: &ast::FunctionArgs) {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => {
                for a in arguments {
                    self.expr(a);
                }
            }
            ast::FunctionArgs::TableConstructor(t) => self.table(t),
            ast::FunctionArgs::String(_) => {}
            _ => {}
        }
    }

    pub(super) fn table(&mut self, t: &ast::TableConstructor) {
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

    pub(super) fn expr(&mut self, e: &Expression) {
        match e {
            Expression::BinaryOperator { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expression::Parentheses { expression, .. } => self.expr(expression),
            Expression::UnaryOperator { expression, .. } => self.expr(expression),
            Expression::Function(f) => {
                self.function(f.body(), false, String::new());
            }
            Expression::FunctionCall(c) => self.call(c),
            Expression::TableConstructor(t) => self.table(t),
            Expression::Number(_) | Expression::String(_) | Expression::Symbol(_) => {}
            Expression::Var(v) => match v {
                Var::Name(token) => {
                    self.use_name(token);
                }
                Var::Expression(v) => self.var_expression(v),
                _ => {}
            },
            _ => {}
        }
    }
}
