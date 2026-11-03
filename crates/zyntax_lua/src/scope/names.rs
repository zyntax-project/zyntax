//! Names support.

use super::*;

pub fn pos_of(token: &TokenReference) -> usize {
    token.token().start_position().bytes()
}

pub fn body_pos(body: &FunctionBody) -> usize {
    pos_of(body.parameters_parentheses().tokens().0)
}

pub(super) fn name_of(token: &TokenReference) -> String {
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::Identifier { identifier } => identifier.to_string(),
        _ => token.token().to_string(),
    }
}

/// The global `_G.name` or `_G["name"]` names (`_ENV` as well), when
/// `_G` is the global of that name and nothing shadows it.
pub(super) fn global_table_member(w: &Walker, v: &ast::VarExpression) -> Option<String> {
    let Prefix::Name(token) = v.prefix() else {
        return None;
    };
    let g = name_of(token);
    if !Scopes::is_globals_name(&g) || w.shadowed(&g) || w.shadowed("_ENV") {
        return None;
    }
    let suffixes: Vec<&Suffix> = v.suffixes().collect();
    match suffixes.as_slice() {
        [Suffix::Index(ast::Index::Dot { name, .. })] => Some(name_of(name)),
        [Suffix::Index(ast::Index::Brackets { expression, .. })] => literal_string(expression),
        _ => None,
    }
}

/// Whether the global `name` with these suffixes is a call the types
/// follow without taking `name` as a value: `setmetatable(t, m)` as a
/// whole expression, `require "file"` heading an expression (a file
/// of the program, whose own metatable calls the program's assembly
/// accounts for), or a `debug` function that hands
/// back no way to set a metatable (`debug.setmetatable(t, m)` again
/// only as a whole expression).
pub(super) fn metatable_call(name: &str, suffixes: &[&Suffix]) -> bool {
    let parenthesized = |s: &Suffix| {
        matches!(
            s,
            Suffix::Call(ast::Call::AnonymousCall(
                ast::FunctionArgs::Parentheses { .. }
            ))
        )
    };
    match (name, suffixes) {
        ("setmetatable", [call]) => parenthesized(call),
        ("require", [Suffix::Call(ast::Call::AnonymousCall(args)), ..]) => {
            required_name(args).is_some_and(|f| !matches!(f.as_str(), "debug" | "_G" | "package"))
        }
        (
            "debug",
            [
                Suffix::Index(ast::Index::Dot { name: member, .. }),
                call,
                rest @ ..,
            ],
        ) => {
            parenthesized(call)
                && match name_of(member).as_str() {
                    "setmetatable" => rest.is_empty(),
                    "getmetatable" | "traceback" | "getinfo" | "sethook" | "gethook" => true,
                    _ => false,
                }
        }
        _ => false,
    }
}

/// The name `require` is called with, when it is a string literal.
pub(super) fn required_name(args: &ast::FunctionArgs) -> Option<String> {
    match args {
        ast::FunctionArgs::String(s) => literal_string_token(s),
        ast::FunctionArgs::Parentheses { arguments, .. } => {
            arguments.iter().next().and_then(literal_string)
        }
        _ => None,
    }
}

/// The string a string literal expression spells, escapes decoded.
pub fn literal_string(e: &Expression) -> Option<String> {
    let Expression::String(token) = e else {
        return None;
    };
    literal_string_token(token)
}

/// The string a string literal token spells, escapes decoded; none when
/// it is not a string literal, is malformed, or is not UTF-8 text.
pub fn literal_string_token(token: &TokenReference) -> Option<String> {
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::StringLiteral { .. } => {
            String::from_utf8(crate::lower::string_bytes(token).ok()?).ok()
        }
        _ => None,
    }
}

impl Walker {
    pub(super) fn frame(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("a function being walked")
    }

    pub(super) fn current(&self) -> FuncId {
        self.frames.last().expect("a function being walked").id
    }

    pub(super) fn declare(&mut self, token: &TokenReference, attribute: Option<String>) -> VarId {
        let name = name_of(token);
        let func = self.current();
        let outermost = func == CHUNK && self.frame().blocks.len() == 1 && !self.no_module_vars;
        let id = VarId(self.out.vars.len() as u32);
        self.out.vars.push(VarInfo {
            name: name.clone(),
            func,
            captured: false,
            assigned: false,
            outermost,
            is_function: false,
            attribute,
            folded: false,
            self_captured: false,
            init: Init::Other,
        });
        self.frame().blocks.last_mut().unwrap().insert(name, id);
        self.out.decls.insert(pos_of(token), id);
        id
    }

    /// Resolve `name` from the current position: a local of this
    /// function, a variable of an enclosing one, or a global.
    pub(super) fn lookup(&mut self, name: &str) -> Binding {
        let depth = self.frames.len();
        for (level, frame) in self.frames.iter().enumerate().rev() {
            for block in frame.blocks.iter().rev() {
                if let Some(&id) = block.get(name) {
                    if level == depth - 1 {
                        return Binding::Local(id);
                    }
                    self.out.vars[id.0 as usize].captured = true;
                    if !self.out.vars[id.0 as usize].folded {
                        for inner in &self.frames[level + 1..] {
                            let f = &mut self.out.funcs[inner.id.0 as usize];
                            if !f.upvalues.contains(&Upvalue::Var(id)) {
                                f.upvalues.push(Upvalue::Var(id));
                            }
                        }
                    }
                    let module_var = self.out.vars[id.0 as usize].is_module_var();
                    if !module_var {
                        // Every function between the use and the
                        // declaration carries the capture.
                        for inner in &self.frames[level + 1..] {
                            let f = &mut self.out.funcs[inner.id.0 as usize];
                            if !f.captures.contains(&id) {
                                f.captures.push(id);
                            }
                        }
                    }
                    return Binding::Upvalue(id);
                }
            }
        }
        // A free name where a variable named `_ENV` is visible is a
        // field of it; `_ENV` not declared as a local is the
        // environment itself, which the lowering reads as the globals
        // table.
        if name != "_ENV" && self.shadowed("_ENV") {
            match self.lookup("_ENV") {
                Binding::Local(id) => return Binding::Field(id, name.to_string(), false),
                Binding::Upvalue(id) => return Binding::Field(id, name.to_string(), true),
                _ => {}
            }
        }
        for frame in &self.frames[1..] {
            let f = &mut self.out.funcs[frame.id.0 as usize];
            if !f.upvalues.contains(&Upvalue::Env) {
                f.upvalues.push(Upvalue::Env);
            }
        }
        self.out.globals.insert(name.to_string());
        Binding::Global(name.to_string())
    }

    pub(super) fn use_global(&mut self, binding: &Binding) {
        if let Binding::Global(name) = binding {
            self.out.mentioned.insert(name.clone());
        }
    }

    pub(super) fn use_name(&mut self, token: &TokenReference) -> Binding {
        let binding = self.lookup(&name_of(token));
        self.use_global(&binding);
        // The globals table reached as a value: every global is then
        // an entry of a real table.
        // A chunk loaded while the program runs shares its globals
        // through the table too.
        if let Binding::Global(name) = &binding
            && (Scopes::is_globals_name(name) || name == "load" || name == "dofile")
        {
            self.out.dynamic_globals = true;
        }
        if let Binding::Global(name) = &binding
            && matches!(name.as_str(), "load" | "loadfile" | "dofile")
        {
            self.out.dynamic_code = true;
        }
        if let Binding::Global(name) = &binding
            && ((name == "package" && !self.tables_indexed.contains(&pos_of(token)))
                || (name == "require" && !self.require_calls.contains(&pos_of(token))))
        {
            self.out.dynamic_code = true;
        }
        if let Binding::Global(name) = &binding
            && !self.metatable_callees.contains(&pos_of(token))
        {
            self.global_value(name);
        }
        let library = match &binding {
            Binding::Global(name) => name == "debug",
            Binding::Local(v) | Binding::Upvalue(v) => self.debug_aliases.contains(v),
            Binding::Field(..) => false,
        };
        if library {
            self.out.debug = true;
            if !self.debug_calls.contains(&pos_of(token)) {
                self.debug_value();
            }
        }
        // The globals table or `package` as a value, or indexed by what
        // may be "debug": the library may be reached through it.
        if let Binding::Global(name) = &binding
            && (Scopes::is_globals_name(name) || name == "package")
            && !self.tables_indexed.contains(&pos_of(token))
        {
            self.debug_value();
        }
        self.out.names.insert(pos_of(token), binding.clone());
        binding
    }

    /// `_G`, `_ENV` or `package` indexed: whether the keys are literals
    /// that do not lead to the `debug` library (`_G.x`, `package.path`,
    /// `package.loaded.x`), noted for [`Self::use_name`].
    pub(super) fn note_table_index(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        let Prefix::Name(token) = prefix else {
            return;
        };
        let name = name_of(token);
        if !Scopes::is_globals_name(&name) && name != "package" {
            return;
        }
        let key = |s: Option<&&Suffix>| match s {
            Some(Suffix::Index(ast::Index::Dot { name, .. })) => Some(name_of(name)),
            Some(Suffix::Index(ast::Index::Brackets { expression, .. })) => {
                literal_string(expression)
            }
            _ => None,
        };
        let safe = match key(suffixes.first()) {
            Some(k) if name == "package" && k == "loaded" => {
                key(suffixes.get(1)).is_some_and(|k| k != "debug")
            }
            Some(k) => k != "debug",
            None => false,
        };
        if safe {
            self.tables_indexed.insert(pos_of(token));
        }
        if name == "package"
            && matches!(
                key(suffixes.first()).as_deref(),
                Some("loadlib" | "searchers")
            )
        {
            self.out.dynamic_code = true;
        }
    }

    /// The local a name resolves to where the walk is, without noting
    /// a capture.
    pub(super) fn visible_local(&self, name: &str) -> Option<VarId> {
        self.frames
            .iter()
            .rev()
            .flat_map(|frame| frame.blocks.iter().rev())
            .find_map(|block| block.get(name).copied())
    }

    /// Whether `e` is `require "debug"`, `require` being the global.
    pub(super) fn requires_debug(&self, e: &Expression) -> bool {
        let Expression::FunctionCall(c) = e else {
            return false;
        };
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        let (Prefix::Name(callee), [Suffix::Call(ast::Call::AnonymousCall(args))]) =
            (c.prefix(), suffixes.as_slice())
        else {
            return false;
        };
        let name = match args {
            ast::FunctionArgs::String(s) => literal_string_token(s),
            ast::FunctionArgs::Parentheses { arguments, .. } => {
                arguments.iter().next().and_then(literal_string)
            }
            _ => None,
        };
        name_of(callee) == "require"
            && !self.shadowed("require")
            && name.as_deref() == Some("debug")
    }

    /// The `debug` library reached as a value: any of its functions
    /// may be called.
    pub(super) fn debug_value(&mut self) {
        self.out.debug = true;
        self.out.debug_rebinds = true;
        self.out.debug_setlocal = true;
        self.out.debug_getlocal = true;
    }

    /// `debug.name(...)`: a call of one of the library's functions by
    /// name, which is no use of the library as a value.
    pub(super) fn note_debug_call(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        let Prefix::Name(token) = prefix else {
            return;
        };
        let name = name_of(token);
        let library = match self.visible_local(&name) {
            Some(v) => self.debug_aliases.contains(&v),
            None => name == "debug",
        };
        if !library {
            return;
        }
        if let [
            Suffix::Index(ast::Index::Dot { name: member, .. }),
            Suffix::Call(_),
            ..,
        ] = suffixes
        {
            self.debug_calls.insert(pos_of(token));
            match name_of(member).as_str() {
                "setupvalue" | "upvaluejoin" => self.out.debug_rebinds = true,
                "setlocal" => self.out.debug_setlocal = true,
                "getlocal" => self.out.debug_getlocal = true,
                "upvalueid" => self.out.upvalue_ids = true,
                _ => {}
            }
        }
    }

    /// The global `name` taken as a value: when it is a way to set a
    /// metatable, metatables may be set where the types do not see.
    pub(super) fn global_value(&mut self, name: &str) {
        if name == "debug" && !self.shadowed("debug") {
            self.out.debug = true;
        }
        if Scopes::is_globals_name(name)
            || matches!(name, "setmetatable" | "debug" | "package" | "require")
        {
            self.out.unseen_metatables = true;
        }
    }

    /// A name called in a way the types follow as a metatable call is
    /// not a use of it as a value.
    pub(super) fn note_metatable_callee(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        if let Prefix::Name(callee) = prefix
            && metatable_call(&name_of(callee), suffixes)
        {
            self.metatable_callees.insert(pos_of(callee));
        }
    }

    /// Whether a local of this name is in scope.
    pub(super) fn shadowed(&self, name: &str) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.blocks.iter().any(|block| block.contains_key(name)))
    }

    /// The name `token` as an assignment's target: the global it
    /// writes, if any, which the caller records once the assigned
    /// values are walked.
    pub(super) fn assign_name(&mut self, token: &TokenReference) -> Option<String> {
        let binding = self.lookup(&name_of(token));
        let mut global = None;
        match &binding {
            Binding::Local(id) | Binding::Upvalue(id) => {
                self.out.vars[id.0 as usize].assigned = true;
                // What the library is known through becomes anything.
                if self.debug_aliases.contains(id) {
                    self.debug_value();
                }
            }
            Binding::Global(name) => {
                // `_ENV = t` replaces the environment: every global is
                // then an entry of a real table.
                if name == "_ENV" {
                    self.out.dynamic_globals = true;
                }
                global = Some(name.clone());
            }
            Binding::Field(..) => {}
        }
        self.out.names.insert(pos_of(token), binding);
        global
    }

    /// A global assigned here; the first mention of a global being an
    /// assignment at the outermost block makes it initialized.
    pub(super) fn global_write(&mut self, name: String) {
        *self.out.global_writes.entry(name.clone()).or_insert(0) += 1;
        let top = self.current() == CHUNK && self.frames[0].blocks.len() == 1;
        if top && !self.out.mentioned.contains(&name) {
            self.out.globals_initialized.insert(name.clone());
        }
        self.out.mentioned.insert(name);
    }
}
