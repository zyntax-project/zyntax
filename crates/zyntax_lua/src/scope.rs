//! Name resolution: which variable each name is, which function each
//! variable belongs to, and what a function captures.
//!
//! Lua's scoping is lexical and block-structured: a `local` is visible
//! from the statement after it to the end of its block, a `local
//! function` inside its own body too, and a name that resolves to no
//! local is a global. One walk over the chunk settles every name; the
//! results are keyed by the byte offset of the name's token, which is
//! how the lowering finds them again.

use std::collections::{BTreeMap, HashMap, HashSet};

use full_moon::ast::{self, Block, Expression, FunctionBody, Prefix, Stmt, Suffix, Var};
use full_moon::tokenizer::TokenReference;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct VarId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FuncId(pub u32);

/// The chunk itself, the function every other one is nested in.
pub const CHUNK: FuncId = FuncId(0);

#[derive(Clone, Debug)]
pub struct VarInfo {
    pub name: String,
    /// The function declaring it.
    pub func: FuncId,
    /// Referenced from a function nested in the declaring one.
    pub captured: bool,
    /// Assigned anywhere other than its declaration.
    pub assigned: bool,
    /// Declared in the chunk's outermost block, so it lives as long as
    /// the program: a module variable rather than a local, once
    /// captured.
    pub outermost: bool,
    /// Declared by `local function`.
    pub is_function: bool,
    pub attribute: Option<String>,
    /// A `<const>` local whose value is known where it is declared: Lua
    /// keeps no slot for it, so the debug library sees it neither as a
    /// local nor as an upvalue.
    pub folded: bool,
    /// A `local function` whose body refers to itself by value: its
    /// record cannot hold a copy of itself, so it goes through a cell.
    pub self_captured: bool,
    /// What its declaration assigns it, as far as that tells a function.
    pub init: Init,
}

/// The value a `local` declaration gives a variable, when it is known
/// to be a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Init {
    /// Nothing known: no value, or one of another kind.
    Other,
    /// This function: `local function f`, or a function expression.
    Function(FuncId),
    /// The first result of calling the name whose token starts at this
    /// byte offset, with the arguments it is given.
    Call(usize),
}

impl VarInfo {
    /// A captured variable that is written after its declaration must
    /// be shared through a cell; one that is not can be copied into
    /// every closure that captures it.
    pub fn needs_cell(&self) -> bool {
        self.captured
            && (self.assigned || self.self_captured)
            && !(self.outermost && self.func == CHUNK)
    }

    /// A chunk-level variable every function can reach directly.
    pub fn is_module_var(&self) -> bool {
        self.outermost && self.func == CHUNK && self.captured
    }
}

#[derive(Clone, Debug)]
pub struct FuncInfo {
    pub params: Vec<VarId>,
    pub is_vararg: bool,
    /// Variables of enclosing functions this one (or one nested in it)
    /// reads or writes, excluding module variables. In a stable order.
    pub captures: Vec<VarId>,
    /// Its upvalues as Lua numbers them: every variable of an enclosing
    /// function it or a nested function reaches, module variables
    /// included, and the environment when it reaches a global, in the
    /// order of their first mention.
    pub upvalues: Vec<Upvalue>,
    /// The line the function starts on, for naming.
    pub line: usize,
    /// What the function is called, for symbol names.
    pub name: String,
    /// Whether the function is used as a value anywhere, so it needs a
    /// record besides its typed entry.
    pub escapes: bool,
    /// Declared at the chunk's outermost block as `function f` or
    /// `local function f`, so it can be a top-level function.
    pub top_level: bool,
    /// Every way out of its body returns a function expression as its
    /// first value: a call to it that returns yields a function.
    pub returns_function: bool,
}

/// An upvalue of a function: a variable of an enclosing function, or
/// the environment globals are read through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Upvalue {
    Var(VarId),
    Env,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Binding {
    Local(VarId),
    Upvalue(VarId),
    Global(String),
    /// A free name in the scope of a variable named `_ENV`: the field
    /// of that variable, whatever it holds; the flag says the variable
    /// is an upvalue at this use.
    Field(VarId, String, bool),
}

#[derive(Default, Debug)]
pub struct Scopes {
    pub vars: Vec<VarInfo>,
    pub funcs: Vec<FuncInfo>,
    /// Every name occurrence, by the byte offset of its token.
    pub names: HashMap<usize, Binding>,
    /// Every declaring token, by byte offset.
    pub decls: HashMap<usize, VarId>,
    /// Every function body, by the byte offset of its parameter list.
    pub func_at: HashMap<usize, FuncId>,
    /// How many times each global is assigned, anywhere.
    pub global_writes: BTreeMap<String, usize>,
    /// Globals declared by `function name()` at the chunk's outermost
    /// block, with the function.
    pub global_functions: HashMap<String, FuncId>,
    /// Every global read or written, so each has a module variable.
    pub globals: HashSet<String>,
    /// Local functions: the variable a `local function` declares, and
    /// the function.
    pub local_functions: HashMap<VarId, FuncId>,
    /// Each label's number, by the byte offset of its token; unique
    /// within the chunk, so two blocks may each have a `::done::`.
    pub labels: HashMap<usize, u32>,
    /// How many blocks enclose each label within its function, the
    /// function's body counting as one: a `goto` from deeper leaves
    /// the blocks between.
    pub label_depths: HashMap<u32, usize>,
    /// The label each `goto` jumps to, by the byte offset of its token:
    /// the nearest enclosing block's label of that name. A `goto` with
    /// no entry names a label that is not visible.
    pub gotos: HashMap<usize, u32>,
    next_label: u32,
    /// Globals whose first mention in the chunk, textually, is an
    /// assignment at the outermost block: never read as nil, so their
    /// type is the join of what is assigned to them.
    pub globals_initialized: HashSet<String>,
    /// Globals mentioned so far, in textual order, for the above.
    mentioned: HashSet<String>,
    /// Whether `_G` or `_ENV` is used as a value anywhere: then the
    /// globals live in a real table and every global is dynamic.
    pub dynamic_globals: bool,
    /// Whether the chunk runs code it does not contain: `load`,
    /// `loadfile`, `dofile`, or `require` of a name that is not a
    /// literal; or may load C code, which reaches the globals through
    /// the globals table: `package.loadlib`, `package.searchers`,
    /// `package` or `require` as a value. What that code defines is not
    /// known here.
    pub dynamic_code: bool,
    /// Whether a metatable may be set where the types do not follow:
    /// `setmetatable` or the `debug` library reached as a value, not
    /// called by name, or the globals table or `package` used, from
    /// which either can be fetched. Set by the program's assembly too
    /// when another of its files may set one.
    pub unseen_metatables: bool,
    /// Whether `#` on a table may be whatever a `__len` metamethod
    /// returns, of any type: some source of the program names one, or
    /// code loaded while it runs might. Set by the program's assembly
    /// over every chunk; off, `#` on a table is an integer.
    pub len_meta: bool,
    /// The names `require` is called with as string literals, in order
    /// of appearance: the files the program is made of besides its
    /// main one.
    pub requires: Vec<String>,
    /// Whether the chunk's outermost block is lowered as several
    /// functions run in sequence, so no one function is as long as a
    /// whole test file: every outermost local is then a module
    /// variable, reachable from any segment.
    pub split_chunk: bool,
    /// Whether the chunk reaches the `debug` library.
    pub debug: bool,
    /// Whether the `debug` library may rebind a variable the types do
    /// not see: it is reached as a value, or `setupvalue`,
    /// `upvaluejoin` or `setlocal` is called. Every captured variable
    /// is then dynamic and shared through a cell.
    pub debug_rebinds: bool,
    /// Whether `debug.setlocal` may be called: every local is then
    /// dynamic.
    pub debug_setlocal: bool,
    /// Whether `debug.getlocal` may be called.
    pub debug_getlocal: bool,
}

/// Outermost statements per chunk segment, and the count past which a
/// chunk is split at all.
pub const SEGMENT_STATEMENTS: usize = 150;

impl Scopes {
    pub fn var(&self, id: VarId) -> &VarInfo {
        &self.vars[id.0 as usize]
    }
    pub fn func(&self, id: FuncId) -> &FuncInfo {
        &self.funcs[id.0 as usize]
    }
    pub fn binding(&self, token: &TokenReference) -> Option<&Binding> {
        self.names.get(&pos_of(token))
    }
    pub fn declared(&self, token: &TokenReference) -> VarId {
        self.decls[&pos_of(token)]
    }
    pub fn function_of(&self, body: &FunctionBody) -> FuncId {
        self.func_at[&body_pos(body)]
    }

    /// The function a global name is, when it is declared once by a
    /// top-level `function name()` and assigned nowhere else, no global
    /// can be reached through the globals table, and a library value of
    /// the name is never read before the declaration.
    pub fn known_global_function(&self, name: &str) -> Option<FuncId> {
        if self.dynamic_globals
            || (crate::types::preset_global(name) && !self.globals_initialized.contains(name))
        {
            return None;
        }
        let f = *self.global_functions.get(name)?;
        (self.global_writes.get(name).copied().unwrap_or(0) == 1).then_some(f)
    }

    /// Whether a variable holds a function from its declaration on and
    /// no statement assigns it again: every read of it is a function,
    /// never nil, since a name is only visible once its declaration has
    /// run (a `local function` inside its own body as well). Which
    /// function it is, the types say.
    pub fn always_function(&self, v: VarId) -> bool {
        let info = self.var(v);
        if info.assigned {
            return false;
        }
        match info.init {
            Init::Other => false,
            Init::Function(_) => true,
            Init::Call(callee) => {
                let f = match self.names.get(&callee) {
                    Some(Binding::Local(u) | Binding::Upvalue(u)) => match self.var(*u).init {
                        Init::Function(f) if !self.var(*u).assigned => f,
                        _ => return false,
                    },
                    Some(Binding::Global(name)) => match self.known_global_function(name) {
                        Some(f) => f,
                        None => return false,
                    },
                    _ => return false,
                };
                self.func(f).returns_function
            }
        }
    }

    /// Whether a name is the globals table: `_G` or `_ENV`, neither
    /// shadowed by a local.
    pub fn is_globals_name(name: &str) -> bool {
        name == "_G" || name == "_ENV"
    }
}

pub fn pos_of(token: &TokenReference) -> usize {
    token.token().start_position().bytes()
}

pub fn body_pos(body: &FunctionBody) -> usize {
    pos_of(body.parameters_parentheses().tokens().0)
}

fn name_of(token: &TokenReference) -> String {
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::Identifier { identifier } => identifier.to_string(),
        _ => token.token().to_string(),
    }
}

/// The global `_G.name` or `_G["name"]` names (`_ENV` as well), when
/// `_G` is the global of that name and nothing shadows it.
fn global_table_member(w: &Walker, v: &ast::VarExpression) -> Option<String> {
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
fn metatable_call(name: &str, suffixes: &[&Suffix]) -> bool {
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
fn required_name(args: &ast::FunctionArgs) -> Option<String> {
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

/// One function being walked: its block scopes, innermost last.
struct Frame {
    id: FuncId,
    blocks: Vec<HashMap<String, VarId>>,
    /// The labels of each open block, innermost last; every label of a
    /// block is known before its statements are walked, since a `goto`
    /// may jump forward.
    labels: Vec<HashMap<String, u32>>,
}

struct Walker {
    out: Scopes,
    frames: Vec<Frame>,
    /// The name tokens, by offset, that are the callee of a call the
    /// types follow as a metatable function (`setmetatable(t, m)`,
    /// `debug.setmetatable(t, m)`): not a use of it as a value.
    metatable_callees: HashSet<usize>,
    /// The name tokens, by offset, of `debug` called as
    /// `debug.name(...)`.
    debug_calls: HashSet<usize>,
    /// The name tokens, by offset, of `_G`, `_ENV` or `package` indexed
    /// by a literal key that is not the way to the `debug` library.
    tables_indexed: HashSet<usize>,
    /// The locals declared as `require "debug"`: the library by another
    /// name.
    debug_aliases: HashSet<VarId>,
    /// Whether the expression being walked initializes such a local.
    requiring_debug: bool,
    /// The name tokens, by offset, of `require` called by name.
    require_calls: HashSet<usize>,
    /// Whether the chunk's outermost locals are kept as any function's
    /// locals rather than module variables.
    no_module_vars: bool,
}

pub fn resolve(ast: &ast::Ast) -> Scopes {
    let scopes = walk(ast, false);
    // A module variable is one variable for every function reaching
    // it, which `debug.upvaluejoin` cannot rebind for one closure, and
    // which a frame's spilled locals cannot hold by reference: a chunk
    // that may join upvalues or read or set locals captures its
    // outermost locals as any function's, unless it runs as segments,
    // which share them.
    let reaches = scopes.debug_rebinds || scopes.debug_setlocal || scopes.debug_getlocal;
    if reaches && !scopes.split_chunk && scopes.vars.iter().any(|v| v.is_module_var()) {
        return walk(ast, true);
    }
    scopes
}

fn walk(ast: &ast::Ast, no_module_vars: bool) -> Scopes {
    let mut w = Walker {
        out: Scopes::default(),
        frames: Vec::new(),
        metatable_callees: HashSet::new(),
        debug_calls: HashSet::new(),
        tables_indexed: HashSet::new(),
        debug_aliases: HashSet::new(),
        requiring_debug: false,
        require_calls: HashSet::new(),
        no_module_vars,
    };
    w.out.funcs.push(FuncInfo {
        params: Vec::new(),
        is_vararg: true,
        captures: Vec::new(),
        upvalues: vec![Upvalue::Env],
        line: 0,
        name: "main".to_string(),
        escapes: false,
        top_level: true,
        returns_function: false,
    });
    w.frames.push(Frame {
        id: CHUNK,
        blocks: vec![HashMap::new()],
        labels: Vec::new(),
    });
    w.stmts(ast.nodes(), true);
    w.frames.pop();
    // A local function capturing itself.
    let self_captures: Vec<VarId> = w
        .out
        .local_functions
        .iter()
        .filter(|(var, func)| w.out.func(**func).captures.contains(var))
        .map(|(var, _)| *var)
        .collect();
    for var in self_captures {
        w.out.vars[var.0 as usize].self_captured = true;
    }
    // A long chunk runs as segments; its outermost locals are module
    // variables, as a captured one already is.
    if ast.nodes().stmts().count() > SEGMENT_STATEMENTS {
        w.out.split_chunk = true;
        for v in &mut w.out.vars {
            if v.outermost && v.func == CHUNK {
                v.captured = true;
            }
        }
    }
    // A variable the debug library may rebind is written where no
    // statement shows it: shared through a cell, never known to hold
    // one function.
    if w.out.debug_rebinds || w.out.debug_setlocal {
        let all = w.out.debug_setlocal;
        for v in &mut w.out.vars {
            if all || v.captured {
                v.assigned = true;
            }
        }
    }
    w.out
}

impl Walker {
    fn frame(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("a function being walked")
    }
    fn current(&self) -> FuncId {
        self.frames.last().expect("a function being walked").id
    }

    fn declare(&mut self, token: &TokenReference, attribute: Option<String>) -> VarId {
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
    fn lookup(&mut self, name: &str) -> Binding {
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

    fn use_global(&mut self, binding: &Binding) {
        if let Binding::Global(name) = binding {
            self.out.mentioned.insert(name.clone());
        }
    }

    fn use_name(&mut self, token: &TokenReference) -> Binding {
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
    fn note_table_index(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
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
    fn visible_local(&self, name: &str) -> Option<VarId> {
        self.frames
            .iter()
            .rev()
            .flat_map(|frame| frame.blocks.iter().rev())
            .find_map(|block| block.get(name).copied())
    }

    /// Whether `e` is `require "debug"`, `require` being the global.
    fn requires_debug(&self, e: &Expression) -> bool {
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
    fn debug_value(&mut self) {
        self.out.debug = true;
        self.out.debug_rebinds = true;
        self.out.debug_setlocal = true;
        self.out.debug_getlocal = true;
    }

    /// `debug.name(...)`: a call of one of the library's functions by
    /// name, which is no use of the library as a value.
    fn note_debug_call(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
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
                _ => {}
            }
        }
    }

    /// The global `name` taken as a value: when it is a way to set a
    /// metatable, metatables may be set where the types do not see.
    fn global_value(&mut self, name: &str) {
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
    fn note_metatable_callee(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        if let Prefix::Name(callee) = prefix
            && metatable_call(&name_of(callee), suffixes)
        {
            self.metatable_callees.insert(pos_of(callee));
        }
    }

    /// Whether a local of this name is in scope.
    fn shadowed(&self, name: &str) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.blocks.iter().any(|block| block.contains_key(name)))
    }

    /// The name `token` as an assignment's target: the global it
    /// writes, if any, which the caller records once the assigned
    /// values are walked.
    fn assign_name(&mut self, token: &TokenReference) -> Option<String> {
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
    fn global_write(&mut self, name: String) {
        *self.out.global_writes.entry(name.clone()).or_insert(0) += 1;
        let top = self.current() == CHUNK && self.frames[0].blocks.len() == 1;
        if top && !self.out.mentioned.contains(&name) {
            self.out.globals_initialized.insert(name.clone());
        }
        self.out.mentioned.insert(name);
    }

    fn block(&mut self, block: &Block) {
        self.frame().blocks.push(HashMap::new());
        self.stmts(block, false);
        self.frame().blocks.pop();
    }

    fn stmts(&mut self, block: &Block, _top: bool) {
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

    fn stmt(&mut self, stmt: &Stmt) {
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

    /// A function body: a new frame with its parameters declared.
    fn function(&mut self, body: &FunctionBody, is_method: bool, name: String) -> FuncId {
        let id = FuncId(self.out.funcs.len() as u32);
        let pos = body_pos(body);
        let line = body
            .parameters_parentheses()
            .tokens()
            .0
            .token()
            .start_position()
            .line();
        self.out.funcs.push(FuncInfo {
            params: Vec::new(),
            is_vararg: false,
            captures: Vec::new(),
            upvalues: Vec::new(),
            line,
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
    fn is_constant(&self, e: &Expression) -> bool {
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
    fn folded(&self, token: &TokenReference) -> bool {
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
    fn init_of(&self, e: &Expression) -> Init {
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

    fn var_expression(&mut self, v: &ast::VarExpression) {
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
    fn note_require(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
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

    fn call(&mut self, c: &ast::FunctionCall) {
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

    fn prefix(&mut self, p: &Prefix) {
        match p {
            Prefix::Name(token) => {
                self.use_name(token);
            }
            Prefix::Expression(e) => self.expr(e),
            _ => {}
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
            ast::FunctionArgs::String(_) => {}
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

    fn expr(&mut self, e: &Expression) {
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

/// Whether every `return` in a block, outside nested functions, has a
/// function expression as its first value.
fn returns_functions(block: &Block) -> bool {
    let first_is_function = |r: &ast::Return| {
        let mut e = r.returns().iter().next();
        while let Some(Expression::Parentheses { expression, .. }) = e {
            e = Some(expression);
        }
        matches!(e, Some(Expression::Function(_)))
    };
    if let Some(ast::LastStmt::Return(r)) = block.last_stmt()
        && !first_is_function(r)
    {
        return false;
    }
    block.stmts().all(|stmt| match stmt {
        Stmt::Do(d) => returns_functions(d.block()),
        Stmt::If(i) => {
            returns_functions(i.block())
                && i.else_if()
                    .is_none_or(|e| e.iter().all(|e| returns_functions(e.block())))
                && i.else_block().is_none_or(returns_functions)
        }
        Stmt::While(w) => returns_functions(w.block()),
        Stmt::Repeat(r) => returns_functions(r.block()),
        Stmt::NumericFor(f) => returns_functions(f.block()),
        Stmt::GenericFor(f) => returns_functions(f.block()),
        _ => true,
    })
}

#[cfg(test)]
mod tests {
    fn unseen(source: &str) -> bool {
        let ast = full_moon::parse_fallible(source, full_moon::LuaVersion::lua54())
            .into_result()
            .expect("parses");
        super::resolve(&ast).unseen_metatables
    }

    #[test]
    fn metatable_calls_by_name_are_followed() {
        for source in [
            "local t = setmetatable({}, {})",
            "setmetatable(t, mt)",
            "debug.setmetatable(t, mt)",
            "debug.getmetatable(t).x = 1",
            "print(debug.traceback())",
            "local m = require \"lib.mod\"",
            "local m = require(\"lib.mod\")",
            "local f = require(\"lib.mod\").f",
            "require(\"lib.mod\").run()",
            "local _G = {}; _G.x = setmetatable({}, {})",
        ] {
            assert!(!unseen(source), "{source}");
        }
    }

    #[test]
    fn metatable_functions_taken_as_values_are_not() {
        for source in [
            "local sm = setmetatable",
            "pcall(setmetatable, t, mt)",
            "(setmetatable)(t, mt)",
            "local x = setmetatable(t, mt).f",
            "setmetatable(t, mt):m()",
            "local d = debug",
            "debug.setmetatable(t, mt).f = 1",
            "debug.getupvalue(f, 1)",
            "local f = _G.setmetatable",
            "local f = rawget(_G, \"setmetatable\")",
            "local g = _G",
            "local d = package.loaded.debug",
            "local d = require \"debug\"",
            "local f = require(\"debug\").setmetatable",
            "local r = require",
        ] {
            assert!(unseen(source), "{source}");
        }
    }

    fn scopes(source: &str) -> super::Scopes {
        let ast = full_moon::parse_fallible(source, full_moon::LuaVersion::lua54())
            .into_result()
            .expect("parses");
        super::resolve(&ast)
    }

    #[test]
    fn debug_calls_by_name_rebind_only_what_they_name() {
        for source in [
            "print(debug.traceback())",
            "debug.getinfo(1)",
            "local d = require \"debug\"; d.getinfo(1)",
        ] {
            let s = scopes(source);
            assert!(s.debug, "{source}");
            assert!(!s.debug_rebinds && !s.debug_setlocal, "{source}");
        }
        assert!(scopes("debug.setupvalue(f, 1, 2)").debug_rebinds);
        assert!(scopes("debug.upvaluejoin(f, 1, g, 1)").debug_rebinds);
        assert!(scopes("debug.setlocal(1, 1, 2)").debug_setlocal);
        assert!(scopes("debug.getlocal(1, 1)").debug_getlocal);
        assert!(!scopes("local debug = {}; debug.traceback()").debug);
        assert!(!scopes("print(1)").debug);
        for source in [
            "print(_G.x, _G[\"y\"])",
            "_G.x = 1",
            "print(package.path, package.loaded.string)",
            "local _ENV = {print = print}; print(1)",
        ] {
            assert!(!scopes(source).debug, "{source}");
        }
    }

    #[test]
    fn the_debug_library_as_a_value_may_rebind_anything() {
        for source in [
            "local d = debug",
            "local f = debug.setupvalue",
            "local d = require \"debug\"; local e = d",
            "local d = require \"debug\"; d = nil",
            "print(require \"debug\")",
            "local x = _G.debug",
            "print(_G.debug.getinfo(1))",
            "print(_ENV[\"debug\"].traceback())",
            "local d = _G[\"deb\" .. \"ug\"]",
            "for k, v in pairs(_G) do end",
            "print(package.loaded.debug)",
            "local m = package.loaded[name]",
            "local m = require(name)",
        ] {
            let s = scopes(source);
            assert!(s.debug_rebinds && s.debug_setlocal, "{source}");
        }
    }

    #[test]
    fn upvalues_are_numbered_by_first_mention() {
        let s = scopes("local a, b; local function f() print(b); return a end");
        let f = s.funcs.iter().find(|f| f.name == "f").expect("f");
        let names: Vec<String> = f
            .upvalues
            .iter()
            .map(|u| match u {
                super::Upvalue::Var(v) => s.var(*v).name.clone(),
                super::Upvalue::Env => "_ENV".to_string(),
            })
            .collect();
        assert_eq!(names, ["_ENV", "b", "a"]);
    }

    #[test]
    fn assignment_targets_are_captured_before_values() {
        let s = scopes("local a, b, t; local function k() a = b; t[b] = a end");
        let k = s.funcs.iter().find(|f| f.name == "k").expect("k");
        let names: Vec<String> = k
            .upvalues
            .iter()
            .filter_map(|u| match u {
                super::Upvalue::Var(v) => Some(s.var(*v).name.clone()),
                super::Upvalue::Env => None,
            })
            .collect();
        assert_eq!(names, ["a", "b", "t"]);
    }

    #[test]
    fn a_chunk_that_may_join_upvalues_has_no_module_variables() {
        let joins =
            scopes("local a = 1; local function f() return a end; debug.upvaluejoin(f, 1, f, 1)");
        assert!(joins.vars.iter().all(|v| !v.is_module_var()));
        let f = joins.funcs.iter().find(|f| f.name == "f").expect("f");
        assert!(!f.top_level && !f.captures.is_empty());
        let plain = scopes("local a = 1; local function f() return a end; print(f())");
        assert!(plain.vars.iter().any(|v| v.is_module_var()));
    }
}
