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
    /// A `local function` whose body refers to itself by value: its
    /// record cannot hold a copy of itself, so it goes through a cell.
    pub self_captured: bool,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    Local(VarId),
    Upvalue(VarId),
    Global(String),
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
    /// Label names by the byte offset of the `goto` referring to them.
    pub gotos: HashMap<usize, String>,
    /// Globals whose first mention in the chunk, textually, is an
    /// assignment at the outermost block: never read as nil, so their
    /// type is the join of what is assigned to them.
    pub globals_initialized: HashSet<String>,
    /// Globals mentioned so far, in textual order, for the above.
    mentioned: HashSet<String>,
    /// Whether `_G` or `_ENV` is used as a value anywhere: then the
    /// globals live in a real table and every global is dynamic.
    pub dynamic_globals: bool,
    /// Whether the chunk's outermost block is lowered as several
    /// functions run in sequence, so no one function is as long as a
    /// whole test file: every outermost local is then a module
    /// variable, reachable from any segment.
    pub split_chunk: bool,
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
    /// top-level `function name()` and assigned nowhere else, and no
    /// global can be reached through the globals table.
    pub fn known_global_function(&self, name: &str) -> Option<FuncId> {
        if self.dynamic_globals {
            return None;
        }
        let f = *self.global_functions.get(name)?;
        (self.global_writes.get(name).copied().unwrap_or(0) == 1).then_some(f)
    }

    /// Whether a name is the globals table: `_G` or `_ENV`, neither
    /// shadowed by a local.
    pub fn is_globals_name(name: &str) -> bool {
        name == "_G" || name == "_ENV"
    }

    /// The function a local variable is, when `local function`
    /// declared it and nothing assigns it again.
    pub fn known_local_function(&self, var: VarId) -> Option<FuncId> {
        let f = *self.local_functions.get(&var)?;
        (!self.var(var).assigned).then_some(f)
    }

    /// The function a name occurrence calls, if it is a known one.
    pub fn known_function(&self, binding: &Binding) -> Option<FuncId> {
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => self.known_local_function(*v),
            Binding::Global(name) => self.known_global_function(name),
        }
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
    if !Scopes::is_globals_name(&g) || w.shadowed(&g) {
        return None;
    }
    let suffixes: Vec<&Suffix> = v.suffixes().collect();
    match suffixes.as_slice() {
        [Suffix::Index(ast::Index::Dot { name, .. })] => Some(name_of(name)),
        [Suffix::Index(ast::Index::Brackets { expression, .. })] => literal_string(expression),
        _ => None,
    }
}

/// The text of a string literal expression.
pub fn literal_string(e: &Expression) -> Option<String> {
    let Expression::String(token) = e else {
        return None;
    };
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::StringLiteral { literal, .. } => Some(literal.to_string()),
        _ => None,
    }
}

/// One function being walked: its block scopes, innermost last.
struct Frame {
    id: FuncId,
    blocks: Vec<HashMap<String, VarId>>,
}

struct Walker {
    out: Scopes,
    frames: Vec<Frame>,
    /// Occurrences of known-function names used as values, resolved
    /// after the walk once assignment counts are known.
    value_uses: Vec<Binding>,
}

pub fn resolve(ast: &ast::Ast) -> Scopes {
    let mut w = Walker {
        out: Scopes::default(),
        frames: Vec::new(),
        value_uses: Vec::new(),
    };
    w.out.funcs.push(FuncInfo {
        params: Vec::new(),
        is_vararg: true,
        captures: Vec::new(),
        line: 0,
        name: "main".to_string(),
        escapes: false,
        top_level: true,
    });
    w.frames.push(Frame {
        id: CHUNK,
        blocks: vec![HashMap::new()],
    });
    w.stmts(ast.nodes(), true);
    w.frames.pop();
    // A known function used as a value escapes.
    let uses = std::mem::take(&mut w.value_uses);
    for binding in uses {
        if let Some(f) = w.out.known_function(&binding) {
            w.out.funcs[f.0 as usize].escapes = true;
        }
    }
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
        let outermost = func == CHUNK && self.frame().blocks.len() == 1;
        let id = VarId(self.out.vars.len() as u32);
        self.out.vars.push(VarInfo {
            name: name.clone(),
            func,
            captured: false,
            assigned: false,
            outermost,
            is_function: false,
            attribute,
            self_captured: false,
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
        // `_ENV` not declared as a local is the globals table.
        let name = if name == "_ENV" { "_G" } else { name };
        self.out.globals.insert(name.to_string());
        Binding::Global(name.to_string())
    }

    fn use_global(&mut self, binding: &Binding) {
        if let Binding::Global(name) = binding {
            self.out.mentioned.insert(name.clone());
        }
    }

    fn use_name(&mut self, token: &TokenReference, as_callee: bool) -> Binding {
        let binding = self.lookup(&name_of(token));
        self.use_global(&binding);
        // The globals table reached as a value: every global is then
        // an entry of a real table.
        if let Binding::Global(name) = &binding
            && Scopes::is_globals_name(name)
        {
            self.out.dynamic_globals = true;
        }
        self.out.names.insert(pos_of(token), binding.clone());
        if !as_callee {
            self.value_uses.push(binding.clone());
        }
        binding
    }

    /// Whether a local of this name is in scope.
    fn shadowed(&self, name: &str) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.blocks.iter().any(|block| block.contains_key(name)))
    }

    fn assign_name(&mut self, token: &TokenReference) {
        let binding = self.lookup(&name_of(token));
        match &binding {
            Binding::Local(id) | Binding::Upvalue(id) => {
                self.out.vars[id.0 as usize].assigned = true;
            }
            Binding::Global(name) => {
                self.global_write(name.clone());
            }
        }
        self.out.names.insert(pos_of(token), binding);
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
        for stmt in block.stmts() {
            self.stmt(stmt);
        }
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
                for e in a.expressions() {
                    self.expr(e);
                }
                for target in a.variables() {
                    match target {
                        Var::Name(token) => self.assign_name(token),
                        Var::Expression(v) => match global_table_member(self, v) {
                            Some(name) => {
                                if let Prefix::Name(token) = v.prefix() {
                                    self.out
                                        .names
                                        .insert(pos_of(token), Binding::Global("_G".to_string()));
                                }
                                self.global_write(name);
                            }
                            None => self.var_expression(v),
                        },
                        _ => {}
                    }
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
                    }
                    self.out.names.insert(pos_of(token), binding.clone());
                    let id = self.function(f.body(), is_method, fname.clone());
                    if let Binding::Global(name) = &binding
                        && top
                        && !self.out.global_functions.contains_key(name)
                    {
                        self.out.global_functions.insert(name.clone(), id);
                        self.out.funcs[id.0 as usize].top_level = true;
                    }
                } else {
                    // `function a.b.c()`: `a` is read, the rest indexed.
                    self.use_name(names[0], false);
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
                for e in l.expressions() {
                    self.expr(e);
                }
                let attributes: Vec<Option<String>> = l
                    .attributes()
                    .map(|a| a.map(|a| name_of(a.name())))
                    .collect();
                for (i, name) in l.names().iter().enumerate() {
                    let attribute = attributes.get(i).cloned().flatten();
                    self.declare(name, attribute);
                }
            }
            Stmt::LocalFunction(f) => {
                let var = self.declare(f.name(), None);
                self.out.vars[var.0 as usize].is_function = true;
                let id = self.function(f.body(), false, name_of(f.name()));
                self.out.local_functions.insert(var, id);
                let top = self.current() == CHUNK && self.frame().blocks.len() == 1;
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
                self.out
                    .gotos
                    .insert(pos_of(g.goto_token()), name_of(g.label_name()));
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
            line,
            name,
            escapes: false,
            top_level: false,
        });
        self.out.func_at.insert(pos, id);
        self.frames.push(Frame {
            id,
            blocks: vec![HashMap::new()],
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
                self_captured: false,
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
        id
    }

    fn var_expression(&mut self, v: &ast::VarExpression) {
        if let Some(name) = global_table_member(self, v) {
            // `_G.name` is the global `name`; the table's own name is
            // bound without being a value.
            if let Prefix::Name(token) = v.prefix() {
                self.out
                    .names
                    .insert(pos_of(token), Binding::Global("_G".to_string()));
            }
            self.out.mentioned.insert(name.clone());
            self.out.globals.insert(name);
            return;
        }
        self.prefix(v.prefix(), false);
        for s in v.suffixes() {
            self.suffix(s);
        }
    }

    fn call(&mut self, c: &ast::FunctionCall) {
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        let callee_first = matches!(suffixes.first(), Some(Suffix::Call(_)));
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
            self.use_name(callee, true);
            let binding = self.lookup(&name_of(g));
            self.out.names.insert(pos_of(g), binding);
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
        self.prefix(c.prefix(), callee_first);
        for s in suffixes {
            self.suffix(s);
        }
    }

    fn prefix(&mut self, p: &Prefix, as_callee: bool) {
        match p {
            Prefix::Name(token) => {
                self.use_name(token, as_callee);
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
            // A function expression is a value from the start: its
            // calls are never known.
            Expression::Function(f) => {
                let id = self.function(f.body(), false, String::new());
                self.out.funcs[id.0 as usize].escapes = true;
            }
            Expression::FunctionCall(c) => self.call(c),
            Expression::TableConstructor(t) => self.table(t),
            Expression::Number(_) | Expression::String(_) | Expression::Symbol(_) => {}
            Expression::Var(v) => match v {
                Var::Name(token) => {
                    self.use_name(token, false);
                }
                Var::Expression(v) => self.var_expression(v),
                _ => {}
            },
            _ => {}
        }
    }
}
