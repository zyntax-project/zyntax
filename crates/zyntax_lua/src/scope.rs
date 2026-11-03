//! Name resolution: which variable each name is, which function each
//! variable belongs to, and what a function captures.
//!
//! Lua's scoping is lexical and block-structured: a `local` is visible
//! from the statement after it to the end of its block, a `local
//! function` inside its own body too, and a name that resolves to no
//! local is a global. One walk over the chunk settles every name; the
//! results are keyed by the byte offset of the name's token, which is
//! how the lowering finds them again.

mod expressions;
mod names;
mod resolve;
mod statements;

pub(crate) use names::*;
pub(crate) use resolve::*;

use std::collections::{BTreeMap, HashMap, HashSet};

use full_moon::ast::{self, Block, Expression, FunctionBody, Prefix, Stmt, Suffix, Var};
use full_moon::node::Node;
use full_moon::tokenizer::TokenReference;

use crate::annotation::{Annotation, LuaType, Returned};

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
    /// Its body's bytes in the source: from its parameter list's `(`
    /// to the end of its `end`. Empty for the main chunk.
    pub body: (usize, usize),
    /// Declared as `function a:b()`, with `self` as a first parameter
    /// the body does not spell.
    pub method: bool,
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
    /// The source bytes of each folded constant's value.
    pub const_inits: HashMap<VarId, (usize, usize)>,
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
    /// The local functions lowering leaves out: see
    /// [`Scopes::prune_unused_functions`].
    pub pruned: HashSet<VarId>,
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
    /// Whether `debug.upvalueid` is called by name: every captured
    /// variable then lives in a cell, whose address is its identity
    /// for each closure instance.
    pub upvalue_ids: bool,
    /// What the chunk's annotations declare.
    pub declared: Declared,
}

/// What a chunk's LuaLS annotations declare (see [`crate::annotation`]),
/// registered with what each annotated statement declares.
#[derive(Default, Debug)]
pub struct Declared {
    /// An annotated function's `@param` types by parameter name, and
    /// its `@return` types.
    pub funcs: HashMap<FuncId, Signature>,
    /// The `@class` a statement binding a variable or global gives it,
    /// with the `@field`s of the class's instances.
    pub classes: HashMap<Holder, Class>,
    /// The `@type` of a field a statement stores into a variable's or a
    /// global's table.
    pub fields: HashMap<(Holder, String), LuaType>,
    /// Annotations that do not fit their statement: the comment's span
    /// and why.
    pub problems: Vec<((usize, usize), String)>,
}

/// What an annotated statement binds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Holder {
    Var(VarId),
    Global(String),
}

#[derive(Clone, Debug, Default)]
pub struct Signature {
    pub params: Vec<(String, LuaType)>,
    pub returns: Vec<Returned>,
}

#[derive(Clone, Debug)]
pub struct Class {
    pub name: String,
    pub fields: Vec<(String, LuaType)>,
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

    /// Mark the local functions no code that can run reaches, for
    /// lowering to leave out: declared by `local function`, never
    /// assigned, and named only inside functions that are themselves
    /// left out or nested in one. Found as the largest such set, so two
    /// unused functions calling each other both go. Making one has no
    /// effect a program sees without the debug library, whose
    /// `getlocal` and line hooks would see the local and the statement,
    /// so a caller asks for this only when no file reaches that library.
    pub fn prune_unused_functions(&mut self) {
        let candidates: HashMap<VarId, FuncId> = self
            .local_functions
            .iter()
            .filter(|(v, _)| {
                let info = self.var(**v);
                info.is_function && !info.assigned && info.attribute.is_none()
            })
            .map(|(v, f)| (*v, *f))
            .collect();
        if candidates.is_empty() {
            return;
        }
        // Each function's parent, from how the body spans nest.
        let mut spans: Vec<(usize, usize, FuncId)> = self
            .funcs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != CHUNK.0 as usize)
            .map(|(i, f)| (f.body.0, f.body.1, FuncId(i as u32)))
            .collect();
        spans.sort();
        let mut parent: HashMap<FuncId, FuncId> = HashMap::new();
        let mut open: Vec<(usize, FuncId)> = Vec::new();
        for &(start, end, id) in &spans {
            while open.last().is_some_and(|(e, _)| *e < start) {
                open.pop();
            }
            parent.insert(id, open.last().map_or(CHUNK, |(_, f)| *f));
            open.push((end, id));
        }
        // The innermost function holding a position: the last to start
        // before it, or the nearest of that one's parents still open.
        let innermost = |at: usize| -> FuncId {
            let i = spans.partition_point(|(start, _, _)| *start <= at);
            let Some(&(_, end, mut id)) = i.checked_sub(1).map(|i| &spans[i]) else {
                return CHUNK;
            };
            let mut end = end;
            while at > end {
                id = parent[&id];
                if id == CHUNK {
                    return CHUNK;
                }
                end = self.func(id).body.1;
            }
            id
        };
        let mut uses: HashMap<VarId, Vec<FuncId>> = HashMap::new();
        for (at, binding) in &self.names {
            let v = match binding {
                Binding::Local(v) | Binding::Upvalue(v) | Binding::Field(v, ..) => *v,
                Binding::Global(_) => continue,
            };
            if candidates.contains_key(&v) {
                uses.entry(v).or_default().push(innermost(*at));
            }
        }
        let mut left_out: HashSet<FuncId> = candidates.values().copied().collect();
        let runs = |f: FuncId, left_out: &HashSet<FuncId>| {
            let mut f = f;
            loop {
                if left_out.contains(&f) {
                    return false;
                }
                if f == CHUNK {
                    return true;
                }
                f = parent[&f];
            }
        };
        loop {
            let reached: Vec<VarId> = candidates
                .iter()
                .filter(|(v, f)| {
                    left_out.contains(f)
                        && uses
                            .get(v)
                            .is_some_and(|at| at.iter().any(|f| runs(*f, &left_out)))
                })
                .map(|(v, _)| *v)
                .collect();
            if reached.is_empty() {
                break;
            }
            for v in reached {
                left_out.remove(&candidates[&v]);
            }
        }
        self.pruned = candidates
            .into_iter()
            .filter(|(_, f)| left_out.contains(f))
            .map(|(v, _)| v)
            .collect();
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

    /// The local functions `prune_unused_functions` leaves out, by name.
    fn pruned(source: &str) -> Vec<String> {
        let mut s = scopes(source);
        s.prune_unused_functions();
        let mut names: Vec<String> = s.pruned.iter().map(|v| s.var(*v).name.clone()).collect();
        names.sort();
        names
    }

    #[test]
    fn local_functions_nothing_running_names_are_pruned() {
        let none: Vec<String> = Vec::new();
        assert_eq!(pruned("local function a() end"), ["a"]);
        // Named only by itself and by a function that is pruned.
        assert_eq!(
            pruned("local function a() return a() end local function b() return a() end"),
            ["a", "b"]
        );
        // Named by a function that runs, through a nested one.
        assert_eq!(
            pruned(
                "local function a() end \
                 local function b() local function c() return a() end return c() end \
                 print(b())"
            ),
            none
        );
        // Nested in a pruned function, and named only there.
        assert_eq!(
            pruned("local function h() local function i() end return i end"),
            ["h", "i"]
        );
        // A use past a nested function's end belongs to the outer one.
        assert_eq!(
            pruned(
                "local function k() end \
                 local function outer() local function inner() end return k() end \
                 print(outer())"
            ),
            ["inner"]
        );
        // The first of two with one name is never named again.
        assert_eq!(
            pruned("local function p() return 1 end local function p() return 2 end print(p())"),
            ["p"]
        );
        // Assigned after its declaration: kept.
        assert_eq!(pruned("local function a() end a = nil"), none);
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
        let ids = scopes("print(debug.upvalueid(f, 1))");
        assert!(ids.upvalue_ids && !ids.debug_rebinds && !ids.debug_setlocal);
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
