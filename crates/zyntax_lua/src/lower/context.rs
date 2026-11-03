//! Context for Lua lowering.

use super::*;

/// The layouts of a chunk's shapes: a shape with fields, and not too
/// many, that no weak metatable may reach, gets slots. With `slots`
/// off (a chunk compiled at run time, whose shapes the program's hooks
/// do not know), none does. Number slots take kind bits from bit 63
/// down while those stay above the presence bits; a Number slot left
/// without one is a Scalar slot.
pub(super) fn shape_layouts(
    inferred: &Inferred,
    chunk_index: i64,
    slots: bool,
) -> Vec<Option<ShapeLayout>> {
    let trace = std::env::var_os("ZYNTAX_TRACE_TYPES").is_some();
    inferred
        .shapes
        .iter()
        .enumerate()
        .map(|(k, info)| {
            if !slots
                || info.fields.is_empty()
                || info.fields.len() > MAX_SLOTS
                || inferred.may_be_weak(ShapeId(k as u32))
            {
                return None;
            }
            let gid = (chunk_index << 20) + k as i64 + 1;
            let mut kbits = (info.fields.len()..64).rev();
            let slots = info
                .fields
                .iter()
                .enumerate()
                .map(|(bit, (name, ty))| {
                    let mut kind = Slot::kind_of(ty.settled());
                    let kbit = match kind {
                        SlotKind::Number => kbits.next(),
                        _ => None,
                    };
                    if kind == SlotKind::Number && kbit.is_none() {
                        if trace {
                            eprintln!(
                                "[types] shape {k}: field {name} has no kind bit, a scalar slot"
                            );
                        }
                        kind = SlotKind::Scalar;
                    }
                    Slot {
                        name: name.clone(),
                        bit,
                        kind,
                        kbit,
                    }
                })
                .collect();
            Some(ShapeLayout {
                id: TypeId::next(),
                name: format!("LuaTable${gid}"),
                gid,
                slots,
            })
        })
        .collect()
}

/// The functions whose depth is bounded by the program's text: not
/// escaping, calling nothing through a value or the library, and not
/// reaching themselves through direct calls. Their depth needs no
/// count, and their depth check cannot raise.
pub(super) fn bounded_functions(
    facts: &HashMap<FuncId, RaiseFact>,
    escaping: impl Fn(FuncId) -> bool,
) -> HashSet<FuncId> {
    let mut unbounded: HashSet<FuncId> = facts
        .iter()
        .filter(|(f, fact)| fact.dynamic || escaping(**f))
        .map(|(f, _)| *f)
        .collect();
    // A function on a cycle of direct calls.
    for f in facts.keys() {
        let mut seen = HashSet::new();
        let mut todo: Vec<FuncId> = facts[f].callees.iter().copied().collect();
        while let Some(g) = todo.pop() {
            if g == *f {
                unbounded.insert(*f);
                break;
            }
            if seen.insert(g)
                && let Some(fact) = facts.get(&g)
            {
                todo.extend(fact.callees.iter().copied());
            }
        }
    }
    loop {
        let before = unbounded.len();
        for (f, fact) in facts {
            if fact.callees.iter().any(|c| unbounded.contains(c)) {
                unbounded.insert(*f);
            }
        }
        if unbounded.len() == before {
            break;
        }
    }
    facts
        .keys()
        .copied()
        .filter(|f| !unbounded.contains(f))
        .collect()
}

/// The functions that may raise: those that check for an error
/// themselves, or their depth without a bound on it, and those calling
/// one of them, and so on.
pub(super) fn raising_functions(
    facts: &HashMap<FuncId, RaiseFact>,
    bounded: &HashSet<FuncId>,
) -> HashSet<FuncId> {
    let mut raising: HashSet<FuncId> = facts
        .iter()
        .filter(|(f, fact)| fact.own || (fact.checks_depth && !bounded.contains(f)))
        .map(|(f, _)| *f)
        .collect();
    loop {
        let before = raising.len();
        for (f, fact) in facts {
            if fact.callees.iter().any(|c| raising.contains(c)) {
                raising.insert(*f);
            }
        }
        if raising.len() == before {
            return raising;
        }
    }
}

impl<'a> Module<'a> {
    pub(super) fn typer(&self) -> Typer<'_> {
        Typer {
            scopes: self.scopes,
            known: self.inferred,
        }
    }

    /// The line a span starts on, counted from one, as positions and
    /// hooks see it: no line in a stripped chunk.
    pub(super) fn line_of(&self, span: Span) -> i64 {
        let line = if self.stripped {
            library::STRIPPED_LINE
        } else {
            self.source_line(span)
        };
        (self.chunk_index << library::LINE_BITS) | line
    }

    /// The line a span starts on in the source, stripped or not.
    pub(super) fn source_line(&self, span: Span) -> i64 {
        self.line_starts
            .partition_point(|&start| start <= span.start) as i64
    }

    /// Whether a program function may raise, as far as is known.
    pub(super) fn raises(&self, f: FuncId) -> bool {
        match &self.raising {
            Some(set) => set.contains(&f),
            None => true,
        }
    }

    /// The struct of each slotted shape, declared and registered, and
    /// its layout handed on for the hooks.
    pub(super) fn declare_shapes(
        &mut self,
        declarations: &mut Vec<TypedNode<TypedDeclaration>>,
        registry: &mut zyntax_typed_ast::TypeRegistry,
        hooks: &mut Vec<ShapeLayout>,
    ) {
        for layout in std::mem::take(&mut self.layouts).into_iter().flatten() {
            let fields = self.layout_fields(&layout);
            library::declare_reference_struct(registry, layout.id, &layout.name, &fields);
            declarations.push(library::reference_struct_class(&layout.name, &fields));
            hooks.push(layout);
        }
    }

    /// Whether tables of shape `k` never get a metatable.
    pub(super) fn plain_shape(&self, k: ShapeId) -> bool {
        let info = self.inferred.shape(k);
        !info.unknown_meta && info.classes.is_empty()
    }

    /// How a shape's tables are laid out, when they have slots.
    pub(super) fn layout(&self, k: ShapeId) -> Option<&ShapeLayout> {
        self.layouts.get(k.0 as usize).and_then(|l| l.as_ref())
    }

    /// The struct type of a layout's tables, as a pointer.
    pub(super) fn shape_ty(&self, layout: &ShapeLayout) -> Type {
        library::table_ty(layout.id)
    }

    /// Every field of a layout's struct: the table header, then the
    /// slots.
    pub(super) fn layout_fields(&self, layout: &ShapeLayout) -> Vec<(String, Type)> {
        let table_ty = self.types.table();
        let mut fields = library::table_header_fields(self.types.table_type);
        for slot in &layout.slots {
            fields.extend(slot.fields(&table_ty));
        }
        fields
    }

    /// Whether a function's depth is known bounded, so it keeps no
    /// count of it. Nothing is, until a first lowering has looked.
    pub(super) fn is_bounded(&self, f: FuncId) -> bool {
        self.bounded.as_ref().is_some_and(|set| set.contains(&f))
    }

    /// Whether the program's code may run inside this lowered body
    /// before it returns: a call through a value, or into a library
    /// function that makes one.
    pub(super) fn calls_dynamically(&self, statements: &[St]) -> bool {
        let mut names = std::collections::BTreeSet::new();
        for s in statements {
            library::callee_names(s, &mut names);
        }
        let helpers = self.reentrant_helpers.borrow();
        names
            .iter()
            .any(|n| self.reentrant.contains(n.as_str()) || helpers.contains(n))
    }

    /// The IR type of a static type.
    pub(super) fn ir(&self, ty: Ty) -> Type {
        match ty {
            Ty::Bool => prim(PrimitiveType::Bool),
            Ty::Int => prim(PrimitiveType::I64),
            Ty::Float => prim(PrimitiveType::F64),
            Ty::Number | Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => number_type(),
            Ty::Str => prim(PrimitiveType::String),
            // A table of a known shape is a table, or null for nil.
            Ty::Table | Ty::Shape(_) => self.types.table(),
            // A known function is still the record every function value
            // is; what is known is where calls through it go.
            Ty::Func(_) | Ty::Nil | Ty::Any | Ty::Unknown => Type::Any,
        }
    }

    pub(super) fn anys(&self) -> Type {
        self.types.anys()
    }

    /// The symbol of a global's module variable.
    pub(super) fn global_symbol(name: &str) -> InternedString {
        intern(&format!("lua$g${name}"))
    }

    /// The symbol of a captured chunk local's module variable.
    pub(super) fn module_local_symbol(&self, v: VarId) -> InternedString {
        intern(&format!(
            "lua$l${}{}${}",
            self.tag,
            self.scopes.var(v).name,
            v.0
        ))
    }

    pub(super) fn declare_module_var(&self, name: InternedString, ty: Ty) {
        let mut vars = self.module_vars.borrow_mut();
        if !vars.iter().any(|(n, _)| *n == name) {
            vars.push((name, ty));
        }
    }

    /// The typed entry's name for a function.
    pub(super) fn entry_name(&self, f: FuncId) -> String {
        let info = self.scopes.func(f);
        if f == CHUNK {
            return if self.tag.is_empty() {
                ENTRY.to_string()
            } else {
                format!("lua${}chunk", self.tag)
            };
        }
        if info.top_level
            && let Some((name, _)) = self
                .scopes
                .global_functions
                .iter()
                .find(|(_, id)| **id == f)
        {
            return format!("lua${}{name}", self.tag);
        }
        let name = if info.name.is_empty() {
            "anon".to_string()
        } else {
            info.name.replace(['.', ':'], "$")
        };
        format!("lua${}{name}${}", self.tag, f.0)
    }

    /// The restores of the line in a body that nothing reads.
    pub(super) fn strip_line_restores(&self, statements: &mut [St]) {
        let reads_line = |name: &str| self.fallible.contains(name) || self.reentrant.contains(name);
        LineRestores::strip(statements, &reads_line);
    }

    /// Whether a function's typed entry takes its record as `env`: a
    /// nested function capturing something, which the record holds.
    pub(super) fn takes_env(&self, f: FuncId) -> bool {
        let info = self.scopes.func(f);
        !info.top_level && !info.captures.is_empty()
    }

    /// The number a function's values carry, unique across the
    /// program's chunks: the chunk's number above the low 32 bits.
    pub(super) fn func_key(&self, f: FuncId) -> i64 {
        (self.chunk_index << 32) | f.0 as i64
    }

    /// A record for the debug library's host, when the program keeps
    /// its call stack.
    pub(super) fn debug_record(&self, fields: &[String]) {
        if let Some(debug) = &self.debug {
            debug.records.borrow_mut().push(
                fields
                    .iter()
                    .map(|f| sanitize(f))
                    .collect::<Vec<_>>()
                    .join("\x1f"),
            );
        }
    }

    /// The record code's name for a function.
    pub(super) fn code_name(&self, f: FuncId) -> String {
        format!("{}$fn", self.entry_name(f))
    }

    pub(super) fn sig(&self, f: FuncId) -> types::Sig {
        self.inferred.sig(f).cloned().unwrap_or_else(|| types::Sig {
            params: vec![Ty::Nil; self.scopes.func(f).params.len()],
            returns: Returns::Fixed(Vec::new()),
        })
    }

    /// The IR return type of a function.
    pub(super) fn return_ir(&self, returns: &Returns) -> Type {
        match returns {
            Returns::Fixed(v) if v.is_empty() => prim(PrimitiveType::Unit),
            Returns::Fixed(v) if v.len() == 1 => self.ir(v[0]),
            Returns::Fixed(_) => self.anys(),
            Returns::Dynamic | Returns::Unknown => Type::Any,
        }
    }
}

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn new(m: &'m Module<'a>, func: FuncId) -> Self {
        let returns = m.sig(func).returns;
        Lowerer {
            m,
            func,
            storage: HashMap::new(),
            varargs: None,
            returns,
            temps: 0,
            bound: std::collections::HashSet::new(),
            raised: false,
            raise_callees: HashSet::new(),
            line_needed: false,
            entry_line: false,
            tbc: Vec::new(),
            counts_depth: false,
            depth: 0,
            loop_depths: Vec::new(),
            frames: false,
            live: Vec::new(),
            for_states: Vec::new(),
            lines: std::collections::BTreeSet::new(),
            tail_span: None,
            end_span: None,
            present: BTreeSet::new(),
            exact_classes: HashMap::new(),
        }
    }

    pub(super) fn typer(&self) -> Typer<'_> {
        self.m.typer()
    }

    pub(super) fn ir(&self, ty: Ty) -> Type {
        self.m.ir(ty)
    }

    pub(super) fn scopes(&self) -> &'a Scopes {
        self.m.scopes
    }

    /// A local no Lua program can spell.
    pub(super) fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("$t{}", self.temps))
    }

    /// The symbol a Lua local is lowered to: its name, or its name and
    /// id once another local of that name has been declared here.
    pub(super) fn local_symbol(&mut self, v: VarId) -> InternedString {
        let name = &self.scopes().var(v).name;
        let plain = intern(name);
        let symbol = if self.bound.contains(&plain) {
            intern(&format!("{name}${}", v.0))
        } else {
            plain
        };
        self.bound.insert(symbol);
        symbol
    }
}
