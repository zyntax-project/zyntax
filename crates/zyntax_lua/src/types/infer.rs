//! Infer support.

use super::*;

/// Infer the chunk: rounds until the types settle, then everything
/// unknown is nil.
pub fn infer(scopes: &Scopes, ast: &ast::Ast) -> Inferred {
    let mut known = Inferred::default();
    // Every constructor's shape exists before the first round, so a
    // constructor is a shape from the start rather than a table for
    // a round, which would lose what is made from it for good.
    // A name bound straight to a constructor names its shape, and a
    // field stored through the name (`Class.method = f`, `function
    // Class.new()`) is a field of the shape from the start, so a read
    // of it before the store's round is undecided rather than nil.
    {
        struct Constructors<'a> {
            round: Round<'a>,
            named: HashMap<Named, ShapeId>,
        }
        #[derive(Clone, PartialEq, Eq, Hash)]
        enum Named {
            Var(VarId),
            Global(String),
        }
        impl Constructors<'_> {
            /// The variable or global a name token is: declared here,
            /// or used.
            fn named(&self, token: &full_moon::tokenizer::TokenReference) -> Option<Named> {
                if let Some(v) = self.round.scopes.decls.get(&crate::scope::pos_of(token)) {
                    return Some(Named::Var(*v));
                }
                match self.round.scopes.binding(token)? {
                    Binding::Local(v) | Binding::Upvalue(v) => Some(Named::Var(*v)),
                    Binding::Global(name) => Some(Named::Global(name.clone())),
                    Binding::Field(..) => None,
                }
            }
            fn constructor_of(e: &Expression) -> Option<&ast::TableConstructor> {
                match e {
                    Expression::TableConstructor(t) => Some(t),
                    Expression::Parentheses { expression, .. } => Self::constructor_of(expression),
                    // `setmetatable({...}, m)` is the constructor.
                    Expression::FunctionCall(c) => {
                        let suffixes: Vec<&Suffix> = c.suffixes().collect();
                        let Prefix::Name(name) = c.prefix() else {
                            return None;
                        };
                        if ident(name) != "setmetatable" {
                            return None;
                        }
                        let [
                            Suffix::Call(ast::Call::AnonymousCall(
                                ast::FunctionArgs::Parentheses { arguments, .. },
                            )),
                        ] = suffixes.as_slice()
                        else {
                            return None;
                        };
                        Self::constructor_of(arguments.iter().next()?)
                    }
                    _ => None,
                }
            }
            fn bind(&mut self, token: &full_moon::tokenizer::TokenReference, e: &Expression) {
                if let Some(t) = Self::constructor_of(e)
                    && let Some(n) = self.named(token)
                {
                    let k = self.round.shape_for(t);
                    self.named.insert(n, k);
                }
            }
            fn field_of(&mut self, token: &full_moon::tokenizer::TokenReference, name: &str) {
                if let Some(n) = self.named(token)
                    && let Some(k) = self.named.get(&n).copied()
                {
                    self.round.out.shapes[k.0 as usize]
                        .fields
                        .entry(name.to_string())
                        .or_insert(Ty::Unknown);
                }
            }
        }
        impl full_moon::visitors::Visitor for Constructors<'_> {
            fn visit_table_constructor(&mut self, t: &ast::TableConstructor) {
                self.round.shape_for(t);
            }
            fn visit_local_assignment(&mut self, l: &ast::LocalAssignment) {
                for (name, e) in l.names().iter().zip(l.expressions().iter()) {
                    self.bind(name, e);
                }
            }
            fn visit_assignment(&mut self, a: &ast::Assignment) {
                for (target, e) in a.variables().iter().zip(a.expressions().iter()) {
                    match target {
                        Var::Name(name) => self.bind(name, e),
                        Var::Expression(v) => {
                            let suffixes: Vec<&Suffix> = v.suffixes().collect();
                            if let (Prefix::Name(name), [Suffix::Index(index)]) =
                                (v.prefix(), suffixes.as_slice())
                                && let Some(key) = constant_key(index)
                            {
                                self.field_of(name, &key);
                            }
                        }
                        _ => {}
                    }
                }
            }
            fn visit_function_declaration(&mut self, f: &ast::FunctionDeclaration) {
                let names: Vec<_> = f.name().names().iter().collect();
                match (names.as_slice(), f.name().method_name()) {
                    ([owner, field], None) => self.field_of(owner, &ident(field)),
                    ([owner], Some(method)) => self.field_of(owner, &ident(method)),
                    _ => {}
                }
            }
        }
        let empty = Inferred::default();
        let mut visitor = Constructors {
            round: Round {
                scopes,
                known: &empty,
                out: Inferred::default(),
                func: CHUNK,
                returns: None,
                blind_reads: HashSet::new(),
                blind_keyed_reads: Reads::default(),
                loop_iterator: None,
                blind_getmetatable: false,
                loops: 0,
                chunk_jumps: false,
            },
            named: HashMap::new(),
        };
        full_moon::visitors::Visitor::visit_ast(&mut visitor, ast);
        known.shapes = visitor.round.out.shapes;
        known.shape_by_keys = visitor.round.out.shape_by_keys;
        known.shape_by_site = visitor.round.out.shape_by_site;
    }
    known.non_nil_vars = non_nil_vars(scopes, ast);
    // Two fixed points, one inside the other. The inner one types the
    // chunk with a given set of escaping functions, whose parameters
    // take anything, and notes what else escapes; the outer one adds
    // those and types again. An escape is thus always read off a
    // settled typing, never off a round that had a type wrong.
    let mut escaping: HashSet<FuncId> = HashSet::new();
    loop {
        known = infer_given(scopes, ast, &known, &escaping);
        if known.escaping.is_subset(&escaping) {
            break;
        }
        escaping.extend(known.escaping.iter().copied());
    }
    for ty in known.vars.values_mut() {
        *ty = ty.settled();
    }
    for ty in known.globals.values_mut() {
        *ty = ty.settled();
    }
    for sig in known.funcs.values_mut() {
        for p in &mut sig.params {
            *p = p.settled();
        }
        sig.returns = sig.returns.clone().settled();
    }
    for shape in &mut known.shapes {
        for ty in shape.fields.values_mut() {
            *ty = ty.settled();
        }
    }
    trace_types(scopes, &known);
    known
}

/// Rounds until the types settle, with `escaping` the functions whose
/// parameters take anything. `seed` carries the shapes' identities.
pub(super) fn infer_given(
    scopes: &Scopes,
    ast: &ast::Ast,
    seed: &Inferred,
    escaping: &HashSet<FuncId>,
) -> Inferred {
    let mut known = Inferred {
        shapes: seed.shapes.clone(),
        shape_by_keys: seed.shape_by_keys.clone(),
        shape_by_site: seed.shape_by_site.clone(),
        escaping: escaping.clone(),
        non_nil_vars: seed.non_nil_vars.clone(),
        ..Default::default()
    };
    for _ in 0..32 {
        let mut round = Round {
            scopes,
            known: &known,
            out: Inferred::default(),
            func: CHUNK,
            returns: None,
            blind_reads: HashSet::new(),
            blind_keyed_reads: Reads::default(),
            loop_iterator: None,
            blind_getmetatable: false,
            loops: 0,
            chunk_jumps: false,
        };
        // The signatures from the last round carry over so a recursive
        // call in this one sees them; params are rejoined from calls.
        // The escapes given are the round's starting point; what it
        // finds is added for the next fixed point. The shapes keep
        // their ids and their fields' order; what is stored in them,
        // their classes and what reaches them are found again.
        round.out.escaping = escaping.clone();
        round.out.settling = known.settling;
        round.out.non_nil_vars = known.non_nil_vars.clone();
        // A variable the debug library may rebind holds anything; a
        // numeric loop keeps its own kind for its counter.
        if scopes.debug_rebinds || scopes.debug_setlocal {
            for (i, v) in scopes.vars.iter().enumerate() {
                if scopes.debug_setlocal || v.captured {
                    round.out.vars.insert(VarId(i as u32), Ty::Any);
                }
            }
        }
        round.out.shape_by_keys = known.shape_by_keys.clone();
        round.out.shape_by_site = known.shape_by_site.clone();
        round.out.shapes = known
            .shapes
            .iter()
            .map(|s| ShapeInfo {
                fields: s.fields.keys().map(|k| (k.clone(), Ty::Unknown)).collect(),
                born: s.born,
                line: s.line,
                ..Default::default()
            })
            .collect();
        for (f, sig) in &known.funcs {
            round.out.funcs.insert(
                *f,
                Sig {
                    params: vec![Ty::Unknown; sig.params.len()],
                    returns: sig.returns.clone(),
                },
            );
        }
        round.block(ast.nodes());
        // The chunk's outermost code runs once, unless it jumps or the
        // chunk is called as a value.
        if round.chunk_jumps || round.is_escaping(CHUNK) {
            for shape in &mut round.out.shapes {
                shape.remade = true;
            }
        }
        let chunk = round.returns.take();
        // A chunk used as a value hands what it returns to callers the
        // types do not see.
        if round.is_escaping(CHUNK)
            && let Some(returns) = &chunk
        {
            round.escape_returns(returns);
        }
        round.out.chunk = chunk;
        round.settle_shapes();
        let out = round.out;
        if std::env::var_os("ZYNTAX_TRACE_ROUNDS").is_some() {
            eprintln!(
                "[round] shapes {:?}",
                out.shapes.iter().map(|s| &s.fields).collect::<Vec<_>>()
            );
            let mut funcs: Vec<(&FuncId, &Sig)> = out.funcs.iter().collect();
            funcs.sort_by_key(|(f, _)| **f);
            for (f, sig) in funcs {
                eprintln!(
                    "[round]   {} {:?} -> {:?}",
                    scopes.func(*f).name,
                    sig.params,
                    sig.returns
                );
            }
            let mut vars: Vec<(&VarId, &Ty)> = out.vars.iter().collect();
            vars.sort_by_key(|(v, _)| **v);
            eprintln!(
                "[round]   vars {:?}",
                vars.iter()
                    .map(|(v, t)| format!("{}={:?}", scopes.var(**v).name, t))
                    .collect::<Vec<_>>()
            );
        }
        if out == known {
            if known.settling {
                break;
            }
            known.settling = true;
            continue;
        }
        known = out;
    }
    known
}

pub(super) fn trace_types(scopes: &Scopes, known: &Inferred) {
    if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
        for (i, shape) in known.shapes.iter().enumerate() {
            let k = ShapeId(i as u32);
            eprintln!(
                "[types] shape {i}@{}: {:?} born={} element={:?} classes={:?}{}{}{}{}{}{}{}",
                shape.line,
                shape.fields,
                shape.born,
                shape.element,
                shape.classes,
                if shape.escapes { " escapes" } else { "" },
                if shape.unknown_meta {
                    " unknown-meta"
                } else {
                    ""
                },
                if shape.dynamic_keys {
                    format!(" dynamic-keys={:?}", shape.dyn_value)
                } else {
                    String::new()
                },
                match shape.untyped_reads {
                    Reads {
                        names: false,
                        elements: false,
                        ..
                    } => String::new(),
                    r => format!(
                        " untyped-reads{}{}{}",
                        if r.names { ":names" } else { "" },
                        if r.elements { ":elements" } else { "" },
                        if r.through_meta { ":meta" } else { "" },
                    ),
                },
                if known.singleton(k) { " once" } else { "" },
                if known.meta_never_nil(k) {
                    " meta-kept"
                } else {
                    ""
                },
                if shape.nil_stores.is_empty() {
                    String::new()
                } else {
                    format!(" nil-stores={:?}", shape.nil_stores)
                },
            );
        }
        eprintln!(
            "[types] blind stores {:?} dynamic={} element={:?} setmetatable={}",
            known.blind_stores,
            known.blind_dynamic_stores,
            known.blind_element,
            known.blind_setmetatable
        );
        let mut escaping: Vec<String> = known
            .escaping
            .iter()
            .map(|f| format!("{}@{}", scopes.func(*f).name, scopes.func(*f).line))
            .collect();
        escaping.sort();
        eprintln!("[types] escaping {escaping:?}");
        let mut funcs: Vec<(&FuncId, &Sig)> = known.funcs.iter().collect();
        funcs.sort_by_key(|(f, _)| **f);
        for (f, sig) in funcs {
            let info = scopes.func(*f);
            eprintln!(
                "[types] {}@{} ({:?}) -> {:?}",
                info.name, info.line, sig.params, sig.returns
            );
        }
        let mut vars: Vec<(&VarId, &Ty)> = known.vars.iter().collect();
        vars.sort_by_key(|(v, _)| **v);
        for (v, ty) in vars {
            let info = scopes.var(*v);
            eprintln!("[types] local {} #{}: {ty:?}", info.name, v.0);
        }
        let mut globals: Vec<(&String, &Ty)> = known.globals.iter().collect();
        globals.sort_by(|a, b| a.0.cmp(b.0));
        for (name, ty) in globals {
            eprintln!("[types] global {name}: {ty:?}");
        }
    }
}
