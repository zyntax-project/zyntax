//! Calls support.

use super::*;

impl<'a> Round<'a> {
    pub(super) fn call(&mut self, c: &ast::FunctionCall) {
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        // `setmetatable(t, m)` links a shape to its class; `rawset`
        // stores; neither loses its table. The `debug` library's
        // metatable functions do the same to a table.
        if let Some(b) = self.typer().builtin_callee(c.prefix(), &suffixes)
            && let Some(Suffix::Call(ast::Call::AnonymousCall(ast::FunctionArgs::Parentheses {
                arguments,
                ..
            }))) = suffixes.last()
        {
            let args: Vec<&Expression> = arguments.iter().collect();
            if (b.lib.is_empty()
                && matches!(
                    b.name,
                    "setmetatable" | "getmetatable" | "rawset" | "rawget"
                ))
                || (b.lib == "debug" && matches!(b.name, "setmetatable" | "getmetatable"))
            {
                self.metatable_builtin(b.name, &args);
                return;
            }
            // `table.sort` of a table with no metatable by a known
            // function calls it with the elements: a direct call, the
            // comparator kept as it is.
            if b.lib == "table"
                && b.name == "sort"
                && args.len() == 2
                && let Ty::Shape(k) = self.typer().ty_of(args[0])
                && let Ty::Func(f) = self.typer().ty_of(args[1])
                && !self.known.shape(k).unknown_meta
                && self.known.shape(k).classes.is_empty()
            {
                self.expr(args[0]);
                self.expr(args[1]);
                // A hole in the array part reaches the comparator as nil.
                let element = self.known.shape(k).element.join(Ty::Nil);
                self.record_call_types(f, vec![element, element]);
                return;
            }
            // `table.insert` and `table.move` store elements into their
            // table; the rest of the call takes values as any library
            // call does.
            if b.lib == "table" && matches!(b.name, "insert" | "move") {
                self.chain(c.prefix(), &suffixes);
                let tys: Vec<Ty> = args.iter().map(|a| self.typer().ty_of(a)).collect();
                match b.name {
                    "insert" if tys.len() >= 2 => {
                        self.store_through(tys[0], None, Ty::Int, tys[tys.len() - 1], false);
                    }
                    // What it reads of a table with a metatable may come
                    // through `__index`.
                    "move" if tys.len() >= 4 => {
                        let element = match tys[0] {
                            Ty::Shape(k)
                                if !self.known.shape(k).unknown_meta
                                    && self.known.shape(k).classes.is_empty() =>
                            {
                                self.known.shape(k).element
                            }
                            Ty::Unknown => Ty::Unknown,
                            source => {
                                self.untyped_read(source, Reads::under(Ty::Int, true));
                                Ty::Any
                            }
                        };
                        let into = tys.get(4).copied().unwrap_or(tys[0]);
                        self.store_through(into, None, Ty::Int, element, false);
                    }
                    _ => {}
                }
                return;
            }
        }
        self.chain(c.prefix(), &suffixes);
    }

    /// What a direct call of the library function `b` reads out of
    /// its table argument and hands out untyped.
    pub(super) fn library_reads_of(
        &mut self,
        b: &Builtin,
        prefix: &Prefix,
        args: &ast::FunctionArgs,
    ) {
        if let Some(reads) = library_reads(b) {
            let table = match args {
                ast::FunctionArgs::Parentheses { arguments, .. } => match arguments.iter().next() {
                    Some(first) => self.typer().ty_of(first),
                    None => Ty::Nil,
                },
                ast::FunctionArgs::TableConstructor(t) => match constructor_shape(self.known, t) {
                    Some(k) => Ty::Shape(k),
                    None => Ty::Table,
                },
                _ => Ty::Str,
            };
            // A typed walk binds each element or value as its type.
            let typed = borrows(b) == Borrows::Walk && self.typed_walk(b, prefix, table);
            if !typed {
                self.untyped_read(table, reads);
            }
        }
        // The iterator `pairs` or `ipairs` returns reads whatever
        // table it is given; outside a loop's header, any.
        let header = match prefix {
            Prefix::Name(token) => self.loop_iterator == Some(crate::scope::pos_of(token)),
            _ => false,
        };
        if b.lib.is_empty() && matches!(b.name, "pairs" | "ipairs") && !header {
            self.loose_reader();
        }
    }

    /// A library function that reads the tables it is given is held
    /// as a value: whatever it is called with, it may hand out as
    /// dynamic values what a table of any escaping shape holds.
    pub(super) fn loose_reader(&mut self) {
        self.blind_keyed_reads = self.blind_keyed_reads.union(Reads {
            names: true,
            elements: true,
            through_meta: true,
        });
    }

    /// Whether the global `name` is a library function that reads a
    /// table it is given, or the library table holding such functions.
    pub(super) fn is_reader(&self, name: &str) -> bool {
        builtin_named(self.scopes, name).is_some_and(|b| library_reads(b).is_some())
            || (name == "table" && !self.scopes.global_writes.contains_key(name))
    }

    /// `setmetatable`, `getmetatable`, `rawset`, `rawget`: what they do
    /// to the shapes involved.
    pub(super) fn metatable_builtin(&mut self, name: &str, args: &[&Expression]) {
        for a in args {
            self.expr(a);
        }
        let tys: Vec<Ty> = args.iter().map(|a| self.typer().ty_of(a)).collect();
        let receiver = tys.first().copied().unwrap_or(Ty::Nil);
        match name {
            "setmetatable" => {
                let meta = tys.get(1).copied().unwrap_or(Ty::Nil);
                // A metatable that is never nil, given to a table being
                // built, is the table's from birth; one that may be nil
                // may take the table's away.
                if let Ty::Shape(k) = receiver {
                    let kept = args.get(1).is_some_and(|m| self.typer().non_nil(m));
                    let built = args.first().and_then(|a| constructor_in(a));
                    let receiver = match args.first() {
                        _ if built.is_some() => MetaReceiver::Built,
                        Some(Expression::Var(Var::Name(t))) => match self.scopes.binding(t) {
                            Some(Binding::Local(v)) => MetaReceiver::Local(*v),
                            _ => MetaReceiver::Other,
                        },
                        _ => MetaReceiver::Other,
                    };
                    let info = &mut self.out.shapes[k.0 as usize];
                    info.meta_receivers.insert(receiver);
                    match built {
                        Some(t) if kept => {
                            info.born_meta.insert(constructor_site(t));
                        }
                        _ if kept => {}
                        _ => info.meta_cleared = true,
                    }
                }
                match (receiver, meta) {
                    (Ty::Shape(k), Ty::Shape(c)) => {
                        self.out.shapes[k.0 as usize].classes.insert(c);
                    }
                    (Ty::Shape(_), Ty::Nil | Ty::Unknown) => {}
                    (Ty::Shape(k), _) => {
                        self.out.shapes[k.0 as usize].unknown_meta = true;
                        self.escape(meta);
                    }
                    (Ty::Unknown, _) => {}
                    (_, _) => {
                        self.out.blind_setmetatable = true;
                        self.escape(meta);
                    }
                }
                for t in tys.iter().skip(2) {
                    self.escape(*t);
                }
            }
            "getmetatable" => {
                match receiver {
                    Ty::Shape(k) => {
                        let classes = self.out.shapes[k.0 as usize].classes.clone();
                        for c in classes {
                            self.escape(Ty::Shape(c));
                        }
                    }
                    Ty::Unknown => {}
                    _ => self.blind_getmetatable = true,
                }
                for t in tys.iter().skip(1) {
                    self.escape(*t);
                }
            }
            "rawset" => {
                let key = args.get(1).and_then(|k| crate::scope::literal_string(k));
                let key_ty = tys.get(1).copied().unwrap_or(Ty::Nil);
                let value = tys.get(2).copied().unwrap_or(Ty::Nil);
                self.store_through(receiver, key, key_ty, value, false);
                for t in tys.iter().skip(3) {
                    self.escape(*t);
                }
            }
            // `rawget` hands out what it finds typed from the shape, or
            // as a dynamic value.
            _ => {
                let key_ty = tys.get(1).copied().unwrap_or(Ty::Nil);
                let typed = self.typer().rawget_ty(args).is_some();
                match args.get(1).and_then(|k| crate::scope::literal_string(k)) {
                    _ if typed => {}
                    Some(key) if !matches!(receiver, Ty::Shape(_)) => {
                        self.read_through(receiver, &key)
                    }
                    _ => self.untyped_read(receiver, Reads::under(key_ty, false)),
                }
                for t in tys.iter().skip(1) {
                    self.escape(*t);
                }
            }
        }
    }

    /// A prefix and its suffixes, left to right: what each call passes
    /// is recorded against the function it reaches, and what each
    /// field read or method call goes through is noted.
    pub(super) fn chain(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        self.prefix(prefix);
        // A reader taken out of the `table` library as a value.
        if let Prefix::Name(token) = prefix
            && let Some(Binding::Global(g)) = self.scopes.binding(token)
            && g == "table"
            && self.is_reader(g)
            && let Some(Suffix::Index(index)) = suffixes.first()
            && !(suffixes.len() >= 2
                && self
                    .typer()
                    .builtin_callee(prefix, &suffixes[..2])
                    .is_some())
            && constant_key(index).is_none_or(|m| {
                builtin_member(self.scopes, "table", &m).is_some_and(|b| library_reads(b).is_some())
            })
        {
            self.loose_reader();
        }
        for (i, s) in suffixes.iter().enumerate() {
            let receiver = self.typer().suffixed_ty(prefix, &suffixes[..i]);
            match s {
                Suffix::Call(ast::Call::AnonymousCall(args)) => {
                    let builtin = self.typer().builtin_callee(prefix, &suffixes[..=i]);
                    if let Some(b) = builtin {
                        self.library_reads_of(b, prefix, args);
                    }
                    // The arguments of a direct call feed the callee's
                    // parameters, and so keep their types; a callee not
                    // typed yet may still turn out known. So do those of
                    // a field call the lookups resolve to known
                    // functions. Those a library function borrows keep
                    // theirs. Anything else takes values, and every
                    // function a field call may reach escapes.
                    let field = match receiver {
                        Ty::Func(_) | Ty::Unknown => None,
                        _ => self.typer().field_callee(prefix, &suffixes[..=i]),
                    };
                    let targets = field
                        .as_ref()
                        .and_then(|(k, name)| self.typer().method_targets(*k, name));
                    let direct = matches!(receiver, Ty::Func(_) | Ty::Unknown) || targets.is_some();
                    match builtin {
                        Some(b) if !direct => self.builtin_args(b, prefix, args),
                        _ => self.args(args, direct),
                    }
                    if let Ty::Func(f) = receiver {
                        self.record_call(f, None, args);
                    }
                    match (targets, field) {
                        (Some(targets), _) => {
                            self.join_target_returns(&targets);
                            for target in targets {
                                if let Ty::Func(f) = target {
                                    self.record_call(f, None, args);
                                }
                            }
                        }
                        (None, Some((k, name))) => self.escape_method(k, &name),
                        (None, None) => {}
                    }
                }
                Suffix::Call(ast::Call::MethodCall(m)) => {
                    let name = ident(m.name());
                    self.handler_calls(receiver);
                    let targets = match receiver {
                        Ty::Shape(k) => self.typer().method_targets(k, &name),
                        Ty::Unknown => Some(Vec::new()),
                        _ => None,
                    };
                    match targets {
                        Some(targets) => {
                            self.args(m.args(), true);
                            self.join_target_returns(&targets);
                            for target in targets {
                                if let Ty::Func(f) = target {
                                    self.record_call(f, Some(receiver), m.args());
                                }
                            }
                        }
                        None => {
                            self.read_through(receiver, &name);
                            if let Ty::Shape(k) = receiver {
                                self.escape_method(k, &name);
                            }
                            self.args(m.args(), false);
                        }
                    }
                }
                Suffix::Index(index) => {
                    if let ast::Index::Brackets { expression, .. } = index {
                        self.value(expression);
                    }
                    match (constant_key(index), index) {
                        (Some(name), _) => {
                            self.handler_calls(receiver);
                            self.read_through(receiver, &name);
                            // A read whose ends join to a dynamic value
                            // loses what they hold, unless a field call
                            // takes it.
                            let called = matches!(
                                suffixes.get(i + 1),
                                Some(Suffix::Call(ast::Call::AnonymousCall(_)))
                            );
                            if let Ty::Shape(k) = receiver
                                && !called
                                && self.typer().field_ty(receiver, &name) == Ty::Any
                            {
                                self.escape_method(k, &name);
                            }
                        }
                        (None, ast::Index::Brackets { expression, .. }) => {
                            let key_ty = self.typer().ty_of(expression);
                            self.keyed_read(receiver, key_ty);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    /// A constant-key lookup from `receiver`: each `__index` function
    /// it may reach is called with the table at that step and the key.
    pub(super) fn handler_calls(&mut self, receiver: Ty) {
        let Ty::Shape(k) = receiver else {
            return;
        };
        if k.0 as usize >= self.known.shapes.len() {
            return;
        }
        for (origin, f) in self.known.handlers_reached(k) {
            self.record_call_types(f, vec![Ty::Shape(origin), Ty::Str]);
        }
    }

    /// A direct call to `f`: what it passes joins the parameters.
    /// `receiver` is the object of a method call, passed first.
    pub(super) fn record_call(
        &mut self,
        f: FuncId,
        receiver: Option<Ty>,
        args: &ast::FunctionArgs,
    ) {
        let exprs: Vec<&Expression> = match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => arguments.iter().collect(),
            _ => Vec::new(),
        };
        let n = self.scopes.func(f).params.len();
        let mut types: Vec<Ty> = receiver.into_iter().collect();
        let wanted = n.saturating_sub(types.len());
        types.extend(match args {
            ast::FunctionArgs::String(_) => vec![Ty::Str],
            ast::FunctionArgs::TableConstructor(t) => vec![match constructor_shape(&self.out, t) {
                Some(k) => Ty::Shape(k),
                None => Ty::Table,
            }],
            _ => self.assigned_types(&exprs, wanted.max(exprs.len())),
        });
        self.record_call_types(f, types);
    }

    /// A direct call to `f` with arguments of `types`: they join the
    /// parameters; the extras go into `...` as values, or are dropped.
    pub(super) fn record_call_types(&mut self, f: FuncId, mut types: Vec<Ty>) {
        let info = self.scopes.func(f);
        let n = info.params.len();
        let is_vararg = info.is_vararg;
        for t in types.iter().skip(n) {
            if is_vararg {
                self.escape(*t);
            }
        }
        types.resize(n, Ty::Nil);
        let params = self
            .out
            .funcs
            .get(&f)
            .map(|s| s.params.clone())
            .unwrap_or_else(|| vec![Ty::Unknown; n]);
        let params: Vec<Ty> = params
            .into_iter()
            .zip(types)
            .map(|(p, t)| self.join_into(p, t))
            .collect();
        let sig = self.out.funcs.entry(f).or_insert_with(|| Sig {
            params: vec![Ty::Unknown; n],
            returns: Returns::Fixed(Vec::new()),
        });
        sig.params = params;
    }

    /// The arguments of a call to the library function `b`: those it
    /// borrows keep their types, the rest take values.
    pub(super) fn builtin_args(&mut self, b: &Builtin, prefix: &Prefix, args: &ast::FunctionArgs) {
        let ast::FunctionArgs::Parentheses { arguments, .. } = args else {
            return self.args(args, false);
        };
        let exprs: Vec<&Expression> = arguments.iter().collect();
        let borrowed = self.borrowed_args(b, prefix, &exprs);
        for (i, a) in exprs.iter().enumerate() {
            if i < borrowed {
                self.expr(a);
            } else {
                self.value(a);
            }
        }
    }

    /// How many of its leading arguments a call to `b` borrows: a
    /// library function may borrow a table only when everything it
    /// hands out of it is typed from the table's shape, or escapes.
    pub(super) fn borrowed_args(
        &self,
        b: &Builtin,
        prefix: &Prefix,
        exprs: &[&Expression],
    ) -> usize {
        let first = || match exprs.first() {
            Some(e) => self.typer().ty_of(e),
            None => Ty::Nil,
        };
        match borrows(b) {
            Borrows::No => 0,
            Borrows::Raw => exprs.len(),
            Borrows::Count => {
                let counts = exprs
                    .first()
                    .and_then(|e| crate::scope::literal_string(e))
                    .is_some_and(|s| s == "#");
                if counts { exprs.len() } else { 0 }
            }
            Borrows::Indexed => match first() {
                Ty::Shape(k) if self.known.plain(k) => 1,
                _ => 0,
            },
            Borrows::Walk => usize::from(self.typed_walk(b, prefix, first())),
        }
    }
}
