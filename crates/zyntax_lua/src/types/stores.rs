//! Stores support.

use super::*;

impl<'a> Round<'a> {
    pub(super) fn typer(&self) -> Typer<'_> {
        Typer {
            scopes: self.scopes,
            known: self.known,
        }
    }

    /// A function reached through this value may be called with
    /// anything from here on.
    pub(super) fn escape(&mut self, ty: Ty) {
        match ty {
            Ty::Func(f) => {
                if self.out.escaping.insert(f) && std::env::var_os("ZYNTAX_TRACE_ESCAPES").is_some()
                {
                    eprintln!(
                        "[escape] function {} ({:?}) in {}",
                        self.scopes.func(f).name,
                        f,
                        std::backtrace::Backtrace::force_capture()
                            .to_string()
                            .lines()
                            .filter(|l| l.contains("types.rs:"))
                            .take(4)
                            .map(|l| l.trim().to_string())
                            .collect::<Vec<_>>()
                            .join(" <- ")
                    );
                }
            }
            Ty::Shape(k) => {
                if !self.out.shapes[k.0 as usize].escapes
                    && std::env::var_os("ZYNTAX_TRACE_ESCAPES").is_some()
                {
                    eprintln!(
                        "[escape] shape {} in {}",
                        k.0,
                        std::backtrace::Backtrace::force_capture()
                            .to_string()
                            .lines()
                            .filter(|l| l.contains("types.rs:"))
                            .take(4)
                            .map(|l| l.trim().to_string())
                            .collect::<Vec<_>>()
                            .join(" <- ")
                    );
                }
                self.out.shapes[k.0 as usize].escapes = true;
            }
            _ => {}
        }
    }

    /// The shape of a constructor's tables, made if none is.
    pub(super) fn shape_for(&mut self, t: &ast::TableConstructor) -> ShapeId {
        let keys = constructor_keys(t);
        let line = t.braces().tokens().0.token().start_position().line();
        let make = |shapes: &mut Vec<ShapeInfo>, keys: &[String]| {
            let id = ShapeId(shapes.len() as u32);
            let mut fields = IndexMap::new();
            for k in keys {
                fields.entry(k.clone()).or_insert(Ty::Unknown);
            }
            let born = fields.len();
            shapes.push(ShapeInfo {
                fields,
                born,
                line,
                ..Default::default()
            });
            id
        };
        if keys.is_empty() {
            let site = constructor_site(t);
            if let Some(k) = self.out.shape_by_site.get(&site) {
                return *k;
            }
            let id = make(&mut self.out.shapes, &keys);
            self.out.shape_by_site.insert(site, id);
            id
        } else {
            let mut sorted = keys.clone();
            sorted.sort();
            sorted.dedup();
            if let Some(k) = self.out.shape_by_keys.get(&sorted) {
                return *k;
            }
            let id = make(&mut self.out.shapes, &sorted);
            self.out.shape_by_keys.insert(sorted, id);
            id
        }
    }

    /// A store of `ty` into field `name` of the tables of shape `k`;
    /// `non_nil` when the value is never nil.
    pub(super) fn store_field(&mut self, k: ShapeId, name: &str, ty: Ty, non_nil: bool) {
        let info = &mut self.out.shapes[k.0 as usize];
        if !non_nil {
            info.nil_stores.insert(name.to_string());
        }
        let slot = info.fields.get(name).copied().unwrap_or(Ty::Unknown);
        let joined = self.join_into(slot, ty);
        self.out.shapes[k.0 as usize]
            .fields
            .insert(name.to_string(), joined);
    }

    /// A store of `ty` under a key that is not a string into the
    /// tables of shape `k`. A nil stored is no element: the element
    /// type is what is present, and a read joins nil in.
    pub(super) fn store_element(&mut self, k: ShapeId, ty: Ty) {
        let element = self.out.shapes[k.0 as usize].element;
        let joined = self.join_into(element, present(ty));
        self.out.shapes[k.0 as usize].element = joined;
    }

    /// A store through `receiver` under `key` of a value of type `ty`:
    /// into the shape's field or element, or blind; `non_nil` when the
    /// value is never nil.
    pub(super) fn store_through(
        &mut self,
        receiver: Ty,
        key: Option<String>,
        key_ty: Ty,
        ty: Ty,
        non_nil: bool,
    ) {
        match (receiver, key) {
            (Ty::Shape(k), Some(name)) => self.store_field(k, &name, ty, non_nil),
            // Under a number the value is an element and stays typed.
            // Under a string it may land in any field, or under a name
            // that is none; under anything else in the hash part: what
            // may land there is `dyn_value`. A table with a metatable
            // may hand the value to `__newindex` instead.
            (Ty::Shape(k), None) => {
                if key_may_be_index(key_ty) {
                    self.store_element(k, ty);
                }
                if key_may_be_string(key_ty) {
                    self.out.shapes[k.0 as usize].dynamic_keys = true;
                    let names: Vec<String> = self.out.shapes[k.0 as usize]
                        .fields
                        .keys()
                        .cloned()
                        .collect();
                    for name in names {
                        self.store_field(k, &name, ty, non_nil);
                    }
                }
                if !key_is_number(key_ty) {
                    let held = self.out.shapes[k.0 as usize].dyn_value;
                    self.out.shapes[k.0 as usize].dyn_value = self.join_into(held, ty);
                }
                if !self.known.plain(k) {
                    self.escape(ty);
                }
            }
            (Ty::Unknown, _) => {}
            (_, Some(name)) => {
                let slot = self
                    .out
                    .blind_stores
                    .get(&name)
                    .copied()
                    .unwrap_or(Ty::Unknown);
                let joined = self.join_into(slot, ty);
                self.out.blind_stores.insert(name, joined);
                self.escape(ty);
            }
            (_, None) => {
                if key_may_be_string(key_ty) {
                    self.out.blind_dynamic_stores = true;
                }
                if key_may_be_index(key_ty) {
                    let element = self.out.blind_element;
                    self.out.blind_element = self.join_into(element, ty);
                }
                self.escape(ty);
            }
        }
    }

    /// A method called through a value: every function `name` may
    /// reach from a table of shape `k`, in the table or along its
    /// classes' `__index` chains, escapes. A chain that goes on where
    /// the types do not follow reads the name blind.
    pub(super) fn escape_method(&mut self, k: ShapeId, name: &str) {
        let mut seen = BTreeSet::new();
        let mut todo = vec![k];
        let mut blind = false;
        while let Some(t) = todo.pop() {
            if !seen.insert(t) || t.0 as usize >= self.out.shapes.len() {
                continue;
            }
            let info = &self.out.shapes[t.0 as usize];
            let own = info.field(name).map(|(_, ty)| ty);
            let classes: Vec<ShapeId> = info.classes.iter().copied().collect();
            blind |= info.unknown_meta;
            if let Some(ty) = own {
                self.escape(ty);
            }
            for c in classes {
                todo.push(c);
                match self.out.shapes[c.0 as usize].field("__index") {
                    Some((_, Ty::Shape(next))) => todo.push(next),
                    Some((_, Ty::Func(f))) => self.escape(self.known.handler_first(f)),
                    Some(_) => blind = true,
                    None => {}
                }
            }
        }
        if blind {
            self.blind_reads.insert(name.to_string());
        }
    }

    /// A read of field `name` through `receiver`, noted when the
    /// receiver is not a shape.
    pub(super) fn read_through(&mut self, receiver: Ty, name: &str) {
        if !matches!(receiver, Ty::Shape(_) | Ty::Unknown | Ty::Str) {
            self.blind_reads.insert(name.to_string());
        }
    }

    /// `receiver[key]` for a key of type `key_ty` not known at compile
    /// time: unless the types know what it yields, what it may yield
    /// escapes.
    pub(super) fn keyed_read(&mut self, receiver: Ty, key_ty: Ty) {
        if let Ty::Shape(k) = receiver
            && self.known.element_read_ty(k, key_ty) != Ty::Any
        {
            return;
        }
        self.untyped_read(receiver, Reads::under(key_ty, true));
    }

    /// Reads out of `receiver` whose results the types do not follow:
    /// noted on its shape, or, through a receiver the types do not
    /// know, against every escaping shape.
    pub(super) fn untyped_read(&mut self, receiver: Ty, reads: Reads) {
        match receiver {
            Ty::Shape(k) => {
                let info = &mut self.out.shapes[k.0 as usize];
                info.untyped_reads = info.untyped_reads.union(reads);
            }
            // A string's reads reach the string library, not a table.
            Ty::Unknown | Ty::Str => {}
            _ => self.blind_keyed_reads = self.blind_keyed_reads.union(reads),
        }
    }
}
