//! Settle support.

use super::*;

impl<'a> Round<'a> {
    /// What reaches the shapes the types lost track of: a table of an
    /// escaping shape may take the stores made through receivers the
    /// types do not know, get any metatable set that way or through a
    /// `setmetatable` the types do not follow, and have
    /// its fields read and its methods called that way. What such a
    /// table holds escapes with it; its classes are reached through
    /// `getmetatable`.
    pub(super) fn settle_shapes(&mut self) {
        let dynamic_code = self.scopes.dynamic_code;
        // The library's readers are reached through the globals table,
        // or by code the chunk does not contain.
        if dynamic_code || self.scopes.dynamic_globals {
            self.loose_reader();
        }
        loop {
            let mut changed = false;
            // A function under a metamethod's name is called by the
            // runtime, with dynamic values, wherever its table serves
            // as a metatable: it escapes, unless the event is one the
            // typed dispatch handles and no table of the shapes it is
            // the metatable of ever reaches the runtime's own dispatch,
            // that is, none of them escapes (nor the class itself).
            for k in 0..self.out.shapes.len() {
                let metamethods: Vec<(String, Ty)> = self.out.shapes[k]
                    .fields
                    .iter()
                    .filter(|(name, _)| name.starts_with("__"))
                    .map(|(name, ty)| (name.clone(), *ty))
                    .collect();
                let class = ShapeId(k as u32);
                let reached = self.out.shapes[k].escapes
                    || self
                        .out
                        .shapes
                        .iter()
                        .any(|s| s.escapes && s.classes.contains(&class));
                for (name, ty) in metamethods {
                    // An `__index` table is looked up in, not handed
                    // out: it escapes with the tables whose lookups
                    // reach it, below.
                    if name == "__index" && matches!(ty, Ty::Shape(_)) {
                        continue;
                    }
                    if reached || !typed_event(&name) {
                        changed |= self.escape_new(ty);
                    }
                }
            }
            for k in 0..self.out.shapes.len() {
                changed |= self.settle_reads(k);
                // A weak table's entries may be collected: no field is
                // sure to be present.
                if self.out.shapes[k].born > 0 && self.out.may_be_weak(ShapeId(k as u32)) {
                    self.out.shapes[k].born = 0;
                    changed = true;
                }
                if !self.out.shapes[k].escapes {
                    continue;
                }
                // A table of the shape may be reached through anything.
                let fields: Vec<(String, Ty)> = self.out.shapes[k]
                    .fields
                    .iter()
                    .map(|(n, t)| (n.clone(), *t))
                    .collect();
                for (name, ty) in &fields {
                    match ty {
                        Ty::Shape(other) if !self.out.shapes[other.0 as usize].escapes => {
                            self.out.shapes[other.0 as usize].escapes = true;
                            changed = true;
                        }
                        Ty::Func(f)
                            if (self.blind_reads.contains(name)
                                || self.blind_getmetatable
                                || dynamic_code)
                                && !self.out.escaping.contains(f) =>
                        {
                            self.escape(Ty::Func(*f));
                            changed = true;
                        }
                        _ => {}
                    }
                }
                if (self.out.blind_setmetatable || self.scopes.unseen_metatables || dynamic_code)
                    && !self.out.shapes[k].unknown_meta
                {
                    self.out.shapes[k].unknown_meta = true;
                    changed = true;
                }
                if (self.out.blind_dynamic_stores || dynamic_code)
                    && !self.out.shapes[k].dynamic_keys
                {
                    self.out.shapes[k].dynamic_keys = true;
                    changed = true;
                }
                // Its array part takes the stores made blind, and what
                // it holds escapes with it.
                let element = self.out.shapes[k].element;
                let reaching = if dynamic_code {
                    Ty::Any
                } else {
                    present(self.out.blind_element)
                };
                let joined = self.join_into(element, reaching);
                if joined != element {
                    self.out.shapes[k].element = joined;
                    changed = true;
                }
                if joined != Ty::Any && !matches!(joined, Ty::Unknown) {
                    changed |= self.escape_new(joined);
                }
                if self.blind_getmetatable || dynamic_code {
                    let classes: Vec<ShapeId> =
                        self.out.shapes[k].classes.iter().copied().collect();
                    for c in classes {
                        if !self.out.shapes[c.0 as usize].escapes {
                            self.out.shapes[c.0 as usize].escapes = true;
                            changed = true;
                        }
                    }
                }
                // Its classes' methods are reached through it, and the
                // tables its lookups go on to.
                let classes: Vec<ShapeId> = self.out.shapes[k].classes.iter().copied().collect();
                for c in classes {
                    if let Some((_, ty @ Ty::Shape(_))) =
                        self.out.shapes[c.0 as usize].field("__index")
                    {
                        changed |= self.escape_new(ty);
                    }
                    let methods: Vec<(String, Ty)> = self.out.shapes[c.0 as usize]
                        .fields
                        .iter()
                        .map(|(n, t)| (n.clone(), *t))
                        .collect();
                    for (name, ty) in methods {
                        if let Ty::Func(f) = ty
                            && (self.blind_reads.contains(&name) || dynamic_code)
                            && !self.out.escaping.contains(&f)
                        {
                            self.escape(Ty::Func(f));
                            changed = true;
                        }
                    }
                }
                let blind: Vec<(String, Ty)> = self
                    .out
                    .blind_stores
                    .iter()
                    .map(|(n, t)| (n.clone(), *t))
                    .collect();
                for (name, ty) in blind {
                    if let Some(slot) = self.out.shapes[k].fields.get(&name).copied() {
                        let joined = self.join_into(slot, ty);
                        if joined != slot {
                            self.out.shapes[k].fields.insert(name, joined);
                            changed = true;
                        }
                    }
                }
                if self.out.shapes[k].dynamic_keys {
                    let names: Vec<String> = self.out.shapes[k].fields.keys().cloned().collect();
                    for name in names {
                        let slot = self.out.shapes[k].fields[&name];
                        if slot != Ty::Any {
                            self.escape(slot);
                            self.out.shapes[k].fields.insert(name, Ty::Any);
                            changed = true;
                        }
                    }
                    let held = self.out.shapes[k].dyn_value;
                    if held != Ty::Any {
                        self.escape(held);
                        self.out.shapes[k].dyn_value = Ty::Any;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// What the untyped reads of a table of shape `k` may yield
    /// escapes, and those that go on through `__index` reach the
    /// tables behind it; an escaping shape takes the reads made
    /// through receivers the types do not know. Whether anything
    /// changed.
    pub(super) fn settle_reads(&mut self, k: usize) -> bool {
        let mut changed = false;
        if self.out.shapes[k].escapes {
            let reads = self.out.shapes[k]
                .untyped_reads
                .union(self.blind_keyed_reads);
            if reads != self.out.shapes[k].untyped_reads {
                self.out.shapes[k].untyped_reads = reads;
                changed = true;
            }
        }
        let reads = self.out.shapes[k].untyped_reads;
        let mut held = Vec::new();
        if reads.names {
            held.extend(self.out.shapes[k].fields.values().copied());
            held.push(self.out.shapes[k].dyn_value);
        }
        if reads.elements {
            held.push(self.out.shapes[k].element);
        }
        for ty in held {
            changed |= self.escape_new(ty);
        }
        if reads.through_meta && (reads.names || reads.elements) {
            // A metatable the types do not follow was lost to them, and
            // its `__index` with it: the reads reach every escaping
            // shape.
            if self.out.shapes[k].unknown_meta {
                let blind = self.blind_keyed_reads.union(reads);
                if blind != self.blind_keyed_reads {
                    self.blind_keyed_reads = blind;
                    changed = true;
                }
            }
            let classes: Vec<ShapeId> = self.out.shapes[k].classes.iter().copied().collect();
            for c in classes {
                match self.out.shapes[c.0 as usize].field("__index") {
                    Some((_, Ty::Shape(next))) => {
                        let info = &mut self.out.shapes[next.0 as usize];
                        let joined = info.untyped_reads.union(reads);
                        if joined != info.untyped_reads {
                            info.untyped_reads = joined;
                            changed = true;
                        }
                    }
                    // A handler called with a key the types do not know.
                    Some((_, ty @ Ty::Func(_))) => changed |= self.escape_new(ty),
                    _ => {}
                }
            }
        }
        changed
    }
}
