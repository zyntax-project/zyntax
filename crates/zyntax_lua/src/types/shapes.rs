//! Shapes support.

use super::*;

impl Inferred {
    pub fn var(&self, v: VarId) -> Ty {
        self.vars.get(&v).copied().unwrap_or(Ty::Unknown)
    }

    pub fn global(&self, name: &str) -> Ty {
        self.globals.get(name).copied().unwrap_or(Ty::Unknown)
    }

    pub fn sig(&self, f: FuncId) -> Option<&Sig> {
        self.funcs.get(&f)
    }

    pub fn shape(&self, k: ShapeId) -> &ShapeInfo {
        &self.shapes[k.0 as usize]
    }

    /// What a field, element or dynamic slot typed `ty` holds as a read
    /// sees it: nothing stored is nil once the rounds are settling.
    pub(super) fn stored(&self, ty: Ty) -> Ty {
        if self.settling && ty == Ty::Unknown {
            Ty::Nil
        } else {
            ty
        }
    }

    /// Whether a table of shape `k` may hold `name` other than in a
    /// field of its shape: stored under a key not known at compile
    /// time, or through a receiver the types do not know.
    pub fn may_hold_unnamed(&self, k: ShapeId, name: &str) -> bool {
        let info = self.shape(k);
        info.field(name).is_none()
            && (info.dynamic_keys || (info.escapes && self.blind_stores.contains_key(name)))
    }

    /// Whether a metatable of shape `c` may hold `event`, in a field or
    /// otherwise.
    pub fn class_may_hold(&self, c: ShapeId, event: &str) -> bool {
        self.shape(c).field(event).is_some() || self.may_hold_unnamed(c, event)
    }

    /// Whether a table of shape `k` may be weak: a metatable it may get
    /// holds `__mode`, or is one the types do not follow. The collector
    /// reaches a weak table's entries only outside slots.
    pub fn may_be_weak(&self, k: ShapeId) -> bool {
        let info = self.shape(k);
        info.unknown_meta
            || info
                .classes
                .iter()
                .any(|c| self.class_may_hold(*c, "__mode"))
    }

    /// What a table of shape `k` itself holds under `name` when that
    /// is no field of its shape: nil, or what such stores put there.
    pub fn unnamed_ty(&self, k: ShapeId, name: &str) -> Ty {
        let info = self.shape(k);
        let mut ty = Ty::Nil;
        // A store under a field's name lands in its slot.
        if info.field(name).is_some() {
            return ty;
        }
        if info.dynamic_keys {
            ty = join_read(ty, info.dyn_value);
        }
        if info.escapes
            && let Some(stored) = self.blind_stores.get(name)
        {
            ty = join_read(ty, *stored);
        }
        ty
    }

    /// Whether a table of shape `k` has no metatable: no handler may
    /// be called with it or with what is stored in it.
    pub fn plain(&self, k: ShapeId) -> bool {
        let info = self.shape(k);
        !info.unknown_meta && info.classes.is_empty()
    }

    /// Whether `#t` on a table of shape `k` is the length of its array
    /// part: no metatable it may have holds a `__len`.
    pub fn plain_len(&self, k: ShapeId) -> bool {
        let info = self.shape(k);
        !info.unknown_meta
            && info.classes.iter().all(|c| {
                self.shape(*c).field("__len").is_none() && !self.may_hold_unnamed(*c, "__len")
            })
    }

    /// Whether at most one table of shape `k` is ever made: its one
    /// constructor is in the main chunk's outermost code, run once.
    pub fn singleton(&self, k: ShapeId) -> bool {
        let info = self.shape(k);
        info.sites.len() == 1 && !info.remade
    }

    /// Whether a table of shape `k` keeps `name` once a store puts it
    /// there: the store lands in the slot, as no metatable it may have
    /// holds a `__newindex`, and nothing stores nil there, by name or
    /// through a key or a receiver the types do not follow.
    pub fn keeps(&self, k: ShapeId, name: &str) -> bool {
        let info = self.shape(k);
        info.field(name).is_some()
            && !info.escapes
            && !info.dynamic_keys
            && !info.unknown_meta
            && !info.nil_stores.contains(name)
            && info
                .classes
                .iter()
                .all(|c| !self.class_may_hold(*c, "__newindex"))
    }

    /// Whether every table of shape `k` holds `name` from birth on:
    /// every constructor sets it to something that is never nil, and
    /// the table keeps it.
    pub fn born_present(&self, k: ShapeId, name: &str) -> bool {
        let info = self.shape(k);
        info.field(name).is_some_and(|(i, _)| i < info.born) && self.keeps(k, name)
    }

    /// Whether every table of shape `k` has a metatable from birth on
    /// and never loses it.
    pub fn meta_never_nil(&self, k: ShapeId) -> bool {
        let info = self.shape(k);
        !info.unknown_meta
            && !info.meta_cleared
            && !info.classes.is_empty()
            && !info.sites.is_empty()
            && info.sites.is_subset(&info.born_meta)
    }

    /// The shape of the metatable every table of shape `k` has from
    /// birth on, when it has one shape only and is never taken away.
    pub fn fixed_meta(&self, k: ShapeId) -> Option<ShapeId> {
        let info = self.shape(k);
        match info.classes.iter().collect::<Vec<_>>().as_slice() {
            [c] if self.meta_never_nil(k) => Some(**c),
            _ => None,
        }
    }

    /// The functions `name` may resolve to for a table whose metatable
    /// has shape `k`, looked up as Lua does when the table itself
    /// lacks it: in the metatable's `__index` table, then in that
    /// table's own metatable's `__index`, and so on. Every function
    /// along the chain is one of them, since a field set later than
    /// the call lets the lookup go on past it; the call site tells
    /// them apart by the value it finds. `None` when a field on the
    /// way is not a function, or the chain leads where the types do
    /// not follow. An `__index` function ends the chain with its first
    /// result, when that is a function.
    pub fn resolve_method(&self, k: ShapeId, name: &str) -> Option<Vec<Ty>> {
        let mut table = match self.shape(k).field("__index") {
            Some((_, Ty::Shape(t))) => t,
            Some((_, Ty::Unknown)) => return Some(vec![Ty::Unknown]),
            Some((_, Ty::Func(f))) => return Some(vec![self.handler_method(f)?]),
            _ => return None,
        };
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(table) {
                break;
            }
            let info = self.shape(table);
            match info.field(name) {
                Some((_, ty @ (Ty::Func(_) | Ty::Unknown))) => {
                    out.push(ty);
                    if info.always_present(name) {
                        break;
                    }
                }
                Some(_) => return None,
                None if self.may_hold_unnamed(table, name) => return None,
                None => {}
            }
            // Behind this table: nothing, when it has no metatable.
            if !info.unknown_meta && info.classes.is_empty() {
                break;
            }
            match self.index_behind(table)? {
                Ty::Shape(next) => table = next,
                Ty::Func(f) => {
                    out.push(self.handler_method(f)?);
                    break;
                }
                _ => return None,
            }
        }
        Some(out)
    }

    /// The first result of an `__index` function, what a lookup
    /// through it yields.
    pub fn handler_first(&self, f: FuncId) -> Ty {
        self.sig(f).map_or(Ty::Unknown, |s| s.returns.first())
    }

    /// An `__index` function's first result as a method: a function,
    /// or not settled yet; None for anything else.
    pub(super) fn handler_method(&self, f: FuncId) -> Option<Ty> {
        match self.handler_first(f) {
            ty @ (Ty::Func(_) | Ty::Unknown) => Some(ty),
            _ => None,
        }
    }

    /// Where reading `name` from a table of shape `k` may end, as the
    /// runtime looks it up: the table's own field, then, when that is
    /// absent, its metatable's `__index` table and the tables behind
    /// it. None when the chain is not one the types follow to its end
    /// (a metatable they do not know, an `__index` that is not a
    /// table, a class with several metatables); an empty list when
    /// the name is nowhere along it.
    ///
    /// Complete when it answers: every class of `k` and every table
    /// behind each `__index` is known, each with at most one
    /// metatable, so a table of shape `k` either ends one of the
    /// listed lookups or holds nothing under `name`. An `__index`
    /// function ends a chain in a `Handler` hop.
    pub fn lookups(&self, k: ShapeId, name: &str) -> Option<Vec<Lookup>> {
        let mut out = Vec::new();
        let mut hops = Vec::new();
        let info = self.shape(k);
        if let Some((_, ty)) = info.field(name) {
            let sure = info.always_present(name);
            out.push(Lookup {
                hops: hops.clone(),
                shape: k,
                ty,
                sure,
            });
            if sure {
                return Some(out);
            }
            hops.push(Hop::Absent(k));
        }
        if info.unknown_meta || self.may_hold_unnamed(k, name) {
            return None;
        }
        for class in &info.classes {
            let mut hops = hops.clone();
            hops.push(Hop::Meta(*class));
            self.lookups_behind(k, *class, name, hops, &mut out)?;
        }
        Some(out)
    }

    /// The lookups of `name` that go through the `__index` of a
    /// metatable of shape `class`, the metatable of a table of shape
    /// `origin`.
    pub(super) fn lookups_behind(
        &self,
        origin: ShapeId,
        class: ShapeId,
        name: &str,
        mut hops: Vec<Hop>,
        out: &mut Vec<Lookup>,
    ) -> Option<()> {
        let mut table = match self.shape(class).field("__index") {
            Some((_, Ty::Shape(t))) => t,
            Some((_, Ty::Func(f))) => {
                self.handler_lookup(origin, class, f, hops, out);
                return Some(());
            }
            // A metatable without `__index` answers nothing.
            None if !self.may_hold_unnamed(class, "__index") => return Some(()),
            _ => return None,
        };
        hops.push(Hop::Index(class));
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(table) {
                return None;
            }
            let info = self.shape(table);
            if let Some((_, ty)) = info.field(name) {
                let sure = info.always_present(name);
                out.push(Lookup {
                    hops: hops.clone(),
                    shape: table,
                    ty,
                    sure,
                });
                if sure {
                    return Some(());
                }
                hops.push(Hop::Absent(table));
            }
            if self.may_hold_unnamed(table, name) {
                return None;
            }
            if !info.unknown_meta && info.classes.is_empty() {
                return Some(());
            }
            if info.unknown_meta || info.classes.len() != 1 {
                return None;
            }
            let behind = *info.classes.iter().next()?;
            hops.push(Hop::Meta(behind));
            table = match self.shape(behind).field("__index") {
                Some((_, Ty::Shape(next))) => next,
                Some((_, Ty::Func(f))) => {
                    self.handler_lookup(table, behind, f, hops, out);
                    return Some(());
                }
                _ => return None,
            };
            hops.push(Hop::Index(behind));
        }
    }

    /// The lookup ending in the `__index` function `f` of a metatable
    /// of shape `class`, called with a table of shape `origin`.
    pub(super) fn handler_lookup(
        &self,
        origin: ShapeId,
        class: ShapeId,
        f: FuncId,
        mut hops: Vec<Hop>,
        out: &mut Vec<Lookup>,
    ) {
        hops.push(Hop::Handler(class, f));
        out.push(Lookup {
            hops,
            shape: origin,
            ty: self.handler_first(f),
            sure: true,
        });
    }

    /// Every `__index` function a lookup from a table of shape `k` may
    /// call, each with the shape of the table it is called with.
    pub fn handlers_reached(&self, k: ShapeId) -> Vec<(ShapeId, FuncId)> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        let mut todo = vec![k];
        while let Some(t) = todo.pop() {
            if !seen.insert(t) {
                continue;
            }
            for class in &self.shape(t).classes {
                match self.shape(*class).field("__index") {
                    Some((_, Ty::Func(f))) => out.push((t, f)),
                    Some((_, Ty::Shape(next))) => todo.push(next),
                    _ => {}
                }
            }
        }
        out
    }

    /// What reading a table of shape `k` under a key of type `key_ty`
    /// not known at compile time yields, when the table has no
    /// metatable to ask: under a number its element type, or nil; under
    /// a key that may be anything else also what stores under such
    /// keys put there, and, when it may be a string, any field. A table
    /// the types lost track of may hold anything under such a key.
    pub fn element_read_ty(&self, k: ShapeId, key_ty: Ty) -> Ty {
        let info = self.shape(k);
        if info.unknown_meta || !info.classes.is_empty() {
            return Ty::Any;
        }
        self.raw_keyed_ty(k, key_ty)
    }

    /// What a table of shape `k` itself holds under a key of type
    /// `key_ty` not known at compile time, as a raw read finds it.
    pub fn raw_keyed_ty(&self, k: ShapeId, key_ty: Ty) -> Ty {
        let info = self.shape(k);
        if key_is_number(key_ty) {
            return join_read(self.stored(info.element), Ty::Nil);
        }
        if key_ty == Ty::Unknown {
            return Ty::Unknown;
        }
        if info.escapes {
            return Ty::Any;
        }
        let mut held = info.dyn_value;
        if key_may_be_string(key_ty) {
            for ty in info.fields.values() {
                held = held.join(*ty);
            }
        }
        if key_may_be_index(key_ty) {
            held = held.join(info.element);
        }
        join_read(self.stored(held), Ty::Nil)
    }

    /// What `pairs(t)` binds as each value, for `t` of type `ty`: any
    /// element, field or value stored under a key not known at compile
    /// time, when the types know every one and no metatable may take
    /// the walk over with `__pairs`. Anything otherwise.
    pub fn pairs_value_ty(&self, ty: Ty) -> Ty {
        let Ty::Shape(k) = ty else {
            return if ty == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Any
            };
        };
        let info = self.shape(k);
        if info.unknown_meta
            || info.escapes
            || info.classes.iter().any(|c| {
                self.shape(*c).field("__pairs").is_some() || self.may_hold_unnamed(*c, "__pairs")
            })
        {
            return Ty::Any;
        }
        let mut held = info.element.join(info.dyn_value);
        for ty in info.fields.values() {
            held = held.join(*ty);
        }
        held
    }

    /// The element type a `for` over `ipairs(t)` binds, for `t` of type
    /// `ty`: the shape's, when its reads are typed.
    pub fn ipairs_value_ty(&self, ty: Ty) -> Ty {
        match ty {
            Ty::Shape(k) if self.element_read_ty(k, Ty::Int) != Ty::Any => self.shape(k).element,
            Ty::Unknown => Ty::Unknown,
            _ => Ty::Any,
        }
    }

    /// The functions the metamethod `event` of a table of shape `k`
    /// may be: one per class, read raw from the metatable as the
    /// runtime reads it. None when the types do not know every
    /// metatable's handler; Unsettled while some class's handler is
    /// not typed yet.
    pub fn metamethod_targets(&self, k: ShapeId, event: &str) -> MetaTargets {
        let info = self.shape(k);
        if info.unknown_meta || info.classes.is_empty() {
            return MetaTargets::None;
        }
        let mut out = Vec::new();
        for class in &info.classes {
            match self.shape(*class).field(event) {
                Some((_, Ty::Func(f))) => out.push((*class, f)),
                Some((_, Ty::Unknown)) => return MetaTargets::Unsettled,
                _ => return MetaTargets::None,
            }
        }
        MetaTargets::Funcs(out)
    }

    /// What a lookup goes on to when a table of shape `t` lacks a
    /// field: its one class's `__index`, when the types know it.
    pub(super) fn index_behind(&self, t: ShapeId) -> Option<Ty> {
        let info = self.shape(t);
        if info.unknown_meta || info.classes.len() != 1 {
            return None;
        }
        let class = *info.classes.iter().next()?;
        self.shape(class).field("__index").map(|(_, ty)| ty)
    }

    /// What reading field `name` of a table of shape `k` yields when
    /// the field is absent: what the table holds under a name that is
    /// no field, else what its metatables' `__index` chains hold, nil
    /// when it has no metatable, anything when the types do not know
    /// what it has.
    pub fn absent_field_ty(&self, k: ShapeId, name: &str) -> Ty {
        let info = self.shape(k);
        if info.unknown_meta {
            return Ty::Any;
        }
        let mut ty = self.unnamed_ty(k, name);
        for class in &info.classes {
            ty = join_read(ty, self.through_index_ty(*class, name, 0));
            if ty == Ty::Any {
                break;
            }
        }
        ty
    }

    /// What `name` yields looked up through the `__index` of a
    /// metatable of shape `class`: the chain of its table, its
    /// function's first result, nil without one.
    pub(super) fn through_index_ty(&self, class: ShapeId, name: &str, depth: usize) -> Ty {
        match self.shape(class).field("__index") {
            Some((_, Ty::Shape(next))) => self.index_chain_ty(next, name, depth + 1),
            Some((_, Ty::Func(f))) => self.handler_first(f),
            Some((_, Ty::Unknown)) => Ty::Unknown,
            None if !self.may_hold_unnamed(class, "__index") => Ty::Nil,
            _ => Ty::Any,
        }
    }

    /// What `name` yields looked up in a table of shape `class` as a
    /// metatable's `__index`: the field when it holds one, else what
    /// the class's own metatable gives, else nil.
    pub(super) fn index_chain_ty(&self, class: ShapeId, name: &str, depth: usize) -> Ty {
        if depth > 8 {
            return Ty::Any;
        }
        let info = self.shape(class);
        let own = match info.field(name).map(|(i, ty)| (i, self.stored(ty))) {
            Some((_, Ty::Unknown)) => return Ty::Unknown,
            Some((_, ty)) if info.always_present(name) => return ty,
            Some((_, ty)) => Some(ty),
            None if self.may_hold_unnamed(class, name) => Some(self.unnamed_ty(class, name)),
            None => None,
        };
        // The field may be absent: what lies behind it.
        let behind = if info.unknown_meta {
            Ty::Any
        } else {
            let mut t = Ty::Nil;
            for c in &info.classes {
                t = join_read(t, self.through_index_ty(*c, name, depth));
            }
            t
        };
        match own {
            Some(own) => join_read(own, behind),
            None => behind,
        }
    }

    /// What reading field `name` of a table of shape `k` yields. A
    /// field nothing has stored in yet decides nothing yet.
    pub fn field_read_ty(&self, k: ShapeId, name: &str) -> Ty {
        let info = self.shape(k);
        match info.field(name).map(|(i, ty)| (i, self.stored(ty))) {
            Some((_, Ty::Unknown)) => Ty::Unknown,
            Some((_, ty)) if info.always_present(name) => ty,
            Some((_, ty)) => join_read(ty, self.absent_field_ty(k, name)),
            None => self.absent_field_ty(k, name),
        }
    }
}
