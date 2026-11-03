//! Method lookup for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// Note the fields an assignment that has just run surely put in
    /// tables made once: a constant key, a value never nil, a table that
    /// keeps the field.
    pub(super) fn note_stores(&mut self, targets: &[&Var], exprs: &[&Expression]) {
        if !self.m.runs_once {
            return;
        }
        for (target, e) in targets.iter().zip(exprs) {
            let Var::Expression(v) = target else {
                continue;
            };
            let suffixes: Vec<&Suffix> = v.suffixes().collect();
            let Some((Suffix::Index(index), init)) = suffixes.split_last() else {
                continue;
            };
            if self.typer().global_member(v.prefix(), &suffixes).is_some() {
                continue;
            }
            if let Some(name) = types::constant_key(index)
                && self.typer().non_nil(e)
            {
                let receiver = self.typer().suffixed_ty(v.prefix(), init);
                self.note_store(receiver, name);
            }
        }
    }

    /// Note the field `function a.b:m()` stores the function in.
    pub(super) fn note_declaration(&mut self, f: &ast::FunctionDeclaration) {
        if !self.m.runs_once {
            return;
        }
        let names: Vec<&TokenReference> = f.name().names().iter().collect();
        let method = f.name().method_name();
        if names.len() == 1 && method.is_none() {
            return;
        }
        let (path, last) = match method {
            Some(m) => (&names[..], ident(m)),
            None => (&names[..names.len() - 1], ident(names[names.len() - 1])),
        };
        let typer = self.typer();
        let mut receiver = match self.scopes().binding(path[0]) {
            Some(b) => typer.binding_ty(b),
            None => return,
        };
        for name in &path[1..] {
            receiver = typer.field_ty(receiver, &ident(name));
        }
        self.note_store(receiver, last);
    }

    pub(super) fn note_store(&mut self, receiver: Ty, name: String) {
        if let Ty::Shape(k) = receiver
            && self.m.inferred.singleton(k)
            && self.m.inferred.keeps(k, &name)
        {
            self.present.insert((k, name));
        }
    }

    /// Whether a table of shape `k` surely holds `name` here: every
    /// table of the shape has it from birth on, or the one table of the
    /// shape has had it stored on every path here.
    pub(super) fn known_present(&self, k: ShapeId, name: &str) -> bool {
        self.m.inferred.born_present(k, name) || self.present.contains(&(k, name.to_string()))
    }

    /// Whether `lookup` surely ends where it says once its first hop
    /// has found a metatable of its shape: each table after it holds
    /// what the lookup goes on through, and the last one the field.
    pub(super) fn lookup_holds(&self, lookup: &types::Lookup, name: &str) -> bool {
        let inferred = self.m.inferred;
        let mut at: Option<ShapeId> = None;
        for hop in &lookup.hops {
            at = match (*hop, at) {
                (Hop::Meta(c), None) => Some(c),
                (Hop::Meta(c), Some(from)) if inferred.fixed_meta(from) == Some(c) => Some(c),
                (Hop::Index(c), Some(from)) if from == c && self.known_present(c, "__index") => {
                    match inferred.shape(c).field("__index") {
                        Some((_, Ty::Shape(next))) => Some(next),
                        _ => return false,
                    }
                }
                _ => return false,
            };
        }
        at == Some(lookup.shape) && (lookup.sure || self.known_present(lookup.shape, name))
    }

    /// Where a lookup of `name` from a table of shape `k` surely ends
    /// here, decided without asking the tables on the way: none when
    /// one of them may lack what the lookup goes through.
    pub(super) fn folded_lookups(&self, k: ShapeId, name: &str) -> Option<Folded> {
        self.folded_class_lookups(k, name, None)
    }

    pub(super) fn folded_class_lookups(
        &self,
        k: ShapeId,
        name: &str,
        exact: Option<ShapeId>,
    ) -> Option<Folded> {
        let inferred = self.m.inferred;
        let lookups = inferred.lookups(k, name)?;
        if let Some(own) = lookups.first().filter(|l| l.hops.is_empty()) {
            return (own.sure || self.known_present(k, name)).then_some(Folded::Own(own.ty));
        }
        let mut out = Vec::new();
        for class in inferred
            .shape(k)
            .classes
            .iter()
            .filter(|c| exact.is_none_or(|e| e == **c))
        {
            // The first lookup through this metatable that may fail
            // leaves the rest to run time.
            let first = lookups
                .iter()
                .find(|l| l.hops.first() == Some(&Hop::Meta(*class)));
            let end = match first {
                Some(l) if self.lookup_holds(l, name) => Some((l.shape, l.ty)),
                Some(_) => return None,
                None => None,
            };
            out.push((*class, end));
        }
        Some(Folded::Classes(out))
    }

    /// The ends of a call through a table of shape `k` whose lookup of
    /// `name` is decided here, and how the site picks one. None when an
    /// end is no known function, or holds a closure's record somewhere
    /// the site cannot read it from: the metatable itself, or the table.
    pub(super) fn fold_plan(
        &self,
        k: ShapeId,
        name: &str,
        folded: Folded,
    ) -> Option<(Vec<(ShapeId, End)>, FoldPlan)> {
        let inferred = self.m.inferred;
        match folded {
            Folded::Own(ty @ Ty::Func(_)) => {
                self.slot_of(k, name)?;
                Some((
                    vec![(k, End::Slot(ty))],
                    FoldPlan {
                        arms: None,
                        via_meta: false,
                    },
                ))
            }
            Folded::Own(_) => None,
            Folded::Classes(classes) => {
                let mut ends: Vec<(ShapeId, End)> = Vec::new();
                let mut arms = Vec::new();
                for (class, end) in &classes {
                    let gid = self.m.layout(*class)?.gid;
                    let Some((shape, ty)) = *end else {
                        continue;
                    };
                    let Ty::Func(f) = ty else {
                        return None;
                    };
                    if self.m.takes_env(f) && !(shape == *class && inferred.singleton(shape)) {
                        return None;
                    }
                    self.slot_of(shape, name)?;
                    let end = (shape, End::Slot(ty));
                    let at = match ends.iter().position(|e| *e == end) {
                        Some(i) => i,
                        None => {
                            ends.push(end);
                            ends.len() - 1
                        }
                    };
                    arms.push((gid, at as i64));
                }
                if ends.is_empty() {
                    return None;
                }
                // One end every metatable leads to, and a metatable every
                // table has: nothing to pick.
                let one =
                    ends.len() == 1 && arms.len() == classes.len() && inferred.meta_never_nil(k);
                Some((
                    ends,
                    FoldPlan {
                        arms: (!one).then_some(arms),
                        via_meta: true,
                    },
                ))
            }
        }
    }
}
