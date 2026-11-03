//! Finders for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The slot a field has in a shape's tables, when they have slots.
    pub(super) fn slot_of(&self, k: ShapeId, name: &str) -> Option<(&'m ShapeLayout, Slot)> {
        let layout = self.m.layout(k)?;
        let slot = layout.slots.iter().find(|s| s.name == name)?;
        Some((layout, slot.clone()))
    }

    /// The finder for `name` looked up from a table of shape `k`, in
    /// `mode`, made on first use: a function checking each lookup in
    /// the order the runtime looks, on to the next where a check fails.
    /// None when the types do not follow the lookups to their ends,
    /// when a table on the way has no slots, or when the name is
    /// nowhere along them.
    pub(super) fn finder(
        &mut self,
        k: ShapeId,
        name: &str,
        mode: FinderMode,
        span: Span,
    ) -> Option<Finder> {
        let key = (k, name.to_string(), mode);
        if let Some(found) = self.m.finders.borrow().get(&key) {
            return found.clone();
        }
        let made = self.make_finder(k, name, mode, span);
        self.m.finders.borrow_mut().insert(key, made.clone());
        made
    }

    /// A `Table` finder covers the lookups through metatables to a
    /// slot; `Which` and `Where` cover every lookup, the table's own
    /// field and `__index` functions included.
    pub(super) fn make_finder(
        &mut self,
        k: ShapeId,
        name: &str,
        mode: FinderMode,
        span: Span,
    ) -> Option<Finder> {
        let mut lookups: Vec<types::Lookup> = self.m.inferred.lookups(k, name)?;
        if mode == FinderMode::Table {
            if lookups
                .iter()
                .any(|l| matches!(l.hops.last(), Some(Hop::Handler(..))))
            {
                return None;
            }
            lookups.retain(|l| !l.hops.is_empty());
        }
        if lookups.is_empty() {
            return None;
        }
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let i64_t = prim(PrimitiveType::I64);
        let t = || var(intern("t"), table_t.clone(), span);
        let not_null = |x: &Node| {
            binary(
                BinaryOp::Ne,
                x.clone(),
                null(table_t.clone(), span),
                bool_t.clone(),
                span,
            )
        };
        let mut ends: Vec<(ShapeId, End)> = Vec::new();
        let mut body = Vec::new();
        if mode == FinderMode::Table {
            body.push(if_(
                binary(
                    BinaryOp::Eq,
                    t(),
                    null(table_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                vec![ret(Some(null(table_t.clone(), span)), span)],
                None,
                span,
            ));
        }
        for lookup in &lookups {
            let end = match lookup.hops.last() {
                Some(Hop::Handler(class, f)) => End::Handler {
                    class: *class,
                    f: *f,
                    ty: lookup.ty,
                },
                _ => {
                    self.slot_of(lookup.shape, name)?;
                    End::Slot(lookup.ty)
                }
            };
            let at = match ends.iter().position(|e| *e == (lookup.shape, end)) {
                Some(i) => i,
                None => {
                    ends.push((lookup.shape, end));
                    ends.len() - 1
                }
            };
            // The table each hop starts from: `t`, then a temporary per
            // hop that moves on. `origin` is the one the last `Meta`
            // hop started from.
            let mut x = t();
            let mut origin = t();
            let mut hops: Vec<(Hop, Node, Node)> = Vec::new();
            for hop in &lookup.hops {
                let to = match hop {
                    Hop::Meta(_) | Hop::Index(_) => var(self.temp(), table_t.clone(), span),
                    Hop::Absent(_) | Hop::Handler(..) => x.clone(),
                };
                if let Hop::Meta(_) = hop {
                    origin = x.clone();
                }
                hops.push((*hop, x, to.clone()));
                x = to;
            }
            // Inside out: the end, then each hop around it.
            let found = match (mode, end) {
                (FinderMode::Which, _) => int_lit(at as i64, span),
                (_, End::Handler { .. }) => origin,
                _ => x.clone(),
            };
            let mut inner = match end {
                End::Slot(_) if !lookup.sure => {
                    let (_, slot) = self.slot_of(lookup.shape, name)?;
                    vec![if_(
                        self.slot_present(&x, &slot, span),
                        vec![ret(Some(found), span)],
                        None,
                        span,
                    )]
                }
                _ => vec![ret(Some(found), span)],
            };
            for (hop, from, to) in hops.into_iter().rev() {
                inner = match hop {
                    Hop::Meta(m) => {
                        let TypedExpression::Variable(to_name) = &to.node else {
                            unreachable!("a hop's table is a variable")
                        };
                        let layout = self.m.layout(m)?;
                        let shape_is = binary(
                            BinaryOp::Eq,
                            field(to.clone(), "shape", i64_t.clone(), span),
                            int_lit(layout.gid, span),
                            bool_t.clone(),
                            span,
                        );
                        vec![
                            let_(
                                *to_name,
                                table_t.clone(),
                                field(from, "meta", table_t.clone(), span),
                                span,
                            ),
                            if_(
                                not_null(&to),
                                vec![if_(shape_is, inner, None, span)],
                                None,
                                span,
                            ),
                        ]
                    }
                    Hop::Absent(a) => {
                        let (_, slot) = self.slot_of(a, name)?;
                        let absent = binary(
                            BinaryOp::Eq,
                            binary(
                                BinaryOp::BitAnd,
                                self.present_of(&from, span),
                                int_lit(slot.mask(), span),
                                i64_t.clone(),
                                span,
                            ),
                            int_lit(0, span),
                            bool_t.clone(),
                            span,
                        );
                        vec![if_(absent, inner, None, span)]
                    }
                    Hop::Index(i) => {
                        let TypedExpression::Variable(to_name) = &to.node else {
                            unreachable!("a hop's table is a variable")
                        };
                        let (layout, slot) = self.slot_of(i, "__index")?;
                        if slot.kind != SlotKind::Table {
                            return None;
                        }
                        let mut through = vec![let_(
                            *to_name,
                            table_t.clone(),
                            self.slot_read(&from, layout, &slot, span).node,
                            span,
                        )];
                        through.extend(inner);
                        vec![if_(
                            self.slot_present(&from, &slot, span),
                            through,
                            None,
                            span,
                        )]
                    }
                    // The metatable holds the function: the lookup ends
                    // in its call, else in nil.
                    Hop::Handler(class, _) => {
                        let (_, slot) = self.slot_of(class, "__index")?;
                        if slot.kind != SlotKind::Any {
                            return None;
                        }
                        vec![if_(
                            self.slot_present(&from, &slot, span),
                            inner,
                            None,
                            span,
                        )]
                    }
                };
            }
            body.extend(inner);
        }
        // Nothing found.
        let (rest, returns, flavor) = match mode {
            FinderMode::Table => (null(table_t.clone(), span), table_t.clone(), "find"),
            FinderMode::Which => (int_lit(-1, span), i64_t, "which"),
            FinderMode::Where => (null(table_t.clone(), span), table_t.clone(), "where"),
        };
        body.push(ret(Some(rest), span));
        let helper = format!("lua${}{flavor}${}${}", self.m.tag, k.0, name);
        self.m.functions.borrow_mut().push(typed_function(
            &helper,
            vec![parameter(intern("t"), table_t, span)],
            returns,
            body,
            span,
        ));
        Some(Finder { helper, ends })
    }

    /// The value a finder found, read from `found` (held, not null) as
    /// `ty`: the slot the field has in the shape of `found`.
    pub(super) fn found_value(
        &mut self,
        finder: &Finder,
        found: &Node,
        name: &str,
        ty: Ty,
        span: Span,
    ) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let mut value: Option<Node> = None;
        for (shape, _) in finder.ends.iter().rev() {
            let Some((layout, slot)) = self.slot_of(*shape, name) else {
                continue;
            };
            let read = self.slot_read(found, layout, &slot, span);
            let read = self.coerce(read, ty);
            value = Some(match value {
                None => read,
                Some(rest) => if_value(
                    binary(
                        BinaryOp::Eq,
                        field(found.clone(), "shape", i64_t.clone(), span),
                        int_lit(layout.gid, span),
                        bool_t.clone(),
                        span,
                    ),
                    read,
                    rest,
                    self.ir(ty),
                    span,
                ),
            });
        }
        value.unwrap_or_else(|| self.zero_of(ty, span))
    }
}
