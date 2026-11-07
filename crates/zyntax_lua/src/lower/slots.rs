//! Slots for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The `present` word of table `t`.
    pub(super) fn present_of(&self, t: &Node, span: Span) -> Node {
        field(t.clone(), "present", prim(PrimitiveType::I64), span)
    }

    /// Whether slot `slot` of table `t` holds a value.
    pub(super) fn slot_present(&self, t: &Node, slot: &Slot, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        binary(
            BinaryOp::Ne,
            binary(
                BinaryOp::BitAnd,
                self.present_of(t, span),
                int_lit(slot.mask(), span),
                i64_t,
                span,
            ),
            int_lit(0, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// The value in slot `slot` of table `t`, as the slot stores it.
    pub(super) fn slot_read(&self, t: &Node, layout: &ShapeLayout, slot: &Slot, span: Span) -> Val {
        let shaped = as_shape(t.clone(), self.m.shape_ty(layout), span);
        let base = format!("s{}", slot.bit);
        let node = match slot.kind {
            SlotKind::Number => return self.number_slot_read(t, layout, slot, Ty::Number, span),
            SlotKind::Scalar => number_value(
                field(
                    shaped.clone(),
                    &format!("{base}t"),
                    prim(PrimitiveType::I64),
                    span,
                ),
                field(
                    shaped.clone(),
                    &format!("{base}i"),
                    prim(PrimitiveType::I64),
                    span,
                ),
                field(shaped, &format!("{base}f"), prim(PrimitiveType::F64), span),
                span,
            ),
            _ => field(shaped, &base, self.ir(slot.stored_ty()), span),
        };
        Val {
            node,
            ty: slot.stored_ty(),
        }
    }

    /// The value in slot `slot` of table `t` read as `ty`: a Number
    /// slot's float word alone for `Float` and its integer word alone
    /// for `Int`, which the caller knows the slot to hold; else as the
    /// slot stores it.
    pub(super) fn slot_read_as(
        &self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        ty: Ty,
        span: Span,
    ) -> Val {
        match slot.kind {
            SlotKind::Number => self.number_slot_read(t, layout, slot, ty, span),
            _ => self.slot_read(t, layout, slot, span),
        }
    }

    /// A word of Number slot `slot` of table `t`: `i` or `f`.
    pub(super) fn number_slot_word(
        &self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        word: char,
        span: Span,
    ) -> Node {
        let shaped = as_shape(t.clone(), self.m.shape_ty(layout), span);
        let ty = match word {
            'i' => prim(PrimitiveType::I64),
            _ => prim(PrimitiveType::F64),
        };
        field(shaped, &format!("s{}{word}", slot.bit), ty, span)
    }

    /// Number slot `slot` of table `t` read as `ty`: one word for
    /// `Float` or `Int`, else the number its kind bit tells.
    pub(super) fn number_slot_read(
        &self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        ty: Ty,
        span: Span,
    ) -> Val {
        let i64_t = prim(PrimitiveType::I64);
        match ty {
            Ty::Float => Val {
                node: self.number_slot_word(t, layout, slot, 'f', span),
                ty: Ty::Float,
            },
            Ty::Int => Val {
                node: self.number_slot_word(t, layout, slot, 'i', span),
                ty: Ty::Int,
            },
            _ => {
                let is_int = binary(
                    BinaryOp::Ne,
                    binary(
                        BinaryOp::BitAnd,
                        self.present_of(t, span),
                        int_lit(slot.kmask(), span),
                        i64_t,
                        span,
                    ),
                    int_lit(0, span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                Val {
                    node: number_value(
                        tag_of_is_int(is_int, span),
                        self.number_slot_word(t, layout, slot, 'i', span),
                        self.number_slot_word(t, layout, slot, 'f', span),
                        span,
                    ),
                    ty: Ty::Number,
                }
            }
        }
    }

    /// Statements storing `value` into slot `slot` of table `t`, and
    /// setting its bit; a value that may be nil clears the bit when it
    /// is. With `settled` the bit is set for good (a field born with
    /// the table that nothing clears) and left alone. A Number slot
    /// takes a value of its own type, `Int` or `Float`, as it is.
    pub(super) fn slot_store(
        &mut self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        value: Val,
        settled: bool,
        span: Span,
    ) -> Vec<St> {
        if slot.kind == SlotKind::Number {
            return self.number_slot_store(t, layout, slot, value, settled, span);
        }
        let i64_t = prim(PrimitiveType::I64);
        let shaped = as_shape(t.clone(), self.m.shape_ty(layout), span);
        let base = format!("s{}", slot.bit);
        let mut out = Vec::new();
        let value = self.coerce(value, slot.stored_ty());
        let value = self
            .hold(
                Val {
                    node: value,
                    ty: slot.stored_ty(),
                },
                &mut out,
            )
            .node;
        match slot.kind {
            SlotKind::Scalar => {
                let parts = NumberParts::of(&value, span);
                out.push(assign(
                    field(shaped.clone(), &format!("{base}t"), i64_t.clone(), span),
                    parts.tag,
                    span,
                ));
                out.push(assign(
                    field(shaped.clone(), &format!("{base}i"), i64_t.clone(), span),
                    parts.int,
                    span,
                ));
                out.push(assign(
                    field(shaped, &format!("{base}f"), prim(PrimitiveType::F64), span),
                    parts.float,
                    span,
                ));
            }
            _ => out.push(assign(
                field(shaped, &base, self.ir(slot.stored_ty()), span),
                value.clone(),
                span,
            )),
        }
        if settled {
            return out;
        }
        let present = self.present_of(t, span);
        let set = binary(
            BinaryOp::BitOr,
            present.clone(),
            int_lit(slot.mask(), span),
            i64_t.clone(),
            span,
        );
        let updated = match self.slot_nil(&value, slot, span) {
            Some(is_nil) => {
                let cleared = binary(
                    BinaryOp::BitAnd,
                    present.clone(),
                    int_lit(!slot.mask(), span),
                    i64_t.clone(),
                    span,
                );
                if_value(is_nil, cleared, set, i64_t, span)
            }
            None => set,
        };
        out.push(assign(present, updated, span));
        out
    }

    /// Statements storing `value` into Number slot `slot` of table `t`:
    /// a float writes `f` and clears the kind bit, an integer writes
    /// `i` and sets it, a number of run-time kind writes both words
    /// and sets the bit from its tag. The presence bit is set unless
    /// `settled`.
    pub(super) fn number_slot_store(
        &mut self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        value: Val,
        settled: bool,
        span: Span,
    ) -> Vec<St> {
        let i64_t = prim(PrimitiveType::I64);
        let mut out = Vec::new();
        let int_word = self.number_slot_word(t, layout, slot, 'i', span);
        let float_word = self.number_slot_word(t, layout, slot, 'f', span);
        let present = self.present_of(t, span);
        let or = |a: Node, b: Node| binary(BinaryOp::BitOr, a, b, i64_t.clone(), span);
        let with_mask = |p: Node| {
            if settled {
                p
            } else {
                or(p, int_lit(slot.mask(), span))
            }
        };
        let without_kind = |p: Node| {
            binary(
                BinaryOp::BitAnd,
                p,
                int_lit(!slot.kmask(), span),
                i64_t.clone(),
                span,
            )
        };
        match value.ty {
            Ty::Float => {
                out.push(assign(float_word, value.node, span));
                out.push(assign(
                    present.clone(),
                    without_kind(with_mask(present.clone())),
                    span,
                ));
            }
            Ty::Int => {
                let held = self.hold(value, &mut out);
                out.push(assign(
                    float_word,
                    cast(held.node.clone(), prim(PrimitiveType::F64), span),
                    span,
                ));
                out.push(assign(int_word, held.node, span));
                let bits = or(present.clone(), int_lit(slot.kmask(), span));
                out.push(assign(present.clone(), with_mask(bits), span));
            }
            _ => {
                let value = Val {
                    node: self.coerce(value, Ty::Number),
                    ty: Ty::Number,
                };
                let (pre, n) = self.number_parts(value);
                out.extend(pre);
                let kind = if_value(
                    n.is_int(span),
                    int_lit(slot.kmask(), span),
                    int_lit(0, span),
                    i64_t.clone(),
                    span,
                );
                out.push(assign(int_word, n.int, span));
                out.push(assign(float_word, n.float, span));
                let bits = or(without_kind(with_mask(present.clone())), kind);
                out.push(assign(present, bits, span));
            }
        }
        out
    }

    /// A new table's Number slot `slot` holding `value`, or nothing:
    /// its `present` bits and its two words' initializers, plain reads.
    pub(super) fn number_slot_init(
        &mut self,
        slot: &Slot,
        value: Option<Val>,
        pre: &mut Vec<St>,
        span: Span,
    ) -> (Node, Vec<TypedFieldInit>) {
        let i64_t = prim(PrimitiveType::I64);
        let (bits, int, float) = match value.map(|v| self.slot_value(v, slot)) {
            None => (int_lit(0, span), int_lit(0, span), float_lit(0.0, span)),
            Some(v) if v.ty == Ty::Int => {
                let held = self.hold(v, pre).node;
                (
                    int_lit(slot.mask() | slot.kmask(), span),
                    held.clone(),
                    cast(held, prim(PrimitiveType::F64), span),
                )
            }
            Some(v) if v.ty == Ty::Float => (
                int_lit(slot.mask(), span),
                int_lit(0, span),
                self.hold(v, pre).node,
            ),
            Some(v) => {
                let (held, n) = self.number_parts(v);
                pre.extend(held);
                let kind = if_value(
                    n.is_int(span),
                    int_lit(slot.mask() | slot.kmask(), span),
                    int_lit(slot.mask(), span),
                    i64_t.clone(),
                    span,
                );
                (kind, n.int, n.float)
            }
        };
        let words = [("i", int), ("f", float)]
            .into_iter()
            .map(|(suffix, value)| TypedFieldInit {
                name: intern(&format!("s{}{suffix}", slot.bit)),
                value: Box::new(value),
            })
            .collect();
        (bits, words)
    }

    /// `v` as a store into `slot` takes it: an integer or a float as it
    /// is for a Number slot, else of the slot's stored type.
    pub(super) fn slot_value(&mut self, v: Val, slot: &Slot) -> Val {
        match (slot.kind, v.ty) {
            (SlotKind::Number, Ty::Int | Ty::Float | Ty::Number) => v,
            _ => Val {
                node: self.coerce(v, slot.stored_ty()),
                ty: slot.stored_ty(),
            },
        }
    }

    /// Whether a value of the slot's stored type is nil, when it can be.
    pub(super) fn slot_nil(&self, value: &Node, slot: &Slot, span: Span) -> Option<Node> {
        let bool_t = prim(PrimitiveType::Bool);
        match slot.kind {
            SlotKind::Scalar => Some(NumberParts::of(value, span).has_tag(TAG_NIL, span)),
            SlotKind::Any => Some(binary(BinaryOp::Eq, value.clone(), nil(span), bool_t, span)),
            SlotKind::Table => Some(binary(
                BinaryOp::Eq,
                value.clone(),
                null(self.ir(Ty::Table), span),
                bool_t,
                span,
            )),
            _ => None,
        }
    }
}
