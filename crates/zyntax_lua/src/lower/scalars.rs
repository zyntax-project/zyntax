//! Scalars for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The parts of a number, held first unless it is a plain read.
    pub(super) fn number_parts(&mut self, v: Val) -> (Vec<St>, NumberParts) {
        let mut pre = Vec::new();
        let parts = self.number_parts_into(v, &mut pre);
        (pre, parts)
    }

    /// The parts of number `v`, with what computes them pushed on
    /// `pre`. A number built in place, alone or at the end of a block,
    /// is held part by part, so no aggregate is made only to be taken
    /// apart.
    pub(super) fn number_parts_into(&mut self, v: Val, pre: &mut Vec<St>) -> NumberParts {
        let span = v.node.span;
        let ty = v.ty;
        match v.node.node {
            TypedExpression::Block(mut block)
                if matches!(
                    block.statements.last().map(|s| &s.node),
                    Some(TypedStatement::Expression(_))
                ) =>
            {
                let Some(St {
                    node: TypedStatement::Expression(tail),
                    ..
                }) = block.statements.pop()
                else {
                    unreachable!("the block ends in an expression")
                };
                pre.extend(block.statements);
                self.number_parts_into(Val { node: *tail, ty }, pre)
            }
            TypedExpression::Tuple(parts) if parts.len() == 3 => {
                let mut held = parts.into_iter().zip([Ty::Int, Ty::Int, Ty::Float]);
                let mut next = || {
                    let (node, ty) = held.next().expect("three parts");
                    self.hold(Val { node, ty }, pre).node
                };
                NumberParts {
                    tag: next(),
                    int: next(),
                    float: next(),
                }
            }
            node => {
                let held = self.hold(
                    Val {
                        node: Node {
                            node,
                            ty: v.node.ty,
                            span,
                        },
                        ty,
                    },
                    pre,
                );
                NumberParts::of(&held.node, span)
            }
        }
    }

    /// A list literal whose items are plain reads: any other item is
    /// bound ahead of the literal, since the lowering stores the elements
    /// in one block and an item may carry control flow (a check after a
    /// call) below a wrapper.
    pub(super) fn array_of(&mut self, items: Vec<Node>, pre: &mut Vec<St>, span: Span) -> Node {
        let mut plain = Vec::with_capacity(items.len());
        for item in items {
            if Self::is_simple(&item) {
                plain.push(item);
            } else {
                let name = self.temp();
                let ty = item.ty.clone();
                pre.push(let_(name, ty.clone(), item, span));
                plain.push(var(name, ty, span));
            }
        }
        node(TypedExpression::Array(plain), self.m.anys(), span)
    }

    /// The truth of a value, as a boolean.
    pub(super) fn truthy(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Bool => v.node,
            // False when nil or false: the tags below true.
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.is_true(span), span)
            }
            // A shaped table is nil when null.
            Ty::Shape(_) => binary(
                BinaryOp::Ne,
                v.node,
                null(self.ir(Ty::Table), span),
                prim(PrimitiveType::Bool),
                span,
            ),
            Ty::Nil => block_value(vec![expr_stmt(v.node)], bool_lit(false, span), span),
            // A function value may be nil.
            Ty::Func(_) => binary(
                BinaryOp::Ne,
                v.node,
                nil(span),
                prim(PrimitiveType::Bool),
                span,
            ),
            Ty::Int | Ty::Float | Ty::Number | Ty::Str | Ty::Table => {
                if Self::is_simple(&v.node) {
                    bool_lit(true, span)
                } else {
                    block_value(vec![expr_stmt(v.node)], bool_lit(true, span), span)
                }
            }
            _ => call("zl_truthy", vec![v.node], prim(PrimitiveType::Bool), span),
        }
    }
}
