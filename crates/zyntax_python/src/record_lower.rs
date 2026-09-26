//! Lowering of records: dict literals of a fixed string-key shape held
//! as instances of their shape's class (see `records`).

use super::{BinaryOp, Lowerer, Node, Val, binary, field_storage, int_lit, node};
use crate::records;
use crate::types::{Elem, Ty};
use crate::{Error, Result, intern};
use ruff_python_ast as py;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    TypedExpression, TypedFieldAccess, TypedFieldInit, TypedLiteral, TypedStructLiteral,
};

impl Lowerer<'_> {
    /// The literal as its record: each value evaluated in order, then
    /// one allocation holding them.
    pub(crate) fn record_literal(
        &mut self,
        k: usize,
        d: &py::ExprDict,
        span: Span,
    ) -> Result<Node> {
        let mut fields = vec![TypedFieldInit {
            name: intern("$class"),
            value: Box::new(int_lit(k as i64, span)),
        }];
        for (i, item) in d.items.iter().enumerate() {
            let name = records::field_name(i);
            let ty = self.module.field(k, &name).map_or(Ty::Object, |(_, t)| t);
            let v = self.expr(&item.value)?;
            let stored = Val {
                node: self.field_in(v, ty),
                ty: field_storage(ty),
            };
            let mut pre = Vec::new();
            let held = self.hold(stored, &mut pre, span);
            self.hoisted.extend(pre);
            fields.push(TypedFieldInit {
                name: intern(&name),
                value: Box::new(held.node),
            });
        }
        Ok(node(
            TypedExpression::Struct(TypedStructLiteral {
                name: intern(&self.module.classes[k].name),
                fields,
            }),
            Ty::Class(k as u16),
            span,
        ))
    }

    /// The field a literal key of the record's shape names; any other
    /// key demotes the shape.
    fn record_field(&self, k: usize, index: &Val, span: Span) -> Result<(String, Ty)> {
        let key = match &index.node.node {
            TypedExpression::Literal(TypedLiteral::String(s)) => s.resolve_global(),
            _ => None,
        };
        match key.and_then(|key| records::field_of(k, &key)) {
            Some(f) => {
                let ty = self.module.field(k, &f).map_or(Ty::Object, |(_, t)| t);
                Ok((f, ty))
            }
            None => {
                records::demote_reached(self.module, Ty::Class(k as u16));
                Err(Error::unsupported_span(
                    "a record read or written by a key outside its shape",
                    span,
                ))
            }
        }
    }

    /// `rec["k"]`: the field's value.
    pub(crate) fn record_read(
        &mut self,
        k: usize,
        seq: Val,
        index: Val,
        span: Span,
    ) -> Result<Val> {
        let (f, ty) = self.record_field(k, &index, span)?;
        let object = self.checked_instance_as(
            seq,
            "TypeError",
            "'NoneType' object is not subscriptable",
            span,
        );
        let field = node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object.node),
                field: intern(&f),
            }),
            field_storage(ty),
            span,
        );
        let node = self.field_out(field, ty, span);
        Ok(Val { node, ty })
    }

    /// `rec["k"] = v`: the field's store.
    pub(crate) fn record_store(
        &mut self,
        k: usize,
        seq: Val,
        index: Val,
        value: Val,
        span: Span,
    ) -> Result<Node> {
        let (f, ty) = self.record_field(k, &index, span)?;
        let object = self.checked_instance_as(
            seq,
            "TypeError",
            "'NoneType' object does not support item assignment",
            span,
        );
        let value = self.field_in(value, ty);
        let field = node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object.node),
                field: intern(&f),
            }),
            field_storage(ty),
            span,
        );
        Ok(binary(BinaryOp::Assign, field, value, Ty::None, span))
    }

    /// A builtin or list method given a record, or a list of them, as
    /// something other than a sequence to walk demotes the shape: only
    /// a dict answers it.
    pub(crate) fn record_call_uses(&self, c: &py::ExprCall) {
        let record_in = |ty: Ty| match ty {
            Ty::Class(k) if records::is_record(k as usize) => Some((k, false)),
            Ty::List(Elem::Class(k)) if records::is_record(k as usize) => Some((k, true)),
            _ => None,
        };
        match &*c.func {
            py::Expr::Attribute(a) => {
                if let Some((k, true)) = record_in(self.ty_of(&a.value))
                    && matches!(
                        a.attr.as_str(),
                        "index" | "count" | "remove" | "sort" | "__contains__"
                    )
                {
                    records::demote_reached(self.module, Ty::Class(k));
                }
            }
            py::Expr::Name(n) => {
                let name = n.id.as_str();
                if self.module.funcs.contains_key(name)
                    || self.module.class_index.contains_key(name)
                {
                    return;
                }
                let walks = matches!(
                    name,
                    "len" | "enumerate" | "zip" | "reversed" | "iter" | "list" | "tuple"
                );
                for arg in &c.arguments.args {
                    let arg = match arg {
                        py::Expr::Starred(s) => &*s.value,
                        a => a,
                    };
                    match record_in(self.ty_of(arg)) {
                        Some((k, false)) => records::demote_reached(self.module, Ty::Class(k)),
                        Some((k, true)) if !walks => {
                            records::demote_reached(self.module, Ty::Class(k))
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}
