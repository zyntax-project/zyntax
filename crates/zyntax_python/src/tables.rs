//! Dicts and sets as the lowering builds them: the functions a dict or
//! set of a shape is lowered to, how its keys, values and elements go
//! in and come out, and how a key of one kind probes a table of another.
//!
//! A shape's store is how it holds its keys and values (see
//! [`types::table_stored`]); shapes of one store share their storage, so
//! a dict read as one of them is the same dict read as another. The
//! library's own dict and set are the store of dynamic values; every
//! other store has its functions generated with the program, suffixed
//! `d<i>` or `s<i>`, and its own box tag. A frozenset is a set of its
//! store under a tag of its own.

use super::{
    Lowerer, Node, Stmt, Val, binary, call, cast, field_of, int_lit, int32_lit, list_type_id, node,
    str_lit, var,
};
use crate::Result;
use crate::types::{self, Elem, Ty};
use ruff_python_ast as py;
use zyntax_builtins::lists::keyed_fn;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedExpression, TypedLet, TypedLiteral, TypedStatement};
use zyntax_typed_ast::{BinaryOp, Mutability, Type, TypedNode};

/// The suffix of the functions of the store of dict or set type `ty`,
/// empty for the library's own.
pub(crate) fn table_suffix(ty: Ty) -> String {
    match ty {
        Ty::Dict(k) => match types::dict_store(k) {
            None => String::new(),
            Some(i) => format!("d{i}"),
        },
        Ty::Set(k) => match types::set_store(k) {
            (_, Ty::Object) => String::new(),
            (i, _) => format!("s{i}"),
        },
        other => unreachable!("a dict or set type, not {other:?}"),
    }
}

/// Function `op` of dict type `ty`.
pub(crate) fn dict_fn(op: &str, ty: Ty) -> String {
    keyed_fn("dict", op, &table_suffix(ty))
}

/// Function `op` of set type `ty`.
pub(crate) fn set_fn(op: &str, ty: Ty) -> String {
    keyed_fn("set", op, &table_suffix(ty))
}

/// Function `op` of dict or set type `ty`.
pub(crate) fn table_fn(op: &str, ty: Ty) -> String {
    match ty {
        Ty::Dict(_) => dict_fn(op, ty),
        _ => set_fn(op, ty),
    }
}

/// Whether set type `ty` is a frozenset.
pub(crate) fn frozen(ty: Ty) -> bool {
    matches!(ty, Ty::Set(k) if types::set_shape(k).1)
}

/// Whether set type `ty` is a frozenset boxed under a tag of its own.
pub(crate) fn frozen_tagged(ty: Ty) -> bool {
    frozen(ty) && types::typed_tables()
}

/// The tag a dict or set of type `ty` is boxed under.
pub(crate) fn table_tag(ty: Ty) -> i64 {
    match ty {
        Ty::Dict(k) => match types::dict_store(k) {
            None => zyntax_builtins::DICT_TAG,
            Some(i) => zyntax_builtins::dicts::dict_shape_tag(i),
        },
        Ty::Set(k) => {
            let (i, stored) = types::set_store(k);
            if frozen_tagged(ty) {
                zyntax_builtins::dicts::frozen_set_shape_tag(i)
            } else if stored == Ty::Object {
                zyntax_builtins::SET_TAG
            } else {
                zyntax_builtins::dicts::set_shape_tag(i)
            }
        }
        other => unreachable!("a dict or set type, not {other:?}"),
    }
}

/// How dict type `ty` stores its keys and its values.
pub(crate) fn dict_stored(ty: Ty) -> (Ty, Ty) {
    match ty {
        Ty::Dict(k) => match types::dict_store(k) {
            None => (Ty::Object, Ty::Object),
            Some(i) => types::dict_stores()[i as usize],
        },
        other => unreachable!("a dict type, not {other:?}"),
    }
}

/// How set type `ty` stores its elements.
pub(crate) fn set_stored(ty: Ty) -> Ty {
    match ty {
        Ty::Set(k) => types::set_store(k).1,
        other => unreachable!("a set type, not {other:?}"),
    }
}

/// How a table of type `ty` stores its keys.
pub(crate) fn key_stored(ty: Ty) -> Ty {
    match ty {
        Ty::Dict(_) => dict_stored(ty).0,
        _ => set_stored(ty),
    }
}

/// Whether two dict or set types share their storage, frozenness
/// aside.
pub(crate) fn same_store(a: Ty, b: Ty) -> bool {
    match (a, b) {
        (Ty::Dict(x), Ty::Dict(y)) => types::dict_store(x) == types::dict_store(y),
        (Ty::Set(x), Ty::Set(y)) => types::set_store(x) == types::set_store(y),
        _ => false,
    }
}

/// The IR type of a dict or set of type `ty`: the entry list of its
/// store.
pub(crate) fn table_ir(ty: Ty) -> Type {
    use zyntax_builtins::dicts;
    match ty {
        Ty::Dict(k) => match types::dict_store(k) {
            None => dicts::dict_type(list_type_id()),
            Some(i) => {
                let (key, value) = types::dict_stores()[i as usize];
                zyntax_builtins::list_of(
                    list_type_id(),
                    dicts::dict_entry_type(&field_of(key), &field_of(value)),
                )
            }
        },
        Ty::Set(k) => match types::set_store(k) {
            (_, Ty::Object) => dicts::set_type(list_type_id()),
            (_, stored) => {
                zyntax_builtins::list_of(list_type_id(), dicts::set_entry_type(&field_of(stored)))
            }
        },
        other => unreachable!("a dict or set type, not {other:?}"),
    }
}

/// How a key of type `key` meets a table whose keys are stored as
/// `stored`, by the rule values of two kinds are equal by.
enum Probe {
    /// As the stored kind: the typed lookup.
    Typed(Node),
    /// Equal to no stored key; the key is still evaluated.
    Absent(Val),
    /// As a dynamic value: the lookup that applies the rule at run time.
    Any(Node),
}

/// A value the IR does not need, evaluated for what it does.
fn effect(v: Node, span: Span) -> Stmt {
    TypedNode::new(TypedStatement::Expression(Box::new(v)), Type::Unknown, span)
}

fn bool_lit(b: bool, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Bool(b)),
        Ty::Bool,
        span,
    )
}

fn not(v: Node, span: Span) -> Node {
    node(
        TypedExpression::Unary(zyntax_typed_ast::typed_ast::TypedUnary {
            op: zyntax_typed_ast::UnaryOp::Not,
            operand: Box::new(v),
        }),
        Ty::Bool,
        span,
    )
}

impl Lowerer<'_> {
    /// A dict or set boxed by reference under its store's tag.
    pub(crate) fn box_table(&mut self, v: Val, span: Span) -> Node {
        if frozen_tagged(v.ty) {
            let tag = table_tag(v.ty) as i32;
            return call(
                &set_fn("box_raw", v.ty),
                vec![v.node, int32_lit(tag, span)],
                Ty::Object,
                span,
            );
        }
        call(&table_fn("box", v.ty), vec![v.node], Ty::Object, span)
    }

    /// A dynamic value read as a dict or set of type `target`, checked
    /// to be one of its store (and, for a set, of its frozenness): the
    /// storage itself, never a copy.
    pub(crate) fn unbox_table(&mut self, v: Val, target: Ty, span: Span) -> Node {
        let checked = if frozen_tagged(target) {
            call(
                &set_fn("unbox_tagged", target),
                vec![v.node, int_lit(table_tag(target), span)],
                target,
                span,
            )
        } else {
            call(&table_fn("unbox", target), vec![v.node], target, span)
        };
        let checked = Val {
            node: checked,
            ty: target,
        };
        if self.guards {
            self.guard(checked, span).node
        } else {
            checked.node
        }
    }

    /// A box the lowering itself made of a dict or set of type `target`,
    /// read back unchecked.
    pub(crate) fn raw_table(&mut self, v: Node, target: Ty, span: Span) -> Node {
        call(&table_fn("raw", target), vec![v], target, span)
    }

    /// A dict or set of one type as one of another: the same storage
    /// when their stores agree (a set becoming a frozenset or the other
    /// way is a copy), else a copy converted value by value.
    pub(crate) fn convert_table(&mut self, v: Val, target: Ty, span: Span) -> Node {
        if v.ty == target {
            return v.node;
        }
        if same_store(v.ty, target) {
            if frozen(v.ty) == frozen(target) {
                return v.node;
            }
            return call(&set_fn("copy", target), vec![v.node], target, span);
        }
        let boxed = self.box_table(v, span);
        let converted = Val {
            node: call(&table_fn("from_dyn", target), vec![boxed], target, span),
            ty: target,
        };
        if self.guards {
            self.guard(converted, span).node
        } else {
            converted.node
        }
    }

    /// A value going into a slot of a table stored as `stored`.
    pub(crate) fn table_in(&mut self, v: Val, stored: Ty) -> Node {
        self.coerce(v, stored)
    }

    /// How `key` probes a typed table of type `table` (not the library's
    /// own, which [`Self::dict_key_of`] probes). A key of the stored kind
    /// is typed; a bool is the int it stands for; a string, None or an
    /// instance is never a number or a tuple, nor a number or a tuple a
    /// string; anything else is boxed and meets the stored keys by the
    /// rule the runtime applies.
    fn probe(&mut self, table: Ty, key: Val, span: Span) -> Probe {
        let stored = key_stored(table);
        let never_number = |t: Ty| matches!(t, Ty::Str | Ty::None | Ty::Class(_) | Ty::Bytes);
        match (stored, key.ty) {
            (s, k) if s == k => Probe::Typed(key.node),
            (Ty::Int, Ty::Bool) => Probe::Typed(cast(key.node, Ty::Int, span)),
            (Ty::Float, Ty::Bool) => {
                Probe::Typed(cast(cast(key.node, Ty::Int, span), Ty::Float, span))
            }
            (Ty::Int | Ty::Float | Ty::Tuple(_), k) if never_number(k) => Probe::Absent(key),
            (Ty::Str, Ty::Int | Ty::Float | Ty::Bool | Ty::None | Ty::Tuple(_) | Ty::Class(_)) => {
                Probe::Absent(key)
            }
            _ => Probe::Any(self.coerce(key, Ty::Object)),
        }
    }

    /// `d[k]` read as `ty`.
    pub(crate) fn dict_get(&mut self, d: Val, key: Val, ty: Ty, span: Span) -> Val {
        let (_, sv) = dict_stored(d.ty);
        if sv == Ty::Object && key_stored(d.ty) == Ty::Object {
            let (k, by) = self.dict_key_of(key);
            let value = Val {
                node: call(
                    &format!("zb_dict_get{by}"),
                    vec![d.node, k],
                    Ty::Object,
                    span,
                ),
                ty: Ty::Object,
            };
            return Val {
                node: self.read_as(value, ty, span),
                ty,
            };
        }
        let got = match self.probe(d.ty, key, span) {
            Probe::Typed(k) => call(&dict_fn("get", d.ty), vec![d.node, k], sv, span),
            Probe::Absent(k) => {
                let k = self.coerce(k, Ty::Object);
                call(&dict_fn("get_any", d.ty), vec![d.node, k], sv, span)
            }
            Probe::Any(k) => call(&dict_fn("get_any", d.ty), vec![d.node, k], sv, span),
        };
        Val {
            node: self.read_as(Val { node: got, ty: sv }, ty, span),
            ty,
        }
    }

    /// `d[k] = v`.
    pub(crate) fn dict_set(&mut self, d: Val, key: Val, value: Val, span: Span) -> Node {
        let (sk, sv) = dict_stored(d.ty);
        if sk == Ty::Object && sv == Ty::Object {
            let (k, by) = self.dict_key_of(key);
            let v = self.coerce(value, Ty::Object);
            return call(
                &format!("zb_dict_set{by}"),
                vec![d.node, k, v],
                Ty::None,
                span,
            );
        }
        let k = self.table_in(key, sk);
        let v = self.table_in(value, sv);
        call(&dict_fn("set", d.ty), vec![d.node, k, v], Ty::None, span)
    }

    /// `del d[k]`.
    pub(crate) fn dict_del(&mut self, d: Val, key: Val, span: Span) -> Node {
        if dict_stored(d.ty) == (Ty::Object, Ty::Object) {
            let k = self.coerce(key, Ty::Object);
            return call("zb_dict_del", vec![d.node, k], Ty::None, span);
        }
        match self.probe(d.ty, key, span) {
            Probe::Typed(k) => call(&dict_fn("del", d.ty), vec![d.node, k], Ty::None, span),
            Probe::Absent(k) => {
                let k = self.coerce(k, Ty::Object);
                call(&dict_fn("del_any", d.ty), vec![d.node, k], Ty::None, span)
            }
            Probe::Any(k) => call(&dict_fn("del_any", d.ty), vec![d.node, k], Ty::None, span),
        }
    }

    /// `k in t` for a dict or set `t`.
    pub(crate) fn table_contains(&mut self, table: Val, key: Val, span: Span) -> Node {
        if key_stored(table.ty) == Ty::Object {
            return match table.ty {
                Ty::Dict(_) => {
                    let (k, by) = self.dict_key_of(key);
                    call(
                        &format!("zb_dict_contains{by}"),
                        vec![table.node, k],
                        Ty::Bool,
                        span,
                    )
                }
                // A tuple of known shape probes by its fields, no box.
                _ => match key.ty {
                    Ty::Tuple(k) => call(
                        &format!("zb_set_contains_{}", types::tuple_suffix(k)),
                        vec![table.node, key.node],
                        Ty::Bool,
                        span,
                    ),
                    _ => {
                        let item = self.coerce(key, Ty::Object);
                        call("zb_set_contains", vec![table.node, item], Ty::Bool, span)
                    }
                },
            };
        }
        let ty = table.ty;
        match self.probe(ty, key, span) {
            Probe::Typed(k) => call(
                &table_fn("contains", ty),
                vec![table.node, k],
                Ty::Bool,
                span,
            ),
            Probe::Absent(k) => {
                let pre = vec![effect(table.node, span), effect(k.node, span)];
                Self::block_value(pre, bool_lit(false, span), Ty::Bool, span)
            }
            Probe::Any(k) => call(
                &table_fn("contains_any", ty),
                vec![table.node, k],
                Ty::Bool,
                span,
            ),
        }
    }

    /// `d.get(k)` and `d.get(k, default)`, read as `ty`: the value as
    /// it is stored when the default is of its kind, else boxed.
    pub(crate) fn dict_get_default(
        &mut self,
        d: Val,
        key: Val,
        default: Val,
        ty: Ty,
        span: Span,
    ) -> Val {
        let (sk, sv) = dict_stored(d.ty);
        if sk == Ty::Object && sv == Ty::Object {
            let (k, by) = self.dict_key_of(key);
            let default = self.coerce(default, Ty::Object);
            let got = call(
                &format!("zb_dict_get_default{by}"),
                vec![d.node, k, default],
                Ty::Object,
                span,
            );
            return Val {
                node: self.read_as(
                    Val {
                        node: got,
                        ty: Ty::Object,
                    },
                    ty,
                    span,
                ),
                ty,
            };
        }
        let typed_default = default.ty == sv && sv != Ty::Object;
        let got = match self.probe(d.ty, key, span) {
            Probe::Typed(k) if typed_default => Val {
                node: call(
                    &dict_fn("get_default", d.ty),
                    vec![d.node, k, default.node],
                    sv,
                    span,
                ),
                ty: sv,
            },
            Probe::Typed(k) => {
                let default = self.coerce(default, Ty::Object);
                Val {
                    node: call(
                        &dict_fn("get_boxed", d.ty),
                        vec![d.node, k, default],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Object,
                }
            }
            Probe::Absent(k) => {
                let pre = vec![effect(d.node, span), effect(k.node, span)];
                Val {
                    node: Self::block_value(pre, default.node, default.ty, span),
                    ty: default.ty,
                }
            }
            Probe::Any(k) => {
                // Found through the rule the runtime applies, else the
                // default: two lookups on a path no typed key takes.
                let mut pre = Vec::new();
                let held_d = self.hold(d, &mut pre, span);
                let held_k = self.hold(
                    Val {
                        node: k,
                        ty: Ty::Object,
                    },
                    &mut pre,
                    span,
                );
                let default = self.coerce(default, Ty::Object);
                let found = call(
                    &dict_fn("contains_any", held_d.ty),
                    vec![held_d.node.clone(), held_k.node.clone()],
                    Ty::Bool,
                    span,
                );
                let value = Val {
                    node: call(
                        &dict_fn("get_any", held_d.ty),
                        vec![held_d.node, held_k.node],
                        sv,
                        span,
                    ),
                    ty: sv,
                };
                let value = self.coerce(value, Ty::Object);
                let picked = node(
                    TypedExpression::If(zyntax_typed_ast::typed_ast::TypedIfExpr {
                        condition: Box::new(found),
                        then_branch: Box::new(value),
                        else_branch: Box::new(default),
                    }),
                    Ty::Object,
                    span,
                );
                Val {
                    node: Self::block_value(pre, picked, Ty::Object, span),
                    ty: Ty::Object,
                }
            }
        };
        Val {
            node: self.read_as(got, ty, span),
            ty,
        }
    }

    /// `d.pop(k)` and `d.pop(k, default)`, read as `ty`.
    pub(crate) fn dict_pop(
        &mut self,
        d: Val,
        key: Val,
        default: Option<Val>,
        ty: Ty,
        span: Span,
    ) -> Val {
        let (sk, sv) = dict_stored(d.ty);
        if sk == Ty::Object && sv == Ty::Object {
            let k = self.coerce(key, Ty::Object);
            let got = match default {
                None => call("zb_dict_pop", vec![d.node, k], Ty::Object, span),
                Some(default) => {
                    let default = self.coerce(default, Ty::Object);
                    call(
                        "zb_dict_pop_default",
                        vec![d.node, k, default],
                        Ty::Object,
                        span,
                    )
                }
            };
            return Val {
                node: self.read_as(
                    Val {
                        node: got,
                        ty: Ty::Object,
                    },
                    ty,
                    span,
                ),
                ty,
            };
        }
        // A key of another kind is found by the runtime's rule, and
        // removed through the same lookup: the value first, then the
        // entry.
        let key = match self.probe(d.ty, key, span) {
            Probe::Typed(k) => Ok(k),
            Probe::Absent(k) => Err(self.coerce(k, Ty::Object)),
            Probe::Any(k) => Err(k),
        };
        let got = match (key, default) {
            (Ok(k), None) => Val {
                node: call(&dict_fn("pop", d.ty), vec![d.node, k], sv, span),
                ty: sv,
            },
            (Ok(k), Some(default)) if default.ty == sv && sv != Ty::Object => Val {
                node: call(
                    &dict_fn("pop_default", d.ty),
                    vec![d.node, k, default.node],
                    sv,
                    span,
                ),
                ty: sv,
            },
            (Ok(k), Some(default)) => {
                let default = self.coerce(default, Ty::Object);
                Val {
                    node: call(
                        &dict_fn("pop_boxed", d.ty),
                        vec![d.node, k, default],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Object,
                }
            }
            (Err(k), default) => {
                let mut pre = Vec::new();
                let held_d = self.hold(d, &mut pre, span);
                let held_k = self.hold(
                    Val {
                        node: k,
                        ty: Ty::Object,
                    },
                    &mut pre,
                    span,
                );
                let read = |this: &mut Self| {
                    let value = Val {
                        node: call(
                            &dict_fn("get_any", held_d.ty),
                            vec![held_d.node.clone(), held_k.node.clone()],
                            sv,
                            span,
                        ),
                        ty: sv,
                    };
                    let value = this.coerce(value, Ty::Object);
                    let name = this.temp();
                    let mut body = vec![TypedNode::new(
                        TypedStatement::Let(TypedLet {
                            name,
                            ty: Type::Any,
                            mutability: Mutability::Immutable,
                            initializer: Some(Box::new(value)),
                            span,
                        }),
                        Type::Unknown,
                        span,
                    )];
                    body.push(effect(
                        call(
                            &dict_fn("del_any", held_d.ty),
                            vec![held_d.node.clone(), held_k.node.clone()],
                            Ty::None,
                            span,
                        ),
                        span,
                    ));
                    Self::block_value(body, var(name, Ty::Object, span), Ty::Object, span)
                };
                let value = match default {
                    None => read(self),
                    Some(default) => {
                        let default = self.coerce(default, Ty::Object);
                        let found = call(
                            &dict_fn("contains_any", held_d.ty),
                            vec![held_d.node.clone(), held_k.node.clone()],
                            Ty::Bool,
                            span,
                        );
                        let value = read(self);
                        node(
                            TypedExpression::If(zyntax_typed_ast::typed_ast::TypedIfExpr {
                                condition: Box::new(found),
                                then_branch: Box::new(value),
                                else_branch: Box::new(default),
                            }),
                            Ty::Object,
                            span,
                        )
                    }
                };
                Val {
                    node: Self::block_value(pre, value, Ty::Object, span),
                    ty: Ty::Object,
                }
            }
        };
        Val {
            node: self.read_as(got, ty, span),
            ty,
        }
    }

    /// `d.setdefault(k, v)`, read as `ty`.
    pub(crate) fn dict_setdefault(
        &mut self,
        d: Val,
        key: Val,
        value: Val,
        ty: Ty,
        span: Span,
    ) -> Val {
        let (sk, sv) = dict_stored(d.ty);
        let k = self.table_in(key, sk);
        let v = self.table_in(value, sv);
        let got = call(&dict_fn("setdefault", d.ty), vec![d.node, k, v], sv, span);
        Val {
            node: self.read_as(Val { node: got, ty: sv }, ty, span),
            ty,
        }
    }

    /// The number of entries of a dict or set.
    pub(crate) fn table_len(&mut self, v: Val, span: Span) -> Node {
        call(&table_fn("len", v.ty), vec![v.node], Ty::Int, span)
    }

    /// A dict or set as `repr` prints it; a frozenset as
    /// `frozenset({..})`.
    pub(crate) fn table_repr(&mut self, v: Val, span: Span) -> Node {
        if !frozen(v.ty) {
            return call(&table_fn("repr", v.ty), vec![v.node], Ty::Str, span);
        }
        let mut pre = Vec::new();
        let held = self.hold(v, &mut pre, span);
        let empty = binary(
            BinaryOp::Eq,
            self.table_len(held.clone(), span),
            int_lit(0, span),
            Ty::Bool,
            span,
        );
        let text = call(&set_fn("repr", held.ty), vec![held.node], Ty::Str, span);
        let wrapped = binary(
            BinaryOp::Add,
            binary(
                BinaryOp::Add,
                str_lit("frozenset(", span),
                text,
                Ty::Str,
                span,
            ),
            str_lit(")", span),
            Ty::Str,
            span,
        );
        let picked = node(
            TypedExpression::If(zyntax_typed_ast::typed_ast::TypedIfExpr {
                condition: Box::new(empty),
                then_branch: Box::new(str_lit("frozenset()", span)),
                else_branch: Box::new(wrapped),
            }),
            Ty::Str,
            span,
        );
        Self::block_value(pre, picked, Ty::Str, span)
    }

    /// The keys of a dict or the elements of a set as a list, of the
    /// kind the store lists them as: numbers and strings unboxed,
    /// anything else as dynamic values.
    pub(crate) fn table_items(&mut self, v: Val, span: Span) -> Val {
        let listed = match key_stored(v.ty) {
            t @ (Ty::Int | Ty::Float | Ty::Str) => Ty::List(Elem::of(t)),
            _ => Ty::List(Elem::Object),
        };
        let op = match v.ty {
            Ty::Dict(_) => "keys",
            _ => "items",
        };
        Val {
            node: call(&table_fn(op, v.ty), vec![v.node], listed, span),
            ty: listed,
        }
    }

    /// The values of a dict as a list, as [`Self::table_items`] lists.
    pub(crate) fn dict_values(&mut self, v: Val, span: Span) -> Val {
        let listed = match dict_stored(v.ty).1 {
            t @ (Ty::Int | Ty::Float | Ty::Str) => Ty::List(Elem::of(t)),
            _ => Ty::List(Elem::Object),
        };
        Val {
            node: call(&dict_fn("values", v.ty), vec![v.node], listed, span),
            ty: listed,
        }
    }

    /// Element `i` of a set by position, as `iter_len` left it, of the
    /// kind [`Self::table_items`] lists it as.
    pub(crate) fn set_iter_at(&mut self, s: Val, i: Node, span: Span) -> Val {
        let at = match set_stored(s.ty) {
            t @ (Ty::Int | Ty::Float | Ty::Str) => t,
            _ => Ty::Object,
        };
        Val {
            node: call(&set_fn("iter_at", s.ty), vec![s.node, i], at, span),
            ty: at,
        }
    }

    /// `s.remove(x)` (`remove`) or `s.discard(x)`: a value of a kind no
    /// element is equal to is a missing value.
    pub(crate) fn set_discard(&mut self, s: Val, item: Val, remove: bool, span: Span) -> Node {
        let op = if remove { "remove" } else { "discard" };
        if set_stored(s.ty) == Ty::Object {
            let item = self.coerce(item, Ty::Object);
            return call(&set_fn(op, s.ty), vec![s.node, item], Ty::None, span);
        }
        let ty = s.ty;
        match self.probe(ty, item, span) {
            Probe::Typed(k) => call(&set_fn(op, ty), vec![s.node, k], Ty::None, span),
            Probe::Absent(k) if !remove => {
                let pre = vec![effect(s.node, span), effect(k.node, span)];
                let none = node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span);
                Self::block_value(pre, none, Ty::None, span)
            }
            Probe::Absent(k) => {
                let k = self.coerce(k, Ty::Object);
                call(&set_fn("remove_any", ty), vec![s.node, k], Ty::None, span)
            }
            Probe::Any(k) => call(
                &set_fn(&format!("{op}_any"), ty),
                vec![s.node, k],
                Ty::None,
                span,
            ),
        }
    }

    /// An empty dict or set of type `ty`.
    pub(crate) fn new_table(&mut self, ty: Ty, span: Span) -> Node {
        call(&table_fn("new", ty), vec![], ty, span)
    }

    /// `s.add(x)` into a set of type `s.ty`.
    pub(crate) fn set_add(&mut self, s: Val, item: Val, span: Span) -> Node {
        let stored = set_stored(s.ty);
        let item = self.table_in(item, stored);
        call(&set_fn("add", s.ty), vec![s.node, item], Ty::None, span)
    }

    /// A dict literal of type `ty`. Keys that are distinct string
    /// literals, or distinct int literals, need no search for an earlier
    /// equal key: the literal lays out the dict's entries itself, each
    /// hash left zero for the index to fill if the dict takes one. Any
    /// other literal stores its pairs in order, later ones winning.
    pub(crate) fn dict_display(&mut self, d: &py::ExprDict, ty: Ty, span: Span) -> Result<Val> {
        let (sk, sv) = dict_stored(ty);
        let mut items = Vec::with_capacity(d.items.len());
        for item in &d.items {
            let Some(key) = &item.key else {
                return super::unsupported("`**` in a dict literal", d);
            };
            items.push((key, &item.value));
        }
        if distinct_literal_keys(d) && (sk == Ty::Object || distinct_kind_fits(d, sk)) {
            let entry_ty = zyntax_builtins::dicts::dict_entry_type(&field_of(sk), &field_of(sv));
            let mut entries = Vec::with_capacity(items.len());
            for (key, value) in items {
                let k = self.expr(key)?;
                let k = self.table_in(k, sk);
                let v = self.expr(value)?;
                let v = self.table_in(v, sv);
                entries.push(TypedNode::new(
                    TypedExpression::Tuple(vec![int_lit(0, span), k, v]),
                    entry_ty.clone(),
                    span,
                ));
            }
            let storage = TypedNode::new(TypedExpression::Array(entries), table_ir(ty), span);
            return Ok(Val {
                node: call(&dict_fn("from_distinct", ty), vec![storage], ty, span),
                ty,
            });
        }
        if sk == Ty::Object && sv == Ty::Object {
            let mut flat = Vec::with_capacity(items.len() * 2);
            for (key, value) in items {
                flat.push(self.expr(key)?);
                flat.push(self.expr(value)?);
            }
            let pairs = self.list_of(flat, Elem::Object, span);
            return Ok(Val {
                node: call("zb_dict_from_pairs", vec![pairs], ty, span),
                ty,
            });
        }
        // Each pair stored as it is evaluated, as Python does.
        let outer = std::mem::take(&mut self.hoisted);
        let mut pre = Vec::new();
        let new = self.new_table(ty, span);
        let held = self.hold(Val { node: new, ty }, &mut pre, span);
        for (key, value) in items {
            let k = self.expr(key)?;
            let v = self.expr(value)?;
            pre.append(&mut self.hoisted);
            let store = self.dict_set(held.clone(), k, v, span);
            pre.append(&mut self.hoisted);
            pre.push(effect(store, span));
        }
        self.hoisted = outer;
        Ok(Val {
            node: Self::block_value(pre, held.node, ty, span),
            ty,
        })
    }

    /// A set literal of type `ty`, its elements added in order.
    pub(crate) fn set_display(&mut self, s: &py::ExprSet, ty: Ty, span: Span) -> Result<Val> {
        if set_stored(ty) == Ty::Object && !frozen(ty) {
            let mut items = Vec::with_capacity(s.elts.len());
            for e in &s.elts {
                items.push(self.expr(e)?);
            }
            let elements = self.list_of(items, Elem::Object, span);
            return Ok(Val {
                node: call("zb_set_from", vec![elements], ty, span),
                ty,
            });
        }
        let outer = std::mem::take(&mut self.hoisted);
        let mut pre = Vec::new();
        let new = self.new_table(ty, span);
        let held = self.hold(Val { node: new, ty }, &mut pre, span);
        for e in &s.elts {
            let v = self.expr(e)?;
            pre.append(&mut self.hoisted);
            let add = self.set_add(held.clone(), v, span);
            pre.append(&mut self.hoisted);
            pre.push(effect(add, span));
        }
        self.hoisted = outer;
        Ok(Val {
            node: Self::block_value(pre, held.node, ty, span),
            ty,
        })
    }

    /// A new set of type `ty` holding what iterating `v` yields: a copy
    /// of a set, the elements of a list of the stored kind at once, and
    /// anything else through its items.
    pub(crate) fn set_from(&mut self, v: Val, ty: Ty, span: Span) -> Node {
        let stored = set_stored(ty);
        match v.ty {
            Ty::Set(_) if same_store(v.ty, ty) => call(&set_fn("copy", ty), vec![v.node], ty, span),
            Ty::Set(_) | Ty::Dict(_) if stored != Ty::Object => self.convert_items(v, ty, span),
            _ => {
                let listed = matches!(stored, Ty::Int | Ty::Float | Ty::Str | Ty::Object);
                let items = match v.ty {
                    Ty::List(e) if listed && e.ty() == stored && e.code().is_none() => v,
                    Ty::List(_) | Ty::Tuple(_) | Ty::Set(_) | Ty::Dict(_) | Ty::Str | Ty::Gen => {
                        Val {
                            node: self.iterable(v, span),
                            ty: Ty::List(Elem::Object),
                        }
                    }
                    _ => {
                        let o = self.coerce(v, Ty::Object);
                        Val {
                            node: call("zb_any_iter", vec![o], Ty::List(Elem::Object), span),
                            ty: Ty::List(Elem::Object),
                        }
                    }
                };
                match stored {
                    Ty::Object => call("zb_set_from", vec![items.node], ty, span),
                    Ty::Int | Ty::Float | Ty::Str => {
                        let typed = Ty::List(Elem::of(stored));
                        let items = self.coerce(items, typed);
                        call(&set_fn("from_list", ty), vec![items], ty, span)
                    }
                    _ => {
                        let library = types::dynamic_set(false);
                        let set = Val {
                            node: call("zb_set_from", vec![items.node], library, span),
                            ty: library,
                        };
                        self.convert_items(set, ty, span)
                    }
                }
            }
        }
    }

    /// A new dict or set of type `ty` from the entries of another of
    /// any store, each converted.
    fn convert_items(&mut self, v: Val, ty: Ty, span: Span) -> Node {
        let boxed = self.coerce(v, Ty::Object);
        let converted = Val {
            node: call(&table_fn("from_dyn", ty), vec![boxed], ty, span),
            ty,
        };
        if self.guards {
            self.guard(converted, span).node
        } else {
            converted.node
        }
    }

    /// `a op b` for two sets, as the set type `ty` inference gave the
    /// result: each operand in the result's store, converted when its
    /// own is another.
    pub(crate) fn set_arith(&mut self, op: &str, a: Val, b: Val, ty: Ty, span: Span) -> Node {
        let a = if same_store(a.ty, ty) {
            a.node
        } else {
            self.convert_items(a, ty, span)
        };
        let b = if same_store(b.ty, ty) {
            b.node
        } else {
            self.convert_items(b, ty, span)
        };
        call(&set_fn(op, ty), vec![a, b], ty, span)
    }

    /// Two sets compared by inclusion or equality: in one store, the
    /// left's when the right shares it, else the store of both kinds.
    pub(crate) fn set_compare(
        &mut self,
        op: py::CmpOp,
        a: Val,
        b: Val,
        span: Span,
    ) -> Option<Node> {
        let (Ty::Set(x), Ty::Set(y)) = (a.ty, b.ty) else {
            return None;
        };
        let shape = if same_store(a.ty, b.ty) {
            a.ty
        } else {
            types::set_of(types::set_shape(x).0.join(types::set_shape(y).0), false)
        };
        let mut pre = Vec::new();
        let a = if same_store(a.ty, shape) {
            a
        } else {
            Val {
                node: self.convert_items(a, shape, span),
                ty: shape,
            }
        };
        let b = if same_store(b.ty, shape) {
            b
        } else {
            Val {
                node: self.convert_items(b, shape, span),
                ty: shape,
            }
        };
        let a = self.hold(a, &mut pre, span);
        let b = self.hold(b, &mut pre, span);
        let f = |op: &str| set_fn(op, shape);
        let subset = |x: &Val, y: &Val| {
            call(
                &f("issubset"),
                vec![x.node.clone(), y.node.clone()],
                Ty::Bool,
                span,
            )
        };
        let len = |x: &Val| call(&f("len"), vec![x.node.clone()], Ty::Int, span);
        let proper = |x: &Val, y: &Val| {
            binary(
                BinaryOp::And,
                binary(BinaryOp::Lt, len(x), len(y), Ty::Bool, span),
                subset(x, y),
                Ty::Bool,
                span,
            )
        };
        let equal = call(
            &f("eq"),
            vec![a.node.clone(), b.node.clone()],
            Ty::Bool,
            span,
        );
        let value = match op {
            py::CmpOp::LtE => subset(&a, &b),
            py::CmpOp::GtE => subset(&b, &a),
            py::CmpOp::Lt => proper(&a, &b),
            py::CmpOp::Gt => proper(&b, &a),
            py::CmpOp::Eq => equal,
            py::CmpOp::NotEq => not(equal, span),
            _ => return None,
        };
        Some(Self::block_value(pre, value, Ty::Bool, span))
    }

    /// Two dicts compared for equality.
    pub(crate) fn dict_equal(&mut self, a: Val, b: Val, span: Span) -> Node {
        if same_store(a.ty, b.ty) {
            return call(&dict_fn("eq", a.ty), vec![a.node, b.node], Ty::Bool, span);
        }
        let other = self.coerce(b, Ty::Object);
        call(
            &dict_fn("eq_any", a.ty),
            vec![a.node, other],
            Ty::Bool,
            span,
        )
    }

    /// A set method whose other operand is any iterable: that operand as
    /// a set of the receiver's store, the receiver's own storage when it
    /// is one already.
    pub(crate) fn as_set_of(&mut self, v: Val, ty: Ty, span: Span) -> Node {
        if matches!(v.ty, Ty::Set(_)) && same_store(v.ty, ty) {
            return v.node;
        }
        self.set_from(v, ty, span)
    }
}

/// Whether every key of a dict literal is a string literal, no two
/// equal, or every key an int literal (not a bool, a float or a negated
/// literal), no two equal: its pairs are then the dict's pairs as
/// written.
pub(crate) fn distinct_literal_keys(d: &py::ExprDict) -> bool {
    let mut texts: Vec<String> = Vec::with_capacity(d.items.len());
    let mut ints: Vec<i64> = Vec::with_capacity(d.items.len());
    for item in &d.items {
        match &item.key {
            Some(py::Expr::StringLiteral(s)) if ints.is_empty() => {
                let text = s.value.to_str().to_string();
                if texts.contains(&text) {
                    return false;
                }
                texts.push(text);
            }
            Some(py::Expr::NumberLiteral(n)) if texts.is_empty() => {
                let py::Number::Int(i) = &n.value else {
                    return false;
                };
                let Some(i) = i.as_i64() else {
                    return false;
                };
                if ints.contains(&i) {
                    return false;
                }
                ints.push(i);
            }
            _ => return false,
        }
    }
    true
}

/// Whether the distinct literal keys of `d` are of the kind its dict
/// stores its keys as.
fn distinct_kind_fits(d: &py::ExprDict, stored: Ty) -> bool {
    match d.items.first().and_then(|i| i.key.as_ref()) {
        Some(py::Expr::StringLiteral(_)) => stored == Ty::Str,
        Some(py::Expr::NumberLiteral(_)) => stored == Ty::Int,
        _ => true,
    }
}
