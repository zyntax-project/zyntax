//! Constructors support.

use super::*;

impl<'a> Round<'a> {
    /// Whether `pairs(t)` or `ipairs(t)`, called by `prefix`, heads a
    /// generic `for` whose bindings are typed from `t`'s shape.
    pub(super) fn typed_walk(&self, b: &Builtin, prefix: &Prefix, table: Ty) -> bool {
        let header = match prefix {
            Prefix::Name(token) => self.loop_iterator == Some(crate::scope::pos_of(token)),
            _ => false,
        };
        let value = match b.name {
            "ipairs" => self.known.ipairs_value_ty(table),
            _ => self.known.pairs_value_ty(table),
        };
        header && matches!(table, Ty::Shape(_)) && value != Ty::Any
    }

    pub(super) fn prefix(&mut self, p: &Prefix) {
        if let Prefix::Expression(e) = p {
            self.expr(e);
        }
    }

    pub(super) fn args(&mut self, args: &ast::FunctionArgs, kept: bool) {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => {
                for a in arguments {
                    if kept {
                        self.expr(a);
                    } else {
                        self.value(a);
                    }
                }
            }
            ast::FunctionArgs::TableConstructor(t) => self.table(t),
            _ => {}
        }
    }

    /// A constructor: its tables have a shape, a constant-key field's
    /// value joins the shape's field, and a positional or other keyed
    /// value stores as through an index.
    pub(super) fn table(&mut self, t: &ast::TableConstructor) {
        let shape = self.shape_for(t);
        let info = &mut self.out.shapes[shape.0 as usize];
        info.sites.insert(constructor_site(t));
        if self.func != CHUNK || self.loops > 0 {
            info.remade = true;
        }
        let count = t.fields().len();
        for (i, field) in t.fields().iter().enumerate() {
            match field {
                ast::Field::ExpressionKey { key, value, .. } => {
                    match crate::scope::literal_string(key) {
                        Some(name) => {
                            self.expr(value);
                            let ty = self.typer().ty_of(value);
                            let non_nil = self.typer().non_nil(value);
                            self.store_field(shape, &name, ty, non_nil);
                        }
                        // Any other key stores as `t[k] = v` does: into
                        // the element, dyn_value or named fields its
                        // type may reach.
                        None => {
                            self.value(key);
                            if self.known.plain(shape) {
                                self.expr(value);
                            } else {
                                self.value(value);
                            }
                            let key_ty = self.typer().ty_of(key);
                            let ty = self.typer().ty_of(value);
                            self.store_through(Ty::Shape(shape), None, key_ty, ty, false);
                        }
                    }
                }
                ast::Field::NameKey { key, value, .. } => {
                    self.expr(value);
                    let ty = self.typer().ty_of(value);
                    let non_nil = self.typer().non_nil(value);
                    self.store_field(shape, &ident(key), ty, non_nil);
                }
                // A positional value is an element; the last one's
                // several values all are.
                ast::Field::NoKey(e) => {
                    self.expr(e);
                    let last = i + 1 == count;
                    let types = match multi_returns(&self.typer(), e) {
                        Some(Returns::Fixed(types)) if last => types,
                        Some(Returns::Dynamic) if last => vec![Ty::Any],
                        Some(Returns::Unknown) if last => vec![Ty::Unknown],
                        _ => vec![self.typer().ty_of(e)],
                    };
                    for ty in types {
                        self.store_element(shape, ty);
                    }
                }
                _ => {}
            }
        }
    }

    /// Walk an expression for the functions and calls inside it, in a
    /// place that keeps the value's type: assigned to a variable,
    /// returned, or passed to a known function.
    pub(super) fn expr(&mut self, e: &Expression) {
        match e {
            Expression::BinaryOperator { lhs, binop, rhs } => {
                // Arithmetic the types dispatch to a handler passes
                // the operands to it; anything else takes values.
                let a = self.typer().ty_of(lhs);
                let b = self.typer().ty_of(rhs);
                let sides =
                    arith_event(binop).and_then(|event| self.typer().metamethod_sides(event, a, b));
                match sides {
                    Some(sides) => {
                        self.expr(lhs);
                        self.expr(rhs);
                        for (_, _, f) in sides {
                            self.record_call_types(f, vec![a, b]);
                        }
                    }
                    None => {
                        self.value(lhs);
                        self.value(rhs);
                    }
                }
            }
            Expression::Parentheses { expression, .. } => self.expr(expression),
            Expression::UnaryOperator { unop, expression } => {
                let t = self.typer().ty_of(expression);
                let sides = match unop {
                    UnOp::Minus(_) => self.typer().metamethod_sides("__unm", t, t),
                    _ => None,
                };
                match sides {
                    Some(sides) => {
                        self.expr(expression);
                        for (_, _, f) in sides {
                            self.record_call_types(f, vec![t, t]);
                        }
                    }
                    // `#t` with no `__len` to call looks at the table.
                    None if matches!(unop, UnOp::Hash(_))
                        && matches!(t, Ty::Shape(k) if self.known.plain_len(k)) =>
                    {
                        self.expr(expression)
                    }
                    None if matches!(unop, UnOp::Not(_)) => self.condition(expression),
                    None => self.value(expression),
                }
            }
            Expression::Function(f) => self.function(f.body()),
            Expression::FunctionCall(c) => self.call(c),
            Expression::TableConstructor(t) => self.table(t),
            Expression::Var(Var::Expression(v)) => {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.chain(v.prefix(), &suffixes);
            }
            Expression::Var(Var::Name(token)) => {
                if let Some(Binding::Global(name)) = self.scopes.binding(token)
                    && self.is_reader(name)
                {
                    self.loose_reader();
                }
            }
            _ => {}
        }
    }
}
