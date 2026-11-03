//! Keys support.

use super::*;

/// What a value of type `ty` is when it is present: nil is nothing, and
/// a number or nil its number.
pub(super) fn present(ty: Ty) -> Ty {
    match ty {
        Ty::Nil => Ty::Unknown,
        t => t.underlying(),
    }
}

/// The join of two things a read may yield, where one not decided yet
/// leaves the read undecided.
pub(super) fn join_read(a: Ty, b: Ty) -> Ty {
    if a == Ty::Unknown || b == Ty::Unknown {
        Ty::Unknown
    } else {
        a.join(b)
    }
}

/// What a library function reads out of the table it takes first and
/// hands to its caller, or to a function it calls, as dynamic values:
/// `pairs` and `next` every field and element, raw; `ipairs`,
/// `table.unpack`, `table.remove` and `table.sort` elements, through
/// `__index`. None for one that hands out nothing of a table.
pub(super) fn library_reads(b: &Builtin) -> Option<Reads> {
    match (b.lib, b.name) {
        ("", "pairs" | "next") => Some(Reads {
            names: true,
            elements: true,
            through_meta: false,
        }),
        ("", "ipairs" | "unpack") | ("table", "unpack" | "remove" | "sort") => {
            Some(Reads::under(Ty::Int, true))
        }
        _ => None,
    }
}

/// Whether a key of this type is a number: then a read under it is
/// of the array part, or of the hash part's numbers.
pub(super) fn key_is_number(ty: Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::Number)
}

/// Whether a key of this type may be a number: then a store under it
/// may land in a shape's array part.
pub(super) fn key_may_be_index(ty: Ty) -> bool {
    !matches!(
        ty,
        Ty::Str | Ty::Bool | Ty::Nil | Ty::Table | Ty::Shape(_) | Ty::Func(_)
    )
}

/// Whether a key of this type may be a string: then a store under it
/// may land on a shape's field.
pub(super) fn key_may_be_string(ty: Ty) -> bool {
    !matches!(
        ty,
        Ty::Int
            | Ty::Float
            | Ty::Number
            | Ty::Scalar
            | Ty::IntOrNil
            | Ty::FloatOrNil
            | Ty::Bool
            | Ty::Nil
            | Ty::Table
            | Ty::Func(_)
            | Ty::Shape(_)
    )
}

/// The constant string an index suffix names: `.name` or `["name"]`.
pub fn constant_key(index: &ast::Index) -> Option<String> {
    match index {
        ast::Index::Dot { name, .. } => Some(ident(name)),
        ast::Index::Brackets { expression, .. } => crate::scope::literal_string(expression),
        _ => None,
    }
}

/// The constant string keys of a constructor, in order, and whether
/// the constructor has other fields too. `None` when it has an entry
/// that is neither a constant key nor positional.
pub fn constructor_keys(t: &ast::TableConstructor) -> Vec<String> {
    let mut keys = Vec::new();
    for field in t.fields() {
        match field {
            ast::Field::NameKey { key, .. } => keys.push(ident(key)),
            ast::Field::ExpressionKey { key, .. } => {
                if let Some(k) = crate::scope::literal_string(key) {
                    keys.push(k);
                }
            }
            _ => {}
        }
    }
    keys
}

/// The byte offset of a constructor's opening brace: what tells one
/// empty constructor from another.
pub fn constructor_site(t: &ast::TableConstructor) -> usize {
    crate::scope::pos_of(t.braces().tokens().0)
}

/// The constructor `e` is, in parentheses or not.
pub(super) fn constructor_in(e: &Expression) -> Option<&ast::TableConstructor> {
    match e {
        Expression::TableConstructor(t) => Some(t),
        Expression::Parentheses { expression, .. } => constructor_in(expression),
        _ => None,
    }
}

/// The shape a constructor's tables have, if one is known.
pub fn constructor_shape(known: &Inferred, t: &ast::TableConstructor) -> Option<ShapeId> {
    let keys = constructor_keys(t);
    if keys.is_empty() {
        known.shape_by_site.get(&constructor_site(t)).copied()
    } else {
        let mut sorted = keys;
        sorted.sort();
        sorted.dedup();
        known.shape_by_keys.get(&sorted).copied()
    }
}

/// What an expression in last position of a list supplies when it may
/// supply several values: a call's returns, or every vararg. `None`
/// for an expression of one value.
pub fn multi_returns(typer: &Typer<'_>, e: &Expression) -> Option<Returns> {
    match e {
        Expression::FunctionCall(c) => {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            Some(typer.call_returns(c.prefix(), &suffixes))
        }
        Expression::Var(Var::Expression(v)) => {
            let suffixes: Vec<&Suffix> = v.suffixes().collect();
            if matches!(suffixes.last(), Some(Suffix::Call(_))) {
                Some(typer.call_returns(v.prefix(), &suffixes))
            } else {
                None
            }
        }
        Expression::Symbol(t) if t.token().to_string().trim() == "..." => Some(Returns::Dynamic),
        _ => None,
    }
}
