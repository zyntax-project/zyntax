//! Operators support.

use super::*;

/// The metamethod an arithmetic operator dispatches to, when an
/// operand is a table.
pub fn arith_event(op: &BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Plus(_) => "__add",
        BinOp::Minus(_) => "__sub",
        BinOp::Star(_) => "__mul",
        BinOp::Slash(_) => "__div",
        BinOp::Percent(_) => "__mod",
        BinOp::Caret(_) => "__pow",
        BinOp::DoubleSlash(_) => "__idiv",
        _ => return None,
    })
}

/// The metamethods the typed dispatch handles: a table operand's
/// handler is called directly when the types know it, so the runtime
/// never reaches it with values of its own. An `__index` handler is
/// called with the table and a constant key, both recorded against
/// it; a read under a key not known at compile time makes it escape.
pub fn typed_event(event: &str) -> bool {
    matches!(
        event,
        "__add" | "__sub" | "__mul" | "__div" | "__mod" | "__pow" | "__idiv" | "__unm" | "__index"
    )
}

/// Whether a value of this type can carry the handler for an
/// arithmetic event of its own: a table can, a string does (the string
/// metatable's, which defers to the other operand's), a number, a
/// boolean or nil cannot.
pub(super) fn may_handle_arith(ty: Ty) -> bool {
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
            | Ty::Str
    )
}

/// The identifier a name token holds.
pub fn ident(token: &full_moon::tokenizer::TokenReference) -> String {
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::Identifier { identifier } => identifier.to_string(),
        _ => token.token().to_string(),
    }
}

/// The type of `a and b` or `a or b`.
pub fn logical_ty(is_and: bool, a: Ty, b: Ty) -> Ty {
    if a == Ty::Unknown || b == Ty::Unknown {
        return Ty::Unknown;
    }
    if is_and {
        if a.always_truthy() {
            b
        } else if a == Ty::Nil {
            Ty::Nil
        } else if a == Ty::Bool && b == Ty::Bool {
            Ty::Bool
        } else {
            a.join(b)
        }
    } else if a.always_truthy() {
        a
    } else if a == Ty::Nil {
        b
    } else if matches!(a, Ty::IntOrNil | Ty::FloatOrNil) {
        // False only when nil: the number it holds, or `b`.
        a.underlying().join(b)
    } else if a == Ty::Bool && b == Ty::Bool {
        Ty::Bool
    } else {
        a.join(b)
    }
}

pub fn binary_ty(op: &BinOp, a: Ty, b: Ty) -> Ty {
    // Nothing known on one side settles nothing yet, except that a
    // comparison is a boolean whatever its operands.
    let comparison = matches!(
        op,
        BinOp::TwoEqual(_)
            | BinOp::TildeEqual(_)
            | BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_)
    );
    if (a == Ty::Unknown || b == Ty::Unknown) && !comparison {
        return Ty::Unknown;
    }
    match op {
        BinOp::Plus(_)
        | BinOp::Minus(_)
        | BinOp::Star(_)
        | BinOp::Percent(_)
        | BinOp::DoubleSlash(_) => {
            // A float operand makes a float; two integers an integer;
            // a number whose kind is not known keeps it open. A scalar
            // that may not be a number raises, or is a number; one that
            // may only be nil besides raises, or is its number.
            let (a, b) = if a.underlying().is_number() && b.underlying().is_number() {
                (a.underlying(), b.underlying())
            } else {
                (a, b)
            };
            if !(a.is_number() && b.is_number()) {
                if a.is_scalar() && b.is_scalar() {
                    Ty::Number
                } else {
                    Ty::Any
                }
            } else if a == Ty::Int && b == Ty::Int {
                Ty::Int
            } else if a == Ty::Float || b == Ty::Float {
                Ty::Float
            } else {
                Ty::Number
            }
        }
        BinOp::Slash(_) | BinOp::Caret(_) => {
            if a.is_scalar() && b.is_scalar() {
                Ty::Float
            } else {
                Ty::Any
            }
        }
        BinOp::Ampersand(_)
        | BinOp::Pipe(_)
        | BinOp::Tilde(_)
        | BinOp::DoubleLessThan(_)
        | BinOp::DoubleGreaterThan(_) => {
            if a.is_number() && b.is_number() {
                Ty::Int
            } else {
                Ty::Any
            }
        }
        BinOp::TwoDots(_) => {
            let text = |t: Ty| t == Ty::Str || t.is_number();
            if text(a) && text(b) { Ty::Str } else { Ty::Any }
        }
        BinOp::TwoEqual(_)
        | BinOp::TildeEqual(_)
        | BinOp::LessThan(_)
        | BinOp::LessThanEqual(_)
        | BinOp::GreaterThan(_)
        | BinOp::GreaterThanEqual(_) => Ty::Bool,
        BinOp::And(_) => logical_ty(true, a, b),
        BinOp::Or(_) => logical_ty(false, a, b),
        _ => Ty::Any,
    }
}

/// Whether the last statement of a block can be run past, so the
/// function returns nothing there.
pub fn falls_through(block: &Block) -> bool {
    match block.last_stmt() {
        Some(ast::LastStmt::Return(_)) => false,
        Some(ast::LastStmt::Break(_)) => false,
        _ => match block.stmts().last() {
            Some(Stmt::If(i)) => {
                let Some(els) = i.else_block() else {
                    return true;
                };
                falls_through(i.block())
                    || i.else_if()
                        .is_some_and(|e| e.iter().any(|e| falls_through(e.block())))
                    || falls_through(els)
            }
            Some(Stmt::Do(d)) => falls_through(d.block()),
            _ => true,
        },
    }
}
