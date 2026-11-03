//! Builtins support.

use super::*;

/// What the library function `b` does with its table arguments.
pub(super) fn borrows(b: &Builtin) -> Borrows {
    match (b.lib, b.name) {
        ("", "rawlen" | "rawequal" | "next") => Borrows::Raw,
        ("", "unpack") | ("table", "concat" | "unpack") => Borrows::Indexed,
        ("", "pairs" | "ipairs") => Borrows::Walk,
        ("", "select") => Borrows::Count,
        _ => Borrows::No,
    }
}

/// The builtin a global name is, when the program never assigns it.
pub fn builtin_named(scopes: &Scopes, name: &str) -> Option<&'static Builtin> {
    if scopes.global_writes.contains_key(name) {
        return None;
    }
    builtin_global(name)
}

/// Whether `e` is never nil: a function, a table, a literal other than
/// nil, a comparison or `not`, a local in `vars`, or `setmetatable` of
/// one of these, which returns its first argument.
pub fn non_nil(scopes: &Scopes, vars: &HashSet<VarId>, e: &Expression) -> bool {
    match e {
        Expression::Function(_)
        | Expression::TableConstructor(_)
        | Expression::Number(_)
        | Expression::String(_) => true,
        Expression::Symbol(t) => matches!(t.token().to_string().trim(), "true" | "false"),
        Expression::Parentheses { expression, .. } => non_nil(scopes, vars, expression),
        Expression::UnaryOperator { unop, .. } => matches!(unop, UnOp::Not(_)),
        Expression::BinaryOperator { binop, .. } => matches!(
            binop,
            BinOp::TwoEqual(_)
                | BinOp::TildeEqual(_)
                | BinOp::LessThan(_)
                | BinOp::LessThanEqual(_)
                | BinOp::GreaterThan(_)
                | BinOp::GreaterThanEqual(_)
        ),
        Expression::Var(Var::Name(token)) => matches!(
            scopes.binding(token),
            Some(Binding::Local(v) | Binding::Upvalue(v)) if vars.contains(v)
        ),
        Expression::FunctionCall(c) => {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            let (
                Prefix::Name(callee),
                [
                    Suffix::Call(ast::Call::AnonymousCall(ast::FunctionArgs::Parentheses {
                        arguments,
                        ..
                    })),
                ],
            ) = (c.prefix(), suffixes.as_slice())
            else {
                return false;
            };
            matches!(scopes.binding(callee), Some(Binding::Global(name)) if name == "setmetatable")
                && builtin_named(scopes, "setmetatable").is_some()
                && arguments
                    .iter()
                    .next()
                    .is_some_and(|first| non_nil(scopes, vars, first))
        }
        _ => false,
    }
}

/// The locals of a chunk that never hold nil: declared with a value
/// that is never nil, by `local function` or a `local` statement, and
/// never assigned again, where the debug library cannot rebind them.
pub(super) fn non_nil_vars(scopes: &Scopes, ast: &ast::Ast) -> HashSet<VarId> {
    struct Walk<'a> {
        scopes: &'a Scopes,
        out: HashSet<VarId>,
    }
    impl Walk<'_> {
        fn fixed(&self, v: VarId) -> bool {
            let info = self.scopes.var(v);
            !info.assigned
                && !self.scopes.debug_setlocal
                && !(self.scopes.debug_rebinds && info.captured)
        }
    }
    impl full_moon::visitors::Visitor for Walk<'_> {
        fn visit_local_assignment(&mut self, l: &ast::LocalAssignment) {
            let exprs: Vec<&Expression> = l.expressions().iter().collect();
            for (i, name) in l.names().iter().enumerate() {
                let v = self.scopes.declared(name);
                // A name past the last expression takes one of its
                // later values, or nil.
                if let Some(e) = exprs.get(i)
                    && self.fixed(v)
                    && non_nil(self.scopes, &self.out, e)
                {
                    self.out.insert(v);
                }
            }
        }
        fn visit_local_function(&mut self, f: &ast::LocalFunction) {
            let v = self.scopes.declared(f.name());
            if self.fixed(v) {
                self.out.insert(v);
            }
        }
    }
    let mut walk = Walk {
        scopes,
        out: HashSet::new(),
    };
    full_moon::visitors::Visitor::visit_ast(&mut walk, ast);
    walk.out
}

/// The variable of a numeric `for` with a start and a step of these
/// types: a float loop when either is a float, an integer loop when
/// both are integers, else the loop's kind is decided when it starts
/// and the variable holds either number.
pub fn numeric_for_ty(start: Ty, step: Ty) -> Ty {
    match (start, step) {
        (Ty::Float, _) | (_, Ty::Float) => Ty::Float,
        (Ty::Int, Ty::Int) => Ty::Int,
        (Ty::Unknown, _) | (_, Ty::Unknown) => Ty::Unknown,
        _ => Ty::Number,
    }
}

/// The library function the global `name` holds when the program starts.
pub fn builtin_global(name: &str) -> Option<&'static Builtin> {
    BUILTINS.iter().find(|b| b.lib.is_empty() && b.name == name)
}

/// Whether the global `name` holds a value when the program starts: a
/// library function or table, `arg` or `_VERSION`.
pub fn preset_global(name: &str) -> bool {
    builtin_global(name).is_some()
        || crate::library::stdlib::LIBS.contains(&name)
        || matches!(name, "arg" | "_VERSION")
}

/// The library function `lib.name` is, when `lib` is the untouched
/// global of that library.
pub fn builtin_member(scopes: &Scopes, lib: &str, name: &str) -> Option<&'static Builtin> {
    if scopes.global_writes.contains_key(lib) {
        return None;
    }
    BUILTINS.iter().find(|b| b.lib == lib && b.name == name)
}

/// The result of a `math` function whose arguments' types decide it:
/// `math.abs` of an integer is an integer, `math.max` of an integer and
/// a float one or the other, `math.floor` of a float the integer it
/// rounds to when that fits, else the float. `None` when they do not,
/// and the function's declared result stands; unknown when an argument
/// is not known yet.
pub fn math_result(b: &Builtin, args: &[Ty]) -> Option<Ty> {
    if b.lib != "math" {
        return None;
    }
    let result = match (b.name, args) {
        ("max" | "min", [first, rest @ ..]) if args.iter().all(|t| t.is_number()) => {
            Some(rest.iter().fold(*first, |acc, t| acc.join(*t)))
        }
        ("abs", [t]) if t.is_number() => Some(*t),
        ("floor" | "ceil", [Ty::Int]) => Some(Ty::Int),
        ("floor" | "ceil", [Ty::Float | Ty::Number]) => Some(Ty::Number),
        ("fmod", [Ty::Int, Ty::Int]) => Some(Ty::Int),
        ("fmod", [Ty::Int | Ty::Float, Ty::Int | Ty::Float]) => Some(Ty::Float),
        ("max" | "min" | "abs" | "floor" | "ceil" | "fmod", _) => None,
        _ => return None,
    };
    if args.contains(&Ty::Unknown) {
        return Some(Ty::Unknown);
    }
    result
}

pub fn ret_ty(r: Ret) -> Ty {
    match r {
        Ret::Unit => Ty::Nil,
        Ret::Bool => Ty::Bool,
        Ret::Int => Ty::Int,
        Ret::Float => Ty::Float,
        Ret::Str => Ty::Str,
        Ret::Any | Ret::Multi => Ty::Any,
        Ret::Table => Ty::Table,
    }
}

/// What a numeral literal is: an integer, or a float when it has a
/// fraction, an exponent, or does not fit.
pub fn numeral_ty(text: &str) -> Ty {
    match crate::host::parse_numeral(text) {
        crate::host::Numeral::Int(_) => Ty::Int,
        _ => Ty::Float,
    }
}
