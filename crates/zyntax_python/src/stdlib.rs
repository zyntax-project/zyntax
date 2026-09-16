//! The standard library modules a program may import, and what each
//! name in them is: a function over typed arguments in the shared
//! library, a constant, or a value read from the host. `typing` is
//! accepted whole and contributes nothing but annotations.

use crate::types::{Elem, Ty};

/// What a module's name stands for.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Member {
    /// A function: its parameter types, its result type, and the
    /// library function that implements it.
    Func {
        params: &'static [Ty],
        ret: Ty,
        zb: &'static str,
    },
    /// A floating-point constant.
    Float(f64),
    /// An integer constant.
    Int(i64),
    /// A value the library computes on each read.
    Value { ty: Ty, zb: &'static str },
}

/// Modules that may be imported, with nothing to resolve at run time.
/// `__future__` is accepted whole: what it names is the language as
/// it is.
pub(crate) fn is_known(module: &str) -> bool {
    matches!(
        module,
        "math" | "sys" | "typing" | "time" | "bisect" | "__future__"
    )
}

/// The names `__future__` exports, every one of them already the case.
pub(crate) fn is_future_name(name: &str) -> bool {
    matches!(
        name,
        "division"
            | "print_function"
            | "absolute_import"
            | "unicode_literals"
            | "generators"
            | "nested_scopes"
            | "with_statement"
            | "generator_stop"
            | "annotations"
            | "barry_as_FLUFL"
    )
}

/// The names `typing` exports; all of them are annotations here.
pub(crate) fn is_typing_name(name: &str) -> bool {
    matches!(
        name,
        "List"
            | "Dict"
            | "Set"
            | "Tuple"
            | "Optional"
            | "Any"
            | "Union"
            | "Callable"
            | "Iterable"
            | "Iterator"
            | "Sequence"
            | "Mapping"
            | "TypeVar"
            | "Generic"
            | "Type"
            | "cast"
            | "NamedTuple"
            | "Protocol"
            | "Final"
            | "ClassVar"
            | "Literal"
            | "NoReturn"
    )
}

const F: Ty = Ty::Float;
const I: Ty = Ty::Int;
const B: Ty = Ty::Bool;

pub(crate) fn member(module: &str, name: &str) -> Option<Member> {
    let func = |params: &'static [Ty], ret: Ty, zb: &'static str| Member::Func { params, ret, zb };
    Some(match (module, name) {
        ("bisect", "bisect" | "bisect_right") => {
            func(&[Ty::List(Elem::Object), Ty::Object], I, "zb_bisect_right")
        }
        ("bisect", "bisect_left") => {
            func(&[Ty::List(Elem::Object), Ty::Object], I, "zb_bisect_left")
        }
        ("bisect", "insort" | "insort_right") => func(
            &[Ty::List(Elem::Object), Ty::Object],
            Ty::None,
            "zb_insort_right",
        ),
        ("bisect", "insort_left") => func(
            &[Ty::List(Elem::Object), Ty::Object],
            Ty::None,
            "zb_insort_left",
        ),
        ("math", "sqrt") => func(&[F], F, "zb_math_sqrt"),
        ("math", "pow") => func(&[F, F], F, "zb_math_pow"),
        ("math", "fabs") => func(&[F], F, "zb_math_fabs"),
        ("math", "floor") => func(&[F], I, "zb_math_floor"),
        ("math", "ceil") => func(&[F], I, "zb_math_ceil"),
        ("math", "trunc") => func(&[F], I, "zb_math_trunc"),
        ("math", "exp") => func(&[F], F, "zb_math_exp"),
        ("math", "log") => func(&[F], F, "zb_math_log"),
        ("math", "log2") => func(&[F], F, "zb_math_log2"),
        ("math", "log10") => func(&[F], F, "zb_math_log10"),
        ("math", "sin") => func(&[F], F, "zb_math_sin"),
        ("math", "cos") => func(&[F], F, "zb_math_cos"),
        ("math", "tan") => func(&[F], F, "zb_math_tan"),
        ("math", "asin") => func(&[F], F, "zb_math_asin"),
        ("math", "acos") => func(&[F], F, "zb_math_acos"),
        ("math", "atan") => func(&[F], F, "zb_math_atan"),
        ("math", "atan2") => func(&[F, F], F, "zb_math_atan2"),
        ("math", "sinh") => func(&[F], F, "zb_math_sinh"),
        ("math", "cosh") => func(&[F], F, "zb_math_cosh"),
        ("math", "tanh") => func(&[F], F, "zb_math_tanh"),
        ("math", "hypot") => func(&[F, F], F, "zb_math_hypot"),
        ("math", "fmod") => func(&[F, F], F, "zb_math_fmod"),
        ("math", "copysign") => func(&[F, F], F, "zb_math_copysign"),
        ("math", "isnan") => func(&[F], B, "zb_math_isnan"),
        ("math", "isinf") => func(&[F], B, "zb_math_isinf"),
        ("math", "isfinite") => func(&[F], B, "zb_math_isfinite"),
        ("math", "degrees") => func(&[F], F, "zb_math_degrees"),
        ("math", "radians") => func(&[F], F, "zb_math_radians"),
        ("math", "gcd") => func(&[I, I], I, "zb_math_gcd"),
        ("math", "factorial") => func(&[I], I, "zb_math_factorial"),
        ("math", "pi") => Member::Float(std::f64::consts::PI),
        ("math", "e") => Member::Float(std::f64::consts::E),
        ("math", "tau") => Member::Float(std::f64::consts::TAU),
        ("math", "inf") => Member::Float(f64::INFINITY),
        ("math", "nan") => Member::Float(f64::NAN),
        ("sys", "maxsize") => Member::Int(i64::MAX),
        ("sys", "version_info") => Member::Value {
            ty: Ty::Tuple,
            zb: "zb_sys_version_info",
        },
        ("sys", "argv") => Member::Value {
            ty: Ty::List(Elem::Str),
            zb: "zb_sys_argv",
        },
        // `exit` takes its status as an int; the lowering fills in the
        // default and the two-argument `log`.
        ("sys", "exit") => func(&[I], Ty::None, "zb_exit"),
        ("time", "time") => func(&[], F, "zb_time_time"),
        // The monotonic clocks are one clock here.
        ("time", "perf_counter" | "monotonic" | "process_time" | "clock") => {
            func(&[], F, "zb_time_perf_counter")
        }
        _ => return None,
    })
}
