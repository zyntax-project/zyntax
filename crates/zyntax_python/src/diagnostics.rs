//! What zypy says when it cannot compile Python, in Python's own terms:
//! the construct by the name a Python programmer uses for it, what zypy
//! does compile in its place, and what to write instead. An error that
//! says "this form" or "N argument(s)" tells the programmer nothing the
//! compiler knew.

use crate::Error;
use ruff_python_ast as py;
use zyntax_typed_ast::source::Span;

/// The methods zypy compiles of a builtin type, by the type's name.
pub(crate) fn methods_of(ty: &str) -> Option<&'static [&'static str]> {
    Some(match ty {
        "str" => &[
            "capitalize",
            "casefold",
            "center",
            "count",
            "encode",
            "endswith",
            "expandtabs",
            "find",
            "format_map",
            "index",
            "isalnum",
            "isalpha",
            "isascii",
            "isdecimal",
            "isdigit",
            "isidentifier",
            "islower",
            "isnumeric",
            "isprintable",
            "isspace",
            "istitle",
            "isupper",
            "join",
            "ljust",
            "lower",
            "lstrip",
            "partition",
            "removeprefix",
            "removesuffix",
            "replace",
            "rfind",
            "rindex",
            "rjust",
            "rpartition",
            "rsplit",
            "rstrip",
            "split",
            "splitlines",
            "startswith",
            "strip",
            "swapcase",
            "title",
            "translate",
            "upper",
            "zfill",
        ],
        "bytes" => &[
            "capitalize",
            "center",
            "count",
            "decode",
            "digest",
            "endswith",
            "expandtabs",
            "find",
            "hex",
            "hexdigest",
            "index",
            "isalnum",
            "isalpha",
            "isascii",
            "isdigit",
            "islower",
            "isspace",
            "istitle",
            "isupper",
            "join",
            "ljust",
            "lower",
            "lstrip",
            "partition",
            "removeprefix",
            "removesuffix",
            "replace",
            "rfind",
            "rindex",
            "rjust",
            "rpartition",
            "rsplit",
            "rstrip",
            "split",
            "splitlines",
            "startswith",
            "strip",
            "swapcase",
            "title",
            "translate",
            "upper",
            "zfill",
        ],
        "list" => &[
            "append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove",
            "reverse", "sort",
        ],
        "tuple" => &["count", "index"],
        "dict" => &[
            "clear",
            "copy",
            "get",
            "items",
            "keys",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "values",
        ],
        "set" => &[
            "add",
            "clear",
            "copy",
            "difference",
            "difference_update",
            "discard",
            "intersection",
            "intersection_update",
            "isdisjoint",
            "issubset",
            "issuperset",
            "pop",
            "remove",
            "symmetric_difference",
            "symmetric_difference_update",
            "union",
            "update",
        ],
        "file" => &[
            "close",
            "flush",
            "getvalue",
            "isatty",
            "read",
            "readable",
            "readline",
            "readlines",
            "seek",
            "seekable",
            "tell",
            "truncate",
            "writable",
            "write",
            "writelines",
        ],
        _ => return None,
    })
}

/// What to do when a type lacks the method, where there is something
/// to say beyond the list.
fn advice(ty: &str) -> &'static str {
    match ty {
        "bytes" => {
            " To use a `str` method on the contents, decode them first, as in `data.decode().expandtabs()`. `data[i]` is an int and `for b in data` yields ints."
        }
        "file" => " Read a text file by line with `for line in f`, or all at once with `f.read()`.",
        "str" => {
            " A method that is missing can often be written with `replace`, `split`, `join`, or a comprehension over the characters."
        }
        _ => "",
    }
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

/// The edit distance of two short names.
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let held = row[j + 1];
            row[j + 1] = if ca == cb {
                prev
            } else {
                1 + prev.min(row[j]).min(row[j + 1])
            };
            prev = held;
        }
    }
    row[b.len()]
}

/// Every method Python gives the builtin type `ty`, compiled by zypy or
/// not, for the types whose methods are all a program can call.
pub(crate) fn python_methods_of(ty: &str) -> Option<&'static [&'static str]> {
    Some(match ty {
        "bytes" => &[
            "capitalize",
            "center",
            "count",
            "decode",
            "digest",
            "endswith",
            "expandtabs",
            "find",
            "hex",
            "hexdigest",
            "index",
            "isalnum",
            "isalpha",
            "isascii",
            "isdigit",
            "islower",
            "isspace",
            "istitle",
            "isupper",
            "join",
            "ljust",
            "lower",
            "lstrip",
            "maketrans",
            "partition",
            "removeprefix",
            "removesuffix",
            "replace",
            "rfind",
            "rindex",
            "rjust",
            "rpartition",
            "rsplit",
            "rstrip",
            "split",
            "splitlines",
            "startswith",
            "strip",
            "swapcase",
            "title",
            "translate",
            "upper",
            "zfill",
        ],
        "str" => &[
            "capitalize",
            "casefold",
            "center",
            "count",
            "encode",
            "endswith",
            "expandtabs",
            "find",
            "format",
            "format_map",
            "index",
            "isalnum",
            "isalpha",
            "isascii",
            "isdecimal",
            "isdigit",
            "isidentifier",
            "islower",
            "isnumeric",
            "isprintable",
            "isspace",
            "istitle",
            "isupper",
            "join",
            "ljust",
            "lower",
            "lstrip",
            "maketrans",
            "partition",
            "removeprefix",
            "removesuffix",
            "replace",
            "rfind",
            "rindex",
            "rjust",
            "rpartition",
            "rsplit",
            "rstrip",
            "split",
            "splitlines",
            "startswith",
            "strip",
            "swapcase",
            "title",
            "translate",
            "upper",
            "zfill",
        ],
        "list" => &[
            "append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove",
            "reverse", "sort",
        ],
        "tuple" => &["count", "index"],
        "dict" => &[
            "clear",
            "copy",
            "fromkeys",
            "get",
            "items",
            "keys",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "values",
        ],
        "set" => &[
            "add",
            "clear",
            "copy",
            "difference",
            "difference_update",
            "discard",
            "intersection",
            "intersection_update",
            "isdisjoint",
            "issubset",
            "issuperset",
            "pop",
            "remove",
            "symmetric_difference",
            "symmetric_difference_update",
            "union",
            "update",
        ],
        _ => return None,
    })
}

/// `ty.method(...)` with `given` arguments, which zypy does not compile.
pub(crate) fn unsupported_method(ty: &str, method: &str, given: usize, span: Span) -> Error {
    let Some(known) = methods_of(ty) else {
        return Error::unsupported_span(format!("`{ty}.{method}()`"), span);
    };
    // A name Python does not define is the error Python raises.
    if let Some(all) = python_methods_of(ty)
        && !all.contains(&method)
    {
        let near = all
            .iter()
            .filter(|m| distance(m, method) <= (method.len() / 3).max(2))
            .min_by_key(|m| distance(m, method));
        let error = no_attribute(ty, method, span);
        return match near {
            Some(near) => error.with_help(format!("Did you mean `{near}`?")),
            None => error,
        };
    }
    let list = known
        .iter()
        .map(|m| format!("`{m}`"))
        .collect::<Vec<_>>()
        .join(", ");
    if known.contains(&method) {
        return Error::unsupported_span(
            format!(
                "`{ty}.{method}()` with {}",
                plural(given, "argument")
            ),
            span,
        )
        .with_help(format!(
            "zypy compiles `{ty}.{method}` with a different number or kind of arguments. Check the call against the Python documentation for `{ty}.{method}`."
        ));
    }
    let near = known
        .iter()
        .filter(|m| distance(m, method) <= (method.len() / 3).max(2))
        .min_by_key(|m| distance(m, method));
    let mut help = format!("zypy compiles these `{ty}` methods: {list}.");
    if let Some(near) = near {
        help = format!("Did you mean `{near}`? {help}");
    }
    help.push_str(advice(ty));
    Error::unsupported_span(format!("the `{ty}` method `{method}()`"), span).with_help(help)
}

/// What to say where keyword arguments cannot be placed.
pub(crate) const KEYWORD_HELP: &str = "zypy matches a keyword argument to a parameter only where it knows which function is called and that the function takes it. Call the function by its name, or pass the arguments by position.";

/// The Python construct the source at `span` is, as a programmer names
/// it, for the label under the code.
pub(crate) fn describe(source: &str, span: (usize, usize)) -> String {
    let end = span.1.min(source.len());
    let text = source.get(span.0.min(end)..end).unwrap_or("");
    if let Ok(parsed) = ruff_python_parser::parse_expression(text.trim()) {
        return match &*parsed.into_syntax().body {
            py::Expr::Call(c) => match &*c.func {
                py::Expr::Attribute(_) => "method call".to_string(),
                _ => "call".to_string(),
            },
            py::Expr::Attribute(_) => "attribute access".to_string(),
            py::Expr::Subscript(_) => "subscript".to_string(),
            py::Expr::Starred(_) => "`*` argument".to_string(),
            py::Expr::Lambda(_) => "lambda".to_string(),
            py::Expr::Compare(_) => "comparison".to_string(),
            py::Expr::BinOp(_) | py::Expr::UnaryOp(_) | py::Expr::BoolOp(_) => {
                "expression".to_string()
            }
            py::Expr::ListComp(_)
            | py::Expr::SetComp(_)
            | py::Expr::DictComp(_)
            | py::Expr::Generator(_) => "comprehension".to_string(),
            py::Expr::FString(_) => "f-string".to_string(),
            py::Expr::Await(_) => "`await` expression".to_string(),
            py::Expr::Yield(_) | py::Expr::YieldFrom(_) => "`yield` expression".to_string(),
            py::Expr::Named(_) => "assignment expression".to_string(),
            py::Expr::Name(_) => "name".to_string(),
            _ => "expression".to_string(),
        };
    }
    let first = text.split_whitespace().next().unwrap_or("");
    let keyword = first.trim_end_matches([':', '(']);
    match keyword {
        "with" => "`with` statement",
        "match" => "`match` statement",
        "try" => "`try` statement",
        "for" => "`for` loop",
        "while" => "`while` loop",
        "if" => "`if` statement",
        "class" => "class definition",
        "def" => "function definition",
        "async" => "`async` construct",
        "import" | "from" => "import",
        "del" => "`del` statement",
        "global" | "nonlocal" => "scope declaration",
        "raise" => "`raise` statement",
        "assert" => "`assert` statement",
        "return" => "`return` statement",
        "type" => "type alias",
        _ if first.starts_with('@') => "decorated definition",
        _ => "code",
    }
    .to_string()
}

/// An object lacks the method a Python protocol needs: Python raises
/// `TypeError` with the message of the protocol.
pub(crate) fn no_protocol(class: &str, dunder: &str, span: Span) -> Error {
    let message = match dunder {
        "__getitem__" => format!("'{class}' object is not subscriptable"),
        "__setitem__" => format!("'{class}' object does not support item assignment"),
        "__enter__" | "__exit__" => {
            format!("'{class}' object does not support the context manager protocol")
        }
        "__len__" => format!("object of type '{class}' has no len()"),
        "__int__" => format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{class}'"
        ),
        "__float__" => format!("float() argument must be a string or a real number, not '{class}'"),
        "__iter__" => format!("'{class}' object is not iterable"),
        "__call__" => format!("'{class}' object is not callable"),
        _ => format!("'{class}' object has no `{dunder}` method"),
    };
    let help = if dunder == "__enter__" || dunder == "__exit__" {
        format!("Define both `__enter__` and `__exit__` in `{class}`, or use `try`/`finally`.")
    } else {
        format!("Define `{dunder}` in `{class}`.")
    };
    Error::raises_span("TypeError", message, span).with_help(help)
}

/// A binary operator no operand's class defines.
pub(crate) fn no_binary(op: &str, left: &str, right: &str, dunder: &str, span: Span) -> Error {
    Error::raises_span(
        "TypeError",
        format!("unsupported operand type(s) for {op}: '{left}' and '{right}'"),
        span,
    )
    .with_help(format!(
        "Define `{dunder}` in `{left}`, or the reflected method in `{right}`."
    ))
}

/// A unary operator the class does not define.
pub(crate) fn no_unary(op: &str, class: &str, dunder: &str, span: Span) -> Error {
    Error::raises_span(
        "TypeError",
        format!("bad operand type for unary {op}: '{class}'"),
        span,
    )
    .with_help(format!("Define `{dunder}` in `{class}`."))
}

/// An ordering comparison the class does not define.
pub(crate) fn no_ordering(op: &str, left: &str, right: &str, dunder: &str, span: Span) -> Error {
    Error::raises_span(
        "TypeError",
        format!("'{op}' not supported between instances of '{left}' and '{right}'"),
        span,
    )
    .with_help(format!(
        "Define `{dunder}` in `{left}`, or `functools.total_ordering` with `__eq__` and one ordering."
    ))
}

/// A method or attribute the class does not have.
pub(crate) fn no_attribute(class: &str, name: &str, span: Span) -> Error {
    Error::raises_span(
        "AttributeError",
        format!("'{class}' object has no attribute '{name}'"),
        span,
    )
}

/// A `bytes` method given an argument that is not bytes, as CPython
/// words it for the method.
pub(crate) fn bytes_needed(method: &str, found: &str, span: Span) -> Error {
    let message = match method {
        "find" | "rfind" | "index" | "rindex" | "count" => {
            format!("argument should be integer or bytes-like object, not '{found}'")
        }
        "startswith" | "endswith" => {
            format!("{method} first arg must be bytes or a tuple of bytes, not {found}")
        }
        _ => format!("a bytes-like object is required, not '{found}'"),
    };
    let help = if found == "str" {
        "Encode the text with `.encode()`, or write the value as a bytes literal such as `b\"...\"`."
    } else {
        "Pass a `bytes` value."
    };
    Error::raises_span("TypeError", message, span).with_help(help)
}

/// A data attribute of a builtin value that zypy does not compile: the
/// error Python raises when the type has no such attribute, and a
/// plain statement when it has one.
pub(crate) fn unsupported_attribute(ty: &str, attr: &str, span: Span) -> Error {
    if let Some(all) = python_methods_of(ty)
        && !all.contains(&attr)
        && !attr.starts_with("__")
    {
        return no_attribute(ty, attr, span);
    }
    Error::unsupported_span(format!("the `{ty}` attribute `{attr}`"), span)
}

/// Whether `ty.method(x, ...)` is the method called on `x`: a method
/// every instance has, not one the type itself offers.
pub(crate) fn is_unbound_method(ty: &str, method: &str) -> bool {
    !matches!(method, "maketrans" | "fromkeys" | "fromhex")
        && python_methods_of(ty).is_some_and(|all| all.contains(&method))
}

/// A method of `int` or `float` that zypy does not compile: Python's
/// error for a name the type lacks, a plain statement otherwise.
pub(crate) fn unsupported_number_method(ty: &str, method: &str, span: Span) -> Error {
    const INT: &[&str] = &[
        "as_integer_ratio",
        "bit_count",
        "bit_length",
        "conjugate",
        "denominator",
        "from_bytes",
        "imag",
        "is_integer",
        "numerator",
        "real",
        "to_bytes",
    ];
    const FLOAT: &[&str] = &[
        "as_integer_ratio",
        "conjugate",
        "fromhex",
        "hex",
        "imag",
        "is_integer",
        "real",
    ];
    let all = if ty == "int" { INT } else { FLOAT };
    if all.contains(&method) {
        return Error::unsupported_span(format!("the `{ty}` method `{method}()`"), span);
    }
    let near = all
        .iter()
        .filter(|m| distance(m, method) <= (method.len() / 3).max(2))
        .min_by_key(|m| distance(m, method));
    let error = no_attribute(ty, method, span);
    match near {
        Some(near) => error.with_help(format!("Did you mean `{near}`?")),
        None => error,
    }
}

/// `T.method(...)` on a builtin type that is not the method called on a
/// value, and not a class method zypy compiles.
pub(crate) fn unsupported_class_call(ty: &str, method: &str, span: Span) -> Error {
    let known = python_methods_of(ty).is_some_and(|all| all.contains(&method))
        || matches!(ty, "int" | "float");
    if known {
        return Error::unsupported_span(format!("`{ty}.{method}()`"), span).with_help(format!(
            "Call the method on a value, as in `x.{method}(...)`, where zypy compiles it."
        ));
    }
    Error::raises_span(
        "AttributeError",
        format!("type object '{ty}' has no attribute '{method}'"),
        span,
    )
}
