//! Arbitrary-precision `int`, for a program that asks for it with the
//! comment `# zypy: bigint` before its first statement, or with the
//! environment variable `ZYPY_BIGINT`. The program is linked with the
//! bundled module `__bigint`, and the operators that can overflow a
//! 64-bit int, and the builtins that make or read ints, call its
//! functions. An int that fits 64 bits stays the machine int; the lists,
//! dicts and sets of the program hold ints as dynamic values so any
//! element may be a bigger one.

use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::Ranged;
use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// The module the functions are in.
pub(crate) const MODULE: &str = "__bigint";

/// The file index of the bundled module's source, or `NONE` when the
/// program did not ask for big ints.
static FILE: AtomicU32 = AtomicU32::new(NONE);
const NONE: u32 = u32::MAX;

/// Record where the bundled module's source is, or that it is not.
pub(crate) fn set_file(file: Option<u32>) {
    FILE.store(file.unwrap_or(NONE), Ordering::Relaxed);
}

/// Whether ints stored in a container are held as dynamic values: the
/// program asked for big ints and the code being typed is not the
/// module that implements them.
pub(crate) fn boxes_ints() -> bool {
    let file = FILE.load(Ordering::Relaxed);
    file != NONE && crate::lower::current_file() != file
}

/// Whether `source` asks for big ints.
pub(crate) fn requested(source: &str) -> bool {
    if std::env::var_os("ZYPY_BIGINT").is_some_and(|v| v != "0") {
        return true;
    }
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(comment) = line.strip_prefix('#') else {
            return false;
        };
        let comment = comment.trim().to_ascii_lowercase();
        if let Some(options) = comment.strip_prefix("zypy:")
            && options.split([',', ' ']).any(|o| o == "bigint")
        {
            return true;
        }
    }
    false
}

/// The import that links the module in.
pub(crate) fn import() -> py::Stmt {
    ruff_python_parser::parse_module(&format!("import {MODULE}\n"))
        .expect("the import parses")
        .into_syntax()
        .body
        .into_iter()
        .next()
        .expect("one statement")
}

/// The module's function `name`, as the linked program names it.
fn function(name: &str) -> String {
    crate::modules::qualified(MODULE, name)
}

/// Rewrite the arithmetic of every statement of the program, not of
/// the prelude or the bundled module.
pub(crate) fn rewrite(body: &mut [py::Stmt], origins: &[Option<String>]) -> Result<()> {
    let mut math: HashMap<String, String> = HashMap::default();
    let mut from_math: HashMap<String, String> = HashMap::default();
    for (s, origin) in body.iter().zip(origins) {
        if origin.as_deref() == Some(MODULE) {
            continue;
        }
        collect_math(s, &mut math, &mut from_math);
    }
    let rewriter = Rewriter {
        math,
        from_math,
        error: std::cell::RefCell::new(None),
    };
    for (s, origin) in body.iter_mut().zip(origins) {
        if origin.as_deref() == Some(MODULE) || origin.as_deref() == Some(crate::PRELUDE) {
            continue;
        }
        rewriter.visit_stmt(s);
    }
    match rewriter.error.into_inner() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The names `import math` and `from math import f` bind.
fn collect_math(
    s: &py::Stmt,
    modules: &mut HashMap<String, String>,
    names: &mut HashMap<String, String>,
) {
    match s {
        py::Stmt::Import(i) => {
            for a in &i.names {
                if a.name.as_str() == "math" {
                    let local = a.asname.as_ref().map_or("math", |n| n.as_str());
                    modules.insert(local.to_string(), "math".to_string());
                }
            }
        }
        py::Stmt::ImportFrom(i) if i.module.as_ref().is_some_and(|m| m.as_str() == "math") => {
            for a in &i.names {
                let local = a.asname.as_ref().map_or(a.name.as_str(), |n| n.as_str());
                names.insert(local.to_string(), a.name.to_string());
            }
        }
        _ => {}
    }
}

/// What the module's `math` replacements are named.
fn math_function(member: &str) -> Option<&'static str> {
    Some(match member {
        "factorial" => "factorial",
        "gcd" => "gcd",
        "lcm" => "lcm",
        "isqrt" => "isqrt",
        "comb" => "comb",
        "prod" => "prod",
        _ => return None,
    })
}

struct Rewriter {
    /// The local names `math` is bound to.
    math: HashMap<String, String>,
    /// The local names of functions imported from `math`, by member.
    from_math: HashMap<String, String>,
    error: std::cell::RefCell<Option<Error>>,
}

fn name(id: &str, range: ruff_text_size::TextRange) -> py::Expr {
    py::Expr::Name(py::ExprName {
        node_index: Default::default(),
        range,
        id: py::name::Name::new(id),
        ctx: py::ExprContext::Load,
    })
}

fn call(func: &str, args: Vec<py::Expr>, range: ruff_text_size::TextRange) -> py::Expr {
    py::Expr::Call(py::ExprCall {
        node_index: Default::default(),
        range_start: range.start(),
        func: Box::new(name(func, range)),
        arguments: py::Arguments {
            node_index: Default::default(),
            range,
            args: args.into_boxed_slice(),
            keywords: Default::default(),
        },
    })
}

/// An operand that is plainly not an int, so the operator on it cannot
/// overflow one.
fn not_int(e: &py::Expr) -> bool {
    match e {
        py::Expr::StringLiteral(_)
        | py::Expr::BytesLiteral(_)
        | py::Expr::FString(_)
        | py::Expr::List(_)
        | py::Expr::Tuple(_)
        | py::Expr::Dict(_)
        | py::Expr::Set(_)
        | py::Expr::ListComp(_)
        | py::Expr::SetComp(_)
        | py::Expr::DictComp(_)
        | py::Expr::Generator(_) => true,
        py::Expr::NumberLiteral(n) => !matches!(n.value, py::Number::Int(_)),
        _ => false,
    }
}

fn int_literal(e: &py::Expr) -> Option<i128> {
    match e {
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i.as_i64().map(i128::from),
            _ => None,
        },
        py::Expr::UnaryOp(u) if u.op == py::UnaryOp::USub => int_literal(&u.operand).map(|v| -v),
        _ => None,
    }
}

/// The module's function an operator calls, when the operator can
/// overflow.
fn operator_function(op: py::Operator) -> Option<&'static str> {
    Some(match op {
        py::Operator::Add => "add",
        py::Operator::Sub => "sub",
        py::Operator::Mult => "mul",
        py::Operator::Pow => "power",
        py::Operator::LShift => "lshift",
        py::Operator::FloorDiv => "floordiv",
        _ => return None,
    })
}

/// Whether `l op r` of two literals fits 64 bits, so it stays as it is.
fn folds_small(op: py::Operator, l: i128, r: i128) -> bool {
    let value = match op {
        py::Operator::Add => l.checked_add(r),
        py::Operator::Sub => l.checked_sub(r),
        py::Operator::Mult => l.checked_mul(r),
        py::Operator::Pow if (0..128).contains(&r) => l.checked_pow(r as u32),
        py::Operator::LShift if (0..64).contains(&r) => l.checked_mul(1i128 << r),
        py::Operator::FloorDiv => l.checked_div(r),
        _ => None,
    };
    value.is_some_and(|v| i64::try_from(v).is_ok())
}

impl Rewriter {
    fn fail(&self, what: &str, at: &impl ruff_text_size::Ranged) {
        self.error
            .borrow_mut()
            .get_or_insert_with(|| Error::unsupported(what.to_string(), at));
    }

    /// `l op r` as the call of the module's function, unless the
    /// operands cannot overflow.
    fn operation(&self, op: py::Operator, l: &py::Expr, r: &py::Expr) -> Option<py::Expr> {
        let f = operator_function(op)?;
        if not_int(l) || not_int(r) {
            return None;
        }
        // Only the smallest int divided by -1 leaves 64 bits.
        if op == py::Operator::FloorDiv && int_literal(r).is_some_and(|k| k != -1) {
            return None;
        }
        if let (Some(a), Some(b)) = (int_literal(l), int_literal(r))
            && folds_small(op, a, b)
        {
            return None;
        }
        let range = l.range().cover(r.range());
        Some(call(&function(f), vec![l.clone(), r.clone()], range))
    }

    /// A load of what an augmented assignment assigns to, when it can
    /// be read twice.
    fn reread(&self, target: &py::Expr) -> Option<py::Expr> {
        fn pure(e: &py::Expr) -> bool {
            match e {
                py::Expr::Name(_) => true,
                py::Expr::Attribute(a) => pure(&a.value),
                py::Expr::Subscript(s) => {
                    pure(&s.value)
                        && matches!(
                            &*s.slice,
                            py::Expr::Name(_)
                                | py::Expr::NumberLiteral(_)
                                | py::Expr::StringLiteral(_)
                        )
                }
                _ => false,
            }
        }
        if !pure(target) {
            return None;
        }
        let mut load = target.clone();
        set_load(&mut load);
        Some(load)
    }
}

fn set_load(e: &mut py::Expr) {
    match e {
        py::Expr::Name(n) => n.ctx = py::ExprContext::Load,
        py::Expr::Attribute(a) => {
            a.ctx = py::ExprContext::Load;
            set_load(&mut a.value);
        }
        py::Expr::Subscript(s) => {
            s.ctx = py::ExprContext::Load;
            set_load(&mut s.value);
        }
        _ => {}
    }
}

impl Transformer for Rewriter {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        if let py::Stmt::AugAssign(a) = s
            && operator_function(a.op).is_some()
            && !not_int(&a.value)
        {
            self.visit_expr(&mut a.value);
            let Some(read) = self.reread(&a.target) else {
                self.fail(
                    "an augmented assignment to a computed target in a program with big ints",
                    &a.range,
                );
                return;
            };
            let Some(combined) = self.operation(a.op, &read, &a.value) else {
                return;
            };
            *s = py::Stmt::Assign(py::StmtAssign {
                node_index: Default::default(),
                range: a.range,
                targets: vec![(*a.target).clone()],
                value: Box::new(combined),
            });
            return;
        }
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        let range = e.range();
        match e {
            // An int literal past 64 bits is read from its digits.
            py::Expr::NumberLiteral(n) => {
                if let py::Number::Int(i) = &n.value
                    && i.as_i64().is_none()
                {
                    let digits = i.to_string();
                    let literal = ruff_python_parser::parse_expression(&format!("{digits:?}"))
                        .expect("a string literal parses")
                        .into_syntax()
                        .body;
                    *e = call(&function("to_int"), vec![*literal], range);
                }
            }
            py::Expr::BinOp(b) => {
                if let Some(combined) = self.operation(b.op, &b.left, &b.right) {
                    *e = combined;
                }
            }
            // Negating the smallest int leaves 64 bits.
            py::Expr::UnaryOp(u)
                if u.op == py::UnaryOp::USub
                    && !not_int(&u.operand)
                    && int_literal(&u.operand).is_none()
                    && !matches!(&*u.operand, py::Expr::NumberLiteral(_)) =>
            {
                let operand = (*u.operand).clone();
                *e = call(&function("neg"), vec![operand], range);
            }
            py::Expr::Call(_) => self.builtin(e, range),
            _ => {}
        }
    }
}

impl Rewriter {
    /// A call of a builtin or `math` function that makes, reads or
    /// combines ints, as the call of the module's function.
    fn builtin(&self, e: &mut py::Expr, range: ruff_text_size::TextRange) {
        let py::Expr::Call(c) = e else {
            return;
        };
        let plain = c.arguments.keywords.is_empty()
            && !c
                .arguments
                .args
                .iter()
                .any(|a| matches!(a, py::Expr::Starred(_)));
        if !plain {
            return;
        }
        let n = c.arguments.args.len();
        let target = match &*c.func {
            py::Expr::Name(f) => {
                let id = f.id.as_str();
                if let Some(member) = self.from_math.get(id) {
                    math_function(member).map(str::to_string)
                } else {
                    match (id, n) {
                        ("abs", 1) => Some("abs_".to_string()),
                        ("hex", 1) => Some("hex_".to_string()),
                        ("oct", 1) => Some("oct_".to_string()),
                        ("bin", 1) => Some("bin_".to_string()),
                        ("sum", 1 | 2) => Some("sum_".to_string()),
                        ("pow", 2) => Some("power".to_string()),
                        ("pow", 3) => Some("powmod".to_string()),
                        ("int", 1) => Some("to_int".to_string()),
                        ("int", 2) => Some("to_int_base".to_string()),
                        ("divmod", 2) => Some("divmod_".to_string()),
                        ("isinstance", 2) if matches!(&c.arguments.args[1], py::Expr::Name(t) if t.id.as_str() == "int") => {
                            Some("is_int".to_string())
                        }
                        _ => None,
                    }
                }
            }
            py::Expr::Attribute(a) => match &*a.value {
                py::Expr::Name(m) if self.math.contains_key(m.id.as_str()) => {
                    math_function(a.attr.as_str()).map(str::to_string)
                }
                _ => None,
            },
            _ => None,
        };
        let Some(target) = target else {
            return;
        };
        let mut args = std::mem::take(&mut c.arguments.args).into_vec();
        // `isinstance(x, int)` is `is_int(x)`.
        if target == "is_int" {
            args.truncate(1);
        }
        *e = call(&function(&target), args, range);
    }
}
