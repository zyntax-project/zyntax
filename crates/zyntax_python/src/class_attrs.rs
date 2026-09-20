//! Class attributes: `class C: X = 1` and `C.X = v`. Each is a module
//! variable named `C$X`, typed like any module variable as the join of
//! its writes, read through the class (`C.X`), an instance whose class
//! has no field of the name (`self.X`), or the method's own class
//! (`cls.X`, `self.__class__.X`). One written only by its declaration,
//! with a value the program fixes, is that constant at every read: a
//! hot method reads an immediate, not a slot.

use crate::classes::ClassDef;
use crate::types::{Item, Ty};
use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
use rustc_hash::FxHashMap as HashMap;

/// A value the program fixes at compile time.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    None,
}

impl Constant {
    pub(crate) fn ty(&self) -> Ty {
        match self {
            Constant::Int(_) => Ty::Int,
            Constant::Float(_) => Ty::Float,
            Constant::Bool(_) => Ty::Bool,
            Constant::Str(_) => Ty::Str,
            Constant::None => Ty::None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Constant::Int(i) => Some(*i as f64),
            Constant::Float(f) => Some(*f),
            Constant::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Constant::Int(i) => Some(*i),
            Constant::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }

    fn truth(&self) -> bool {
        match self {
            Constant::Int(i) => *i != 0,
            Constant::Float(f) => *f != 0.0,
            Constant::Bool(b) => *b,
            Constant::Str(s) => !s.is_empty(),
            Constant::None => false,
        }
    }
}

/// One class attribute.
#[derive(Debug, Clone)]
pub(crate) struct ClassAttr {
    /// The module variable it is stored in, `Owner$name`.
    pub(crate) global: String,
    /// Its value when the program fixes it: declared once, in the class
    /// body, from constants, and written nowhere else.
    pub(crate) constant: Option<Constant>,
}

/// The class attributes of a program.
#[derive(Debug, Default)]
pub(crate) struct ClassAttrs {
    attrs: Vec<ClassAttr>,
    index: HashMap<(usize, String), usize>,
}

pub(crate) fn global_of(class: &str, attr: &str) -> String {
    format!("{class}${attr}")
}

impl ClassAttrs {
    /// The attribute `name` as class `k` sees it: its own or the nearest
    /// base's.
    pub(crate) fn lookup(
        &self,
        bases: &[Option<usize>],
        k: usize,
        name: &str,
    ) -> Option<&ClassAttr> {
        let mut at = Some(k);
        while let Some(c) = at {
            if let Some(&i) = self.index.get(&(c, name.to_string())) {
                return Some(&self.attrs[i]);
            }
            at = bases[c];
        }
        None
    }

    /// The module variables the attributes not fixed as constants live in.
    pub(crate) fn globals(&self) -> impl Iterator<Item = &str> {
        self.attrs
            .iter()
            .filter(|a| a.constant.is_none())
            .map(|a| a.global.as_str())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
}

/// A class attribute declared in a class body, with the value it is
/// declared with.
pub(crate) struct Declared<'a> {
    pub(crate) name: String,
    pub(crate) value: &'a py::Expr,
}

/// The `name = value` statements of a class body, in order. Any other
/// statement of the body is a method, a docstring or `pass`, or an error.
pub(crate) fn declared_in(c: &py::StmtClassDef) -> Result<Vec<Declared<'_>>> {
    let mut out = Vec::new();
    for s in &c.body {
        match s {
            py::Stmt::FunctionDef(_) | py::Stmt::Pass(_) => {}
            py::Stmt::Expr(e) if matches!(*e.value, py::Expr::StringLiteral(_)) => {}
            py::Stmt::Assign(a) => {
                let [py::Expr::Name(n)] = a.targets.as_slice() else {
                    return Err(Error::unsupported(
                        "a class body assignment to anything but one name",
                        s,
                    ));
                };
                out.push(Declared {
                    name: n.id.to_string(),
                    value: &a.value,
                });
            }
            py::Stmt::AnnAssign(a) => {
                let (py::Expr::Name(n), Some(value)) = (&*a.target, &a.value) else {
                    return Err(Error::unsupported(
                        "a class body annotation without a value, or on anything but a name",
                        s,
                    ));
                };
                out.push(Declared {
                    name: n.id.to_string(),
                    value,
                });
            }
            other => {
                return Err(Error::unsupported(
                    "a class body statement other than a method or an attribute",
                    other,
                ));
            }
        }
    }
    Ok(out)
}

/// The class attributes of the program: those the class bodies declare
/// and those first written as `C.X = v` in the module body or a
/// function. `class_index` names the classes; `bases` their bases.
pub(crate) fn collect(
    defs: &[ClassDef<'_>],
    class_index: &HashMap<String, usize>,
    bases: &[Option<usize>],
    top_level: &[&py::Stmt],
    items: &[Item<'_>],
) -> Result<ClassAttrs> {
    let mut table = ClassAttrs::default();
    // Every `C.X = v` in the program, by class and name, with how many.
    let mut writes: HashMap<(usize, String), usize> = HashMap::default();
    let mut finder = WriteFinder {
        class_index,
        writes: &mut writes,
    };
    for s in top_level {
        finder.visit_stmt(s);
    }
    for item in items {
        for s in &item.def.body {
            finder.visit_stmt(s);
        }
    }
    let located = |e: Error, module: &Option<String>| match module {
        Some(m) => e.in_module(m),
        None => e,
    };
    // Body declarations, in class order, constants folded as they go.
    for def in defs {
        let k = class_index[&def.name];
        for decl in &def.attrs {
            if let Some(base) = bases[k]
                && table.lookup(bases, base, &decl.name).is_some()
            {
                return Err(located(
                    Error::unsupported(
                        format!(
                            "class attribute `{}` of {}, which a base class declares too",
                            decl.name, def.name
                        ),
                        decl.value,
                    ),
                    &def.module,
                ));
            }
            let written_elsewhere = writes.contains_key(&(k, decl.name.clone()));
            let constant = if written_elsewhere {
                None
            } else {
                Folder {
                    table: &table,
                    class_index,
                    bases,
                    class: k,
                }
                .expr(decl.value)
            };
            let i = table.attrs.len();
            table.attrs.push(ClassAttr {
                global: global_of(&def.name, &decl.name),
                constant,
            });
            table.index.insert((k, decl.name.clone()), i);
        }
    }
    // Attributes a write introduces: never constants.
    let mut introduced: Vec<(usize, String)> = writes
        .keys()
        .filter(|(k, name)| table.lookup(bases, *k, name).is_none())
        .cloned()
        .collect();
    introduced.sort();
    for (k, name) in introduced {
        let class = &defs[k].name;
        let i = table.attrs.len();
        table.attrs.push(ClassAttr {
            global: global_of(class, &name),
            constant: None,
        });
        table.index.insert((k, name), i);
    }
    Ok(table)
}

/// The class a name stands for at an attribute's base: `C` in `C.X`.
pub(crate) fn class_named(class_index: &HashMap<String, usize>, e: &py::Expr) -> Option<usize> {
    match e {
        py::Expr::Name(n) => class_index.get(n.id.as_str()).copied(),
        _ => None,
    }
}

/// Every `C.X = v` store target under a body.
struct WriteFinder<'a> {
    class_index: &'a HashMap<String, usize>,
    writes: &'a mut HashMap<(usize, String), usize>,
}

impl WriteFinder<'_> {
    fn target(&mut self, t: &py::Expr) {
        match t {
            py::Expr::Attribute(a) => {
                if let Some(k) = class_named(self.class_index, &a.value) {
                    *self.writes.entry((k, a.attr.to_string())).or_insert(0) += 1;
                }
            }
            py::Expr::Tuple(t) => t.elts.iter().for_each(|e| self.target(e)),
            py::Expr::List(l) => l.elts.iter().for_each(|e| self.target(e)),
            py::Expr::Starred(s) => self.target(&s.value),
            _ => {}
        }
    }
}

impl<'a> Visitor<'a> for WriteFinder<'_> {
    fn visit_stmt(&mut self, s: &'a py::Stmt) {
        match s {
            py::Stmt::Assign(a) => a.targets.iter().for_each(|t| self.target(t)),
            py::Stmt::AugAssign(a) => self.target(&a.target),
            py::Stmt::AnnAssign(a) => self.target(&a.target),
            py::Stmt::For(f) => self.target(&f.target),
            py::Stmt::With(w) => {
                for item in &w.items {
                    if let Some(v) = &item.optional_vars {
                        self.target(v);
                    }
                }
            }
            py::Stmt::Delete(d) => d.targets.iter().for_each(|t| self.target(t)),
            _ => {}
        }
        walk_stmt(self, s);
    }

    fn visit_expr(&mut self, e: &'a py::Expr) {
        if let py::Expr::Named(n) = e {
            self.target(&n.target);
        }
        walk_expr(self, e);
    }
}

/// Evaluates a class body initialiser that the program fixes: literals,
/// the constants of this class declared before it and of any class,
/// and arithmetic on them as Python computes it. Anything else, an
/// overflow included, is not a constant.
struct Folder<'a> {
    table: &'a ClassAttrs,
    class_index: &'a HashMap<String, usize>,
    bases: &'a [Option<usize>],
    class: usize,
}

impl Folder<'_> {
    fn attr(&self, k: usize, name: &str) -> Option<Constant> {
        self.table.lookup(self.bases, k, name)?.constant.clone()
    }

    fn expr(&self, e: &py::Expr) -> Option<Constant> {
        use Constant as C;
        Some(match e {
            py::Expr::NumberLiteral(n) => match &n.value {
                py::Number::Int(i) => C::Int(i.as_i64()?),
                py::Number::Float(f) => C::Float(*f),
                py::Number::Complex { .. } => return None,
            },
            py::Expr::BooleanLiteral(b) => C::Bool(b.value),
            py::Expr::NoneLiteral(_) => C::None,
            py::Expr::StringLiteral(s) => C::Str(s.value.to_str().to_string()),
            py::Expr::Name(n) => self.attr(self.class, n.id.as_str())?,
            py::Expr::Attribute(a) => {
                let k = class_named(self.class_index, &a.value)?;
                self.attr(k, a.attr.as_str())?
            }
            py::Expr::UnaryOp(u) => {
                let v = self.expr(&u.operand)?;
                match u.op {
                    py::UnaryOp::Not => C::Bool(!v.truth()),
                    py::UnaryOp::Invert => C::Int(!v.as_i64()?),
                    py::UnaryOp::UAdd => match v {
                        C::Bool(b) => C::Int(b as i64),
                        C::Int(_) | C::Float(_) => v,
                        _ => return None,
                    },
                    py::UnaryOp::USub => match v {
                        C::Bool(b) => C::Int(-(b as i64)),
                        C::Int(i) => C::Int(i.checked_neg()?),
                        C::Float(f) => C::Float(-f),
                        _ => return None,
                    },
                }
            }
            py::Expr::BinOp(b) => {
                let l = self.expr(&b.left)?;
                let r = self.expr(&b.right)?;
                binop(b.op, &l, &r)?
            }
            py::Expr::BoolOp(b) => {
                let mut acc = self.expr(&b.values[0])?;
                for v in &b.values[1..] {
                    let stop = match b.op {
                        py::BoolOp::And => !acc.truth(),
                        py::BoolOp::Or => acc.truth(),
                    };
                    if stop {
                        break;
                    }
                    acc = self.expr(v)?;
                }
                acc
            }
            py::Expr::Compare(c) if c.comparators.len() == 1 => {
                let l = self.expr(&c.left)?;
                let r = self.expr(&c.comparators[0])?;
                C::Bool(compare(c.ops[0], &l, &r)?)
            }
            // `float(x)` and `int(x)`: the builtins, which a class body
            // that shadows them cannot mean here since a shadowing name
            // would be an attribute of the class.
            py::Expr::Call(c) if c.arguments.keywords.is_empty() && c.arguments.args.len() == 1 => {
                let py::Expr::Name(f) = &*c.func else {
                    return None;
                };
                if self.class_index.contains_key(f.id.as_str())
                    || self.attr(self.class, f.id.as_str()).is_some()
                {
                    return None;
                }
                let v = self.expr(&c.arguments.args[0])?;
                match f.id.as_str() {
                    "float" => C::Float(v.as_f64()?),
                    "int" => match v {
                        C::Int(_) => v,
                        C::Bool(b) => C::Int(b as i64),
                        C::Float(f) if f.is_finite() && f.abs() < 9.2e18 => {
                            C::Int(f.trunc() as i64)
                        }
                        _ => return None,
                    },
                    _ => return None,
                }
            }
            _ => return None,
        })
    }
}

/// `l op r` as Python computes it on ints, bools and floats; `None` on
/// overflow, division by zero or an operand the operator does not take.
fn binop(op: py::Operator, l: &Constant, r: &Constant) -> Option<Constant> {
    use Constant as C;
    if let (Some(a), Some(b)) = (l.as_i64(), r.as_i64())
        && !matches!(l, C::Float(_))
        && !matches!(r, C::Float(_))
    {
        return Some(match op {
            py::Operator::Add => C::Int(a.checked_add(b)?),
            py::Operator::Sub => C::Int(a.checked_sub(b)?),
            py::Operator::Mult => C::Int(a.checked_mul(b)?),
            py::Operator::Div => C::Float(a as f64 / nonzero(b as f64)?),
            py::Operator::FloorDiv => C::Int(a.checked_div_euclid(b)?.checked_sub(
                // Euclidean and floor division differ when the divisor
                // is negative and the remainder is not zero.
                (b < 0 && a.checked_rem_euclid(b)? != 0) as i64,
            )?),
            py::Operator::Mod => {
                if b == 0 {
                    return None;
                }
                let m = a.checked_rem(b)?;
                C::Int(if m != 0 && (m < 0) != (b < 0) {
                    m + b
                } else {
                    m
                })
            }
            py::Operator::Pow => {
                if b < 0 {
                    C::Float((a as f64).powf(b as f64))
                } else {
                    C::Int(a.checked_pow(u32::try_from(b).ok()?)?)
                }
            }
            py::Operator::LShift => {
                let shift = u32::try_from(b).ok()?;
                if shift >= 63 || a.checked_shl(shift)? >> shift != a {
                    return None;
                }
                C::Int(a << shift)
            }
            py::Operator::RShift => C::Int(a >> u32::try_from(b).ok()?.min(63)),
            py::Operator::BitAnd => C::Int(a & b),
            py::Operator::BitOr => C::Int(a | b),
            py::Operator::BitXor => C::Int(a ^ b),
            py::Operator::MatMult => return None,
        });
    }
    if let (C::Str(a), C::Str(b), py::Operator::Add) = (l, r, op) {
        return Some(C::Str(format!("{a}{b}")));
    }
    let (a, b) = (l.as_f64()?, r.as_f64()?);
    Some(C::Float(match op {
        py::Operator::Add => a + b,
        py::Operator::Sub => a - b,
        py::Operator::Mult => a * b,
        py::Operator::Div => a / nonzero(b)?,
        py::Operator::FloorDiv => (a / nonzero(b)?).floor(),
        py::Operator::Mod => {
            let m = a % nonzero(b)?;
            if m != 0.0 && (m < 0.0) != (b < 0.0) {
                m + b
            } else {
                m
            }
        }
        py::Operator::Pow => a.powf(b),
        _ => return None,
    }))
}

fn nonzero(b: f64) -> Option<f64> {
    (b != 0.0).then_some(b)
}

fn compare(op: py::CmpOp, l: &Constant, r: &Constant) -> Option<bool> {
    use Constant as C;
    match (l, r) {
        (C::Str(a), C::Str(b)) => Some(match op {
            py::CmpOp::Eq => a == b,
            py::CmpOp::NotEq => a != b,
            py::CmpOp::Lt => a < b,
            py::CmpOp::LtE => a <= b,
            py::CmpOp::Gt => a > b,
            py::CmpOp::GtE => a >= b,
            _ => return None,
        }),
        (C::None, C::None) => match op {
            py::CmpOp::Eq | py::CmpOp::Is => Some(true),
            py::CmpOp::NotEq | py::CmpOp::IsNot => Some(false),
            _ => None,
        },
        _ => {
            let (a, b) = (l.as_f64()?, r.as_f64()?);
            Some(match op {
                py::CmpOp::Eq => a == b,
                py::CmpOp::NotEq => a != b,
                py::CmpOp::Lt => a < b,
                py::CmpOp::LtE => a <= b,
                py::CmpOp::Gt => a > b,
                py::CmpOp::GtE => a >= b,
                _ => return None,
            })
        }
    }
}

/// Whether `value` reads a name that is one of `names`: a class body
/// initialiser reading a sibling attribute that is not a constant.
pub(crate) fn reads_any(value: &py::Expr, names: &[String]) -> bool {
    struct Reader<'a> {
        names: &'a [String],
        found: bool,
    }
    impl<'a> Visitor<'a> for Reader<'_> {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Name(n) = e
                && self.names.iter().any(|x| x == n.id.as_str())
            {
                self.found = true;
            }
            walk_expr(self, e);
        }
    }
    let mut r = Reader {
        names,
        found: false,
    };
    r.visit_expr(value);
    r.found
}
