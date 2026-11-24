//! `match` as the `if` statements a program would write. The subject is
//! held once; each case is the test its pattern makes of the held value
//! (and of the parts of it, by index, key or attribute), the assignments
//! of the names the pattern captures, and its body. A case with a guard
//! falls through to the next when the guard fails, so a program with a
//! guard marks the case it ran with a flag.
//!
//! Refused: an or-pattern that captures names, and a mapping pattern
//! with `**rest`.

use crate::dataclass::Fill;
use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::walk_stmt;
use ruff_text_size::Ranged;
use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;

/// The classes `__match_args__` names, by class: what a positional
/// pattern of the class reads.
type MatchArgs = HashMap<String, Vec<String>>;

/// Rewrite every `match` of `body`, at any depth.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) -> Result<()> {
    let mut args = MatchArgs::default();
    for s in body.iter() {
        collect(s, &mut args);
    }
    let error: RefCell<Option<Error>> = RefCell::new(None);
    let mut count = 0usize;
    crate::walk::expand_module(body, origins, &mut |s| match s {
        py::Stmt::Match(m) => match lower(m, &args, &mut count) {
            Ok(stmts) => stmts,
            Err(e) => {
                error.borrow_mut().get_or_insert(e);
                Vec::new()
            }
        },
        other => vec![other],
    });
    match error.into_inner() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// What a positional pattern of each class takes: the names in its
/// `__match_args__`, or the fields of a dataclass.
fn collect(s: &py::Stmt, out: &mut MatchArgs) {
    let py::Stmt::ClassDef(c) = s else {
        return;
    };
    let mut names: Option<Vec<String>> = None;
    for m in &c.body {
        if let py::Stmt::Assign(a) = m
            && let [py::Expr::Name(t)] = &a.targets[..]
            && t.id.as_str() == "__match_args__"
            && let py::Expr::Tuple(t) = &*a.value
        {
            let strs: Option<Vec<String>> = t
                .elts
                .iter()
                .map(|e| match e {
                    py::Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
                    _ => None,
                })
                .collect();
            names = strs;
        }
    }
    if names.is_none()
        && c.decorator_list.iter().any(|d| {
            let e = match &d.expression {
                py::Expr::Call(call) => &*call.func,
                other => other,
            };
            crate::dataclass::is_dataclass_name(e)
        })
    {
        let mut fields: Vec<String> = Vec::new();
        if let Some(args) = &c.arguments {
            for b in &args.args {
                if let py::Expr::Name(b) = b
                    && let Some(inherited) = out.get(b.id.as_str())
                {
                    fields.extend(inherited.iter().cloned());
                }
            }
        }
        for m in &c.body {
            if let py::Stmt::AnnAssign(a) = m
                && let py::Expr::Name(t) = &*a.target
                && !crate::dataclass::is_class_var(&a.annotation)
                && !fields.iter().any(|f| f == t.id.as_str())
            {
                fields.push(t.id.to_string());
            }
        }
        names = Some(fields);
    }
    if let Some(names) = names {
        out.insert(c.name.to_string(), names);
    }
}

/// What a pattern asks of its subject: a test, and the names it binds.
struct Test {
    cond: String,
    binds: Vec<(String, String)>,
}

struct Lowering<'a> {
    args: &'a MatchArgs,
    /// The expressions of the program the generated source names.
    held: Vec<py::Expr>,
}

impl Lowering<'_> {
    fn hold(&mut self, e: &py::Expr) -> String {
        self.held.push(e.clone());
        format!("__pv{}", self.held.len() - 1)
    }

    fn fail<T>(&self, what: &str, at: &impl Ranged) -> Result<T> {
        Err(Error::unsupported(what.to_string(), at))
    }

    fn test(&mut self, p: &py::Pattern, subject: &str) -> Result<Test> {
        match p {
            py::Pattern::MatchValue(v) => {
                let value = self.hold(&v.value);
                Ok(Test {
                    cond: format!("{subject} == {value}"),
                    binds: Vec::new(),
                })
            }
            py::Pattern::MatchSingleton(s) => {
                let word = match s.value {
                    py::Singleton::None => "None",
                    py::Singleton::True => "True",
                    py::Singleton::False => "False",
                };
                Ok(Test {
                    cond: format!("{subject} is {word}"),
                    binds: Vec::new(),
                })
            }
            py::Pattern::MatchAs(a) => {
                let mut t = match &a.pattern {
                    Some(inner) => self.test(inner, subject)?,
                    None => Test {
                        cond: "True".to_string(),
                        binds: Vec::new(),
                    },
                };
                if let Some(n) = &a.name {
                    t.binds.push((n.to_string(), subject.to_string()));
                }
                Ok(t)
            }
            py::Pattern::MatchOr(o) => {
                let mut conds = Vec::new();
                for alternative in &o.patterns {
                    let t = self.test(alternative, subject)?;
                    if !t.binds.is_empty() {
                        return self.fail("an or-pattern that captures names", &o.range);
                    }
                    conds.push(format!("({})", t.cond));
                }
                Ok(Test {
                    cond: conds.join(" or "),
                    binds: Vec::new(),
                })
            }
            py::Pattern::MatchSequence(s) => self.sequence(s, subject),
            py::Pattern::MatchMapping(m) => {
                if m.rest.is_some() {
                    return self.fail("a mapping pattern with **rest", &m.range);
                }
                let mut conds = vec![format!("isinstance({subject}, dict)")];
                let mut binds = Vec::new();
                for (key, pattern) in m.keys.iter().zip(m.patterns.iter()) {
                    let key = self.hold(key);
                    conds.push(format!("{key} in {subject}"));
                    let t = self.test(pattern, &format!("{subject}[{key}]"))?;
                    conds.push(t.cond);
                    binds.extend(t.binds);
                }
                Ok(Test {
                    cond: conds.join(" and "),
                    binds,
                })
            }
            py::Pattern::MatchClass(c) => self.class(c, subject),
            py::Pattern::MatchStar(s) => self.fail("a `*` pattern outside a sequence", &s.range),
        }
    }

    fn sequence(&mut self, s: &py::PatternMatchSequence, subject: &str) -> Result<Test> {
        let star = s
            .patterns
            .iter()
            .position(|p| matches!(p, py::Pattern::MatchStar(_)));
        let n = s.patterns.len();
        let mut conds = vec![format!("isinstance({subject}, (list, tuple))")];
        conds.push(match star {
            None => format!("len({subject}) == {n}"),
            Some(_) => format!("len({subject}) >= {}", n - 1),
        });
        let mut binds = Vec::new();
        for (i, p) in s.patterns.iter().enumerate() {
            if let py::Pattern::MatchStar(star) = p {
                if let Some(name) = &star.name {
                    let after = n - i - 1;
                    let slice = if after == 0 {
                        format!("list({subject}[{i}:])")
                    } else {
                        format!("list({subject}[{i}:len({subject}) - {after}])")
                    };
                    binds.push((name.to_string(), slice));
                }
                continue;
            }
            let at = match star {
                Some(st) if i > st => format!("-{}", n - i),
                _ => i.to_string(),
            };
            let t = self.test(p, &format!("{subject}[{at}]"))?;
            conds.push(t.cond);
            binds.extend(t.binds);
        }
        Ok(Test {
            cond: conds.join(" and "),
            binds,
        })
    }

    fn class(&mut self, c: &py::PatternMatchClass, subject: &str) -> Result<Test> {
        let class = self.hold(&c.cls);
        let mut conds = vec![format!("isinstance({subject}, {class})")];
        let mut binds = Vec::new();
        let builtin = matches!(
            &*c.cls,
            py::Expr::Name(n) if matches!(
                n.id.as_str(),
                "int" | "float" | "str" | "bool" | "bytes" | "list" | "tuple" | "dict" | "set" | "frozenset"
            )
        );
        let positional: Vec<String> = if c.arguments.patterns.is_empty() {
            Vec::new()
        } else if builtin {
            // `int(n)`: the one subpattern is the subject itself.
            if c.arguments.patterns.len() != 1 {
                return self.fail("a builtin class pattern with several subpatterns", &c.range);
            }
            vec![String::new()]
        } else {
            let py::Expr::Name(n) = &*c.cls else {
                return self.fail("a positional class pattern of a dotted class", &c.range);
            };
            let Some(names) = self.args.get(n.id.as_str()) else {
                return self.fail(
                    "a positional class pattern of a class with no __match_args__ or dataclass fields",
                    &c.range,
                );
            };
            if c.arguments.patterns.len() > names.len() {
                return self.fail(
                    "more subpatterns than the class has match arguments",
                    &c.range,
                );
            }
            names.clone()
        };
        for (p, attr) in c.arguments.patterns.iter().zip(&positional) {
            let sub = if attr.is_empty() {
                subject.to_string()
            } else {
                format!("{subject}.{attr}")
            };
            let t = self.test(p, &sub)?;
            conds.push(t.cond);
            binds.extend(t.binds);
        }
        for k in &c.arguments.keywords {
            let t = self.test(&k.pattern, &format!("{subject}.{}", k.attr))?;
            conds.push(t.cond);
            binds.extend(t.binds);
        }
        Ok(Test {
            cond: conds.join(" and "),
            binds,
        })
    }
}

/// The statements `m` stands for.
fn lower(m: py::StmtMatch, args: &MatchArgs, count: &mut usize) -> Result<Vec<py::Stmt>> {
    let held_name = format!("__match{}", *count);
    let flag = format!("__matched{}", *count);
    *count += 1;
    let guarded = m.cases.iter().any(|c| c.guard.is_some());
    let mut lowering = Lowering {
        args,
        held: Vec::new(),
    };
    // The source of the whole statement, with the bodies as `pass`.
    let mut source = format!("{held_name} = __subject\n");
    if guarded {
        source.push_str(&format!("{flag} = True\n"));
    }
    let mut bodies: Vec<Vec<py::Stmt>> = Vec::new();
    let last = m.cases.len().saturating_sub(1);
    let mut chained = false;
    for (i, case) in m.cases.into_iter().enumerate() {
        let t = lowering.test(&case.pattern, &held_name)?;
        let mut body_source = String::new();
        for (name, value) in &t.binds {
            body_source.push_str(&format!("    {name} = {value}\n"));
        }
        let mut body: Vec<py::Stmt> = Vec::new();
        let guard = case.guard.as_deref().cloned();
        let ends_chain = t.cond == "True" && guard.is_none();
        if guarded {
            let cond = if t.cond == "True" {
                flag.clone()
            } else {
                format!("{flag} and ({})", t.cond)
            };
            source.push_str(&format!("if {cond}:\n{body_source}"));
            if let Some(g) = &guard {
                lowering.held.push(g.clone());
                source.push_str(&format!(
                    "    if __pv{}:\n        {flag} = False\n        pass\n",
                    lowering.held.len() - 1
                ));
            } else {
                source.push_str(&format!("    {flag} = False\n    pass\n"));
            }
        } else {
            let keyword = if chained { "elif" } else { "if" };
            if i == last && ends_chain && chained {
                source.push_str(&format!("else:\n{body_source}    pass\n"));
            } else {
                source.push_str(&format!("{keyword} {}:\n{body_source}    pass\n", t.cond));
            }
            chained = true;
        }
        body.extend(case.body);
        bodies.push(body);
    }
    let mut stmts: Vec<py::Stmt> = ruff_python_parser::parse_module(&source)
        .expect("a match lowers to source that parses")
        .into_syntax()
        .body
        .into_iter()
        .collect();
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    fill.insert("__subject".to_string(), *m.subject);
    for (i, e) in lowering.held.into_iter().enumerate() {
        fill.insert(format!("__pv{i}"), e);
    }
    let fill = Fill(fill);
    for s in stmts.iter_mut() {
        walk_stmt(&fill, s);
    }
    // Each body goes where its `pass` is, in case order.
    let mut bodies = bodies.into_iter();
    for s in stmts.iter_mut() {
        let py::Stmt::If(chain) = s else {
            continue;
        };
        if guarded {
            let body = bodies.next().unwrap_or_default();
            // A guard makes the last statement an `if` of its own.
            if let Some(py::Stmt::If(inner)) = chain.body.last_mut() {
                inner.body.pop();
                inner.body.extend(body);
            } else {
                chain.body.pop();
                chain.body.extend(body);
            }
        } else {
            let body = bodies.next().unwrap_or_default();
            chain.body.pop();
            chain.body.extend(body);
            for clause in chain.elif_else_clauses.iter_mut() {
                let body = bodies.next().unwrap_or_default();
                clause.body.pop();
                clause.body.extend(body);
            }
        }
    }
    Ok(stmts)
}
