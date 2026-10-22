//! The `Math` global: constants, compiler intrinsics where the semantics match,
//! and native primitives for the ECMAScript edge cases.

use super::*;

/// Shared compiler intrinsics, reached by the source-level name the compiler aliases.
const INTRINSICS: &[(&str, &str)] = &[("sqrt", "sqrt"), ("abs", "abs"), ("floor", "floor")];

/// `$TS$math_<name>` primitives, with their arity.
const PRIMITIVES: &[(&str, usize)] = &[
    ("ceil", 1),
    ("trunc", 1),
    ("round", 1),
    ("sin", 1),
    ("cos", 1),
    ("tan", 1),
    ("asin", 1),
    ("acos", 1),
    ("atan", 1),
    ("atan2", 2),
    ("exp", 1),
    ("log", 1),
    ("log2", 1),
    ("log10", 1),
    ("cbrt", 1),
    ("min", 2),
    ("max", 2),
];

pub(super) fn primitive_externs() -> impl Iterator<Item = (String, usize)> {
    PRIMITIVES
        .iter()
        .map(|(name, arity)| (format!("$TS$math_{name}"), *arity))
}

fn constant(name: &str) -> Option<f64> {
    use std::f64::consts;
    Some(match name {
        "PI" => consts::PI,
        "E" => consts::E,
        "LN2" => consts::LN_2,
        "LN10" => consts::LN_10,
        "LOG2E" => consts::LOG2_E,
        "LOG10E" => consts::LOG10_E,
        "SQRT2" => consts::SQRT_2,
        "SQRT1_2" => consts::FRAC_1_SQRT_2,
        _ => return None,
    })
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// True when `Math` names the global, not a local, import or namespace.
    fn is_math_global(&self, object: &js::Expression<'ast>) -> bool {
        let js::Expression::Identifier(id) = object else {
            return false;
        };
        id.name == "Math"
            && !self.namespace_references.contains_key(&id.node_id.get())
            && !self.imported_exports.contains_key(&self.name("Math"))
            && !self
                .scopes
                .iter()
                .any(|s| s.contains_key(&self.name("Math")))
    }

    pub(super) fn math_constant(&self, m: &js::StaticMemberExpression<'ast>) -> Option<Expr> {
        if !self.is_math_global(&m.object) {
            return None;
        }
        constant(m.property.name.as_str()).map(|value| num(value, m.span))
    }

    /// `Math.name(...)`; `None` when the callee is not the `Math` global.
    pub(super) fn math_call(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Option<Expr>> {
        if !self.is_math_global(&m.object) {
            return Ok(None);
        }
        let at = c.span;
        let name = m.property.name.as_str();
        let variadic = matches!(name, "min" | "max");
        let (callee, arity) = if let Some((_, alias)) = INTRINSICS.iter().find(|(n, _)| *n == name)
        {
            (intern(alias), 1)
        } else if name == "pow" {
            (intern("$TS$pow"), 2)
        } else if let Some((_, arity)) = PRIMITIVES.iter().find(|(n, _)| *n == name) {
            (intern(&format!("$TS$math_{name}")), *arity)
        } else {
            return self.unsupported("this Math function", m.span);
        };
        if c.type_arguments.is_some() {
            return self.error(ErrorKind::Type, "Math functions take no type arguments", at);
        }
        let hints = [number(), number()];
        let mut args = self.arguments(&c.arguments, at, hints.iter().cycle(), None)?;
        if variadic && args.len() > 2 {
            // Fold left; each step keeps NaN and signed-zero behaviour.
            for arg in &args {
                self.expect(&arg.ty, &number(), at)?;
            }
            let mut rest = args.into_iter();
            let mut result = rest.next().unwrap();
            for next in rest {
                result = call(callee, vec![result, next], number(), at);
            }
            return Ok(Some(result));
        }
        if variadic && args.is_empty() {
            return self.unsupported("Math.min and Math.max without arguments", at);
        }
        if variadic && args.len() == 1 {
            self.expect(&args[0].ty, &number(), at)?;
            return Ok(Some(args.remove(0)));
        }
        if args.len() != arity {
            return self.error(ErrorKind::Type, "wrong argument count", at);
        }
        for arg in &args {
            self.expect(&arg.ty, &number(), at)?;
        }
        Ok(Some(call(callee, args, number(), at)))
    }
}
