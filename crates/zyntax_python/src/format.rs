//! f-strings and the format spec mini-language.
//!
//! A spec such as `>+08.3f` is parsed here once into its fields and
//! handed to the library's formatters as scalar arguments, so the
//! program pays for nothing it did not write.

use crate::lower::{Lowerer, Node, Val};
use crate::types::Ty;
use crate::{span_of, Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedExpression, TypedLiteral};
use zyntax_typed_ast::BinaryOp;

/// The fields of a format spec, with the library's encodings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Spec {
    pub fill: String,
    /// 0 unset, 1 `<`, 2 `>`, 3 `^`, 4 `=`.
    pub align: i64,
    /// 0 `-`, 1 `+`, 2 space.
    pub sign: i64,
    pub alt: bool,
    pub zero: bool,
    /// -1 when unset.
    pub width: i64,
    /// 0 none, 1 `,`, 2 `_`.
    pub grouping: i64,
    /// -1 when unset.
    pub precision: i64,
    /// The presentation type's character, 0 when unset.
    pub ty: i64,
}

impl Spec {
    /// Whether the spec asks for a float presentation.
    fn wants_float(&self) -> bool {
        matches!(
            self.ty_char(),
            Some('f' | 'F' | 'e' | 'E' | 'g' | 'G' | '%')
        )
    }

    fn ty_char(&self) -> Option<char> {
        char::from_u32(self.ty as u32).filter(|_| self.ty != 0)
    }
}

/// Parse `[[fill]align][sign][z][#][0][width][grouping][.precision][type]`.
pub(crate) fn parse_spec(text: &str) -> std::result::Result<Spec, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut spec = Spec {
        fill: " ".to_string(),
        align: 0,
        sign: 0,
        alt: false,
        zero: false,
        width: -1,
        grouping: 0,
        precision: -1,
        ty: 0,
    };
    let align_of = |c: char| match c {
        '<' => Some(1),
        '>' => Some(2),
        '^' => Some(3),
        '=' => Some(4),
        _ => None,
    };
    if chars.len() >= 2 && align_of(chars[1]).is_some() {
        spec.fill = chars[0].to_string();
        spec.align = align_of(chars[1]).unwrap();
        i = 2;
    } else if let Some(a) = chars.first().and_then(|c| align_of(*c)) {
        spec.align = a;
        i = 1;
    }
    match chars.get(i) {
        Some('+') => {
            spec.sign = 1;
            i += 1;
        }
        Some('-') => i += 1,
        Some(' ') => {
            spec.sign = 2;
            i += 1;
        }
        _ => {}
    }
    if chars.get(i) == Some(&'z') {
        i += 1;
    }
    if chars.get(i) == Some(&'#') {
        spec.alt = true;
        i += 1;
    }
    if chars.get(i) == Some(&'0') {
        spec.zero = true;
        i += 1;
    }
    let mut digits = String::new();
    while let Some(c) = chars.get(i).filter(|c| c.is_ascii_digit()) {
        digits.push(*c);
        i += 1;
    }
    if !digits.is_empty() {
        spec.width = digits.parse().map_err(|_| "width too large".to_string())?;
    }
    match chars.get(i) {
        Some(',') => {
            spec.grouping = 1;
            i += 1;
        }
        Some('_') => {
            spec.grouping = 2;
            i += 1;
        }
        _ => {}
    }
    if chars.get(i) == Some(&'.') {
        i += 1;
        let mut digits = String::new();
        while let Some(c) = chars.get(i).filter(|c| c.is_ascii_digit()) {
            digits.push(*c);
            i += 1;
        }
        if digits.is_empty() {
            return Err("Format specifier missing precision".to_string());
        }
        spec.precision = digits
            .parse()
            .map_err(|_| "precision too large".to_string())?;
    }
    if let Some(c) = chars.get(i) {
        if "bcdeEfFgGnosxX%".contains(*c) {
            spec.ty = *c as i64;
            i += 1;
        }
    }
    if i != chars.len() {
        return Err(format!("Invalid format specifier '{text}'"));
    }
    Ok(spec)
}

impl Lowerer<'_> {
    /// An f-string: its pieces concatenated left to right.
    pub(crate) fn fstring(&mut self, f: &py::ExprFString, span: Span) -> Result<Val> {
        let mut pieces: Vec<Node> = Vec::new();
        for part in f.value.iter() {
            match part {
                py::FStringPart::Literal(s) => {
                    pieces.push(crate::lower::str_lit(&s.value, span_of(s)));
                }
                py::FStringPart::FString(fs) => {
                    for element in fs.elements.iter() {
                        pieces.push(self.fstring_element(element)?);
                    }
                }
            }
        }
        let mut it = pieces.into_iter();
        let first = it.next().unwrap_or_else(|| crate::lower::str_lit("", span));
        let node = it.fold(first, |acc, piece| {
            crate::lower::binary(BinaryOp::Add, acc, piece, Ty::Str, span)
        });
        Ok(Val { node, ty: Ty::Str })
    }

    fn fstring_element(&mut self, element: &py::InterpolatedStringElement) -> Result<Node> {
        match element {
            py::InterpolatedStringElement::Literal(l) => {
                Ok(crate::lower::str_lit(&l.value, span_of(l)))
            }
            py::InterpolatedStringElement::Interpolation(e) => {
                let span = span_of(e);
                let value = self.expr(&e.expression)?;
                if e.debug_text.is_some() {
                    return Err(Error::Unsupported {
                        what: "the `=` debug form in an f-string".to_string(),
                        at: span.start,
                    });
                }
                // `!r` and `!a` convert before formatting; `!s` is str().
                let value = match e.conversion {
                    py::ConversionFlag::None => value,
                    py::ConversionFlag::Str => Val {
                        node: self.str_of(value),
                        ty: Ty::Str,
                    },
                    py::ConversionFlag::Repr | py::ConversionFlag::Ascii => Val {
                        node: self.repr_of(value),
                        ty: Ty::Str,
                    },
                };
                let Some(spec) = &e.format_spec else {
                    return Ok(self.str_of(value));
                };
                let spec = self.spec_text(spec)?;
                let spec = parse_spec(&spec).map_err(|message| Error::Unsupported {
                    what: format!("format spec `{spec}` ({message})"),
                    at: span.start,
                })?;
                Ok(self.format(value, &spec, span))
            }
        }
    }

    /// The spec's text, which must be literal here.
    fn spec_text(&self, spec: &py::InterpolatedStringFormatSpec) -> Result<String> {
        let mut text = String::new();
        for element in spec.elements.iter() {
            match element {
                py::InterpolatedStringElement::Literal(l) => text.push_str(&l.value),
                py::InterpolatedStringElement::Interpolation(e) => {
                    return Err(Error::Unsupported {
                        what: "a nested expression in a format spec".to_string(),
                        at: e.range().start().to_usize(),
                    })
                }
            }
        }
        Ok(text)
    }

    /// `format(value, spec)`.
    pub(crate) fn format(&mut self, value: Val, spec: &Spec, span: Span) -> Node {
        let lit_str = |s: &str| crate::lower::str_lit(s, span);
        let lit_int = |v: i64| crate::lower::int_lit(v, span);
        let lit_bool = |b: bool| {
            crate::lower::node(
                TypedExpression::Literal(TypedLiteral::Bool(b)),
                Ty::Bool,
                span,
            )
        };
        let common =
            |spec: &Spec| vec![lit_str(&spec.fill), lit_int(spec.align), lit_int(spec.sign)];
        match value.ty {
            Ty::Int | Ty::Bool if spec.wants_float() => {
                let v = self.coerce(value, Ty::Float);
                let mut args = vec![v];
                args.extend(common(spec));
                args.extend([
                    lit_bool(spec.zero),
                    lit_int(spec.width),
                    lit_int(spec.grouping),
                    lit_int(spec.precision),
                    lit_int(spec.ty),
                ]);
                crate::lower::call("zb_fmt_float", args, Ty::Str, span)
            }
            Ty::Bool if spec.ty == 0 => {
                let text = self.str_of(value);
                self.format(
                    Val {
                        node: text,
                        ty: Ty::Str,
                    },
                    spec,
                    span,
                )
            }
            Ty::Int | Ty::Bool => {
                let v = self.coerce(value, Ty::Int);
                let mut args = vec![v];
                args.extend(common(spec));
                args.extend([
                    lit_bool(spec.alt),
                    lit_bool(spec.zero),
                    lit_int(spec.width),
                    lit_int(spec.grouping),
                    lit_int(spec.ty),
                ]);
                crate::lower::call("zb_fmt_int", args, Ty::Str, span)
            }
            Ty::Float => {
                let mut args = vec![value.node];
                args.extend(common(spec));
                args.extend([
                    lit_bool(spec.zero),
                    lit_int(spec.width),
                    lit_int(spec.grouping),
                    lit_int(spec.precision),
                    lit_int(spec.ty),
                ]);
                crate::lower::call("zb_fmt_float", args, Ty::Str, span)
            }
            Ty::Object | Ty::Unknown => {
                let mut args = vec![value.node];
                args.extend(common(spec));
                args.extend([
                    lit_bool(spec.alt),
                    lit_bool(spec.zero),
                    lit_int(spec.width),
                    lit_int(spec.grouping),
                    lit_int(spec.precision),
                    lit_int(spec.ty),
                ]);
                crate::lower::call("zb_any_fmt", args, Ty::Str, span)
            }
            _ => {
                let text = self.str_of(value);
                crate::lower::call(
                    "zb_fmt_str",
                    vec![
                        text,
                        lit_str(&spec.fill),
                        lit_int(spec.align),
                        lit_int(spec.width),
                        lit_int(spec.precision),
                    ],
                    Ty::Str,
                    span,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_spec;

    #[test]
    fn specs_parse_into_fields() {
        let s = parse_spec("05d").unwrap();
        assert!(s.zero);
        assert_eq!(s.width, 5);
        assert_eq!(s.ty, 'd' as i64);
        let s = parse_spec(".3f").unwrap();
        assert_eq!(s.precision, 3);
        assert_eq!(s.ty, 'f' as i64);
        let s = parse_spec("*^+10,.2f").unwrap();
        assert_eq!(s.fill, "*");
        assert_eq!(s.align, 3);
        assert_eq!(s.sign, 1);
        assert_eq!(s.width, 10);
        assert_eq!(s.grouping, 1);
        assert_eq!(s.precision, 2);
        assert!(parse_spec("q").is_err());
    }
}
