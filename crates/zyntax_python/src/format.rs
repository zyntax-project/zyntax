//! f-strings and the format spec mini-language.
//!
//! A spec such as `>+08.3f` is parsed here once into its fields and
//! handed to the library's formatters as scalar arguments, so the
//! program pays for nothing it did not write.

use crate::lower::{Lowerer, Node, Val};
use crate::types::Ty;
use crate::{Error, Result, span_of};
use ruff_python_ast as py;
use zyntax_typed_ast::BinaryOp;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedExpression, TypedLiteral};

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

/// One conversion of a `%` format: `%[flags][width][.precision]type`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PercentField {
    pub left: bool,
    pub zero: bool,
    pub plus: bool,
    pub space: bool,
    pub alt: bool,
    /// -1 when unset.
    pub width: i64,
    /// -1 when unset.
    pub precision: i64,
    pub conversion: char,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Percent {
    Text(String),
    Field(PercentField),
}

/// Split a `%` format into its literal text and its conversions. A
/// `*` width, a mapping key `%(name)s` and a length modifier are not
/// taken.
pub(crate) fn parse_percent(text: &str) -> std::result::Result<Vec<Percent>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut literal = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '%' {
            literal.push(chars[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i < chars.len() && chars[i] == '%' {
            literal.push('%');
            i += 1;
            continue;
        }
        if !literal.is_empty() {
            out.push(Percent::Text(std::mem::take(&mut literal)));
        }
        let mut field = PercentField {
            left: false,
            zero: false,
            plus: false,
            space: false,
            alt: false,
            width: -1,
            precision: -1,
            conversion: ' ',
        };
        while i < chars.len() && matches!(chars[i], '-' | '0' | '+' | ' ' | '#') {
            match chars[i] {
                '-' => field.left = true,
                '0' => field.zero = true,
                '+' => field.plus = true,
                ' ' => field.space = true,
                _ => field.alt = true,
            }
            i += 1;
        }
        let number = |i: &mut usize| -> Option<i64> {
            let start = *i;
            while *i < chars.len() && chars[*i].is_ascii_digit() {
                *i += 1;
            }
            chars[start..*i].iter().collect::<String>().parse().ok()
        };
        if i < chars.len() && chars[i] == '*' {
            return Err("a `*` width".to_string());
        }
        if let Some(w) = number(&mut i) {
            field.width = w;
        }
        if i < chars.len() && chars[i] == '.' {
            i += 1;
            field.precision = number(&mut i).unwrap_or(0);
        }
        if i < chars.len() && chars[i] == '(' {
            return Err("a mapping key".to_string());
        }
        let Some(&conversion) = chars.get(i) else {
            return Err("an incomplete conversion".to_string());
        };
        i += 1;
        field.conversion = conversion;
        out.push(Percent::Field(field));
    }
    if !literal.is_empty() {
        out.push(Percent::Text(literal));
    }
    Ok(out)
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
    if let Some(c) = chars.get(i)
        && "bcdeEfFgGnosxX%".contains(*c)
    {
        spec.ty = *c as i64;
        i += 1;
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

    /// `"...{}...".format(args)` with a literal template: each field is
    /// the argument it names, positional by count or index or keyword
    /// by name, converted and formatted as an f-string's would be.
    pub(crate) fn str_format(
        &mut self,
        template: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        span: Span,
    ) -> Result<Val> {
        let mut pieces: Vec<Node> = Vec::new();
        let mut literal = String::new();
        let mut next_positional = 0usize;
        let chars: Vec<char> = template.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '{' && chars.get(i + 1) == Some(&'{') {
                literal.push('{');
                i += 2;
                continue;
            }
            if c == '}' && chars.get(i + 1) == Some(&'}') {
                literal.push('}');
                i += 2;
                continue;
            }
            if c != '{' {
                literal.push(c);
                i += 1;
                continue;
            }
            // A field: `{name!conv:spec}` up to the matching brace.
            let start = i + 1;
            let mut depth = 1;
            let mut end = start;
            while end < chars.len() {
                match chars[end] {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                end += 1;
            }
            if end >= chars.len() {
                return Err(Error::unsupported_span(
                    "str.format with an unclosed `{`".to_string(),
                    span,
                ));
            }
            let field: String = chars[start..end].iter().collect();
            i = end + 1;
            if !literal.is_empty() {
                pieces.push(crate::lower::str_lit(&literal, span));
                literal.clear();
            }
            let (name, rest) = match field.find(['!', ':']) {
                Some(at) => (&field[..at], &field[at..]),
                None => (field.as_str(), ""),
            };
            let (conversion, spec) = match rest.strip_prefix('!') {
                Some(r) => {
                    let conv = r.chars().next();
                    let spec = r.get(1..).and_then(|t| t.strip_prefix(':')).unwrap_or("");
                    (conv, spec)
                }
                None => (None, rest.strip_prefix(':').unwrap_or(rest)),
            };
            let arg = if name.is_empty() {
                let a = args.get(next_positional);
                next_positional += 1;
                a
            } else if let Ok(index) = name.parse::<usize>() {
                args.get(index)
            } else {
                keywords
                    .iter()
                    .find(|k| k.arg.as_ref().is_some_and(|a| a.as_str() == name))
                    .map(|k| &k.value)
            };
            let Some(arg) = arg else {
                return Err(Error::unsupported_span(
                    format!("str.format with no argument for `{{{field}}}`"),
                    span,
                ));
            };
            let value = self.expr(arg)?;
            let value = match conversion {
                Some('r') | Some('a') => Val {
                    node: self.repr_of(value),
                    ty: Ty::Str,
                },
                Some('s') => Val {
                    node: self.str_of(value),
                    ty: Ty::Str,
                },
                Some(other) => {
                    return Err(Error::unsupported_span(
                        format!("str.format conversion `!{other}`"),
                        span,
                    ));
                }
                None => value,
            };
            if spec.is_empty() {
                pieces.push(self.str_of(value));
            } else {
                let parsed = parse_spec(spec).map_err(|message| {
                    Error::unsupported_span(format!("format spec `{spec}` ({message})"), span)
                })?;
                pieces.push(self.format(value, &parsed, span));
            }
        }
        if !literal.is_empty() {
            pieces.push(crate::lower::str_lit(&literal, span));
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
                    return Err(Error::unsupported_span(
                        "the `=` debug form in an f-string".to_string(),
                        span,
                    ));
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
                let spec = parse_spec(&spec).map_err(|message| {
                    Error::unsupported_span(format!("format spec `{spec}` ({message})"), span)
                })?;
                Ok(self.format(value, &spec, span))
            }
        }
    }

    /// `"..." % values` with a literal format: each conversion is
    /// mapped onto the format mini-language and the pieces concatenated.
    /// The values are a tuple literal, one per conversion, or a single
    /// value for a single conversion.
    pub(crate) fn percent_format(
        &mut self,
        template: &str,
        values: &py::Expr,
        span: Span,
    ) -> Result<Val> {
        let conversions = parse_percent(template)
            .map_err(|message| Error::unsupported_span(format!("`%` format ({message})"), span))?;
        let wanted = conversions
            .iter()
            .filter(|p| matches!(p, Percent::Field(_)))
            .count();
        let args: Vec<&py::Expr> = match values {
            py::Expr::Tuple(t) => t.elts.iter().collect(),
            single => vec![single],
        };
        if args.len() != wanted {
            return Err(Error::unsupported_span(
                format!(
                    "`%` format with {wanted} conversion(s) and {} value(s)",
                    args.len()
                ),
                span,
            ));
        }
        let mut next = args.into_iter();
        let mut pieces: Vec<Node> = Vec::new();
        for piece in conversions {
            match piece {
                Percent::Text(t) => pieces.push(crate::lower::str_lit(&t, span)),
                Percent::Field(field) => {
                    let value = self.expr(next.next().expect("counted"))?;
                    pieces.push(self.percent_field(value, &field, span)?);
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

    pub(crate) fn percent_field(
        &mut self,
        value: Val,
        field: &PercentField,
        span: Span,
    ) -> Result<Node> {
        let mut spec = Spec {
            fill: if field.zero { "0" } else { " " }.to_string(),
            align: if field.left {
                1
            } else if field.zero {
                4
            } else {
                2
            },
            sign: if field.plus {
                1
            } else if field.space {
                2
            } else {
                0
            },
            alt: field.alt,
            zero: field.zero,
            width: field.width,
            grouping: 0,
            precision: field.precision,
            ty: 0,
        };
        let conversion = field.conversion;
        let value = match conversion {
            's' => Val {
                node: self.str_of(value),
                ty: Ty::Str,
            },
            'r' | 'a' => Val {
                node: self.repr_of(value),
                ty: Ty::Str,
            },
            'c' => {
                let node = match value.ty {
                    Ty::Str => value.node,
                    _ => {
                        let code = self.coerce(value, Ty::Int);
                        crate::lower::call("zb_str_chr", vec![code], Ty::Str, span)
                    }
                };
                Val { node, ty: Ty::Str }
            }
            'd' | 'i' | 'u' => {
                spec.ty = 'd' as i64;
                match value.ty {
                    Ty::Float => Val {
                        node: crate::lower::cast(value.node, Ty::Int, span),
                        ty: Ty::Int,
                    },
                    // A float found at run time truncates as a typed one.
                    Ty::Object => {
                        let node = crate::lower::call(
                            "zb_any_pct_int",
                            vec![value.node],
                            Ty::Object,
                            span,
                        );
                        self.checked(Val {
                            node,
                            ty: Ty::Object,
                        })
                    }
                    _ => value,
                }
            }
            'x' | 'X' | 'o' => {
                spec.ty = conversion as i64;
                value
            }
            'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                spec.ty = conversion as i64;
                if spec.precision < 0 {
                    spec.precision = 6;
                }
                value
            }
            other => {
                return Err(Error::unsupported_span(
                    format!("`%{other}` in a format"),
                    span,
                ));
            }
        };
        // A string conversion with a precision is a truncation.
        if matches!(conversion, 's' | 'r' | 'a' | 'c') {
            spec.ty = 's' as i64;
            if spec.width < 0 && spec.precision < 0 {
                return Ok(value.node);
            }
        }
        Ok(self.format(value, &spec, span))
    }

    /// The spec's text, which must be literal here.
    fn spec_text(&self, spec: &py::InterpolatedStringFormatSpec) -> Result<String> {
        let mut text = String::new();
        for element in spec.elements.iter() {
            match element {
                py::InterpolatedStringElement::Literal(l) => text.push_str(&l.value),
                py::InterpolatedStringElement::Interpolation(e) => {
                    return Err(Error::unsupported(
                        "a nested expression in a format spec".to_string(),
                        &e,
                    ));
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
