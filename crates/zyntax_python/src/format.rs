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

/// The width and precision a spec takes from fields nested in it
/// (`{:>{w}.{p}f}`), which are known when the program runs.
#[derive(Default)]
pub(crate) struct Nested {
    pub width: Option<Node>,
    pub precision: Option<Node>,
}

/// A spec's text with each nested field replaced by `1`, and the fields
/// in order: whether each is the precision (it follows the `.`) and what
/// it holds. A field anywhere but the width or precision is refused.
pub(crate) fn split_nested<T>(
    parts: impl IntoIterator<Item = SpecPart<T>>,
) -> std::result::Result<(String, Vec<(bool, T)>), String> {
    let mut text = String::new();
    let mut fields: Vec<(bool, T)> = Vec::new();
    for part in parts {
        match part {
            SpecPart::Text(t) => text.push_str(&t),
            SpecPart::Field(source) => {
                let precision = text.ends_with('.');
                // A lone `0` before the field is the zero flag.
                let digits = text.chars().rev().take_while(char::is_ascii_digit).count();
                if (digits > 0 && !(digits == 1 && text.ends_with('0')))
                    || fields.iter().any(|(p, _)| *p == precision)
                {
                    return Err("a nested field must be the whole width or precision".to_string());
                }
                text.push('1');
                fields.push((precision, source));
            }
        }
    }
    Ok((text, fields))
}

/// A piece of a spec: literal text, or a nested field.
pub(crate) enum SpecPart<T> {
    Text(String),
    Field(T),
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
    /// `%(key)s`: the mapping key the value is read by.
    pub key: Option<String>,
    /// `%*d`, `%.*f`: the width or precision is the next value.
    pub star_width: bool,
    pub star_precision: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Percent {
    Text(String),
    Field(PercentField),
}

/// Split a `%` format into its literal text and its conversions. A
/// length modifier is not taken.
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
            key: None,
            star_width: false,
            star_precision: false,
        };
        if i < chars.len() && chars[i] == '(' {
            let Some(close) = chars[i..].iter().position(|&c| c == ')') else {
                return Err("an unclosed mapping key".to_string());
            };
            field.key = Some(chars[i + 1..i + close].iter().collect());
            i += close + 1;
        }
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
            field.star_width = true;
            field.width = 1;
            i += 1;
        } else if let Some(w) = number(&mut i) {
            field.width = w;
        }
        if i < chars.len() && chars[i] == '.' {
            i += 1;
            if i < chars.len() && chars[i] == '*' {
                field.star_precision = true;
                field.precision = 1;
                i += 1;
            } else {
                field.precision = number(&mut i).unwrap_or(0);
            }
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

/// The pieces of a concatenation flattened into `out`, in order: a sum
/// of `ty` that is itself a sum contributes its own operands.
fn flatten_sum(node: Node, ty: &zyntax_typed_ast::Type, out: &mut Vec<Node>) {
    if let TypedExpression::Binary(b) = &node.node
        && b.op == BinaryOp::Add
        && node.ty == *ty
    {
        let TypedExpression::Binary(b) = node.node else {
            unreachable!("matched above")
        };
        flatten_sum(*b.left, ty, out);
        flatten_sum(*b.right, ty, out);
    } else {
        out.push(node);
    }
}

/// Most pieces one concatenation call takes.
const MAX_CONCAT: usize = 8;

/// The pieces, strings or bytes per `ty`, as one value in order. Three
/// or more go to a call that builds the result in one allocation; a
/// longer run is joined `MAX_CONCAT` at a time, each call taking the
/// result so far as its first piece. None for no pieces.
pub(crate) fn concat_pieces(pieces: Vec<Node>, ty: Ty, span: Span) -> Option<Node> {
    let element = crate::lower::ir(ty);
    let mut flat = Vec::with_capacity(pieces.len());
    for piece in pieces {
        flatten_sum(piece, &element, &mut flat);
    }
    let mut rest = flat.into_iter();
    let mut acc = rest.next()?;
    let mut pending: Vec<Node> = Vec::new();
    loop {
        pending.clear();
        let room = MAX_CONCAT - 1;
        pending.extend(rest.by_ref().take(room));
        if pending.is_empty() {
            return Some(acc);
        }
        acc = if pending.len() == 1 {
            let piece = pending.pop().expect("one piece");
            crate::lower::binary(BinaryOp::Add, acc, piece, ty, span)
        } else {
            let mut args = vec![acc];
            args.append(&mut pending);
            let name = format!("zb_str_concat{}", args.len());
            crate::lower::call(&name, args, ty, span)
        };
    }
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
        let node =
            concat_pieces(pieces, Ty::Str, span).unwrap_or_else(|| crate::lower::str_lit("", span));
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
            let lookup = |name: &str, next: &mut usize| -> Option<&py::Expr> {
                if name.is_empty() {
                    let a = args.get(*next);
                    *next += 1;
                    a
                } else if let Ok(index) = name.parse::<usize>() {
                    args.get(index)
                } else {
                    keywords
                        .iter()
                        .find(|k| k.arg.as_ref().is_some_and(|a| a.as_str() == name))
                        .map(|k| &k.value)
                }
            };
            // `{0.attr}` and `{1[key]}` read from the argument.
            let (name, accessors) = match name.find(['.', '[']) {
                Some(at) => (&name[..at], &name[at..]),
                None => (name, ""),
            };
            let arg = lookup(name, &mut next_positional);
            let owned;
            let arg = match arg {
                Some(base) if !accessors.is_empty() => {
                    let Some(read) = field_accessors(base, accessors) else {
                        return Err(Error::unsupported_span(
                            format!("str.format field `{{{field}}}`"),
                            span,
                        ));
                    };
                    owned = read;
                    Some(&owned)
                }
                other => other,
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
                // Fields nested in the spec name arguments too.
                let mut parts = Vec::new();
                let mut rest = spec;
                while let Some(open) = rest.find('{') {
                    parts.push(SpecPart::Text(rest[..open].to_string()));
                    let Some(close) = rest[open..].find('}') else {
                        return Err(Error::unsupported_span(
                            "str.format with an unclosed `{` in a spec".to_string(),
                            span,
                        ));
                    };
                    let inner = &rest[open + 1..open + close];
                    let Some(e) = lookup(inner, &mut next_positional) else {
                        return Err(Error::unsupported_span(
                            format!("str.format with no argument for `{{{inner}}}`"),
                            span,
                        ));
                    };
                    parts.push(SpecPart::Field(e));
                    rest = &rest[open + close + 1..];
                }
                parts.push(SpecPart::Text(rest.to_string()));
                let (text, fields) = split_nested(parts)
                    .map_err(|message| Error::unsupported_span(message, span))?;
                let parsed = parse_spec(&text).map_err(|message| {
                    Error::unsupported_span(format!("format spec `{text}` ({message})"), span)
                })?;
                let mut nested = Nested::default();
                for (is_precision, e) in fields {
                    let n = self.expr_as(e, Ty::Int)?;
                    if is_precision {
                        nested.precision = Some(n);
                    } else {
                        nested.width = Some(n);
                    }
                }
                pieces.push(self.format_with(value, &parsed, &nested, span));
            }
        }
        if !literal.is_empty() {
            pieces.push(crate::lower::str_lit(&literal, span));
        }
        let node =
            concat_pieces(pieces, Ty::Str, span).unwrap_or_else(|| crate::lower::str_lit("", span));
        Ok(Val { node, ty: Ty::Str })
    }

    fn fstring_element(&mut self, element: &py::InterpolatedStringElement) -> Result<Node> {
        match element {
            py::InterpolatedStringElement::Literal(l) => {
                Ok(crate::lower::str_lit(&l.value, span_of(l)))
            }
            py::InterpolatedStringElement::Interpolation(e) => {
                let span = span_of(e);
                // `{x=}` shows the source text first, and converts with
                // `!r` unless it has a spec or a conversion.
                if let Some(debug) = &e.debug_text {
                    let text = crate::lower::str_lit(debug.as_str(), span);
                    let mut inner = e.clone();
                    inner.debug_text = None;
                    if inner.format_spec.is_none()
                        && matches!(inner.conversion, py::ConversionFlag::None)
                    {
                        inner.conversion = py::ConversionFlag::Repr;
                    }
                    let shown =
                        self.fstring_element(&py::InterpolatedStringElement::Interpolation(inner))?;
                    return Ok(concat_pieces(vec![text, shown], Ty::Str, span).expect("two pieces"));
                }
                let value = self.expr(&e.expression)?;
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
                let mut parts = Vec::new();
                for element in spec.elements.iter() {
                    parts.push(match element {
                        py::InterpolatedStringElement::Literal(l) => {
                            SpecPart::Text(l.value.to_string())
                        }
                        py::InterpolatedStringElement::Interpolation(n)
                            if n.format_spec.is_none()
                                && matches!(n.conversion, py::ConversionFlag::None)
                                && n.debug_text.is_none() =>
                        {
                            SpecPart::Field(&*n.expression)
                        }
                        py::InterpolatedStringElement::Interpolation(n) => {
                            return Err(Error::unsupported(
                                "a conversion or spec inside a format spec".to_string(),
                                &n,
                            ));
                        }
                    });
                }
                let (text, fields) = split_nested(parts)
                    .map_err(|message| Error::unsupported_span(message, span))?;
                let parsed = parse_spec(&text).map_err(|message| {
                    Error::unsupported_span(format!("format spec `{text}` ({message})"), span)
                })?;
                let mut nested = Nested::default();
                for (is_precision, e) in fields {
                    let n = self.expr_as(e, Ty::Int)?;
                    if is_precision {
                        nested.precision = Some(n);
                    } else {
                        nested.width = Some(n);
                    }
                }
                Ok(self.format_with(value, &parsed, &nested, span))
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
        let keyed = conversions
            .iter()
            .any(|p| matches!(p, Percent::Field(f) if f.key.is_some()));
        if keyed {
            return self.percent_format_keyed(conversions, values, span);
        }
        let args: Vec<&py::Expr> = match values {
            py::Expr::Tuple(t) => t.elts.iter().collect(),
            single => vec![single],
        };
        let stars: usize = conversions
            .iter()
            .map(|p| match p {
                Percent::Field(f) => f.star_width as usize + f.star_precision as usize,
                Percent::Text(_) => 0,
            })
            .sum();
        if args.len() != wanted + stars {
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
                    // A `*` takes its number from the values, ahead of
                    // the value it formats.
                    let mut nested = Nested::default();
                    if field.star_width {
                        nested.width = Some(self.expr_as(next.next().expect("counted"), Ty::Int)?);
                    }
                    if field.star_precision {
                        nested.precision =
                            Some(self.expr_as(next.next().expect("counted"), Ty::Int)?);
                    }
                    let value = self.expr(next.next().expect("counted"))?;
                    pieces.push(self.percent_field_with(value, &field, &nested, span)?);
                }
            }
        }
        let node =
            concat_pieces(pieces, Ty::Str, span).unwrap_or_else(|| crate::lower::str_lit("", span));
        Ok(Val { node, ty: Ty::Str })
    }

    /// `"%(a)s %(b)d" % mapping`: each conversion reads its key from a
    /// dict display or from a name bound to a dict.
    fn percent_format_keyed(
        &mut self,
        conversions: Vec<Percent>,
        values: &py::Expr,
        span: Span,
    ) -> Result<Val> {
        let refused = |what: &str| Error::unsupported_span(format!("`%` format ({what})"), span);
        let mut pieces: Vec<Node> = Vec::new();
        for piece in conversions {
            let Percent::Field(field) = piece else {
                let Percent::Text(t) = piece else {
                    unreachable!()
                };
                pieces.push(crate::lower::str_lit(&t, span));
                continue;
            };
            let Some(key) = &field.key else {
                return Err(refused("a mapping key mixed with a positional one"));
            };
            if field.star_width || field.star_precision {
                return Err(refused("a `*` beside a mapping key"));
            }
            let read = match values {
                py::Expr::Dict(d) => {
                    // The last entry of a key wins. The display is not
                    // built, so no value in it may call.
                    let mut found = None;
                    for item in &d.items {
                        let Some(k) = &item.key else {
                            return Err(refused("a `**` in the mapping"));
                        };
                        let py::Expr::StringLiteral(lit) = k else {
                            return Err(refused("a mapping with a key that is not a string"));
                        };
                        if lit.value.to_str() == key {
                            found = Some(&item.value);
                        }
                    }
                    let Some(e) = found else {
                        return Err(refused("a key the mapping does not have"));
                    };
                    if d.items.iter().any(|i| has_call(&i.value)) {
                        return Err(refused("a mapping whose values call"));
                    }
                    e.clone()
                }
                py::Expr::Name(n) => py::Expr::Subscript(py::ExprSubscript {
                    node_index: Default::default(),
                    range: n.range,
                    value: Box::new(values.clone()),
                    slice: Box::new(py::Expr::StringLiteral(py::ExprStringLiteral {
                        node_index: Default::default(),
                        range: n.range,
                        value: py::StringLiteralValue::single(py::StringLiteral {
                            range: n.range,
                            node_index: Default::default(),
                            value: key.clone().into_boxed_str(),
                            flags: py::StringLiteralFlags::empty(),
                        }),
                    })),
                    ctx: py::ExprContext::Load,
                }),
                _ => return Err(refused("a mapping that is not a dict display or a name")),
            };
            let value = self.expr(&read)?;
            pieces.push(self.percent_field(value, &field, span)?);
        }
        let node =
            concat_pieces(pieces, Ty::Str, span).unwrap_or_else(|| crate::lower::str_lit("", span));
        Ok(Val { node, ty: Ty::Str })
    }

    pub(crate) fn percent_field(
        &mut self,
        value: Val,
        field: &PercentField,
        span: Span,
    ) -> Result<Node> {
        self.percent_field_with(value, field, &Nested::default(), span)
    }

    fn percent_field_with(
        &mut self,
        value: Val,
        field: &PercentField,
        nested: &Nested,
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
        Ok(self.format_with(value, &spec, nested, span))
    }

    /// `format(value, spec)`.
    pub(crate) fn format(&mut self, value: Val, spec: &Spec, span: Span) -> Node {
        self.format_with(value, spec, &Nested::default(), span)
    }

    /// `format(value, spec)`, the width and precision of `nested` taking
    /// the place of the spec's own.
    pub(crate) fn format_with(
        &mut self,
        value: Val,
        spec: &Spec,
        nested: &Nested,
        span: Span,
    ) -> Node {
        let lit_str = |s: &str| crate::lower::str_lit(s, span);
        let lit_int = |v: i64| crate::lower::int_lit(v, span);
        let lit_bool = |b: bool| {
            crate::lower::node(
                TypedExpression::Literal(TypedLiteral::Bool(b)),
                Ty::Bool,
                span,
            )
        };
        let width = || nested.width.clone().unwrap_or_else(|| lit_int(spec.width));
        let precision = || {
            nested
                .precision
                .clone()
                .unwrap_or_else(|| lit_int(spec.precision))
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
                    width(),
                    lit_int(spec.grouping),
                    precision(),
                    lit_int(spec.ty),
                ]);
                crate::lower::call("zb_fmt_float", args, Ty::Str, span)
            }
            Ty::Bool if spec.ty == 0 => {
                let text = self.str_of(value);
                self.format_with(
                    Val {
                        node: text,
                        ty: Ty::Str,
                    },
                    spec,
                    nested,
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
                    width(),
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
                    width(),
                    lit_int(spec.grouping),
                    precision(),
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
                    width(),
                    lit_int(spec.grouping),
                    precision(),
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
                        width(),
                        precision(),
                    ],
                    Ty::Str,
                    span,
                )
            }
        }
    }
}

/// `base` read through the accessors of a replacement field: `.name`
/// and `[index]`, a number or a string key.
fn field_accessors(base: &py::Expr, mut rest: &str) -> Option<py::Expr> {
    use ruff_text_size::Ranged;
    let range = base.range();
    let mut read = base.clone();
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('.') {
            let end = after.find(['.', '[']).unwrap_or(after.len());
            let name = &after[..end];
            if name.is_empty() {
                return None;
            }
            read = py::Expr::Attribute(py::ExprAttribute {
                node_index: Default::default(),
                range,
                value: Box::new(read),
                attr: py::Identifier::new(name, range),
                ctx: py::ExprContext::Load,
            });
            rest = &after[end..];
        } else {
            let after = rest.strip_prefix('[')?;
            let end = after.find(']')?;
            let key = &after[..end];
            let slice = match key.parse::<u64>() {
                Ok(n) => py::Expr::NumberLiteral(py::ExprNumberLiteral {
                    node_index: Default::default(),
                    range,
                    value: py::Number::Int(py::Int::from(n)),
                }),
                Err(_) if !key.is_empty() => py::Expr::StringLiteral(py::ExprStringLiteral {
                    node_index: Default::default(),
                    range,
                    value: py::StringLiteralValue::single(py::StringLiteral {
                        range,
                        node_index: Default::default(),
                        value: key.to_string().into_boxed_str(),
                        flags: py::StringLiteralFlags::empty(),
                    }),
                }),
                Err(_) => return None,
            };
            read = py::Expr::Subscript(py::ExprSubscript {
                node_index: Default::default(),
                range,
                value: Box::new(read),
                slice: Box::new(slice),
                ctx: py::ExprContext::Load,
            });
            rest = &after[end + 1..];
        }
    }
    Some(read)
}

/// Whether evaluating `e` may call into the program.
fn has_call(e: &py::Expr) -> bool {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    struct Calls(bool);
    impl<'a> Visitor<'a> for Calls {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if matches!(
                e,
                py::Expr::Call(_) | py::Expr::Await(_) | py::Expr::Yield(_) | py::Expr::Named(_)
            ) {
                self.0 = true;
            }
            walk_expr(self, e);
        }
    }
    let mut v = Calls(false);
    v.visit_expr(e);
    v.0
}

/// The names a template's fields read, first segment only, in order and
/// once each; the empty and numbered fields are left out.
pub(crate) fn template_names(template: &str) -> Vec<String> {
    let chars: Vec<char> = template.chars().collect();
    let mut names: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match (chars[i], chars.get(i + 1)) {
            ('{', Some('{')) | ('}', Some('}')) => i += 2,
            ('{', _) => {
                let mut depth = 1;
                let mut end = i + 1;
                while end < chars.len() && depth > 0 {
                    match chars[end] {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                    end += 1;
                }
                let field: String = chars[i + 1..end.saturating_sub(1).max(i + 1)]
                    .iter()
                    .collect();
                let name = field
                    .split(['!', ':', '.', '['])
                    .next()
                    .unwrap_or_default()
                    .to_string();
                if !name.is_empty() && name.parse::<usize>().is_err() && !names.contains(&name) {
                    names.push(name);
                }
                i = end;
            }
            _ => i += 1,
        }
    }
    names
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
