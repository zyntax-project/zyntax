//! Byte strings and files. A byte string is carried as a string is,
//! `[length][bytes]`, and only ever read by functions that take it as
//! bytes; a file is the record the library keeps for it, written out
//! whole when it is closed.

use crate::format::{Percent, parse_percent};
use crate::lower::{
    Lowerer, Node, Stmt, Val, binary, call, int_lit, node, str_lit, unsupported, var,
};
use crate::types::{Elem, Ty};
use crate::{Error, Result, intern};
use ruff_python_ast as py;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedExpression, TypedLet, TypedLiteral, TypedStatement};
use zyntax_typed_ast::{BinaryOp, Mutability, Type, TypedNode};

impl Lowerer<'_> {
    /// A bytes literal. UTF-8 bytes are a string literal, laid out in
    /// the program; anything else is built from its values.
    pub(crate) fn bytes_lit(&mut self, bytes: &[u8], span: Span) -> Val {
        let node = match std::str::from_utf8(bytes) {
            Ok(text) => node(
                TypedExpression::Literal(TypedLiteral::String(intern(text))),
                Ty::Bytes,
                span,
            ),
            Err(_) => {
                let values = bytes
                    .iter()
                    .map(|b| Val {
                        node: int_lit(*b as i64, span),
                        ty: Ty::Int,
                    })
                    .collect();
                let ints = self.list_of(values, Elem::Int, span);
                call("zb_bytes_from_ints", vec![ints], Ty::Bytes, span)
            }
        };
        Val {
            node,
            ty: Ty::Bytes,
        }
    }

    /// `bytes(x)`: zero bytes counted by an int, the values of a list
    /// of ints, a string's UTF-8, or a copy of bytes.
    pub(crate) fn bytes_call(
        &mut self,
        args: &[py::Expr],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        let node = match args {
            [] => self.bytes_lit(&[], span).node,
            [source] => {
                let v = self.expr(source)?;
                match v.ty {
                    Ty::Bytes => v.node,
                    Ty::Int | Ty::Bool => {
                        let n = self.coerce(v, Ty::Int);
                        call("zb_bytes_zeros", vec![n], Ty::Bytes, span)
                    }
                    Ty::List(_) | Ty::Tuple(_) | Ty::Object => {
                        let ints = self.coerce(v, Ty::List(Elem::Int));
                        call("zb_bytes_from_ints", vec![ints], Ty::Bytes, span)
                    }
                    Ty::Str => return unsupported("bytes() of a string without an encoding", c),
                    _ => return unsupported("bytes() of this value", c),
                }
            }
            [source, encoding] if self.is_utf8_name(encoding) => {
                let v = self.expr(source)?;
                match v.ty {
                    Ty::Str => v.node,
                    _ => return unsupported("bytes() with an encoding of a non-string", c),
                }
            }
            _ => return unsupported("bytes() with these arguments", c),
        };
        Ok(Val {
            node,
            ty: Ty::Bytes,
        })
    }

    /// Whether `e` names UTF-8, the one encoding strings have.
    pub(crate) fn is_utf8_name(&self, e: &py::Expr) -> bool {
        matches!(e, py::Expr::StringLiteral(s)
            if matches!(s.value.to_str().to_ascii_lowercase().as_str(), "utf-8" | "utf8" | "ascii"))
    }

    /// `open(path, mode)`: the file's record, with the mode a literal so
    /// what it reads and writes is known.
    pub(crate) fn open_call(&mut self, c: &py::ExprCall, span: Span) -> Result<Val> {
        let args = &c.arguments.args;
        let Some(mode) = crate::types::open_mode(c) else {
            return unsupported("open() with a mode that is not a string literal", c);
        };
        let Some(path) = args.first() else {
            return unsupported("open() without a path", c);
        };
        let path = self.expr_as(path, Ty::Str)?;
        let mode_text = match args
            .get(1)
            .or_else(|| c.arguments.find_keyword("mode").map(|k| &k.value))
        {
            Some(py::Expr::StringLiteral(s)) => s.value.to_str().to_string(),
            _ => "r".to_string(),
        };
        Ok(Val {
            node: call(
                "zb_file_open",
                vec![path, str_lit(&mode_text, span)],
                Ty::File(mode),
                span,
            ),
            ty: Ty::File(mode),
        })
    }

    /// `with open(...) as f: body`: the file bound, the body, the file
    /// closed. Only files are context managers here.
    pub(crate) fn with_stmt(
        &mut self,
        w: &py::StmtWith,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<()> {
        if w.is_async {
            return unsupported("async with", w);
        }
        let mut files = Vec::with_capacity(w.items.len());
        for item in &w.items {
            let v = self.expr(&item.context_expr)?;
            let Ty::File(_) = v.ty else {
                return unsupported("`with` on something other than a file", &item.context_expr);
            };
            let ty = v.ty;
            // The open and its check run before the binding.
            out.append(&mut self.hoisted);
            let held = match item.optional_vars.as_deref() {
                Some(target @ py::Expr::Name(n)) => {
                    self.bind_after(target, v, span, out)?;
                    var(self.local_symbol(n.id.as_str()), ty, span)
                }
                Some(target) => {
                    return unsupported("`with ... as` a target that is not a name", target);
                }
                None => {
                    let name = self.temp();
                    out.push(TypedNode::new(
                        TypedStatement::Let(TypedLet {
                            name,
                            ty: crate::lower::ir(ty),
                            mutability: Mutability::Immutable,
                            initializer: Some(Box::new(v.node)),
                            span,
                        }),
                        Type::Unknown,
                        span,
                    ));
                    var(name, ty, span)
                }
            };
            files.push(held);
        }
        for s in &w.body {
            self.stmt(s, out)?;
        }
        for f in files.into_iter().rev() {
            out.push(TypedNode::new(
                TypedStatement::Expression(Box::new(call(
                    "zb_file_close",
                    vec![f],
                    Ty::None,
                    span,
                ))),
                Type::Unknown,
                span,
            ));
        }
        Ok(())
    }

    /// `b"..." % values` with a literal format: the conversions are
    /// done as a string's are, since every number formats as ASCII, and
    /// the pieces concatenated as bytes.
    pub(crate) fn percent_format_bytes(
        &mut self,
        template: &[u8],
        values: &py::Expr,
        span: Span,
    ) -> Result<Val> {
        // Each byte as one char keeps the parser's positions the bytes'.
        let text: String = template.iter().map(|b| *b as char).collect();
        let conversions = parse_percent(&text)
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
                Percent::Text(t) => {
                    let bytes: Vec<u8> = t.chars().map(|c| c as u8).collect();
                    pieces.push(self.bytes_lit(&bytes, span).node);
                }
                Percent::Field(field) => {
                    let value = self.expr(next.next().expect("counted"))?;
                    let node = match (field.conversion, value.ty) {
                        // `%c` of an int is that byte; of bytes, them.
                        ('c', Ty::Bytes) => value.node,
                        ('c', _) => {
                            let code = self.coerce(value, Ty::Int);
                            call("zb_bytes_byte", vec![code], Ty::Bytes, span)
                        }
                        // `%s` and `%b` take bytes only.
                        ('s' | 'b', Ty::Bytes) => value.node,
                        ('s' | 'b', _) => {
                            return Err(Error::unsupported_span(
                                "`%s` in a bytes format of a value that is not bytes".to_string(),
                                span,
                            ));
                        }
                        _ => self.percent_field(value, &field, span)?,
                    };
                    pieces.push(node);
                }
            }
        }
        let mut it = pieces.into_iter();
        let first = it.next().unwrap_or_else(|| self.bytes_lit(&[], span).node);
        let node = it.fold(first, |acc, piece| {
            binary(BinaryOp::Add, acc, piece, Ty::Bytes, span)
        });
        Ok(Val {
            node,
            ty: Ty::Bytes,
        })
    }
}
