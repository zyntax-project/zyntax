//! Lua's AST rewritten as typed AST.
//!
//! Every expression is lowered with the static type inference gave it:
//! arithmetic on two integers is an `i64` add, on two dynamic values a
//! call into the library's `zl_arith`. Crossings are explicit: a typed
//! value is boxed where a dynamic one is needed, and a dynamic value
//! known to hold a type is read back without a check, because
//! inference only settles a type it saw every assignment of.
//!
//! Functions become typed entries, called directly where the callee is
//! known; one used as a value gets a record of the shape every function
//! value has, whose code adapts the record's dynamic arguments to the
//! typed entry. Closures capture through cells or copies in the record.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use full_moon::ast::{self, BinOp, Block, Expression, Prefix, Stmt, Suffix, UnOp, Var};
use full_moon::tokenizer::TokenReference;

use zyntax_builtins::functions::{VARIADIC_ARITY, arity_word};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    BinaryOp, ParamOwnership, TypedBinary, TypedBlock, TypedCall, TypedCast, TypedDeclaration,
    TypedExpression, TypedFunction, TypedIf, TypedIfExpr, TypedIndex, TypedLet, TypedLiteral,
    TypedParameter, TypedStatement, TypedUnary, TypedVariable, TypedWhile, UnaryOp,
};
use zyntax_typed_ast::{
    InternedString, Mutability, PrimitiveType, Type, TypedNode, TypedProgram, Visibility,
};

use crate::library::stdlib::{Builtin, Param, Ret};
use crate::library::values::*;
use crate::library::{self, Types};
use crate::scope::{Binding, CHUNK, FuncId, Scopes, VarId};
use crate::types::{self, Inferred, Returns, Ty, Typer, ident};
use crate::{ENTRY, Error, Library, Result, intern, prim};

type Node = TypedNode<TypedExpression>;
type St = TypedNode<TypedStatement>;

/// A lowered expression and the static type it has.
#[derive(Clone)]
struct Val {
    node: Node,
    ty: Ty,
}

/// What an expression in last position produces: a fixed number of
/// values, or one dynamic value that may hold several, or nothing: a
/// call that runs for its effects.
enum Multi {
    Fixed(Vec<Val>),
    Dynamic(Node),
    None(Node),
}

/// Where a variable lives in the function being lowered.
#[derive(Clone)]
enum Storage {
    /// A local of the function, or a parameter.
    Local(InternedString, Ty),
    /// A one-element list shared with closures.
    Cell(InternedString, Ty),
    /// A module-level variable.
    Module(InternedString, Ty),
}

/// Where the record of a captured variable sits: `env[RECORD_CELLS_AT
/// + i]` for the `i`th capture of the function.
const RECORD_CELLS_AT: usize = 2;

/// The whole program as the lowerers share it.
struct Module<'a> {
    scopes: &'a Scopes,
    inferred: &'a Inferred,
    types: Types,
    /// The chunk's name, as error positions spell it.
    chunk: &'a str,
    /// What this chunk's symbols carry after `lua$`: nothing for the
    /// main chunk, `m$<name>$` for a required file, so two files'
    /// functions and locals never share a name.
    tag: String,
    /// This chunk's number in the bits above `LINE_BITS` of a stored
    /// line: 0 for the main chunk.
    chunk_index: i64,
    /// Byte offsets where each line starts, for positions.
    line_starts: Vec<usize>,
    /// The library functions that can raise.
    fallible: HashSet<&'static str>,
    /// The program's functions that can raise, once a first lowering
    /// has found out; every one, before.
    raising: Option<HashSet<FuncId>>,
    /// Functions lowered so far, in the order they were reached.
    functions: RefCell<Vec<TypedFunction>>,
    /// Module-level variables: globals, and chunk locals every function
    /// reaches. Name and type.
    module_vars: RefCell<Vec<(InternedString, Ty)>>,
    /// What each function's lowering found about its raising.
    facts: RefCell<HashMap<FuncId, RaiseFact>>,
}

/// Whether a function raises itself, and which functions it calls
/// that may raise into it.
#[derive(Clone, Default)]
struct RaiseFact {
    own: bool,
    callees: HashSet<FuncId>,
}

/// The functions that may raise: those that check for an error
/// themselves, and those calling one of them, and so on.
fn raising_functions(facts: &HashMap<FuncId, RaiseFact>) -> HashSet<FuncId> {
    let mut raising: HashSet<FuncId> = facts
        .iter()
        .filter(|(_, fact)| fact.own)
        .map(|(f, _)| *f)
        .collect();
    loop {
        let before = raising.len();
        for (f, fact) in facts {
            if fact.callees.iter().any(|c| raising.contains(c)) {
                raising.insert(*f);
            }
        }
        if raising.len() == before {
            return raising;
        }
    }
}

impl<'a> Module<'a> {
    fn typer(&self) -> Typer<'_> {
        Typer {
            scopes: self.scopes,
            known: self.inferred,
        }
    }

    /// The line a span starts on, counted from one.
    fn line_of(&self, span: Span) -> i64 {
        let line = self
            .line_starts
            .partition_point(|&start| start <= span.start) as i64;
        (self.chunk_index << library::LINE_BITS) | line
    }

    /// Whether a program function may raise, as far as is known.
    fn raises(&self, f: FuncId) -> bool {
        match &self.raising {
            Some(set) => set.contains(&f),
            None => true,
        }
    }

    /// The IR type of a static type.
    fn ir(&self, ty: Ty) -> Type {
        match ty {
            Ty::Bool => prim(PrimitiveType::Bool),
            Ty::Int => prim(PrimitiveType::I64),
            Ty::Float => prim(PrimitiveType::F64),
            Ty::Str => prim(PrimitiveType::String),
            Ty::Table => self.types.table(),
            Ty::Nil | Ty::Any | Ty::Unknown => Type::Any,
        }
    }

    fn anys(&self) -> Type {
        self.types.anys()
    }

    /// The symbol of a global's module variable.
    fn global_symbol(name: &str) -> InternedString {
        intern(&format!("lua$g${name}"))
    }

    /// The symbol of a captured chunk local's module variable.
    fn module_local_symbol(&self, v: VarId) -> InternedString {
        intern(&format!(
            "lua$l${}{}${}",
            self.tag,
            self.scopes.var(v).name,
            v.0
        ))
    }

    fn declare_module_var(&self, name: InternedString, ty: Ty) {
        let mut vars = self.module_vars.borrow_mut();
        if !vars.iter().any(|(n, _)| *n == name) {
            vars.push((name, ty));
        }
    }

    /// The typed entry's name for a function.
    fn entry_name(&self, f: FuncId) -> String {
        let info = self.scopes.func(f);
        if f == CHUNK {
            return if self.tag.is_empty() {
                ENTRY.to_string()
            } else {
                format!("lua${}chunk", self.tag)
            };
        }
        if info.top_level
            && let Some((name, _)) = self
                .scopes
                .global_functions
                .iter()
                .find(|(_, id)| **id == f)
        {
            return format!("lua${}{name}", self.tag);
        }
        let name = if info.name.is_empty() {
            "anon".to_string()
        } else {
            info.name.replace(['.', ':'], "$")
        };
        format!("lua${}{name}${}", self.tag, f.0)
    }

    /// The record code's name for a function.
    fn code_name(&self, f: FuncId) -> String {
        format!("{}$fn", self.entry_name(f))
    }

    fn sig(&self, f: FuncId) -> types::Sig {
        self.inferred.sig(f).cloned().unwrap_or_else(|| types::Sig {
            params: vec![Ty::Nil; self.scopes.func(f).params.len()],
            returns: Returns::Fixed(Vec::new()),
        })
    }

    /// The IR return type of a function.
    fn return_ir(&self, returns: &Returns) -> Type {
        match returns {
            Returns::Fixed(v) if v.is_empty() => prim(PrimitiveType::Unit),
            Returns::Fixed(v) if v.len() == 1 => self.ir(v[0]),
            Returns::Fixed(_) => self.anys(),
            Returns::Dynamic => Type::Any,
        }
    }
}

/// A function being lowered.
struct Lowerer<'m, 'a> {
    m: &'m Module<'a>,
    func: FuncId,
    storage: HashMap<VarId, Storage>,
    /// The list of extra arguments, for a variadic function.
    varargs: Option<InternedString>,
    returns: Returns,
    temps: usize,
    /// Names already bound by a `let` in this function, so a second
    /// declaration of a Lua local of the same name is a fresh symbol.
    bound: std::collections::HashSet<InternedString>,
    /// The last global read as nil, for the message of a call to it.
    nil_global: Option<String>,
    /// Whether this function checks for an error anywhere, so it may
    /// leave with one pending.
    raised: bool,
    /// The program functions this one calls directly and that may
    /// raise into it.
    raise_callees: HashSet<FuncId>,
    /// Whether the statement being lowered checks for an error, so its
    /// line is recorded ahead of it.
    line_needed: bool,
    /// Whether the body reads the line it was entered at, for
    /// `error(v, 2)`.
    entry_line: bool,
    /// The `<close>` variables in scope, each with the depth of the
    /// block declaring it, innermost last. Leaving a block closes its
    /// variables in reverse order.
    tbc: Vec<(usize, VarId)>,
    /// How many blocks are open, the function's body counting as one.
    depth: usize,
    /// The depth of each enclosing loop's body.
    loop_depths: Vec<usize>,
}

fn unsupported<T>(what: impl Into<String>, span: Span) -> Result<T> {
    Err(Error::unsupported(what, span))
}

fn span_of<N: full_moon::node::Node>(node: &N) -> Span {
    match node.range() {
        Some((a, b)) => Span::new(a.bytes(), b.bytes()),
        None => Span::new(0, 0),
    }
}

fn node(x: TypedExpression, ty: Type, span: Span) -> Node {
    TypedNode::new(x, ty, span)
}

fn stmt(s: TypedStatement, span: Span) -> St {
    TypedNode::new(s, Type::Unknown, span)
}

fn expr_stmt(e: Node) -> St {
    let span = e.span;
    stmt(TypedStatement::Expression(Box::new(e)), span)
}

fn var(name: InternedString, ty: Type, span: Span) -> Node {
    node(TypedExpression::Variable(name), ty, span)
}

fn int_lit(v: i64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        prim(PrimitiveType::I64),
        span,
    )
}

fn int32_lit(v: i32, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        prim(PrimitiveType::I32),
        span,
    )
}

fn float_lit(v: f64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Float(v)),
        prim(PrimitiveType::F64),
        span,
    )
}

fn bool_lit(v: bool, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Bool(v)),
        prim(PrimitiveType::Bool),
        span,
    )
}

fn str_lit(s: &str, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::String(intern(s))),
        prim(PrimitiveType::String),
        span,
    )
}

fn null(ty: Type, span: Span) -> Node {
    node(TypedExpression::Literal(TypedLiteral::Null), ty, span)
}

fn nil(span: Span) -> Node {
    null(Type::Any, span)
}

fn call(name: &str, args: Vec<Node>, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(var(intern(name), Type::Unknown, span)),
            positional_args: args,
            named_args: Vec::new(),
            type_args: Vec::new(),
        }),
        ty,
        span,
    )
}

fn binary(op: BinaryOp, left: Node, right: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }),
        ty,
        span,
    )
}

fn unary(op: UnaryOp, operand: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Unary(TypedUnary {
            op,
            operand: Box::new(operand),
        }),
        ty,
        span,
    )
}

fn cast(value: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ty.clone(),
        }),
        ty,
        span,
    )
}

fn index(xs: Node, i: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(xs),
            index: Box::new(i),
        }),
        ty,
        span,
    )
}

fn assign(target: Node, value: Node, span: Span) -> St {
    let ty = target.ty.clone();
    expr_stmt(binary(BinaryOp::Assign, target, value, ty, span))
}

fn let_(name: InternedString, ty: Type, value: Node, span: Span) -> St {
    stmt(
        TypedStatement::Let(TypedLet {
            name,
            ty,
            mutability: Mutability::Mutable,
            initializer: Some(Box::new(value)),
            span,
        }),
        span,
    )
}

fn if_(cond: Node, then: Vec<St>, els: Option<Vec<St>>, span: Span) -> St {
    stmt(
        TypedStatement::If(TypedIf {
            condition: Box::new(cond),
            then_block: TypedBlock {
                statements: then,
                span,
            },
            else_block: els.map(|statements| TypedBlock { statements, span }),
            span,
        }),
        span,
    )
}

fn while_(cond: Node, body: Vec<St>, span: Span) -> St {
    stmt(
        TypedStatement::While(TypedWhile {
            condition: Box::new(cond),
            body: TypedBlock {
                statements: body,
                span,
            },
            span,
        }),
        span,
    )
}

fn ret(value: Option<Node>, span: Span) -> St {
    stmt(TypedStatement::Return(value.map(Box::new)), span)
}

/// Statements followed by a value, as one expression.
fn block_value(statements: Vec<St>, value: Node, span: Span) -> Node {
    if statements.is_empty() {
        return value;
    }
    let ty = value.ty.clone();
    let mut statements = statements;
    statements.push(expr_stmt(value));
    node(
        TypedExpression::Block(TypedBlock { statements, span }),
        ty,
        span,
    )
}

/// The length of a list.
fn list_len(xs: Node, span: Span) -> Node {
    node(
        TypedExpression::MethodCall(zyntax_typed_ast::typed_ast::TypedMethodCall {
            receiver: Box::new(xs),
            method: intern("len"),
            type_args: Vec::new(),
            positional_args: Vec::new(),
            named_args: Vec::new(),
        }),
        prim(PrimitiveType::I64),
        span,
    )
}

/// The address of function `name`.
fn code_of(name: &str, span: Span) -> Node {
    var(intern(name), prim(PrimitiveType::USize), span)
}

fn parameter(name: InternedString, ty: Type, span: Span) -> TypedParameter {
    let mut p = TypedParameter::regular(name, ty.clone(), Mutability::Mutable, span);
    // A dynamic parameter is dynamic by the language's rules, not by
    // omission, so the lowering does not warn about it.
    if ty == Type::Any {
        p.attributes
            .push(zyntax_typed_ast::typed_ast::ParameterAttribute {
                name: intern("dynamic"),
                args: Vec::new(),
                span,
            });
    }
    // Anything on the heap that a function receives may be kept by
    // it: a table stored into another, a value captured.
    if !matches!(
        ty,
        Type::Primitive(PrimitiveType::I64)
            | Type::Primitive(PrimitiveType::F64)
            | Type::Primitive(PrimitiveType::Bool)
            | Type::Primitive(PrimitiveType::Unit)
    ) {
        p.ownership = ParamOwnership::Shared;
    }
    p
}

fn typed_function(
    name: &str,
    params: Vec<TypedParameter>,
    return_type: Type,
    statements: Vec<St>,
    span: Span,
) -> TypedFunction {
    TypedFunction {
        name: intern(name),
        annotations: Vec::new(),
        effects: Vec::new(),
        with_handlers: Vec::new(),
        type_params: Vec::new(),
        params,
        return_type,
        body: Some(TypedBlock { statements, span }),
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
        link_name: None,
        module: None,
    }
}

/// The bytes of a string literal token, escapes decoded, and a byte the
/// source could not spell as text restored from its stand-in.
fn string_bytes(token: &TokenReference) -> std::result::Result<Vec<u8>, String> {
    let bytes = literal_bytes(token)?;
    Ok(restore_bytes(bytes))
}

/// Private-use characters standing for bytes (see `source_text`) back
/// to the bytes.
fn restore_bytes(bytes: Vec<u8>) -> Vec<u8> {
    // U+F700..U+F7FF are EF 9C 80 .. EF 9F BF.
    if !bytes
        .windows(2)
        .any(|w| w[0] == 0xEF && (0x9C..=0x9F).contains(&w[1]))
    {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 < bytes.len()
            && bytes[i] == 0xEF
            && (0x9C..=0x9F).contains(&bytes[i + 1])
            && (0x80..=0xBF).contains(&bytes[i + 2])
        {
            let code = ((bytes[i] as u32 & 0x0F) << 12)
                | ((bytes[i + 1] as u32 & 0x3F) << 6)
                | (bytes[i + 2] as u32 & 0x3F);
            if (crate::ESCAPED_BYTES..crate::ESCAPED_BYTES + 0x100).contains(&code) {
                out.push((code - crate::ESCAPED_BYTES) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn literal_bytes(token: &TokenReference) -> std::result::Result<Vec<u8>, String> {
    use full_moon::tokenizer::{StringLiteralQuoteType, TokenType};
    let TokenType::StringLiteral {
        literal,
        quote_type,
        ..
    } = token.token().token_type()
    else {
        return Ok(token.token().to_string().into_bytes());
    };
    match quote_type {
        StringLiteralQuoteType::Brackets => {
            // A long string starts after its first line break.
            let s = literal.as_str();
            let s = s
                .strip_prefix("\r\n")
                .or_else(|| s.strip_prefix('\n'))
                .or_else(|| s.strip_prefix('\r'))
                .unwrap_or(s);
            Ok(s.as_bytes().to_vec())
        }
        _ => decode_escapes(literal.as_str()),
    }
}

/// A string literal as a node: text when it is UTF-8, otherwise built
/// at run time from its bytes spelled in hex, since a literal in the
/// typed AST is text.
fn string_literal(bytes: Vec<u8>, span: Span) -> Node {
    match String::from_utf8(bytes) {
        Ok(text) => str_lit(&text, span),
        Err(e) => {
            let hex: String = e.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
            call(
                "zl_bytes",
                vec![str_lit(&hex, span)],
                prim(PrimitiveType::String),
                span,
            )
        }
    }
}

/// Lua's escapes: `\n` and the others, `\ddd`, `\xXX`, `\u{XXX}`, `\z`,
/// and a backslash before a line break.
fn decode_escapes(s: &str) -> std::result::Result<Vec<u8>, String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b != b'\\' {
            out.push(b);
            i += 1;
            continue;
        }
        i += 1;
        let Some(&c) = bytes.get(i) else {
            return Err("unfinished string".to_string());
        };
        i += 1;
        match c {
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'a' => out.push(7),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'v' => out.push(11),
            b'\\' => out.push(b'\\'),
            b'"' => out.push(b'"'),
            b'\'' => out.push(b'\''),
            b'\n' => {
                out.push(b'\n');
                if bytes.get(i) == Some(&b'\r') {
                    i += 1;
                }
            }
            b'\r' => {
                out.push(b'\n');
                if bytes.get(i) == Some(&b'\n') {
                    i += 1;
                }
            }
            b'x' => {
                let hex = s.get(i..i + 2).ok_or("hexadecimal digit expected")?;
                let v = u8::from_str_radix(hex, 16).map_err(|_| "hexadecimal digit expected")?;
                out.push(v);
                i += 2;
            }
            b'z' => {
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
            }
            b'u' => {
                if bytes.get(i) != Some(&b'{') {
                    return Err("missing '{' in \\u{xxxx}".to_string());
                }
                let end = s[i..].find('}').ok_or("missing '}' in \\u{xxxx}")? + i;
                let digits = &s[i + 1..end];
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("hexadecimal digit expected".to_string());
                }
                // Lua's own UTF-8 reaches 0x7FFFFFFF in six bytes.
                let code = u64::from_str_radix(digits, 16).unwrap_or(u64::MAX);
                if code > 0x7FFF_FFFF {
                    return Err("UTF-8 value too large".to_string());
                }
                crate::host::utf8_encode(code as u32, &mut out);
                i = end + 1;
            }
            d if d.is_ascii_digit() => {
                let mut v: u32 = (d - b'0') as u32;
                let mut n = 1;
                while n < 3 && i < bytes.len() && bytes[i].is_ascii_digit() {
                    v = v * 10 + (bytes[i] - b'0') as u32;
                    i += 1;
                    n += 1;
                }
                if v > 255 {
                    return Err("decimal escape too large".to_string());
                }
                out.push(v as u8);
            }
            other => return Err(format!("invalid escape sequence '\\{}'", other as char)),
        }
    }
    Ok(out)
}

impl<'m, 'a> Lowerer<'m, 'a> {
    fn new(m: &'m Module<'a>, func: FuncId) -> Self {
        let returns = m.sig(func).returns;
        Lowerer {
            m,
            func,
            storage: HashMap::new(),
            varargs: None,
            returns,
            temps: 0,
            bound: std::collections::HashSet::new(),
            nil_global: None,
            raised: false,
            raise_callees: HashSet::new(),
            line_needed: false,
            entry_line: false,
            tbc: Vec::new(),
            depth: 0,
            loop_depths: Vec::new(),
        }
    }

    fn typer(&self) -> Typer<'_> {
        self.m.typer()
    }

    fn ir(&self, ty: Ty) -> Type {
        self.m.ir(ty)
    }

    fn scopes(&self) -> &'a Scopes {
        self.m.scopes
    }

    /// A local no Lua program can spell.
    fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("$t{}", self.temps))
    }

    /// The symbol a Lua local is lowered to: its name, or its name and
    /// id once another local of that name has been declared here.
    fn local_symbol(&mut self, v: VarId) -> InternedString {
        let name = &self.scopes().var(v).name;
        let plain = intern(name);
        let symbol = if self.bound.contains(&plain) {
            intern(&format!("{name}${}", v.0))
        } else {
            plain
        };
        self.bound.insert(symbol);
        symbol
    }

    // ─── values ─────────────────────────────────────────────────

    fn nil_val(&self, span: Span) -> Val {
        Val {
            node: nil(span),
            ty: Ty::Nil,
        }
    }

    /// `v` as a value of `target`.
    fn coerce(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        let target = if target == Ty::Unknown {
            Ty::Nil
        } else {
            target
        };
        match (v.ty, target) {
            (a, b) if a == b => v.node,
            (Ty::Unknown, _) => v.node,
            (Ty::Nil, Ty::Any) => v.node,
            (Ty::Int, Ty::Float) => cast(v.node, prim(PrimitiveType::F64), span),
            (Ty::Bool, Ty::Any) => call("zb_box_bool", vec![v.node], Type::Any, span),
            (Ty::Int, Ty::Any) => call("zb_box_i64", vec![v.node], Type::Any, span),
            (Ty::Float, Ty::Any) => call("zb_box_f64", vec![v.node], Type::Any, span),
            (Ty::Str, Ty::Any) => call("zb_box_str", vec![v.node], Type::Any, span),
            (Ty::Table, Ty::Any) => self.box_table(v.node),
            // A dynamic value known to hold the type: read straight out.
            (Ty::Any, Ty::Bool) => binary(
                BinaryOp::Ne,
                call(
                    "zb_box_payload_bool",
                    vec![v.node],
                    prim(PrimitiveType::I32),
                    span,
                ),
                int32_lit(0, span),
                prim(PrimitiveType::Bool),
                span,
            ),
            (Ty::Any, Ty::Int) => call(
                "zb_box_payload_i64",
                vec![v.node],
                prim(PrimitiveType::I64),
                span,
            ),
            (Ty::Any, Ty::Float) => call(
                "zb_box_payload_f64",
                vec![v.node],
                prim(PrimitiveType::F64),
                span,
            ),
            (Ty::Any, Ty::Str) => call(
                "zb_box_get_str",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            (Ty::Any, Ty::Table) => self.unbox_table(v.node),
            (Ty::Any, Ty::Nil) => v.node,
            // A nil where a typed value was expected: the value's zero,
            // after the expression ran. Inference joins nil into the
            // type, so this is only reached for a slot never read.
            (Ty::Nil, t) => {
                let zero = match t {
                    Ty::Int => int_lit(0, span),
                    Ty::Float => float_lit(0.0, span),
                    Ty::Bool => bool_lit(false, span),
                    Ty::Str => str_lit("", span),
                    Ty::Table => null(self.ir(Ty::Table), span),
                    _ => nil(span),
                };
                block_value(vec![expr_stmt(v.node)], zero, span)
            }
            (Ty::Float, Ty::Int) => cast(v.node, prim(PrimitiveType::I64), span),
            (a, b) => {
                // Anything else goes through the box.
                let boxed = self.coerce(v, Ty::Any);
                if b == Ty::Any {
                    return boxed;
                }
                let _ = a;
                self.coerce(
                    Val {
                        node: boxed,
                        ty: Ty::Any,
                    },
                    b,
                )
            }
        }
    }

    fn boxed(&mut self, v: Val) -> Node {
        self.coerce(v, Ty::Any)
    }

    fn box_table(&mut self, t: Node) -> Node {
        let span = t.span;
        call(
            "zb_box_instance",
            vec![
                cast(t, prim(PrimitiveType::I64), span),
                int32_lit(library::table_tag() as i32, span),
            ],
            Type::Any,
            span,
        )
    }

    fn unbox_table(&mut self, x: Node) -> Node {
        let span = x.span;
        cast(
            call(
                "zb_unbox_instance_raw",
                vec![x],
                prim(PrimitiveType::I64),
                span,
            ),
            self.ir(Ty::Table),
            span,
        )
    }

    /// Whether the node is a plain read, safe to repeat.
    fn is_simple(node: &Node) -> bool {
        matches!(
            node.node,
            TypedExpression::Variable(_) | TypedExpression::Literal(_)
        )
    }

    /// The value held in a temporary, so it can be read more than
    /// once: the statement binding it, and the read.
    fn hold(&mut self, v: Val, pre: &mut Vec<St>) -> Val {
        if Self::is_simple(&v.node) {
            return v;
        }
        let span = v.node.span;
        let name = self.temp();
        let ty = self.ir(v.ty);
        pre.push(let_(name, ty.clone(), v.node, span));
        Val {
            node: var(name, ty, span),
            ty: v.ty,
        }
    }

    /// A list literal whose items are plain reads: any other item is
    /// bound ahead of the literal, since the lowering stores the elements
    /// in one block and an item may carry control flow (a check after a
    /// call) below a wrapper.
    fn array_of(&mut self, items: Vec<Node>, pre: &mut Vec<St>, span: Span) -> Node {
        let mut plain = Vec::with_capacity(items.len());
        for item in items {
            if Self::is_simple(&item) {
                plain.push(item);
            } else {
                let name = self.temp();
                let ty = item.ty.clone();
                pre.push(let_(name, ty.clone(), item, span));
                plain.push(var(name, ty, span));
            }
        }
        node(TypedExpression::Array(plain), self.m.anys(), span)
    }

    /// The truth of a value, as a boolean.
    fn truthy(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Bool => v.node,
            Ty::Nil => block_value(vec![expr_stmt(v.node)], bool_lit(false, span), span),
            Ty::Int | Ty::Float | Ty::Str | Ty::Table => {
                if Self::is_simple(&v.node) {
                    bool_lit(true, span)
                } else {
                    block_value(vec![expr_stmt(v.node)], bool_lit(true, span), span)
                }
            }
            _ => call("zl_truthy", vec![v.node], prim(PrimitiveType::Bool), span),
        }
    }

    // ─── errors ─────────────────────────────────────────────────

    /// The placeholder this function leaves with once an error is
    /// pending: whoever called it checks next.
    fn placeholder_return(&mut self, span: Span) -> St {
        if !self.tbc.is_empty() {
            let mut statements = Vec::new();
            self.closes_from(1, Some(Self::pending(span)), span, &mut statements);
            statements.push(self.placeholder_return_plain(span));
            return stmt(TypedStatement::Block(TypedBlock { statements, span }), span);
        }
        self.placeholder_return_plain(span)
    }

    fn placeholder_return_plain(&mut self, span: Span) -> St {
        let value = match self.returns.clone() {
            Returns::Fixed(types) if types.is_empty() => None,
            Returns::Fixed(types) if types.len() == 1 => {
                Some(self.zero_of(types[0].settled(), span))
            }
            Returns::Fixed(_) => Some(node(
                TypedExpression::Array(Vec::new()),
                self.m.anys(),
                span,
            )),
            Returns::Dynamic => Some(nil(span)),
        };
        ret(value, span)
    }

    fn zero_of(&mut self, ty: Ty, span: Span) -> Node {
        match ty {
            Ty::Int => int_lit(0, span),
            Ty::Float => float_lit(0.0, span),
            Ty::Bool => bool_lit(false, span),
            Ty::Str => str_lit("", span),
            Ty::Table => call("zl_table_new", vec![], self.ir(Ty::Table), span),
            _ => nil(span),
        }
    }

    fn pending(span: Span) -> Node {
        var(intern(library::PENDING), Type::Any, span)
    }

    /// `if an error is pending, leave`; then the line is this one
    /// again, since a callee sets its own.
    fn pending_check(&mut self, span: Span) -> St {
        self.raised = true;
        self.line_needed = true;
        let cond = binary(
            BinaryOp::Ne,
            Self::pending(span),
            nil(span),
            prim(PrimitiveType::Bool),
            span,
        );
        let leave = self.placeholder_return(span);
        let restore = self.set_line(span);
        if_(cond, vec![leave], Some(vec![restore]), span)
    }

    /// `zl_line = <the line of span>`.
    fn set_line(&mut self, span: Span) -> St {
        let line = self.m.line_of(span);
        assign(
            var(intern(library::LINE), prim(PrimitiveType::I64), span),
            int_lit(line, span),
            span,
        )
    }

    /// Whether a call node names a library function that can raise.
    fn call_can_raise(&self, node: &Node) -> bool {
        let TypedExpression::Call(c) = &node.node else {
            return false;
        };
        let TypedExpression::Variable(name) = &c.callee.node else {
            return false;
        };
        name.resolve_global()
            .is_some_and(|n| self.m.fallible.contains(n.as_str()))
    }

    /// A value from a call that may have raised: held, then checked
    /// before anything uses it. A call that cannot raise is left alone.
    fn guarded(&mut self, v: Val) -> Val {
        if !self.call_can_raise(&v.node) {
            return v;
        }
        self.guard(v)
    }

    fn guard(&mut self, v: Val) -> Val {
        let span = v.node.span;
        let mut pre = Vec::new();
        let held = if v.node.ty == prim(PrimitiveType::Unit) {
            pre.push(expr_stmt(v.node));
            Val {
                node: nil(span),
                ty: Ty::Nil,
            }
        } else {
            self.hold(v, &mut pre)
        };
        pre.push(self.pending_check(span));
        Val {
            node: block_value(pre, held.node, span),
            ty: held.ty,
        }
    }

    /// A statement calling something that may have raised, checked.
    fn guarded_stmt(&mut self, node: Node) -> St {
        if !self.call_can_raise(&node) {
            return expr_stmt(node);
        }
        let span = node.span;
        let mut statements = vec![expr_stmt(node)];
        statements.push(self.pending_check(span));
        stmt(TypedStatement::Block(TypedBlock { statements, span }), span)
    }

    // ─── variables ──────────────────────────────────────────────

    fn var_ty(&self, v: VarId) -> Ty {
        self.m.inferred.var(v).settled()
    }

    /// Where a variable of this function or an enclosing one lives,
    /// deciding for one not seen yet.
    fn storage_of(&mut self, v: VarId) -> Storage {
        if let Some(s) = self.storage.get(&v) {
            return s.clone();
        }
        let info = self.scopes().var(v);
        let ty = self.var_ty(v);
        let s = if info.is_module_var() {
            let symbol = self.m.module_local_symbol(v);
            self.m.declare_module_var(symbol, ty);
            Storage::Module(symbol, ty)
        } else if info.needs_cell() {
            let symbol = intern(&format!("{}$cell{}", info.name, v.0));
            Storage::Cell(symbol, ty)
        } else {
            let symbol = self.local_symbol(v);
            Storage::Local(symbol, ty)
        };
        self.storage.insert(v, s.clone());
        s
    }

    /// Read a variable.
    fn read_var(&mut self, v: VarId, span: Span) -> Val {
        match self.storage_of(v) {
            Storage::Local(name, ty) | Storage::Module(name, ty) => Val {
                node: var(name, self.ir(ty), span),
                ty,
            },
            Storage::Cell(name, ty) => {
                let cell = var(name, self.m.anys(), span);
                let element = index(cell, int_lit(0, span), Type::Any, span);
                let node = self.coerce(
                    Val {
                        node: element,
                        ty: Ty::Any,
                    },
                    ty,
                );
                Val { node, ty }
            }
        }
    }

    /// Store into a variable.
    fn write_var(&mut self, v: VarId, value: Val, span: Span) -> St {
        match self.storage_of(v) {
            Storage::Local(name, ty) | Storage::Module(name, ty) => {
                let value = self.coerce(value, ty);
                assign(var(name, self.ir(ty), span), value, span)
            }
            Storage::Cell(name, ty) => {
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, Ty::Any);
                let cell = var(name, self.m.anys(), span);
                assign(index(cell, int_lit(0, span), Type::Any, span), value, span)
            }
        }
    }

    /// Declare a local with a value; its first declaration is a `let`,
    /// a cell's is the cell.
    fn declare_var(&mut self, v: VarId, value: Val, span: Span) -> St {
        match self.storage_of(v) {
            Storage::Local(name, ty) => {
                let value = self.coerce(value, ty);
                let_(name, self.ir(ty), value, span)
            }
            Storage::Module(name, ty) => {
                let value = self.coerce(value, ty);
                assign(var(name, self.ir(ty), span), value, span)
            }
            Storage::Cell(name, ty) => {
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, Ty::Any);
                let mut pre = Vec::new();
                let cell = self.array_of(vec![value], &mut pre, span);
                let_(name, self.m.anys(), block_value(pre, cell, span), span)
            }
        }
    }

    /// Read a name.
    fn read_name(&mut self, token: &TokenReference) -> Result<Val> {
        let span = span_of(token);
        let binding = self
            .scopes()
            .binding(token)
            .cloned()
            .unwrap_or_else(|| Binding::Global(ident(token)));
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => Ok(self.read_var(v, span)),
            Binding::Global(name) => self.read_global(&name, span),
        }
    }

    fn read_global(&mut self, name: &str, span: Span) -> Result<Val> {
        if self.scopes().dynamic_globals {
            let key = Val {
                node: str_lit(name, span),
                ty: Ty::Str,
            };
            let g = self.globals_table(span);
            return Ok(self.index_read(g, key, span));
        }
        if let Some(f) = self.scopes().known_global_function(name) {
            return Ok(self.function_value(f, span));
        }
        if let Some(b) = types::builtin_named(self.scopes(), name) {
            return Ok(self.builtin_value(b, span));
        }
        if !self.scopes().global_writes.contains_key(name) {
            if crate::library::stdlib::LIBS.contains(&name) {
                return Ok(Val {
                    node: call(
                        &crate::library::stdlib::lib_table_fn(name),
                        vec![],
                        Type::Any,
                        span,
                    ),
                    ty: Ty::Any,
                });
            }
            if name == "arg" {
                return Ok(Val {
                    node: call("zl_arg_table", vec![], Type::Any, span),
                    ty: Ty::Any,
                });
            }
            if name == "_G" || name == "_ENV" {
                return unsupported("`_G` other than as `_G.name`", span);
            }
            self.nil_global = Some(name.to_string());
            return Ok(self.nil_val(span));
        }
        let ty = self.typer().global_ty(name);
        let symbol = Module::global_symbol(name);
        self.m.declare_module_var(symbol, ty);
        Ok(Val {
            node: var(symbol, self.ir(ty), span),
            ty,
        })
    }

    fn write_global(&mut self, name: &str, value: Val, span: Span) -> Result<St> {
        if self.scopes().dynamic_globals {
            if name == "_ENV" {
                return unsupported("assigning `_ENV`", span);
            }
            let key = Val {
                node: str_lit(name, span),
                ty: Ty::Str,
            };
            let g = self.globals_table(span);
            return Ok(self.index_write(g, key, value, span));
        }
        if self.scopes().known_global_function(name).is_some() {
            // The one declaration of a known function is its typed
            // entry; nothing is stored.
            return Ok(stmt(
                TypedStatement::Block(TypedBlock {
                    statements: Vec::new(),
                    span,
                }),
                span,
            ));
        }
        let ty = self.typer().global_ty(name);
        let symbol = Module::global_symbol(name);
        self.m.declare_module_var(symbol, ty);
        let value = self.coerce(value, ty);
        Ok(assign(var(symbol, self.ir(ty), span), value, span))
    }

    /// The globals table, when the program reaches its globals through
    /// one: every global is an entry, the builtins included.
    fn globals_table(&mut self, span: Span) -> Val {
        Val {
            node: var(intern(library::GLOBALS), self.ir(Ty::Table), span),
            ty: Ty::Table,
        }
    }

    /// A builtin as a function value.
    fn builtin_value(&mut self, b: &Builtin, span: Span) -> Val {
        Val {
            node: call(
                "zl_func_of",
                vec![
                    code_of(&crate::library::stdlib::wrapper_name(b), span),
                    int_lit(VARIADIC_ARITY, span),
                ],
                Type::Any,
                span,
            ),
            ty: Ty::Any,
        }
    }

    // ─── functions as values ────────────────────────────────────

    /// The record of a function: its code, its arity and what it
    /// captures, made where the function is defined or named.
    fn function_value(&mut self, f: FuncId, span: Span) -> Val {
        let info = self.scopes().func(f);
        let arity = if info.is_vararg {
            VARIADIC_ARITY
        } else {
            arity_word(info.params.len(), info.params.len())
        };
        let captures = info.captures.clone();
        let mut cells = Vec::with_capacity(captures.len());
        for v in captures {
            cells.push(self.capture_value(v, span));
        }
        let mut pre = Vec::new();
        let cells = self.array_of(cells, &mut pre, span);
        let record = call(
            "zb_func_new",
            vec![
                code_of(&self.m.code_name(f), span),
                int_lit(arity, span),
                cells,
            ],
            Type::Any,
            span,
        );
        Val {
            node: block_value(pre, record, span),
            ty: Ty::Any,
        }
    }

    /// What a closure's record holds for a captured variable: the cell
    /// itself, boxed, or the value copied.
    fn capture_value(&mut self, v: VarId, span: Span) -> Node {
        match self.storage_of(v) {
            Storage::Cell(name, _) => call(
                "zb_list_box_any",
                vec![var(name, self.m.anys(), span)],
                Type::Any,
                span,
            ),
            Storage::Local(..) | Storage::Module(..) => {
                let value = self.read_var(v, span);
                self.boxed(value)
            }
        }
    }

    /// The prologue of a function taking its record: each capture
    /// read out of `env`.
    fn env_prologue(&mut self, span: Span) -> Vec<St> {
        let mut out = Vec::new();
        let captures = self.scopes().func(self.func).captures.clone();
        for (i, v) in captures.iter().enumerate() {
            let slot = index(
                var(intern("env"), self.m.anys(), span),
                int_lit((RECORD_CELLS_AT + i) as i64, span),
                Type::Any,
                span,
            );
            let ty = self.var_ty(*v);
            let info = self.scopes().var(*v);
            if info.needs_cell() {
                let symbol = intern(&format!("{}$cell{}", info.name, v.0));
                self.storage.insert(*v, Storage::Cell(symbol, ty));
                out.push(let_(
                    symbol,
                    self.m.anys(),
                    call("zb_unbox_list_raw_any", vec![slot], self.m.anys(), span),
                    span,
                ));
            } else {
                let symbol = self.local_symbol(*v);
                self.storage.insert(*v, Storage::Local(symbol, ty));
                let value = self.coerce(
                    Val {
                        node: slot,
                        ty: Ty::Any,
                    },
                    ty,
                );
                out.push(let_(symbol, self.ir(ty), value, span));
            }
        }
        out
    }

    // ─── expressions ────────────────────────────────────────────

    /// An expression in single-value position.
    fn expr(&mut self, e: &Expression) -> Result<Val> {
        let span = span_of(e);
        match e {
            Expression::Number(t) => {
                let text = t.token().to_string();
                Ok(match crate::host::parse_numeral(text.trim()) {
                    crate::host::Numeral::Int(v) => Val {
                        node: int_lit(v, span),
                        ty: Ty::Int,
                    },
                    crate::host::Numeral::Float(v) => Val {
                        node: float_lit(v, span),
                        ty: Ty::Float,
                    },
                    crate::host::Numeral::None => {
                        return Err(Error::Syntax {
                            message: format!("malformed number near '{}'", text.trim()),
                            span: (span.start, span.end),
                        });
                    }
                })
            }
            Expression::String(t) => {
                let bytes = string_bytes(t).map_err(|message| Error::Syntax {
                    message,
                    span: (span.start, span.end),
                })?;
                Ok(Val {
                    node: string_literal(bytes, span),
                    ty: Ty::Str,
                })
            }
            Expression::Symbol(t) => match t.token().to_string().trim() {
                "true" => Ok(Val {
                    node: bool_lit(true, span),
                    ty: Ty::Bool,
                }),
                "false" => Ok(Val {
                    node: bool_lit(false, span),
                    ty: Ty::Bool,
                }),
                "nil" => Ok(self.nil_val(span)),
                "..." => {
                    let Some(varargs) = self.varargs else {
                        return unsupported("`...` outside a vararg function", span);
                    };
                    Ok(Val {
                        node: call(
                            "zl_value_at",
                            vec![var(varargs, self.m.anys(), span), int_lit(1, span)],
                            Type::Any,
                            span,
                        ),
                        ty: Ty::Any,
                    })
                }
                other => unsupported(format!("the symbol `{other}`"), span),
            },
            Expression::Parentheses { expression, .. } => {
                // Parentheses truncate to one value.
                self.expr(expression)
            }
            Expression::Function(f) => {
                let id = self.scopes().function_of(f.body());
                self.lower_function(id, f.body(), false)?;
                Ok(self.function_value(id, span))
            }
            Expression::TableConstructor(t) => self.table_constructor(t, span),
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                let multi = self.suffixed(c.prefix(), &suffixes, span)?;
                Ok(self.first_of(multi, span))
            }
            Expression::Var(v) => match v {
                Var::Name(token) => self.read_name(token),
                Var::Expression(v) => {
                    let suffixes: Vec<&Suffix> = v.suffixes().collect();
                    let multi = self.suffixed(v.prefix(), &suffixes, span)?;
                    Ok(self.first_of(multi, span))
                }
                _ => unsupported("this variable form", span),
            },
            Expression::UnaryOperator { unop, expression } => {
                let v = self.expr(expression)?;
                self.unary_op(unop, v, span)
            }
            Expression::BinaryOperator { lhs, binop, rhs } => self.binary_op(binop, lhs, rhs, span),
            _ => unsupported("this expression form", span),
        }
    }

    /// The single value of what may be several.
    fn first_of(&mut self, multi: Multi, span: Span) -> Val {
        match multi {
            Multi::Fixed(mut vals) => {
                if vals.is_empty() {
                    self.nil_val(span)
                } else if vals.len() == 1 {
                    vals.pop().unwrap()
                } else {
                    // Several values of one call: the first carries the
                    // call, the rest are reads of its result.
                    vals.swap_remove(0)
                }
            }
            Multi::Dynamic(node) => Val {
                node: call("zl_first", vec![node], Type::Any, span),
                ty: Ty::Any,
            },
            Multi::None(node) => Val {
                node: block_value(vec![expr_stmt(node)], nil(span), span),
                ty: Ty::Nil,
            },
        }
    }

    /// An expression in last position, where it may supply several
    /// values.
    fn expr_multi(&mut self, e: &Expression) -> Result<Multi> {
        let span = span_of(e);
        match e {
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                self.suffixed(c.prefix(), &suffixes, span)
            }
            Expression::Var(Var::Expression(v))
                if matches!(v.suffixes().last(), Some(Suffix::Call(_))) =>
            {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.suffixed(v.prefix(), &suffixes, span)
            }
            Expression::Symbol(t) if t.token().to_string().trim() == "..." => {
                let Some(varargs) = self.varargs else {
                    return unsupported("`...` outside a vararg function", span);
                };
                Ok(Multi::Dynamic(call(
                    "zl_values_from",
                    vec![var(varargs, self.m.anys(), span), int_lit(1, span)],
                    Type::Any,
                    span,
                )))
            }
            _ => Ok(Multi::Fixed(vec![self.expr(e)?])),
        }
    }

    /// A list of expressions as the values it supplies, the last one
    /// expanded: statements to run first, the fixed values, and the
    /// dynamic tail if any. A last call supplying no values still runs,
    /// after the values before it.
    fn expr_list(&mut self, exprs: &[&Expression]) -> Result<(Vec<St>, Vec<Val>, Option<Node>)> {
        let mut pre = Vec::new();
        let mut vals: Vec<Val> = Vec::new();
        let mut tail = None;
        for (i, e) in exprs.iter().enumerate() {
            if i + 1 == exprs.len() {
                match self.expr_multi(e)? {
                    Multi::Fixed(more) => vals.extend(more),
                    Multi::Dynamic(node) => tail = Some(node),
                    Multi::None(node) => {
                        let held = std::mem::take(&mut vals);
                        for v in held {
                            let h = self.hold(v, &mut pre);
                            vals.push(h);
                        }
                        pre.push(expr_stmt(node));
                    }
                }
            } else {
                vals.push(self.expr(e)?);
            }
        }
        Ok((pre, vals, tail))
    }

    /// The values of an expression list as a `List<Any>`.
    fn packed_list(&mut self, exprs: &[&Expression], span: Span) -> Result<Node> {
        let (mut pre, vals, tail) = self.expr_list(exprs)?;
        let mut items = Vec::with_capacity(vals.len());
        for v in vals {
            items.push(self.boxed(v));
        }
        let list = self.array_of(items, &mut pre, span);
        Ok(match tail {
            None => block_value(pre, list, span),
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(name, self.m.anys(), list, span));
                pre.push(expr_stmt(call(
                    "zl_append_values",
                    vec![var(name, self.m.anys(), span), tail],
                    prim(PrimitiveType::Unit),
                    span,
                )));
                block_value(pre, var(name, self.m.anys(), span), span)
            }
        })
    }

    /// `n` values from an expression list, Lua's way: the last
    /// expression's several values fill the rest, missing ones are nil.
    fn adjusted(
        &mut self,
        exprs: &[&Expression],
        n: usize,
        span: Span,
    ) -> Result<(Vec<St>, Vec<Val>)> {
        let (mut pre, vals, tail) = self.expr_list(exprs)?;
        let mut out: Vec<Val> = Vec::with_capacity(n);
        for v in vals {
            if out.len() < n {
                let held = self.hold(v, &mut pre);
                out.push(held);
            } else {
                // Extra values run for their effects.
                pre.push(expr_stmt(v.node));
            }
        }
        if let Some(tail) = tail {
            if out.len() < n {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                let mut k = 1;
                while out.len() < n {
                    out.push(Val {
                        node: call(
                            "zl_value_at",
                            vec![var(name, self.m.anys(), span), int_lit(k, span)],
                            Type::Any,
                            span,
                        ),
                        ty: Ty::Any,
                    });
                    k += 1;
                }
            } else {
                pre.push(expr_stmt(tail));
            }
        }
        while out.len() < n {
            out.push(self.nil_val(span));
        }
        Ok((pre, out))
    }

    // ─── operators ──────────────────────────────────────────────

    fn unary_op(&mut self, op: &UnOp, v: Val, span: Span) -> Result<Val> {
        Ok(match op {
            UnOp::Not(_) => {
                let t = self.truthy(v);
                Val {
                    node: unary(UnaryOp::Not, t, prim(PrimitiveType::Bool), span),
                    ty: Ty::Bool,
                }
            }
            UnOp::Minus(_) => match v.ty {
                Ty::Int => Val {
                    node: binary(
                        BinaryOp::Sub,
                        int_lit(0, span),
                        v.node,
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    ty: Ty::Int,
                },
                Ty::Float => Val {
                    node: unary(UnaryOp::Minus, v.node, prim(PrimitiveType::F64), span),
                    ty: Ty::Float,
                },
                _ => {
                    let b = self.boxed(v);
                    self.guard(Val {
                        node: call("zl_unm", vec![b], Type::Any, span),
                        ty: Ty::Any,
                    })
                }
            },
            UnOp::Hash(_) => match v.ty {
                Ty::Str => Val {
                    node: call("zb_str_len", vec![v.node], prim(PrimitiveType::I64), span),
                    ty: Ty::Int,
                },
                Ty::Table => Val {
                    node: call("zl_table_len", vec![v.node], prim(PrimitiveType::I64), span),
                    ty: Ty::Int,
                },
                _ => {
                    let b = self.boxed(v);
                    self.guard(Val {
                        node: call("zl_len_any", vec![b], Type::Any, span),
                        ty: Ty::Any,
                    })
                }
            },
            UnOp::Tilde(_) => match v.ty {
                Ty::Int => Val {
                    node: binary(
                        BinaryOp::BitXor,
                        v.node,
                        int_lit(-1, span),
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    ty: Ty::Int,
                },
                _ => {
                    let b = self.boxed(v);
                    self.guard(Val {
                        node: call("zl_bnot", vec![b], Type::Any, span),
                        ty: Ty::Any,
                    })
                }
            },
            _ => return unsupported("this unary operator", span),
        })
    }

    fn binary_op(
        &mut self,
        op: &BinOp,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<Val> {
        // `and` and `or` evaluate their right side conditionally.
        match op {
            BinOp::And(_) | BinOp::Or(_) => {
                return self.logical(matches!(op, BinOp::And(_)), lhs, rhs, span);
            }
            _ => {}
        }
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let v = self.binary_vals(op, a, b, span)?;
        Ok(self.guarded(v))
    }

    fn binary_vals(&mut self, op: &BinOp, a: Val, b: Val, span: Span) -> Result<Val> {
        let ints = a.ty == Ty::Int && b.ty == Ty::Int;
        let numbers = a.ty.is_number() && b.ty.is_number();
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let as_float = |this: &mut Self, v: Val| this.coerce(v, Ty::Float);
        let int_result = |n: Node| Val {
            node: n,
            ty: Ty::Int,
        };
        let float_result = |n: Node| Val {
            node: n,
            ty: Ty::Float,
        };
        let bool_result = |n: Node| Val {
            node: n,
            ty: Ty::Bool,
        };
        let arith = |this: &mut Self, code: i64, a: Val, b: Val| {
            let a = this.boxed(a);
            let b = this.boxed(b);
            Val {
                node: call("zl_arith", vec![int_lit(code, span), a, b], Type::Any, span),
                ty: Ty::Any,
            }
        };
        Ok(match op {
            BinOp::Plus(_) | BinOp::Minus(_) | BinOp::Star(_) => {
                let bop = match op {
                    BinOp::Plus(_) => BinaryOp::Add,
                    BinOp::Minus(_) => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                let code = match op {
                    BinOp::Plus(_) => OP_ADD,
                    BinOp::Minus(_) => OP_SUB,
                    _ => OP_MUL,
                };
                if ints {
                    int_result(binary(bop, a.node, b.node, i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(binary(bop, x, y, f64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::Slash(_) => {
                if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(binary(BinaryOp::Div, x, y, f64_t, span))
                } else {
                    arith(self, OP_DIV, a, b)
                }
            }
            BinOp::Caret(_) => {
                if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_pow", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_POW, a, b)
                }
            }
            BinOp::DoubleSlash(_) => {
                if ints {
                    int_result(call("zl_idiv_i64", vec![a.node, b.node], i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_idiv_f64", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_IDIV, a, b)
                }
            }
            BinOp::Percent(_) => {
                if ints {
                    int_result(call("zl_mod_i64", vec![a.node, b.node], i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_mod_f64", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_MOD, a, b)
                }
            }
            BinOp::Ampersand(_) | BinOp::Pipe(_) | BinOp::Tilde(_) => {
                let (bop, code) = match op {
                    BinOp::Ampersand(_) => (BinaryOp::BitAnd, OP_BAND),
                    BinOp::Pipe(_) => (BinaryOp::BitOr, OP_BOR),
                    _ => (BinaryOp::BitXor, OP_BXOR),
                };
                if ints {
                    int_result(binary(bop, a.node, b.node, i64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::DoubleLessThan(_) | BinOp::DoubleGreaterThan(_) => {
                let (f, code) = match op {
                    BinOp::DoubleLessThan(_) => ("zl_shl_i64", OP_SHL),
                    _ => ("zl_shr_i64", OP_SHR),
                };
                if ints {
                    int_result(call(f, vec![a.node, b.node], i64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::TwoDots(_) => {
                let text = |t: Ty| matches!(t, Ty::Str | Ty::Int | Ty::Float);
                if text(a.ty) && text(b.ty) {
                    let x = self.text_of(a);
                    let y = self.text_of(b);
                    Val {
                        node: binary(BinaryOp::Add, x, y, prim(PrimitiveType::String), span),
                        ty: Ty::Str,
                    }
                } else {
                    let x = self.boxed(a);
                    let y = self.boxed(b);
                    Val {
                        node: call("zl_concat", vec![x, y], Type::Any, span),
                        ty: Ty::Any,
                    }
                }
            }
            BinOp::TwoEqual(_) | BinOp::TildeEqual(_) => {
                let negate = matches!(op, BinOp::TildeEqual(_));
                let eq = self.equal(a, b, span);
                bool_result(if negate {
                    unary(UnaryOp::Not, eq, bool_t, span)
                } else {
                    eq
                })
            }
            BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_) => {
                // `a > b` is `b < a`.
                let (a, b, or_equal) = match op {
                    BinOp::LessThan(_) => (a, b, false),
                    BinOp::LessThanEqual(_) => (a, b, true),
                    BinOp::GreaterThan(_) => (b, a, false),
                    _ => (b, a, true),
                };
                bool_result(self.less(a, b, or_equal, span))
            }
            _ => return unsupported("this binary operator", span),
        })
    }

    /// The text a string or number contributes to `..`.
    fn text_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Str => v.node,
            Ty::Int => call(
                "zb_str_of_int",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            Ty::Float => call(
                "zl_float_str",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            _ => {
                let b = self.boxed(v);
                self.guard(Val {
                    node: call("zl_concat_text", vec![b], prim(PrimitiveType::String), span),
                    ty: Ty::Str,
                })
                .node
            }
        }
    }

    fn equal(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        match (a.ty, b.ty) {
            (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float) | (Ty::Bool, Ty::Bool) => {
                binary(BinaryOp::Eq, a.node, b.node, bool_t, span)
            }
            (Ty::Int, Ty::Float) => call("zl_eq_if", vec![a.node, b.node], bool_t, span),
            (Ty::Float, Ty::Int) => call("zl_eq_if", vec![b.node, a.node], bool_t, span),
            (Ty::Str, Ty::Str) => call("zb_str_eq", vec![a.node, b.node], bool_t, span),
            (Ty::Nil, Ty::Nil) => block_value(
                vec![expr_stmt(a.node), expr_stmt(b.node)],
                bool_lit(true, span),
                span,
            ),
            // Two different known types are never equal; the operands
            // still run.
            (x, y)
                if x != Ty::Any && y != Ty::Any && x != y && x != Ty::Table && y != Ty::Table =>
            {
                block_value(
                    vec![expr_stmt(a.node), expr_stmt(b.node)],
                    bool_lit(false, span),
                    span,
                )
            }
            _ => {
                let x = self.boxed(a);
                let y = self.boxed(b);
                call("zl_eq", vec![x, y], bool_t, span)
            }
        }
    }

    fn less(&mut self, a: Val, b: Val, or_equal: bool, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let op = if or_equal { BinaryOp::Le } else { BinaryOp::Lt };
        if a.ty.is_number() && b.ty.is_number() {
            if a.ty == b.ty {
                return binary(op, a.node, b.node, bool_t, span);
            }
            // An integer against a float compares exactly.
            let f = match (a.ty, or_equal) {
                (Ty::Int, false) => "zl_lt_if",
                (Ty::Int, true) => "zl_le_if",
                (_, false) => "zl_lt_fi",
                _ => "zl_le_fi",
            };
            return call(f, vec![a.node, b.node], bool_t, span);
        }
        if a.ty == Ty::Str && b.ty == Ty::Str {
            let cmp = call(
                "zb_str_cmp",
                vec![a.node, b.node],
                prim(PrimitiveType::I32),
                span,
            );
            return binary(op, cmp, int32_lit(0, span), bool_t, span);
        }
        let x = self.boxed(a);
        let y = self.boxed(b);
        call(
            if or_equal { "zl_le" } else { "zl_lt" },
            vec![x, y],
            bool_t,
            span,
        )
    }

    /// `a and b` is `a` when that is false, else `b`; `a or b` the
    /// other way round. The right side runs only when needed.
    fn logical(
        &mut self,
        is_and: bool,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<Val> {
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let bool_t = prim(PrimitiveType::Bool);
        if a.ty == Ty::Bool && b.ty == Ty::Bool {
            let op = if is_and { BinaryOp::And } else { BinaryOp::Or };
            return Ok(Val {
                node: binary(op, a.node, b.node, bool_t, span),
                ty: Ty::Bool,
            });
        }
        // The result type is the join; `a` is read twice, so it is held.
        let ty = types::logical_ty(is_and, a.ty, b.ty).settled();
        let mut pre = Vec::new();
        let a = self.hold(a, &mut pre);
        let test = self.truthy(a.clone());
        let a_node = self.coerce(a, ty);
        let b_node = self.coerce(b, ty);
        let (then, els) = if is_and {
            (b_node, a_node)
        } else {
            (a_node, b_node)
        };
        let value = node(
            TypedExpression::If(TypedIfExpr {
                condition: Box::new(test),
                then_branch: Box::new(then),
                else_branch: Box::new(els),
            }),
            self.ir(ty),
            span,
        );
        Ok(Val {
            node: block_value(pre, value, span),
            ty,
        })
    }

    // ─── tables ─────────────────────────────────────────────────

    fn table_constructor(&mut self, t: &ast::TableConstructor, span: Span) -> Result<Val> {
        let fields: Vec<&ast::Field> = t.fields().iter().collect();
        let positional: Vec<&Expression> = fields
            .iter()
            .filter_map(|f| match f {
                ast::Field::NoKey(e) => Some(e),
                _ => None,
            })
            .collect();
        let arr = self.packed_list(&positional, span)?;
        let table_t = self.ir(Ty::Table);
        let table = call("zl_table_with_arr", vec![arr], table_t.clone(), span);
        let keyed: Vec<&ast::Field> = fields
            .iter()
            .copied()
            .filter(|f| !matches!(f, ast::Field::NoKey(_)))
            .collect();
        if keyed.is_empty() {
            return Ok(Val {
                node: table,
                ty: Ty::Table,
            });
        }
        let name = self.temp();
        let mut pre = vec![let_(name, table_t.clone(), table, span)];
        for f in keyed {
            let tb = var(name, table_t.clone(), span);
            match f {
                ast::Field::NameKey { key, value, .. } => {
                    let v = self.expr(value)?;
                    let k = Val {
                        node: str_lit(&ident(key), span),
                        ty: Ty::Str,
                    };
                    pre.push(self.raw_store(tb, k, v, span));
                }
                ast::Field::ExpressionKey { key, value, .. } => {
                    let k = self.expr(key)?;
                    let v = self.expr(value)?;
                    pre.push(self.raw_store(tb, k, v, span));
                }
                _ => {}
            }
        }
        Ok(Val {
            node: block_value(pre, var(name, table_t, span), span),
            ty: Ty::Table,
        })
    }

    /// `t[k] = v` without metamethods, on a table.
    fn raw_store(&mut self, tb: Node, k: Val, v: Val, span: Span) -> St {
        let v = self.boxed(v);
        let unit = prim(PrimitiveType::Unit);
        let node = if let Some(kb) = self.constant_key(&k) {
            call("zl_rawset_key", vec![tb, kb, v], unit, span)
        } else {
            match k.ty {
                Ty::Int => call("zl_rawseti", vec![tb, k.node, v], unit, span),
                Ty::Str => call("zl_rawset_str", vec![tb, k.node, v], unit, span),
                _ => {
                    let k = self.boxed(k);
                    call("zl_rawset", vec![tb, k, v], unit, span)
                }
            }
        };
        self.guarded_stmt(node)
    }

    /// A name the program spells, boxed once and shared: the compiler
    /// keeps a boxed string literal as one constant.
    fn constant_key(&mut self, key: &Val) -> Option<Node> {
        match &key.node.node {
            TypedExpression::Literal(TypedLiteral::String(_)) if key.ty == Ty::Str => {
                Some(self.boxed(key.clone()))
            }
            _ => None,
        }
    }

    /// `obj[key]`, with `__index`.
    fn index_read(&mut self, obj: Val, key: Val, span: Span) -> Val {
        if let Some(k) = self.constant_key(&key) {
            let node = match obj.ty {
                Ty::Table => call("zl_table_index_key", vec![obj.node, k], Type::Any, span),
                _ => {
                    let o = self.boxed(obj);
                    call("zl_index_key", vec![o, k], Type::Any, span)
                }
            };
            return self.guarded(Val { node, ty: Ty::Any });
        }
        let node = match obj.ty {
            Ty::Table => match key.ty {
                Ty::Int => call("zl_table_geti", vec![obj.node, key.node], Type::Any, span),
                Ty::Str => call(
                    "zl_table_index_str",
                    vec![obj.node, key.node],
                    Type::Any,
                    span,
                ),
                _ => {
                    let k = self.boxed(key);
                    call("zl_table_index", vec![obj.node, k], Type::Any, span)
                }
            },
            _ => {
                let o = self.boxed(obj);
                match key.ty {
                    Ty::Int => call("zl_geti", vec![o, key.node], Type::Any, span),
                    Ty::Str => call("zl_index_str", vec![o, key.node], Type::Any, span),
                    _ => {
                        let k = self.boxed(key);
                        call("zl_index", vec![o, k], Type::Any, span)
                    }
                }
            }
        };
        self.guarded(Val { node, ty: Ty::Any })
    }

    /// `obj[key] = value`, with `__newindex`.
    fn index_write(&mut self, obj: Val, key: Val, value: Val, span: Span) -> St {
        let unit = prim(PrimitiveType::Unit);
        let v = self.boxed(value);
        if let Some(k) = self.constant_key(&key) {
            let node = match obj.ty {
                Ty::Table => call("zl_table_setindex_key", vec![obj.node, k, v], unit, span),
                _ => {
                    let o = self.boxed(obj);
                    call("zl_setindex_key", vec![o, k, v], unit, span)
                }
            };
            return self.guarded_stmt(node);
        }
        let node = match obj.ty {
            Ty::Table => match key.ty {
                Ty::Int => call("zl_table_seti", vec![obj.node, key.node, v], unit, span),
                Ty::Str => call(
                    "zl_table_setindex_str",
                    vec![obj.node, key.node, v],
                    unit,
                    span,
                ),
                _ => {
                    let k = self.boxed(key);
                    call("zl_table_setindex", vec![obj.node, k, v], unit, span)
                }
            },
            _ => {
                let o = self.boxed(obj);
                match key.ty {
                    Ty::Int => call("zl_seti", vec![o, key.node, v], unit, span),
                    Ty::Str => call("zl_setindex_str", vec![o, key.node, v], unit, span),
                    _ => {
                        let k = self.boxed(key);
                        call("zl_setindex", vec![o, k, v], unit, span)
                    }
                }
            }
        };
        self.guarded_stmt(node)
    }

    // ─── calls ──────────────────────────────────────────────────

    /// A prefix followed by suffixes: indexing and calls, left to
    /// right. The last suffix may produce several values.
    fn suffixed(&mut self, prefix: &Prefix, suffixes: &[&Suffix], span: Span) -> Result<Multi> {
        // `_G.name` is the global.
        if let Some(name) = self.typer().global_member(prefix, suffixes) {
            let v = self.read_global(&name, span)?;
            return Ok(Multi::Fixed(vec![v]));
        }
        if let Some(name) = self
            .typer()
            .global_member(prefix, &suffixes[..suffixes.len().min(1)])
        {
            let mut multi = None;
            let head = self.read_global(&name, span)?;
            return self.suffixes_from(head, suffixes, 1, &mut multi, span);
        }
        // A direct call to a known function or a builtin, possibly
        // followed by more suffixes on its result.
        let mut multi: Option<Multi> = None;
        let mut first = 0;
        if let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.first()
            && let Some(f) = self.typer().known_callee(prefix)
        {
            multi = Some(self.direct_call(f, args, span)?);
            first = 1;
        }
        if multi.is_none() {
            let head = suffixes.len().min(2);
            for n in (1..=head).rev() {
                if let Some(b) = self.typer().builtin_callee(prefix, &suffixes[..n]) {
                    let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes[..n].last()
                    else {
                        unreachable!("a builtin callee ends in a call");
                    };
                    multi = Some(self.builtin_call(b, None, args, span)?);
                    first = n;
                    break;
                }
            }
        }
        if first == suffixes.len()
            && let Some(m) = multi
        {
            return Ok(m);
        }
        let current = match multi.take() {
            Some(m) => self.first_of(m, span),
            None => match prefix {
                Prefix::Name(token) => self.read_name(token)?,
                Prefix::Expression(e) => self.expr(e)?,
                _ => return unsupported("this prefix", span),
            },
        };
        self.suffixes_from(current, suffixes, first, &mut multi, span)
    }

    /// The suffixes from `first` on, applied to `current`.
    fn suffixes_from(
        &mut self,
        mut current: Val,
        suffixes: &[&Suffix],
        first: usize,
        multi: &mut Option<Multi>,
        span: Span,
    ) -> Result<Multi> {
        for (i, s) in suffixes.iter().enumerate().skip(first) {
            if let Some(m) = multi.take() {
                current = self.first_of(m, span);
            }
            let last = i + 1 == suffixes.len();
            match s {
                Suffix::Index(ast::Index::Dot { name, .. }) => {
                    let key = Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    };
                    current = self.index_read(current, key, span);
                }
                Suffix::Index(ast::Index::Brackets { expression, .. }) => {
                    let key = self.expr(expression)?;
                    current = self.index_read(current, key, span);
                }
                Suffix::Call(ast::Call::AnonymousCall(args)) => {
                    let m = self.value_call(current.clone(), None, args, span)?;
                    if last {
                        return Ok(m);
                    }
                    *multi = Some(m);
                }
                Suffix::Call(ast::Call::MethodCall(mc)) => {
                    let m = self.method_call(current.clone(), mc, span)?;
                    if last {
                        return Ok(m);
                    }
                    *multi = Some(m);
                }
                _ => return unsupported("this suffix", span),
            }
        }
        Ok(Multi::Fixed(vec![current]))
    }

    fn args_exprs<'e>(&self, args: &'e ast::FunctionArgs) -> Vec<&'e Expression> {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => arguments.iter().collect(),
            _ => Vec::new(),
        }
    }

    /// The single argument a call written `f "x"` or `f {…}` passes.
    fn literal_arg(&mut self, args: &ast::FunctionArgs, span: Span) -> Result<Option<Val>> {
        Ok(match args {
            ast::FunctionArgs::String(t) => {
                let bytes = string_bytes(t).map_err(|message| Error::Syntax {
                    message,
                    span: (span.start, span.end),
                })?;
                Some(Val {
                    node: string_literal(bytes, span),
                    ty: Ty::Str,
                })
            }
            ast::FunctionArgs::TableConstructor(t) => Some(self.table_constructor(t, span)?),
            _ => None,
        })
    }

    /// The values a call's arguments supply: `receiver` first for a
    /// method call.
    fn call_values(
        &mut self,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<(Vec<St>, Vec<Val>, Option<Node>)> {
        let mut vals = Vec::new();
        if let Some(r) = receiver {
            vals.push(r);
        }
        if let Some(v) = self.literal_arg(args, span)? {
            vals.push(v);
            return Ok((Vec::new(), vals, None));
        }
        let exprs = self.args_exprs(args);
        let (mut pre, more, tail) = self.expr_list(&exprs)?;
        if !pre.is_empty() {
            // The receiver runs before the arguments.
            let held = std::mem::take(&mut vals);
            let mut first = Vec::new();
            for v in held {
                let h = self.hold(v, &mut first);
                vals.push(h);
            }
            first.append(&mut pre);
            pre = first;
        }
        vals.extend(more);
        Ok((pre, vals, tail))
    }

    /// A call to a known function: its typed entry, the arguments
    /// adjusted to its parameters.
    fn direct_call(&mut self, f: FuncId, args: &ast::FunctionArgs, span: Span) -> Result<Multi> {
        let info = self.scopes().func(f);
        let sig = self.m.sig(f);
        let n = info.params.len();
        let is_vararg = info.is_vararg;
        let has_env = !info.top_level;
        let (mut pre, vals, tail) = self.call_values(None, args, span)?;
        // A tail of several values is read through a list.
        let tail_name = match tail {
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                Some(name)
            }
            None => None,
        };
        let mut lowered: Vec<Node> = Vec::new();
        if has_env {
            // A nested known function's record holds its captures; the
            // typed entry takes it as `env`. The record lives in the
            // function's variable.
            let record = match self
                .scopes()
                .local_functions
                .iter()
                .find(|(_, id)| **id == f)
            {
                Some((v, _)) => self.read_var(*v, span),
                None => self.function_value(f, span),
            };
            lowered.push(call(
                "zb_unbox_list_raw_any",
                vec![record.node],
                self.m.anys(),
                span,
            ));
        }
        let mut fixed = vals.into_iter();
        let mut from_tail: i64 = 0;
        for i in 0..n {
            let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
            let v = match fixed.next() {
                Some(v) => v,
                None => match tail_name {
                    Some(t) => {
                        from_tail += 1;
                        Val {
                            node: call(
                                "zl_value_at",
                                vec![var(t, self.m.anys(), span), int_lit(from_tail, span)],
                                Type::Any,
                                span,
                            ),
                            ty: Ty::Any,
                        }
                    }
                    None => self.nil_val(span),
                },
            };
            let v = self.hold(v, &mut pre);
            lowered.push(self.coerce(v, param_ty));
        }
        // The extras: the variadic list, or run for their effects.
        let extras: Vec<Val> = fixed.collect();
        if is_vararg {
            let mut items = Vec::with_capacity(extras.len());
            for v in extras {
                items.push(self.boxed(v));
            }
            let list = self.array_of(items, &mut pre, span);
            match tail_name {
                Some(t) => {
                    let list_name = self.temp();
                    pre.push(let_(list_name, self.m.anys(), list, span));
                    let rest = call(
                        "zl_slice",
                        vec![
                            var(t, self.m.anys(), span),
                            int_lit(from_tail, span),
                            list_len(var(t, self.m.anys(), span), span),
                        ],
                        self.m.anys(),
                        span,
                    );
                    pre.push(expr_stmt(call(
                        "zb_list_extend_any",
                        vec![var(list_name, self.m.anys(), span), rest],
                        prim(PrimitiveType::Unit),
                        span,
                    )));
                    lowered.push(var(list_name, self.m.anys(), span));
                }
                None => lowered.push(list),
            }
        } else {
            for v in extras {
                pre.push(expr_stmt(v.node));
            }
        }
        let value = call(
            &self.m.entry_name(f),
            lowered,
            self.m.return_ir(&sig.returns),
            span,
        );
        let raises = self.m.raises(f);
        if raises {
            self.raise_callees.insert(f);
        }
        Ok(self.call_result(value, &sig.returns, pre, raises, span))
    }

    /// The values a typed entry returned, checked for an error when the
    /// callee may raise.
    fn call_result(
        &mut self,
        value: Node,
        returns: &Returns,
        pre: Vec<St>,
        raises: bool,
        span: Span,
    ) -> Multi {
        match returns {
            Returns::Fixed(types) if types.is_empty() => {
                let mut pre = pre;
                pre.push(expr_stmt(value));
                if raises {
                    pre.push(self.pending_check(span));
                }
                Multi::None(block_value(pre, nil(span), span))
            }
            Returns::Fixed(types) if types.len() == 1 => {
                let ty = types[0].settled();
                let mut pre = pre;
                let value = if raises {
                    let held = self.hold(Val { node: value, ty }, &mut pre);
                    pre.push(self.pending_check(span));
                    held.node
                } else {
                    value
                };
                Multi::Fixed(vec![Val {
                    node: block_value(pre, value, span),
                    ty,
                }])
            }
            Returns::Fixed(types) => {
                // Several values: a list, each read out as its type.
                let name = self.temp();
                let mut pre = pre;
                pre.push(let_(name, self.m.anys(), value, span));
                if raises {
                    pre.push(self.pending_check(span));
                }
                let mut vals = Vec::with_capacity(types.len());
                for (i, ty) in types.iter().enumerate() {
                    let ty = ty.settled();
                    let element = index(
                        var(name, self.m.anys(), span),
                        int_lit(i as i64, span),
                        Type::Any,
                        span,
                    );
                    let read = self.coerce(
                        Val {
                            node: element,
                            ty: Ty::Any,
                        },
                        ty,
                    );
                    let read = if i == 0 {
                        block_value(std::mem::take(&mut pre), read, span)
                    } else {
                        read
                    };
                    vals.push(Val { node: read, ty });
                }
                Multi::Fixed(vals)
            }
            Returns::Dynamic => {
                let value = if raises {
                    self.guard(Val {
                        node: value,
                        ty: Ty::Any,
                    })
                    .node
                } else {
                    value
                };
                Multi::Dynamic(block_value(pre, value, span))
            }
        }
    }

    /// A call through a function value.
    fn value_call(
        &mut self,
        callee: Val,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<Multi> {
        // A value known to be nil cannot be called; the error names
        // what it was, as Lua's does.
        if callee.ty == Ty::Nil {
            let (pre, _, _) = self.call_values(receiver, args, span)?;
            let what = match callee.node.node {
                TypedExpression::Variable(name) => {
                    let text = name.resolve_global().unwrap_or_default();
                    match text.strip_prefix("lua$g$") {
                        Some(g) => format!("attempt to call a nil value (global '{g}')"),
                        None => "attempt to call a nil value".to_string(),
                    }
                }
                _ => match self.nil_global.take() {
                    Some(g) => format!("attempt to call a nil value (global '{g}')"),
                    None => "attempt to call a nil value".to_string(),
                },
            };
            let error = call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit(&what, span)],
                prim(PrimitiveType::Unit),
                span,
            );
            return Ok(Multi::None(block_value(pre, error, span)));
        }
        let f = self.boxed(callee);
        let (mut pre, vals, tail) = self.call_values(receiver, args, span)?;
        let f = if pre.is_empty() {
            f
        } else {
            self.hold(
                Val {
                    node: f,
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node
        };
        if tail.is_none() && vals.len() <= zyntax_builtins::functions::MAX_CALL_ARITY {
            let mut lowered = vec![f];
            let n = vals.len();
            for v in vals {
                lowered.push(self.boxed(v));
            }
            let v = self.guard(Val {
                node: call(&format!("zl_call_{n}"), lowered, Type::Any, span),
                ty: Ty::Any,
            });
            return Ok(Multi::Dynamic(block_value(pre, v.node, span)));
        }
        let mut items = Vec::with_capacity(vals.len());
        for v in vals {
            items.push(self.boxed(v));
        }
        let list = self.array_of(items, &mut pre, span);
        let list = match tail {
            None => list,
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(name, self.m.anys(), list, span));
                pre.push(expr_stmt(call(
                    "zl_append_values",
                    vec![var(name, self.m.anys(), span), tail],
                    prim(PrimitiveType::Unit),
                    span,
                )));
                var(name, self.m.anys(), span)
            }
        };
        let v = self.guard(Val {
            node: call("zl_call_packed", vec![f, list], Type::Any, span),
            ty: Ty::Any,
        });
        Ok(Multi::Dynamic(block_value(pre, v.node, span)))
    }

    /// `obj:name(args)`: `obj.name(obj, args)` with `obj` evaluated once.
    fn method_call(&mut self, obj: Val, mc: &ast::MethodCall, span: Span) -> Result<Multi> {
        let name = ident(mc.name());
        let mut pre = Vec::new();
        let obj = self.hold(obj, &mut pre);
        // A string's methods are the string library's.
        if obj.ty == Ty::Str
            && let Some(b) = types::builtin_member(self.scopes(), "string", &name)
        {
            let multi = self.builtin_call(b, Some(obj), mc.args(), span)?;
            return Ok(self.prefixed(pre, multi, span));
        }
        let key = Val {
            node: str_lit(&name, span),
            ty: Ty::Str,
        };
        let callee = self.index_read(obj.clone(), key, span);
        let multi = self.value_call(callee, Some(obj), mc.args(), span)?;
        Ok(self.prefixed(pre, multi, span))
    }

    /// Statements run before several values.
    fn prefixed(&mut self, pre: Vec<St>, multi: Multi, span: Span) -> Multi {
        if pre.is_empty() {
            return multi;
        }
        match multi {
            Multi::Dynamic(node) => Multi::Dynamic(block_value(pre, node, span)),
            Multi::None(node) => Multi::None(block_value(pre, node, span)),
            Multi::Fixed(mut vals) => {
                if vals.is_empty() {
                    Multi::Fixed(vec![Val {
                        node: block_value(pre, nil(span), span),
                        ty: Ty::Nil,
                    }])
                } else {
                    let first = vals.remove(0);
                    let mut out = vec![Val {
                        node: block_value(pre, first.node, span),
                        ty: first.ty,
                    }];
                    out.extend(vals);
                    Multi::Fixed(out)
                }
            }
        }
    }

    /// A call to a library function: each argument as the
    /// implementation takes it.
    fn builtin_call(
        &mut self,
        b: &Builtin,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<Multi> {
        // `rawget(_G, "x")` and `rawset(_G, "x", v)` are the global.
        if b.lib.is_empty() && (b.name == "rawget" || b.name == "rawset") {
            let exprs = self.args_exprs(args);
            if let Some(Expression::Var(Var::Name(token))) = exprs.first() {
                let is_g = matches!(self.scopes().binding(token), Some(Binding::Global(g)) if Scopes::is_globals_name(g))
                    && !self.scopes().global_writes.contains_key(&ident(token))
                    && !self.scopes().dynamic_globals;
                if is_g {
                    let name = exprs.get(1).and_then(|e| crate::scope::literal_string(e));
                    let Some(name) = name else {
                        return unsupported("`_G` indexed by anything but a string literal", span);
                    };
                    if b.name == "rawget" {
                        let v = self.read_global(&name, span)?;
                        return Ok(Multi::Fixed(vec![v]));
                    }
                    let Some(value) = exprs.get(2) else {
                        return unsupported("`rawset(_G, name)` without a value", span);
                    };
                    let v = self.expr(value)?;
                    let st = self.write_global(&name, v, span)?;
                    return Ok(Multi::None(block_value(vec![st], nil(span), span)));
                }
            }
        }
        let is_method = receiver.is_some();
        // `error(v, level)` above level 1 positions at the caller: the
        // line this function was entered at.
        if b.lib.is_empty() && b.name == "error" {
            let exprs = self.args_exprs(args);
            let level = exprs.get(1).and_then(|e| match e {
                Expression::Number(n) => {
                    match crate::host::parse_numeral(n.token().to_string().trim()) {
                        crate::host::Numeral::Int(v) => Some(v),
                        _ => None,
                    }
                }
                _ => None,
            });
            if level.is_some_and(|l| l >= 2) {
                self.entry_line = true;
                let (mut pre, vals, _) = self.call_values(None, args, span)?;
                let v = vals.into_iter().next().map(|v| self.boxed(v));
                let v = v.unwrap_or_else(|| nil(span));
                let call = call(
                    "zl_error_at",
                    vec![v, var(intern(ENTRY_LINE), prim(PrimitiveType::I64), span)],
                    prim(PrimitiveType::Unit),
                    span,
                );
                pre.push(expr_stmt(call));
                pre.push(self.pending_check(span));
                return Ok(Multi::None(block_value(pre, nil(span), span)));
            }
        }
        let (mut pre, vals, tail) = self.call_values(receiver, args, span)?;
        let mut vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut pre)).collect();
        // A tail of several values fills what follows through a list.
        let tail_list = match tail {
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                Some(name)
            }
            None => None,
        };
        let mut lowered = Vec::with_capacity(b.params.len());
        let mut consumed = 0;
        for (i, p) in b.params.iter().enumerate() {
            if let Param::Rest = p {
                // Everything left, boxed, plus the tail.
                let mut items = Vec::new();
                for v in vals.drain(..) {
                    items.push(self.boxed(v));
                }
                let list = self.array_of(items, &mut pre, span);
                let list = match tail_list {
                    Some(name) => {
                        let list_name = self.temp();
                        pre.push(let_(list_name, self.m.anys(), list, span));
                        pre.push(expr_stmt(call(
                            "zb_list_extend_any",
                            vec![
                                var(list_name, self.m.anys(), span),
                                var(name, self.m.anys(), span),
                            ],
                            prim(PrimitiveType::Unit),
                            span,
                        )));
                        var(list_name, self.m.anys(), span)
                    }
                    None => list,
                };
                lowered.push(list);
                consumed = usize::MAX;
                break;
            }
            let v = if !vals.is_empty() {
                Some(vals.remove(0))
            } else if let Some(name) = tail_list {
                consumed += 1;
                Some(Val {
                    node: call(
                        "zl_value_at",
                        vec![
                            var(name, self.m.anys(), span),
                            int_lit(consumed as i64, span),
                        ],
                        Type::Any,
                        span,
                    ),
                    ty: Ty::Any,
                })
            } else {
                None
            };
            // A method's receiver is not counted, and is not "bad
            // argument #0".
            let what = match (is_method, i) {
                (true, 0) => format!("calling '{}' on bad self", b.name),
                (true, i) => format!("bad argument #{i} to '{}'", b.name),
                (false, i) => format!("bad argument #{} to '{}'", i + 1, b.name),
            };
            let what = str_lit(&what, span);
            lowered.push(self.builtin_arg(p, v, what, span));
        }
        // Arguments past the parameters run for their effects.
        if consumed != usize::MAX {
            for v in vals {
                pre.push(expr_stmt(v.node));
            }
        }
        let ret_ir = crate::library::stdlib::ret_type(b.ret, &self.m.types);
        // An argument's conversion can raise as well as the function.
        let converts = lowered.iter().any(|a| self.call_can_raise(a));
        let value = call(b.func, lowered, ret_ir, span);
        let raises = converts || self.call_can_raise(&value);
        let ty = match b.ret {
            Ret::Unit => Ty::Nil,
            Ret::Multi => Ty::Any,
            r => types::ret_ty(r),
        };
        let v = Val { node: value, ty };
        let v = if raises { self.guard(v) } else { v };
        Ok(match b.ret {
            Ret::Unit => Multi::None(block_value(pre, v.node, span)),
            Ret::Multi => Multi::Dynamic(block_value(pre, v.node, span)),
            _ => Multi::Fixed(vec![Val {
                node: block_value(pre, v.node, span),
                ty: v.ty,
            }]),
        })
    }

    /// One argument of a builtin as its parameter takes it.
    fn builtin_arg(&mut self, p: &Param, v: Option<Val>, what: Node, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let str_t = prim(PrimitiveType::String);
        match p {
            Param::Any => match v {
                Some(v) => self.boxed(v),
                None => nil(span),
            },
            Param::Value => match v {
                Some(v) => self.boxed(v),
                None => call("zl_arg_value_missing", vec![what], Type::Any, span),
            },
            Param::Int => match v {
                Some(v) if v.ty == Ty::Int => v.node,
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_int", vec![b, what], i64_t, span)
                }
                None => call("zl_arg_int_missing", vec![what], i64_t, span),
            },
            Param::Float => match v {
                Some(v) if v.ty.is_number() => self.coerce(v, Ty::Float),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_float", vec![b, what], f64_t, span)
                }
                None => call("zl_arg_float_missing", vec![what], f64_t, span),
            },
            Param::Str => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty.is_number() => self.text_of(v),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_str", vec![b, what], str_t, span)
                }
                None => call("zl_arg_str_missing", vec![what], str_t, span),
            },
            Param::Table => match v {
                Some(v) if v.ty == Ty::Table => v.node,
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_as_table", vec![b, what], self.ir(Ty::Table), span)
                }
                None => call("zl_as_table_missing", vec![what], self.ir(Ty::Table), span),
            },
            Param::OptInt(d) => match v {
                Some(v) if v.ty == Ty::Int => v.node,
                Some(v) if v.ty == Ty::Nil => {
                    block_value(vec![expr_stmt(v.node)], int_lit(*d, span), span)
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call(
                        "zl_arg_opt_int",
                        vec![b, int_lit(*d, span), what],
                        i64_t,
                        span,
                    )
                }
                None => int_lit(*d, span),
            },
            Param::OptStr(d) => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty == Ty::Nil => {
                    block_value(vec![expr_stmt(v.node)], str_lit(d, span), span)
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call(
                        "zl_arg_opt_str",
                        vec![b, str_lit(d, span), what],
                        str_t,
                        span,
                    )
                }
                None => str_lit(d, span),
            },
            Param::Rest => unreachable!("the rest is gathered by the caller"),
        }
    }

    // ─── statements ─────────────────────────────────────────────

    /// The chunk's outermost block as segments of at most
    /// `SEGMENT_STATEMENTS` statements each, the final `return` in the
    /// last.
    fn segments(&mut self, block: &Block) -> Result<Vec<Vec<St>>> {
        self.depth = 1;
        let mut segments = Vec::new();
        let mut out = Vec::new();
        for s in block.stmts() {
            let at = out.len();
            self.line_needed = false;
            self.stmt(s, &mut out)?;
            self.record_line(span_of(s), at, &mut out);
            if out.len() >= crate::scope::SEGMENT_STATEMENTS {
                segments.push(std::mem::take(&mut out));
            }
        }
        if let Some(last) = block.last_stmt() {
            let span = span_of(last);
            let at = out.len();
            self.line_needed = false;
            match last {
                ast::LastStmt::Return(r) => {
                    let exprs: Vec<&Expression> = r.returns().iter().collect();
                    self.return_stmt(&exprs, span, &mut out)?;
                }
                _ => return unsupported("this statement", span),
            }
            self.record_line(span, at, &mut out);
        }
        if !out.is_empty() || segments.is_empty() {
            segments.push(out);
        }
        Ok(segments)
    }

    fn block(&mut self, block: &Block) -> Result<Vec<St>> {
        self.depth += 1;
        let mut out = Vec::new();
        for s in block.stmts() {
            let at = out.len();
            self.line_needed = false;
            self.stmt(s, &mut out)?;
            self.record_line(span_of(s), at, &mut out);
        }
        match block.last_stmt() {
            Some(last) => {
                let span = span_of(last);
                let at = out.len();
                self.line_needed = false;
                match last {
                    ast::LastStmt::Break(_) => {
                        // The loop's body and everything within it is left.
                        let loop_depth = self.loop_depths.last().copied().unwrap_or(1);
                        self.closes_from(loop_depth, None, span, &mut out);
                        out.push(stmt(TypedStatement::Break(None), span));
                    }
                    ast::LastStmt::Return(r) => {
                        let exprs: Vec<&Expression> = r.returns().iter().collect();
                        self.return_stmt(&exprs, span, &mut out)?;
                    }
                    _ => return unsupported("this statement", span),
                }
                self.record_line(span, at, &mut out);
            }
            None => {
                let span = span_of(block);
                self.closes_from(self.depth, None, span, &mut out);
            }
        }
        self.tbc.retain(|(d, _)| *d < self.depth);
        self.depth -= 1;
        Ok(out)
    }

    /// Close every `<close>` variable declared at `depth` or deeper,
    /// innermost first, with `error` (the pending error on an error
    /// exit) or nil. The variables stay in scope for the paths that
    /// go on.
    fn closes_from(&mut self, depth: usize, error: Option<Node>, span: Span, out: &mut Vec<St>) {
        let vars: Vec<VarId> = self
            .tbc
            .iter()
            .rev()
            .filter(|(d, _)| *d >= depth)
            .map(|(_, v)| *v)
            .collect();
        for v in vars {
            let value = self.read_var(v, span);
            let value = self.boxed(value);
            let err = error.clone().unwrap_or_else(|| nil(span));
            out.push(expr_stmt(call(
                "zl_close",
                vec![value, err],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
    }

    /// A statement that checks for an error stores its line first, for
    /// the position the error's message carries.
    fn record_line(&mut self, span: Span, at: usize, out: &mut Vec<St>) {
        if !self.line_needed {
            return;
        }
        self.line_needed = false;
        let st = self.set_line(span);
        out.insert(at, st);
    }

    fn return_stmt(&mut self, exprs: &[&Expression], span: Span, out: &mut Vec<St>) -> Result<()> {
        self.return_values(exprs, span, out)?;
        if self.tbc.is_empty() {
            return Ok(());
        }
        // The values are computed before anything is closed.
        let Some(last) = out.pop() else {
            return Ok(());
        };
        let TypedStatement::Return(value) = last.node else {
            out.push(last);
            return Ok(());
        };
        let value = value.map(|value| {
            let name = self.temp();
            let ty = value.ty.clone();
            out.push(let_(name, ty.clone(), *value, span));
            var(name, ty, span)
        });
        self.closes_from(1, None, span, out);
        out.push(ret(value, span));
        Ok(())
    }

    fn return_values(
        &mut self,
        exprs: &[&Expression],
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let returns = self.returns.clone();
        match returns {
            Returns::Fixed(types) if types.is_empty() => {
                for e in exprs {
                    let v = self.expr(e)?;
                    out.push(expr_stmt(v.node));
                }
                if self.func == CHUNK && self.scopes().split_chunk {
                    let flag = intern(RETURNED);
                    self.m.declare_module_var(flag, Ty::Bool);
                    out.push(assign(
                        var(flag, prim(PrimitiveType::Bool), span),
                        bool_lit(true, span),
                        span,
                    ));
                }
                out.push(ret(None, span));
            }
            Returns::Fixed(types) if types.len() == 1 => {
                let (pre, mut vals) = self.adjusted(exprs, 1, span)?;
                out.extend(pre);
                let v = vals.pop().unwrap();
                let value = self.coerce(v, types[0].settled());
                out.push(ret(Some(value), span));
            }
            Returns::Fixed(types) => {
                let (pre, vals) = self.adjusted(exprs, types.len(), span)?;
                out.extend(pre);
                let mut items = Vec::with_capacity(vals.len());
                for v in vals {
                    items.push(self.boxed(v));
                }
                let mut pre = Vec::new();
                let list = self.array_of(items, &mut pre, span);
                out.extend(pre);
                out.push(ret(Some(list), span));
            }
            Returns::Dynamic => {
                let (pre, vals, tail) = self.expr_list(exprs)?;
                out.extend(pre);
                let value = match (vals.len(), tail) {
                    (1, None) => {
                        let v = vals.into_iter().next().unwrap();
                        self.boxed(v)
                    }
                    (0, Some(tail)) => tail,
                    (_, tail) => {
                        let mut items = Vec::with_capacity(vals.len());
                        for v in vals {
                            items.push(self.boxed(v));
                        }
                        let mut pre = Vec::new();
                        let list = self.array_of(items, &mut pre, span);
                        out.extend(pre);
                        match tail {
                            None => call("zl_pack", vec![list], Type::Any, span),
                            Some(tail) => {
                                let name = self.temp();
                                let pre = vec![
                                    let_(name, self.m.anys(), list, span),
                                    expr_stmt(call(
                                        "zl_append_values",
                                        vec![var(name, self.m.anys(), span), tail],
                                        prim(PrimitiveType::Unit),
                                        span,
                                    )),
                                ];
                                block_value(
                                    pre,
                                    call(
                                        "zl_pack",
                                        vec![var(name, self.m.anys(), span)],
                                        Type::Any,
                                        span,
                                    ),
                                    span,
                                )
                            }
                        }
                    }
                };
                out.push(ret(Some(value), span));
            }
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt, out: &mut Vec<St>) -> Result<()> {
        let span = span_of(s);
        match s {
            Stmt::LocalAssignment(l) => {
                let names: Vec<&TokenReference> = l.names().iter().collect();
                let exprs: Vec<&Expression> = l.expressions().iter().collect();
                let (pre, vals) = if exprs.is_empty() {
                    (
                        Vec::new(),
                        names.iter().map(|_| self.nil_val(span)).collect(),
                    )
                } else {
                    self.adjusted(&exprs, names.len(), span)?
                };
                out.extend(pre);
                for (name, v) in names.iter().zip(vals) {
                    let id = self.scopes().declared(name);
                    out.push(self.declare_var(id, v, span));
                    // A `<close>` variable must hold something closable,
                    // and is closed when its block is left.
                    if self.scopes().var(id).attribute.as_deref() == Some("close") {
                        let value = self.read_var(id, span);
                        let value = self.boxed(value);
                        let check = call(
                            "zl_closable",
                            vec![value, str_lit(&ident(name), span)],
                            prim(PrimitiveType::Unit),
                            span,
                        );
                        let st = self.guarded_stmt(check);
                        out.push(st);
                        self.tbc.push((self.depth, id));
                    }
                }
            }
            Stmt::Assignment(a) => {
                let targets: Vec<&Var> = a.variables().iter().collect();
                let exprs: Vec<&Expression> = a.expressions().iter().collect();
                if targets.len() == 1 && exprs.len() == 1 {
                    // The common case: no temporaries.
                    let v = self.expr(exprs[0])?;
                    let st = self.assign_target(targets[0], v, span)?;
                    out.push(st);
                    return Ok(());
                }
                // Every right side runs before any target is stored.
                let (pre, vals) = self.adjusted(&exprs, targets.len(), span)?;
                out.extend(pre);
                for (target, v) in targets.iter().zip(vals) {
                    let st = self.assign_target(target, v, span)?;
                    out.push(st);
                }
            }
            Stmt::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                let multi = self.suffixed(c.prefix(), &suffixes, span)?;
                match multi {
                    Multi::Fixed(vals) => {
                        for v in vals {
                            out.push(expr_stmt(v.node));
                        }
                    }
                    Multi::Dynamic(node) | Multi::None(node) => out.push(expr_stmt(node)),
                }
            }
            Stmt::Do(d) => {
                let body = self.block(d.block())?;
                out.push(stmt(
                    TypedStatement::Block(TypedBlock {
                        statements: body,
                        span,
                    }),
                    span,
                ));
            }
            Stmt::If(i) => {
                let st = self.if_stmt(i, span)?;
                out.push(st);
            }
            Stmt::While(w) => {
                let cond = self.expr(w.condition())?;
                let cond = self.truthy(cond);
                let body = self.loop_body(w.block())?;
                out.push(while_(cond, body, span));
            }
            Stmt::Repeat(r) => {
                // `repeat body until c` is a loop leaving once `c` holds;
                // the condition sees the body's locals.
                let mut body = self.loop_body(r.block())?;
                let cond = self.expr(r.until())?;
                let cond = self.truthy(cond);
                body.push(if_(
                    cond,
                    vec![stmt(TypedStatement::Break(None), span)],
                    None,
                    span,
                ));
                out.push(while_(bool_lit(true, span), body, span));
            }
            Stmt::NumericFor(f) => self.numeric_for(f, span, out)?,
            Stmt::GenericFor(f) => self.generic_for(f, span, out)?,
            Stmt::LocalFunction(f) => {
                let v = self.scopes().declared(f.name());
                let id = self.scopes().function_of(f.body());
                self.lower_function(id, f.body(), false)?;
                // The record is made once, here, and lives in the
                // variable; a direct call reads its captures from it. A
                // function capturing its own variable needs the cell
                // before the record.
                if matches!(self.storage_of(v), Storage::Cell(..)) {
                    let placeholder = self.nil_val(span);
                    out.push(self.declare_var(v, placeholder, span));
                    let record = self.function_value(id, span);
                    out.push(self.write_var(v, record, span));
                } else {
                    let record = self.function_value(id, span);
                    out.push(self.declare_var(v, record, span));
                }
            }
            Stmt::FunctionDeclaration(f) => {
                let id = self.scopes().function_of(f.body());
                let is_method = f.name().method_name().is_some();
                self.lower_function(id, f.body(), is_method)?;
                let names: Vec<&TokenReference> = f.name().names().iter().collect();
                if names.len() == 1 && !is_method {
                    let token = names[0];
                    let binding = self
                        .scopes()
                        .binding(token)
                        .cloned()
                        .unwrap_or_else(|| Binding::Global(ident(token)));
                    if let Binding::Global(name) = &binding
                        && self.scopes().known_global_function(name) == Some(id)
                    {
                        // The typed entry is the function.
                        return Ok(());
                    }
                    let record = self.function_value(id, span);
                    let st = match binding {
                        Binding::Local(v) | Binding::Upvalue(v) => self.write_var(v, record, span),
                        Binding::Global(name) => self.write_global(&name, record, span)?,
                    };
                    out.push(st);
                    return Ok(());
                }
                // `function a.b.c()` / `function a.b:m()`: index down to
                // the holder, then store.
                let mut obj = self.read_name(names[0])?;
                let mut pre = Vec::new();
                for name in &names[1..names.len() - if is_method { 0 } else { 1 }] {
                    let key = Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    };
                    obj = self.index_read(obj, key, span);
                    obj = self.hold(obj, &mut pre);
                }
                let last = match f.name().method_name() {
                    Some(m) => ident(m),
                    None => ident(names[names.len() - 1]),
                };
                let record = self.function_value(id, span);
                let key = Val {
                    node: str_lit(&last, span),
                    ty: Ty::Str,
                };
                out.extend(pre);
                let st = self.index_write(obj, key, record, span);
                out.push(st);
            }
            Stmt::Goto(g) => {
                let Some(&id) = self
                    .scopes()
                    .gotos
                    .get(&crate::scope::pos_of(g.goto_token()))
                else {
                    return Err(Error::Syntax {
                        message: format!("no visible label '{}' for goto", ident(g.label_name())),
                        span: (span.start, span.end),
                    });
                };
                let target_depth = self.scopes().label_depths[&id];
                self.closes_from(target_depth + 1, None, span, out);
                out.push(stmt(TypedStatement::Goto(label_name(id)), span));
            }
            Stmt::Label(l) => {
                let id = self.scopes().labels[&crate::scope::pos_of(l.name())];
                out.push(stmt(TypedStatement::Label(label_name(id)), span));
            }
            _ => return unsupported("this statement", span),
        }
        Ok(())
    }

    /// Store into an assignment target.
    fn assign_target(&mut self, target: &Var, v: Val, span: Span) -> Result<St> {
        match target {
            Var::Name(token) => {
                let binding = self
                    .scopes()
                    .binding(token)
                    .cloned()
                    .unwrap_or_else(|| Binding::Global(ident(token)));
                match binding {
                    Binding::Local(id) | Binding::Upvalue(id) => {
                        if self.scopes().var(id).attribute.as_deref() == Some("const") {
                            return Err(Error::Syntax {
                                message: format!(
                                    "attempt to assign to const variable '{}'",
                                    ident(token)
                                ),
                                span: (span.start, span.end),
                            });
                        }
                        Ok(self.write_var(id, v, span))
                    }
                    Binding::Global(name) => self.write_global(&name, v, span),
                }
            }
            Var::Expression(ve) => {
                let suffixes: Vec<&Suffix> = ve.suffixes().collect();
                if let Some(name) = self.typer().global_member(ve.prefix(), &suffixes) {
                    return self.write_global(&name, v, span);
                }
                let Some((last, init)) = suffixes.split_last() else {
                    return unsupported("this assignment target", span);
                };
                let obj_multi = self.suffixed(ve.prefix(), init, span)?;
                let obj = self.first_of(obj_multi, span);
                let key = match last {
                    Suffix::Index(ast::Index::Dot { name, .. }) => Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    },
                    Suffix::Index(ast::Index::Brackets { expression, .. }) => {
                        self.expr(expression)?
                    }
                    _ => return unsupported("this assignment target", span),
                };
                Ok(self.index_write(obj, key, v, span))
            }
            _ => unsupported("this assignment target", span),
        }
    }

    fn if_stmt(&mut self, i: &ast::If, span: Span) -> Result<St> {
        let cond = self.expr(i.condition())?;
        let cond = self.truthy(cond);
        let then = self.block(i.block())?;
        // `elseif` chains nest as else-ifs.
        let mut els: Option<Vec<St>> = i.else_block().map(|b| self.block(b)).transpose()?;
        if let Some(elseifs) = i.else_if() {
            for e in elseifs.iter().rev() {
                let c = self.expr(e.condition())?;
                let c = self.truthy(c);
                let body = self.block(e.block())?;
                els = Some(vec![if_(c, body, els, span_of(e))]);
            }
        }
        Ok(if_(cond, then, els, span))
    }

    /// A loop body. A label ending it is where a `goto` continues to,
    /// and lands ahead of whatever the loop does after the body.
    fn loop_body(&mut self, block: &Block) -> Result<Vec<St>> {
        self.loop_depths.push(self.depth + 1);
        let body = self.block(block);
        self.loop_depths.pop();
        body
    }

    fn numeric_for(&mut self, f: &ast::NumericFor, span: Span, out: &mut Vec<St>) -> Result<()> {
        let v = self.scopes().declared(f.index_variable());
        let loop_ty = self.var_ty(v);
        let start = self.expr(f.start())?;
        let limit = self.expr(f.end())?;
        let step = f.step().map(|s| self.expr(s)).transpose()?;
        let is_float = loop_ty == Ty::Float;
        let num_ty = if is_float { Ty::Float } else { Ty::Int };
        let ir = self.ir(num_ty);
        let bool_t = prim(PrimitiveType::Bool);
        // A limit or start of another type is converted as Lua would:
        // a float limit of an integer loop is floored.
        let start_node = match start.ty {
            t if t == num_ty => start.node,
            Ty::Int | Ty::Float => self.coerce(start, num_ty),
            _ => {
                let b = self.boxed(start);
                let node = if is_float {
                    call(
                        "zl_for_float",
                        vec![b, str_lit("initial value", span)],
                        ir.clone(),
                        span,
                    )
                } else {
                    call(
                        "zl_for_int",
                        vec![b, str_lit("initial value", span)],
                        ir.clone(),
                        span,
                    )
                };
                self.guard(Val { node, ty: num_ty }).node
            }
        };
        let limit_literal = match &limit.node.node {
            TypedExpression::Literal(TypedLiteral::Integer(n)) if limit.ty == Ty::Int => {
                Some(*n as i64)
            }
            _ => None,
        };
        let limit_node = match limit.ty {
            t if t == num_ty => limit.node,
            Ty::Int => self.coerce(limit, num_ty),
            Ty::Float if !is_float => cast(
                call("floor", vec![limit.node], prim(PrimitiveType::F64), span),
                ir.clone(),
                span,
            ),
            _ => {
                let b = self.boxed(limit);
                let node = if is_float {
                    call(
                        "zl_for_float",
                        vec![b, str_lit("limit", span)],
                        ir.clone(),
                        span,
                    )
                } else {
                    call("zl_for_limit", vec![b], ir.clone(), span)
                };
                self.guard(Val { node, ty: num_ty }).node
            }
        };
        // The step's sign, when it is a literal, picks the test.
        let (step_node, step_literal): (Node, Option<f64>) = match step {
            None => (
                if is_float {
                    float_lit(1.0, span)
                } else {
                    int_lit(1, span)
                },
                Some(1.0),
            ),
            Some(s) => {
                let literal = match &s.node.node {
                    TypedExpression::Literal(TypedLiteral::Integer(n)) => Some(*n as f64),
                    TypedExpression::Literal(TypedLiteral::Float(x)) => Some(*x),
                    _ => None,
                };
                let n = match s.ty {
                    t if t == num_ty => s.node,
                    Ty::Int | Ty::Float => self.coerce(s, num_ty),
                    _ => {
                        let b = self.boxed(s);
                        let node = if is_float {
                            call(
                                "zl_for_float",
                                vec![b, str_lit("step", span)],
                                ir.clone(),
                                span,
                            )
                        } else {
                            call(
                                "zl_for_int",
                                vec![b, str_lit("step", span)],
                                ir.clone(),
                                span,
                            )
                        };
                        self.guard(Val { node, ty: num_ty }).node
                    }
                };
                (n, literal)
            }
        };
        let counter = self.temp();
        let lim = self.temp();
        let st = self.temp();
        out.push(let_(counter, ir.clone(), start_node, span));
        out.push(let_(lim, ir.clone(), limit_node, span));
        out.push(let_(st, ir.clone(), step_node, span));
        let zero = if is_float {
            float_lit(0.0, span)
        } else {
            int_lit(0, span)
        };
        let counter_v = || var(counter, ir.clone(), span);
        let lim_v = || var(lim, ir.clone(), span);
        let st_v = || var(st, ir.clone(), span);
        if step_literal.is_none() {
            out.push(if_(
                binary(BinaryOp::Eq, st_v(), zero.clone(), bool_t.clone(), span),
                vec![expr_stmt(call(
                    "zb_fatal",
                    vec![str_lit("error", span), str_lit("'for' step is zero", span)],
                    prim(PrimitiveType::Unit),
                    span,
                ))],
                None,
                span,
            ));
        }
        let cond = match step_literal {
            Some(x) if x > 0.0 => binary(BinaryOp::Le, counter_v(), lim_v(), bool_t.clone(), span),
            Some(_) => binary(BinaryOp::Ge, counter_v(), lim_v(), bool_t.clone(), span),
            None => binary(
                BinaryOp::Or,
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Gt, st_v(), zero.clone(), bool_t.clone(), span),
                    binary(BinaryOp::Le, counter_v(), lim_v(), bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Lt, st_v(), zero, bool_t.clone(), span),
                    binary(BinaryOp::Ge, counter_v(), lim_v(), bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            ),
        };
        // An integer counter must not run past the limit into a wrap:
        // the loop leaves at the limit itself, unless the limit is a
        // literal the step cannot carry past the range.
        let bounded = !is_float
            && !matches!(
                (limit_literal, step_literal),
                (Some(l), Some(s)) if (s as i64).checked_add(l).is_some() && l.checked_add(s as i64).is_some()
                    && (s as i64) != i64::MIN
            );
        // The body's variable is a copy of the counter each iteration.
        let mut body = vec![self.declare_var(
            v,
            Val {
                node: counter_v(),
                ty: num_ty,
            },
            span,
        )];
        let inner = self.loop_body(f.block())?;
        let mut after = Vec::new();
        if bounded {
            after.push(if_(
                binary(BinaryOp::Eq, counter_v(), lim_v(), bool_t.clone(), span),
                vec![stmt(TypedStatement::Break(None), span)],
                None,
                span,
            ));
        }
        after.push(assign(
            counter_v(),
            binary(BinaryOp::Add, counter_v(), st_v(), ir.clone(), span),
            span,
        ));
        body.extend(inner);
        body.extend(after);
        out.push(while_(cond, body, span));
        Ok(())
    }

    fn generic_for(&mut self, f: &ast::GenericFor, span: Span, out: &mut Vec<St>) -> Result<()> {
        let names: Vec<VarId> = f
            .names()
            .iter()
            .map(|n| self.scopes().declared(n))
            .collect();
        let exprs: Vec<&Expression> = f.expressions().iter().collect();
        // `pairs(t)`, `ipairs(t)` and `next, t` walk the table itself.
        if let [Expression::FunctionCall(c)] = exprs.as_slice() {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            if let Some(b) = self.typer().builtin_callee(c.prefix(), &suffixes)
                && b.lib.is_empty()
                && (b.name == "pairs" || b.name == "ipairs")
            {
                let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.last() else {
                    unreachable!("a builtin callee ends in a call")
                };
                let arg_exprs = self.args_exprs(args);
                if let (1, None) = (arg_exprs.len(), self.literal_arg(args, span)?) {
                    let t = self.expr(arg_exprs[0])?;
                    return if b.name == "ipairs" {
                        self.ipairs_loop(&names, t, f.block(), span, out)
                    } else {
                        self.pairs_loop(&names, t, f.block(), span, out)
                    };
                }
            }
        }
        if let [Expression::Var(Var::Name(token)), table] = exprs.as_slice()
            && let Some(Binding::Global(name)) = self.scopes().binding(token)
            && name == "next"
            && types::builtin_named(self.scopes(), "next").is_some()
        {
            let t = self.expr(table)?;
            return self.pairs_loop(&names, t, f.block(), span, out);
        }
        // The general protocol: `f, s, control`, then `f(s, control)`
        // until its first value is nil.
        let (pre, vals) = self.adjusted(&exprs, 3, span)?;
        out.extend(pre);
        let mut it = vals.into_iter();
        let (fv, sv, cv) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        let fname = self.temp();
        let sname = self.temp();
        let cname = self.temp();
        let fb = self.boxed(fv);
        let sb = self.boxed(sv);
        let cb = self.boxed(cv);
        out.push(let_(fname, Type::Any, fb, span));
        out.push(let_(sname, Type::Any, sb, span));
        out.push(let_(cname, Type::Any, cb, span));
        let vals_name = self.temp();
        let step = call(
            "zl_call_2",
            vec![
                var(fname, Type::Any, span),
                var(sname, Type::Any, span),
                var(cname, Type::Any, span),
            ],
            Type::Any,
            span,
        );
        let step = self
            .guard(Val {
                node: step,
                ty: Ty::Any,
            })
            .node;
        let mut body = vec![let_(
            vals_name,
            self.m.anys(),
            call("zl_values", vec![step], self.m.anys(), span),
            span,
        )];
        let first = call(
            "zl_value_at",
            vec![var(vals_name, self.m.anys(), span), int_lit(1, span)],
            Type::Any,
            span,
        );
        body.push(assign(var(cname, Type::Any, span), first, span));
        body.push(if_(
            binary(
                BinaryOp::Eq,
                var(cname, Type::Any, span),
                nil(span),
                prim(PrimitiveType::Bool),
                span,
            ),
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        for (i, v) in names.iter().enumerate() {
            let value = if i == 0 {
                var(cname, Type::Any, span)
            } else {
                call(
                    "zl_value_at",
                    vec![
                        var(vals_name, self.m.anys(), span),
                        int_lit(i as i64 + 1, span),
                    ],
                    Type::Any,
                    span,
                )
            };
            body.push(self.declare_var(
                *v,
                Val {
                    node: value,
                    ty: Ty::Any,
                },
                span,
            ));
        }
        body.extend(self.loop_body(f.block())?);
        out.push(while_(bool_lit(true, span), body, span));
        Ok(())
    }

    /// `for i, v in ipairs(t)`: `t[1], t[2], …` until nil.
    fn ipairs_loop(
        &mut self,
        names: &[VarId],
        t: Val,
        block: &Block,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let i64_t = prim(PrimitiveType::I64);
        let tname = self.temp();
        let counter = self.temp();
        let value = self.temp();
        let t_ir = self.ir(t.ty);
        let t_ty = t.ty;
        out.push(let_(tname, t_ir.clone(), t.node, span));
        out.push(let_(counter, i64_t.clone(), int_lit(1, span), span));
        let read = self.index_read(
            Val {
                node: var(tname, t_ir, span),
                ty: t_ty,
            },
            Val {
                node: var(counter, i64_t.clone(), span),
                ty: Ty::Int,
            },
            span,
        );
        let mut body = vec![let_(value, Type::Any, read.node, span)];
        body.push(if_(
            binary(
                BinaryOp::Eq,
                var(value, Type::Any, span),
                nil(span),
                prim(PrimitiveType::Bool),
                span,
            ),
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        if let Some(k) = names.first() {
            body.push(self.declare_var(
                *k,
                Val {
                    node: var(counter, i64_t.clone(), span),
                    ty: Ty::Int,
                },
                span,
            ));
        }
        if let Some(v) = names.get(1) {
            body.push(self.declare_var(
                *v,
                Val {
                    node: var(value, Type::Any, span),
                    ty: Ty::Any,
                },
                span,
            ));
        }
        for v in names.iter().skip(2) {
            let n = self.nil_val(span);
            body.push(self.declare_var(*v, n, span));
        }
        let inner = self.loop_body(block)?;
        let increment = assign(
            var(counter, i64_t.clone(), span),
            binary(
                BinaryOp::Add,
                var(counter, i64_t.clone(), span),
                int_lit(1, span),
                i64_t,
                span,
            ),
            span,
        );
        self.push_loop_with_increment(body, inner, increment, span, out);
        Ok(())
    }

    /// `for k, v in pairs(t)`: every position holding a value.
    fn pairs_loop(
        &mut self,
        names: &[VarId],
        t: Val,
        block: &Block,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let i64_t = prim(PrimitiveType::I64);
        let table_t = self.ir(Ty::Table);
        let tname = self.temp();
        let pos = self.temp();
        let t = if t.ty == Ty::Table {
            t.node
        } else {
            let b = self.boxed(t);
            let node = call(
                "zl_as_table",
                vec![b, str_lit("for iterator", span)],
                table_t.clone(),
                span,
            );
            self.guard(Val {
                node,
                ty: Ty::Table,
            })
            .node
        };
        out.push(let_(tname, table_t.clone(), t, span));
        out.push(let_(
            pos,
            i64_t.clone(),
            call(
                "zl_next_pos",
                vec![var(tname, table_t.clone(), span), int_lit(0, span)],
                i64_t.clone(),
                span,
            ),
            span,
        ));
        let mut body = Vec::new();
        if let Some(k) = names.first() {
            let key = call(
                "zl_pos_key",
                vec![
                    var(tname, table_t.clone(), span),
                    var(pos, i64_t.clone(), span),
                ],
                Type::Any,
                span,
            );
            body.push(self.declare_var(
                *k,
                Val {
                    node: key,
                    ty: Ty::Any,
                },
                span,
            ));
        }
        if let Some(v) = names.get(1) {
            let value = call(
                "zl_pos_value",
                vec![
                    var(tname, table_t.clone(), span),
                    var(pos, i64_t.clone(), span),
                ],
                Type::Any,
                span,
            );
            body.push(self.declare_var(
                *v,
                Val {
                    node: value,
                    ty: Ty::Any,
                },
                span,
            ));
        }
        for v in names.iter().skip(2) {
            let n = self.nil_val(span);
            body.push(self.declare_var(*v, n, span));
        }
        let inner = self.loop_body(block)?;
        let advance = assign(
            var(pos, i64_t.clone(), span),
            call(
                "zl_next_pos",
                vec![
                    var(tname, table_t, span),
                    binary(
                        BinaryOp::Add,
                        var(pos, i64_t.clone(), span),
                        int_lit(1, span),
                        i64_t.clone(),
                        span,
                    ),
                ],
                i64_t.clone(),
                span,
            ),
            span,
        );
        let cond = binary(
            BinaryOp::Ge,
            var(pos, i64_t, span),
            int_lit(0, span),
            prim(PrimitiveType::Bool),
            span,
        );
        let mut whole = body;
        whole.extend(inner);
        whole.push(advance);
        out.push(while_(cond, whole, span));
        Ok(())
    }

    /// A `while true` loop whose body ends in an increment; a label
    /// ending the body lands ahead of it.
    fn push_loop_with_increment(
        &mut self,
        head: Vec<St>,
        inner: Vec<St>,
        increment: St,
        span: Span,
        out: &mut Vec<St>,
    ) {
        let mut body = head;
        body.extend(inner);
        body.push(increment);
        out.push(while_(bool_lit(true, span), body, span));
    }

    // ─── functions ──────────────────────────────────────────────

    /// Lower a function's body into its typed entry, and its record
    /// code if it is used as a value. Nested functions take their
    /// record as `env`.
    fn lower_function(
        &mut self,
        id: FuncId,
        body: &ast::FunctionBody,
        _is_method: bool,
    ) -> Result<()> {
        let span = span_of(body);
        let info = self.scopes().func(id).clone();
        let sig = self.m.sig(id);
        let has_env = !info.top_level;
        let mut child = Lowerer::new(self.m, id);
        let mut params = Vec::new();
        let mut statements = Vec::new();
        if has_env {
            params.push(parameter(intern("env"), self.m.anys(), span));
            statements.extend(child.env_prologue(span));
        }
        for (i, v) in info.params.iter().enumerate() {
            let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
            let local_ty = child.var_ty(*v);
            let vinfo = self.scopes().var(*v).clone();
            let symbol = child.local_symbol(*v);
            let arg = intern(&format!("{}$arg", vinfo.name));
            params.push(parameter(arg, child.ir(param_ty), span));
            // The parameter as the body's variable: its own type, or a cell.
            let value = Val {
                node: var(arg, child.ir(param_ty), span),
                ty: param_ty,
            };
            if vinfo.needs_cell() {
                let cell_symbol = intern(&format!("{}$cell{}", vinfo.name, v.0));
                child
                    .storage
                    .insert(*v, Storage::Cell(cell_symbol, local_ty));
                statements.push(child.declare_var(*v, value, span));
            } else {
                child.storage.insert(*v, Storage::Local(symbol, local_ty));
                let value = child.coerce(value, local_ty);
                statements.push(let_(symbol, child.ir(local_ty), value, span));
            }
        }
        if info.is_vararg {
            let name = intern("$varargs");
            params.push(parameter(name, self.m.anys(), span));
            child.varargs = Some(name);
        }
        let body_statements = child.block(body.block())?;
        if child.entry_line {
            statements.push(entry_line_save(span));
        }
        statements.extend(body_statements);
        // Falling off the end returns nothing.
        if types::falls_through(body.block()) {
            child.return_stmt(&[], span, &mut statements)?;
        }
        let entry = self.m.entry_name(id);
        let function = typed_function(
            &entry,
            params,
            self.m.return_ir(&sig.returns),
            statements,
            span,
        );
        self.m.facts.borrow_mut().insert(
            id,
            RaiseFact {
                own: child.raised,
                callees: child.raise_callees.clone(),
            },
        );
        self.m.functions.borrow_mut().push(function);
        // The record code, whether or not anything takes the function
        // as a value: unreached, it costs nothing.
        self.record_code(id, &sig, span);
        Ok(())
    }

    /// The record code of a function: the shape every function value
    /// has, calling the typed entry.
    fn record_code(&mut self, id: FuncId, sig: &types::Sig, span: Span) {
        let info = self.scopes().func(id).clone();
        let mut params = vec![parameter(intern("env"), self.m.anys(), span)];
        let mut args: Vec<Node> = Vec::new();
        if !info.top_level {
            args.push(var(intern("env"), self.m.anys(), span));
        }
        let mut statements = Vec::new();
        let n = info.params.len();
        let mut child = Lowerer::new(self.m, id);
        if info.is_vararg {
            // Variadic: one packed list, split into the parameters and
            // the rest.
            params.push(parameter(intern("packed"), Type::Any, span));
            let all = intern("$all");
            statements.push(let_(
                all,
                self.m.anys(),
                call(
                    "zl_values",
                    vec![var(intern("packed"), Type::Any, span)],
                    self.m.anys(),
                    span,
                ),
                span,
            ));
            for i in 0..n {
                let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
                let value = call(
                    "zl_value_at",
                    vec![var(all, self.m.anys(), span), int_lit(i as i64 + 1, span)],
                    Type::Any,
                    span,
                );
                args.push(child.coerce(
                    Val {
                        node: value,
                        ty: Ty::Any,
                    },
                    param_ty,
                ));
            }
            args.push(call(
                "zl_slice",
                vec![
                    var(all, self.m.anys(), span),
                    int_lit(n as i64, span),
                    list_len(var(all, self.m.anys(), span), span),
                ],
                self.m.anys(),
                span,
            ));
        } else {
            for i in 0..n {
                let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
                let arg = intern(&format!("a{i}"));
                params.push(parameter(arg, Type::Any, span));
                args.push(child.coerce(
                    Val {
                        node: var(arg, Type::Any, span),
                        ty: Ty::Any,
                    },
                    param_ty,
                ));
            }
        }
        let value = call(
            &self.m.entry_name(id),
            args,
            self.m.return_ir(&sig.returns),
            span,
        );
        // The result as one dynamic value.
        let result = match &sig.returns {
            Returns::Fixed(types) if types.is_empty() => block_value(
                vec![expr_stmt(value)],
                call("zl_none", vec![], Type::Any, span),
                span,
            ),
            Returns::Fixed(types) if types.len() == 1 => child.coerce(
                Val {
                    node: value,
                    ty: types[0].settled(),
                },
                Ty::Any,
            ),
            Returns::Fixed(_) => call("zb_box_tuple", vec![value], Type::Any, span),
            Returns::Dynamic => value,
        };
        statements.push(ret(Some(result), span));
        let function = typed_function(&self.m.code_name(id), params, Type::Any, statements, span);
        self.m.functions.borrow_mut().push(function);
    }
}

/// The chunk's statements as a function; the entry names the chunk,
/// runs it, and reports an error nothing caught.
const CHUNK_FN: &str = "lua$chunk";

/// A label's name in the typed program, from its number.
fn label_name(id: u32) -> InternedString {
    intern(&format!("$label{id}"))
}

/// Set by a `return` at the chunk's outermost level when the chunk
/// runs as segments, so the driver stops.
const RETURNED: &str = "lua$returned";

/// The line a function was entered at: the caller's, for `error(v, 2)`.
const ENTRY_LINE: &str = "$entry_line";

fn entry_line_save(span: Span) -> St {
    let i64_t = prim(PrimitiveType::I64);
    let_(
        intern(ENTRY_LINE),
        i64_t.clone(),
        var(intern(library::LINE), i64_t, span),
        span,
    )
}

/// The chunk name as positions spell it (`luaO_chunkid`): a file name
/// longer than the reference's buffer keeps its tail after `...`.
fn chunk_id(file: &str) -> String {
    const IDSIZE: usize = 60;
    const KEPT: usize = IDSIZE - "...".len() - 1;
    if file.len() < IDSIZE {
        return file.to_string();
    }
    let tail = file.len() - KEPT;
    let mut cut = tail;
    while !file.is_char_boundary(cut) {
        cut += 1;
    }
    format!("...{}", &file[cut..])
}

/// A file the program requires, parsed and resolved on its own.
struct Loaded {
    name: String,
    file: String,
    source: String,
    ast: ast::Ast,
    scopes: Scopes,
}

/// The files `require`d by name, transitively, each parsed once. A
/// name that is a standard library or has no file is left to
/// `require` at run time.
fn load_required(first: &[String], main_file: &str) -> Result<Vec<Loaded>> {
    let dir = std::path::Path::new(main_file)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let mut loaded: Vec<Loaded> = Vec::new();
    let mut queue: Vec<String> = first.to_vec();
    let mut seen: HashSet<String> = HashSet::new();
    while let Some(name) = queue.pop() {
        if !seen.insert(name.clone())
            || crate::library::stdlib::LIBS.contains(&name.as_str())
            || name == "_G"
        {
            continue;
        }
        let relative = format!("{}.lua", name.replace('.', "/"));
        let path = dir.join(&relative);
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let source = crate::source_text(&bytes).into_owned();
        let file = path.display().to_string();
        let ast = match full_moon::parse_fallible(&source, full_moon::LuaVersion::lua54())
            .into_result()
        {
            Ok(ast) => ast,
            Err(errors) => {
                let first = errors.into_iter().next().expect("an error");
                let message = match &first {
                    full_moon::Error::AstError(e) => e.error_message().to_string(),
                    full_moon::Error::TokenizerError(e) => e.error().to_string(),
                };
                return Err(Error::Library(format!("{file}: {message}")));
            }
        };
        let scopes = crate::scope::resolve(&ast);
        queue.extend(scopes.requires.iter().cloned());
        loaded.push(Loaded {
            name,
            file,
            source,
            ast,
            scopes,
        });
    }
    Ok(loaded)
}

fn line_starts_of(source: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(source.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

/// The whole chunk as a program.
pub(crate) fn program(
    ast: &ast::Ast,
    source: &str,
    file: &str,
    library: Library,
) -> Result<TypedProgram> {
    let started = std::time::Instant::now();
    let mut scopes = crate::scope::resolve(ast);
    let mut loaded = load_required(&scopes.requires, file)?;
    // Files share their globals through the table.
    if !loaded.is_empty() {
        scopes.dynamic_globals = true;
        for m in &mut loaded {
            m.scopes.dynamic_globals = true;
        }
    }
    crate::trace_phase("scopes", started);
    let started = std::time::Instant::now();
    let inferred = types::infer(&scopes, ast);
    let module_inferred: Vec<Inferred> = loaded
        .iter()
        .map(|m| types::infer(&m.scopes, &m.ast))
        .collect();
    crate::trace_phase("infer", started);
    let started = std::time::Instant::now();
    let line_starts = line_starts_of(source);
    let mut module = Module {
        scopes: &scopes,
        inferred: &inferred,
        types: library.types.clone(),
        chunk: file,
        tag: String::new(),
        chunk_index: 0,
        line_starts,
        fallible: crate::fallible::FALLIBLE.iter().copied().collect(),
        raising: None,
        functions: RefCell::new(Vec::new()),
        module_vars: RefCell::new(Vec::new()),
        facts: RefCell::new(HashMap::new()),
    };
    let span = Span::new(0, source.len());
    // Lowered twice: the first time finds which functions may raise, so
    // the second checks after calls to those alone.
    let lower_chunk = |module: &Module<'_>| -> Result<Vec<St>> {
        let mut main = Lowerer::new(module, CHUNK);
        main.returns = Returns::Fixed(Vec::new());
        let statements = if scopes.split_chunk {
            // Each segment is a function; the chunk calls them in turn
            // and stops at an error or a `return`.
            let segments = main.segments(ast.nodes())?;
            let unit = prim(PrimitiveType::Unit);
            let mut driver = Vec::new();
            let returned = module
                .module_vars
                .borrow()
                .iter()
                .any(|(n, _)| *n == intern(RETURNED));
            for (k, mut body) in segments.into_iter().enumerate() {
                if main.entry_line {
                    body.insert(0, entry_line_save(span));
                }
                body.push(ret(None, span));
                let name = format!("{CHUNK_FN}${k}");
                module.functions.borrow_mut().push(typed_function(
                    &name,
                    Vec::new(),
                    unit.clone(),
                    body,
                    span,
                ));
                driver.push(expr_stmt(call(&name, vec![], unit.clone(), span)));
                driver.push(if_(
                    binary(
                        BinaryOp::Ne,
                        Lowerer::pending(span),
                        nil(span),
                        prim(PrimitiveType::Bool),
                        span,
                    ),
                    vec![ret(None, span)],
                    None,
                    span,
                ));
                if returned {
                    driver.push(if_(
                        var(intern(RETURNED), prim(PrimitiveType::Bool), span),
                        vec![ret(None, span)],
                        None,
                        span,
                    ));
                }
            }
            driver.push(ret(None, span));
            driver
        } else {
            let mut statements = main.block(ast.nodes())?;
            if main.entry_line {
                statements.insert(0, entry_line_save(span));
            }
            if types::falls_through(ast.nodes()) {
                statements.push(ret(None, span));
            }
            statements
        };
        module.facts.borrow_mut().insert(
            CHUNK,
            RaiseFact {
                own: main.raised,
                callees: main.raise_callees.clone(),
            },
        );
        Ok(statements)
    };
    lower_chunk(&module)?;
    let raising = raising_functions(&module.facts.borrow());
    module.raising = Some(raising);
    module.functions.borrow_mut().clear();
    module.module_vars.borrow_mut().clear();
    module.facts.borrow_mut().clear();
    crate::trace_phase("lower 1", started);
    let started = std::time::Instant::now();
    let statements = lower_chunk(&module)?;
    crate::trace_phase("lower 2", started);
    let chunk_fn = typed_function(
        CHUNK_FN,
        Vec::new(),
        prim(PrimitiveType::Unit),
        statements,
        span,
    );
    let mut declarations = Vec::new();
    let declare = |module: &Module<'_>, declarations: &mut Vec<TypedNode<TypedDeclaration>>| {
        for (name, ty) in module.module_vars.borrow().iter() {
            declarations.push(TypedNode::new(
                TypedDeclaration::Variable(TypedVariable {
                    name: *name,
                    ty: module.ir(*ty),
                    mutability: Mutability::Mutable,
                    initializer: None,
                    visibility: Visibility::Public,
                }),
                Type::Unknown,
                Span::new(0, 0),
            ));
        }
        for f in module.functions.borrow().iter() {
            let span = f.body.as_ref().map(|b| b.span).unwrap_or(Span::new(0, 0));
            declarations.push(TypedNode::new(
                TypedDeclaration::Function(f.clone()),
                Type::Unknown,
                span,
            ));
        }
    };
    declare(&module, &mut declarations);

    // Each required file is a chunk of its own, a function the program
    // enters through `package.preload`, and named in positions by its
    // number.
    let mut preloads: Vec<St> = Vec::new();
    for (k, m) in loaded.iter().enumerate() {
        let tag = format!("m${}$", m.name.replace('.', "$"));
        let mut file_module = Module {
            scopes: &m.scopes,
            inferred: &module_inferred[k],
            types: library.types.clone(),
            chunk: &m.file,
            tag: tag.clone(),
            chunk_index: k as i64 + 1,
            line_starts: line_starts_of(&m.source),
            fallible: crate::fallible::FALLIBLE.iter().copied().collect(),
            raising: None,
            functions: RefCell::new(Vec::new()),
            module_vars: RefCell::new(Vec::new()),
            facts: RefCell::new(HashMap::new()),
        };
        let file_span = Span::new(0, m.source.len());
        let lower_file = |module: &Module<'_>| -> Result<Vec<St>> {
            let mut main = Lowerer::new(module, CHUNK);
            main.returns = Returns::Dynamic;
            main.varargs = Some(intern("$varargs"));
            let mut statements = main.block(m.ast.nodes())?;
            if main.entry_line {
                statements.insert(0, entry_line_save(file_span));
            }
            if types::falls_through(m.ast.nodes()) {
                main.return_stmt(&[], file_span, &mut statements)?;
            }
            module.facts.borrow_mut().insert(
                CHUNK,
                RaiseFact {
                    own: main.raised,
                    callees: main.raise_callees.clone(),
                },
            );
            Ok(statements)
        };
        lower_file(&file_module)?;
        let raising = raising_functions(&file_module.facts.borrow());
        file_module.raising = Some(raising);
        file_module.functions.borrow_mut().clear();
        file_module.module_vars.borrow_mut().clear();
        file_module.facts.borrow_mut().clear();
        let statements = lower_file(&file_module)?;
        let chunk_name = format!("lua${tag}chunk");
        let code_name = format!("{chunk_name}$fn");
        file_module.functions.borrow_mut().push(typed_function(
            &chunk_name,
            vec![parameter(intern("$varargs"), file_module.anys(), file_span)],
            Type::Any,
            statements,
            file_span,
        ));
        // Its record code: a variadic taking the packed arguments.
        file_module.functions.borrow_mut().push(typed_function(
            &code_name,
            vec![
                parameter(intern("env"), file_module.anys(), file_span),
                parameter(intern("packed"), Type::Any, file_span),
            ],
            Type::Any,
            vec![ret(
                Some(call(
                    &chunk_name,
                    vec![call(
                        "zl_values",
                        vec![var(intern("packed"), Type::Any, file_span)],
                        file_module.anys(),
                        file_span,
                    )],
                    Type::Any,
                    file_span,
                )),
                file_span,
            )],
            file_span,
        ));
        declare(&file_module, &mut declarations);
        // Positions name the file as `require` found it.
        let found_as = format!("./{}.lua", m.name.replace('.', "/"));
        preloads.push(expr_stmt(call(
            "zl_chunk_add",
            vec![str_lit(&chunk_id(&found_as), span)],
            prim(PrimitiveType::Unit),
            span,
        )));
        preloads.push(expr_stmt(call(
            "zl_preload_module",
            vec![
                str_lit(&m.name, span),
                call(
                    "zl_func_of",
                    vec![code_of(&code_name, span), int_lit(VARIADIC_ARITY, span)],
                    Type::Any,
                    span,
                ),
            ],
            prim(PrimitiveType::Unit),
            span,
        )));
    }

    let mut entry_body = vec![
        assign(
            var(intern(library::CHUNK), prim(PrimitiveType::String), span),
            str_lit(&chunk_id(module.chunk), span),
            span,
        ),
        if scopes.dynamic_globals {
            assign(
                var(intern(library::GLOBALS), module.ir(Ty::Table), span),
                call("zl_globals_table", vec![], module.ir(Ty::Table), span),
                span,
            )
        } else {
            stmt(
                TypedStatement::Block(TypedBlock {
                    statements: Vec::new(),
                    span,
                }),
                span,
            )
        },
    ];
    entry_body.extend(preloads);
    entry_body.extend([
        expr_stmt(call(CHUNK_FN, vec![], prim(PrimitiveType::Unit), span)),
        expr_stmt(call(
            "zl_report_pending",
            vec![],
            prim(PrimitiveType::Unit),
            span,
        )),
        ret(None, span),
    ]);
    let entry = typed_function(
        ENTRY,
        Vec::new(),
        prim(PrimitiveType::Unit),
        entry_body,
        span,
    );
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(chunk_fn),
        Type::Unknown,
        span,
    ));
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(entry),
        Type::Unknown,
        span,
    ));
    // The library itself arrives by import: its declarations for
    // typing, its HIR to link against.
    declarations.push(TypedNode::new(
        TypedDeclaration::Import(zyntax_typed_ast::typed_ast::TypedImport {
            language: Some(intern("lua")),
            module_path: vec![intern(crate::policy::LIBRARY_MODULE)],
            items: Vec::new(),
            span: Span::new(0, 0),
        }),
        Type::Unknown,
        Span::new(0, 0),
    ));
    let mut source_files = vec![zyntax_typed_ast::source::SourceFile::new(
        file.to_string(),
        source.to_string(),
    )];
    for m in &loaded {
        source_files.push(zyntax_typed_ast::source::SourceFile::new(
            m.file.clone(),
            m.source.clone(),
        ));
    }
    Ok(TypedProgram {
        declarations,
        language: Some(intern("lua")),
        span,
        source_files,
        type_registry: library.type_registry,
    })
}
