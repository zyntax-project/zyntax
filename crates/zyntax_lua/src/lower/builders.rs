//! Builders for Lua lowering.

use super::*;

/// An exactly representable integer literal needs no float-to-int guard.
pub(super) fn mixed_numeric_compare(
    int: Node,
    float: Node,
    how: Compare,
    int_first: bool,
    span: Span,
) -> Node {
    let bool_t = prim(PrimitiveType::Bool);
    let op = match how {
        Compare::Eq => BinaryOp::Eq,
        Compare::Lt => BinaryOp::Lt,
        Compare::Le => BinaryOp::Le,
    };
    if let TypedExpression::Literal(TypedLiteral::Integer(n)) = &int.node
        && (-(1i128 << 53)..=(1i128 << 53)).contains(n)
    {
        let constant = float_lit(*n as f64, span);
        let (left, right) = if int_first {
            (constant, float)
        } else {
            (float, constant)
        };
        return binary(op, left, right, bool_t, span);
    }
    let helper = match (how, int_first) {
        (Compare::Eq, _) => "zl_eq_if",
        (Compare::Lt, true) => "zl_lt_if",
        (Compare::Lt, false) => "zl_lt_fi",
        (Compare::Le, true) => "zl_le_if",
        (Compare::Le, false) => "zl_le_fi",
    };
    let args = if int_first || how == Compare::Eq {
        vec![int, float]
    } else {
        vec![float, int]
    };
    call(helper, args, bool_t, span)
}

pub(super) fn unsupported<T>(what: impl Into<String>, span: Span) -> Result<T> {
    Err(Error::unsupported(what, span))
}

pub(super) fn span_of<N: full_moon::node::Node>(node: &N) -> Span {
    match node.range() {
        Some((a, b)) => Span::new(a.bytes(), b.bytes()),
        None => Span::new(0, 0),
    }
}

pub(super) fn node(x: TypedExpression, ty: Type, span: Span) -> Node {
    TypedNode::new(x, ty, span)
}

pub(super) fn stmt(s: TypedStatement, span: Span) -> St {
    TypedNode::new(s, Type::Unknown, span)
}

pub(super) fn expr_stmt(e: Node) -> St {
    let span = e.span;
    stmt(TypedStatement::Expression(Box::new(e)), span)
}

impl NamedReferences {
    pub(super) fn start() -> Self {
        let previous = NAMED.with(|named| named.replace(Some(HashSet::new())));
        Self {
            previous,
            active: true,
        }
    }

    pub(super) fn finish(mut self) -> HashSet<InternedString> {
        let found = NAMED.with(|named| named.replace(self.previous.take()));
        self.active = false;
        found.unwrap_or_default()
    }
}

pub(super) fn var(name: InternedString, ty: Type, span: Span) -> Node {
    NAMED.with(|named| {
        if let Some(named) = named.borrow_mut().as_mut() {
            named.insert(name);
        }
    });
    node(TypedExpression::Variable(name), ty, span)
}

pub(super) fn int_lit(v: i64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        prim(PrimitiveType::I64),
        span,
    )
}

pub(super) fn int32_lit(v: i32, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        prim(PrimitiveType::I32),
        span,
    )
}

pub(super) fn float_lit(v: f64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Float(v)),
        prim(PrimitiveType::F64),
        span,
    )
}

pub(super) fn bool_lit(v: bool, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Bool(v)),
        prim(PrimitiveType::Bool),
        span,
    )
}

pub(super) fn str_lit(s: &str, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::String(intern(s))),
        prim(PrimitiveType::String),
        span,
    )
}

pub(super) fn null(ty: Type, span: Span) -> Node {
    node(TypedExpression::Literal(TypedLiteral::Null), ty, span)
}

pub(super) fn nil(span: Span) -> Node {
    null(Type::Any, span)
}

pub(super) fn call(name: &str, args: Vec<Node>, ty: Type, span: Span) -> Node {
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

pub(super) fn binary(op: BinaryOp, left: Node, right: Node, ty: Type, span: Span) -> Node {
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

pub(super) fn unary(op: UnaryOp, operand: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Unary(TypedUnary {
            op,
            operand: Box::new(operand),
        }),
        ty,
        span,
    )
}

pub(super) fn if_value(cond: Node, then: Node, els: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::If(TypedIfExpr {
            condition: Box::new(cond),
            then_branch: Box::new(then),
            else_branch: Box::new(els),
        }),
        ty,
        span,
    )
}

pub(super) fn cast(value: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ty.clone(),
        }),
        ty,
        span,
    )
}

pub(super) fn index(xs: Node, i: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(xs),
            index: Box::new(i),
        }),
        ty,
        span,
    )
}

/// `obj.field`, a struct field read.
pub(super) fn field(obj: Node, name: &str, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Field(TypedFieldAccess {
            object: Box::new(obj),
            field: intern(name),
        }),
        ty,
        span,
    )
}

/// `t` as a pointer to the struct of a shape's tables.
pub(super) fn as_shape(t: Node, shape_ty: Type, span: Span) -> Node {
    cast(cast(t, prim(PrimitiveType::I64), span), shape_ty, span)
}

pub(super) fn assign(target: Node, value: Node, span: Span) -> St {
    let ty = target.ty.clone();
    expr_stmt(binary(BinaryOp::Assign, target, value, ty, span))
}

pub(super) fn let_(name: InternedString, ty: Type, value: Node, span: Span) -> St {
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

pub(super) fn if_(cond: Node, then: Vec<St>, els: Option<Vec<St>>, span: Span) -> St {
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

pub(super) fn while_(cond: Node, body: Vec<St>, span: Span) -> St {
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

pub(super) fn ret(value: Option<Node>, span: Span) -> St {
    stmt(TypedStatement::Return(value.map(Box::new)), span)
}

/// Statements followed by a value, as one expression.
/// A scalar whose kind is decided at run time: a tag, the integer it
/// holds and the float it holds. A value struct, so it lives in
/// registers; the field the tag does not select is unspecified, so
/// every reader selects by the tag. A number is one whose tag is never
/// nil or boolean.
pub(super) fn number_type() -> Type {
    Type::Tuple(vec![
        prim(PrimitiveType::I64),
        prim(PrimitiveType::I64),
        prim(PrimitiveType::F64),
    ])
}

/// Whether a number of run-time kind has its own lowering of `op`.
pub(super) fn takes_number_path(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Plus(_)
            | BinOp::Minus(_)
            | BinOp::Star(_)
            | BinOp::Slash(_)
            | BinOp::Caret(_)
            | BinOp::DoubleSlash(_)
            | BinOp::Percent(_)
            | BinOp::TwoEqual(_)
            | BinOp::TildeEqual(_)
            | BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_)
    )
}

/// Whether arithmetic or ordering on `a` and `b` takes the numbers'
/// own operation once a nil operand has raised: each is a number of
/// known kind or such a number or nil, and one may be nil.
pub(super) fn or_nil_operands(a: Ty, b: Ty) -> bool {
    let known = |t: Ty| matches!(t.underlying(), Ty::Int | Ty::Float);
    (matches!(a, Ty::IntOrNil | Ty::FloatOrNil) || matches!(b, Ty::IntOrNil | Ty::FloatOrNil))
        && known(a)
        && known(b)
}

/// Whether `a == b` is decided by tags and one number part: one side
/// is an integer or nil and the other an integer, nil or the same; or
/// the like for floats.
pub(super) fn or_nil_comparable(a: Ty, b: Ty) -> bool {
    let fits = |x: Ty, y: Ty| match x {
        Ty::IntOrNil => matches!(y, Ty::IntOrNil | Ty::Int | Ty::Nil),
        Ty::FloatOrNil => matches!(y, Ty::FloatOrNil | Ty::Float | Ty::Nil),
        _ => false,
    };
    fits(a, b) || fits(b, a)
}

pub(super) fn number_value(tag: Node, int: Node, float: Node, span: Span) -> Node {
    node(
        TypedExpression::Tuple(vec![tag, int, float]),
        number_type(),
        span,
    )
}

/// The tuple of integer `n`, which is read twice: a variable or a
/// literal. An integer's float word holds its value as a double, so
/// the float of any number is its float word.
pub(super) fn number_of_int(n: Node, span: Span) -> Node {
    let float = match &n.node {
        TypedExpression::Literal(TypedLiteral::Integer(k)) => float_lit(*k as i64 as f64, span),
        _ => cast(n.clone(), prim(PrimitiveType::F64), span),
    };
    number_value(int_lit(TAG_INT, span), n, float, span)
}

/// The float word of an integer-or-float result: the integer's value
/// as a double when it is the integer, else `float`.
pub(super) fn float_word(is_int: Node, int: Node, float: Node, span: Span) -> Node {
    if_value(
        is_int,
        cast(int, prim(PrimitiveType::F64), span),
        float,
        prim(PrimitiveType::F64),
        span,
    )
}

pub(super) fn number_of_float(f: Node, span: Span) -> Node {
    number_value(int_lit(TAG_FLOAT, span), int_lit(0, span), f, span)
}

pub(super) fn scalar_of_nil(span: Span) -> Node {
    number_value(
        int_lit(TAG_NIL, span),
        int_lit(0, span),
        float_lit(0.0, span),
        span,
    )
}

pub(super) fn scalar_of_bool(b: Node, span: Span) -> Node {
    let tag = if_value(
        b,
        int_lit(TAG_TRUE, span),
        int_lit(TAG_FALSE, span),
        prim(PrimitiveType::I64),
        span,
    );
    number_value(tag, int_lit(0, span), float_lit(0.0, span), span)
}

/// The tag an integer-or-float result carries, from whether it is
/// the integer.
pub(super) fn tag_of_is_int(is_int: Node, span: Span) -> Node {
    if_value(
        is_int,
        int_lit(TAG_INT, span),
        int_lit(TAG_FLOAT, span),
        prim(PrimitiveType::I64),
        span,
    )
}

impl NumberParts {
    pub(super) fn of(n: &Node, span: Span) -> NumberParts {
        let part = |i: i64, ty: PrimitiveType| index(n.clone(), int_lit(i, span), prim(ty), span);
        NumberParts {
            tag: part(0, PrimitiveType::I64),
            int: part(1, PrimitiveType::I64),
            float: part(2, PrimitiveType::F64),
        }
    }

    pub(super) fn has_tag(&self, tag: i64, span: Span) -> Node {
        binary(
            BinaryOp::Eq,
            self.tag.clone(),
            int_lit(tag, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    pub(super) fn is_int(&self, span: Span) -> Node {
        self.has_tag(TAG_INT, span)
    }

    /// Whether the value is a number at all.
    pub(super) fn is_numeric(&self, span: Span) -> Node {
        binary(
            BinaryOp::Ge,
            self.tag.clone(),
            int_lit(TAG_INT, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// Whether the value is true: anything but nil and false.
    pub(super) fn is_true(&self, span: Span) -> Node {
        binary(
            BinaryOp::Ge,
            self.tag.clone(),
            int_lit(TAG_TRUE, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// Whether both values are numbers, without a short-circuit: the
    /// tags are small, so their product says.
    pub(super) fn both_numeric(&self, other: &NumberParts, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        binary(
            BinaryOp::Ge,
            binary(
                BinaryOp::Mul,
                self.tag.clone(),
                other.tag.clone(),
                i64_t,
                span,
            ),
            int_lit(TAG_INT * TAG_INT, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// Whether two numbers are both integers, without a short-circuit:
    /// the tags of numbers are `TAG_INT` and `TAG_FLOAT`, and only two
    /// `TAG_INT`s or together to `TAG_INT`.
    pub(super) fn both_int(&self, other: &NumberParts, span: Span) -> Node {
        binary(
            BinaryOp::Eq,
            binary(
                BinaryOp::BitOr,
                self.tag.clone(),
                other.tag.clone(),
                prim(PrimitiveType::I64),
                span,
            ),
            int_lit(TAG_INT, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// The value as a float, whichever number it holds: an integer's
    /// float word is its value.
    pub(super) fn as_float(&self, _span: Span) -> Node {
        self.float.clone()
    }
}

pub(super) fn block_value(statements: Vec<St>, value: Node, span: Span) -> Node {
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

/// `x % n` for an integer `x` and a positive power of two `n`: the low
/// bits, which are the floored modulo whatever `x`'s sign.
pub(super) fn pow2_mod(x: Node, n: i64, span: Span) -> Node {
    let i64_t = prim(PrimitiveType::I64);
    binary(BinaryOp::BitAnd, x, int_lit(n - 1, span), i64_t, span)
}

/// `x // n` for an integer `x` and a positive power of two `n`: an
/// arithmetic shift, which rounds toward negative infinity.
pub(super) fn pow2_floordiv(x: Node, n: i64, span: Span) -> Node {
    let i64_t = prim(PrimitiveType::I64);
    let shift = int_lit(n.trailing_zeros() as i64, span);
    binary(BinaryOp::Shr, x, shift, i64_t, span)
}

/// The length of a list.
pub(super) fn list_len(xs: Node, span: Span) -> Node {
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
pub(super) fn code_of(name: &str, span: Span) -> Node {
    var(intern(name), prim(PrimitiveType::USize), span)
}

pub(super) fn parameter(name: InternedString, ty: Type, span: Span) -> TypedParameter {
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

pub(super) fn typed_function(
    name: &str,
    params: Vec<TypedParameter>,
    return_type: Type,
    statements: Vec<St>,
    span: Span,
) -> TypedFunction {
    TypedFunction {
        name: intern(name),
        annotations: vec![library::strict_fp()],
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

/// A label's name in the typed program, from its number.
pub(super) fn label_name(id: u32) -> InternedString {
    intern(&format!("$label{id}"))
}

/// `zl_depth += by`.
pub(super) fn depth_step(by: i64, span: Span) -> St {
    let i64_t = prim(PrimitiveType::I64);
    let depth = var(intern(library::DEPTH), i64_t.clone(), span);
    assign(
        depth.clone(),
        binary(BinaryOp::Add, depth, int_lit(by, span), i64_t, span),
        span,
    )
}

pub(super) fn entry_line_save(span: Span) -> St {
    let i64_t = prim(PrimitiveType::I64);
    let_(
        intern(ENTRY_LINE),
        i64_t.clone(),
        var(intern(library::LINE), i64_t, span),
        span,
    )
}

/// A field of a record for the debug library's host: no record or
/// field separator inside.
pub(super) fn sanitize(field: &str) -> String {
    field.replace(['\n', '\x1f'], "?")
}

/// The chunk name as positions spell it (`luaO_chunkid`): a file name
/// longer than the reference's buffer keeps its tail after `...`.
pub(super) fn chunk_id(file: &str) -> String {
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
