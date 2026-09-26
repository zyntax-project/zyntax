#![cfg(feature = "cranelift-backend")]

//! Control flow inside a block used as a value, lowered from typed AST
//! and run on every backend.
//!
//! A `TypedExpression::Block` in a let initializer, a call argument or
//! a short-circuit operand is lowered on demand by the SSA builder, not
//! by the function-level CFG builder. Each case below builds such a
//! block with an `if`, a loop, `break` or `continue` in it, places it
//! in each of those positions, and checks the value it produces on
//! Cranelift, LLVM, the wasm backend and the HIR interpreter.

use std::collections::HashMap;
use std::sync::Arc;

use zyntax_compiler::hir::{HirFunction, HirId, HirModule};
use zyntax_compiler::{CompilationConfig, compile_to_hir};
use zyntax_typed_ast::type_registry::Mutability;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedFor, TypedIf, TypedLet, TypedParameter, TypedPattern,
    TypedRange, TypedWhile,
};
use zyntax_typed_ast::{
    BinaryOp, CallingConvention, InternedString, PrimitiveType, Span, Type, TypeRegistry,
    TypedDeclaration, TypedExpression, TypedFunction, TypedLiteral, TypedNode, TypedProgram,
    TypedStatement, Visibility,
};

type Expr = TypedNode<TypedExpression>;
type Stmt = TypedNode<TypedStatement>;

fn span() -> Span {
    Span::new(0, 10)
}

fn node<T>(inner: T, ty: Type) -> TypedNode<T> {
    TypedNode {
        node: inner,
        ty,
        span: span(),
    }
}

fn name(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn i64_ty() -> Type {
    Type::Primitive(PrimitiveType::I64)
}

fn bool_ty() -> Type {
    Type::Primitive(PrimitiveType::Bool)
}

fn unit_ty() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}

fn int(v: i128) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Integer(v)), i64_ty())
}

fn truth() -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::Bool(true)),
        bool_ty(),
    )
}

fn var(s: &str) -> Expr {
    node(TypedExpression::Variable(name(s)), i64_ty())
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    let ty = match op {
        BinaryOp::Eq
        | BinaryOp::Ne
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge
        | BinaryOp::And
        | BinaryOp::Or => bool_ty(),
        _ => i64_ty(),
    };
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }),
        ty,
    )
}

fn add(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Add, l, r)
}

fn let_(s: &str, init: Expr) -> Stmt {
    node(
        TypedStatement::Let(TypedLet {
            name: name(s),
            ty: i64_ty(),
            mutability: Mutability::Mutable,
            initializer: Some(Box::new(init)),
            span: span(),
        }),
        unit_ty(),
    )
}

fn let_bool(s: &str, init: Expr) -> Stmt {
    node(
        TypedStatement::Let(TypedLet {
            name: name(s),
            ty: bool_ty(),
            mutability: Mutability::Immutable,
            initializer: Some(Box::new(init)),
            span: span(),
        }),
        unit_ty(),
    )
}

fn set(s: &str, value: Expr) -> Stmt {
    expr_stmt(bin(BinaryOp::Assign, var(s), value))
}

fn expr_stmt(e: Expr) -> Stmt {
    node(TypedStatement::Expression(Box::new(e)), unit_ty())
}

fn block(statements: Vec<Stmt>) -> TypedBlock {
    TypedBlock {
        statements,
        span: span(),
    }
}

fn if_(cond: Expr, then: Vec<Stmt>, otherwise: Option<Vec<Stmt>>) -> Stmt {
    node(
        TypedStatement::If(TypedIf {
            condition: Box::new(cond),
            then_block: block(then),
            else_block: otherwise.map(block),
            span: span(),
        }),
        unit_ty(),
    )
}

fn while_(cond: Expr, body: Vec<Stmt>) -> Stmt {
    node(
        TypedStatement::While(TypedWhile {
            condition: Box::new(cond),
            body: block(body),
            span: span(),
        }),
        unit_ty(),
    )
}

fn for_range(v: &str, start: Expr, end: Expr, body: Vec<Stmt>) -> Stmt {
    node(
        TypedStatement::For(TypedFor {
            pattern: Box::new(node(
                TypedPattern::Identifier {
                    name: name(v),
                    mutability: Mutability::Mutable,
                },
                i64_ty(),
            )),
            iterator: Box::new(node(
                TypedExpression::Range(TypedRange {
                    start: Some(Box::new(start)),
                    end: Some(Box::new(end)),
                    inclusive: false,
                }),
                i64_ty(),
            )),
            body: block(body),
        }),
        unit_ty(),
    )
}

fn brk() -> Stmt {
    node(TypedStatement::Break(None), unit_ty())
}

fn cont() -> Stmt {
    node(TypedStatement::Continue, unit_ty())
}

fn ret(e: Expr) -> Stmt {
    node(TypedStatement::Return(Some(Box::new(e))), unit_ty())
}

fn block_value(statements: Vec<Stmt>) -> Expr {
    node(TypedExpression::Block(block(statements)), i64_ty())
}

fn call(callee: &str, args: Vec<Expr>) -> Expr {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(node(TypedExpression::Variable(name(callee)), unit_ty())),
            positional_args: args,
            named_args: vec![],
            type_args: vec![],
        }),
        i64_ty(),
    )
}

fn param(s: &str) -> TypedParameter {
    TypedParameter {
        name: name(s),
        ty: i64_ty(),
        mutability: Mutability::Mutable,
        ..Default::default()
    }
}

fn function(fn_name: &str, params: &[&str], body: Vec<Stmt>) -> TypedNode<TypedDeclaration> {
    node(
        TypedDeclaration::Function(TypedFunction {
            name: name(fn_name),
            params: params.iter().map(|p| param(p)).collect(),
            return_type: i64_ty(),
            body: Some(block(body)),
            visibility: Visibility::Public,
            calling_convention: CallingConvention::Rust,
            ..Default::default()
        }),
        unit_ty(),
    )
}

/// A block whose statements end in its value. Every case reads the
/// parameters `a` and `b`.
struct Case {
    name: &'static str,
    statements: fn() -> Vec<Stmt>,
    expected: fn(i64, i64) -> i64,
}

fn odd_sum_to(n: i64) -> i64 {
    (1..=n).filter(|i| i % 2 == 1).sum()
}

const CASES: &[Case] = &[
    // let t = 10; if a > 0 { t = b } else { let u = b + 1 }; t
    Case {
        name: "write_in_one_branch",
        statements: || {
            vec![
                let_("t", int(10)),
                if_(
                    bin(BinaryOp::Gt, var("a"), int(0)),
                    vec![set("t", var("b"))],
                    Some(vec![let_("u", add(var("b"), int(1)))]),
                ),
                expr_stmt(var("t")),
            ]
        },
        expected: |a, b| if a > 0 { b } else { 10 },
    },
    // let t = 10; if a > 0 { t = b } else { t = a + b }; t
    Case {
        name: "write_in_both_branches",
        statements: || {
            vec![
                let_("t", int(10)),
                if_(
                    bin(BinaryOp::Gt, var("a"), int(0)),
                    vec![set("t", var("b"))],
                    Some(vec![set("t", add(var("a"), var("b")))]),
                ),
                expr_stmt(var("t")),
            ]
        },
        expected: |a, b| if a > 0 { b } else { a + b },
    },
    // let t = 10
    // if a > 0 { if b > 4 { t = 1 } else { t = 2 } } else { if b > 4 { t = 3 } }
    // t
    Case {
        name: "nested_ifs",
        statements: || {
            vec![
                let_("t", int(10)),
                if_(
                    bin(BinaryOp::Gt, var("a"), int(0)),
                    vec![if_(
                        bin(BinaryOp::Gt, var("b"), int(4)),
                        vec![set("t", int(1))],
                        Some(vec![set("t", int(2))]),
                    )],
                    Some(vec![if_(
                        bin(BinaryOp::Gt, var("b"), int(4)),
                        vec![set("t", int(3))],
                        None,
                    )]),
                ),
                expr_stmt(var("t")),
            ]
        },
        expected: |a, b| match (a > 0, b > 4) {
            (true, true) => 1,
            (true, false) => 2,
            (false, true) => 3,
            (false, false) => 10,
        },
    },
    // let t = 10; if a > 0 { t = t + b }; t
    Case {
        name: "if_without_else",
        statements: || {
            vec![
                let_("t", int(10)),
                if_(
                    bin(BinaryOp::Gt, var("a"), int(0)),
                    vec![set("t", add(var("t"), var("b")))],
                    None,
                ),
                expr_stmt(var("t")),
            ]
        },
        expected: |a, b| if a > 0 { 10 + b } else { 10 },
    },
    // let i = 0; let s = 0; while i < b { s = s + a; i = i + 1 }; s
    Case {
        name: "while_reads_body_write",
        statements: || {
            vec![
                let_("i", int(0)),
                let_("s", int(0)),
                while_(
                    bin(BinaryOp::Lt, var("i"), var("b")),
                    vec![
                        set("s", add(var("s"), var("a"))),
                        set("i", add(var("i"), int(1))),
                    ],
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |a, b| a * b.max(0),
    },
    // let s = 0; for i in 0..b { s = s + i }; s
    Case {
        name: "for_loop",
        statements: || {
            vec![
                let_("s", int(0)),
                for_range(
                    "i",
                    int(0),
                    var("b"),
                    vec![set("s", add(var("s"), var("i")))],
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |_, b| (0..b).sum(),
    },
    // let i = 0; while true { if i >= b { break }; i = i + 1 }; i
    Case {
        name: "while_break",
        statements: || {
            vec![
                let_("i", int(0)),
                while_(
                    truth(),
                    vec![
                        if_(bin(BinaryOp::Ge, var("i"), var("b")), vec![brk()], None),
                        set("i", add(var("i"), int(1))),
                    ],
                ),
                expr_stmt(var("i")),
            ]
        },
        expected: |_, b| b.max(0),
    },
    // let i = 0; let s = 0
    // while i < b { i = i + 1; if i % 2 == 0 { continue }; s = s + i }
    // s
    Case {
        name: "while_continue",
        statements: || {
            vec![
                let_("i", int(0)),
                let_("s", int(0)),
                while_(
                    bin(BinaryOp::Lt, var("i"), var("b")),
                    vec![
                        set("i", add(var("i"), int(1))),
                        if_(
                            bin(BinaryOp::Eq, bin(BinaryOp::Rem, var("i"), int(2)), int(0)),
                            vec![cont()],
                            None,
                        ),
                        set("s", add(var("s"), var("i"))),
                    ],
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |_, b| odd_sum_to(b),
    },
    // let s = 0; for i in 0..100 { if i >= b { break }; s = s + i }; s
    Case {
        name: "for_break",
        statements: || {
            vec![
                let_("s", int(0)),
                for_range(
                    "i",
                    int(0),
                    int(100),
                    vec![
                        if_(bin(BinaryOp::Ge, var("i"), var("b")), vec![brk()], None),
                        set("s", add(var("s"), var("i"))),
                    ],
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |_, b| (0..b.clamp(0, 100)).sum(),
    },
    // let s = 0; for i in 0..b { if i % 2 == 0 { continue }; s = s + i }; s
    Case {
        name: "for_continue",
        statements: || {
            vec![
                let_("s", int(0)),
                for_range(
                    "i",
                    int(0),
                    var("b"),
                    vec![
                        if_(
                            bin(BinaryOp::Eq, bin(BinaryOp::Rem, var("i"), int(2)), int(0)),
                            vec![cont()],
                            None,
                        ),
                        set("s", add(var("s"), var("i"))),
                    ],
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |_, b| odd_sum_to(b - 1),
    },
    // let s = 0
    // if a > 0 { let i = 0; while i < b { s = s + 2; i = i + 1 } } else { s = 7 }
    // s
    Case {
        name: "loop_inside_if",
        statements: || {
            vec![
                let_("s", int(0)),
                if_(
                    bin(BinaryOp::Gt, var("a"), int(0)),
                    vec![
                        let_("i", int(0)),
                        while_(
                            bin(BinaryOp::Lt, var("i"), var("b")),
                            vec![
                                set("s", add(var("s"), int(2))),
                                set("i", add(var("i"), int(1))),
                            ],
                        ),
                    ],
                    Some(vec![set("s", int(7))]),
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |a, b| if a > 0 { 2 * b.max(0) } else { 7 },
    },
    // let i = 0; let s = 0
    // while true { let r = { if i >= b { break }; i + a }; s = s + r; i = i + 1 }
    // s
    Case {
        name: "break_from_inner_block_value",
        statements: || {
            vec![
                let_("i", int(0)),
                let_("s", int(0)),
                while_(
                    truth(),
                    vec![
                        let_(
                            "r",
                            block_value(vec![
                                if_(bin(BinaryOp::Ge, var("i"), var("b")), vec![brk()], None),
                                expr_stmt(add(var("i"), var("a"))),
                            ]),
                        ),
                        set("s", add(var("s"), var("r"))),
                        set("i", add(var("i"), int(1))),
                    ],
                ),
                expr_stmt(var("s")),
            ]
        },
        expected: |a, b| (0..b.max(0)).map(|i| i + a).sum(),
    },
];

/// Where the block sits. Every function takes `(a, b, e)`.
#[derive(Clone, Copy, Debug)]
enum Position {
    /// `let r = BLOCK; return r`
    LetInitializer,
    /// `return plus(a, BLOCK)`
    CallArgument,
    /// `let ok = e != -1 && BLOCK == e; if ok { return 1 } else { return 0 }`
    ShortCircuitOperand,
}

const POSITIONS: [Position; 3] = [
    Position::LetInitializer,
    Position::CallArgument,
    Position::ShortCircuitOperand,
];

fn fn_name(case: &Case, position: Position) -> String {
    format!("{}_{:?}", case.name, position)
}

fn body_for(case: &Case, position: Position) -> Vec<Stmt> {
    let value = block_value((case.statements)());
    match position {
        Position::LetInitializer => vec![let_("r", value), ret(var("r"))],
        Position::CallArgument => vec![ret(call("plus", vec![var("a"), value]))],
        Position::ShortCircuitOperand => vec![
            let_bool(
                "ok",
                bin(
                    BinaryOp::And,
                    bin(BinaryOp::Ne, var("e"), int(-1)),
                    bin(BinaryOp::Eq, value, var("e")),
                ),
            ),
            if_(
                node(TypedExpression::Variable(name("ok")), bool_ty()),
                vec![ret(int(1))],
                Some(vec![ret(int(0))]),
            ),
        ],
    }
}

/// Function-level loops whose body writes a variable only inside a
/// block value: the loop's own phis must see that write.
struct Extra {
    name: &'static str,
    body: fn() -> Vec<Stmt>,
    expected: fn(i64, i64) -> i64,
}

const EXTRAS: &[Extra] = &[
    // let acc = 0; let k = 0
    // while k < 3 { let r = { if a > 0 { acc = acc + b }; 0 }; k = k + 1 }
    // return acc
    Extra {
        name: "loop_writes_through_block_value",
        body: || {
            vec![
                let_("acc", int(0)),
                let_("k", int(0)),
                while_(
                    bin(BinaryOp::Lt, var("k"), int(3)),
                    vec![
                        let_(
                            "r",
                            block_value(vec![
                                if_(
                                    bin(BinaryOp::Gt, var("a"), int(0)),
                                    vec![set("acc", add(var("acc"), var("b")))],
                                    None,
                                ),
                                expr_stmt(int(0)),
                            ]),
                        ),
                        set("k", add(var("k"), int(1))),
                    ],
                ),
                ret(var("acc")),
            ]
        },
        expected: |a, b| if a > 0 { 3 * b } else { 0 },
    },
    // let r = { if a > 0 { b = b + 1 }; 0 }; return b + r
    Extra {
        name: "parameter_written_in_block_value",
        body: || {
            vec![
                let_(
                    "r",
                    block_value(vec![
                        if_(
                            bin(BinaryOp::Gt, var("a"), int(0)),
                            vec![set("b", add(var("b"), int(1)))],
                            None,
                        ),
                        expr_stmt(int(0)),
                    ]),
                ),
                ret(add(var("b"), var("r"))),
            ]
        },
        expected: |a, b| if a > 0 { b + 1 } else { b },
    },
    // let i = 0; let s = 0
    // while i < 100 { let r = { if i >= b { break }; i * 2 }; s = s + r; i = i + 1 }
    // return s
    Extra {
        name: "break_from_block_value_in_loop",
        body: || {
            vec![
                let_("i", int(0)),
                let_("s", int(0)),
                while_(
                    bin(BinaryOp::Lt, var("i"), int(100)),
                    vec![
                        let_(
                            "r",
                            block_value(vec![
                                if_(bin(BinaryOp::Ge, var("i"), var("b")), vec![brk()], None),
                                expr_stmt(bin(BinaryOp::Mul, var("i"), int(2))),
                            ]),
                        ),
                        set("s", add(var("s"), var("r"))),
                        set("i", add(var("i"), int(1))),
                    ],
                ),
                ret(var("s")),
            ]
        },
        expected: |_, b| (0..b.clamp(0, 100)).map(|i| 2 * i).sum(),
    },
    // let s = 0
    // for i in 0..b { s = s + plus(0, { if i % 2 == 0 { continue }; i }) }
    // return s
    Extra {
        name: "continue_from_block_value_in_for",
        body: || {
            vec![
                let_("s", int(0)),
                for_range(
                    "i",
                    int(0),
                    var("b"),
                    vec![set(
                        "s",
                        add(
                            var("s"),
                            call(
                                "plus",
                                vec![
                                    int(0),
                                    block_value(vec![
                                        if_(
                                            bin(
                                                BinaryOp::Eq,
                                                bin(BinaryOp::Rem, var("i"), int(2)),
                                                int(0),
                                            ),
                                            vec![cont()],
                                            None,
                                        ),
                                        expr_stmt(var("i")),
                                    ]),
                                ],
                            ),
                        ),
                    )],
                ),
                ret(var("s")),
            ]
        },
        expected: |_, b| odd_sum_to(b - 1),
    },
];

const INPUTS: [(i64, i64); 5] = [(3, 5), (-2, 4), (0, 7), (6, 0), (1, 9)];

/// One call and the value it must return.
struct Probe {
    function: String,
    args: [i64; 3],
    expected: i64,
}

/// `BLOCK_VALUES_ONLY=<text>` keeps only the probes whose function name
/// contains the text; safe to leave unset.
fn probes() -> Vec<Probe> {
    let mut out = all_probes();
    if let Ok(only) = std::env::var("BLOCK_VALUES_ONLY") {
        out.retain(|p| p.function.contains(&only));
    }
    out
}

fn all_probes() -> Vec<Probe> {
    let mut out = Vec::new();
    for case in CASES {
        for position in POSITIONS {
            for (a, b) in INPUTS {
                let v = (case.expected)(a, b);
                let function = fn_name(case, position);
                match position {
                    Position::LetInitializer => out.push(Probe {
                        function,
                        args: [a, b, 0],
                        expected: v,
                    }),
                    Position::CallArgument => out.push(Probe {
                        function,
                        args: [a, b, 0],
                        expected: a + v,
                    }),
                    Position::ShortCircuitOperand => {
                        for (e, expected) in [(v, 1), (v + 1, 0), (-1, 0)] {
                            // `e == -1` skips the block; when the block's
                            // value is itself -1 the first probe is that.
                            let expected = if e == -1 { 0 } else { expected };
                            out.push(Probe {
                                function: function.clone(),
                                args: [a, b, e],
                                expected,
                            });
                        }
                    }
                }
            }
        }
    }
    for extra in EXTRAS {
        for (a, b) in INPUTS {
            out.push(Probe {
                function: extra.name.to_string(),
                args: [a, b, 0],
                expected: (extra.expected)(a, b),
            });
        }
    }
    out
}

fn module() -> HirModule {
    let mut declarations = vec![function(
        "plus",
        &["x", "y"],
        vec![ret(add(var("x"), var("y")))],
    )];
    for case in CASES {
        for position in POSITIONS {
            declarations.push(function(
                &fn_name(case, position),
                &["a", "b", "e"],
                body_for(case, position),
            ));
        }
    }
    for extra in EXTRAS {
        declarations.push(function(extra.name, &["a", "b", "e"], (extra.body)()));
    }
    let mut program = TypedProgram {
        language: None,
        declarations,
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };
    compile_to_hir(
        &mut program,
        Arc::new(TypeRegistry::new()),
        CompilationConfig {
            opt_level: 0,
            debug_info: false,
            enable_monomorphization: true,
            ..Default::default()
        },
    )
    .expect("the cases lower")
}

fn find<'m>(module: &'m HirModule, fn_name: &str) -> &'m HirFunction {
    module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(fn_name))
        .unwrap_or_else(|| panic!("{fn_name} is in the module"))
}

/// How long one probe may run. A loop that never reads its own writes
/// spins forever, so a probe past this is reported rather than waited on.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Run every probe through `run` on a thread of its own and report every
/// mismatch at once. A probe that does not finish ends the run there.
fn check(backend: &str, mut run: impl FnMut(&Probe) -> Result<i64, String> + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for probe in probes() {
            if tx.send(run(&probe)).is_err() {
                return;
            }
        }
    });
    let mut failures = Vec::new();
    for probe in probes() {
        match rx.recv_timeout(PROBE_TIMEOUT) {
            Ok(Ok(got)) if got == probe.expected => {}
            Ok(Ok(got)) => failures.push(format!(
                "{}{:?}: got {got}, expected {}",
                probe.function, probe.args, probe.expected
            )),
            Ok(Err(e)) => failures.push(format!("{}{:?}: {e}", probe.function, probe.args)),
            Err(_) => {
                failures.push(format!(
                    "{}{:?}: did not finish; the probes after it were not run",
                    probe.function, probe.args
                ));
                break;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{backend}: {} probe(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

type Entry = extern "C" fn(i64, i64, i64) -> i64;

/// Every probed function's entry point, as an address a thread can take.
fn entries(
    module: &HirModule,
    mut lookup: impl FnMut(HirId) -> Option<usize>,
) -> Arc<HashMap<String, usize>> {
    let mut out = HashMap::new();
    for probe in probes() {
        if !out.contains_key(&probe.function)
            && let Some(ptr) = lookup(find(module, &probe.function).id)
        {
            out.insert(probe.function.clone(), ptr);
        }
    }
    Arc::new(out)
}

fn run_native(entries: &HashMap<String, usize>, p: &Probe) -> Result<i64, String> {
    let ptr = *entries
        .get(&p.function)
        .ok_or_else(|| "not compiled".to_string())?;
    let f: Entry = unsafe { std::mem::transmute(ptr) };
    Ok(f(p.args[0], p.args[1], p.args[2]))
}

#[test]
fn block_values_on_cranelift() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let module = module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let entries = entries(&module, |id| {
        backend.get_function_ptr(id).map(|p| p as usize)
    });
    // The code must outlive a probe left running on its thread.
    std::mem::forget(backend);
    check("cranelift", move |p| run_native(&entries, p));
}

#[test]
fn block_values_on_the_interpreter() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::value::ZyntaxValue;
    let module = Arc::new(module());
    check("interpreter", move |p| {
        let mut interp = HirInterpreter::new();
        let args = p.args.iter().map(|v| ZyntaxValue::Int(*v)).collect();
        let v = interp
            .call(&module, &p.function, args)
            .map_err(|e| format!("{e:?}"))?;
        value_to_i64(&v).ok_or_else(|| format!("not an integer: {v:?}"))
    });
}

#[cfg(feature = "llvm-backend")]
#[test]
fn block_values_on_llvm() {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;
    if zyntax_compiler::llvm_link::find_linker().is_err() {
        eprintln!("no system linker; skipping the LLVM leg");
        return;
    }
    let module = module();
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let mut backend = LLVMJitBackend::new(context).expect("backend");
    backend.compile_module(&module).expect("compile");
    let entries = entries(&module, |id| {
        backend.get_function_pointer(id).map(|p| p as usize)
    });
    std::mem::forget(backend);
    check("llvm", move |p| run_native(&entries, p));
}

/// The wasm backend emits one module per function; node runs them,
/// with `plus` supplied as the import a call to it becomes.
#[cfg(feature = "wasm-jit")]
#[test]
fn block_values_on_wasm() {
    use std::fmt::Write as _;
    use zyntax_compiler::wasm_backend::WasmBackend;
    if std::process::Command::new("node")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("no node; skipping the wasm leg");
        return;
    }
    let module = module();
    let dir = std::env::temp_dir().join(format!("zyntax_block_values_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let backend = WasmBackend::new();
    let mut compile_errors = Vec::new();
    let mut names: Vec<String> = probes().iter().map(|p| p.function.clone()).collect();
    names.push("plus".to_string());
    names.sort();
    names.dedup();
    for fn_name in &names {
        match backend.compile_function(find(&module, fn_name)) {
            Ok(m) => std::fs::write(dir.join(format!("{fn_name}.wasm")), &m.bytes).expect("write"),
            Err(e) => compile_errors.push(format!("{fn_name}: {e}")),
        }
    }
    assert!(
        compile_errors.is_empty(),
        "wasm: functions that did not compile:\n{}",
        compile_errors.join("\n")
    );
    let mut calls = String::from("[");
    for p in probes() {
        let _ = write!(
            calls,
            "[\"{}\",{},{},{}],",
            p.function, p.args[0], p.args[1], p.args[2]
        );
    }
    calls.push(']');
    let script = format!(
        r#"
const fs = require('fs');
const dir = {dir:?};
const read = (n) => new WebAssembly.Module(fs.readFileSync(dir + '/' + n + '.wasm'));
const plus = new WebAssembly.Instance(read('plus'), {{}}).exports.entry;
const cache = {{}};
const get = (n) => {{
  if (!cache[n]) {{
    const m = read(n);
    const imports = {{}};
    for (const imp of WebAssembly.Module.imports(m)) {{
      if (imp.module !== 'internal') throw new Error(n + ' imports ' + imp.module + '.' + imp.name);
      imports.internal = imports.internal || {{}};
      imports.internal[imp.name] = plus;
    }}
    cache[n] = new WebAssembly.Instance(m, imports).exports.entry;
  }}
  return cache[n];
}};
for (const [n, a, b, e] of {calls}) {{
  try {{ fs.writeSync(1, String(get(n)(BigInt(a), BigInt(b), BigInt(e))) + '\n'); }}
  catch (err) {{ fs.writeSync(1, 'error ' + err.message + '\n'); }}
}}
"#,
        dir = dir.to_string_lossy()
    );
    let script_path = dir.join("run.js");
    std::fs::write(&script_path, script).expect("write script");
    let out_path = dir.join("out.txt");
    let mut child = std::process::Command::new("node")
        .arg(&script_path)
        .stdout(std::fs::File::create(&out_path).expect("out file"))
        .spawn()
        .expect("run node");
    // A spinning probe hangs node; what it printed before that is kept.
    let started = std::time::Instant::now();
    while child.try_wait().expect("wait").is_none() {
        if started.elapsed() > PROBE_TIMEOUT * 3 {
            let _ = child.kill();
            let _ = child.wait();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let stdout = std::fs::read_to_string(&out_path).expect("read output");
    let _ = std::fs::remove_dir_all(&dir);
    let mut lines: Vec<String> = stdout.lines().map(str::to_string).collect();
    lines.reverse();
    check("wasm", move |_| match lines.pop() {
        Some(line) => line.parse::<i64>().map_err(|_| line),
        None => loop {
            std::thread::park();
        },
    });
}
