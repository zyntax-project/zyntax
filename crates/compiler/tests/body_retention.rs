//! What the tiers keep of a lazy function's HIR. Its interp body and its
//! optimised body go once it has native code and no interpreted frame
//! runs them; a frame still running one finds it for as long as it
//! runs; a large body whose first frame returns before there is native
//! code goes then; and a body asked for again after it went is made
//! again, once.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirModule, HirParam, HirPhi, HirTerminator, HirType, HirValue, HirValueKind,
};
use zyntax_compiler::opt_audit;
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

/// Steps of the chain each body runs: two instructions a step, more than
/// any size a body is kept at.
const STEPS: usize = 300;

fn key(k: usize) -> i32 {
    (k as i32).wrapping_mul(0x9e37) ^ 0x5a5a
}

/// The chain, as the compiled code computes it.
fn chain(mut x: i32) -> i32 {
    for k in 0..STEPS {
        x = x.wrapping_mul(31) ^ key(k);
    }
    x
}

struct Builder {
    values: IndexMap<HirId, HirValue>,
}

impl Builder {
    fn value(&mut self, ty: HirType, kind: HirValueKind) -> HirId {
        let id = HirId::new();
        self.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn constant(&mut self, v: i32) -> HirId {
        self.value(HirType::I32, HirValueKind::Constant(HirConstant::I32(v)))
    }

    fn binary(
        &mut self,
        out: &mut Vec<HirInstruction>,
        op: BinaryOp,
        left: HirId,
        right: HirId,
    ) -> HirId {
        let ty = if matches!(op, BinaryOp::Lt) {
            HirType::Bool
        } else {
            HirType::I32
        };
        let result = self.value(ty.clone(), HirValueKind::Instruction);
        out.push(HirInstruction::Binary {
            op,
            result,
            ty,
            left,
            right,
        });
        result
    }

    /// `x * 31 ^ key(k)` for each step, into `out`.
    fn chain(&mut self, out: &mut Vec<HirInstruction>, mut x: HirId) -> HirId {
        let thirty_one = self.constant(31);
        for k in 0..STEPS {
            let key = self.constant(key(k));
            let m = self.binary(out, BinaryOp::Mul, x, thirty_one);
            x = self.binary(out, BinaryOp::Xor, m, key);
        }
        x
    }
}

fn block(
    id: HirId,
    phis: Vec<HirPhi>,
    instructions: Vec<HirInstruction>,
    terminator: HirTerminator,
) -> HirBlock {
    HirBlock {
        id,
        label: None,
        phis,
        instructions,
        terminator,
        dominance_frontier: Default::default(),
        predecessors: vec![],
        successors: vec![],
    }
}

fn function(name: &str, param: HirId, b: Builder, blocks: Vec<HirBlock>) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global(name),
        HirFunctionSignature {
            params: vec![HirParam {
                id: param,
                name: InternedString::new_global("n"),
                ty: HirType::I32,
                attributes: Default::default(),
                ownership: Default::default(),
            }],
            returns: vec![HirType::I32],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: true,
        },
    );
    f.entry_block = blocks[0].id;
    f.blocks = blocks.into_iter().map(|b| (b.id, b)).collect();
    f.values = b.values;
    // Left for its first call, as the runtime leaves a program's
    // functions.
    f.attributes.optimized = true;
    f.attributes.deferred = true;
    f
}

/// `looped(n)`: `x = n`, then `n` times `x = chain(x)`; returns `x`.
fn looped() -> HirFunction {
    let mut b = Builder {
        values: IndexMap::new(),
    };
    let n = b.value(HirType::I32, HirValueKind::Parameter(0));
    let zero = b.constant(0);
    let one = b.constant(1);
    let (entry, header, body, exit) = (HirId::new(), HirId::new(), HirId::new(), HirId::new());
    let i = b.value(HirType::I32, HirValueKind::Instruction);
    let x = b.value(HirType::I32, HirValueKind::Instruction);
    let mut test = Vec::new();
    let cmp = b.binary(&mut test, BinaryOp::Lt, i, n);
    let mut step = Vec::new();
    let x_next = b.chain(&mut step, x);
    let i_next = b.binary(&mut step, BinaryOp::Add, i, one);
    let blocks = vec![
        block(
            entry,
            vec![],
            vec![],
            HirTerminator::Branch { target: header },
        ),
        block(
            header,
            vec![
                HirPhi {
                    result: i,
                    ty: HirType::I32,
                    incoming: vec![(zero, entry), (i_next, body)],
                },
                HirPhi {
                    result: x,
                    ty: HirType::I32,
                    incoming: vec![(n, entry), (x_next, body)],
                },
            ],
            test,
            HirTerminator::CondBranch {
                condition: cmp,
                true_target: body,
                false_target: exit,
            },
        ),
        block(body, vec![], step, HirTerminator::Branch { target: header }),
        block(
            exit,
            vec![],
            vec![],
            HirTerminator::Return { values: vec![x] },
        ),
    ];
    function("looped", n, b, blocks)
}

/// `straight(n)`: `chain(n)`, with no loop.
fn straight() -> HirFunction {
    let mut b = Builder {
        values: IndexMap::new(),
    };
    let n = b.value(HirType::I32, HirValueKind::Parameter(0));
    let entry = HirId::new();
    let mut out = Vec::new();
    let x = b.chain(&mut out, n);
    let blocks = vec![block(
        entry,
        vec![],
        out,
        HirTerminator::Return { values: vec![x] },
    )];
    function("straight", n, b, blocks)
}

fn call(entry: *const u8, n: i32) -> i32 {
    // SAFETY: both functions are `fn(i32) -> i32`.
    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(entry) };
    f(n)
}

/// One test: the lazy compiler a backend installs is the process's.
#[test]
fn bodies_go_once_nothing_can_ask_for_them() {
    opt_audit::enable();
    let (looped, straight) = (looped(), straight());
    let (a, b) = (looped.id, straight.id);
    let mut module = HirModule::new(InternedString::new_global("body_retention"));
    module.functions.insert(a, looped);
    module.functions.insert(b, straight);
    let mut backend = TieredBackend::new(TieredConfig::default()).expect("tiered backend");
    backend.set_emit_osr_probes(true);
    backend
        .compile_module_lazily(module, None, HashSet::from([a, b]), HashSet::new(), false)
        .expect("module compiles");
    let (_, entry, bead) = backend.interpreter_bridge();
    let Some(bead_a) = bead(a) else {
        // OSR off in this environment: no bead to ask for bodies by.
        return;
    };
    let mut source = backend.interpreter_body_source();
    let mut frame_exit = backend.interpreter_frame_exit_hook();

    // A frame of `looped` starts in the interpreter on its interp body,
    // and holds it while it runs.
    let frame = source(a).expect("an interp body");
    let frame_tag = osr::body_tag(&frame);
    // The optimised body, made before the first compile, which reads it.
    let optimized = Arc::downgrade(&osr::lazy_optimized_body(bead_a).expect("an optimised body"));
    assert_eq!(opt_audit::pipeline_runs(a), 1);

    // The first call compiles it: native code from here on.
    assert_eq!(call(entry(a).expect("a stub"), 3), chain(chain(chain(3))));
    assert!(
        optimized.upgrade().is_none(),
        "the optimised body outlived the compile that was its last reader"
    );
    // The frame still running finds its body by the tag it asks at.
    let found = source(a).expect("the frame's body");
    assert!(Arc::ptr_eq(&found, &frame));
    assert_eq!(osr::body_tag(&found), frame_tag);
    drop(found);
    // Its frame returns: nothing holds the interp body any more.
    let gone = Arc::downgrade(&frame);
    drop(frame);
    assert!(
        gone.upgrade().is_none(),
        "the interp body outlived its last frame"
    );

    // Asked for again, each body is made again: the optimised one once
    // in its new cell, from the scratch while one still holds it.
    let remade = source(a).expect("an interp body made again");
    assert_eq!(opt_audit::pipeline_runs(a), 1);
    drop(remade);
    let again = osr::lazy_optimized_body(bead_a).expect("an optimised body made again");
    let runs = opt_audit::pipeline_runs(a);
    assert!(runs <= 2, "the pipeline ran {runs} times");
    let same = osr::lazy_optimized_body(bead_a).expect("the optimised body");
    assert!(Arc::ptr_eq(&again, &same));
    assert_eq!(opt_audit::pipeline_runs(a), runs);
    assert_eq!(call(entry(a).expect("code"), 2), chain(chain(2)));
    drop((again, same));

    // `straight` runs once in the interpreter: a large body whose first
    // frame returns with no native code for it goes then, once.
    let body = source(b).expect("an interp body");
    let gone = Arc::downgrade(&body);
    drop(body);
    assert!(frame_exit(b), "a large body run once was kept");
    assert!(gone.upgrade().is_none());
    let body = source(b).expect("an interp body made again");
    let kept = Arc::downgrade(&body);
    drop(body);
    assert!(!frame_exit(b), "a body made again was let go again");
    assert!(kept.upgrade().is_some());
    assert_eq!(call(entry(b).expect("a stub"), 5), chain(5));
}
