//! An exclusive argument may not be a second name for another one.
//!
//! A parameter declared `mut` says the callee may write through it and
//! nobody else is looking; one declared `own` says the callee is the
//! only holder left. Both are claims about the caller, not the callee,
//! and both are worth nothing unless the caller is held to them. Passing
//! one buffer as an exclusive argument and again as any other argument
//! of the same call breaks the claim at the moment it is made.
//!
//! This matters beyond tidiness. Deciding whether a loop's iterations
//! can run at once comes down to whether two base pointers reach the
//! same storage, and a function body cannot see that: its parameters
//! arrive as separate values whatever the caller did with them. An
//! exclusive parameter is the one piece of evidence that settles it,
//! which is why the promise has to be checked rather than assumed.
//!
//! ## What it can and cannot see
//!
//! Two arguments are compared by the storage they name, chasing casts
//! and element addresses back to a root, so `f(mut p, p)` and
//! `f(mut p, p + 3)` are both caught, and either spelling of an
//! element address counts. Two roots that differ are left alone:
//! whether they alias may depend on a caller further out, or on
//! nothing this module can read.
//!
//! So this is enforcement where the aliasing is visible and a declared
//! promise where it is not, which is the same standing `&mut` and
//! `restrict` have. It is deliberately not silent about the difference.

use std::collections::HashMap;

use crate::hir::{
    BinaryOp, HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirTerminator, HirType,
    ParamOwnership,
};

/// One call passing the same storage to an exclusive parameter and to
/// something else.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The function containing the call.
    pub caller: HirId,
    /// What was called, for the message.
    pub callee: String,
    /// Position of the parameter declared exclusive.
    pub exclusive: usize,
    /// Position of the other argument naming the same storage.
    pub other: usize,
}

impl Conflict {
    /// A message for whoever wrote the call, in terms of the call.
    pub fn message(&self) -> String {
        format!(
            "argument {} of `{}` is declared exclusive, so nothing else \
             may name the same value; argument {} does",
            self.exclusive + 1,
            self.callee,
            self.other + 1
        )
    }
}

/// Whether this ownership gives the callee sole access.
fn is_exclusive(o: ParamOwnership) -> bool {
    matches!(o, ParamOwnership::BorrowedMut | ParamOwnership::Owned)
}

/// How one value can be reached from another: an address chain worth
/// following back to the storage it started from.
enum Step {
    /// Same storage under another type.
    Cast(HirId),
    /// An element inside it.
    Element(HirId),
}

/// Whether a value names storage rather than a number.
fn is_pointer(func: &HirFunction, value: HirId) -> bool {
    func.values
        .get(&value)
        .is_some_and(|v| matches!(v.ty, HirType::Ptr(_) | HirType::Ref { .. }))
}

/// Index the address chains in `func`, so a root lookup is a walk
/// rather than a scan per question.
fn chains(func: &HirFunction) -> HashMap<HirId, Step> {
    let mut steps = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            match inst {
                HirInstruction::Cast {
                    result, operand, ..
                } => {
                    steps.insert(*result, Step::Cast(*operand));
                }
                HirInstruction::GetElementPtr { result, ptr, .. } => {
                    steps.insert(*result, Step::Element(*ptr));
                }
                // Adding to a pointer names an element of it the same
                // way indexing does. Both spellings reach a buffer, so
                // a check that follows only one of them can be walked
                // around by writing the other.
                HirInstruction::Binary {
                    op: BinaryOp::Add | BinaryOp::Sub,
                    result,
                    left,
                    right,
                    ..
                } => match (is_pointer(func, *left), is_pointer(func, *right)) {
                    (true, false) => {
                        steps.insert(*result, Step::Element(*left));
                    }
                    (false, true) => {
                        steps.insert(*result, Step::Element(*right));
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    steps
}

/// The storage a value names, as far back as the chain goes.
fn root(steps: &HashMap<HirId, Step>, mut value: HirId) -> HirId {
    // A chain this long is not a shape worth following, and the bound
    // keeps a malformed one from spinning.
    for _ in 0..32 {
        match steps.get(&value) {
            Some(Step::Cast(next)) | Some(Step::Element(next)) => value = *next,
            None => break,
        }
    }
    value
}

/// Every call in `module` that breaks an exclusive parameter's claim.
pub fn check_module(module: &HirModule) -> Vec<Conflict> {
    let mut found = Vec::new();
    for func in module.functions.values() {
        if func.is_external || func.blocks.is_empty() {
            continue;
        }
        let steps = chains(func);
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let HirInstruction::Call { callee, args, .. } = inst {
                    check_call(module, func, &steps, callee, args, &mut found);
                }
            }
            if let HirTerminator::Invoke { callee, args, .. } = &block.terminator {
                check_call(module, func, &steps, callee, args, &mut found);
            }
        }
    }
    found
}

fn check_call(
    module: &HirModule,
    caller: &HirFunction,
    steps: &HashMap<HirId, Step>,
    callee: &HirCallable,
    args: &[HirId],
    found: &mut Vec<Conflict>,
) {
    // Only a call naming a function in this module says what its
    // parameters claim. An indirect one does not, and a symbol is
    // somebody else's declaration.
    let HirCallable::Function(id) = callee else {
        return;
    };
    let Some(target) = module.functions.get(id) else {
        return;
    };
    let name = target
        .name
        .resolve_global()
        .unwrap_or_else(|| "<unnamed>".to_string());

    let roots: Vec<HirId> = args.iter().map(|a| root(steps, *a)).collect();
    for (i, param) in target.signature.params.iter().enumerate() {
        if !is_exclusive(param.ownership) || i >= roots.len() {
            continue;
        }
        for (j, other) in roots.iter().enumerate() {
            if i != j && roots[i] == *other {
                found.push(Conflict {
                    caller: caller.id,
                    callee: name.clone(),
                    exclusive: i,
                    other: j,
                });
            }
        }
    }
}

/// Whether a parameter of `func` is one nothing else may name.
///
/// This is what a caller established by getting past [`check_module`],
/// and what an analysis deciding whether two bases can be the same
/// storage is entitled to rely on.
pub fn parameter_is_exclusive(func: &HirFunction, value: HirId) -> bool {
    func.values
        .get(&value)
        .and_then(|v| match v.kind {
            crate::hir::HirValueKind::Parameter(idx) => func.signature.params.get(idx as usize),
            _ => None,
        })
        .is_some_and(|p| is_exclusive(p.ownership))
}

/// Whether `value` is any parameter of `func`.
pub fn is_parameter(func: &HirFunction, value: HirId) -> bool {
    func.values
        .get(&value)
        .is_some_and(|v| matches!(v.kind, crate::hir::HirValueKind::Parameter(_)))
}
