//! One box per string constant, made when the program starts.
//!
//! A string literal that flows into a dynamic slot is boxed at the use:
//! a box and a copy of the bytes each time the use runs, which for a
//! dict key inside a loop is once a row. The bytes are immutable and no
//! box of them is distinguishable from another, so one box serves every
//! use. Each boxed constant gets a global holding its box, a function
//! makes the boxes once when the module is installed, and each use
//! loads the global.
//!
//! Runs before the release analysis: a load is not an allocation, so
//! no use releases a box it did not make. The globals are writable, so
//! a collector that reads the program's globals keeps the boxes.
//!
//! A library stored for linking runs the pass on itself, under its own
//! name, so the bodies a program adopts are the bodies its release
//! facts were recorded on; the program's pass leaves them alone and
//! makes its own initializer. A program runs every initializer it holds.

use std::collections::{HashMap, HashSet};

use crate::hir::{
    BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirGlobal,
    HirId, HirInstruction, HirModule, HirTerminator, HirType, HirValue, HirValueKind, Linkage,
    Visibility,
};
use zyntax_typed_ast::InternedString;

/// The name of the function that makes the program's boxes.
pub const INIT_FUNCTION: &str = "__zyntax_box_constants";

/// The name of the function that makes a library's boxes.
pub fn library_init_function(library: &str) -> String {
    format!("{INIT_FUNCTION}${library}")
}

/// Whether `name` is an initializer this pass made, the program's or a
/// library's.
pub fn is_init_function(name: &str) -> bool {
    name.strip_prefix(INIT_FUNCTION)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('$'))
}

/// The initializers of libraries linked into `module`, by name.
pub fn library_init_functions(module: &HirModule) -> Vec<String> {
    module
        .functions
        .values()
        .filter(|f| !f.is_external)
        .filter_map(|f| f.name.resolve_global())
        .filter(|name| name != INIT_FUNCTION && is_init_function(name))
        .collect()
}

/// Calls that box a string, by the symbol reached or the function's
/// name, taking the string as their only argument.
const BOXERS: &[&str] = &["$IO$string_to_dynamic", "zyntax_box_str", "zb_box_str"];

/// Prepare one box per string constant for the host to initialize after
/// installing the module. Returns the number of replaced boxing sites.
pub fn run_module(module: &mut HirModule) -> usize {
    run_module_as(module, None)
}

/// [`run_module`] for a library about to be stored: its globals and
/// initializer are named after it, apart from those of the program
/// that links it.
pub fn run_library(module: &mut HirModule, library: &str) -> usize {
    run_module_as(module, Some(library))
}

fn run_module_as(module: &mut HirModule, library: Option<&str>) -> usize {
    // Which callees box a string.
    let mut boxers: HashSet<HirId> = HashSet::new();
    for (id, f) in &module.functions {
        let by_link = f.link_name.as_deref().is_some_and(|n| BOXERS.contains(&n));
        let by_name = f
            .name
            .resolve_global()
            .is_some_and(|n| BOXERS.contains(&n.as_str()));
        if by_link || by_name {
            boxers.insert(*id);
        }
    }
    if boxers.is_empty() {
        return 0;
    }
    let string_globals: HashSet<HirId> = module
        .globals
        .iter()
        .filter(|(_, g)| matches!(g.initializer, Some(HirConstant::String(_))))
        .map(|(id, _)| *id)
        .collect();

    // Sites: (function, block, index) with the string global and the
    // callee, so the box can be made the same way.
    let mut sites: Vec<(HirId, HirId, usize, HirId, HirCallable, HirType)> = Vec::new();
    for (fid, f) in &module.functions {
        if f.is_external
            || f.name
                .resolve_global()
                .is_some_and(|name| is_init_function(&name))
        {
            continue;
        }
        for (bid, block) in &f.blocks {
            for (i, inst) in block.instructions.iter().enumerate() {
                let HirInstruction::Call {
                    result: Some(result),
                    callee,
                    args,
                    ..
                } = inst
                else {
                    continue;
                };
                let boxing = match callee {
                    HirCallable::Function(c) => boxers.contains(c),
                    HirCallable::Symbol(name) => BOXERS.contains(&name.as_str()),
                    _ => false,
                };
                if !boxing || args.len() != 1 {
                    continue;
                }
                let Some(arg) = f.values.get(&args[0]) else {
                    continue;
                };
                let HirValueKind::Global(gid) = arg.kind else {
                    continue;
                };
                if !string_globals.contains(&gid) {
                    continue;
                }
                let Some(ty) = f.values.get(result).map(|v| v.ty.clone()) else {
                    continue;
                };
                sites.push((*fid, *bid, i, gid, callee.clone(), ty));
            }
        }
    }
    if sites.is_empty() {
        return 0;
    }

    // One global per constant.
    let box_ty = crate::zrtl::dynamic_box_pointer_type();
    let mut box_global: HashMap<HirId, HirId> = HashMap::new();
    let mut made: Vec<(HirId, HirId, HirCallable, HirType)> = Vec::new();
    let scope = library.map(|l| format!("${l}")).unwrap_or_default();
    for (_, _, _, gid, callee, ty) in &sites {
        if box_global.contains_key(gid) {
            continue;
        }
        let id = HirId::new();
        module.globals.insert(
            id,
            HirGlobal {
                id,
                name: InternedString::new_global(&format!(
                    "__zyntax$box{scope}${}",
                    box_global.len()
                )),
                ty: box_ty.clone(),
                initializer: None,
                is_const: false,
                is_thread_local: false,
                linkage: Linkage::Private,
                visibility: Visibility::Default,
            },
        );
        box_global.insert(*gid, id);
        made.push((*gid, id, callee.clone(), ty.clone()));
    }

    // Each use loads its global. The body is no longer the one any
    // recorded release facts describe.
    for (fid, bid, i, gid, _, ty) in &sites {
        let Some(f) = module.functions.get_mut(fid) else {
            continue;
        };
        f.attributes.release_facts = None;
        let global_ref = global_ref(f, box_global[gid], &box_ty);
        let Some(block) = f.blocks.get_mut(bid) else {
            continue;
        };
        let HirInstruction::Call {
            result: Some(result),
            ..
        } = block.instructions[*i]
        else {
            continue;
        };
        block.instructions[*i] = HirInstruction::Load {
            result,
            ty: ty.clone(),
            ptr: global_ref,
            align: 8,
            volatile: false,
        };
    }

    // The function that makes them is called by the host after codegen.
    // Keep the flag so a repeated initialization cannot replace boxes
    // already reachable from compiled code.
    let flag = HirId::new();
    module.globals.insert(
        flag,
        HirGlobal {
            id: flag,
            name: InternedString::new_global(&format!("__zyntax$boxes_made{scope}")),
            ty: HirType::I64,
            initializer: Some(HirConstant::I64(0)),
            is_const: false,
            is_thread_local: false,
            linkage: Linkage::Private,
            visibility: Visibility::Default,
        },
    );
    let name = match library {
        Some(library) => library_init_function(library),
        None => INIT_FUNCTION.to_owned(),
    };
    let init = make_init(&name, &made, &box_ty, flag, module);
    let init_id = init.id;
    module.functions.insert(init_id, init);
    sites.len()
}

/// A value naming `global` in `f`.
fn global_ref(f: &mut HirFunction, global: HirId, ty: &HirType) -> HirId {
    let id = HirId::new();
    f.values.insert(
        id,
        HirValue {
            id,
            ty: HirType::Ptr(Box::new(ty.clone())),
            kind: HirValueKind::Global(global),
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn value(f: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    f.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn make_init(
    name: &str,
    made: &[(HirId, HirId, HirCallable, HirType)],
    box_ty: &HirType,
    flag: HirId,
    module: &HirModule,
) -> HirFunction {
    let signature = HirFunctionSignature {
        params: Vec::new(),
        returns: vec![HirType::Void],
        type_params: Vec::new(),
        const_params: Vec::new(),
        lifetime_params: Vec::new(),
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: Vec::new(),
        is_pure: false,
    };
    let mut f = HirFunction::new(InternedString::new_global(name), signature);
    let entry = f.entry_block;
    let make = HirId::new();
    let done = HirId::new();

    // entry: if the flag is set, done; else make.
    let flag_ref = global_ref(&mut f, flag, &HirType::I64);
    let was_made = value(&mut f, HirType::I64, HirValueKind::Instruction);
    let zero = value(
        &mut f,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(0)),
    );
    let one = value(
        &mut f,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(1)),
    );
    let already = value(&mut f, HirType::Bool, HirValueKind::Instruction);
    let entry_block = HirBlock {
        instructions: vec![
            HirInstruction::Load {
                result: was_made,
                ty: HirType::I64,
                ptr: flag_ref,
                align: 8,
                volatile: false,
            },
            HirInstruction::Binary {
                op: BinaryOp::Ne,
                result: already,
                ty: HirType::Bool,
                left: was_made,
                right: zero,
            },
        ],
        terminator: HirTerminator::CondBranch {
            condition: already,
            true_target: done,
            false_target: make,
        },
        successors: vec![done, make],
        ..HirBlock::new(entry)
    };
    f.blocks.insert(entry, entry_block);

    let mut instructions = Vec::new();
    for (string_gid, box_gid, callee, ty) in made {
        let string_ty = module
            .globals
            .get(string_gid)
            .map(|g| g.ty.clone())
            .unwrap_or(HirType::I8);
        let string_ref = global_ref(&mut f, *string_gid, &string_ty);
        let boxed = HirId::new();
        f.values.insert(
            boxed,
            HirValue {
                id: boxed,
                ty: ty.clone(),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        instructions.push(HirInstruction::Call {
            result: Some(boxed),
            callee: callee.clone(),
            args: vec![string_ref],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        let slot = global_ref(&mut f, *box_gid, box_ty);
        instructions.push(HirInstruction::Store {
            value: boxed,
            ptr: slot,
            align: 8,
            volatile: false,
        });
    }
    let flag_ref = global_ref(&mut f, flag, &HirType::I64);
    instructions.push(HirInstruction::Store {
        value: one,
        ptr: flag_ref,
        align: 8,
        volatile: false,
    });
    f.blocks.insert(
        make,
        HirBlock {
            instructions,
            terminator: HirTerminator::Branch { target: done },
            predecessors: vec![entry],
            successors: vec![done],
            ..HirBlock::new(make)
        },
    );
    f.blocks.insert(
        done,
        HirBlock {
            terminator: HirTerminator::Return { values: Vec::new() },
            predecessors: vec![entry, make],
            ..HirBlock::new(done)
        },
    );
    f
}
