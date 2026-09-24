//! LLVM backend: a runtime symbol whose registered signature returns a
//! 64-bit integer, called where the HIR types the result as an address,
//! yields that address. The value reaches its uses and the function's
//! return as the same bits, never as a zero of the declared type.
#![cfg(feature = "llvm-backend")]

use inkwell::context::Context;
use zyntax_compiler::hir::{
    HirCallable, HirFunction, HirFunctionSignature, HirInstruction, HirModule, HirParam,
    HirTerminator, HirType, HirValueKind, ParamAttributes,
};
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_compiler::zrtl::{MAX_PARAMS, RuntimeSymbolInfo, TypeTag, ZrtlSigFlags, ZrtlSymbolSig};
use zyntax_typed_ast::InternedString;

fn address() -> HirType {
    HirType::Ptr(Box::new(HirType::I8))
}

/// `f(p: *i8) -> *i8 { return box(p) }`, `box` registered as i64 -> i64.
fn compile_boxing_function() -> String {
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: zyntax_compiler::hir::HirId::new(),
            name: InternedString::new_global("p"),
            ty: address(),
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
        returns: vec![address()],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let mut func = HirFunction::new(InternedString::new_global("f"), sig);
    let p = func.create_value(address(), HirValueKind::Parameter(0));
    let boxed = func.create_value(address(), HirValueKind::Instruction);
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::Call {
        result: Some(boxed),
        callee: HirCallable::Symbol("test$box".to_string()),
        args: vec![p],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    block.set_terminator(HirTerminator::Return {
        values: vec![boxed],
    });

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(func.id, func);

    let mut params = [TypeTag::VOID; MAX_PARAMS];
    params[0] = TypeTag::I64;
    let symbols = [RuntimeSymbolInfo {
        name: "test$box",
        ptr: std::ptr::null(),
        sig: Some(ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::NONE,
            return_type: TypeTag::I64,
            params,
        }),
    }];

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "symbol_result_type");
    backend.register_symbol_signatures(&symbols);
    backend
        .compile_module(&module)
        .expect("a symbol call returning an address should compile on LLVM")
}

#[test]
fn an_address_returned_as_an_integer_is_returned_as_the_address() {
    let ir = compile_boxing_function();
    assert!(
        ir.contains("inttoptr"),
        "the integer result must become the address it holds:\n{ir}"
    );
    assert!(
        !ir.contains("ret ptr null"),
        "the call's result must be what is returned:\n{ir}"
    );
}
