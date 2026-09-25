//! The worked examples in docs/BYTECODE_FORMAT_SPEC.md section 4 are the
//! bytes the serializer writes, and read back.

use zyntax_compiler::bytecode::{Format, deserialize_module, serialize_module};
use zyntax_compiler::hir::{BinaryOp, HirId, HirInstruction, HirModule, HirTerminator, HirType};
use zyntax_typed_ast::InternedString;

/// Section 4.1: an empty module with id 1 named `m`, Postcard payload.
const EMPTY_MODULE_FILE: [u8; 57] = [
    0x00, 0x43, 0x42, 0x5a, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x57, 0x08, 0x07, 0xd2, 0x01, 0x01, 0x6d, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

fn empty_module() -> HirModule {
    let mut module = HirModule::new(InternedString::new_global("m"));
    module.id = HirId::from_raw(1);
    module
}

#[test]
fn the_empty_module_example_is_what_the_serializer_writes() {
    let bytes = serialize_module(&empty_module(), Format::Postcard).expect("serializes");
    assert_eq!(bytes, EMPTY_MODULE_FILE);
}

#[test]
fn the_empty_module_example_reads_back() {
    let module = deserialize_module(&EMPTY_MODULE_FILE).expect("deserializes");
    assert_eq!(module.name.resolve_global().as_deref(), Some("m"));
    // The id is relocated on read, so only its presence is fixed.
    assert_ne!(module.id.as_u32(), 0);
    assert!(module.functions.is_empty());
    assert!(module.globals.is_empty());
    assert!(module.types.is_empty());
    assert!(module.imports.is_empty());
    assert!(module.exports.is_empty());
    assert_eq!(module.version, 0);
    assert!(module.dependencies.is_empty());
    assert!(module.effects.is_empty());
    assert!(module.handlers.is_empty());
    assert!(!module.automatic_release);
}

/// Section 4.2. Postcard is the codec `serialize_module` applies to the
/// whole module, so a value encodes the same on its own as inside one.
#[test]
fn the_instruction_examples_are_what_postcard_writes() {
    let binary = HirInstruction::Binary {
        op: BinaryOp::Add,
        result: HirId::from_raw(5),
        ty: HirType::I32,
        left: HirId::from_raw(3),
        right: HirId::from_raw(4),
    };
    let bytes = postcard::to_allocvec(&binary).expect("serializes");
    assert_eq!(bytes, [0x00, 0x00, 0x05, 0x04, 0x03, 0x04]);

    let ret = HirTerminator::Return {
        values: vec![HirId::from_raw(5)],
    };
    let bytes = postcard::to_allocvec(&ret).expect("serializes");
    assert_eq!(bytes, [0x00, 0x01, 0x05]);
}

#[test]
fn the_instruction_examples_read_back() {
    // Outside a module read no relocation is in force, so ids stay raw.
    let binary: HirInstruction =
        postcard::from_bytes(&[0x00, 0x00, 0x05, 0x04, 0x03, 0x04]).expect("deserializes");
    assert!(matches!(
        binary,
        HirInstruction::Binary {
            op: BinaryOp::Add,
            ty: HirType::I32,
            result,
            left,
            right,
        } if result.as_u32() == 5 && left.as_u32() == 3 && right.as_u32() == 4
    ));

    let ret: HirTerminator = postcard::from_bytes(&[0x00, 0x01, 0x05]).expect("deserializes");
    assert!(matches!(
        ret,
        HirTerminator::Return { ref values } if values.len() == 1 && values[0].as_u32() == 5
    ));
}
