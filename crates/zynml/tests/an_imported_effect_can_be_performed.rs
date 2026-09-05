//! An effect an import brought in can be performed.
//!
//! A function that declares an effect resolves the operations it may
//! perform against the effects lowered so far. An import appends what
//! it brought to the end of the program, so every imported effect was
//! lowered after every function that could name it, and a perform of
//! one read as a call to a function that does not exist. Effects are
//! lowered before anything that can name them now.
//!
//! The op-table cases matter beyond the import. A perform names its
//! operation by position in the effect's declaration list and reads
//! that slot of the handler's table, so the two have to agree slot for
//! slot. Building the table used to drop a slot whose implementation it
//! could not find, which moved every later operation one slot early and
//! sent a perform of the last one past the end of the table, into
//! whatever followed it in memory.

use zynml::{ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::{LanguageGrammar, ZyntaxRuntime, ZyntaxValue};

const LIB: &str = r#"
effect Ports {
    def first(): i64
    def second(): i64
    def third(): i64
}

handler Wired for Ports {
    def first(): i64 { return 11 }
    def second(): i64 { return 22 }
    def third(): i64 { return 33 }
}
"#;

fn run(src: &str) -> i64 {
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "ports" => Ok(Some(LIB.to_string())),
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        _ => Ok(None),
    }));
    let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar");
    rt.register_grammar("zynml", g);
    rt.load_module("zynml", src).expect("should compile");
    // Compiled on the first call. A handler scope pushes its frame
    // through a runtime symbol the bytecode interpreter does not carry,
    // so an interpreted run says nothing about where a perform lands.
    let mut cfg = TieredConfig::default();
    cfg.profile_config = ProfileConfig {
        warm_threshold: 0,
        hot_threshold: u32::MAX as u64,
        ..Default::default()
    };
    rt.install_interp_jit_with(cfg).expect("install jit");
    match rt.call_function_raw("main", vec![]).expect("should run") {
        ZyntaxValue::Int(v) => v,
        other => panic!("expected an integer, got {other:?}"),
    }
}

/// The first operation, which is the one a broken index defaults to, so
/// it passing says the least. Here to make the others mean something.
#[test]
fn an_imported_effect_is_performable() {
    assert_eq!(
        run(r#"
import ports

@effect(Ports)
def ask(): i64 { return first() }

def main(): i64 {
    with Wired {
        return ask()
    }
}
"#),
        11
    );
}

/// The last operation of the effect. A table missing a slot sends this
/// one past its end.
#[test]
fn the_last_operation_reaches_its_own_implementation() {
    assert_eq!(
        run(r#"
import ports

@effect(Ports)
def ask(): i64 { return third() }

def main(): i64 {
    with Wired {
        return ask()
    }
}
"#),
        33
    );
}

/// Every operation, each answering for itself. One standing in for
/// another shows as a different number rather than passing by luck.
#[test]
fn every_operation_reaches_its_own_implementation() {
    assert_eq!(
        run(r#"
import ports

@effect(Ports)
def a(): i64 { return first() }

@effect(Ports)
def b(): i64 { return second() }

@effect(Ports)
def c(): i64 { return third() }

def main(): i64 {
    with Wired {
        return a() * 10000 + b() * 100 + c()
    }
}
"#),
        112233
    );
}
