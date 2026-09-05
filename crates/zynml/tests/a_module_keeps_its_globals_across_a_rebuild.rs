//! A module stays usable for as long as it is loaded.
//!
//! Loading a module can rebuild the JIT, and a rebuild throws away the
//! module object and with it the address of every global. Only what is
//! recompiled afterwards gets an address back, and that was the newest
//! module alone. Every earlier one kept its HIR and its compiled code
//! and lost its globals, so a handler declared in the first file of a
//! program stopped being installable as soon as the second file loaded.
//!
//! It has two endings, both this defect. An op-table address taken
//! before the rebuild keeps pointing where the table used to be, which
//! is some other module's data, and the perform site calls it. Ask for
//! the address after the rebuild instead and there is none to give, so
//! no handler can be installed, the perform finds no frame, and the
//! static op it falls back to reads an implicit `self` from null.
//!
//! Three files rather than two: restoring only the module that just
//! loaded is enough to make two files work, and leaves the first of
//! three broken.

use zynml::{ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE};
use zyntax_embed::{LanguageGrammar, TieredConfig, TieredRuntime};

/// The file that declares the effect, its stateful handler, and the
/// frames that perform it.
const FIRST: &str = r#"
effect HostEvents {
    def tick(): i64
    def other(): i64
}

handler Tick for HostEvents {
    var count: i64 = 41
    def tick(): i64 { self.count = self.count + 1  return self.count }
    def other(): i64 { return 0 }
}

@effect(HostEvents)
def read_tick(): i64 { return tick() }

def view(): i64 { return read_tick() }
"#;

const SECOND: &str = "def second_file(): i64 { return 2 }";
const THIRD: &str = "def third_file(): i64 { return 3 }";

fn runtime_with(files: &[&str]) -> TieredRuntime {
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        _ => Ok(None),
    }));
    let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar");
    rt.register_grammar("zynml", g);
    for f in files {
        rt.load_module("zynml", f).expect("module should load");
        // What a host does between files so the next one can link
        // against this one. This is the rebuild: it replaces the JIT
        // module and every address in it.
        rt.finalize_runtime_symbols().expect("publish symbols");
    }
    rt
}

/// The handler is declared in the first of three files and installed
/// after all three have loaded, which is the ordering the runtime asks
/// for. Its op table has to still have an address.
#[test]
fn a_handler_from_the_first_file_installs_after_the_last_one_loads() {
    let mut rt = runtime_with(&[FIRST, SECOND, THIRD]);
    let token = rt
        .get_effect_handler("Tick")
        .expect("the first file declares this handler");
    let instance = rt.new_handler_instance(token).expect("instance");
    let _frame = rt
        .push_handler_instance(instance)
        .expect("its op table should still have an address");

    assert_eq!(
        rt.call_raw("view", &[]).expect("should run"),
        zyntax_embed::ZyntaxValue::Int(42),
        "the perform should reach the handler's own state"
    );
}

/// And an instance created before the later files load, which is what a
/// host does when it builds each file's handlers as that file compiles.
/// The address its table had then is not the address it has now.
#[test]
fn an_instance_made_before_the_later_files_still_reaches_its_table() {
    let mut rt = runtime_with(&[FIRST]);
    let token = rt.get_effect_handler("Tick").expect("handler");
    let instance = rt.new_handler_instance(token).expect("instance");

    // Two more files, each of which may rebuild the JIT.
    rt.load_module("zynml", SECOND).expect("second");
    rt.load_module("zynml", THIRD).expect("third");

    let _frame = rt.push_handler_instance(instance).expect("push");
    assert_eq!(
        rt.call_raw("view", &[]).expect("should run"),
        zyntax_embed::ZyntaxValue::Int(42),
        "an instance older than the rebuild should still find its table"
    );
}

/// With nothing installed, the reachable perform is reported rather
/// than reached. The walk crosses into other modules to find it, which
/// is where a program of several files keeps most of its calls.
#[test]
fn the_reachable_perform_is_found_across_files() {
    let rt = runtime_with(&[FIRST, SECOND, THIRD]);
    assert_eq!(
        rt.missing_stateful_handler("view").map(|m| m.1),
        Some("HostEvents".to_string()),
        "`view` reaches a perform of an effect no handler is installed for"
    );
}

/// Loading the same file twice, which a host doing a reload does. The
/// restore recompiles what it kept, so keeping two copies of one module
/// would redefine everything in it.
#[test]
fn loading_the_same_file_twice_still_rebuilds() {
    let mut rt = runtime_with(&[FIRST, FIRST]);
    rt.finalize_runtime_symbols().expect("rebuild again");
    let token = rt.get_effect_handler("Tick").expect("handler");
    let instance = rt.new_handler_instance(token).expect("instance");
    rt.push_handler_instance(instance).expect("push");
}
