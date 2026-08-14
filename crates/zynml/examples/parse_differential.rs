//! Digests every `.zynml` file's parse tree, for comparing engines.
//!
//! A grammar this size has corners no unit test covers, so the corpus
//! is the evidence that the machine parses the language the
//! interpreter defines. Run it once per engine and compare:
//!
//! ```text
//! ZYNPEG_MACHINE=1 parse_differential . > machine.txt
//! ZYNPEG_MACHINE=0 parse_differential . > interpreter.txt
//! diff machine.txt interpreter.txt
//! ```
//!
//! Each engine runs in its own process on purpose. A tree carries type
//! ids handed out by a counter that keeps climbing across parses, so
//! two parses in one process disagree over ids alone and a comparison
//! made that way reports differences that are not there.
//!
//! `--tree <file>` prints one file's whole tree, for reading a
//! difference the digests turned up.

use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParsedValue, ParserState};
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_typed_ast::type_registry::TypeRegistry;
use zyntax_typed_ast::TypedASTBuilder;

fn parse(interp: &GrammarInterpreter<'_>, source: &str) -> String {
    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new(source, &mut builder, &mut registry);
    match interp.parse(&mut state) {
        // The declarations and their spans are what the parser
        // produced. The rest of a program is the type registry, whose
        // `Debug` walks a hash map in an order that changes per
        // process, so including it compares the allocator rather than
        // the parse.
        ParseResult::Success(ParsedValue::Program(program), pos) => {
            format!("ok {pos} {:?}\n{:?}", program.span, program.declarations)
        }
        ParseResult::Success(other, pos) => format!("ok {pos}\n{other:?}"),
        ParseResult::Failure(e) => format!("failed at line {} column {}", e.line, e.column),
    }
}

fn digest(tree: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tree.hash(&mut h);
    h.finish()
}

fn files(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path
                .file_name()
                .is_some_and(|n| n == "target" || n == ".git")
            {
                continue;
            }
            files(&path, out);
        } else if path.extension().is_some_and(|e| e == "zynml") {
            out.push(path);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let one_tree = args.first().is_some_and(|a| a == "--tree");
    let target = args
        .get(usize::from(one_tree))
        .cloned()
        .unwrap_or_else(|| ".".to_string());

    // The interpreter recurses through the pattern tree deeply enough
    // to outgrow the main thread's stack, the same reason the embedder
    // parses on a worker.
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
            let ir = grammar.grammar_ir();
            let interp = GrammarInterpreter::new(ir);
            let engine = match interp.program() {
                Some(program) => format!(
                    "machine: {} rules compiled, {} left to the interpreter",
                    program.supported(),
                    program.unsupported.len()
                ),
                None => "interpreter".to_string(),
            };

            if one_tree {
                let source = std::fs::read_to_string(&target).expect("read");
                println!("{}", parse(&interp, &source));
                return 0;
            }

            let mut corpus = Vec::new();
            files(std::path::Path::new(&target), &mut corpus);
            corpus.sort();
            if corpus.is_empty() {
                eprintln!("no .zynml files under {target}");
                return 1;
            }

            eprintln!("{engine}, {} files", corpus.len());
            for path in &corpus {
                let Ok(source) = std::fs::read_to_string(path) else {
                    continue;
                };
                let tree = parse(&interp, &source);
                let kind = if tree.starts_with("ok ") {
                    "ok  "
                } else {
                    "FAIL"
                };
                println!("{kind} {:016x}  {}", digest(&tree), path.display());
            }
            0
        })
        .expect("worker");

    std::process::exit(handle.join().expect("worker finished"));
}
