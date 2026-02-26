//! ZynPEG grammar-based compilation format
//!
//! Two runtimes are supported:
//!
//! ## Grammar1 (Legacy)
//! 1. Parse .zyn grammar with ZynPEG
//! 2. Compile grammar to zpeg module (pest grammar + JSON commands)
//! 3. Parse source code with pest_vm (dynamic grammar)
//! 4. Execute JSON commands via host functions to build TypedAST
//! 5. Lower TypedAST to HIR and compile
//!
//! ## Grammar2 (Default)
//! 1. Parse .zyn grammar with ZynPEG to GrammarIR
//! 2. Use GrammarInterpreter to parse source directly to TypedProgram
//! 3. Lower TypedProgram to HIR and compile
//!
//! Usage:
//! ```bash
//! # Default (Grammar2)
//! zyntax compile --source my_code.lang --grammar my_lang.zyn --format zyn -o output --jit
//!
//! # Legacy (Grammar1)
//! zyntax compile --source my_code.lang --grammar my_lang.zyn --format zyn --grammar1 -o output --jit
//! ```

use colored::Colorize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use zyntax_compiler::hir::HirModule;
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_embed::Grammar2;
use zyntax_typed_ast::{AstArena, InternedString, TypeRegistry, TypedProgram};

/// Load and compile source using a ZynPEG grammar
pub fn load(
    grammar_path: &PathBuf,
    source_path: &PathBuf,
    verbose: bool,
) -> Result<HirModule, Box<dyn std::error::Error>> {
    if verbose {
        println!("{} Grammar file: {:?}", "info:".blue(), grammar_path);
        println!("{} Source file: {:?}", "info:".blue(), source_path);
    }

    // Verify files exist
    if !grammar_path.exists() {
        return Err(format!("Grammar file not found: {:?}", grammar_path).into());
    }
    if !source_path.exists() {
        return Err(format!("Source file not found: {:?}", source_path).into());
    }

    // Read source code
    let source_code = std::fs::read_to_string(source_path)?;
    if verbose {
        println!(
            "{} Read {} bytes of source code",
            "info:".blue(),
            source_code.len()
        );
    }

    // Read grammar
    let grammar_code = std::fs::read_to_string(grammar_path)?;
    if verbose {
        println!(
            "{} Read {} bytes of grammar",
            "info:".blue(),
            grammar_code.len()
        );
    }

    // Get the grammar name from the file
    let grammar_name = grammar_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Step 1: Compile .zyn grammar to zpeg module
    let zpeg_module = compile_grammar(&grammar_code, grammar_name, verbose)?;

    // Step 2: Parse source code using zpeg runtime
    let source_file_name = source_path.to_string_lossy().to_string();
    let typed_ast_json = parse_with_zpeg(&zpeg_module, &source_code, &source_file_name, verbose)?;

    // Step 3: Deserialize TypedAST JSON to TypedProgram
    if verbose {
        // Pretty print the JSON for debugging
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&typed_ast_json) {
            println!(
                "{} TypedAST:\n{}",
                "debug:".yellow(),
                serde_json::to_string_pretty(&parsed).unwrap_or_default()
            );
        }
    }

    let mut typed_program: TypedProgram = serde_json::from_str(&typed_ast_json)
        .map_err(|e| format!("Failed to deserialize TypedAST: {}", e))?;

    // Add the source file to the program for proper diagnostics
    use zyntax_typed_ast::source::SourceFile;
    typed_program.source_files = vec![SourceFile::new(
        source_file_name.clone(),
        source_code.clone(),
    )];

    if verbose {
        println!(
            "{} Parsed to TypedProgram with {} declarations",
            "info:".blue(),
            typed_program.declarations.len()
        );
    }

    // Step 4: Lower TypedProgram to HIR
    let hir_module = lower_to_hir(typed_program, verbose)?;

    if verbose {
        println!(
            "{} Lowered to HIR with {} functions",
            "info:".blue(),
            hir_module.functions.len()
        );
    }

    Ok(hir_module)
}

/// Load and compile source using Grammar2 runtime (GrammarInterpreter)
///
/// This is the new default runtime that uses named bindings and direct TypedAST construction.
pub fn load_grammar2(
    grammar_path: &PathBuf,
    source_path: &PathBuf,
    verbose: bool,
) -> Result<HirModule, Box<dyn std::error::Error>> {
    if verbose {
        println!("{} Grammar file: {:?}", "info:".blue(), grammar_path);
        println!("{} Source file: {:?}", "info:".blue(), source_path);
    }

    // Verify files exist
    if !grammar_path.exists() {
        return Err(format!("Grammar file not found: {:?}", grammar_path).into());
    }
    if !source_path.exists() {
        return Err(format!("Source file not found: {:?}", source_path).into());
    }

    // Read grammar
    let grammar_code = std::fs::read_to_string(grammar_path)?;
    if verbose {
        println!(
            "{} Read {} bytes of grammar",
            "info:".blue(),
            grammar_code.len()
        );
    }

    // Read source code
    let source_code = std::fs::read_to_string(source_path)?;
    if verbose {
        println!(
            "{} Read {} bytes of source code",
            "info:".blue(),
            source_code.len()
        );
    }

    // Step 1: Compile grammar with Grammar2
    if verbose {
        println!("{} Compiling grammar with Grammar2...", "info:".blue());
    }
    let grammar = Grammar2::from_source(&grammar_code)
        .map_err(|e| format!("Failed to compile grammar: {}", e))?;

    if verbose {
        println!(
            "{} Language: {} v{}",
            "info:".blue(),
            grammar.name(),
            grammar.version()
        );
    }

    // Step 2: Parse source to TypedProgram
    if verbose {
        println!("{} Parsing source with Grammar2...", "info:".blue());
    }
    let source_file_name = source_path.to_string_lossy().to_string();
    let mut typed_program = grammar
        .parse_with_filename(&source_code, &source_file_name)
        .map_err(|e| format!("Failed to parse source: {}", e))?;

    // Add the source file to the program for proper diagnostics
    use zyntax_typed_ast::source::SourceFile;
    typed_program.source_files = vec![SourceFile::new(
        source_file_name.clone(),
        source_code.clone(),
    )];

    if verbose {
        println!(
            "{} Parsed to TypedProgram with {} declarations",
            "info:".blue(),
            typed_program.declarations.len()
        );
    }

    // Step 3: Lower TypedProgram to HIR
    let hir_module = lower_to_hir(typed_program, verbose)?;

    if verbose {
        println!(
            "{} Lowered to HIR with {} functions",
            "info:".blue(),
            hir_module.functions.len()
        );
    }

    Ok(hir_module)
}

/// Compile .zyn grammar to zpeg module
fn compile_grammar(
    grammar_code: &str,
    grammar_name: &str,
    verbose: bool,
) -> Result<zyn_peg::runtime::ZpegModule, Box<dyn std::error::Error>> {
    use pest::Parser;
    use zyn_peg::ast::build_grammar;
    use zyn_peg::runtime::ZpegCompiler;
    use zyn_peg::{Rule as ZynRule, ZynGrammarParser};

    if verbose {
        println!("{} Parsing .zyn grammar with ZynPEG...", "info:".blue());
    }

    // Parse the .zyn grammar file
    let pairs = ZynGrammarParser::parse(ZynRule::program, grammar_code)
        .map_err(|e| format!("Failed to parse .zyn grammar: {}", e))?;

    let grammar = build_grammar(pairs).map_err(|e| format!("Failed to build grammar: {}", e))?;

    if verbose {
        let name = if grammar.language.name.is_empty() {
            grammar_name.to_string()
        } else {
            grammar.language.name.clone()
        };
        println!("{} Grammar '{}' parsed successfully", "info:".blue(), name);
        println!("{} Found {} rules", "info:".blue(), grammar.rules.len());
    }

    // Compile to zpeg module
    let zpeg_module = ZpegCompiler::compile(&grammar)
        .map_err(|e| format!("Failed to compile grammar to zpeg: {}", e))?;

    if verbose {
        println!("{} Compiled to zpeg module", "info:".blue());
        println!(
            "{} Pest grammar: {} bytes",
            "info:".blue(),
            zpeg_module.pest_grammar.len()
        );
        println!(
            "{} Rule commands: {} rules",
            "info:".blue(),
            zpeg_module.rules.len()
        );
    }

    Ok(zpeg_module)
}

/// Parse source code using zpeg module (via pest_vm)
fn parse_with_zpeg(
    zpeg_module: &zyn_peg::runtime::ZpegModule,
    source_code: &str,
    source_file_name: &str,
    verbose: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    use pest::iterators::{Pair, Pairs};
    use pest_meta::optimizer;
    use pest_meta::parser;
    use pest_vm::Vm;
    use zyn_peg::runtime::{AstHostFunctions, CommandInterpreter, RuntimeValue, TypedAstBuilder};

    if verbose {
        println!("{} Parsing source with pest_vm...", "info:".blue());
    }

    // Parse the pest grammar to get the grammar AST
    let pest_grammar = &zpeg_module.pest_grammar;
    let pairs = parser::parse(parser::Rule::grammar_rules, pest_grammar)
        .map_err(|e| format!("Failed to parse pest grammar: {:?}", e))?;

    // Convert to AST and optimize
    let ast = parser::consume_rules(pairs)
        .map_err(|e| format!("Failed to consume grammar rules: {:?}", e))?;
    let optimized = optimizer::optimize(ast);

    // Create VM and parse source
    let vm = Vm::new(optimized);
    let parse_result: Pairs<'_, &str> = vm
        .parse("program", source_code)
        .map_err(|e| format!("Failed to parse source: {}", e))?;

    if verbose {
        println!("{} Source parsed successfully", "info:".blue());
    }

    // Create interpreter with TypedAstBuilder host
    let mut builder = TypedAstBuilder::new();
    // Set source file information for proper span tracking
    builder.set_source(source_file_name.to_string(), source_code.to_string());
    let mut interpreter = CommandInterpreter::new(zpeg_module, builder);

    // Walk the parse tree and execute commands
    let result = walk_parse_tree(&mut interpreter, parse_result, verbose)?;

    // Finalize the AST
    let json = match result {
        RuntimeValue::Node(handle) => interpreter.host_mut().finalize_program(handle),
        _ => {
            // Create empty program if we got something unexpected
            let handle = interpreter.host_mut().create_program();
            interpreter.host_mut().finalize_program(handle)
        }
    };

    if verbose {
        println!(
            "{} Generated TypedAST JSON ({} bytes)",
            "info:".blue(),
            json.len()
        );
    }

    Ok(json)
}

/// Recursively walk the pest parse tree and execute zpeg commands
fn walk_parse_tree<'a, H: zyn_peg::runtime::AstHostFunctions>(
    interpreter: &mut zyn_peg::runtime::CommandInterpreter<'_, H>,
    pairs: pest::iterators::Pairs<'a, &'a str>,
    verbose: bool,
) -> Result<zyn_peg::runtime::RuntimeValue, Box<dyn std::error::Error>> {
    use zyn_peg::runtime::RuntimeValue;

    let mut results = Vec::new();

    for pair in pairs {
        let rule_name = pair.as_rule().to_string();
        let text = pair.as_str().to_string();
        let span_start = pair.as_span().start();
        let span_end = pair.as_span().end();

        if verbose {
            println!(
                "  {} Processing rule '{}' at {}..{}: {:?}",
                "trace:".cyan(),
                rule_name,
                span_start,
                span_end,
                if text.len() > 40 {
                    format!("{}...", &text[..40])
                } else {
                    text.clone()
                }
            );
        }

        // Set current span in interpreter and host before processing
        interpreter.set_current_span(span_start, span_end);
        interpreter
            .host_mut()
            .set_current_span(span_start, span_end);

        // Recursively process children first
        let children: Vec<RuntimeValue> = pair
            .into_inner()
            .map(|child| walk_pair_to_value(child, interpreter, verbose))
            .collect();

        if verbose && !children.is_empty() {
            println!("    {} children: {}", "trace:".cyan(), children.len());
        }

        // Execute commands for this rule
        let result = interpreter
            .execute_rule(&rule_name, &text, children)
            .map_err(|e| format!("Error executing rule '{}': {}", rule_name, e))?;

        if verbose {
            println!("    {} result: {:?}", "trace:".cyan(), result);
        }

        results.push(result);
    }

    // Return the last result (typically the program node)
    Ok(results.into_iter().last().unwrap_or(RuntimeValue::Null))
}

/// Recursively walk a single pair and return its RuntimeValue
fn walk_pair_to_value<'a, H: zyn_peg::runtime::AstHostFunctions>(
    pair: pest::iterators::Pair<'a, &'a str>,
    interpreter: &mut zyn_peg::runtime::CommandInterpreter<'_, H>,
    verbose: bool,
) -> zyn_peg::runtime::RuntimeValue {
    use zyn_peg::runtime::RuntimeValue;

    let rule_name = pair.as_rule().to_string();
    let text = pair.as_str().to_string();
    let span_start = pair.as_span().start();
    let span_end = pair.as_span().end();

    // Always trace for debugging
    log::trace!(
        "[walk_pair] rule='{}', text='{}'",
        rule_name,
        text.chars().take(50).collect::<String>()
    );

    if verbose {
        println!(
            "      {} walk_pair '{}': {:?}",
            "trace:".cyan(),
            rule_name,
            if text.len() > 30 {
                format!("{}...", &text[..30])
            } else {
                text.clone()
            }
        );
    }

    // Recursively process children first
    let children: Vec<RuntimeValue> = pair
        .into_inner()
        .map(|c| walk_pair_to_value(c, interpreter, verbose))
        .collect();

    // Set current span for THIS node (after children have been processed)
    // This ensures the span corresponds to the current rule, not a child
    interpreter.set_current_span(span_start, span_end);
    interpreter
        .host_mut()
        .set_current_span(span_start, span_end);

    // Execute commands for this rule with the correct span
    interpreter
        .execute_rule(&rule_name, &text, children)
        .unwrap_or(RuntimeValue::Null)
}

/// Lower TypedProgram to HIR module
fn lower_to_hir(
    mut program: TypedProgram,
    verbose: bool,
) -> Result<HirModule, Box<dyn std::error::Error>> {
    use zyntax_typed_ast::{type_registry::*, TypedDeclaration};

    // Register struct/class types in the type registry before lowering
    // This is needed because the Grammar2 parser creates TypedDeclaration::Class
    // but doesn't register them in the type registry
    for decl_node in &program.declarations {
        if let TypedDeclaration::Class(class) = &decl_node.node {
            // Check if type is already registered
            if program.type_registry.get_type_by_name(class.name).is_some() {
                if verbose {
                    eprintln!(
                        "[DEBUG] Type '{}' already registered, skipping",
                        class.name.resolve_global().unwrap_or_default()
                    );
                }
                continue;
            }

            // Register the struct type
            let type_id = TypeId::next();
            let fields: Vec<FieldDef> = class
                .fields
                .iter()
                .map(|f| FieldDef {
                    name: f.name,
                    ty: f.ty.clone(),
                    visibility: f.visibility,
                    mutability: f.mutability,
                    is_static: f.is_static,
                    is_synthetic: false,
                    span: f.span,
                    getter: None,
                    setter: None,
                })
                .collect();

            let type_def = TypeDefinition {
                id: type_id,
                name: class.name,
                kind: TypeKind::Struct {
                    fields: fields.clone(),
                    is_tuple: false,
                },
                type_params: vec![],
                constraints: vec![],
                fields,
                methods: vec![],
                constructors: vec![],
                metadata: Default::default(),
                span: class.span,
            };
            program.type_registry.register_type(type_def);
            if verbose {
                eprintln!(
                    "[DEBUG] Registered struct type '{}' with TypeId {:?}",
                    class.name.resolve_global().unwrap_or_default(),
                    type_id
                );
            }
        }
    }

    // Run linear type checking (ownership/borrowing validation)
    // Skip if SKIP_LINEAR_CHECK is set (for debugging or legacy code)
    if std::env::var("SKIP_LINEAR_CHECK").is_err() {
        zyntax_compiler::run_linear_type_check(&program)
            .map_err(|e| format!("Linear type check failed: {:?}", e))?;
    }

    let arena = AstArena::new();
    let module_name = InternedString::new_global("main");
    // Use the program's type registry which now has registered types
    let type_registry = Arc::new(program.type_registry.clone());

    let mut lowering_ctx = LoweringContext::new(
        module_name,
        type_registry.clone(),
        Arc::new(Mutex::new(arena)),
        LoweringConfig::default(),
    );

    // Skip type checking (parser already produced typed AST)
    std::env::set_var("SKIP_TYPE_CHECK", "1");

    let mut hir_module = lowering_ctx
        .lower_program(&mut program)
        .map_err(|e| format!("Lowering error: {:?}", e))?;

    if verbose {
        println!("{} Lowering complete", "info:".blue());
    }

    // Monomorphization
    zyntax_compiler::monomorphize_module(&mut hir_module)
        .map_err(|e| format!("Monomorphization error: {:?}", e))?;

    if verbose {
        println!("{} Monomorphization complete", "info:".blue());
    }

    Ok(hir_module)
}
