//! TypedAST JSON format loading and HIR conversion
//!
//! Phase 2: Full TypedAST → HIR conversion implementation
//! Converts language-agnostic TypedAST into Zyntax HIR for compilation.

use colored::Colorize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use zyntax_compiler::hir::{CallingConvention, HirId, HirModule, HirType};
use zyntax_compiler::hir_builder::HirBuilder;
use zyntax_typed_ast::{
    typed_ast::TypedBlock, AstArena, BinaryOp, PrimitiveType, Type, TypedDeclaration,
    TypedExpression, TypedFunction, TypedLiteral, TypedNode, TypedProgram, TypedStatement,
};

/// Load TypedAST from JSON file(s) and convert to HIR
pub fn load(inputs: &[PathBuf], verbose: bool) -> Result<HirModule, Box<dyn std::error::Error>> {
    let json_files = collect_json_files(inputs)?;

    if verbose {
        println!("{} Found {} JSON file(s)", "info:".blue(), json_files.len());
    }

    let programs = parse_json_files(&json_files, verbose)?;
    let merged_program = merge_programs(programs)?;

    if verbose {
        println!("{} Building HIR...", "info:".blue());
    }

    typed_ast_to_hir(&merged_program)
}

fn collect_json_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut json_files = Vec::new();

    for input in inputs {
        if input.is_file() {
            if input.extension().and_then(|s| s.to_str()) == Some("json") {
                json_files.push(input.clone());
            }
        } else if input.is_dir() {
            for entry in walkdir::WalkDir::new(input)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            {
                json_files.push(entry.path().to_path_buf());
            }
        }
    }

    if json_files.is_empty() {
        return Err("No JSON files found in input".into());
    }

    Ok(json_files)
}

fn parse_json_files(
    json_files: &[PathBuf],
    verbose: bool,
) -> Result<Vec<TypedProgram>, Box<dyn std::error::Error>> {
    let mut programs = Vec::new();

    for json_file in json_files {
        if verbose {
            println!("{} Parsing {}", "info:".blue(), json_file.display());
        }

        let json_content = fs::read_to_string(json_file)
            .map_err(|e| format!("Failed to read {}: {}", json_file.display(), e))?;

        let program: TypedProgram = serde_json::from_str(&json_content)
            .map_err(|e| format!("Failed to parse {}: {}", json_file.display(), e))?;

        programs.push(program);
    }

    Ok(programs)
}

fn merge_programs(programs: Vec<TypedProgram>) -> Result<TypedProgram, Box<dyn std::error::Error>> {
    if programs.is_empty() {
        return Err("No programs to merge".into());
    }

    if programs.len() == 1 {
        return Ok(programs.into_iter().next().unwrap());
    }

    // Merge all declarations
    let mut merged = programs[0].clone();
    for program in programs.into_iter().skip(1) {
        merged.declarations.extend(program.declarations);
    }

    Ok(merged)
}

/// Add runtime function declarations to the TypedProgram so lowering pass knows about them
fn add_runtime_function_declarations(program: &mut TypedProgram, arena: &mut AstArena) {
    use zyntax_typed_ast::typed_ast::{ParameterKind, TypedParameter};
    use zyntax_typed_ast::{
        AsyncKind, CallingConvention, Mutability, NullabilityKind, ParamInfo, PrimitiveType, Span,
        Type, TypedDeclaration, TypedFunction, TypedNode, Visibility,
    };

    let string_type = Type::Primitive(PrimitiveType::String);
    let i32_type = Type::Primitive(PrimitiveType::I32);
    let bool_type = Type::Primitive(PrimitiveType::Bool);
    let unit_type = Type::Primitive(PrimitiveType::Unit);
    let zero_span = Span { start: 0, end: 0 };

    // Helper to create an external function declaration
    let mut add_extern_func = |name: &str, params: Vec<(&str, Type)>, return_type: Type| {
        let func_name = arena.intern_string(name);
        let typed_params: Vec<TypedParameter> = params
            .iter()
            .map(|(pname, ty)| TypedParameter {
                name: arena.intern_string(pname),
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: zero_span,
            })
            .collect();

        let param_infos: Vec<ParamInfo> = params
            .iter()
            .map(|(pname, ty)| ParamInfo {
                name: Some(arena.intern_string(pname)),
                ty: ty.clone(),
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            })
            .collect();

        let func = TypedFunction {
            name: func_name,
            type_params: vec![],
            params: typed_params,
            return_type: return_type.clone(),
            body: None,
            visibility: Visibility::Public,
            is_async: false,
            is_external: true,
            calling_convention: CallingConvention::Cdecl,
            link_name: None,
            annotations: vec![],
            effects: vec![],
            is_pure: false,
        };

        program.declarations.insert(
            0,
            TypedNode {
                node: TypedDeclaration::Function(func),
                ty: Type::Function {
                    params: param_infos,
                    return_type: Box::new(return_type),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: AsyncKind::Sync,
                    calling_convention: CallingConvention::Cdecl,
                    nullability: NullabilityKind::NonNull,
                },
                span: zero_span,
            },
        );
    };

    // ========== String Runtime Functions ==========
    // $String$println - print string with newline
    add_extern_func(
        "$String$println",
        vec![("s", string_type.clone())],
        unit_type.clone(),
    );

    // $String$concat - concatenate two strings
    add_extern_func(
        "$String$concat",
        vec![("s1", string_type.clone()), ("s2", string_type.clone())],
        string_type.clone(),
    );

    // $String$length - get string length
    add_extern_func(
        "$String$length",
        vec![("s", string_type.clone())],
        i32_type.clone(),
    );

    // $String$charAt - get character at index
    add_extern_func(
        "$String$charAt",
        vec![("s", string_type.clone()), ("index", i32_type.clone())],
        string_type.clone(),
    );

    // $String$charCodeAt - get char code at index
    add_extern_func(
        "$String$charCodeAt",
        vec![("s", string_type.clone()), ("index", i32_type.clone())],
        i32_type.clone(),
    );

    // $String$substring - extract substring
    add_extern_func(
        "$String$substring",
        vec![
            ("s", string_type.clone()),
            ("start", i32_type.clone()),
            ("end", i32_type.clone()),
        ],
        string_type.clone(),
    );

    // $String$substr - extract substr (different semantics)
    add_extern_func(
        "$String$substr",
        vec![
            ("s", string_type.clone()),
            ("pos", i32_type.clone()),
            ("len", i32_type.clone()),
        ],
        string_type.clone(),
    );

    // $String$indexOf - find substring
    add_extern_func(
        "$String$indexOf",
        vec![("s", string_type.clone()), ("sub", string_type.clone())],
        i32_type.clone(),
    );

    // $String$lastIndexOf - find last occurrence of substring
    add_extern_func(
        "$String$lastIndexOf",
        vec![("s", string_type.clone()), ("sub", string_type.clone())],
        i32_type.clone(),
    );

    // $String$toUpperCase - convert to uppercase
    add_extern_func(
        "$String$toUpperCase",
        vec![("s", string_type.clone())],
        string_type.clone(),
    );

    // $String$toLowerCase - convert to lowercase
    add_extern_func(
        "$String$toLowerCase",
        vec![("s", string_type.clone())],
        string_type.clone(),
    );

    // $String$equals - string equality comparison
    add_extern_func(
        "$String$equals",
        vec![("s1", string_type.clone()), ("s2", string_type.clone())],
        bool_type.clone(),
    );

    // $String$fromCString - convert C string to runtime string
    add_extern_func(
        "$String$fromCString",
        vec![("cstr", string_type.clone())],
        string_type.clone(),
    );

    // ========== Array Runtime Functions ==========
    // Note: Arrays are polymorphic, but we use i32 as default element type for now
    // $Array$length - get array length
    add_extern_func(
        "$Array$length",
        vec![("arr", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$push - add element to end
    add_extern_func(
        "$Array$push",
        vec![("arr", i32_type.clone()), ("elem", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$pop - remove and return last element
    add_extern_func(
        "$Array$pop",
        vec![("arr", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$shift - remove and return first element
    add_extern_func(
        "$Array$shift",
        vec![("arr", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$unshift - add element to beginning
    add_extern_func(
        "$Array$unshift",
        vec![("arr", i32_type.clone()), ("elem", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$indexOf - find element index
    add_extern_func(
        "$Array$indexOf",
        vec![("arr", i32_type.clone()), ("elem", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$contains - check if element exists
    add_extern_func(
        "$Array$contains",
        vec![("arr", i32_type.clone()), ("elem", i32_type.clone())],
        bool_type.clone(),
    );

    // $Array$remove - remove first occurrence of element
    add_extern_func(
        "$Array$remove",
        vec![("arr", i32_type.clone()), ("elem", i32_type.clone())],
        bool_type.clone(),
    );

    // $Array$insert - insert element at index
    add_extern_func(
        "$Array$insert",
        vec![
            ("arr", i32_type.clone()),
            ("index", i32_type.clone()),
            ("elem", i32_type.clone()),
        ],
        unit_type.clone(),
    );

    // $Array$copy - create a copy of the array
    add_extern_func(
        "$Array$copy",
        vec![("arr", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$reverse - reverse array in place
    add_extern_func(
        "$Array$reverse",
        vec![("arr", i32_type.clone())],
        unit_type.clone(),
    );

    // $Array$get - get element at index
    add_extern_func(
        "$Array$get",
        vec![("arr", i32_type.clone()), ("index", i32_type.clone())],
        i32_type.clone(),
    );

    // $Array$set - set element at index
    add_extern_func(
        "$Array$set",
        vec![
            ("arr", i32_type.clone()),
            ("index", i32_type.clone()),
            ("elem", i32_type.clone()),
        ],
        unit_type.clone(),
    );

    // $Array$create - create a new array (variadic, but simplified)
    add_extern_func("$Array$create", vec![], i32_type.clone());

    // ========== Integer/Number Utility Functions ==========
    // println_i32 - print integer
    add_extern_func(
        "println_i32",
        vec![("n", i32_type.clone())],
        unit_type.clone(),
    );

    // print_i32 - print integer without newline
    add_extern_func(
        "print_i32",
        vec![("n", i32_type.clone())],
        unit_type.clone(),
    );
}

fn typed_ast_to_hir(program: &TypedProgram) -> Result<HirModule, Box<dyn std::error::Error>> {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::lowering::{LoweringConfig, LoweringContext};
    use zyntax_compiler::AstLowering;
    use zyntax_typed_ast::TypeRegistry;

    // Use the lowering pass which properly uses TypedCFG + SSA
    let mut arena = AstArena::new();

    // Add runtime function declarations to the program
    let mut augmented_program = program.clone();
    add_runtime_function_declarations(&mut augmented_program, &mut arena);

    // Run linear type checking (ownership/borrowing validation)
    // Skip if SKIP_LINEAR_CHECK is set (for debugging or legacy code)
    if std::env::var("SKIP_LINEAR_CHECK").is_err() {
        zyntax_compiler::run_linear_type_check(&augmented_program)
            .map_err(|e| format!("Linear type check failed: {:?}", e))?;
    }

    let module_name = arena.intern_string("main");
    let type_registry = Arc::new(TypeRegistry::new());
    let arena_arc = Arc::new(Mutex::new(arena));

    let config = LoweringConfig::default();
    let mut lowering_ctx = LoweringContext::new(module_name, type_registry, arena_arc, config);

    // Lower the program using the proper pipeline
    let hir_module = lowering_ctx.lower_program(&mut augmented_program)?;

    return Ok(hir_module);

    /* MANUAL APPROACH - keeping for reference but using lowering pass instead
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("main", &mut arena);

        // Register standard library functions

        // Two-pass conversion: First declare all functions, then convert bodies
        // This allows forward references

        // Pass 1: Collect all function names (both declared and called)
        let mut functions_to_convert = Vec::new();
        let mut declared_functions = std::collections::HashSet::new();
        let mut called_functions = std::collections::HashSet::new();

        for decl_node in &program.declarations {
            if let TypedDeclaration::Function(func) = &decl_node.node {
                functions_to_convert.push(func);
                if let Some(name) = func.name.resolve_global() {
                    declared_functions.insert(name);
                }

                // Scan function body for calls
                if let Some(body) = &func.body {
                    scan_for_function_calls(&body, &mut called_functions);
                }
            }
        }

        // Pass 2: Declare all functions (including missing extern ones)
        let mut function_registry = HashMap::new();

        // Declare all defined functions
        for func in &functions_to_convert {
            let return_type = convert_type(&func.return_type);
            let func_name = func.name.resolve_global()
                .ok_or_else(|| format!("Failed to resolve function name: {:?}", func.name))?;

            let mut func_builder = builder.begin_function(&func_name);

            for param in &func.params {
                let param_type = convert_type(&param.ty);
                let param_name = param.name.resolve_global()
                    .ok_or_else(|| format!("Failed to resolve parameter name: {:?}", param.name))?;
                func_builder = func_builder.param(&param_name, param_type);
            }

            func_builder = func_builder.returns(return_type.clone());
            let func_id = func_builder.build();

            function_registry.insert(func_name, func_id);
        }

        // Pre-declare runtime functions that may be generated during SSA transformation
        // (e.g., string concatenation operators)
        declare_runtime_functions(&mut builder, &mut function_registry);

        // Declare any called-but-not-defined functions as external
        for called_name in &called_functions {
            // Skip if already declared or pre-declared in runtime functions
            if !declared_functions.contains(called_name) && !function_registry.contains_key(called_name) {
                // Declare with appropriate signature based on function name
                // TODO: Properly infer signatures from call sites
                let func_id = match called_name.as_str() {
                    "println_i32" | "print_i32" => {
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("value", HirType::I32)
                            .returns(HirType::Void)
                            .build()
                    }
                    "println_i64" | "print_i64" => {
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("value", HirType::I64)
                            .returns(HirType::Void)
                            .build()
                    }
                    "print_cstr" | "puts" => {
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("ptr", HirType::Ptr(Box::new(HirType::I8)))
                            .returns(HirType::I32)
                            .build()
                    }
                    "$Array$create" => {
                        // Create array from 2 elements
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("elem0", HirType::I32)
                            .param("elem1", HirType::I32)
                            .returns(HirType::Ptr(Box::new(HirType::I32)))
                            .build()
                    }
                    "$Array$push" => {
                        // Push element, returns new pointer
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("array", HirType::Ptr(Box::new(HirType::I32)))
                            .param("element", HirType::I32)
                            .returns(HirType::Ptr(Box::new(HirType::I32)))
                            .build()
                    }
                    "$Array$get" => {
                        // Get element by index
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("array", HirType::Ptr(Box::new(HirType::I32)))
                            .param("index", HirType::I32)
                            .returns(HirType::I32)
                            .build()
                    }
                    "$Array$length" => {
                        // Get array length
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("array", HirType::Ptr(Box::new(HirType::I32)))
                            .returns(HirType::I32)
                            .build()
                    }
                    "$String$concat" => {
                        // String concatenation - returns new string pointer
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .param("str1", HirType::Ptr(Box::new(HirType::I32)))
                            .param("str2", HirType::Ptr(Box::new(HirType::I32)))
                            .returns(HirType::Ptr(Box::new(HirType::I32)))
                            .build()
                    }
                    _ => {
                        // Unknown extern - try to infer from usage
                        // For now, just declare with void signature
                        builder.begin_extern_function(called_name, CallingConvention::C)
                            .returns(HirType::Void)
                            .build()
                    }
                };
                function_registry.insert(called_name.clone(), func_id);
            }
        }

        // Pass 3: Convert each function body
        for func in functions_to_convert {
            convert_function_body(&mut builder, func, &function_registry, &arena)?;
        }

        Ok(builder.finish())
    }

    /// Convert a TypedFunction body to HIR (signature already declared)
    fn convert_function_body(
        builder: &mut HirBuilder,
        func: &TypedFunction,
        function_registry: &HashMap<String, HirId>,
        arena: &Arc<std::sync::Mutex<zyntax_typed_ast::AstArena>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Convert return type
        let return_type = convert_type(&func.return_type);

        // Get function name and look up its HirId
        let func_name = func.name.resolve_global()
            .ok_or_else(|| format!("Failed to resolve function name: {:?}", func.name))?;
        let func_id = *function_registry
            .get(&func_name)
            .ok_or_else(|| format!("Function {} not in registry", func_name))?;

        // Set as current function for body conversion
        builder.set_current_function(func_id);
        let entry = builder.entry_block();
        builder.set_insert_point(entry);

        // Convert function body if present
        if let Some(body) = &func.body {
            // Use TypedCfgBuilder + SsaBuilder for proper CFG/SSA construction
            use zyntax_compiler::typed_cfg::TypedCfgBuilder;
            use zyntax_compiler::ssa::SsaBuilder;
            use std::sync::Arc;
            use zyntax_typed_ast::TypeRegistry;

            // Build TypedCFG from function body
            let mut typed_cfg_builder = TypedCfgBuilder::new();
            let typed_cfg = typed_cfg_builder.build_from_block(body, entry)?;

            // Get the HirFunction from the builder (need to extract it temporarily)
            let mut hir_module = builder.finish();
            let mut hir_func = hir_module.functions.remove(&func_id)
                .ok_or_else(|| format!("Function {} not found in module", func_name))?;

            // Convert TypedCFG to SSA/HIR
            let type_registry = Arc::new(TypeRegistry::new());

            // Build function_symbols map with InternedString keys
            let mut function_symbols = HashMap::new();
            {
                let mut arena_guard = arena.lock().unwrap();
                for (name_str, &func_id) in function_registry {
                    let interned_name = arena_guard.intern_string(name_str);
                    function_symbols.insert(interned_name, func_id);
                }
            }

            let ssa_builder = SsaBuilder::new(hir_func, type_registry, function_symbols);
            let ssa = ssa_builder.build_from_typed_cfg(&typed_cfg)?;

            // Verify and optimize
            ssa.verify()?;
            let mut ssa = ssa;
            ssa.optimize_trivial_phis();

            hir_func = ssa.function;

            // Add string globals generated during SSA construction
            for global in ssa.string_globals {
                hir_module.globals.insert(global.id, global);
            }

            // Put the function back into the module
            hir_module.functions.insert(func_id, hir_func);

            // Re-create builder from module (this is a hack - we should restructure this)
            // For now, we'll just return early and reconstruct the builder later
            *builder = HirBuilder::from_module(hir_module, builder.arena);
        } else {
            // No body - just return void or default value
            if matches!(return_type, HirType::Void) {
                builder.ret_void();
            } else {
                let zero = builder.const_i32(0);
                builder.ret(zero);
            }
        }

        Ok(())
    }

    /// Check if block already has a return statement
    fn body_has_return_block(block: &TypedBlock) -> bool {
        block.statements.iter().any(|stmt| {
            matches!(stmt.node, TypedStatement::Return(_))
        })
    }

    /// Scan a block for all function calls
    fn scan_for_function_calls(block: &TypedBlock, called_functions: &mut std::collections::HashSet<String>) {
        for stmt in &block.statements {
            scan_statement_for_calls(&stmt.node, called_functions);
        }
    }

    /// Scan a statement for function calls
    fn scan_statement_for_calls(stmt: &TypedStatement, called_functions: &mut std::collections::HashSet<String>) {
        match stmt {
            TypedStatement::Expression(expr) => scan_expr_for_calls(&expr.node, called_functions),
            TypedStatement::Let(let_stmt) => {
                if let Some(init) = &let_stmt.initializer {
                    scan_expr_for_calls(&init.node, called_functions);
                }
            }
            TypedStatement::Return(Some(expr)) => scan_expr_for_calls(&expr.node, called_functions),
            _ => {}
        }
    }

    /// Scan an expression for function calls
    fn scan_expr_for_calls(expr: &TypedExpression, called_functions: &mut std::collections::HashSet<String>) {
        match expr {
            TypedExpression::Call(call) => {
                // Extract function name from callee
                if let TypedExpression::Variable(name) = &call.callee.node {
                    if let Some(func_name) = name.resolve_global() {
                        called_functions.insert(func_name);
                    }
                }
                // Scan arguments too
                for arg in &call.positional_args {
                    scan_expr_for_calls(&arg.node, called_functions);
                }
            }
            TypedExpression::Binary(binary) => {
                scan_expr_for_calls(&binary.left.node, called_functions);
                scan_expr_for_calls(&binary.right.node, called_functions);
            }
            _ => {}
        }
    }

    /// Convert TypedAST Type to HIR Type
    fn convert_type(ty: &Type) -> HirType {
        match ty {
            Type::Primitive(prim) => match prim {
                PrimitiveType::Unit => HirType::Void,
                PrimitiveType::I32 => HirType::I32,
                PrimitiveType::I64 => HirType::I64,
                PrimitiveType::F32 => HirType::F32,
                PrimitiveType::F64 => HirType::F64,
                PrimitiveType::Bool => HirType::Bool,
                _ => HirType::I32, // Default for other types
            },
            _ => HirType::I32, // Default for complex types
        }
    }

    /// Expression converter with variable and function tracking
    struct ExpressionConverter<'a, 'arena> {
        builder: &'a mut HirBuilder<'arena>,
        variables: HashMap<String, HirId>,
        functions: HashMap<String, HirId>,
    }

    impl<'a, 'arena> ExpressionConverter<'a, 'arena> {
        fn new(builder: &'a mut HirBuilder<'arena>, functions: HashMap<String, HirId>) -> Self {
            Self {
                builder,
                variables: HashMap::new(),
                functions,
            }
        }

        fn convert_expression(
            &mut self,
            expr_node: &TypedNode<TypedExpression>,
        ) -> Result<HirId, Box<dyn std::error::Error>> {
            match &expr_node.node {
                TypedExpression::Literal(lit) => self.convert_literal(lit),

                TypedExpression::Variable(name) => {
                    // Resolve InternedString to actual string
                    let name_str = name.resolve_global()
                        .ok_or_else(|| format!("Failed to resolve variable name: {:?}", name))?;
                    self.variables
                        .get(&name_str)
                        .copied()
                        .ok_or_else(|| format!("Undefined variable: {}", name_str).into())
                }

                TypedExpression::Binary(binary) => {
                    let left = self.convert_expression(&binary.left)?;
                    let right = self.convert_expression(&binary.right)?;
                    let hir_ty = HirType::I32; // Simplified for now

                    use zyntax_compiler::hir::BinaryOp as HirBinaryOp;

                    match binary.op {
                        // Arithmetic operators
                        BinaryOp::Add => Ok(self.builder.add(left, right, hir_ty)),
                        BinaryOp::Sub => Ok(self.builder.sub(left, right, hir_ty)),
                        BinaryOp::Mul => Ok(self.builder.mul(left, right, hir_ty)),
                        BinaryOp::Div => Ok(self.builder.div(left, right, hir_ty)),
                        // Comparison operators - use icmp
                        BinaryOp::Eq => Ok(self.builder.icmp(HirBinaryOp::Eq, left, right, hir_ty)),
                        BinaryOp::Ne => Ok(self.builder.icmp(HirBinaryOp::Ne, left, right, hir_ty)),
                        BinaryOp::Lt => Ok(self.builder.icmp(HirBinaryOp::Lt, left, right, hir_ty)),
                        BinaryOp::Le => Ok(self.builder.icmp(HirBinaryOp::Le, left, right, hir_ty)),
                        BinaryOp::Gt => Ok(self.builder.icmp(HirBinaryOp::Gt, left, right, hir_ty)),
                        BinaryOp::Ge => Ok(self.builder.icmp(HirBinaryOp::Ge, left, right, hir_ty)),
                        _ => Err(format!("Unsupported binary operator: {:?}", binary.op).into()),
                    }
                }

                TypedExpression::Call(call) => {
                    // Convert arguments (using positional_args from TypedCall)
                    let mut args = Vec::new();
                    for arg in &call.positional_args {
                        args.push(self.convert_expression(arg)?);
                    }

                    // Get callee name - for now only support simple variable references
                    let callee_name = if let TypedExpression::Variable(name) = &call.callee.node {
                        name.resolve_global()
                            .ok_or_else(|| format!("Failed to resolve function name: {:?}", name))?
                    } else {
                        return Err("Only simple function calls supported (no method calls yet)".into());
                    };

                    // Find function by name in the module
                    let func_id = self.functions.get(&callee_name)
                        .copied()
                        .or_else(|| {
                            // Try to find in builder's module (for stdlib functions)
                            let interned_name = self.builder.intern(&callee_name);
                            Some(self.builder.get_function_by_name(interned_name))
                        })
                        .ok_or_else(|| format!("Undefined function: {}", callee_name))?;

                    // Call function
                    Ok(self.builder.call(func_id, args).unwrap_or_else(|| {
                        // Void return - create dummy value
                        self.builder.const_i32(0)
                    }))
                }

                TypedExpression::Block(block) => {
                    // Convert each statement in the block
                    let mut last_value = None;
                    for stmt in &block.statements {
                        last_value = Some(self.convert_statement(stmt)?);
                    }
                    // Return the last value, or unit if empty
                    Ok(last_value.unwrap_or_else(|| self.builder.const_i32(0)))
                }

                TypedExpression::If(if_expr) => {
                    // Create the three blocks for if/else control flow
                    let then_block = self.builder.create_block("then");
                    let else_block = self.builder.create_block("else");
                    let merge_block = self.builder.create_block("merge");

                    // Convert the condition in the current block
                    let cond = self.convert_expression(&if_expr.condition)?;

                    // Emit conditional branch
                    self.builder.cond_br(cond, then_block, else_block);

                    // Fill in then_block
                    self.builder.set_insert_point(then_block);
                    let then_value = self.convert_expression(&if_expr.then_branch)?;
                    self.builder.br(merge_block);

                    // Fill in else_block
                    self.builder.set_insert_point(else_block);
                    let else_value = self.convert_expression(&if_expr.else_branch)?;
                    self.builder.br(merge_block);

                    // Set up merge block with PHI node
                    self.builder.set_insert_point(merge_block);
                    let result = self.builder.add_phi(
                        HirType::I32,
                        vec![
                            (then_value, then_block),
                            (else_value, else_block),
                        ],
                    );

                    Ok(result)
                }

                _ => {
                    // For unsupported expressions, just return a constant
                    // TODO: Implement remaining expression types
                    Ok(self.builder.const_i32(0))
                }
            }
        }

        fn convert_literal(&mut self, lit: &TypedLiteral) -> Result<HirId, Box<dyn std::error::Error>> {
            match lit {
                TypedLiteral::Integer(n) => Ok(self.builder.const_i32(*n as i32)),
                TypedLiteral::Float(f) => Ok(self.builder.const_f64(*f)),
                TypedLiteral::Bool(b) => Ok(self.builder.const_bool(*b)),
                TypedLiteral::Unit => Ok(self.builder.const_i32(0)), // Unit as 0
                TypedLiteral::String(s) => {
                    // Resolve the string from the interned value
                    let string_val = s.resolve_global()
                        .ok_or_else(|| format!("Failed to resolve string literal: {:?}", s))?;

                    // Create a null-terminated C string constant
                    let c_string = format!("{}\0", string_val);
                    let c_str_ptr = self.builder.string_constant(&c_string);

                    // Convert C string to Haxe runtime string using $String$fromCString
                    // This ensures string literals have the proper runtime representation
                    let from_cstring_name = self.builder.intern("$String$fromCString");
                    let from_cstring_fn = self.builder.get_function_by_name(from_cstring_name);

                    // Call $String$fromCString(c_str_ptr) to create a Haxe string
                    let haxe_string = self.builder.call(from_cstring_fn, vec![c_str_ptr])
                        .ok_or_else(|| format!("Failed to generate call to $String$fromCString"))?;
                    Ok(haxe_string)
                }
                _ => Err(format!("Unsupported literal: {:?}", lit).into()),
            }
        }

        fn convert_statement(
            &mut self,
            stmt_node: &TypedNode<TypedStatement>,
        ) -> Result<HirId, Box<dyn std::error::Error>> {
            match &stmt_node.node {
                TypedStatement::Expression(expr) => self.convert_expression(expr),

                TypedStatement::Return(ret_expr) => {
                    if let Some(expr) = ret_expr {
                        let value = self.convert_expression(expr)?;
                        self.builder.ret(value);
                        Ok(value)
                    } else {
                        self.builder.ret_void();
                        Ok(self.builder.const_i32(0))
                    }
                }

                TypedStatement::Let(let_stmt) => {
                    let value = if let Some(init) = &let_stmt.initializer {
                        self.convert_expression(init)?
                    } else {
                        self.builder.const_i32(0)
                    };

                    let name_str = let_stmt.name.resolve_global()
                        .ok_or_else(|| format!("Failed to resolve variable name: {:?}", let_stmt.name))?;
                    self.variables.insert(name_str, value);
                    Ok(value)
                }

                _ => Err(format!("Unsupported statement: {:?}", stmt_node.node).into()),
            }
        }

    }

    /// Declare runtime functions that may be generated during SSA transformation
    fn declare_runtime_functions(
        builder: &mut HirBuilder,
        function_registry: &mut HashMap<String, HirId>,
    ) {
        // String runtime functions
        if !function_registry.contains_key("$String$concat") {
            let func_id = builder.begin_extern_function("$String$concat", CallingConvention::C)
                .param("str1", HirType::Ptr(Box::new(HirType::I32)))
                .param("str2", HirType::Ptr(Box::new(HirType::I32)))
                .returns(HirType::Ptr(Box::new(HirType::I32)))
                .build();
            function_registry.insert("$String$concat".to_string(), func_id);
        }

        // String literal conversion function
        if !function_registry.contains_key("$String$fromCString") {
            let func_id = builder.begin_extern_function("$String$fromCString", CallingConvention::C)
                .param("cstr", HirType::Ptr(Box::new(HirType::I8)))
                .returns(HirType::Ptr(Box::new(HirType::I32)))
                .build();
            function_registry.insert("$String$fromCString".to_string(), func_id);
        }

        // String print function
        if !function_registry.contains_key("$String$println") {
            let func_id = builder.begin_extern_function("$String$println", CallingConvention::C)
                .param("str", HirType::Ptr(Box::new(HirType::I32)))
                .returns(HirType::Void)
                .build();
            function_registry.insert("$String$println".to_string(), func_id);
        }

        // Future: Add more runtime functions as needed
        // - $String$length
        // - $String$charAt
        // - $Array operations that might be generated
        // etc.
        */
}
