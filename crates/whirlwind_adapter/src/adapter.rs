//! Main adapter implementation for converting Whirlwind Standpoint to TypedAST

use crate::error::{AdapterError, AdapterResult};
use crate::expression_converter::ExpressionConverter;
use crate::statement_converter::StatementConverter;
use crate::symbol_extractor::SymbolExtractor;
use crate::type_converter::TypeConverter;
use crate::typed_expression_converter::TypedExpressionConverter;
use whirlwind_analyzer::Standpoint;
use zyntax_typed_ast::{AstArena, Type, TypeRegistry, TypedProgram};

/// The main adapter that converts Whirlwind's Standpoint IR to Zyntax's TypedAST
///
/// # Example Usage
///
/// ```rust,ignore
/// use whirlwind_adapter::WhirlwindAdapter;
///
/// // Create the adapter
/// let mut adapter = WhirlwindAdapter::new();
///
/// // Convert a Whirlwind Standpoint to TypedAST
/// let typed_program = adapter.convert_standpoint(&standpoint)?;
///
/// // Now you can pass typed_program to Zyntax's HIR converter
/// // let hir = convert_to_hir(typed_program)?;
/// ```
pub struct WhirlwindAdapter {
    /// Converts types
    type_converter: TypeConverter,

    /// Converts expressions (raw AST expressions)
    expression_converter: ExpressionConverter,

    /// Converts statements (raw AST statements)
    statement_converter: StatementConverter,

    /// Converts typed expressions (analyzed expressions with inferred types)
    typed_expression_converter: TypedExpressionConverter,

    /// Extracts type information from symbols
    symbol_extractor: SymbolExtractor,

    /// Type registry for registering and resolving types
    type_registry: TypeRegistry,

    /// String arena for interning
    arena: AstArena,
}

impl WhirlwindAdapter {
    /// Create a new Whirlwind adapter
    pub fn new() -> Self {
        let type_converter = TypeConverter::new();
        let expression_converter = ExpressionConverter::new(TypeConverter::new());
        let statement_converter = StatementConverter::new(TypeConverter::new());
        let typed_expression_converter = TypedExpressionConverter::new();
        let symbol_extractor = SymbolExtractor::new();
        let type_registry = TypeRegistry::new();

        Self {
            type_converter,
            expression_converter,
            statement_converter,
            typed_expression_converter,
            symbol_extractor,
            type_registry,
            arena: AstArena::new(),
        }
    }

    /// Convert a Whirlwind Standpoint to a TypedAST Program
    ///
    /// # Process:
    ///
    /// 1. **Extract modules** from Standpoint.module_map
    /// 2. **For each module:**
    ///    - Convert all statements to TypedDeclarations
    ///    - Build symbol table from SymbolLibrary
    ///    - Handle imports and dependencies
    /// 3. **Build TypedProgram:**
    ///    - Set entry point from Standpoint.entry_module
    ///    - Include all converted modules
    ///    - Preserve diagnostics
    ///
    /// # Arguments
    ///
    /// * `standpoint` - The Whirlwind Standpoint IR
    ///
    /// # Returns
    ///
    /// A `TypedProgram` ready for HIR conversion
    pub fn convert_standpoint(&mut self, standpoint: &Standpoint) -> AdapterResult<TypedProgram> {
        use whirlwind_analyzer::TypedStmnt;

        // First pass: Register all types in the TypeRegistry
        self.register_all_types(standpoint)?;

        let mut all_declarations = Vec::new();

        // Iterate through all modules and convert their statements
        for (_path_index, module) in standpoint.module_map.paths() {
            // Convert each TypedStmnt to our TypedDeclaration
            for typed_stmt in &module.statements {
                match typed_stmt {
                    TypedStmnt::FunctionDeclaration(func_decl) => {
                        // Convert function declarations
                        if let Ok(typed_func) = self.convert_typed_function_declaration(
                            func_decl,
                            &standpoint.symbol_library,
                            &standpoint.literals,
                        ) {
                            all_declarations.push(typed_func);
                        }
                    }
                    TypedStmnt::VariableDeclaration(var_decl) => {
                        // Convert variable declarations to top-level variables
                        if let Ok(typed_var) = self.convert_typed_variable_declaration(
                            var_decl,
                            &standpoint.symbol_library,
                            &standpoint.literals,
                        ) {
                            all_declarations.push(typed_var);
                        }
                    }
                    TypedStmnt::ShorthandVariableDeclaration(short_decl) => {
                        // Convert shorthand variable declarations
                        if let Ok(typed_var) = self.convert_typed_shorthand_variable(
                            short_decl,
                            &standpoint.symbol_library,
                            &standpoint.literals,
                        ) {
                            all_declarations.push(typed_var);
                        }
                    }
                    TypedStmnt::EnumDeclaration(enum_decl) => {
                        // Convert enum declarations
                        if let Ok(typed_enum) = self
                            .convert_typed_enum_declaration(enum_decl, &standpoint.symbol_library)
                        {
                            all_declarations.push(typed_enum);
                        }
                    }
                    TypedStmnt::TypedTypeEquation(type_eq) => {
                        // Convert type aliases
                        if let Ok(typed_alias) =
                            self.convert_typed_type_equation(type_eq, &standpoint.symbol_library)
                        {
                            all_declarations.push(typed_alias);
                        }
                    }
                    TypedStmnt::ImportDeclaration(import_decl) => {
                        // Convert import declarations
                        if let Ok(typed_import) =
                            self.convert_typed_import_declaration(import_decl, &standpoint.literals)
                        {
                            all_declarations.push(typed_import);
                        }
                    }
                    TypedStmnt::ModelDeclaration(model_decl) => {
                        // Convert class/model declarations
                        if let Ok(typed_class) = self.convert_typed_model_declaration(
                            model_decl,
                            &standpoint.symbol_library,
                            &standpoint.literals,
                        ) {
                            all_declarations.push(typed_class);
                        }
                    }
                    TypedStmnt::InterfaceDeclaration(interface_decl) => {
                        // Convert interface declarations
                        if let Ok(typed_interface) = self.convert_typed_interface_declaration(
                            interface_decl,
                            &standpoint.symbol_library,
                            &standpoint.literals,
                        ) {
                            all_declarations.push(typed_interface);
                        }
                    }
                    TypedStmnt::ModuleDeclaration(_) => {
                        // Module declarations are handled at a higher level
                    }
                    TypedStmnt::UseDeclaration(_) => {
                        // Use declarations are internal imports, already resolved
                    }
                    TypedStmnt::TestDeclaration(_) => {
                        // Skip test declarations for now
                    }
                    TypedStmnt::RecordDeclaration => {
                        // TODO: Convert record declarations
                    }
                    // Skip non-declaration statements (these would be in function bodies)
                    TypedStmnt::ExpressionStatement(_)
                    | TypedStmnt::FreeExpression(_)
                    | TypedStmnt::ReturnStatement(_)
                    | TypedStmnt::BreakStatement(_)
                    | TypedStmnt::ForStatement(_)
                    | TypedStmnt::WhileStatement(_)
                    | TypedStmnt::ContinueStatement(_) => {
                        // These are not top-level declarations
                    }
                }
            }
        }

        let span = zyntax_typed_ast::Span::default();

        let mut program = TypedProgram {
            declarations: all_declarations,
            span,
            source_files: vec![],
            type_registry: zyntax_typed_ast::TypeRegistry::new(),
        };

        // Run our type inference engine to fill in Unknown types
        self.run_type_inference(&mut program);

        Ok(program)
    }

    /// Run type inference on the converted program to fill in Unknown types
    fn run_type_inference(&mut self, program: &mut TypedProgram) {
        use zyntax_typed_ast::type_checker::{TypeCheckOptions, TypeChecker};

        // Create a type checker with our type registry
        let registry = Box::new(self.type_registry.clone());
        let mut type_checker = TypeChecker::with_options(
            registry,
            TypeCheckOptions {
                strict_nulls: false, // Be lenient since Whirlwind doesn't have strict nulls
                strict_functions: false,
                no_implicit_any: false, // Allow Unknown types
                check_unreachable: false,
            },
        );

        // Run type checking/inference
        type_checker.check_program(program);

        // Log any diagnostics (optional - for debugging)
        if type_checker.has_errors() {
            eprintln!(
                "Type inference generated {} diagnostics",
                type_checker.diagnostics().error_count()
            );
        }
    }

    /// Convert a Whirlwind TypedVariableDeclaration to our TypedDeclaration::Variable
    fn convert_typed_variable_declaration(
        &mut self,
        var_decl: &whirlwind_analyzer::TypedVariableDeclaration,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::{
            Mutability, Span, TypedDeclaration, TypedNode, TypedVariable, Visibility,
        };

        // Get the first variable name from the symbol library
        if var_decl.names.is_empty() {
            return Err(AdapterError::statement_conversion(
                "Empty variable declaration",
            ));
        }

        let symbol_idx = var_decl.names[0];
        let symbol = symbol_library
            .get(symbol_idx)
            .ok_or_else(|| AdapterError::statement_conversion("Symbol not found in library"))?;

        let var_name = self.arena.intern_string(&symbol.name);

        // Extract actual type from symbol
        let ty = self.symbol_extractor.extract_variable_type(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract visibility
        let visibility = SymbolExtractor::extract_visibility(symbol);

        // Convert the initializer expression if present
        let initializer = if let Some(ref expr) = var_decl.value {
            Some(Box::new(self.convert_typed_expression(
                expr,
                symbol_library,
                literals,
            )?))
        } else {
            None
        };

        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedNode::new(
            TypedDeclaration::Variable(TypedVariable {
                name: var_name,
                ty: ty.clone(),
                mutability: Mutability::Mutable, // Whirlwind variables are mutable by default
                initializer,
                visibility,
            }),
            ty,
            span,
        ))
    }

    /// Convert a Whirlwind TypedShorthandVariableDeclaration
    fn convert_typed_shorthand_variable(
        &mut self,
        short_decl: &whirlwind_analyzer::TypedShorthandVariableDeclaration,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::{
            Mutability, Span, TypedDeclaration, TypedNode, TypedVariable, Visibility,
        };

        let symbol_idx = short_decl.name;
        let symbol = symbol_library
            .get(symbol_idx)
            .ok_or_else(|| AdapterError::statement_conversion("Symbol not found in library"))?;

        let var_name = self.arena.intern_string(&symbol.name);

        // Extract actual type from symbol
        let ty = self.symbol_extractor.extract_variable_type(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract visibility
        let visibility = SymbolExtractor::extract_visibility(symbol);

        let initializer =
            self.convert_typed_expression(&short_decl.value, symbol_library, literals)?;
        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedNode::new(
            TypedDeclaration::Variable(TypedVariable {
                name: var_name,
                ty: ty.clone(),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(initializer)),
                visibility,
            }),
            ty,
            span,
        ))
    }

    /// Convert a Whirlwind TypedExpression to our TypedExpression
    fn convert_typed_expression(
        &mut self,
        expr: &whirlwind_analyzer::TypedExpression,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>> {
        self.typed_expression_converter.convert_typed_expression(
            expr,
            symbol_library,
            literals,
            &mut self.symbol_extractor,
            &self.type_registry,
            &mut self.arena,
        )
    }

    /// Convert a Whirlwind TypedFunctionDeclaration to our TypedDeclaration::Function
    fn convert_typed_function_declaration(
        &mut self,
        func_decl: &whirlwind_analyzer::TypedFunctionDeclaration,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::{
            Span, Type, TypedDeclaration, TypedFunction, TypedNode, Visibility,
        };

        let symbol = symbol_library
            .get(func_decl.name)
            .ok_or_else(|| AdapterError::statement_conversion("Function symbol not found"))?;

        let func_name = self.arena.intern_string(&symbol.name);

        // Extract parameters and return type from symbol
        let (params, return_type) = self.symbol_extractor.extract_function_signature(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract visibility
        let visibility = SymbolExtractor::extract_visibility(symbol);

        // Extract is_async flag
        let is_async = matches!(&symbol.kind, whirlwind_analyzer::SemanticSymbolKind::Function { is_async, .. } if *is_async);

        // Convert function body
        let body = self.typed_expression_converter.convert_typed_block(
            &func_decl.body,
            symbol_library,
            literals,
            &mut self.symbol_extractor,
            &self.type_registry,
            &mut self.arena,
        )?;

        let span = Span::default();

        Ok(TypedNode::new(
            TypedDeclaration::Function(TypedFunction {
                name: func_name,
                type_params: vec![],
                params,
                return_type: return_type.clone(),
                body: Some(body),
                visibility,
                is_async,
                is_external: false,
                calling_convention: zyntax_typed_ast::CallingConvention::Default,
                link_name: None,
                annotations: vec![],
                effects: vec![],
                is_pure: false,
            }),
            return_type,
            span,
        ))
    }

    /// Convert a Whirlwind TypedEnumDeclaration to our TypedDeclaration::Enum
    fn convert_typed_enum_declaration(
        &mut self,
        enum_decl: &whirlwind_analyzer::TypedEnumDeclaration,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::typed_ast::TypedEnum;
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedNode, Visibility};

        let symbol = symbol_library
            .get(enum_decl.name)
            .ok_or_else(|| AdapterError::statement_conversion("Enum symbol not found"))?;

        let enum_name = self.arena.intern_string(&symbol.name);

        // Extract enum variants from symbol information
        let variants = self.symbol_extractor.extract_enum_variants(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract generic type parameters
        let type_params = self.symbol_extractor.extract_generic_params(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract visibility
        let visibility = SymbolExtractor::extract_visibility(symbol);

        let span = Span::default();

        Ok(TypedNode::new(
            TypedDeclaration::Enum(TypedEnum {
                name: enum_name,
                type_params,
                variants,
                visibility,
                span,
            }),
            Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
            span,
        ))
    }

    /// Convert a Whirlwind TypedTypeEquation to our TypedDeclaration::TypeAlias
    fn convert_typed_type_equation(
        &mut self,
        type_eq: &whirlwind_analyzer::TypedTypeEquation,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::typed_ast::TypedTypeAlias;
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedNode, Visibility};

        let symbol = symbol_library
            .get(type_eq.name)
            .ok_or_else(|| AdapterError::statement_conversion("Type alias symbol not found"))?;

        let alias_name = self.arena.intern_string(&symbol.name);

        // Extract actual type from symbol information
        let target = self.symbol_extractor.extract_type_alias_target(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract generic type parameters
        let type_params = self.symbol_extractor.extract_generic_params(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;

        // Extract visibility
        let visibility = SymbolExtractor::extract_visibility(symbol);

        let span = Span::default();

        Ok(TypedNode::new(
            TypedDeclaration::TypeAlias(TypedTypeAlias {
                name: alias_name,
                type_params,
                target: target.clone(),
                visibility,
                span,
            }),
            target,
            span,
        ))
    }

    /// Convert a Whirlwind TypedImportDeclaration to our TypedDeclaration::Import
    fn convert_typed_import_declaration(
        &mut self,
        import_decl: &whirlwind_analyzer::TypedImportDeclaration,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::typed_ast::TypedImport;
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedNode};

        // Get the import source string from literals
        use whirlwind_analyzer::Literal;

        let literal = literals
            .get(import_decl.name)
            .ok_or_else(|| AdapterError::statement_conversion("Import source literal not found"))?;

        let source_str = match literal {
            Literal::StringLiteral { value, .. } => &value.value,
            _ => {
                return Err(AdapterError::statement_conversion(
                    "Import source must be a string literal",
                ))
            }
        };

        // Parse module path from source string (e.g., "foo.bar.baz" -> ["foo", "bar", "baz"])
        let module_path: Vec<_> = source_str
            .split('.')
            .map(|part| self.arena.intern_string(part))
            .collect();

        // TODO: Convert imported items from import_decl.imports
        let span = Span::default();

        Ok(TypedNode::new(
            TypedDeclaration::Import(TypedImport {
                module_path,
                items: vec![], // TODO: Convert import items
                span,
            }),
            Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
            span,
        ))
    }

    /// Convert a Whirlwind TypedModelDeclaration to our TypedDeclaration::Class
    fn convert_typed_model_declaration(
        &mut self,
        model_decl: &whirlwind_analyzer::TypedModelDeclaration,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::typed_ast::{TypedClass, TypedField, TypedMethod, TypedParameter};
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedNode, Visibility};

        let symbol = symbol_library
            .get(model_decl.name)
            .ok_or_else(|| AdapterError::statement_conversion("Model symbol not found"))?;

        let class_name = self.arena.intern_string(&symbol.name);

        // Convert properties to fields and methods
        let mut fields = Vec::new();
        let mut methods = Vec::new();

        for property in &model_decl.body.properties {
            let prop_symbol = symbol_library
                .get(property.name)
                .ok_or_else(|| AdapterError::statement_conversion("Property symbol not found"))?;

            let prop_name = self.arena.intern_string(&prop_symbol.name);

            match &property._type {
                whirlwind_analyzer::TypedModelPropertyType::TypedAttribute => {
                    // Convert to field - extract actual type from symbol
                    let field_ty = self.symbol_extractor.extract_variable_type(
                        prop_symbol,
                        symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let field_visibility = SymbolExtractor::extract_visibility(prop_symbol);

                    fields.push(TypedField {
                        name: prop_name,
                        ty: field_ty,
                        initializer: None,
                        visibility: field_visibility,
                        mutability: zyntax_typed_ast::Mutability::Mutable,
                        is_static: false,
                        span: Span::default(),
                    });
                }
                whirlwind_analyzer::TypedModelPropertyType::TypedMethod { body } => {
                    // Convert to method - extract full signature
                    let (method_params, return_type, is_static, is_async) =
                        self.symbol_extractor.extract_method_signature(
                            prop_symbol,
                            symbol_library,
                            &self.type_registry,
                            &mut self.arena,
                        )?;

                    let type_params = self.symbol_extractor.extract_generic_params(
                        prop_symbol,
                        symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let visibility = SymbolExtractor::extract_visibility(prop_symbol);
                    let method_body = self.typed_expression_converter.convert_typed_block(
                        body,
                        symbol_library,
                        literals,
                        &mut self.symbol_extractor,
                        &self.type_registry,
                        &mut self.arena,
                    )?;

                    methods.push(TypedMethod {
                        name: prop_name,
                        type_params,
                        params: method_params,
                        return_type,
                        body: Some(method_body),
                        visibility,
                        is_static,
                        is_async,
                        is_override: false,
                        span: Span::default(),
                    });
                }
                whirlwind_analyzer::TypedModelPropertyType::InterfaceImpl { body, .. } => {
                    // Interface implementation method - extract signature
                    let (method_params, return_type, is_static, is_async) =
                        self.symbol_extractor.extract_method_signature(
                            prop_symbol,
                            symbol_library,
                            &self.type_registry,
                            &mut self.arena,
                        )?;

                    let type_params = self.symbol_extractor.extract_generic_params(
                        prop_symbol,
                        symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let visibility = SymbolExtractor::extract_visibility(prop_symbol);
                    let method_body = self.typed_expression_converter.convert_typed_block(
                        body,
                        symbol_library,
                        literals,
                        &mut self.symbol_extractor,
                        &self.type_registry,
                        &mut self.arena,
                    )?;

                    methods.push(TypedMethod {
                        name: prop_name,
                        type_params,
                        params: method_params,
                        return_type,
                        body: Some(method_body),
                        visibility,
                        is_static,
                        is_async,
                        is_override: true, // This is an interface implementation
                        span: Span::default(),
                    });
                }
            }
        }

        // Extract class-level information
        let type_params = self.symbol_extractor.extract_generic_params(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;
        let (extends, implements) = self.symbol_extractor.extract_model_inheritance(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;
        let visibility = SymbolExtractor::extract_visibility(symbol);

        // Convert constructor if present
        let mut constructors = Vec::new();
        if let Some(constructor_params) = self.symbol_extractor.extract_constructor_params(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )? {
            if let Some(ref constructor_body) = model_decl.body.constructor {
                let body = self.typed_expression_converter.convert_typed_block(
                    constructor_body,
                    symbol_library,
                    literals,
                    &mut self.symbol_extractor,
                    &self.type_registry,
                    &mut self.arena,
                )?;

                constructors.push(zyntax_typed_ast::typed_ast::TypedConstructor {
                    params: constructor_params,
                    body,
                    visibility: visibility.clone(),
                    span: zyntax_typed_ast::Span::default(),
                });
            }
        }

        let span = Span::default();

        Ok(TypedNode::new(
            TypedDeclaration::Class(TypedClass {
                name: class_name,
                type_params,
                extends,
                implements,
                fields,
                methods,
                constructors,
                visibility,
                is_abstract: false,
                is_final: false,
                span,
            }),
            Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
            span,
        ))
    }

    /// Convert a Whirlwind TypedInterfaceDeclaration to our TypedDeclaration::Interface
    fn convert_typed_interface_declaration(
        &mut self,
        interface_decl: &whirlwind_analyzer::TypedInterfaceDeclaration,
        symbol_library: &whirlwind_analyzer::SymbolLibrary,
        literals: &whirlwind_analyzer::LiteralMap,
    ) -> AdapterResult<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>> {
        use zyntax_typed_ast::typed_ast::{TypedInterface, TypedMethodSignature};
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedNode, Visibility};

        let symbol = symbol_library
            .get(interface_decl.name)
            .ok_or_else(|| AdapterError::statement_conversion("Interface symbol not found"))?;

        let interface_name = self.arena.intern_string(&symbol.name);

        // Convert properties to method signatures
        let mut methods = Vec::new();

        for property in &interface_decl.body.properties {
            let prop_symbol = symbol_library.get(property.name).ok_or_else(|| {
                AdapterError::statement_conversion("Interface property symbol not found")
            })?;

            let method_name = self.arena.intern_string(&prop_symbol.name);

            // Extract method signature information
            let (method_params, return_type, is_static, is_async) =
                self.symbol_extractor.extract_method_signature(
                    prop_symbol,
                    symbol_library,
                    &self.type_registry,
                    &mut self.arena,
                )?;
            let method_type_params = self.symbol_extractor.extract_generic_params(
                prop_symbol,
                symbol_library,
                &self.type_registry,
                &mut self.arena,
            )?;

            match &property._type {
                whirlwind_analyzer::TypedInterfacePropertyType::Signature => {
                    // Abstract method signature
                    methods.push(TypedMethodSignature {
                        name: method_name,
                        type_params: method_type_params,
                        params: method_params,
                        return_type,
                        is_static,
                        is_async,
                        span: Span::default(),
                    });
                }
                whirlwind_analyzer::TypedInterfacePropertyType::Method { body } => {
                    // Default implementation - for now, just add as signature
                    // TODO: Support default implementations in interfaces
                    methods.push(TypedMethodSignature {
                        name: method_name,
                        type_params: method_type_params,
                        params: method_params,
                        return_type,
                        is_static,
                        is_async,
                        span: Span::default(),
                    });
                }
            }
        }

        // Extract interface-level information
        let type_params = self.symbol_extractor.extract_generic_params(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;
        let extends = self.symbol_extractor.extract_interface_extends(
            symbol,
            symbol_library,
            &self.type_registry,
            &mut self.arena,
        )?;
        let visibility = SymbolExtractor::extract_visibility(symbol);

        let span = Span::default();

        Ok(TypedNode::new(
            TypedDeclaration::Interface(TypedInterface {
                name: interface_name,
                type_params,
                extends,
                methods,
                associated_types: vec![],
                visibility,
                span,
            }),
            Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
            span,
        ))
    }

    /// Convert a single Whirlwind Module to TypedAST representation
    ///
    /// # Process:
    ///
    /// 1. Convert module name and path
    /// 2. Convert all statements to declarations
    /// 3. Extract imports
    /// 4. Build exports list
    fn convert_module(&mut self, _module: &str) -> AdapterResult<()> {
        // TODO: Implement module conversion

        /*
        let mut declarations = Vec::new();

        // Convert each statement in the module
        for statement in &module.statements {
            let typed_stmt = self.statement_converter.convert_statement(statement)?;

            // Wrap statement in a declaration if needed
            let typed_decl = match typed_stmt {
                TypedStatement::Function(f) => TypedDeclaration::Function(f),
                TypedStatement::Struct(s) => TypedDeclaration::Struct(s),
                TypedStatement::Enum(e) => TypedDeclaration::Enum(e),
                // ... etc
                _ => {
                    // Regular statement, not a top-level declaration
                    continue;
                }
            };

            declarations.push(typed_decl);
        }

        Ok(TypedModule {
            name: module.name.clone(),
            path: module.module_path.clone(),
            declarations,
            imports: vec![], // TODO: Extract from module.statements (Use declarations)
            exports: vec![], // TODO: Determine exports
        })
        */

        Err(AdapterError::unsupported(
            "Module conversion not yet implemented",
        ))
    }

    /// Access the type converter
    pub fn type_converter(&self) -> &TypeConverter {
        &self.type_converter
    }

    /// Access the type converter mutably
    pub fn type_converter_mut(&mut self) -> &mut TypeConverter {
        &mut self.type_converter
    }

    /// Access the expression converter
    pub fn expression_converter(&self) -> &ExpressionConverter {
        &self.expression_converter
    }

    /// Access the expression converter mutably
    pub fn expression_converter_mut(&mut self) -> &mut ExpressionConverter {
        &mut self.expression_converter
    }

    /// Access the statement converter
    pub fn statement_converter(&self) -> &StatementConverter {
        &self.statement_converter
    }

    /// Access the statement converter mutably
    pub fn statement_converter_mut(&mut self) -> &mut StatementConverter {
        &mut self.statement_converter
    }

    /// Access the type registry
    pub fn type_registry(&self) -> &TypeRegistry {
        &self.type_registry
    }

    /// Access the type registry mutably
    pub fn type_registry_mut(&mut self) -> &mut TypeRegistry {
        &mut self.type_registry
    }

    /// Access the arena for string resolution
    pub fn arena(&self) -> &AstArena {
        &self.arena
    }

    /// Access the arena mutably for string interning
    pub fn arena_mut(&mut self) -> &mut AstArena {
        &mut self.arena
    }

    /// Register all types from Whirlwind's SymbolLibrary into the TypeRegistry
    /// This is called as a first pass before converting declarations
    ///
    /// We walk the entire SymbolLibrary to capture all types (models, interfaces, enums, type aliases)
    /// regardless of where they are defined in the module hierarchy.
    ///
    /// NOTE: This only registers type *names* and basic structure. Full field/method extraction
    /// happens during HIR conversion, after all types are registered and can be resolved.
    fn register_all_types(&mut self, standpoint: &Standpoint) -> AdapterResult<()> {
        use whirlwind_analyzer::SemanticSymbolKind;
        use zyntax_typed_ast::{Span, TypeDefinition, TypeId, TypeKind, TypeMetadata};

        // Iterate through ALL symbols in the SymbolLibrary
        for (_symbol_index, symbol) in standpoint.symbol_library.symbols() {
            match &symbol.kind {
                SemanticSymbolKind::Model {
                    methods: method_indices,
                    attributes: attribute_indices,
                    ..
                } => {
                    // Register model as a Class type with full information
                    let type_name = self.arena.intern_string(&symbol.name);
                    let type_id = TypeId::next();

                    // Extract generic parameters and convert to TypeParam
                    let typed_type_params = self.symbol_extractor.extract_generic_params(
                        symbol,
                        &standpoint.symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let type_params: Vec<zyntax_typed_ast::TypeParam> = typed_type_params
                        .iter()
                        .map(|tp| {
                            zyntax_typed_ast::TypeParam {
                                name: tp.name,
                                bounds: vec![], // TODO: Extract bounds from tp.constraints
                                variance: zyntax_typed_ast::Variance::Invariant,
                                default: None,
                                span: tp.span,
                            }
                        })
                        .collect();

                    // Extract fields from attributes
                    let mut fields = Vec::new();
                    for &attr_idx in attribute_indices {
                        if let Some(attr_symbol) = standpoint.symbol_library.get(attr_idx) {
                            if let SemanticSymbolKind::Attribute {
                                declared_type,
                                is_public,
                                ..
                            } = &attr_symbol.kind
                            {
                                let field_name = self.arena.intern_string(&attr_symbol.name);
                                let field_type = self.symbol_extractor.convert_intermediate_type(
                                    declared_type,
                                    &standpoint.symbol_library,
                                    &self.type_registry,
                                    &mut self.arena,
                                )?;
                                let visibility = if *is_public {
                                    zyntax_typed_ast::Visibility::Public
                                } else {
                                    zyntax_typed_ast::Visibility::Private
                                };

                                // Convert to FieldDef for TypeRegistry
                                fields.push(zyntax_typed_ast::FieldDef {
                                    name: field_name,
                                    ty: field_type,
                                    visibility,
                                    mutability: zyntax_typed_ast::Mutability::Mutable,
                                    is_static: false,
                                    is_synthetic: false,
                                    span: Span::default(),
                                    getter: None,
                                    setter: None,
                                });
                            }
                        }
                    }

                    // Extract methods
                    let mut methods_sigs = Vec::new();
                    for &method_idx in method_indices {
                        if let Some(method_symbol) = standpoint.symbol_library.get(method_idx) {
                            let method_name = self.arena.intern_string(&method_symbol.name);
                            let (typed_params, return_type, is_static, is_async) =
                                self.symbol_extractor.extract_method_signature(
                                    method_symbol,
                                    &standpoint.symbol_library,
                                    &self.type_registry,
                                    &mut self.arena,
                                )?;

                            let is_public =
                                if let SemanticSymbolKind::Method { is_public, .. } =
                                    &method_symbol.kind
                                {
                                    *is_public
                                } else {
                                    false
                                };

                            // Convert TypedMethodParam to ParamDef for TypeRegistry
                            let params: Vec<zyntax_typed_ast::ParamDef> = typed_params
                                .iter()
                                .map(|p| zyntax_typed_ast::ParamDef {
                                    name: p.name,
                                    ty: p.ty.clone(),
                                    is_self: p.is_self,
                                    is_varargs: false,
                                    is_mut: p.mutability == zyntax_typed_ast::Mutability::Mutable,
                                })
                                .collect();

                            methods_sigs.push(zyntax_typed_ast::MethodSig {
                                name: method_name,
                                type_params: vec![], // TODO: Extract method-level generic params
                                params,
                                return_type,
                                where_clause: vec![],
                                is_static,
                                is_async,
                                visibility: if is_public {
                                    zyntax_typed_ast::Visibility::Public
                                } else {
                                    zyntax_typed_ast::Visibility::Private
                                },
                                span: Span::default(),
                                is_extension: false,
                            });
                        }
                    }

                    // Extract constructor if present
                    let mut constructors = Vec::new();
                    if let Some(constructor_params) =
                        self.symbol_extractor.extract_constructor_params(
                            symbol,
                            &standpoint.symbol_library,
                            &self.type_registry,
                            &mut self.arena,
                        )?
                    {
                        // Convert TypedMethodParam to ParamDef for ConstructorSig
                        let params: Vec<zyntax_typed_ast::ParamDef> = constructor_params
                            .iter()
                            .map(|p| zyntax_typed_ast::ParamDef {
                                name: p.name,
                                ty: p.ty.clone(),
                                is_self: p.is_self,
                                is_varargs: false,
                                is_mut: p.mutability == zyntax_typed_ast::Mutability::Mutable,
                            })
                            .collect();

                        constructors.push(zyntax_typed_ast::ConstructorSig {
                            type_params: vec![],
                            params,
                            visibility: zyntax_typed_ast::Visibility::Public,
                            span: Span::default(),
                        });
                    }

                    let type_def = TypeDefinition {
                        id: type_id,
                        name: type_name,
                        kind: TypeKind::Class,
                        type_params,
                        constraints: vec![],
                        fields,
                        methods: methods_sigs,
                        constructors,
                        metadata: TypeMetadata::default(),
                        span: Span::default(),
                    };

                    self.type_registry.register_type(type_def);
                }
                SemanticSymbolKind::Interface {
                    generic_params,
                    methods: method_indices,
                    interfaces,
                    ..
                } => {
                    // Register interface type with full information
                    let type_name = self.arena.intern_string(&symbol.name);
                    let type_id = TypeId::next();

                    // Extract generic parameters and convert to TypeParam
                    let typed_type_params = self.symbol_extractor.extract_generic_params(
                        symbol,
                        &standpoint.symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let type_params: Vec<zyntax_typed_ast::TypeParam> = typed_type_params
                        .iter()
                        .map(|tp| {
                            zyntax_typed_ast::TypeParam {
                                name: tp.name,
                                bounds: vec![], // TODO: Extract bounds from tp.constraints
                                variance: zyntax_typed_ast::Variance::Invariant,
                                default: None,
                                span: tp.span,
                            }
                        })
                        .collect();

                    // Extract super traits (interfaces this interface extends)
                    let super_traits = interfaces
                        .iter()
                        .filter_map(|int_type| {
                            self.symbol_extractor
                                .convert_intermediate_type(
                                    int_type,
                                    &standpoint.symbol_library,
                                    &self.type_registry,
                                    &mut self.arena,
                                )
                                .ok()
                        })
                        .collect();

                    // Extract methods
                    let mut methods_sigs = Vec::new();
                    for &method_idx in method_indices {
                        if let Some(method_symbol) = standpoint.symbol_library.get(method_idx) {
                            let method_name = self.arena.intern_string(&method_symbol.name);
                            let (typed_params, return_type, is_static, is_async) =
                                self.symbol_extractor.extract_method_signature(
                                    method_symbol,
                                    &standpoint.symbol_library,
                                    &self.type_registry,
                                    &mut self.arena,
                                )?;

                            let is_public =
                                if let SemanticSymbolKind::Method { is_public, .. } =
                                    &method_symbol.kind
                                {
                                    *is_public
                                } else {
                                    false
                                };

                            // Convert TypedMethodParam to ParamDef for TypeRegistry
                            let params: Vec<zyntax_typed_ast::ParamDef> = typed_params
                                .iter()
                                .map(|p| zyntax_typed_ast::ParamDef {
                                    name: p.name,
                                    ty: p.ty.clone(),
                                    is_self: p.is_self,
                                    is_varargs: false,
                                    is_mut: p.mutability == zyntax_typed_ast::Mutability::Mutable,
                                })
                                .collect();

                            methods_sigs.push(zyntax_typed_ast::MethodSig {
                                name: method_name,
                                type_params: vec![],
                                params,
                                return_type,
                                where_clause: vec![],
                                is_static,
                                is_async,
                                visibility: if is_public {
                                    zyntax_typed_ast::Visibility::Public
                                } else {
                                    zyntax_typed_ast::Visibility::Private
                                },
                                span: Span::default(),
                                is_extension: false,
                            });
                        }
                    }

                    let type_def = TypeDefinition {
                        id: type_id,
                        name: type_name,
                        kind: TypeKind::Interface {
                            methods: methods_sigs.clone(),
                            associated_types: vec![],
                            super_traits,
                        },
                        type_params,
                        constraints: vec![],
                        fields: vec![],
                        methods: methods_sigs,
                        constructors: vec![],
                        metadata: TypeMetadata::default(),
                        span: Span::default(),
                    };

                    self.type_registry.register_type(type_def);
                }
                SemanticSymbolKind::Enum {
                    generic_params,
                    variants: variant_indices,
                    ..
                } => {
                    // Register enum type with full information
                    let type_name = self.arena.intern_string(&symbol.name);
                    let type_id = TypeId::next();

                    // Extract generic parameters and convert to TypeParam
                    let typed_type_params = self.symbol_extractor.extract_generic_params(
                        symbol,
                        &standpoint.symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let type_params: Vec<zyntax_typed_ast::TypeParam> = typed_type_params
                        .iter()
                        .map(|tp| {
                            zyntax_typed_ast::TypeParam {
                                name: tp.name,
                                bounds: vec![], // TODO: Extract bounds from tp.constraints
                                variance: zyntax_typed_ast::Variance::Invariant,
                                default: None,
                                span: tp.span,
                            }
                        })
                        .collect();

                    // Extract and convert variants
                    let typed_variants = self.symbol_extractor.extract_enum_variants(
                        symbol,
                        &standpoint.symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;

                    let variants: Vec<zyntax_typed_ast::VariantDef> = typed_variants
                        .iter()
                        .map(|v| {
                            let fields = match &v.fields {
                                zyntax_typed_ast::typed_ast::TypedVariantFields::Unit => {
                                    zyntax_typed_ast::VariantFields::Unit
                                }
                                zyntax_typed_ast::typed_ast::TypedVariantFields::Tuple(types) => {
                                    zyntax_typed_ast::VariantFields::Tuple(types.clone())
                                }
                                zyntax_typed_ast::typed_ast::TypedVariantFields::Named(
                                    typed_fields,
                                ) => {
                                    let field_defs: Vec<zyntax_typed_ast::FieldDef> = typed_fields
                                        .iter()
                                        .map(|f| zyntax_typed_ast::FieldDef {
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
                                    zyntax_typed_ast::VariantFields::Named(field_defs)
                                }
                            };
                            zyntax_typed_ast::VariantDef {
                                name: v.name,
                                fields,
                                discriminant: None, // TypedVariant has discriminant as expression, we'd need to evaluate it
                                span: v.span,
                            }
                        })
                        .collect();

                    let type_def = TypeDefinition {
                        id: type_id,
                        name: type_name,
                        kind: TypeKind::Enum { variants },
                        type_params,
                        constraints: vec![],
                        fields: vec![],
                        methods: vec![],
                        constructors: vec![],
                        metadata: TypeMetadata::default(),
                        span: Span::default(),
                    };

                    self.type_registry.register_type(type_def);
                }
                SemanticSymbolKind::TypeName {
                    value,
                    generic_params: _generic_params,
                    ..
                } => {
                    // Register type alias with full information
                    let type_name = self.arena.intern_string(&symbol.name);
                    let type_id = TypeId::next();

                    // Extract generic parameters and convert to TypeParam
                    let typed_type_params = self.symbol_extractor.extract_generic_params(
                        symbol,
                        &standpoint.symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;
                    let type_params: Vec<zyntax_typed_ast::TypeParam> = typed_type_params
                        .iter()
                        .map(|tp| {
                            zyntax_typed_ast::TypeParam {
                                name: tp.name,
                                bounds: vec![], // TODO: Extract bounds from tp.constraints
                                variance: zyntax_typed_ast::Variance::Invariant,
                                default: None,
                                span: tp.span,
                            }
                        })
                        .collect();

                    // Convert the target type
                    let target = self.symbol_extractor.convert_intermediate_type(
                        value,
                        &standpoint.symbol_library,
                        &self.type_registry,
                        &mut self.arena,
                    )?;

                    let type_def = TypeDefinition {
                        id: type_id,
                        name: type_name,
                        kind: TypeKind::Alias { target },
                        type_params,
                        constraints: vec![],
                        fields: vec![],
                        methods: vec![],
                        constructors: vec![],
                        metadata: TypeMetadata::default(),
                        span: Span::default(),
                    };

                    self.type_registry.register_type(type_def);
                }
                _ => {
                    // Skip non-type symbols (variables, functions, methods, etc.)
                }
            }
        }

        Ok(())
    }
}

impl Default for WhirlwindAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = WhirlwindAdapter::new();
        // Basic smoke test
        assert!(adapter
            .type_converter()
            .lookup_type("nonexistent")
            .is_none());
    }

    #[test]
    fn test_adapter_default() {
        let _adapter = WhirlwindAdapter::default();
        // Test that default constructor works
    }
}
