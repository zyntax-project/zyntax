//! ZynPEG Runtime - JSON Command Pattern for TypedAST Construction
//!
//! This module provides a runtime interpreter for ZynPEG grammars that doesn't
//! require Rust compilation. Instead of generating Rust code, it:
//!
//! 1. Compiles .zyn grammar to .zpeg format (pest grammar + JSON command mappings)
//! 2. At runtime, parses source with pest and executes JSON commands via host functions
//! 3. Uses the TypedASTBuilder fluent API from zyntax_typed_ast to construct TypedProgram
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Compile Time                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  .zyn grammar  →  ZynPEG Compiler  →  .zpeg bytecode        │
//! └─────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Runtime                                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  source.lang + .zpeg  →  Runtime Interpreter  →  TypedAST  │
//! │                                                             │
//! │  Uses TypedASTBuilder fluent API to construct TypedProgram  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! // Compile grammar to zpeg format
//! let zpeg = ZpegCompiler::compile(&zyn_grammar)?;
//! zpeg.save("my_lang.zpeg")?;
//!
//! // At runtime, load and execute
//! let zpeg = ZpegModule::load("my_lang.zpeg")?;
//! let typed_ast = zpeg.parse_source(&source_code)?;
//! ```

use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, ZynPegError};
use crate::{ZynGrammar, BuiltinMappings, TypeDeclarations};

// Re-export types from typed_ast for host function implementations
pub use zyntax_typed_ast::{
    TypedASTBuilder, TypedProgram, TypedNode, TypedDeclaration, TypedExpression,
    TypedStatement, TypedBlock, BinaryOp, UnaryOp, Span, InternedString,
    TypedClass, TypedEnum, TypedField, TypedVariant,
    typed_ast::{TypedVariantFields, TypedMatchExpr, TypedMatchArm, TypedPattern, TypedLiteralPattern, TypedLiteral, TypedFieldPattern, TypedMethod, TypedMethodParam, TypedTypeParam, ParameterKind, ParameterAttribute, TypedRange, TypedExtern, TypedExternStruct, TypedTypeAlias},
    type_registry::{Type, PrimitiveType, Mutability, Visibility, ConstValue},
};

// ============================================================================
// ZPEG Module Format
// ============================================================================

/// Compiled ZynPEG module - contains pest grammar and AST builder commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZpegModule {
    /// Module metadata
    pub metadata: ZpegMetadata,
    /// The pest grammar string (embedded)
    pub pest_grammar: String,
    /// Rule definitions with their AST builder commands
    pub rules: HashMap<String, RuleCommands>,
}

/// Module metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZpegMetadata {
    /// Language name from @language directive
    pub name: String,
    /// Language version
    pub version: String,
    /// File extensions this grammar handles
    pub file_extensions: Vec<String>,
    /// Entry point function name (declared by the grammar)
    #[serde(default)]
    pub entry_point: Option<String>,
    /// ZynPEG version used to compile
    pub zpeg_version: String,
    /// Built-in mappings (functions, methods, and operators)
    /// - Functions: "println" -> "$IO$println"
    /// - Methods: "@sum" -> "tensor_sum" (x.sum() -> tensor_sum(x))
    /// - Operators: "$*" -> "vec_dot" (x * y -> vec_dot(x, y))
    #[serde(default)]
    pub builtins: BuiltinMappings,
    /// Type declarations for opaque types and function return types
    /// Used for proper type tracking during lowering for operator trait dispatch
    #[serde(default)]
    pub types: TypeDeclarations,
}

/// Commands for a single grammar rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCommands {
    /// The TypedAST type this rule produces (e.g., "TypedExpression", "TypedStatement")
    pub return_type: Option<String>,
    /// Sequence of commands to build the AST node
    pub commands: Vec<AstCommand>,
}

/// Named arguments for define commands (object-based)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NamedArgs(pub HashMap<String, CommandArg>);

/// A single AST construction command
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AstCommand {
    /// Define an AST node using a host function (preferred term over "call")
    /// e.g., {"type": "define", "node": "int_literal", "args": {"value": "$result"}}
    Define { node: String, args: NamedArgs },
    /// Legacy: Call a host function to create an AST node (deprecated, use Define)
    /// e.g., {"type": "call", "func": "binary_op", "args": ["$op", "$1", "$3"]}
    Call { func: String, args: Vec<CommandArg> },
    /// Get a child node by index (0-based) or name
    /// e.g., {"type": "get_child", "index": 0}
    GetChild {
        #[serde(default)]
        index: Option<usize>,
        #[serde(default)]
        name: Option<String>,
    },
    /// Get all children as a list (for rule* patterns)
    /// e.g., {"type": "get_all_children"}
    GetAllChildren,
    /// Get the text content of current node
    /// e.g., {"type": "get_text"}
    GetText,
    /// Parse text as integer
    /// e.g., {"type": "parse_int"}
    ParseInt,
    /// Parse text as float
    /// e.g., {"type": "parse_float"}
    ParseFloat,
    /// Create a span from current node
    /// e.g., {"type": "span"}
    Span,
    /// Fold binary operations (for left-associative parsing)
    /// e.g., {"type": "fold_binary", "operand_rule": "term", "operator_rule": "addop"}
    FoldBinary {
        operand_rule: String,
        operator_rule: String,
    },
    /// Fold postfix operations (primary followed by zero or more postfix ops)
    /// e.g., {"fold_postfix": true}
    FoldPostfix,
    /// Apply unary operator to operand (handles optional unary prefix)
    /// e.g., {"apply_unary": true}
    ApplyUnary,
    /// Fold left for binary operations with operators in the child list
    /// e.g., {"fold_left_ops": true}
    FoldLeftOps,
    /// Fold left with custom operation (for pipe operator, etc.)
    /// e.g., {"fold_left": {"op": "pipe", "transform": "prepend_arg"}}
    FoldLeft {
        op: String,
        transform: Option<String>,
    },
    /// Iterate over all children matching a rule
    /// e.g., {"type": "map_children", "rule": "statement", "commands": [...]}
    MapChildren {
        rule: String,
        commands: Vec<AstCommand>,
    },
    /// Conditional command based on rule match
    /// e.g., {"type": "match_rule", "cases": {"number": [...], "ident": [...]}}
    MatchRule {
        cases: HashMap<String, Vec<AstCommand>>,
    },
    /// Store result in a named variable
    /// e.g., {"type": "store", "name": "left"}
    Store { name: String },
    /// Load from a named variable
    /// e.g., {"type": "load", "name": "left"}
    Load { name: String },
    /// Return the current value as the rule's result
    /// e.g., {"type": "return"}
    Return,
}

/// Argument to a command - can be a literal, reference, list, or nested command
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CommandArg {
    /// Reference to a child by index: "$1", "$2", etc. (1-based for compatibility)
    ChildRef(String),
    /// String literal
    StringLit(String),
    /// Integer literal
    IntLit(i64),
    /// Boolean literal
    BoolLit(bool),
    /// List of arguments
    List(Vec<CommandArg>),
    /// Nested command that produces a value
    Nested(Box<AstCommand>),
}

// ============================================================================
// Host Functions Interface
// ============================================================================

/// Handle to an AST node being constructed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeHandle(pub u32);

/// Value that can be passed between commands
#[derive(Debug, Clone)]
pub enum RuntimeValue {
    /// Handle to an AST node
    Node(NodeHandle),
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Span value
    Span { start: usize, end: usize },
    /// List of values
    List(Vec<RuntimeValue>),
    /// Null/None value
    Null,
}

/// Host functions that the runtime calls to construct TypedAST
pub trait AstHostFunctions {
    // ========== Program Structure ==========

    /// Create a new program node
    fn create_program(&mut self) -> NodeHandle;

    /// Add a declaration to a program
    fn program_add_decl(&mut self, program: NodeHandle, decl: NodeHandle);

    /// Finalize program and return serialized TypedAST JSON
    fn finalize_program(&mut self, program: NodeHandle) -> String;

    /// Set the current source span for tracking locations
    fn set_current_span(&mut self, _start: usize, _end: usize) {
        // Default implementation does nothing - can be overridden
    }

    // ========== Functions ==========

    /// Create a function declaration
    fn create_function(
        &mut self,
        name: &str,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
        body: NodeHandle,
    ) -> NodeHandle;

    /// Create an extern function declaration (no body)
    fn create_extern_function(
        &mut self,
        name: &str,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
    ) -> NodeHandle;

    /// Create an async function declaration
    fn create_async_function(
        &mut self,
        name: &str,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
        body: NodeHandle,
    ) -> NodeHandle;

    /// Create an import declaration
    fn create_import(&mut self, module_name: &str) -> NodeHandle;

    /// Create a trait declaration
    fn create_trait(&mut self, name: &str, methods: Vec<NodeHandle>) -> NodeHandle;

    /// Create a trait implementation block
    fn create_impl_block(
        &mut self,
        trait_name: &str,
        for_type_name: &str,
        trait_args: Vec<NodeHandle>,
        items: Vec<NodeHandle>,
    ) -> NodeHandle;

    /// Create an inherent impl block for abstract types
    fn create_abstract_inherent_impl(
        &mut self,
        type_name: &str,
        underlying_type: NodeHandle,
        items: Vec<NodeHandle>,
    ) -> NodeHandle;

    /// Create an opaque type declaration (@opaque("$ExternName") type TypeName)
    fn create_opaque_type(&mut self, name: &str, external_name: &str) -> NodeHandle;

    /// Create a struct type definition (struct Name: field1: Type1, field2: Type2)
    fn create_struct_def(&mut self, name: &str, fields: Vec<NodeHandle>) -> NodeHandle;

    /// Create an abstract type definition (abstract Name(UnderlyingType) or abstract Name(UnderlyingType): fields, Suffix("x"))
    fn create_abstract_def(&mut self, name: &str, underlying_type: NodeHandle, fields: Vec<NodeHandle>, suffixes: Vec<String>) -> NodeHandle;

    /// Lookup suffix in suffix registry to find the abstract type name
    /// Returns None if suffix is not registered
    fn lookup_suffix(&self, suffix: &str) -> Option<String>;

    /// Lookup a declared type by name
    /// Returns None if type is not declared
    fn lookup_declared_type(&self, type_name: &str) -> Option<Type>;

    /// Get an expression node by handle (for extracting string literals, etc.)
    fn get_expr(&self, handle: NodeHandle) -> Option<TypedNode<TypedExpression>>;

    /// Get the type name from a type node handle (for generic types like List<T> returns "List")
    fn get_type_name(&self, handle: NodeHandle) -> Option<String>;

    /// Create a typed integer literal with explicit type annotation
    fn create_typed_int_literal(&mut self, value: i64, ty: Type) -> NodeHandle;

    /// Create a function parameter
    fn create_param(&mut self, name: &str, ty: NodeHandle) -> NodeHandle;

    // ========== Expressions ==========

    /// Create a binary operation expression
    fn create_binary_op(&mut self, op: &str, left: NodeHandle, right: NodeHandle) -> NodeHandle;

    /// Create a unary operation expression
    fn create_unary_op(&mut self, op: &str, operand: NodeHandle) -> NodeHandle;

    /// Create an integer literal
    fn create_int_literal(&mut self, value: i64) -> NodeHandle;

    /// Create a float literal
    fn create_float_literal(&mut self, value: f64) -> NodeHandle;

    /// Create a string literal
    fn create_string_literal(&mut self, value: &str) -> NodeHandle;

    /// Create a boolean literal
    fn create_bool_literal(&mut self, value: bool) -> NodeHandle;

    /// Create an identifier expression
    fn create_identifier(&mut self, name: &str) -> NodeHandle;

    /// Create a character literal
    fn create_char_literal(&mut self, value: char) -> NodeHandle;

    /// Create a variable reference expression
    fn create_variable(&mut self, name: &str) -> NodeHandle;

    /// Create a function call expression
    fn create_call(&mut self, callee: NodeHandle, args: Vec<NodeHandle>) -> NodeHandle;

    /// Create a function call expression with explicit return type
    /// Used when we know the return type from @types directive
    fn create_call_with_return_type(&mut self, callee: NodeHandle, args: Vec<NodeHandle>, return_type: Option<&str>) -> NodeHandle {
        // Default implementation ignores return type
        self.create_call(callee, args)
    }

    /// Create a function call expression with builtin and type resolution
    /// If the callee is a simple identifier matching a builtin function, use the runtime symbol instead
    /// Also looks up return types from @types directive for proper opaque type tracking
    fn create_call_with_builtin_resolution(
        &mut self,
        callee: NodeHandle,
        args: Vec<NodeHandle>,
        builtins: &BuiltinMappings,
        types: &TypeDeclarations,
    ) -> NodeHandle {
        // Check if callee is an identifier that matches a builtin function
        if let Some(name) = self.get_identifier_name(callee) {
            log::trace!("[builtin resolution] callee name='{}', checking {} builtins", name, builtins.functions.len());

            // Look up return type from @types directive
            let return_type = types.function_returns.get(&name).map(|s| s.as_str());
            if return_type.is_some() {
                log::trace!("[builtin resolution] found return type for '{}': {:?}", name, return_type);
            }

            if let Some(symbol) = builtins.functions.get(&name) {
                log::trace!("[builtin resolution] found builtin '{}' -> '{}'", name, symbol);
                // Create a new identifier with the runtime symbol name
                let resolved_callee = self.create_identifier(symbol);
                return self.create_call_with_return_type(resolved_callee, args, return_type);
            }

            // Not a builtin, but might have return type info
            return self.create_call_with_return_type(callee, args, return_type);
        } else {
            log::trace!("[builtin resolution] callee is not an identifier");
        }
        // Fall back to normal call
        self.create_call(callee, args)
    }

    /// Get the name of an identifier node, if it is one
    fn get_identifier_name(&self, handle: NodeHandle) -> Option<String> {
        None // Default implementation returns None
    }

    /// Get the value of a string literal node, if it is one
    fn get_string_literal_value(&self, handle: NodeHandle) -> Option<String> {
        None // Default implementation returns None
    }

    /// Allocate a new node handle
    fn alloc_handle(&mut self) -> NodeHandle;

    /// Store a type associated with a handle (for type resolution)
    fn store_type(&mut self, handle: NodeHandle, ty: zyntax_typed_ast::type_registry::Type);

    /// Create a method call expression
    fn create_method_call(&mut self, receiver: NodeHandle, method: &str, args: Vec<NodeHandle>) -> NodeHandle;

    /// Create a static method call expression: TypeName::method(args)
    fn create_static_method_call(&mut self, type_name: &str, method: &str, args: Vec<NodeHandle>) -> NodeHandle;

    /// Create an array/index access expression
    fn create_index(&mut self, array: NodeHandle, index: NodeHandle) -> NodeHandle;

    /// Create a field access expression
    fn create_field_access(&mut self, object: NodeHandle, field: &str) -> NodeHandle;

    /// Create an array literal expression
    fn create_array(&mut self, elements: Vec<NodeHandle>) -> NodeHandle;

    /// Create a struct literal expression
    fn create_struct_literal(&mut self, name: &str, fields: Vec<(String, NodeHandle)>) -> NodeHandle;

    /// Store a struct field initialization (for later lookup by struct_init)
    fn store_struct_field_init(&mut self, name: &str, value: NodeHandle) -> NodeHandle;

    /// Get a stored struct field initialization
    fn get_struct_field_init(&self, handle: NodeHandle) -> Option<(String, NodeHandle)>;

    /// Create a cast expression
    fn create_cast(&mut self, expr: NodeHandle, target_type: NodeHandle) -> NodeHandle;

    /// Create a lambda expression
    fn create_lambda(&mut self, params: Vec<NodeHandle>, body: NodeHandle) -> NodeHandle;

    /// Create an await expression (for async functions)
    fn create_await(&mut self, expr: NodeHandle) -> NodeHandle;

    // ========== Statements ==========

    /// Create a variable declaration statement
    fn create_var_decl(
        &mut self,
        name: &str,
        ty: Option<NodeHandle>,
        init: Option<NodeHandle>,
        is_const: bool,
    ) -> NodeHandle;

    /// Create an assignment statement
    fn create_assignment(&mut self, target: NodeHandle, value: NodeHandle) -> NodeHandle;

    /// Create a return statement
    fn create_return(&mut self, value: Option<NodeHandle>) -> NodeHandle;

    /// Create an if statement
    fn create_if(
        &mut self,
        condition: NodeHandle,
        then_branch: NodeHandle,
        else_branch: Option<NodeHandle>,
    ) -> NodeHandle;

    /// Create a while loop
    fn create_while(&mut self, condition: NodeHandle, body: NodeHandle) -> NodeHandle;

    /// Create a for loop
    fn create_for(&mut self, iterator: &str, iterable: NodeHandle, body: NodeHandle) -> NodeHandle;

    /// Create a block statement
    fn create_block(&mut self, statements: Vec<NodeHandle>) -> NodeHandle;

    /// Create an expression statement
    fn create_expr_stmt(&mut self, expr: NodeHandle) -> NodeHandle;

    /// Create a let/variable declaration (alias for create_var_decl)
    fn create_let(
        &mut self,
        name: &str,
        ty: Option<NodeHandle>,
        init: Option<NodeHandle>,
        is_const: bool,
    ) -> NodeHandle;

    /// Create a break statement
    fn create_break(&mut self, value: Option<NodeHandle>) -> NodeHandle;

    /// Create a continue statement
    fn create_continue(&mut self) -> NodeHandle;

    /// Create an expression statement (alias)
    fn create_expression_stmt(&mut self, expr: NodeHandle) -> NodeHandle;

    /// Create a range expression (start..end or start...end)
    fn create_range(&mut self, start: Option<NodeHandle>, end: Option<NodeHandle>, inclusive: bool) -> NodeHandle;

    /// Apply a postfix operation (call, field access, index) to a base expression
    fn apply_postfix(&mut self, base: NodeHandle, postfix_op: NodeHandle) -> NodeHandle;

    // ========== Pattern Matching ==========

    /// Create a match/switch expression
    fn create_match_expr(&mut self, scrutinee: NodeHandle, arms: Vec<NodeHandle>) -> NodeHandle;

    /// Create a match arm (pattern => body)
    fn create_match_arm(&mut self, pattern: NodeHandle, body: NodeHandle) -> NodeHandle;

    /// Create a literal pattern (matches a specific value)
    fn create_literal_pattern(&mut self, value: NodeHandle) -> NodeHandle;

    /// Create a wildcard pattern (matches anything, like _ or else)
    fn create_wildcard_pattern(&mut self) -> NodeHandle;

    /// Create an identifier pattern (variable binding)
    fn create_identifier_pattern(&mut self, name: &str) -> NodeHandle;

    /// Create a struct pattern: Point { x, y } or Point { x: px }
    fn create_struct_pattern(&mut self, name: &str, fields: Vec<NodeHandle>) -> NodeHandle;

    /// Create a struct field pattern: x or x: pattern
    fn create_field_pattern(&mut self, name: &str, pattern: Option<NodeHandle>) -> NodeHandle;

    /// Create an enum/variant pattern: Some(x), None, Ok(v)
    fn create_enum_pattern(&mut self, name: &str, variant: &str, fields: Vec<NodeHandle>) -> NodeHandle;

    /// Create an array pattern: [x, y, z]
    fn create_array_pattern(&mut self, elements: Vec<NodeHandle>) -> NodeHandle;

    /// Create a tuple pattern: (x, y, _)
    fn create_tuple_pattern(&mut self, elements: Vec<NodeHandle>) -> NodeHandle;

    /// Create a range pattern: 1..10 or 1..=10
    fn create_range_pattern(&mut self, start: NodeHandle, end: NodeHandle, inclusive: bool) -> NodeHandle;

    /// Create an or pattern: x | y | z
    fn create_or_pattern(&mut self, patterns: Vec<NodeHandle>) -> NodeHandle;

    /// Create a pointer/reference pattern: *x, &x
    fn create_pointer_pattern(&mut self, inner: NodeHandle, mutable: bool) -> NodeHandle;

    /// Create a slice pattern: arr[0..5]
    fn create_slice_pattern(&mut self, prefix: Vec<NodeHandle>, middle: Option<NodeHandle>, suffix: Vec<NodeHandle>) -> NodeHandle;

    /// Create an error pattern: error.X (Zig-style)
    fn create_error_pattern(&mut self, error_name: &str) -> NodeHandle;

    // ========== Types ==========

    /// Create a primitive type (i32, i64, f32, f64, bool, etc.)
    fn create_primitive_type(&mut self, name: &str) -> NodeHandle;

    /// Create a pointer type
    fn create_pointer_type(&mut self, pointee: NodeHandle) -> NodeHandle;

    /// Create an array type
    fn create_array_type(&mut self, element: NodeHandle, size: Option<usize>) -> NodeHandle;

    /// Create a function type
    fn create_function_type(
        &mut self,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
    ) -> NodeHandle;

    /// Create a named/user type
    fn create_named_type(&mut self, name: &str) -> NodeHandle;

    // ========== Struct/Enum Declarations ==========

    /// Create a struct declaration
    fn create_struct(&mut self, name: &str, fields: Vec<NodeHandle>) -> NodeHandle;

    /// Create an enum declaration
    fn create_enum(&mut self, name: &str, variants: Vec<NodeHandle>) -> NodeHandle;

    /// Create a struct field
    fn create_field(&mut self, name: &str, ty: NodeHandle) -> NodeHandle;

    /// Create an enum variant
    fn create_variant(&mut self, name: &str) -> NodeHandle;

    // ========== Class/OOP Declarations (Haxe) ==========

    /// Create a class declaration with type parameters and members
    fn create_class(&mut self, name: &str, type_params: Vec<String>, members: Vec<NodeHandle>) -> NodeHandle;

    /// Create a method declaration
    fn create_method(
        &mut self,
        name: &str,
        is_static: bool,
        visibility: &str,
        params: Vec<NodeHandle>,
        return_type: Option<NodeHandle>,
        body: NodeHandle,
    ) -> NodeHandle;

    /// Create a ternary conditional expression (condition ? then_expr : else_expr)
    fn create_ternary(&mut self, condition: NodeHandle, then_expr: NodeHandle, else_expr: NodeHandle) -> NodeHandle;

    // ========== Span/Location ==========

    /// Set span on a node
    fn set_span(&mut self, node: NodeHandle, start: usize, end: usize);
}

// ============================================================================
// ZPEG Compiler - Converts ZynGrammar to ZpegModule
// ============================================================================

/// Compiler that converts ZynGrammar to ZpegModule
pub struct ZpegCompiler;

impl ZpegCompiler {
    /// Compile a ZynGrammar to a ZpegModule
    pub fn compile(grammar: &ZynGrammar) -> Result<ZpegModule> {
        let metadata = ZpegMetadata {
            name: grammar.language.name.clone(),
            version: grammar.language.version.clone(),
            file_extensions: grammar.language.file_extensions.clone(),
            entry_point: grammar.language.entry_point.clone(),
            zpeg_version: env!("CARGO_PKG_VERSION").to_string(),
            builtins: grammar.builtins.clone(),
            types: grammar.types.clone(),
        };

        // Generate pest grammar
        let pest_grammar = crate::generator::generate_pest_grammar_string(grammar)?;

        // Convert rules to commands
        let mut rules = HashMap::new();
        for rule in &grammar.rules {
            let commands = Self::compile_rule(rule)?;
            rules.insert(rule.name.clone(), commands);
        }

        Ok(ZpegModule {
            metadata,
            pest_grammar,
            rules,
        })
    }

    /// Compile a single rule's action block to commands
    fn compile_rule(rule: &crate::RuleDef) -> Result<RuleCommands> {
        let return_type = rule.action.as_ref().map(|a| a.return_type.clone());

        let commands = if let Some(action) = &rule.action {
            Self::compile_action(action)?
        } else {
            // No action block - just pass through child
            vec![AstCommand::GetChild {
                index: Some(0),
                name: None,
            }]
        };

        Ok(RuleCommands {
            return_type,
            commands,
        })
    }

    /// Compile an action block to commands
    fn compile_action(action: &crate::ActionBlock) -> Result<Vec<AstCommand>> {
        let mut commands = Vec::new();

        // Prefer JSON commands if available (new format)
        if let Some(json_str) = &action.json_commands {
            commands.extend(Self::parse_json_commands(json_str)?);
        }
        // Otherwise, if there's raw code, parse it into commands (legacy)
        else if let Some(raw) = &action.raw_code {
            commands.extend(Self::parse_raw_action(raw)?);
        } else {
            // Structured fields - generate commands for each
            for field in &action.fields {
                commands.push(Self::compile_field_value(&field.name, &field.value)?);
            }
        }

        // Add return at the end
        commands.push(AstCommand::Return);

        Ok(commands)
    }

    /// Parse JSON commands from action block
    fn parse_json_commands(json_str: &str) -> Result<Vec<AstCommand>> {
        // Parse the JSON - can be an array of commands or a single object
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| ZynPegError::ParseError(format!("Invalid JSON in action block: {}", e)))?;

        let mut commands = Vec::new();

        // Support array of command objects
        if let serde_json::Value::Array(arr) = value {
            for item in arr {
                if let serde_json::Value::Object(map) = item {
                    commands.extend(Self::parse_json_object_commands(&map)?);
                }
            }
            return Ok(commands);
        }

        // Single object (original format or "commands" wrapper)
        if let serde_json::Value::Object(map) = value {
            // Check for "commands": [...] wrapper format
            if let Some(serde_json::Value::Array(arr)) = map.get("commands") {
                for item in arr {
                    if let serde_json::Value::Object(cmd_map) = item {
                        commands.extend(Self::parse_json_object_commands(cmd_map)?);
                    }
                }
            } else {
                commands.extend(Self::parse_json_object_commands(&map)?);
            }
        }

        Ok(commands)
    }

    /// Parse commands from a single JSON object
    fn parse_json_object_commands(map: &serde_json::Map<String, serde_json::Value>) -> Result<Vec<AstCommand>> {
        let mut commands = Vec::new();

        if true {  // Scope for borrowing map
            // Check for different command types

            // "get_child": { "index": 0 }
            if let Some(val) = map.get("get_child") {
                let index = val.get("index").and_then(|v| v.as_u64()).map(|n| n as usize);
                let name = val.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
                commands.push(AstCommand::GetChild { index, name });
            }

            // "get_all_children": true - collect all children as a list
            // Can also include "store": "var_name" to store the result
            if map.get("get_all_children").is_some() {
                commands.push(AstCommand::GetAllChildren);
                // If "store" field exists, add a Store command after get_all_children
                if let Some(store_name) = map.get("store").and_then(|v| v.as_str()) {
                    commands.push(AstCommand::Store { name: store_name.to_string() });
                }
            }

            // "get_text": true
            if map.get("get_text").is_some() {
                commands.push(AstCommand::GetText);
            }

            // "parse_int": true
            if map.get("parse_int").is_some() {
                commands.push(AstCommand::ParseInt);
            }

            // "parse_float": true
            if map.get("parse_float").is_some() {
                commands.push(AstCommand::ParseFloat);
            }

            // "define": "node_type", "args": {...}, "store": "var_name" (optional)
            // This is the preferred new format with named arguments
            if let Some(node_type) = map.get("define").and_then(|v| v.as_str()) {
                let args = if let Some(serde_json::Value::Object(obj)) = map.get("args") {
                    let mut named_args = HashMap::new();
                    for (key, val) in obj {
                        named_args.insert(key.clone(), Self::json_to_command_arg(val));
                    }
                    NamedArgs(named_args)
                } else {
                    NamedArgs::default()
                };
                commands.push(AstCommand::Define {
                    node: node_type.to_string(),
                    args,
                });
                // If "store" field exists, add a Store command after the define
                if let Some(store_name) = map.get("store").and_then(|v| v.as_str()) {
                    commands.push(AstCommand::Store { name: store_name.to_string() });
                }
            }

            // "call": "func_name", "args": [...], "store": "var_name" (optional)
            // Legacy format with positional arguments (deprecated but still supported)
            if let Some(func) = map.get("call").and_then(|v| v.as_str()) {
                let args = if let Some(serde_json::Value::Array(arr)) = map.get("args") {
                    arr.iter().map(|v| Self::json_to_command_arg(v)).collect()
                } else {
                    vec![]
                };
                commands.push(AstCommand::Call {
                    func: func.to_string(),
                    args,
                });
                // If "store" field exists, add a Store command after the call
                if let Some(store_name) = map.get("store").and_then(|v| v.as_str()) {
                    commands.push(AstCommand::Store { name: store_name.to_string() });
                }
            }

            // "fold_binary": { "operand": "term", "operator_map": {...} }
            if let Some(val) = map.get("fold_binary") {
                let operand_rule = val.get("operand")
                    .or(val.get("operand_rule"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("operand")
                    .to_string();
                let operator_rule = val.get("operator")
                    .or(val.get("operator_rule"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("operator")
                    .to_string();
                commands.push(AstCommand::FoldBinary { operand_rule, operator_rule });
            }

            // "match_rule": { "rule_name": [...], ... }
            if let Some(serde_json::Value::Object(cases_map)) = map.get("match_rule") {
                let mut cases = std::collections::HashMap::new();
                for (rule_name, case_cmds) in cases_map {
                    if let serde_json::Value::Array(arr) = case_cmds {
                        let cmds: Vec<AstCommand> = arr.iter()
                            .filter_map(|v| Self::json_value_to_command(v).ok())
                            .collect();
                        cases.insert(rule_name.clone(), cmds);
                    }
                }
                commands.push(AstCommand::MatchRule { cases });
            }

            // "store": { "name": "var_name" }
            if let Some(val) = map.get("store") {
                if let Some(name) = val.get("name").and_then(|v| v.as_str()) {
                    commands.push(AstCommand::Store { name: name.to_string() });
                }
            }

            // "load": { "name": "var_name" }
            if let Some(val) = map.get("load") {
                if let Some(name) = val.get("name").and_then(|v| v.as_str()) {
                    commands.push(AstCommand::Load { name: name.to_string() });
                }
            }

            // "fold_postfix": true
            if map.get("fold_postfix").is_some() {
                commands.push(AstCommand::FoldPostfix);
            }

            // "apply_unary": true
            if map.get("apply_unary").is_some() {
                commands.push(AstCommand::ApplyUnary);
            }

            // "fold_left_ops": true
            if map.get("fold_left_ops").is_some() {
                commands.push(AstCommand::FoldLeftOps);
            }

            // "fold_left": { "op": "pipe", "transform": "prepend_arg" }
            if let Some(val) = map.get("fold_left") {
                let op = val.get("op")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let transform = val.get("transform")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                commands.push(AstCommand::FoldLeft { op, transform });
            }
        }

        Ok(commands)
    }

    /// Convert a JSON value to a command argument
    fn json_to_command_arg(value: &serde_json::Value) -> CommandArg {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with('$') {
                    CommandArg::ChildRef(s.clone())
                } else {
                    CommandArg::StringLit(s.clone())
                }
            }
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    CommandArg::IntLit(i)
                } else {
                    CommandArg::StringLit(n.to_string())
                }
            }
            serde_json::Value::Bool(b) => CommandArg::BoolLit(*b),
            serde_json::Value::Array(arr) => {
                // Convert array to a list of command args
                CommandArg::List(arr.iter().map(Self::json_to_command_arg).collect())
            }
            serde_json::Value::Object(_) => {
                // Try to convert to a nested command
                if let Ok(cmd) = Self::json_value_to_command(value) {
                    CommandArg::Nested(Box::new(cmd))
                } else {
                    CommandArg::StringLit(value.to_string())
                }
            }
            _ => CommandArg::StringLit(value.to_string()),
        }
    }

    /// Convert a JSON value to an AstCommand
    fn json_value_to_command(value: &serde_json::Value) -> Result<AstCommand> {
        if let serde_json::Value::Object(map) = value {
            if let Some(val) = map.get("get_child") {
                let index = val.get("index").and_then(|v| v.as_u64()).map(|n| n as usize);
                let name = val.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
                return Ok(AstCommand::GetChild { index, name });
            }
            if map.get("get_text").is_some() {
                return Ok(AstCommand::GetText);
            }
            if map.get("parse_int").is_some() {
                return Ok(AstCommand::ParseInt);
            }
            if let Some(func) = map.get("call").and_then(|v| v.as_str()) {
                let args = if let Some(serde_json::Value::Array(arr)) = map.get("args") {
                    arr.iter().map(|v| Self::json_to_command_arg(v)).collect()
                } else {
                    vec![]
                };
                return Ok(AstCommand::Call { func: func.to_string(), args });
            }
            // Handle nested "define" commands
            if let Some(node_type) = map.get("define").and_then(|v| v.as_str()) {
                let args = if let Some(serde_json::Value::Object(obj)) = map.get("args") {
                    let mut named_args = HashMap::new();
                    for (key, val) in obj {
                        named_args.insert(key.clone(), Self::json_to_command_arg(val));
                    }
                    NamedArgs(named_args)
                } else {
                    NamedArgs::default()
                };
                return Ok(AstCommand::Define {
                    node: node_type.to_string(),
                    args,
                });
            }
        }
        Err(ZynPegError::ParseError("Invalid JSON command".into()))
    }

    /// Parse raw action code into commands
    fn parse_raw_action(code: &str) -> Result<Vec<AstCommand>> {
        // For now, simple pattern matching on common patterns
        // This can be expanded to handle more complex expressions

        let code = code.trim();
        let mut commands = Vec::new();

        // Pattern: IntLiteral($1.parse())
        if code.contains("IntLiteral") && code.contains(".parse()") {
            commands.push(AstCommand::GetChild {
                index: Some(0),
                name: None,
            });
            commands.push(AstCommand::GetText);
            commands.push(AstCommand::ParseInt);
            commands.push(AstCommand::Call {
                func: "int_literal".to_string(),
                args: vec![CommandArg::ChildRef("$result".to_string())],
            });
        }
        // Pattern: BinaryOp($op, $1, $3) or similar
        else if code.contains("BinaryOp") {
            commands.push(AstCommand::FoldBinary {
                operand_rule: "term".to_string(),
                operator_rule: "operator".to_string(),
            });
        }
        // Default: pass through
        else {
            commands.push(AstCommand::GetChild {
                index: Some(0),
                name: None,
            });
        }

        Ok(commands)
    }

    /// Compile a field value expression to a command
    fn compile_field_value(field_name: &str, value: &str) -> Result<AstCommand> {
        // Parse the value expression and generate appropriate command
        let value = value.trim();

        // $N reference
        if value.starts_with('$') && value.len() > 1 {
            if let Ok(idx) = value[1..].parse::<usize>() {
                return Ok(AstCommand::GetChild {
                    index: Some(idx - 1),
                    name: None,
                });
            }
        }

        // Function call pattern: func(args)
        if let Some(paren_pos) = value.find('(') {
            let func_name = &value[..paren_pos];
            let args_str = &value[paren_pos + 1..value.len() - 1];
            let args = Self::parse_args(args_str)?;

            return Ok(AstCommand::Call {
                func: func_name.to_string(),
                args,
            });
        }

        // Default: treat as identifier/constant
        Ok(AstCommand::Call {
            func: "constant".to_string(),
            args: vec![CommandArg::StringLit(value.to_string())],
        })
    }

    /// Parse function arguments
    fn parse_args(args_str: &str) -> Result<Vec<CommandArg>> {
        let mut args = Vec::new();

        for arg in args_str.split(',') {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }

            // $N reference
            if arg.starts_with('$') {
                args.push(CommandArg::ChildRef(arg.to_string()));
            }
            // Quoted string
            else if arg.starts_with('"') && arg.ends_with('"') {
                args.push(CommandArg::StringLit(arg[1..arg.len() - 1].to_string()));
            }
            // Number
            else if let Ok(n) = arg.parse::<i64>() {
                args.push(CommandArg::IntLit(n));
            }
            // Boolean
            else if arg == "true" {
                args.push(CommandArg::BoolLit(true));
            } else if arg == "false" {
                args.push(CommandArg::BoolLit(false));
            }
            // Identifier/other
            else {
                args.push(CommandArg::StringLit(arg.to_string()));
            }
        }

        Ok(args)
    }
}

impl ZpegModule {
    /// Save module to a file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ZynPegError::CodeGenError(format!("Failed to serialize zpeg: {}", e)))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load module from a file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let module: ZpegModule = serde_json::from_str(&json)
            .map_err(|e| ZynPegError::ParseError(format!("Failed to parse zpeg: {}", e)))?;
        Ok(module)
    }

    /// Get the pest grammar for runtime parsing
    pub fn pest_grammar(&self) -> &str {
        &self.pest_grammar
    }

    /// Get commands for a rule
    pub fn rule_commands(&self, rule_name: &str) -> Option<&RuleCommands> {
        self.rules.get(rule_name)
    }
}

// ============================================================================
// Default TypedAST Host Functions Implementation
// ============================================================================

/// Default implementation of AstHostFunctions that uses the TypedASTBuilder
/// fluent API from zyntax_typed_ast to build TypedProgram directly.
///
/// This approach:
/// 1. Uses NodeHandle as a reference to stored TypedNode<TypedExpression> etc.
/// 2. Builds actual typed AST nodes using the fluent API
/// 3. Serializes to JSON for pipeline integration (can also return TypedProgram directly)
pub struct TypedAstBuilder {
    /// The underlying fluent builder from zyntax_typed_ast
    inner: TypedASTBuilder,
    /// Next handle ID
    next_id: u32,
    /// Stored expression nodes by handle
    expressions: HashMap<NodeHandle, TypedNode<TypedExpression>>,
    /// Stored statement nodes by handle
    statements: HashMap<NodeHandle, TypedNode<TypedStatement>>,
    /// Stored block nodes by handle
    blocks: HashMap<NodeHandle, TypedBlock>,
    /// Stored declaration nodes by handle
    declarations: HashMap<NodeHandle, TypedNode<TypedDeclaration>>,
    /// Stored field nodes by handle
    fields: HashMap<NodeHandle, TypedField>,
    /// Stored variant nodes by handle
    variants: HashMap<NodeHandle, TypedVariant>,
    /// Stored method nodes by handle
    methods: HashMap<NodeHandle, TypedMethod>,
    /// Stored struct field initializers (name, value) by handle
    struct_field_inits: HashMap<NodeHandle, (String, NodeHandle)>,
    /// Stored patterns by handle
    patterns: HashMap<NodeHandle, TypedNode<TypedPattern>>,
    /// Stored match arms by handle
    match_arms: HashMap<NodeHandle, TypedMatchArm>,
    /// Stored field patterns by handle (for struct patterns)
    field_patterns: HashMap<NodeHandle, TypedFieldPattern>,
    /// Stored function parameters (name, type) by handle
    params: HashMap<NodeHandle, (String, Type)>,
    /// Stored types by handle (for type resolution)
    types: HashMap<NodeHandle, Type>,
    /// Variable name to type mapping (for proper variable references)
    variable_types: HashMap<String, Type>,
    /// Enum type name to variant names (in order, for discriminant calculation)
    enum_types: HashMap<String, Vec<String>>,
    /// Type declarations defined in the current file (name -> Type)
    /// Tracks types from @opaque, struct, enum declarations so type references can be resolved
    declared_types: HashMap<String, Type>,
    /// Suffix to type name mapping for abstract types (suffix -> type_name)
    /// e.g., "ms" -> "Duration", "s" -> "Duration", "px" -> "Length"
    suffix_registry: HashMap<String, String>,
    /// Program declaration handles (in order)
    program_decls: Vec<NodeHandle>,
    /// Current span being processed (start, end)
    current_span: (usize, usize),
}

impl Default for TypedAstBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TypedAstBuilder {
    pub fn new() -> Self {
        Self {
            inner: TypedASTBuilder::new(),
            next_id: 0,
            expressions: HashMap::new(),
            statements: HashMap::new(),
            blocks: HashMap::new(),
            declarations: HashMap::new(),
            fields: HashMap::new(),
            variants: HashMap::new(),
            methods: HashMap::new(),
            struct_field_inits: HashMap::new(),
            patterns: HashMap::new(),
            field_patterns: HashMap::new(),
            match_arms: HashMap::new(),
            params: HashMap::new(),
            types: HashMap::new(),
            variable_types: HashMap::new(),
            enum_types: HashMap::new(),
            declared_types: HashMap::new(),
            suffix_registry: HashMap::new(),
            program_decls: Vec::new(),
            current_span: (0, 0),
        }
    }

    /// Set the current span for the next node being created
    pub fn set_current_span(&mut self, start: usize, end: usize) {
        self.current_span = (start, end);
    }

    /// Get the current span
    pub fn get_current_span(&self) -> (usize, usize) {
        self.current_span
    }

    /// Set the source file information for span tracking and diagnostics
    pub fn set_source(&mut self, file_name: String, content: String) {
        self.inner.set_source(file_name, content);
    }

    /// Store an expression and return its handle
    fn store_expr(&mut self, expr: TypedNode<TypedExpression>) -> NodeHandle {
        let handle = self.alloc_handle();
        self.expressions.insert(handle, expr);
        handle
    }

    /// Store a statement and return its handle
    fn store_stmt(&mut self, stmt: TypedNode<TypedStatement>) -> NodeHandle {
        let handle = self.alloc_handle();
        self.statements.insert(handle, stmt);
        handle
    }

    /// Store a declaration and return its handle
    fn store_decl(&mut self, decl: TypedNode<TypedDeclaration>) -> NodeHandle {
        let handle = self.alloc_handle();
        self.declarations.insert(handle, decl);
        handle
    }

    /// Store a block and return its handle
    fn store_block(&mut self, block: TypedBlock) -> NodeHandle {
        let handle = self.alloc_handle();
        self.blocks.insert(handle, block);
        handle
    }

    /// Get an expression by handle (cloning it)
    fn get_expr(&self, handle: NodeHandle) -> Option<TypedNode<TypedExpression>> {
        self.expressions.get(&handle).cloned()
    }

    /// Get a statement by handle (cloning it)
    fn get_stmt(&self, handle: NodeHandle) -> Option<TypedNode<TypedStatement>> {
        self.statements.get(&handle).cloned()
    }

    /// Get a block by handle (cloning it)
    fn get_block(&self, handle: NodeHandle) -> Option<TypedBlock> {
        self.blocks.get(&handle).cloned()
    }

    /// Get a field by handle (cloning it)
    fn get_field(&self, handle: NodeHandle) -> Option<TypedField> {
        self.fields.get(&handle).cloned()
    }

    /// Get a variant by handle (cloning it)
    fn get_variant(&self, handle: NodeHandle) -> Option<TypedVariant> {
        self.variants.get(&handle).cloned()
    }

    /// Get a pattern by handle (cloning it)
    fn get_pattern(&self, handle: NodeHandle) -> Option<TypedNode<TypedPattern>> {
        self.patterns.get(&handle).cloned()
    }

    /// Get a match arm by handle (cloning it)
    fn get_match_arm(&self, handle: NodeHandle) -> Option<TypedMatchArm> {
        self.match_arms.get(&handle).cloned()
    }

    /// Store a pattern and return its handle
    fn store_pattern(&mut self, pattern: TypedNode<TypedPattern>) -> NodeHandle {
        let handle = self.alloc_handle();
        self.patterns.insert(handle, pattern);
        handle
    }

    /// Store a match arm and return its handle
    fn store_match_arm(&mut self, arm: TypedMatchArm) -> NodeHandle {
        let handle = self.alloc_handle();
        self.match_arms.insert(handle, arm);
        handle
    }

    /// Get a type from a handle
    fn get_type_from_handle(&self, handle: NodeHandle) -> Option<Type> {
        self.types.get(&handle).cloned()
    }

    /// Get a named type by string name (looks up declared types or creates unresolved)
    fn get_type_by_name(&mut self, name: &str) -> Type {
        let interned_name = self.inner.intern(name);

        // First check if this type was already declared (struct, enum, etc.)
        debug!("[DEBUG get_type_by_name] Looking up '{}', declared_types has {} entries", name, self.declared_types.len());
        if let Some(ty) = self.declared_types.get(name) {
            debug!("[DEBUG get_type_by_name] Found declared type for '{}': {:?}", name, ty);
            return ty.clone();
        }

        // If not declared yet, return an unresolved type for later resolution
        debug!("[DEBUG get_type_by_name] Type '{}' not found in declared_types, returning Unresolved", name);
        Type::Unresolved(interned_name)
    }

    /// Get the default span for nodes (uses current span from parsing)
    fn default_span(&self) -> Span {
        Span::new(self.current_span.0, self.current_span.1)
    }

    /// Build and return the final TypedProgram
    pub fn build_program(&self) -> TypedProgram {
        use zyntax_typed_ast::source::SourceFile;

        let decls: Vec<TypedNode<TypedDeclaration>> = self.program_decls.iter()
            .filter_map(|h| self.declarations.get(h).cloned())
            .collect();

        // Create SourceFile if we have source information
        let source_files = if let (Some(name), Some(content)) = (self.inner.source_file(), self.inner.source_content()) {
            vec![SourceFile::new(name.to_string(), content.to_string())]
        } else {
            vec![]
        };

        debug!("[DEBUG build_program] Building program with {} declarations, registry has {} types",
            decls.len(), self.inner.registry.get_all_types().count());

        TypedProgram {
            declarations: decls,
            span: self.default_span(),
            source_files,
            type_registry: self.inner.registry.clone(),
        }
    }

    /// Convert string operator to BinaryOp enum
    fn string_to_binary_op(op: &str) -> BinaryOp {
        match op.to_lowercase().as_str() {
            "add" | "+" => BinaryOp::Add,
            "sub" | "-" => BinaryOp::Sub,
            "mul" | "*" => BinaryOp::Mul,
            "div" | "/" => BinaryOp::Div,
            "mod" | "%" | "rem" => BinaryOp::Rem,
            "eq" | "==" => BinaryOp::Eq,
            "ne" | "!=" => BinaryOp::Ne,
            "lt" | "<" => BinaryOp::Lt,
            "le" | "<=" => BinaryOp::Le,
            "gt" | ">" => BinaryOp::Gt,
            "ge" | ">=" => BinaryOp::Ge,
            "and" | "&&" => BinaryOp::And,
            "or" | "||" => BinaryOp::Or,
            "bitand" | "&" => BinaryOp::BitAnd,
            "bitor" | "|" => BinaryOp::BitOr,
            "bitxor" | "^" => BinaryOp::BitXor,
            "shl" | "<<" => BinaryOp::Shl,
            "shr" | ">>" => BinaryOp::Shr,
            _ => BinaryOp::Add, // Default fallback
        }
    }

    /// Convert string operator to UnaryOp enum
    fn string_to_unary_op(op: &str) -> UnaryOp {
        match op.to_lowercase().as_str() {
            "neg" | "-" | "minus" => UnaryOp::Minus,
            "pos" | "+" | "plus" => UnaryOp::Plus,
            "not" | "!" => UnaryOp::Not,
            "bitnot" | "~" => UnaryOp::BitNot,
            _ => UnaryOp::Minus, // Default fallback
        }
    }
}

impl AstHostFunctions for TypedAstBuilder {
    fn create_program(&mut self) -> NodeHandle {
        // Just return a handle - program tracks declarations separately
        self.alloc_handle()
    }

    fn program_add_decl(&mut self, _program: NodeHandle, decl: NodeHandle) {
        // Only add actual declarations - grammar is responsible for defining entry points
        if self.declarations.contains_key(&decl) {
            self.program_decls.push(decl);
        }
        // Note: If the grammar passes an expression or statement handle here,
        // it's a grammar error - the grammar should use wrap_main or similar
        // to create proper function declarations with entry points
    }

    fn finalize_program(&mut self, _program: NodeHandle) -> String {
        // Build the TypedProgram and serialize to JSON
        let typed_program = self.build_program();
        serde_json::to_string(&typed_program)
            .unwrap_or_else(|e| format!(r#"{{"declarations": [], "error": "{}"}}"#, e))
    }

    fn set_current_span(&mut self, start: usize, end: usize) {
        self.current_span = (start, end);
    }

    fn alloc_handle(&mut self) -> NodeHandle {
        let handle = NodeHandle(self.next_id);
        self.next_id += 1;
        handle
    }

    fn store_type(&mut self, handle: NodeHandle, ty: zyntax_typed_ast::type_registry::Type) {
        self.types.insert(handle, ty);
    }

    fn create_function(
        &mut self,
        name: &str,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
        body: NodeHandle,
    ) -> NodeHandle {
        let span = self.default_span();

        // Convert param handles to TypedParameter using stored parameter info
        let typed_params: Vec<_> = params.iter()
            .map(|h| {
                if let Some((name, ty)) = self.params.get(h) {
                    // Register parameter type in variable_types for later variable references
                    self.variable_types.insert(name.clone(), ty.clone());
                    debug!("[DEBUG create_function] Registered parameter '{}' with type {:?}", name, ty);
                    self.inner.parameter(name, ty.clone(), Mutability::Immutable, span)
                } else {
                    // Fallback for unknown params
                    self.inner.parameter("arg", Type::Primitive(PrimitiveType::I32), Mutability::Immutable, span)
                }
            })
            .collect();

        // Get body block - first try as a block, then as a single statement
        let body_block = if let Some(block) = self.get_block(body) {
            block
        } else if let Some(stmt) = self.get_stmt(body) {
            TypedBlock { statements: vec![stmt], span }
        } else if let Some(expr) = self.get_expr(body) {
            TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
        } else {
            TypedBlock { statements: vec![], span }
        };

        // Get return type from handle, default to Unit (void)
        let ret_type = self.types.get(&return_type)
            .cloned()
            .unwrap_or(Type::Primitive(PrimitiveType::Unit));

        let func = self.inner.function(
            name,
            typed_params,
            ret_type,
            body_block,
            Visibility::Public,
            false,
            span,
        );

        self.store_decl(func)
    }

    fn create_extern_function(
        &mut self,
        name: &str,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
    ) -> NodeHandle {
        let span = self.default_span();

        // Convert param handles to TypedParameter using stored parameter info
        let typed_params: Vec<_> = params.iter()
            .map(|h| {
                if let Some((name, ty)) = self.params.get(h) {
                    self.inner.parameter(name, ty.clone(), Mutability::Immutable, span)
                } else {
                    // Fallback for unknown params
                    self.inner.parameter("arg", Type::Primitive(PrimitiveType::I32), Mutability::Immutable, span)
                }
            })
            .collect();

        // Get return type from handle, default to Unit (void)
        let ret_type = self.types.get(&return_type)
            .cloned()
            .unwrap_or(Type::Primitive(PrimitiveType::Unit));

        // Create extern function (no body, is_external = true)
        let func = self.inner.extern_function(
            name,
            typed_params,
            ret_type,
            Visibility::Public,
            span,
        );

        self.store_decl(func)
    }

    fn create_async_function(
        &mut self,
        name: &str,
        params: Vec<NodeHandle>,
        return_type: NodeHandle,
        body: NodeHandle,
    ) -> NodeHandle {
        let span = self.default_span();

        // Convert param handles to TypedParameter using stored parameter info
        let typed_params: Vec<_> = params.iter()
            .map(|h| {
                if let Some((name, ty)) = self.params.get(h) {
                    self.inner.parameter(name, ty.clone(), Mutability::Immutable, span)
                } else {
                    // Fallback for unknown params
                    self.inner.parameter("arg", Type::Primitive(PrimitiveType::I32), Mutability::Immutable, span)
                }
            })
            .collect();

        // Get body block - first try as a block, then as a single statement
        let body_block = if let Some(block) = self.get_block(body) {
            block
        } else if let Some(stmt) = self.get_stmt(body) {
            TypedBlock { statements: vec![stmt], span }
        } else if let Some(expr) = self.get_expr(body) {
            TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
        } else {
            TypedBlock { statements: vec![], span }
        };

        // Get return type from handle, default to Unit (void)
        let ret_type = self.types.get(&return_type)
            .cloned()
            .unwrap_or(Type::Primitive(PrimitiveType::Unit));

        // Create async function (is_async = true)
        let func = self.inner.function(
            name,
            typed_params,
            ret_type,
            body_block,
            Visibility::Public,
            true,  // is_async = true
            span,
        );

        self.store_decl(func)
    }

    fn create_import(&mut self, module_name: &str) -> NodeHandle {
        let span = self.default_span();

        // Create import declaration using the builder's import method
        let import_decl = self.inner.import(module_name, span);

        self.store_decl(import_decl)
    }

    fn create_trait(&mut self, name: &str, methods: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // For now, create a simple trait with no type params or associated types
        // Methods will be empty vec - full implementation needs method signature handling
        let trait_decl = self.inner.trait_def(
            name,
            vec![], // type_params
            vec![], // methods (TODO: convert handles to TypedMethodSignature)
            vec![], // associated_types
            span,
        );

        self.store_decl(trait_decl)
    }

    fn create_impl_block(
        &mut self,
        trait_name: &str,
        for_type_name: &str,
        trait_args: Vec<NodeHandle>,
        items: Vec<NodeHandle>,
    ) -> NodeHandle {
        let span = self.default_span();
        debug!("DEBUG: create_impl_block span=({}, {})", span.start, span.end);

        // Convert trait type arguments from handles to Type
        let trait_type_args: Vec<Type> = trait_args.iter()
            .filter_map(|h| self.get_type_from_handle(*h))
            .collect();

        // Create named type for the implementing type
        // Type inference will resolve the actual TypeId later
        let for_type = self.get_type_by_name(for_type_name);

        // Separate items into methods and associated types
        let mut methods = Vec::new();
        let mut associated_types = Vec::new();

        debug!("[DEBUG] Processing {} items for impl block", items.len());
        for item_handle in items {
            debug!("[DEBUG] Processing item handle: {:?}", item_handle);
            if let Some(decl) = self.declarations.get(&item_handle) {
                debug!("[DEBUG] Found declaration: {:?}", decl.node);
                match &decl.node {
                    TypedDeclaration::Function(func) => {
                        debug!("[DEBUG] Processing function: {:?}", func.name);

                        // Convert function parameters to method parameters
                        // Set is_self flag based on parameter name matching
                        // The compiler will handle type resolution for self parameters
                        let self_name = self.inner.intern("self");
                        let method_params: Vec<TypedMethodParam> = func.params.iter().map(|p| {
                            let is_self_param = p.name == self_name;
                            TypedMethodParam {
                                name: p.name,
                                ty: p.ty.clone(),
                                mutability: p.mutability,
                                is_self: is_self_param,  // Mark self params - compiler will resolve type
                                default_value: None,
                                attributes: vec![],
                                kind: ParameterKind::Regular,
                                span: p.span,
                            }
                        }).collect();

                        // Convert function to method
                        debug!("[DEBUG] Impl method return type: {:?}", func.return_type);

                        let method = TypedMethod {
                            name: func.name,
                            type_params: func.type_params.clone(),
                            params: method_params,
                            return_type: func.return_type.clone(),
                            body: func.body.clone(),
                            visibility: func.visibility.clone(),
                            is_static: false,
                            is_async: func.is_async,
                            is_override: false,
                            span: span,
                        };
                        methods.push(method);
                    }
                    // TODO: Handle associated types when impl_assoc_type creates proper structures
                    _ => {}
                }
            }
        }

        let impl_decl = self.inner.impl_block(
            trait_name,
            trait_type_args,
            for_type,
            methods,
            associated_types,
            span,
        );

        self.store_decl(impl_decl)
    }

    fn create_abstract_inherent_impl(
        &mut self,
        type_name: &str,
        underlying_type: NodeHandle,
        items: Vec<NodeHandle>,
    ) -> NodeHandle {
        let span = self.default_span();
        debug!("DEBUG: create_abstract_inherent_impl type_name={} span=({}, {})", type_name, span.start, span.end);

        // Get the underlying type from the handle
        let underlying = self.get_type_from_handle(underlying_type)
            .unwrap_or(Type::Any);

        // Create named type for the abstract type
        let for_type = self.get_type_by_name(type_name);

        // Process items (methods) - same as trait impl blocks
        let mut methods = Vec::new();

        debug!("[DEBUG] Processing {} items for inherent impl block", items.len());
        for item_handle in items {
            debug!("[DEBUG] Processing item handle: {:?}", item_handle);
            if let Some(decl) = self.declarations.get(&item_handle) {
                debug!("[DEBUG] Found declaration: {:?}", decl.node);
                match &decl.node {
                    TypedDeclaration::Function(func) => {
                        debug!("[DEBUG] Processing function: {:?}", func.name);

                        // Convert function parameters to method parameters
                        let self_name = self.inner.intern("self");
                        let method_params: Vec<TypedMethodParam> = func.params.iter().map(|p| {
                            let is_self_param = p.name == self_name;
                            TypedMethodParam {
                                name: p.name,
                                ty: p.ty.clone(),
                                mutability: p.mutability,
                                is_self: is_self_param,
                                default_value: None,
                                attributes: vec![],
                                kind: ParameterKind::Regular,
                                span: p.span,
                            }
                        }).collect();

                        let method = TypedMethod {
                            name: func.name,
                            type_params: func.type_params.clone(),
                            params: method_params,
                            return_type: func.return_type.clone(),
                            body: func.body.clone(),
                            visibility: func.visibility.clone(),
                            is_static: false,
                            is_async: func.is_async,
                            is_override: false,
                            span: span,
                        };
                        methods.push(method);
                    }
                    _ => {}
                }
            }
        }

        // Create an inherent impl block (no trait)
        // We'll use empty trait name to indicate this is an inherent impl
        let impl_decl = self.inner.impl_block(
            "", // Empty trait name for inherent impl
            vec![], // No trait type args
            for_type,
            methods,
            vec![], // No associated types in inherent impls
            span,
        );

        debug!("[DEBUG impl_decl] Inherent impl_decl.ty = {:?}", impl_decl.ty);

        self.store_decl(impl_decl)
    }

    fn create_opaque_type(&mut self, name: &str, external_name: &str) -> NodeHandle {
        // Create an external struct declaration for the opaque type
        // This declares a type like: extern struct Tensor (backed by $Tensor)
        let name_interned = self.inner.intern(name);
        let runtime_prefix = self.inner.intern(external_name);

        // Register this extern type so type references can be resolved
        let extern_type = Type::Extern {
            name: runtime_prefix,
            layout: None,
        };
        self.declared_types.insert(name.to_string(), extern_type);
        debug!("[DEBUG create_opaque_type] Registered extern type '{}' -> Extern({})", name, external_name);

        let extern_struct = TypedExternStruct {
            name: name_interned,
            runtime_prefix,
            type_params: vec![],  // No type parameters for now
        };

        let decl = TypedDeclaration::Extern(TypedExtern::Struct(extern_struct));
        let decl_node = TypedNode {
            node: decl,
            span: self.default_span(),
            ty: Type::Never,  // Declarations don't have a type
        };

        self.store_decl(decl_node)
    }

    fn create_struct_def(&mut self, name: &str, fields: Vec<NodeHandle>) -> NodeHandle {
        // Create a struct as a Class with fields but no methods
        // struct Tensor:
        //     ptr: TensorPtr
        debug!("[DEBUG create_struct_def] Creating struct '{}' with {} fields", name, fields.len());

        let name_interned = self.inner.intern(name);

        // Convert field handles to TypedField nodes
        let mut typed_fields = Vec::new();
        for field_handle in fields {
            // Each field should be a "field" node created by struct_def_field rule
            // The grammar creates these via the "field" command
            if let Some(params_val) = self.params.get(&field_handle) {
                let (field_name, field_type) = params_val;
                let field_name_interned = self.inner.intern(field_name);

                typed_fields.push(TypedField {
                    name: field_name_interned,
                    ty: field_type.clone(),
                    initializer: None,
                    visibility: Visibility::Public,
                    mutability: Mutability::Immutable,
                    is_static: false,
                    span: self.default_span(),
                });
                debug!("[DEBUG create_struct_def] Added field: {}", field_name);
            }
        }

        // Pre-allocate TypeId before creating TypeDefinition
        let type_id = zyntax_typed_ast::type_registry::TypeId::next();
        debug!("[DEBUG create_struct_def] Allocated TypeId: {:?}", type_id);

        // Create a Named type for the struct
        let struct_type = Type::Named {
            id: type_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
        };

        // Register the struct type so it can be referenced
        self.declared_types.insert(name.to_string(), struct_type.clone());
        debug!("[DEBUG create_struct_def] Registered struct type '{}' with ID {:?}", name, type_id);

        // Register the type definition in the type registry
        let field_defs: Vec<zyntax_typed_ast::type_registry::FieldDef> = typed_fields.clone().into_iter().map(|f| zyntax_typed_ast::type_registry::FieldDef {
            name: f.name,
            ty: f.ty,
            visibility: f.visibility,
            mutability: f.mutability,
            is_static: f.is_static,
            span: f.span,
            getter: None,
            setter: None,
            is_synthetic: false,
        }).collect();

        let type_def = zyntax_typed_ast::type_registry::TypeDefinition {
            id: type_id,
            name: name_interned,
            kind: zyntax_typed_ast::type_registry::TypeKind::Struct {
                fields: field_defs.clone(),
                is_tuple: false,
            },
            type_params: vec![],
            constraints: vec![],
            fields: field_defs,
            methods: vec![],
            constructors: vec![],
            metadata: Default::default(),
            span: self.default_span(),
        };
        self.inner.registry.register_type(type_def);
        debug!("[DEBUG create_struct_def] Registered type definition in registry");

        // Create the struct as a Class declaration (no methods, just fields)
        let class_decl = TypedDeclaration::Class(TypedClass {
            name: name_interned,
            type_params: vec![],
            extends: None,
            implements: vec![],
            fields: typed_fields,
            methods: vec![],
            constructors: vec![],
            visibility: Visibility::Public,
            is_abstract: false,
            is_final: false,
            span: self.default_span(),
        });

        let decl_node = TypedNode {
            node: class_decl,
            span: self.default_span(),
            ty: struct_type,
        };

        self.store_decl(decl_node)
    }

    fn create_abstract_def(&mut self, name: &str, underlying_type_handle: NodeHandle, field_handles: Vec<NodeHandle>, suffixes: Vec<String>) -> NodeHandle {
        // Create an abstract type (Haxe-style zero-cost wrapper)
        // abstract Duration(i64): ms: i64, Suffix("ms"), Suffix("s")
        debug!("[DEBUG create_abstract_def] Creating abstract type '{}' with {} fields, suffixes={:?}", name, field_handles.len(), suffixes);

        let name_interned = self.inner.intern(name);

        // Get the underlying type from the handle
        let underlying_type = self.get_type_from_handle(underlying_type_handle)
            .unwrap_or(Type::Any);

        // Convert field handles to TypedField nodes (same as struct_def)
        let mut typed_fields = Vec::new();
        for field_handle in field_handles {
            if let Some(params_val) = self.params.get(&field_handle) {
                let (field_name, field_type) = params_val;
                let field_name_interned = self.inner.intern(field_name);

                typed_fields.push(TypedField {
                    name: field_name_interned,
                    ty: field_type.clone(),
                    initializer: None,
                    visibility: Visibility::Public,
                    mutability: Mutability::Immutable,
                    is_static: false,
                    span: self.default_span(),
                });
                debug!("[DEBUG create_abstract_def] Added field: {}", field_name);
            }
        }

        // Pre-allocate TypeId before creating TypeDefinition
        let type_id = zyntax_typed_ast::type_registry::TypeId::next();
        debug!("[DEBUG create_abstract_def] Allocated TypeId: {:?}", type_id);

        // Create a Named type for the abstract
        let abstract_type = Type::Named {
            id: type_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
        };

        // Register the abstract type so it can be referenced
        self.declared_types.insert(name.to_string(), abstract_type.clone());
        debug!("[DEBUG create_abstract_def] Registered abstract type '{}' with ID {:?}", name, type_id);

        // Register suffixes in the suffix registry for literal parsing
        for suffix in &suffixes {
            debug!("[DEBUG create_abstract_def] Registering suffix '{}' -> '{}'", suffix, name);
            self.suffix_registry.insert(suffix.clone(), name.to_string());
        }

        // Register the type definition in the type registry
        // Implicit conversions will be populated later via From/Into trait impls
        let field_defs: Vec<zyntax_typed_ast::type_registry::FieldDef> = typed_fields.clone().into_iter().map(|f| zyntax_typed_ast::type_registry::FieldDef {
            name: f.name,
            ty: f.ty,
            visibility: f.visibility,
            mutability: f.mutability,
            is_static: f.is_static,
            span: f.span,
            getter: None,
            setter: None,
            is_synthetic: false,
        }).collect();

        // Validate abstract types with suffixes
        if !suffixes.is_empty() {
            // 1. Check numeric underlying type
            let is_numeric = matches!(&underlying_type,
                zyntax_typed_ast::Type::Primitive(zyntax_typed_ast::type_registry::PrimitiveType::I8 |
                    zyntax_typed_ast::type_registry::PrimitiveType::I16 |
                    zyntax_typed_ast::type_registry::PrimitiveType::I32 |
                    zyntax_typed_ast::type_registry::PrimitiveType::I64 |
                    zyntax_typed_ast::type_registry::PrimitiveType::U8 |
                    zyntax_typed_ast::type_registry::PrimitiveType::U16 |
                    zyntax_typed_ast::type_registry::PrimitiveType::U32 |
                    zyntax_typed_ast::type_registry::PrimitiveType::U64 |
                    zyntax_typed_ast::type_registry::PrimitiveType::F32 |
                    zyntax_typed_ast::type_registry::PrimitiveType::F64)
            );
            if !is_numeric {
                debug!("[WARNING] Abstract type '{}' uses Suffixes with non-numeric underlying type. This will be reported as an error during type checking.", name);
            }

            // 2. Enforce 'value' field convention
            // Abstract types with suffixes MUST have exactly one field named 'value'
            let has_value_field = typed_fields.iter().any(|f| {
                f.name.resolve_global().unwrap_or_default() == "value"
            });

            if !has_value_field {
                debug!("[ERROR] Abstract type '{}' with Suffixes must have a 'value' field", name);
                debug!("       Convention: abstract {}({}) with Suffixes(...): value: {}",
                    name,
                    format!("{:?}", underlying_type).split("::").last().unwrap_or("Type"),
                    format!("{:?}", underlying_type).split("::").last().unwrap_or("Type")
                );
                debug!("       The 'value' field represents the canonical IR representation.");
            } else if typed_fields.len() > 1 {
                debug!("[WARNING] Abstract type '{}' with Suffixes has multiple fields. Only the 'value' field will be used in IR.", name);
            }
        }

        let type_def = zyntax_typed_ast::type_registry::TypeDefinition {
            id: type_id,
            name: name_interned,
            kind: zyntax_typed_ast::type_registry::TypeKind::Abstract {
                underlying_type: underlying_type.clone(),
                implicit_to: vec![],
                implicit_from: vec![],
                suffixes,
            },
            type_params: vec![],
            constraints: vec![],
            fields: field_defs.clone(),
            methods: vec![],
            constructors: vec![],
            metadata: Default::default(),
            span: self.default_span(),
        };
        self.inner.registry.register_type(type_def);
        debug!("[DEBUG create_abstract_def] Registered abstract type definition in registry");

        // Create as Class declaration if it has fields, or TypeAlias if not
        let abstract_decl = if typed_fields.is_empty() {
            TypedDeclaration::TypeAlias(TypedTypeAlias {
                name: name_interned,
                target: underlying_type,
                type_params: vec![],
                visibility: Visibility::Public,
                span: self.default_span(),
            })
        } else {
            TypedDeclaration::Class(TypedClass {
                name: name_interned,
                type_params: vec![],
                extends: None,
                implements: vec![],
                fields: typed_fields,
                methods: vec![],
                constructors: vec![],
                visibility: Visibility::Public,
                is_abstract: false,
                is_final: false,
                span: self.default_span(),
            })
        };

        let decl_node = TypedNode {
            node: abstract_decl,
            span: self.default_span(),
            ty: abstract_type,
        };

        self.store_decl(decl_node)
    }

    fn lookup_suffix(&self, suffix: &str) -> Option<String> {
        self.suffix_registry.get(suffix).cloned()
    }

    fn lookup_declared_type(&self, type_name: &str) -> Option<Type> {
        self.declared_types.get(type_name).cloned()
    }

    fn get_expr(&self, handle: NodeHandle) -> Option<TypedNode<TypedExpression>> {
        self.expressions.get(&handle).cloned()
    }

    fn get_type_name(&self, handle: NodeHandle) -> Option<String> {
        // Try to get the type from the handle and extract its name
        if let Some(ty) = self.get_type_from_handle(handle) {
            match &ty {
                Type::Named { id, .. } => {
                    // Look up the type name in declared_types by matching TypeId
                    for (name, declared_ty) in &self.declared_types {
                        if let Type::Named { id: declared_id, .. } = declared_ty {
                            if id == declared_id {
                                return Some(name.clone());
                            }
                        }
                    }
                }
                Type::Unresolved(name) => {
                    return Some(name.resolve_global().unwrap_or_default().to_string());
                }
                _ => {}
            }
        }
        // Try to find in types map
        if let Some(ty) = self.types.get(&handle) {
            match ty {
                Type::Named { id, .. } => {
                    // Look up by TypeId match
                    for (name, declared_ty) in &self.declared_types {
                        if let Type::Named { id: declared_id, .. } = declared_ty {
                            if id == declared_id {
                                return Some(name.clone());
                            }
                        }
                    }
                }
                Type::Unresolved(name) => {
                    return Some(name.resolve_global().unwrap_or_default().to_string());
                }
                _ => {}
            }
        }
        None
    }

    fn create_param(&mut self, name: &str, ty: NodeHandle) -> NodeHandle {
        // Store parameter name and type for later use in create_function
        let handle = self.alloc_handle();
        // Get the type from the type handle, default to Any if not found
        let param_type = self.get_type_from_handle(ty).unwrap_or(Type::Any);
        // IMPORTANT: Register parameter type IMMEDIATELY so that variable references
        // in the function body (which is parsed after params but before create_function is called)
        // can find the correct type
        self.variable_types.insert(name.to_string(), param_type.clone());
        debug!("[DEBUG create_param] Registered parameter '{}' with type {:?}", name, param_type);
        self.params.insert(handle, (name.to_string(), param_type));
        handle
    }

    fn create_binary_op(&mut self, op: &str, left: NodeHandle, right: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let left_expr = self.get_expr(left).unwrap_or_else(|| self.inner.int_literal(0, span));
        let right_expr = self.get_expr(right).unwrap_or_else(|| self.inner.int_literal(0, span));

        let binary_op = Self::string_to_binary_op(op);

        // Infer result type from operands
        // Comparison operators return bool, arithmetic operators return left operand's type
        let result_type = match binary_op {
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le
            | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::And | BinaryOp::Or
                => Type::Primitive(PrimitiveType::Bool),
            _ => left_expr.ty.clone(),  // Arithmetic ops preserve left operand's type
        };

        debug!("[DEBUG create_binary_op] op={:?}, left_type={:?}, right_type={:?}, result_type={:?}",
            binary_op, left_expr.ty, right_expr.ty, result_type);

        let expr = self.inner.binary(binary_op, left_expr, right_expr, result_type, span);
        self.store_expr(expr)
    }

    fn create_unary_op(&mut self, op: &str, operand: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let operand_expr = self.get_expr(operand).unwrap_or_else(|| self.inner.int_literal(0, span));

        // Handle special cases that aren't true unary ops
        match op.to_lowercase().as_str() {
            // `try` unwraps error union - for now just pass through the value
            "try" => return operand,
            _ => {}
        }

        let unary_op = Self::string_to_unary_op(op);

        // Infer result type from operand
        // Most unary ops (neg, not) preserve the operand's type
        let result_type = operand_expr.ty.clone();

        debug!("[DEBUG create_unary_op] op={:?}, operand_type={:?}, result_type={:?}",
            unary_op, operand_expr.ty, result_type);

        let expr = self.inner.unary(unary_op, operand_expr, result_type, span);
        self.store_expr(expr)
    }

    fn create_await(&mut self, expr_handle: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        // Get the expression being awaited
        let awaited_expr = self.get_expr(expr_handle)
            .unwrap_or_else(|| self.inner.int_literal(0, span));

        // The result type of await is the inner type of the Promise
        // For simplicity, we assume i32 for now
        let result_type = Type::Primitive(PrimitiveType::I32);

        // Create the await expression node
        let expr = self.inner.await_expr(awaited_expr, result_type, span);
        self.store_expr(expr)
    }

    fn create_int_literal(&mut self, value: i64) -> NodeHandle {
        let span = self.default_span();
        let expr = self.inner.int_literal(value as i128, span);
        self.store_expr(expr)
    }

    fn create_typed_int_literal(&mut self, value: i64, ty: Type) -> NodeHandle {
        // Create an integer literal with an explicit type annotation
        // This preserves Abstract types through the compilation pipeline
        let span = self.default_span();
        let expr = TypedNode::new(
            TypedExpression::Literal(TypedLiteral::Integer(value as i128)),
            ty,  // Use the provided type instead of inferring
            span
        );
        self.store_expr(expr)
    }

    fn create_float_literal(&mut self, value: f64) -> NodeHandle {
        let span = self.default_span();
        let expr = self.inner.float_literal(value, span);
        self.store_expr(expr)
    }

    fn create_string_literal(&mut self, value: &str) -> NodeHandle {
        let span = self.default_span();
        let expr = self.inner.string_literal(value, span);
        self.store_expr(expr)
    }

    fn create_bool_literal(&mut self, value: bool) -> NodeHandle {
        let span = self.default_span();
        let expr = self.inner.bool_literal(value, span);
        self.store_expr(expr)
    }

    fn create_identifier(&mut self, name: &str) -> NodeHandle {
        let span = self.default_span();

        // Look up the variable's actual type from our tracking map
        // If not found, default to Any (will be resolved during type inference)
        let var_type = self.variable_types.get(name)
            .cloned()
            .unwrap_or(Type::Any);

        debug!("[DEBUG create_identifier] Variable '{}' has type {:?}", name, var_type);
        let expr = self.inner.variable(name, var_type, span);
        self.store_expr(expr)
    }

    fn create_call(&mut self, callee: NodeHandle, args: Vec<NodeHandle>) -> NodeHandle {
        self.create_call_with_return_type(callee, args, None)
    }

    fn create_call_with_return_type(&mut self, callee: NodeHandle, args: Vec<NodeHandle>, return_type: Option<&str>) -> NodeHandle {
        let span = self.default_span();

        let callee_expr = self.get_expr(callee)
            .unwrap_or_else(|| self.inner.variable("unknown", Type::Primitive(PrimitiveType::I32), span));

        // Check if this is a method call: obj.method(args)
        // If callee is a field access, transform into MethodCall
        if let TypedExpression::Field(field_access) = &callee_expr.node {
            let receiver_expr = *field_access.object.clone();
            let method_name_str = field_access.field.resolve_global()
                .unwrap_or_else(|| "unknown".to_string());

            debug!("[PARSER] Detected method call: receiver_ty={:?}, method={}",
                receiver_expr.ty, method_name_str);

            // Store the receiver expression and use create_method_call
            let receiver_handle = self.store_expr(receiver_expr);
            return self.create_method_call(receiver_handle, &method_name_str, args);
        }

        let arg_exprs: Vec<_> = args.iter()
            .filter_map(|h| self.get_expr(*h))
            .collect();

        // Try to resolve the return type from the function's signature if callee is a simple variable
        let resolved_return_type = if let TypedExpression::Variable(func_name) = &callee_expr.node {
            // Look up the function in declarations to get its actual return type
            let func_name_str = func_name.resolve_global().unwrap_or_default();
            debug!("[PARSER] Looking up function '{}' to get return type", func_name_str);

            // Search through declarations for this function
            // Try exact match first, then try as a suffix (for mangled names like Type$method)
            let found_func = self.declarations.values().find_map(|decl| {
                if let TypedDeclaration::Function(func) = &decl.node {
                    let this_name = func.name.resolve_global().unwrap_or_default();
                    // Exact match
                    if this_name == func_name_str {
                        debug!("[PARSER] Found function '{}' with return type: {:?}", func_name_str, func.return_type);
                        return Some(func.return_type.clone());
                    }
                    // Mangled name match: if func_name contains Type$method, match on just the method name
                    // or if we're looking for Type$method, match the full mangled name
                    if this_name.ends_with(&format!("${}", func_name_str)) || this_name == func_name_str || func_name_str.ends_with(&format!("${}", this_name.split('$').last().unwrap_or(""))) {
                        debug!("[PARSER] Found mangled function '{}' for call to '{}' with return type: {:?}", this_name, func_name_str, func.return_type);
                        return Some(func.return_type.clone());
                    }
                }
                None
            });

            found_func
        } else {
            None
        };

        // Convert return type string to Type
        // Opaque types (starting with $) become Extern types which are pointers at the HIR level
        let ty = if let Some(resolved_ty) = resolved_return_type {
            // Use the resolved type from the function signature
            resolved_ty
        } else {
            match return_type {
            Some(type_name) if type_name.starts_with('$') => {
                // This is an opaque type - create an Extern type
                // The type name without $ is used as the extern type name
                log::trace!("[create_call] opaque return type: {}", type_name);
                Type::Extern {
                    name: InternedString::new_global(type_name),
                    layout: None,
                }
            }
            Some(type_name) => {
                // Regular named type - try to parse it
                log::trace!("[create_call] named return type: {}", type_name);
                match type_name {
                    "i32" | "I32" => Type::Primitive(PrimitiveType::I32),
                    "i64" | "I64" => Type::Primitive(PrimitiveType::I64),
                    "f32" | "F32" => Type::Primitive(PrimitiveType::F32),
                    "f64" | "F64" => Type::Primitive(PrimitiveType::F64),
                    "bool" | "Bool" => Type::Primitive(PrimitiveType::Bool),
                    "void" | "Void" | "()" => Type::Primitive(PrimitiveType::Unit),
                    _ => Type::Any, // Use Any for unknown types - will be resolved during type inference
                }
            }
            None => Type::Any, // Use Any when no return type specified - will be resolved during type inference
            }
        };

        let expr = self.inner.call_positional(callee_expr, arg_exprs, ty, span);
        self.store_expr(expr)
    }

    fn create_index(&mut self, array: NodeHandle, index: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let array_expr = self.get_expr(array)
            .unwrap_or_else(|| self.inner.variable("array", Type::Primitive(PrimitiveType::I32), span));
        let index_expr = self.get_expr(index)
            .unwrap_or_else(|| self.inner.int_literal(0, span));

        let expr = self.inner.index(array_expr, index_expr, Type::Primitive(PrimitiveType::I32), span);
        self.store_expr(expr)
    }

    fn create_field_access(&mut self, object: NodeHandle, field: &str) -> NodeHandle {
        let span = self.default_span();

        let object_expr = self.get_expr(object)
            .unwrap_or_else(|| self.inner.variable("object", Type::Primitive(PrimitiveType::I32), span));

        // Check if this is an enum variant access (EnumType.Variant)
        if let TypedExpression::Variable(var_name) = &object_expr.node {
            let type_name = var_name.resolve_global().unwrap_or_default();
            if let Some(variants) = self.enum_types.get(&type_name) {
                // This is an enum variant access - find the discriminant
                if let Some(discriminant) = variants.iter().position(|v| v == field) {
                    // Return an integer literal representing the enum variant
                    let expr = self.inner.int_literal(discriminant as i128, span);
                    return self.store_expr(expr);
                }
            }
        }

        // Infer field type from object's struct type
        let field_type = match &object_expr.ty {
            Type::Struct { fields, .. } => {
                // Find the field by name and get its type
                let field_name = InternedString::new_global(field);
                fields.iter()
                    .find(|f| f.name == field_name)
                    .map(|f| f.ty.clone())
                    .unwrap_or(Type::Primitive(PrimitiveType::I32))
            }
            _ => Type::Primitive(PrimitiveType::I32),
        };

        let expr = self.inner.field_access(object_expr, field, field_type, span);
        self.store_expr(expr)
    }

    fn create_var_decl(
        &mut self,
        name: &str,
        _ty: Option<NodeHandle>,
        init: Option<NodeHandle>,
        is_const: bool,
    ) -> NodeHandle {
        let span = self.default_span();

        let init_expr = init.and_then(|h| self.get_expr(h));
        let mutability = if is_const { Mutability::Immutable } else { Mutability::Mutable };

        // Infer type from initializer expression if available
        let var_type = init_expr.as_ref()
            .map(|expr| expr.ty.clone())
            .unwrap_or(Type::Any);

        // Register the variable type for later lookup
        self.variable_types.insert(name.to_string(), var_type.clone());

        let stmt = self.inner.let_statement(
            name,
            var_type,
            mutability,
            init_expr,
            span,
        );
        self.store_stmt(stmt)
    }

    fn create_assignment(&mut self, target: NodeHandle, value: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        log::trace!("[create_assignment] target={:?}, value={:?}", target, value);

        // Get target and value expressions
        let target_expr = self.get_expr(target)
            .unwrap_or_else(|| {
                log::trace!("[create_assignment] FAILED to get target expression!");
                self.inner.variable("target", Type::Primitive(PrimitiveType::I32), span)
            });
        let value_expr = self.get_expr(value)
            .unwrap_or_else(|| {
                log::trace!("[create_assignment] FAILED to get value expression!");
                self.inner.int_literal(0, span)
            });

        // Create assignment as a binary expression: target = value
        // Store as EXPRESSION - the expr_stmt wrapper will handle statement wrapping
        let assign_expr = self.inner.binary(
            BinaryOp::Assign,
            target_expr,
            value_expr,
            Type::Primitive(PrimitiveType::Unit),
            span,
        );

        self.store_expr(assign_expr)
    }

    fn create_return(&mut self, value: Option<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        if let Some(h) = value {
            if let Some(expr) = self.get_expr(h) {
                let stmt = self.inner.return_stmt(expr, span);
                return self.store_stmt(stmt);
            }
        }

        let stmt = self.inner.return_void(span);
        self.store_stmt(stmt)
    }

    fn create_if(
        &mut self,
        condition: NodeHandle,
        then_branch: NodeHandle,
        else_branch: Option<NodeHandle>,
    ) -> NodeHandle {
        let span = self.default_span();

        let cond_expr = self.get_expr(condition)
            .unwrap_or_else(|| self.inner.bool_literal(true, span));

        // Get the then block - check for block first, then statement, then expression
        let then_block = if let Some(block) = self.get_block(then_branch) {
            block
        } else if let Some(stmt) = self.get_stmt(then_branch) {
            TypedBlock { statements: vec![stmt], span }
        } else if let Some(expr) = self.get_expr(then_branch) {
            TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
        } else {
            TypedBlock { statements: vec![], span }
        };

        // Get the else block if present
        let else_block = else_branch.map(|h| {
            if let Some(block) = self.get_block(h) {
                block
            } else if let Some(stmt) = self.get_stmt(h) {
                TypedBlock { statements: vec![stmt], span }
            } else if let Some(expr) = self.get_expr(h) {
                TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
            } else {
                TypedBlock { statements: vec![], span }
            }
        });

        let stmt = self.inner.if_statement(cond_expr, then_block, else_block, span);
        self.store_stmt(stmt)
    }

    fn create_while(&mut self, condition: NodeHandle, body: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let cond_expr = self.get_expr(condition)
            .unwrap_or_else(|| self.inner.bool_literal(true, span));

        // Get body block - check for block first, then statement, then expression
        let body_block = if let Some(block) = self.get_block(body) {
            block
        } else if let Some(stmt) = self.get_stmt(body) {
            TypedBlock { statements: vec![stmt], span }
        } else if let Some(expr) = self.get_expr(body) {
            TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
        } else {
            TypedBlock { statements: vec![], span }
        };

        let stmt = self.inner.while_loop(cond_expr, body_block, span);
        self.store_stmt(stmt)
    }

    fn create_for(&mut self, iterator: &str, iterable: NodeHandle, body: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let iter_expr = self.get_expr(iterable)
            .unwrap_or_else(|| self.inner.variable("iter", Type::Primitive(PrimitiveType::I32), span));

        log::trace!("[create_for] iterator={}, iterable={:?}, iter_expr.node={:?}",
            iterator, iterable, iter_expr.node);

        // Get body block - check for block first, then statement, then expression
        let body_block = if let Some(block) = self.get_block(body) {
            block
        } else if let Some(stmt) = self.get_stmt(body) {
            TypedBlock { statements: vec![stmt], span }
        } else if let Some(expr) = self.get_expr(body) {
            TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
        } else {
            TypedBlock { statements: vec![], span }
        };

        // Check if iterable is a Range - if so, desugar to while loop
        // for (i in 0...5) { body } => { var i = 0; while (i < 5) { body; i = i + 1; } }
        if let TypedExpression::Range(range) = &iter_expr.node {
            // Extract start and end values
            let start_expr = range.start.as_ref()
                .map(|e| (**e).clone())
                .unwrap_or_else(|| self.inner.int_literal(0, span));
            let end_expr = range.end.as_ref()
                .map(|e| (**e).clone())
                .unwrap_or_else(|| self.inner.int_literal(0, span));

            // Register the iterator variable type
            self.variable_types.insert(iterator.to_string(), Type::Primitive(PrimitiveType::I32));

            // Transform for-loop to handle continue correctly:
            // The challenge: continue always jumps to loop header, but we need increment between iterations.
            // Solution: Start one before range and unconditionally increment at loop header.
            //
            // Desugar to:
            // {
            //   let mut i = start - 1
            //   loop {
            //     i = i + 1         // <-- Continue jumps here, ALWAYS incrementing
            //     if i >= end { break }
            //     { body }
            //   }
            // }
            //
            // With this structure:
            // - First iteration: i = (start-1) + 1 = start
            // - continue jumps to loop header, incrementing i before next iteration
            // - Body always sees the post-increment value

            // Create: let mut i = start - 1
            let one = self.inner.int_literal(1, span);
            let start_minus_one = self.inner.binary(BinaryOp::Sub, start_expr.clone(), one.clone(), Type::Primitive(PrimitiveType::I32), span);
            let iter_str = &iterator.to_string();
            let init_stmt = self.inner.let_statement(
                iter_str,
                Type::Primitive(PrimitiveType::I32),
                Mutability::Mutable,
                Some(start_minus_one),
                span,
            );

            // Create: i = i + 1
            let iter_ref1 = self.inner.variable(iter_str, Type::Primitive(PrimitiveType::I32), span);
            let iter_inc = self.inner.binary(BinaryOp::Add, iter_ref1.clone(), one, Type::Primitive(PrimitiveType::I32), span);
            let iter_assign = self.inner.binary(BinaryOp::Assign, iter_ref1, iter_inc, Type::Primitive(PrimitiveType::I32), span);
            let incr_stmt = self.inner.expression_statement(iter_assign, span);

            // Create condition check: if i >= end { break }
            let iter_ref2 = self.inner.variable(iter_str, Type::Primitive(PrimitiveType::I32), span);
            let cond_op = if range.inclusive { BinaryOp::Gt } else { BinaryOp::Ge };
            let break_cond = self.inner.binary(cond_op, iter_ref2, end_expr, Type::Primitive(PrimitiveType::Bool), span);

            let break_stmt = TypedNode::new(
                TypedStatement::Break(None),
                Type::Primitive(PrimitiveType::Unit),
                span,
            );
            let break_block = TypedBlock {
                statements: vec![break_stmt],
                span,
            };
            let guard_if = self.inner.if_statement(break_cond, break_block, None, span);

            // Wrap user body in a block
            let body_wrapped = TypedNode::new(
                TypedStatement::Block(body_block.clone()),
                Type::Primitive(PrimitiveType::Unit),
                span,
            );

            // Create loop body: { i++; if i >= end break; body }
            let loop_body = TypedBlock {
                statements: vec![incr_stmt, guard_if, body_wrapped],
                span,
            };

            // Create infinite loop
            let infinite_loop = self.inner.loop_stmt(loop_body, span);

            // Wrap in outer block: { let i = start - 1; loop { ... } }
            let outer_block = TypedBlock {
                statements: vec![init_stmt, infinite_loop],
                span,
            };
            let block_stmt = TypedNode::new(
                TypedStatement::Block(outer_block),
                Type::Primitive(PrimitiveType::Unit),
                span,
            );
            return self.store_stmt(block_stmt);
        }

        // For non-range iterables, we don't have full iterator trait support yet
        // Fall back to the original TypedLoop::For which will need to be implemented in lowering
        // TODO: Implement proper iterator desugaring when method calls and trait resolution work

        let stmt = self.inner.for_loop(iterator, iter_expr, body_block, span);
        self.store_stmt(stmt)
    }

    fn create_block(&mut self, statements: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        log::trace!("[create_block] statements: {:?}", statements);
        log::trace!("[create_block] statements keys: {:?}", self.statements.keys().collect::<Vec<_>>());

        // First collect all the statements we can find
        let mut stmts: Vec<TypedNode<TypedStatement>> = Vec::new();
        for h in &statements {
            log::trace!("[create_block] checking handle {:?}", h);
            if let Some(stmt) = self.get_stmt(*h) {
                log::trace!("[create_block]   -> found statement: {:?}", stmt.node);
                stmts.push(stmt);
            } else if let Some(expr) = self.get_expr(*h) {
                log::trace!("[create_block]   -> found expression, wrapping: {:?}", expr.node);
                let expr_stmt = self.inner.expression_statement(expr, span);
                stmts.push(expr_stmt);
            } else {
                log::trace!("[create_block]   -> NOT FOUND in statements or expressions!");
            }
        }

        log::trace!("[create_block] collected {} statements", stmts.len());

        // Store the entire block and return its handle
        let block = TypedBlock { statements: stmts, span };
        self.store_block(block)
    }

    fn create_expr_stmt(&mut self, expr: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let expr_node = self.get_expr(expr)
            .unwrap_or_else(|| self.inner.unit_literal(span));

        let stmt = self.inner.expression_statement(expr_node, span);
        self.store_stmt(stmt)
    }

    fn create_primitive_type(&mut self, name: &str) -> NodeHandle {
        // Parse the type name and store the actual type
        let handle = self.alloc_handle();
        let ty = match name {
            "i8" => Type::Primitive(PrimitiveType::I8),
            "i16" => Type::Primitive(PrimitiveType::I16),
            "i32" => Type::Primitive(PrimitiveType::I32),
            "i64" => Type::Primitive(PrimitiveType::I64),
            "i128" => Type::Primitive(PrimitiveType::I128),
            "u8" => Type::Primitive(PrimitiveType::U8),
            "u16" => Type::Primitive(PrimitiveType::U16),
            "u32" => Type::Primitive(PrimitiveType::U32),
            "u64" => Type::Primitive(PrimitiveType::U64),
            "u128" => Type::Primitive(PrimitiveType::U128),
            "f32" => Type::Primitive(PrimitiveType::F32),
            "f64" => Type::Primitive(PrimitiveType::F64),
            "bool" => Type::Primitive(PrimitiveType::Bool),
            "void" | "unit" => Type::Primitive(PrimitiveType::Unit),
            _ => {
                // Unknown type - create unresolved for compiler resolution
                debug!("[DEBUG create_primitive_type] Unknown type '{}', creating Unresolved", name);
                let name_interned = self.inner.intern(name);
                Type::Unresolved(name_interned)
            }
        };
        self.types.insert(handle, ty);
        handle
    }

    fn create_pointer_type(&mut self, _pointee: NodeHandle) -> NodeHandle {
        self.alloc_handle()
    }

    fn create_array_type(&mut self, _element: NodeHandle, _size: Option<usize>) -> NodeHandle {
        self.alloc_handle()
    }

    fn create_function_type(
        &mut self,
        _params: Vec<NodeHandle>,
        _return_type: NodeHandle,
    ) -> NodeHandle {
        self.alloc_handle()
    }

    fn create_named_type(&mut self, name: &str) -> NodeHandle {
        let handle = self.alloc_handle();

        // First check if it's a primitive type
        let ty = match name {
            "i8" => Type::Primitive(PrimitiveType::I8),
            "i16" => Type::Primitive(PrimitiveType::I16),
            "i32" => Type::Primitive(PrimitiveType::I32),
            "i64" => Type::Primitive(PrimitiveType::I64),
            "i128" => Type::Primitive(PrimitiveType::I128),
            "u8" => Type::Primitive(PrimitiveType::U8),
            "u16" => Type::Primitive(PrimitiveType::U16),
            "u32" => Type::Primitive(PrimitiveType::U32),
            "u64" => Type::Primitive(PrimitiveType::U64),
            "u128" => Type::Primitive(PrimitiveType::U128),
            "f32" => Type::Primitive(PrimitiveType::F32),
            "f64" => Type::Primitive(PrimitiveType::F64),
            "bool" => Type::Primitive(PrimitiveType::Bool),
            "void" | "unit" => Type::Primitive(PrimitiveType::Unit),
            _ => {
                // Check if this type was declared in the current file (e.g., via @opaque, struct, enum)
                if let Some(declared_ty) = self.declared_types.get(name) {
                    debug!("[DEBUG create_named_type] Found declared type '{}': {:?}", name, declared_ty);
                    declared_ty.clone()
                } else {
                    // Type not found in current file - create unresolved for compiler resolution
                    // This handles types from imports or forward references (including language keywords like "Self")
                    debug!("[DEBUG create_named_type] Creating unresolved type '{}'", name);
                    let name_interned = self.inner.intern(name);
                    Type::Unresolved(name_interned)
                }
            }
        };

        self.types.insert(handle, ty);
        handle
    }

    fn create_struct(&mut self, name: &str, field_handles: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect fields from handles
        let fields: Vec<TypedField> = field_handles
            .iter()
            .filter_map(|h| self.get_field(*h))
            .collect();

        let class = TypedClass {
            name: InternedString::new_global(name),
            type_params: Vec::new(),
            extends: None,
            implements: Vec::new(),
            fields,
            methods: Vec::new(),
            constructors: Vec::new(),
            visibility: Visibility::Public,
            is_abstract: false,
            is_final: false,
            span,
        };

        let decl = TypedNode {
            node: TypedDeclaration::Class(class),
            ty: Type::Never,
            span,
        };

        self.store_decl(decl)
    }

    fn create_enum(&mut self, name: &str, variant_handles: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect variants from handles
        let variants: Vec<TypedVariant> = variant_handles
            .iter()
            .filter_map(|h| self.get_variant(*h))
            .collect();

        // Register enum type with variant names for later lookup
        let variant_names: Vec<String> = variants
            .iter()
            .filter_map(|v| v.name.resolve_global())
            .collect();
        self.enum_types.insert(name.to_string(), variant_names);

        let enum_decl = TypedEnum {
            name: InternedString::new_global(name),
            type_params: Vec::new(),
            variants,
            visibility: Visibility::Public,
            span,
        };

        let decl = TypedNode {
            node: TypedDeclaration::Enum(enum_decl),
            ty: Type::Never,
            span,
        };

        self.store_decl(decl)
    }

    fn create_field(&mut self, name: &str, ty: NodeHandle) -> NodeHandle {
        // Store field name and type for later use in create_struct_def
        let handle = self.alloc_handle();
        // Get the type from the type handle, default to Any if not found
        let field_type = self.get_type_from_handle(ty).unwrap_or(Type::Any);
        self.params.insert(handle, (name.to_string(), field_type));
        handle
    }

    fn create_variant(&mut self, name: &str) -> NodeHandle {
        let span = self.default_span();

        let variant = TypedVariant {
            name: InternedString::new_global(name),
            fields: TypedVariantFields::Unit,  // Simple Zig-style enum variants
            discriminant: None,
            span,
        };

        let handle = self.alloc_handle();
        self.variants.insert(handle, variant);
        handle
    }

    fn create_class(&mut self, name: &str, type_params: Vec<String>, member_handles: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Convert type params to TypedTypeParam
        let typed_type_params: Vec<TypedTypeParam> = type_params
            .iter()
            .map(|s| TypedTypeParam {
                name: InternedString::new_global(s),
                bounds: Vec::new(),
                default: None,
                span,
            })
            .collect();

        // Separate fields and methods from member handles
        let mut fields: Vec<TypedField> = Vec::new();
        let mut methods: Vec<TypedMethod> = Vec::new();

        for handle in member_handles {
            // Check if it's a field
            if let Some(field) = self.get_field(handle) {
                fields.push(field);
            }
            // Check if it's a method (stored in declarations as function)
            else if let Some(decl) = self.declarations.get(&handle) {
                if let TypedDeclaration::Function(func) = &decl.node {
                    // Convert function to method - convert TypedParameter to TypedMethodParam
                    let method_params: Vec<TypedMethodParam> = func.params.iter()
                        .map(|p| TypedMethodParam {
                            name: p.name.clone(),
                            ty: p.ty.clone(),
                            mutability: p.mutability,
                            is_self: false,
                            kind: ParameterKind::Regular,
                            default_value: None,
                            attributes: Vec::new(),
                            span: p.span,
                        })
                        .collect();
                    let method = TypedMethod {
                        name: func.name.clone(),
                        type_params: Vec::new(),
                        params: method_params,
                        return_type: func.return_type.clone(),
                        body: func.body.clone(),
                        visibility: func.visibility,
                        is_static: false, // Will be updated by create_method
                        is_async: false,
                        is_override: false,
                        span,
                    };
                    methods.push(method);
                }
            }
            // Check if it's a method stored in our methods map
            else if let Some(method) = self.methods.get(&handle).cloned() {
                methods.push(method);
            }
        }

        let class = TypedClass {
            name: InternedString::new_global(name),
            type_params: typed_type_params,
            extends: None,
            implements: Vec::new(),
            fields,
            methods,
            constructors: Vec::new(),
            visibility: Visibility::Public,
            is_abstract: false,
            is_final: false,
            span,
        };

        let decl = TypedNode {
            node: TypedDeclaration::Class(class),
            ty: Type::Never,
            span,
        };

        self.store_decl(decl)
    }

    fn create_method(
        &mut self,
        name: &str,
        is_static: bool,
        visibility: &str,
        param_handles: Vec<NodeHandle>,
        return_type_handle: Option<NodeHandle>,
        body_handle: NodeHandle,
    ) -> NodeHandle {
        let span = self.default_span();

        // Convert visibility string to Visibility enum
        let vis = match visibility {
            "private" => Visibility::Private,
            "protected" => Visibility::Private, // No protected in our type system yet
            _ => Visibility::Public,
        };

        // Get parameters - convert to TypedMethodParam
        let params: Vec<TypedMethodParam> = param_handles.iter()
            .filter_map(|h| {
                self.params.get(h).map(|(param_name, ty)| {
                    TypedMethodParam {
                        name: InternedString::new_global(param_name),
                        ty: ty.clone(),
                        mutability: Mutability::Immutable,
                        is_self: false,
                        kind: ParameterKind::Regular,
                        default_value: None,
                        attributes: Vec::new(),
                        span,
                    }
                })
            })
            .collect();

        // Get return type
        let return_type = return_type_handle
            .and_then(|h| self.get_type_from_handle(h))
            .unwrap_or(Type::Primitive(PrimitiveType::Unit));

        // Get body block
        let body = if let Some(block) = self.get_block(body_handle) {
            block
        } else if let Some(stmt) = self.get_stmt(body_handle) {
            TypedBlock { statements: vec![stmt], span }
        } else if let Some(expr) = self.get_expr(body_handle) {
            TypedBlock { statements: vec![self.inner.expression_statement(expr, span)], span }
        } else {
            TypedBlock { statements: vec![], span }
        };

        let method = TypedMethod {
            name: InternedString::new_global(name),
            type_params: Vec::new(),
            params,
            return_type,
            body: Some(body),
            visibility: vis,
            is_static,
            is_async: false,
            is_override: false,
            span,
        };

        // Store method and return handle
        let handle = self.alloc_handle();
        self.methods.insert(handle, method);
        handle
    }

    fn create_ternary(&mut self, condition: NodeHandle, then_expr_handle: NodeHandle, else_expr_handle: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let cond_expr = self.get_expr(condition)
            .unwrap_or_else(|| self.inner.bool_literal(true, span));
        let then_expr = self.get_expr(then_expr_handle)
            .unwrap_or_else(|| self.inner.int_literal(0, span));
        let else_expr = self.get_expr(else_expr_handle)
            .unwrap_or_else(|| self.inner.int_literal(0, span));

        // Type of ternary expression is Any initially - will be resolved during type resolution
        let result_type = Type::Any;

        let expr = self.inner.if_expr(cond_expr, then_expr, else_expr, result_type, span);
        self.store_expr(expr)
    }

    fn set_span(&mut self, _node: NodeHandle, _start: usize, _end: usize) {
        // Spans are handled inline during node creation
        // This could be extended to update spans if needed
    }

    fn create_char_literal(&mut self, value: char) -> NodeHandle {
        let span = self.default_span();
        let expr = self.inner.char_literal(value, span);
        self.store_expr(expr)
    }

    fn create_let(
        &mut self,
        name: &str,
        ty: Option<NodeHandle>,
        init: Option<NodeHandle>,
        is_const: bool,
    ) -> NodeHandle {
        // Delegate to create_var_decl
        self.create_var_decl(name, ty, init, is_const)
    }

    fn create_break(&mut self, value: Option<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        if let Some(h) = value {
            if let Some(expr) = self.get_expr(h) {
                let stmt = self.inner.break_with_value(expr, span);
                return self.store_stmt(stmt);
            }
        }

        let stmt = self.inner.break_stmt(span);
        self.store_stmt(stmt)
    }

    fn create_continue(&mut self) -> NodeHandle {
        let span = self.default_span();
        let stmt = self.inner.continue_stmt(span);
        self.store_stmt(stmt)
    }

    fn create_expression_stmt(&mut self, expr: NodeHandle) -> NodeHandle {
        // Delegate to create_expr_stmt
        self.create_expr_stmt(expr)
    }

    fn create_range(&mut self, start: Option<NodeHandle>, end: Option<NodeHandle>, inclusive: bool) -> NodeHandle {
        let span = self.default_span();

        let start_expr = start.and_then(|h| self.get_expr(h));
        let end_expr = end.and_then(|h| self.get_expr(h));

        let range = TypedRange {
            start: start_expr.map(Box::new),
            end: end_expr.map(Box::new),
            inclusive,
        };

        let expr = TypedNode::new(
            TypedExpression::Range(range),
            Type::Primitive(PrimitiveType::I32), // Range type - will be resolved during type checking
            span,
        );
        self.store_expr(expr)
    }

    fn apply_postfix(&mut self, base: NodeHandle, postfix_op: NodeHandle) -> NodeHandle {
        // This is now primarily handled by FoldPostfix command in the interpreter
        // which directly creates call/field/index nodes
        // This fallback just returns base for any unhandled cases
        log::trace!("[apply_postfix] base={:?}, postfix_op={:?} (fallback)", base, postfix_op);
        base
    }

    fn create_variable(&mut self, name: &str) -> NodeHandle {
        let span = self.default_span();
        // Look up the variable's declared type, default to I32 if not found
        let var_type = self.variable_types.get(name)
            .cloned()
            .unwrap_or_else(|| {
                debug!("[DEBUG create_variable] Variable '{}' NOT FOUND in variable_types, defaulting to I32", name);
                debug!("[DEBUG create_variable] Available variables: {:?}",
                    self.variable_types.keys().collect::<Vec<_>>());
                Type::Primitive(PrimitiveType::I32)
            });
        debug!("[DEBUG create_variable] Variable '{}' has type {:?}", name, var_type);
        let expr = self.inner.variable(name, var_type, span);
        self.store_expr(expr)
    }

    fn create_method_call(&mut self, receiver: NodeHandle, method: &str, args: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Get the receiver expression - it must exist
        let receiver_expr = match self.get_expr(receiver) {
            Some(expr) => expr,
            None => {
                // If receiver doesn't exist, create a placeholder error literal
                // Don't assume language-specific constructs like "self"
                self.inner.int_literal(0, span)
            }
        };

        let arg_exprs: Vec<_> = args.iter()
            .filter_map(|h| self.get_expr(*h))
            .collect();

        // Use Type::Any for method call return type - will be resolved during type checking/lowering
        let expr = self.inner.method_call(receiver_expr, method, arg_exprs, Type::Any, span);
        self.store_expr(expr)
    }

    fn create_static_method_call(&mut self, type_name: &str, method: &str, args: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Get the type from the registry
        let type_ty = self.get_type_by_name(type_name);

        let arg_exprs: Vec<_> = args.iter()
            .filter_map(|h| self.get_expr(*h))
            .collect();

        // Create a call to the static method using the mangled name format: TypeName$method
        // This matches the inherent method naming convention used in SSA/lowering
        let mangled_name = format!("{}${}", type_name, method);
        debug!("[STATIC_CALL] Creating static method call to '{}' with return type {:?}", mangled_name, type_ty);

        let callee = self.inner.variable(&mangled_name, Type::Any, span);
        let call_expr = self.inner.call_positional(callee, arg_exprs, type_ty, span);

        self.store_expr(call_expr)
    }

    fn create_array(&mut self, elements: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        let elem_exprs: Vec<_> = elements.iter()
            .filter_map(|h| self.get_expr(*h))
            .collect();

        let array_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: Some(zyntax_typed_ast::ConstValue::Int(elem_exprs.len() as i64)),
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        };

        let expr = self.inner.array_literal(elem_exprs, array_type, span);
        self.store_expr(expr)
    }

    fn create_struct_literal(&mut self, name: &str, fields: Vec<(String, NodeHandle)>) -> NodeHandle {
        let span = self.default_span();

        let field_exprs: Vec<(&str, TypedNode<TypedExpression>)> = fields.iter()
            .filter_map(|(field_name, h)| {
                self.get_expr(*h).map(|expr| (field_name.as_str(), expr))
            })
            .collect();

        // Look up the struct type by name - use Named type reference, not inline Struct
        // The type should have been declared earlier with create_struct_def
        let struct_type = self.get_type_by_name(name);

        let expr = self.inner.struct_literal(name, field_exprs, struct_type, span);
        self.store_expr(expr)
    }

    fn store_struct_field_init(&mut self, name: &str, value: NodeHandle) -> NodeHandle {
        let handle = self.alloc_handle();
        self.struct_field_inits.insert(handle, (name.to_string(), value));
        handle
    }

    fn get_struct_field_init(&self, handle: NodeHandle) -> Option<(String, NodeHandle)> {
        self.struct_field_inits.get(&handle).cloned()
    }

    fn create_cast(&mut self, expr_handle: NodeHandle, _target_type: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let expr = self.get_expr(expr_handle)
            .unwrap_or_else(|| self.inner.int_literal(0, span));

        // For now, cast to i64
        let target = Type::Primitive(PrimitiveType::I64);
        let cast_expr = self.inner.cast(expr, target, span);
        self.store_expr(cast_expr)
    }

    fn create_lambda(&mut self, _params: Vec<NodeHandle>, body: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        let body_expr = self.get_expr(body)
            .unwrap_or_else(|| self.inner.unit_literal(span));

        // Simple lambda with no params for now
        let lambda_type = Type::Function {
            params: vec![],
            return_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: zyntax_typed_ast::AsyncKind::Sync,
            calling_convention: zyntax_typed_ast::CallingConvention::Default,
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        };

        let expr = self.inner.lambda(vec![], body_expr, lambda_type, span);
        self.store_expr(expr)
    }

    fn create_match_expr(&mut self, scrutinee: NodeHandle, arms: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        let scrutinee_expr = self.get_expr(scrutinee)
            .unwrap_or_else(|| self.inner.int_literal(0, span));

        // Collect match arms from handles
        let typed_arms: Vec<TypedMatchArm> = arms.iter()
            .filter_map(|h| self.get_match_arm(*h))
            .collect();

        let match_expr = TypedMatchExpr {
            scrutinee: Box::new(scrutinee_expr),
            arms: typed_arms,
        };

        let expr = TypedNode {
            node: TypedExpression::Match(match_expr),
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };
        self.store_expr(expr)
    }

    fn create_match_arm(&mut self, pattern: NodeHandle, body: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        // Get the pattern
        let typed_pattern = self.get_pattern(pattern)
            .unwrap_or_else(|| {
                TypedNode {
                    node: TypedPattern::Wildcard,
                    ty: Type::Primitive(PrimitiveType::I32),
                    span,
                }
            });

        // Get the body expression
        let body_expr = self.get_expr(body)
            .unwrap_or_else(|| self.inner.int_literal(0, span));

        let arm = TypedMatchArm {
            pattern: Box::new(typed_pattern),
            guard: None,
            body: Box::new(body_expr),
        };

        self.store_match_arm(arm)
    }

    fn create_literal_pattern(&mut self, value: NodeHandle) -> NodeHandle {
        let span = self.default_span();

        // Get the expression and extract its literal value for the pattern
        let literal_pattern = if let Some(expr) = self.get_expr(value) {
            match &expr.node {
                TypedExpression::Literal(TypedLiteral::Integer(n)) => {
                    TypedLiteralPattern::Integer(*n)
                }
                TypedExpression::Literal(TypedLiteral::Bool(b)) => {
                    TypedLiteralPattern::Bool(*b)
                }
                TypedExpression::Literal(TypedLiteral::String(s)) => {
                    TypedLiteralPattern::String(s.clone())
                }
                TypedExpression::Literal(TypedLiteral::Char(c)) => {
                    TypedLiteralPattern::Char(*c)
                }
                _ => TypedLiteralPattern::Integer(0),
            }
        } else {
            TypedLiteralPattern::Integer(0)
        };

        let pattern = TypedNode {
            node: TypedPattern::Literal(literal_pattern),
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_wildcard_pattern(&mut self) -> NodeHandle {
        let span = self.default_span();

        let pattern = TypedNode {
            node: TypedPattern::Wildcard,
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_identifier_pattern(&mut self, name: &str) -> NodeHandle {
        let span = self.default_span();

        let pattern = TypedNode {
            node: TypedPattern::Identifier {
                name: InternedString::new_global(name),
                mutability: Mutability::Immutable,
            },
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_struct_pattern(&mut self, name: &str, fields: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect field patterns from handles
        let typed_fields: Vec<TypedFieldPattern> = fields.iter()
            .filter_map(|h| self.field_patterns.get(h).cloned())
            .collect();

        let pattern = TypedNode {
            node: TypedPattern::Struct {
                name: InternedString::new_global(name),
                fields: typed_fields,
            },
            // Use a simple primitive type for now - actual type would be resolved by type checker
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_field_pattern(&mut self, name: &str, pattern: Option<NodeHandle>) -> NodeHandle {
        let span = self.default_span();
        let handle = self.alloc_handle();

        // Get the nested pattern, or create an identifier pattern with the same name
        let nested_pattern = if let Some(p_handle) = pattern {
            self.get_pattern(p_handle).unwrap_or_else(|| TypedNode {
                node: TypedPattern::Identifier {
                    name: InternedString::new_global(name),
                    mutability: Mutability::Immutable,
                },
                ty: Type::Primitive(PrimitiveType::I32),
                span,
            })
        } else {
            // No explicit pattern means binding with the same name
            TypedNode {
                node: TypedPattern::Identifier {
                    name: InternedString::new_global(name),
                    mutability: Mutability::Immutable,
                },
                ty: Type::Primitive(PrimitiveType::I32),
                span,
            }
        };

        let field_pattern = TypedFieldPattern {
            name: InternedString::new_global(name),
            pattern: Box::new(nested_pattern),
        };

        self.field_patterns.insert(handle, field_pattern);
        handle
    }

    fn create_enum_pattern(&mut self, name: &str, variant: &str, fields: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect nested patterns from handles for the enum variant
        let typed_fields: Vec<TypedNode<TypedPattern>> = fields.iter()
            .filter_map(|h| self.get_pattern(*h))
            .collect();

        let pattern = TypedNode {
            node: TypedPattern::Enum {
                name: InternedString::new_global(name),
                variant: InternedString::new_global(variant),
                fields: typed_fields,
            },
            // Use a simple primitive type for now - actual type would be resolved by type checker
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_array_pattern(&mut self, elements: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect nested patterns from handles
        let typed_elements: Vec<TypedNode<TypedPattern>> = elements.iter()
            .filter_map(|h| self.get_pattern(*h))
            .collect();

        let pattern = TypedNode {
            node: TypedPattern::Array(typed_elements),
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                size: Some(ConstValue::Int(elements.len() as i64)),
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_tuple_pattern(&mut self, elements: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect nested patterns from handles
        let typed_elements: Vec<TypedNode<TypedPattern>> = elements.iter()
            .filter_map(|h| self.get_pattern(*h))
            .collect();

        // Build tuple type from element types
        let element_types: Vec<Type> = typed_elements.iter()
            .map(|p| p.ty.clone())
            .collect();

        let pattern = TypedNode {
            node: TypedPattern::Tuple(typed_elements),
            ty: Type::Tuple(element_types),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_range_pattern(&mut self, start: NodeHandle, end: NodeHandle, inclusive: bool) -> NodeHandle {
        let span = self.default_span();

        // Helper to extract literal pattern from a TypedPattern
        let extract_literal = |pattern_opt: Option<TypedNode<TypedPattern>>| -> TypedNode<TypedLiteralPattern> {
            if let Some(pattern) = pattern_opt {
                if let TypedPattern::Literal(lit) = pattern.node {
                    return TypedNode {
                        node: lit,
                        ty: pattern.ty,
                        span: pattern.span,
                    };
                }
            }
            // Default to integer 0
            TypedNode {
                node: TypedLiteralPattern::Integer(0),
                ty: Type::Primitive(PrimitiveType::I32),
                span,
            }
        };

        let start_lit = extract_literal(self.get_pattern(start));
        let end_lit = extract_literal(self.get_pattern(end));

        let pattern = TypedNode {
            node: TypedPattern::Range {
                start: Box::new(start_lit),
                end: Box::new(end_lit),
                inclusive,
            },
            ty: Type::Primitive(PrimitiveType::I32),
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_or_pattern(&mut self, patterns: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect nested patterns from handles
        let typed_patterns: Vec<TypedNode<TypedPattern>> = patterns.iter()
            .filter_map(|h| self.get_pattern(*h))
            .collect();

        // Use the type of the first pattern
        let ty = typed_patterns.first()
            .map(|p| p.ty.clone())
            .unwrap_or(Type::Primitive(PrimitiveType::I32));

        let pattern = TypedNode {
            node: TypedPattern::Or(typed_patterns),
            ty,
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_pointer_pattern(&mut self, inner: NodeHandle, mutable: bool) -> NodeHandle {
        let span = self.default_span();

        // Get the inner pattern
        let inner_pattern = self.get_pattern(inner)
            .unwrap_or_else(|| {
                TypedNode {
                    node: TypedPattern::Wildcard,
                    ty: Type::Primitive(PrimitiveType::I32),
                    span,
                }
            });

        let mutability = if mutable { Mutability::Mutable } else { Mutability::Immutable };
        let inner_ty = inner_pattern.ty.clone();

        // Create a Reference pattern (Zig uses * for pointers, Rust uses &)
        let pattern = TypedNode {
            node: TypedPattern::Reference {
                pattern: Box::new(inner_pattern),
                mutability,
            },
            ty: Type::Reference {
                ty: Box::new(inner_ty),
                mutability,
                lifetime: None,
                nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
            },
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_slice_pattern(&mut self, prefix: Vec<NodeHandle>, middle: Option<NodeHandle>, suffix: Vec<NodeHandle>) -> NodeHandle {
        let span = self.default_span();

        // Collect prefix patterns
        let prefix_patterns: Vec<TypedNode<TypedPattern>> = prefix.iter()
            .filter_map(|h| self.get_pattern(*h))
            .collect();

        // Get middle (rest) pattern if provided
        let middle_pattern = middle.and_then(|h| self.get_pattern(h));

        // Collect suffix patterns
        let suffix_patterns: Vec<TypedNode<TypedPattern>> = suffix.iter()
            .filter_map(|h| self.get_pattern(*h))
            .collect();

        // Determine element type from first available pattern
        let elem_ty = prefix_patterns.first()
            .or(middle_pattern.as_ref())
            .or(suffix_patterns.first())
            .map(|p| p.ty.clone())
            .unwrap_or(Type::Primitive(PrimitiveType::I32));

        let pattern = TypedNode {
            node: TypedPattern::Slice {
                prefix: prefix_patterns,
                middle: middle_pattern.map(Box::new),
                suffix: suffix_patterns,
            },
            // Use Array type for slices (similar to how Zig handles slices)
            ty: Type::Array {
                element_type: Box::new(elem_ty),
                size: None, // dynamic size for slices
                nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
            },
            span,
        };

        self.store_pattern(pattern)
    }

    fn create_error_pattern(&mut self, error_name: &str) -> NodeHandle {
        let span = self.default_span();
        let name = InternedString::new_global(error_name);

        // Error patterns in Zig are like enum variant patterns
        // error.OutOfMemory is essentially an enum variant
        let pattern = TypedNode {
            node: TypedPattern::Path {
                path: vec![InternedString::new_global("error"), name],
                args: None,
            },
            ty: Type::Error,
            span,
        };

        self.store_pattern(pattern)
    }

    fn get_identifier_name(&self, handle: NodeHandle) -> Option<String> {
        // Get the expression and check if it's a Variable (identifier)
        if let Some(expr_node) = self.get_expr(handle) {
            if let TypedExpression::Variable(name) = &expr_node.node {
                // Use resolve_global() to get the actual string value
                // (to_string() returns the debug format for InternedString)
                return name.resolve_global();
            }
        }
        None
    }

    fn get_string_literal_value(&self, handle: NodeHandle) -> Option<String> {
        // Get the expression and check if it's a string literal
        if let Some(expr_node) = self.get_expr(handle) {
            if let TypedExpression::Literal(TypedLiteral::String(value)) = &expr_node.node {
                // Use resolve_global() to get the actual string value
                return value.resolve_global();
            }
        }
        None
    }
}

// ============================================================================
// Command Interpreter
// ============================================================================

/// Interpreter that executes zpeg commands against a parse tree
pub struct CommandInterpreter<'a, H: AstHostFunctions> {
    /// The zpeg module containing rule commands
    module: &'a ZpegModule,
    /// Host functions for AST construction
    host: H,
    /// Current value stack
    value_stack: Vec<RuntimeValue>,
    /// Named variables
    variables: HashMap<String, RuntimeValue>,
    /// Current span being processed (start, end)
    current_span: (usize, usize),
}

impl<'a, H: AstHostFunctions> CommandInterpreter<'a, H> {
    /// Create a new interpreter
    pub fn new(module: &'a ZpegModule, host: H) -> Self {
        Self {
            module,
            host,
            value_stack: Vec::new(),
            variables: HashMap::new(),
            current_span: (0, 0),
        }
    }

    /// Set the current span for the node being processed
    pub fn set_current_span(&mut self, start: usize, end: usize) {
        self.current_span = (start, end);
    }

    /// Get the current span
    pub fn get_current_span(&self) -> (usize, usize) {
        self.current_span
    }

    /// Unescape a string literal, processing escape sequences
    fn unescape_string(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('\'') => result.push('\''),
                    Some('0') => result.push('\0'),
                    Some(other) => {
                        // Unknown escape, keep as-is
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Get the host functions (consuming the interpreter)
    pub fn into_host(self) -> H {
        self.host
    }

    /// Get mutable reference to host functions
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.host
    }

    /// Look up a builtin function by name, returning the runtime symbol if found
    pub fn lookup_builtin(&self, name: &str) -> Option<&str> {
        self.module.metadata.builtins.functions.get(name).map(|s| s.as_str())
    }

    /// Get all builtin mappings
    pub fn builtins(&self) -> &BuiltinMappings {
        &self.module.metadata.builtins
    }

    /// Execute commands for a rule given its parse tree node
    pub fn execute_rule(&mut self, rule_name: &str, text: &str, children: Vec<RuntimeValue>) -> Result<RuntimeValue> {
        log::trace!("[EXECUTE_RULE] {}: text='{}', children.len()={}, children={:?}", rule_name, text, children.len(), children);
        // Get commands for this rule
        let commands = match self.module.rule_commands(rule_name) {
            Some(cmds) => {
                log::trace!("[execute_rule] rule='{}' has {} commands, children={:?}", rule_name, cmds.commands.len(), children);
                cmds.commands.clone()
            }
            None => {
                log::trace!("[execute_rule] rule='{}' has NO commands, using defaults, children={:?}", rule_name, children);
                // No commands defined - default behavior depends on children
                if children.len() == 1 {
                    return Ok(children.into_iter().next().unwrap());
                } else if children.is_empty() {
                    // Leaf node - return text
                    return Ok(RuntimeValue::String(text.to_string()));
                } else {
                    // Multiple children - return as list
                    return Ok(RuntimeValue::List(children));
                }
            }
        };

        // Clear previous child variables before storing new ones
        // This is critical - otherwise old $2, $3, etc. from child rules pollute parent rules
        // But preserve $postfix_* variables which are used for postfix operation info
        self.variables.retain(|k, _| !k.starts_with('$') || k == "$text" || k.starts_with("$postfix_"));

        // Store children as $1, $2, etc.
        for (i, child) in children.into_iter().enumerate() {
            self.variables.insert(format!("${}", i + 1), child);
        }
        // Store text as $text
        self.variables.insert("$text".to_string(), RuntimeValue::String(text.to_string()));

        // Execute commands
        for cmd in &commands {
            log::trace!("[execute_rule] executing command: {:?}", cmd);
            self.execute_command(cmd)?;
        }

        // Return top of stack or null
        let result = self.value_stack.pop().unwrap_or(RuntimeValue::Null);
        Ok(result)
    }

    /// Execute a single command
    fn execute_command(&mut self, cmd: &AstCommand) -> Result<()> {
        match cmd {
            AstCommand::Define { node, args } => {
                // Resolve named arguments to a map of RuntimeValues
                let mut resolved_args: HashMap<String, RuntimeValue> = HashMap::new();
                for (key, arg) in &args.0 {
                    resolved_args.insert(key.clone(), self.resolve_arg(arg)?);
                }

                let result = self.define_node(node, resolved_args)?;
                self.value_stack.push(result);
            }

            AstCommand::Call { func, args } => {
                let resolved_args: Vec<RuntimeValue> = args.iter()
                    .map(|a| self.resolve_arg(a))
                    .collect::<Result<Vec<_>>>()?;

                let result = self.call_host_function(func, resolved_args)?;
                self.value_stack.push(result);
            }

            AstCommand::GetChild { index, name } => {
                let value = if let Some(idx) = index {
                    let key = format!("${}", idx + 1);
                    self.variables.get(&key).cloned().unwrap_or(RuntimeValue::Null)
                } else if let Some(n) = name {
                    let key = format!("${}", n);
                    self.variables.get(&key).cloned().unwrap_or(RuntimeValue::Null)
                } else {
                    RuntimeValue::Null
                };
                self.value_stack.push(value);
            }

            AstCommand::GetAllChildren => {
                // Collect all $N variables as a list
                let mut children: Vec<RuntimeValue> = Vec::new();
                let mut i = 1;
                while let Some(val) = self.variables.get(&format!("${}", i)) {
                    children.push(val.clone());
                    i += 1;
                }
                self.value_stack.push(RuntimeValue::List(children));
            }

            AstCommand::GetText => {
                let text = self.variables.get("$text")
                    .cloned()
                    .unwrap_or(RuntimeValue::String(String::new()));
                self.value_stack.push(text);
            }

            AstCommand::ParseInt => {
                if let Some(RuntimeValue::String(s)) = self.value_stack.pop() {
                    let value: i64 = s.trim().parse().unwrap_or(0);
                    self.value_stack.push(RuntimeValue::Int(value));
                }
            }

            AstCommand::ParseFloat => {
                if let Some(RuntimeValue::String(s)) = self.value_stack.pop() {
                    let value: f64 = s.trim().parse().unwrap_or(0.0);
                    self.value_stack.push(RuntimeValue::Float(value));
                }
            }

            AstCommand::Span => {
                // TODO: Get span from current parse position
                self.value_stack.push(RuntimeValue::Span { start: 0, end: 0 });
            }

            AstCommand::FoldBinary { operand_rule: _, operator_rule: _ } => {
                // Binary fold: takes alternating operands and operators from variables
                // e.g., for "1 + 2 - 3": $1=1, $2="+", $3=2, $4="-", $5=3
                // Result: ((1 + 2) - 3)

                // Collect all children in order
                let mut children: Vec<RuntimeValue> = Vec::new();
                let mut i = 1;
                while let Some(val) = self.variables.get(&format!("${}", i)) {
                    children.push(val.clone());
                    i += 1;
                }
                log::trace!("[FoldBinary] children.len()={}, children={:?}", children.len(), children);

                // Helper closure to convert RuntimeValue to NodeHandle, creating nodes for immediates
                let value_to_node = |host: &mut H, val: &RuntimeValue| -> Option<NodeHandle> {
                    match val {
                        RuntimeValue::Node(h) => Some(*h),
                        RuntimeValue::Int(n) => Some(host.create_int_literal(*n)),
                        RuntimeValue::Float(f) => Some(host.create_float_literal(*f)),
                        RuntimeValue::String(s) => {
                            // If it looks like a number, parse it
                            if let Ok(n) = s.parse::<i64>() {
                                Some(host.create_int_literal(n))
                            } else if let Ok(f) = s.parse::<f64>() {
                                Some(host.create_float_literal(f))
                            } else {
                                // Treat as variable reference
                                Some(host.create_variable(s))
                            }
                        }
                        RuntimeValue::Bool(b) => Some(host.create_bool_literal(*b)),
                        _ => None,
                    }
                };

                if children.is_empty() {
                    self.value_stack.push(RuntimeValue::Null);
                } else if children.len() == 1 {
                    // Single operand - convert to node if needed, then return it
                    log::trace!("[FoldBinary] single child, passing through");
                    let child = children.into_iter().next().unwrap();
                    // Convert immediate values to nodes
                    let result = if let Some(h) = value_to_node(&mut self.host, &child) {
                        RuntimeValue::Node(h)
                    } else {
                        child
                    };
                    self.value_stack.push(result);
                } else {
                    // Fold left-to-right: operand op operand op operand ...
                    // children[0], children[2], children[4], ... are operands (nodes or immediates)
                    // children[1], children[3], children[5], ... are operators (strings)

                    // Convert first operand to node
                    let first_node = value_to_node(&mut self.host, &children[0]);
                    let mut result = if let Some(h) = first_node {
                        RuntimeValue::Node(h)
                    } else {
                        children[0].clone()
                    };

                    let mut idx = 1;
                    while idx + 1 < children.len() {
                        let op = &children[idx];
                        let right = &children[idx + 1];

                        // Get operator string
                        let op_str = match op {
                            RuntimeValue::String(s) => s.clone(),
                            RuntimeValue::Node(_) => {
                                // Operator was parsed as a node, try to get its text
                                // This happens with rules like add_op = { "+" }
                                "+".to_string() // Default fallback
                            }
                            _ => "+".to_string(),
                        };

                        // Convert right operand to node if needed
                        let right_node = value_to_node(&mut self.host, right);

                        // Get node handles and create binary op
                        if let (RuntimeValue::Node(left_h), Some(right_h)) = (&result, right_node) {
                            let new_node = self.host.create_binary_op(&op_str, *left_h, right_h);
                            result = RuntimeValue::Node(new_node);
                        }

                        idx += 2;
                    }

                    self.value_stack.push(result);
                }
            }

            AstCommand::FoldPostfix => {
                // Fold postfix operations: primary ~ postfix_op*
                // $1 is the primary (base), $2, $3, ... are postfix operations
                // For now, just pass through the first child (primary)
                // If there are postfix ops, apply them left-to-right

                // Helper to unwrap nested single-element lists
                fn unwrap_nested_list(val: RuntimeValue) -> RuntimeValue {
                    match val {
                        RuntimeValue::List(list) if list.len() == 1 => {
                            unwrap_nested_list(list.into_iter().next().unwrap())
                        }
                        other => other,
                    }
                }

                // First check value stack (from get_all_children)
                let children: Vec<RuntimeValue> = if let Some(RuntimeValue::List(list)) = self.value_stack.pop() {
                    // Flatten nested single-element lists
                    list.into_iter().map(unwrap_nested_list).collect()
                } else {
                    // Fallback: collect from $N variables directly
                    let mut children: Vec<RuntimeValue> = Vec::new();
                    let mut i = 1;
                    while let Some(val) = self.variables.get(&format!("${}", i)) {
                        children.push(val.clone());
                        i += 1;
                    }
                    children
                };
                log::trace!("[FoldPostfix] children.len()={}, children={:?}", children.len(), children);

                if children.is_empty() {
                    self.value_stack.push(RuntimeValue::Null);
                } else if children.len() == 1 {
                    // No postfix ops - just pass through the primary
                    self.value_stack.push(children.into_iter().next().unwrap());
                } else {
                    // Apply postfix ops left-to-right
                    // children[0] is primary, children[1..] are postfix ops
                    let mut result = children[0].clone();
                    let postfix_ops = &children[1..];
                    let mut idx = 0;

                    while idx < postfix_ops.len() {
                        let postfix_op = &postfix_ops[idx];
                        // Each postfix_op was created by call_postfix/field_postfix/index_postfix
                        // and has info stored in self.variables
                        if let (RuntimeValue::Node(base_h), RuntimeValue::Node(op_h)) = (&result, postfix_op) {
                            // Check what kind of postfix operation this is
                            let postfix_key = format!("$postfix_{}", op_h.0);
                            log::trace!("[FoldPostfix] Looking up postfix_key={} for op_h={:?}", postfix_key, op_h);
                            log::trace!("[FoldPostfix] All variables with $postfix: {:?}",
                                self.variables.iter()
                                    .filter(|(k, _)| k.starts_with("$postfix"))
                                    .collect::<Vec<_>>());
                            if let Some(RuntimeValue::String(op_info)) = self.variables.get(&postfix_key) {
                                if op_info.starts_with("call:") {
                                    // Get the call arguments
                                    let args_key = format!("$postfix_{}_args", op_h.0);
                                    let call_args: Vec<NodeHandle> = self.variables.get(&args_key)
                                        .and_then(|v| match v {
                                            RuntimeValue::List(list) => Some(
                                                list.iter()
                                                    .filter_map(|v| match v {
                                                        RuntimeValue::Node(h) => Some(*h),
                                                        _ => None,
                                                    })
                                                    .collect()
                                            ),
                                            _ => None,
                                        })
                                        .unwrap_or_default();
                                    log::trace!("[FoldPostfix] creating call with {} args", call_args.len());
                                    // Use builtin resolution to map function names to runtime symbols
                                    let new_node = self.host.create_call_with_builtin_resolution(
                                        base_h.clone(),
                                        call_args,
                                        &self.module.metadata.builtins,
                                        &self.module.metadata.types,
                                    );
                                    result = RuntimeValue::Node(new_node);
                                } else if op_info.starts_with("field:") {
                                    let field_name = op_info.strip_prefix("field:").unwrap_or("");

                                    // Check if this is a method call: field access followed by call
                                    // and the method name is in the methods map
                                    let next_is_call = if idx + 1 < postfix_ops.len() {
                                        if let RuntimeValue::Node(next_op_h) = &postfix_ops[idx + 1] {
                                            let next_key = format!("$postfix_{}", next_op_h.0);
                                            self.variables.get(&next_key)
                                                .map(|v| matches!(v, RuntimeValue::String(s) if s.starts_with("call:")))
                                                .unwrap_or(false)
                                        } else {
                                            false
                                        }
                                    } else {
                                        false
                                    };

                                    // Check if this method name has a builtin mapping
                                    if next_is_call {
                                        if let Some(builtin_names) = self.module.metadata.builtins.methods.get(field_name) {
                                            // Method mapping found! Transform x.method(args) -> builtin(x, args)
                                            // Use the first builtin in the list (type-based dispatch can be added later)
                                            let builtin_name = &builtin_names[0];
                                            let next_op_h = if let RuntimeValue::Node(h) = &postfix_ops[idx + 1] { h } else { unreachable!() };
                                            let args_key = format!("$postfix_{}_args", next_op_h.0);
                                            let call_args: Vec<NodeHandle> = self.variables.get(&args_key)
                                                .and_then(|v| match v {
                                                    RuntimeValue::List(list) => Some(
                                                        list.iter()
                                                            .filter_map(|v| match v {
                                                                RuntimeValue::Node(h) => Some(*h),
                                                                _ => None,
                                                            })
                                                            .collect()
                                                    ),
                                                    _ => None,
                                                })
                                                .unwrap_or_default();

                                            // Resolve the builtin name through function mappings (e.g., vec_dot -> $Vector$dot_product)
                                            let resolved_name = self.module.metadata.builtins.functions.get(builtin_name)
                                                .map(|s| s.clone())
                                                .unwrap_or_else(|| builtin_name.to_string());

                                            // Create call with receiver as first argument
                                            let base_h_copy = base_h.clone();
                                            let mut all_args = vec![base_h_copy];
                                            all_args.extend(call_args);

                                            // Create callee identifier for the resolved builtin
                                            let callee = self.host.create_identifier(&resolved_name);
                                            let new_node = self.host.create_call(callee, all_args);
                                            log::trace!("[FoldPostfix] Method mapping: {}.{}() -> {}() [resolved: {}]",
                                                base_h.0, field_name, builtin_name, resolved_name);
                                            result = RuntimeValue::Node(new_node);

                                            // Skip the next postfix op (the call) since we consumed it
                                            idx += 1;
                                            log::trace!("[FoldPostfix] Method mapping: {}.{}() -> {}()", field_name, field_name, builtin_name);
                                        } else {
                                            // No method mapping, create normal field access
                                            let new_node = self.host.create_field_access(base_h.clone(), field_name);
                                            result = RuntimeValue::Node(new_node);
                                        }
                                    } else {
                                        // Not a method call, just field access
                                        let new_node = self.host.create_field_access(base_h.clone(), field_name);
                                        result = RuntimeValue::Node(new_node);
                                    }
                                } else if op_info.starts_with("index:") {
                                    let index_key = format!("$postfix_{}_index", op_h.0);
                                    if let Some(RuntimeValue::Node(index_h)) = self.variables.get(&index_key) {
                                        let new_node = self.host.create_index(base_h.clone(), index_h.clone());
                                        result = RuntimeValue::Node(new_node);
                                    }
                                }
                            } else {
                                // Fallback to apply_postfix (original behavior)
                                let new_node = self.host.apply_postfix(base_h.clone(), op_h.clone());
                                result = RuntimeValue::Node(new_node);
                            }
                        }
                        idx += 1;
                    }

                    self.value_stack.push(result);
                }
            }

            AstCommand::ApplyUnary => {
                // Apply unary operator: children = [op?, operand]
                // If there's one child, it's just the operand - pass through
                // If there are two children, first is operator string, second is operand
                let mut children: Vec<RuntimeValue> = Vec::new();
                let mut i = 1;
                while let Some(val) = self.variables.get(&format!("${}", i)) {
                    children.push(val.clone());
                    i += 1;
                }
                log::trace!("[ApplyUnary] children.len()={}, children={:?}", children.len(), children);

                // Helper to convert RuntimeValue to NodeHandle
                let value_to_node = |host: &mut H, val: &RuntimeValue| -> Option<NodeHandle> {
                    match val {
                        RuntimeValue::Node(h) => Some(*h),
                        RuntimeValue::Int(n) => Some(host.create_int_literal(*n)),
                        RuntimeValue::Float(f) => Some(host.create_float_literal(*f)),
                        RuntimeValue::Bool(b) => Some(host.create_bool_literal(*b)),
                        RuntimeValue::String(s) => Some(host.create_variable(s)),
                        RuntimeValue::List(list) => {
                            // Unwrap single-element lists
                            if list.len() == 1 {
                                match &list[0] {
                                    RuntimeValue::Node(h) => Some(*h),
                                    RuntimeValue::Int(n) => Some(host.create_int_literal(*n)),
                                    RuntimeValue::Float(f) => Some(host.create_float_literal(*f)),
                                    RuntimeValue::Bool(b) => Some(host.create_bool_literal(*b)),
                                    RuntimeValue::String(s) => Some(host.create_variable(s)),
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                };

                // Helper to recursively unwrap nested single-element lists
                fn unwrap_nested_list(val: RuntimeValue) -> RuntimeValue {
                    match val {
                        RuntimeValue::List(list) if list.len() == 1 => {
                            unwrap_nested_list(list.into_iter().next().unwrap())
                        }
                        other => other,
                    }
                }

                if children.is_empty() {
                    self.value_stack.push(RuntimeValue::Null);
                } else if children.len() == 1 {
                    // No unary operator - pass through, unwrapping nested lists
                    let result = unwrap_nested_list(children.into_iter().next().unwrap());
                    self.value_stack.push(result);
                } else {
                    // First child is operator, second is operand
                    let op_str = match &children[0] {
                        RuntimeValue::String(s) => s.clone(),
                        _ => "-".to_string(),
                    };
                    let operand = unwrap_nested_list(children[1].clone());
                    if let Some(operand_h) = value_to_node(&mut self.host, &operand) {
                        let new_node = self.host.create_unary_op(&op_str, operand_h);
                        self.value_stack.push(RuntimeValue::Node(new_node));
                    } else {
                        self.value_stack.push(RuntimeValue::Null);
                    }
                }
            }

            AstCommand::FoldLeftOps => {
                // Fold binary operations with operators interleaved
                // children = [operand, op, operand, op, operand, ...]
                // Same as FoldBinary but doesn't use named rules
                let mut children: Vec<RuntimeValue> = Vec::new();
                let mut i = 1;
                while let Some(val) = self.variables.get(&format!("${}", i)) {
                    children.push(val.clone());
                    i += 1;
                }
                log::trace!("[FoldLeftOps] children.len()={}, children={:?}", children.len(), children);

                // Helper to recursively unwrap nested single-element lists
                fn unwrap_nested_list(val: RuntimeValue) -> RuntimeValue {
                    match val {
                        RuntimeValue::List(list) if list.len() == 1 => {
                            unwrap_nested_list(list.into_iter().next().unwrap())
                        }
                        other => other,
                    }
                }

                let value_to_node = |host: &mut H, val: &RuntimeValue| -> Option<NodeHandle> {
                    let unwrapped = unwrap_nested_list(val.clone());
                    match unwrapped {
                        RuntimeValue::Node(h) => Some(h),
                        RuntimeValue::Int(n) => Some(host.create_int_literal(n)),
                        RuntimeValue::Float(f) => Some(host.create_float_literal(f)),
                        RuntimeValue::Bool(b) => Some(host.create_bool_literal(b)),
                        RuntimeValue::String(s) => {
                            // If it looks like a number, parse it
                            if let Ok(n) = s.parse::<i64>() {
                                Some(host.create_int_literal(n))
                            } else if let Ok(f) = s.parse::<f64>() {
                                Some(host.create_float_literal(f))
                            } else {
                                Some(host.create_variable(&s))
                            }
                        }
                        _ => None,
                    }
                };

                if children.is_empty() {
                    self.value_stack.push(RuntimeValue::Null);
                } else if children.len() == 1 {
                    // Single operand - unwrap and pass through
                    let result = unwrap_nested_list(children.into_iter().next().unwrap());
                    self.value_stack.push(result);
                } else {
                    // Fold: operand op operand op ...
                    let first_val = unwrap_nested_list(children[0].clone());
                    let mut result = if let Some(h) = value_to_node(&mut self.host, &first_val) {
                        RuntimeValue::Node(h)
                    } else {
                        first_val
                    };

                    let mut idx = 1;
                    while idx + 1 < children.len() {
                        let op = &children[idx];
                        let right = unwrap_nested_list(children[idx + 1].clone());

                        let op_str = match op {
                            RuntimeValue::String(s) => s.clone(),
                            _ => "+".to_string(),
                        };

                        let right_node = value_to_node(&mut self.host, &right);

                        if let (RuntimeValue::Node(left_h), Some(right_h)) = (&result, right_node) {
                            let left_h = *left_h; // Copy to avoid borrow
                            // Check if this operator has a builtin overload
                            if let Some(builtin_names) = self.module.metadata.builtins.operators.get(&op_str) {
                                // Operator overload found! Transform x op y -> builtin(x, y)
                                // Use the first builtin in the list (type-based dispatch can be added later)
                                let builtin_name = &builtin_names[0];
                                // Resolve the builtin name through function mappings (e.g., vec_dot -> $Vector$dot_product)
                                let resolved_name = self.module.metadata.builtins.functions.get(builtin_name)
                                    .map(|s| s.as_str())
                                    .unwrap_or(builtin_name);
                                let callee = self.host.create_identifier(resolved_name);
                                let new_node = self.host.create_call(callee, vec![left_h, right_h]);
                                result = RuntimeValue::Node(new_node);
                                log::trace!("[FoldLeftOps] Operator overload: {} {} {} -> {}({}, {}) [resolved: {}]",
                                    left_h.0, op_str, right_h.0, builtin_name, left_h.0, right_h.0, resolved_name);
                            } else {
                                // No overload, use standard binary op
                                let new_node = self.host.create_binary_op(&op_str, left_h, right_h);
                                result = RuntimeValue::Node(new_node);
                            }
                        }

                        idx += 2;
                    }

                    self.value_stack.push(result);
                }
            }

            AstCommand::FoldLeft { op, transform } => {
                // Fold left with custom operation
                // For pipes: a |> f(b) |> g() becomes g(f(a, b))
                let mut children: Vec<RuntimeValue> = Vec::new();
                let mut i = 1;
                while let Some(val) = self.variables.get(&format!("${}", i)) {
                    children.push(val.clone());
                    i += 1;
                }
                log::trace!("[FoldLeft] op={:?}, transform={:?}, children.len()={}, children={:?}", op, transform, children.len(), children);

                // Helper to recursively unwrap nested single-element lists
                fn unwrap_nested_list(val: RuntimeValue) -> RuntimeValue {
                    match val {
                        RuntimeValue::List(list) if list.len() == 1 => {
                            unwrap_nested_list(list.into_iter().next().unwrap())
                        }
                        other => other,
                    }
                }

                let value_to_node = |host: &mut H, val: &RuntimeValue| -> Option<NodeHandle> {
                    let unwrapped = unwrap_nested_list(val.clone());
                    match unwrapped {
                        RuntimeValue::Node(h) => Some(h),
                        RuntimeValue::Int(n) => Some(host.create_int_literal(n)),
                        RuntimeValue::Float(f) => Some(host.create_float_literal(f)),
                        RuntimeValue::Bool(b) => Some(host.create_bool_literal(b)),
                        RuntimeValue::String(s) => Some(host.create_variable(&s)),
                        _ => None,
                    }
                };

                if children.is_empty() {
                    self.value_stack.push(RuntimeValue::Null);
                } else if children.len() == 1 {
                    let result = unwrap_nested_list(children.into_iter().next().unwrap());
                    self.value_stack.push(result);
                } else if op == "pipe" {
                    // Pipe transform: a |> f(b) becomes f(a, b)
                    // children[0] is the initial value
                    // children[1..] are pipe targets with callee and args
                    let first_val = unwrap_nested_list(children[0].clone());
                    let mut result = if let Some(h) = value_to_node(&mut self.host, &first_val) {
                        RuntimeValue::Node(h)
                    } else {
                        first_val
                    };

                    for pipe_target in &children[1..] {
                        // pipe_target should be a PipeTarget node with callee and args
                        // For now, handle it as a call where we prepend the current result
                        if let RuntimeValue::Node(target_h) = pipe_target {
                            // Get the target info from variables
                            let target_key = format!("$pipe_target_{}", target_h.0);
                            if let Some(info) = self.variables.get(&target_key) {
                                log::trace!("[FoldLeft] pipe target info: {:?}", info);
                            }
                            // For now, apply as a function call with result prepended
                            if let RuntimeValue::Node(res_h) = &result {
                                // Use apply_postfix for now
                                let new_node = self.host.apply_postfix(*res_h, *target_h);
                                result = RuntimeValue::Node(new_node);
                            }
                        }
                    }

                    self.value_stack.push(result);
                } else {
                    // Generic fold_left for logical operators (||, &&)
                    let first_val = unwrap_nested_list(children[0].clone());
                    let mut result = if let Some(h) = value_to_node(&mut self.host, &first_val) {
                        RuntimeValue::Node(h)
                    } else {
                        first_val
                    };

                    // For ||, &&: children alternate between operands
                    let mut idx = 1;
                    while idx < children.len() {
                        let right = unwrap_nested_list(children[idx].clone());
                        let right_node = value_to_node(&mut self.host, &right);

                        if let (RuntimeValue::Node(left_h), Some(right_h)) = (&result, right_node) {
                            let new_node = self.host.create_binary_op(&op, *left_h, right_h);
                            result = RuntimeValue::Node(new_node);
                        }

                        idx += 1;
                    }

                    self.value_stack.push(result);
                }
            }

            AstCommand::MapChildren { rule, commands } => {
                let _ = rule; // Filter children by rule name
                let results: Vec<RuntimeValue> = Vec::new();

                // Execute commands for each matching child
                for cmd in commands {
                    self.execute_command(cmd)?;
                }

                self.value_stack.push(RuntimeValue::List(results));
            }

            AstCommand::MatchRule { cases } => {
                // Get the rule name from first child
                // For now, execute first case that exists
                for (_rule_name, cmds) in cases {
                    for cmd in cmds {
                        self.execute_command(cmd)?;
                    }
                    break;
                }
            }

            AstCommand::Store { name } => {
                if let Some(value) = self.value_stack.last().cloned() {
                    // Store with $ prefix for consistency with resolve_arg
                    let key = if name.starts_with('$') {
                        name.clone()
                    } else {
                        format!("${}", name)
                    };
                    self.variables.insert(key, value);
                }
            }

            AstCommand::Load { name } => {
                // Look up with $ prefix if not provided
                let key = if name.starts_with('$') {
                    name.clone()
                } else {
                    format!("${}", name)
                };
                let value = self.variables.get(&key).cloned().unwrap_or(RuntimeValue::Null);
                self.value_stack.push(value);
            }

            AstCommand::Return => {
                // Value is already on stack - nothing to do
            }
        }

        Ok(())
    }

    /// Resolve a command argument to a value
    fn resolve_arg(&mut self, arg: &CommandArg) -> Result<RuntimeValue> {
        match arg {
            CommandArg::ChildRef(ref_str) => {
                // $1, $2, $name, $text, $result
                if ref_str == "$result" {
                    Ok(self.value_stack.last().cloned().unwrap_or(RuntimeValue::Null))
                } else {
                    Ok(self.variables.get(ref_str).cloned().unwrap_or(RuntimeValue::Null))
                }
            }
            CommandArg::StringLit(s) => Ok(RuntimeValue::String(s.clone())),
            CommandArg::IntLit(n) => Ok(RuntimeValue::Int(*n)),
            CommandArg::BoolLit(b) => Ok(RuntimeValue::Bool(*b)),
            CommandArg::List(items) => {
                // Resolve each item in the list - need to collect to avoid borrow issues
                let items_clone: Vec<CommandArg> = items.clone();
                let resolved: Vec<RuntimeValue> = items_clone.iter()
                    .map(|item| self.resolve_arg(item))
                    .collect::<Result<Vec<_>>>()?;
                Ok(RuntimeValue::List(resolved))
            }
            CommandArg::Nested(cmd) => {
                // Execute nested command inline
                match cmd.as_ref() {
                    AstCommand::GetChild { index, name } => {
                        let value = if let Some(idx) = index {
                            let key = format!("${}", idx + 1);
                            self.variables.get(&key).cloned().unwrap_or(RuntimeValue::Null)
                        } else if let Some(n) = name {
                            let key = format!("${}", n);
                            self.variables.get(&key).cloned().unwrap_or(RuntimeValue::Null)
                        } else {
                            RuntimeValue::Null
                        };
                        Ok(value)
                    }
                    AstCommand::GetText => {
                        let text = self.variables.get("$text")
                            .cloned()
                            .unwrap_or(RuntimeValue::String(String::new()));
                        Ok(text)
                    }
                    AstCommand::GetAllChildren => {
                        let mut children: Vec<RuntimeValue> = Vec::new();
                        let mut i = 1;
                        while let Some(val) = self.variables.get(&format!("${}", i)) {
                            children.push(val.clone());
                            i += 1;
                        }
                        Ok(RuntimeValue::List(children))
                    }
                    AstCommand::Define { node, args } => {
                        // Execute nested define command - resolve its args first
                        let args_clone = args.clone();
                        let mut resolved_args: HashMap<String, RuntimeValue> = HashMap::new();
                        for (key, arg) in &args_clone.0 {
                            resolved_args.insert(key.clone(), self.resolve_arg(arg)?);
                        }
                        self.define_node(node, resolved_args)
                    }
                    _ => {
                        // For other commands that need mutation, we can't execute them here
                        // This should be rare - most nested args are get_child or define
                        Ok(RuntimeValue::Null)
                    }
                }
            }
        }
    }

    /// Define an AST node with named arguments (new format)
    fn define_node(&mut self, node_type: &str, args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
        match node_type {
            "int_literal" | "integer" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::Int(n)) => *n,
                    Some(RuntimeValue::String(s)) => s.parse().unwrap_or(0),
                    _ => 0,
                };
                let handle = self.host.create_int_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "float_literal" | "float" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::Float(n)) => *n,
                    Some(RuntimeValue::String(s)) => s.parse().unwrap_or(0.0),
                    _ => 0.0,
                };
                let handle = self.host.create_float_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "string_literal" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                // Strip surrounding quotes if present (from grammar capture)
                let stripped = if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
                    &value[1..value.len()-1]
                } else if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
                    &value[1..value.len()-1]
                } else {
                    &value
                };
                // Process escape sequences
                let unescaped = Self::unescape_string(stripped);
                let handle = self.host.create_string_literal(&unescaped);
                Ok(RuntimeValue::Node(handle))
            }

            "bool_literal" | "bool" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::Bool(b)) => *b,
                    Some(RuntimeValue::String(s)) => s == "true",
                    _ => false,
                };
                let handle = self.host.create_bool_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "suffixed_literal" => {
                // Parse suffixed literals like "1000ms", "5s", "3ns"
                // Lookup suffix in registry to find abstract type and create Type::from_{suffix}(value) call
                let text = match args.get("text") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("suffixed_literal: missing text".into())),
                };

                // Split into number and suffix
                let mut num_end = 0;
                for (i, ch) in text.char_indices() {
                    if ch.is_ascii_digit() || ch == '-' {
                        num_end = i + ch.len_utf8();
                    } else {
                        break;
                    }
                }

                let num_str = &text[..num_end];
                let suffix = &text[num_end..];

                // Validate that we have both a number and a suffix
                if num_str.is_empty() {
                    return Err(crate::error::ZynPegError::CodeGenError(
                        format!("Suffix literal '{}' must start with a number", text)
                    ));
                }
                if suffix.is_empty() {
                    return Err(crate::error::ZynPegError::CodeGenError(
                        format!("Suffix literal '{}' must have a suffix after the number", text)
                    ));
                }

                debug!("[SUFFIX LITERAL] Parsing '{}': num='{}', suffix='{}'", text, num_str, suffix);

                // Look up suffix in registry to find the abstract type
                let type_name = match self.host.lookup_suffix(suffix) {
                    Some(name) => name,
                    None => {
                        // Suffix not registered - provide helpful error message
                        return Err(crate::error::ZynPegError::CodeGenError(
                            format!(
                                "Unknown suffix '{}' in literal '{}'. \
                                No abstract type has been declared with this suffix. \
                                \nHint: Declare an abstract type like: abstract Duration(i64) with Suffixes(\"ms, s\") to use '{}' suffix literals.",
                                suffix, text, suffix
                            )
                        ));
                    }
                };

                debug!("[SUFFIX LITERAL] Suffix '{}' maps to abstract type '{}'", suffix, type_name);

                // Parse the number value
                let num: i64 = num_str.parse()
                    .map_err(|e| crate::error::ZynPegError::CodeGenError(
                        format!("Invalid number '{}' in suffix literal '{}': {}", num_str, text, e)
                    ))?;

                // Validate that the abstract type is actually registered in declared_types
                if self.host.lookup_declared_type(&type_name).is_none() {
                    return Err(crate::error::ZynPegError::CodeGenError(
                        format!(
                            "Abstract type '{}' referenced by suffix '{}' is not defined. \
                            \nThe suffix was registered but the type definition is missing.",
                            type_name, suffix
                        )
                    ));
                }

                debug!("[SUFFIX LITERAL] Abstract type '{}' is registered in type registry", type_name);

                // Call the appropriate from_<suffix> constructor function
                // e.g., "1000ms" -> Duration::from_ms(1000)
                // e.g., "2s" -> Duration::from_s(2)

                // Get the abstract type from the registry
                let abstract_type = if let Some(ty) = self.host.lookup_declared_type(&type_name) {
                    ty
                } else {
                    // Should not reach here since we validated above, but handle gracefully
                    return Err(crate::error::ZynPegError::CodeGenError(
                        format!("Failed to retrieve abstract type '{}' after validation", type_name)
                    ));
                };

                // Construct the constructor function name: from_<suffix>
                let constructor_name = format!("from_{}", suffix);
                debug!("[SUFFIX LITERAL] Looking for constructor function '{}' for type '{}'", constructor_name, type_name);

                // Create a function call: TypeName::from_suffix(num)
                // This is represented as a static method call
                let num_literal = self.host.create_int_literal(num);
                let handle = self.host.create_static_method_call(
                    &type_name,
                    &constructor_name,
                    vec![num_literal]
                );

                debug!("[SUFFIX LITERAL] Successfully parsed suffix literal '{}' as constructor call {}::{}({})",
                    text, type_name, constructor_name, num);
                Ok(RuntimeValue::Node(handle))
            }

            "duration_literal" => {
                // Parse duration like "5s", "1h", "500ms" into milliseconds as i64
                let text = match args.get("value") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("duration_literal: missing value".into())),
                };

                // Extract number and unit
                let (num_str, unit) = if text.ends_with("ms") {
                    (&text[..text.len()-2], "ms")
                } else if text.ends_with('s') {
                    (&text[..text.len()-1], "s")
                } else if text.ends_with('m') {
                    (&text[..text.len()-1], "m")
                } else if text.ends_with('h') {
                    (&text[..text.len()-1], "h")
                } else if text.ends_with('d') {
                    (&text[..text.len()-1], "d")
                } else {
                    return Err(crate::error::ZynPegError::CodeGenError(format!("duration_literal: unknown unit in '{}'", text)));
                };

                let num: i64 = num_str.parse().unwrap_or(0);

                // Convert to milliseconds
                let ms = match unit {
                    "ms" => num,
                    "s" => num * 1000,
                    "m" => num * 60 * 1000,
                    "h" => num * 60 * 60 * 1000,
                    "d" => num * 24 * 60 * 60 * 1000,
                    _ => num,
                };

                let handle = self.host.create_int_literal(ms);
                Ok(RuntimeValue::Node(handle))
            }

            "identifier" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let handle = self.host.create_identifier(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "binary_op" => {
                let op = match args.get("op").or(args.get("operator")) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "+".to_string(),
                };
                let left = match args.get("left") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("binary_op: missing left operand".into())),
                };
                let right = match args.get("right") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("binary_op: missing right operand".into())),
                };
                let handle = self.host.create_binary_op(&op, left, right);
                Ok(RuntimeValue::Node(handle))
            }

            "unary_op" | "unary" => {
                let op = match args.get("op").or(args.get("operator")) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "-".to_string(),
                };
                let operand = match args.get("operand") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("unary_op: missing operand".into())),
                };
                let handle = self.host.create_unary_op(&op, operand);
                Ok(RuntimeValue::Node(handle))
            }

            "return_stmt" => {
                log::trace!("[RETURN_STMT] args = {:?}", args);
                let value = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => {
                        log::trace!("[RETURN_STMT] value is Node: {:?}", h);
                        Some(*h)
                    },
                    Some(other) => {
                        log::trace!("[RETURN_STMT] value is not Node: {:?}", other);
                        None
                    }
                    None => {
                        log::trace!("[RETURN_STMT] value is missing");
                        None
                    }
                };
                let handle = self.host.create_return(value);
                Ok(RuntimeValue::Node(handle))
            }

            "function" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    Some(_) | None => "anonymous".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_block(vec![]),
                };

                // Get return type from args, default to void for ZynML-style dynamic functions
                let return_type = match args.get("return_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    Some(RuntimeValue::String(s)) => self.host.create_primitive_type(s),
                    _ => self.host.create_primitive_type("void"),
                };
                let handle = self.host.create_function(&name, params, return_type, body);
                Ok(RuntimeValue::Node(handle))
            }

            "async_function" => {
                // Async function declaration - same as function but with is_async=true
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "anonymous".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_block(vec![]),
                };

                // Get return type from args, default to void
                let return_type = match args.get("return_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    Some(RuntimeValue::String(s)) => self.host.create_primitive_type(s),
                    _ => self.host.create_primitive_type("void"),
                };
                let handle = self.host.create_async_function(&name, params, return_type, body);
                Ok(RuntimeValue::Node(handle))
            }

            "extern_function" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "anonymous".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let return_type = match args.get("return_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };

                // Create an extern function declaration (no body, is_external = true)
                let handle = self.host.create_extern_function(&name, params, return_type);
                Ok(RuntimeValue::Node(handle))
            }

            "async_function" => {
                // Handle name: can be String (from get_text) or Node (from identifier rule)
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    Some(RuntimeValue::Node(h)) => {
                        // Try to extract the identifier name from the node
                        self.host.get_identifier_name(*h)
                            .unwrap_or_else(|| "anonymous_async".to_string())
                    },
                    _ => "anonymous_async".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_block(vec![]),
                };

                let return_type = match args.get("return_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };

                // Create an async function declaration (is_async = true)
                let handle = self.host.create_async_function(&name, params, return_type, body);
                Ok(RuntimeValue::Node(handle))
            }

            "struct" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousStruct".to_string(),
                };
                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_struct(&name, fields);
                Ok(RuntimeValue::Node(handle))
            }

            "enum" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousEnum".to_string(),
                };
                let variants: Vec<NodeHandle> = match args.get("variants") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_enum(&name, variants);
                Ok(RuntimeValue::Node(handle))
            }

            "field" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "field".to_string(),
                };
                let ty = match args.get("type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };
                let handle = self.host.create_field(&name, ty);
                Ok(RuntimeValue::Node(handle))
            }

            "variant" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Variant".to_string(),
                };
                let handle = self.host.create_variant(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "program" => {
                let handle = self.host.create_program();
                // Add declarations from args
                if let Some(RuntimeValue::List(decls)) = args.get("declarations") {
                    for decl in decls {
                        if let RuntimeValue::Node(decl_h) = decl {
                            self.host.program_add_decl(handle, *decl_h);
                        }
                    }
                } else if let Some(RuntimeValue::Node(decl)) = args.get("declarations") {
                    self.host.program_add_decl(handle, *decl);
                }
                Ok(RuntimeValue::Node(handle))
            }

            "block" => {
                let statements: Vec<NodeHandle> = match args.get("statements") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    // Handle single statement (when statement* matches exactly one)
                    Some(RuntimeValue::Node(h)) => vec![*h],
                    _ => vec![],
                };
                let handle = self.host.create_block(statements);
                Ok(RuntimeValue::Node(handle))
            }

            "expr_block" => {
                // Expression block: wraps a single expression in a block
                let expr = match args.get("expr") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("expr_block: missing expr".into())),
                };

                // create_block will automatically wrap expressions in expression statements
                let handle = self.host.create_block(vec![expr]);
                Ok(RuntimeValue::Node(handle))
            }

            "param" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    // If name is a node (e.g., identifier), try to extract its string value
                    Some(RuntimeValue::Node(h)) => {
                        self.host.get_identifier_name(*h)
                            .unwrap_or_else(|| "arg".to_string())
                    }
                    _ => "arg".to_string(),
                };
                // If type is not provided, default to Any (dynamic box pointer)
                let ty = match args.get("type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => {
                        // Create a handle for Type::Any
                        let handle = self.host.alloc_handle();
                        self.host.store_type(handle, zyntax_typed_ast::type_registry::Type::Any);
                        handle
                    }
                };
                let handle = self.host.create_param(&name, ty);
                Ok(RuntimeValue::Node(handle))
            }

            "primitive_type" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => {
                        // Map language-specific type names to internal primitive names
                        match s.as_str() {
                            // Haxe types
                            "Int" => "i32".to_string(),
                            "Float" => "f64".to_string(),
                            "Bool" => "bool".to_string(),
                            "Void" => "void".to_string(),
                            "String" => "string".to_string(),
                            // Already internal names
                            other => other.to_string(),
                        }
                    }
                    _ => "i32".to_string(),
                };
                let handle = self.host.create_primitive_type(&name);
                Ok(RuntimeValue::Node(handle))
            }

            // Type inference marker - tells type checker to infer the type
            "infer_type" | "auto" | "var" => {
                // Return None/null to indicate type should be inferred
                // The type checker will infer the type from the initializer
                Ok(RuntimeValue::Null)
            }

            // ===== ADDITIONAL EXPRESSIONS =====

            "char_literal" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::String(s)) if !s.is_empty() => s.chars().next().unwrap_or('\0'),
                    _ => '\0',
                };
                let handle = self.host.create_char_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "variable" | "var_ref" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let handle = self.host.create_variable(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "path" => {
                // Path expression: Type::method or module::function
                // Creates a variable with the mangled name "Type::method"
                let segments = match args.get("segments") {
                    Some(RuntimeValue::List(arr)) => {
                        let mut path_parts = Vec::new();
                        for val in arr {
                            if let RuntimeValue::String(s) = val {
                                path_parts.push(s.clone());
                            }
                        }
                        path_parts
                    }
                    _ => return Err(crate::error::ZynPegError::CodeGenError("path: missing segments".into())),
                };

                if segments.len() != 2 {
                    return Err(crate::error::ZynPegError::CodeGenError(
                        format!("path: expected 2 segments, got {}", segments.len())
                    ));
                }

                // Create mangled name: "Type::method"
                let mangled_name = format!("{}::{}", segments[0], segments[1]);
                let handle = self.host.create_variable(&mangled_name);
                Ok(RuntimeValue::Node(handle))
            }

            "call_expr" | "call" => {
                // Callee can be either a Node (expression) or String (identifier)
                let callee = match args.get("callee") {
                    Some(RuntimeValue::Node(h)) => *h,
                    Some(RuntimeValue::String(name)) => {
                        // Create a variable reference for the callee
                        self.host.create_variable(name)
                    }
                    _ => return Err(crate::error::ZynPegError::CodeGenError("call: missing callee".into())),
                };
                let call_args: Vec<NodeHandle> = match args.get("args") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    Some(RuntimeValue::Null) => vec![],
                    None => vec![],
                    _ => vec![],
                };
                // Use builtin resolution to map function names to runtime symbols
                let handle = self.host.create_call_with_builtin_resolution(
                    callee,
                    call_args,
                    &self.module.metadata.builtins,
                    &self.module.metadata.types,
                );
                Ok(RuntimeValue::Node(handle))
            }

            "call_or_primary" => {
                // call_or_primary handles the common pattern: primary ~ ("(" ~ args? ~ ")")?
                // If args is Null (no parentheses), just return the callee as-is
                // If args exists (even empty), create a function call
                let callee = match args.get("callee") {
                    Some(RuntimeValue::Node(h)) => *h,
                    Some(RuntimeValue::String(name)) => {
                        self.host.create_variable(name)
                    }
                    _ => return Err(crate::error::ZynPegError::CodeGenError("call_or_primary: missing callee".into())),
                };

                // Check if args is Null (no parentheses) vs empty list (parentheses but no args)
                match args.get("args") {
                    Some(RuntimeValue::Null) | None => {
                        // No parentheses - just pass through the callee
                        Ok(RuntimeValue::Node(callee))
                    }
                    Some(RuntimeValue::List(list)) => {
                        // Has parentheses - create a call
                        let call_args: Vec<NodeHandle> = list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect();
                        let handle = self.host.create_call(callee, call_args);
                        Ok(RuntimeValue::Node(handle))
                    }
                    _ => {
                        // Other value - treat as call with no args
                        let handle = self.host.create_call(callee, vec![]);
                        Ok(RuntimeValue::Node(handle))
                    }
                }
            }

            "method_call" => {
                let receiver = match args.get("receiver") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("method_call: missing receiver".into())),
                };
                let method = match args.get("method") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let call_args: Vec<NodeHandle> = match args.get("args") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_method_call(receiver, &method, call_args);
                Ok(RuntimeValue::Node(handle))
            }

            "field_access" => {
                let object = match args.get("object") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("field_access: missing object".into())),
                };
                let field = match args.get("field") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let handle = self.host.create_field_access(object, &field);
                Ok(RuntimeValue::Node(handle))
            }

            "index" | "index_expr" => {
                let object = match args.get("object").or(args.get("array")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("index: missing object".into())),
                };
                let index = match args.get("index") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("index: missing index".into())),
                };
                let handle = self.host.create_index(object, index);
                Ok(RuntimeValue::Node(handle))
            }

            // Postfix operation markers - store info for later application in apply_postfix
            // These are handled specially since they need to interact with TypedAstBuilder's postfix_ops
            // "call_args" is an alias used in ZynML grammar
            "call_postfix" | "call_args" => {
                // Extract call arguments
                let call_args: Vec<NodeHandle> = match args.get("args") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    Some(RuntimeValue::Node(h)) => vec![*h], // Single argument
                    Some(RuntimeValue::Null) => vec![],
                    None => vec![],
                    _ => vec![],
                };
                log::trace!("[define_node] call_postfix with {} args", call_args.len());
                // Create a marker handle - we need to downcast to TypedAstBuilder
                // For now, create a placeholder node and store postfix info
                let handle = self.host.create_int_literal(0); // Marker placeholder
                // Store postfix info in a special way - use the handle as key
                // Note: This requires TypedAstBuilder to have postfix_ops accessible
                // For the generic case, we'll store this info as a variable
                self.variables.insert(
                    format!("$postfix_{}", handle.0),
                    RuntimeValue::String(format!("call:{}", call_args.len()))
                );
                // Store args for later retrieval
                self.variables.insert(
                    format!("$postfix_{}_args", handle.0),
                    RuntimeValue::List(call_args.iter().map(|h| RuntimeValue::Node(*h)).collect())
                );
                Ok(RuntimeValue::Node(handle))
            }

            // "member" is an alias used in ZynML grammar for field access
            "field_postfix" | "member" => {
                let field_name = match args.get("field") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    Some(RuntimeValue::Node(h)) => {
                        // Field might be stored as an identifier node - get text from variable
                        self.variables.get(&format!("${}", h.0))
                            .and_then(|v| match v {
                                RuntimeValue::String(s) => Some(s.clone()),
                                _ => None
                            })
                            .unwrap_or_else(|| format!("field_{}", h.0))
                    }
                    _ => String::new(),
                };
                log::trace!("[define_node] field_postfix with field={}", field_name);
                let handle = self.host.create_int_literal(0); // Marker placeholder
                self.variables.insert(
                    format!("$postfix_{}", handle.0),
                    RuntimeValue::String(format!("field:{}", field_name))
                );
                Ok(RuntimeValue::Node(handle))
            }

            // "index" is an alias used in ZynML grammar
            "index_postfix" | "index" => {
                let index = match args.get("index") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("index_postfix/index: missing index".into())),
                };
                log::trace!("[define_node] index_postfix with index={:?}", index);
                let handle = self.host.create_int_literal(0); // Marker placeholder
                self.variables.insert(
                    format!("$postfix_{}", handle.0),
                    RuntimeValue::String(format!("index:{}", index.0))
                );
                self.variables.insert(
                    format!("$postfix_{}_index", handle.0),
                    RuntimeValue::Node(index)
                );
                Ok(RuntimeValue::Node(handle))
            }

            "field_init" => {
                let field_name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let value_handle = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => NodeHandle(0),
                };
                // Store the field init for later retrieval by struct_literal
                let handle = self.host.store_struct_field_init(&field_name, value_handle);
                Ok(RuntimeValue::Node(handle))
            }

            "array" | "array_literal" => {
                let elements: Vec<NodeHandle> = match args.get("elements") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_array(elements);
                Ok(RuntimeValue::Node(handle))
            }

            "struct_literal" => {
                let name = match args.get("type_name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => {
                        debug!("[ERROR] struct_literal missing type_name, args: {:?}", args.keys());
                        String::new()
                    }
                };
                // Fields should be struct_literal_field nodes with field name and expression
                let fields: Vec<(String, NodeHandle)> = match args.get("fields") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => {
                                    // Each field is a struct_literal_field with field_name and expr
                                    // For now we just pass the handle, field name extraction happens in host
                                    self.host.get_struct_field_init(*h)
                                }
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_struct_literal(&name, fields);
                Ok(RuntimeValue::Node(handle))
            }

            // Struct instantiation: Point{ .x = 10, .y = 20 }
            "struct_init" => {
                let type_name = match args.get("type_name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousStruct".to_string(),
                };
                // Fields come as a list of struct_field_init results
                // Each struct_field_init stores (name, value) in host's struct_init_fields map
                let fields: Vec<(String, NodeHandle)> = match args.get("fields") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => {
                                    // Look up the field init data from host
                                    self.host.get_struct_field_init(*h)
                                }
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_struct_literal(&type_name, fields);
                Ok(RuntimeValue::Node(handle))
            }

            // Struct field initialization: .x = 10
            "struct_field_init" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "field".to_string(),
                };
                let value = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("struct_field_init: missing value".into())),
                };
                // Store field init and return a handle for later lookup
                let handle = self.host.store_struct_field_init(&name, value);
                Ok(RuntimeValue::Node(handle))
            }

            "cast" => {
                let expr = match args.get("expr") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("cast: missing expr".into())),
                };
                let target_type = match args.get("target_type").or(args.get("type")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };
                let handle = self.host.create_cast(expr, target_type);
                Ok(RuntimeValue::Node(handle))
            }

            "lambda" => {
                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("lambda: missing body".into())),
                };
                let handle = self.host.create_lambda(params, body);
                Ok(RuntimeValue::Node(handle))
            }

            // ===== ADDITIONAL STATEMENTS =====

            node_type @ ("let_stmt" | "var_decl" | "var_stmt" | "const_stmt") => {
                log::trace!("[define_node] {}: args={:?}", node_type, args);
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let ty = match args.get("type") {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                // Handle init as either a Node handle or an immediate value
                let init = match args.get("init").or(args.get("value")) {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    Some(RuntimeValue::Int(n)) => {
                        // Create an int literal node for the immediate value
                        Some(self.host.create_int_literal(*n))
                    }
                    Some(RuntimeValue::Float(n)) => {
                        Some(self.host.create_float_literal(*n))
                    }
                    Some(RuntimeValue::String(s)) => {
                        Some(self.host.create_string_literal(s))
                    }
                    Some(RuntimeValue::Bool(b)) => {
                        Some(self.host.create_bool_literal(*b))
                    }
                    _ => None,
                };
                log::trace!("[define_node] {} name={}, ty={:?}, init={:?}", node_type, name, ty, init);
                // Determine is_const: true if node_type is const_stmt, or if is_const arg is true
                let is_const = node_type == "const_stmt" || match args.get("is_const").or(args.get("const")) {
                    Some(RuntimeValue::Bool(b)) => *b,
                    _ => false,
                };
                let handle = self.host.create_let(&name, ty, init, is_const);
                Ok(RuntimeValue::Node(handle))
            }

            "assignment" | "assign" => {
                log::trace!("[define_node] assignment args: {:?}", args);
                // Target can be either a Node (expression) or String (identifier)
                let target = match args.get("target") {
                    Some(RuntimeValue::Node(h)) => *h,
                    Some(RuntimeValue::String(name)) => {
                        // Create a variable reference for the target
                        self.host.create_variable(name)
                    }
                    _ => return Err(crate::error::ZynPegError::CodeGenError("assignment: missing target".into())),
                };
                let value = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("assignment: missing value".into())),
                };
                let handle = self.host.create_assignment(target, value);
                Ok(RuntimeValue::Node(handle))
            }

            "field_assignment" => {
                log::trace!("[define_node] field_assignment args: {:?}", args);
                // Field assignment: object.field = value
                let object_name = match args.get("object") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("field_assignment: missing object".into())),
                };
                let field_name = match args.get("field") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("field_assignment: missing field".into())),
                };
                let value = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("field_assignment: missing value".into())),
                };

                // Create field access expression as target
                let object = self.host.create_variable(&object_name);
                let field_access = self.host.create_field_access(object, &field_name);
                let handle = self.host.create_assignment(field_access, value);
                Ok(RuntimeValue::Node(handle))
            }

            "if_stmt" | "if" => {
                let condition = match args.get("condition") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("if: missing condition".into())),
                };
                let then_block = match args.get("then").or(args.get("then_block")).or(args.get("then_branch")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("if: missing then block".into())),
                };
                let else_block = match args.get("else").or(args.get("else_block")).or(args.get("else_branch")) {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                let handle = self.host.create_if(condition, then_block, else_block);
                Ok(RuntimeValue::Node(handle))
            }

            "while_stmt" | "while" => {
                log::trace!("[define_node] while args: {:?}", args);
                let condition = match args.get("condition") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("while: missing condition".into())),
                };
                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("while: missing body".into())),
                };
                let handle = self.host.create_while(condition, body);
                log::trace!("[define_node] while created handle={:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "for_stmt" | "for" => {
                log::trace!("[define_node] for args: {:?}", args);
                let variable = match args.get("variable").or(args.get("iterator")).or(args.get("binding")) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    Some(RuntimeValue::Node(h)) => {
                        // If it's a node, try to extract the name
                        format!("iter_{:?}", h)
                    }
                    _ => "it".to_string(),
                };
                let iterable = match args.get("iterable") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("for: missing iterable".into())),
                };
                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("for: missing body".into())),
                };
                let handle = self.host.create_for(&variable, iterable, body);
                Ok(RuntimeValue::Node(handle))
            }

            "break_stmt" | "break" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                let handle = self.host.create_break(value);
                Ok(RuntimeValue::Node(handle))
            }

            "continue_stmt" | "continue" => {
                let handle = self.host.create_continue();
                Ok(RuntimeValue::Node(handle))
            }

            "expression_stmt" | "expr_stmt" => {
                log::trace!("[define_node] expression_stmt args: {:?}", args);
                let expr = match args.get("expr") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("expression_stmt: missing expr".into())),
                };
                let handle = self.host.create_expression_stmt(expr);
                Ok(RuntimeValue::Node(handle))
            }

            // Range expression (for for-loop iterables): 0...5 or 0..5
            "range" | "range_expr" => {
                log::trace!("[define_node] range/range_expr args: {:?}", args);
                let start = args.get("start").and_then(|v| match v {
                    RuntimeValue::Node(h) => Some(*h),
                    _ => None,
                });
                let end = args.get("end").and_then(|v| match v {
                    RuntimeValue::Node(h) => Some(*h),
                    _ => None,
                });
                let inclusive = match args.get("inclusive") {
                    Some(RuntimeValue::Bool(b)) => *b,
                    _ => false, // Haxe 0...5 is exclusive (0 to 4)
                };
                let handle = self.host.create_range(start, end, inclusive);
                Ok(RuntimeValue::Node(handle))
            }

            // ===== TYPES =====

            "pointer_type" => {
                let pointee = match args.get("pointee") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };
                let handle = self.host.create_pointer_type(pointee);
                Ok(RuntimeValue::Node(handle))
            }

            "array_type" => {
                let element = match args.get("element") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };
                let size = match args.get("size") {
                    Some(RuntimeValue::Int(n)) => Some(*n as usize),
                    _ => None,
                };
                let handle = self.host.create_array_type(element, size);
                Ok(RuntimeValue::Node(handle))
            }

            "named_type" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Unknown".to_string(),
                };
                let handle = self.host.create_named_type(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "generic_type" => {
                // Generic type like List<T>, Option<Item>, etc.
                // Extract the base type name (e.g., "List" from "List<T>")
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Unknown".to_string(),
                };
                // For now, just create a named type with the base name
                // The type arguments are stored separately but not used yet
                let handle = self.host.create_named_type(&name);
                debug!("[DEBUG generic_type] Created generic type: name='{}', handle={:?}", name, handle);
                Ok(RuntimeValue::Node(handle))
            }

            "function_type" => {
                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let return_type = match args.get("return_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("void"),
                };
                let handle = self.host.create_function_type(params, return_type);
                Ok(RuntimeValue::Node(handle))
            }

            // ===== ZIG-SPECIFIC NODES =====

            "null_literal" => {
                // Create null literal - for now use an int with special marker
                // TypedASTBuilder may need a null_literal method added
                let handle = self.host.create_identifier("null");
                Ok(RuntimeValue::Node(handle))
            }

            "try_expr" | "try" => {
                // try expression unwraps error union
                let expr = match args.get("expr").or(args.get("operand")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("try: missing expr".into())),
                };
                // For now, represent as unary with special op
                let handle = self.host.create_unary_op("try", expr);
                Ok(RuntimeValue::Node(handle))
            }

            "await_expr" | "await" => {
                // await expression awaits a Promise/Future
                let expr = match args.get("expr").or(args.get("operand")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("await: missing expr".into())),
                };
                // Create await expression node
                let handle = self.host.create_await(expr);
                Ok(RuntimeValue::Node(handle))
            }

            "defer_stmt" | "defer" => {
                // defer statement - execute on scope exit
                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("defer: missing body".into())),
                };
                // Represent as expression statement for now
                let handle = self.host.create_expr_stmt(body);
                Ok(RuntimeValue::Node(handle))
            }

            "errdefer_stmt" | "errdefer" => {
                // errdefer - execute on scope exit if error
                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("errdefer: missing body".into())),
                };
                let handle = self.host.create_expr_stmt(body);
                Ok(RuntimeValue::Node(handle))
            }

            "optional_type" => {
                // ?T - optional type
                let inner = match args.get("inner").or(args.get("element")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };
                // Represent as pointer for now (optional is like nullable pointer)
                let handle = self.host.create_pointer_type(inner);
                Ok(RuntimeValue::Node(handle))
            }

            "error_union_type" => {
                // !T - error union type
                let ok_type = match args.get("ok_type").or(args.get("inner")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };
                // For now, just use the ok type (error handling comes later)
                Ok(RuntimeValue::Node(ok_type))
            }

            "struct_decl" => {
                // Struct type declaration
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousStruct".to_string(),
                };
                // Create as named type for now
                let handle = self.host.create_named_type(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "enum_decl" => {
                // Enum type declaration
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousEnum".to_string(),
                };
                let handle = self.host.create_named_type(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "union_decl" => {
                // Union type declaration
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousUnion".to_string(),
                };
                let handle = self.host.create_named_type(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "orelse_expr" | "orelse" => {
                // x orelse y - unwrap optional or use default
                let lhs = match args.get("lhs").or(args.get("left")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("orelse: missing lhs".into())),
                };
                let rhs = match args.get("rhs").or(args.get("right")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("orelse: missing rhs".into())),
                };
                // Represent as binary op with special operator
                let handle = self.host.create_binary_op("orelse", lhs, rhs);
                Ok(RuntimeValue::Node(handle))
            }

            "catch_expr" | "catch" => {
                // x catch y - unwrap error or use default/handler
                let lhs = match args.get("lhs").or(args.get("left")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("catch: missing lhs".into())),
                };
                let rhs = match args.get("rhs").or(args.get("right")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("catch: missing rhs".into())),
                };
                let handle = self.host.create_binary_op("catch", lhs, rhs);
                Ok(RuntimeValue::Node(handle))
            }

            // ===== PATTERN MATCHING =====

            "match_expr" | "switch_expr" | "switch" => {
                let scrutinee = match args.get("scrutinee").or(args.get("expr")).or(args.get("value")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("match_expr: missing scrutinee".into())),
                };
                let arms: Vec<NodeHandle> = match args.get("arms").or(args.get("cases")) {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_match_expr(scrutinee, arms);
                Ok(RuntimeValue::Node(handle))
            }

            "match_arm" | "switch_case" | "case" => {
                let pattern = match args.get("pattern").or(args.get("value")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_wildcard_pattern(),
                };
                let body = match args.get("body").or(args.get("result")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("match_arm: missing body".into())),
                };
                let handle = self.host.create_match_arm(pattern, body);
                Ok(RuntimeValue::Node(handle))
            }

            "literal_pattern" => {
                let value = match args.get("value") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_int_literal(0),
                };
                let handle = self.host.create_literal_pattern(value);
                Ok(RuntimeValue::Node(handle))
            }

            "wildcard_pattern" | "else_pattern" => {
                let handle = self.host.create_wildcard_pattern();
                Ok(RuntimeValue::Node(handle))
            }

            "identifier_pattern" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "x".to_string(),
                };
                let handle = self.host.create_identifier_pattern(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "struct_pattern" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Struct".to_string(),
                };
                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_struct_pattern(&name, fields);
                Ok(RuntimeValue::Node(handle))
            }

            "field_pattern" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "field".to_string(),
                };
                let pattern = match args.get("pattern") {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                let handle = self.host.create_field_pattern(&name, pattern);
                Ok(RuntimeValue::Node(handle))
            }

            "enum_pattern" | "variant_pattern" => {
                let name = match args.get("name").or(args.get("enum_name")) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Enum".to_string(),
                };
                let variant = match args.get("variant") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Variant".to_string(),
                };
                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_enum_pattern(&name, &variant, fields);
                Ok(RuntimeValue::Node(handle))
            }

            "array_pattern" => {
                let elements: Vec<NodeHandle> = match args.get("elements") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_array_pattern(elements);
                Ok(RuntimeValue::Node(handle))
            }

            "tuple_pattern" => {
                let elements: Vec<NodeHandle> = match args.get("elements") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_tuple_pattern(elements);
                Ok(RuntimeValue::Node(handle))
            }

            "range_pattern" => {
                // Create default literal patterns first to avoid double borrow
                let default_start = {
                    let int_lit = self.host.create_int_literal(0);
                    self.host.create_literal_pattern(int_lit)
                };
                let default_end = {
                    let int_lit = self.host.create_int_literal(0);
                    self.host.create_literal_pattern(int_lit)
                };

                let start = match args.get("start") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => default_start,
                };
                let end = match args.get("end") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => default_end,
                };
                let inclusive = match args.get("inclusive") {
                    Some(RuntimeValue::Bool(b)) => *b,
                    _ => true,
                };
                let handle = self.host.create_range_pattern(start, end, inclusive);
                Ok(RuntimeValue::Node(handle))
            }

            "or_pattern" => {
                let patterns: Vec<NodeHandle> = match args.get("patterns") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_or_pattern(patterns);
                Ok(RuntimeValue::Node(handle))
            }

            "pointer_pattern" | "reference_pattern" => {
                let inner = match args.get("inner").or(args.get("pattern")) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_wildcard_pattern(),
                };
                let mutable = match args.get("mutable") {
                    Some(RuntimeValue::Bool(b)) => *b,
                    _ => false,
                };
                let handle = self.host.create_pointer_pattern(inner, mutable);
                Ok(RuntimeValue::Node(handle))
            }

            "slice_pattern" => {
                let prefix: Vec<NodeHandle> = match args.get("prefix") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let middle = match args.get("middle").or(args.get("rest")) {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                let suffix: Vec<NodeHandle> = match args.get("suffix") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_slice_pattern(prefix, middle, suffix);
                Ok(RuntimeValue::Node(handle))
            }

            "error_pattern" => {
                let name = match args.get("name").or(args.get("error_name")) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Error".to_string(),
                };
                let handle = self.host.create_error_pattern(&name);
                Ok(RuntimeValue::Node(handle))
            }

            // ===== CLASS/OOP SUPPORT (Haxe) =====

            "class" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "AnonymousClass".to_string(),
                };
                let type_params: Vec<String> = match args.get("type_params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let members: Vec<NodeHandle> = match args.get("members") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let handle = self.host.create_class(&name, type_params, members);
                Ok(RuntimeValue::Node(handle))
            }

            "method" => {
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "method".to_string(),
                };
                let is_static = match args.get("is_static") {
                    Some(RuntimeValue::String(s)) => s == "static",
                    Some(RuntimeValue::Bool(b)) => *b,
                    _ => false,
                };
                let visibility = match args.get("visibility") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "public".to_string(),
                };
                let params: Vec<NodeHandle> = match args.get("params") {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };
                let return_type = match args.get("return_type") {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                let body = match args.get("body") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("method: missing body".into())),
                };
                let handle = self.host.create_method(&name, is_static, &visibility, params, return_type, body);
                Ok(RuntimeValue::Node(handle))
            }

            "package" => {
                // Package declarations are structural - just return a placeholder for now
                let path = match args.get("path") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                // For now, packages don't generate AST nodes - they're metadata
                // Return a dummy node that won't be included in declarations
                let handle = self.host.create_identifier(&path);
                Ok(RuntimeValue::Node(handle))
            }

            "import" => {
                // Import a module (e.g., "import prelude")
                let module_name = match args.get("module_name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    Some(RuntimeValue::Node(h)) => {
                        // If it's a node, it might be an identifier - get its text
                        // For now, just use "unknown" as fallback
                        "unknown".to_string()
                    }
                    _ => String::new(),
                };

                if module_name.is_empty() {
                    return Err(crate::error::ZynPegError::CodeGenError("import: missing module_name".into()));
                }

                let handle = self.host.create_import(&module_name);
                Ok(RuntimeValue::Node(handle))
            }

            "trait" => {
                // Trait declaration (e.g., "trait Display { ... }")
                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("trait: missing name".into())),
                };

                // TODO: Handle type_params and items (methods/associated types)
                let methods = vec![];

                let handle = self.host.create_trait(&name, methods);
                Ok(RuntimeValue::Node(handle))
            }

            "impl_abstract_inherent" => {
                debug!("[GRAMMAR impl_abstract_inherent] Creating inherent impl block");
                debug!("[GRAMMAR impl_abstract_inherent] All args: {:?}", args);

                let type_name = match args.get("type_name") {
                    Some(RuntimeValue::String(s)) => {
                        debug!("[GRAMMAR impl_abstract_inherent] type_name: {}", s);
                        s.clone()
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_abstract_inherent] ERROR: type_name is not a string, it's: {:?}", other);
                        return Err(crate::error::ZynPegError::CodeGenError("impl_abstract_inherent: type_name is not a string".into()));
                    },
                    None => {
                        debug!("[GRAMMAR impl_abstract_inherent] ERROR: missing type_name");
                        return Err(crate::error::ZynPegError::CodeGenError("impl_abstract_inherent: missing type_name".into()));
                    }
                };

                let underlying_type = match args.get("underlying_type") {
                    Some(RuntimeValue::Node(h)) => {
                        debug!("[GRAMMAR impl_abstract_inherent] underlying_type handle: {:?}", h);
                        *h
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_abstract_inherent] ERROR: underlying_type is not a node, it's: {:?}", other);
                        return Err(crate::error::ZynPegError::CodeGenError("impl_abstract_inherent: underlying_type is not a node".into()));
                    },
                    None => {
                        debug!("[GRAMMAR impl_abstract_inherent] ERROR: missing underlying_type");
                        return Err(crate::error::ZynPegError::CodeGenError("impl_abstract_inherent: missing underlying_type".into()));
                    }
                };

                let items: Vec<NodeHandle> = match args.get("items") {
                    Some(RuntimeValue::List(vals)) => {
                        debug!("[GRAMMAR impl_abstract_inherent] items list has {} values", vals.len());
                        vals.iter().filter_map(|v| {
                            if let RuntimeValue::Node(h) = v {
                                Some(*h)
                            } else {
                                None
                            }
                        }).collect()
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_abstract_inherent] items is not a list, it's: {:?}", other);
                        vec![]
                    },
                    None => {
                        debug!("[GRAMMAR impl_abstract_inherent] items arg is None");
                        vec![]
                    }
                };

                let handle = self.host.create_abstract_inherent_impl(&type_name, underlying_type, items);
                debug!("[GRAMMAR impl_abstract_inherent] Created inherent impl block with handle: {:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "impl_block" => {
                debug!("[GRAMMAR impl_block] Creating impl block");
                debug!("[GRAMMAR impl_block] All args: {:?}", args);
                // Trait implementation (e.g., "impl Add<Tensor> for Tensor { ... }")
                let trait_name = match args.get("trait_name") {
                    Some(RuntimeValue::String(s)) => {
                        debug!("[GRAMMAR impl_block] trait_name: {}", s);
                        s.clone()
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_block] ERROR: trait_name is not a string, it's: {:?}", other);
                        return Err(crate::error::ZynPegError::CodeGenError("impl_block: trait_name is not a string".into()));
                    },
                    None => {
                        debug!("[GRAMMAR impl_block] ERROR: missing trait_name");
                        return Err(crate::error::ZynPegError::CodeGenError("impl_block: missing trait_name".into()));
                    }
                };

                let for_type = match args.get("type_name") {
                    Some(RuntimeValue::String(s)) => {
                        debug!("[GRAMMAR impl_block] type_name string: {}", s);
                        s.clone()
                    },
                    Some(RuntimeValue::Node(h)) => {
                        // type_expr returns a Node, extract the type name from it
                        if let Some(name) = self.host.get_type_name(*h) {
                            debug!("[GRAMMAR impl_block] type_name from node: {}", name);
                            name
                        } else {
                            debug!("[GRAMMAR impl_block] ERROR: could not extract type name from node {:?}", h);
                            return Err(crate::error::ZynPegError::CodeGenError("impl_block: could not extract type name from node".into()));
                        }
                    },
                    Some(RuntimeValue::Null) => {
                        debug!("[GRAMMAR impl_block] ERROR: type_name is Null");
                        return Err(crate::error::ZynPegError::CodeGenError("impl_block: type_name is Null".into()));
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_block] ERROR: type_name is unexpected type: {:?}", other);
                        return Err(crate::error::ZynPegError::CodeGenError("impl_block: type_name is not a string or node".into()));
                    },
                    None => {
                        debug!("[GRAMMAR impl_block] ERROR: missing type_name");
                        return Err(crate::error::ZynPegError::CodeGenError("impl_block: missing type_name".into()));
                    }
                };

                // Extract trait type arguments (e.g., <Tensor> in Add<Tensor>)
                let trait_args: Vec<NodeHandle> = match args.get("trait_args") {
                    Some(RuntimeValue::String(s)) if s == "none" => {
                        debug!("[GRAMMAR impl_block] No trait type arguments");
                        vec![]
                    },
                    Some(RuntimeValue::List(vals)) => {
                        debug!("[GRAMMAR impl_block] trait_args list with {} items", vals.len());
                        vals.iter().filter_map(|v| {
                            if let RuntimeValue::Node(h) = v {
                                Some(*h)
                            } else {
                                None
                            }
                        }).collect()
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_block] WARNING: trait_args is unexpected type: {:?}", other);
                        vec![]
                    },
                    None => {
                        debug!("[GRAMMAR impl_block] No trait_args provided");
                        vec![]
                    }
                };

                // Extract impl items (methods and associated types)
                debug!("[DEBUG impl_block] items arg: {:?}", args.get("items"));
                let items: Vec<NodeHandle> = match args.get("items") {
                    Some(RuntimeValue::List(vals)) => {
                        debug!("[DEBUG impl_block] items list has {} values", vals.len());
                        vals.iter().filter_map(|v| {
                            debug!("[DEBUG impl_block] item value: {:?}", v);
                            if let RuntimeValue::Node(h) = v {
                                Some(*h)
                            } else {
                                None
                            }
                        }).collect()
                    },
                    Some(other) => {
                        debug!("[DEBUG impl_block] items is not a list, it's: {:?}", other);
                        vec![]
                    },
                    None => {
                        debug!("[DEBUG impl_block] items arg is None");
                        vec![]
                    }
                };

                let handle = self.host.create_impl_block(&trait_name, &for_type, trait_args, items);
                debug!("[GRAMMAR impl_block] Created impl block with handle: {:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "impl_inherent" => {
                debug!("[GRAMMAR impl_inherent] Creating inherent impl block");
                debug!("[GRAMMAR impl_inherent] All args: {:?}", args);

                // Extract the type name - can be a simple identifier or a generic type like List<T>
                let type_name = match args.get("type_name") {
                    Some(RuntimeValue::String(s)) => {
                        debug!("[GRAMMAR impl_inherent] type_name string: {}", s);
                        s.clone()
                    },
                    Some(RuntimeValue::List(vals)) => {
                        // For generic types like List<T>, extract the base name from the node
                        if let Some(RuntimeValue::Node(h)) = vals.first() {
                            // Get the type name from the node
                            if let Some(name) = self.host.get_type_name(*h) {
                                debug!("[GRAMMAR impl_inherent] type_name from node: {}", name);
                                name
                            } else {
                                debug!("[GRAMMAR impl_inherent] Could not get type name from node {:?}", h);
                                return Err(crate::error::ZynPegError::CodeGenError("impl_inherent: could not resolve type_name".into()));
                            }
                        } else {
                            debug!("[GRAMMAR impl_inherent] Empty list for type_name");
                            return Err(crate::error::ZynPegError::CodeGenError("impl_inherent: empty type_name list".into()));
                        }
                    },
                    Some(RuntimeValue::Node(h)) => {
                        // Direct node - get type name from it
                        if let Some(name) = self.host.get_type_name(*h) {
                            debug!("[GRAMMAR impl_inherent] type_name from direct node: {}", name);
                            name
                        } else {
                            debug!("[GRAMMAR impl_inherent] Could not get type name from direct node {:?}", h);
                            return Err(crate::error::ZynPegError::CodeGenError("impl_inherent: could not resolve type_name from node".into()));
                        }
                    },
                    Some(other) => {
                        debug!("[GRAMMAR impl_inherent] ERROR: type_name is unexpected type: {:?}", other);
                        return Err(crate::error::ZynPegError::CodeGenError("impl_inherent: type_name is not a string".into()));
                    },
                    None => {
                        debug!("[GRAMMAR impl_inherent] ERROR: missing type_name");
                        return Err(crate::error::ZynPegError::CodeGenError("impl_inherent: missing type_name".into()));
                    }
                };

                // Extract impl items (methods)
                debug!("[DEBUG impl_inherent] items arg: {:?}", args.get("items"));
                let items: Vec<NodeHandle> = match args.get("items") {
                    Some(RuntimeValue::List(vals)) => {
                        debug!("[DEBUG impl_inherent] items list has {} values", vals.len());
                        vals.iter().filter_map(|v| {
                            debug!("[DEBUG impl_inherent] item value: {:?}", v);
                            if let RuntimeValue::Node(h) = v {
                                Some(*h)
                            } else {
                                None
                            }
                        }).collect()
                    },
                    Some(other) => {
                        debug!("[DEBUG impl_inherent] items is not a list, it's: {:?}", other);
                        vec![]
                    },
                    None => {
                        debug!("[DEBUG impl_inherent] items arg is None");
                        vec![]
                    }
                };

                // Create inherent impl block - uses empty string for trait_name to indicate inherent impl
                let handle = self.host.create_impl_block("", &type_name, vec![], items);
                debug!("[GRAMMAR impl_inherent] Created inherent impl block with handle: {:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "opaque_type" => {
                // @opaque("$Tensor") type Tensor
                // Declares an opaque/extern type backed by external implementation

                // Get the full matched text to extract both external name and type name
                let text = match args.get("text") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => {
                        return Err(crate::error::ZynPegError::CodeGenError("opaque_type: missing text".into()));
                    }
                };

                // Parse text like: @opaque("$Tensor") type Tensor
                // Extract the string between quotes for external_name
                let external_name = if let Some(start) = text.find('"') {
                    if let Some(end) = text[start + 1..].find('"') {
                        text[start + 1..start + 1 + end].to_string()
                    } else {
                        return Err(crate::error::ZynPegError::CodeGenError("opaque_type: malformed external_name".into()));
                    }
                } else {
                    return Err(crate::error::ZynPegError::CodeGenError("opaque_type: missing external_name".into()));
                };

                // Extract the identifier after "type"
                let name = text.split_whitespace()
                    .last()
                    .unwrap_or("Unknown")
                    .to_string();

                // FIXME: The grammar only captures the string literal text, not the full match
                // For "@opaque("$Tensor") type Tensor", text is just "$Tensor"
                // Since we can't get the actual type name, use the external_name without "$" prefix
                let actual_name = external_name.trim_start_matches('$');
                let actual_external_name = external_name.clone();

                debug!("[DEBUG opaque_type] Using derived name='{}', external_name='{}'", actual_name, actual_external_name);
                log::debug!("[opaque_type] Creating opaque type: name='{}', external_name='{}'", actual_name, actual_external_name);
                let handle = self.host.create_opaque_type(actual_name, &actual_external_name);
                log::debug!("[opaque_type] Created opaque type with handle: {:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "struct_type" => {
                // struct Tensor:
                //     ptr: TensorPtr
                // Declares a struct type with named fields

                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => {
                        return Err(crate::error::ZynPegError::CodeGenError("struct_type: missing name".into()));
                    }
                };

                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(values)) => {
                        // Extract node handles from the list
                        values.iter().filter_map(|v| {
                            if let RuntimeValue::Node(handle) = v {
                                Some(*handle)
                            } else {
                                None
                            }
                        }).collect()
                    }
                    _ => {
                        return Err(crate::error::ZynPegError::CodeGenError("struct_type: missing fields".into()));
                    }
                };

                debug!("[DEBUG struct_type] Creating struct type: name='{}', {} fields", name, fields.len());
                log::debug!("[struct_type] Creating struct type: name='{}', {} fields", name, fields.len());
                let handle = self.host.create_struct_def(&name, fields);
                log::debug!("[struct_type] Created struct type with handle: {:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "abstract_type" => {
                // abstract Duration(i64): ms: i64
                // Declares an abstract type (zero-cost wrapper with implicit conversions)

                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => {
                        return Err(crate::error::ZynPegError::CodeGenError("abstract_type: missing name".into()));
                    }
                };

                let underlying_type = match args.get("underlying_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => {
                        return Err(crate::error::ZynPegError::CodeGenError("abstract_type: missing underlying_type".into()));
                    }
                };

                // Fields are optional
                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(values)) => {
                        values.iter().filter_map(|v| {
                            if let RuntimeValue::Node(handle) = v {
                                Some(*handle)
                            } else {
                                None
                            }
                        }).collect()
                    }
                    _ => vec![], // No fields
                };

                // Suffixes are optional (can be multiple)
                // Debug what we're receiving
                debug!("[DEBUG abstract_type] suffixes arg: {:?}", args.get("suffixes"));

                let suffixes = match args.get("suffixes") {
                    Some(RuntimeValue::List(values)) => {
                        debug!("[DEBUG abstract_type] Got suffixes list with {} items", values.len());
                        values.iter().filter_map(|v| {
                            debug!("[DEBUG abstract_type] Suffix item: {:?}", v);
                            if let RuntimeValue::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        }).collect()
                    }
                    _ => {
                        debug!("[DEBUG abstract_type] No suffixes list found");
                        vec![]
                    }
                };

                debug!("[DEBUG abstract_type] Creating abstract type: name='{}', {} fields, suffixes={:?}", name, fields.len(), suffixes);
                log::debug!("[abstract_type] Creating abstract type: name='{}', {} fields, suffixes={:?}", name, fields.len(), suffixes);
                let handle = self.host.create_abstract_def(&name, underlying_type, fields, suffixes);
                log::debug!("[abstract_type] Created abstract type with handle: {:?}", handle);
                Ok(RuntimeValue::Node(handle))
            }

            "abstract_with_single_suffix" => {
                debug!("[DEBUG abstract_with_single_suffix] Handler called with args: {:?}", args);

                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("abstract_with_single_suffix: missing name".into())),
                };

                let underlying_type = match args.get("underlying_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("abstract_with_single_suffix: missing underlying_type".into())),
                };

                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(values)) => {
                        values.iter().filter_map(|v| {
                            if let RuntimeValue::Node(handle) = v {
                                Some(*handle)
                            } else {
                                None
                            }
                        }).collect()
                    }
                    _ => vec![],
                };

                // Extract suffix from the string literal node
                let suffix_str = match args.get("suffix_literal") {
                    Some(RuntimeValue::Node(h)) => {
                        if let Some(expr) = self.host.get_expr(*h) {
                            if let TypedExpression::Literal(TypedLiteral::String(s)) = &expr.node {
                                s.resolve_global().unwrap_or_default()
                            } else {
                                return Err(crate::error::ZynPegError::CodeGenError(format!("Expected string literal for suffix, got {:?}", expr.node)));
                            }
                        } else {
                            return Err(crate::error::ZynPegError::CodeGenError("Failed to get string literal expression".into()));
                        }
                    }
                    _ => return Err(crate::error::ZynPegError::CodeGenError("abstract_with_single_suffix: missing or invalid suffix_literal".into())),
                };

                let suffix = suffix_str.trim_matches('"');
                let suffixes = vec![suffix.to_string()];

                let handle = self.host.create_abstract_def(&name, underlying_type, fields, suffixes);
                Ok(RuntimeValue::Node(handle))
            }

            "abstract_with_multiple_suffixes" => {
                debug!("[DEBUG abstract_with_multiple_suffixes] Handler called with args: {:?}", args);

                let name = match args.get("name") {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => return Err(crate::error::ZynPegError::CodeGenError("abstract_with_multiple_suffixes: missing name".into())),
                };

                let underlying_type = match args.get("underlying_type") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("abstract_with_multiple_suffixes: missing underlying_type".into())),
                };

                let fields: Vec<NodeHandle> = match args.get("fields") {
                    Some(RuntimeValue::List(values)) => {
                        values.iter().filter_map(|v| {
                            if let RuntimeValue::Node(handle) = v {
                                Some(*handle)
                            } else {
                                None
                            }
                        }).collect()
                    }
                    _ => vec![],
                };

                // Extract suffixes from the string literal node or direct string
                let suffixes_str = match args.get("suffixes_literal") {
                    Some(RuntimeValue::Node(h)) => {
                        if let Some(expr) = self.host.get_expr(*h) {
                            if let TypedExpression::Literal(TypedLiteral::String(s)) = &expr.node {
                                s.resolve_global().unwrap_or_default()
                            } else {
                                return Err(crate::error::ZynPegError::CodeGenError(format!("Expected string literal for suffixes, got {:?}", expr.node)));
                            }
                        } else {
                            return Err(crate::error::ZynPegError::CodeGenError("Failed to get string literal expression".into()));
                        }
                    }
                    Some(RuntimeValue::String(s)) => {
                        // Direct string (for abstract types without suffix clause)
                        s.clone()
                    }
                    _ => return Err(crate::error::ZynPegError::CodeGenError("abstract_with_multiple_suffixes: missing or invalid suffixes_literal".into())),
                };

                let suffixes_clean = suffixes_str.trim_matches('"');
                let suffixes: Vec<String> = suffixes_clean
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();

                let handle = self.host.create_abstract_def(&name, underlying_type, fields, suffixes);
                Ok(RuntimeValue::Node(handle))
            }

            "ternary" => {
                // Ternary conditional: condition ? then_expr : else_expr
                let condition = match args.get("condition") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("ternary: missing condition".into())),
                };
                let then_expr = match args.get("then_expr") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("ternary: missing then_expr".into())),
                };
                let else_expr = match args.get("else_expr") {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("ternary: missing else_expr".into())),
                };
                let handle = self.host.create_ternary(condition, then_expr, else_expr);
                Ok(RuntimeValue::Node(handle))
            }

            "range" => {
                // Range expression: start..end or start..=end
                let start = args.get("start")
                    .and_then(|v| if let RuntimeValue::Node(h) = v { Some(*h) } else { None });
                let end = args.get("end")
                    .and_then(|v| if let RuntimeValue::Node(h) = v { Some(*h) } else { None });
                let inclusive = args.get("inclusive")
                    .and_then(|v| if let RuntimeValue::Bool(b) = v { Some(*b) } else { None })
                    .unwrap_or(false);
                let handle = self.host.create_range(start, end, inclusive);
                Ok(RuntimeValue::Node(handle))
            }

            _ => {
                Err(crate::error::ZynPegError::CodeGenError(format!("Unknown node type: {}", node_type)))
            }
        }
    }

    /// Call a host function with resolved arguments (legacy positional format)
    fn call_host_function(&mut self, func: &str, args: Vec<RuntimeValue>) -> Result<RuntimeValue> {
        match func {
            "int_literal" | "integer" => {
                let value = match args.first() {
                    Some(RuntimeValue::Int(n)) => *n,
                    Some(RuntimeValue::String(s)) => s.parse().unwrap_or(0),
                    _ => 0,
                };
                let handle = self.host.create_int_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "float_literal" | "float" => {
                let value = match args.first() {
                    Some(RuntimeValue::Float(n)) => *n,
                    Some(RuntimeValue::String(s)) => s.parse().unwrap_or(0.0),
                    _ => 0.0,
                };
                let handle = self.host.create_float_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "string_literal" => {
                let value = match args.first() {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let handle = self.host.create_string_literal(&value);
                Ok(RuntimeValue::Node(handle))
            }

            "bool_literal" | "bool" => {
                let value = match args.first() {
                    Some(RuntimeValue::Bool(b)) => *b,
                    Some(RuntimeValue::String(s)) => s == "true",
                    _ => false,
                };
                let handle = self.host.create_bool_literal(value);
                Ok(RuntimeValue::Node(handle))
            }

            "identifier" => {
                let name = match args.first() {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let handle = self.host.create_identifier(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "binary_op" => {
                let op = match args.get(0) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Unknown".to_string(),
                };
                let left = match args.get(1) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("binary_op: missing left operand".into())),
                };
                let right = match args.get(2) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("binary_op: missing right operand".into())),
                };
                let handle = self.host.create_binary_op(&op, left, right);
                Ok(RuntimeValue::Node(handle))
            }

            "unary_op" => {
                let op = match args.get(0) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "Unknown".to_string(),
                };
                let operand = match args.get(1) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => return Err(crate::error::ZynPegError::CodeGenError("unary_op: missing operand".into())),
                };
                let handle = self.host.create_unary_op(&op, operand);
                Ok(RuntimeValue::Node(handle))
            }

            "return" => {
                let value = match args.first() {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    Some(RuntimeValue::Null) => None,
                    None => None,
                    _ => None,
                };
                let handle = self.host.create_return(value);
                Ok(RuntimeValue::Node(handle))
            }

            "block" => {
                let statements: Vec<NodeHandle> = args.iter()
                    .filter_map(|a| match a {
                        RuntimeValue::Node(h) => Some(*h),
                        RuntimeValue::List(items) => {
                            // Flatten list of nodes
                            None // TODO: Handle nested lists
                        }
                        _ => None,
                    })
                    .collect();
                let handle = self.host.create_block(statements);
                Ok(RuntimeValue::Node(handle))
            }

            "primitive_type" => {
                let name = match args.first() {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "unknown".to_string(),
                };
                let handle = self.host.create_primitive_type(&name);
                Ok(RuntimeValue::Node(handle))
            }

            "program" => {
                let handle = self.host.create_program();
                debug!("[GRAMMAR program] Adding {} declarations to program", args.len());
                // Add declarations from args
                for (idx, arg) in args.iter().enumerate() {
                    if let RuntimeValue::Node(decl) = arg {
                        debug!("[GRAMMAR program] Adding decl {} with handle: {:?}", idx, decl);
                        self.host.program_add_decl(handle, *decl);
                    } else {
                        debug!("[GRAMMAR program] Skipping non-node arg {} : {:?}", idx, arg);
                    }
                }
                Ok(RuntimeValue::Node(handle))
            }

            "return_stmt" => {
                // Create a return statement from an expression
                let value = match args.first() {
                    Some(RuntimeValue::Node(h)) => Some(*h),
                    _ => None,
                };
                let handle = self.host.create_return(value);
                Ok(RuntimeValue::Node(handle))
            }

            "function" => {
                // Create a function: function(name, params, body)
                // args: [name: string, params: list of nodes, body: node]
                let name = match args.get(0) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "anonymous".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get(1) {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let body = match args.get(2) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => {
                        // Create empty block as default body
                        self.host.create_block(vec![])
                    }
                };

                let return_type = self.host.create_primitive_type("i32");
                let handle = self.host.create_function(&name, params, return_type, body);
                Ok(RuntimeValue::Node(handle))
            }

            "extern_function" => {
                // Create an extern function declaration (no body)
                let name = match args.get(0) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    _ => "anonymous".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get(1) {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let return_type = match args.get(2) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };

                let handle = self.host.create_extern_function(&name, params, return_type);
                Ok(RuntimeValue::Node(handle))
            }

            "async_function" => {
                // Create an async function declaration
                // args: [name: string or node, params: list of nodes, body: node]
                let name = match args.get(0) {
                    Some(RuntimeValue::String(s)) => s.clone(),
                    Some(RuntimeValue::Node(h)) => {
                        // Try to extract the identifier name from the node
                        self.host.get_identifier_name(*h)
                            .unwrap_or_else(|| "anonymous_async".to_string())
                    },
                    _ => "anonymous_async".to_string(),
                };

                let params: Vec<NodeHandle> = match args.get(1) {
                    Some(RuntimeValue::List(list)) => {
                        list.iter()
                            .filter_map(|v| match v {
                                RuntimeValue::Node(h) => Some(*h),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                let body = match args.get(2) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_block(vec![]),
                };

                let return_type = match args.get(3) {
                    Some(RuntimeValue::Node(h)) => *h,
                    _ => self.host.create_primitive_type("i32"),
                };

                let handle = self.host.create_async_function(&name, params, return_type, body);
                Ok(RuntimeValue::Node(handle))
            }

            // Add more host functions as needed...

            _ => {
                Err(crate::error::ZynPegError::CodeGenError(format!("Unknown host function: {}", func)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zpeg_module_serialization() {
        let module = ZpegModule {
            metadata: ZpegMetadata {
                name: "Test".to_string(),
                version: "1.0".to_string(),
                file_extensions: vec![".test".to_string()],
                entry_point: None,
                zpeg_version: "0.1.0".to_string(),
                builtins: BuiltinMappings::default(),
                types: TypeDeclarations::default(),
            },
            pest_grammar: "program = { expr }".to_string(),
            rules: HashMap::from([(
                "expr".to_string(),
                RuleCommands {
                    return_type: Some("TypedExpression".to_string()),
                    commands: vec![
                        AstCommand::GetChild {
                            index: Some(0),
                            name: None,
                        },
                        AstCommand::Return,
                    ],
                },
            )]),
        };

        let json = serde_json::to_string_pretty(&module).unwrap();
        let parsed: ZpegModule = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.metadata.name, "Test");
        assert_eq!(parsed.rules.len(), 1);
    }

    #[test]
    fn test_command_serialization() {
        let cmd = AstCommand::Call {
            func: "binary_op".to_string(),
            args: vec![
                CommandArg::StringLit("Add".to_string()),
                CommandArg::ChildRef("$1".to_string()),
                CommandArg::ChildRef("$3".to_string()),
            ],
        };

        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("binary_op"));
        assert!(json.contains("$1"));
    }
}
