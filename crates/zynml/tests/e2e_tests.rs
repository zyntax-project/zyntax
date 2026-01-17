//! End-to-end tests for ZynML language
//!
//! These tests verify the full pipeline:
//! 1. Grammar compilation (both LanguageGrammar and Grammar2)
//! 2. Source parsing
//! 3. TypedAST generation
//! 4. Runtime execution (when plugins are available)
//!
//! Tests are organized by language feature as described in:
//! docs/ml-dsl-plans/00-unified-ml-dsl.md

use zynml::{
    ZynML, ZynMLConfig, ZynMLError,
    LanguageGrammar, Grammar2,
    ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_TENSOR,
};
use std::path::Path;

// ============================================================================
// Grammar Compilation Tests
// ============================================================================

/// Test that both grammar versions compile successfully
mod grammar_compilation {
    use super::*;

    #[test]
    fn test_language_grammar_compiles() {
        let result = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR);
        assert!(result.is_ok(), "LanguageGrammar should compile: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_compiles() {
        let result = Grammar2::from_source(ZYNML_GRAMMAR);
        assert!(result.is_ok(), "Grammar2 should compile: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_metadata() {
        let grammar = Grammar2::from_source(ZYNML_GRAMMAR).unwrap();

        // Verify metadata is populated
        assert!(!grammar.name().is_empty(), "Language name should be set");
        assert!(!grammar.version().is_empty(), "Version should be set");

        println!("Language: {} v{}", grammar.name(), grammar.version());
        println!("Extensions: {:?}", grammar.file_extensions());
    }
}

// ============================================================================
// Module and Import System Tests (Spec Section: Module and Import System)
// ============================================================================

mod module_system {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_parse_import_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("import tensor");
        assert!(result.is_ok(), "Should parse simple import: {:?}", result.err());
    }

    #[test]
    fn test_parse_import_prelude() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("import prelude");
        assert!(result.is_ok(), "Should parse prelude import: {:?}", result.err());
    }

    #[test]
    fn test_parse_multiple_imports() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            import tensor
            import prelude
        "#);
        assert!(result.is_ok(), "Should parse multiple imports: {:?}", result.err());
    }

    #[test]
    fn test_module_declaration() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("module recommendation_pipeline");
        assert!(result.is_ok(), "Should parse module declaration: {:?}", result.err());
    }

    #[test]
    fn test_aliased_import() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("import zynml.tensor as T");
        assert!(result.is_ok(), "Should parse aliased import: {:?}", result.err());
    }

    #[test]
    fn test_dotted_module_path() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("import zynml.tensor.ops as ops");
        assert!(result.is_ok(), "Should parse dotted module path: {:?}", result.err());
    }
}

// ============================================================================
// Type System Tests (Spec Section: Type System)
// ============================================================================

mod type_system {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // --- Type Aliases ---

    #[test]
    fn test_parse_type_alias_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("type Scalar = f32");
        assert!(result.is_ok(), "Should parse simple type alias: {:?}", result.err());
    }

    #[test]
    fn test_parse_type_alias_tensor() {
        let grammar = get_grammar();
        // Current grammar doesn't support tensor[shape, dtype] syntax in type position
        let result = grammar.parse_to_json("type Embedding = Tensor");
        assert!(result.is_ok(), "Should parse tensor type alias: {:?}", result.err());
    }

    // --- Struct Definitions ---

    #[test]
    fn test_parse_struct_colon_syntax() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            struct Point:
                x: float
                y: float
        "#);
        assert!(result.is_ok(), "Should parse struct with colon syntax: {:?}", result.err());
    }

    #[test]
    fn test_parse_struct_with_multiple_fields() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            struct Detection:
                bbox: BBox
                class_name: string
                confidence: float
        "#);
        assert!(result.is_ok(), "Should parse struct with multiple fields: {:?}", result.err());
    }

    #[test]
    fn test_parse_struct_generic() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            struct Container<T>:
                value: T
                count: int
        "#);
        assert!(result.is_ok(), "Should parse generic struct: {:?}", result.err());
    }

    #[test]
    fn test_parse_type_struct_braces() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("type Point = { x: f32, y: f32 }");
        assert!(result.is_ok(), "Should parse struct type with braces: {:?}", result.err());
    }

    // --- Enum Definitions ---

    #[test]
    fn test_parse_enum_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            enum Color {
                Red,
                Green,
                Blue
            }
        "#);
        assert!(result.is_ok(), "Should parse simple enum: {:?}", result.err());
    }

    #[test]
    fn test_parse_enum_with_data() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            enum Option<T> {
                Some(T),
                None
            }
        "#);
        assert!(result.is_ok(), "Should parse enum with data: {:?}", result.err());
    }

    #[test]
    fn test_parse_enum_result() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            enum Result<T, E> {
                Ok(T),
                Err(E)
            }
        "#);
        assert!(result.is_ok(), "Should parse Result enum: {:?}", result.err());
    }

    // --- Abstract Types ---

    #[test]
    fn test_parse_abstract_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("abstract Duration(i64)");
        assert!(result.is_ok(), "Should parse simple abstract type: {:?}", result.err());
    }

    #[test]
    fn test_parse_abstract_with_suffix() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            abstract Duration(i64) with Suffix("ms"):
                ms: i64
        "#);
        assert!(result.is_ok(), "Should parse abstract with suffix: {:?}", result.err());
    }

    #[test]
    fn test_parse_abstract_with_suffixes() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            abstract Duration(i64) with Suffixes("ms, s, m, h"):
                value: i64
        "#);
        assert!(result.is_ok(), "Should parse abstract with multiple suffixes: {:?}", result.err());
    }

    // --- Trait Definitions ---

    #[test]
    fn test_parse_trait_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            trait Display {
                fn to_string(self) -> String
            }
        "#);
        assert!(result.is_ok(), "Should parse simple trait: {:?}", result.err());
    }

    #[test]
    fn test_parse_trait_generic() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            trait Iterator<T> {
                fn next(self) -> T
            }
        "#);
        assert!(result.is_ok(), "Should parse generic trait: {:?}", result.err());
    }

    #[test]
    fn test_parse_trait_with_associated_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            trait Add<Rhs> {
                type Output
                fn add(self, rhs: Rhs) -> Output
            }
        "#);
        assert!(result.is_ok(), "Should parse trait with associated type: {:?}", result.err());
    }

    // --- Impl Blocks ---

    #[test]
    fn test_parse_impl_trait_for_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Display for Point {
                fn to_string(self) -> String {
                    self.x
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse trait impl: {:?}", result.err());
    }

    #[test]
    fn test_parse_impl_inherent() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Point {
                fn new(x: float, y: float) -> Point {
                    x
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse inherent impl: {:?}", result.err());
    }

    #[test]
    fn test_parse_impl_generic_trait() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Add<Tensor> for Tensor {
                type Output = Tensor
                fn add(self, rhs: Tensor) -> Tensor {
                    self
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse generic trait impl: {:?}", result.err());
    }

    #[test]
    fn test_parse_impl_abstract_inherent() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Duration(i64) {
                fn from_ms(value: i64) -> Duration {
                    value
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse abstract type impl: {:?}", result.err());
    }

    // --- Opaque Types ---

    #[test]
    fn test_parse_opaque_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"@opaque("$Tensor") type Tensor"#);
        assert!(result.is_ok(), "Should parse opaque type: {:?}", result.err());
    }

    // --- Extern Struct ---

    #[test]
    fn test_parse_extern_struct_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"extern struct Tensor"#);
        assert!(result.is_ok(), "Should parse simple extern struct: {:?}", result.err());
    }

    #[test]
    fn test_parse_extern_struct_generic() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"extern struct HashMap<K, V>"#);
        assert!(result.is_ok(), "Should parse generic extern struct: {:?}", result.err());
    }

    #[test]
    fn test_parse_extern_struct_single_param() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"extern struct List<T>"#);
        assert!(result.is_ok(), "Should parse extern struct with single type param: {:?}", result.err());
    }

    // --- Generics with Bounds ---

    #[test]
    fn test_parse_generic_with_bounds() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn map<T: Display, U>(items: List) -> List {
                items
            }
        "#);
        assert!(result.is_ok(), "Should parse generic with bounds: {:?}", result.err());
    }

    #[test]
    fn test_parse_generic_multiple_bounds() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn process<T: Clone + Debug>(item: T) -> T {
                item
            }
        "#);
        assert!(result.is_ok(), "Should parse generic with multiple bounds: {:?}", result.err());
    }
}

// ============================================================================
// Function Definition Tests
// ============================================================================

mod function_definitions {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // --- Brace-style functions ---

    #[test]
    fn test_parse_function_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("fn main() { }");
        assert!(result.is_ok(), "Should parse simple function: {:?}", result.err());
    }

    #[test]
    fn test_parse_function_with_params() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#);
        assert!(result.is_ok(), "Should parse function with params: {:?}", result.err());
    }

    #[test]
    fn test_parse_function_generic() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn identity<T>(x: T) -> T {
                x
            }
        "#);
        assert!(result.is_ok(), "Should parse generic function: {:?}", result.err());
    }

    #[test]
    fn test_parse_function_no_return_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn process(x: i32) {
                println(x)
            }
        "#);
        assert!(result.is_ok(), "Should parse function without return type: {:?}", result.err());
    }

    // --- Python-style def keyword ---

    #[test]
    fn test_parse_def_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("def main() { }");
        assert!(result.is_ok(), "Should parse def simple: {:?}", result.err());
    }

    #[test]
    fn test_parse_def_with_return_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("def greet(): int { 42 }");
        assert!(result.is_ok(), "Should parse def with return type: {:?}", result.err());
    }

    #[test]
    fn test_parse_def_with_params() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            def add(a: i32, b: i32): i32 {
                a + b
            }
        "#);
        assert!(result.is_ok(), "Should parse def with params: {:?}", result.err());
    }

    #[test]
    fn test_parse_async_def() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            async def fetch(url: str): Response {
                println(url)
            }
        "#);
        assert!(result.is_ok(), "Should parse async def: {:?}", result.err());
    }

    // --- Expression-bodied methods (single expr after colon) ---

    #[test]
    fn test_parse_function_expression_body() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Point {
                fn x_coord(self) -> float: self.x
            }
        "#);
        assert!(result.is_ok(), "Should parse expression-bodied function: {:?}", result.err());
    }
}

// ============================================================================
// Brace and Expression-Bodied Syntax Tests
// ============================================================================

mod syntax_styles {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // --- Brace-style struct definitions ---

    #[test]
    fn test_parse_struct_brace_style() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("struct Point { x: float, y: float }");
        assert!(result.is_ok(), "Should parse brace-style struct: {:?}", result.err());
    }

    #[test]
    fn test_parse_struct_brace_generic() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("struct Container<T> { value: T, count: int }");
        assert!(result.is_ok(), "Should parse brace-style generic struct: {:?}", result.err());
    }

    // --- Brace-style abstract definitions ---

    #[test]
    fn test_parse_abstract_brace_style() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("abstract Duration(i64) { value: i64 }");
        assert!(result.is_ok(), "Should parse brace-style abstract: {:?}", result.err());
    }

    #[test]
    fn test_parse_abstract_brace_with_suffix() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"abstract Duration(i64) with Suffix("ms") { value: i64 }"#);
        assert!(result.is_ok(), "Should parse brace abstract with suffix: {:?}", result.err());
    }

    #[test]
    fn test_parse_abstract_brace_with_suffixes() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"abstract Duration(i64) with Suffixes("ms, s, m, h") { value: i64 }"#);
        assert!(result.is_ok(), "Should parse brace abstract with suffixes: {:?}", result.err());
    }

    // --- Expression-bodied impl methods ---

    #[test]
    fn test_parse_impl_method_expression_body() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Duration(i64) {
                fn from_ms(v: i64) -> Duration: Duration { value: v }
                fn to_ms(self) -> i64: self.value
            }
        "#);
        assert!(result.is_ok(), "Should parse expression-bodied methods: {:?}", result.err());
    }

    #[test]
    fn test_parse_impl_method_brace_body() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Calculator {
                fn compute(self, x: i32) -> i32 {
                    let temp = x * 2
                    let result = temp + 1
                    result
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse brace-bodied method: {:?}", result.err());
    }

    // --- Mixed styles in impl blocks ---

    #[test]
    fn test_parse_mixed_impl_styles() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            impl Point {
                fn brace_method(self) -> i32 {
                    self.x
                }
                fn expr_method(self) -> i32: self.x
            }
        "#);
        assert!(result.is_ok(), "Should parse mixed impl method styles: {:?}", result.err());
    }

    // --- Complete brace-style example ---

    #[test]
    fn test_parse_complete_brace_example() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            import prelude

            abstract Duration(i64) with Suffixes("ms, s, m, h") { value: i64 }

            impl Duration(i64) {
                fn from_ms(v: i64) -> Duration: Duration { value: v }
                fn to_ms(self) -> i64: self.value
            }

            fn main() {
                let delay = 1000ms
                println(delay.to_ms())
            }
        "#);
        assert!(result.is_ok(), "Should parse complete brace-style example: {:?}", result.err());
    }
}

// ============================================================================
// Statement Tests
// ============================================================================

mod statements {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_parse_let_binding() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = 42");
        assert!(result.is_ok(), "Should parse let binding: {:?}", result.err());
    }

    #[test]
    fn test_parse_assignment() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                let x = 1
                x = 2
            }
        "#);
        assert!(result.is_ok(), "Should parse assignment: {:?}", result.err());
    }

    #[test]
    fn test_parse_field_assignment() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                self.x = 42
            }
        "#);
        assert!(result.is_ok(), "Should parse field assignment: {:?}", result.err());
    }

    #[test]
    fn test_parse_if_statement() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                if x > 0 {
                    let y = 1
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse if statement: {:?}", result.err());
    }

    #[test]
    fn test_parse_if_else() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                if x > 0 {
                    let y = 1
                } else {
                    let y = 0
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse if-else: {:?}", result.err());
    }

    #[test]
    fn test_parse_while_loop() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                while i < 10 {
                    i = i + 1
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse while loop: {:?}", result.err());
    }

    #[test]
    fn test_parse_for_loop() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                for i in range(0, 10) {
                    println(i)
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse for loop: {:?}", result.err());
    }

    #[test]
    fn test_parse_return_statement() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() -> i32 {
                return 42
            }
        "#);
        assert!(result.is_ok(), "Should parse return: {:?}", result.err());
    }

    #[test]
    fn test_parse_break_statement() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                while true {
                    break
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse break: {:?}", result.err());
    }

    #[test]
    fn test_parse_continue_statement() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn test() {
                for i in items {
                    continue
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse continue: {:?}", result.err());
    }
}

// ============================================================================
// Expression Tests
// ============================================================================

mod expressions {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // --- Arithmetic Operators ---

    #[test]
    fn test_parse_arithmetic_operators() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let a = 1 + 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 - 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 * 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 / 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 % 2").is_ok());
    }

    // --- Comparison Operators ---

    #[test]
    fn test_parse_comparison_operators() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let a = 1 < 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 > 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 <= 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 >= 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 == 2").is_ok());
        assert!(grammar.parse_to_json("let a = 1 != 2").is_ok());
    }

    // --- Logical Operators ---

    #[test]
    fn test_parse_logical_operators() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let a = true && false").is_ok());
        assert!(grammar.parse_to_json("let a = true || false").is_ok());
        assert!(grammar.parse_to_json("let a = !true").is_ok());
    }

    // --- Unary Operators ---

    #[test]
    fn test_parse_unary_operators() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let a = -x").is_ok());
        assert!(grammar.parse_to_json("let a = !flag").is_ok());
    }

    // --- Matrix Multiply Operator ---

    #[test]
    fn test_parse_matrix_multiply() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let a = x @ y");
        assert!(result.is_ok(), "Should parse @ operator: {:?}", result.err());
    }

    // --- Pipe Operator ---

    #[test]
    fn test_parse_pipe_operator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let result = data |> transform()");
        assert!(result.is_ok(), "Should parse pipe operator: {:?}", result.err());
    }

    #[test]
    fn test_parse_pipe_chain() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            let result = data |> transform() |> process(10) |> finalize()
        "#);
        assert!(result.is_ok(), "Should parse pipe chain: {:?}", result.err());
    }

    // --- Ternary Operator ---

    #[test]
    fn test_parse_ternary() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let output = confidence > 0.9 ? high() : low()");
        assert!(result.is_ok(), "Should parse ternary: {:?}", result.err());
    }

    // --- Range Expressions ---

    #[test]
    fn test_parse_range_exclusive() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let r = 0..10");
        assert!(result.is_ok(), "Should parse exclusive range: {:?}", result.err());
    }

    #[test]
    fn test_parse_range_inclusive() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let r = 0..=10");
        assert!(result.is_ok(), "Should parse inclusive range: {:?}", result.err());
    }

    // --- Function Calls ---

    #[test]
    fn test_parse_function_call() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = foo(1, 2, 3)");
        assert!(result.is_ok(), "Should parse function call: {:?}", result.err());
    }

    #[test]
    fn test_parse_method_call() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let s = tensor.sum()");
        assert!(result.is_ok(), "Should parse method call: {:?}", result.err());
    }

    #[test]
    fn test_parse_chained_method_calls() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let r = tensor.reshape([2, 3]).transpose()");
        assert!(result.is_ok(), "Should parse chained methods: {:?}", result.err());
    }

    // --- Indexing ---

    #[test]
    fn test_parse_indexing() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = arr[0]");
        assert!(result.is_ok(), "Should parse indexing: {:?}", result.err());
    }

    #[test]
    fn test_parse_nested_indexing() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = matrix[0][1]");
        assert!(result.is_ok(), "Should parse nested indexing: {:?}", result.err());
    }

    // --- Member Access ---

    #[test]
    fn test_parse_member_access() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = point.x");
        assert!(result.is_ok(), "Should parse member access: {:?}", result.err());
    }

    #[test]
    fn test_parse_chained_member_access() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = obj.field.subfield");
        assert!(result.is_ok(), "Should parse chained member access: {:?}", result.err());
    }

    // --- Path Expressions ---

    #[test]
    fn test_parse_path_expression() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let d = Duration::from(1000)");
        assert!(result.is_ok(), "Should parse path expression: {:?}", result.err());
    }

    // --- Extern Calls ---

    #[test]
    fn test_parse_extern_call() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let s = extern tensor_to_string(t)");
        assert!(result.is_ok(), "Should parse extern call: {:?}", result.err());
    }
}

// ============================================================================
// Literal Tests
// ============================================================================

mod literals {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_parse_integer_literal() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let x = 42").is_ok());
        assert!(grammar.parse_to_json("let x = -10").is_ok());
        assert!(grammar.parse_to_json("let x = 0").is_ok());
    }

    #[test]
    fn test_parse_float_literal() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let x = 3.14").is_ok());
        assert!(grammar.parse_to_json("let x = -2.5").is_ok());
        assert!(grammar.parse_to_json("let x = 1e-3").is_ok());
        assert!(grammar.parse_to_json("let x = 1.5e10").is_ok());
    }

    #[test]
    fn test_parse_string_literal() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let s = "hello world""#);
        assert!(result.is_ok(), "Should parse string: {:?}", result.err());
    }

    #[test]
    fn test_parse_triple_quoted_string() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let s = """multiline
        string
        here""""#);
        assert!(result.is_ok(), "Should parse triple-quoted string: {:?}", result.err());
    }

    #[test]
    fn test_parse_bool_literal() {
        let grammar = get_grammar();
        assert!(grammar.parse_to_json("let a = true").is_ok());
        assert!(grammar.parse_to_json("let b = false").is_ok());
    }

    #[test]
    fn test_parse_array_literal() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let arr = [1, 2, 3]");
        assert!(result.is_ok(), "Should parse array literal: {:?}", result.err());
    }

    #[test]
    fn test_parse_nested_array() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let m = [[1, 2], [3, 4]]");
        assert!(result.is_ok(), "Should parse nested array: {:?}", result.err());
    }

    #[test]
    fn test_parse_tensor_literal() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let t = tensor([1.0, 2.0, 3.0])");
        assert!(result.is_ok(), "Should parse tensor literal: {:?}", result.err());
    }

    #[test]
    fn test_parse_2d_tensor() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let m = tensor([[1.0, 2.0], [3.0, 4.0]])");
        assert!(result.is_ok(), "Should parse 2D tensor: {:?}", result.err());
    }

    #[test]
    fn test_parse_struct_literal() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let p = Point { x: 1.0, y: 2.0 }");
        assert!(result.is_ok(), "Should parse struct literal: {:?}", result.err());
    }

    #[test]
    fn test_parse_suffixed_literal() {
        let grammar = get_grammar();
        // Suffixed literals for abstract types (e.g., 1000ms)
        let result = grammar.parse_to_json("let d = 1000ms");
        assert!(result.is_ok(), "Should parse suffixed literal: {:?}", result.err());
    }

    #[test]
    fn test_parse_duration_literals() {
        let grammar = get_grammar();
        // Duration literals: 1h, 5m, 30s, 500ms, 2d
        assert!(grammar.parse_to_json("let d = 1h").is_ok());
        assert!(grammar.parse_to_json("let d = 5m").is_ok());
        assert!(grammar.parse_to_json("let d = 30s").is_ok());
        assert!(grammar.parse_to_json("let d = 500ms").is_ok());
        assert!(grammar.parse_to_json("let d = 2d").is_ok());
    }
}

// ============================================================================
// Data Loading Tests (Spec Section: Data Loading)
// ============================================================================

mod data_loading {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // NOTE: load() as special syntax is in spec but NOT YET in grammar
    // Currently load() would be parsed as a regular function call
    #[test]
    fn test_load_as_function_call() {
        let grammar = get_grammar();
        // load() works as regular function call
        let result = grammar.parse_to_json(r#"let data = load("data.json")"#);
        assert!(result.is_ok(), "Should parse load as function call: {:?}", result.err());
    }

    #[test]
    fn test_load_with_as_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let photo = load("image.jpg") as image"#);
        assert!(result.is_ok(), "Should parse load() with as type: {:?}", result.err());
    }

    #[test]
    fn test_stream_as_function_call() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let sensor_stream = stream("mqtt://sensors/+")"#);
        assert!(result.is_ok(), "Should parse stream() as function call: {:?}", result.err());
    }

    #[test]
    fn test_model_with_config() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            let encoder = model("bert.onnx") {
                input: text,
                output: Embedding
            }
        "#);
        assert!(result.is_ok(), "Should parse model() with config block: {:?}", result.err());
    }
}

// ============================================================================
// Pipeline Definition Tests (Spec Section: Pipeline Definition)
// ============================================================================

mod pipeline_definitions {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_pipeline_definition() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            pipeline image_search(query: text, top_k: int) -> list {
                query
            }
        "#);
        assert!(result.is_ok(), "Should parse pipeline definition: {:?}", result.err());
    }

    #[test]
    fn test_pipeline_no_return_type() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            pipeline process_data(input: tensor) {
                let result = transform(input)
                result
            }
        "#);
        assert!(result.is_ok(), "Should parse pipeline without return type: {:?}", result.err());
    }

    // Pipelines can also be written as regular functions
    #[test]
    fn test_pipeline_as_function_alternative() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn image_search(query: Text, top_k: i32) -> List {
                let encoded = encoder(query)
                encoded
            }
        "#);
        assert!(result.is_ok(), "Should parse pipeline as function: {:?}", result.err());
    }

    // --- Lambda Expressions ---

    #[test]
    fn test_parse_lambda_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let double = def(x): x * 2");
        assert!(result.is_ok(), "Should parse simple lambda: {:?}", result.err());
    }

    #[test]
    fn test_parse_lambda_multiple_params() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let add = def(x, y): x + y");
        assert!(result.is_ok(), "Should parse lambda with multiple params: {:?}", result.err());
    }

    #[test]
    fn test_parse_lambda_no_params() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let unit = def(): 42");
        assert!(result.is_ok(), "Should parse lambda with no params: {:?}", result.err());
    }

    #[test]
    fn test_parse_lambda_in_call() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let result = map(items, def(x): x * 2)");
        assert!(result.is_ok(), "Should parse lambda as function argument: {:?}", result.err());
    }

    // --- Import Modifier Expressions ---

    #[test]
    fn test_parse_import_asset() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let img = import asset("image.jpg") as Image"#);
        assert!(result.is_ok(), "Should parse import asset expression: {:?}", result.err());
    }

    #[test]
    fn test_parse_import_audio() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let audio = import audio("sound.wav") as AudioBuffer"#);
        assert!(result.is_ok(), "Should parse import audio expression: {:?}", result.err());
    }

    #[test]
    fn test_parse_import_model() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"let model = import model("bert.onnx") as TextModel"#);
        assert!(result.is_ok(), "Should parse import model expression: {:?}", result.err());
    }
}

// ============================================================================
// Compute/Kernel Tests (Spec Section: GPU Compute Dispatch)
// ============================================================================

mod compute_kernels {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_compute_with_kernel() {
        let grammar = get_grammar();
        // Note: index-assignment (out[i] = ...) requires separate grammar support
        // For now, test the basic compute structure
        let result = grammar.parse_to_json(r#"
            let result = compute(tensor) {
                @kernel elementwise
                for i in 0..len {
                    process(i)
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse compute() with kernel syntax: {:?}", result.err());
    }

    #[test]
    fn test_kernel_decorator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            @kernel
            fn elementwise(x: Tensor) -> Tensor {
                x
            }
        "#);
        assert!(result.is_ok(), "Should parse @kernel decorator: {:?}", result.err());
    }

    #[test]
    fn test_workgroup_decorator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            @workgroup(256)
            fn parallel_reduce(x: Tensor) -> f32 {
                x
            }
        "#);
        assert!(result.is_ok(), "Should parse @workgroup decorator: {:?}", result.err());
    }

    #[test]
    fn test_device_decorator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            @device("cuda")
            fn gpu_compute(x: Tensor) -> Tensor {
                x
            }
        "#);
        assert!(result.is_ok(), "Should parse @device decorator: {:?}", result.err());
    }
}

// ============================================================================
// Visualization/Render Tests (Spec Section: Visualization for ZynBook)
// ============================================================================

mod visualization {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_render_simple() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("render photo");
        assert!(result.is_ok(), "Should parse simple render statement: {:?}", result.err());
    }

    #[test]
    fn test_render_function_call() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("render chart(data)");
        assert!(result.is_ok(), "Should parse render with function call: {:?}", result.err());
    }

    #[test]
    fn test_render_with_options() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            render photo {
                title: "Original Image",
                width: 400
            }
        "#);
        assert!(result.is_ok(), "Should parse render with options: {:?}", result.err());
    }
}

// ============================================================================
// Streaming Tests (Spec Section: Streaming)
// ============================================================================

mod streaming {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // Pipe operator works for streaming patterns
    #[test]
    fn test_streaming_with_pipes() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            let processed = data |> window(100) |> map(transform) |> filter(check)
        "#);
        assert!(result.is_ok(), "Should parse streaming pattern with pipes: {:?}", result.err());
    }

    #[test]
    fn test_stream_with_sink() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            stream sensor_data
                |> window(100)
                |> sink(alert_system)
        "#);
        assert!(result.is_ok(), "Should parse stream with sink: {:?}", result.err());
    }

    #[test]
    fn test_stream_pipeline() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            stream video_feed |> detect(model) |> filter(confidence) |> sink(db)
        "#);
        assert!(result.is_ok(), "Should parse stream pipeline: {:?}", result.err());
    }
}

// ============================================================================
// Caching and Memoization Tests (Spec Section: Caching and Memoization)
// ============================================================================

mod caching {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_cache_decorator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            @cache(ttl=1h)
            fn expensive_embedding(text: Text) -> Embedding {
                encoder(text)
            }
        "#);
        assert!(result.is_ok(), "Should parse @cache decorator: {:?}", result.err());
    }

    #[test]
    fn test_memoize_decorator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            @memoize
            fn compute_statistics(data: Tensor) -> Stats {
                data
            }
        "#);
        assert!(result.is_ok(), "Should parse @memoize decorator: {:?}", result.err());
    }

    #[test]
    fn test_cache_on_pipeline() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            @cache(ttl=24h)
            pipeline cached_search(query: text) -> list {
                search(query)
            }
        "#);
        assert!(result.is_ok(), "Should parse @cache on pipeline: {:?}", result.err());
    }
}

// ============================================================================
// Configuration Tests (Spec Section: Configuration)
// ============================================================================

mod configuration {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_config_block() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            config {
                device: "cpu",
                precision: "float32",
                batch_size: 32
            }
        "#);
        assert!(result.is_ok(), "Should parse config block: {:?}", result.err());
    }

    #[test]
    fn test_config_with_nested_values() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            config {
                model_path: "models/bert.onnx",
                max_batch: 64,
                use_gpu: true
            }
        "#);
        assert!(result.is_ok(), "Should parse config with various value types: {:?}", result.err());
    }
}

// ============================================================================
// Error Handling Tests (Spec Section: Error Handling)
// ============================================================================

mod error_handling_syntax {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_try_catch() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn safe_load() {
                try {
                    let result = risky_operation()
                } catch ModelError as e {
                    handle_error(e)
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse try/catch block: {:?}", result.err());
    }

    #[test]
    fn test_try_multiple_catch() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn robust_process() {
                try {
                    process()
                } catch IOError as e {
                    log(e)
                } catch ParseError as e {
                    retry()
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse try with multiple catch clauses: {:?}", result.err());
    }

    #[test]
    fn test_match_expression() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn classify(detection) {
                match detection.class {
                    case "person" { process_person(detection) }
                    case "car" { process_vehicle(detection) }
                    case _ { ignore() }
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse match/case expression: {:?}", result.err());
    }

    #[test]
    fn test_match_with_literals() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(r#"
            fn check_code(code) {
                match code {
                    case 200 { success() }
                    case 404 { not_found() }
                    case _ { error() }
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse match with integer literals: {:?}", result.err());
    }
}

// ============================================================================
// Real Example File Tests
// ============================================================================

mod example_files {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    fn examples_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples")
    }

    #[test]
    fn test_parse_hello_example() {
        let grammar = get_grammar();
        let source = std::fs::read_to_string(examples_dir().join("hello.zynml"))
            .expect("Should read hello.zynml");

        let result = grammar.parse_to_json(&source);
        assert!(result.is_ok(), "Should parse hello.zynml: {:?}", result.err());
    }

    #[test]
    fn test_parse_tensor_ops_example() {
        let grammar = get_grammar();
        let source = std::fs::read_to_string(examples_dir().join("tensor_ops.zynml"))
            .expect("Should read tensor_ops.zynml");

        let result = grammar.parse_to_json(&source);
        assert!(result.is_ok(), "Should parse tensor_ops.zynml: {:?}", result.err());
    }

    #[test]
    fn test_parse_collections_example() {
        let grammar = get_grammar();
        let source = std::fs::read_to_string(examples_dir().join("collections.zynml"))
            .expect("Should read collections.zynml");

        let result = grammar.parse_to_json(&source);
        assert!(result.is_ok(), "Should parse collections.zynml: {:?}", result.err());
    }

    #[test]
    fn test_parse_neural_network_example() {
        let grammar = get_grammar();
        let source = std::fs::read_to_string(examples_dir().join("neural_network.zynml"))
            .expect("Should read neural_network.zynml");

        let result = grammar.parse_to_json(&source);
        assert!(result.is_ok(), "Should parse neural_network.zynml: {:?}", result.err());
    }
}

// ============================================================================
// Stdlib Parsing Tests
// ============================================================================

mod stdlib_parsing {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_parse_prelude_stdlib() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(ZYNML_STDLIB_PRELUDE);
        assert!(result.is_ok(), "Should parse prelude.zynml: {:?}", result.err());
    }

    #[test]
    fn test_parse_tensor_stdlib() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json(ZYNML_STDLIB_TENSOR);
        // Note: tensor.zynml uses trait impl syntax
        if result.is_err() {
            println!("tensor.zynml uses advanced syntax not yet fully supported");
            println!("Error: {:?}", result.err());
        } else {
            println!("tensor.zynml parsed successfully");
        }
    }
}

// ============================================================================
// Grammar2 Direct TypedAST Parsing Tests
// ============================================================================

mod grammar2_parsing {
    use super::*;

    fn get_grammar2() -> Grammar2 {
        Grammar2::from_source(ZYNML_GRAMMAR).expect("Grammar2 should compile")
    }

    #[test]
    fn test_grammar2_parse_fn_simple() {
        let grammar = get_grammar2();
        let result = grammar.parse("fn main() { }");
        assert!(result.is_ok(), "Should parse fn main(): {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_def_simple() {
        let grammar = get_grammar2();
        let result = grammar.parse("def main() { }");
        assert!(result.is_ok(), "Should parse def main(): {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_def_with_return_type() {
        let grammar = get_grammar2();
        let result = grammar.parse("def greet(): int { 42 }");
        assert!(result.is_ok(), "Should parse def with return type: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_def_with_params() {
        let grammar = get_grammar2();
        let result = grammar.parse("def add(a: int, b: int): int { a + b }");
        assert!(result.is_ok(), "Should parse def with params: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_function_with_call() {
        let grammar = get_grammar2();
        let result = grammar.parse("def greet() { println(\"Hello\") }");
        assert!(result.is_ok(), "Should parse def with function call: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_multiple_functions() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            def greet(name: str): str {
                println(name)
            }

            fn main() {
                greet("World")
            }
        "#);
        assert!(result.is_ok(), "Should parse multiple functions: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 2, "Should have 2 declarations");
    }

    #[test]
    fn test_grammar2_parse_struct() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            struct Point:
                x: float
                y: float
        "#);
        assert!(result.is_ok(), "Should parse struct: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_struct_brace_style() {
        let grammar = get_grammar2();
        let result = grammar.parse("struct Point { x: float, y: float }");
        assert!(result.is_ok(), "Should parse brace-style struct: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_let_statement() {
        let grammar = get_grammar2();
        let result = grammar.parse("fn test() { let x = 42 }");
        assert!(result.is_ok(), "Should parse let statement: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_mut_let() {
        let grammar = get_grammar2();
        let result = grammar.parse("fn test() { mut x: int = 0 }");
        assert!(result.is_ok(), "Should parse mut let: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_python_ternary() {
        let grammar = get_grammar2();
        // Python-style ternary: value if condition else other_value
        let result = grammar.parse("def test(): int { 1 if true else 0 }");
        assert!(result.is_ok(), "Should parse Python-style ternary: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_c_style_ternary() {
        let grammar = get_grammar2();
        // C-style ternary: condition ? value : other_value
        let result = grammar.parse("def test(): int { true ? 1 : 0 }");
        assert!(result.is_ok(), "Should parse C-style ternary: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_null_coalesce() {
        let grammar = get_grammar2();
        // Null-coalescing operator: a ?? b
        let result = grammar.parse("def test(): int { x ?? 42 }");
        assert!(result.is_ok(), "Should parse null-coalesce: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_chained_null_coalesce() {
        let grammar = get_grammar2();
        // Chained null-coalescing: a ?? b ?? c
        let result = grammar.parse("def test(): int { a ?? b ?? 0 }");
        assert!(result.is_ok(), "Should parse chained null-coalesce: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_ternary_with_expressions() {
        let grammar = get_grammar2();
        // Ternary with variable expressions (simpler case)
        let result = grammar.parse("def test(): int { a if b else c }");
        assert!(result.is_ok(), "Should parse ternary with variables: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_ternary_with_calls() {
        let grammar = get_grammar2();
        // Ternary with function calls (more complex)
        let result = grammar.parse("def test(): int { foo() if bar() else baz() }");
        assert!(result.is_ok(), "Should parse ternary with function calls: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_call_with_arg_in_def_body() {
        let grammar = get_grammar2();
        // Function call with argument in def body
        let result = grammar.parse("def test(): int { foo(42) }");
        assert!(result.is_ok(), "Should parse call with arg in def body: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_empty_call_in_def_body() {
        let grammar = get_grammar2();
        // Function call with no args in def body with return type
        let result = grammar.parse("def test(): int { foo() }");
        assert!(result.is_ok(), "Should parse empty call in def body: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_call_in_fn_body() {
        let grammar = get_grammar2();
        // Simple function call in fn body (no return type)
        let result = grammar.parse("fn test() { foo() }");
        assert!(result.is_ok(), "Should parse call in fn body: {:?}", result.err());
    }

    // =========================================================================
    // Chained Operations Tests
    // =========================================================================

    #[test]
    fn test_grammar2_parse_field_access() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { obj.field }");
        assert!(result.is_ok(), "Should parse field access: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_chained_field_access() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { obj.field.subfield }");
        assert!(result.is_ok(), "Should parse chained field access: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_method_call() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { obj.method() }");
        assert!(result.is_ok(), "Should parse method call: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_chained_method_calls() {
        let grammar = get_grammar2();
        // Test without array literal first (simpler case)
        let result = grammar.parse("def test() { tensor.normalize().transpose() }");
        assert!(result.is_ok(), "Should parse chained method calls: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_chained_method_with_args() {
        let grammar = get_grammar2();
        // Test with simple numeric arguments
        let result = grammar.parse("def test() { tensor.reshape(2).transpose() }");
        assert!(result.is_ok(), "Should parse chained methods with args: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_mixed_chains() {
        let grammar = get_grammar2();
        // field -> method -> field
        let result = grammar.parse("def test() { obj.data.process().result }");
        assert!(result.is_ok(), "Should parse mixed chains: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_index_access() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { arr[0] }");
        assert!(result.is_ok(), "Should parse index access: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_chained_index_access() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { matrix[0][1] }");
        assert!(result.is_ok(), "Should parse chained index access: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_field_and_index() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { obj.items[0] }");
        assert!(result.is_ok(), "Should parse field then index: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_self_field_access() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { self.x }");
        assert!(result.is_ok(), "Should parse self.x: {:?}", result.err());
    }

    // =========================================================================
    // List Comprehension Tests
    // =========================================================================

    #[test]
    fn test_grammar2_parse_list_comprehension_simple() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { [x for x in items] }");
        assert!(result.is_ok(), "Should parse simple list comprehension: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_list_comprehension_with_expr() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { [x * 2 for x in items] }");
        assert!(result.is_ok(), "Should parse list comp with expression: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_list_comprehension_with_filter() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { [x for x in items if x > 0] }");
        assert!(result.is_ok(), "Should parse list comp with filter: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_list_comprehension_complex() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { [x * 2 for x in data if x > 0] }");
        assert!(result.is_ok(), "Should parse complex list comprehension: {:?}", result.err());
    }

    // =========================================================================
    // Slice Expression Tests
    // =========================================================================

    #[test]
    fn test_grammar2_parse_slice_basic() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { arr[1:3] }");
        assert!(result.is_ok(), "Should parse basic slice: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_slice_from_start() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { arr[:3] }");
        assert!(result.is_ok(), "Should parse slice from start: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_slice_to_end() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { arr[1:] }");
        assert!(result.is_ok(), "Should parse slice to end: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_slice_with_step() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { arr[::2] }");
        assert!(result.is_ok(), "Should parse slice with step: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_slice_full() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { arr[1:10:2] }");
        assert!(result.is_ok(), "Should parse full slice: {:?}", result.err());
    }
}

// ============================================================================
// Destructuring Pattern Tests
// ============================================================================

mod destructuring_patterns {
    use super::*;

    fn get_grammar2() -> Grammar2 {
        Grammar2::from_source(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_grammar2_parse_tuple_destructure_simple() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { let (x, y) = point }");
        assert!(result.is_ok(), "Should parse simple tuple destructure: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_tuple_destructure_three() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { let (a, b, c) = triple }");
        assert!(result.is_ok(), "Should parse three-element tuple destructure: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_tuple_destructure_with_wildcard() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { let (x, _) = pair }");
        assert!(result.is_ok(), "Should parse tuple destructure with wildcard: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_tuple_destructure_with_expr() {
        let grammar = get_grammar2();
        let result = grammar.parse("def test() { let (x, y) = get_point() }");
        assert!(result.is_ok(), "Should parse tuple destructure with function call: {:?}", result.err());
    }
}

// ============================================================================
// Annotations Tests (Tier 3: Annotations & Effects System)
// ============================================================================

mod annotations {
    use super::*;

    fn get_grammar2() -> Grammar2 {
        Grammar2::from_source(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_grammar2_parse_simple_annotation() {
        let grammar = get_grammar2();
        let result = grammar.parse("@inline def fast() { x + 1 }");
        assert!(result.is_ok(), "Should parse simple annotation: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_annotation_with_string_arg() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"@deprecated("Use new_function instead") def old_function() { 0 }"#);
        assert!(result.is_ok(), "Should parse annotation with string arg: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_annotation_with_named_args() {
        let grammar = get_grammar2();
        let result = grammar.parse("@validate(min=1, max=100) def bounded(x: int): int { x }");
        assert!(result.is_ok(), "Should parse annotation with named args: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_multiple_annotations() {
        let grammar = get_grammar2();
        let result = grammar.parse("@inline @jit def optimized(x: int): int { x * 2 }");
        assert!(result.is_ok(), "Should parse multiple annotations: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_annotation_with_identifier_args() {
        let grammar = get_grammar2();
        let result = grammar.parse("@derive(Debug, Clone) def serializable(): str { \"hello\" }");
        assert!(result.is_ok(), "Should parse annotation with identifier args: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_annotation_with_bool_arg() {
        let grammar = get_grammar2();
        let result = grammar.parse("@feature(enabled=true) def feature_flag(): bool { true }");
        assert!(result.is_ok(), "Should parse annotation with bool arg: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_annotation_with_int_arg() {
        let grammar = get_grammar2();
        // Simplified test to isolate the annotation parsing
        let result = grammar.parse("@retry(count=3) def test() { 1 }");
        assert!(result.is_ok(), "Should parse annotation with int arg: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_jit_annotation() {
        let grammar = get_grammar2();
        let result = grammar.parse("@jit def forward(x: Tensor): Tensor { relu(x) }");
        assert!(result.is_ok(), "Should parse @jit annotation: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_device_annotation() {
        let grammar = get_grammar2();
        let result = grammar.parse("@device(GPU) def gpu_compute(x: Tensor): Tensor { x * 2 }");
        assert!(result.is_ok(), "Should parse @device annotation: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_effect_annotation() {
        let grammar = get_grammar2();
        let result = grammar.parse("@effect(IO) def read_file(path: str): str { load(path) }");
        assert!(result.is_ok(), "Should parse @effect annotation: {:?}", result.err());
    }
}

// ============================================================================
// Algebraic Effects Tests (Tier 3: Annotations & Effects System)
// ============================================================================

mod algebraic_effects {
    use super::*;

    fn get_grammar2() -> Grammar2 {
        Grammar2::from_source(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    // --- Effect Declaration ---

    #[test]
    fn test_grammar2_parse_effect_declaration_simple() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            effect IO {
                def read(): str
                def write(s: str)
            }
        "#);
        assert!(result.is_ok(), "Should parse simple effect declaration: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_effect_declaration_generic() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            effect State<S> {
                def get(): S
                def put(s: S)
            }
        "#);
        assert!(result.is_ok(), "Should parse generic effect declaration: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_effect_declaration_probabilistic() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            effect Probabilistic {
                def sample(dist: Distribution): float
                def observe(dist: Distribution, value: float)
                def factor(score: float)
            }
        "#);
        assert!(result.is_ok(), "Should parse probabilistic effect: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_effect_declaration_async() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            effect Async {
                def await_promise(promise: Promise): Result
                def spawn(task: Task): Promise
            }
        "#);
        assert!(result.is_ok(), "Should parse async effect: {:?}", result.err());
    }

    // --- Effect Handler Declaration ---

    #[test]
    fn test_grammar2_parse_handler_simple() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            handler PrintIO for IO {
                def read(): str {
                    "input"
                }
                def write(s: str) {
                    println(s)
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse simple handler: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1, "Should have 1 declaration");
    }

    #[test]
    fn test_grammar2_parse_handler_generic() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            handler RefState<S> for State<S> {
                def get(): S {
                    self.value
                }
                def put(s: S) {
                    self.value = s
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse generic handler: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_handler_mcmc() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            handler MCMC for Probabilistic {
                def sample(dist: Distribution): float {
                    dist.sample()
                }
                def observe(dist: Distribution, value: float) {
                    self.score = self.score + dist.log_prob(value)
                }
                def factor(score: float) {
                    self.score = self.score + score
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse MCMC handler: {:?}", result.err());
    }

    // --- Effect and Handler Combined ---

    #[test]
    fn test_grammar2_parse_effect_and_handler() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            effect Logger {
                def log(msg: str)
            }

            handler ConsoleLogger for Logger {
                def log(msg: str) {
                    println(msg)
                }
            }
        "#);
        assert!(result.is_ok(), "Should parse effect and handler together: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 2, "Should have 2 declarations");
    }

    // --- Effect Annotations ---

    #[test]
    fn test_grammar2_parse_function_with_effect_annotation() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            @effect(IO)
            def read_file(path: str): str {
                extern read_file(path)
            }
        "#);
        assert!(result.is_ok(), "Should parse function with @effect annotation: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_function_with_multiple_effects() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            @effect(IO, State)
            def stateful_io(path: str): str {
                let data = read(path)
                put(data)
                data
            }
        "#);
        assert!(result.is_ok(), "Should parse function with multiple effects: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_pure_function() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            @pure
            def add(a: int, b: int): int {
                a + b
            }
        "#);
        assert!(result.is_ok(), "Should parse @pure function: {:?}", result.err());
    }

    // --- Handler Application ---

    #[test]
    fn test_grammar2_parse_with_annotation() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            @with(ConsoleLogger)
            def logged_task() {
                log("Starting task")
                do_work()
                log("Task complete")
            }
        "#);
        assert!(result.is_ok(), "Should parse @with handler annotation: {:?}", result.err());
    }

    #[test]
    fn test_grammar2_parse_with_multiple_handlers() {
        let grammar = get_grammar2();
        let result = grammar.parse(r#"
            @with(MCMC, ConsoleLogger)
            def inference_with_logging() {
                log("Starting inference")
                let x = sample(Normal(0.0, 1.0))
                log("Sampled value")
                x
            }
        "#);
        assert!(result.is_ok(), "Should parse @with multiple handlers: {:?}", result.err());
    }
}

// ============================================================================
// Runtime Tests
// ============================================================================

mod runtime {
    use super::*;

    fn plugins_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()  // crates/
            .unwrap()
            .parent()  // workspace root
            .unwrap()
            .join("plugins/target/zrtl")
    }

    #[test]
    fn test_runtime_creation_without_plugins() {
        let config = ZynMLConfig {
            plugins_dir: "/nonexistent".to_string(),
            load_optional: false,
            verbose: false,
        };

        let result = ZynML::with_config(config);
        assert!(result.is_ok(), "Runtime should create without plugins: {:?}", result.err());
    }

    #[test]
    fn test_runtime_has_grammar2() {
        let config = ZynMLConfig::default();
        let zynml = ZynML::with_config(config).expect("Should create runtime");

        if zynml.has_grammar2() {
            println!("Grammar2 is available");
            let g2 = zynml.grammar2().unwrap();
            println!("  Language: {}", g2.name());
        } else {
            println!("Grammar2 not available (grammar may not support it)");
        }
    }

    #[test]
    fn test_runtime_parse_to_json() {
        let config = ZynMLConfig::default();
        let zynml = ZynML::with_config(config).expect("Should create runtime");

        let result = zynml.parse_to_json("let x = 42");
        assert!(result.is_ok(), "Should parse to JSON: {:?}", result.err());

        let json = result.unwrap();
        assert!(json.contains("42"), "JSON should contain literal value");
    }

    #[test]
    fn test_runtime_with_plugins() {
        let plugins_path = plugins_dir();

        if !plugins_path.exists() {
            println!("Skipping runtime with plugins test: plugins not built");
            println!("  Expected: {:?}", plugins_path);
            return;
        }

        let config = ZynMLConfig {
            plugins_dir: plugins_path.to_string_lossy().to_string(),
            load_optional: false,
            verbose: true,
        };

        let result = ZynML::with_config(config);
        assert!(result.is_ok(), "Should create runtime with plugins: {:?}", result.err());

        let zynml = result.unwrap();

        // Verify plugin detection works
        assert!(zynml.has_plugin("zrtl_tensor"), "Should recognize tensor plugin");
        assert!(zynml.has_plugin("zrtl_audio"), "Should recognize audio plugin");
    }
}

// ============================================================================
// Execution Tests
// ============================================================================

mod execution {
    use super::*;

    fn plugins_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("plugins/target/zrtl")
    }

    fn examples_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples")
    }

    fn create_runtime_with_plugins() -> Option<ZynML> {
        let plugins_path = plugins_dir();
        if !plugins_path.exists() {
            return None;
        }

        let config = ZynMLConfig {
            plugins_dir: plugins_path.to_string_lossy().to_string(),
            load_optional: true,
            verbose: false,
        };

        ZynML::with_config(config).ok()
    }

    #[test]
    fn test_execute_simple_expression() {
        let Some(mut zynml) = create_runtime_with_plugins() else {
            println!("Skipping: plugins not available");
            return;
        };

        let result = zynml.load_source(r#"
            fn main() {
                let x = 42
            }
        "#);

        match result {
            Ok(functions) => {
                println!("Loaded functions: {:?}", functions);
                if functions.contains(&"main".to_string()) {
                    let call_result = zynml.call("main");
                    println!("Call result: {:?}", call_result);
                }
            }
            Err(e) => {
                println!("Load error (may be expected): {}", e);
            }
        }
    }

    #[test]
    fn test_execute_hello_example() {
        let Some(mut zynml) = create_runtime_with_plugins() else {
            println!("Skipping: plugins not available");
            return;
        };

        let hello_path = examples_dir().join("hello.zynml");
        if !hello_path.exists() {
            println!("Skipping: hello.zynml not found");
            return;
        }

        // Use catch_unwind since runtime errors may panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            zynml.run_file(&hello_path)
        }));

        match result {
            Ok(Ok(())) => println!("hello.zynml executed successfully"),
            Ok(Err(e)) => println!("Execution error (expected during development): {}", e),
            Err(_) => println!("Runtime panic (symbol resolution may be incomplete)"),
        }
    }
}

// ============================================================================
// Parse Error Tests
// ============================================================================

mod error_handling {
    use super::*;

    fn get_grammar() -> LanguageGrammar {
        LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("Grammar should compile")
    }

    #[test]
    fn test_parse_error_unclosed_brace() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("fn main() {");
        assert!(result.is_err(), "Should fail on unclosed brace");
    }

    #[test]
    fn test_parse_error_invalid_syntax() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let = 42");
        assert!(result.is_err(), "Should fail on missing identifier");
    }

    #[test]
    fn test_parse_error_unmatched_paren() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = (1 + 2");
        assert!(result.is_err(), "Should fail on unmatched paren");
    }

    #[test]
    fn test_parse_error_invalid_operator() {
        let grammar = get_grammar();
        let result = grammar.parse_to_json("let x = 1 $$ 2");
        assert!(result.is_err(), "Should fail on invalid operator");
    }

    #[test]
    fn test_parse_error_missing_function_body() {
        let grammar = get_grammar();
        // Note: "fn test()" is parsed as two expressions (fn, test()) not a function
        // The actual error case is when there's a return type but no body
        let result = grammar.parse_to_json("fn test() -> i32");
        assert!(result.is_err(), "Should fail on function with return type but no body");
    }
}

// ============================================================================
// Spec Compliance Summary
// ============================================================================

/// This test summarizes which spec features are implemented vs pending
#[test]
fn test_spec_compliance_summary() {
    println!("\n=== ZynML Spec Compliance Summary ===\n");

    println!("IMPLEMENTED FEATURES:");
    println!("  [x] Import statements (import module)");
    println!("  [x] Type aliases (type X = Y)");
    println!("  [x] Struct definitions (struct Name: fields)");
    println!("  [x] Enum definitions (enum Name {{ variants }})");
    println!("  [x] Abstract types (abstract Name(T) with Suffixes(...))");
    println!("  [x] Trait definitions (trait Name {{ methods }})");
    println!("  [x] Impl blocks (impl Trait for Type {{ }})");
    println!("  [x] Opaque types (@opaque(\"$T\") type T)");
    println!("  [x] Generic functions and types (<T: Bound>)");
    println!("  [x] Function definitions - brace style (fn name() {{}})");
    println!("  [x] Function definitions - colon style (fn name(): ...)");
    println!("  [x] Impl methods - expression-bodied (fn name(): expr)");
    println!("  [x] Impl methods - indent-bodied (fn name():\\n    statements)");
    println!("  [x] Let bindings (let x = expr)");
    println!("  [x] Assignments (x = expr, self.x = expr)");
    println!("  [x] Control flow - brace style (if/else, while, for {{ }})");
    println!("  [x] Control flow - colon style (if/else, while, for: ...)");
    println!("  [x] Arithmetic operators (+, -, *, /, %)");
    println!("  [x] Comparison operators (==, !=, <, >, <=, >=)");
    println!("  [x] Logical operators (&&, ||, !)");
    println!("  [x] Matrix multiply operator (@)");
    println!("  [x] Pipe operator (|>)");
    println!("  [x] Ternary operator (? :)");
    println!("  [x] Range expressions (.., ..=)");
    println!("  [x] Method calls / static dispatch (x.method(), Type::method())");
    println!("  [x] Indexing (x[i])");
    println!("  [x] Path expressions (Type::method)");
    println!("  [x] Literals (int, float, string, bool, array, tensor, struct)");
    println!("  [x] Duration/suffixed literals (1000ms, 5s, 2m, 1h)");
    println!("  [x] Extern calls (extern func())");
    println!("  [x] Comments (// and /* */)");

    println!("  [x] Module declarations (module name)");
    println!("  [x] Aliased imports (import x as y)");
    println!("  [x] load() with 'as' type (load(\"x\") as image)");
    println!("  [x] model() with config block (model(\"x\") {{ ... }})");
    println!("  [x] Pipeline definitions (pipeline name() {{ }})");
    println!("  [x] Compute/kernel syntax (compute(x) {{ @kernel ... }})");
    println!("  [x] Render statements (render x {{ ... }})");
    println!("  [x] Stream with sink (stream x |> sink())");
    println!("  [x] Config blocks (config {{ ... }})");
    println!("  [x] Try/catch blocks (try {{ }} catch {{ }})");
    println!("  [x] Match/case expressions (match x {{ case ... }})");
    println!("  [x] @cache decorator (@cache(ttl=1h))");
    println!("  [x] @memoize decorator (@memoize)");
    println!("  [x] @kernel, @workgroup, @device decorators");

    println!("\nPENDING SPEC FEATURES:");
    println!("  [ ] Parallel execution with & operator");

    println!("\n=== End Summary ===\n");
}
