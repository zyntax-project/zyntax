//! Grammar Intermediate Representation for ZynPEG 2.0
//!
//! This module defines the IR types that represent a parsed grammar with named bindings.
//! Unlike the legacy system which uses positional `$N` captures, this IR:
//!
//! - Uses named bindings: `name:identifier` instead of `$1`
//! - Handles optionals natively: `params?` returns `Option<T>`
//! - Represents repetitions directly: `items*` returns `Vec<T>`
//! - Supports semantic actions as direct Rust code

use std::collections::HashMap;

/// A complete grammar definition with all its rules and metadata
#[derive(Debug, Clone)]
pub struct GrammarIR {
    /// Language metadata (name, version, extensions)
    pub metadata: GrammarMetadata,
    /// Import statements for generated code
    pub imports: Vec<String>,
    /// Type declarations (opaque types, function returns)
    pub type_decls: TypeDeclarations,
    /// Built-in function mappings
    pub builtins: BuiltinMappings,
    /// All grammar rules indexed by name
    pub rules: HashMap<String, RuleIR>,
    /// Entry rule name (usually "program" or "file")
    pub entry_rule: String,
}

/// Language metadata from @language directive
#[derive(Debug, Clone, Default)]
pub struct GrammarMetadata {
    pub name: String,
    pub version: String,
    pub file_extensions: Vec<String>,
    pub entry_point: Option<String>,
}

/// Type declarations from @types directive
#[derive(Debug, Clone, Default)]
pub struct TypeDeclarations {
    /// Opaque type names (ZRTL-backed)
    pub opaque_types: Vec<String>,
    /// Function name -> return type mapping
    pub function_returns: HashMap<String, String>,
}

/// Built-in function mappings from @builtin directive
#[derive(Debug, Clone, Default)]
pub struct BuiltinMappings {
    /// Direct function mappings: name -> symbol
    pub functions: HashMap<String, String>,
    /// Method mappings: method_name -> [possible_implementations]
    pub methods: HashMap<String, Vec<String>>,
    /// Operator mappings: operator -> [possible_implementations]
    pub operators: HashMap<String, Vec<String>>,
}

/// A single grammar rule with pattern and optional action
#[derive(Debug, Clone)]
pub struct RuleIR {
    /// Rule name (e.g., "fn_def", "expression")
    pub name: String,
    /// Rule modifier (atomic, silent, etc.)
    pub modifier: Option<RuleModifier>,
    /// The parsing pattern with named bindings
    pub pattern: PatternIR,
    /// Semantic action for AST construction
    pub action: Option<ActionIR>,
    /// Return type of this rule
    pub return_type: Option<String>,
}

/// Rule modifiers that affect parsing behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleModifier {
    /// `@` - Atomic rule (no whitespace between elements)
    Atomic,
    /// `_` - Silent rule (doesn't appear in AST)
    Silent,
    /// `$` - Compound rule (captures as single string)
    Compound,
    /// `!` - Non-atomic rule (explicit whitespace handling)
    NonAtomic,
}

/// Pattern IR with named bindings
///
/// Patterns describe what to parse and how to capture the results.
/// Named bindings (`name:rule`) create local variables accessible in actions.
#[derive(Debug, Clone)]
pub enum PatternIR {
    /// Literal string match: `"fn"`, `"("`, etc.
    /// Literals never produce bindings.
    Literal(String),

    /// Character class: `'a'..'z'`, `ASCII_DIGIT`, etc.
    CharClass(CharClass),

    /// Reference to another rule with optional binding
    /// - `identifier` - no binding, result is anonymous
    /// - `name:identifier` - binds result to `name`
    RuleRef {
        rule_name: String,
        binding: Option<String>,
    },

    /// Sequence of patterns: `p1 ~ p2 ~ p3`
    Sequence(Vec<PatternIR>),

    /// Ordered choice: `p1 | p2 | p3`
    Choice(Vec<PatternIR>),

    /// Optional pattern: `p?`
    /// Returns `Option<T>` where T is the pattern's type
    Optional(Box<PatternIR>),

    /// Zero or more repetitions: `p*`
    /// Returns `Vec<T>`
    Repeat {
        pattern: Box<PatternIR>,
        min: usize,                        // 0 for *, 1 for +
        max: Option<usize>,                // None for unlimited
        separator: Option<Box<PatternIR>>, // For comma-separated lists
    },

    /// Positive lookahead: `&p`
    /// Succeeds if p matches but doesn't consume input
    PositiveLookahead(Box<PatternIR>),

    /// Negative lookahead: `!p`
    /// Succeeds if p doesn't match, doesn't consume input
    NegativeLookahead(Box<PatternIR>),

    /// Any single character: `ANY`
    Any,

    /// Start of input: `SOI`
    StartOfInput,

    /// End of input: `EOI`
    EndOfInput,

    /// Whitespace handling: `WHITESPACE` or `_`
    Whitespace,
}

/// Character class for matching character ranges
#[derive(Debug, Clone)]
pub enum CharClass {
    /// Single character: `'a'`
    Single(char),
    /// Character range: `'a'..'z'`
    Range(char, char),
    /// Built-in class: `ASCII_DIGIT`, `ASCII_ALPHA`, etc.
    Builtin(String),
    /// Union of classes: `ASCII_DIGIT | ASCII_ALPHA`
    Union(Vec<CharClass>),
    /// Negation: `!('a'..'z')`
    Negation(Box<CharClass>),
}

/// Semantic action for constructing TypedAST nodes
///
/// Actions describe how to construct AST nodes from parsed results.
/// They can reference named bindings from the pattern.
#[derive(Debug, Clone)]
pub enum ActionIR {
    /// Direct AST node construction
    /// ```zyn
    /// -> TypedExpression::Binary {
    ///     left: left,
    ///     op: op,
    ///     right: right,
    /// }
    /// ```
    Construct {
        /// The TypedAST type path (e.g., "TypedExpression::Binary")
        type_path: String,
        /// Field name -> expression mapping
        fields: Vec<(String, ExprIR)>,
    },

    /// Call a helper function
    /// ```zyn
    /// -> fold_binary_left(items)
    /// ```
    HelperCall { function: String, args: Vec<ExprIR> },

    /// Pass through a binding directly (for wrapper rules)
    /// ```zyn
    /// -> inner
    /// ```
    PassThrough { binding: String },

    /// Match on the type of a binding
    /// ```zyn
    /// -> match kind {
    ///     "let" => TypedStatement::Let { ... },
    ///     "const" => TypedStatement::Const { ... },
    /// }
    /// ```
    Match {
        binding: String,
        cases: Vec<(String, Box<ActionIR>)>,
    },

    /// Conditional action
    /// ```zyn
    /// -> if ret.is_some() {
    ///     TypedDeclaration::Function { ... }
    /// } else {
    ///     TypedDeclaration::Procedure { ... }
    /// }
    /// ```
    Conditional {
        condition: ExprIR,
        then_action: Box<ActionIR>,
        else_action: Option<Box<ActionIR>>,
    },

    /// Legacy JSON command syntax (for backwards compatibility)
    /// ```zyn
    /// -> TypedStatement {
    ///     "commands": [
    ///         { "define": "let_stmt", "args": { "name": "$1", ... }}
    ///     ]
    /// }
    /// ```
    LegacyJson {
        /// The return type (e.g., "TypedStatement")
        return_type: String,
        /// Raw JSON content as string
        json_content: String,
    },
}

/// Expression IR for action code
///
/// Expressions can reference bindings, call methods, access fields, etc.
#[derive(Debug, Clone)]
pub enum ExprIR {
    /// Reference to a named binding: `name`
    Binding(String),

    /// Field access: `binding.field`
    FieldAccess { base: Box<ExprIR>, field: String },

    /// Method call: `binding.method(args)`
    MethodCall {
        receiver: Box<ExprIR>,
        method: String,
        args: Vec<ExprIR>,
    },

    /// Function call: `func(args)`
    FunctionCall { function: String, args: Vec<ExprIR> },

    /// String literal: `"text"`
    StringLit(String),

    /// Integer literal: `42`
    IntLit(i64),

    /// Boolean literal: `true`, `false`
    BoolLit(bool),

    /// List construction: `[a, b, c]`
    List(Vec<ExprIR>),

    /// Unwrap optional with default: `opt.unwrap_or(default)`
    UnwrapOr {
        optional: Box<ExprIR>,
        default: Box<ExprIR>,
    },

    /// Map over optional: `opt.map(|x| expr)`
    MapOption {
        optional: Box<ExprIR>,
        param: String,
        body: Box<ExprIR>,
    },

    /// Struct construction: `Struct { field: value, ... }`
    StructLit {
        type_name: String,
        fields: Vec<(String, ExprIR)>,
    },

    /// Enum variant: `Enum::Variant` or `Enum::Variant(value)`
    EnumVariant {
        type_name: String,
        variant: String,
        value: Option<Box<ExprIR>>,
    },

    /// Type cast: `expr as Type`
    Cast {
        expr: Box<ExprIR>,
        target_type: String,
    },

    /// Arena intern string: `arena.intern(text)`
    Intern(Box<ExprIR>),

    /// Get text from a parse result
    Text(Box<ExprIR>),

    /// Get span from a parse result
    GetSpan(Box<ExprIR>),

    /// Check if optional is some: `opt.is_some()`
    IsSome(Box<ExprIR>),

    /// Binary operation: `a + b`, `a == b`, etc.
    Binary {
        left: Box<ExprIR>,
        op: String,
        right: Box<ExprIR>,
    },

    /// Default value for a type
    Default(String),
}

impl GrammarIR {
    /// Create an empty grammar IR
    pub fn new() -> Self {
        GrammarIR {
            metadata: GrammarMetadata::default(),
            imports: Vec::new(),
            type_decls: TypeDeclarations::default(),
            builtins: BuiltinMappings::default(),
            rules: HashMap::new(),
            entry_rule: "program".to_string(),
        }
    }

    /// Add a rule to the grammar
    pub fn add_rule(&mut self, rule: RuleIR) {
        self.rules.insert(rule.name.clone(), rule);
    }

    /// Get a rule by name
    pub fn get_rule(&self, name: &str) -> Option<&RuleIR> {
        self.rules.get(name)
    }

    /// Get all rule names
    pub fn rule_names(&self) -> impl Iterator<Item = &str> {
        self.rules.keys().map(|s| s.as_str())
    }

    /// Collect all named bindings from a pattern
    pub fn collect_bindings(pattern: &PatternIR) -> Vec<(String, String)> {
        let mut bindings = Vec::new();
        Self::collect_bindings_recursive(pattern, &mut bindings);
        bindings
    }

    fn collect_bindings_recursive(pattern: &PatternIR, bindings: &mut Vec<(String, String)>) {
        match pattern {
            PatternIR::RuleRef { rule_name, binding } => {
                if let Some(name) = binding {
                    bindings.push((name.clone(), rule_name.clone()));
                }
            }
            PatternIR::Sequence(patterns) => {
                for p in patterns {
                    Self::collect_bindings_recursive(p, bindings);
                }
            }
            PatternIR::Choice(patterns) => {
                // For choices, bindings from any branch might be available
                for p in patterns {
                    Self::collect_bindings_recursive(p, bindings);
                }
            }
            PatternIR::Optional(inner) => {
                Self::collect_bindings_recursive(inner, bindings);
            }
            PatternIR::Repeat { pattern, .. } => {
                Self::collect_bindings_recursive(pattern, bindings);
            }
            PatternIR::PositiveLookahead(inner) | PatternIR::NegativeLookahead(inner) => {
                // Lookaheads don't consume, but might still have bindings for type checking
                Self::collect_bindings_recursive(inner, bindings);
            }
            // Patterns that don't produce bindings
            PatternIR::Literal(_)
            | PatternIR::CharClass(_)
            | PatternIR::Any
            | PatternIR::StartOfInput
            | PatternIR::EndOfInput
            | PatternIR::Whitespace => {}
        }
    }
}

impl Default for GrammarIR {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternIR {
    /// Check if this pattern is optional (can match empty input)
    pub fn is_optional(&self) -> bool {
        matches!(
            self,
            PatternIR::Optional(_) | PatternIR::Repeat { min: 0, .. }
        )
    }

    /// Check if this pattern has any named bindings
    pub fn has_bindings(&self) -> bool {
        !GrammarIR::collect_bindings(self).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_bindings_simple() {
        let pattern = PatternIR::Sequence(vec![
            PatternIR::Literal("fn".to_string()),
            PatternIR::RuleRef {
                rule_name: "identifier".to_string(),
                binding: Some("name".to_string()),
            },
            PatternIR::Literal("(".to_string()),
            PatternIR::Optional(Box::new(PatternIR::RuleRef {
                rule_name: "params".to_string(),
                binding: Some("params".to_string()),
            })),
            PatternIR::Literal(")".to_string()),
        ]);

        let bindings = GrammarIR::collect_bindings(&pattern);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("name".to_string(), "identifier".to_string()));
        assert_eq!(bindings[1], ("params".to_string(), "params".to_string()));
    }

    #[test]
    fn test_pattern_is_optional() {
        let required = PatternIR::RuleRef {
            rule_name: "expr".to_string(),
            binding: None,
        };
        assert!(!required.is_optional());

        let optional = PatternIR::Optional(Box::new(required.clone()));
        assert!(optional.is_optional());

        let star = PatternIR::Repeat {
            pattern: Box::new(required.clone()),
            min: 0,
            max: None,
            separator: None,
        };
        assert!(star.is_optional());

        let plus = PatternIR::Repeat {
            pattern: Box::new(required),
            min: 1,
            max: None,
            separator: None,
        };
        assert!(!plus.is_optional());
    }
}
