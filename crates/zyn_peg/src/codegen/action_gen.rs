//! Action Code Generator for ZynPEG 2.0
//!
//! Generates Rust code from ActionIR to construct TypedAST nodes.
//!
//! # Action Types
//!
//! - `Construct`: Direct struct/enum construction
//! - `HelperCall`: Call helper functions like `fold_binary_left`
//! - `PassThrough`: Return a binding directly
//! - `Match`: Pattern match on a binding
//! - `Conditional`: If/else based on a condition

use crate::grammar::{ActionIR, ExprIR};

/// Code generator for semantic actions
pub struct ActionGenerator {
    /// Generated code buffer
    code: String,
    /// Indentation level
    indent: usize,
}

impl ActionGenerator {
    pub fn new() -> Self {
        ActionGenerator {
            code: String::new(),
            indent: 0,
        }
    }

    /// Generate action code from ActionIR
    pub fn generate(&mut self, action: &ActionIR) -> String {
        self.code.clear();
        self.generate_action(action);
        self.code.clone()
    }

    fn generate_action(&mut self, action: &ActionIR) {
        match action {
            ActionIR::Construct { type_path, fields } => {
                self.line(&format!("{} {{", type_path));
                self.indent += 1;
                for (name, expr) in fields {
                    let expr_code = self.generate_expr(expr);
                    self.line(&format!("{}: {},", name, expr_code));
                }
                self.indent -= 1;
                self.line("}");
            }

            ActionIR::HelperCall { function, args } => {
                let args_code: Vec<String> = args.iter().map(|e| self.generate_expr(e)).collect();
                self.line(&format!("{}({})", function, args_code.join(", ")));
            }

            ActionIR::PassThrough { binding } => {
                self.line(binding);
            }

            ActionIR::Match { binding, cases } => {
                self.line(&format!("match {} {{", binding));
                self.indent += 1;
                for (pattern, action) in cases {
                    self.line(&format!("{:?} => {{", pattern));
                    self.indent += 1;
                    self.generate_action(action);
                    self.indent -= 1;
                    self.line("}");
                }
                self.line("_ => panic!(\"unmatched case\"),");
                self.indent -= 1;
                self.line("}");
            }

            ActionIR::Conditional {
                condition,
                then_action,
                else_action,
            } => {
                let cond_code = self.generate_expr(condition);
                self.line(&format!("if {} {{", cond_code));
                self.indent += 1;
                self.generate_action(then_action);
                self.indent -= 1;
                if let Some(else_act) = else_action {
                    self.line("} else {");
                    self.indent += 1;
                    self.generate_action(else_act);
                    self.indent -= 1;
                }
                self.line("}");
            }

            ActionIR::LegacyJson {
                return_type,
                json_content,
            } => {
                // For legacy JSON actions, we generate a comment and placeholder
                // The actual execution uses the runtime interpreter
                self.line(&format!("// Legacy JSON action for {}", return_type));
                self.line(&format!("// JSON: {{{}}}", json_content.trim()));
                self.line("unimplemented!(\"Legacy JSON actions require runtime interpreter\")");
            }
        }
    }

    /// Generate expression code
    fn generate_expr(&self, expr: &ExprIR) -> String {
        match expr {
            ExprIR::Binding(name) => name.clone(),

            ExprIR::FieldAccess { base, field } => {
                format!("{}.{}", self.generate_expr(base), field)
            }

            ExprIR::MethodCall {
                receiver,
                method,
                args,
            } => {
                let args_code: Vec<String> = args.iter().map(|e| self.generate_expr(e)).collect();
                format!(
                    "{}.{}({})",
                    self.generate_expr(receiver),
                    method,
                    args_code.join(", ")
                )
            }

            ExprIR::FunctionCall { function, args } => {
                let args_code: Vec<String> = args.iter().map(|e| self.generate_expr(e)).collect();
                format!("{}({})", function, args_code.join(", "))
            }

            ExprIR::StringLit(s) => format!("{:?}", s),

            ExprIR::IntLit(n) => format!("{}", n),

            ExprIR::BoolLit(b) => format!("{}", b),

            ExprIR::List(items) => {
                let items_code: Vec<String> = items.iter().map(|e| self.generate_expr(e)).collect();
                format!("vec![{}]", items_code.join(", "))
            }

            ExprIR::UnwrapOr { optional, default } => {
                format!(
                    "{}.unwrap_or({})",
                    self.generate_expr(optional),
                    self.generate_expr(default)
                )
            }

            ExprIR::MapOption {
                optional,
                param,
                body,
            } => {
                format!(
                    "{}.map(|{}| {})",
                    self.generate_expr(optional),
                    param,
                    self.generate_expr(body)
                )
            }

            ExprIR::StructLit { type_name, fields } => {
                let fields_code: Vec<String> = fields
                    .iter()
                    .map(|(name, expr)| format!("{}: {}", name, self.generate_expr(expr)))
                    .collect();
                format!("{} {{ {} }}", type_name, fields_code.join(", "))
            }

            ExprIR::EnumVariant {
                type_name,
                variant,
                value,
            } => {
                if let Some(v) = value {
                    format!("{}::{}({})", type_name, variant, self.generate_expr(v))
                } else {
                    format!("{}::{}", type_name, variant)
                }
            }

            ExprIR::Cast { expr, target_type } => {
                format!("{} as {}", self.generate_expr(expr), target_type)
            }

            ExprIR::Intern(expr) => {
                format!("self.state.intern(&{})", self.generate_expr(expr))
            }

            ExprIR::Text(expr) => {
                format!("{}.text()", self.generate_expr(expr))
            }

            ExprIR::GetSpan(expr) => {
                format!("{}.span()", self.generate_expr(expr))
            }

            ExprIR::IsSome(expr) => {
                format!("{}.is_some()", self.generate_expr(expr))
            }

            ExprIR::Binary { left, op, right } => {
                format!(
                    "({} {} {})",
                    self.generate_expr(left),
                    op,
                    self.generate_expr(right)
                )
            }

            ExprIR::Default(type_name) => {
                if type_name.is_empty() {
                    "Default::default()".to_string()
                } else {
                    format!("{}::default()", type_name)
                }
            }
        }
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.code.push_str("    ");
        }
        self.code.push_str(text);
        self.code.push('\n');
    }
}

impl Default for ActionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_construct() {
        let action = ActionIR::Construct {
            type_path: "TypedExpression::Binary".to_string(),
            fields: vec![
                ("left".to_string(), ExprIR::Binding("left".to_string())),
                ("op".to_string(), ExprIR::Binding("op".to_string())),
                ("right".to_string(), ExprIR::Binding("right".to_string())),
            ],
        };

        let mut gen = ActionGenerator::new();
        let code = gen.generate(&action);

        assert!(code.contains("TypedExpression::Binary"));
        assert!(code.contains("left: left"));
        assert!(code.contains("op: op"));
        assert!(code.contains("right: right"));
    }

    #[test]
    fn test_generate_helper_call() {
        let action = ActionIR::HelperCall {
            function: "fold_binary_left".to_string(),
            args: vec![ExprIR::Binding("items".to_string())],
        };

        let mut gen = ActionGenerator::new();
        let code = gen.generate(&action);

        assert!(code.contains("fold_binary_left(items)"));
    }

    #[test]
    fn test_generate_pass_through() {
        let action = ActionIR::PassThrough {
            binding: "inner".to_string(),
        };

        let mut gen = ActionGenerator::new();
        let code = gen.generate(&action);

        assert!(code.contains("inner"));
    }

    #[test]
    fn test_generate_expr_unwrap_or() {
        let gen = ActionGenerator::new();
        let expr = ExprIR::UnwrapOr {
            optional: Box::new(ExprIR::Binding("params".to_string())),
            default: Box::new(ExprIR::List(vec![])),
        };

        let code = gen.generate_expr(&expr);
        assert!(code.contains("params.unwrap_or(vec![])"));
    }
}
