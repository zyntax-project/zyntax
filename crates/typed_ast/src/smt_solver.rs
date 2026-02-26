//! # SMT Solver Integration
//!
//! This module provides integration with external SMT solvers for constraint
//! satisfiability checking in dependent types and refinement predicates.

use crate::dependent_types::RefinementPredicate;
use crate::source::Span;
use crate::type_registry::{ConstValue, Type};
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

/// Result of SMT solver query
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtResult {
    /// Constraint is satisfiable
    Satisfiable,
    /// Constraint is unsatisfiable
    Unsatisfiable,
    /// Solver could not determine satisfiability (timeout, unknown)
    Unknown,
    /// Solver error occurred
    Error(String),
}

/// SMT solver configuration
#[derive(Debug, Clone)]
pub struct SmtConfig {
    /// Path to SMT solver binary (e.g., "z3", "cvc4")
    pub solver_path: String,
    /// Timeout in seconds
    pub timeout: u32,
    /// Additional solver flags
    pub flags: Vec<String>,
    /// Enable model generation
    pub produce_models: bool,
}

impl Default for SmtConfig {
    fn default() -> Self {
        Self {
            solver_path: "z3".to_string(),
            timeout: 5,
            flags: vec!["-in".to_string()],
            produce_models: false,
        }
    }
}

/// SMT solver interface
pub struct SmtSolver {
    config: SmtConfig,
    variable_counter: u32,
    type_context: HashMap<String, SmtType>,
}

/// SMT-LIB types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtType {
    Int,
    Bool,
    Real,
    String,
    Array(Box<SmtType>, Box<SmtType>),
    BitVector(u32),
}

/// SMT-LIB expression
#[derive(Debug, Clone)]
pub enum SmtExpr {
    Variable(String),
    Constant(SmtConstant),
    Application {
        function: String,
        args: Vec<SmtExpr>,
    },
    Let {
        bindings: Vec<(String, SmtExpr)>,
        body: Box<SmtExpr>,
    },
    Quantified {
        quantifier: Quantifier,
        variables: Vec<(String, SmtType)>,
        body: Box<SmtExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum SmtConstant {
    Int(i64),
    Bool(bool),
    Real(String), // Store as string to avoid floating point precision issues
    String(String),
    BitVector { value: u64, width: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    ForAll,
    Exists,
}

impl SmtSolver {
    /// Create a new SMT solver with default configuration
    pub fn new() -> Self {
        Self::with_config(SmtConfig::default())
    }

    /// Create a new SMT solver with custom configuration
    pub fn with_config(config: SmtConfig) -> Self {
        Self {
            config,
            variable_counter: 0,
            type_context: HashMap::new(),
        }
    }

    /// Check satisfiability of a refinement predicate
    pub fn check_predicate_satisfiable(
        &mut self,
        predicate: &RefinementPredicate,
        value_type: &Type,
        context: &HashMap<String, Type>,
    ) -> Result<SmtResult, String> {
        // Generate SMT-LIB formula
        let smt_formula = self.translate_predicate(predicate, value_type, context)?;

        // Create SMT-LIB script
        let script = self.generate_smt_script(&smt_formula)?;

        // Execute solver
        self.execute_solver(&script)
    }

    /// Generate a fresh variable name
    fn fresh_variable(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.variable_counter);
        self.variable_counter += 1;
        name
    }

    /// Translate refinement predicate to SMT expression
    fn translate_predicate(
        &mut self,
        predicate: &RefinementPredicate,
        _value_type: &Type,
        _context: &HashMap<String, Type>,
    ) -> Result<SmtExpr, String> {
        use RefinementPredicate::*;

        match predicate {
            Constant(value) => Ok(SmtExpr::Constant(SmtConstant::Bool(*value))),

            Variable(var) => Ok(SmtExpr::Variable(var.to_string())),

            Comparison { op, left, right } => {
                let left_expr = self.translate_refinement_expr(left)?;
                let right_expr = self.translate_refinement_expr(right)?;
                let op_name = self.comparison_op_to_smt(op);

                Ok(SmtExpr::Application {
                    function: op_name,
                    args: vec![left_expr, right_expr],
                })
            }

            And(left, right) => {
                let left_expr = self.translate_predicate(left, _value_type, _context)?;
                let right_expr = self.translate_predicate(right, _value_type, _context)?;

                Ok(SmtExpr::Application {
                    function: "and".to_string(),
                    args: vec![left_expr, right_expr],
                })
            }

            Or(left, right) => {
                let left_expr = self.translate_predicate(left, _value_type, _context)?;
                let right_expr = self.translate_predicate(right, _value_type, _context)?;

                Ok(SmtExpr::Application {
                    function: "or".to_string(),
                    args: vec![left_expr, right_expr],
                })
            }

            Not(inner) => {
                let inner_expr = self.translate_predicate(inner, _value_type, _context)?;

                Ok(SmtExpr::Application {
                    function: "not".to_string(),
                    args: vec![inner_expr],
                })
            }

            Implies(antecedent, consequent) => {
                let ant_expr = self.translate_predicate(antecedent, _value_type, _context)?;
                let cons_expr = self.translate_predicate(consequent, _value_type, _context)?;

                Ok(SmtExpr::Application {
                    function: "=>".to_string(),
                    args: vec![ant_expr, cons_expr],
                })
            }

            _ => {
                // For other predicates, return a placeholder
                Ok(SmtExpr::Constant(SmtConstant::Bool(true)))
            }
        }
    }

    /// Translate refinement expression to SMT expression
    fn translate_refinement_expr(
        &mut self,
        expr: &crate::dependent_types::RefinementExpr,
    ) -> Result<SmtExpr, String> {
        use crate::dependent_types::RefinementExpr;

        match expr {
            RefinementExpr::Variable(var) => Ok(SmtExpr::Variable(var.to_string())),

            RefinementExpr::Constant(value) => {
                Ok(SmtExpr::Constant(self.translate_const_value(value)?))
            }

            RefinementExpr::Binary { op, left, right } => {
                let left_expr = self.translate_refinement_expr(left)?;
                let right_expr = self.translate_refinement_expr(right)?;
                let op_name = self.binary_op_to_smt(op);

                Ok(SmtExpr::Application {
                    function: op_name,
                    args: vec![left_expr, right_expr],
                })
            }

            RefinementExpr::Unary { op, operand } => {
                let operand_expr = self.translate_refinement_expr(operand)?;
                let op_name = self.unary_op_to_smt(op);

                Ok(SmtExpr::Application {
                    function: op_name,
                    args: vec![operand_expr],
                })
            }

            RefinementExpr::Call { func, args } => {
                let arg_exprs: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| self.translate_refinement_expr(arg))
                    .collect();

                Ok(SmtExpr::Application {
                    function: func.to_string(),
                    args: arg_exprs?,
                })
            }

            _ => {
                // For complex expressions, use placeholder
                Ok(SmtExpr::Constant(SmtConstant::Int(0)))
            }
        }
    }

    /// Translate const value to SMT constant
    fn translate_const_value(&self, value: &ConstValue) -> Result<SmtConstant, String> {
        match value {
            ConstValue::Int(i) => Ok(SmtConstant::Int(*i)),
            ConstValue::UInt(u) => Ok(SmtConstant::Int(*u as i64)), // Simplified
            ConstValue::Bool(b) => Ok(SmtConstant::Bool(*b)),
            ConstValue::String(s) => Ok(SmtConstant::String(s.to_string())),
            ConstValue::Char(c) => Ok(SmtConstant::Int(*c as i64)), // Treat as ASCII
            _ => Err(format!("Unsupported const value for SMT: {:?}", value)),
        }
    }

    /// Convert comparison operator to SMT function name
    fn comparison_op_to_smt(&self, op: &crate::dependent_types::ComparisonOp) -> String {
        use crate::dependent_types::ComparisonOp::*;

        match op {
            Equal => "=".to_string(),
            NotEqual => "distinct".to_string(),
            Less => "<".to_string(),
            LessEqual => "<=".to_string(),
            Greater => ">".to_string(),
            GreaterEqual => ">=".to_string(),
            In => "member".to_string(), // Would need set theory
            NotIn => "not-member".to_string(),
        }
    }

    /// Convert binary operator to SMT function name
    fn binary_op_to_smt(&self, op: &crate::dependent_types::ArithmeticOp) -> String {
        use crate::dependent_types::ArithmeticOp::*;

        match op {
            Add => "+".to_string(),
            Sub => "-".to_string(),
            Mul => "*".to_string(),
            Div => "div".to_string(),
            Mod => "mod".to_string(),
            BitAnd => "and".to_string(),
            BitOr => "or".to_string(),
            BitXor => "xor".to_string(),
            Shl => "shl".to_string(),
            Shr => "shr".to_string(),
        }
    }

    /// Convert unary operator to SMT function name
    fn unary_op_to_smt(&self, op: &crate::dependent_types::UnaryOp) -> String {
        use crate::dependent_types::UnaryOp::*;

        match op {
            Neg => "-".to_string(),
            BitNot => "not".to_string(),
        }
    }

    /// Generate complete SMT-LIB script
    fn generate_smt_script(&self, formula: &SmtExpr) -> Result<String, String> {
        let mut script = String::new();

        // Set logic
        script.push_str("(set-logic LIA)\n"); // Linear Integer Arithmetic

        // Set options
        if self.config.produce_models {
            script.push_str("(set-option :produce-models true)\n");
        }

        // Declare variables (would need to track variable types)
        // TODO: Add variable declarations based on type context

        // Assert the formula
        script.push_str(&format!("(assert {})\n", self.expr_to_smt_string(formula)?));

        // Check satisfiability
        script.push_str("(check-sat)\n");

        if self.config.produce_models {
            script.push_str("(get-model)\n");
        }

        script.push_str("(exit)\n");

        Ok(script)
    }

    /// Convert SMT expression to SMT-LIB string format
    fn expr_to_smt_string(&self, expr: &SmtExpr) -> Result<String, String> {
        match expr {
            SmtExpr::Variable(name) => Ok(name.clone()),

            SmtExpr::Constant(constant) => Ok(self.constant_to_smt_string(constant)),

            SmtExpr::Application { function, args } => {
                if args.is_empty() {
                    Ok(function.clone())
                } else {
                    let arg_strings: Result<Vec<_>, _> = args
                        .iter()
                        .map(|arg| self.expr_to_smt_string(arg))
                        .collect();

                    Ok(format!("({} {})", function, arg_strings?.join(" ")))
                }
            }

            SmtExpr::Let { bindings, body } => {
                let binding_strings: Result<Vec<String>, String> = bindings
                    .iter()
                    .map(|(var, expr)| Ok(format!("({} {})", var, self.expr_to_smt_string(expr)?)))
                    .collect();

                Ok(format!(
                    "(let ({}) {})",
                    binding_strings?.join(" "),
                    self.expr_to_smt_string(body)?
                ))
            }

            SmtExpr::Quantified {
                quantifier,
                variables,
                body,
            } => {
                let quant_name = match quantifier {
                    Quantifier::ForAll => "forall",
                    Quantifier::Exists => "exists",
                };

                let var_decls: Vec<String> = variables
                    .iter()
                    .map(|(name, typ)| format!("({} {})", name, self.type_to_smt_string(typ)))
                    .collect();

                Ok(format!(
                    "({} ({}) {})",
                    quant_name,
                    var_decls.join(" "),
                    self.expr_to_smt_string(body)?
                ))
            }
        }
    }

    /// Convert SMT constant to string
    fn constant_to_smt_string(&self, constant: &SmtConstant) -> String {
        match constant {
            SmtConstant::Int(i) => i.to_string(),
            SmtConstant::Bool(b) => b.to_string(),
            SmtConstant::Real(r) => r.clone(),
            SmtConstant::String(s) => format!("\"{}\"", s),
            SmtConstant::BitVector { value, width } => {
                format!("(_ bv{} {})", value, width)
            }
        }
    }

    /// Convert SMT type to string
    fn type_to_smt_string(&self, typ: &SmtType) -> String {
        match typ {
            SmtType::Int => "Int".to_string(),
            SmtType::Bool => "Bool".to_string(),
            SmtType::Real => "Real".to_string(),
            SmtType::String => "String".to_string(),
            SmtType::Array(index, element) => {
                format!(
                    "(Array {} {})",
                    self.type_to_smt_string(index),
                    self.type_to_smt_string(element)
                )
            }
            SmtType::BitVector(width) => format!("(_ BitVec {})", width),
        }
    }

    /// Execute SMT solver with the given script
    fn execute_solver(&self, script: &str) -> Result<SmtResult, String> {
        // Try to execute the solver
        let mut child = Command::new(&self.config.solver_path)
            .args(&self.config.flags)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "Failed to start SMT solver '{}': {}",
                    self.config.solver_path, e
                )
            })?;

        // Send script to solver
        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(script.as_bytes())
                .map_err(|e| format!("Failed to write to SMT solver: {}", e))?;
        }

        // Get result
        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to read SMT solver output: {}", e))?;

        if !output.status.success() {
            return Ok(SmtResult::Error(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse result
        if stdout.contains("sat") && !stdout.contains("unsat") {
            Ok(SmtResult::Satisfiable)
        } else if stdout.contains("unsat") {
            Ok(SmtResult::Unsatisfiable)
        } else if stdout.contains("unknown") {
            Ok(SmtResult::Unknown)
        } else {
            Ok(SmtResult::Error(format!(
                "Unexpected solver output: {}",
                stdout
            )))
        }
    }

    /// Check if SMT solver is available
    pub fn is_available(&self) -> bool {
        Command::new(&self.config.solver_path)
            .arg("--version")
            .output()
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependent_types::*;
    use crate::type_registry::*;
    use crate::AstArena;

    #[test]
    fn test_smt_solver_creation() {
        let solver = SmtSolver::new();
        assert_eq!(solver.config.solver_path, "z3");
    }

    #[test]
    fn test_simple_predicate_translation() {
        let mut solver = SmtSolver::new();

        // Create a simple predicate: x == 5
        let predicate = RefinementPredicate::Comparison {
            op: crate::dependent_types::ComparisonOp::Equal,
            left: Box::new(RefinementExpr::Variable(AstArena::new().intern_string("x"))),
            right: Box::new(RefinementExpr::Constant(ConstValue::Int(5))),
        };

        let context = HashMap::new();
        let value_type = Type::Primitive(PrimitiveType::I32);

        let result = solver.translate_predicate(&predicate, &value_type, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_smt_script_generation() {
        let solver = SmtSolver::new();

        let formula = SmtExpr::Application {
            function: "=".to_string(),
            args: vec![
                SmtExpr::Variable("x".to_string()),
                SmtExpr::Constant(SmtConstant::Int(5)),
            ],
        };

        let script = solver.generate_smt_script(&formula);
        assert!(script.is_ok());

        let script_text = script.unwrap();
        assert!(script_text.contains("(set-logic LIA)"));
        assert!(script_text.contains("(assert (= x 5))"));
        assert!(script_text.contains("(check-sat)"));
    }
}
