//! Const Expression Evaluator
//!
//! Evaluates compile-time constant expressions for const generics and dependent types.
//! Supports Rust-style const generics, C++ template value parameters, and value-dependent types.

use crate::arena::{AstArena, InternedString};
use crate::source::Span;
use crate::type_registry::{ConstBinaryOp, ConstUnaryOp, ConstValue, Type, TypeId};
use crate::PrimitiveType;
use std::collections::HashMap;

/// Const evaluation context
#[derive(Debug, Clone)]
pub struct ConstEvalContext {
    /// Const variable values
    pub const_vars: HashMap<InternedString, ConstValue>,
    /// Const function definitions
    pub const_functions: HashMap<InternedString, ConstFunction>,
    /// Type size/alignment information
    pub type_context: HashMap<TypeId, TypeLayout>,
}

/// Type layout information
#[derive(Debug, Clone)]
pub struct TypeLayout {
    pub size: usize,
    pub alignment: usize,
    pub is_packed: bool,
}

/// Const function definition
#[derive(Debug, Clone)]
pub struct ConstFunction {
    pub name: InternedString,
    pub params: Vec<(InternedString, Type)>,
    pub return_type: Type,
    pub body: ConstExpr,
}

/// Const expression for evaluation
#[derive(Debug, Clone)]
pub struct ConstExpr {
    pub kind: ConstExprKind,
    pub ty: Type,
    pub span: Span,
}

/// Const expression kinds
#[derive(Debug, Clone)]
pub enum ConstExprKind {
    /// Literal value
    Literal(Literal),
    /// Variable reference
    Variable(InternedString),
    /// Binary operation
    BinaryOp {
        op: ConstBinaryOp,
        left: Box<ConstExpr>,
        right: Box<ConstExpr>,
    },
    /// Unary operation
    UnaryOp {
        op: ConstUnaryOp,
        operand: Box<ConstExpr>,
    },
    /// Type cast
    Cast {
        expr: Box<ConstExpr>,
        target_ty: Box<PrimitiveType>,
    },
    /// Function call
    Call {
        func: InternedString,
        args: Vec<ConstExpr>,
    },
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    UInteger(u64),
    Float(f64),
    Boolean(bool),
    String(InternedString),
    Char(char),
    Null,
    Unit,
    Placeholder,
}

// Use the ConstBinaryOp and ConstUnaryOp from type_registry

/// Const constraint for generic parameters
#[derive(Debug, Clone, PartialEq)]
pub enum ConstConstraint {
    /// Value must equal specific value
    Equal(ConstValue),
    /// Value must be within range
    Range { min: ConstValue, max: ConstValue },
    /// Custom predicate
    Predicate(InternedString),
    /// Conjunction of constraints
    And(Vec<ConstConstraint>),
    /// Disjunction of constraints
    Or(Vec<ConstConstraint>),
}

/// Const expression evaluator
pub struct ConstEvaluator {
    /// Evaluation context with const variables and functions
    pub context: ConstEvalContext,
    /// String interner for creating const variable names
    pub arena: AstArena,
    /// Maximum evaluation depth to prevent infinite recursion
    pub max_depth: usize,
    /// Current evaluation depth
    current_depth: usize,
}

/// Const evaluation error
#[derive(Debug, Clone, PartialEq)]
pub enum ConstEvalError {
    /// Undefined const variable
    UndefinedVariable(InternedString),
    /// Undefined const function
    UndefinedFunction(InternedString),
    /// Type mismatch in const operation
    TypeMismatch {
        expected: Type,
        found: Type,
        operation: String,
    },
    /// Division by zero
    DivisionByZero,
    /// Integer overflow
    IntegerOverflow,
    /// Invalid cast
    InvalidCast {
        from: Type,
        to: Type,
        value: ConstValue,
    },
    /// Recursion limit exceeded
    RecursionLimitExceeded,
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Runtime-only operation in const context
    RuntimeOperation(String),
}

/// Const evaluation result
pub type ConstEvalResult = Result<ConstValue, ConstEvalError>;

impl ConstEvaluator {
    pub fn new() -> Self {
        Self {
            context: ConstEvalContext {
                const_vars: HashMap::new(),
                const_functions: HashMap::new(),
                type_context: HashMap::new(),
            },
            arena: AstArena::new(),
            max_depth: 256,
            current_depth: 0,
        }
    }

    /// Create evaluator with pre-populated context
    pub fn with_context(context: ConstEvalContext) -> Self {
        Self {
            context,
            arena: AstArena::new(),
            max_depth: 256,
            current_depth: 0,
        }
    }

    /// Create a new const variable name
    pub fn create_const_var(&mut self, name: &str) -> InternedString {
        self.arena.intern_string(name)
    }

    /// Evaluate a const value (may need further evaluation)
    pub fn eval_const_value(&mut self, value: &ConstValue) -> ConstEvalResult {
        if self.current_depth >= self.max_depth {
            return Err(ConstEvalError::RecursionLimitExceeded);
        }

        self.current_depth += 1;
        let result = self.eval_const_value_impl(value);
        self.current_depth -= 1;
        result
    }

    fn eval_const_value_impl(&mut self, value: &ConstValue) -> ConstEvalResult {
        match value {
            // Already evaluated literals
            ConstValue::Int(i) => Ok(ConstValue::Int(*i)),
            ConstValue::UInt(u) => Ok(ConstValue::UInt(*u)),
            ConstValue::Bool(b) => Ok(ConstValue::Bool(*b)),
            ConstValue::String(s) => Ok(ConstValue::String(*s)),
            ConstValue::Char(c) => Ok(ConstValue::Char(*c)),

            // Variable lookup
            ConstValue::Variable(name) => {
                // Look up variable by name in context
                if let Some(value) = self.context.const_vars.get(name) {
                    Ok(value.clone())
                } else {
                    Err(ConstEvalError::UndefinedVariable(*name))
                }
            }

            // Binary operations
            ConstValue::BinaryOp { op, left, right } => {
                let left_val = self.eval_const_value(left)?;
                let right_val = self.eval_const_value(right)?;
                self.eval_binary_op(*op, &left_val, &right_val)
            }

            // Unary operations
            ConstValue::UnaryOp { op, operand } => {
                let operand_val = self.eval_const_value(operand)?;
                self.eval_unary_op(*op, &operand_val)
            }

            // Function calls
            ConstValue::FunctionCall { name, args } => self.eval_const_function(*name, args),

            // Arrays, tuples, structs - just return as-is for now
            ConstValue::Array(_) => Ok(value.clone()),
            ConstValue::Tuple(_) => Ok(value.clone()),
            ConstValue::Struct(_) => Ok(value.clone()),
        }
    }

    /// Evaluate a const expression
    pub fn eval_const_expr(&mut self, expr: &ConstExpr) -> ConstEvalResult {
        match &expr.kind {
            ConstExprKind::Literal(lit) => self.eval_literal(lit),

            ConstExprKind::Variable(name) => {
                // Look up variable by name
                if let Some(value) = self.context.const_vars.get(name) {
                    Ok(value.clone())
                } else {
                    Err(ConstEvalError::UndefinedVariable(*name))
                }
            }

            ConstExprKind::BinaryOp { op, left, right } => {
                let left_val = self.eval_const_expr(left)?;
                let right_val = self.eval_const_expr(right)?;
                self.eval_binary_op(*op, &left_val, &right_val)
            }

            ConstExprKind::UnaryOp { op, operand } => {
                let operand_val = self.eval_const_expr(operand)?;
                self.eval_unary_op(*op, &operand_val)
            }

            ConstExprKind::Cast { expr, target_ty } => {
                let value = self.eval_const_expr(expr)?;
                let target_type = Type::Primitive(**target_ty);
                self.eval_const_cast(&value, &target_type)
            }

            ConstExprKind::Call { func, args } => {
                let arg_vals: Result<Vec<_>, _> =
                    args.iter().map(|arg| self.eval_const_expr(arg)).collect();
                let arg_vals = arg_vals?;
                self.eval_const_function(*func, &arg_vals)
            }
        }
    }

    /// Evaluate sizeof for a type
    fn eval_sizeof(&self, ty: &Type) -> ConstEvalResult {
        match ty {
            Type::Primitive(prim) => {
                let size = match prim {
                    PrimitiveType::I8 | PrimitiveType::U8 => 1,
                    PrimitiveType::I16 | PrimitiveType::U16 => 2,
                    PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
                    PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 8,
                    PrimitiveType::ISize | PrimitiveType::USize => 8, // Assuming 64-bit
                    PrimitiveType::Bool => 1,
                    PrimitiveType::Char => 4,    // Unicode scalar value
                    PrimitiveType::String => 24, // Fat pointer (ptr + len + cap)
                    _ => {
                        return Err(ConstEvalError::UnsupportedOperation(
                            "sizeof for this type".to_string(),
                        ))
                    }
                };
                Ok(ConstValue::UInt(size))
            }

            Type::Named { id, .. } => {
                if let Some(layout) = self.context.type_context.get(id) {
                    Ok(ConstValue::UInt(layout.size as u64))
                } else {
                    Err(ConstEvalError::UnsupportedOperation(
                        "sizeof for unknown type".to_string(),
                    ))
                }
            }

            Type::Array {
                element_type, size, ..
            } => {
                let elem_size = self.eval_sizeof(element_type)?;
                if let Some(array_size) = size {
                    let array_len = match array_size {
                        ConstValue::Int(len) if *len >= 0 => *len as u64,
                        ConstValue::UInt(len) => *len,
                        _ => {
                            return Err(ConstEvalError::TypeMismatch {
                                expected: Type::Primitive(PrimitiveType::USize),
                                found: Type::Error,
                                operation: "array size".to_string(),
                            })
                        }
                    };

                    if let ConstValue::UInt(elem_size_val) = elem_size {
                        Ok(ConstValue::UInt(elem_size_val * array_len))
                    } else {
                        Err(ConstEvalError::TypeMismatch {
                            expected: Type::Primitive(PrimitiveType::USize),
                            found: Type::Error,
                            operation: "element size".to_string(),
                        })
                    }
                } else {
                    // Unsized array - return size of pointer
                    Ok(ConstValue::UInt(8)) // Assuming 64-bit
                }
            }

            _ => Err(ConstEvalError::UnsupportedOperation(format!(
                "sizeof for {:?}",
                ty
            ))),
        }
    }

    /// Evaluate alignof for a type
    fn eval_alignof(&self, ty: &Type) -> ConstEvalResult {
        match ty {
            Type::Primitive(prim) => {
                let align = match prim {
                    PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool => 1,
                    PrimitiveType::I16 | PrimitiveType::U16 => 2,
                    PrimitiveType::I32
                    | PrimitiveType::U32
                    | PrimitiveType::F32
                    | PrimitiveType::Char => 4,
                    PrimitiveType::I64
                    | PrimitiveType::U64
                    | PrimitiveType::F64
                    | PrimitiveType::ISize
                    | PrimitiveType::USize => 8,
                    PrimitiveType::String => 8, // Pointer alignment
                    _ => {
                        return Err(ConstEvalError::UnsupportedOperation(
                            "alignof for this type".to_string(),
                        ))
                    }
                };
                Ok(ConstValue::UInt(align))
            }

            Type::Named { id, .. } => {
                if let Some(layout) = self.context.type_context.get(id) {
                    Ok(ConstValue::UInt(layout.alignment as u64))
                } else {
                    Err(ConstEvalError::UnsupportedOperation(
                        "alignof for unknown type".to_string(),
                    ))
                }
            }

            _ => Err(ConstEvalError::UnsupportedOperation(format!(
                "alignof for {:?}",
                ty
            ))),
        }
    }

    /// Evaluate binary operations
    fn eval_binary_op(
        &self,
        op: ConstBinaryOp,
        left: &ConstValue,
        right: &ConstValue,
    ) -> ConstEvalResult {
        match (left, right) {
            // Integer arithmetic
            (ConstValue::Int(l), ConstValue::Int(r)) => match op {
                ConstBinaryOp::Add => l
                    .checked_add(*r)
                    .map(ConstValue::Int)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstBinaryOp::Sub => l
                    .checked_sub(*r)
                    .map(ConstValue::Int)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstBinaryOp::Mul => l
                    .checked_mul(*r)
                    .map(ConstValue::Int)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstBinaryOp::Div => {
                    if *r == 0 {
                        Err(ConstEvalError::DivisionByZero)
                    } else {
                        l.checked_div(*r)
                            .map(ConstValue::Int)
                            .ok_or(ConstEvalError::IntegerOverflow)
                    }
                }
                ConstBinaryOp::Mod => {
                    if *r == 0 {
                        Err(ConstEvalError::DivisionByZero)
                    } else {
                        Ok(ConstValue::Int(l % r))
                    }
                }
                ConstBinaryOp::And => Ok(ConstValue::Int(l & r)),
                ConstBinaryOp::Or => Ok(ConstValue::Int(l | r)),
                ConstBinaryOp::Xor => Ok(ConstValue::Int(l ^ r)),
                ConstBinaryOp::Shl => Ok(ConstValue::Int(l << r)),
                ConstBinaryOp::Shr => Ok(ConstValue::Int(l >> r)),
                ConstBinaryOp::Eq => Ok(ConstValue::Bool(l == r)),
                ConstBinaryOp::Ne => Ok(ConstValue::Bool(l != r)),
                ConstBinaryOp::Lt => Ok(ConstValue::Bool(l < r)),
                ConstBinaryOp::Le => Ok(ConstValue::Bool(l <= r)),
                ConstBinaryOp::Gt => Ok(ConstValue::Bool(l > r)),
                ConstBinaryOp::Ge => Ok(ConstValue::Bool(l >= r)),
                _ => Err(ConstEvalError::UnsupportedOperation(format!(
                    "integer {:?}",
                    op
                ))),
            },

            // Unsigned integer arithmetic
            (ConstValue::UInt(l), ConstValue::UInt(r)) => match op {
                ConstBinaryOp::Add => l
                    .checked_add(*r)
                    .map(ConstValue::UInt)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstBinaryOp::Sub => l
                    .checked_sub(*r)
                    .map(ConstValue::UInt)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstBinaryOp::Mul => l
                    .checked_mul(*r)
                    .map(ConstValue::UInt)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstBinaryOp::Div => {
                    if *r == 0 {
                        Err(ConstEvalError::DivisionByZero)
                    } else {
                        Ok(ConstValue::UInt(l / r))
                    }
                }
                ConstBinaryOp::Mod => {
                    if *r == 0 {
                        Err(ConstEvalError::DivisionByZero)
                    } else {
                        Ok(ConstValue::UInt(l % r))
                    }
                }
                ConstBinaryOp::And => Ok(ConstValue::UInt(l & r)),
                ConstBinaryOp::Or => Ok(ConstValue::UInt(l | r)),
                ConstBinaryOp::Xor => Ok(ConstValue::UInt(l ^ r)),
                ConstBinaryOp::Shl => Ok(ConstValue::UInt(l << r)),
                ConstBinaryOp::Shr => Ok(ConstValue::UInt(l >> r)),
                ConstBinaryOp::Eq => Ok(ConstValue::Bool(l == r)),
                ConstBinaryOp::Ne => Ok(ConstValue::Bool(l != r)),
                ConstBinaryOp::Lt => Ok(ConstValue::Bool(l < r)),
                ConstBinaryOp::Le => Ok(ConstValue::Bool(l <= r)),
                ConstBinaryOp::Gt => Ok(ConstValue::Bool(l > r)),
                ConstBinaryOp::Ge => Ok(ConstValue::Bool(l >= r)),
                _ => Err(ConstEvalError::UnsupportedOperation(format!(
                    "unsigned integer {:?}",
                    op
                ))),
            },

            // Boolean operations
            (ConstValue::Bool(l), ConstValue::Bool(r)) => match op {
                ConstBinaryOp::LogicalAnd => Ok(ConstValue::Bool(*l && *r)),
                ConstBinaryOp::LogicalOr => Ok(ConstValue::Bool(*l || *r)),
                ConstBinaryOp::Eq => Ok(ConstValue::Bool(l == r)),
                ConstBinaryOp::Ne => Ok(ConstValue::Bool(l != r)),
                _ => Err(ConstEvalError::UnsupportedOperation(format!(
                    "boolean {:?}",
                    op
                ))),
            },

            _ => Err(ConstEvalError::TypeMismatch {
                expected: Type::Error,
                found: Type::Error,
                operation: format!("{:?}", op),
            }),
        }
    }

    /// Evaluate unary operations
    fn eval_unary_op(&self, op: ConstUnaryOp, operand: &ConstValue) -> ConstEvalResult {
        match operand {
            ConstValue::Int(val) => match op {
                ConstUnaryOp::Neg => val
                    .checked_neg()
                    .map(ConstValue::Int)
                    .ok_or(ConstEvalError::IntegerOverflow),
                ConstUnaryOp::Not => Ok(ConstValue::Int(!val)),
                _ => Err(ConstEvalError::TypeMismatch {
                    expected: Type::Primitive(PrimitiveType::Bool),
                    found: Type::Primitive(PrimitiveType::I64),
                    operation: "logical not".to_string(),
                }),
            },

            ConstValue::UInt(val) => match op {
                ConstUnaryOp::Not => Ok(ConstValue::UInt(!val)),
                _ => Err(ConstEvalError::UnsupportedOperation(format!(
                    "unsigned {:?}",
                    op
                ))),
            },

            ConstValue::Bool(val) => match op {
                ConstUnaryOp::Not => Ok(ConstValue::Bool(!val)),
                _ => Err(ConstEvalError::TypeMismatch {
                    expected: Type::Primitive(PrimitiveType::I64),
                    found: Type::Primitive(PrimitiveType::Bool),
                    operation: format!("{:?}", op),
                }),
            },

            _ => Err(ConstEvalError::UnsupportedOperation(format!(
                "{:?} on {:?}",
                op, operand
            ))),
        }
    }

    /// Evaluate const cast
    fn eval_const_cast(&self, value: &ConstValue, target_ty: &Type) -> ConstEvalResult {
        match (value, target_ty) {
            // Integer to integer casts
            (ConstValue::Int(val), Type::Primitive(PrimitiveType::I32)) => {
                if *val >= i32::MIN as i64 && *val <= i32::MAX as i64 {
                    Ok(ConstValue::Int(*val))
                } else {
                    Err(ConstEvalError::IntegerOverflow)
                }
            }

            (ConstValue::Int(val), Type::Primitive(PrimitiveType::U32)) => {
                if *val >= 0 && *val <= u32::MAX as i64 {
                    Ok(ConstValue::UInt(*val as u64))
                } else {
                    Err(ConstEvalError::IntegerOverflow)
                }
            }

            // More cast implementations would go here...
            _ => Err(ConstEvalError::InvalidCast {
                from: Type::Error, // Would need to track source type
                to: target_ty.clone(),
                value: value.clone(),
            }),
        }
    }

    /// Evaluate const function call
    fn eval_const_function(
        &mut self,
        name: InternedString,
        args: &[ConstValue],
    ) -> ConstEvalResult {
        if let Some(func) = self.context.const_functions.get(&name).cloned() {
            if args.len() != func.params.len() {
                return Err(ConstEvalError::TypeMismatch {
                    expected: Type::Error,
                    found: Type::Error,
                    operation: format!("function {} argument count", name),
                });
            }

            // Create new scope with function parameters
            let old_vars = self.context.const_vars.clone();

            // Bind parameters
            for (i, (param_name, _param_ty)) in func.params.iter().enumerate() {
                self.context.const_vars.insert(*param_name, args[i].clone());
            }

            // Evaluate function body
            let result = self.eval_const_expr(&func.body);

            // Restore old scope
            self.context.const_vars = old_vars;

            result
        } else {
            Err(ConstEvalError::UndefinedFunction(name))
        }
    }

    /// Evaluate literal
    fn eval_literal(&self, lit: &Literal) -> ConstEvalResult {
        match lit {
            Literal::Integer(val) => Ok(ConstValue::Int(*val)),
            Literal::UInteger(val) => Ok(ConstValue::UInt(*val)),
            Literal::Float(_) => Err(ConstEvalError::UnsupportedOperation(
                "float literals in const context".to_string(),
            )),
            Literal::Boolean(val) => Ok(ConstValue::Bool(*val)),
            Literal::String(val) => Ok(ConstValue::String(*val)),
            Literal::Char(val) => Ok(ConstValue::Char(*val)),
            Literal::Null => Err(ConstEvalError::UnsupportedOperation(
                "null in const context".to_string(),
            )),
            Literal::Unit => Err(ConstEvalError::UnsupportedOperation(
                "unit in const context".to_string(),
            )),
            Literal::Placeholder => Err(ConstEvalError::UnsupportedOperation(
                "placeholder literal".to_string(),
            )),
        }
    }

    /// Check if a const constraint is satisfied
    pub fn check_constraint(
        &mut self,
        value: &ConstValue,
        constraint: &ConstConstraint,
    ) -> Result<bool, ConstEvalError> {
        match constraint {
            ConstConstraint::Equal(expected) => {
                let expected_val = self.eval_const_value(expected)?;
                Ok(value == &expected_val)
            }

            ConstConstraint::Range { min, max } => {
                let min_val = self.eval_const_value(min)?;
                let max_val = self.eval_const_value(max)?;

                match (value, &min_val, &max_val) {
                    (ConstValue::Int(v), ConstValue::Int(min), ConstValue::Int(max)) => {
                        Ok(*v >= *min && *v <= *max)
                    }
                    (ConstValue::UInt(v), ConstValue::UInt(min), ConstValue::UInt(max)) => {
                        Ok(*v >= *min && *v <= *max)
                    }
                    _ => Err(ConstEvalError::TypeMismatch {
                        expected: Type::Error,
                        found: Type::Error,
                        operation: "range constraint".to_string(),
                    }),
                }
            }

            ConstConstraint::Predicate(pred) => {
                // Would call predicate function with arguments
                Err(ConstEvalError::UnsupportedOperation(
                    "predicate constraints".to_string(),
                ))
            }

            ConstConstraint::And(constraints) => {
                for constraint in constraints {
                    if !self.check_constraint(value, constraint)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            ConstConstraint::Or(constraints) => {
                for constraint in constraints {
                    if self.check_constraint(value, constraint)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }
}

/// Helper function to create common const values
impl ConstValue {
    pub fn int(val: i64) -> Self {
        ConstValue::Int(val)
    }

    pub fn uint(val: u64) -> Self {
        ConstValue::UInt(val)
    }

    pub fn bool(val: bool) -> Self {
        ConstValue::Bool(val)
    }

    pub fn char(val: char) -> Self {
        ConstValue::Char(val)
    }

    pub fn string(val: InternedString) -> Self {
        ConstValue::String(val)
    }
}

/// Helper to create basic type layouts
impl TypeLayout {
    pub fn primitive(size: usize, alignment: usize) -> Self {
        Self {
            size,
            alignment,
            is_packed: false,
        }
    }

    pub fn packed(size: usize) -> Self {
        Self {
            size,
            alignment: 1,
            is_packed: true,
        }
    }
}
