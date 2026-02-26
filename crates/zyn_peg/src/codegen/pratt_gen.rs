//! Pratt Parser Generator for ZynPEG 2.0
//!
//! Generates a Pratt parser for expression precedence handling.
//! This is used for binary and unary operators with proper precedence and associativity.
//!
//! # Precedence Climbing
//!
//! Pratt parsing uses precedence climbing:
//! 1. Parse primary expression (literals, identifiers, grouping)
//! 2. While there's an operator with binding power >= min:
//!    - Parse the operator
//!    - Recursively parse right side with appropriate min binding power
//!    - Combine into binary expression

/// Operator precedence and associativity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorInfo {
    /// Left binding power
    pub lbp: u8,
    /// Right binding power (for associativity)
    pub rbp: u8,
}

impl OperatorInfo {
    /// Create left-associative operator
    pub fn left(precedence: u8) -> Self {
        OperatorInfo {
            lbp: precedence,
            rbp: precedence + 1,
        }
    }

    /// Create right-associative operator
    pub fn right(precedence: u8) -> Self {
        OperatorInfo {
            lbp: precedence,
            rbp: precedence,
        }
    }
}

/// Configuration for Pratt parser generation
#[derive(Debug, Clone, Default)]
pub struct PrattConfig {
    /// Binary operators with their info
    pub binary_ops: Vec<(String, OperatorInfo)>,
    /// Unary prefix operators
    pub prefix_ops: Vec<(String, u8)>, // (operator, precedence)
    /// Unary postfix operators
    pub postfix_ops: Vec<(String, u8)>,
    /// Name of the primary expression rule
    pub primary_rule: String,
}

/// Code generator for Pratt parser
pub struct PrattGenerator {
    code: String,
    indent: usize,
}

impl PrattGenerator {
    pub fn new() -> Self {
        PrattGenerator {
            code: String::new(),
            indent: 0,
        }
    }

    /// Generate Pratt parser code from configuration
    pub fn generate(&mut self, config: &PrattConfig) -> String {
        self.code.clear();

        self.generate_expr_method(config);
        self.generate_operator_info(config);

        self.code.clone()
    }

    fn generate_expr_method(&mut self, config: &PrattConfig) {
        self.line("/// Parse expression using Pratt algorithm");
        self.line("pub fn parse_expr(&mut self) -> ParseResult<TypedExpression> {");
        self.indent += 1;
        self.line("self.parse_expr_bp(0)");
        self.indent -= 1;
        self.line("}");
        self.line("");

        // Main Pratt parsing method with min binding power
        self.line("/// Parse expression with minimum binding power");
        self.line("fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<TypedExpression> {");
        self.indent += 1;

        // Parse prefix operators or primary
        self.line("self.state.skip_ws();");
        self.line("");
        self.line("// Parse prefix operator or primary expression");
        self.line("let mut left = if let Some(op_info) = self.get_prefix_op() {");
        self.indent += 1;
        self.line("let op = self.consume_operator();");
        self.line("self.state.skip_ws();");
        self.line("let operand = self.parse_expr_bp(op_info)?;");
        self.line("self.make_unary(op, operand)");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line(&format!("match self.parse_{}() {{", config.primary_rule));
        self.indent += 1;
        self.line("ParseResult::Success(v, _) => v,");
        self.line("ParseResult::Failure(e) => return ParseResult::Failure(e),");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("};");
        self.line("");

        // Main precedence climbing loop
        self.line("loop {");
        self.indent += 1;
        self.line("self.state.skip_ws();");
        self.line("");

        // Check for postfix operator
        self.line("// Check for postfix operator");
        self.line("if let Some(postfix_bp) = self.get_postfix_op() {");
        self.indent += 1;
        self.line("if postfix_bp < min_bp {");
        self.indent += 1;
        self.line("break;");
        self.indent -= 1;
        self.line("}");
        self.line("let op = self.consume_operator();");
        self.line("left = self.make_postfix(left, op);");
        self.line("continue;");
        self.indent -= 1;
        self.line("}");
        self.line("");

        // Check for binary operator
        self.line("// Check for binary operator");
        self.line("let Some((lbp, rbp)) = self.get_binary_op() else {");
        self.indent += 1;
        self.line("break;");
        self.indent -= 1;
        self.line("};");
        self.line("");
        self.line("if lbp < min_bp {");
        self.indent += 1;
        self.line("break;");
        self.indent -= 1;
        self.line("}");
        self.line("");
        self.line("let op = self.consume_operator();");
        self.line("self.state.skip_ws();");
        self.line("");
        self.line("let right = match self.parse_expr_bp(rbp) {");
        self.indent += 1;
        self.line("ParseResult::Success(v, _) => v,");
        self.line("ParseResult::Failure(e) => return ParseResult::Failure(e),");
        self.indent -= 1;
        self.line("};");
        self.line("");
        self.line("left = self.make_binary(left, op, right);");

        self.indent -= 1;
        self.line("}");
        self.line("");
        self.line("ParseResult::Success(left, self.state.pos())");

        self.indent -= 1;
        self.line("}");
        self.line("");
    }

    fn generate_operator_info(&mut self, config: &PrattConfig) {
        // Generate prefix operator lookup
        self.line("/// Check for prefix operator at current position");
        self.line("fn get_prefix_op(&self) -> Option<u8> {");
        self.indent += 1;
        if config.prefix_ops.is_empty() {
            self.line("None");
        } else {
            for (op, prec) in &config.prefix_ops {
                self.line(&format!(
                    "if self.state.check({:?}) {{ return Some({}); }}",
                    op, prec
                ));
            }
            self.line("None");
        }
        self.indent -= 1;
        self.line("}");
        self.line("");

        // Generate postfix operator lookup
        self.line("/// Check for postfix operator at current position");
        self.line("fn get_postfix_op(&self) -> Option<u8> {");
        self.indent += 1;
        if config.postfix_ops.is_empty() {
            self.line("None");
        } else {
            for (op, prec) in &config.postfix_ops {
                self.line(&format!(
                    "if self.state.check({:?}) {{ return Some({}); }}",
                    op, prec
                ));
            }
            self.line("None");
        }
        self.indent -= 1;
        self.line("}");
        self.line("");

        // Generate binary operator lookup
        self.line("/// Check for binary operator at current position, return (lbp, rbp)");
        self.line("fn get_binary_op(&self) -> Option<(u8, u8)> {");
        self.indent += 1;
        if config.binary_ops.is_empty() {
            self.line("None");
        } else {
            // Sort by operator length (longest first) to handle overlapping operators
            let mut sorted_ops = config.binary_ops.clone();
            sorted_ops.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

            for (op, info) in &sorted_ops {
                self.line(&format!(
                    "if self.state.check({:?}) {{ return Some(({}, {})); }}",
                    op, info.lbp, info.rbp
                ));
            }
            self.line("None");
        }
        self.indent -= 1;
        self.line("}");
        self.line("");

        // Generate operator consumption
        self.line("/// Consume the current operator and return it");
        self.line("fn consume_operator(&mut self) -> String {");
        self.indent += 1;
        self.line("let start = self.state.pos();");

        // Try all operators (longest first)
        let mut all_ops: Vec<&str> = config
            .binary_ops
            .iter()
            .map(|(s, _)| s.as_str())
            .chain(config.prefix_ops.iter().map(|(s, _)| s.as_str()))
            .chain(config.postfix_ops.iter().map(|(s, _)| s.as_str()))
            .collect();
        all_ops.sort_by(|a, b| b.len().cmp(&a.len()));
        all_ops.dedup();

        for op in &all_ops {
            self.line(&format!("if self.state.check({:?}) {{", op));
            self.indent += 1;
            for _ in op.chars() {
                self.line("self.state.advance();");
            }
            self.line(&format!("return {:?}.to_string();", op));
            self.indent -= 1;
            self.line("}");
        }

        self.line("// Fallback: consume one char");
        self.line("if let Some(c) = self.state.advance() {");
        self.indent += 1;
        self.line("return c.to_string();");
        self.indent -= 1;
        self.line("}");
        self.line("String::new()");

        self.indent -= 1;
        self.line("}");
        self.line("");

        // Generate AST construction helpers
        self.generate_ast_helpers();
    }

    fn generate_ast_helpers(&mut self) {
        self.line("/// Create binary expression node");
        self.line("fn make_binary(&mut self, left: TypedExpression, op: String, right: TypedExpression) -> TypedExpression {");
        self.indent += 1;
        self.line("let binary_op = match op.as_str() {");
        self.indent += 1;
        self.line("\"+\" => BinaryOp::Add,");
        self.line("\"-\" => BinaryOp::Sub,");
        self.line("\"*\" => BinaryOp::Mul,");
        self.line("\"/\" => BinaryOp::Div,");
        self.line("\"%\" => BinaryOp::Mod,");
        self.line("\"==\" => BinaryOp::Eq,");
        self.line("\"!=\" => BinaryOp::Ne,");
        self.line("\"<\" => BinaryOp::Lt,");
        self.line("\">\" => BinaryOp::Gt,");
        self.line("\"<=\" => BinaryOp::Le,");
        self.line("\">=\" => BinaryOp::Ge,");
        self.line("\"&&\" => BinaryOp::And,");
        self.line("\"||\" => BinaryOp::Or,");
        self.line("\"&\" => BinaryOp::BitAnd,");
        self.line("\"|\" => BinaryOp::BitOr,");
        self.line("\"^\" => BinaryOp::BitXor,");
        self.line("\"<<\" => BinaryOp::Shl,");
        self.line("\">>\" => BinaryOp::Shr,");
        self.line("\"=\" => BinaryOp::Assign,");
        self.line("_ => BinaryOp::Add, // fallback");
        self.indent -= 1;
        self.line("};");
        self.line("");
        self.line("TypedExpression::Binary(TypedBinary {");
        self.indent += 1;
        self.line(
            "left: Box::new(TypedNode { node: left, ty: Type::Unknown, span: Span::new(0, 0) }),",
        );
        self.line("op: binary_op,");
        self.line(
            "right: Box::new(TypedNode { node: right, ty: Type::Unknown, span: Span::new(0, 0) }),",
        );
        self.indent -= 1;
        self.line("})");
        self.indent -= 1;
        self.line("}");
        self.line("");

        self.line("/// Create unary expression node");
        self.line(
            "fn make_unary(&mut self, op: String, operand: TypedExpression) -> TypedExpression {",
        );
        self.indent += 1;
        self.line("let unary_op = match op.as_str() {");
        self.indent += 1;
        self.line("\"-\" => UnaryOp::Neg,");
        self.line("\"!\" => UnaryOp::Not,");
        self.line("\"~\" => UnaryOp::BitNot,");
        self.line("\"*\" => UnaryOp::Deref,");
        self.line("\"&\" => UnaryOp::Ref,");
        self.line("_ => UnaryOp::Neg, // fallback");
        self.indent -= 1;
        self.line("};");
        self.line("");
        self.line("TypedExpression::Unary(TypedUnary {");
        self.indent += 1;
        self.line("op: unary_op,");
        self.line("operand: Box::new(TypedNode { node: operand, ty: Type::Unknown, span: Span::new(0, 0) }),");
        self.indent -= 1;
        self.line("})");
        self.indent -= 1;
        self.line("}");
        self.line("");

        self.line("/// Create postfix expression node");
        self.line(
            "fn make_postfix(&mut self, operand: TypedExpression, op: String) -> TypedExpression {",
        );
        self.indent += 1;
        self.line("// TODO: Handle postfix operators like ++, --, etc.");
        self.line("operand");
        self.indent -= 1;
        self.line("}");
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.code.push_str("    ");
        }
        self.code.push_str(text);
        self.code.push('\n');
    }
}

impl Default for PrattGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a standard expression precedence configuration
pub fn standard_expr_config() -> PrattConfig {
    PrattConfig {
        binary_ops: vec![
            // Assignment (lowest precedence, right-associative)
            ("=".to_string(), OperatorInfo::right(10)),
            // Logical OR
            ("||".to_string(), OperatorInfo::left(20)),
            // Logical AND
            ("&&".to_string(), OperatorInfo::left(30)),
            // Bitwise OR
            ("|".to_string(), OperatorInfo::left(40)),
            // Bitwise XOR
            ("^".to_string(), OperatorInfo::left(50)),
            // Bitwise AND
            ("&".to_string(), OperatorInfo::left(60)),
            // Equality
            ("==".to_string(), OperatorInfo::left(70)),
            ("!=".to_string(), OperatorInfo::left(70)),
            // Comparison
            ("<".to_string(), OperatorInfo::left(80)),
            (">".to_string(), OperatorInfo::left(80)),
            ("<=".to_string(), OperatorInfo::left(80)),
            (">=".to_string(), OperatorInfo::left(80)),
            // Shift
            ("<<".to_string(), OperatorInfo::left(90)),
            (">>".to_string(), OperatorInfo::left(90)),
            // Additive
            ("+".to_string(), OperatorInfo::left(100)),
            ("-".to_string(), OperatorInfo::left(100)),
            // Multiplicative
            ("*".to_string(), OperatorInfo::left(110)),
            ("/".to_string(), OperatorInfo::left(110)),
            ("%".to_string(), OperatorInfo::left(110)),
        ],
        prefix_ops: vec![
            ("-".to_string(), 120),
            ("!".to_string(), 120),
            ("~".to_string(), 120),
            ("*".to_string(), 120), // dereference
            ("&".to_string(), 120), // reference
        ],
        postfix_ops: vec![
            // ("++".to_string(), 130),
            // ("--".to_string(), 130),
        ],
        primary_rule: "primary".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_info() {
        let left = OperatorInfo::left(100);
        assert_eq!(left.lbp, 100);
        assert_eq!(left.rbp, 101);

        let right = OperatorInfo::right(100);
        assert_eq!(right.lbp, 100);
        assert_eq!(right.rbp, 100);
    }

    #[test]
    fn test_generate_pratt_parser() {
        let config = standard_expr_config();
        let mut gen = PrattGenerator::new();
        let code = gen.generate(&config);

        assert!(code.contains("fn parse_expr"));
        assert!(code.contains("fn parse_expr_bp"));
        assert!(code.contains("fn get_binary_op"));
        assert!(code.contains("fn make_binary"));
    }

    #[test]
    fn test_standard_config_precedence() {
        let config = standard_expr_config();

        // Find + and * operators
        let plus = config.binary_ops.iter().find(|(op, _)| op == "+").unwrap();
        let mult = config.binary_ops.iter().find(|(op, _)| op == "*").unwrap();

        // * should have higher precedence than +
        assert!(mult.1.lbp > plus.1.lbp);
    }
}
