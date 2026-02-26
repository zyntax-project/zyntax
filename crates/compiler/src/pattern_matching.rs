//! # Pattern Matching Compilation
//!
//! Implements efficient pattern matching compilation using decision trees.
//! Converts pattern matches into optimized control flow graphs.

use crate::hir::*;
use crate::{CompilerError, CompilerResult};
use std::collections::{HashMap, HashSet, VecDeque};

/// Decision tree node for pattern matching
#[derive(Debug, Clone)]
pub enum DecisionNode {
    /// Leaf node - pattern matched successfully
    Success {
        target: HirId,
        bindings: Vec<PatternBinding>,
    },
    /// Test a constant value
    ConstantTest {
        value: HirId,
        constant: HirConstant,
        success: Box<DecisionNode>,
        failure: Box<DecisionNode>,
    },
    /// Test union variant discriminant
    UnionTest {
        value: HirId,
        variant_index: u32,
        success: Box<DecisionNode>,
        failure: Box<DecisionNode>,
    },
    /// Test struct field patterns
    StructTest {
        value: HirId,
        field_index: u32,
        field_test: Box<DecisionNode>,
        next_field: Box<DecisionNode>,
    },
    /// Bind a variable
    Binding {
        name: zyntax_typed_ast::InternedString,
        value: HirId,
        ty: HirType,
        next: Box<DecisionNode>,
    },
    /// Guard test (boolean condition)
    Guard {
        condition: HirId,
        success: Box<DecisionNode>,
        failure: Box<DecisionNode>,
    },
    /// Failure node - no pattern matched
    Failure,
}

/// Variable binding from pattern matching
#[derive(Debug, Clone)]
pub struct PatternBinding {
    pub name: zyntax_typed_ast::InternedString,
    pub value: HirId,
    pub ty: HirType,
}

/// Pattern matching compiler
pub struct PatternMatchCompiler {
    next_block_id: u64,
}

impl PatternMatchCompiler {
    pub fn new() -> Self {
        Self { next_block_id: 0 }
    }

    /// Compile a pattern match into a decision tree
    pub fn compile_pattern_match(
        &mut self,
        scrutinee: HirId,
        patterns: &[HirPattern],
        default_target: Option<HirId>,
    ) -> CompilerResult<DecisionNode> {
        // Convert patterns into a decision tree
        let mut rows = Vec::new();

        // Create pattern rows for decision tree construction
        for pattern in patterns {
            let row = PatternRow {
                patterns: vec![pattern.clone()],
                target: pattern.target,
                bindings: pattern.bindings.clone(),
            };
            rows.push(row);
        }

        // Build decision tree
        let default_block = default_target.unwrap_or_else(|| {
            self.next_block_id += 1;
            HirId::new()
        });

        let tree = self.build_decision_tree(vec![scrutinee], rows, default_block)?;

        Ok(tree)
    }

    /// Build decision tree using pattern matrix algorithm
    fn build_decision_tree(
        &mut self,
        values: Vec<HirId>,
        rows: Vec<PatternRow>,
        default: HirId,
    ) -> CompilerResult<DecisionNode> {
        // Base case: no patterns left
        if rows.is_empty() {
            return Ok(DecisionNode::Failure);
        }

        // Base case: first row has no patterns left - success
        if rows[0].patterns.is_empty() {
            return Ok(DecisionNode::Success {
                target: rows[0].target,
                bindings: rows[0]
                    .bindings
                    .iter()
                    .map(|b| PatternBinding {
                        name: b.name,
                        value: b.value_id,
                        ty: b.ty.clone(),
                    })
                    .collect(),
            });
        }

        // Find the best column to split on
        let split_column = self.choose_split_column(&values, &rows)?;
        let split_value = values[split_column];

        // Group patterns by constructor
        let mut groups = HashMap::new();
        let mut wildcard_rows = Vec::new();

        // Find the first constructor in pattern order before moving rows
        let first_constructor_opt = rows.iter().find_map(|row| {
            if split_column < row.patterns.len() {
                match &row.patterns[split_column].kind {
                    HirPatternKind::Constant(c) => Some(Constructor::Constant(c.clone())),
                    HirPatternKind::UnionVariant { variant_index, .. } => {
                        Some(Constructor::UnionVariant(*variant_index))
                    }
                    HirPatternKind::Struct { .. } => Some(Constructor::Struct),
                    _ => None,
                }
            } else {
                None
            }
        });

        for row in rows {
            let pattern = &row.patterns[split_column];
            match &pattern.kind {
                HirPatternKind::Constant(c) => {
                    groups
                        .entry(Constructor::Constant(c.clone()))
                        .or_insert_with(Vec::new)
                        .push(row);
                }
                HirPatternKind::UnionVariant { variant_index, .. } => {
                    groups
                        .entry(Constructor::UnionVariant(*variant_index))
                        .or_insert_with(Vec::new)
                        .push(row);
                }
                HirPatternKind::Struct { .. } => {
                    groups
                        .entry(Constructor::Struct)
                        .or_insert_with(Vec::new)
                        .push(row);
                }
                HirPatternKind::Wildcard | HirPatternKind::Binding(_) => {
                    wildcard_rows.push(row);
                }
                HirPatternKind::Guard { .. } => {
                    // Guards are handled specially
                    wildcard_rows.push(row);
                }
            }
        }

        // Build subtrees for each constructor
        if groups.is_empty() {
            // Only wildcards - continue with wildcard expansion
            let specialized_rows = self.specialize_wildcard_rows(wildcard_rows, split_column)?;
            return self.build_decision_tree(values, specialized_rows, default);
        }

        // Use the first constructor found, or fallback to any available
        let first_constructor = first_constructor_opt
            .or_else(|| groups.keys().next().cloned())
            .ok_or_else(|| CompilerError::Analysis("No constructors found".into()))?;
        match first_constructor {
            Constructor::Constant(const_val) => {
                let const_rows = groups
                    .remove(&Constructor::Constant(const_val.clone()))
                    .unwrap();
                let specialized_const = self.specialize_constant_rows(const_rows, split_column)?;

                // Combine remaining constructors with wildcards for failure case
                let failure_rows = self.combine_remaining_rows(groups, wildcard_rows);

                let success_tree =
                    self.build_decision_tree(values.clone(), specialized_const, default)?;
                let failure_tree = self.build_decision_tree(values, failure_rows, default)?;

                Ok(DecisionNode::ConstantTest {
                    value: split_value,
                    constant: const_val,
                    success: Box::new(success_tree),
                    failure: Box::new(failure_tree),
                })
            }

            Constructor::UnionVariant(variant_idx) => {
                let variant_rows = groups
                    .remove(&Constructor::UnionVariant(variant_idx))
                    .unwrap();
                let specialized_variant = self.specialize_union_rows(variant_rows, split_column)?;

                let failure_rows = self.combine_remaining_rows(groups, wildcard_rows);

                let success_tree =
                    self.build_decision_tree(values.clone(), specialized_variant, default)?;
                let failure_tree = self.build_decision_tree(values, failure_rows, default)?;

                Ok(DecisionNode::UnionTest {
                    value: split_value,
                    variant_index: variant_idx,
                    success: Box::new(success_tree),
                    failure: Box::new(failure_tree),
                })
            }

            Constructor::Struct => {
                let struct_rows = groups.remove(&Constructor::Struct).unwrap();

                // Get the number of fields from the first struct pattern
                let num_fields = if let Some(row) = struct_rows.first() {
                    if let HirPatternKind::Struct { field_patterns, .. } =
                        &row.patterns[split_column].kind
                    {
                        field_patterns.len()
                    } else {
                        0
                    }
                } else {
                    0
                };

                // Specialize rows by expanding struct field patterns
                let specialized_struct =
                    self.specialize_struct_rows(struct_rows, split_column, num_fields)?;

                let failure_rows = self.combine_remaining_rows(groups, wildcard_rows);

                // Create new values list with field values replacing the struct
                let mut new_values = values.clone();
                let struct_value = new_values.remove(split_column);

                // Add HIR IDs for each field (would be ExtractValue instructions in real compilation)
                for field_idx in 0..num_fields {
                    let field_id = HirId::new(); // Placeholder - in real compilation, this would be ExtractValue
                    new_values.insert(split_column + field_idx, field_id);
                }

                let success_tree =
                    self.build_decision_tree(new_values.clone(), specialized_struct, default)?;
                let failure_tree = self.build_decision_tree(values, failure_rows, default)?;

                // Build struct test node with field extraction
                if num_fields > 0 {
                    // Create a chain of struct field tests
                    self.build_struct_test_chain(
                        struct_value,
                        0,
                        num_fields,
                        Box::new(success_tree),
                        Box::new(failure_tree),
                    )
                } else {
                    // Empty struct or no field patterns - just return success
                    Ok(success_tree)
                }
            }
        }
    }

    /// Choose the best column to split the decision tree on
    fn choose_split_column(&self, values: &[HirId], rows: &[PatternRow]) -> CompilerResult<usize> {
        // Simple heuristic: choose the leftmost column with a constructor pattern
        for (col_idx, _) in values.iter().enumerate() {
            for row in rows {
                if col_idx < row.patterns.len() {
                    match &row.patterns[col_idx].kind {
                        HirPatternKind::Constant(_)
                        | HirPatternKind::UnionVariant { .. }
                        | HirPatternKind::Struct { .. } => {
                            return Ok(col_idx);
                        }
                        _ => continue,
                    }
                }
            }
        }

        // Fallback to first column
        Ok(0)
    }

    /// Specialize rows for wildcard patterns
    fn specialize_wildcard_rows(
        &self,
        rows: Vec<PatternRow>,
        col: usize,
    ) -> CompilerResult<Vec<PatternRow>> {
        let mut result = Vec::new();

        for mut row in rows {
            // Remove the wildcard pattern from this column
            if col < row.patterns.len() {
                row.patterns.remove(col);
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Specialize rows for constant patterns
    fn specialize_constant_rows(
        &self,
        rows: Vec<PatternRow>,
        col: usize,
    ) -> CompilerResult<Vec<PatternRow>> {
        let mut result = Vec::new();

        for mut row in rows {
            // Remove the constant pattern from this column
            if col < row.patterns.len() {
                row.patterns.remove(col);
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Specialize rows for union variant patterns
    fn specialize_union_rows(
        &self,
        rows: Vec<PatternRow>,
        col: usize,
    ) -> CompilerResult<Vec<PatternRow>> {
        let mut result = Vec::new();

        for mut row in rows {
            if col < row.patterns.len() {
                // If the pattern has an inner pattern, add it
                if let HirPatternKind::UnionVariant { inner_pattern, .. } = &row.patterns[col].kind
                {
                    if let Some(inner) = inner_pattern {
                        row.patterns[col] = inner.as_ref().clone();
                    } else {
                        row.patterns.remove(col);
                    }
                } else {
                    row.patterns.remove(col);
                }
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Specialize rows for struct patterns
    fn specialize_struct_rows(
        &self,
        rows: Vec<PatternRow>,
        col: usize,
        num_fields: usize,
    ) -> CompilerResult<Vec<PatternRow>> {
        let mut result = Vec::new();

        for mut row in rows {
            if col < row.patterns.len() {
                // For struct patterns, expand field patterns into separate columns
                if let HirPatternKind::Struct { field_patterns, .. } = &row.patterns[col].kind {
                    // Clone field patterns before removing
                    let field_patterns_clone = field_patterns.clone();

                    // Remove the struct pattern
                    row.patterns.remove(col);

                    // Insert field patterns at the same position
                    // field_patterns is Vec<(field_index, pattern)>
                    let mut field_pats: Vec<Option<HirPattern>> = vec![None; num_fields];

                    for (field_idx, pattern) in &field_patterns_clone {
                        if (*field_idx as usize) < num_fields {
                            field_pats[*field_idx as usize] = Some(pattern.clone());
                        }
                    }

                    // Insert patterns (or wildcards for missing fields) in reverse order
                    for (idx, maybe_pattern) in field_pats.into_iter().enumerate().rev() {
                        let pattern = maybe_pattern.unwrap_or_else(|| {
                            // Create a wildcard pattern for unmatched fields
                            HirPattern {
                                kind: HirPatternKind::Wildcard,
                                target: row.target,
                                bindings: vec![],
                            }
                        });
                        row.patterns.insert(col, pattern);
                    }
                } else {
                    row.patterns.remove(col);
                }
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Combine remaining constructor groups with wildcard rows
    fn combine_remaining_rows(
        &self,
        groups: HashMap<Constructor, Vec<PatternRow>>,
        wildcards: Vec<PatternRow>,
    ) -> Vec<PatternRow> {
        let mut result = wildcards;

        for (_, mut rows) in groups {
            result.append(&mut rows);
        }

        result
    }

    /// Build a chain of struct field test nodes
    fn build_struct_test_chain(
        &self,
        struct_value: HirId,
        current_field: u32,
        total_fields: usize,
        success: Box<DecisionNode>,
        failure: Box<DecisionNode>,
    ) -> CompilerResult<DecisionNode> {
        if current_field >= total_fields as u32 {
            // All fields processed, return success
            return Ok(*success);
        }

        if current_field == total_fields as u32 - 1 {
            // Last field - use success directly
            Ok(DecisionNode::StructTest {
                value: struct_value,
                field_index: current_field,
                field_test: success,
                next_field: Box::new(DecisionNode::Failure), // No more fields
            })
        } else {
            // Not the last field - recursively build chain
            let next_chain = self.build_struct_test_chain(
                struct_value,
                current_field + 1,
                total_fields,
                success,
                failure.clone(),
            )?;

            Ok(DecisionNode::StructTest {
                value: struct_value,
                field_index: current_field,
                field_test: Box::new(next_chain.clone()), // Use the recursive chain as field test
                next_field: Box::new(next_chain),
            })
        }
    }
}

/// Pattern constructor for grouping
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Constructor {
    Constant(HirConstant),
    UnionVariant(u32),
    Struct,
}

/// Pattern row for decision tree construction
#[derive(Debug, Clone)]
struct PatternRow {
    patterns: Vec<HirPattern>,
    target: HirId,
    bindings: Vec<HirPatternBinding>,
}

/// Check if patterns are exhaustive
pub fn check_exhaustiveness(
    patterns: &[HirPattern],
    scrutinee_ty: &HirType,
) -> CompilerResult<bool> {
    match scrutinee_ty {
        HirType::Bool => {
            // Boolean needs true and false cases
            let mut has_true = false;
            let mut has_false = false;
            let mut has_wildcard = false;

            for pattern in patterns {
                match &pattern.kind {
                    HirPatternKind::Constant(HirConstant::Bool(true)) => has_true = true,
                    HirPatternKind::Constant(HirConstant::Bool(false)) => has_false = true,
                    HirPatternKind::Wildcard | HirPatternKind::Binding(_) => has_wildcard = true,
                    _ => {}
                }
            }

            Ok(has_wildcard || (has_true && has_false))
        }

        HirType::Union(union_ty) => {
            // Union needs all variants covered
            let mut covered_variants = HashSet::new();
            let mut has_wildcard = false;

            for pattern in patterns {
                match &pattern.kind {
                    HirPatternKind::UnionVariant { variant_index, .. } => {
                        covered_variants.insert(*variant_index);
                    }
                    HirPatternKind::Wildcard | HirPatternKind::Binding(_) => has_wildcard = true,
                    _ => {}
                }
            }

            let total_variants = union_ty.variants.len() as u32;
            Ok(has_wildcard || covered_variants.len() == total_variants as usize)
        }

        _ => {
            // For other types, check if there's a wildcard
            let has_wildcard = patterns.iter().any(|p| {
                matches!(
                    p.kind,
                    HirPatternKind::Wildcard | HirPatternKind::Binding(_)
                )
            });
            Ok(has_wildcard)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::arena::AstArena;

    fn create_test_arena() -> AstArena {
        AstArena::new()
    }

    fn intern_str(arena: &mut AstArena, s: &str) -> zyntax_typed_ast::InternedString {
        arena.intern_string(s)
    }

    #[test]
    fn test_pattern_compiler_creation() {
        let compiler = PatternMatchCompiler::new();
        assert_eq!(compiler.next_block_id, 0);
    }

    #[test]
    fn test_exhaustiveness_bool() {
        let mut arena = create_test_arena();

        // Create true and false patterns
        let true_pattern = HirPattern {
            kind: HirPatternKind::Constant(HirConstant::Bool(true)),
            target: HirId::new(),
            bindings: vec![],
        };

        let false_pattern = HirPattern {
            kind: HirPatternKind::Constant(HirConstant::Bool(false)),
            target: HirId::new(),
            bindings: vec![],
        };

        let patterns = vec![true_pattern, false_pattern];
        let result = check_exhaustiveness(&patterns, &HirType::Bool).unwrap();
        assert!(result);
    }

    #[test]
    fn test_exhaustiveness_wildcard() {
        let mut arena = create_test_arena();

        let wildcard_pattern = HirPattern {
            kind: HirPatternKind::Wildcard,
            target: HirId::new(),
            bindings: vec![],
        };

        let patterns = vec![wildcard_pattern];
        let result = check_exhaustiveness(&patterns, &HirType::I32).unwrap();
        assert!(result);
    }

    #[test]
    fn test_exhaustiveness_incomplete() {
        let true_pattern = HirPattern {
            kind: HirPatternKind::Constant(HirConstant::Bool(true)),
            target: HirId::new(),
            bindings: vec![],
        };

        let patterns = vec![true_pattern];
        let result = check_exhaustiveness(&patterns, &HirType::Bool).unwrap();
        assert!(!result); // Missing false case
    }
}
