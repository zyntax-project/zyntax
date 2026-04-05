use crate::metadata::MetadataTable;
use crate::trace::FiredSet;
use zyntax_typed_ast::advanced_analysis::AnalysisContext;
use zyntax_typed_ast::effect_system::EffectSystem;
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedProgram};
use zyntax_typed_ast::InternedString;
use zyntax_typed_ast::TypeRegistry;

/// Snapshot of effect operation names, built before the walk.
/// Maps operation name → effect name.
#[derive(Default, Clone)]
pub struct EffectOpIndex {
    ops: std::collections::HashMap<String, InternedString>,
}

impl EffectOpIndex {
    /// Build from a TypedProgram by scanning its Effect declarations.
    pub fn build(program: &TypedProgram) -> Self {
        let mut ops = std::collections::HashMap::new();
        for decl in &program.declarations {
            if let TypedDeclaration::Effect(effect) = &decl.node {
                for op in &effect.operations {
                    let op_name = op.name.resolve_global().unwrap_or_default();
                    ops.insert(op_name, effect.name);
                }
            }
        }
        Self { ops }
    }

    /// Check if a name resolves to an effect operation.
    /// Returns the effect name if found.
    pub fn resolve(&self, name: &str) -> Option<InternedString> {
        self.ops.get(name).copied()
    }
}

/// Everything a pattern predicate can query about the surrounding program.
/// Immutable during matching.
pub struct MatchContext<'a> {
    /// Resolved types and trait implementations
    pub registry: &'a TypeRegistry,
    /// DFG/CFG for data flow queries
    pub analysis: &'a AnalysisContext,
    /// Active effects at the match site
    pub effects: &'a EffectSystem,
    /// Current lowering target (CPU/GPU/FPGA)
    pub target: LoweringTarget,
    /// Metadata attached to TypedAST nodes (target-specific)
    pub metadata: &'a MetadataTable,
    /// Previously fired rewrites in this iteration (for ordering)
    pub fired: &'a FiredSet,
    /// Snapshot of effect operation names for op resolution
    pub effect_ops: &'a EffectOpIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoweringTarget {
    Cpu,
    Nvptx,
    Rtlil,
    Wasm,
}

impl Default for LoweringTarget {
    fn default() -> Self {
        LoweringTarget::Cpu
    }
}
