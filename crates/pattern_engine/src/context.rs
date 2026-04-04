use crate::metadata::MetadataTable;
use crate::trace::FiredSet;
use zyntax_typed_ast::advanced_analysis::AnalysisContext;
use zyntax_typed_ast::effect_system::EffectSystem;
use zyntax_typed_ast::TypeRegistry;

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
