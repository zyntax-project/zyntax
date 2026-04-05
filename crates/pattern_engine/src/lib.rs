#![allow(
    unused,
    dead_code,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::derivable_impls
)]

pub mod bindings;
pub mod context;
pub mod engine;
pub mod metadata;
pub mod node_id;
pub mod pass;
pub mod pattern;
pub mod rewrite;
pub mod trace;
pub mod verify;
pub mod walk;

pub use bindings::Bindings;
pub use context::{EffectOpIndex, LoweringTarget, MatchContext};
pub use engine::{EngineConfig, EngineResult, PassOrderError, PatternEngine};
pub use metadata::MetadataTable;
pub use node_id::NodeId;
pub use pass::PatternPass;
pub use pattern::{Pattern, PatternId};
pub use rewrite::{
    DeclRewrite, ExprRewrite, Priority, RewriteBenefit, RewriteId, RewriteOutput, StmtRewrite,
};
pub use trace::{FiredRewrite, FiredSet};
