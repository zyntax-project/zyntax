//! Code Generation for ZynPEG 2.0
//!
//! This module generates Rust code from GrammarIR:
//! - `parser_gen.rs`: Generate parse_* methods for each rule
//! - `action_gen.rs`: Generate action code from ActionIR
//! - `pratt_gen.rs`: Generate Pratt parser for expression precedence

pub mod action_gen;
pub mod parser_gen;
pub mod pratt_gen;

pub use action_gen::*;
pub use parser_gen::*;
pub use pratt_gen::*;
