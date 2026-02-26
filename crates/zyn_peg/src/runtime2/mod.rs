//! ZynPEG 2.0 Runtime - Direct TypedAST Construction
//!
//! This module provides the runtime components for the new PEG parser:
//! - `state.rs`: ParserState with AstArena integration
//! - `memo.rs`: Packrat memoization for O(n) parsing
//! - `combinator.rs`: Parser combinator functions
//! - `interpreter.rs`: Runtime interpreter for GrammarIR

pub mod combinator;
pub mod interpreter;
pub mod memo;
pub mod state;

pub use combinator::*;
pub use interpreter::*;
pub use memo::*;
pub use state::*;
