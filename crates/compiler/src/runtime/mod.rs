//! Async Runtime for Zyntax
//!
//! This module provides a minimal async runtime for executing async functions.
//! It includes:
//! - Task: Represents a spawned async function
//! - Executor: Schedules and runs tasks
//! - Waker: Mechanism for waking tasks
//! - block_on: Simple runtime for blocking on a future
//!
//! ## Design
//!
//! The runtime follows a similar design to Rust's async ecosystem:
//! - Tasks are state machines (AsyncStateMachine from async_support)
//! - Executor polls tasks until completion
//! - Wakers notify the executor when tasks should be re-polled
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use zyntax_compiler::runtime::{Runtime, block_on};
//!
//! async fn fetch_data() -> i32 {
//!     // ... async code ...
//!     42
//! }
//!
//! let result = block_on(fetch_data());
//! assert_eq!(result, 42);
//! ```

pub mod executor;
pub mod task;
pub mod waker;

pub use executor::{block_on, Executor};
pub use task::Task;
pub use waker::{Context, Waker};
