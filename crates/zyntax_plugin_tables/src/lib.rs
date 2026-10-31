//! The symbol tables of the plugins the language frontends link into
//! their executables, for the frontends' build scripts, which compile the
//! library's calls into those plugins ahead of time.
//!
//! Each plugin's source compiles here as a module. A plugin crate also
//! builds as a cdylib, so its rlib's file name carries no hash: a build
//! script depending on the plugin itself gets a second build of it,
//! under its own profile and features, written to the same file as the
//! executable's.

#[path = "../../../plugins/zrtl_io/src/lib.rs"]
pub mod zrtl_io;

#[path = "../../../plugins/zrtl_math/src/lib.rs"]
pub mod zrtl_math;

#[path = "../../../plugins/zrtl_string/src/lib.rs"]
pub mod zrtl_string;
