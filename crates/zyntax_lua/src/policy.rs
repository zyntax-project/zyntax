//! Lua's spellings for the built-in library. Included by the crate
//! and by its build script, which lowers the library once for the
//! snapshot the crate carries.

pub const POLICY: zyntax_builtins::Policy = zyntax_builtins::Policy {
    true_text: "true",
    false_text: "false",
    none_text: "nil",
    single_quotes: false,
    // A float that is a whole number still prints as one: `1.0`.
    float_fraction: true,
    // Tables and coroutines are instances of this frontend's kinds.
    instance_hooks: true,
    // A library error goes through `zb_hook_raise`, which this frontend
    // defines.
    exceptions: true,
    type_names: zyntax_builtins::TypeNames {
        none: "nil",
        bool: "boolean",
        int: "number",
        float: "number",
        str: "string",
        list: "table",
        tuple: "table",
        dict: "table",
        set: "table",
        function: "function",
        object: "userdata",
    },
};

/// The name the library is imported under.
pub const LIBRARY_MODULE: &str = zyntax_builtins::MODULE;
