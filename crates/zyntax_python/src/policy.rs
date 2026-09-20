//! Python's spellings for the built-in library. Included by the crate
//! and by its build script, which lowers the library once for the
//! snapshot the crate carries.

pub const POLICY: zyntax_builtins::Policy = zyntax_builtins::Policy {
    true_text: "True",
    false_text: "False",
    none_text: "None",
    single_quotes: true,
    float_fraction: true,
    instance_hooks: true,
    exceptions: true,
    bool_is_number: true,
    type_names: zyntax_builtins::TypeNames {
        none: "NoneType",
        bool: "bool",
        int: "int",
        float: "float",
        str: "str",
        list: "list",
        tuple: "tuple",
        dict: "dict",
        set: "set",
        function: "function",
        object: "object",
    },
};

/// The name the library is imported under.
pub const LIBRARY_MODULE: &str = zyntax_builtins::MODULE;
