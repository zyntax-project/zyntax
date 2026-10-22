use std::fmt::Write;
use zyntax_typed_ast::InternedString;

/// A host-resolved dependency; shared compiled imports and snapshots supply its program.
#[derive(Clone, Copy)]
pub struct ModuleDependency<'a> {
    pub specifier: &'a str,
    pub program: &'a zyntax_typed_ast::TypedProgram,
}

pub(crate) fn append_component(buffer: &mut String, component: &str) {
    for byte in component.bytes() {
        if byte.is_ascii_alphanumeric() {
            buffer.push(char::from(byte));
        } else {
            write!(buffer, "_{byte:02x}").unwrap();
        }
    }
}

/// A resolver's stable, canonical identity, independent of diagnostic filenames.
///
/// File resolvers should canonicalize paths before constructing an identity.
/// Package and snapshot resolvers can supply their own logical module names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleId {
    name: InternedString,
    prefix: InternedString,
}

impl ModuleId {
    pub fn new(identity: &str) -> Self {
        let length: usize = identity
            .bytes()
            .map(|byte| if byte.is_ascii_alphanumeric() { 1 } else { 3 })
            .sum();
        let mut prefix = String::with_capacity(length + 6);
        prefix.push_str("ts$m$");
        // Escape the escape character too: punctuation must remain distinct
        // after either native backend's symbol mangling.
        append_component(&mut prefix, identity);
        prefix.push('$');
        Self {
            name: InternedString::new_global(identity),
            prefix: InternedString::new_global(&prefix),
        }
    }

    pub fn name(self) -> InternedString {
        self.name
    }

    /// The native name of a source-level function, for shared linking or host calls.
    pub fn function_name(self, source_name: &str) -> InternedString {
        let mut name = self.prefix();
        name.push_str("fn$");
        append_component(&mut name, source_name);
        InternedString::new_global(&name)
    }

    pub(crate) fn prefix(self) -> String {
        self.prefix.resolve_global().unwrap()
    }
}
