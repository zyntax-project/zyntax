# Language snapshots

A snapshot is everything a language brings to a runtime, in one
artifact: its grammar, compiled; its standard library, already parsed;
and the type ids both were built against. A host installs one and the
language is ready to compile source.

This replaces five hand-written pieces per language with two lines,
and it closes a hole that only opens when a runtime runs more than one
language at a time.

## What a language writes today

ZynML is the worked example. To ship a stdlib it writes a `build.rs`
calling `CompiledImport::new` per module, three `include_bytes!`
constants, a `OnceLock` per module with a decode helper, two resolver
callbacks whose match arms have to agree, and a loop that decodes each
artifact before any user source is parsed so the build-time type ids
are reserved in order.

The last one is the tell. It is an ordering rule of the runtime, not of
ZynML, and today it lives in a comment in the language crate. Every
language on Zyntax has to rediscover it, and getting it wrong produces
type ids that drift from the ones the artifacts were built against,
which surfaces a long way from the cause.

## What multi-language breaks

A runtime holds a grammar per language, keyed by name, with a file
extension map beside it (`register_grammar`). Imports do not work that
way. `import_resolvers` and `compiled_import_resolvers` are flat lists
on the runtime, each callback takes a bare module name, and the first
resolver to answer wins.

So two languages that both ship a module called `prelude` resolve to
whichever one registered first. Nothing detects it. The artifact
carries the language it belongs to and the import chain checks that
language is registered, but by then the wrong module has already been
chosen.

The source fallback is worse. When a module resolves to source rather
than to an artifact, the chain tries to parse it with every registered
grammar in turn. With one language that is a no-op. With three it is
two wasted parses and an ambiguity, because the first grammar that
accepts the text wins whether or not the module was written in it.

## The shape

    // build.rs
    SnapshotBuilder::new("zynml")
        .grammar(ZYNML_GRAMMAR)
        .module("prelude", prelude_src)
        .module("tensor", tensor_src)
        .build_in(&out)?;

    // lib.rs
    static SNAPSHOT: &[u8] = zyntax_embed::include_snapshot!("zynml");

    // runtime setup
    runtime.install_snapshot(Snapshot::load(SNAPSHOT)?)?;

The builder owns the file name, composing it from the language name and
the format's own extension, and the macro resolves the same name from
`OUT_DIR`. Neither side spells the extension, so a build that writes
one name and a load that reads another cannot happen. The extension is
a convenience; a magic header and a schema constant are the authority,
as compiled grammars already do.

`install_snapshot` is one call that registers the grammar and its
extensions, reserves the type ids, and registers the modules. The
ordering rule moves into the runtime, which is where it belongs, and
every language gets it right by construction.

## Modules become language-scoped

A snapshot registers its modules under `(language, module)`. An import
written inside a ZynML file resolves against ZynML's modules first, so
existing source is unchanged and two languages can both ship a
`prelude`. A module from another language is named explicitly, spelled
however each grammar chooses to spell it, and resolves against that
language's table.

The source fallback stops guessing. A module registered under a
language parses with that language's grammar and no other. Source that
belongs to no snapshot keeps going through the callbacks below.

## What stays

The existing callbacks stay as the layer underneath. They are how a
host resolves a module it cannot know at build time, from a directory
or over a network, and a snapshot does not replace that. What changes
is that shipping a fixed standard library stops going through the
dynamic path.

`CompiledImport` stays as the per-module unit inside a snapshot, so the
wire format and its versioning carry over rather than being reinvented.

## Why the grammar belongs inside

A grammar is compiled to a parsing machine before it can parse, and
that compile currently happens when the grammar loads: about 0.8 ms for
ZynML's 417 rules, once per runtime. In a snapshot it is a section
written at build time, so a runtime maps it rather than building it,
and the cost goes to zero instead of being amortised.

## The contract a grammar harnesses

An import names a language or it does not:

    pub struct TypedImport {
        /// The language the module belongs to. `None` means the
        /// language of the file doing the importing.
        #[serde(default)]
        pub language: Option<InternedString>,
        pub module_path: Vec<InternedString>,
        pub items: Vec<TypedImportItem>,
        pub span: Span,
    }

That field is the whole runtime-facing surface. Resolution reads it,
falls back to the importing file's language when it is `None`, and
looks the pair up in that language's module table. Everything above it
belongs to the language: whether it writes `import zig::math`, `from
zig import math`, or `@zig import math` is a matter for its grammar,
and its action sets the field.

The language goes in a field of its own rather than as the first
segment of `module_path`, which would make a module actually named
`zig` indistinguishable from the `zig` language. `serde(default)`
keeps artifacts built before the field readable, since a compiled
import is a serialised `TypedProgram`.

A host that registers no second language never sees any of this. The
field stays `None`, resolution stays within the one language
installed, and nothing about existing source changes.

## TypeScript importing Python

Resolution is the easy half. TypeScript writes whatever its grammar
says, its action sets `language: Some("python")`, and the pair lands in
Python's module table. The question that decides whether any of this is
usable is what the TypeScript checker sees on the other side.

It sees the Python module's exported declarations, carrying the types
Python gave them. A `def add(a, b)` arrives with parameters typed
`Any`, because that is what the Python front end knew. TypeScript
already has a rule for values it cannot type, so it applies its own:
`Any` is what `any` means, and its checker decides how strict to be.
Nothing asks TypeScript to understand Python's type system, only to
read a signature written in the shared vocabulary.

That vocabulary is what the two languages actually agree on. The
primitives and `Any` are common ground because they are the type
registry's, not any language's. A nominal type is not: Python's `list`
and TypeScript's `Array` are separate entries with separate ids even
when they mean the same thing. So a nominal type crossing the boundary
is opaque to the importer, passable and returnable but not inspectable,
unless both languages name the same registry entry deliberately.

Values cross without conversion glue because both languages lower to
one HIR and run on one runtime. A static caller handing a value to an
`Any` parameter boxes it, and the same is true in reverse, which is the
autoboxing the compiler already does within a single language. Crossing
a language boundary is that case and not a new one.

Type ids are reserved per snapshot in registration order, so a
language's artifacts agree with the registry they were built against
and a second language's reservations start after the first rather than
overlapping it.

Exported symbols are the part that does not work yet. `export_function`
publishes a compiled function under a bare name for later modules to
link against, and when a name is already exported it logs a warning and
overwrites the pointer. A Python `add` and a TypeScript `add` in one
runtime are therefore the same symbol, and the second one loaded wins.
The type registry's `current_module` does not help here: it qualifies
type names, not function symbols.

That is the third flat namespace this design has to answer, after
module resolution and the grammar the source fallback guesses at. A
snapshot knows the language of everything it installs, so it can
publish symbols qualified by it and resolve a bare name within the
importing language first, the same rule modules follow. A collision
between two languages then stops being a warning and a silent
overwrite, which is the correct outcome: nothing about `add` in Python
says it should replace `add` in TypeScript.

Qualifying them costs more than qualifying modules did, and it is
worth being clear why. A module is resolved by a runtime that can be
handed a language alongside the name. A symbol is linked by name, at
the backend, against a table that is a plain map of name to address. A
qualified export that nothing else knows about is an export nothing
can call: the name a call site emits has to be qualified in the same
way, which makes this a mangling scheme reaching back into lowering
rather than a lookup rule at the edge.

So it splits in two. Refusing the collision needs only the language
that exported each symbol, which the runtime already has when it loads
a module, and it turns silent corruption into an error at the point of
the second export. Qualifying the names, so two languages can both
have an `add` and each call its own, is the mangling work and belongs
with whatever decides how a language's symbols are spelled.

Renaming an import does not stand in for either. An import item
carries the name in the module it came from and an optional local
alias, and the alias binds a name in the file doing the importing; the
call still links against the name the module exported. So `add as
pyAdd` reads like it settles which `add` is meant and settles nothing:
both are one entry in the symbol table, and the second export replaces
the first. That makes aliasing worth naming in the error, because it
is the first thing anyone will reach for, and it is the case where the
damage is hardest to see.

Once names are qualified, aliasing becomes what it looks like. Two
`add`s are two symbols, an import picks one, and a local alias is
simply a convenience for writing it.

What this design does not attempt is making Python's semantics
available to TypeScript. Calling across the boundary is a call. Whether
the callee raises where the caller expects a rejected promise is a
question for the languages, and the runtime's answer is the effect
system, which is already language-neutral at HIR.

## Lowered HIR, symbol and sourcemap aware

A snapshot carries lowered HIR, not only the typed AST. A compile today
lowers 462 stdlib declarations for a kernel with a handful of its own,
because the artifact skips parsing the standard library but not
lowering it, and that is the largest item left in a cold compile.

The encoding is the bytecode format, which already exists: a versioned,
platform-independent HIR serialisation with a documented spec, written
for exactly this and already used by the benchmark's `.zbc` cache.
Nothing new gets invented for the snapshot's HIR section.

Two things that format needs for this use.

**Symbols.** `HirModule` already carries `exports: Vec<HirExport>`, each
a name, an internal name, and whether it is a function or a global.
What it does not carry is which language produced it, and that is the
field that decides whether installing two snapshots is safe. With it,
installation publishes symbols qualified by language, resolves a bare
name within the importing language first, and reports a real
cross-language collision instead of the bare-name overwrite
`export_function` does today. Without it, the collision is a log line.

**Source maps.** HIR values carry `span: Option<Span>`, which is a pair
of byte offsets into a source nobody named. Inside one compile that is
fine, because the source is in hand. In a snapshot it is not: the
standard library was compiled at build time and its text is not on the
running machine. So a snapshot carries a source table per module, path
and text or a line index, and spans resolve through it.

That table is what makes the rest legible. A panic inside prelude code
points at a prelude line rather than an offset into nothing. A
TypeScript frame calling a Python one names both files and both
languages. And a debugger stepping through installed code has
somewhere to step to. A snapshot without it is faster and blind, which
is a bad trade for a standard library, because the code a user did not
write is exactly the code they most need named when it fails.

## What the bytecode format has to promise

Shipping HIR in a snapshot asks more of the bytecode format than the
benchmark cache does, and the difference is worth being exact about.

The format is agnostic in the way its spec claims: a frontend in any
language can emit HIR without writing Rust, and the container is
versioned, checksummed, and rejects a major mismatch. ZRTL settled the
other boundary, the plugin ABI, with a version checked when a package
loads.

Neither of those covers this. The payload is postcard over the
compiler's own HIR structs, and postcard is positional rather than
self-describing, so adding a field to `HirFunction` changes the wire
layout while the major version stays where it was. The tree already
knows: the benchmark keeps a `CACHE_SCHEMA_VERSION` it bumps by hand,
because, in its own words, neither the checksum nor postcard catches a
valid old payload deserialised into a subtly incompatible new struct.

A hand-bumped constant is fine for a cache one commit writes and reads.
It is not fine for an artifact a language builds once and ships, which
is what a snapshot is. So the format owes a schema identity that moves
on its own when the HIR types move, derived rather than remembered, and
checked on load. A snapshot built against a compiler whose HIR has
since changed should be rejected on sight, not decoded into something
that looks plausible.

Given that, a snapshot carries both sections: the typed AST it has
always had, and lowered HIR beside it. HIR that the running compiler
accepts is used and the lowering is skipped. HIR it rejects is
discarded, and the typed AST is lowered as it is today. That is slower
and correct, and it means a language does not have to rebuild in
lockstep with a compiler release to keep working.
