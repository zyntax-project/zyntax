# Symbol identity

A symbol is identified by its path, and the path begins with the
language. Two languages in one runtime can each have an `add`, two
modules in one language can each have a `parse`, and neither takes the
other's name.

Today a symbol is identified by its last segment. `export_function`
refuses a name a second language already exported, which stops the
damage but does not make the two nameable. This is how they become
nameable.

## What the backends already do

A local function cannot collide with another local function, and that
is worth knowing before designing a scheme to stop it. Cranelift
declares one as its name joined to its `HirId`, with a generation
suffix so recompiling under tier-up or hot reload produces a fresh
symbol. LLVM declares one as `func_` and the id, dropping the source
name. Both are unique by construction.

`link_name` does not enter into that. Cranelift, LLVM and the
interpreter all read it only when a function is external, which is what
it is for: an extern's declared name is an alias, and the link name is
the symbol a plugin actually provides. Setting it on an ordinary
function changes nothing anywhere.

So the collision this opened with does not live in the backends. It
lives in the one table still keyed by a bare name, the symbols a host
exports for later modules to link against. Qualifying that is a change
at the export boundary, not a mangling scheme reaching into lowering.

One thing found on the way belongs in its own change. LLVM keeps the
name `main` verbatim while mangling every other local function, so a
backend is deciding which name is an entry point. An entry point is
configured, and nothing below the host should be naming one.

## The two halves

Naming is not the same problem as resolving, and conflating them is
what made this look small twice already.

**Identity** is which symbol a function registers as, and for anything
a backend compiles it is already settled. What is not settled is the
name a host exports a function under, which is its source name and
nothing else.

**Resolution** is which function a call means, and this is the real
gap. A call to `add` inside a TypeScript file has to find TypeScript's,
and today it cannot, for a reason that has nothing to do with symbols:
imports are merged. An imported module's declarations are drained into
the importing program and lowered together, so both `add`s are
declarations in one program and a lookup by source name has nothing to
choose on. Two functions with one name, each already compiled to its
own symbol, and no rule for deciding which one a call meant.

## Where the path comes from

A snapshot module knows both parts already: the language installed it
and the module is named in the artifact. `CompiledImport` carries
both, and a program parsed from source carries the language the host
registered it under.

That gives `zynml::prelude::sum` for a standard library function. What
it gives for a function in a user's file is the question, and the
answer today is wrong: lowering is handed the module name `main`, as a
literal, for every program it compiles. So two files would both be
`zynml::main::` and still collide, which makes the middle segment of
the path a constant and the path no better than the leaf.

The name is already available. `module_name_of` takes it from the file
stem, and the runtime has the filename when it parses. Identity
depends on that being what reaches lowering, so it is part of this
work rather than a tidy-up beside it.

Nothing here decides how a language spells nested modules. Whatever it
spells is what the symbol carries.

Externs keep their own names. A function whose `link_name` is already
set is naming a symbol somebody else owns, and prefixing it would name
a symbol nobody exports.

## What a rename reaches

The blast radius is the argument for doing this deliberately rather
than as a patch.

A host calls a function by name, and it has to keep working when the
symbol is qualified. Which name that is comes from configuration: a
grammar declares an entry point in its metadata and a host can name
one itself. Zyntax has no entry point of its own and must not assume
one; a language deciding that its own is `main` is that language's
business. So the fallback resolves whatever name it is handed, not a
name this layer picked.

That wants a bare-name index alongside the qualified one, resolving
when a bare name is unambiguous and refusing when it is not, which is
the same rule the export check uses now.

Hot reload identifies what to replace by name. Snapshot caches key on
it. The interpreter keeps its own symbol table. A test asserting a
plugin registered `$Image$load` is asserting on a spelling. None of
these break if bare names keep resolving, and all of them break at
once if they do not, so the fallback is not a convenience.

## Resolution, when it comes

The honest version needs the merge to stop throwing provenance away.
An imported declaration should arrive knowing the module it came from,
and lowering should resolve a call against the names visible to the
calling module: its own first, then what it imported, honouring the
alias each import chose.

That is a scope table, built where the merge happens, keyed by the
module doing the calling. It is ordinary name resolution, and the
reason it does not exist is that a single-language runtime never
needed it: one program, one namespace, no ambiguity to resolve.

The larger question underneath is whether modules should be merged at
all. Compiling each separately and linking them is what makes a
namespace per module fall out for free rather than being rebuilt on
top of a merge. That is a much bigger change and it is not proposed
here, but it is the shape this keeps pointing at.

## Order

The export boundary first. It is the only place a bare name is still
the identity, the change is local to it, and refusing the collision is
already there to build on.

Resolution after, with the scope table, because it is the one that
decides whether a call is right rather than whether a link is unique.
Bare-name fallback goes with whichever of them first changes a name a
host can ask for.

Nothing about `link_name` on ordinary functions, which was the plan
until the backends were read. It is an externs-only mechanism and the
functions it would have qualified are unique already.
