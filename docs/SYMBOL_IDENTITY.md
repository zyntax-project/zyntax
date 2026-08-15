# Symbol identity

A symbol is identified by its path, and the path begins with the
language. Two languages in one runtime can each have an `add`, two
modules in one language can each have a `parse`, and neither takes the
other's name.

Today a symbol is identified by its last segment. `export_function`
refuses a name a second language already exported, which stops the
damage but does not make the two nameable. This is how they become
nameable.

## What already carries the weight

A function can already say what it links as. `TypedFunction` has a
`link_name`, lowering copies it onto `HirFunction`, and every backend
reads it: Cranelift and the interpreter when they register a symbol,
LLVM as `link_name.unwrap_or(actual_name)`. The mechanism for giving a
function a symbol other than its source name is built, shipped, and
exercised by every extern in the tree.

So nothing here needs a new field in HIR, a new backend path, or a new
convention. ZRTL already spells its symbols `$Image$load`, which is the
same idea reached by hand.

## The two halves

Naming is not the same problem as resolving, and conflating them is
what made this look small twice already.

**Identity** is which symbol a function registers as. Giving each
merged declaration a `link_name` of `language::module::name` makes
every symbol distinct, and the backends already do the rest. A Python
`add` and a TypeScript `add` stop being one address.

**Resolution** is which function a call means. A call to `add` inside
a TypeScript file has to find TypeScript's, and today it cannot, for a
reason that has nothing to do with symbols: imports are merged. An
imported module's declarations are drained into the importing program
and lowered together, so both `add`s are declarations in one program
and a lookup by source name has nothing to choose on.

Identity without resolution is worth having. It converts a silent
wrong call into a link that fails, and it is what makes a debugger, a
profile, or a stack trace say which `add` ran. But it does not by
itself make the right one run.

## Where the path comes from

A snapshot module knows both parts already: the language installed it
and the module is named in the artifact. `CompiledImport` carries
both, and a program parsed from source carries the language the host
registered it under.

That gives `zynml::prelude::sum` for a standard library function and
`zynml::main::train` for one in a user's file. Modules that nest give
their own path; nothing here decides how a language spells them,
only that whatever it spells is what the symbol carries.

Externs keep their own names. A function whose `link_name` is already
set is naming a symbol somebody else owns, and prefixing it would name
a symbol nobody exports.

## What a rename reaches

The blast radius is the argument for doing this deliberately rather
than as a patch.

A host calls a function by name, and `call_function("main")` has to
keep working when the symbol is `zynml::main::main`. That wants a
bare-name index alongside the qualified one, resolving when a bare
name is unambiguous and refusing when it is not, which is the same
rule the export check uses now.

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

Identity first, since it stands alone, needs no new machinery, and
turns a silent overwrite into a loud failure. Bare-name fallback with
it, because everything that looks a symbol up by name depends on it.
Resolution after, with the scope table, once there is something to
resolve between.
