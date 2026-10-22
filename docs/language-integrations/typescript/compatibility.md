# TypeScript compatibility

[Quick start](../../../crates/zyntax_ts/README.md) · [Architecture](architecture.md) · [Testing](testing.md)

`zynode` implements native TypeScript execution without a JavaScript output step.
The table describes implemented behavior, not permanent language restrictions.
Unimplemented constructs report `Unsupported`; the conformance suite checks these
gaps against TypeScript rather than treating them as compatibility passes.

## Language support

| Area | Implemented | Remaining gaps |
| --- | --- | --- |
| Values and control flow | Numbers, booleans, strings, primitive aliases, operators, branches and loops | Full JavaScript coercion and string behavior |
| Functions and inference | Typed parameters; forward calls; primitive, object and void return inference; annotated recursive groups; direct recursive returns inferred from base cases | Arithmetic and mutual inference cycles need annotations; recursive function-value types and broader contextual inference |
| Generics | Inferred and explicit arguments, nested calls, recursion and cached native specializations | Constraints, defaults, const parameters and operations on unresolved generic operands |
| Function values | Capture-free arrows, named references, contextual parameter types and higher-order scalar signatures | Captures, function object identity/properties, parameter-count adaptation and aggregate callback signatures |
| Objects | Scalar fields, reference identity, aliases, source-order initializers, readonly views and excess-property checks | Reference-valued fields, general layout adaptation and assignments that change concrete layout |
| Arrays | Dense `number[]`, `boolean[]`, `string[]`, `Array<T>`, proven index access, length and push | Sparse extension, length mutation, other methods and reads that may yield `undefined` |
| Unions | Scalar alternatives among number, boolean and string; number/boolean with `undefined`; narrowing, equality, truthiness and formatting | Owned/reference payloads, broader nullable combinations and optional callback signatures |
| Modules | Named/type-only/side-effect imports; function/type/interface exports; aliases; named and star re-exports; relative filesystem projects | Default/namespace objects, exported variables/live bindings, cycles/TDZ and package resolution |

String arrays and string union payloads require a proven static string lifetime.
Generic element mutation and owned reference containers need additional lifetime
analysis. Classes, capturing closures and reference-bearing unions are not yet
implemented. `void` and `undefined` remain distinct.

Local object views can expose a subset of fields without copying. Calls and
annotated returns require matching native offsets for every exposed field.
Ordinary pointer casts preserve the allocation and its ownership.

## Checking policies

Strict static checking is enabled. The default `--soundness typescript` policy
uses supported TypeScript assignment rules. `--soundness checked` additionally
rejects weakening readonly views through initialization, assignment, arguments,
returns or assertions.

For example, the default policy prints `2`; checked mode rejects the writable alias:

```ts
type Read = { readonly x: number };
type Write = { x: number };
const r: Read = { x: 1 };
const w: Write = r;
w.x = 2;
console.log(r.x);
```

Checked mode adds a policy; it does not claim a complete sound type system or
permit unsafe native layout conversions. Numbers use `f64`, including NaN,
infinities and signed zero. The compiler's `strict_fp` annotation preserves
separate source arithmetic rounding.

## Module behavior

Relative imports accept `.ts`, extensionless files, directory indexes and `.js`
specifiers backed by TypeScript source. Each canonical file is checked once.
Explicit type-only and unused named imports do not evaluate dependencies; bare
imports retain side effects. Dependency initializers run once before importers.

Re-exports retain original binding identity and type-only provenance. Explicit
exports shadow stars; conflicting star bindings are errors, while repeated paths
to the same binding remain valid. Stars exclude `default`. Ordinary stars evaluate
their source; type-only forwarding is erased. Cycles report `Unsupported`.

Diagnostics use the shared Ariadne formatter, including source labels and related
locations. `NO_COLOR` disables colors; `CLICOLOR_FORCE` enables them.
`--diagnostics=json` emits categories, filenames and byte ranges.
