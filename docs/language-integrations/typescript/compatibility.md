# TypeScript compatibility

[Quick start](../../../crates/zyntax_ts/README.md) · [Architecture](architecture.md) · [Testing](testing.md)

`zynode` implements native TypeScript execution without a JavaScript output step.
The table describes implemented behavior, not permanent language restrictions.
Unimplemented constructs report `Unsupported`; the conformance suite checks these
gaps against TypeScript rather than treating them as compatibility passes.

## Language support

| Area | Implemented | Remaining gaps |
| --- | --- | --- |
| Values and control flow | Numbers, booleans, strings, primitive aliases, operators (unary `+` and `-` on numbers and booleans), branches and loops; `switch` with fall-through, `do`/`while`, labeled `break` and `continue`; module bindings that functions read and write | Full JavaScript coercion and string behavior; `delete`; `as const`; library globals as values (`Date`, `JSON`, `Symbol`, typed arrays and others); module bindings holding unions or computed strings |
| Functions and inference | Typed parameters; optional and default number or boolean parameters; rest parameters; forward calls; primitive, object and void return inference; annotated recursive groups; direct recursive returns inferred from base cases | Arithmetic and mutual inference cycles need annotations; recursive function-value types and broader contextual inference; optional parameters of other types, and function values or exports with optional, default or rest parameters |
| Generics | Inferred and explicit arguments, trailing defaults, dependent constraints, scalar-bound operations, nested calls, recursion and cached native specializations | Generic aliases/interfaces, const parameters, richer inference and layout-dependent aggregate operations |
| Function values | Capture-free arrows, named references, contextual parameter types and higher-order scalar signatures | Captures, function object identity/properties, parameter-count adaptation and aggregate callback signatures |
| Objects | Number, boolean and static string fields, reference identity, aliases, interfaces extending record interfaces, classes implementing them, source-order initializers, readonly views and excess-property checks | Reference-valued fields, general layout adaptation and assignments that change concrete layout |
| Classes | Fields, constructors with parameter properties, instance and static methods, `private`/`protected` members and constructors, `implements`; single inheritance among the module's classes with `super(...)`, `super.method()`, overriding methods dispatched on the instance's class, implicit constructors forwarding base parameters, redeclared fields with a value and polymorphic `this` results | Abstract, generic and nested classes, accessors, static fields, redeclared fields without a value, overrides with a different signature, mixins and extending library classes |
| Arrays | Dense numbers, booleans, static strings, class instances and nested arrays; typed indexing, length, push, array `for…of`, and filled length constructors | Sparse extension, length mutation, other methods, optional reads and general iterator protocols |
| Enums | Numeric, string, `const` and ambient `const` enums with constant members; member access, member and union-of-member types, assignability, literal comparisons and narrowing by equality | Reverse mapping, enum objects as values, computed or mixed members, ambient values, declaration merging, exported enums, and enums in arrays, fields, unions or function types |
| Unions | Scalar alternatives among number, boolean and string; number/boolean with `undefined`; narrowing, equality, truthiness and formatting | Owned/reference payloads, broader nullable combinations and optional callback signatures |
| Modules | Named/default/type-only/side-effect imports; static namespace calls, function references and qualified types; function/type/interface exports; default function declarations and bindings; aliases; named and star re-exports; relative filesystem projects | Other default values, namespace object values/re-exports and runtime indexing, exported variables/live bindings, cycles/TDZ and package resolution |

Top-level bindings that functions, arrows or methods use are module globals shared by
every tier. A function that may run before a `let` or `const` declaration checks the
binding on each access and raises `ReferenceError`, as the temporal dead zone requires;
literal constants that nothing can observe early are substituted. A function read of a
binding that may still hold `undefined` (declared without a value, or a `var` read before
its declaration) is checked and traps, as an array index TypeScript would read as
`undefined` does.

A `switch` evaluates its case tests in order with strict equality and falls through between
clauses. Case values must be comparable with the discriminant, and an enum discriminant is
narrowed in each clause. When the clauses cover every member of an enum, a number outside
the enum traps instead of running past the switch, since TypeScript treats that path as
unreachable. Clauses that share block-scoped declarations, and switches over unions, report
`Unsupported`.

An enum value is its native number or string; enum identity and the members a binding may
hold are checker facts, so comparisons that cannot overlap and assignments of other enums or
non-member literals are rejected as TypeScript rejects them. Number and string `const`
bindings keep their literal types for the same checks. A function that may read a regular
enum before its declaration runs checks for it, as the enum object would still be undefined.

String arrays, string fields and string union payloads require a proven static string lifetime: a literal, a template of literal parts, a constant, or a value read from such storage.
Generic element mutation, capturing closures and broader reference-bearing unions
need additional implementation. `void` and `undefined` remain distinct.

Array `for…of` supports `let` and `const` identifier bindings, observes growth and
keeps iterating the original array if its source variable is reassigned. Numeric
length constructors support `new Array<T>(n).fill(value)` and `Array<T>(n).fill(value)`;
reference elements share identity. Invalid lengths fail before evaluating the fill
argument. Bare length constructors, fill ranges, destructuring, `var` loop bindings
and asynchronous/custom iterators remain unsupported.

A subclass's layout begins with its base's, so a base-typed view reads an instance in
place. Every instance in a hierarchy carries its class in a hidden field; a call through a
base-typed receiver runs the override of the instance's class. A derived constructor runs
the statements before `super(...)` without `this`, then the base constructor, then its own
parameter properties and field initializers, as JavaScript does.

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
Type-only and unused bindings, including namespace bindings used only for types,
do not evaluate dependencies. Bare imports retain side effects. Dependency
initializers run once before importers.

Re-exports retain original binding identity and type-only provenance. Explicit
exports shadow stars; conflicting star bindings are errors, while repeated paths
to the same binding remain valid. Stars exclude `default`. Ordinary stars evaluate
their source; type-only forwarding is erased. Cycles report `Unsupported`.

Default function declarations may be named or anonymous. Import aliases and
re-exports retain the same native signature, including generic parameters and
array ownership facts. Default exports of type bindings remain type-only and
are erased. Other default expression values need module binding support.

Namespace imports resolve `lib.fn`, `lib["fn"]` and qualified types such as
`lib.Point` through exported checker interfaces. Calls retain native generic,
readonly and array ownership facts. Members cannot be assigned. Namespace object
values, namespace re-export objects and runtime property indexing remain gaps.

Diagnostics use the shared Ariadne formatter, including source labels and related
locations. `NO_COLOR` disables colors; `CLICOLOR_FORCE` enables them.
`--diagnostics=json` emits categories, filenames and byte ranges. A grammar error
TypeScript reports from its checker, such as a decorator on a function or an empty
variable declaration list, is a type error with TypeScript's message, even where the
parser rejects it first.
