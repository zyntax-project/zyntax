# Native TypeScript

`zynode` runs statically checked TypeScript through Zyntax's interpreter-first
native runtime. The `zyntax_ts` crate uses Oxc for parsing and Zyntax for compilation
and execution. It does not emit JavaScript or require Node.js to run programs.
TypeScript compatibility is still growing; see the [compatibility guide](../../docs/language-integrations/typescript/compatibility.md).

## Quick start

Save this as `example.ts`:

```ts
function sum(xs: number[]): number {
    let total = 0;
    for (let i = 0; i < xs.length; i++) {
        total += xs[i];
    }
    return total;
}

console.log(sum([1, 2, 3])); // 6
```

From the repository root:

```sh
cargo build --release -p zyntax_ts
target/release/zynode check example.ts
target/release/zynode run example.ts
```

Use `run --soundness checked example.ts` for additional readonly checks,
`run --cache-dir .zynode-cache example.ts` for persistent project caching, or
`check --diagnostics=json example.ts` for structured diagnostics.

## Further reading

- [Compatibility and checking policies](../../docs/language-integrations/typescript/compatibility.md)
- [Architecture, memory and caching](../../docs/language-integrations/typescript/architecture.md)
- [Testing and conformance](../../docs/language-integrations/typescript/testing.md)
