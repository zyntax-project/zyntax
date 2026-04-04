# ZynML Roadmap

## Current Status

ZynML compiles and runs programs through the full pipeline:
**ZynML source -> PEG parse -> Typed AST -> HIR (SSA) -> Cranelift IR -> native code**

### Working Features

| Feature | Status | Example |
|---------|--------|---------|
| String/int/float literals | Done | `println("hello")`, `println(42)` |
| Let bindings | Done | `let x = 10` |
| Arithmetic (`+`, `-`, `*`, `/`, `%`) | Done | `let z = x + y * 2` |
| String concatenation (`+`) | Done | `println("Hello, " + name)` |
| User-defined functions | Done | `def add(x, y) { return x + y }` |
| Dynamic return type inference | Done | Functions with `return <expr>` but no annotation |
| If/else | Done | `if x > 0 { ... } else { ... }` |
| While loops | Done | `while i < 10 { ... }` |
| For loops with range() | Done | `for i in range(0, 10) { ... }` |
| F-strings (inside print) | Done | `println(f"x = {x}")` |
| Comparisons (`<`, `>`, `<=`, `>=`, `==`, `!=`) | Done | `if a > b { ... }` |
| Boolean literals | Done | `let flag = true` |
| Tensor creation (arange, zeros, ones) | Done | `let a = Tensor::arange(1.0, 5.0, 1.0)` |
| Tensor arithmetic (`+`, `-`, `*`) | Done | `let c = a + b` |
| Tensor reductions (sum, mean) | Done | `println(a.sum())` |
| Tensor Display (f-string interpolation) | Done | `println(f"Tensor: {a}")` |
| compute() with @kernel SIMD | Done | `compute { @kernel elementwise ... }` |
| @kernel reduce + yield | Done | `@kernel reduce ... yield val` |
| Import prelude / tensor | Done | `import prelude` |
| Struct declarations | Done | `struct Point { x: i64, y: i64 }` |
| Trait declarations (parsing) | Done | `trait Display { ... }` |
| Impl blocks (parsing) | Done | `impl Display for Tensor { ... }` |

### Working Examples

```
hello_simple.zynml        # Tensor ops, f-strings, arithmetic
hello.zynml               # Full tensor demo with f-strings
hello_no_fstring.zynml    # Tensor demo without f-strings
hello_basic.zynml         # Basic println
compute_simd.zynml        # @kernel SIMD (elementwise + reduce)
test_01 through test_13   # Unit test examples (all pass)
```

---

## Known Issues

### P0 — Correctness Bugs

**2D tensor shape corruption**
- `Tensor::zeros(2, 3)` displays shape `[2, 4294967299]` instead of `[2, 3]`
- The value `4294967299 = 2^32 + 3` suggests an i32/i64 overflow in multi-dim shape representation
- 1D tensors are unaffected
- Location: likely in `plugins/zrtl_tensor/` shape handling or the Cranelift argument passing for multi-arg tensor constructors

**Function call return with implicit last-expression**
- `def add(a, b) { a + b }` (no explicit `return`) — the last expression value is wrapped as `Return(Some(expr))` by the CFG builder, but without explicit `return` the value may be discarded as void
- Workaround: always use explicit `return` in functions that return values

**test_06_function_call.zynml prints opaque pointers**
- `def add(a: int, b: int): int { a + b }` — the `: int` and `: int` type annotation syntax is parsed but the grammar maps `int` to an unresolved type, so params are treated as Dynamic
- The `a + b` implicit return doesn't propagate correctly without explicit `return`

### P1 — Missing Language Features

**Type annotations in function signatures**
- `def foo(x: i32) -> i32` is not fully supported by the Grammar2 parser path
- Parameters default to `Type::Any` (Dynamic) with a W0001 warning
- Return types default to void unless `return <expr>` is detected (W0002 warning)
- Impact: all values flow as i64 at runtime, losing type precision

**F-strings outside print calls**
- `let s = f"hello {x}"` does not work
- F-strings are only intercepted when directly inside `println`/`print`/`eprintln`/`eprint` calls
- The SSA builder uses a "closure" approach that inlines individual `print_dynamic()` calls for each f-string part
- A general-purpose f-string would need to concatenate parts into a single string value

**Standalone string methods**
- `"hello".len()`, `s.upper()`, etc. are not supported
- String type (`Primitive(String)`) doesn't go through trait dispatch
- String concat via `+` works (dispatches to `$IO$string_concat`)

**Generic type instantiation**
- `Option<T>`, `Result<T,E>`, `List<T>` are defined in the prelude but their methods don't fully lower
- Struct field access on generic types fails because pre-registered types have 0 fields
- The `collections.zynml` example partially runs but stops early

**Pattern matching**
- `match` expressions are parsed by the grammar but not lowered to HIR/Cranelift
- No exhaustiveness checking

**Closures / lambdas**
- `let f = |x| x + 1` — parsed but not compiled
- Lambda capture, closure conversion not implemented

**Error handling**
- `try`/`catch` or `Result` propagation (`?` operator) not implemented
- `Result<T,E>` type exists in prelude but is not usable at runtime

### P2 — Tensor Runtime Gaps

**Missing tensor operations in runtime dispatch**
- The grammar (`ml.zyn`) registers 65+ tensor operations as `@builtin` mappings
- The ZRTL tensor plugin (`plugins/zrtl_tensor/`) implements most of them
- But the SSA builder's method dispatch doesn't resolve all `Tensor::method()` calls to the correct `$Tensor$method` symbol
- Specifically: `Tensor::linspace()`, `Tensor::reshape()`, `Tensor::transpose()` fail with "can't resolve symbol"
- Root cause: static method calls (`Type::method()`) vs instance method calls (`value.method()`) have different dispatch paths

**Tensor matmul (`@` operator)**
- Grammar defines `@` as `MatMul` operator
- Trait dispatch requires `MatMul::matmul` implementation
- Not wired to `$Tensor$dot` in the ZRTL plugin

**Neural network example**
- `neural_network.zynml` compiles without error but produces no output
- Struct method calls (`self.weights`, `self.forward()`) are not fully lowered
- impl block methods are registered but the function bodies reference `self` which isn't resolved

### P3 — Infrastructure / DX

**Compiler warnings for stdlib generics**
- Generic types from prelude (`Option<T>`, `Result<T,E>`, type params `T`, `U`) generate "Could not resolve type" traces during trait lowering
- These are suppressed for single-letter type names but still clutter trace logs

**LLVM backend tests broken**
- `llvm_backend_integration_tests.rs` references outdated struct layouts (HashMap vs IndexMap, missing fields)
- Tests are disabled pending a rewrite of test helpers

**Two parsing paths**
- `parse_to_json` / `parse_with_builder` uses the old pest VM path
- `parse_to_typed_program` uses Grammar2 (packrat memoized, used by `run_file`)
- Grammar2 doesn't handle all constructs (top-level `let`, some `impl` block patterns)
- F-strings work on both paths now

---

## Roadmap

### Phase 1 — Language Completeness

1. **Type annotations** — Wire Grammar2 to parse `param: Type` and `-> ReturnType` syntax, map `int`/`float`/`str`/`bool` to proper primitive types
2. **Implicit return** — Make last-expression-in-block return the value (like Rust) without requiring explicit `return`
3. **F-strings as values** — Support `let s = f"..."` by concatenating parts into a string at compile time
4. **Pattern matching** — Lower `match` arms to conditional branches in HIR
5. **Closures** — Implement closure conversion (capture environment, generate trampoline functions)

### Phase 2 — Type System

1. **Generic instantiation** — Monomorphize `Option<i64>`, `List<Tensor>` etc. at call sites
2. **Struct field access** — Fix generic struct field resolution for stdlib types
3. **Method resolution** — Unify static (`Type::method()`) and instance (`value.method()`) dispatch paths
4. **Trait method dispatch** — Wire impl block methods so `self.field` and `self.method()` work in method bodies

### Phase 3 — Tensor Runtime

1. **Fix 2D shape corruption** — Debug the i32/i64 overflow in multi-dimensional tensor constructors
2. **Static method dispatch** — Wire `Tensor::linspace()`, `Tensor::reshape()`, etc. through the SSA builder
3. **Matmul operator** — Connect `@` to `$Tensor$dot`
4. **Neural network support** — Get `neural_network.zynml` producing output (requires Phase 2 struct/method work)

### Phase 4 — Ergonomics

1. **String methods** — `.len()`, `.upper()`, `.lower()`, `.split()`, `.strip()` via ZRTL string plugin
2. **Error handling** — `try`/`catch` or `?` operator for Result propagation
3. **Collections** — Working `List`, `Dict`/`HashMap` with iteration support
4. **Better error messages** — Source-location annotations on all diagnostics (currently only some have spans)
5. **REPL** — Interactive mode for quick tensor experiments
