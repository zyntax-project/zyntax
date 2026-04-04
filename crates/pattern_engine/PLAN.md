# Pattern Engine — Implementation Plan

## Insertion Point

Two compilation paths exist. Both need the engine in the same position.

### Path 1: `compile_typed_program` (`crates/compiler/src/lib.rs:1437`)

```
register_impl_blocks(&mut program)
generate_abstract_trait_impls(&mut program)
register_impl_blocks(&mut program)           // second pass for generated impls
                                             // ← engine.run() goes here
lowering_ctx.lower_program(&mut program)     // → HIR → Cranelift/LLVM
```

### Path 2: ZynML runtime (`crates/zyntax_embed/src/runtime.rs:1138`)

```
register_impl_blocks(&mut program)
generate_abstract_trait_impls(&mut program)
register_impl_blocks(&mut program)
                                             // ← engine.run() goes here
lowering_ctx.lower_program(&mut program)     // → HIR → Cranelift JIT
```

Both paths share the same shape: impl registration, then lowering. The engine inserts between them. `lower_program` already calls `run_type_checking()` internally (line 567), so the engine receives a type-checked program.

The engine takes `&mut TypedProgram` and the `TypeRegistry` (available as `program.type_registry` or as the `Arc<TypeRegistry>` created just before lowering). It returns the program mutated in-place.

---

## What Exists Today

| Component | Location | Status |
|---|---|---|
| `TypedNode<T>` | `typed_ast.rs:19` | `{ node: T, ty: Type, span: Span }` — no NodeId |
| `TypedProgram` | `typed_ast.rs:43` | `{ declarations: Vec<TypedNode<TypedDeclaration>>, span, source_files, type_registry }` |
| `TypedASTBuilder` | `typed_builder.rs` | Full coverage — all expression/statement/declaration types, fluent API |
| `TypeRegistry` | `type_registry.rs` | Types, traits, impls, aliases, coherence caches |
| `AnalysisContext` | `advanced_analysis.rs:40` | DFG, CFG, ownership, lifetime analysis. `analyze_program()` runs full pipeline |
| `EffectSystem` | `effect_system.rs:32` | Effect types, function signatures, inference context |
| `EffectAnalyzer` | `compiler/effect_analysis.rs` | Handler scope analysis, pure function validation, transitive propagation |
| `EffectCodegenContext` | `compiler/effect_codegen.rs` | Handler dispatch strategies (DirectCall, Inline, RuntimeDispatch) |
| `AsyncCompiler` | `compiler/async_support.rs` | State machine builder at HIR level — await detection, capture analysis, poll generation |
| `TypedEffect`, `TypedEffectHandler`, `TypedEffectOp` | `typed_ast.rs:250-335` | Full AST node types for algebraic effects |
| `TypedFunction.is_async`, `TypedExpression::Await` | `typed_ast.rs` | Async markers in the AST |
| `Span` | `source.rs:69` | `{ start: usize, end: usize }` — `Copy`, `Eq`, `Hash` |
| TypedAST walker | — | Does not exist. `LoweringContext` manually iterates `program.declarations`. |

---

## What Needs Building

### 1. `NodeId` assignment (`node_id.rs`)

`TypedNode` has no unique ID. The `MetadataTable` needs one.

Add a `NodeId` newtype (`u32`) and a pre-pass that walks `TypedProgram` depth-first, assigns a monotonic ID to every `TypedNode`, and stores the mapping in a `NodeIdMap`. The map is keyed by `Span` for parsed nodes. Synthesized nodes (produced by rewrites via `TypedASTBuilder`) get fresh IDs from the same counter.

`MetadataTable` keys on `NodeId`. The `NodeIdMap` is built once at the start of `engine.run()` and rebuilt each fixpoint iteration (since rewrites create new nodes).

### 2. Mutable AST walker (`walk.rs`)

The engine needs to walk `TypedProgram` depth-first, trying rewrites at each node. No visitor trait exists today.

Implement as a concrete function, not a trait — the engine is the only consumer:

```
walk_program(program, |node_kind, node, ctx| -> Option<RewriteOutput>)
```

Walk order: depth-first, children before parent (bottom-up). This means inner rewrites fire first; outer rewrites see already-transformed subtrees.

The walker must handle `RewriteOutput::Expand` by collecting new declarations into a pending buffer, inserted into `program.declarations` after the current walk completes. The next fixpoint iteration picks them up.

The walker descends into:
- `TypedDeclaration::Function` → body statements → expressions
- `TypedDeclaration::Class` → methods → bodies
- `TypedDeclaration::Impl` → method bodies
- `TypedDeclaration::Effect`, `TypedDeclaration::EffectHandler` → handler bodies
- `TypedStatement::If`, `While`, `For`, `Match` → nested blocks
- `TypedExpression::Call`, `Binary`, `If`, `Lambda`, `Block` → sub-expressions

### 3. Post-rewrite verification (`verify.rs`)

In debug builds, after each fixpoint iteration, run `TypeChecker::check_program()` on the transformed program. This catches rewrites that produce type-unsound AST at the specific iteration that broke things. Controlled by `EngineConfig.verify_after`.

Use the same `TypeChecker::with_options()` setup that `LoweringContext::run_type_checking()` uses (line 600).

---

## Implementation Phases

### Phase 0 — Crate skeleton + engine loop

**Create `crates/pattern_engine/`:**

```
src/
  lib.rs           pub mod + re-exports
  bindings.rs      Bindings struct
  context.rs       MatchContext, LoweringTarget
  pattern.rs       Pattern<T>, combinators (and/or/when/for_target)
  rewrite.rs       Rewrite<T>, RewriteOutput, Priority, RewriteBenefit
  pass.rs          PatternPass trait
  engine.rs        PatternEngine, EngineConfig, registration, finalize(), run()
  metadata.rs      MetadataTable
  node_id.rs       NodeId, NodeIdMap, assignment walk
  walk.rs          depth-first mutable walker
  trace.rs         FiredRewrite, trace emission
  verify.rs        post-rewrite type checker call
```

**Cargo.toml dependencies:**
- `zyntax_typed_ast` — TypedProgram, TypedASTBuilder, TypeRegistry, AnalysisContext, EffectSystem
- No dependency on `zyntax_compiler` — the engine operates purely on TypedAST

**Wire into the pipeline:**

In `crates/compiler/src/lib.rs` after `register_impl_blocks`, before `lowering_ctx.lower_program`:
```rust
let mut engine = PatternEngine::new(EngineConfig {
    target: LoweringTarget::Cpu,
    max_iterations: 64,
    trace: cfg!(debug_assertions),
    verify_after: cfg!(debug_assertions),
});
engine.finalize()?;
let _result = engine.run(&mut program, &registry);
```

Same in `crates/zyntax_embed/src/runtime.rs` at the equivalent point.

With no passes registered, `engine.run()` walks the program, fires nothing, returns `EngineResult { changed: false, iterations: 1, rewrites_fired: vec![] }`.

**Test:** Register one expression rewrite matching `IntLiteral(42)` → `IntLiteral(0)`. Build a `TypedProgram` with a `println(42)` call. Run engine. Assert the literal changed. Assert `rewrites_fired.len() == 1`.

### Phase 1 — Normalization pass

**Create `crates/passes/normalization/`:**

Three rewrites at `Priority::NORMALIZATION (100)`:

| Rewrite | Match | Output |
|---|---|---|
| `flatten_nested_blocks` | `Block` whose only content is another `Block` | Hoist inner statements to outer |
| `unit_return_explicit` | `Function` returning `Unit` with no terminal `Return` | Append `Return(None)` |
| `dead_let_elimination` | `Let(name)` where `AnalysisContext` DFG shows zero uses and initializer is pure | `Delete` |

`dead_let_elimination` is the first rewrite that uses `MatchContext.analysis`. This exercises the `AnalysisContext::analyze_program()` integration — the DFG must be rebuilt each iteration so deleted lets don't leave stale edges.

**Test:** Verify idempotency — running the engine twice produces the same AST as running it once.

### Phase 2 — Algebraic effects pass

**Create `crates/passes/algebraic_effects/`:**

Three rewrites at `Priority::SEMANTIC (200)`:

| Rewrite | Match | Output |
|---|---|---|
| `effect_op_to_continuation` | `Call` where callee resolves to a `TypedEffectOp` | `Replace` with handler dispatch call |
| `effect_decl_to_vtable` | `TypedDeclaration::Effect` | `Expand` into `Class(OpTable)` |
| `handler_decl_to_impl` | `TypedDeclaration::EffectHandler` | `Expand` into vtable instance + handler run function |

The `MatchContext.effects` field wraps the existing `EffectSystem` from `typed_ast/effect_system.rs`. Handler scope resolution uses `EffectAnalyzer` results from `compiler/effect_analysis.rs`, exposed through `MatchContext.analysis`.

After this pass, the program contains no `TypedEffect` or `TypedEffectHandler` declarations — they've been rewritten into classes and functions that the existing `LoweringContext` can lower to HIR without special-casing effects.

**Dependencies:** `normalization` (effects pass assumes normalized control flow)

### Phase 3 — Async-IR pass

**Create `crates/passes/async_ir/`:**

Two rewrites at `Priority::SEMANTIC (200)`:

| Rewrite | Match | Output |
|---|---|---|
| `async_fn_to_state_machine` | `Function` where `is_async == true` and body contains `Await` | `Expand` into `Enum(StateEnum)` + `Class(FutureStruct)` + `Function(poll_fn)` |
| `await_expr_to_poll` | `TypedExpression::Await(inner)` where `inner.ty` implements `Future` | `Replace` with `Match(poll_dispatch)` |

The state machine construction uses `TypedASTBuilder` to synthesize the enum, struct, and poll function at TypedAST level. The existing `AsyncCompiler` in `async_support.rs` operates at HIR level — once the async-ir pass is complete, `AsyncCompiler` is no longer needed because async/await is already eliminated from the TypedAST before lowering.

**Dependencies:** `normalization`; if effects can be async, also `algebraic-effects`.

### Phase 4 — Target-specific passes

**NVPTX pass** (`Priority::TARGET (400)`, target: `LoweringTarget::Nvptx`):
- Annotate `@kernel` functions with `NvptxMetadata` (thread block dims, shared memory) in the `MetadataTable`
- No AST rewriting — purely metadata population for the NVPTX backend to consume

**RTLIL pass** (`Priority::TARGET (400)`, target: `LoweringTarget::Rtlil`):
- Clock domain analysis, combinational/sequential classification
- Metadata-only

These passes use `Pattern::for_target()` so they never fire when targeting CPU.

---

## Crate Dependency Graph

```
zyntax_typed_ast
       ↑
pattern_engine          (core framework)
       ↑
  ┌────┼────────┐
  │    │        │
normalization  algebraic_effects  async_ir    (passes)
  │    │        │
  └────┼────────┘
       ↑
zyntax_compiler         (wires engine.run() into the pipeline)
       ↑
zyntax_embed            (wires engine.run() into the runtime path)
```

No pass crate depends on `zyntax_compiler`. Passes only depend on `pattern_engine` and `zyntax_typed_ast`. The compiler and runtime wire the engine into their respective pipelines.
