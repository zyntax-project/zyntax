# Pattern Engine — Implementation Plan

## Spec Review

The spec is architecturally sound. The term-rewriting model, fixpoint loop, priority-based first-match-wins strategy, and external metadata table are all good choices for this stage of the compiler. Three adjustments are needed based on what actually exists in the codebase.

### What the spec gets right

- **Pipeline position** — after TypeChecker, before lowering. This is exactly where the existing `LoweringContext` sits. The engine slots in naturally.
- **MetadataTable** — external to TypedNode. This is the right call; `TypedNode<T>` has no ID field and shouldn't grow per-target data.
- **Pass ordering via dependencies** — topological sort catches cycles at registration time. Better than implicit priority-only ordering.
- **Analysis rebuild per iteration** — `AnalysisContext::analyze_program()` exists and is real. Rebuilding per iteration is correct; incremental invalidation can come later.
- **Effect system types** — `TypedEffect`, `TypedEffectHandler`, `TypedEffectOp` all exist in `typed_ast.rs` with full structure. Effect analysis (`effect_analysis.rs`) and codegen (`effect_codegen.rs`) are real implementations, not stubs.
- **Async state machine** — `AsyncCompiler` in `async_support.rs` already builds state machines with await point detection, capture analysis, and poll generation. The async-ir pass can delegate to this.
- **TypedASTBuilder** — comprehensive. Covers all expression, statement, and declaration types with a fluent API. Rewrites can use it directly.

### What needs adjustment

**1. TypedNode has no NodeId**

`TypedNode<T>` is `{ node: T, ty: Type, span: Span }`. There is no unique ID. The spec's `MetadataTable` keys on `NodeId` which doesn't exist.

Options:
- **A.** Add a `NodeId` field to `TypedNode` (invasive — touches every node constructor)
- **B.** Use `Span` as the key (works if spans are unique per node, which they are for parsed programs but not for synthesized nodes)
- **C.** Assign IDs in a pre-pass walk before the engine runs (the `node_id.rs` file in the spec). IDs live in a side table `HashMap<*const (), NodeId>` keyed by pointer, or in a parallel vec built during the walk.

**Recommendation:** Option C. The spec already has `node_id.rs` in the crate structure. Implement it as a pre-pass that walks `TypedProgram` once, assigns monotonic IDs, and stores them in a `NodeIdMap` side table. The `MetadataTable` then keys on these IDs. Synthesized nodes from rewrites get fresh IDs from the same counter.

**2. No TypedAST walker exists**

The compiler's `LoweringContext` manually walks `TypedProgram` by iterating `program.declarations` and pattern-matching each `TypedDeclaration` variant. There is no generic visitor or mutable walker trait.

The spec lists `walk.rs` in the crate structure but doesn't define the walker interface. This is the critical missing piece.

**Recommendation:** Implement a bottom-up mutable walker as a trait:

```rust
pub trait TypedWalker {
    fn walk_program(&mut self, program: &mut TypedProgram);
    fn walk_declaration(&mut self, decl: &mut TypedNode<TypedDeclaration>);
    fn walk_statement(&mut self, stmt: &mut TypedNode<TypedStatement>);
    fn walk_expression(&mut self, expr: &mut TypedNode<TypedExpression>);
}
```

The engine implements this trait. Each `walk_*` method iterates the applicable rewrites, tries to match, and applies the first successful rewrite. Child nodes are walked recursively before the parent (bottom-up), so inner rewrites fire first and outer rewrites see already-transformed children.

**3. Async transformation lives at HIR level, not TypedAST level**

The spec proposes `async_fn_to_state_machine` as a TypedAST rewrite that produces `Enum(StateEnum)`, `Class(FutureStruct)`, `Function(poll_fn)`. But the existing `AsyncCompiler` in `async_support.rs` operates on `HirFunction`, not `TypedFunction`. It builds `AsyncStateMachine` with `AsyncState` structs containing `HirInstruction` and `HirTerminator`.

Moving this to TypedAST level means reimplementing the state machine builder at a higher abstraction level.

**Recommendation:** Keep async transformation at HIR level for now. The pattern engine can handle the *detection* and *marking* phase (identify async functions, annotate suspension points in metadata), and the actual state machine construction stays in `async_support.rs` where it already works. The `async-ir` pass becomes:

1. **At TypedAST level (pattern engine):** Mark async functions and await sites in the MetadataTable. Validate that awaited expressions implement Future.
2. **At HIR level (existing):** `AsyncCompiler::compile_async_function()` consumes the metadata and builds the state machine.

This avoids duplicating the state machine builder and leverages existing working code.

---

## Implementation Order

### Phase 0 — Foundation (the crate skeleton)

Create `crates/pattern_engine/` with the core types. No passes yet, just the framework.

Files:
- `Cargo.toml` — depends on `zyntax_typed_ast`
- `src/lib.rs` — re-exports
- `src/bindings.rs` — `Bindings` struct (typed map from names to AST fragments)
- `src/context.rs` — `MatchContext`, `LoweringTarget`
- `src/pattern.rs` — `Pattern<T>` with `and`/`or`/`when`/`for_target` combinators
- `src/rewrite.rs` — `Rewrite<T>`, `RewriteOutput`, `Priority`, `RewriteBenefit`
- `src/pass.rs` — `PatternPass` trait
- `src/metadata.rs` — `MetadataTable` keyed by `NodeId`
- `src/node_id.rs` — `NodeId` type + assignment walk
- `src/walk.rs` — `TypedWalker` trait + default depth-first implementation
- `src/engine.rs` — `PatternEngine`, `EngineConfig`, registration, `finalize()`, `run()`
- `src/trace.rs` — `FiredRewrite`, rewrite trace emission
- `src/verify.rs` — post-rewrite type soundness check (calls `TypeChecker`)

**Milestone:** `PatternEngine::new()` compiles. Registration API works. `run()` walks the program and returns `EngineResult` with `changed: false` (no passes registered).

**Test:** Register a no-op pass with one expression rewrite that matches `IntLiteral(42)` and replaces it with `IntLiteral(0)`. Verify the program is transformed. Verify `EngineResult.rewrites_fired` contains the rewrite.

### Phase 1 — Normalization pass

Implement `crates/passes/normalization/` with the three rewrites from the spec:
- `flatten_nested_blocks`
- `unit_return_explicit`
- `dead_let_elimination` (requires DFG query — uses `AnalysisContext`)

This is the simplest pass and exercises every part of the engine: pattern matching, AST mutation, the walker, analysis integration, and the fixpoint loop.

**Test:** Write ZynML programs that trigger each rewrite. Verify before/after AST structure. Verify idempotency (running twice produces same result).

### Phase 2 — Algebraic effects pass

Implement `crates/passes/algebraic_effects/`. This is the first semantically complex pass and the primary motivator for the pattern engine.

Leverage existing infrastructure:
- `TypedEffect`/`TypedEffectHandler`/`TypedEffectOp` — already defined in typed_ast
- `EffectAnalyzer` in `effect_analysis.rs` — already does handler scope analysis
- `EffectCodegenContext` in `effect_codegen.rs` — already has handler dispatch strategies

The pass translates these into plain TypedAST constructs (classes, functions, vtables) that the existing lowering pipeline already handles.

**Key integration:** The `MatchContext.effects` field should wrap the existing `EffectAnalyzer` results, not reimplement effect scope analysis.

### Phase 3 — Async-IR pass (marking only)

Implement `crates/passes/async_ir/` as a metadata-only pass:
- Walk async functions, record suspension points in `MetadataTable`
- Validate Future trait implementations
- Does NOT build state machines (that stays at HIR level in `AsyncCompiler`)

Wire the metadata into the existing `LoweringContext` → `AsyncCompiler` path so that async lowering consumes pattern engine metadata instead of re-analyzing.

### Phase 4 — Target-specific passes

Once the framework is proven with Phases 1-3, add target-specific passes:
- **NVPTX pass** — annotate kernel functions, thread hierarchy in metadata
- **RTLIL pass** — clock domain analysis, combinational/sequential classification

These are metadata-heavy (they don't rewrite the AST much, they populate the MetadataTable for backend consumption).

---

## Open Questions

1. **Should `RewriteOutput::Expand` inject declarations at program scope or at the nearest enclosing scope?** The spec says program scope, which works for effect-to-vtable. But async state machines may need to inject structs in the same module scope as the original function, not necessarily top-level.

2. **Should the walker be top-down or bottom-up?** The spec says depth-first but doesn't specify direction. Bottom-up (children first) means rewrites see already-normalized children. Top-down means a rewrite can prevent walking into children it's about to delete. The normalization pass wants bottom-up; the async pass might want top-down. Consider making direction configurable per pass.

3. **How does `Expand` interact with the walker?** When a rewrite expands a declaration into multiple declarations, the walker needs to know where to insert them and whether to walk the new declarations in the current iteration. Inserting into a `pending_declarations` buffer and processing them in the next iteration avoids mutation-during-iteration issues.

4. **Should `Bindings` use `&'static str` or `InternedString` for keys?** The spec uses `&'static str` which is ergonomic for hardcoded pattern names. But if patterns are ever generated dynamically (e.g., from a DSL), `InternedString` would be more flexible. Start with `&'static str` and migrate if needed.
