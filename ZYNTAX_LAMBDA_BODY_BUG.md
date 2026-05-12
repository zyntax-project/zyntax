# Zyntax: lambda body drops `Call` / `MethodCall` / `Block` expressions

Closure bodies whose only content is function or method calls compile to
no-op functions that return `0`. Authors get no diagnostic — the lambda
just silently doesn't execute its statements.

## Where

`crates/compiler/src/ssa.rs`:

- `fn translate_lambda_expr(&self, …)` — around **line 8493**
- `fn translate_closure(&mut self, …)` — around **line 8292**

## What the code does today (upstream rev `5b0dfab7…`)

### `translate_lambda_expr`

Only three `TypedExpression` variants are translated:

- `Literal` → emit a `HirValue { kind: Constant(…) }`
- `Variable` → look up in lambda params / outer captures
- `Binary` → recurse + emit `HirInstruction::Binary`

Everything else hits the fallback:

```rust
_ => {
    // Fallback - return constant 0
    let val_id = HirId::new();
    func.values.insert(
        val_id,
        crate::hir::HirValue {
            id: val_id,
            ty: result_ty.clone(),
            kind: crate::hir::HirValueKind::Constant(
                crate::hir::HirConstant::I32(0),
            ),
            uses: HashSet::new(),
            span: None,
        },
    );
    Ok(val_id)
}
```

So `Call`, `MethodCall`, `Block`, `Field`, `If`, `Index`, etc. inside a
lambda body all silently become `Constant::I32(0)` — the source is
discarded.

### `translate_closure` (the `TypedLambdaBody::Block` arm)

Around line 8435, block-bodied lambdas hit the *same* placeholder
without going through `translate_lambda_expr` at all:

```rust
TypedLambdaBody::Block(_block) => {
    // Block body - for now, just return 0
    let val_id = HirId::new();
    lambda_func.values.insert(
        val_id,
        crate::hir::HirValue {
            id: val_id,
            ty: return_type.clone(),
            kind: crate::hir::HirValueKind::Constant(crate::hir::HirConstant::I32(0)),
            uses: HashSet::new(),
            span: None,
        },
    );
    val_id
}
```

So `|| { stmt; stmt; … }` (the `Block` body shape) compiles to a no-op
even before `translate_lambda_expr` gets a chance — the whole body is
dropped one level higher.

## Minimal repro

Zyntax-only, no embedder needed:

```text
extern fn sink(value: i32)

fn main() {
    let f = || { sink(42) }
    f()
}
```

Expected: `sink(42)` runs.  
Observed: the JIT'd closure ignores `sink(42)`, evaluates to `0`,
and returns. `sink` is never called.

## Why it matters

Closures in Zyntax-embedded DSLs typically wrap side-effectful logic:
host calls, signal writes, event dispatches, etc. The bodies are
predominantly `Call`s — exactly the variant the current implementation
discards. Even a single-statement closure of the form
`|| { do_thing() }` produces a no-op function.

## Proposed fix

Two viable shapes; both yield the same end-user behaviour.

### (A) Run the lambda body through the existing `translate_expression`

Today `translate_lambda_expr` is a hand-rolled mini-translator
maintained in parallel with the real one. Anytime
`translate_expression`'s `Call` arm grows a new special case
(f-string flattening, effect resume, indirect calls), the lambda
translator stays behind.

The straightforward fix is to lift the lambda body into a regular
`HirFunction` and run it through `translate_expression`, the same way
FSM tick guards already are. The `self.function` context-swap that
that requires can mirror what `translate_closure` already does when it
builds `lambda_func`. After the swap, every variant `translate_expression`
handles works inside a closure too — and stays in sync forever.

### (B) Add explicit handlers for `Block`, `Call`, `MethodCall`, `Field`

If a full delegate is too invasive for one PR, the minimum to unblock
DSL embedders is:

- `TypedExpression::Block(b)` — iterate `b.statements`, translate each
  `Statement::Expression(e)` via recursion (and similar for `Let`).
- `TypedExpression::Call(call)` — recursively translate args, emit
  `HirInstruction::Call { callee: HirCallable::Symbol(name), args, … }`
  for the simple direct-call case (callee is `Variable`).
- `TypedExpression::MethodCall(mc)` — translate receiver + args, emit
  the appropriate call instruction.
- `TypedExpression::Field(f)` — emit `ExtractValue` like the main
  translator does at ssa.rs:3878.

The `TypedLambdaBody::Block` arm in `translate_closure` (line 8435)
should iterate the block's statements the same way.

This doesn't carry the f-string / effect-resume special cases the main
translator has, but covers every Blinc-style "call externs inside a
closure" use case, which is the most common one.

## Captures

Out of scope for this fix. Today closures that don't capture (touching
globals via externs) cover the most common embedder use cases. The
existing `outer_captures` / `translate_closure_environment` machinery
can continue to be no-op'd for the bodies this patch unblocks.

## Status of the local checkout at `/Users/amaterasu/Vibranium/zyntax`

The local checkout's two-PR delta does fix the lambda-body drop:
`crates/compiler/tests/closure_body_lowering_tests.rs` now covers
both the original drop and the sibling-fn regression case and all 4
tests pass against the local code.

**The end-to-end regression still reproduces in Blinc, though, and the
new synthetic tests don't catch it.** Re-pointed Blinc at the local
checkout via `[patch]` and reran the probe. Same shape Blinc uses:

```text
signal count: i32
view {
    Div(on_click = || { count.set(count.get() + 1) }) { Text("+1") }
}
```

Instrumented `zyntax_embed::runtime::compile_typed_program` to dump
`hir_module.functions` at three points + dumped the typed program
right after `parse_to_typed_ast`. Result:

```
[BLINC] typed program after parse_to_typed_ast — 20 functions:
  render_view                       ← NON-extern, body present
  __set_overlay_corner_radius__ (extern)
  __signal_get_i32 (extern)
  __set_overlay_border_width__ (extern)
  text (extern)
  $Blinc$text (extern)
  __set_overlay_border_color__ (extern)
  __signal_get_string (extern)
  __signal_get_f64 (extern)
  __new_child_list__ (extern)
  __push_child__ (extern)
  text_int (extern)
  $Blinc$text_int (extern)
  __set_overlay_opacity__ (extern)
  __set_overlay_bg__ (extern)
  __new_style_overlay__ (extern)
  __fstring_format__ (extern)
  $Blinc$format_int (extern)
  string_concat (extern)
  $Blinc$string_concat (extern)

[BLINC-DEBUG] after lower_typed_program: 19 fns (0 non-extern). Names:
  [all 19 externs above, render_view missing]
[BLINC-DEBUG] after apply_krio_async_lowering: 19 fns (0 non-extern). ← unchanged
[BLINC-DEBUG] after apply_krio_effect_lowering: 19 fns (0 non-extern). ← unchanged
compile result: Ok([])
render_main result: Err(Backend("Function not found: render_view"))
```

### What this tells us

- **The regression is inside `lower_typed_program`**, not in
  `apply_krio_async_lowering` / `apply_krio_effect_lowering`. The
  module is already missing `render_view` at the first snapshot.
- The typed program input is healthy — `render_view` is present,
  non-extern, with a body, alongside 19 extern declarations.
- The lowering accepts the program without error (no
  `CompilerError::Lowering(...)` surfaced), but silently drops the
  one non-extern function on the way to HIR.
- Both synthetic regression tests pass because they don't replicate
  this shape — they have at most ~3 declarations and don't go through
  the `parse_with_signatures` extern-injection path that gets Blinc
  to 19 externs.

### Suggested minimum to reproduce in a Zyntax-only test

Build a `TypedProgram` with:

1. A non-extern `render_view`: `fn render_view(): i64 { return $Blinc$Div$view(0, 0, "", 0) }`.
2. ~15-20 sibling extern function declarations covering the union of
   the names from Blinc's dump above (the specific names don't
   matter, only the count + the mix of `$Blinc$…` /
   `__set_overlay_…` / `__signal_get_…` shapes).
3. A `Lambda` expression nested in `render_view`'s body.

Lower it and assert `render_view` survives. Today against the local
checkout that program would produce a module with the 19 externs and
no `render_view` — same observed behaviour as Blinc.

### Blinc-side workaround

`[patch."https://github.com/darmie/zyntax"]` block in Blinc's
workspace `Cargo.toml` is currently active so this is reproducible
without bumping the rev. Once the lowering regression is resolved,
the patch can stay in (the closure-body fix is the thing Blinc
actually needs). If the patch needs to be reverted to unblock
something else in the meantime, comment the three lines under
`[patch."https://github.com/darmie/zyntax"]` — Blinc reverts to the
silent-drop-but-renders-fine behaviour against upstream rev
`5b0dfab7…`.

---

## Update — 2026-05-12: silent-drop mechanism identified + fixed

Found the mechanism. `crates/compiler/src/lowering.rs::lower_declaration`
was swallowing every `lower_function` error at `log::trace!` level
and removing the function from the symbol table:

```rust
if let Err(e) = self.lower_function(func) {
    log::trace!("[LOWERING WARN] Skipping function '{}': {:?}", func_name, e);
    self.symbols.functions.remove(&func.name);
}
```

`log::trace!` is invisible without `RUST_LOG=trace`. The function
disappeared from the module, compilation kept going as if nothing
happened, the caller saw `Ok([])`. That's the `render_view`-vanishes
mechanism.

The blanket swallow was historical generic-function leniency:
monomorphic instantiations come from call sites via
`monomorphize_module`, so failing to lower the original generic
decl is benign. Applying that same leniency to non-generic
non-extern functions hid genuine bugs.

### Fix

`lower_declaration` now distinguishes:

- **Generic functions** (non-empty `type_params`): keep the
  `log::trace!` swallow — instantiations are emitted at call sites
  by `monomorphize_module`, so the original generic skip is
  harmless.
- **Non-generic non-extern functions**: propagate the error
  directly via `?`. Compilation fails loudly with the underlying
  SSA / lowering error instead of producing a module silently
  missing functions.

`tests/expression_lowering_tests.rs::test_matmul_missing_impl_reports_clear_error`
was updated to match its own name — it now asserts that an
unresolvable `MatMul` impl produces `CompilerError::Analysis(...)`,
not a silent drop.

### What Blinc should do next

Re-run the failing program against the latest local checkout.
Compilation will either:

1. **Fail with the actual `lower_function` error** that was
   silently dropped before. Share that error here so we can chase
   it directly (likely a method-resolution path through the
   lambda's captured `count.set` / `count.get`).
2. **Succeed entirely** — meaning the upstream changes are
   sufficient on their own.

Either way the path forward is unambiguous now: no more `Ok([])`
hiding the real cause.

---

## Update — 2026-05-12 (Blinc side): silent-drop fix confirmed, next layer revealed

The `lower_declaration` fix lands cleanly. Re-ran a minimal probe
against the latest local checkout:

```rust
// crates/blinc_dsl_core/examples/_probe_closure.rs
let res = dsl.compile_source(
    r##"
    signal count: i32
    view {
        Div(on_click = || { count.set(count.get() + 1) }) { Text("+1") }
    }
    "##,
    "probe_closure.blinc",
);
```

**Before the fix** (`log::trace!` swallow): `compile_source` returned
`Ok([])`, `render_main` then failed with `Backend("Function not found:
render_view")`. The lowering error was invisible.

**After the fix**: `compile_source` now returns

```
Err(Compile("Execution error: Lowering error: Analysis(\"Cannot access fields on non-struct type: Any\")"))
```

So the silent-drop mechanism is gone — exactly as the update intended.
The error that was previously being swallowed is now front and centre:
something inside the lambda body is trying to do field/method access
on a value typed as `Any`.

### Likely site

This is the path your update predicted: "method-resolution path through
the lambda's captured `count.set` / `count.get`."

Inside the lambda `|| { count.set(count.get() + 1) }`, `count` is an
outer signal that is *not* a typed struct — Blinc's signal substrate
exposes it via `__signal_get_i32` / `__signal_set_i32` externs, not as
a struct with `.get` / `.set` methods. The Blinc DSL has a post-parse
pass `resolve_signal_calls` that rewrites `<signal>.get()` /
`<signal>.set(v)` into direct extern calls before lowering.

When the lambda body is processed, it looks like the rewrite either
(a) isn't seeing inside the lambda body, or (b) the lambda body is
being analyzed with `count` typed as `Any` before the rewrite gets a
chance, so the analyser sees `Any.set(...)` and bails with the new
"Cannot access fields on non-struct type" error.

If you can share the file:line where this analysis error is raised
(it's "Cannot access fields on non-struct type: Any" — likely in
`crates/compiler/src/...` near a field-access or method-resolution
arm), we can narrow whether the fix is Blinc-side (run the signal
rewrite earlier / inside lambda bodies) or Zyntax-side (lambda body
analysis missing some context other top-level statements have).

### Repro

The probe above is the minimal repro — single signal, single lambda
calling `.set` / `.get` on it. No FSMs, no view-renderer, no
component, no externs other than what `BlincDsl::new()` registers.

The `[patch."https://github.com/darmie/zyntax"]` block in Blinc's
`Cargo.toml` remains active.

---

## Update — 2026-05-12 (later): the captured-type bug is patched

Confirmed your diagnosis. `get_field_index` (in
`crates/compiler/src/ssa.rs:6863`) was the error site, and the
root cause was on my side: the lambda-body context swap I added
in the previous fix mirrored the captured variable's HIR-side
type into `self.var_types`, but did NOT mirror the TypedAST-side
type into `self.var_typed_ast_types`.

`process_statement` / `translate_expression`'s field-access and
method-resolution arms read `var_typed_ast_types` to look up the
receiver's nominal type. With no entry, they default to
`Type::Any`, and `get_field_index` raises:

    Cannot access fields on non-struct type: Any

That's exactly what your `_probe_closure.rs` saw. The fix at
`ssa.rs:8533–8546` now copies the matching entry from the
saved outer state (`saved_var_typed_ast_types`) into the
lambda's `var_typed_ast_types` alongside the existing
`var_types` copy. Field / method resolution inside the lambda
body now sees the same nominal type the outer scope did.

So `count` inside `|| { count.set(count.get() + 1) }` should
now be visible to Blinc's `resolve_signal_calls` rewrite as the
signal type it actually is, and the rewrite to
`__signal_get_i32 / __signal_set_i32` should fire. No
Blinc-side change required.

Re-run `_probe_closure.rs` against the latest checkout and let
me know what happens. If you still see the `Cannot access
fields on non-struct type` error, share the new `count.ty` value
the error surfaces — that'll tell us whether the captured-type
mirror is still wrong, or whether `resolve_signal_calls` itself
isn't recursing into lambda bodies for some other reason.

---

## Update — 2026-05-12 (later still): cranelift indirect-call now surfaces too

Blinc-side now reports:

```
zyntax_compiler::cranelift_backend: Indirect call: function pointer
HirId(...) not in value_map
```

Three sites in `cranelift_backend.rs` had the same anti-pattern as
`lower_declaration`'s silent swallow:

1. `HirCallable::Indirect` in the main call-emit arm (line 2444):
   `warn!` + `iconst(0)` → `call_indirect` into address 0 → silent
   SIGSEGV at runtime, no path back to the cause.
2. `HirInstruction::IndirectCall` (line 2857): `log::trace!` + the
   same `iconst(0)` fallback.
3. `HirCallable::Indirect` in `FuncRef` emit (line 5770):
   `self.value_map[func_ptr]` bare indexing → panic with an
   unhelpful out-of-bounds message.

All three now return `CompilerError::Backend(...)` with the
specific HirId. Blinc-side callers get a usable diagnostic
instead of a runtime crash or an obscure panic.

### What to expect now on the Blinc side

Re-run the probe. The error you see should be the new
`CompilerError::Backend("indirect call: function-pointer value
HirId(...) is referenced but never defined ...")`. That's
actually surfacing a real SSA-lowering bug — the closure value's
`CreateClosure` instruction isn't reaching Cranelift's
`value_map` for some shape Blinc emits.

To narrow it: the HirId in the new error is the indirect-call's
callee. Trace back where that HirId is supposed to be defined
in the HIR. Likely candidates (in priority order):

1. A `CreateClosure` for the on-click lambda is emitted but its
   `result` HirId never enters Cranelift's `value_map` (maybe
   because the SSA put it in a block that Cranelift skipped, or
   the `function_map` lookup at line 3403 fell through to the
   else branch and `null_ptr` got stored but under a DIFFERENT
   HirId than the indirect-call references).
2. The closure value was passed AS AN ARGUMENT to an extern, and
   the SSA tried to re-call the captured value INSIDE that
   extern's body (won't happen — externs have no body).
3. Some `HirCallable::Indirect` is being emitted with a
   parameter HirId, but parameter HirIds should be in
   `value_map` from `build_param_locals` at the top of every
   function.

If you can share the HIR dump (`hir_dump.rs` should print it
for a single function), or `RUST_LOG=debug` logs showing what
instructions were emitted in `render_view`'s body, we can
pinpoint which of the above (or something else) is firing.

---

## Update — 2026-05-12 (Blinc side, layer 4): every call in the lambda body lowers as `Indirect`

### Confirmed from layers 2 + 3

- Captured-variable type fix lands: `Cannot access fields on
  non-struct type: Any` no longer fires. `count` inside the lambda
  resolves to its real signal type.
- Blinc-side `resolve_signal_calls` / `resolve_fsm_trigger_calls`
  walkers now descend into `TypedExpression::Lambda` bodies (the
  walkers previously had no Lambda arm — added that, see
  `crates/blinc_dsl_core/src/lib.rs` `rewrite_expr` matches), so the
  typed AST handed to Zyntax has the in-lambda `count.set(...)` /
  `CounterFsm.trigger(...)` rewritten to direct `Call(Variable
  ("__signal_set_i32"), [...])` / `Call(Variable ("__fsm_runtime_trigger__"), [...])`
  before lowering. Identical to the shape the top-level body uses.
- Cranelift backend's improved error messages land: we now see a
  `CompilerError::Backend("indirect call: function-pointer value
  HirId(...) is referenced but never defined ...")` instead of a
  silent SIGSEGV.

### New observation: not just one indirect call — *all* of them

With `RUST_LOG=zyntax_compiler::cranelift_backend=debug`, the lambda
body (`|| { count.set(count.get() + 1); CounterFsm.trigger("Idle.Increment"); CounterFsm.trigger("Counting.Increment") }`) dumps as:

```
[Cranelift]   inst[0]: Call { callee: Indirect(HirId(...)), args: [HirId(...), HirId(...)], ... }
[Cranelift]   inst[1]: Call { callee: Indirect(HirId(...)), args: [HirId(...), HirId(...)], ... }
[Cranelift]   inst[2]: Call { callee: Indirect(HirId(...)), args: [HirId(...), HirId(...)], ... }
[Cranelift]   inst[3]: Call { callee: Indirect(HirId(...)), args: [HirId(...), HirId(...)], ... }
[Cranelift]   inst[4]: Call { callee: Indirect(HirId(...)), args: [HirId(...), HirId(...)], ... }

[CRANELIFT] Skipping function: Backend("indirect call: function-pointer value
   HirId(caadeed6-…) is referenced but never defined in this function's value_map…")
```

So *every* call site in the lambda body emits as `HirCallable::Indirect(callee_value)`,
including the calls to known extern symbols (`__signal_set_i32`,
`__signal_get_i32`, `__fsm_runtime_trigger__`). At the top level
`translate_expression`'s `Call` arm recognises a `Variable` callee that
resolves to a known function name and emits `HirCallable::Symbol(name)`;
inside a lambda body that resolution doesn't happen, so each call gets
the `Variable`'s value HirId as an `Indirect` callee — and nothing
defines those HirIds in the function's value_map.

This is exactly the failure mode Proposal (A) called out:

> "Today `translate_lambda_expr` is a hand-rolled mini-translator
> maintained in parallel with the real one. ... The straightforward
> fix is to lift the lambda body into a regular `HirFunction` and run
> it through `translate_expression`, the same way FSM tick guards
> already are."

The current closure-body-lowering tests cover `Block` / `Call` /
`MethodCall` / `Field` *shapes* but don't assert the `HirCallable`
variant — adding an assertion that
`Call { callee: Variable(name) }` where `name` resolves to a known
function symbol lowers to `HirCallable::Symbol(name)` (not `Indirect`)
inside a lambda body would catch this.

### Cascade panic (still present): `FunctionBuilderContext` not finalised on error

When `compile_function_body` returns `Err` via `?` between
`FunctionBuilder::new` (cranelift_backend.rs:1341) and
`builder.finalize()` (around line 4702), the `FunctionBuilderContext`
stays dirty. The next iteration's `FunctionBuilder::new` panics:

```
thread 'main' panicked at cranelift-frontend-0.106.2/src/frontend.rs:295:9:
assertion failed: func_ctx.is_empty()
   3: <FunctionBuilder>::new
   4: <CraneliftBackend>::compile_function_body  (cranelift_backend.rs:1341)
   5: <CraneliftBackend>::compile_module        (cranelift_backend.rs:486)
```

The new `Backend(...)` returns from layer 3 surface the underlying
error cleanly, but the same `?` path that returns them leaves
`self.builder_context` dirty for the next iteration's
`FunctionBuilder::new` — which panics. Once layer 4 (lambda lowering
emitting `Symbol` for direct externs) is fixed the trigger goes
away in this specific case, but the panic remains latent: any future
`compile_function_body` error path will hit the same cascade.

Minimum fix: ensure cleanup on all error paths — structure
`compile_function_body` so every error path falls through
`builder.finalize()` first, or reset
`self.builder_context = FunctionBuilderContext::new()` (and similarly
`self.codegen_context.clear()`) before returning an early-error from
between create-builder and finalize.

### Repro

`crates/blinc_dsl_core/examples/counter_dsl.{rs,blinc}` against the
current local Zyntax checkout, run with
`RUST_LOG=zyntax_compiler::cranelift_backend=debug`, gives the trace
above. The minimal `_probe_closure.rs` from layer 2 *might not* trip
the cascade panic (only one non-extern function — `render_view` — so
the Err is returned, no next-iter `FunctionBuilder::new`); but it
will still skip the lambda function and so leave `render_main` with
a no-op closure.

---

## Update — 2026-05-12 (layer 4 response): InternedString-mismatch fallback + diagnostics

### Synthetic shape doesn't reproduce

Wrote three new layer-4 tests in
`closure_body_lowering_tests.rs`:

- `expression_bodied_closure_emits_call_to_extern` (tightened) —
  asserts the inner Call lowers as Symbol/Function, NOT Indirect.
- `lambda_as_call_arg_with_capture_survives` (tightened) — same
  shape with a captured outer var; same assertion.
- `lambda_body_extern_call_resolves_same_as_outer` (new) — outer
  AND inner call the same extern; asserts both resolve the same
  way.

**All three pass on the current local checkout.** So my synthetic
TypedAST doesn't reproduce Blinc's layer-4 symptom. The most
likely difference: the `InternedString` instance Blinc's
`resolve_signal_calls` produces for the rewritten
`Variable("__signal_set_i32")` callee compares unequal to the
matching key in `function_symbols` / `extern_link_names`, even
though both `resolve_global()` to the same string. Different
interner arenas → different `InternedString` handles → `IndexMap`
lookup misses.

### Defensive fix

Added a string-name fallback to `translate_expression`'s Call
resolution path in `ssa.rs:3409–3500`:

1. Direct `InternedString` lookup in `function_symbols` (existing).
2. Direct `InternedString` lookup in `extern_link_names` (existing).
3. **NEW**: walk `function_symbols.iter()`, match by
   `key.resolve_global() == name_str`. O(N), only runs after the
   fast lookups miss.
4. **NEW**: same fallback for `extern_link_names`.
5. Only after all four miss does the resolution fall through to
   `HirCallable::Indirect`. The miss site now logs the full
   `function_symbols` + `extern_link_names` key list (resolved
   strings) at `debug!` so the failure mode is observable.

Two small helpers (`lookup_by_resolved_name`,
`lookup_link_by_resolved_name`) at the top of the file
encapsulate the string-equality walk.

This should silently fix Blinc's case if my InternedString-mismatch
theory is right. If the bug persists, the new debug log will
print exactly which keys are available vs the name being looked
up — that'll narrow it to the real cause.

### What Blinc should do next

Re-run with `RUST_LOG=zyntax_compiler::ssa=debug` against the
latest checkout. Either:

1. **The closure body compiles cleanly** — the fallback caught the
   InternedString mismatch and resolved the externs as Symbol /
   Function. The Cranelift `value_map` error goes away.
2. **The Indirect fallback still fires** — the debug log will
   show `function_symbols` keys + the name being looked up.
   Share that log; we'll see if the name truly isn't in either
   map (a different bug), or if some shape escapes both
   fallbacks.
