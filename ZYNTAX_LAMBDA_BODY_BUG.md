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

The local checkout (HEAD `0aea7aa92fd7e6c1a5c190b25c76e4c9f84ad2fb`)
already contains a partial fix for this in `ssa.rs::translate_closure`
— `TypedLambdaBody::Block` now iterates statements and calls
`translate_expression` instead of returning constant 0. The closure
unit tests in that checkout
(`crates/compiler/tests/closure_body_lowering_tests.rs`) assert the
HIR contains `Call` instructions and pass against the local code.

**However, pointing Blinc at the local checkout via `[patch]`
regresses regular function emission.** Concretely:

- Source:
  ```text
  signal count: i32
  view {
      Div(on_click = || { count.set(count.get() + 1) }) { Text("+1") }
  }
  ```
- Compile output against upstream rev `5b0dfab7`:
  `Ok(["__lambda_HirId(…)", "render_view"])` — both functions emitted.
  `render_main` resolves `render_view` and returns the widget handle.
- Compile output against the local checkout:
  `Ok([])` — empty. Both `render_view` and the lambda are missing.
  `render_main` fails with `Backend("Function not found: render_view")`.

So the closure-body fix as it stands regresses the value-returning
view shape Blinc relies on — something the local lambda work touches
must be interfering with regular function registration for
`render_view` / component view methods. The HIR-level closure tests
don't exercise this path (they build their own `TypedProgram` and
call `lower_program` directly), so the regression doesn't show up in
the local checkout's own test suite.

When the upstream fix lands, please verify that for a program of the
form

```text
extern fn $Blinc$Div$view(children: i64, style: i64, class: string, on_click: i64): i64
extern fn sink(value: i32)
fn render_view(): i64 {
    return $Blinc$Div$view(0, 0, "", 0)
}
```

`render_view` ends up in the compiled module's symbol list, JIT-calling
it returns the inner widget handle, and a closure declared elsewhere
in the same program (e.g. `|| { sink(42) }`) still emits a `Call`
instruction in its lowered body.

Blinc's `[patch."https://github.com/darmie/zyntax"]` block in
`Cargo.toml` is currently commented out for this reason; uncomment
it to retest once the regression is resolved.

---

## Update — 2026-05-12: synthetic Blinc shapes pass HIR-level lowering

Added two regression tests in
`crates/compiler/tests/closure_body_lowering_tests.rs`:

- `sibling_top_level_fns_survive_closure_lowering` — three decls
  (extern `sink`, `def helper(): i64 { return 99 }`, `def main():
  i32 { let f = def(): sink(42); return 0 }`). Asserts all three
  non-extern functions + the synthesised `__lambda_*` end up in
  `module.functions`.
- `lambda_as_call_arg_with_capture_survives` — the closer-to-Blinc
  shape: lambda passed as a CALL ARGUMENT (not a let binding),
  body references a captured outer variable. Asserts both
  `render_view` and `__lambda_*` survive lowering.

Both pass against HEAD `28d503c…` (Phase E.5 landed). The
sibling-fn case + the call-arg-with-capture case do NOT reproduce
the `Ok([])` symptom on this checkout.

Possible interpretations:
1. The Blinc report was against a stale checkout state — maybe
   the lambda-fix commit (`5453cc0…`) without the subsequent
   compile-pass churn from the wasm track. Re-test against the
   latest HEAD with `[patch]` re-enabled and see if `Ok([])`
   still reproduces.
2. The repro depends on shapes the synthetic TypedProgram doesn't
   reach — e.g. signal-runtime struct field access, method-call
   trait dispatch on the captured value, or a Blinc-frontend
   syntax desugaring that produces an unusual TypedAst the
   synthetic version doesn't replicate.
3. The repro depends on a code path OUTSIDE SSA lowering — e.g.
   the wasm-target krio passes (`apply_krio_async_lowering` /
   `apply_krio_effect_lowering` in `compile_typed_program`) that
   run AFTER `lower_typed_program` and could in principle
   silently drop functions on certain shapes.

If Blinc still observes the empty-function-list regression after
re-pulling, the most productive next step is to capture
`compile_typed_program`'s `hir_module.functions` count BEFORE the
post-lowering passes (`apply_krio_*`) and AFTER, then dump the
function names. That isolates whether SSA produces them and one
of the subsequent passes drops them, vs. SSA never produces them
in the first place.
