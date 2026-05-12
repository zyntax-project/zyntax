# Zyntax: lambda body drops `Call` / `MethodCall` / `Block` expressions

Closure bodies whose only content is function or method calls compile to
no-op functions that return `0`. Authors get no diagnostic — the lambda
just silently doesn't execute its statements.

## Where

`crates/compiler/src/ssa.rs`:

- `fn translate_lambda_expr(&self, …)` — around **line 8493**
- `fn translate_closure(&mut self, …)` — around **line 8292**

## What the code does today

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

## Notes for testing

If a regression test is wanted, the repro above can sit next to the
existing closure tests in `crates/compiler/tests/` and assert the
extern was invoked (e.g. via a `static AtomicI32` the extern bumps).
