# Capturing a handler context for a deferred host callback

**Status**: requested, not implemented
**Raised by**: Blinc (`blinc_dsl_core`), 2026-08-13
**Related**: `crates/zyntax_embed/src/effect_runtime.rs`
(`HANDLER_SEGMENTS`, `__zyntax_effect_fiber_enter` / `_leave`)

## The gap

A host that registers a callback to run **later** — outside the extent
that installed the handlers — has no way to record the handler context
in force at registration and reinstate it when the callback fires.

The runtime already does exactly this for fibers. A fiber shares the
thread-local `HANDLER_STACK` with its caller, so
`__zyntax_effect_fiber_enter` records a baseline and re-pushes the
frames the fiber had open, and `__zyntax_effect_fiber_leave` lifts them
back out, keyed by fiber pointer in `HANDLER_SEGMENTS`. The same
mechanism is needed keyed by something a host holds.

## Why a host needs it

A UI DSL compiles `computed { ... }` and `on_click = || { ... }` into
zero-argument functions the host stores and calls on its own schedule:
a reactive-graph flush, an input event. Reads inside those bodies
perform against a handler, so they need the context that was installed
where the closure was **written**, not whatever happens to be installed
when it runs — which is usually nothing.

This is not specific to a UI. It is the general shape of "register a
continuation now, run it later" for any embedder whose effects are
scoped.

## What a host is forced into without it

Blinc keyed callbacks to a `u64` it invented, held in a thread-local,
and resolved a handler instance from that key at call time. That is a
parallel handler stack, and it produced a failure a real one cannot:
resolution was get-or-create over the invented key, so a request from
an unfamiliar scope silently minted a **second machine** with fresh
state, where a perform with no handler installed would have failed
loudly.

## Shape that would work

```rust
// at registration, on the thread where the handlers are installed
let ctx: HandlerContext = rt.capture_handler_context();

// later, on the same thread, around the callback body
let restored = rt.enter_handler_context(&ctx);
let out = callback();
rt.leave_handler_context(restored);
```

`HandlerFrame` is small and plain, so a capture is a `Vec` clone.

Two properties that are easy to miss and expensive to get wrong:

- **The runtime lock must not be held across the callback body.** The
  body is compiled code that may call host externs which take the
  runtime lock, so an API that holds it for the duration deadlocks.
  Installing is a thread-local push, so releasing the lock before
  running the body is sound — but the API should make that the obvious
  reading rather than something each caller has to work out.
- **Entering must layer, not replace.** A callback may itself register
  another, so entering has to stack above whatever is installed,
  exactly as `fiber_enter` layers on the caller's baseline.

## What this unblocks

Deleting the invented layer on the Blinc side: a thread-local scope
`u64`, a `(scope, fsm)` map, the scope-bracket injection pass, and the
compile-time-baked-id fallback that existed only to paper over it.
