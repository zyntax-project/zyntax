# A perform two frames deep jumps to garbage, and no guard covers it

**Status**: reported, cause not identified
**Raised by**: Blinc (`blinc_dsl_core`), 2026-08-23
**Rev**: `262a7d1` (current `origin/main`)
**Platform**: macOS arm64, Cranelift tier 0
**Related**: `crates/zyntax_embed/src/runtime/tiered.rs` (`call_raw`'s
stateful-effect guard), `crates/zyntax_embed/src/effect_runtime.rs`

## What happens

A host renders by taking a compiled function's pointer and calling it.
Deep inside that call, a function that declares an effect performs one
of its operations, and the process dies jumping to an address that is
not code:

```
__blinc_with_4$view
  -> reactive$ReactivePage$view (+0x388)
    -> reactive$Play$readid_pct (+0x64)
      -> 0x695700000007            SIGBUS
```

`reactive$Play$readid_pct` is a one-line function: it declares
`@effect(reactive$Play$Events)` and its body performs
`reactive$Play$id_pct()`, an operation of that effect.

The frames above it are anonymous in a debugger. They were named by
dumping `symbol -> get_function_ptr` after compile and matching each
frame address to the nearest preceding symbol.

## What is already known to be right

- A handler **is** installed on the calling thread. The host pushes the
  machine's handler instance (`push_handler_instance`) immediately
  before the render and pops it after; it logs `installed=true` for
  `reactive$Play` on every render.
- The operation **is** declared and implemented. The effect declares
  `get_`, `id_` and `set_` per context field, and the handler
  implements all three, emitted in the same loop.
- The instance **is** bound: it was created with `new_handler_instance`
  and bound to the machine's fiber with `bind_fiber_handler_instance`.
- Calling the same function **through the runtime** works. The host's
  other route resolves the identical reader via
  `push_handler_instance` + `TieredRuntime::call`, and that path is
  exercised by passing tests.

## What `call_raw` already says about this shape

`call_raw` refuses a call whose stateful effects have no frame in
scope, and the comment there describes precisely this failure:

> A perform whose effect has no frame in scope resolves its handler op
> statically, and a stateful op then reads an implicit `self` that
> nothing supplied. Refusing the call is the difference between an
> error the host can report and a null dereference inside compiled
> code.

So the shape is understood upstream. Two things follow.

**The guard does not cover this case.** Routing the render through
`call_raw` instead of the raw pointer still crashes, with no guard
error: a handler is active for the *entry* symbol, and the guard checks
only the effects that symbol itself declares. The perform that dies is
two frames deeper, in a function the guard never inspected.

**A host that must call by pointer cannot run the guard at all.**
`stateful_effects_of` lives on the backend and is not public;
`effect_runtime::has_handler_for` is public but takes an `effect_id` a
host has no way to obtain. The host calls by pointer deliberately: the
runtime mutex cannot be held across compiled code that re-enters the
host, so `call`/`call_raw` are not usable on the render path.

## Reproducing

In the Blinc tree at `wip/fsm-machine-scope`, four tests fail this way,
all loading the same multi-module program:

```
cargo test -p blinc_cn_dsl --test pg_nav --test pg_grow_cost \
  --test pg_trigger --test pg_reactive_baseline --no-fail-fast
```

It needs the multi-module build; the same FSM and view in one file does
not reproduce it. It predates the host-side work that made these reads
in-language, so it is not a regression from that.

## What would help

1. **Say why the jump target is not code.** With a frame in scope, an
   instance bound, and the op implemented, what else decides the
   address a perform dispatches to? A pointer-shaped value of
   `0x6957_00000007` looks like a slot read from uninitialised or
   wrongly-based handler state rather than a null.

2. **Make the guard reachable and total.** Two separate asks:
   - expose `stateful_effects_of(name)`, or a
     `missing_stateful_handler(name) -> Option<(u64, String)>`, so an
     embedder that must call by pointer can run the same check before
     jumping;
   - consider whether the check can cover effects performed
     *transitively*, since the entry symbol declaring nothing is
     exactly the case that crashed here.

Either one turns this class of bug from a SIGBUS into a message, which
is the difference the existing comment is already arguing for.
