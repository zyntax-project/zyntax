# Async, algebraic effects and fibers

Zyntax gives every frontend three ways for code to suspend and be resumed:

- **Algebraic effects.** A function performs an operation; a handler chosen by dynamic scope runs it. A handler operation that takes a continuation can resume the rest of the performing function, several times or not at all.
- **Async functions.** An `async` function is compiled to a stackless state machine that a cooperative executor polls.
- **Fibers.** A `fiber def` runs on its own stack and yields values to whoever resumes it.

This document describes how each one is represented in the typed AST and HIR, how it is lowered to code in each tier, what runs at runtime, how the three compose, and where they stop. It is written for contributors who change these subsystems and for frontend authors who target them. Every statement is about the code as it stands; file paths are relative to the repository root.

## Contents

1. [Overview](#1-overview)
2. [ZynML surface](#2-zynml-surface)
3. [Typed AST and the rewrites before lowering](#3-typed-ast-and-the-rewrites-before-lowering)
4. [HIR](#4-hir)
5. [Algebraic effects](#5-algebraic-effects)
6. [Async functions](#6-async-functions)
7. [Fibers](#7-fibers)
8. [Host API](#8-host-api)
9. [Frontends](#9-frontends)
10. [Code generation by tier](#10-code-generation-by-tier)
11. [Composition](#11-composition)
12. [Limits](#12-limits)
13. [Source map](#13-source-map)

## 1. Overview

| | Algebraic effects | Async functions | Fibers |
|---|---|---|---|
| Suspends by | calling the handler operation; a resumable operation holds the rest of the performing function as a state machine | returning Pending from its poll function | switching to another stack |
| Suspended state lives in | the thread's handler stack, and for resumable operations the performer's state machine | a heap array of 8-byte slots | the fiber's own stack |
| Built by | `SsaBuilder` (`PerformEffect`), `LoweringContext::lower_with_scopes`, `krio_adapter` for resumable operations | `krio_adapter` | `apply_krio_fiber_lowering` |
| Runtime | `crates/zyntax_embed/src/effect_runtime.rs` | `crates/zyntax_embed/src/host_futures.rs`, `crates/zyntax_embed/src/runtime/promise.rs` | `crates/passes/krio_adapter/src/fiber.rs` over the `krio-fiber` crate |

Two of the three share one substrate. Async functions and resumable effects are both lowered by the krio captures-lift transform (`krio_async::transform_to_state_machine_with_options`, driven by `crates/passes/krio_adapter`): values live across a suspension point are saved into numbered slots of a heap state machine, and a dispatcher switch at the top of the poll function jumps back to the right resume block. Fibers are the other substrate: a real stack per fiber and a context switch, provided by `krio-fiber`.

**Scheduling is cooperative and single-threaded.** Every runtime table is `thread_local!`: the handler stack, the future table, the timer queue, the fiber registries. A fiber must be resumed on the thread that made it; a task is driven by whichever thread calls the driver. The design keeps these tables thread-local because every suspension point is a call the program itself makes, so nothing needs synchronizing. What a real task or fiber object would own (its open handlers, its fibers) is kept in side tables keyed by fiber pointer or task id and swapped in around each resume (section 5.9).

```
 source: ZynML, zypy, zylua
   |
   |  typed AST: TypedEffect, TypedEffectHandler, TypedWith, Type::Fiber,
   |             TypedFunction { effects, is_async, is_fiber }
   v
 algebraic_effects pattern pass        synthesize_handler_state
   handler ops -> `H$op` functions       `H$state`, `H$new`, implicit `self`
   |
   v
 LoweringContext + SsaBuilder
   PerformEffect, FiberNew / FiberResume / FiberYield / FiberCancel,
   Call(Intrinsic::Await), push/pop/free around `with` bodies,
   FiberDrop before returns
   |
   v
 apply_krio_async_lowering    async fn  -> promise entry + `NAME$poll`
 apply_krio_effect_lowering   sync fn performing a resumable effect
                              -> sync entry + `NAME$poll`
 apply_krio_fiber_lowering    fiber ops -> `krio_fiber_*` calls
   |
   v
 interpreter / Cranelift / LLVM / wasm backend
   |  call into
   v
 runtime (one set per thread)
   effect_runtime.rs      handler stack, fiber and task segments, resume, launch
   host_futures.rs        future table, timer queue, completion latches
   runtime/promise.rs     ZyntaxPromise, drive_until, combinators
   krio_adapter/fiber.rs  KrioFiberBackend: fiber stacks, errors, task registry
```

The three krio passes run in that order after HIR lowering, in every compile entry point of `ZyntaxRuntime` (`crates/zyntax_embed/src/runtime/classic.rs`) and `TieredRuntime` (`crates/zyntax_embed/src/runtime/tiered.rs`). The wasm entry points in `crates/zyntax_wasm/src/lib.rs` run the first two only.

## 2. ZynML surface

The grammar is `crates/zynml/ml.zyn`. The examples below are taken from the tests under `crates/zynml/tests`.

### 2.1 Effects and handlers

- `effect E { def op(params): T }` declares an effect and its operations (`effect_def`, `effect_op`).
- `handler H for E { ... }` implements them (`handler_def`). State fields come first, as `var name: T = init` or `let name: T = init` (`handler_field`), then one `def` or `async def` per operation (`handler_impl_sync`, `handler_impl_async`).
- A function that performs an effect is annotated `@effect(E)` and calls the operation by its bare name. The annotation works on free functions and on impl methods (`decorated_impl_method`).
- `with H { body }` installs `H` for the dynamic extent of `body` (`with_stmt`). Outside any `with`, a perform goes to the effect's static default handler, the first handler declared for it.
- An operation is **resumable** when one of its parameters has type `Resume<T>`. That parameter must come last. Inside the body, `k(v)` resumes the performer with `v` and returns what the performer then returns; `abort(v)` evaluates to `v` without resuming.

A stateful, non-resumable handler:

```
effect Counter {
    def next(): i64
}

handler Seq for Counter {
    var n: i64 = 0
    def next(): i64 {
        self.n = self.n + 1
        return self.n
    }
}

@effect(Counter)
def tick(): i64 {
    return next()
}

def main(): i64 {
    let mut total: i64 = 0
    with Seq {
        total = tick()
        total = total + tick()
    }
    return total
}
```

`self` is never written in the operation's parameter list; it is added by `synthesize_handler_state` (section 5.6). Each `with Seq` scope starts from the field initializers.

A resumable handler:

```
effect E {
    def op(): i64
}

handler H for E {
    def op(k: Resume<i64>): i64 {
        return k(21) + 1000
    }
}

@effect(E)
def run(): i64 {
    let x = op()
    return x * 2
}
```

`run()` returns 1042: `k(21)` runs the rest of `run` with `x = 21`, which returns 42 to the handler, and the handler's own return value becomes `run`'s result (section 5.7).

### 2.2 Fibers

- `fiber def name(): T { ... }` declares a fiber whose yields have type `T` (`fiber_def_*` rules). The return type may be omitted.
- `yield expr` inside a fiber body suspends it with a value (`yield_stmt`; outside a fiber body the same statement feeds a `compute` reduction).
- Calling a fiber def does not run it: it returns a paused `Fiber<T>`.
- On a `Fiber<T>`: `next()` resumes it and returns `Option<T>` (`Some` for a yield, `None` once it has finished, was cancelled, or aborted); `cancel()` asks it to stop; `error()` returns `Option<FiberError<String>>`.
- Inside a fiber body, `Fiber.abort(x)` ends it with an error payload.

```
effect Log { def emit(): i64 }
handler H for Log { def emit(): i64 { return 7 } }

@effect(Log)
fiber def gen(): i64 {
    yield emit()
    yield emit()
}

def main(): i64 {
    let mut total: i64 = 0
    with H {
        let f = gen()
        while let Some(x) = f.next() {
            total = total + x
        }
    }
    return total
}
```

The grammar has no `async fiber def`.

### 2.3 Async

- `async def name(...): T { ... }` declares an async function (`async_def_*` rules). Calling it returns a promise; the host drives it (section 8).
- `await expr` suspends until the awaited promise is Ready (`await_expr`). The operand is usually a call to another async function or a host bridge.
- ZynML declares no async library functions. A host maps names onto host bridges with builtin aliases, for example `sleep` to `__zyntax_async_set_timeout` through `ZyntaxRuntime::config_mut().builtins` or `TieredRuntime::builtin_aliases_mut`; the wasm entry points install that alias themselves.
- `@cooperative` (alias `@coop`) sets `FunctionAttributes::cooperative`. Nothing reads the flag (section 12).

A resumable effect performed from an async function that owns a fiber:

```
effect E { def op(): i64 }
handler H for E { def op(k: Resume<i64>): i64 { return k(1) } }
fiber def gen() { yield 10 yield 20 }

@effect(E)
async def work(): i64 {
    let f = gen()
    let x = op()
    await sleep(10)
    var s: i64 = 0
    while let Some(y) = f.next() { s = s + y }
    return x + s
}

async def run(): i64 { var v: i64 = 0 with H { v = await work() } return v }
```

A handler operation may itself be `async def op(k: Resume<i64>): i64 { await sleep(30) return k(7) }` (section 5.8).

## 3. Typed AST and the rewrites before lowering

**Declarations** (`crates/typed_ast/src/typed_ast.rs`):

- `TypedEffect { name, type_params, operations: Vec<TypedEffectOp> }`.
- `TypedEffectHandler { name, effect_name, type_params, fields: Vec<TypedField>, handlers: Vec<TypedEffectHandlerImpl> }`. `TypedEffectHandlerImpl::is_async` marks an `async def` operation.
- `TypedFunction` carries `effects` (names from `@effect`), `with_handlers` (names from `@with`, which lowering refuses), `is_async` and `is_fiber`. `TypedMethod` carries annotations, so an impl method takes `@effect` too.
- `TypedStatement::With(TypedWith)`, where `TypedWith` has `handlers: Vec<TypedWithHandler>` (each a `name` and `args`) and a `body`. The list and `args` have room for `with A, B` and `with H(args)`, which the grammar does not parse.
- `TypedStatement::Yield`, `TypedExpression::Await`.

**Types.** `Type::Fiber(Box<Type>)` (`crates/typed_ast/src/type_registry.rs`) is compiler-known, not a named struct: the parser produces it for `Fiber<T>`, and method calls on it dispatch through `builtin_class::FiberClass`. `Resume<T>` is an `extern struct` in `crates/zynml/stdlib/prelude.zynml`, recognized by its name wherever resumability is decided. The prelude also declares `extern struct Fiber<T>`, its `Iterator` impl and `FiberError<T>`; the stub bodies there never run.

**The `algebraic_effects` pattern pass** (`crates/passes/algebraic_effects`) runs from `rewrite_patterns` in `crates/zyntax_embed/src/lower.rs` when the program declares an effect, a handler, or an `@effect`/`@with` annotation (`declares_effects`), or when a frontend turns pattern rewrites on. It registers three declaration rewrites:

- `extract_effect_annotations` copies `@effect(...)` and `@with(...)` identifiers into `TypedFunction::effects` and `with_handlers` (shared with impl methods through `effect_annotation_lists`).
- `handler_decl_to_impl` adds one standalone function per handler operation, named `H$op` (`effect_codegen::mangle_handler_op_name`), with the operation's `is_async`. The `EffectHandler` declaration stays, because lowering builds `HirEffectHandler` from it.
- `effect_decl_to_vtable` adds an `E$OpTable` class. Nothing downstream reads it; op tables are built during lowering (section 5.2).

**Handler state** is synthesized by `synthesize_handler_state` (`crates/zyntax_embed/src/runtime/handler_state.rs`) before the type registry is snapshotted, because later passes see a read-only registry and cannot add types. For each handler with fields it registers a `@reference` struct `H$state`, adds a constructor `H$new(): H$state` that builds it from the field initializers, and prepends `self: H$state` to every operation that has no `Resume<T>` parameter (section 5.6).

**`@with` is refused.** `LoweringContext::lower_function` and the `Impl` arm of `lower_declaration_inner` (`crates/compiler/src/lowering.rs`) reject any function or method carrying it ("a handler-scoping annotation on `NAME` is no longer supported"). The annotation installs no handler, so accepting it would be a silent no-op; handler scoping is the `with` statement.

**`Fiber.abort(x)`** is rewritten in `crates/zyntax_embed/src/import_chain.rs` into a call with a variant tag inserted first (1 for a `String` argument, 2 otherwise), and the builtin alias `Fiber$abort` (`crates/zyntax_embed/src/lower.rs`) routes that call to `krio_fiber_abort_with`.

## 4. HIR

**Module tables** (`crates/compiler/src/hir.rs`): `HirModule::effects: IndexMap<HirId, HirEffect>` and `HirModule::handlers: IndexMap<HirId, HirEffectHandler>`. A `HirEffect` lists its `HirEffectOp`s in declaration order; that order defines every operation's index. A `HirEffectHandler` has `state_fields` and one `HirEffectHandlerImpl` per operation, with `is_resumable` (it has a `Resume<T>` parameter) and `is_async`. They are filled by `LoweringContext::lower_effect` and `lower_effect_handler`.

**Signatures.** `HirFunctionSignature` has `is_async`, `is_fiber` (the grammar keeps them exclusive) and `effects: Vec<InternedString>`. `FunctionAttributes::cooperative` records `@cooperative`.

**Types.** `HirType::Fiber(Box<HirType>)` is a pointer-sized handle in every backend. `Resume<T>` converts to `HirType::Ptr(U8)`.

**Instructions.**

| Instruction | Produced by | Consumed by |
|---|---|---|
| `PerformEffect { result, effect_id, op_name, args, return_ty }` | `SsaBuilder`, for a call to an operation name inside an `@effect` function | krio lowering, which replaces resumable performs with dispatch sequences and, in async functions, makes every perform a suspension site; the Cranelift and LLVM arms compile what remains |
| `Call { callee: Intrinsic(Await), args: [promise] }` | `SsaBuilder`, for `await` | krio async lowering; an error if it reaches Cranelift |
| `AsyncSaveSlot { frame, slot, value }`, `AsyncLoadSlot { result, ty, frame, slot }` | `krio_adapter` | interpreter, Cranelift, wasm backend |
| `FiberNew { closure, stack_size }` | `SsaBuilder`, for a call to a fiber def | `apply_krio_fiber_lowering` |
| `FiberResume { fiber }` | `SsaBuilder::emit_fiber_next`, for `next()` | `apply_krio_fiber_lowering` |
| `FiberResumeWith { fiber, value }` | nothing in ZynML | `apply_krio_fiber_lowering` |
| `FiberYield { value }` | `SsaBuilder`, for `yield` in a fiber body | `apply_krio_fiber_lowering` |
| `FiberCancel { fiber }` | `SsaBuilder::emit_fiber_cancel`, for `cancel()` | `apply_krio_fiber_lowering` |
| `FiberDrop { fiber }` | `LoweringContext::emit_fiber_drops` | `apply_krio_fiber_lowering` |
| `FiberTransfer { target, value }` | nothing | `apply_krio_fiber_lowering`; the runtime panics |
| `HandleEffect`, `Resume`, `AbortEffect`, `CaptureContinuation` | nothing | placeholder arms in Cranelift and LLVM |

`HandleEffect`, `Resume`, `AbortEffect` and `CaptureContinuation` describe a handler model the lowering does not use. Handler installation is a pair of runtime calls around a `with` body, and resumption is a call to `__zyntax_effect_resume`. The modules built around the unused model, `effect_analysis`, `effect_handler_resolution` and `effect_codegen`, are reached from `zyntax_compiler::compile_to_hir` (effect checking, on by default in `CompilationConfig`) and from tests; the embedding runtime's lowering does not call them, apart from the name mangler.

**Runtime calls.** Everything else is an ordinary `Call` to a `HirCallable::Symbol` that the runtime registers (`register_effect_runtime_symbols`, `register_fiber_runtime_symbols` and `fiber_runtime_symbol_infos` in `crates/zyntax_embed/src/effect_runtime.rs`; `zrtl::fiber_runtime_symbols` in `crates/compiler/src/zrtl.rs`):

| Symbol | Emitted by | Role |
|---|---|---|
| `__zyntax_effect_push_handler(effect_id, state, op_table, async_mask) -> frame_id` | `lower_with_scopes` | push a handler frame |
| `__zyntax_effect_pop_handler(frame_id)` | `lower_with_scopes` | pop it |
| `__zyntax_effect_lookup_op(effect_id, op_index) -> fn` | every perform site | innermost frame's operation, or null |
| `__zyntax_effect_lookup_state(effect_id) -> state` | non-resumable perform sites of a stateful effect | innermost frame's state, or null |
| `__zyntax_effect_lookup_op_is_async`, `__zyntax_effect_finish_op` | resumable perform sites with mixed async and sync handlers | pick the calling convention at run time |
| `__zyntax_effect_launch_handler(promise, resume)` | resumable perform sites of an async operation | start the handler's task |
| `__zyntax_effect_resume(resume, value)` | `k(v)` in a resumable operation | resume the performer |
| `__zyntax_effect_abort(value)` | `abort(v)` in a resumable operation | returns `value` |
| `__zyntax_effect_fiber_enter`, `_leave`, `_forget` | `apply_krio_fiber_lowering` | per-fiber handler segments |
| `__zyntax_register_future` | `lower_await_calls`, before a host-bridge call (`__zyntax_async_*`, reached from source through a builtin alias) | park on a host event |
| `__zyntax_runtime_release_sm_by_offset` | `generate_sync_entry` | release a sync performer's state machine |
| `krio_fiber_*` | `apply_krio_fiber_lowering`, and frontends directly | fiber operations |

## 5. Algebraic effects

### 5.1 Perform sites

`LoweringContext` builds a per-function `effect_op_map` from the function's `effects` list: every operation of every named effect, by name, with its effect id and return type. When `SsaBuilder` translates a call whose callee is a bare name in that map, it emits `PerformEffect` instead of resolving the name as a function. A function without `@effect(E)` therefore cannot perform `E`, and inside one, an operation name shadows any function with the same name.

The same builder rewrites calls inside a handler operation: a call through a parameter of type `Resume<T>` becomes `__zyntax_effect_resume(k, v)`, and `abort(v)` becomes `__zyntax_effect_abort(v)` (`resume_param_names` in `crates/compiler/src/ssa.rs`).

### 5.2 Op tables

`LoweringContext::build_op_table` (`crates/compiler/src/lowering.rs`) builds one global per handler, named `$optable$H`: an array of function pointers where slot `i` is the handler's implementation of the effect's `i`-th operation. It is laid out like a trait vtable and materialized through the same path (`trait_lowering::create_vtable_global`). `lower_program` builds a table for every declared handler, not only for handlers some `with` names, because a host can install any handler around code it drives (section 8).

An operation the handler does not implement keeps its slot, left null. Operations are addressed by position, so dropping a slot would shift every later operation onto its neighbour's code. A null slot is visible at the perform site, which falls back to the static handler.

### 5.3 The handler stack

`effect_runtime.rs` keeps one `HANDLER_STACK: Vec<HandlerFrame>` per thread. A `HandlerFrame` is `{ effect_id, handler_state, op_table, async_mask }`, where bit `i` of `async_mask` says operation `i` is async in this handler. The effect id is the effect's `HirId` as a `u64`, the same value on push and lookup.

- `__zyntax_effect_push_handler` appends a frame and returns its index as the frame id.
- `__zyntax_effect_pop_handler(frame_id)` pops the top frame. A mismatched id still pops (and returns 2), because leaving a stale frame would break every later lookup.
- `__zyntax_effect_lookup_op`, `_lookup_state` and `_lookup_op_is_async` walk the stack from the top and answer from the first frame with a matching effect id, or return null (`-1` for `_is_async`).

### 5.4 `with` scopes

`TypedCfgBuilder` (`crates/compiler/src/typed_cfg.rs`) splits a `with` body into its own blocks and records a `WithScopeInfo { handler_name, entry, after, body_blocks }`. Push and pop are emitted later, by `LoweringContext::lower_with_scopes`, in a pass that runs after every function is lowered, since the scope needs the handler's `H$op` functions and op table to exist.

For each scope the pass:

1. For a stateful handler, calls `H$new()` at the top of the entry block.
2. Emits `__zyntax_effect_push_handler(effect_id, state or 0, &$optable$H, async_mask)` there.
3. On every scope block whose terminator returns or branches to a block outside the scope, appends `__zyntax_effect_pop_handler(frame_id)` and, for a stateful handler, `Intrinsic::Free(state)`.

A return, or an unconditional branch out of the scope (how `break`, `continue` and falling off the end of the body leave it), gets the pop. A conditional branch with one target inside the scope and one outside, and a `Switch`, get none (section 12).

### 5.5 Non-resumable dispatch

A `PerformEffect` that no krio pass rewrote is compiled by the backend. The Cranelift arm (`crates/compiler/src/cranelift_backend.rs`) emits:

```
static  = address of H0$op            (H0 = first handler declared for the effect)
dyn     = __zyntax_effect_lookup_op(effect_id, op_index)
fn_ptr  = select(dyn != 0, dyn, static)
result  = call_indirect fn_ptr(self?, args...)
```

There is no branch: a `with` in scope wins, and otherwise the static default runs. `op_index` is the operation's position in the effect; an operation name the effect does not declare is a compile error ("effect operation `op` is performed here but is not one of the operations its effect declares"). A perform of an effect that has no handler at all compiles to a trap.

**The calling convention belongs to the effect, not to a handler.** The perform site calls whatever handler is in scope at run time, so every handler of an effect must take the same arguments. The rule is applied on both sides: `synthesize_handler_state` gives every handler of a stateful effect the leading `self` slot (a handler without state receives an `i64` it never reads), and the perform site passes the state when any handler of the effect has state and the operation is not resumable. Deriving the convention from whichever handler happened to be declared first would make one scope's arguments land one slot off in another.

### 5.6 Handler state

A stateful handler's fields live in an `H$state` region allocated by `H$new` through `Intrinsic::Malloc`, which is the runtime's pool allocator. The region is passed to `push_handler`, and a non-resumable operation receives it as `self`, fetched at the perform site with `__zyntax_effect_lookup_state`. Field reads and writes are ordinary loads and stores through `self`.

Lifetime follows who installed the frame:

- **A `with` scope** owns its region. `lower_with_scopes` frees it on every exit edge, right after the pop, so no later perform can reach it. The region escapes into `push_handler`, so drop insertion does not see it; the free is syntax-directed.
- **A host install** is reference counted (section 8.2). One region can back a fiber binding and any number of pushes, so no single pop may free it.

A stateful handler must be active when its operations run. Outside every `with`, the static default runs with a null `self`. `TieredRuntime::call_raw` refuses such a call before entering compiled code, through `missing_stateful_handler` (`TieredBackend::stateful_effects_reached_by`), which follows direct calls transitively from the entry point and treats a function that opens a `with` for an effect as covering everything it calls. It does not follow indirect calls or handler operations reached by dispatch, so an empty answer means nothing reachable that way needs a frame, not that the call is safe. `ZyntaxRuntime` does not check.

Resumable operations receive no `self`, and a region is freed without dropping its fields (section 12).

### 5.7 Resumable handlers and continuations

A resumable perform needs a continuation: the rest of the performing function, runnable later and possibly more than once. Zyntax builds it with the same transform as async functions, so the continuation is a state machine and resuming is re-polling it.

**Which functions are lowered.** `apply_krio_effect_lowering` (`crates/zyntax_embed/src/krio_lowering.rs`) takes every function that is not async, still contains a raw `PerformEffect`, and names in `@effect` an effect that has a handler with a resumable operation. Gating on a raw `PerformEffect` keeps it off functions the async pass already lowered. Async functions that perform resumable effects are handled by `apply_krio_async_lowering` (section 6.2).

Each selected function goes through `orchestrator::lower_async_function` and `reshape_to_poll_abi` (section 6.2) and becomes `NAME$poll`. Its public name becomes a **sync entry** (`abi_emit::generate_sync_entry`) with the original signature: it allocates the state machine, stores the arguments, sets the refcount slot to 1, polls until the poll function returns non-zero, releases the state machine through `__zyntax_runtime_release_sm_by_offset`, and returns the value.

**The perform site** (`abi_emit::upgrade_resume_struct_at_perform_sites`) replaces the `PerformEffect` in its suspension block with:

1. A `Resume` record in the state machine's scratch region, five 8-byte fields: `poll_fn_ptr` (the function's own poll function), `state_machine_ptr`, `result_slot_offset`, `next_state` (the resume state) and `refcount_offset`. Each perform site has its own record.
2. The same regional dispatch as section 5.5, built from HIR ops: `select(__zyntax_effect_lookup_op(...) != 0, dyn, static)`, where the static default is the first declared handler whose implementation of the operation is resumable (`record_handler_resolution`).
3. An `IndirectCall` of the operation with its arguments and the `Resume` pointer last.
4. A `Return` of the handler's result from the poll function.

**`__zyntax_effect_resume(resume, v)`** (`effect_runtime.rs`) writes `v` into the result slot, writes `next_state` into slot 0 as a full `i64`, and polls the performer until it returns non-zero, which it returns to the handler. When a poll returns Pending and a timer is parked (the continuation reached an `await`), it drives the nearest timer and polls again; if that drive already completed the performer, it takes the recorded completion (`host_futures::take_sm_completion`) instead of polling a finished machine. Pending with no timer counts toward `LOOP_BUDGET`, and re-entry depth toward `REENTRY_BUDGET`; either overrun panics with the state-machine pointer and state, since both mean the dispatcher is not reaching the resume state.

**Semantics.** The handler's return value is the performing function's return value: the perform site returns it straight out of the poll function. Consequently:

- `return k(v)` makes the perform evaluate to `v` and the function finish normally.
- `return k(a) + k(b)` runs the continuation twice and returns the combination (multi-shot).
- Not calling `k` (for example `return abort(99)`) ends the performing function with the handler's value. `abort` itself only returns its argument.
- The continuation extends to the end of the function that performed, not to the end of the `with` body. A caller of that function sees an ordinary return.

A second `k` re-polls the same state machine from the same resume state; the continuation is not copied (section 12).

**Continuations that outlive their perform.** A host that keeps a `Resume` pointer past the call that produced it pins the state machine with `__zyntax_runtime_retain_sm` and later releases it with `__zyntax_runtime_release_sm`; the sync entry's release then leaves it alive. ZynML source has no way to store `k`.

### 5.8 Async handler operations

An `async def` operation is an async function (`handler_decl_to_impl` keeps `is_async`), so its `H$op` becomes a promise entry. Resumable async operations are supported: `HirEffectHandlerImpl::is_async` reaches `handler_resolution`, and the perform site in an async performer:

1. Calls the operation, receiving a `*Promise`.
2. Calls `__zyntax_effect_launch_handler(promise, resume)`, which registers the handler's state machine as handling the performer's (`host_futures::register_handler_performer`), polls the handler once under a fresh negative task id with its own handler segment, and returns the handler's result if it finished, or 0 if it parked.
3. Returns that value from the performer's poll function: Ready if the handler finished inline, Pending if it parked.

The parked handler's timer belongs to the executor, so other tasks run while it waits. When it resumes the performer with `k(v)`, `__zyntax_effect_resume` runs the continuation. When the handler finishes, `route_handler_completion` records its return value as the performer's completion by state-machine pointer, and the executor's `harvest` step marks the performer's promise Ready without polling it again.

**Mixed handlers.** One operation may have an async handler in one scope and a sync handler in another. `record_handler_resolution` marks the operation as mixed; its perform site computes the convention at run time, taking the `async_mask` bit of the frame in scope (`__zyntax_effect_lookup_op_is_async`) or the static default's, and calls `__zyntax_effect_finish_op(raw, is_async, resume)`, which launches a promise or passes a value through. Operations that are uniformly sync or async keep the compile-time path.

### 5.9 Per-fiber and per-task handler segments

Fibers and cooperative tasks share one thread, and so one handler stack. Each gets its own **segment** of that stack, layered on top of whatever is installed when it runs:

- `apply_krio_fiber_lowering` brackets every fiber resume: `baseline = __zyntax_effect_fiber_enter(fiber)` records the stack depth and re-pushes the fiber's saved frames; after the switch, `__zyntax_effect_fiber_leave(fiber, baseline)` lifts every frame above the baseline into the fiber's segment and truncates the stack back. `FiberDrop` first calls `__zyntax_effect_fiber_forget`, so a fiber that reuses a freed address inherits nothing. Segments are kept in `HANDLER_SEGMENTS`, keyed by fiber pointer.
- The executor brackets every poll of a task the same way with `task_handler_enter` and `task_handler_leave` (`TASK_HANDLER_SEGMENTS`, keyed by task id), in `drive_until` and `drive_next_timer_with_task`. `deregister_task` calls `task_handler_forget`.

Three properties follow. A perform inside a fiber resolves against the handlers active where it was resumed, plus its own. A fiber's or task's open `with` scopes never show through to its resumer or to another task. And a fiber abandoned inside its own `with` scope does not leak that frame to its caller. Truncating the stack after each resume without saving the segment would lose the fiber's open frames before it could pop them on the next resume; saving and re-pushing them is what keeps them.

A host binding (`fiber_bind_handler`, section 8.2) inserts a frame at the bottom of a fiber's segment, beneath the fiber's own scopes, which keep precedence.

## 6. Async functions

### 6.1 Two lowerings

**The default is the krio transform.** `zyntax_embed`'s default `native` feature includes `krio-async-backend`, which sets `LoweringConfig::use_krio_async` (`crates/zyntax_embed/src/lower.rs`, `runtime/classic.rs`); `LoweringContext::lower_function` then leaves async functions untransformed, still marked `is_async`, for `apply_krio_async_lowering`.

**The legacy lowering** is `async_support::AsyncCompiler` (`crates/compiler/src/async_support.rs`). It runs from `LoweringContext::transform_async_function`, and from `zyntax_compiler::compile_to_hir` when `CompilationConfig::async_runtime` is set, whenever `use_krio_async` is false. The embedding runtimes exist only under `native`, which implies the krio feature, so the legacy path is reached only by code that calls the compiler directly with the default `CompilationConfig` (where `use_krio_async` is false). `zyntax_wasm` sets `use_krio_async` itself. The rest of this section describes the krio path.

### 6.2 The transform

`apply_krio_async_lowering` (`crates/zyntax_embed/src/krio_lowering.rs`) takes every function with `is_async`. It computes their liveness (`AnalysisRunner`) and the module's suspending set (`krio_adapter::HirSuspendingFns`: async functions, functions with effects or a `PerformEffect`, and their transitive callers), then for each function:

1. `orchestrator::lower_async_function` (`crates/passes/krio_adapter/src/orchestrator.rs`) runs `krio_async::transform_to_state_machine_with_options` over a `HirCoroCfg` view of the function. `HirAsyncHooks::classify` (`krio_adapter/src/lib.rs`) reports both `Call(Intrinsic::Await)` and `PerformEffect` as suspension sites, so awaits and performs share one numbering and one dispatcher. Then it:
   - saves each value live across a suspension into a slot before the suspension and reloads it at the resume block (`emit::emit_save_load`);
   - adds the dispatcher, a `Switch` on slot 0 (`emit::emit_dispatcher`);
   - rebuilds SSA for the reloads (`emit::repair_ssa_for_reloads`), treating each await or perform result as defined at its resume block, which is where it is reloaded (the suspension block returns and dominates nothing after it);
   - lowers awaits (`abi_emit::lower_await_calls`) and performs (`abi_emit::lower_perform_effect_calls`), and repairs phi predecessors and phi types (`repair_phi_predecessors`, `insert_phi_coercions`).
2. `abi_emit::reshape_to_poll_abi` gives the function the poll signature, prepends a prologue that loads the parameters from their slots, rewrites every suspension return to Pending, and renames it `NAME$poll`.
3. When the function performs an effect with a resumable handler, `upgrade_resume_struct_at_perform_sites` builds the perform sites of section 5.7.
4. `abi_emit::generate_promise_entry` builds the public `NAME`: same parameters, returns a `*Promise`.

**Slot layout.** The state machine is an array of 8-byte slots, zeroed by the entry. Slot 0 is the state id; slots 1 to N hold the parameters; krio's captures follow; each await adds a promise slot and a result slot (a host-bridge await adds only a result slot); each perform adds a result slot; after those comes one five-slot `Resume` record per perform site (at least one), then a final refcount slot. Values of every type are stored as 64-bit patterns: integers widened, floats and pointers by their bits.

### 6.3 Poll ABI and the promise

A poll function is `extern "C" fn(state_machine: *mut u8) -> i64`, pinned to the platform C convention because the runtime calls it through a transmuted pointer. It returns 0 for Pending and the result, cast to `i64`, for Ready; a function returning nothing returns 1.

The entry allocates the state machine and a 16-byte promise `{ state_machine @0, poll_fn @8 }` and returns the promise. `ZyntaxPromise::from_async_call` (`crates/zyntax_embed/src/runtime/promise.rs`) copies both fields and frees the 16 bytes at once. `ZyntaxPromise::poll` calls the poll function once and maps the result: 0 is Pending, a negative value is `Failed("Async operation failed with code N")`, and anything else is `Ready(ZyntaxValue::Int(v))`.

`PromiseInner::drop` frees the state machine once the task is not Pending, on the thread that ran it, and only when `host_futures::sm_is_referenced` says no table still names it (a parked future, a latched completion, a handler-performer pairing). Before freeing, it clears the task's own completion latch, since a later allocation at the same address would otherwise look already complete.

### 6.4 Await

`SsaBuilder` lowers `await call(args)` to the call, typed as returning a promise, followed by `Call(Intrinsic::Await, [promise])`. An `await` inside a `fiber def` is refused there: "await is not allowed inside a fiber function". An await that reaches Cranelift untransformed, typically one outside an `async def`, is a codegen error: "unlowered `await` reached codegen".

`lower_await_calls` rewrites each await into one of two shapes.

**Awaiting another async function** polls it in place. On first entry the block calls the callee and stores its promise in the await's promise slot; every entry loads the promise and calls its `poll_fn` on its state machine. If the callee is Pending, the caller returns Pending. If it is Ready, the caller saves the result, clears the promise slot (so an await inside a loop starts a fresh call next time), writes the next state, and returns Pending; the next poll continues from the resume block.

**Awaiting a host bridge** (a call to a `Symbol` whose name starts with `__zyntax_async_`, `HOST_BRIDGE_PREFIX`) parks instead. The block calls `__zyntax_register_future(poll_fn, state_machine, result_offset, next_state, 0, 0)`, saves every live value and the next state, and only then calls the bridge with the returned handle prepended, and returns Pending. The saves come before the bridge because a bridge may resolve the future before it returns. When the host event fires, `host_futures::resolve_future` resumes the machine.

### 6.5 The executor

The executor is `host_futures.rs` plus the drivers in `runtime/promise.rs`.

- **`FUTURE_TABLE`** maps a handle to a `ParkedFuture { poll_fn_ptr, state_machine_ptr, result_slot_offset, next_state, refcount_offset, task_id }`. `__zyntax_register_future` fills `task_id` from `CURRENT_TASK_ID` when codegen passes 0, which it always does.
- **`TIMER_QUEUE`** (native only) holds `(deadline, handle)` pairs. `__zyntax_async_set_timeout(handle, ms)` only schedules a timer and returns, so the task parks instead of blocking the thread.
- **`resolve_future(handle, value)`** removes the entry, writes `value` into the result slot and `next_state` into slot 0, and polls the machine exactly once. If it reaches Ready, the value is routed to the performer when the machine is an async handler (`route_handler_completion`); otherwise it is latched by state-machine pointer (`record_sm_completion`) and reported through `complete_task`. One poll per resolve is what keeps the host's event loop in control on wasm.
- **`drive_until(promises, deadline, done)`** is the core driver. It polls every task once, stamping each poll with the task's index as its task id and bracketing it with the task's handler segment. It then loops: tear down tasks marked cancelled, harvest completions latched for the tasks' own machines, stop when `done` holds or the deadline passes, and otherwise sleep until the nearest timer and resolve it (`drive_next_timer_with_task`), marking the owning promise Ready when its machine completes. With no timers left, it re-polls the pending tasks.

`await_raw`, `drive_tasks`, `await_with_timeout`, `PromiseAll::await_all` (and its timeout form) and `PromiseRace::await_first` all run through `drive_until`. `await_result`, `poll_with_limit`, `PromiseRace::await_first_with_timeout`, `PromiseAllSettled` and the `std::future::Future` impl only call `poll` and never drive the timer queue, so a task parked on a host bridge does not finish under them on native.

Task ids are indices into the slice a driver was given. Async handler drives use negative ids from `alloc_driver_task_id`, so they never collide with a top-level task.

### 6.6 Cancellation

`ZyntaxPromise::cancel` marks a pending promise Cancelled. The teardown happens when a driver sees the mark: `drive_until` calls `host_futures::deregister_task(id)` for each cancelled task, `PromiseRace::await_first` cancels and deregisters the losers, and `await_with_timeout` cancels the overrunner. `deregister_task` removes the task's parked futures and timers, frees every fiber the task created and has not dropped (`krio_adapter::fiber::free_task_fibers`, from the `TASK_FIBERS` registry that `fiber_new` fills under the current task id), and forgets the task's handler segment. No code of the cancelled task runs: it gets no cleanup path and no pops.

### 6.7 wasm

The wasm path (`crates/zyntax_wasm/src/lib.rs`) has no `ZyntaxRuntime` or `ZyntaxPromise`; it compiles with `use_krio_async` and runs the result in the bytecode interpreter. On wasm32, the poll pointer stored in a promise or a parked future is an interpreter closure handle, so `resolve_future` polls through `WASM_POLL_DISPATCH`, installed by the shim. `__zyntax_async_set_timeout` is routed to a JS `setTimeout` that calls the exported `_zyntax_resolve_future`, and completion reaches JS through `set_complete_task_callback`. The JS event loop plays the role of the timer queue and `drive_until`.

## 7. Fibers

### 7.1 The contract

`FiberCfg` (`crates/compiler/src/fiber_backend.rs`) is the contract a fiber backend implements: `fiber_new`, `fiber_new_with_env`, `fiber_env`, `fiber_resume`, `fiber_resume_with`, `fiber_yield`, `fiber_take_input`, `fiber_transfer`, `fiber_cancel`, `fiber_free`, `fiber_abort_with`, `fiber_take_error`, plus `stack_windows` and `held_addresses` for the collector. One backend is installed per process through a set-once slot (`install_fiber_backend`); both runtimes install `krio_adapter::fiber::KrioFiberBackend` at construction. The `krio_fiber_*` extern functions in `crates/compiler/src/zrtl.rs` forward to the installed backend and panic when none is installed. Handles are opaque `*mut FiberRepr`.

The HIR carries fibers as operations rather than calls so that the choice of implementation stays in one lowering pass: a backend without stack switching can replace `apply_krio_fiber_lowering` without touching any frontend or code generator.

### 7.2 The krio-fiber backend

`KrioFiberBackend` (`crates/passes/krio_adapter/src/fiber.rs`) boxes a `krio_fiber::Fiber` per handle; a stack size of 0 selects krio-fiber's default (`DEFAULT_STACK_SIZE`, 64 KiB), and the stack is mmap'd with a guard page. The body is an `extern "C" fn()` with no parameters. Thread-local tables hold what does not fit in the fiber object:

- `RUNNING`: fibers being resumed, innermost last, so a body can find its own handle.
- `ENV_MAP`: each fiber's environment pointer (`fiber_new_with_env`, read back by `krio_fiber_env`).
- `RESUME_INPUT`: the value `resume_with` delivered, until `krio_fiber_take_input` reads it.
- `ABORT_PAYLOAD` and `ERROR_MAP`: section 7.5.
- `FIBER_TASK_ID` and `TASK_FIBERS`: which task made which fiber, for cancellation.

For the collector, `stack_windows` reports the live window of every running and suspended fiber stack, and `held_addresses` reports environments and pending error payloads as roots. Compiling a function the first time it is called happens on a separate thread (`TieredBackend`'s lazy compiler), because the call may come from a fiber stack too small for a compile.

### 7.3 Lowering

A call to a fiber def lowers to `CreateClosure` of the body with no captures, which yields the body's raw address, and `FiberNew { closure, stack_size: 0 }` (`SsaBuilder`, in the `Call` arm). The call's arguments are not passed (section 12). `f.next()` lowers through `FiberClass::dispatch` to `SsaBuilder::emit_fiber_next`: `FiberResume`, then an inline decode of the step into `Option<T>` (variant 1 `Some(payload)` for a yield, variant 0 `None` otherwise) using a `Select`.

`apply_krio_fiber_lowering` (`crates/compiler/src/fiber_lowering.rs`) rewrites every fiber operation into runtime calls before any backend runs, so no backend needs a fiber arm:

| HIR | Calls |
|---|---|
| `FiberNew` | `krio_fiber_new(closure, stack_size)` |
| `FiberResume` | `__zyntax_effect_fiber_enter`, `krio_fiber_resume(fiber)`, `__zyntax_effect_fiber_leave` |
| `FiberResumeWith` | `__zyntax_effect_fiber_enter`, `krio_fiber_resume_with(fiber, value)`, `__zyntax_effect_fiber_leave` |
| `FiberYield` | `krio_fiber_yield(value)` |
| `FiberTransfer` | `krio_fiber_transfer(target, value)` |
| `FiberCancel` | `krio_fiber_cancel(fiber)` |
| `FiberDrop` | `__zyntax_effect_fiber_forget(fiber)`, `krio_fiber_free(fiber)` |

### 7.4 Steps

A resume returns a packed `i64`: bits 0 and 1 are the tag (0 Yielded, 1 Done, 2 Errored) and bits 2 to 63 hold the payload (`pack_fiber_step`, `unpack_fiber_step`). A yielded value therefore travels as a 62-bit signed integer. Resuming a fiber whose body has returned, or whose body panicked, returns Done again rather than failing.

### 7.5 Cancel, abort and errors

`cancel()` sets krio-fiber's cancel flag. `KrioFiberBackend::fiber_resume` checks the flag at the resume boundary and returns Done without running the body, so `next()` returns `None` from then on. The fiber is not freed.

`Fiber.abort(x)` is deferred: `fiber_abort_with` latches `(variant, payload)` in `ABORT_PAYLOAD` and yields. The resumer's `encode_step` sees the latch, records the pair in `ERROR_MAP` under the fiber's handle, and reports Errored, so `next()` returns `None`. The abort is deferred because an immediate one would have to unwind through JIT-compiled frames on the fiber's stack, and the runtime does not unwind through those frames. The fiber stays suspended at the abort point. A panic caught by krio-fiber's trampoline also reports Errored, with no payload.

`error()` calls `krio_fiber_take_error`, which removes the entry and packs it (bit 0 present, bits 1 and 2 the variant, the payload from bit 3 up); `SsaBuilder::emit_fiber_error` builds `Option<FiberError<String>>` from it.

### 7.6 Drop

`LoweringContext::emit_fiber_drops` runs on each function after SSA construction. For every `FiberNew` in the entry block whose result is not returned and does not flow into a phi, it inserts `FiberDrop` before every `Return`. Entry-block fibers are made unconditionally and dominate every return, so a drop there never frees an uninitialized handle; a fiber that is returned becomes the caller's to free. The rule trades leaks for safety: fibers created anywhere else are never freed by compiled code (section 12). `KrioFiberBackend::fiber_free` removes the fiber's error and environment entries and its task registration before dropping the box, which unmaps the stack.

### 7.7 Environments, resume values and transfer

- **Environments.** `krio_fiber_new_with_env(body, env, stack)` attaches an address that the body reads with `krio_fiber_env()`. That is how a body with no parameters receives arguments and captures. zypy's generators and zylua's coroutines use it; ZynML's `fiber def` does not.
- **Resume values.** `krio_fiber_resume_with(fiber, v)` delivers `v` as the result of the fiber's pending yield. `FiberYield` is a void call, so the backend keeps `v` until the body reads it with `krio_fiber_take_input()`. ZynML's `yield` is a statement and has no receiving form.
- **Transfer.** `FiberTransfer` exists in the HIR and lowers to `krio_fiber_transfer`, which panics: krio-fiber has no symmetric switch.

## 8. Host API

### 8.1 `ZyntaxRuntime`

`crates/zyntax_embed/src/runtime/classic.rs`.

- `call_async(name, args) -> ZyntaxPromise` calls a promise entry. Drive the promise with `await_raw`, `await_with_timeout`, the combinators, or `drive_tasks(&[...])` for several tasks at once (section 6.5).
- `call_fiber(name, args) -> ZyntaxFiber` makes a fiber from a parameterless fiber def. `ZyntaxFiber` (`crates/zyntax_embed/src/fiber.rs`) is `!Send`, refuses use from any thread but its creator's, offers `resume`, `resume_with` and `cancel`, and frees the fiber on drop. It resumes without a handler-segment bracket (section 12).

```rust
let mut rt = ZyntaxRuntime::new()?;
rt.config_mut().builtins.insert("sleep".into(), "__zyntax_async_set_timeout".into());
rt.compile_typed_program(program)?;
let a = rt.call_async("taskA", &[])?;
let b = rt.call_async("taskB", &[])?;
let results = drive_tasks(&[a, b]);
```

### 8.2 `TieredRuntime`

`crates/zyntax_embed/src/runtime/tiered.rs` lets a host own and drive machines built from `fiber def`s, with effect handlers installed around them. Tokens name machines, not code, so they survive OSR and reloads.

**Fibers.**

- `get_fiber(name) -> FiberToken` makes a fresh paused instance of a parameterless fiber def. Names resolve against `module::name` when unambiguous, and an ambiguous name is an error that lists the candidates.
- `resume_fiber(token)` runs one step and returns a `HostFiberStep`: `Yielded(value)`, `Done` (repeated on every later resume), `Errored`, or `MachineGone` when a reload removed the function (the fiber is not resumed).
- `resume_fiber_within(token, &["H", ...])` does the same with those handlers pushed around the step, leftmost outermost, the host's equivalent of `with H { f.next() }`. The step is bracketed with the fiber's segment, so the machine's own scopes keep precedence. A stateful handler gets fresh state on every call.
- `resume_fiber_handled(token, &[EffectHandlerToken])` is the same with pinned handler tokens.
- `drop_fiber(token)` forgets the segment and frees the fiber. `fiber_info(token)` reports the yield shape the handle decodes with, its generation, and whether a reload changed the function's yield shape (`shape_stale`) or removed it (`machine_gone`).

**Handlers.**

- `get_effect_handler(name) -> EffectHandlerToken` resolves a handler name once and pins the result, so later edits that add same-named handlers cannot re-aim the binding.
- `new_handler_instance(token) -> HandlerInstance` allocates one state region by running `H$new`.
- `bind_fiber_handler_instance(fiber, instance)` inserts the instance's frame at the bottom of the fiber's segment for its lifetime.
- `push_handler_instance(instance) -> HandlerFrame` and `pop_effect_handler(frame)` install it around whatever the host does next.
- `push_effect_handler(token)` and `bind_fiber_handler(fiber, token)` are shorthands that allocate an instance the runtime owns and immediately disowns.
- `drop_handler_instance(instance)` gives up the owner's claim.

An instance counts its installs: `HandlerInstanceEntry { installs, dropped_by_owner }`, reaped by `reap_handler_instance` and released with `effect_runtime::free_handler_state` when the owner has let go and no install remains. Handler state is owner-driven rather than scope-driven because one region can back a binding and any number of pushes.

**Callbacks.** `capture_handler_context()` snapshots the frames in force (claiming an install on each instance a frame names); `enter_handler_context` and `leave_handler_context` reinstate them around a callback that runs later; `release_handler_context` gives the claims back.

**Guards and reload.** `call_raw` refuses a call that reaches a stateful effect with no frame in scope (section 5.6). A reload migrates live handler state into an edited layout through `effect_runtime::migrate_handler_states`, which visits the shared stack and every fiber and task segment and migrates each distinct region once.

```rust
let fiber = rt.get_fiber("machine")?;
let feed = rt.get_effect_handler("Feed")?;
rt.bind_fiber_handler(fiber, feed)?;
while let HostFiberStep::Yielded(state) = rt.resume_fiber(fiber)? {
    // observe `state`
}
rt.drop_fiber(fiber)?;
```

## 9. Frontends

### 9.1 ZynML

ZynML uses every construct in sections 2 to 8 directly.

### 9.2 zypy

`crates/zyntax_python`.

**Generators are fibers.** A function containing `yield` is typed `Ty::Gen` (`types::is_generator`), lowered to `Type::Fiber(Any)`, and compiled as a body with `is_fiber: true`, no parameters and return type `Any`. Arguments and cells reach it through the fiber environment (`generator_prologue`, reading `zb_fiber_env`). `yield v` becomes `TypedStatement::Yield`, hence `FiberYield`. A generator starts through `zb_fiber_start`, a library function over `krio_fiber_new_with_env` (`crates/zyntax_builtins/src/functions.rs`), not through `FiberNew`. `for`, `list`, `sum`, `next` and generator expressions consume it with `next()` on `Fiber<Any>`, so the resume is `FiberResume` and carries the segment bracket. A fresh generator drained on the spot is freed with `zb_fiber_free`. Expression `yield`, `yield from`, returning a value, and `send`, `throw` and `close` are not supported (4f32f2991b9e94a1078ee397c15f7911903ea4405bdb3f75b836ed3a692e366a).

**Async is refused.** `async def`, `async for`, `async with` and `await` fail with "`FORM` is not supported yet" (`crates/zyntax_python/src/lower.rs`, `bytes.rs`). There is no `asyncio` module; `import asyncio` resolves to the embedding host's module. The design that maps asyncio onto the machinery in section 6 is on e9086df7a988adbd490fd072b65be7ffa70295a5b4aaecc653b39fd078d4ce51.

**No effects.** zypy programs declare no effects or handlers and turn the effect pattern rewrites off.

### 9.3 zylua

`crates/zyntax_lua/src/library/coroutines.rs`.

**Coroutines are fibers, reached through runtime symbols rather than HIR fiber operations.** The coroutine library is written as typed-AST library functions that declare externs bound to `krio_fiber_new_with_env`, `krio_fiber_resume_with`, `krio_fiber_yield`, `krio_fiber_take_input` and `krio_fiber_free`. A coroutine is a record holding the fiber handle, its status, its function, and a slot through which values pass in both directions. The fiber's body is one trampoline, `zl_co_body`, which reads the record back as its environment. `coroutine.resume` resumes with a value and decodes the packed step; `coroutine.yield` yields and takes the next resume's arguments with `krio_fiber_take_input`. Each coroutine reserves a 64 MiB stack, committed as it is touched.

Being stackful, a yield works at any depth: inside `pcall`, metamethods and `for` iterators. A yield inside a library callback (a `table.sort` comparator, a `gsub` function) suspends through the library call, where the reference implementation refuses (c33bf811417b293c2a1a1b826ed32d318a79732e16ca767a922054eb8550f441). The C API's `lua_yieldk`, `lua_resume` and `lua_newthread` raise "not provided by zylua yet". A suspended or never-started coroutine is never collected (224cd147eb57b76b29c83e088dcf50fcc0481481b9f2fb16561622d3751f1166).

Lua has no effects or async; zylua turns the effect pattern rewrites off.

## 10. Code generation by tier

The table describes code after the krio and fiber passes, which is what the tiers see.

| Construct | HIR interpreter | Cranelift | LLVM | wasm backend |
|---|---|---|---|---|
| raw `PerformEffect` | refused; the function runs on Cranelift | static default plus `lookup_op` dispatch, `self` from `lookup_state` | direct call to one handler, no `with`, no state | no arm |
| resumable perform site (`Select`, `IndirectCall`, runtime calls) | runs | runs | no `IndirectCall` arm | runs |
| `AsyncSaveSlot`, `AsyncLoadSlot` | runs | runs | no arm | runs |
| fiber operations (runtime calls) | run as foreign calls | run | run | not produced: the wasm path skips the fiber lowering, and the interpreter refuses the raw ops |
| `Intrinsic::Await` | refused | error | error | refused |
| `HandleEffect`, `Resume`, `AbortEffect`, `CaptureContinuation` | refused | placeholders | placeholders | no arm |

**Interpreter** (`crates/compiler/src/hir_interp.rs`). A refused instruction is named by `instruction_name`, for example "performing effect operation `op`: the bytecode interpreter cannot run algebraic effects, so this function needs a JIT tier". Under `TieredRuntime` the refusal is not an error: the function is marked uncompilable and runs through the native bridge (`run_natively_or`), whose stub compiles it with Cranelift. In `InterpRuntime` (the wasm path) there is no native bridge, and the refusal stands. `hir_interp::unsupported_constructs` lets a caller for which the interpreter is the only engine ask up front. A function whose address is taken, which includes every fiber body and poll function, gets no interpreter OSR sites.

**Cranelift** (`crates/compiler/src/cranelift_backend.rs`) implements everything in sections 5 to 7. Raw fiber operations have no arm because none reach it.

**LLVM** (`crates/compiler/src/llvm_backend.rs`) is the optimizing tier when `TieredConfig::tier2_backend` is `Tier2Backend::LLVM` (as in `TieredConfig::production_llvm()`, with the `llvm-backend` feature); the default is Cranelift. Its `PerformEffect` arm calls the handler recorded in `effect_handler_index` directly, and it has no arm for the slot ops or `IndirectCall`, so compiling a poll function or a resumable perform site fails with "Instruction not yet implemented", which `TieredBackend` logs. See section 12.

**OSR** (`crates/compiler/src/osr.rs`). `layout_supports` whitelists the instructions a loop may reach before an OSR entry is built; `PerformEffect`, the slot ops, `IndirectCall` and `CreateClosure` are not on it, so such loops are not entered mid-flight.

**wasm backend** (`crates/compiler/src/wasm_backend.rs`) lowers the slot ops and calls. The wasm JIT hook in `crates/zyntax_wasm/src/lib.rs` takes only parameterless functions returning `i64`, so poll functions stay in the interpreter.

## 11. Composition

### 11.1 Effects inside fibers

A fiber body may perform effects whose handlers are non-resumable; annotate the `fiber def` with `@effect(E)`. The perform resolves against the handlers in force where the fiber was resumed, layered under the fiber's own `with` scopes (section 5.9). A handler scope around the resume site, one inside the body, and a host binding all apply as their nesting says, and interleaved fibers keep their own scopes.

A fiber def that performs an effect with a resumable handler is refused by `apply_krio_effect_lowering`: "fiber function `NAME` performs a resumable algebraic effect". A resumable perform needs its performer to be a state machine that `k(v)` can re-poll, and a fiber body is a stackful function the fiber runtime switches into; the two disagree on any value live across both a perform and a yield.

### 11.2 Fibers inside async functions

An async function may create and drive fibers. The fiber handle is an ordinary value, saved across each suspension point like any other, including a resumable perform, where the save is placed before the suspension. A fiber the task creates is filed under the task's id and freed if the task is cancelled (section 6.6).

`f.next()` runs the fiber's step synchronously on the task's stack: the executor runs nothing else until the task reaches an `await`. Interleaving fiber steps of different tasks is what `@cooperative` is reserved for (section 12).

### 11.3 Await inside a fiber

Refused in `SsaBuilder`: "await is not allowed inside a fiber function". A fiber suspends by switching stacks, not by returning Pending, so there is no state machine for the executor to resume. Drive the fiber from an async function and await there.

### 11.4 Effects inside async functions

An async function performs effects like any other `@effect` function. A non-resumable perform is compiled as in section 5.5 and is also a suspension site of the state machine: the result is saved and the task returns Pending once. A resumable perform becomes a suspension site whose resume state `k(v)` targets (section 5.7). A continuation that reaches an `await` parks on its timer, and `__zyntax_effect_resume` drives timers until the performer is Ready. Values produced by an await or a perform stay valid across later conditional suspensions, because SSA repair treats them as defined at their resume blocks.

### 11.5 Async handler operations

Section 5.8. The handler's awaits park on the executor's timers under its own negative task id and handler segment, so it interleaves with other tasks, and its return value completes the performer. One operation may have async and sync handlers in different scopes.

### 11.6 Handler scopes across suspension points

A `with` scope in an async function may span `await`s: its frame lives in the task's segment while the task is parked and is re-pushed on the next poll. The isolation holds only when the task is driven through `drive_until` or `drive_next_timer_with_task`; a host that calls `ZyntaxPromise::poll` itself gets no bracket. A `with` inside a fiber body may span `yield`s in the same way, through the fiber's segment.

### 11.7 Cancellation and drop

A cancelled task's fibers are freed and its handler segment is forgotten, but none of its code runs (section 6.6). A fiber created in a function's entry block and not returned is freed at every return, in async functions too, since the drop is placed before the async transform. A `with` scope's handler state is freed on its exit edges.

### 11.8 Host-driven machines

`TieredRuntime` fibers compose with effects through `resume_fiber_within`, `bind_fiber_handler` and handler instances (section 8.2). A host-driven machine takes its inputs through effects, which is why `get_fiber` requires a parameterless fiber def.

## 12. Limits

Each item states what the code does. The trailing ids are git-bug issues (`git-bug bug show <id>`).

**Effects**

- `with` takes one handler and no constructor arguments; nest `with` blocks for several. Handler state starts from its declared initializers. d39c155e769981fca45f1a86a46a13e43ae15a0542c990b4ad5bb9d0b852ab74
- A `with` scope left through a conditional branch with one target outside the scope, or through a `Switch`, keeps its frame pushed; lowering logs "mixed in/out CondBranch in handler scope". 812731d26b7d79dd4598bfff7f3e033fbf3609bafa896b04725fb03396bc6124
- Resumable operations have no `self`, so a handler cannot keep a continuation in its state and resume it later. 1acfd94a62e0c05bcdd02ca383269bce1714739a954196cbf46a246bbb63b75e
- Handler state is freed without dropping its fields. d40f93ff54d09cc33463bfae1f1153307ef2a2332c0afd0f57a8277ae00100a5
- A second `k(v)` re-polls the same state machine: the continuation is not copied, so it runs over what the first resumption left, including resources it released. 0084b0d0a872e4b708cdb3c75fd49dccc762d0bfb0eb48f69c2222951dda1051
- Two async handler shapes compile and misbehave: an `async def` operation without a `Resume<T>` parameter (the perform's value is the promise pointer), and an async resumable operation performed from a function that is not `async` (its sync entry never drives the handler's timer). fc27dba84e52f4b16c5595a18a4cf01143d70179d04c147dd4685489a564a46b
- A stateful handler outside any `with` or host install runs with a null `self`; `TieredRuntime::call_raw` refuses such calls, `ZyntaxRuntime` does not.
- Type arguments on `handler H for E<T>` are parsed and discarded.
- `__zyntax_effect_lookup_op` does not bounds-check the operation index against the table. e2ebd373ef36f7817981295bec331c4d91feae8deeb4374a437e25803cae67db

**Async**

- A poll result of 0 is Pending, so an async function whose result is 0, `false` or null never completes, and the host reports a negative result as a failure. ab01238df74d256eacb6482c2ec879540eec3e04305ba0c8c6f71ff183ff09ff
- The executor re-polls a task only through timers its own machine parked; a task that returns Pending with nothing parked waits until no timer is left. 2af438da470f05d30aa913ae21b68617ed51fb34ada9c3ee2b96b14b3ab24ec5
- The promise and state machine of an awaited async function, and those of an async handler, are never freed. 90679a74258281aef8322c383110dbe466d18131d13677012e4326e576648d62
- Cancellation runs none of the task's code and frees its handler segment without freeing the state of its open `with` scopes. Task ids are slice indices, and 0 doubles as "no task". 3cde2035080ecd7ff650450ebb650ea2e356a152b634a98562bf9615bfc56b33
- An async function called without `await` returns its promise pointer typed as its declared result. ZynML has no promise type, `spawn` or `select`, and `zynml::ZynML` has no async entry point: hosts drive async functions through `call_async`. The only host bridge the runtimes register is `__zyntax_async_set_timeout`.
- `ZyntaxPromise`'s results are always `ZyntaxValue::Int`.
- The interpreter stores a float into an async slot by value and reads it back by bits. bc1a1740cd6e59aae9ac234bc724b449c20cd10d71579ec3327e16990ae4e80a

**Fibers**

- A fiber def's parameters receive no arguments. 327a3a220c076ee63f56dcdfe5c6c0cde8a807a068fbdebc4fe68bcc8523405e
- Compiled code frees only fibers made in a function's entry block that are never returned; others leak, and an entry-block fiber stored into memory is still freed at return. 1e74c68a113719046d075411cba8bf8977d7e78bb3ef42ddaa5564db8dc464a6
- `error()` decodes the payload as a `String` whatever `Fiber.abort` was given, and resuming after an abort continues the body past the abort point. 4bf11db6488368b067c22e9c36dea6517ebd3a1956734737bfa04f8c974d8f0c
- A yielded value must fit in 62 bits.
- `FiberTransfer` panics at run time; `FiberResumeWith` and expression `yield` are not reachable from ZynML source.
- `ZyntaxFiber` resumes without the handler-segment bracket, and `TieredRuntime::resume_fiber_within` leaks the state it allocates for each step. a7adf968f352b2f3c516a28386c8781e3eb052ab4daef972ccdbe8cf2745d368
- `krio_fiber_resume` accepts a call from any thread; only `ZyntaxFiber` checks. A single scheduler for fibers, tasks and continuations, with per-task storage in place of the thread-locals, is part of 754e04e0a256d001130951f71acd7db20d2ffa2e608f443123d18b96546f626c.

**Composition**

- A fiber def cannot perform a resumable effect. b61285e0d0b6f8a1ce0d654121ab9824eb9e16018ace640fb6c6b9735c456048
- A fiber body cannot `await`, and a fiber step inside an async task holds the executor; `@cooperative` is recorded and read by nothing. 84adfc065267ef0e2332ba2718e9faba6f013782f6e5eb6a8ceb3b9ceee7b0ce

**Tiers**

- The LLVM tier compiles a perform as a direct call that ignores `with` scopes and handler state, and it cannot compile poll functions or resumable perform sites. 73f2922a03fd37e5fff88f12fcee4324c26ad834e3344eac3fe3b0b3e578ccc0
- On wasm, effects and fibers do not run; async functions do. b991f13ba3899b642a7f7ddd1eecd7cf8c4aa8e0fe7935a45dea5b166cb239dc

## 13. Source map

| Area | Files |
|---|---|
| Grammar and prelude | `crates/zynml/ml.zyn`, `crates/zynml/stdlib/prelude.zynml` |
| Typed AST | `crates/typed_ast/src/typed_ast.rs`, `crates/typed_ast/src/type_registry.rs` |
| Effect pattern pass | `crates/passes/algebraic_effects/src/{lib,annotations,dispatch,vtable}.rs` |
| Handler state synthesis | `crates/zyntax_embed/src/runtime/handler_state.rs` |
| Lowering driver | `crates/zyntax_embed/src/lower.rs` |
| HIR | `crates/compiler/src/hir.rs` |
| SSA: performs, fibers, await | `crates/compiler/src/ssa.rs`, `crates/compiler/src/builtin_class.rs` |
| `with` scopes, op tables, fiber drops, `@with` refusal | `crates/compiler/src/lowering.rs`, `crates/compiler/src/typed_cfg.rs` |
| krio orchestration | `crates/zyntax_embed/src/krio_lowering.rs` |
| State-machine transform | `crates/passes/krio_adapter/src/{lib,orchestrator,emit,abi_emit}.rs` |
| Legacy async lowering | `crates/compiler/src/async_support.rs` |
| Fiber lowering and contract | `crates/compiler/src/fiber_lowering.rs`, `crates/compiler/src/fiber_backend.rs`, `crates/compiler/src/zrtl.rs` |
| Fiber backend | `crates/passes/krio_adapter/src/fiber.rs` |
| Effect runtime | `crates/zyntax_embed/src/effect_runtime.rs` |
| Futures, timers, completions | `crates/zyntax_embed/src/host_futures.rs` |
| Promises and drivers | `crates/zyntax_embed/src/runtime/promise.rs` |
| Host fiber and handler API | `crates/zyntax_embed/src/runtime/tiered.rs`, `crates/zyntax_embed/src/fiber.rs` |
| Tiers | `crates/compiler/src/hir_interp.rs`, `cranelift_backend.rs`, `llvm_backend.rs`, `wasm_backend.rs`, `osr.rs`, `tiered_backend.rs` |
| wasm runtime | `crates/zyntax_wasm/src/lib.rs` |
| Frontends | `crates/zyntax_python/src/lower.rs`, `crates/zyntax_lua/src/library/coroutines.rs` |
| Tests | `crates/zynml/tests/{handler_state,handler_state_lifetime,handler_declaration_order,handler_fiber_stack,handler_instance_repro,effect_regional_dispatch,effect_regional_resumable,effect_source_e2e,effect_on_impl_method,fiber_execution,fiber_drop,fiber_in_async,async_effect_composition,cooperative_executor,host_fiber_api}.rs`, `crates/zyntax_embed/tests/{async_runtime_tests,cooperative_resume,effect_runtime_tests}.rs` |
