# A rebuild leaves every loaded module but one without its globals

**Status**: reported, cause identified, repro attached
**Raised by**: Blinc (`blinc_dsl_core`), 2026-09-05
**Rev**: `0a169cc`
**Platform**: macOS arm64, Cranelift tier 0
**Related**: `crates/compiler/src/cranelift_backend.rs`
(`rebuild_with_accumulated_symbols`), `crates/compiler/src/tiered_backend.rs`
(`compile_module`, `try_handler_push_info`),
`docs/PERFORM_WITHOUT_A_FRAME_IN_A_CALLED_FUNCTION.md` (the same crash,
seen from the outside before the cause was known)

## Summary

`rebuild_with_accumulated_symbols` replaces the JIT module and clears
`global_map`. `compile_module` then recompiles at most one module, the
one in `current_module`. Every other module that has been loaded keeps
its HIR, keeps its compiled code, and loses the addresses of its
globals. A handler's `$optable$` global is one of those, so after the
second or third file of a program is loaded, the first file's handlers
can no longer be installed correctly.

There is no ordering that avoids it. Ask for the op table before the
rebuild and you get an address that stops being the table. Ask after
and you get an error saying the global was never declared.

## What the host sees

A program of two files. `page.blinc` declares an FSM with a context
field and a component that reads it; `main.blinc` imports the
component and mounts it. Each file compiles into its own module, and
after each one the host exports that module's `$view` symbols and calls
`finalize_runtime_symbols`, so the next file can link against them.

Creating the handler instance right after its own module compiles,
which is what Blinc used to do:

```
__zyntax_effect_lookup_op(effect, 2) -> 0x6764695700000007   SIGBUS
```

The frame is there and its `op_table` is non-null, so the perform site
takes the dynamic branch and calls what it read. Reading the table
through the same lookup shows what it actually points at:

```
op_table[0] = 0x697461676976616e   "navigati"
op_table[1] = 0x0000000000006e6f   "on"
op_table[2] = 0x6764695700000007   {7, "Widg"}
```

String constants from another module. The address was recorded in
`HandlerInstanceEntry.table` when the instance was created, and the
rebuild that followed made it point at something else.

Creating the instance after every file has loaded instead, which is the
ordering the runtime asks for:

```
`$optable$page$Tick$HostEvents` (HirId(7994)) is in the module but was
never declared to the backend, so it has no address. A JIT rebuild
clears the backend's global map; anything installed before it needs
recompiling.
```

That message is `try_handler_push_info`'s own, and it describes the bug
exactly. No handler can be installed, so the perform site finds no
frame, falls back to the static op, and that op reads an implicit
`self` from a null pointer:

```
ldr x0, [x0]     ; x0 = 0        SIGSEGV
```

Both endings are the same defect at two different moments.

## Where it comes from

`rebuild_with_accumulated_symbols` ends with:

```rust
self.module = JITModule::new(builder);
self.function_map.clear();
self.global_map.clear();
self.compiled_functions.clear();
```

`compile_module` restores one module afterwards:

```rust
if rebuilt {
    if let Some(previous) = self.current_module.clone() {
        self.cranelift.with_lock(|be| be.compile_module(&previous))?;
    }
}
```

The comment above it states the requirement correctly: "Anything
already installed loses its entries, and nothing re-declares them: a
handler's `$optable$` global then has no address". The code satisfies
it for one module. `loaded_modules()` already exists a few hundred
lines away and knows about all of them.

A host cannot compensate. Re-declaring globals is not on the public
API, and the address a `HandlerInstance` holds is captured inside
`new_handler_instance`.

## The ask

1. A rebuild should leave every loaded module addressable. Recompiling
   all of `loaded_modules()` in load order is the direct reading; so is
   re-declaring their globals without re-emitting their code, if that
   is cheaper. Either way the invariant worth stating is that a module
   stays usable for as long as it is loaded, whatever loads after it.

2. `HandlerInstanceEntry` caches `table` as a raw address, so even with
   (1) an instance created before a rebuild points into the module that
   was discarded. Resolving the table at push time, or re-resolving the
   cached entries after a rebuild, would make instance creation
   independent of load order rather than merely legal at one moment.

3. `stateful_effects_reached_by` answers `None` here, which is how the
   crash reached compiled code with the guard already wired in.
   The walk resolves callees with `module.functions.get(&id)` against
   the single module that holds the entry point, so a call into another
   module ends it silently. The reachable perform in this program is
   two modules away from `render_view`. The same cross-module lookup
   that `0a169cc` gave `module_holding` is what the queue needs.

## Repro

In Blinc, `crates/blinc_dsl_core/tests/imported_module_fsm.rs`:

```
cargo test -p blinc_dsl_core --test imported_module_fsm -- --ignored
```

Two tests, two files and three files. Both fault rather than fail,
which is why they are ignored by default. The three-file case is the
one that survives a fix that only restores `current_module`.

Blinc's four playground tests (`pg_nav`, `pg_trigger`, `pg_grow_cost`,
`pg_reactive_baseline`) are the same defect at six files.
