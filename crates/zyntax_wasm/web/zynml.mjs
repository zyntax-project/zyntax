// Thin JS shim over the wasm-pack output.
//
// Lets host pages / tests import a clean ESM API instead of poking at
// `pkg-web/zyntax_wasm.js`'s wasm-bindgen surface directly:
//
//     import { initZynml, run, version, ErrorKind } from "./zynml.mjs";
//     await initZynml();                        // browser only
//     const r = run("def main(): i64 { return 7 }");
//     console.log(r.output, r.ok, r.errorKind);
//
// Two consumers:
//   * Browser pages — point `--target web` output at this. The
//     wasm-bindgen `default` export must be called once with the
//     wasm URL or a fetch promise before `run()` is callable.
//   * Node.js — `--target nodejs` output is auto-initialised, so
//     `initZynml()` returns immediately.
//
// Keeping the shim small means the user-visible API stays stable
// across wasm-bindgen versions; if `pkg-*` regenerates with new
// internals we only have to update the import path here.

/**
 * Initialise the wasm module. Call once at startup.
 *
 * Also installs the JIT host shims (`_zyntax_jit_install`,
 * `_zyntax_jit_call_0_i64`, ...) on `globalThis` so that the
 * interpreter's wasm-JIT tier-up path (Phase E.6) can hand emitted
 * wasm bytes back to the host for `WebAssembly.compile` +
 * `WebAssembly.instantiate`.
 *
 * @param {Object} [opts]
 * @param {URL|RequestInfo} [opts.module]
 *        URL of the `.wasm` file. Browser-only; ignored under Node.
 *        When omitted, wasm-bindgen's default resolution (same dir
 *        as the JS glue) is used.
 * @returns {Promise<void>}
 */
export async function initZynml(opts = {}) {
    // Install JIT host shims BEFORE the wasm module loads so any
    // `start` hook that touches them sees the live globals.
    installJitHost();

    // Lazy load so the shim works under both targets without
    // up-front conditional imports.
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
        // Node: the wasm-pack output auto-initialises on require.
        await import("../pkg-node/zyntax_wasm.js");
    } else {
        const mod = await import("../pkg-web/zyntax_wasm.js");
        await mod.default(opts.module);
        zynmlBindings = mod;
    }
}

// ---------------------------------------------------------------------------
// Wasm-JIT host (Phase E.7)
// ---------------------------------------------------------------------------
//
// Phase E.6 lives in the Rust crate: when a function gets hot, the
// interpreter calls `WasmBackend::compile_function`, ships the bytes
// to JS via `globalThis._zyntax_jit_install`, and caches the
// returned funcref-table index. Future calls dispatch through
// `globalThis._zyntax_jit_call_0_i64`.
//
// We can't store wasm funcrefs in a wasm `WebAssembly.Table` here
// because the BC interpreter (Rust running inside `zyntax_wasm.wasm`)
// can't easily emit `call_indirect` against a host-provided table.
// Instead JS keeps a plain array of `inst.exports.entry` references
// and dispatches them as ordinary JS function calls — the wasm boundary
// crossing is the same as any wasm-bindgen extern.

/** Plain JS array indexed by handle. `jitFuncs[handle]` is the
 *  exported `entry` function from the JIT'd module. */
const jitFuncs = [];

/** Install the JIT-host globals once. Idempotent. */
function installJitHost() {
    if (globalThis._zyntax_jit_install) return; // already installed

    /**
     * Compile + instantiate a single-function wasm module emitted
     * by `WasmBackend::compile_function`. Returns the array index
     * the dispatch shim uses as the JIT handle, or 0xFFFFFFFF on
     * failure (Rust side maps that back to `None`).
     *
     * `bytes` arrives as a `Uint8Array` — wasm-bindgen marshals
     * the `&[u8]` for us. `WebAssembly.Module` takes a copy, so
     * we don't need to worry about wasm-memory lifetimes.
     *
     * Phase E.5: builds an `importObject` for instantiation by
     * parsing the wasm module's imports list. Each import lives
     * under module name `"extern"` and is named `<symbol>@<arity>`;
     * we strip the suffix to recover the symbol name + pick the
     * matching `_zyntax_call_extern_<arity>` dispatcher.
     */
    globalThis._zyntax_jit_install = function _zyntax_jit_install(bytes) {
        try {
            const mod = new WebAssembly.Module(bytes);

            // Build the importObject from the module's declared
            // imports. Two namespaces today:
            //   - `"extern"` — function imports, one per Symbol
            //     callee with an `@arity` suffix in the name.
            //     Each maps to a JS shim that calls back into the
            //     host wasm's `_zyntax_call_extern_<arity>` exports.
            //   - `"host"` — non-function imports the JIT'd module
            //     needs from the runtime. The only entry today is
            //     `"memory"`, the host wasm's linear memory, so
            //     JIT'd `i64.load` / `i64.store` ops read/write the
            //     same heap the BC interpreter built.
            const importObj = {};
            const imports = WebAssembly.Module.imports(mod);
            for (const imp of imports) {
                if (imp.module === "host" && imp.name === "memory") {
                    if (!importObj.host) importObj.host = {};
                    importObj.host.memory = zynmlBindings.memory;
                    continue;
                }
                if (imp.kind !== "function") continue;
                const at = imp.name.lastIndexOf("@");
                if (at < 0) {
                    console.warn(
                        `_zyntax_jit_install: import "${imp.name}" missing @arity suffix`,
                    );
                    continue;
                }
                const baseName = imp.name.slice(0, at);
                const arity = parseInt(imp.name.slice(at + 1), 10);
                if (imp.module === "extern") {
                    if (!importObj.extern) importObj.extern = {};
                    importObj.extern[imp.name] = makeExternDispatcher(baseName, arity);
                    continue;
                }
                if (imp.module === "internal") {
                    // Internal HIR-function call. `baseName` is the
                    // callee's HirId as 32-char lowercase hex.
                    // The host's `_zyntax_call_internal_<arity>`
                    // dispatcher routes back into the runtime's
                    // function table (BC interpreter today; a sibling
                    // JIT'd module in a future tier-up of multi-hot
                    // function programs).
                    if (!importObj.internal) importObj.internal = {};
                    importObj.internal[imp.name] = makeInternalDispatcher(baseName, arity);
                    continue;
                }
                if (imp.module === "host") {
                    // Allocator intrinsics. `baseName` is `malloc` or
                    // `free`; the WasmBackend emits these for
                    // `HirCallable::Intrinsic(Malloc|Free)`. Route via
                    // `_zyntax_call_host_alloc` / `_zyntax_call_host_free`
                    // which call wasm-bindgen's allocator under the
                    // hood.
                    if (!importObj.host) importObj.host = {};
                    importObj.host[imp.name] = makeHostDispatcher(baseName, arity);
                    continue;
                }
                console.warn(
                    `_zyntax_jit_install: unexpected import module "${imp.module}"`,
                );
            }

            const inst = new WebAssembly.Instance(mod, importObj);
            const fn = inst.exports.entry;
            if (typeof fn !== "function") return 0xFFFFFFFF;
            jitFuncs.push(fn);
            return jitFuncs.length - 1;
        } catch (e) {
            // Surface for debugging but don't propagate — the Rust
            // side handles `0xFFFFFFFF` as "JIT install failed,
            // keep the function in BC".
            console.error("_zyntax_jit_install failed:", e);
            return 0xFFFFFFFF;
        }
    };

    /** Build a JS dispatcher that calls the right
     *  `_zyntax_call_extern_<arity>` export with `symbolName` as
     *  the first arg + the JIT'd module's actual args after.
     *
     *  Arities 0-8 are covered today; that range fits every plugin
     *  symbol we ship (zrtl_io / string / math / vector / simd /
     *  tensor / audio / model), the largest of which —
     *  `$Tensor$matmul` — takes 8 i64-funneled args. Higher arities
     *  fall through to a throwing thunk so unsupported calls fault
     *  loudly at the JIT call site instead of silently miscoercing. */
    function makeExternDispatcher(symbolName, arity) {
        const exports = zynmlBindings;
        switch (arity) {
            case 0:
                return () => exports._zyntax_call_extern_0(symbolName);
            case 1:
                return (a0) =>
                    exports._zyntax_call_extern_1(symbolName, a0);
            case 2:
                return (a0, a1) =>
                    exports._zyntax_call_extern_2(symbolName, a0, a1);
            case 3:
                return (a0, a1, a2) =>
                    exports._zyntax_call_extern_3(symbolName, a0, a1, a2);
            case 4:
                return (a0, a1, a2, a3) =>
                    exports._zyntax_call_extern_4(symbolName, a0, a1, a2, a3);
            case 5:
                return (a0, a1, a2, a3, a4) =>
                    exports._zyntax_call_extern_5(symbolName, a0, a1, a2, a3, a4);
            case 6:
                return (a0, a1, a2, a3, a4, a5) =>
                    exports._zyntax_call_extern_6(symbolName, a0, a1, a2, a3, a4, a5);
            case 7:
                return (a0, a1, a2, a3, a4, a5, a6) =>
                    exports._zyntax_call_extern_7(
                        symbolName,
                        a0, a1, a2, a3, a4, a5, a6,
                    );
            case 8:
                return (a0, a1, a2, a3, a4, a5, a6, a7) =>
                    exports._zyntax_call_extern_8(
                        symbolName,
                        a0, a1, a2, a3, a4, a5, a6, a7,
                    );
            default:
                // Out-of-coverage arity: a JIT'd call to this would
                // fault on instantiate (no matching dispatcher).
                // Returning a throwing thunk surfaces the issue at
                // the call site instead of silently coercing.
                return () => {
                    throw new Error(
                        `_zyntax_jit: extern "${symbolName}" arity ${arity} not supported (max 8)`,
                    );
                };
        }
    }

    /** Build a JS dispatcher for an `internal.<hex_id>@<arity>`
     *  import — a call to another HIR function in the same runtime.
     *  Routes through `_zyntax_call_internal_<arity>(hex_id, args)`
     *  which calls back into the BC interpreter (or a sibling JIT'd
     *  module once cross-function JIT lands). Same arity contract
     *  as the extern dispatcher. */
    function makeInternalDispatcher(hexId, arity) {
        const exports = zynmlBindings;
        switch (arity) {
            case 0:
                return () => exports._zyntax_call_internal_0(hexId);
            case 1:
                return (a0) =>
                    exports._zyntax_call_internal_1(hexId, a0);
            case 2:
                return (a0, a1) =>
                    exports._zyntax_call_internal_2(hexId, a0, a1);
            case 3:
                return (a0, a1, a2) =>
                    exports._zyntax_call_internal_3(hexId, a0, a1, a2);
            case 4:
                return (a0, a1, a2, a3) =>
                    exports._zyntax_call_internal_4(hexId, a0, a1, a2, a3);
            case 5:
                return (a0, a1, a2, a3, a4) =>
                    exports._zyntax_call_internal_5(hexId, a0, a1, a2, a3, a4);
            case 6:
                return (a0, a1, a2, a3, a4, a5) =>
                    exports._zyntax_call_internal_6(hexId, a0, a1, a2, a3, a4, a5);
            case 7:
                return (a0, a1, a2, a3, a4, a5, a6) =>
                    exports._zyntax_call_internal_7(
                        hexId,
                        a0, a1, a2, a3, a4, a5, a6,
                    );
            case 8:
                return (a0, a1, a2, a3, a4, a5, a6, a7) =>
                    exports._zyntax_call_internal_8(
                        hexId,
                        a0, a1, a2, a3, a4, a5, a6, a7,
                    );
            default:
                return () => {
                    throw new Error(
                        `_zyntax_jit: internal "${hexId}" arity ${arity} not supported (max 8)`,
                    );
                };
        }
    }

    /** Build a JS dispatcher for `host.malloc@1` / `host.free@1`
     *  imports. The WasmBackend emits these for
     *  `HirCallable::Intrinsic(Malloc|Free)` calls. Routes through
     *  matching `_zyntax_call_host_alloc` / `_zyntax_call_host_free`
     *  exports that delegate to wasm-bindgen's `__wbindgen_malloc` /
     *  `__wbindgen_free`. Only arity 1 is meaningful for either
     *  intrinsic; anything else is a structural bug and throws. */
    function makeHostDispatcher(baseName, arity) {
        const exports = zynmlBindings;
        if (baseName === "malloc" && arity === 1) {
            return (size) => exports._zyntax_call_host_alloc(size);
        }
        if (baseName === "free" && arity === 1) {
            return (ptr) => exports._zyntax_call_host_free(ptr);
        }
        // Per-frame stack ops for Alloca. The WasmBackend emits a
        // prologue (`stack_save@0` → captured into a local) and a
        // matching epilogue at every Return (`stack_restore@1` with
        // the saved mark), so every Alloca'd region auto-frees on
        // function exit.
        if (baseName === "stack_save" && arity === 0) {
            return () => exports._zyntax_call_host_stack_save();
        }
        if (baseName === "stack_alloc" && arity === 1) {
            return (size) => exports._zyntax_call_host_stack_alloc(size);
        }
        if (baseName === "stack_restore" && arity === 1) {
            return (mark) => exports._zyntax_call_host_stack_restore(mark);
        }
        // Cooperative-async host bridges (Phase I+). The wasm
        // module imports `host.async_set_timeout@2` and friends
        // when a krio_adapter-emitted SM reaches an `await
        // __builtin_*` site; the dispatcher here wires each to the
        // matching JS Promise primitive and reports completion
        // back via `_zyntax_resolve_future` (Phase G plumbing).
        if (baseName === "async_set_timeout" && arity === 2) {
            return (handle, ms) => {
                // `handle` arrives as a BigInt (wasm i64). The
                // resolve export also takes i64; pass it through
                // verbatim so the round-trip is exact. ms arrives
                // as BigInt; setTimeout wants a Number.
                setTimeout(
                    () => zynmlBindings._zyntax_resolve_future(handle, 0n),
                    Number(ms),
                );
                // All wasm imports in our pipeline are typed
                // `(N i64) → i64` for ABI uniformity (see
                // `wasm_backend.rs::prepare_type_indices`). Void-
                // returning host bridges still need to push an i64
                // onto the wasm stack — return 0n so the wasm side
                // sees a well-typed value and either Drops it or
                // stores it into an unused local.
                return 0n;
            };
        }

        // Host-routed indirect calls. `arity` here is `handle + N args`,
        // so the dispatcher takes `handle + (arity-1)` args. CreateClosure
        // produces the handle as a folded HirId hash; the host's
        // `_zyntax_call_host_indirect_<arity-1>` resolves to the
        // function name and re-enters call_function.
        if (baseName === "indirect_call") {
            // arity = 1 + n_args. Min arity 1 (handle only).
            switch (arity) {
                case 1:
                    return (h) => exports._zyntax_call_host_indirect_0(h);
                case 2:
                    return (h, a0) => exports._zyntax_call_host_indirect_1(h, a0);
                case 3:
                    return (h, a0, a1) =>
                        exports._zyntax_call_host_indirect_2(h, a0, a1);
                case 4:
                    return (h, a0, a1, a2) =>
                        exports._zyntax_call_host_indirect_3(h, a0, a1, a2);
                case 5:
                    return (h, a0, a1, a2, a3) =>
                        exports._zyntax_call_host_indirect_4(h, a0, a1, a2, a3);
                default:
                    return () => {
                        throw new Error(
                            `_zyntax_jit: host.indirect_call arity ${arity} not supported (max 5)`,
                        );
                    };
            }
        }
        return () => {
            throw new Error(
                `_zyntax_jit: host "${baseName}" arity ${arity} not supported`,
            );
        };
    }

    /** Zero-arg / i64-return dispatch shim. JIT'd `entry` exports
     *  return a BigInt under wasm-bindgen's i64 ABI; we coerce to
     *  Number ONLY where it fits, otherwise return the BigInt
     *  through wasm-bindgen's existing i64 marshalling. */
    globalThis._zyntax_jit_call_0_i64 = function _zyntax_jit_call_0_i64(handle) {
        return jitFuncs[handle]();
    };
}


// In Node mode the bindings come from the require()d module; in
// browser mode we capture them inside `initZynml`. Either way,
// `run()` and `version()` route through this binding.
let zynmlBindings = null;

async function bindings() {
    if (zynmlBindings) return zynmlBindings;
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
        // Node-target wasm-pack output auto-instantiates on
        // import. Wasm-bindgen resolves all extern-fn imports at
        // INSTANTIATE time, so the JIT-host shims have to be on
        // `globalThis` BEFORE the `import` line below — otherwise
        // the wasm module trips trying to bind
        // `wbg._zyntax_jit_install` and aborts. Browser path
        // handles this inside `initZynml`.
        installJitHost();
        zynmlBindings = await import("../pkg-node/zyntax_wasm.js");
        return zynmlBindings;
    }
    throw new Error("zynml: call initZynml() before run()/version() in browser");
}

/**
 * Parse a ZynML source string and execute its `main()` function.
 * Returns a `RunResult` object exposing `output`, `ok`, `errorKind`.
 *
 * @param {string} source - ZynML source code
 */
export async function run(source) {
    const b = await bindings();
    return b.run(source);
}

// ---------------------------------------------------------------------------
// Cooperative-async resolver registry (Phase H)
// ---------------------------------------------------------------------------
//
// `call_async` returns a Promise that resolves when the runtime
// hands back a result through `_zyntax_complete_task(taskId,
// value, ok)`. For sync programs the callback fires inline during
// `_zyntax_run_async`'s call; for programs that park on host
// bridges (Phase I+) it fires later when the SM eventually
// reaches Ready.
//
// `taskResolvers` lives on `globalThis` so the wasm-imported
// `_zyntax_complete_task` (which wasm-bindgen resolves through
// `js_namespace = globalThis`) can find it regardless of module
// scope. Worker mode installs its own copy in the worker's
// `globalThis`.

const taskResolvers = new Map();
let nextTaskId = 1;

// BC-interpreter path uses a JS hook (not a wasm import) for the
// host-bridge call, because zyntax_embed (where the symbol table
// lives) doesn't depend on wasm-bindgen. The wasm-side wrapper
// (`__zw_async_set_timeout_via_js` in zyntax_wasm/src/lib.rs)
// invokes this hook through wasm-bindgen's globalThis lookup.
//
// The JIT path imports `host.async_set_timeout@2` directly and
// hits `makeHostDispatcher` instead; that route doesn't use this
// hook. Both paths converge on `_zyntax_resolve_future` for the
// SM advancement.
function installHostBridgeHooks() {
    if (globalThis._zyntax_call_host_async_set_timeout) return;
    globalThis._zyntax_call_host_async_set_timeout = (handle, ms) => {
        const ms_n = typeof ms === "bigint" ? Number(ms) : ms;
        setTimeout(() => {
            if (zynmlBindings) {
                zynmlBindings._zyntax_resolve_future(handle, 0n);
            }
        }, ms_n);
    };
}

function installCompleteTaskHook() {
    installHostBridgeHooks();
    if (globalThis._zyntax_complete_task) return;
    globalThis._zyntax_complete_task = (taskId, value, ok) => {
        // Bigint inputs from wasm-bindgen i64 marshalling.
        const id = typeof taskId === "bigint" ? Number(taskId) : taskId;
        const r = taskResolvers.get(id);
        if (!r) return;
        taskResolvers.delete(id);
        // For Phase H we marshal the value through `RunResult.output`
        // (the synchronous return value the wasm-side `run_impl`
        // already formatted). The bare `value` (an i64) is the
        // resume-slot bridge value used by Phase I+ parking; for
        // sync completions it's the parsed numeric form of the
        // same output and either field works.
        r.resolve({
            output: r.output != null ? r.output : String(value),
            ok: ok === 1 || ok === 1n,
            errorKind: 0,
        });
    };
}

/**
 * Promise-returning entry that routes through `_zyntax_run_async`.
 * For sync programs, the Promise resolves on the next microtask
 * after `_zyntax_complete_task` fires inline from wasm. For
 * programs that park on host bridges (sleep / fetch / WebSocket
 * — Phase I+), the Promise stays pending until the parked SM
 * reaches Ready and host_futures' completion callback drives
 * `_zyntax_complete_task` asynchronously.
 *
 * The shape is forward-compatible: today's call site continues to
 * work when Phase I lands actual yielding behavior without
 * changing this signature.
 */
export async function call_async(source) {
    const b = await bindings();
    installCompleteTaskHook();
    return new Promise((resolve, reject) => {
        const taskId = nextTaskId++;
        // Pre-register the resolver before kicking off the wasm
        // call. The wasm-side `_zyntax_run_async` may fire
        // `_zyntax_complete_task` synchronously, in which case
        // the resolver entry must already be in place.
        taskResolvers.set(taskId, { resolve, reject, output: null });
        let initial;
        try {
            initial = b._zyntax_run_async(source, BigInt(taskId));
        } catch (e) {
            taskResolvers.delete(taskId);
            reject(e);
            return;
        }
        // Compile / runtime errors don't fire complete_task. Stash
        // the RunResult's `output` so the synchronous-success path
        // below also has it available, then resolve directly with
        // the failure payload.
        if (!initial.ok) {
            taskResolvers.delete(taskId);
            resolve({
                output: initial.output,
                ok: false,
                errorKind: initial.errorKind,
            });
            return;
        }
        // Synchronous-success path: wasm already fired
        // _zyntax_complete_task, which removed the resolver entry
        // and resolved the Promise. If the entry's still around,
        // wasm didn't fire complete_task — that means the task
        // parked (Phase I+). Either way nothing else to do here.
        const stillPending = taskResolvers.get(taskId);
        if (stillPending) {
            // Phase I+: task is parked, attach the wasm-formatted
            // output to the resolver entry so complete_task's
            // String(value) fallback isn't used when we know the
            // semantic output (this matters once parking arrives).
            stillPending.output = initial.output;
        }
    });
}

/**
 * Build identifier of the loaded wasm module.
 */
export async function version() {
    const b = await bindings();
    return b.version();
}

/** Mirror of the Rust-side `ErrorKind` enum. */
export const ErrorKind = Object.freeze({
    None: 0,
    CompileError: 1,
    RuntimeError: 2,
});

// ---------------------------------------------------------------------------
// Worker-mode entry (mirrors wren_lift's `createWlift`)
// ---------------------------------------------------------------------------
//
// The default ESM API above (`initZynml` + `run` + `call_async`) runs
// the interpreter inline on whatever thread imported the module — the
// UI thread in a browser, or the Node main thread under `node`. That
// works but blocks the page on long-running programs.
//
// `createZyntax({ mode: "worker" })` (browser only) spawns a Web
// Worker hosting `worker.js`, which loads its own copy of the wasm
// module and runs the interpreter off the UI thread. The returned
// handle has the same `.run(source)` / `.call_async(source)` /
// `.version()` surface, but each call is async-by-construction and
// the UI thread stays responsive regardless of how long the program
// takes to finish.
//
// `mode: "main"` is the inline path — same as calling `initZynml` +
// `run` directly. Exists so embedders can pick at construction time
// and keep the call site identical across modes.

let __workerNextId = 1;

class WorkerZyntax {
    constructor({ workerUrl } = {}) {
        // `import.meta.url` resolves the worker beside this file so
        // the consumer doesn't have to thread URLs through itself.
        // Override with `workerUrl` for bundlers that rewrite paths.
        const url = workerUrl ?? new URL("./worker.js", import.meta.url);
        this.worker = new Worker(url, { type: "module" });
        this.pending = new Map();
        this.ready = new Promise((resolve) => {
            const onReady = (ev) => {
                if (ev.data?.cmd === "ready") {
                    this.worker.removeEventListener("message", onReady);
                    this.version_ = ev.data.version;
                    resolve(this);
                }
            };
            this.worker.addEventListener("message", onReady);
        });
        this.worker.addEventListener("message", (ev) => {
            const m = ev.data;
            if (!m || typeof m.cmd !== "string") return;
            if (m.cmd === "ready") return; // handled above
            const entry = this.pending.get(m.id);
            if (!entry) return;
            this.pending.delete(m.id);
            if (m.cmd === "error") {
                entry.reject(new Error(m.message));
                return;
            }
            entry.resolve(m);
        });
    }

    _send(cmd, payload) {
        const id = __workerNextId++;
        const p = new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
        });
        this.worker.postMessage({ cmd, id, ...payload });
        return p;
    }

    /**
     * Run a ZynML source in the worker. Returns a Promise of a
     * `RunResult`-shaped object: `{ output, ok, errorKind }`.
     */
    async run(source) {
        const r = await this._send("run", { source });
        return { output: r.output, ok: r.ok, errorKind: r.errorKind };
    }

    /** Same surface as `run`; exists so worker-mode + main-mode have
     *  identical APIs. The worker is already async by construction. */
    async call_async(source) {
        return this.run(source);
    }

    /** Worker-side wasm build identifier. */
    async version() {
        if (this.version_) return this.version_;
        const r = await this._send("version", {});
        return r.version;
    }

    /** Mirror of `isolated()` above. Worker contexts inherit the
     *  page's isolation state, so this still answers the page-level
     *  question. */
    isolated() {
        return isolated();
    }

    /** Tear down the Worker. Pending operations reject with the
     *  given reason (or a default). After `terminate`, the handle
     *  is unusable. */
    terminate(reason = "terminated") {
        for (const { reject } of this.pending.values()) {
            reject(new Error(reason));
        }
        this.pending.clear();
        this.worker.terminate();
    }
}

class MainZyntax {
    constructor() {
        this.version_ = null;
    }

    async init(opts) {
        await initZynml(opts);
        this.version_ = (await bindings()).version();
        return this;
    }

    async run(source) {
        const b = await bindings();
        return b.run(source);
    }

    async call_async(source) {
        return call_async(source);
    }

    async version() {
        return this.version_ ?? (await bindings()).version();
    }

    isolated() {
        return isolated();
    }

    terminate() {
        // No-op for main mode — the wasm module stays loaded for
        // the lifetime of the page. Provided so embedders can call
        // `terminate()` without branching on mode.
    }
}

/**
 * Create a Zyntax handle backed by either a Web Worker (default) or
 * the calling thread. Mirrors `wren_lift`'s `createWlift({ mode })`.
 *
 * @param {Object} [opts]
 * @param {"worker"|"main"} [opts.mode="worker"]
 *        `"worker"` (browser only) spawns a Worker hosting
 *        `worker.js`. `"main"` runs inline on the calling thread.
 *        Node has no Workers in the browser sense, so `mode` is
 *        forced to `"main"` there.
 * @param {URL|string} [opts.workerUrl]
 *        Override the Worker script URL. Defaults to `./worker.js`
 *        resolved relative to this file.
 * @param {URL|RequestInfo} [opts.module]
 *        Passed through to `initZynml` in main mode; ignored in
 *        worker mode (the worker resolves its own wasm URL).
 * @returns {Promise<WorkerZyntax|MainZyntax>}
 */
export async function createZyntax(opts = {}) {
    const isNode =
        typeof process !== "undefined" &&
        process.versions &&
        process.versions.node;
    const mode = isNode ? "main" : opts.mode ?? "worker";
    if (mode === "worker") {
        const z = new WorkerZyntax({ workerUrl: opts.workerUrl });
        await z.ready;
        return z;
    }
    if (mode === "main") {
        return await new MainZyntax().init({ module: opts.module });
    }
    throw new Error(
        `createZyntax: unknown mode '${mode}', expected 'worker' or 'main'`,
    );
}

/**
 * Whether the host page is cross-origin isolated. Returns `false` in
 * Node, `globalThis.crossOriginIsolated` in browsers. Cross-origin
 * isolation is what lets a page use `SharedArrayBuffer`, `Atomics`,
 * and the threading-flavoured wasm features that a future Phase F
 * worker-pool runtime would need; it requires the page to ship
 * with both:
 *
 *     Cross-Origin-Opener-Policy:   same-origin
 *     Cross-Origin-Embedder-Policy: require-corp
 *
 * Without isolation, `run()` still works on a single thread — the
 * algebraic-effects runtime falls back to direct synchronous polling
 * — but features that need atomics will throw at the call site
 * instead of silently corrupting. See `docs/wasm-deployment.md` for
 * hosting recipes.
 */
export function isolated() {
    if (typeof globalThis === "undefined") return false;
    return globalThis.crossOriginIsolated === true;
}
