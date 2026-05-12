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
     */
    globalThis._zyntax_jit_install = function _zyntax_jit_install(bytes) {
        try {
            const mod  = new WebAssembly.Module(bytes);
            const inst = new WebAssembly.Instance(mod, {});
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
