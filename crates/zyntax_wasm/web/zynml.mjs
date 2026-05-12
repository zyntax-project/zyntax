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
 * @param {Object} [opts]
 * @param {URL|RequestInfo} [opts.module]
 *        URL of the `.wasm` file. Browser-only; ignored under Node.
 *        When omitted, wasm-bindgen's default resolution (same dir as
 *        the JS glue) is used.
 * @returns {Promise<void>}
 */
export async function initZynml(opts = {}) {
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

// In Node mode the bindings come from the require()d module; in
// browser mode we capture them inside `initZynml`. Either way,
// `run()` and `version()` route through this binding.
let zynmlBindings = null;

async function bindings() {
    if (zynmlBindings) return zynmlBindings;
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
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
