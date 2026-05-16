// Web Worker host for the zyntax_wasm interpreter.
//
// The main page used to import zyntax_wasm directly and call
// `run(source)` on the UI thread. That works for short scripts
// but freezes the page on:
//
//   * heavy compute — wasm can't yield to the JS event loop while
//     we're inside `run()`,
//   * algebraic-effects handlers that loop many times without
//     hitting a real async boundary,
//   * any program whose `main()` takes more than a frame budget.
//
// Moving the interpreter into a Worker means the UI stays
// responsive regardless. The protocol is intentionally tiny:
//
//   main → worker:  { cmd: "run",     id, source }
//   main → worker:  { cmd: "version", id }
//   worker → main:  { cmd: "ready",  version }
//   worker → main:  { cmd: "result", id, output, ok, errorKind }
//   worker → main:  { cmd: "version-result", id, version }
//   worker → main:  { cmd: "error",  id, message }
//
// Mirrors `wren_lift/wasm/web/worker.js` — the wlift project has
// shaken out the bridge ergonomics in production, this is the
// minimal subset zyntax_wasm needs today. Bridges that touch the
// DOM (querySelector, OffscreenCanvas, localStorage) will be added
// here as ZRTL plugins that depend on them land; the pattern is to
// `postMessage` the op to the page, run it on `window`, and
// `postMessage` the result back through a future-handle map.

// Worker resolves the wasm-pack output the same way `zynml.mjs`
// does for main mode: `../pkg-web/zyntax_wasm.js` next to this
// file. Bundlers that rewrite paths can override by replacing
// this file with their own worker entry that imports from the
// rewritten location.
import * as zyntax_wasm from "../pkg-web/zyntax_wasm.js";

const { default: init, version, run } = zyntax_wasm;

// `init()` runs once at startup. After it resolves, the BC
// interpreter is ready and we can drain queued messages.
await init();

self.postMessage({ cmd: "ready", version: version() });

self.addEventListener("message", (ev) => {
    const m = ev.data;
    if (!m || typeof m.cmd !== "string") return;

    switch (m.cmd) {
        case "run": {
            try {
                const r = run(m.source);
                // r is a `RunResult` wasm-bindgen wrapper; pull
                // the fields out before postMessage (wasm pointers
                // can't cross the worker boundary).
                self.postMessage({
                    cmd: "result",
                    id: m.id,
                    output: r.output,
                    ok: r.ok,
                    errorKind: r.errorKind,
                });
            } catch (e) {
                self.postMessage({
                    cmd: "error",
                    id: m.id,
                    message: String(e?.message ?? e),
                });
            }
            return;
        }
        case "version": {
            try {
                self.postMessage({
                    cmd: "version-result",
                    id: m.id,
                    version: version(),
                });
            } catch (e) {
                self.postMessage({
                    cmd: "error",
                    id: m.id,
                    message: String(e?.message ?? e),
                });
            }
            return;
        }
        default:
            self.postMessage({
                cmd: "error",
                id: m.id,
                message: `unknown cmd: ${m.cmd}`,
            });
    }
});
