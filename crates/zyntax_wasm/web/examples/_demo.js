// Shared driver for ZynML browser demos.
//
// Each demo HTML page calls `runDemo({ source, expected })` after the
// DOM is ready. We load the wasm, wire the "Run" button, and surface
// the elapsed wall-clock (proof the cooperative-async parking really
// yielded back to the event loop instead of busy-waiting).
//
// Browser-only — the import path resolves the wasm-bindgen `--target
// web` bundle at `../pkg-web/zyntax_wasm.js` via `zynml.mjs`. Build
// it first: `cd crates/zyntax_wasm && ./build.sh web`.

import { initZynml, call_async, ErrorKind, version } from "../zynml.mjs";

const VERSION_EL_ID = "zynml-version";
const STATUS_EL_ID = "demo-status";
const RUN_BTN_ID = "demo-run";
const RESULT_EL_ID = "demo-result";

export async function runDemo({ source, expected }) {
    const statusEl = document.getElementById(STATUS_EL_ID);
    const runBtn = document.getElementById(RUN_BTN_ID);
    const resultEl = document.getElementById(RESULT_EL_ID);
    const versionEl = document.getElementById(VERSION_EL_ID);

    statusEl.textContent = "loading wasm…";
    runBtn.disabled = true;

    try {
        await initZynml({
            // `zynml.mjs` resolves `../pkg-web/zyntax_wasm.js`. The
            // wasm-bindgen `default()` call then fetches the sibling
            // `.wasm` file automatically; passing no explicit URL
            // lets it use the JS-glue's relative path.
        });
    } catch (err) {
        statusEl.textContent = `failed to load wasm: ${err.message}`;
        renderResult(resultEl, {
            ok: false,
            output: String(err),
            errorKind: "LoadFailed",
            elapsed: 0,
        });
        return;
    }

    if (versionEl) {
        try {
            versionEl.textContent = await version();
        } catch {
            /* non-fatal */
        }
    }

    statusEl.textContent = "ready — click Run to compile + execute";
    runBtn.disabled = false;

    // Headless verification escape hatch: visiting any demo with
    // `?autorun=1` clicks Run as soon as the wasm is ready, so
    // automated tooling (e.g. Chrome `--headless --dump-dom`) can
    // confirm the full async cycle without scripting a click via CDP.
    const autorun =
        typeof window !== "undefined" &&
        new URLSearchParams(window.location.search).has("autorun");

    runBtn.addEventListener("click", async () => {
        runBtn.disabled = true;
        statusEl.textContent = "running…";
        resultEl.innerHTML = "";

        const t0 = performance.now();
        let runResult;
        try {
            runResult = await call_async(source);
        } catch (err) {
            statusEl.textContent = "errored";
            renderResult(resultEl, {
                ok: false,
                output: String(err),
                errorKind: "RuntimeError",
                elapsed: performance.now() - t0,
            });
            runBtn.disabled = false;
            return;
        }
        const elapsed = performance.now() - t0;

        statusEl.textContent = runResult.ok ? "done" : "errored";
        renderResult(resultEl, {
            ok: runResult.ok,
            output: runResult.output,
            errorKind: errorKindName(runResult.errorKind),
            elapsed,
            expected,
        });
        runBtn.disabled = false;
    });

    if (autorun) runBtn.click();
}

function renderResult(el, { ok, output, errorKind, elapsed, expected }) {
    const lines = [];
    lines.push(line("status", ok ? "ok" : "error", ok ? "ok" : "bad"));
    lines.push(line("output", output ?? "(empty)"));
    if (errorKind && errorKind !== "None") {
        lines.push(line("errorKind", errorKind, "bad"));
    }
    lines.push(line("elapsed", `${elapsed.toFixed(1)} ms`, "timing"));
    if (expected) {
        const match = String(output).trim() === String(expected.output).trim();
        lines.push(
            line(
                "matches expected",
                match ? "yes" : `no (expected ${expected.output})`,
                match ? "ok" : "bad",
            ),
        );
        if (expected.minElapsedMs != null) {
            const passed = elapsed >= expected.minElapsedMs;
            lines.push(
                line(
                    `elapsed ≥ ${expected.minElapsedMs} ms`,
                    passed ? "yes" : "no",
                    passed ? "ok" : "bad",
                ),
            );
        }
    }
    el.innerHTML = lines.join("");
}

function line(label, value, cls = "") {
    return `<div class="result-line">
        <span class="result-label">${escapeHtml(label)}</span>
        <span class="result-value ${cls}">${escapeHtml(value)}</span>
    </div>`;
}

function errorKindName(k) {
    for (const [name, v] of Object.entries(ErrorKind)) {
        if (v === k) return name;
    }
    return String(k);
}

function escapeHtml(s) {
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}
