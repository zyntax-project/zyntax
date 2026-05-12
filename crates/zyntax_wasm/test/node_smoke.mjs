// Node.js smoke test for the zyntax_wasm shim.
//
// Loads the wasm-pack `--target nodejs` output (`pkg-node/`) and
// exercises the same `run()` API the browser will. This is the
// real end-to-end check that:
//
//   1. wasm-bindgen's bindings work when invoked from JS.
//   2. The whole ZynML pipeline (parse → lower → interpret) executes
//      inside the wasm module.
//   3. Surface types (RunResult getters, ErrorKind enum) round-trip
//      between JS and Rust correctly.
//
// Run:
//
//   ./build.sh node
//   node test/node_smoke.mjs
//
// Or via the package.json shortcut (no extra deps needed):
//
//   npm test

import { run, version, ErrorKind } from "../web/zynml.mjs";

let failed = 0;
function check(label, cond, detail) {
    const status = cond ? "ok" : "FAIL";
    console.log(`  ${status} - ${label}${detail ? `  ${detail}` : ""}`);
    if (!cond) failed++;
}

async function main() {
    console.log("zyntax_wasm Node.js smoke test");
    console.log("==============================");

    const v = await version();
    check("version() returns a string", typeof v === "string");
    check("version() identifies zyntax_wasm", v.includes("zyntax_wasm"), `(${v})`);

    {
        const r = await run("def main(): i64 { return 42 }");
        check("trivial main() ok", r.ok === true, `output=${r.output}`);
        check("trivial main() output is '42'", r.output === "42", `got '${r.output}'`);
        check(
            "trivial main() errorKind is None",
            r.errorKind === ErrorKind.None,
            `(got ${r.errorKind})`,
        );
    }

    {
        const r = await run("def main(): i64 { return 6 * 7 }");
        check("arithmetic main() ok", r.ok === true);
        check("arithmetic main() output is '42'", r.output === "42", `got '${r.output}'`);
    }

    {
        const r = await run("def main(): i64 { return 7"); // missing brace
        check("parse error is not ok", r.ok === false);
        check(
            "parse error classifies as CompileError",
            r.errorKind === ErrorKind.CompileError,
            `(got ${r.errorKind})`,
        );
    }

    // Phase E.8: hot-function tier-up smoke test. The interpreter's
    // default `wasm_jit_threshold` is 1, so the SECOND call to a
    // function should hit the wasm-encoder JIT. We can't observe
    // the tier transition from the host today (no diagnostic
    // counter export yet), but we CAN verify two things:
    //   1. A program that calls `main()` once still works (already
    //      covered by the cases above).
    //   2. A program that calls a helper function twice from main
    //      — exercising the second-call JIT path — still returns
    //      the correct value. If the JIT install failed midway or
    //      the dispatch hook returned a wrong value, this would
    //      regress against the interpreter answer.
    {
        // Two calls to the same helper from `main` — the FIRST call
        // installs the wasm-JIT handle (threshold = 1), the SECOND
        // call routes through the JS dispatch shim. If JIT install
        // or dispatch were broken we'd either crash here or get the
        // wrong value back.
        const src =
            "def trivial(): i64 { return 99 }\n" +
            "def main(): i64 { return trivial() + trivial() }\n";
        const r = await run(src);
        check("hot-function via helper call ok", r.ok === true, `output=${r.output}`);
        check(
            "hot-function tier-up preserves return value",
            r.output === "198",
            `got '${r.output}'`,
        );
    }

    console.log("");
    if (failed === 0) {
        console.log("All checks passed.");
        process.exit(0);
    } else {
        console.error(`${failed} check(s) failed.`);
        process.exit(1);
    }
}

main().catch((e) => {
    console.error("uncaught error:", e);
    process.exit(2);
});
