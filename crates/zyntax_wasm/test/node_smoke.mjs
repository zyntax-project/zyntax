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
