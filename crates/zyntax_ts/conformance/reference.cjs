// The reference compiler is a test dependency; native execution never loads it.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");

const [action, compiler, requestFile] = process.argv.slice(2);
const ts = require(path.resolve(compiler));
if (ts.version !== "6.0.2") throw new Error(`Expected TypeScript 6.0.2, found ${ts.version}`);
const request = JSON.parse(fs.readFileSync(requestFile, "utf8"));
const converted = ts.convertCompilerOptionsFromJson({
  target: "es2022", module: "commonjs", strict: true, noEmit: true,
  skipLibCheck: true, types: [], lib: ["es2022", "dom"],
  ...request.options,
}, path.dirname(request.files[0]));
if (converted.errors.length) throw new Error(ts.formatDiagnosticsWithColorAndContext(converted.errors, {
  getCurrentDirectory: () => process.cwd(), getCanonicalFileName: x => x, getNewLine: () => "\n"
}));
if (action === "check") {
  const program = ts.createProgram(request.files, converted.options);
  const syntax = program.getSyntacticDiagnostics();
  const diagnostics = [...syntax, ...program.getSemanticDiagnostics(), ...program.getOptionsDiagnostics()];
  process.stdout.write(JSON.stringify({
    compiler: ts.version,
    accepted: diagnostics.length === 0,
    kind: syntax.length ? "Syntax" : diagnostics.length ? "Type" : "Accept",
    diagnostics: diagnostics.map(d => ({code: d.code, start: d.start, length: d.length,
      file: d.file && path.relative(path.dirname(request.files[0]), d.file.fileName),
      message: ts.flattenDiagnosticMessageText(d.messageText, "\n")})),
  }));
} else if (action === "run") {
  Module._extensions[".ts"] = (module, filename) => {
    const source = fs.readFileSync(filename, "utf8");
    const output = ts.transpileModule(source, {compilerOptions: {...converted.options, noEmit: false}});
    module._compile(output.outputText, filename);
  };
  require(path.resolve(request.files[0]));
} else {
  throw new Error(`Unknown reference action ${action}`);
}
