// The reference compiler is a test dependency; native execution never loads it.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const {pathToFileURL} = require("node:url");

const expectedVersion = require("./package.json").devDependencies.typescript;

async function main() {
  const [action, compiler, requestFile] = process.argv.slice(2);
  if (action !== "check" && action !== "run") throw new Error(`Unknown reference action ${action}`);
  const root = path.resolve(compiler);
  const {version} = require(path.join(root, "package.json"));
  if (version !== expectedVersion) throw new Error(`Expected TypeScript ${expectedVersion}, found ${version}`);
  const load = Module.createRequire(path.join(root, "package.json"));
  const {API, ModuleKind, formatDiagnosticsWithColorAndContext} = await import(pathToFileURL(load.resolve("typescript/sync")).href);
  const request = JSON.parse(fs.readFileSync(requestFile, "utf8"));
  const base = path.dirname(request.files[0]);
  const api = new API({cwd: base});
  try {
    const parsed = api.parseJsonConfigFileContent({
      files: request.files,
      compilerOptions: {
        target: "es2022", module: "commonjs", strict: true, noEmit: true,
        skipLibCheck: true, types: [], lib: ["es2022", "dom"],
        ...request.options,
        ...(action === "run" ? {noEmit: false, declaration: false,
          emitDeclarationOnly: false, rewriteRelativeImportExtensions: true} : {}),
      },
    }, {configDirectory: base});
    if (parsed.errors.length) throw new Error(formatDiagnosticsWithColorAndContext(parsed.errors, api));
    const program = api.createProgram(parsed.fileNames, parsed.options);
    try {
      if (action === "check") {
        const syntax = program.getSyntacticDiagnostics();
        const diagnostics = [...syntax, ...program.getSemanticDiagnostics(),
          ...program.getProgramDiagnostics(), ...program.getGlobalDiagnostics()];
        const message = (d, depth = 0) => ["  ".repeat(depth) + d.text,
          ...(d.messageChain || []).map(child => message(child, depth + 1))].join("\n");
        process.stdout.write(JSON.stringify({
          compiler: version,
          accepted: diagnostics.length === 0,
          kind: syntax.length ? "Syntax" : diagnostics.length ? "Type" : "Accept",
          diagnostics: diagnostics.map(d => ({code: d.code,
            start: d.fileName ? d.pos : undefined,
            length: d.fileName ? d.end - d.pos : undefined,
            file: d.fileName && path.relative(base, d.fileName), message: message(d)})),
        }));
        return;
      }

      const output = program.emitToString();
      if (output.emitSkipped || output.diagnostics.length) {
        throw new Error("reference emission failed: " + formatDiagnosticsWithColorAndContext(output.diagnostics, api));
      }
      const emitted = new Map();
      for (const [filename, file] of output.outputFiles) {
        if (filename.endsWith(".js")) {
          if (!file.sourceFileName) throw new Error(`reference output lacks its source: ${filename}`);
          emitted.set(fs.realpathSync(file.sourceFileName), file.text);
        }
      }
      const resolver = api.createModuleResolver(parsed.options);
      const resolve = Module._resolveFilename;
      const extension = Module._extensions[".ts"];
      try {
        Module._resolveFilename = function(specifier, parent, ...rest) {
          if (parent?.filename.endsWith(".ts") && /^\.\.?\//.test(specifier)) {
            const {resolvedModule} = resolver.resolveModuleName(specifier, path.dirname(parent.filename), ModuleKind.CommonJS);
            if (resolvedModule && !resolvedModule.resolvedFileName.endsWith(".d.ts")) {
              return fs.realpathSync(resolvedModule.resolvedFileName);
            }
          }
          return resolve.call(this, specifier, parent, ...rest);
        };
        Module._extensions[".ts"] = (module, filename) => {
          const text = emitted.get(fs.realpathSync(filename));
          if (text === undefined) throw new Error(`reference did not emit ${filename}`);
          module._compile(text, filename);
        };
        require(path.resolve(request.files[0]));
      } finally {
        Module._resolveFilename = resolve;
        if (extension === undefined) delete Module._extensions[".ts"];
        else Module._extensions[".ts"] = extension;
        resolver.dispose();
      }
    } finally {
      program.dispose();
    }
  } finally {
    api.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
