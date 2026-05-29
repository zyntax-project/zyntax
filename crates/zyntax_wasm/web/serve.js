#!/usr/bin/env node
// Static-file server tailored for the ZynML browser demos.
//
//   * Sets the cross-origin-isolation headers Chrome wants for
//     SharedArrayBuffer / advanced wasm features (COOP, COEP, CORP).
//   * Emits the right MIME for .wasm / .mjs / .js / .html / .css.
//   * Serves the *crate root* so `web/` and `pkg-web/` are reachable
//     as siblings — the same layout the demo HTML expects.
//
// Usage:
//   cd crates/zyntax_wasm && node web/serve.js [port]
//
// No npm deps. Built-in `http` + `fs` only.
//
// ES-module style (the crate's package.json sets `"type": "module"`).

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = parseInt(process.argv[2] || process.env.PORT || "8000", 10);
// Crate root — one level above this file. `pkg-web/` and `web/` are
// both children of this path.
const ROOT = path.resolve(__dirname, "..");

const MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".map": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
    ".png": "image/png",
};

function sendError(res, code, msg) {
    res.statusCode = code;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(msg + "\n");
}

const server = http.createServer((req, res) => {
    // Strip query string, normalise the URL path.
    const url = new URL(req.url, `http://${req.headers.host}`);
    let urlPath = decodeURIComponent(url.pathname);
    if (urlPath === "/") urlPath = "/web/examples/index.html";

    // Resolve against ROOT and reject directory escapes.
    const fsPath = path.normalize(path.join(ROOT, urlPath));
    if (!fsPath.startsWith(ROOT)) {
        return sendError(res, 403, "forbidden");
    }

    fs.stat(fsPath, (statErr, stat) => {
        if (statErr || !stat.isFile()) {
            return sendError(res, 404, `not found: ${urlPath}`);
        }
        const ext = path.extname(fsPath).toLowerCase();
        const mime = MIME[ext] || "application/octet-stream";

        // Cross-origin isolation headers.
        //
        //   COOP=same-origin + COEP=require-corp put the page in a
        //   "cross-origin isolated" context, which unlocks
        //   SharedArrayBuffer and the more capable wasm features.
        //   Without these Chrome silently disables SharedArrayBuffer
        //   even though wasm itself still loads.
        //
        //   CORP=cross-origin on every response tells the browser
        //   that fetching this resource from an isolated context is
        //   OK — required for the .wasm fetch from the page.
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
        res.setHeader("Content-Type", mime);
        res.setHeader("Content-Length", stat.size);
        res.setHeader("Cache-Control", "no-store");
        fs.createReadStream(fsPath).pipe(res);
    });
});

server.listen(PORT, () => {
    process.stdout.write(
        `\nZynML demo server\n` +
            `  root:  ${ROOT}\n` +
            `  port:  ${PORT}\n` +
            `  open:  http://localhost:${PORT}/web/examples/index.html\n\n` +
            `Cross-origin isolation headers are set on every response:\n` +
            `  Cross-Origin-Opener-Policy: same-origin\n` +
            `  Cross-Origin-Embedder-Policy: require-corp\n` +
            `  Cross-Origin-Resource-Policy: cross-origin\n\n`,
    );
});

// Clean stop on SIGINT so cargo / shell scripts don't leak the port.
for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => {
        server.close(() => process.exit(0));
    });
}
