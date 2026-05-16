# Deploying Zyntax to a browser

This page covers the HTTP-header requirements for hosting the
`zyntax_wasm` bundle in a browser, plus deployment recipes for the
common static-hosting setups. The required headers are
**Cross-Origin-Opener-Policy: `same-origin`** and
**Cross-Origin-Embedder-Policy: `require-corp`** — both are needed
to put the page into a *cross-origin isolated* execution context.

Zyntax's runtime works on a single thread without isolation
(`Zyntax.run` returns the program's result; the wasm-emitting JIT
still operates). Isolation matters for future surface area:

- `SharedArrayBuffer`, `Atomics.wait` / `Atomics.notify` — the
  primitives a worker-pool runtime would use.
- Threaded wasm via `wasm-bindgen-rayon`.
- High-precision timers (`performance.now()` regains microsecond
  resolution under isolation in modern browsers).

You can check whether your page is isolated at runtime via the
`isolated()` helper:

```js
import { initZynml, run, isolated } from "/path/to/zynml.mjs";
await initZynml();
if (isolated()) {
    console.log("cross-origin isolated — full async + workers available");
} else {
    console.warn("not isolated — async runs single-threaded");
}
```

## Worker mode vs. main mode

There are two ways to run the interpreter in a browser. Both work on
stable Rust with the default `zyntax_wasm` build; the choice is
purely about whether you want the interpreter to share a thread with
your UI.

**Worker mode** (default) — `createZyntax({ mode: "worker" })`
spawns a Web Worker that loads its own copy of the wasm module and
runs the BC interpreter off the UI thread. The page stays responsive
regardless of how long the program takes to finish. Returned handle:

```js
import { createZyntax } from "/path/to/zynml.mjs";
const zx = await createZyntax({ mode: "worker" });
const r  = await zx.run("def main(): i64 { return 42 }");
console.log(r.output); // "42"
```

**Main mode** — `createZyntax({ mode: "main" })` runs the
interpreter inline on the calling thread. Faster (no postMessage
hop, direct access to the wasm linear memory), but a long program
freezes the UI. Choose this for pages where memory throughput
matters more than responsiveness (e.g. an in-page tool that streams
many KB/frame between Zyntax and a canvas).

Both modes expose the same `.run` / `.call_async` / `.version` /
`.isolated` / `.terminate` surface, so the call site doesn't change
when you swap them. In Node, `mode` is forced to `"main"`.

## What COOP and COEP do, briefly

| Header | Value | Effect |
|---|---|---|
| `Cross-Origin-Opener-Policy` | `same-origin` | Detaches the page from any opener / popup window that isn't same-origin. Prevents `window.opener` and `BroadcastChannel` channels into other documents. |
| `Cross-Origin-Embedder-Policy` | `require-corp` | Every cross-origin subresource (images, scripts, fonts, …) must opt in via `Cross-Origin-Resource-Policy: cross-origin` (or `same-site`/`same-origin`). Subresources without it fail to load. |

Once both are set, `globalThis.crossOriginIsolated` becomes `true`
and `SharedArrayBuffer` becomes constructible.

## Recipe: local development

`examples/web/serve.sh` uses [`miniserve`](https://github.com/svenstaro/miniserve)
to serve the repo root with the required headers:

```bash
cargo install miniserve
cd /path/to/zyntax
crates/zyntax_wasm/build.sh web   # produce pkg-web/
./examples/web/serve.sh
# open http://localhost:8080/examples/web/
```

The script passes `--header "Cross-Origin-Opener-Policy: same-origin"`
and `--header "Cross-Origin-Embedder-Policy: require-corp"`.

### Python stdlib fallback

If you don't want to install miniserve, this Python 3 snippet
serves the current directory with the same headers:

```python
# save as serve.py, run with `python3 serve.py 8080`
import http.server, sys

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
http.server.ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
```

## Recipe: Nginx

```nginx
location /your-zyntax-app/ {
    root /var/www/zyntax;
    add_header Cross-Origin-Opener-Policy   "same-origin";
    add_header Cross-Origin-Embedder-Policy "require-corp";
    types {
        application/wasm  wasm;
    }
}
```

The `types` block ensures `.wasm` files get the right MIME type;
browsers reject `WebAssembly.instantiateStreaming` responses that
arrive with `application/octet-stream`.

## Recipe: Caddy

```caddyfile
your-zyntax-app.example.com {
    root * /var/www/zyntax
    file_server
    header Cross-Origin-Opener-Policy   "same-origin"
    header Cross-Origin-Embedder-Policy "require-corp"
}
```

Caddy auto-detects `.wasm` and serves it with `application/wasm`
out of the box.

## Recipe: Cloudflare Workers (or any edge-compute platform)

```js
export default {
    async fetch(req, env) {
        const res = await env.ASSETS.fetch(req);
        const headers = new Headers(res.headers);
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        return new Response(res.body, {
            status: res.status,
            headers,
        });
    },
};
```

Bind the static assets via `[site] bucket = "./dist"` in
`wrangler.toml` and the worker rewrites every response's headers
before they reach the browser.

## When you can't set the headers

CDN-hosted widgets that embed into pages you don't control can't
require COOP/COEP — the embedding page's headers govern. The
default `zyntax_wasm` build works fine without isolation (worker
mode and main mode both run single-threaded), so the only thing
isolation gates is the future SAB-backed worker pool.

If you do want isolation on a host you don't control,
`crates/zyntax_wasm/web/coi-serviceworker.js` (vendored MIT, from
the `coi-serviceworker` project) installs a service worker that
re-writes responses to add the COOP/COEP headers. Drop it next to
your page and include it inline:

```html
<script src="coi-serviceworker.js"></script>
```

Caveats:

- First-load misses the headers — the service worker isn't
  controlling the page yet. It reloads itself once on first install
  to engage. Users see a single auto-refresh.
- The service worker itself has to be served from the same origin
  as the page; the script can't be CDN-hosted.
- Works on GitHub Pages, Cloudflare R2 static sites, S3 + CloudFront
  without origin-side header config — anywhere you can drop a JS
  file alongside the page.

`examples/web/index.html` ships with this script included so the
demo works on hosts without server-side header control. When the
host already sends COOP/COEP, the service worker is a no-op.

## Future items not yet shipped

The current browser delivery is end-to-end functional: single-
Worker offload, optional COI service worker, vendored worker.js,
isolated() runtime check, headless-Chrome CI smoke. Still to come:

- **SAB-backed Worker pool** (`crates/zyntax_wasm/src/worker_pool.rs`)
  driving parallel poll-fn invocations across N Workers via shared
  `WebAssembly.Memory`. Requires the threaded-wasm toolchain
  (`wasm-bindgen-rayon` + nightly Rust + custom RUSTFLAGS today,
  stable when the threading proposal lands). Strict superset of
  the single-Worker mode; falls back when isolation isn't
  available.
- **Cross-Worker effect resume** — when the SAB-backed pool lands,
  algebraic-effect resume continuations can hop Workers instead of
  serialising through the main thread.

Each of these is independent and can land in its own phase. Today's
single-Worker mode covers every test we have and is what the
headless-Chrome CI exercises.
