#!/usr/bin/env bash
# Dev-mode static server for the Phase F browser example.
#
# Serves `examples/web/` (and the sibling `crates/zyntax_wasm/web/`
# + `crates/zyntax_wasm/pkg-web/`) with the COOP+COEP headers required
# for cross-origin-isolated execution. Without these headers the
# page still runs (single-threaded), but `Zyntax.isolated()` returns
# `false` and any future SharedArrayBuffer-based code would throw.
#
# Required headers:
#   Cross-Origin-Opener-Policy:   same-origin
#   Cross-Origin-Embedder-Policy: require-corp
#
# Prerequisites:
#   * Built wasm: `cd crates/zyntax_wasm && ./build.sh web`
#   * `miniserve` (https://github.com/svenstaro/miniserve):
#       `cargo install miniserve`
#     If you'd rather use the Python stdlib server, a script that
#     emits the headers via a custom handler also works — sketched
#     in docs/wasm-deployment.md.
#
# Usage:
#   ./examples/web/serve.sh                 # serves on 0.0.0.0:8080
#   ./examples/web/serve.sh 0.0.0.0 8000    # custom host/port
#
# Browse to: http://localhost:8080/examples/web/

set -euo pipefail

HOST="${1:-0.0.0.0}"
PORT="${2:-8080}"

if ! command -v miniserve >/dev/null 2>&1; then
    cat >&2 <<EOF
serve.sh: 'miniserve' not found.

Install with:
    cargo install miniserve

Or see docs/wasm-deployment.md for a Python-based fallback.
EOF
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Quick smoke check — fail fast if the wasm bundle hasn't been built.
if [[ ! -f "$REPO_ROOT/crates/zyntax_wasm/pkg-web/zyntax_wasm_bg.wasm" ]]; then
    cat >&2 <<EOF
serve.sh: pkg-web/zyntax_wasm_bg.wasm missing.

Build the wasm bundle first:
    cd $REPO_ROOT/crates/zyntax_wasm
    ./build.sh web

Then re-run ./examples/web/serve.sh from the repo root.
EOF
    exit 1
fi

echo "Serving $REPO_ROOT on http://$HOST:$PORT/"
echo "Open: http://$HOST:$PORT/examples/web/"
echo

# `--header` flag is honored by miniserve >= 0.20.
exec miniserve \
    "$REPO_ROOT" \
    --interfaces "$HOST" \
    --port "$PORT" \
    --header "Cross-Origin-Opener-Policy: same-origin" \
    --header "Cross-Origin-Embedder-Policy: require-corp"
