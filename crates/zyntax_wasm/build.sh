#!/usr/bin/env bash
# Build the zyntax_wasm crate for both browser and Node.js targets.
# Outputs land in `pkg-web/` and `pkg-node/` next to this script.
#
# Usage:
#   ./build.sh          # build both targets
#   ./build.sh web      # browser-only
#   ./build.sh node     # nodejs-only
#
# Requires: wasm-pack (cargo install wasm-pack), Rust wasm32 target
# (rustup target add wasm32-unknown-unknown).

set -euo pipefail

cd "$(dirname "$0")"

target="${1:-both}"

build_web() {
    echo "==> wasm-pack build --target web --out-dir pkg-web"
    wasm-pack build --target web --out-dir pkg-web
}

build_node() {
    echo "==> wasm-pack build --target nodejs --out-dir pkg-node"
    wasm-pack build --target nodejs --out-dir pkg-node
}

case "$target" in
    web)  build_web ;;
    node) build_node ;;
    both) build_web; build_node ;;
    *) echo "unknown target: $target (web|node|both)"; exit 1 ;;
esac
