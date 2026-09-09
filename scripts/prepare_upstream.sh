#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_DIR=${LINGBOT_UPSTREAM_DIR:-"$ROOT_DIR/.upstream/lingbot-world-v2"}
UPSTREAM_URL=https://github.com/robbyant/lingbot-world-v2.git
UPSTREAM_SHA=45fa40673607c9acba6cf96a1f9396c95bcef25f

mkdir -p "$(dirname "$UPSTREAM_DIR")"
if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    git clone "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi

actual=$(git -C "$UPSTREAM_DIR" rev-parse HEAD)
[[ "$actual" == "$UPSTREAM_SHA" ]] || {
    echo "error: $UPSTREAM_DIR is at $actual; expected $UPSTREAM_SHA" >&2
    exit 1
}

apply_once() {
    local patch_file=$1
    if patch --dry-run --forward --batch -d "$UPSTREAM_DIR" -p1 < "$patch_file" >/dev/null 2>&1; then
        patch --forward --batch -d "$UPSTREAM_DIR" -p1 < "$patch_file" >/dev/null
    elif patch --dry-run --reverse --batch -d "$UPSTREAM_DIR" -p1 < "$patch_file" >/dev/null 2>&1; then
        :
    else
        echo "error: cannot apply or identify already-applied patch $patch_file" >&2
        exit 1
    fi
}

apply_once "$ROOT_DIR/patches/0001-strix-halo-sdpa-cross-attention.patch"
apply_once "$ROOT_DIR/patches/0002-experiment-metrics.patch"

echo "upstream: $(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
echo "working tree patches:"
git -C "$UPSTREAM_DIR" diff --stat
