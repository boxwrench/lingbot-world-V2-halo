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
    elif patch_is_present "$patch_file"; then
        # A later experiment patch may add lines inside an earlier patch's
        # context, making reverse-dry-run too strict for an already-stacked
        # checkout. Require distinctive markers before accepting that state.
        :
    else
        echo "error: cannot apply or identify already-applied patch $patch_file" >&2
        exit 1
    fi
}

search_q() {
    local pattern=$1
    local file=$2
    if command -v rg >/dev/null 2>&1; then
        rg -q "$pattern" "$file"
    else
        # Keep reproduction usable on minimal hosts that do not have ripgrep.
        # These markers are extended-regular-expression compatible.
        grep -Eq "$pattern" "$file"
    fi
}

patch_is_present() {
    case "$(basename "$1")" in
        0001-strix-halo-sdpa-cross-attention.patch)
            search_q 'from \.attention import attention' "$UPSTREAM_DIR/wan/modules/model_fast.py" &&
                search_q 'x = attention\(q, k, v, k_lens=context_lens\)' "$UPSTREAM_DIR/wan/modules/model_fast.py"
            ;;
        0002-experiment-metrics.patch)
            search_q 'self\.metrics = metrics' "$UPSTREAM_DIR/wan/image2video.py" &&
                search_q 'self\.metrics\["vae_decode_ms"\]' "$UPSTREAM_DIR/wan/image2video.py"
            ;;
        0003-per-forward-metrics.patch)
            search_q 'forward_t0 = time\.perf_counter\(\)' "$UPSTREAM_DIR/wan/image2video.py" &&
                search_q 'cache_t0 = time\.perf_counter\(\)' "$UPSTREAM_DIR/wan/image2video.py"
            ;;
        0004-opt-in-host-kv-cursor.patch)
            search_q 'host_cursor = "global_end_index_py" in kv_cache' "$UPSTREAM_DIR/wan/modules/model_fast.py" &&
                search_q 'kv_cache\["local_end_index_py"\] = int\(local_end_index\)' "$UPSTREAM_DIR/wan/modules/model_fast.py"
            ;;
        *)
            return 1
            ;;
    esac
}

apply_once "$ROOT_DIR/patches/0001-strix-halo-sdpa-cross-attention.patch"
apply_once "$ROOT_DIR/patches/0002-experiment-metrics.patch"
apply_once "$ROOT_DIR/patches/0003-per-forward-metrics.patch"
apply_once "$ROOT_DIR/patches/0004-opt-in-host-kv-cursor.patch"

echo "upstream: $(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
echo "working tree patches:"
git -C "$UPSTREAM_DIR" diff --stat
