from __future__ import annotations

import inspect
from pathlib import Path

from pure_compile_helpers import make_pure_block_forward, PureTensorIslands

ROOT = Path(__file__).resolve().parent.parent


def test_pure_block_forward_accepts_kv_write_only() -> None:
    # The model loop always passes kv_write_only (False by default); the pure
    # path must accept it or every RC1 run breaks at block dispatch.
    islands = PureTensorIslands.compile()
    forward = make_pure_block_forward(islands)
    assert "kv_write_only" in inspect.signature(forward).parameters


def test_patch_0005_wired() -> None:
    assert (ROOT / "patches/0005-kv-write-only-clean-forward.patch").is_file()
    wiring = (ROOT / "scripts/prepare_upstream.sh").read_text()
    assert "patches/0005-kv-write-only-clean-forward.patch" in wiring
    assert wiring.count("0005-kv-write-only-clean-forward.patch") >= 2  # apply + marker


def test_commit_clean_kv_sets_write_only() -> None:
    text = (ROOT / "scripts/run_interactive.py").read_text()
    call = text[text.index("def commit_clean_kv"):]
    call = call[:call.index("\n    sync_fn()")]
    assert "kv_write_only=True" in call
