from __future__ import annotations

from state_cache_validation import reset_cursors_at_bootstrap, warmed_input_strict_equal


def frozen_tensor(sha: str) -> dict:
    return {"kind": "tensor", "sha256": sha}


def prepare_shaped(shas: dict) -> dict:
    # run_prepare_capture holds input hashes only, no DiT outputs.
    return {"frozen": {name: frozen_tensor(shas[name]) for name in shas}}


def test_warmup_input_compare_works_on_prepare_shaped_capture() -> None:
    warmup = prepare_shaped({"noise_chunk_0": "n", "condition_chunk_0": "x", "plucker_chunk_0": "p", "text_context": "t"})
    capture = {
        "frozen": {
            "noise_chunk_0": frozen_tensor("n"),
            "condition_chunk_0": frozen_tensor("x"),
            "plucker_chunk_0": frozen_tensor("p"),
            "text_context": frozen_tensor("t"),
            "bootstrap_x0": frozen_tensor("y0"),
            "layer0_clean_k": frozen_tensor("yk"),
            "layer0_clean_v": frozen_tensor("yv"),
        }
    }
    passed, detail = warmed_input_strict_equal(warmup, capture)
    assert passed is True
    assert detail == {
        "noise_chunk_0": True,
        "condition_chunk_0": True,
        "plucker_chunk_0": True,
        "text_context": True,
    }


def test_warmup_input_compare_detects_condition_drift() -> None:
    warmup = prepare_shaped({"noise_chunk_0": "n", "condition_chunk_0": "x", "plucker_chunk_0": "p", "text_context": "t"})
    capture = prepare_shaped({"noise_chunk_0": "n", "condition_chunk_0": "y", "plucker_chunk_0": "p", "text_context": "t"})
    passed, detail = warmed_input_strict_equal(warmup, capture)
    assert passed is False
    assert detail["condition_chunk_0"] is False
    assert detail["noise_chunk_0"] is True


def test_bootstrap_cursors_ignore_capacity_key() -> None:
    # cache_positions() reports cache_capacity_tokens alongside the two
    # cursors; a correct bootstrap must still pass.
    reset_pos = {
        "global_end_index": 1008,
        "local_end_index": 1008,
        "cache_capacity_tokens": 12096,
    }
    assert reset_cursors_at_bootstrap(reset_pos, 1008) is True


def test_bootstrap_cursors_reject_wrong_positions() -> None:
    assert reset_cursors_at_bootstrap(
        {"global_end_index": 0, "local_end_index": 0, "cache_capacity_tokens": 12096},
        1008,
    ) is False
    assert reset_cursors_at_bootstrap(
        {"global_end_index": 1008, "local_end_index": 0},
        1008,
    ) is False
