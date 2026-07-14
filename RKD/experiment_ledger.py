"""Campaign ledger: marks completed experiments and caches their results,
so that an interrupted campaign is resumed without redoing work.

Each experiment is identified by its ``save_dir`` (unique). We write a
``result.json`` upon completion; on re-run, an experiment with ``result.json``
is skipped (and its result is read from disk, important for the N selection).
"""

import json
import os

__all__ = ["is_done", "load_result", "mark_done"]


def _result_path(save_dir):
    return os.path.join(save_dir, "result.json")


def is_done(save_dir):
    """True if this experiment has already been completed (has result.json)."""
    return save_dir is not None and os.path.exists(_result_path(save_dir))


def load_result(save_dir):
    """Cached result (dict) or None."""
    if not is_done(save_dir):
        return None
    with open(_result_path(save_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def mark_done(save_dir, result=None):
    """Writes result.json marking completion (result must be JSON-serializable)."""
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    with open(_result_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(result if result is not None else {"done": True}, f)
