"""Helpers for fetching model checkpoints (e.g. teachers) from Weights & Biases.

Teacher checkpoints are logged by run.py as W&B artifacts named
``model-best-<base>-<seed>`` (file ``best.pth``, alias ``best``) and
``model-last-<base>-<seed>`` (file ``last.pth``, alias ``last``). Student
training can pull them directly instead of relying on a local path.
"""

import glob
import hashlib
import os
from datetime import timedelta

import wandb


def stable_run_id(name):
    """Deterministic W&B run ID from a stable name.

    Reusing the same id (with resume='allow') makes a restarted job RESUME the same
    run in W&B instead of creating another. 16 hex of sha1 -> safe for a run id.
    """
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def log_model_artifact(run, file_path, name, aliases=None, metadata=None,
                       ttl_days=None):
    """Ship a saved checkpoint to W&B as a versioned ``model`` artifact.

    No-op when there is no active run or the run is disabled, so callers can
    invoke it unconditionally. ``name`` should be stable across a run so W&B
    versions (v0, v1, ...) accumulate under one artifact; ``aliases`` (e.g.
    ["best"], ["last"]) tag specific versions.

    ``ttl_days`` sets a time-to-live so old versions expire automatically -- a
    safety net against unbounded storage growth. Avoid logging a new version
    every epoch and avoid unique per-epoch aliases (e.g. "epoch-N"), which keep
    every version pinned forever.
    """
    if run is None or getattr(run, "disabled", False):
        return
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
    artifact.add_file(file_path)
    if ttl_days is not None:
        artifact.ttl = timedelta(days=ttl_days)
    run.log_artifact(artifact, aliases=aliases or [])


def _find_checkpoint(artifact_dir):
    """Locate the .pth checkpoint inside a downloaded artifact directory."""
    for preferred in ("best.pth", "last.pth"):
        candidate = os.path.join(artifact_dir, preferred)
        if os.path.exists(candidate):
            return candidate

    pths = sorted(glob.glob(os.path.join(artifact_dir, "*.pth")))
    if not pths:
        raise FileNotFoundError(
            "No .pth checkpoint found in artifact dir %s" % artifact_dir
        )
    if len(pths) > 1:
        raise ValueError(
            "Multiple .pth files in artifact %s; expected one of best.pth/last.pth. "
            "Found: %s" % (artifact_dir, pths)
        )
    return pths[0]


def resolve_teacher_checkpoint(teacher_load=None, teacher_artifact=None, run=None):
    """Return a local path to a teacher checkpoint.

    Provide exactly one of:
      * ``teacher_load``: a local checkpoint path (used as-is), or
      * ``teacher_artifact``: a W&B artifact reference. Within an active run
        the short form ``name:alias`` (e.g. ``model-best-resnet50-7:best``)
        resolves in the run's project/entity; otherwise pass the fully
        qualified ``entity/project/name:alias``.

    When an active ``run`` is given, ``use_artifact`` is used so the teacher is
    recorded in the run's lineage; otherwise the public W&B Api is used.
    """
    if (teacher_load is None) == (teacher_artifact is None):
        raise ValueError(
            "Provide exactly one of teacher_load (local path) or "
            "teacher_artifact (W&B reference)."
        )

    if teacher_load is not None:
        return teacher_load

    if run is not None:
        artifact = run.use_artifact(teacher_artifact, type="model")
    else:
        artifact = wandb.Api().artifact(teacher_artifact, type="model")
    artifact_dir = artifact.download()
    return _find_checkpoint(artifact_dir)
