"""Entrypoint executed INSIDE the SageMaker training job.

SageMaker invokes this script (entry_point) with the estimator's hyperparameters
converted into ``--flag value``. We pass ONE single hyperparameter ``--spec`` with
the serialized JobSpec (JSON) to avoid list/quote issues. This script:

  1. reads the ``--spec`` (kind + trainer params, already with wandb_* and teacher);
  2. resolves, from the SageMaker environment, the DATA directory (mounted input
     channel, or downloads via torchvision) and the CHECKPOINT one (resumable);
  3. dispatches to the right trainer (finetune_metric / train_metric_baseline /
     distill_metric) via ``run_with_params``.

cwd in the container = source_dir (RKD/), so ``import distill_metric`` etc. works.
W&B: needs ``WANDB_API_KEY`` in the environment (the launcher forwards it).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # RKD/sm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # RKD
import data_prep  # noqa: E402  (same cache layer as local/Modal)

# "last" checkpoint names per trainer (for resume)
LAST_NAME = {"teacher": "last.pth", "baseline": "baseline_last.pth",
             "distill": "student_last.pth"}


def resolve_data_dir():
    """CACHE dir for the datasets (same `data_prep` layer as local/Modal).

    Preference: an already-mounted input channel (``SM_CHANNEL_DATA`` — pre-staged
    data), else ``/opt/ml/checkpoints`` (synced with S3, survives
    resume/spot => cache across re-runs of the same job), else ``/opt/ml/data``.
    The download itself is done by ``data_prep.ensure`` (public HTTPS, no creds)."""
    ch = os.environ.get("SM_CHANNEL_DATA") or os.environ.get("DATA_DIR")
    if ch and os.path.isdir(ch):
        return ch
    if os.path.isdir("/opt/ml/checkpoints"):
        d = "/opt/ml/checkpoints/_data"
        os.makedirs(d, exist_ok=True)
        return d
    os.makedirs("/opt/ml/data", exist_ok=True)
    return "/opt/ml/data"


def resolve_ckpt_dir():
    """Resumable checkpoint directory: uses /opt/ml/checkpoints (synced
    with checkpoint_s3_uri, survives spot/retry) if it exists; else the model dir."""
    ck = "/opt/ml/checkpoints"
    if os.path.isdir(ck):
        return ck
    return os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


def dispatch(kind, params):
    if kind == "teacher":
        import finetune_metric as trainer
    elif kind == "baseline":
        import train_metric_baseline as trainer
    elif kind == "distill":
        import distill_metric as trainer
    else:
        raise SystemExit(f"unknown kind: {kind}")
    return trainer.run_with_params(params)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="JobSpec JSON (kind + params)")
    opts = p.parse_args(argv)
    spec = json.loads(opts.spec)
    kind = spec["kind"]
    params = dict(spec["params"])

    save_dir = os.path.join(resolve_ckpt_dir(), spec.get("name", kind))
    params["save_dir"] = save_dir
    params["resume"] = os.path.join(save_dir, LAST_NAME[kind])
    cache = resolve_data_dir()
    ds = params.get("dataset")
    if ds:                                          # download (public) + cache
        data_prep.ensure(cache, [ds], s3_prefix=data_prep.DEFAULT_S3)
    params["data"] = cache
    params.setdefault("wandb_mode", "online")

    print(f"[entry] kind={kind} name={spec.get('name')} "
          f"data={params['data']} save_dir={save_dir}", flush=True)
    print(f"[entry] wandb: project={params.get('wandb_project')} "
          f"entity={params.get('wandb_entity')} id={params.get('wandb_id')}", flush=True)
    result = dispatch(kind, params)
    print(f"[entry] done: {result if isinstance(result, (int, float)) else 'ok'}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
