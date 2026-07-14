"""Modal backend: runs the Graph-RKD plan on RENTED GPUs, in parallel, pulling
the datasets from S3 and logging to the user's W&B. No AWS quota, pay per second.

Prerequisites (one time):
    pip install modal
    modal setup                                   # authenticates the Modal account
    modal secret create wandb WANDB_API_KEY=xxxxx
    modal secret create aws AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx \
        AWS_DEFAULT_REGION=us-east-1

Usage:
    modal run --detach sm/run_modal.py --phases "teachers phase0 phase1"
    modal run --detach sm/run_modal.py --phases "phase5" --gpu A10G

Parallelism: `train_job.map(...)` spins up one GPU container per job and autoscales. The
teachers run first (barrier); the students pull the teacher via the W&B artifact
``metric-<arch>-<dataset>:best`` (same mechanism as the SageMaker backend).
"""

import os
import sys

import modal

RKD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../RKD

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.7.0", "torchvision==0.22.0", "numpy<2.0", "scipy>=1.15",
        "wandb==0.27.0", "tqdm>=4.67", "kagglehub>=1.0.1", "awscli", "boto3",
    )
    .add_local_dir(RKD_DIR, remote_path="/root/RKD")
)

app = modal.App("graph-rkd")

# Persistent Volume = CACHE of the extracted datasets, shared across ALL
# containers. We download/extract ONCE (prepare_data) and the train_jobs only read —
# before, each container re-downloaded the Cars.tar (4 GB). Public download (no creds).
data_vol = modal.Volume.from_name("graph-rkd-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": data_vol}, timeout=2 * 3600)
def prepare_data(datasets: list, data_s3: str):
    """Populates the /data Volume (once) with the extracted datasets and commits."""
    sys.path.insert(0, "/root/RKD/sm")
    import data_prep
    data_prep.ensure("/data", datasets, s3_prefix=data_s3)
    data_vol.commit()                       # persists for the other containers
    return sorted(datasets)


@app.function(image=image, gpu=os.environ.get("GRAPH_RKD_GPU", "A10G"),
              timeout=8 * 3600, volumes={"/data": data_vol},
              secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("aws")])
def train_job(spec: dict):
    """Runs ONE job (teacher/baseline/distill) in a GPU container. The dataset is
    already in the /data Volume (cache) — data_prep sees the marker and downloads nothing."""
    sys.path.insert(0, "/root/RKD")
    sys.path.insert(0, "/root/RKD/sm")
    os.chdir("/root/RKD")
    import data_prep

    ds = spec["params"].get("dataset")
    if ds:
        data_prep.ensure("/data", [ds], s3_prefix=spec["data_s3"])  # cache hit on the Volume
    params = dict(spec["params"])
    params["data"] = "/data"
    params["save_dir"] = f"/root/out/{spec['name']}"
    params["resume"] = os.path.join(params["save_dir"],
                                    {"teacher": "last.pth", "baseline": "baseline_last.pth",
                                     "distill": "student_last.pth"}[spec["kind"]])
    if spec["kind"] == "teacher":
        import finetune_metric as t
    elif spec["kind"] == "baseline":
        import train_metric_baseline as t
    else:
        import distill_metric as t
    t.run_with_params(params)
    return spec["name"]


@app.function(image=image, timeout=24 * 3600)
def driver(teachers: list, rest: list, data_s3: str):
    """Orchestrates the campaign INSIDE Modal (not on the laptop): caches the datasets
    on the Volume ONCE, then the teacher barrier and the student fan-out. Running the
    orchestration server-side makes the campaign immune to the local connection (the old
    version failed because the ``local_entrypoint`` held the ``.map()`` for ~40 min
    on the laptop and the network dropped -> 'function is stopped')."""
    datasets = sorted({s["params"]["dataset"] for s in (teachers + rest)
                       if s["params"].get("dataset")})
    if datasets:
        print("preparing datasets in the cache (Volume):", datasets)
        prepare_data.remote(datasets, data_s3)     # single download/extract
    if teachers:
        tres = list(train_job.map(teachers, return_exceptions=True))
        failed = [str(r) for r in tres if isinstance(r, Exception)]
        if failed:
            print("teacher(s) FAILED, aborting students:", failed)
            return {"teacher_failed": failed}
        print("teachers ok:", [r for r in tres if not isinstance(r, Exception)])
    sres = list(train_job.map(rest, return_exceptions=True))
    ok = [r for r in sres if not isinstance(r, Exception)]
    bad = [str(r) for r in sres if isinstance(r, Exception)]
    print(f"students: {len(ok)} ok, {len(bad)} failures", ("| failures: " + str(bad)) if bad else "")
    return {"ok": ok, "failed": bad}


@app.local_entrypoint()
def main(phases: str = "teachers phase0 phase1",
         wandb_entity: str = "gabomfim-unicamp",
         wandb_project: str = "graph-rkd",
         data_s3: str = "s3://graph-rkd-832271495954/graph-rkd/data",
         only: str = "", student_epochs: int = 0, search_epochs: int = 0,
         trimmed: bool = False,
         headline_method: str = "", headline_norm: str = "",
         headline_objective: str = "", headline_nodes: int = 0,
         headline_lambda: float = 0.0):
    """``only`` = filter (comma-separated substrings) over the job NAMES,
    to (re)run a subset — e.g.: --only lg100-s0 runs only that λg point.
    If the filter excludes the teachers, they are skipped (the W&B artifact is used).
    ``student_epochs``/``search_epochs`` (>0) override the epoch budget
    (cost/convergence control)."""
    sys.path.insert(0, os.path.join(RKD_DIR, "sm"))
    import plan
    import launch

    ov = dict(wandb_entity=wandb_entity, wandb_project=wandb_project)
    if student_epochs:
        ov["student_epochs"] = student_epochs
    if search_epochs:
        ov["search_epochs"] = search_epochs
    if headline_method:
        ov["headline_method"] = headline_method
    if headline_norm:
        ov["headline_norm"] = headline_norm
    if headline_objective:
        ov["headline_objective"] = headline_objective
    if headline_nodes:
        ov["headline_nodes"] = headline_nodes
    if headline_lambda:
        ov["headline_lambda"] = headline_lambda
    cfg = plan.merged_config(trimmed=trimmed, **ov)
    jobs = plan.build_plan(cfg, phases.split())
    specs = []
    for s in jobs:
        payload = launch.build_spec_payload(cfg, s)       # {kind,name,params(+wandb,teacher)}
        payload["dataset"] = s["dataset"]
        payload["data_s3"] = data_s3
        specs.append(payload)

    if only:
        subs = [x.strip() for x in only.split(",") if x.strip()]
        specs = [s for s in specs if any(x in s["name"] for x in subs)]
        print("--only filter ->", [s["name"] for s in specs])

    teachers = [s for s in specs if s["kind"] == "teacher"]
    rest = [s for s in specs if s["kind"] != "teacher"]
    # Spawn (fire-and-forget) the driver: ALL orchestration + training runs on Modal.
    # With `modal run --detach`, the app stays alive even if the laptop disconnects.
    call = driver.spawn(teachers, rest)
    print(f"Modal: {len(teachers)} teachers -> {len(rest)} students "
          f"| W&B {wandb_entity}/{wandb_project}")
    print(f"driver spawned (call id: {call.object_id}); run with `modal run --detach`. "
          "Track it at modal.com/apps and on W&B.")
