"""Backend Modal: roda o plano Graph-RKD em GPUs ALUGADAS, em paralelo, puxando
os datasets do S3 e logando no W&B do usuário. Sem cota AWS, pagamento por segundo.

Pré-requisitos (uma vez):
    pip install modal
    modal setup                                   # autentica a conta Modal
    modal secret create wandb WANDB_API_KEY=xxxxx
    modal secret create aws AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx \
        AWS_DEFAULT_REGION=us-east-1

Uso:
    modal run sm/run_modal.py --phases "teachers phase0 phase1"
    modal run sm/run_modal.py --phases "phase5" --gpu A10G

Paralelismo: `train_job.map(...)` sobe um container GPU por job e autoescala. Os
teachers rodam primeiro (barreira); os alunos puxam o teacher pelo artefato W&B
``metric-<arch>-<dataset>:best`` (mesmo mecanismo do backend SageMaker).
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


@app.function(image=image, gpu=os.environ.get("GRAPH_RKD_GPU", "A10G"),
              timeout=8 * 3600,
              secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("aws")])
def train_job(spec: dict):
    """Roda UM job (teacher/baseline/distill) num container GPU."""
    sys.path.insert(0, "/root/RKD")
    sys.path.insert(0, "/root/RKD/sm")
    os.chdir("/root/RKD")
    import data_prep

    ds = spec["params"].get("dataset")
    if ds:
        data_prep.ensure("/data", [ds], s3_prefix=spec["data_s3"])  # creds via secret
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


@app.local_entrypoint()
def main(phases: str = "teachers phase0 phase1",
         wandb_entity: str = "gabomfim-unicamp",
         wandb_project: str = "graph-rkd",
         data_s3: str = "s3://graph-rkd-832271495954/graph-rkd/data"):
    sys.path.insert(0, os.path.join(RKD_DIR, "sm"))
    import plan
    import launch

    cfg = plan.merged_config(wandb_entity=wandb_entity, wandb_project=wandb_project)
    jobs = plan.build_plan(cfg, phases.split())
    specs = []
    for s in jobs:
        payload = launch.build_spec_payload(cfg, s)       # {kind,name,params(+wandb,teacher)}
        payload["dataset"] = s["dataset"]
        payload["data_s3"] = data_s3
        specs.append(payload)

    teachers = [s for s in specs if s["kind"] == "teacher"]
    rest = [s for s in specs if s["kind"] != "teacher"]
    print(f"Modal: {len(teachers)} teachers -> depois {len(rest)} alunos "
          f"| W&B {wandb_entity}/{wandb_project}")
    if teachers:
        for name in train_job.map(teachers):             # barreira: espera teachers
            print("teacher ok:", name)
    for name in train_job.map(rest):                     # alunos em paralelo
        print("done:", name)
    print("campanha Modal concluída.")
