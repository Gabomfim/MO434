"""Backend Modal: roda o plano Graph-RKD em GPUs ALUGADAS, em paralelo, puxando
os datasets do S3 e logando no W&B do usuário. Sem cota AWS, pagamento por segundo.

Pré-requisitos (uma vez):
    pip install modal
    modal setup                                   # autentica a conta Modal
    modal secret create wandb WANDB_API_KEY=xxxxx
    modal secret create aws AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx \
        AWS_DEFAULT_REGION=us-east-1

Uso:
    modal run --detach sm/run_modal.py --phases "teachers phase0 phase1"
    modal run --detach sm/run_modal.py --phases "phase5" --gpu A10G

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


@app.function(image=image, timeout=24 * 3600)
def driver(teachers: list, rest: list):
    """Orquestra a campanha DENTRO do Modal (não no laptop): barreira dos teachers
    e então o fan-out dos alunos. Rodar a orquestração server-side é o que torna a
    campanha imune à conexão local — a versão anterior falhava porque o
    ``local_entrypoint`` segurava o ``.map()`` dos teachers por ~40 min no laptop
    e a rede caía antes do ``.map()`` dos alunos ('function is stopped')."""
    if teachers:
        tres = list(train_job.map(teachers, return_exceptions=True))
        failed = [str(r) for r in tres if isinstance(r, Exception)]
        if failed:
            print("teacher(s) FALHARAM, abortando alunos:", failed)
            return {"teacher_failed": failed}
        print("teachers ok:", [r for r in tres if not isinstance(r, Exception)])
    sres = list(train_job.map(rest, return_exceptions=True))
    ok = [r for r in sres if not isinstance(r, Exception)]
    bad = [str(r) for r in sres if isinstance(r, Exception)]
    print(f"alunos: {len(ok)} ok, {len(bad)} falhas", ("| falhas: " + str(bad)) if bad else "")
    return {"ok": ok, "failed": bad}


@app.local_entrypoint()
def main(phases: str = "teachers phase0 phase1",
         wandb_entity: str = "gabomfim-unicamp",
         wandb_project: str = "graph-rkd",
         data_s3: str = "s3://graph-rkd-832271495954/graph-rkd/data",
         only: str = "", student_epochs: int = 0, search_epochs: int = 0):
    """``only`` = filtro (substrings separadas por vírgula) sobre os NOMES dos jobs,
    p/ (re)rodar um subconjunto — ex.: --only lg100-s0 roda só aquele ponto de λg.
    Se o filtro excluir os teachers, eles são pulados (usa-se o artefato W&B).
    ``student_epochs``/``search_epochs`` (>0) sobrescrevem o orçamento de época
    (controle de custo/convergência)."""
    sys.path.insert(0, os.path.join(RKD_DIR, "sm"))
    import plan
    import launch

    ov = dict(wandb_entity=wandb_entity, wandb_project=wandb_project)
    if student_epochs:
        ov["student_epochs"] = student_epochs
    if search_epochs:
        ov["search_epochs"] = search_epochs
    cfg = plan.merged_config(**ov)
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
        print("filtro --only ->", [s["name"] for s in specs])

    teachers = [s for s in specs if s["kind"] == "teacher"]
    rest = [s for s in specs if s["kind"] != "teacher"]
    # Spawn (fire-and-forget) do driver: TODA a orquestração + treino roda no Modal.
    # Com `modal run --detach`, o app segue vivo mesmo que o laptop desconecte.
    call = driver.spawn(teachers, rest)
    print(f"Modal: {len(teachers)} teachers -> {len(rest)} alunos "
          f"| W&B {wandb_entity}/{wandb_project}")
    print(f"driver spawned (call id: {call.object_id}); rode com `modal run --detach`. "
          "Acompanhe em modal.com/apps e no W&B.")
