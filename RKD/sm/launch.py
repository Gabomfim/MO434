"""Dispara o plano de experimentos Graph-RKD como jobs de treino do SageMaker,
EM PARALELO, logando no W&B do usuário.

Por padrão faz **dry-run**: monta o plano, imprime o resumo e grava ``plan.json``
SEM tocar na AWS. Só com ``--launch`` cria os jobs (custa dinheiro).

Paralelismo e dependências:
  * onda 1: teachers + baselines (sem dependência) sobem juntos;
  * espera SÓ os teachers concluírem (baselines seguem em paralelo);
  * onda 2: todos os jobs de destilação sobem juntos (puxam o teacher por
    artefato W&B ``metric-<arch>-<dataset>:best``).
Cada job é 1 instância; a config (role/bucket/região/instância/spot) vem por flag
ou variável de ambiente.

Resumibilidade: o nome LÓGICO do job (plan) fixa o ``wandb_id`` (retoma a mesma
run) e o ``checkpoint_s3_uri`` (retoma o treino); o nome do job SageMaker recebe
um sufixo único (jobs SageMaker não podem reusar nome).

Exemplo (dry-run):
    python sm/launch.py --phases teachers phase0 phase1
Exemplo (lançar de verdade):
    export WANDB_API_KEY=...   # obrigatório
    python sm/launch.py --phases teachers phase0 --launch \
        --region us-east-1 --role arn:aws:iam::123:role/SageMakerRole \
        --bucket my-bucket --wandb-entity myuser --wandb-project graph-rkd
"""

import argparse
import hashlib
import json
import os
import sys
import time

import plan  # módulo irmão (RKD/sm/plan.py)

RKD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../RKD
ENTRY_POINT = "sm/entry.py"
# arquivo (objeto S3) por dataset -> canal de dados enxuto por job
ARCHIVE = {"cars196": "Cars196.tar", "cub200": "CUB_200_2011.tgz"}


def _stable_id(name):
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _job_name(logical, run_tag):
    """Nome de job SageMaker único e válido (<=63, [a-zA-Z0-9-])."""
    base = "".join(c if c.isalnum() or c == "-" else "-" for c in logical)
    suffix = "-" + run_tag
    return base[: 63 - len(suffix)].strip("-") + suffix


def teacher_ref(cfg, spec):
    ent, proj = cfg["wandb_entity"], cfg["wandb_project"]
    art = plan.teacher_artifact_name(spec["arch"], spec["dataset"])
    return f"{ent}/{proj}/{art}:best" if ent else f"{proj}/{art}:best"


def job_params(cfg, spec):
    """Params finais do trainer p/ este job: injeta W&B e o teacher."""
    params = dict(spec["params"])
    params["wandb_project"] = cfg["wandb_project"]
    if cfg["wandb_entity"]:
        params["wandb_entity"] = cfg["wandb_entity"]
    params["wandb_run_name"] = spec["wandb"]["run_name"]
    params["wandb_group"] = spec["wandb"]["group"]
    params["wandb_tags"] = spec["wandb"]["tags"]
    params["wandb_id"] = _stable_id(spec["name"])
    params["wandb_mode"] = "online"
    if spec["kind"] == "distill":
        params["teacher_artifact"] = teacher_ref(cfg, spec)
    return params


def build_spec_payload(cfg, spec):
    """JobSpec compacto passado ao entry.py via hiperparâmetro --spec."""
    return {"kind": spec["kind"], "name": spec["name"],
            "params": job_params(cfg, spec)}


# --------------------------------------------------------------------------- #
# AWS (importado só quando --launch)                                           #
# --------------------------------------------------------------------------- #
def make_estimator(cfg, spec, sm_session, run_tag):
    from sagemaker.pytorch import PyTorch

    payload = build_spec_payload(cfg, spec)
    spec_json = json.dumps(payload, separators=(",", ":"))
    if len(spec_json) > 2400:
        raise SystemExit(f"spec JSON grande demais ({len(spec_json)}B) p/ hiperparâmetro")

    env = {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
           "WANDB_START_METHOD": "thread"}
    kwargs = dict(
        entry_point=ENTRY_POINT, source_dir=RKD_DIR,
        role=cfg["role"] or "arn:aws:iam::000000000000:role/dummy",
        framework_version=cfg["framework_version"], py_version=cfg["py_version"],
        instance_count=1, hyperparameters={"spec": spec_json}, environment=env,
        sagemaker_session=sm_session, base_job_name=spec["name"][:40],
    )
    if cfg.get("local"):
        # SageMaker LOCAL MODE: roda o container na GPU local via Docker (sem cota).
        kwargs.update(instance_type="local_gpu" if cfg["local_gpu"] else "local",
                      output_path=f"file://{os.path.abspath('sm_local_output')}")
        est = PyTorch(**kwargs)
        data = os.path.abspath(cfg["data_local"])
        return est, {"data": f"file://{data}"}, _job_name(spec["name"], run_tag)

    s3_prefix = f"s3://{cfg['bucket']}/{cfg['prefix']}"
    kwargs.update(
        instance_type=cfg["instance_type"],
        output_path=f"{s3_prefix}/output", code_location=f"{s3_prefix}/code",
        checkpoint_s3_uri=f"{s3_prefix}/checkpoints/{spec['name']}",
        checkpoint_local_path="/opt/ml/checkpoints",
        max_run=cfg["max_run"], volume_size=cfg["volume_size"],
    )
    if cfg["use_spot"]:
        kwargs.update(use_spot_instances=True,
                      max_wait=max(cfg["max_run"], cfg["max_wait"]))
    est = PyTorch(**kwargs)
    # canal de dados = APENAS o arquivo do dataset deste job (download enxuto);
    # entry.py extrai. Se data_s3 apontar a uma árvore extraída, use-a direto.
    inputs = None
    if cfg["data_s3"]:
        base = cfg["data_s3"].rstrip("/")
        arch = ARCHIVE.get(spec["dataset"])
        inputs = {"data": f"{base}/{arch}" if arch else cfg["data_s3"]}
    return est, inputs, _job_name(spec["name"], run_tag)


def wait_for(sm_client, job_names, poll=30, timeout=None):
    """Espera todos os job_names atingirem Completed; falha se algum Failed/Stopped."""
    start = time.time()
    pending = set(job_names)
    while pending:
        done = set()
        for jn in list(pending):
            st = sm_client.describe_training_job(TrainingJobName=jn)["TrainingJobStatus"]
            if st == "Completed":
                done.add(jn)
            elif st in ("Failed", "Stopped"):
                raise SystemExit(f"Job de teacher {jn} terminou em {st} — abortando "
                                 "(os alunos dependem dele).")
        pending -= done
        if done:
            print(f"[wait] concluídos: {sorted(done)} | restam {len(pending)}", flush=True)
        if not pending:
            break
        if timeout and time.time() - start > timeout:
            raise SystemExit(f"[wait] timeout esperando teachers: {sorted(pending)}")
        time.sleep(poll)


def do_launch_local(cfg, jobs):
    """SageMaker LOCAL MODE: roda cada job no container, na GPU local (Docker),
    SEM cota AWS. Sequencial (local mode é síncrono): teachers -> alunos. Os
    alunos puxam o teacher pelo artefato W&B, então precisam de WANDB online."""
    from sagemaker.local import LocalSession
    sm_session = LocalSession()
    sm_session.config = {"local": {"local_code": True}}
    run_tag = "local"
    order = ([j for j in jobs if j["kind"] == "teacher" and not cfg["skip_teachers"]]
             + [j for j in jobs if j["kind"] == "baseline"]
             + [j for j in jobs if j["kind"] == "distill"])
    print(f"[local] {len(order)} jobs sequenciais via Docker "
          f"(instance={'local_gpu' if cfg['local_gpu'] else 'local'}).")
    for s in order:
        est, inputs, jn = make_estimator(cfg, s, sm_session, run_tag)
        print(f"[local] treinando {s['name']} ({s['phase']}/{s['kind']})...", flush=True)
        est.fit(inputs=inputs, wait=True, job_name=jn)
    print(f"\n[local] {len(order)} jobs concluídos. Resultados no W&B "
          f"({cfg['wandb_entity'] or '<default>'}/{cfg['wandb_project']}).")


def do_launch(cfg, jobs):
    if cfg.get("local"):
        return do_launch_local(cfg, jobs)
    import boto3
    import sagemaker

    boto_sess = boto3.Session(region_name=cfg["region"])
    sm_session = sagemaker.Session(boto_session=boto_sess)
    sm_client = boto_sess.client("sagemaker")
    run_tag = format(int(time.time()), "x")[-6:]

    teachers = [j for j in jobs if j["kind"] == "teacher" and not cfg["skip_teachers"]]
    independents = [j for j in jobs if j["kind"] == "baseline"]
    distills = [j for j in jobs if j["kind"] == "distill"]

    launched = {}

    def submit(spec):
        est, inputs, jn = make_estimator(cfg, spec, sm_session, run_tag)
        print(f"[launch] {jn}  ({spec['phase']}/{spec['kind']})", flush=True)
        est.fit(inputs=inputs, wait=False, job_name=jn)
        launched[spec["name"]] = jn
        return jn

    # onda 1: teachers + baselines
    teacher_jn = [submit(s) for s in teachers]
    for s in independents:
        submit(s)

    # espera só os teachers; então onda 2 (destilações)
    if distills:
        if teacher_jn:
            print(f"[wait] aguardando {len(teacher_jn)} teacher(s) concluírem...", flush=True)
            wait_for(sm_client, teacher_jn, timeout=cfg["wait_timeout"])
        for s in distills:
            submit(s)

    print(f"\n[done] {len(launched)} jobs disparados na região {cfg['region']}.")
    print("Acompanhe no console SageMaker (Training jobs) e no W&B "
          f"({cfg['wandb_entity'] or '<default>'}/{cfg['wandb_project']}).")
    return launched


# --------------------------------------------------------------------------- #
# dry-run / CLI                                                                #
# --------------------------------------------------------------------------- #
def print_plan(cfg, jobs):
    by_phase, by_kind = plan.summarize(jobs)
    print("\n================ PLANO Graph-RKD (SageMaker) ================")
    print(f"phases={cfg['phases']}")
    print(f"datasets={cfg['datasets']} teachers={cfg['teachers']} "
          f"methods={cfg['methods']} objectives={cfg['objectives']}")
    print(f"norms={cfg['norms']} n_list={cfg['n_list']} λg_grid={cfg['lambda_grid']} "
          f"seeds={cfg['seeds']}")
    print(f"W&B -> entity={cfg['wandb_entity'] or '<default>'} project={cfg['wandb_project']}")
    print(f"instância={cfg['instance_type']} spot={cfg['use_spot']} "
          f"region={cfg['region'] or '<unset>'}")
    print("por fase: " + " ".join(f"{k}={v}" for k, v in sorted(by_phase.items())))
    print("por kind: " + " ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    print(f"TOTAL de jobs: {len(jobs)}")
    print("-" * 60)
    for s in jobs:
        dep = f"  <- {s['depends_on']}" if s.get("depends_on") else ""
        print(f"  [{s['phase']:8s}] {s['kind']:8s} {s['name']}{dep}")


def build_parser():
    p = argparse.ArgumentParser(description="Launcher SageMaker do Graph-RKD")
    p.add_argument("--phases", nargs="+", default=["teachers", "phase0", "phase1"],
                   choices=list(plan.PHASES))
    p.add_argument("--launch", action="store_true",
                   help="cria os jobs de verdade (default: dry-run)")
    p.add_argument("--local", action="store_true",
                   help="SageMaker LOCAL MODE: roda na GPU local via Docker (sem cota)")
    p.add_argument("--local-cpu", action="store_true",
                   help="com --local, usar instance 'local' (CPU) em vez de local_gpu")
    p.add_argument("--data-local", default="data",
                   help="dir local dos dados p/ --local (montado como canal file://)")
    p.add_argument("--out", default="plan.json", help="onde gravar o plano (JSON)")

    # W&B
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-project", default="graph-rkd")

    # AWS
    p.add_argument("--region", default=os.environ.get("AWS_REGION"))
    p.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    p.add_argument("--bucket", default=os.environ.get("SAGEMAKER_BUCKET"))
    p.add_argument("--prefix", default="graph-rkd")
    p.add_argument("--data-s3", default=os.environ.get("GRAPH_RKD_DATA_S3"),
                   help="s3://.../ com Cars196/ e CUB_200_2011/ (canal 'data'); "
                        "se omitido, cada job baixa via torchvision")
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--framework-version", default="2.2")
    p.add_argument("--py-version", default="py310")
    p.add_argument("--volume-size", type=int, default=100)
    p.add_argument("--max-run", type=int, default=48 * 3600)
    p.add_argument("--max-wait", type=int, default=72 * 3600)
    p.add_argument("--no-spot", action="store_true", help="desliga managed spot")
    p.add_argument("--skip-teachers", action="store_true",
                   help="não treina teachers (assume artefatos W&B já existentes)")
    p.add_argument("--wait-timeout", type=int, default=None)

    # overrides do plano (senão usa DEFAULTS)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--teachers", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--objectives", nargs="+")
    p.add_argument("--norms", nargs="+")
    p.add_argument("--n-list", nargs="+", type=int)
    p.add_argument("--lambda-grid", nargs="+", type=float)
    p.add_argument("--seeds", type=int)
    p.add_argument("--student-epochs", type=int)
    p.add_argument("--search-epochs", type=int)
    p.add_argument("--teacher-epochs", type=int)
    p.add_argument("--gate-dataset", choices=["cars196", "cub200"])
    p.add_argument("--gate-teacher", choices=["resnet18", "convnext_tiny"])
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = plan.merged_config(
        datasets=a.datasets, teachers=a.teachers, methods=a.methods,
        objectives=a.objectives, norms=a.norms, n_list=a.n_list,
        lambda_grid=a.lambda_grid, seeds=a.seeds,
        student_epochs=a.student_epochs, search_epochs=a.search_epochs,
        teacher_epochs=a.teacher_epochs,
        gate_dataset=a.gate_dataset, gate_teacher=a.gate_teacher,
        wandb_entity=a.wandb_entity, wandb_project=a.wandb_project,
    )
    # config de execução (não faz parte do plano puro)
    cfg.update(phases=a.phases, region=a.region, role=a.role, bucket=a.bucket,
               prefix=a.prefix, data_s3=a.data_s3, instance_type=a.instance_type,
               framework_version=a.framework_version, py_version=a.py_version,
               volume_size=a.volume_size, max_run=a.max_run, max_wait=a.max_wait,
               use_spot=not a.no_spot, skip_teachers=a.skip_teachers,
               wait_timeout=a.wait_timeout, local=a.local,
               local_gpu=not a.local_cpu, data_local=a.data_local)

    jobs = plan.build_plan(cfg, a.phases)
    print_plan(cfg, jobs)

    payload = [build_spec_payload(cfg, s) | {"phase": s["phase"],
                                             "depends_on": s.get("depends_on")}
               for s in jobs]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nPlano gravado em {a.out} ({len(jobs)} jobs).")

    if not a.launch:
        mode = "LOCAL MODE (Docker)" if a.local else "AWS"
        print(f"\n(dry-run: nada criado. Use --launch p/ disparar em {mode}.)")
        return
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("--launch exige WANDB_API_KEY no ambiente (p/ logar no W&B).")
    if not cfg["local"]:
        missing = [k for k in ("region", "role", "bucket") if not cfg[k]]
        if missing:
            raise SystemExit(f"--launch exige {missing} (flag ou variável de ambiente). "
                             "Ou use --local p/ rodar na GPU local via Docker.")
    do_launch(cfg, jobs)


if __name__ == "__main__":
    sys.exit(main())
