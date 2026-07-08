"""Roda o plano de experimentos (plan.py) LOCALMENTE, em PARALELO, maximizando
o uso da(s) GPU(s). Sem AWS.

Os modelos são pequenos (ConvNextMicro + teacher), então UM treino sozinho
subutiliza uma GPU grande. Este runner empacota VÁRIOS experimentos por GPU ao
mesmo tempo (cada job é um subprocesso próprio), dimensionando a concorrência
pela VRAM livre, e distribui jobs entre múltiplas GPUs em round-robin.

Escalonamento:
  * onda 1: teachers (+baselines) em paralelo;
  * um aluno (distill) só entra quando o teacher dele terminou com sucesso
    (best.pth presente); teacher que falha pula seus dependentes.
Resumível: jobs com ledger concluído são pulados; treinos retomam do *_last.pth.

Handoff de teacher: local, por CAMINHO (teacher_load = <save_root>/<teacher>/best.pth).

Exemplos:
    export WANDB_API_KEY=...
    # concorrência AUTO pela VRAM, todas as GPUs:
    python sm/run_local.py --phases teachers phase0 phase1 --data data \
        --wandb-entity gabomfim-unicamp --wandb-project graph-rkd --amp
    # fixar 6 jobs simultâneos:
    python sm/run_local.py --max-parallel 6 ...
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RKD = os.path.dirname(HERE)
sys.path.insert(0, RKD)
sys.path.insert(0, HERE)

import plan  # noqa: E402
import data_prep  # noqa: E402
from experiment_ledger import is_done, mark_done  # noqa: E402

LAST = {"teacher": "last.pth", "baseline": "baseline_last.pth",
        "distill": "student_last.pth"}


def _stable_id(name):
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _teacher_ckpt(save_root, arch, ds):
    return os.path.join(save_root, plan.teacher_name(arch, ds), "best.pth")


# --------------------------------------------------------------------------- #
# GPU detection / concurrency sizing                                          #
# --------------------------------------------------------------------------- #
def detect_gpus():
    """[(index, free_MiB), ...] via nvidia-smi; [] se não houver GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, free = (x.strip() for x in line.split(","))
        gpus.append((int(idx), int(free)))
    return gpus


def auto_parallel(gpus, per_job_gb):
    """Nº de jobs simultâneos = soma, por GPU, de floor(VRAM_livre / per_job)."""
    if not gpus:
        return 1, []
    slots = []                      # lista de índices de GPU (um por slot)
    for idx, free_mib in gpus:
        n = max(1, int((free_mib / 1024.0) / per_job_gb))
        slots.extend([idx] * n)
    return len(slots), slots


# --------------------------------------------------------------------------- #
# params de um job                                                            #
# --------------------------------------------------------------------------- #
def build_job(spec, cfg, save_root, workers):
    save_dir = os.path.join(save_root, spec["name"])
    params = dict(spec["params"])
    params["data"] = cfg["data"]
    params["save_dir"] = save_dir
    params["resume"] = os.path.join(save_dir, LAST[spec["kind"]])
    params["workers"] = workers
    params["wandb_project"] = cfg["wandb_project"]
    if cfg["wandb_entity"]:
        params["wandb_entity"] = cfg["wandb_entity"]
    params["wandb_run_name"] = spec["wandb"]["run_name"]
    params["wandb_group"] = spec["wandb"]["group"]
    params["wandb_tags"] = spec["wandb"]["tags"]
    params["wandb_id"] = _stable_id(spec["name"])
    params["wandb_mode"] = cfg["wandb_mode"]
    if spec["kind"] == "distill":
        params["teacher_load"] = _teacher_ckpt(save_root, spec["arch"], spec["dataset"])
    return save_dir, params


def _dispatch(kind):
    if kind == "teacher":
        import finetune_metric as t
    elif kind == "baseline":
        import train_metric_baseline as t
    else:
        import distill_metric as t
    return t


def run_worker(spec_file):
    """Executado no SUBPROCESSO: roda exatamente um job in-process."""
    with open(spec_file, encoding="utf-8") as f:
        job = json.load(f)
    _dispatch(job["kind"]).run_with_params(job["params"])


# --------------------------------------------------------------------------- #
# scheduler                                                                    #
# --------------------------------------------------------------------------- #
def schedule(jobs, cfg, save_root, max_parallel, gpu_slots, workers, poll=5):
    done, failed, skipped = set(), set(), set()
    remaining = list(jobs)
    running = {}          # name -> (Popen, slot_index)
    free_slots = list(range(max_parallel))

    def dep_ok(spec):
        d = spec.get("depends_on")
        if not d:
            return True
        if d in failed or d in skipped:
            return None      # dependência morta -> pular
        return d in done or is_done(os.path.join(save_root, d))

    def launch(spec, slot):
        save_dir, params = build_job(spec, cfg, save_root, workers)
        os.makedirs(save_dir, exist_ok=True)
        spec_file = os.path.join(save_dir, "_job.json")
        with open(spec_file, "w", encoding="utf-8") as f:
            json.dump({"kind": spec["kind"], "name": spec["name"],
                       "params": params}, f)
        env = dict(os.environ)
        if gpu_slots:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_slots[slot % len(gpu_slots)])
        log = open(os.path.join(save_dir, "run.log"), "w")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                              "--worker", spec_file], env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        running[spec["name"]] = (p, slot, log)
        gpu = env.get("CUDA_VISIBLE_DEVICES", "cpu")
        print(f"[run  ] {spec['name']}  (gpu={gpu}, {len(running)}/{max_parallel} busy)")

    total = len(jobs)
    while remaining or running:
        # preenche slots livres com jobs prontos
        progressed = True
        while progressed and free_slots and remaining:
            progressed = False
            for spec in list(remaining):
                sd = os.path.join(save_root, spec["name"])
                if is_done(sd):
                    remaining.remove(spec); done.add(spec["name"])
                    print(f"[skip ] {spec['name']} (ledger)"); progressed = True; continue
                ok = dep_ok(spec)
                if ok is None:
                    remaining.remove(spec); skipped.add(spec["name"])
                    print(f"[SKIP ] {spec['name']} (teacher falhou)"); progressed = True; continue
                if ok and free_slots:
                    remaining.remove(spec); launch(spec, free_slots.pop(0))
                    progressed = True
        # coleta terminados
        for name, (p, slot, log) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            del running[name]
            free_slots.append(slot)
            if rc == 0:
                done.add(name)
                mark_done(os.path.join(save_root, name), {"name": name})
                print(f"[done ] {name}  ({len(done)}/{total})")
            else:
                failed.add(name)
                print(f"[FAIL ] {name} (rc={rc}) — ver {save_root}/{name}/run.log")
        if running and not free_slots:
            time.sleep(poll)
        elif not remaining and running:
            time.sleep(poll)
    return done, failed, skipped


def build_parser():
    p = argparse.ArgumentParser(description="Runner local PARALELO do plano Graph-RKD")
    p.add_argument("--worker", help="(interno) roda um job a partir de um _job.json")
    p.add_argument("--phases", nargs="+", default=["teachers", "phase0", "phase1"],
                   choices=list(plan.PHASES))
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
    p.add_argument("--trimmed", action="store_true",
                   help="config enxuta (drop hybrid, λg={0.01,0.1,1}, N={3,4,8})")
    p.add_argument("--data", default="data")
    p.add_argument("--data-s3", default=data_prep.DEFAULT_S3,
                   help="prefixo S3 com Cars196.tar/CUB_200_2011.tgz; "
                        "'' desliga o pull (usa dados locais / download do trainer)")
    p.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", "gabomfim"))
    p.add_argument("--save-root", default="experiments_local")
    p.add_argument("--max-parallel", type=int, default=0,
                   help="jobs simultâneos (0 = auto pela VRAM livre)")
    p.add_argument("--per-job-gb", type=float, default=4.0,
                   help="VRAM estimada por job, p/ dimensionar o auto")
    p.add_argument("--workers", type=int, default=0,
                   help="dataloader workers por job (0 = auto p/ não saturar a CPU)")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY",
                                                            "gabomfim-unicamp"))
    p.add_argument("--wandb-project", default="graph-rkd")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default="online")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.worker:                       # caminho do subprocesso
        return run_worker(a.worker)

    cfg = plan.merged_config(
        trimmed=a.trimmed,
        datasets=a.datasets, teachers=a.teachers, methods=a.methods,
        objectives=a.objectives, norms=a.norms, n_list=a.n_list,
        lambda_grid=a.lambda_grid, seeds=a.seeds,
        student_epochs=a.student_epochs, search_epochs=a.search_epochs,
        teacher_epochs=a.teacher_epochs,
        gate_dataset=a.gate_dataset, gate_teacher=a.gate_teacher,
        wandb_entity=a.wandb_entity, wandb_project=a.wandb_project)
    cfg.update(data=a.data, wandb_mode=a.wandb_mode)

    jobs = plan.build_plan(cfg, a.phases)

    # garante os datasets usados (puxa do S3 e extrai se ainda não estão locais)
    datasets_used = sorted({s["dataset"] for s in jobs if s.get("dataset")})
    if a.data_s3:
        data_prep.ensure(a.data, datasets_used, a.data_s3, a.aws_profile)
    else:
        print("[data] --data-s3 vazio: usando dados locais / download do trainer")

    gpus = detect_gpus()
    if a.max_parallel > 0:
        max_parallel = a.max_parallel
        gpu_slots = [g[0] for g in gpus] if gpus else []
    else:
        max_parallel, gpu_slots = auto_parallel(gpus, a.per_job_gb)
    workers = a.workers or max(2, (os.cpu_count() or 8) // max(1, max_parallel))

    by_phase, by_kind = plan.summarize(jobs)
    gpu_desc = ", ".join(f"gpu{idx}:{free//1024}GB" for idx, free in gpus) or "CPU"
    print(f"Plano local: {len(jobs)} jobs | fases {by_phase} | kinds {by_kind}")
    print(f"GPUs: {gpu_desc} | paralelismo={max_parallel} "
          f"(~{a.per_job_gb}GB/job) | workers/job={workers}")
    print(f"W&B -> {a.wandb_entity}/{a.wandb_project} | data={a.data} "
          f"| save_root={a.save_root}\n")

    done, failed, skipped = schedule(jobs, cfg, a.save_root, max_parallel,
                                     gpu_slots, workers)
    print(f"\nConcluído: {len(done)} ok, {len(failed)} falhas, "
          f"{len(skipped)} pulados.")
    if failed:
        print("Falhas:", sorted(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
