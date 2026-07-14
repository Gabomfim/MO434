"""Graph-RKD experiment plan as an independent LIST OF JOBS.

This module is PURE (no AWS, no torch): it turns the campaign configuration
into a list of ``JobSpec`` — one per experiment — that ``launch.py`` fires
as SageMaker training jobs IN PARALLEL. Each job runs a single
training (``entry.py`` -> trainer) and logs to the user's W&B.

Maps EXPERIMENTS_EN.md:
  * §3 phases (0 smoke, 1 λg gate, 2 normalization, 3 descriptor, 4 objective,
    5 multi-seed headline) — each phase generates its subset of jobs;
  * §4 the 5 students (triplet-only, +RKD-D 25, +RKD-A 50, +RKD-D+RKD-A 1:2,
    +Graph-RKD) — see ``PHASE 5`` and ``CLASSIC``;
  * §2 invariants: fixed classic weights (I2), warmup ramping to λg (I6),
    selection by val mAP@R (I4), λg tuned PER CONFIG on a grid (I1), ≥3 seeds
    in the finals (I5), N ∈ {3,4,8,16,17} without N=2 (§1).

Teacher handoff between parallel jobs: each teacher logs the checkpoint as
a W&B artifact ``metric-<arch>-<dataset>:best`` (see finetune_metric.py); the
student jobs pull it by artifact reference (assembled in the launcher). This way
the jobs do not depend on coupled S3 paths.

A ``JobSpec`` is a JSON-serializable dict:
  name        stable logical id (base of wandb_id and checkpoint_s3_uri)
  kind        "teacher" | "baseline" | "distill"
  phase       phase label
  dataset     cars196 | cub200
  arch        teacher arch (kind teacher: what it trains; distill: which to pull)
  depends_on  name of the required teacher job (or None)
  params      trainer kwargs (without wandb/data/save_dir/teacher path)
  wandb       {group, run_name, tags}
"""

# --- compact abbreviations for job names (<=63 chars, [a-z0-9-]) ------------
DS_AB = {"cars196": "cars", "cub200": "cub"}
ARCH_AB = {"resnet18": "r18", "convnext_tiny": "cvt"}
MET_AB = {"profile": "prof", "mds": "mds"}
OBJ_AB = {"regression": "reg", "contrastive": "con"}
NORM_AB = {"per_graph": "pg", "minibatch": "mb", "none": "no", "hybrid": "hy"}

# §4 classic students: name -> list of (ratio_key, weight). Weights fixed by I2
# (RKD-D=25, RKD-A=50, combined 1:2). Do NOT re-tune (use the authors' validated
# weights so as not to handicap the baselines).
CLASSIC = {
    "rkd_dist": [("dist_ratio", 25.0)],
    "rkd_angle": [("angle_ratio", 50.0)],
    "rkd_both": [("dist_ratio", 25.0), ("angle_ratio", 50.0)],
}

# default λg per objective when not swept (contrastive is ~O(1); regression needs
# a larger weight). Just a starting point — the phase 1 grid replaces this (I1).
LAMBDA_DEFAULT = {"regression": 100.0, "contrastive": 1.0}

DEFAULTS = dict(
    datasets=["cars196", "cub200"],
    teachers=["resnet18", "convnext_tiny"],
    methods=["profile", "mds"],
    objectives=["regression", "contrastive"],
    norms=["per_graph", "minibatch", "none", "hybrid"],
    n_list=[3, 4, 8, 16, 17],                 # §1: no N=2, no N=1
    lambda_grid=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],   # §3 phase 1
    seeds=3,                                   # I5
    # per-phase epoch budgets. search_epochs bumped 30->60: the 30-ep runs
    # sat near the floor (λg/norm/descriptor selection unreliable).
    teacher_epochs=60, student_epochs=120, search_epochs=60, smoke_epochs=2,
    batch=128, recall=[1, 2, 4, 8], select_metric="mapr", rel_warmup_frac=0.1,
    triplet_sample="distance", num_negatives=10, temperature=0.07,
    graph_rkd_sampling="log", graph_rkd_alpha=0.5, graph_rkd_gmax=64,
    amp=True,
    # cheap gate slice (phase 1): Cars-196 + ResNet-18, profile, reg, minibatch, N=4
    gate_dataset="cars196", gate_teacher="resnet18", gate_method="profile",
    gate_objective="regression", gate_norm="minibatch", gate_nodes=4,
    gate_seeds=1,
    # normalization ablation slice (phase 2)
    norm_datasets=["cars196", "cub200"], norm_nodes=[3, 4, 8],
    # CHOSEN config for the headline (phase 5) — fill in from phases 1-4
    headline_norm="minibatch", headline_method="profile",
    headline_objective="regression", headline_nodes=4, headline_lambda=100.0,
    # W&B (single user project by default)
    wandb_entity=None, wandb_project="graph-rkd",
)


# TRIMMED config derived from Modal iteration (gate + dev), for the full local run:
#   * drop `hybrid` (consistently the worst norm);
#   * small λg — 3 points instead of 6 (the signal lives at low λg);
#   * N ∈ {3,4,8} — drop 16/17 (MDS degenerates >90% there, and they are costly);
#   * headline = mds/regression/per_graph/N4/λ0.01 (best on dev — reconfirm on conv).
# Roughly halves the search phases (2/3/4); phase 5 (headline) stays.
TRIMMED = dict(
    norms=["per_graph", "minibatch", "none"],
    lambda_grid=[0.01, 0.1, 1.0],
    n_list=[3, 4, 8],
    norm_nodes=[3, 4, 8],
    headline_method="mds", headline_norm="per_graph",
    headline_objective="regression", headline_nodes=4, headline_lambda=0.01,
)


def merged_config(trimmed=False, **overrides):
    cfg = dict(DEFAULTS)
    if trimmed:
        cfg.update(TRIMMED)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def teacher_name(arch, ds):
    return f"teacher-{ARCH_AB[arch]}-{DS_AB[ds]}"


def teacher_artifact_name(arch, ds):
    """Name of the W&B artifact logged by finetune_metric (art_name)."""
    return f"metric-{arch}-{ds}"


# --------------------------------------------------------------------------- #
# JobSpec construction helpers                                                #
# --------------------------------------------------------------------------- #
def _common_train(cfg, epochs):
    return dict(
        data="data", batch=cfg["batch"], epochs=epochs,
        recall=list(cfg["recall"]), select_metric=cfg["select_metric"],
        triplet_sample=cfg["triplet_sample"], amp=bool(cfg["amp"]),
    )


def _teacher_spec(cfg, arch, ds):
    return {
        "name": teacher_name(arch, ds), "kind": "teacher", "phase": "teachers",
        "dataset": ds, "arch": arch, "depends_on": None,
        "params": {**_common_train(cfg, cfg["teacher_epochs"]),
                   "arch": arch, "dataset": ds, "seed": 0},
        "wandb": {"group": f"teacher-{arch}-{ds}",
                  "run_name": f"teacher-{arch}-{ds}",
                  "tags": ["teacher", "metric", arch, ds]},
    }


def _baseline_spec(cfg, ds, epochs, seed, phase, tag="baseline"):
    name = f"{tag}-{DS_AB[ds]}-s{seed}"
    return {
        "name": name, "kind": "baseline", "phase": phase,
        "dataset": ds, "arch": None, "depends_on": None,
        # baseline = pure triplet: no relational term, hence no rel_warmup_frac
        "params": {**_common_train(cfg, epochs), "dataset": ds, "seed": seed},
        "wandb": {"group": f"baseline-{ds}", "run_name": name,
                  "tags": [tag, "triplet-only", "metric", ds]},
    }


def _classic_spec(cfg, ds, arch, classic_name, epochs, seed, phase):
    ratios = {"dist_ratio": 0.0, "angle_ratio": 0.0}
    for k, v in CLASSIC[classic_name]:
        ratios[k] = v
    name = f"{phase}-{classic_name.replace('_','')}-{ARCH_AB[arch]}-{DS_AB[ds]}-s{seed}"
    return {
        "name": name, "kind": "distill", "phase": phase,
        "dataset": ds, "arch": arch, "depends_on": teacher_name(arch, ds),
        "params": {**_common_train(cfg, epochs),
                   "dataset": ds, "teacher_arch": arch, "seed": seed,
                   "graph_rkd_mode": "off", "triplet_ratio": 1.0,
                   "rel_warmup_frac": cfg["rel_warmup_frac"], **ratios},
        "wandb": {"group": f"classic-{arch}-{ds}", "run_name": name,
                  "tags": ["classic", classic_name, "metric", arch, ds]},
    }


def _graph_spec(cfg, ds, arch, method, objective, norm, nodes, lam, epochs,
                seed, phase):
    name = (f"{phase}-{DS_AB[ds]}-{ARCH_AB[arch]}-{MET_AB[method]}-"
            f"{OBJ_AB[objective]}-{NORM_AB[norm]}-N{nodes}-lg{_fmt_lam(lam)}-s{seed}")
    params = {
        **_common_train(cfg, epochs),
        "dataset": ds, "teacher_arch": arch, "seed": seed,
        "triplet_ratio": 1.0, "rel_warmup_frac": cfg["rel_warmup_frac"],
        "graph_rkd_mode": objective, "graph_rkd_method": method,
        "graph_rkd_norm": norm, "graph_rkd_nodes": nodes,
        "graph_rkd_ratio": lam,
        "graph_rkd_sampling": cfg["graph_rkd_sampling"],
        "graph_rkd_alpha": cfg["graph_rkd_alpha"],
        "graph_rkd_gmax": cfg["graph_rkd_gmax"],
    }
    if objective == "contrastive":
        params["num_negatives"] = cfg["num_negatives"]
        params["temperature"] = cfg["temperature"]
    return {
        "name": name, "kind": "distill", "phase": phase,
        "dataset": ds, "arch": arch, "depends_on": teacher_name(arch, ds),
        "params": params,
        "wandb": {"group": f"graph-{arch}-{ds}", "run_name": name,
                  "tags": ["graph-rkd", method, objective, norm, f"N{nodes}",
                           "metric", arch, ds]},
    }


def _fmt_lam(lam):
    """λg -> short string for job name (1e2, 1e-1, 5)."""
    if lam == 0:
        return "0"
    if lam >= 1 and float(lam).is_integer():
        return str(int(lam))
    return ("%g" % lam).replace(".", "p")


# --------------------------------------------------------------------------- #
# phases (§3)                                                                  #
# --------------------------------------------------------------------------- #
def phase_teachers(cfg):
    return [_teacher_spec(cfg, a, d) for d in cfg["datasets"] for a in cfg["teachers"]]


def phase0_smoke(cfg):
    """Smoke: 1 short baseline + 1 short graph on the gate slice, to validate the
    end-to-end pipeline (trains, evaluates, logs per-term, pulls teacher)."""
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    e = cfg["smoke_epochs"]
    jobs = [_baseline_spec(cfg, ds, e, 0, "phase0", tag="smoke"),
            _graph_spec(cfg, ds, arch, cfg["gate_method"], cfg["gate_objective"],
                        cfg["gate_norm"], cfg["gate_nodes"],
                        LAMBDA_DEFAULT[cfg["gate_objective"]], e, 0, "phase0")]
    return jobs


def phase1_lambda_gate(cfg):
    """Gate H0: sweeps λg over a wide log range on a single cheap slice + the
    triplet-only floor. Later selection by val mAP@R (I1/I4)."""
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    e = cfg["search_epochs"]
    jobs = []
    for seed in range(cfg["gate_seeds"]):
        jobs.append(_baseline_spec(cfg, ds, e, seed, "phase1", tag="floor"))
        for lam in cfg["lambda_grid"]:
            jobs.append(_graph_spec(cfg, ds, arch, cfg["gate_method"],
                                    cfg["gate_objective"], cfg["gate_norm"],
                                    cfg["gate_nodes"], lam, e, seed, "phase1"))
    return jobs


def phase2_norm(cfg):
    """Normalization ablation (H2): varies norm ∈ {per_graph,minibatch,none,hybrid}
    with λg re-tuned per config (grid) on a fixed slice."""
    e = cfg["search_epochs"]
    jobs = []
    for ds in cfg["norm_datasets"]:
        arch = cfg["gate_teacher"]
        for norm in cfg["norms"]:
            for N in cfg["norm_nodes"]:
                for lam in cfg["lambda_grid"]:
                    jobs.append(_graph_spec(cfg, ds, arch, cfg["gate_method"],
                                            cfg["gate_objective"], norm, N, lam,
                                            e, 0, "phase2"))
    return jobs


def phase3_descriptor(cfg):
    """Descriptor characterization (H3): profile vs mds over N∈n_list, both
    datasets/teachers, best norm, regression objective, λg re-tuned per (desc,N)."""
    e = cfg["search_epochs"]
    jobs = []
    for ds in cfg["datasets"]:
        for arch in cfg["teachers"]:
            for method in cfg["methods"]:
                for N in cfg["n_list"]:
                    for lam in cfg["lambda_grid"]:
                        jobs.append(_graph_spec(cfg, ds, arch, method,
                                                "regression", cfg["headline_norm"],
                                                N, lam, e, 0, "phase3"))
    return jobs


def phase4_objective(cfg):
    """Objective robustness (H4): λg overlay of regression vs contrastive on a
    slice with active order (best norm+descriptor)."""
    e = cfg["search_epochs"]
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    jobs = []
    for objective in ["regression", "contrastive"]:
        for lam in cfg["lambda_grid"]:
            jobs.append(_graph_spec(cfg, ds, arch, cfg["headline_method"],
                                    objective, cfg["headline_norm"],
                                    cfg["headline_nodes"], lam, e, 0, "phase4"))
    return jobs


def phase5_headline(cfg):
    """Multi-seed headline (H1/H5): the 5 students in the chosen config, full
    budget, ≥3 seeds, both datasets and teachers. Includes N=3 (H5 vs RKD-A)."""
    e = cfg["student_epochs"]
    jobs = []
    for ds in cfg["datasets"]:
        # student 1: triplet-only (floor) — no teacher, once per dataset/seed
        for seed in range(cfg["seeds"]):
            jobs.append(_baseline_spec(cfg, ds, e, seed, "phase5"))
        for arch in cfg["teachers"]:
            for seed in range(cfg["seeds"]):
                # students 2-4: classic (RKD-D, RKD-A, combined)
                for cname in CLASSIC:
                    jobs.append(_classic_spec(cfg, ds, arch, cname, e, seed, "phase5"))
                # student 5: Graph-RKD in the chosen config
                jobs.append(_graph_spec(cfg, ds, arch, cfg["headline_method"],
                                        cfg["headline_objective"],
                                        cfg["headline_norm"], cfg["headline_nodes"],
                                        cfg["headline_lambda"], e, seed, "phase5"))
                # H5: N=3 arity-matched vs RKD-A (profile AND mds) if not yet covered
                for method in cfg["methods"]:
                    if not (cfg["headline_nodes"] == 3
                            and method == cfg["headline_method"]):
                        jobs.append(_graph_spec(cfg, ds, arch, method, "regression",
                                                cfg["headline_norm"], 3,
                                                cfg["headline_lambda"], e, seed,
                                                "phase5"))
    return jobs


def phase_dev(cfg):
    """CHEAP grid to ITERATE the method on Modal within the free credits (reuses
    the already-trained teacher via W&B artifact — run with `--only dev-` to skip the
    teacher). Compares descriptor × objective × normalization at a small λg, short
    schedule, 1 seed, on the cars196/r18/N4 slice. ~12 jobs. Used to decide the
    promising config BEFORE the full local run. Adjust the axes via the cfg overrides."""
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    e = cfg["search_epochs"]
    lam = cfg["lambda_grid"][0] if cfg["lambda_grid"] else 0.1
    jobs = []
    for method in cfg["methods"]:                 # profile, mds
        for obj in cfg["objectives"]:             # regression, contrastive
            for norm in cfg["norms"]:             # default: the 4 schemes
                jobs.append(_graph_spec(cfg, ds, arch, method, obj, norm,
                                        cfg["gate_nodes"], lam, e, 0, "dev"))
    return jobs


def phase_conv(cfg):
    """Cheap CONVERGENCE test on Modal: triplet-only floor + the 2 best dev
    configs, on a LONGER schedule (student_epochs), to see whether the Graph-RKD
    gain persists once the student actually trains (30-ep gate/dev sit near the
    floor). Reuses the teacher (run with `--only conv`)."""
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    e = cfg["student_epochs"]
    lam = 0.01
    jobs = [_baseline_spec(cfg, ds, e, 0, "conv", tag="convfloor")]
    for method, norm in [("mds", "per_graph"), ("profile", "minibatch")]:
        jobs.append(_graph_spec(cfg, ds, arch, method, "regression", norm,
                                cfg["gate_nodes"], lam, e, 0, "conv"))
    return jobs


PHASES = {
    "teachers": phase_teachers,
    "dev": phase_dev,
    "conv": phase_conv,
    "phase0": phase0_smoke,
    "phase1": phase1_lambda_gate,
    "phase2": phase2_norm,
    "phase3": phase3_descriptor,
    "phase4": phase4_objective,
    "phase5": phase5_headline,
}


def build_plan(cfg, phases):
    """Concatenates the jobs of the requested phases, ensuring the teachers they
    depend on are included (dedup by name)."""
    jobs, seen = [], set()

    def add(spec):
        if spec["name"] not in seen:
            if spec["phase"] not in spec["wandb"]["tags"]:   # tag for per-phase analysis
                spec["wandb"]["tags"] = list(spec["wandb"]["tags"]) + [spec["phase"]]
            seen.add(spec["name"])
            jobs.append(spec)

    # always include the needed teachers first (dependencies)
    need_teachers = any(p != "teachers" for p in phases)
    if "teachers" in phases or need_teachers:
        for t in phase_teachers(cfg):
            add(t)
    for p in phases:
        if p == "teachers":
            continue
        for spec in PHASES[p](cfg):
            add(spec)

    # prune teachers not referenced by any job (avoids training a teacher for nothing)
    used = {s["depends_on"] for s in jobs if s.get("depends_on")}
    if phases != ["teachers"] and "teachers" not in phases:
        jobs = [s for s in jobs if s["kind"] != "teacher" or s["name"] in used]
    return jobs


def summarize(jobs):
    """Count per phase and per kind, for the dry-run."""
    by_phase, by_kind = {}, {}
    for s in jobs:
        by_phase[s["phase"]] = by_phase.get(s["phase"], 0) + 1
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    return by_phase, by_kind
