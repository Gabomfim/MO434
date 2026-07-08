"""Plano de experimentos Graph-RKD como uma LISTA DE JOBS independentes.

Este módulo é PURO (sem AWS, sem torch): transforma a configuração da campanha
em uma lista de ``JobSpec`` — um por experimento — que o ``launch.py`` dispara
como jobs de treino do SageMaker EM PARALELO. Cada job roda um único
treinamento (``entry.py`` -> trainer) e loga no W&B do usuário.

Mapeia o EXPERIMENTS_EN.md:
  * §3 fases (0 smoke, 1 gate de λg, 2 normalização, 3 descritor, 4 objetivo,
    5 headline multi-seed) — cada fase gera seu subconjunto de jobs;
  * §4 os 5 alunos (triplet-only, +RKD-D 25, +RKD-A 50, +RKD-D+RKD-A 1:2,
    +Graph-RKD) — ver ``PHASE 5`` e ``CLASSIC``;
  * §2 invariantes: pesos clássicos fixos (I2), warmup rampando p/ λg (I6),
    seleção por val mAP@R (I4), λg tunado POR CONFIG num grid (I1), ≥3 seeds
    nos finais (I5), N ∈ {3,4,8,16,17} sem N=2 (§1).

Handoff de teacher entre jobs paralelos: cada teacher loga o checkpoint como
artefato W&B ``metric-<arch>-<dataset>:best`` (ver finetune_metric.py); os jobs
de aluno o puxam por referência de artefato (montada no launcher). Assim os
jobs não dependem de caminhos S3 acoplados.

Um ``JobSpec`` é um dict JSON-serializável:
  name        id lógico estável (base do wandb_id e do checkpoint_s3_uri)
  kind        "teacher" | "baseline" | "distill"
  phase       rótulo da fase
  dataset     cars196 | cub200
  arch        teacher arch (kind teacher: o que treina; distill: qual puxar)
  depends_on  name do job de teacher requerido (ou None)
  params      kwargs do trainer (sem wandb/data/save_dir/teacher path)
  wandb       {group, run_name, tags}
"""

# --- abreviações compactas p/ nomes de job (<=63 chars, [a-z0-9-]) ----------
DS_AB = {"cars196": "cars", "cub200": "cub"}
ARCH_AB = {"resnet18": "r18", "convnext_tiny": "cvt"}
MET_AB = {"profile": "prof", "mds": "mds"}
OBJ_AB = {"regression": "reg", "contrastive": "con"}
NORM_AB = {"per_graph": "pg", "minibatch": "mb", "none": "no", "hybrid": "hy"}

# §4 alunos clássicos: nome -> lista de (ratio_key, peso). Pesos fixos por I2
# (RKD-D=25, RKD-A=50, combinado 1:2). NÃO re-tunar (usa pesos validados dos
# autores p/ não handicapar os baselines).
CLASSIC = {
    "rkd_dist": [("dist_ratio", 25.0)],
    "rkd_angle": [("angle_ratio", 50.0)],
    "rkd_both": [("dist_ratio", 25.0), ("angle_ratio", 50.0)],
}

# λg default por objetivo qdo não varrido (contrastiva é ~O(1); regressão precisa
# de peso maior). Só um ponto de partida — o grid da fase 1 substitui isto (I1).
LAMBDA_DEFAULT = {"regression": 100.0, "contrastive": 1.0}

DEFAULTS = dict(
    datasets=["cars196", "cub200"],
    teachers=["resnet18", "convnext_tiny"],
    methods=["profile", "mds"],
    objectives=["regression", "contrastive"],
    norms=["per_graph", "minibatch", "none", "hybrid"],
    n_list=[3, 4, 8, 16, 17],                 # §1: sem N=2, sem N=1
    lambda_grid=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],   # §3 fase 1
    seeds=3,                                   # I5
    # orçamentos de época por fase
    teacher_epochs=60, student_epochs=120, search_epochs=30, smoke_epochs=2,
    batch=128, recall=[1, 2, 4, 8], select_metric="mapr", rel_warmup_frac=0.1,
    triplet_sample="distance", num_negatives=10, temperature=0.07,
    graph_rkd_sampling="log", graph_rkd_alpha=0.5, graph_rkd_gmax=64,
    amp=True,
    # fatia barata do gate (fase 1): Cars-196 + ResNet-18, profile, reg, minibatch, N=4
    gate_dataset="cars196", gate_teacher="resnet18", gate_method="profile",
    gate_objective="regression", gate_norm="minibatch", gate_nodes=4,
    gate_seeds=1,
    # fatia da ablação de normalização (fase 2)
    norm_datasets=["cars196", "cub200"], norm_nodes=[3, 4, 8],
    # config ESCOLHIDA p/ o headline (fase 5) — preencher a partir das fases 1-4
    headline_norm="minibatch", headline_method="profile",
    headline_objective="regression", headline_nodes=4, headline_lambda=100.0,
    # W&B (projeto único do usuário por padrão)
    wandb_entity=None, wandb_project="graph-rkd",
)


# Config ENXUTA derivada da iteração no Modal (gate + dev), p/ o run completo local:
#   * drop `hybrid` (pior norm consistentemente);
#   * λg pequeno — 3 pontos em vez de 6 (o sinal vive em λg baixo);
#   * N ∈ {3,4,8} — dropa 16/17 (MDS degenera >90% lá, e são caros);
#   * headline = mds/regression/per_graph/N4/λ0.01 (melhor no dev — reconfirmar no conv).
# Reduz ~pela metade as fases de busca (2/3/4); a fase 5 (headline) permanece.
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
    """Nome do artefato W&B logado pelo finetune_metric (art_name)."""
    return f"metric-{arch}-{ds}"


# --------------------------------------------------------------------------- #
# helpers de construção de JobSpec                                            #
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
        # baseline = triplet puro: sem termo relacional, logo sem rel_warmup_frac
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
    """λg -> string curta p/ nome de job (1e2, 1e-1, 5)."""
    if lam == 0:
        return "0"
    if lam >= 1 and float(lam).is_integer():
        return str(int(lam))
    return ("%g" % lam).replace(".", "p")


# --------------------------------------------------------------------------- #
# fases (§3)                                                                   #
# --------------------------------------------------------------------------- #
def phase_teachers(cfg):
    return [_teacher_spec(cfg, a, d) for d in cfg["datasets"] for a in cfg["teachers"]]


def phase0_smoke(cfg):
    """Fumaça: 1 baseline curto + 1 graph curto na fatia do gate, p/ validar o
    pipeline ponta-a-ponta (treina, avalia, loga per-term, puxa teacher)."""
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    e = cfg["smoke_epochs"]
    jobs = [_baseline_spec(cfg, ds, e, 0, "phase0", tag="smoke"),
            _graph_spec(cfg, ds, arch, cfg["gate_method"], cfg["gate_objective"],
                        cfg["gate_norm"], cfg["gate_nodes"],
                        LAMBDA_DEFAULT[cfg["gate_objective"]], e, 0, "phase0")]
    return jobs


def phase1_lambda_gate(cfg):
    """Gate H0: varre λg num range log largo numa única fatia barata + o piso
    triplet-only. Seleção posterior por val mAP@R (I1/I4)."""
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
    """Ablação de normalização (H2): varia norm ∈ {per_graph,minibatch,none,hybrid}
    com λg re-tunado por config (grid) numa fatia fixa."""
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
    """Caracterização do descritor (H3): profile vs mds em N∈n_list, ambos
    datasets/teachers, melhor norm, objetivo regressão, λg re-tunado por (desc,N)."""
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
    """Robustez ao objetivo (H4): overlay λg de regressão vs contrastiva numa
    fatia de ordem ativa (melhor norm+descritor)."""
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
    """Headline multi-seed (H1/H5): os 5 alunos na config escolhida, orçamento
    cheio, ≥3 seeds, ambos datasets e teachers. Inclui N=3 (H5 vs RKD-A)."""
    e = cfg["student_epochs"]
    jobs = []
    for ds in cfg["datasets"]:
        # aluno 1: triplet-only (piso) — sem teacher, uma vez por dataset/seed
        for seed in range(cfg["seeds"]):
            jobs.append(_baseline_spec(cfg, ds, e, seed, "phase5"))
        for arch in cfg["teachers"]:
            for seed in range(cfg["seeds"]):
                # alunos 2-4: clássicos (RKD-D, RKD-A, combinado)
                for cname in CLASSIC:
                    jobs.append(_classic_spec(cfg, ds, arch, cname, e, seed, "phase5"))
                # aluno 5: Graph-RKD na config escolhida
                jobs.append(_graph_spec(cfg, ds, arch, cfg["headline_method"],
                                        cfg["headline_objective"],
                                        cfg["headline_norm"], cfg["headline_nodes"],
                                        cfg["headline_lambda"], e, seed, "phase5"))
                # H5: N=3 arity-matched vs RKD-A (profile E mds) se ainda não coberto
                for method in cfg["methods"]:
                    if not (cfg["headline_nodes"] == 3
                            and method == cfg["headline_method"]):
                        jobs.append(_graph_spec(cfg, ds, arch, method, "regression",
                                                cfg["headline_norm"], 3,
                                                cfg["headline_lambda"], e, seed,
                                                "phase5"))
    return jobs


def phase_dev(cfg):
    """Grade BARATA p/ ITERAR o método no Modal dentro dos créditos grátis (reusa
    o teacher já treinado via artefato W&B — rode com `--only dev-` p/ pular o
    teacher). Compara descritor × objetivo × normalização num λg pequeno, schedule
    curto, 1 seed, na fatia cars196/r18/N4. ~12 jobs. Serve p/ decidir a config
    promissora ANTES do run completo local. Ajuste os eixos via os overrides do cfg."""
    ds, arch = cfg["gate_dataset"], cfg["gate_teacher"]
    e = cfg["search_epochs"]
    lam = cfg["lambda_grid"][0] if cfg["lambda_grid"] else 0.1
    jobs = []
    for method in cfg["methods"]:                 # profile, mds
        for obj in cfg["objectives"]:             # regression, contrastive
            for norm in cfg["norms"]:             # default: os 4 esquemas
                jobs.append(_graph_spec(cfg, ds, arch, method, obj, norm,
                                        cfg["gate_nodes"], lam, e, 0, "dev"))
    return jobs


def phase_conv(cfg):
    """Teste de CONVERGÊNCIA barato no Modal: piso triplet-only + as 2 melhores
    configs do dev, num schedule MAIS LONGO (student_epochs), p/ ver se o ganho do
    Graph-RKD persiste quando o aluno de fato treina (gate/dev de 30 ep ficam perto
    do piso). Reusa o teacher (rode com `--only conv`)."""
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
    """Concatena os jobs das fases pedidas, garantindo que os teachers de que
    dependem estejam incluídos (dedup por name)."""
    jobs, seen = [], set()

    def add(spec):
        if spec["name"] not in seen:
            if spec["phase"] not in spec["wandb"]["tags"]:   # tag p/ análise por fase
                spec["wandb"]["tags"] = list(spec["wandb"]["tags"]) + [spec["phase"]]
            seen.add(spec["name"])
            jobs.append(spec)

    # sempre inclui os teachers necessários primeiro (dependências)
    need_teachers = any(p != "teachers" for p in phases)
    if "teachers" in phases or need_teachers:
        for t in phase_teachers(cfg):
            add(t)
    for p in phases:
        if p == "teachers":
            continue
        for spec in PHASES[p](cfg):
            add(spec)

    # poda teachers não referenciados por nenhum job (evita treinar teacher à toa)
    used = {s["depends_on"] for s in jobs if s.get("depends_on")}
    if phases != ["teachers"] and "teachers" not in phases:
        jobs = [s for s in jobs if s["kind"] != "teacher" or s["name"] in used]
    return jobs


def summarize(jobs):
    """Contagem por fase e por kind, p/ o dry-run."""
    by_phase, by_kind = {}, {}
    for s in jobs:
        by_phase[s["phase"]] = by_phase.get(s["phase"], 0) + 1
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    return by_phase, by_kind
