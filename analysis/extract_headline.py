#!/usr/bin/env python3
"""Extrai os resultados da Fase 5 (headline) do que já concluiu.
Gera 2 CSVs em analysis/:
  - headline_partial.csv       : tabela resumo test mAP@R por célula×método×seed
  - headline_raw_metrics.csv   : TODAS as métricas de teste (mAP@R, R_prec, R@1/2/4/8)
                                 por job concluído (o que os notebooks do Gabriel usam)
Graph-RKD nas 3 configs (mds-N4=headline; mds-N3/prof-N3=H5). Rode quantas vezes
quiser p/ atualizar conforme os jobs fecham. Só conta job concluído (sem processo vivo).
"""
import os, re, subprocess, collections

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RKD",
                    "experiments_local")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

try:
    ps = subprocess.run(["pgrep", "-af", "run_local.py --worker"],
                        capture_output=True, text=True).stdout
    running = set(re.findall(r"experiments_local/([^/]+)/_job\.json", ps))
except Exception:
    running = set()

MET = ["mAP@R", "R_precision", "recall@1", "recall@2", "recall@4", "recall@8"]

def get_metrics(d):
    lg = os.path.join(ROOT, d, "run.log")
    if not os.path.exists(lg):
        return None
    txt = open(lg, "rb").read().replace(b"\r", b"\n").decode("utf-8", "ignore")
    out = {}
    for k in MET:
        m = re.findall(r"final/test/%s\s+([0-9.]+)" % re.escape(k), txt)
        if m:
            out[k] = float(m[-1])
    # fallback p/ mAP@R via linha 'Done ... test mAP@R=X' (em %)
    if "mAP@R" not in out:
        m = re.findall(r"test mAP@R=([0-9.]+)", txt)
        if m:
            out["mAP@R"] = float(m[-1]) / 100.0
    return out or None

def classify(name):
    seed = int(name.rsplit("-s", 1)[-1])
    ds = "cub" if "cub" in name else "cars"
    if name.startswith("baseline-"):
        return ds, "*", "triplet-only", seed
    tc = "cvt" if "cvt" in name else "r18"
    if "rkddist" in name:  return ds, tc, "RKD-D", seed
    if "rkdangle" in name: return ds, tc, "RKD-A", seed
    if "rkdboth" in name:  return ds, tc, "RKD-D+A", seed
    md = re.search(r"-(mds|prof)-reg-pg-N(\d+)-", name)
    if md:
        return ds, tc, "Graph-RKD/%s-N%s" % (md.group(1), md.group(2)), seed
    return None

summary = collections.defaultdict(dict)      # (ds,tc,method) -> {seed: mAP@R%}
baselines = collections.defaultdict(dict)     # ds -> {seed: mAP@R%}
raw_rows = ["dataset,teacher,method,seed,test_mAPR,test_Rprec,test_R1,test_R2,test_R4,test_R8,job"]

for d in sorted(os.listdir(ROOT)):
    if not os.path.isdir(os.path.join(ROOT, d)):
        continue
    if d.startswith("teacher-") or re.match(r"^phase[234]-", d) or d in running:
        continue
    c = classify(d)
    if not c:
        continue
    ds, tc, method, seed = c
    m = get_metrics(d)
    if not m or "mAP@R" not in m:
        continue
    mapr_pct = m["mAP@R"] * 100.0
    (baselines[ds] if method == "triplet-only" else summary[(ds, tc, method)])[seed] = mapr_pct
    g = lambda k: ("%.4f" % m[k]) if k in m else ""
    raw_rows.append(",".join([ds, tc, method, str(seed), g("mAP@R"), g("R_precision"),
                              g("recall@1"), g("recall@2"), g("recall@4"), g("recall@8"), d]))

METHOD_ORDER = ["triplet-only", "Graph-RKD/mds-N4", "Graph-RKD/mds-N3",
                "Graph-RKD/prof-N3", "RKD-D", "RKD-A", "RKD-D+A"]
CELLS = [("cars", "r18"), ("cars", "cvt"), ("cub", "r18"), ("cub", "cvt")]

def fmt(sm):
    vals = [sm.get(s) for s in (0, 1, 2)]
    cells = [("%.2f" % v) if v is not None else "-" for v in vals]
    got = [v for v in vals if v is not None]
    mean = ("%.2f" % (sum(got) / len(got))) if got else "-"
    return cells, mean

sum_rows = ["cell,method,s0,s1,s2,mean_test_mAPR_pct"]
for ds, tc in CELLS:
    print("\n=== %s / %s (test mAP@R %%) ===" % (ds.upper(), tc))
    print("%-18s  %6s %6s %6s | %6s" % ("method", "s0", "s1", "s2", "mean"))
    for method in METHOD_ORDER:
        sm = baselines.get(ds, {}) if method == "triplet-only" else summary.get((ds, tc, method), {})
        if not sm:
            continue
        cells, mean = fmt(sm)
        print("%-18s  %6s %6s %6s | %6s" % (method, cells[0], cells[1], cells[2], mean))
        sum_rows.append("%s-%s,%s,%s,%s,%s,%s" % (ds, tc, method, cells[0], cells[1], cells[2], mean))

open(os.path.join(OUTDIR, "headline_partial.csv"), "w").write("\n".join(sum_rows) + "\n")
open(os.path.join(OUTDIR, "headline_raw_metrics.csv"), "w").write("\n".join(raw_rows) + "\n")
print("\nCSVs: analysis/headline_partial.csv  +  analysis/headline_raw_metrics.csv")
print("jobs headline concluídos processados:", len(raw_rows) - 1)
