"""Entrypoint executado DENTRO do job de treino do SageMaker.

O SageMaker invoca este script (entry_point) com os hiperparâmetros do estimator
convertidos em ``--flag valor``. Passamos UM único hiperparâmetro ``--spec`` com
o JobSpec serializado (JSON) para evitar problemas de listas/aspas. Este script:

  1. lê o ``--spec`` (kind + params do trainer, já com wandb_* e teacher);
  2. resolve, a partir do ambiente do SageMaker, o diretório de DADOS (canal de
     input montado, ou baixa via torchvision) e o de CHECKPOINT (resumível);
  3. despacha para o trainer certo (finetune_metric / train_metric_baseline /
     distill_metric) via ``run_with_params``.

cwd no container = source_dir (RKD/), então ``import distill_metric`` etc. funciona.
W&B: precisa de ``WANDB_API_KEY`` no ambiente (o launcher repassa).
"""

import argparse
import glob
import json
import os
import sys
import tarfile

# nomes de checkpoint "last" por trainer (p/ resume)
LAST_NAME = {"teacher": "last.pth", "baseline": "baseline_last.pth",
             "distill": "student_last.pth"}

# marcadores que indicam dataset JÁ extraído sob um diretório de dados
MARKERS = [os.path.join("Cars196", "cars_annos.mat"),
           os.path.join("CUB_200_2011", "images.txt")]


def _has_dataset(d):
    return any(os.path.exists(os.path.join(d, m)) for m in MARKERS)


def _extract_archives(src, dst):
    """Extrai todos os .tar/.tgz/.tar.gz de ``src`` em ``dst`` (idempotente)."""
    archives = (glob.glob(os.path.join(src, "*.tar")) +
                glob.glob(os.path.join(src, "*.tgz")) +
                glob.glob(os.path.join(src, "*.tar.gz")))
    os.makedirs(dst, exist_ok=True)
    for a in archives:
        print(f"[entry] extraindo {a} -> {dst}", flush=True)
        with tarfile.open(a) as t:
            t.extractall(dst)
    return archives


def resolve_data_dir():
    """Diretório com datasets EXTRAÍDOS (Cars196/, CUB_200_2011/).

    Aceita o canal de input montado como (a) árvore já extraída, ou (b) ARQUIVOS
    (Cars196.tar / CUB_200_2011.tgz) — o modo recomendado p/ S3/SageMaker e para
    plataformas de GPU alugada, pois são poucos objetos grandes (download rápido).
    No caso (b) extrai uma vez num dir gravável. Sem canal, cai p/ download via
    torchvision (URLs upstream podem falhar; ver README)."""
    channel = None
    for env in ("SM_CHANNEL_DATA", "DATA_DIR"):
        v = os.environ.get(env)
        if v and os.path.isdir(v):
            channel = v
            break
    if channel is None and os.path.isdir("/opt/ml/input/data/data"):
        channel = "/opt/ml/input/data/data"

    if channel and _has_dataset(channel):
        return channel                              # (a) já extraído
    if channel and _extract_archives(channel, "/opt/ml/data"):
        return "/opt/ml/data"                       # (b) arquivos -> extrai
    fallback = "/opt/ml/data"                       # (c) download em job
    os.makedirs(fallback, exist_ok=True)
    return fallback


def resolve_ckpt_dir():
    """Diretório de checkpoint resumível: usa /opt/ml/checkpoints (sincronizado
    com checkpoint_s3_uri, sobrevive a spot/retry) se existir; senão o model dir."""
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
        raise SystemExit(f"kind desconhecido: {kind}")
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
    params["data"] = resolve_data_dir()
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
