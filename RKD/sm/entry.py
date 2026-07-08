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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # RKD/sm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # RKD
import data_prep  # noqa: E402  (mesma camada de cache do local/Modal)

# nomes de checkpoint "last" por trainer (p/ resume)
LAST_NAME = {"teacher": "last.pth", "baseline": "baseline_last.pth",
             "distill": "student_last.pth"}


def resolve_data_dir():
    """Dir de CACHE dos datasets (mesma camada `data_prep` do local/Modal).

    Preferência: um canal de input já montado (``SM_CHANNEL_DATA`` — dados
    pré-staged), senão ``/opt/ml/checkpoints`` (sincronizado com S3, sobrevive a
    resume/spot => cache entre re-execuções do mesmo job), senão ``/opt/ml/data``.
    O download em si é feito por ``data_prep.ensure`` (HTTPS público, sem creds)."""
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
    cache = resolve_data_dir()
    ds = params.get("dataset")
    if ds:                                          # baixa (público) + cacheia
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
