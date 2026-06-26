"""Ledger de campanha: marca experimentos concluídos e cacheia seus resultados,
para que uma campanha interrompida seja retomada sem refazer trabalho.

Cada experimento é identificado pelo seu ``save_dir`` (único). Gravamos um
``result.json`` ao concluir; na re-execução, um experimento com ``result.json``
é pulado (e seu resultado é lido do disco, importante para a seleção de N).
"""

import json
import os

__all__ = ["is_done", "load_result", "mark_done"]


def _result_path(save_dir):
    return os.path.join(save_dir, "result.json")


def is_done(save_dir):
    """True se este experimento já foi concluído (tem result.json)."""
    return save_dir is not None and os.path.exists(_result_path(save_dir))


def load_result(save_dir):
    """Resultado cacheado (dict) ou None."""
    if not is_done(save_dir):
        return None
    with open(_result_path(save_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def mark_done(save_dir, result=None):
    """Grava result.json marcando conclusão (result deve ser JSON-serializável)."""
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    with open(_result_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(result if result is not None else {"done": True}, f)
