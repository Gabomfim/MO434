"""Garante que os datasets estejam EXTRAÍDOS localmente, puxando os arquivos do
S3 quando necessário. Compartilhado pelo runner local e pelo backend Modal.

Layout no S3 (mesma região do treino; melhor acesso p/ SageMaker/Modal/etc.):
    s3://<bucket>/<prefix>/Cars196.tar
    s3://<bucket>/<prefix>/CUB_200_2011.tgz

`ensure(...)` é idempotente: se o marcador do dataset já existe em ``data_dir``
não faz nada; senão baixa o arquivo do S3 e extrai. Sem ``s3_prefix``, deixa o
trainer tentar o download via torchvision (URLs upstream podem falhar)."""

import os
import subprocess
import tarfile

ARCHIVE = {"cars196": "Cars196.tar", "cub200": "CUB_200_2011.tgz"}
MARKER = {"cars196": os.path.join("Cars196", "cars_annos.mat"),
          "cub200": os.path.join("CUB_200_2011", "images.txt")}
DEFAULT_S3 = "s3://graph-rkd-832271495954/graph-rkd/data"


def has_dataset(data_dir, dataset):
    return os.path.exists(os.path.join(data_dir, MARKER[dataset]))


def _s3_cp(uri, dst, profile=None):
    env = dict(os.environ, AWS_MAX_ATTEMPTS="15", AWS_RETRY_MODE="standard")
    if profile:
        env["AWS_PROFILE"] = profile
    subprocess.run(["aws", "s3", "cp", uri, dst, "--no-progress"],
                   check=True, env=env)


def ensure(data_dir, datasets, s3_prefix=DEFAULT_S3, profile=None,
           keep_archive=False):
    """Garante Cars196/ e/ou CUB_200_2011/ extraídos em ``data_dir``.

    Retorna a lista de datasets prontos. Levanta se um S3 cp/extração falhar.
    """
    os.makedirs(data_dir, exist_ok=True)
    ready = []
    for ds in datasets:
        if has_dataset(data_dir, ds):
            ready.append(ds)
            continue
        if not s3_prefix:
            print(f"[data] {ds}: sem s3_prefix; trainer tentará baixar via torchvision")
            continue
        arch = ARCHIVE[ds]
        local = os.path.join(data_dir, arch)
        uri = f"{s3_prefix.rstrip('/')}/{arch}"
        print(f"[data] {ds}: baixando {uri} -> {local}", flush=True)
        _s3_cp(uri, local, profile)
        print(f"[data] {ds}: extraindo {arch}", flush=True)
        with tarfile.open(local) as t:
            t.extractall(data_dir)
        if not keep_archive:
            os.remove(local)
        if not has_dataset(data_dir, ds):
            raise RuntimeError(f"{ds}: marcador ausente após extrair {arch}")
        ready.append(ds)
    return ready


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prepara datasets a partir do S3")
    p.add_argument("--data", default="data")
    p.add_argument("--datasets", nargs="+", default=["cars196", "cub200"])
    p.add_argument("--s3-prefix", default=DEFAULT_S3)
    p.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE"))
    a = p.parse_args()
    print("prontos:", ensure(a.data, a.datasets, a.s3_prefix, a.aws_profile))
