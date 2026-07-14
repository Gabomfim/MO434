"""Garante que os datasets estejam EXTRAÍDOS localmente, puxando os arquivos do
S3 quando necessário. Compartilhado pelo runner local e pelo backend Modal.

Layout no S3 (mesma região do treino; melhor acesso p/ SageMaker/Modal/etc.):
    s3://<bucket>/<prefix>/Cars196.tar
    s3://<bucket>/<prefix>/CUB_200_2011.tgz

O prefixo S3 é PÚBLICO (read-only em graph-rkd/data/*), então o download usa
HTTPS e **não precisa de credenciais AWS** — o colega roda de qualquer máquina.
`ensure(...)` é idempotente e CACHEIA: se o marcador do dataset já existe em
``data_dir`` não baixa nada (basta apontar sempre p/ o mesmo ``data_dir`` — ou,
no Modal, um Volume persistente — p/ baixar uma única vez). Sem ``s3_prefix``,
deixa o trainer tentar o download via torchvision."""

import os
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

ARCHIVE = {"cars196": "Cars196.tar", "cub200": "CUB_200_2011.tgz"}
MARKER = {"cars196": os.path.join("Cars196", "cars_annos.mat"),
          "cub200": os.path.join("CUB_200_2011", "images.txt")}
DEFAULT_S3 = "s3://graph-rkd-832271495954/graph-rkd/data"


def has_dataset(data_dir, dataset):
    return os.path.exists(os.path.join(data_dir, MARKER[dataset]))


def s3_to_https(s3_prefix):
    """s3://bucket/key... -> https://bucket.s3.amazonaws.com/key... (acesso público)."""
    rest = s3_prefix[len("s3://"):] if s3_prefix.startswith("s3://") else s3_prefix
    bucket, _, key = rest.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}".rstrip("/")


def _download_https(url, dst, attempts=25, progress=False):
    """Baixa ``url`` -> ``dst`` com RESUME (Range) e retries/backoff. Sem creds.
    ``progress=True`` mostra uma barra tqdm (para uso interativo/notebook); fica
    desligada por padrão para não poluir logs de treino (SageMaker/Modal/local)."""
    bar_cls = None
    if progress:
        try:
            from tqdm.auto import tqdm as bar_cls  # notebook ou terminal
        except ImportError:
            bar_cls = None
    last = None
    for i in range(1, attempts + 1):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resume = bool(have) and r.status == 206
                mode = "ab" if resume else "wb"
                clen = r.getheader("Content-Length")
                total = int(clen) + (have if resume else 0) if clen else None
                bar = (bar_cls(total=total, initial=have if resume else 0,
                               unit="B", unit_scale=True, unit_divisor=1024,
                               desc=os.path.basename(dst), leave=False)
                       if bar_cls else None)
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
                if bar is not None:
                    bar.close()
            return
        except (urllib.error.URLError, OSError) as e:  # noqa: PERF203
            last = e
            if i == attempts:
                break
            time.sleep(min(30, 3 * i))
    raise RuntimeError(f"download HTTPS falhou ({url}): {last}")


def _s3_cp(uri, dst, profile=None):
    """Fallback com credenciais (se o bucket voltar a ser privado)."""
    env = dict(os.environ, AWS_MAX_ATTEMPTS="15", AWS_RETRY_MODE="standard")
    if profile:
        env["AWS_PROFILE"] = profile
    subprocess.run(["aws", "s3", "cp", uri, dst, "--no-progress"], check=True, env=env)


def ensure(data_dir, datasets, s3_prefix=DEFAULT_S3, profile=None,
           keep_archive=False, progress=False):
    """Garante Cars196/ e/ou CUB_200_2011/ EXTRAÍDOS em ``data_dir`` (cacheia:
    pula o que já está lá). Baixa por HTTPS público; se falhar e houver creds,
    tenta ``aws s3 cp``. Retorna a lista de datasets prontos."""
    os.makedirs(data_dir, exist_ok=True)
    https_base = s3_to_https(s3_prefix) if s3_prefix else None
    ready = []
    for ds in datasets:
        if has_dataset(data_dir, ds):
            print(f"[data] {ds}: já extraído em {data_dir} (cache hit)")
            ready.append(ds)
            continue
        if not s3_prefix:
            print(f"[data] {ds}: sem s3_prefix; trainer tentará baixar via torchvision")
            continue
        arch = ARCHIVE[ds]
        local = os.path.join(data_dir, arch)
        url = f"{https_base}/{arch}"
        print(f"[data] {ds}: baixando (público) {url} -> {local}", flush=True)
        try:
            _download_https(url, local, progress=progress)
        except Exception as e:  # noqa: BLE001 - fallback com creds se privado
            print(f"[data] {ds}: HTTPS falhou ({e}); tentando aws s3 cp")
            _s3_cp(f"{s3_prefix.rstrip('/')}/{arch}", local, profile)
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
