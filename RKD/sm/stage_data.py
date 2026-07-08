"""Prepara (baixa se preciso) e ENVIA os datasets para o S3, de forma robusta.

Faz o caminho source -> local -> S3 com resume, retries com backoff, verificação
de md5 e idempotência (pula o que já está no S3 com o tamanho certo). É o inverso
de ``data_prep.py`` (que faz S3 -> local nos jobs de treino).

Layout de destino (mesma região do treino; melhor acesso p/ SageMaker/Modal):
    s3://<bucket>/<prefix>/Cars196.tar
    s3://<bucket>/<prefix>/CUB_200_2011.tgz

Observações:
  * Cars-196 NÃO tem fonte pública viva (Stanford caiu). Forneça o .tar local
    (``--archive-dir``); sem ele e sem estar no S3, o script reporta erro claro.
  * CUB-200 baixa do mirror do Caltech Data (URL do repo antiga está 404).

Exemplos:
    python sm/stage_data.py                      # ambos, bucket/prefix default
    python sm/stage_data.py --datasets cub200    # só CUB
    python sm/stage_data.py --check              # só diagnóstico (não baixa/envia)
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# dataset -> (arquivo, url_fonte ou None, md5 esperado ou None)
SOURCES = {
    "cars196": ("Cars196.tar", None,
                None),  # sem fonte pública; exige .tar local
    "cub200": ("CUB_200_2011.tgz",
               "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
               "97eceeb196236b17998738112f37df78"),
}
DEFAULT_BUCKET = "graph-rkd-832271495954"
DEFAULT_PREFIX = "graph-rkd/data"


class StageError(Exception):
    """Falha de staging de um dataset (mensagem legível)."""


def log(msg):
    print(f"[stage] {msg}", flush=True)


def retry(fn, attempts=8, base_delay=3.0, what="operação"):
    """Executa ``fn`` com retries e backoff exponencial. Levanta a última exceção."""
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - queremos re-tentar tudo transitório
            last = e
            if i == attempts:
                break
            delay = base_delay * (2 ** (i - 1))
            log(f"{what}: tentativa {i}/{attempts} falhou ({e}); "
                f"aguardando {delay:.0f}s")
            time.sleep(delay)
    raise StageError(f"{what} falhou após {attempts} tentativas: {last}")


# --------------------------------------------------------------------------- #
# md5                                                                          #
# --------------------------------------------------------------------------- #
def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# download com resume                                                          #
# --------------------------------------------------------------------------- #
def _download_once(url, dest):
    """Uma passada de download COM RESUME (Range a partir do que já existe)."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            # 206 = resume aceito (append); 200 = servidor ignorou Range (recomeça)
            mode = "ab" if (have and r.status == 206) else "wb"
            if mode == "wb":
                have = 0
            total = r.getheader("Content-Length")
            with open(dest, mode) as f:
                shutil.copyfileobj(r, f, length=1 << 20)
    except (urllib.error.URLError, OSError) as e:
        raise StageError(f"download interrompido em {os.path.getsize(dest)}B: {e}")
    got = os.path.getsize(dest)
    if have and got <= have:
        raise StageError("download não avançou (sem progresso)")
    return got


def download_with_resume(url, dest, expected_md5=None, attempts=25):
    """Baixa ``url`` -> ``dest`` resumindo entre tentativas; valida md5 no fim."""
    def one():
        size = _download_once(url, dest)
        log(f"baixado {size/1e6:.1f} MB")
        return size
    retry(one, attempts=attempts, base_delay=4.0, what=f"download {os.path.basename(dest)}")
    if expected_md5:
        log("verificando md5...")
        got = md5sum(dest)
        if got != expected_md5:
            raise StageError(f"md5 divergente: {got} != {expected_md5} "
                             "(arquivo corrompido/incompleto)")
        log("md5 OK")


# --------------------------------------------------------------------------- #
# S3 (via aws cli; usa o profile/credenciais do ambiente)                      #
# --------------------------------------------------------------------------- #
def _aws(args, profile=None, capture=False):
    env = dict(os.environ, AWS_MAX_ATTEMPTS="15", AWS_RETRY_MODE="standard")
    if profile:
        env["AWS_PROFILE"] = profile
    cp = subprocess.run(["aws", *args], env=env, text=True,
                        capture_output=capture)
    if cp.returncode != 0:
        raise StageError(f"aws {' '.join(args[:2])} rc={cp.returncode}: "
                         f"{(cp.stderr or '').strip()[:200]}")
    return cp.stdout if capture else None


def s3_size(bucket, key, profile=None):
    """ContentLength do objeto, ou None se não existir."""
    try:
        out = _aws(["s3api", "head-object", "--bucket", bucket, "--key", key,
                    "--query", "ContentLength", "--output", "text"],
                   profile=profile, capture=True)
        return int(out.strip())
    except StageError:
        return None


def s3_upload(local, bucket, key, profile=None, attempts=6):
    uri = f"s3://{bucket}/{key}"
    retry(lambda: _aws(["s3", "cp", local, uri, "--no-progress"], profile=profile),
          attempts=attempts, base_delay=5.0, what=f"upload {os.path.basename(local)}")
    remote = s3_size(bucket, key, profile)
    local_sz = os.path.getsize(local)
    if remote != local_sz:
        raise StageError(f"tamanho no S3 ({remote}) != local ({local_sz}) após upload")
    log(f"S3 OK: {uri} ({remote} bytes)")


def check_aws_cli():
    if shutil.which("aws") is None:
        raise StageError("aws CLI não encontrado no PATH (instale/config. o profile)")


# --------------------------------------------------------------------------- #
# staging por dataset                                                          #
# --------------------------------------------------------------------------- #
def stage_one(ds, archive_dir, bucket, prefix, profile, force, check_only):
    fname, url, md5 = SOURCES[ds]
    key = f"{prefix.rstrip('/')}/{fname}"
    local = os.path.join(archive_dir, fname)

    remote_sz = s3_size(bucket, key, profile)
    if remote_sz and not force:
        log(f"{ds}: já no S3 ({remote_sz} bytes) — pulando")
        return "skip(s3)"
    if check_only:
        have_local = os.path.exists(local)
        log(f"{ds}: S3={'ausente' if not remote_sz else remote_sz} "
            f"local={'sim' if have_local else 'não'} url={'sim' if url else 'não'}")
        return "checked"

    # garante o arquivo local
    if not os.path.exists(local):
        if url is None:
            raise StageError(
                f"{ds}: sem cópia local em {local} e sem fonte pública "
                "(forneça o arquivo via --archive-dir)")
        os.makedirs(archive_dir, exist_ok=True)
        download_with_resume(url, local, md5)
    elif md5:
        log(f"{ds}: arquivo local existe; verificando md5...")
        if md5sum(local) != md5:
            raise StageError(f"{ds}: md5 do arquivo local diverge do esperado")
        log(f"{ds}: md5 local OK")

    s3_upload(local, bucket, key, profile)
    return "uploaded"


def main(argv=None):
    p = argparse.ArgumentParser(description="Envia os datasets Graph-RKD ao S3 (robusto)")
    p.add_argument("--datasets", nargs="+", default=["cars196", "cub200"],
                   choices=list(SOURCES))
    p.add_argument("--archive-dir", default=os.path.join(os.path.dirname(__file__),
                                                         "..", "dataset"),
                   help="onde procurar/salvar os arquivos (Cars196.tar, CUB_200_2011.tgz)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", "gabomfim"))
    p.add_argument("--force", action="store_true", help="reenvia mesmo se já no S3")
    p.add_argument("--check", action="store_true",
                   help="só diagnóstico (S3/local/fonte), não baixa nem envia")
    a = p.parse_args(argv)

    try:
        check_aws_cli()
    except StageError as e:
        log(f"ERRO: {e}"); return 2

    results, failures = {}, 0
    for ds in a.datasets:
        try:
            results[ds] = stage_one(ds, os.path.abspath(a.archive_dir), a.bucket,
                                    a.prefix, a.aws_profile, a.force, a.check)
        except StageError as e:
            failures += 1
            results[ds] = f"FALHA: {e}"
            log(f"{ds}: {e}")
        except KeyboardInterrupt:
            log("interrompido pelo usuário"); return 130

    log("resumo: " + " | ".join(f"{k}={v}" for k, v in results.items()))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
