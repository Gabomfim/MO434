"""Prepares (downloads if needed) and UPLOADS the datasets to S3, robustly.

Does the source -> local -> S3 path with resume, retries with backoff, md5
verification and idempotency (skips what is already on S3 with the right size). It
is the inverse of ``data_prep.py`` (which does S3 -> local in the training jobs).

Destination layout (same region as training; better access for SageMaker/Modal):
    s3://<bucket>/<prefix>/Cars196.tar
    s3://<bucket>/<prefix>/CUB_200_2011.tgz

Notes:
  * Cars-196 has NO live public source (Stanford went down). Provide the local .tar
    (``--archive-dir``); without it and not being on S3, the script reports a clear error.
  * CUB-200 downloads from the Caltech Data mirror (the old repo URL is 404).

Examples:
    python sm/stage_data.py                      # both, default bucket/prefix
    python sm/stage_data.py --datasets cub200    # only CUB
    python sm/stage_data.py --check              # diagnostics only (no download/upload)
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

# dataset -> (file, source_url or None, expected md5 or None)
SOURCES = {
    "cars196": ("Cars196.tar", None,
                None),  # no public source; requires local .tar
    "cub200": ("CUB_200_2011.tgz",
               "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
               "97eceeb196236b17998738112f37df78"),
}
DEFAULT_BUCKET = "graph-rkd-832271495954"
DEFAULT_PREFIX = "graph-rkd/data"


class StageError(Exception):
    """Staging failure of a dataset (readable message)."""


def log(msg):
    print(f"[stage] {msg}", flush=True)


def retry(fn, attempts=8, base_delay=3.0, what="operation"):
    """Runs ``fn`` with retries and exponential backoff. Raises the last exception."""
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - we want to retry everything transient
            last = e
            if i == attempts:
                break
            delay = base_delay * (2 ** (i - 1))
            log(f"{what}: attempt {i}/{attempts} failed ({e}); "
                f"waiting {delay:.0f}s")
            time.sleep(delay)
    raise StageError(f"{what} failed after {attempts} attempts: {last}")


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
# download with resume                                                         #
# --------------------------------------------------------------------------- #
def _download_once(url, dest):
    """One download pass WITH RESUME (Range from what already exists)."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            # 206 = resume accepted (append); 200 = server ignored Range (restart)
            mode = "ab" if (have and r.status == 206) else "wb"
            if mode == "wb":
                have = 0
            total = r.getheader("Content-Length")
            with open(dest, mode) as f:
                shutil.copyfileobj(r, f, length=1 << 20)
    except (urllib.error.URLError, OSError) as e:
        raise StageError(f"download interrupted at {os.path.getsize(dest)}B: {e}")
    got = os.path.getsize(dest)
    if have and got <= have:
        raise StageError("download did not advance (no progress)")
    return got


def download_with_resume(url, dest, expected_md5=None, attempts=25):
    """Downloads ``url`` -> ``dest`` resuming between attempts; validates md5 at the end."""
    def one():
        size = _download_once(url, dest)
        log(f"downloaded {size/1e6:.1f} MB")
        return size
    retry(one, attempts=attempts, base_delay=4.0, what=f"download {os.path.basename(dest)}")
    if expected_md5:
        log("verifying md5...")
        got = md5sum(dest)
        if got != expected_md5:
            raise StageError(f"md5 mismatch: {got} != {expected_md5} "
                             "(corrupted/incomplete file)")
        log("md5 OK")


# --------------------------------------------------------------------------- #
# S3 (via aws cli; uses the profile/credentials from the environment)          #
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
    """ContentLength of the object, or None if it does not exist."""
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
        raise StageError(f"S3 size ({remote}) != local ({local_sz}) after upload")
    log(f"S3 OK: {uri} ({remote} bytes)")


def check_aws_cli():
    if shutil.which("aws") is None:
        raise StageError("aws CLI not found on PATH (install/configure the profile)")


# --------------------------------------------------------------------------- #
# staging per dataset                                                          #
# --------------------------------------------------------------------------- #
def stage_one(ds, archive_dir, bucket, prefix, profile, force, check_only):
    fname, url, md5 = SOURCES[ds]
    key = f"{prefix.rstrip('/')}/{fname}"
    local = os.path.join(archive_dir, fname)

    remote_sz = s3_size(bucket, key, profile)
    if remote_sz and not force:
        log(f"{ds}: already on S3 ({remote_sz} bytes) — skipping")
        return "skip(s3)"
    if check_only:
        have_local = os.path.exists(local)
        log(f"{ds}: S3={'missing' if not remote_sz else remote_sz} "
            f"local={'yes' if have_local else 'no'} url={'yes' if url else 'no'}")
        return "checked"

    # ensure the local file
    if not os.path.exists(local):
        if url is None:
            raise StageError(
                f"{ds}: no local copy at {local} and no public source "
                "(provide the file via --archive-dir)")
        os.makedirs(archive_dir, exist_ok=True)
        download_with_resume(url, local, md5)
    elif md5:
        log(f"{ds}: local file exists; verifying md5...")
        if md5sum(local) != md5:
            raise StageError(f"{ds}: local file md5 differs from expected")
        log(f"{ds}: local md5 OK")

    s3_upload(local, bucket, key, profile)
    return "uploaded"


def main(argv=None):
    p = argparse.ArgumentParser(description="Uploads the Graph-RKD datasets to S3 (robust)")
    p.add_argument("--datasets", nargs="+", default=["cars196", "cub200"],
                   choices=list(SOURCES))
    p.add_argument("--archive-dir", default=os.path.join(os.path.dirname(__file__),
                                                         "..", "dataset"),
                   help="where to look for/save the files (Cars196.tar, CUB_200_2011.tgz)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", "gabomfim"))
    p.add_argument("--force", action="store_true", help="re-uploads even if already on S3")
    p.add_argument("--check", action="store_true",
                   help="diagnostics only (S3/local/source), does not download or upload")
    a = p.parse_args(argv)

    try:
        check_aws_cli()
    except StageError as e:
        log(f"ERROR: {e}"); return 2

    results, failures = {}, 0
    for ds in a.datasets:
        try:
            results[ds] = stage_one(ds, os.path.abspath(a.archive_dir), a.bucket,
                                    a.prefix, a.aws_profile, a.force, a.check)
        except StageError as e:
            failures += 1
            results[ds] = f"FAILURE: {e}"
            log(f"{ds}: {e}")
        except KeyboardInterrupt:
            log("interrupted by the user"); return 130

    log("summary: " + " | ".join(f"{k}={v}" for k, v in results.items()))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
