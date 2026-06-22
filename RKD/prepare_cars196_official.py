"""Materialize the official Stanford Cars-196 train/test split locally.

The local ``car_ims/`` + ``cars_annos.mat`` we had is a re-indexed copy without
the official ``test`` flag (every image marked train), so classification could
not be evaluated. This script fetches the canonical split (8144 train / 8041
test, the standard literature split) from the Hugging Face dataset
``Donghyun99/Stanford-Cars`` (parquet with image bytes + 0-indexed label) and
writes it as an ImageFolder under ``<data>/Cars196/official/{train,test}/<label>/``.

``Cars196Classification`` reads this official layout when present.

Requires: huggingface_hub, pyarrow. Run once:
    python prepare_cars196_official.py --data ../data
"""

import argparse
import io
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "Donghyun99/Stanford-Cars"
FILES = {
    "train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
    "test": ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"],
}


def materialize(data_root):
    out_root = os.path.join(data_root, "Cars196", "official")
    for split, files in FILES.items():
        n = 0
        for f in files:
            path = hf_hub_download(REPO, f, repo_type="dataset")
            table = pq.read_table(path)
            images = table.column("image").to_pylist()
            labels = table.column("label").to_pylist()
            for img, label in zip(images, labels):
                cls_dir = os.path.join(out_root, split, "%03d" % int(label))
                os.makedirs(cls_dir, exist_ok=True)
                # image bytes are already encoded JPEGs -- write as-is (lossless).
                with open(os.path.join(cls_dir, "%06d.jpg" % n), "wb") as fh:
                    fh.write(img["bytes"])
                n += 1
        print("%s: %d images -> %s" % (split, n, os.path.join(out_root, split)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data")
    opts = p.parse_args()
    materialize(opts.data)


if __name__ == "__main__":
    main()
