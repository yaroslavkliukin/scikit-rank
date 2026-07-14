"""Single source of truth for the pre-split BARS CTR datasets used across ``exps/``.

Stdlib-only (do not import torch/polars/pandas). Add a dataset by appending one
:class:`Dataset` entry to ``DATASETS``; every consumer picks it up. Reference MD5s come
from each dataset's BARS/reczoo ``README.md``.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    """Metadata for one pre-split CTR dataset (BARS layout: train/valid/test.csv)."""

    name: str
    label: str = "label"
    # filename -> expected MD5 (from the dataset's BARS/reczoo README).
    md5s: dict[str, str] = field(default_factory=dict)


DATASETS: dict[str, Dataset] = {
    "criteo_x1": Dataset(
        name="criteo_x1",
        md5s={
            "train.csv": "30b89c1c7213013b92df52ec44f52dc5",
            "valid.csv": "f73c71fb3c4f66b6ebdfa032646bea72",
            "test.csv": "2c48b26e84c04a69b948082edae46f8c",
        },
    ),
    "avazu_x1": Dataset(
        name="avazu_x1",
        md5s={
            "train.csv": "f1114a07aea9e996842c71648e0f6395",
            "valid.csv": "d9568f246357d156c4b8030fadb8b623",
            "test.csv": "9e2fe9c48705c9315ae7a0953eb57acf",
        },
    ),
}
