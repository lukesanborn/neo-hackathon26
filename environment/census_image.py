"""Pinned CPU image for the Track A CELLxGENE Census kidney download.

Track A — Cell-JEPA reimplemented from the paper (unofficial).

Kept separate from environment/image.py: the Census client needs a newer tiledbsoma stack
than the scGPT training images, and this image has no torch / scGPT at all.

Primary pin: cellxgene-census 1.18.0 (python>=3.10, tiledbsoma>=1.15.3).
Fallback, if that stack cannot read the 2023-05-15 LTS build: cellxgene-census 1.15.0 with
tiledbsoma~=1.11.4 and numpy<2. Pick it with CENSUS_STACK=legacy when running modal.
"""

import os

import modal

PYTHON_VERSION = "3.11"  # Modal's builder has no 3.10

STACKS = {
    "current": {
        "cellxgene-census": "1.18.0",
        "tiledbsoma": "1.17.1",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "pandas": "2.2.3",
        "anndata": "0.11.4",
        "pyarrow": "18.1.0",
        "s3fs": "2024.12.0",
    },
    "legacy": {
        "cellxgene-census": "1.15.0",
        "tiledbsoma": "1.11.4",
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "pandas": "2.1.4",
        "anndata": "0.10.5.post1",
        "pyarrow": "14.0.2",
        "s3fs": "2024.6.1",
    },
}

CENSUS_STACK = os.environ.get("CENSUS_STACK", "current")


def build_census_image(stack: str = CENSUS_STACK) -> modal.Image:
    pins = STACKS[stack]
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*[f"{k}=={v}" for k, v in pins.items()])
        .env({"PYTHONHASHSEED": "0", "CENSUS_STACK": stack})
        .add_local_python_source("src", "environment")
    )
