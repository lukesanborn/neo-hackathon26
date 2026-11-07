"""Pinned Modal container images.

Two stacks, identical except for torch / torchtext / CUDA:

STACK "hopper" (default; prepare / H100 training / aggregate)
  * torch 2.1.2 + CUDA 12.1 wheels <-> torchtext 0.16.2 (scGPT's declared dependency)
  * Python 3.11 (Modal's 2026.10 image builder has no 3.10; scGPT's metadata says <3.11,
    so it is installed with --ignore-requires-python — verified importable)
  * cell-gears 0.0.2 for the processed-data URL and DataSplitter (needs torch_geometric)

STACK "blackwell" (B200/B300 benchmark; also runs on H100 for an apples-to-apples check)
  * torch 2.12.1 + CUDA 13.0 wheels (B300 requires CUDA 13.1+ per Modal docs)
  * NO torchtext (no releases after torch 2.3). A tiny import shim
    (environment/torchtext_shim) satisfies `import torchtext.vocab` inside scGPT; our code
    never instantiates GeneVocab. No GEARS / torch_geometric (prepare runs on "hopper").

Common to both
  * numpy 1.26 (torch 2.1 ABI); scGPT 0.2.4 installed with --no-deps plus only the
    runtime imports it actually needs (ipython, datasets, pyarrow).
  * flash-attn NOT installed; scGPT uses use_fast_transformer=False (nn.TransformerEncoder,
    SDPA attention). Pretrained flash 'Wqkv' weights are renamed to 'in_proj_' on load.
"""

import os

import modal

DEMO_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "demo_template.html")
SHIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchtext_shim")
PYTHON_VERSION = "3.11"

PINNED = {
    "hopper": {"torch": "2.1.2", "torchtext": "0.16.2", "cuda_index": "cu121"},
    "blackwell": {"torch": "2.12.1", "torchtext": None, "cuda_index": "cu130"},
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "pandas": "2.1.4",
    "anndata": "0.10.5.post1",
    "scanpy": "1.9.8",
    "torch_geometric": "2.4.0",
    "cell-gears": "0.0.2",
    "scgpt": "0.2.4",
}

COMMON_PIP = [
    f"numpy=={PINNED['numpy']}",
    f"scipy=={PINNED['scipy']}",
    f"pandas=={PINNED['pandas']}",
    f"anndata=={PINNED['anndata']}",
    f"scanpy=={PINNED['scanpy']}",
    "gdown==5.1.0",
    "matplotlib==3.8.2",
    "scikit-learn==1.3.2",
    "h5py==3.10.0",
    "tqdm==4.66.1",
    "networkx==3.2.1",
    "requests==2.31.0",
    "typing_extensions==4.12.2",
    # undeclared-at-runtime scGPT imports (scgpt.utils / scgpt.scbank)
    "ipython==8.20.0",
    "datasets==2.16.1",
    "pyarrow==14.0.2",
]

ENV = {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def _torch(stack: str) -> list[str]:
    p = PINNED[stack]
    pkgs = [f"torch=={p['torch']}"]
    if p["torchtext"]:
        pkgs.append(f"torchtext=={p['torchtext']}")
    return pkgs


def build_image(stack: str = "hopper") -> modal.Image:
    p = PINNED[stack]
    img = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git", "unzip", "wget")
        .pip_install(
            *_torch(stack),
            index_url=f"https://download.pytorch.org/whl/{p['cuda_index']}",
            extra_index_url="https://pypi.org/simple",
        )
        .pip_install(*COMMON_PIP)
    )
    if stack == "hopper":
        img = img.pip_install(f"torch_geometric=={PINNED['torch_geometric']}", "dcor==0.6").pip_install(
            f"cell-gears=={PINNED['cell-gears']}",
            f"scgpt=={PINNED['scgpt']}",
            extra_options="--no-deps --ignore-requires-python",
        )
        env = {**ENV, "JEPA_STACK": "hopper"}
    else:
        img = img.pip_install(
            f"scgpt=={PINNED['scgpt']}", extra_options="--no-deps --ignore-requires-python"
        ).add_local_dir(SHIM_DIR, "/opt/torchtext_shim", copy=True)
        env = {**ENV, "JEPA_STACK": "blackwell", "PYTHONPATH": "/opt/torchtext_shim"}
    # add_local_python_source ships only .py files; the demo template is mounted separately
    return (img.env({**env, "JEPA_DEMO_TEMPLATE": "/opt/demo/demo_template.html"})
            .add_local_python_source("src", "environment")
            .add_local_file(DEMO_TEMPLATE, "/opt/demo/demo_template.html"))
