"""Minimal import shim for the Blackwell image (no real torchtext; it stopped at torch 2.3).

scGPT imports `torchtext.vocab` / `torchtext.vocab.Vocab` only to define GeneVocab. Our
pipeline never instantiates GeneVocab (src/models.py reads vocab.json directly), so an
importable placeholder is sufficient. Any accidental use fails loudly.
"""
__version__ = "0.0.0+shim"
from . import vocab  # noqa: F401
