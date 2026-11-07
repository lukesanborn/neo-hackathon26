"""Dataset registry, cache-path compatibility and unrepresentable-target exclusion."""

import numpy as np
import pytest

from src.data import (ADAMSON_URL, DATASET_LABELS, DATASETS, cache_dir_name, download_dataset,
                      screen_targets)


def test_registry_urls():
    assert DATASETS["adamson"] == "https://dataverse.harvard.edu/api/access/datafile/6154417"
    assert DATASETS["replogle_k562_essential"] == "https://dataverse.harvard.edu/api/access/datafile/7458695"
    assert ADAMSON_URL == DATASETS["adamson"]
    assert set(DATASET_LABELS) == set(DATASETS)
    # the Adamson label is part of existing manifests; keep it byte-identical
    assert DATASET_LABELS["adamson"] == "adamson (GEARS processed, cell-gears 0.0.2 dataverse file 6154417)"


def test_adamson_cache_path_unchanged():
    # existing experiments (split hash 5618e72828c1) point at this exact directory
    assert cache_dir_name("adamson", 1) == "adamson_gears002_simulation_seed1"
    assert cache_dir_name("replogle_k562_essential", 1) == "replogle_k562_essential_gears_simulation_seed1"
    assert cache_dir_name("replogle_k562_essential", 3) != cache_dir_name("replogle_k562_essential", 1)


def test_modal_cache_dir_compat():
    modal_app = pytest.importorskip("modal_app", exc_type=Exception)
    assert modal_app.cache_dir(1) == "/vol/data/adamson_gears002_simulation_seed1"
    assert modal_app.cache_dir(1, "adamson") == "/vol/data/adamson_gears002_simulation_seed1"
    assert modal_app.cache_dir(1, "replogle_k562_essential") == "/vol/data/replogle_k562_essential_gears_simulation_seed1"


def test_unknown_dataset_rejected(tmp_path):
    with pytest.raises(KeyError):
        cache_dir_name("norman", 1)
    with pytest.raises(KeyError):
        download_dataset("norman", str(tmp_path))


def test_screen_targets_excludes_unrepresentable():
    genes = ["A", "B", "OLDSYM", "GONE", "C"]
    vocab = {"A", "B", "C", "NEWSYM"}
    conds = ["ctrl", "A+ctrl", "B+ctrl", "OLDSYM+ctrl", "GONE+ctrl", "UNMEASURED+ctrl", "A+B", "ctrl"]
    excluded, aliased = screen_targets(np.array(conds), genes, vocab, {"OLDSYM": "NEWSYM"})
    assert excluded == {"GONE+ctrl": "target_not_in_scgpt_vocab",
                        "UNMEASURED+ctrl": "target_not_in_measured_genes",
                        "A+B": "not_single_gene"}
    assert aliased == {"OLDSYM": "NEWSYM"}
    # alias whose target is itself absent from the vocab does not rescue the perturbation
    excluded, _ = screen_targets(conds, genes, vocab, {"GONE": "ALSO_GONE"})
    assert excluded["GONE+ctrl"] == "target_not_in_scgpt_vocab"
    # no vocab -> only structural / measured-gene checks
    excluded, aliased = screen_targets(conds, genes, None)
    assert set(excluded) == {"UNMEASURED+ctrl", "A+B"} and aliased == {}


def test_screen_targets_ambiguous_and_colliding():
    genes = ["A", "DUP", "DUP", "OLD", "NEW"]
    vocab = {"A", "DUP", "NEW"}
    excluded, aliased = screen_targets(["ctrl", "A+ctrl", "DUP+ctrl", "OLD+ctrl"], genes, vocab, {"OLD": "NEW"})
    assert excluded == {"DUP+ctrl": "target_name_duplicated_in_gene_list",
                        "OLD+ctrl": "alias_collides_with_measured_gene"}
    assert aliased == {}


def test_build_cache_requires_base_aliases_with_vocab(tmp_path):
    from src.data import build_cache

    with pytest.raises(ValueError, match="base_aliases"):
        build_cache("unused.h5ad", str(tmp_path), 1, 8, 0, {}, {}, vocab_genes={"A"})


def test_kept_targets_never_map_to_pad():
    """The Trainer's targets_mapped_to_pad failure must be impossible after screening."""
    from src.models import HGNC_ALIASES, gene_vocab_ids

    class V(dict):
        pass

    extra = {"OLDSYM": "NEWSYM"}
    genes = ["A", "OLDSYM", "GONE", "AARS"]
    vocab = V({"<pad>": 0, "A": 1, "NEWSYM": 2, "AARS1": 3})
    conds = ["A+ctrl", "OLDSYM+ctrl", "GONE+ctrl", "AARS+ctrl"]
    excluded, _ = screen_targets(conds, genes, set(vocab), {**HGNC_ALIASES, **extra})
    gids, info = gene_vocab_ids(genes, vocab, extra_aliases=extra)
    for c in conds:
        if c not in excluded:
            assert gids[genes.index(c.split("+")[0])] != vocab["<pad>"], c
    assert set(excluded) == {"GONE+ctrl"}
    # without the dataset-scoped alias the default mapping is unchanged (Adamson behaviour)
    gids0, _ = gene_vocab_ids(genes, vocab)
    assert gids0[1] == vocab["<pad>"] and gids0[3] == vocab["AARS1"]


def test_dataset_aliases_do_not_touch_adamson():
    from src.models import DATASET_HGNC_ALIASES, HGNC_ALIASES

    assert DATASET_HGNC_ALIASES["adamson"] == {}
    # Replogle-only renames must not leak into the global table used for Adamson
    assert not set(DATASET_HGNC_ALIASES["replogle_k562_essential"]) & set(HGNC_ALIASES)
