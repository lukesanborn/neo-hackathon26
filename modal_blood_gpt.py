"""Blood GPT: Modal inference over Vincent's strongest retrieval model.

The checkpoint is the completed run with the highest validation identification percentile:
``celljepa_s5_s2bjepa_r1`` (blood-JEPA continued pretraining, then Cell-JEPA fine-tune
with the contrastive retrieval loss, seed 5, selected epoch 8, val id percentile 0.818).
An uploaded cell is ranked by cosine similarity of its expression change against that
run's predicted profiles for held-out Replogle genes.

    modal run modal_blood_gpt.py --stage prep --force
    modal run modal_blood_gpt.py --stage samples
    modal deploy modal_blood_gpt.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from environment.image import build_image

APP_NAME = "blood-gpt"
VOL_PATH = "/vol"
DEMO_DIR = "/opt/blood_gpt"
PREP_DIR = f"{VOL_PATH}/blood_gpt"
REPLOGLE_CACHE = f"{VOL_PATH}/data/replogle_k562_essential_gears_simulation_seed1"
# Highest val id_pct among completed blood-JEPA -> Cell-JEPA + contrastive runs (seeds 1-5).
STRONGEST_RUN = f"{VOL_PATH}/experiments/replogle_k562_v1/runs/celljepa_s5_s2bjepa_r1"
LIBRARY_VERSION = "celljepa_s5_s2bjepa_r1"

ROOT = Path(__file__).resolve().parent
LOCAL_DEMO = ROOT / "website"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb", create_if_missing=True)

image = build_image(
    "blackwell",
    extra_pip=("fastapi==0.115.6", "uvicorn[standard]==0.32.1", "python-multipart==0.0.17"),
).add_local_dir(str(LOCAL_DEMO), DEMO_DIR, ignore=["**/node_modules/**", "**/samples/**"])


def _plain(x):
    return json.loads(json.dumps(x, default=str))


def _make_asgi(svc: "BloodGPT"):
    """ASGI app factory.

    Handlers use builtin ``dict`` (not local Pydantic models) so FastAPI can resolve
    annotations under ``from __future__ import annotations``.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(title="Blood JEPA")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/api/health")
    def health():
        return {"ok": True, "n_library": len(svc.lib_names), "ckpt": svc.ckpt_info}

    @web.get("/api/reference")
    def reference():
        return svc.reference

    @web.get("/api/examples")
    def examples():
        out = []
        for ex in svc.examples:
            out.append({
                "id": ex["id"],
                "label": ex["label"],
                "hint": ex["hint"],
                "genes": ex["genes"],
                "values": ex["values"],
            })
        return {"examples": out}

    @web.post("/api/infer")
    async def infer(payload: dict):
        genes = payload.get("genes")
        values = payload.get("values")
        top_k = int(payload.get("top_k", 8))
        if not isinstance(genes, list) or not isinstance(values, list) or not genes:
            raise HTTPException(400, "genes and values must be non-empty lists")
        if len(genes) != len(values):
            raise HTTPException(400, "genes and values must have the same length")
        try:
            return svc._infer(genes, [float(v) for v in values], top_k=max(1, min(top_k, 20)))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            return JSONResponse({"detail": f"infer failed: {e}"}, status_code=500)

    dist = os.path.join(DEMO_DIR, "web", "dist")
    legacy = DEMO_DIR

    if os.path.isdir(dist) and os.path.isfile(os.path.join(dist, "index.html")):
        # API routes above take precedence; html=True falls back to index.html.
        web.mount("/", StaticFiles(directory=dist, html=True), name="spa")
    else:
        @web.get("/")
        def index():
            return FileResponse(os.path.join(legacy, "index.html"))

        web.mount("/static", StaticFiles(directory=legacy), name="static")

    return web


def _prediction_rows(run_dir: str):
    """Held-out profiles written by the selected epoch of one fine-tune."""
    import numpy as np

    conds, preds = [], []
    for name in ("val_predictions.npz", "test_predictions.npz"):
        z = np.load(os.path.join(run_dir, name), allow_pickle=True)
        conds.extend(str(c) for c in z["conditions"].tolist())
        preds.append(z["preds"].astype(np.float32))
    return conds, np.concatenate(preds, axis=0)


def _profile_csv(genes: list[str], values) -> str:
    lines = ["gene,value"]
    for g, v in zip(genes, values):
        lines.append(f"{g},{float(v):.6f}")
    return "\n".join(lines) + "\n"


def _top_gene(delta, pred_deltas, names: list[str]) -> str:
    from src.blood_demo import rank_knockdowns

    return rank_knockdowns(delta, pred_deltas, names, top_k=1)[0]["gene"]


def _pool_profile(ds, cond: str, n: int, rng):
    import numpy as np

    idx = ds.cells_of(cond)
    take = rng.choice(idx, size=min(n, len(idx)), replace=False)
    values = np.asarray(ds.X[int(take[0])]) if len(take) == 1 else np.asarray(ds.X[np.sort(take)]).mean(0)
    return values, int(len(take))


def _smallest_hit(ds, conds, pred_deltas, names, ctrl, rng):
    """Smallest real-cell pool whose expression change ranks the true gene first."""
    from src.blood_demo import HGNC_ALIASES
    from src.data import target_gene

    best = None
    for cond in conds:
        gene = HGNC_ALIASES.get(target_gene(cond), target_gene(cond))
        n_avail = len(ds.cells_of(cond))
        single = None
        for n in (8, 16, 32, 64, 128, n_avail):
            values, used = _pool_profile(ds, cond, n, rng)
            if single is None:
                single = _pool_profile(ds, cond, 1, rng)[0]
            if _top_gene(values - ctrl, pred_deltas, names) != gene:
                continue
            if best is None or used < best[0]:
                best = (used, cond, gene, values, single)
            break
    if best is None:
        raise RuntimeError("no knockdown pool ranked its own gene first")
    return best


def _demo_uploads(ds, pred_conds: list[str], pred_deltas, names: list[str]) -> list[dict]:
    """Real Replogle test-split cells. Each pool is the noisiest size that still ranks its gene first."""
    import numpy as np

    rng = np.random.default_rng(7)
    ctrl = np.asarray(ds.control_reference())
    ranked = sorted(
        [c for c in pred_conds if c != "ctrl"],
        key=lambda c: float(np.linalg.norm(np.asarray(ds.observed_mean(c)) - ctrl)),
        reverse=True,
    )
    genes = list(ds.gene_names)
    out = []
    remaining = [c for c in ranked if c in set(ds.split["test"])]
    for eid in ("test_pool", "test_pool_2"):
        n_cells, cond, gene, values, single = _smallest_hit(
            ds, remaining[:12], pred_deltas, names, ctrl, rng,
        )
        remaining = [c for c in remaining if c != cond]
        out.append({
            "id": eid,
            "label": f"Noisy K562 pool ({n_cells} cells)",
            "hint": f"Mean of {n_cells} real Replogle test-split cells from one knockdown",
            "genes": genes,
            "values": [round(float(v), 6) for v in values],
            "true_gene": gene,
            "split": "test",
            "n_cells": n_cells,
            "filename": f"replogle_{gene}_{n_cells}cell.csv",
        })
        out.append({
            "id": eid + "_single",
            "label": "Very noisy single cell",
            "hint": "One real test-split cell from the same knockdown. Dropout often hides the gene.",
            "genes": genes,
            "values": [round(float(v), 6) for v in single],
            "true_gene": gene,
            "split": "test",
            "n_cells": 1,
            "filename": f"replogle_{gene}_single_cell.csv",
        })
    ctrl_i = int(rng.choice(ds.ctrl_idx))
    out.append({
        "id": "control_single",
        "label": "K562 control cell",
        "hint": "One real unperturbed Replogle K562 cell",
        "genes": genes,
        "values": [round(float(v), 6) for v in np.asarray(ds.X[ctrl_i])],
        "true_gene": None,
        "split": "control",
        "n_cells": 1,
        "filename": "replogle_control_single_cell.csv",
    })
    return out


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=1800)
def prep(force: bool = False) -> dict:
    """Library of this run's predicted profiles, plus measured responses for the heatmap."""
    import numpy as np

    from src.blood_demo import HGNC_ALIASES, top_delta_genes
    from src.data import load_cache, target_gene

    vol.reload()
    os.makedirs(PREP_DIR, exist_ok=True)
    marker = os.path.join(PREP_DIR, "ready.json")
    if not force and os.path.exists(marker):
        with open(marker) as f:
            prev = json.load(f)
        if prev.get("version") == LIBRARY_VERSION:
            return _plain(prev)

    if not os.path.exists(os.path.join(REPLOGLE_CACHE, "manifest.json")):
        raise FileNotFoundError(f"Replogle cache missing at {REPLOGLE_CACHE}")
    if not os.path.exists(os.path.join(STRONGEST_RUN, "best.pt")):
        raise FileNotFoundError(f"strongest checkpoint missing: {STRONGEST_RUN}/best.pt")
    ds = load_cache(REPLOGLE_CACHE, mmap=True)
    conds, preds = _prediction_rows(STRONGEST_RUN)
    if preds.shape[1] != ds.n_genes:
        raise RuntimeError(f"prediction width {preds.shape[1]} != panel {ds.n_genes}")
    ctrl = np.asarray(ds.control_reference(), dtype=np.float64)
    names, pred_deltas, measured = [], [], []
    for cond, pred in zip(conds, preds):
        gene = target_gene(cond)
        if gene is None:
            continue
        names.append(HGNC_ALIASES.get(gene, gene))
        pred_deltas.append(np.asarray(pred, dtype=np.float64) - ctrl)
        measured.append(np.asarray(ds.observed_mean(cond), dtype=np.float64) - ctrl)
    pred_deltas = np.stack(pred_deltas).astype(np.float32)
    measured = np.stack(measured).astype(np.float32)
    card_deltas = {
        name: top_delta_genes(delta, ds.gene_names, k=6) for name, delta in zip(names, measured)
    }
    np.savez_compressed(
        os.path.join(PREP_DIR, "library.npz"),
        names=np.array(names),
        pred_deltas=pred_deltas,
        measured_deltas=measured,
        gene_names=np.array(ds.gene_names),
        control=ctrl.astype(np.float32),
        conditions=np.array(conds),
    )
    with open(os.path.join(PREP_DIR, "card_deltas.json"), "w") as f:
        json.dump(card_deltas, f)
    reference = {
        "source": "Replogle K562 essential control mean (same panel as the retrieval model)",
        "genes": list(ds.gene_names),
        "values": [round(float(v), 4) for v in ctrl],
    }
    examples = _demo_uploads(ds, conds, pred_deltas, names)
    with open(os.path.join(PREP_DIR, "reference.json"), "w") as f:
        json.dump(reference, f)
    with open(os.path.join(PREP_DIR, "examples.json"), "w") as f:
        json.dump(examples, f)
    metrics = json.load(open(os.path.join(STRONGEST_RUN, "metrics.json")))
    info = {
        "version": LIBRARY_VERSION,
        "prep_dir": PREP_DIR,
        "run": os.path.basename(STRONGEST_RUN),
        "val_id_pct": (metrics.get("val_retrieval_selected") or {}).get("id_pct"),
        "selected_epoch": metrics.get("selected_epoch"),
        "n_library": len(names),
        "n_genes": ds.n_genes,
        "n_examples": len(examples),
        "example_genes": [e["true_gene"] for e in examples],
    }
    with open(marker, "w") as f:
        json.dump(info, f, indent=2)
    vol.commit()
    return _plain(info)


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=8192, timeout=600)
def sample_files() -> list[dict]:
    """CSV text for the noisy demo uploads written by prep."""
    vol.reload()
    path = os.path.join(PREP_DIR, "examples.json")
    if not os.path.exists(path):
        raise FileNotFoundError("examples.json missing; run prep first")
    with open(path) as f:
        examples = json.load(f)
    out = []
    for ex in examples:
        out.append({
            "filename": ex["filename"],
            "true_gene": ex.get("true_gene"),
            "label": ex["label"],
            "split": ex.get("split"),
            "csv": _profile_csv(ex["genes"], ex["values"]),
        })
    return out


@app.cls(
    image=image,
    volumes={VOL_PATH: vol},
    cpu=2,
    memory=8192,
    timeout=10 * 60,
    scaledown_window=10 * 60,
)
class BloodGPT:
    @modal.enter()
    def load(self):
        import numpy as np

        vol.reload()
        lib_path = os.path.join(PREP_DIR, "library.npz")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} missing; run: modal run modal_blood_gpt.py --stage prep --force")
        metrics_path = os.path.join(STRONGEST_RUN, "metrics.json")
        ckpt = os.path.join(STRONGEST_RUN, "best.pt")
        if not os.path.exists(ckpt) or os.path.getsize(ckpt) == 0:
            raise FileNotFoundError(ckpt)
        metrics = json.load(open(metrics_path))
        retr = metrics.get("val_retrieval_selected") or {}
        self.ckpt_info = {
            "run": os.path.basename(STRONGEST_RUN),
            "run_dir": STRONGEST_RUN,
            "ckpt": ckpt,
            "method": metrics.get("method"),
            "seed": metrics.get("seed"),
            "selected_epoch": metrics.get("selected_epoch"),
            "val_id_pct": retr.get("id_pct"),
            "init_ckpt": metrics.get("ckpt"),
        }
        z = np.load(lib_path, allow_pickle=True)
        self.lib_names = [str(x) for x in z["names"].tolist()]
        self.pred_deltas = z["pred_deltas"].astype(np.float32)
        self.measured_deltas = z["measured_deltas"].astype(np.float32)
        self.lib_control = z["control"].astype(np.float32)
        self.lib_gene_names = [str(x) for x in z["gene_names"].tolist()]
        with open(os.path.join(PREP_DIR, "reference.json")) as f:
            self.reference = json.load(f)
        with open(os.path.join(PREP_DIR, "examples.json")) as f:
            self.examples = json.load(f)

    def _infer(self, genes: list[str], values: list[float], top_k: int = 8) -> dict:
        import numpy as np

        from src.blood_demo import abnormal_genes, deltas_for_genes, heatmap_matrix, rank_knockdowns

        values_arr = np.asarray(values, dtype=np.float64)
        ref_genes = self.reference["genes"]
        ref_vals = np.asarray(self.reference["values"], dtype=np.float64)
        idx = {g: i for i, g in enumerate(genes)}
        aligned = np.zeros(len(ref_genes), dtype=np.float64)
        present = np.zeros(len(ref_genes), dtype=bool)
        for i, g in enumerate(ref_genes):
            if g in idx:
                aligned[i] = values_arr[idx[g]]
                present[i] = True
        n_overlap = int(present.sum())
        if n_overlap < 50:
            raise ValueError("upload shares too few genes with the Replogle K562 panel")
        delta = aligned - ref_vals
        ranked = rank_knockdowns(delta[present], self.pred_deltas[:, present], self.lib_names, top_k=top_k)
        abn = abnormal_genes(aligned, ref_vals, ref_genes, top_k=20)
        # Measured change for the same genes the chart lists, including real zeros.
        focus = [row["gene"] for row in abn]
        row_of = {name: i for i, name in enumerate(self.lib_names)}
        for row in ranked:
            i = row_of.get(row["gene"])
            row["deltas"] = (
                []
                if i is None
                else deltas_for_genes(self.measured_deltas[i], self.lib_gene_names, focus)
            )
        heat = heatmap_matrix(
            self.measured_deltas,
            self.lib_names,
            self.lib_gene_names,
            ranked,
            query_aligned=aligned,
            control=self.lib_control,
            n_overlap=n_overlap,
        )
        return {
            "matches": ranked,
            "abnormal": abn,
            "heatmap": heat,
            "n_genes_aligned": n_overlap,
            "disclaimer": (
                "Match weights compare this cell's expression change with predicted profiles from "
                "celljepa_s5_s2bjepa_r1 (blood-JEPA, then Cell-JEPA fine-tune with contrastive loss). "
                "They are not clinical probabilities. "
                "The heatmap shows measured Replogle K562 responses for the closest signatures."
            ),
            "ckpt": self.ckpt_info,
        }

    @modal.asgi_app()
    def api(self):
        # Build routes with no postponed local annotations: with
        # ``from __future__ import annotations``, nested FastAPI handlers cannot resolve
        # types imported only inside this method (they become query params).
        return _make_asgi(self)


@app.local_entrypoint()
def main(stage: str = "prep", force: bool = False):
    if stage == "prep":
        print(json.dumps(prep.remote(force=force), indent=2))
    elif stage == "samples":
        rows = sample_files.remote()
        out_dir = ROOT / "test_set_examples"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.csv"):
            old.unlink()
        manifest = []
        for row in rows:
            path = out_dir / row["filename"]
            path.write_text(row["csv"])
            manifest.append({
                "file": row["filename"],
                "label": row["label"],
                "true_gene": row["true_gene"],
                "split": row.get("split"),
            })
            print(f"wrote {path}  true_gene={row['true_gene']}  split={row.get('split')}")
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        raise SystemExit(f"unknown stage {stage!r}; use prep or samples")