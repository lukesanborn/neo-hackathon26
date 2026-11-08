#!/usr/bin/env python3
"""Local Blood JEPA UI server with mock APIs (no Modal, no GPU, no model).

    python website/dev_server.py
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import math
import random
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765

# Compact synthetic gene panel so the page stays snappy while you design.
GENES = [
    "HBZ", "HBA1", "HBG1", "HBG2", "FARSB", "DNAJC19", "DHDDS", "SAMM50",
    "EIF2B4", "TRAPPC6A", "MDK", "HIST1H1C", "CLCA1", "PHGDH", "SERPINF1",
    "ERP29", "RPS21", "FXYD5", "MS4A3", "PRSS57", "TIMP1", "RPS29", "PTMA",
    "B2M", "ACTB", "GAPDH", "MYC", "TP53", "BCL2", "CD34", "GATA1", "SPI1",
    "RUNX1", "IKZF1", "PAX5", "IRF4", "STAT3", "JAK2", "KRAS", "EGFR",
]

LIBRARY = ["FARSB", "DNAJC19", "DHDDS", "SAMM50", "EIF2B4", "TRAPPC6A", "MDK", "PHGDH"]


def _ref_values(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [round(2.0 + 2.5 * rng.random(), 4) for _ in GENES]


REF = _ref_values(1)


def _profile(seed: int, bump: dict[str, float] | None = None) -> list[float]:
    rng = random.Random(seed)
    vals = [round(v + rng.uniform(-0.15, 0.15), 4) for v in REF]
    for g, d in (bump or {}).items():
        i = GENES.index(g)
        vals[i] = round(vals[i] + d, 4)
    return vals


def _example(eid: str, label: str, hint: str, seed: int, bump: dict[str, float] | None = None) -> dict:
    return {
        "id": eid,
        "label": label,
        "hint": hint,
        "genes": GENES,
        "values": _profile(seed, bump),
    }


def _blood_bins(seed: int, bump: dict[str, float] | None = None) -> tuple[list[str], list[float]]:
    """Sparse within-cell bins (1..50) so the heatmap query-scale gate drops the upload column."""
    rng = random.Random(seed)
    # Cover only a fraction of the mock panel, like a real census cell vs Adamson genes.
    keep = [g for g in GENES if rng.random() < 0.35]
    if bump:
        for g in bump:
            if g not in keep:
                keep.append(g)
    vals = []
    for g in keep:
        base = 8 + int(rng.random() * 35)  # 8..42
        if bump and g in bump:
            base = max(1, min(50, int(base + bump[g])))
        vals.append(float(base))
    return keep, vals


def _blood_example(eid: str, label: str, hint: str, seed: int, bump: dict[str, float] | None = None) -> dict:
    genes, values = _blood_bins(seed, bump)
    return {"id": eid, "label": label, "hint": hint, "genes": genes, "values": values}


EXAMPLES = {
    "examples": [
        _blood_example(
            "blood_cell_1",
            "Blood census cell 1",
            "Mock blood cell (bin values 1..50)",
            11,
            {"HBZ": 12, "HBA1": -8},
        ),
        _blood_example(
            "blood_cell_2",
            "Blood census cell 2",
            "Mock blood cell (bin values 1..50)",
            22,
            {"MDK": 10, "MYC": 6},
        ),
        _example(
            "hidden_knockdown",
            "Mystery knockdown signature",
            "Control shifted by a mock FARSB-like signature",
            33,
            {"FARSB": -1.1, "HBZ": -1.2, "HBA1": 0.8, "TRAPPC6A": 0.55},
        ),
        _example("control", "K562 control mean", "Unperturbed mock control", 1),
    ]
}

REFERENCE = {
    "source": "Mock healthy reference (local UI only)",
    "genes": GENES,
    "values": REF,
}

def _lib_deltas() -> dict[str, list[float]]:
    """Each knockdown moves a short gene block. The rest is background below 0.15."""
    rng = random.Random(7)
    out = {g: [0.0] * len(GENES) for g in LIBRARY}
    for i, g in enumerate(LIBRARY):
        start = (i * 4) % len(GENES)
        for k in range(5):
            gi = (start + k) % len(GENES)
            sign = 1.0 if k < 3 else -1.0
            out[g][gi] = round(sign * (0.42 + 0.16 * (4 - k)), 3)
        for gi in range(len(GENES)):
            if out[g][gi] == 0.0:
                out[g][gi] = round(rng.uniform(-0.04, 0.04), 3)
    return out


_LIB_DELTAS = _lib_deltas()


def _mock_heatmap(matches: list[dict], aligned: list[float], n_overlap: int) -> dict:
    columns = [{"gene": m["gene"], "cosine": m["cosine"]} for m in matches]
    selected: list[str] = []
    seen: set[str] = set()
    min_abs = 0.15
    for m in matches:
        ranked_genes = sorted(
            zip(GENES, _LIB_DELTAS[m["gene"]]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        taken = 0
        for gene, delta in ranked_genes:
            if abs(delta) < min_abs:
                break
            if gene in seen:
                continue
            seen.add(gene)
            selected.append(gene)
            taken += 1
            if taken >= 14 or len(selected) >= 48:
                break
        if len(selected) >= 48:
            break
    # Group by the matched column where |change| peaks, then by signed change.
    def row_vec(gene: str) -> list[float]:
        return [_LIB_DELTAS[m["gene"]][GENES.index(gene)] for m in matches]

    groups: dict[int, list[str]] = {}
    for gene in selected:
        vec = row_vec(gene)
        peak = max(range(len(vec)), key=lambda i: abs(vec[i]))
        groups.setdefault(peak, []).append(gene)
    ordered: list[str] = []
    for col in range(len(matches)):
        band = groups.get(col, [])
        band.sort(key=lambda gene: -row_vec(gene)[col])
        ordered.extend(band)
    selected = ordered
    values = [row_vec(gene) for gene in selected]
    # Scale gate: full-panel mock uploads use log1p-ish values; blood cells use bins 1..50.
    query = None
    coverage = n_overlap / max(len(GENES), 1)
    present = [v for v in aligned if v != 0.0]
    if coverage >= 0.5 and present and max(abs(v) for v in present) <= 12.0:
        query = [round(aligned[GENES.index(g)] - REF[GENES.index(g)], 4) for g in selected]
    return {
        "genes": selected,
        "columns": columns,
        "values": values,
        "unit": "log1p change vs K562 control",
        "query": query,
    }


def _mock_infer(genes: list[str], values: list[float], top_k: int = 8) -> dict:
    idx = {g: i for i, g in enumerate(genes)}
    aligned = [float(values[idx[g]]) if g in idx else 0.0 for g in GENES]
    n_overlap = sum(1 for g in GENES if g in idx)
    delta = [a - r for a, r in zip(aligned, REF)]
    abn = sorted(
        [{"gene": g, "delta": d, "query": a, "ref": r} for g, d, a, r in zip(GENES, delta, aligned, REF)],
        key=lambda x: abs(x["delta"]),
        reverse=True,
    )[:20]

    # Fake ranking: prefer library genes whose name appears among the strongest deltas,
    # with a deterministic score so the mystery example still crowns FARSB.
    strength = {row["gene"]: abs(row["delta"]) for row in abn}
    raw = []
    for g in LIBRARY:
        score = 0.55 + 0.08 * strength.get(g, 0.0)
        if g == "FARSB" and strength.get("FARSB", 0) > 0.5:
            score += 0.35
        raw.append((g, score))
    raw.sort(key=lambda x: -x[1])
    raw = raw[: max(1, min(top_k, len(raw)))]
    # Softmax-ish weights over the shown cards.
    mx = raw[0][1]
    exps = [math.exp((s - mx) / 0.08) for _, s in raw]
    z = sum(exps) or 1.0
    focus = [row["gene"] for row in abn]
    matches = []
    for (g, s), e in zip(raw, exps):
        profile = _LIB_DELTAS[g]
        matches.append({
            "gene": g,
            "cosine": round(min(0.999, 0.92 + 0.07 * (s / (mx or 1))), 4),
            "weight": e / z,
            "deltas": [
                {"gene": gene, "delta": profile[GENES.index(gene)]}
                for gene in focus
            ],
        })
    heat = _mock_heatmap(matches, aligned, n_overlap)
    # Simulate a short "inference" pause so the loader is visible while designing.
    time.sleep(0.7)
    return {
        "matches": matches,
        "abnormal": abn,
        "heatmap": heat,
        "n_tokens": min(600, len([v for v in values if v])),
        "disclaimer": (
            "LOCAL MOCK: match weights are fake. Deployed Modal inference is not used here. "
            "Heatmap values are synthetic measured-style deltas."
        ),
        "ckpt": {"run_dir": "local-mock", "ckpt": "none", "vocab": "none"},
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {args[0]} {args[1]} {args[2]}")

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"ok": True, "n_library": len(LIBRARY), "mock": True})
        if path == "/api/reference":
            return self._json(200, REFERENCE)
        if path == "/api/examples":
            return self._json(200, EXAMPLES)
        if path == "/":
            self.path = "/index.html"
        elif path.startswith("/static/"):
            # Modal mounts the demo dir at /static; locally the files live at the root.
            self.path = path[len("/static"):] or "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/infer":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"detail": "invalid JSON"})
        genes = payload.get("genes")
        values = payload.get("values")
        top_k = int(payload.get("top_k", 8))
        if not isinstance(genes, list) or not isinstance(values, list) or not genes:
            return self._json(400, {"detail": "genes and values must be non-empty lists"})
        if len(genes) != len(values):
            return self._json(400, {"detail": "genes and values must have the same length"})
        return self._json(200, _mock_infer(genes, [float(v) for v in values], top_k=top_k))


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Blood JEPA mock API")
    print(f"  http://{HOST}:{PORT}/api/health")
    print("  UI: cd website/web && npm run dev  → http://127.0.0.1:5173")
    print("  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
