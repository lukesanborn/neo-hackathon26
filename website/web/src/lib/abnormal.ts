import type { CellProfile, GeneDelta, Reference } from "../types";

export const ABN_ROWS = 8;

export function computeAbnormal(cell: CellProfile, reference: Reference): GeneDelta[] {
  const idx = Object.fromEntries(cell.genes.map((g, i) => [g, i]));
  const deltas = reference.genes.map((g, i) => {
    const q = idx[g] === undefined ? 0 : cell.values[idx[g]];
    return { gene: g, delta: q - reference.values[i] };
  });
  deltas.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return deltas;
}

export function alignedCompareRows(queryRows: GeneDelta[], matchDeltas: GeneDelta[]) {
  const qMap = Object.fromEntries((queryRows || []).map((d) => [d.gene, d.delta]));
  const mMap = Object.fromEntries((matchDeltas || []).map((d) => [d.gene, d.delta]));
  const genes: string[] = [];
  for (const d of (queryRows || []).slice(0, ABN_ROWS)) {
    if (!genes.includes(d.gene)) genes.push(d.gene);
  }
  for (const d of matchDeltas || []) {
    if (genes.length >= ABN_ROWS) break;
    if (!genes.includes(d.gene)) genes.push(d.gene);
  }
  const rows = genes.map((g) => ({
    gene: g,
    // Missing is not a measured zero. The knockdown payload only includes genes
    // we actually have a change-from-control for.
    left: qMap[g] ?? null,
    right: mMap[g] ?? null,
  }));
  const magnitudes = rows.flatMap((d) =>
    [d.left, d.right].filter((v): v is number => v != null).map((v) => Math.abs(v)),
  );
  const lim = Math.max(...magnitudes, 1e-6);
  return { rows, lim };
}
