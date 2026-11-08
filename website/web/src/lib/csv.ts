import type { CellProfile } from "../types";

export function parseCsv(text: string): CellProfile {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) throw new Error("empty CSV");
  const start = /gene/i.test(lines[0]) ? 1 : 0;
  const genes: string[] = [];
  const values: number[] = [];
  for (let i = start; i < lines.length; i++) {
    const parts = lines[i].split(",").map((s) => s.trim());
    if (parts.length < 2) continue;
    const v = Number(parts[1]);
    if (!Number.isFinite(v)) continue;
    genes.push(parts[0]);
    values.push(v);
  }
  if (!genes.length) throw new Error("no gene,value rows found");
  return { genes, values };
}
