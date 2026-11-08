import type { HeatmapPayload } from "../types";
import { pane, paneLabel } from "./ui";

/** Same green-up / berry-down pair as the bar charts. */
const SCALE: [number, number, number][] = [
  [175, 59, 110],
  [196, 146, 177],
  [247, 251, 254],
  [125, 255, 192],
  [15, 159, 92],
];

const TOP_GENES = 10;

function mix(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function cellColor(v: number, lim: number): string {
  const t = Math.max(-1, Math.min(1, v / lim));
  const x = ((t + 1) / 2) * (SCALE.length - 1);
  const i = Math.min(SCALE.length - 2, Math.floor(x));
  const f = x - i;
  const c0 = SCALE[i];
  const c1 = SCALE[i + 1];
  return `rgb(${mix(c0[0], c1[0], f)}, ${mix(c0[1], c1[1], f)}, ${mix(c0[2], c1[2], f)})`;
}

function ink(v: number, lim: number): string {
  return Math.abs(v) / lim > 0.55 ? "#ffffff" : "#1c1917";
}

function robustLimit(values: number[]): number {
  const abs = values.map((v) => Math.abs(v)).sort((a, b) => a - b);
  if (!abs.length) return 0.5;
  const idx = Math.min(abs.length - 1, Math.floor(0.98 * (abs.length - 1)));
  return Math.max(abs[idx], 0.4);
}

function fmt(v: number): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}`;
}

/** Ten genes with the largest |change| in the knockdown on screen, strongest drop first. */
function focusGeneIndexes(data: HeatmapPayload, selectedGene: string): number[] {
  let ci = data.columns.findIndex((c) => c.gene === selectedGene);
  if (ci < 0) ci = 0;
  const ranked = data.genes.map((_, ri) => ({
    ri,
    delta: data.values[ri]?.[ci] ?? 0,
  }));
  ranked.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.ri - b.ri);
  const top = ranked.slice(0, TOP_GENES);
  top.sort((a, b) => a.delta - b.delta || a.ri - b.ri);
  return top.map((row) => row.ri);
}

function Legend({ lim }: { lim: number }) {
  const gradient = `linear-gradient(to right, rgb(${SCALE[0].join(",")}), rgb(${SCALE[2].join(",")}), rgb(${SCALE[4].join(",")}))`;
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2 text-[13px] font-semibold">
      <span className="text-down">Lower</span>
      <div className="h-2.5 w-36 overflow-hidden rounded-full border border-border" style={{ background: gradient }} />
      <span className="text-up">Higher</span>
      <span className="font-normal text-muted">
        {fmt(-lim)} to {fmt(lim)}
      </span>
    </div>
  );
}

type Series = { key: string; label: string; title: string; sub: string; values: number[] };

function Matrix({
  genes,
  series,
  lim,
  selectedKey,
}: {
  genes: string[];
  series: Series[];
  lim: number;
  selectedKey: string;
}) {
  const cols = `minmax(84px, 112px) repeat(${series.length}, minmax(64px, 1fr))`;
  return (
    <div className="overflow-x-auto">
      <div
        className="grid min-w-[560px] gap-px overflow-hidden rounded-lg bg-border"
        style={{ gridTemplateColumns: cols }}
      >
        <div className="bg-panel" />
        {series.map((col) => {
          const active = col.key === selectedKey;
          return (
            <div
              key={col.key}
              title={col.title}
              className={`flex min-h-14 flex-col items-center justify-end bg-panel px-1 py-2 text-center ${
                active ? "shadow-[inset_0_-3px_0_0_var(--color-berry)]" : ""
              }`}
            >
              <span
                className={`text-[13px] leading-tight font-bold ${active ? "text-berry" : "text-charcoal"}`}
              >
                {col.label}
              </span>
              <span className={`mt-0.5 text-[11px] font-semibold ${active ? "text-berry" : "text-muted"}`}>
                {col.sub}
              </span>
            </div>
          );
        })}
        {genes.map((gene, ri) => (
          <div key={gene} className="contents">
            <div
              title={gene}
              className="flex items-center justify-end bg-panel pr-2 text-right text-[13px] font-bold text-charcoal"
            >
              <span className="truncate">{gene}</span>
            </div>
            {series.map((col) => {
              const v = col.values[ri] ?? 0;
              const active = col.key === selectedKey;
              return (
                <div
                  key={col.key}
                  title={`${gene}, ${col.label}, ${fmt(v)}`}
                  className={`flex min-h-12 items-center justify-center px-1 text-[13px] font-bold tabular-nums ${
                    active ? "shadow-[inset_0_0_0_2px_var(--color-berry)]" : ""
                  }`}
                  style={{ background: cellColor(v, lim), color: ink(v, lim) }}
                >
                  {fmt(v)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Heatmap({
  data,
  selectedGene,
}: {
  data: HeatmapPayload;
  selectedGene: string;
}) {
  if (!data.genes.length || !data.columns.length) return null;

  const focus = focusGeneIndexes(data, selectedGene);
  const genes = focus.map((ri) => data.genes[ri]);
  const hasQuery = Array.isArray(data.query) && data.query.length === data.genes.length;
  const query: Series | null = hasQuery
    ? {
        key: "query",
        label: "This cell",
        title: "Uploaded cell, change from the K562 control",
        sub: "upload",
        values: focus.map((ri) => (data.query as number[])[ri]),
      }
    : null;
  const columns: Series[] = data.columns.map((c, ci) => ({
    key: c.gene,
    label: c.gene,
    title: `${c.gene}, cosine ${c.cosine.toFixed(3)}`,
    sub: c.cosine.toFixed(2),
    values: focus.map((ri) => data.values[ri][ci]),
  }));
  const selected = columns.find((c) => c.key === selectedGene) ?? columns[0];
  const rest = columns.filter((c) => c.key !== selected.key);
  const series = [...(query ? [query] : []), selected, ...rest];
  const shown = series.flatMap((col) => col.values);
  const lim = robustLimit(shown);

  return (
    <section className={`${pane} min-w-0 max-w-full`}>
      <p className={paneLabel}>Measured response</p>
      <Legend lim={lim} />
      <Matrix genes={genes} series={series} lim={lim} selectedKey={selected.key} />
      {hasQuery ? null : (
        <p className="m-0 mt-2 text-[13px] text-muted">
          This cell is not shown. Its values are not on the K562 log-expression scale.
        </p>
      )}
    </section>
  );
}
