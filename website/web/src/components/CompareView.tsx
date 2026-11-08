import { useEffect, useState } from "react";
import { alignedCompareRows } from "../lib/abnormal";
import { fetchGeneInfo, geneCardsUrl, uniprotUrl } from "../lib/genes";
import type { GeneDelta, GeneInfo, HeatmapPayload, Match } from "../types";
import { DualBars } from "./DeltaBars";
import { Heatmap } from "./Heatmap";
import { StructureViewer } from "./StructureViewer";
import { pane, paneCandidate, paneLabel } from "./ui";

export function CompareView({
  match,
  abnormal,
  heatmap,
}: {
  match: Match;
  abnormal: GeneDelta[];
  heatmap: HeatmapPayload | null;
}) {
  const [info, setInfo] = useState<GeneInfo | null>(null);
  const [blurb, setBlurb] = useState("Loading gene summary…");
  const { rows, lim } = alignedCompareRows(abnormal, match.deltas || []);
  const w = Math.round((match.weight || 0) * 1000) / 10;

  useEffect(() => {
    let cancelled = false;
    setInfo(null);
    setBlurb("Loading gene summary…");
    fetchGeneInfo(match.gene)
      .then((g) => {
        if (cancelled) return;
        setInfo(g);
        setBlurb(g.function || "No function summary available.");
      })
      .catch(() => {
        if (!cancelled) {
          setInfo(null);
          setBlurb("Could not load gene summary.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [match.gene]);

  const chips: string[] = [];
  if (info?.length) chips.push(`${info.length} aa`);
  for (const loc of info?.locations || []) chips.push(loc);
  for (const d of info?.diseases || []) chips.push(d);
  for (const a of info?.aliases || []) chips.push(a);

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-2.5">
      <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] lg:items-stretch">
      <div className={paneCandidate}>
        <p className={paneLabel}>Knockdown candidate</p>
        <p className="m-0 mb-1 font-sans text-[28px] font-bold text-berry">{match.gene}</p>
        <p className="m-0 mb-2 text-[15px] font-semibold text-charcoal">
          {info?.proteinName || ""}
        </p>
        <div className="mb-1.5 flex items-baseline gap-2">
          <span className="font-sans text-5xl leading-none font-bold text-berry tabular-nums">
            {w}
          </span>
          <span className="text-[15px] font-semibold text-muted">% match</span>
        </div>
        <div className="mb-2 text-[13px] text-muted">
          cosine {Number(match.cosine).toFixed(3)}
        </div>
        <div className="mb-2 flex flex-wrap gap-2.5 text-[13px]">
          <a
            className="font-semibold text-berry no-underline hover:underline"
            href={geneCardsUrl(match.gene)}
            target="_blank"
            rel="noopener noreferrer"
          >
            GeneCards
          </a>
          {info?.accession && (
            <a
              className="font-semibold text-berry no-underline hover:underline"
              href={uniprotUrl(info.accession)}
              target="_blank"
              rel="noopener noreferrer"
            >
              UniProt
            </a>
          )}
        </div>
        <p className="m-0 mb-2 line-clamp-3 text-sm leading-snug text-charcoal">{blurb}</p>
        <div className="mb-2.5 flex flex-wrap gap-1.5">
          {chips.slice(0, 3).map((c) => (
            <span
              className="rounded-full border border-[#c5e4f5] bg-icy/45 px-2 py-0.5 text-xs font-semibold text-charcoal"
              key={c}
            >
              {c}
            </span>
          ))}
        </div>
      </div>

      <StructureViewer gene={match.gene} />
      </div>

      <div className={pane}>
        <p className={paneLabel}>Change from control</p>
        <DualBars rows={rows} lim={lim} leftLabel="This cell" rightLabel={match.gene} />
      </div>

      {heatmap ? <Heatmap data={heatmap} selectedGene={match.gene} /> : null}
    </div>
  );
}
