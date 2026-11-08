import { useEffect, useState } from "react";
import { fetchGeneInfo, geneCardsUrl } from "../lib/genes";
import type { GeneInfo, Match } from "../types";
import { MiniStructure } from "./MiniStructure";

export function CandidatePick({
  matches,
  selectedIdx,
  onSelect,
}: {
  matches: Match[];
  selectedIdx: number;
  onSelect: (idx: number) => void;
}) {
  const [infoByGene, setInfoByGene] = useState<Record<string, GeneInfo>>({});

  useEffect(() => {
    let cancelled = false;
    for (const m of matches) {
      const gene = m.gene;
      fetchGeneInfo(gene)
        .then((info) => {
          if (cancelled) return;
          setInfoByGene((prev) => ({ ...prev, [gene]: info }));
        })
        .catch(() => {
          /* leave card without UniProt extras */
        });
    }
    return () => {
      cancelled = true;
    };
  }, [matches]);

  // Cap interactive 3D previews so we don't spawn too many Mol* instances.
  const previewLimit = Math.min(matches.length, 6);

  return (
    <div className="min-w-0 rounded-[10px] border border-border bg-panel p-2.5">
      <h3 className="mb-2 font-sans text-xs font-bold tracking-wider text-muted uppercase">
        Other candidates
      </h3>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-2">
        {matches.map((m, i) => {
          const w = Math.round((m.weight || 0) * 1000) / 10;
          const active = i === selectedIdx;
          const info = infoByGene[m.gene];
          const showProtein = i < previewLimit;
          return (
            <button
              key={m.gene + i}
              type="button"
              data-candidate-idx={i}
              className={`flex min-w-0 flex-col overflow-hidden rounded-lg border text-left transition ${
                active
                  ? "border-berry bg-berry/5"
                  : "border-transparent bg-white hover:border-[#c5e4f5] hover:bg-[#eef8fd]"
              }`}
              onClick={() => onSelect(i)}
            >
              <div className="h-[120px] w-full shrink-0 overflow-hidden border-b border-border">
                {showProtein ? (
                  <MiniStructure gene={m.gene} />
                ) : (
                  <div className="flex h-full items-center justify-center bg-[#111318] text-[11px] text-muted">
                    Select to view structure
                  </div>
                )}
              </div>
              <div className="min-w-0 flex-1 p-2.5">
                <div className="flex items-baseline gap-1.5">
                  <span
                    className={`truncate text-sm font-bold ${active ? "text-berry" : "text-charcoal"}`}
                  >
                    #{i + 1} {m.gene}
                  </span>
                  <span className="shrink-0 text-xs font-semibold text-berry">{w}%</span>
                </div>
                <div className="mt-0.5 truncate text-xs font-semibold text-charcoal">
                  {info?.proteinName || "Loading…"}
                </div>
                {info?.function ? (
                  <p className="mt-1 line-clamp-2 text-[11px] leading-snug text-charcoal/80">
                    {info.function}
                  </p>
                ) : null}
                <a
                  className="mt-1.5 inline-block text-[11px] font-semibold text-berry hover:underline"
                  href={geneCardsUrl(m.gene)}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(ev) => ev.stopPropagation()}
                >
                  GeneCards
                </a>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
