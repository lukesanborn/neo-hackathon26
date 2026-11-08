import { useCallback, useEffect, useRef, useState } from "react";
import { fetchReference, runInfer, wakeHealth } from "./api";
import { CandidatePick } from "./components/CandidatePick";
import { CompareView } from "./components/CompareView";
import { Loader } from "./components/Loader";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { EmptyUpload, ReadyStage } from "./components/UploadStage";
import { computeAbnormal } from "./lib/abnormal";
import { parseCsv } from "./lib/csv";
import {
  loadRecent,
  rememberUpload,
  saveRecent,
  updateRecentHit,
} from "./lib/recent";
import type {
  CellProfile,
  GeneDelta,
  HeatmapPayload,
  Match,
  RecentUpload,
  Reference,
  StageMode,
} from "./types";
export default function App() {
  const [reference, setReference] = useState<Reference | null>(null);
  const [cell, setCell] = useState<CellProfile | null>(null);
  const [cellLabel, setCellLabel] = useState("");
  const [recent, setRecent] = useState<RecentUpload[]>(() => loadRecent());
  const [activeRecentId, setActiveRecentId] = useState<string | null>(null);
  const [previewAbn, setPreviewAbn] = useState<GeneDelta[]>([]);
  const [abnormal, setAbnormal] = useState<GeneDelta[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [heatmap, setHeatmap] = useState<HeatmapPayload | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [stage, setStage] = useState<StageMode>("idle");
  const [status, setStatus] = useState("Estimating gene perturbations…");
  const [err, setErr] = useState("");
  const [coldHintShown, setColdHintShown] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await wakeHealth();
        const ref = await fetchReference();
        if (cancelled) return;
        setReference(ref);
      } catch (e) {
        if (!cancelled) setErr(`Failed to load demo data: ${(e as Error).message || e}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const applyCell = useCallback(
    (
      next: CellProfile,
      label: string,
      opts: { skipHistory?: boolean; recentId?: string } = {},
    ) => {
      setCell(next);
      setCellLabel(label || "Uploaded");
      setErr("");
      setStage("idle");
      setMatches([]);
      setHeatmap(null);
      setSelectedIdx(0);
      if (reference) {
        const deltas = computeAbnormal(next, reference);
        setPreviewAbn(deltas);
        setAbnormal(deltas);
      } else {
        setPreviewAbn([]);
      }
      if (!opts.skipHistory) {
        const entry = rememberUpload(label || "Uploaded", next);
        setActiveRecentId(entry.id);
        setRecent(loadRecent());
      } else if (opts.recentId) {
        setActiveRecentId(opts.recentId);
        setRecent(loadRecent());
      }
    },
    [reference],
  );

  const resetToEmpty = () => {
    setCell(null);
    setCellLabel("");
    setPreviewAbn([]);
    setAbnormal([]);
    setMatches([]);
    setHeatmap(null);
    setSelectedIdx(0);
    setActiveRecentId(null);
    setErr("");
    setStage("idle");
  };

  const onFile = async (file: File) => {
    try {
      applyCell(parseCsv(await file.text()), file.name);
    } catch (e) {
      setErr((e as Error).message || String(e));
    }
  };

  const onUploadNew = (file?: File) => {
    if (file) {
      void onFile(file);
      return;
    }
    resetToEmpty();
  };

  const onRecent = (item: RecentUpload) => {
    applyCell(
      { genes: item.genes, values: item.values },
      item.label,
      { skipHistory: true, recentId: item.id },
    );
  };

  const onClearRecent = () => {
    saveRecent([]);
    setRecent([]);
    setActiveRecentId(null);
  };

  const onRun = async () => {
    if (!cell) return;
    setErr("");
    setStage("loading");
    const t0 = performance.now();
    setStatus(
      coldHintShown
        ? "Estimating gene perturbations…"
        : "Starting the model (cold start can take a minute)…",
    );
    const timer = window.setInterval(() => {
      const s = (performance.now() - t0) / 1000;
      if (s > 8 && !coldHintShown) {
        setStatus("Starting the model… still waking the GPU");
      } else if (s > 3 && coldHintShown) {
        setStatus("Estimating gene perturbations…");
      }
    }, 500);
    try {
      setStatus("Estimating gene perturbations…");
      const body = await runInfer(cell);
      setColdHintShown(true);
      const abn = body.abnormal?.length ? body.abnormal : abnormal;
      setAbnormal(abn);
      setMatches(body.matches || []);
      setHeatmap(body.heatmap || null);
      setSelectedIdx(0);
      setStage("results");
      const top = body.matches?.[0]?.gene || null;
      setRecent(updateRecentHit(activeRecentId, top));
    } catch (e) {
      setErr((e as Error).message || String(e));
      setStage("idle");
      if (cell && reference) setPreviewAbn(computeAbnormal(cell, reference));
    } finally {
      window.clearInterval(timer);
    }
  };

  const selected = matches[selectedIdx];
  const showEmpty = stage === "idle" && !cell;
  const showReady = stage === "idle" && !!cell;

  const results = stage === "results" && !!selected;
  const selectedIdxRef = useRef(selectedIdx);
  selectedIdxRef.current = selectedIdx;

  useEffect(() => {
    if (!results || matches.length < 2) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      if (ev.altKey || ev.metaKey || ev.ctrlKey) return;
      const target = ev.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (
          tag === "INPUT" ||
          tag === "TEXTAREA" ||
          tag === "SELECT" ||
          target.isContentEditable
        ) {
          return;
        }
      }
      ev.preventDefault();
      const delta = ev.key === "ArrowLeft" ? -1 : 1;
      const next = selectedIdxRef.current + delta;
      if (next < 0 || next >= matches.length) return;
      selectedIdxRef.current = next;
      if (document.activeElement instanceof HTMLElement) {
        document.activeElement.blur();
      }
      window.scrollTo(0, 0);
      setSelectedIdx(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [results, matches.length]);

  return (
    <div className="flex min-h-dvh w-full max-w-full flex-col overflow-x-clip">
      <header className="sticky top-0 z-20 flex min-h-[58px] items-center gap-3 border-b border-border bg-white px-4 py-2.5">
        <img
          className="h-10 w-10 shrink-0 rounded-lg bg-[#111111] object-contain p-0.5"
          src="/logo.png"
          alt="blood JEPA logo"
        />
        <h1 className="m-0 whitespace-nowrap font-sans text-[22px] font-bold tracking-wide text-charcoal">
          blood JEPA
        </h1>
        <Toolbar onUploadNew={onUploadNew} disabled={stage === "loading"} />
      </header>

      <div className="grid flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-[200px_minmax(0,1fr)] lg:items-start">
        <div className="lg:sticky lg:top-[70px] lg:max-h-[calc(100dvh-82px)]">
          <Sidebar
            items={recent}
            activeId={activeRecentId}
            onSelect={onRecent}
            onClear={onClearRecent}
          />
        </div>

        <main className="grid min-w-0 gap-2.5">
          {err ? (
            <div className="rounded-lg border border-berry bg-berry/10 px-3 py-2 text-[13px] text-[#7a2148]">
              {err}
            </div>
          ) : null}

          <section
            className={`flex min-w-0 max-w-full flex-col rounded-xl border border-border bg-white p-3.5 ${
              results ? "" : "min-h-[calc(100dvh-90px)]"
            }`}
          >
            {stage === "loading" ? <Loader status={status} /> : null}
            {showEmpty ? (
              <EmptyUpload onFile={onFile} />
            ) : null}
            {showReady && cell ? (
              <ReadyStage
                fileName={cellLabel}
                nGenes={cell.genes.length}
                previewAbn={previewAbn}
                onRun={onRun}
                running={false}
              />
            ) : null}
            {results && selected ? (
              <div className="flex min-w-0 max-w-full flex-col gap-3">
                <CompareView match={selected} abnormal={abnormal} heatmap={heatmap} />
                <CandidatePick
                  matches={matches}
                  selectedIdx={selectedIdx}
                  onSelect={setSelectedIdx}
                />
              </div>
            ) : null}
          </section>
        </main>
      </div>
    </div>
  );
}
