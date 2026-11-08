import { useEffect, useRef, useState } from "react";
import { loadAlphaFold, type StructureViewer as Viewer } from "../lib/molstar";
import { pane, paneLabel } from "./ui";

export function StructureViewer({ gene }: { gene: string }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const [show, setShow] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const mount = mountRef.current;
    if (!mount || !gene) return;

    setShow(false);
    setError("");

    (async () => {
      try {
        if (viewerRef.current) {
          try {
            await viewerRef.current.plugin.clear();
          } catch {
            /* ignore */
          }
          viewerRef.current = null;
        }
        const result = await loadAlphaFold(gene, mount, { zoom: 0.5 });
        if (cancelled) {
          try {
            await result.viewer.plugin.clear();
          } catch {
            /* ignore */
          }
          return;
        }
        viewerRef.current = result.viewer;
        setShow(true);
      } catch {
        if (!cancelled) {
          setShow(false);
          setError("No AlphaFold structure available for this gene.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [gene]);

  return (
    <div className={`${pane} flex h-full min-h-0 flex-col`}>
      <p className={paneLabel}>AlphaFold predicted structure</p>
      {error ? <div className="mb-1.5 text-[13px] text-muted">{error}</div> : null}
      <div className={`min-h-0 flex-1 flex-col ${show ? "flex" : error ? "hidden" : "flex"}`}>
        <div
          className="structure-view"
          ref={mountRef}
          role="region"
          aria-label={`AlphaFold predicted structure for ${gene}`}
        />
      </div>
    </div>
  );
}
