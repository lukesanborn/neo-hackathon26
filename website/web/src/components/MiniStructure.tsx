import { useEffect, useRef, useState } from "react";
import { loadAlphaFold, type StructureViewer } from "../lib/molstar";

/** Compact Mol* AlphaFold preview for candidate cards. */
export function MiniStructure({ gene }: { gene: string }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<StructureViewer | null>(null);
  const [status, setStatus] = useState<"loading" | "ok" | "fail">("loading");

  useEffect(() => {
    let cancelled = false;
    const mount = mountRef.current;
    if (!mount || !gene) return;

    setStatus("loading");
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
        mount.innerHTML = "";
        const { viewer } = await loadAlphaFold(gene, mount, { zoom: 1 });
        if (cancelled) {
          try {
            await viewer.plugin.clear();
          } catch {
            /* ignore */
          }
          return;
        }
        viewerRef.current = viewer;
        setStatus("ok");
      } catch {
        if (!cancelled) setStatus("fail");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [gene]);

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#111318]">
      <div ref={mountRef} className="structure-view !min-h-0 h-full rounded-none border-0" />
      {status === "loading" ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px] text-muted">
          Loading…
        </div>
      ) : null}
      {status === "fail" ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px] text-muted">
          No structure
        </div>
      ) : null}
    </div>
  );
}
