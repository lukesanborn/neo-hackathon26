import { useRef, useState } from "react";
import type { GeneDelta } from "../types";
import { AbnPreview } from "./AbnPreview";
import { btnPrimary } from "./ui";

export function EmptyUpload({ onFile }: { onFile: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 overflow-auto p-5 text-center">
      <img
        className="h-16 w-16 rounded-[14px] bg-[#111111] object-contain p-1.5"
        src="/logo.png"
        alt=""
      />
      <h2 className="m-0 max-w-md truncate font-sans text-[26px] font-bold text-charcoal">
        Load a cell state
      </h2>
      <p className="m-0 text-[15px] text-muted">CSV with columns gene, value.</p>

      <div
        className={`flex w-full max-w-[420px] items-center justify-center gap-1.5 rounded-[14px] border border-dashed px-5 py-7 text-[15px] ${
          over
            ? "border-spring bg-spring/15 text-charcoal"
            : "border-[#9fc7db] bg-icy/30 text-muted"
        }`}
        onDragOver={(ev) => {
          ev.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(ev) => {
          ev.preventDefault();
          setOver(false);
          const f = ev.dataTransfer.files?.[0];
          if (f) onFile(f);
        }}
      >
        Drop a CSV here, or{" "}
        <label className="file cursor-pointer font-semibold text-berry underline-offset-2 hover:underline">
          choose a file
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(ev) => {
              const f = ev.target.files?.[0];
              if (f) onFile(f);
              ev.target.value = "";
            }}
          />
        </label>
      </div>
    </div>
  );
}

export function ReadyStage({
  fileName,
  nGenes,
  previewAbn,
  onRun,
  running,
}: {
  fileName: string;
  nGenes: number;
  previewAbn: GeneDelta[];
  onRun: () => void;
  running: boolean;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 overflow-auto p-5 text-center">
      <img
        className="h-16 w-16 rounded-[14px] bg-[#111111] object-contain p-1.5"
        src="/logo.png"
        alt=""
      />
      <h2 className="m-0 max-w-md truncate font-sans text-[26px] font-bold text-charcoal">
        {fileName}
      </h2>
      <p className="m-0 text-[15px] text-muted">{nGenes.toLocaleString()} genes loaded</p>
      {previewAbn.length > 0 ? (
        <div className="w-full max-w-[720px] text-left">
          <AbnPreview rows={previewAbn} />
        </div>
      ) : null}
      <button
        className={`${btnPrimary} mt-2 px-5.5 py-3 text-base`}
        type="button"
        disabled={running}
        onClick={onRun}
      >
        Estimate gene perturbations
      </button>
    </div>
  );
}
