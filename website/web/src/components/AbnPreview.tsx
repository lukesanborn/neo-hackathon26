import { ABN_ROWS } from "../lib/abnormal";
import type { GeneDelta } from "../types";
import { SingleBars } from "./DeltaBars";

export function AbnPreview({ rows }: { rows: GeneDelta[] }) {
  const top = rows.slice(0, ABN_ROWS);
  if (!top.length) return null;
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="m-0 font-sans text-[13px] font-bold tracking-wider text-muted uppercase">
          Cell abnormalities
        </h3>
        <div className="flex gap-2.5 text-[13px] font-semibold text-muted">
          <span className="before:mr-1 before:inline-block before:h-2 before:w-2 before:rounded-sm before:bg-up-fill before:content-['']">
            Higher
          </span>
          <span className="before:mr-1 before:inline-block before:h-2 before:w-2 before:rounded-sm before:bg-down before:content-['']">
            Lower
          </span>
        </div>
      </div>
      <SingleBars rows={top} />
    </div>
  );
}
