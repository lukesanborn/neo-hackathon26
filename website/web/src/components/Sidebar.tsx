import { relativeTime } from "../lib/recent";
import type { RecentUpload } from "../types";
import { btn } from "./ui";

export function Sidebar({
  items,
  activeId,
  onSelect,
  onClear,
}: {
  items: RecentUpload[];
  activeId: string | null;
  onSelect: (item: RecentUpload) => void;
  onClear: () => void;
}) {
  return (
    <aside
      className="flex max-h-[calc(100dvh-82px)] min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border border-border bg-white px-2 py-2.5"
      aria-label="Recent uploads"
    >
      <h2 className="mx-1 mb-2 font-sans text-xs font-bold tracking-wider text-muted uppercase">
        Recent
      </h2>
      {items.length === 0 ? (
        <p className="m-0 shrink-0 px-1 py-2 text-[13px] text-muted">No uploads yet.</p>
      ) : (
        <ul className="m-0 flex min-h-0 flex-1 list-none flex-col justify-start gap-1 overflow-x-hidden overflow-y-auto p-0">
          {items.map((it) => (
            <li key={it.id} className="min-w-0 shrink-0">
              <button
                type="button"
                className={`box-border w-full max-w-full overflow-hidden rounded-lg border px-2.5 py-2 text-left font-inherit text-charcoal ${
                  it.id === activeId
                    ? "border-berry bg-berry/5"
                    : "border-transparent bg-panel hover:border-[#c5e4f5] hover:bg-[#eef8fd]"
                }`}
                onClick={() => onSelect(it)}
              >
                <span className="block max-w-full truncate text-sm font-bold text-charcoal">
                  {it.label}
                </span>
                <span className="mt-0.5 block truncate text-xs text-muted">
                  {relativeTime(it.ts)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-auto shrink-0 border-t border-border pt-2">
        <button type="button" className={`${btn} w-full text-[13px]`} onClick={onClear}>
          Clear
        </button>
      </div>
    </aside>
  );
}
