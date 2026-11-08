type SingleRow = { gene: string; delta: number };

type DualRow = { gene: string; left: number | null; right: number | null };

function Track({ delta, lim }: { delta: number; lim: number }) {
  const up = delta >= 0;
  const pct = Math.max(4, (Math.abs(delta) / lim) * 50);
  const left = up ? 50 : 50 - pct;
  return (
    <div className="track">
      <div
        className={`track-fill ${up ? "track-fill-up" : "track-fill-down"}`}
        style={{ left: `${left}%`, width: `${pct}%` }}
      />
    </div>
  );
}

export function SingleBars({ rows, lim }: { rows: SingleRow[]; lim?: number }) {
  const scale = lim || Math.max(...rows.map((d) => Math.abs(d.delta)), 1e-6);
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 sm:gap-x-4 sm:gap-y-2">
      {rows.map((d) => {
        const up = d.delta >= 0;
        return (
          <div className="grid grid-cols-[64px_1fr_48px] items-center gap-2 text-sm" key={d.gene}>
            <div className="truncate text-right font-semibold text-charcoal" title={d.gene}>
              {d.gene}
            </div>
            <Track delta={d.delta} lim={scale} />
            <div
              className={`text-right text-sm font-bold tabular-nums ${up ? "text-up" : "text-down"}`}
            >
              {d.delta >= 0 ? "+" : ""}
              {d.delta.toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function DualBars({
  rows,
  lim,
  leftLabel,
  rightLabel,
}: {
  rows: DualRow[];
  lim: number;
  leftLabel: string;
  rightLabel: string;
}) {
  const cell = (delta: number | null) => {
    if (delta == null) {
      return (
        <div className="grid grid-cols-[1fr_48px] items-center gap-2" aria-hidden="true">
          <div className="h-3.5" />
          <div />
        </div>
      );
    }
    const up = delta >= 0;
    return (
      <div className="grid grid-cols-[1fr_48px] items-center gap-2">
        <Track delta={delta} lim={lim} />
        <div
          className={`text-right text-sm font-bold tabular-nums ${up ? "text-up" : "text-down"}`}
        >
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(2)}
        </div>
      </div>
    );
  };
  return (
    <>
      <div className="mb-2 grid grid-cols-[72px_1fr_1fr] items-end gap-3">
        <span className="text-[13px] font-bold tracking-wider text-muted uppercase">Gene</span>
        <span className="text-center text-[13px] font-bold tracking-wider text-muted uppercase">
          {leftLabel}
        </span>
        <span className="text-center text-[13px] font-bold tracking-wider text-berry uppercase">
          {rightLabel}
        </span>
      </div>
      <div className="grid gap-1.5">
        {rows.map((d) => (
          <div
            className="grid grid-cols-[72px_1fr_1fr] items-center gap-3 text-[15px]"
            key={d.gene}
          >
            <div className="truncate text-right font-semibold text-charcoal" title={d.gene}>
              {d.gene}
            </div>
            {cell(d.left)}
            {cell(d.right)}
          </div>
        ))}
      </div>
    </>
  );
}
