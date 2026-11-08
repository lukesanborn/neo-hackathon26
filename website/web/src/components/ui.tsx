/** Shared Tailwind class bundles for repeated controls. */

export const btn =
  "rounded-lg border border-border bg-white px-3 py-2 text-[15px] text-charcoal transition";

export const btnPrimary =
  "rounded-lg border border-berry bg-berry px-3 py-2 text-[15px] font-semibold text-white transition hover:border-[#c44a7f] hover:bg-[#c44a7f] disabled:cursor-not-allowed disabled:opacity-45";

export const pane =
  "min-h-0 min-w-0 rounded-[10px] border border-border bg-panel p-3.5";

export const paneCandidate =
  `${pane} flex h-full flex-col border-berry shadow-[0_0_0_1px_rgba(175,59,110,0.18)]`;

export const paneLabel =
  "mb-2 font-sans text-xs font-bold tracking-wider text-muted uppercase";
