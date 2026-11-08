export function AlphafoldBadge({
  compact = false,
  className = "",
}: {
  compact?: boolean;
  className?: string;
}) {
  if (compact) {
    return (
      <span
        className={`inline-flex items-center gap-1 rounded bg-[#0A2540] px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-white ${className}`}
        title="Structure predicted by AlphaFold (DeepMind / EMBL-EBI)"
      >
        <img src="/alphafold-mark.svg" alt="" className="h-3.5 w-auto" />
      </span>
    );
  }
  return (
    <a
      href="https://alphafold.ebi.ac.uk/"
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center gap-2 no-underline ${className}`}
      title="Open AlphaFold Protein Structure Database"
    >
      <img src="/alphafold-mark.svg" alt="AlphaFold" className="h-6 w-auto" />
      <span className="text-[12px] font-semibold text-muted">
        Predicted by AlphaFold
      </span>
    </a>
  );
}
