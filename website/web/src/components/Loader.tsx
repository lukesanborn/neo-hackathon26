export function Loader({ status }: { status: string }) {
  return (
    <div className="flex h-full flex-1 flex-col items-center justify-center gap-2.5">
      <div className="orbit" aria-hidden="true">
        <div className="orbit-ring" />
        <img className="orbit-logo" src="/logo.png" alt="" />
        <div className="orbit-dot" />
        <div className="orbit-dot orbit-dot-2" />
        <div className="orbit-dot orbit-dot-3" />
        <div className="orbit-dot orbit-dot-4" />
      </div>
      <div className="text-[15px] font-semibold text-charcoal">{status}</div>
    </div>
  );
}
