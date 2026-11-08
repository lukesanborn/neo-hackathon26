import { useRef } from "react";
import { btnPrimary } from "./ui";

/** Compact header control: start a new upload from the top right. */
export function Toolbar({
  onUploadNew,
  disabled,
}: {
  onUploadNew: (file?: File) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-2.5">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        onChange={(ev) => {
          const f = ev.target.files?.[0];
          if (f) onUploadNew(f);
          else onUploadNew();
          ev.target.value = "";
        }}
      />
      <button
        type="button"
        className={btnPrimary}
        disabled={disabled}
        onClick={() => {
          onUploadNew();
          inputRef.current?.click();
        }}
      >
        Upload new
      </button>
    </div>
  );
}
