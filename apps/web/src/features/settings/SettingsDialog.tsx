import { useId } from "react";
import { Button } from "../../components/ui";
import { useDialogFocus } from "../../components/use-dialog-focus";
import { SettingsPanel } from "./SettingsPanel";

export function SettingsDialog({ onClose }: { onClose: () => void }) {
  const { dialogRef, onKeyDown } = useDialogFocus(onClose);
  const titleId = useId();

  return (
    <div
      className="animate-fade fixed inset-0 z-50 flex items-center justify-center bg-scrim p-3 sm:p-6"
      onClick={onClose}
    >
      {/* The dialog itself no longer scrolls — the body does. A modal whose own
       * header scrolls away leaves no visible way out mid-form. */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        className="animate-enter flex h-full max-h-[46rem] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-border bg-bg shadow-overlay outline-none"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-border px-5 py-3.5">
          <h2 id={titleId} className="text-base font-semibold">
            Settings
          </h2>
          <Button onClick={onClose}>Close</Button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          <SettingsPanel />
        </div>
      </div>
    </div>
  );
}
