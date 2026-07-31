import { useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent } from "react";

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Keyboard/focus behaviour shared by the app's inline confirmation dialogs.
 * The dialog is conditionally mounted, so mounting is the open transition and
 * unmounting is the close transition.
 */
export function useDialogFocus(onClose: () => void, active = true) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active) return;
    previouslyFocused.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const frame = requestAnimationFrame(() => {
      dialogRef.current
        ?.querySelector<HTMLElement>(FOCUSABLE)
        ?.focus();
    });
    return () => {
      cancelAnimationFrame(frame);
      previouslyFocused.current?.focus();
    };
  }, [active]);

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = [
      ...(dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []),
    ];
    if (focusable.length === 0) {
      event.preventDefault();
      dialogRef.current?.focus();
      return;
    }
    const first = focusable[0]!;
    const last = focusable.at(-1)!;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return { dialogRef, onKeyDown };
}
