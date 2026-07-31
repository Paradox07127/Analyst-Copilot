import { createContext, useContext } from "react";

const SettingsDialogContext = createContext<(() => void) | null>(null);

export const SettingsDialogProvider = SettingsDialogContext.Provider;

/** Opens the shell-owned Settings dialog without changing the current route. */
export function useOpenSettingsDialog(): () => void {
  const openSettings = useContext(SettingsDialogContext);
  if (!openSettings) {
    throw new Error("Settings dialog is unavailable outside the application shell.");
  }
  return openSettings;
}
