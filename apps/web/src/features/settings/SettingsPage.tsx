/* Route wrapper. The rail no longer links here, but the path stays a
 * bookmarkable deep link; the body it renders is the same SettingsPanel the
 * top-bar dialog shows. */

import { useSearchParams } from "react-router";
import {
  SettingsPanel,
  type SettingsSectionId,
} from "./SettingsPanel";

const SETTINGS_SECTIONS = new Set<SettingsSectionId>([
  "model",
  "analysis",
  "appearance",
  "about",
]);

export function Component() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requested = searchParams.get("section") as SettingsSectionId | null;
  const section =
    requested && SETTINGS_SECTIONS.has(requested) ? requested : "model";

  const changeSection = (nextSection: SettingsSectionId) => {
    const next = new URLSearchParams(searchParams);
    if (nextSection === "model") next.delete("section");
    else next.set("section", nextSection);
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="mx-auto flex w-[95%] max-w-data min-w-0 flex-col gap-4 p-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-sm text-status-neutral">
          Workspace-wide model, privacy and appearance preferences.
        </p>
      </div>
      <SettingsPanel section={section} onSectionChange={changeSection} />
    </div>
  );
}
