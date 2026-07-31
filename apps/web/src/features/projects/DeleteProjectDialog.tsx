/* Owned by the projects feature rather than the rail, because both the rail's
 * per-project menu and the Projects page open the same dialog. */

import { useNavigate } from "react-router";
import type { ProjectSummary } from "../../api/client";
import { useDeleteProject } from "../../api/hooks";
import { ErrorState } from "../../components/async-states";
import { Button } from "../../components/ui";
import { useDialogFocus } from "../../components/use-dialog-focus";

export function DeleteProjectDialog({
  project,
  onClose,
}: {
  project: ProjectSummary;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const remove = useDeleteProject();
  const { dialogRef, onKeyDown } = useDialogFocus(onClose);

  const confirm = () => {
    remove.mutate(project.project_id, {
      onSuccess: () => {
        onClose();
        navigate("/projects");
      },
    });
  };

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label={`Delete project ${project.name}`}
      onKeyDown={onKeyDown}
      className="animate-fade fixed inset-0 z-50 flex items-center justify-center bg-scrim p-4"
    >
      <div className="animate-enter flex max-w-md flex-col gap-3 rounded-xl border border-status-critical/40 bg-bg p-5 shadow-overlay">
        <p className="text-xs font-semibold tracking-widest text-status-critical uppercase">
          Irreversible action
        </p>
        <h2 className="text-base font-semibold">Delete {project.name}?</h2>
        <p className="text-sm text-status-neutral">
          Every session, uploaded dataset, report and project setting will be
          permanently removed. This cannot be undone.
        </p>
        {remove.isError && <ErrorState error={remove.error} />}
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="danger" onClick={confirm} disabled={remove.isPending}>
            {remove.isPending ? "Deleting…" : "Delete project"}
          </Button>
        </div>
      </div>
    </div>
  );
}
