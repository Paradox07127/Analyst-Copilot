/* Skills slice (§10.3): browse the project's saved skills plus the builtin
 * seed templates, bind a replay to this session's datasets/columns, then
 * prepare (server-side approval) → confirm card with the SQL → execute, which
 * queues a skill_replay job onto a derived run the activity drawer tracks.
 * Replay is deterministic SQL through the read-only gate: no model call. */

import { useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type SkillPlanCandidate,
  type SkillReplayPrepared,
  type SkillReplayStarted,
  type SkillSummary,
  type SkillTargetDataset,
} from "../../api/client";
import { queryKeys, useSkills } from "../../api/hooks";
import {
  approvalGuidance,
  type ApprovalGuidance,
} from "../../api/stale-approval";
import { useJobActivity } from "../../app/job-activity";
import { sessionSectionPath } from "../../app/paths";
import { useRouteSearchParam } from "../../app/route-state";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Disclosure,
  SectionHeader,
  formatCompact,
} from "../../components/ui";
import { useDialogFocus } from "../../components/use-dialog-focus";

/* `library` and `seed` are storage words. What the reader needs is whether this
 * survives the session and whether they may delete it. */
function isSaved(skill: SkillSummary): boolean {
  return skill.source === "library";
}

function staleApprovalGuidance(error: unknown): ApprovalGuidance | null {
  return approvalGuidance(error, {
    approval_expired: {
      message: "The approval expired.",
      hint: "Prepare the replay again to request a fresh approval.",
    },
    approval_consumed: {
      message: "This approval was already used.",
      hint:
        "Its replay already ran — check the activity drawer. Preparing again " +
        "replays the skill once more.",
    },
    job_conflict: {
      message: "This request conflicts with an earlier replay.",
      hint:
        "The retry key was already used by a different job. Prepare the " +
        "replay again to start a fresh one.",
    },
  });
}

/* Same arity rule the server enforces (skill_service._require_target_arity /
 * _instantiate_seed_for), stated up front so the form can refuse locally. */
function targetArity(skill: SkillSummary): {
  allowed: (count: number) => boolean;
  hint: string;
} {
  if ((skill.params ?? []).length > 0) {
    return {
      allowed: (count) => count === 1,
      hint: "A seed template is instantiated on exactly 1 dataset.",
    };
  }
  const expected = (skill.expected_datasets ?? []).length;
  if (expected <= 1) {
    return {
      allowed: (count) => count === 1,
      hint: "Select 1 dataset to replay this skill on.",
    };
  }
  return {
    allowed: (count) => count === 1 || count === expected,
    hint:
      `Select 1 dataset to run the whole analysis on it, or ${expected} to ` +
      `map onto the ${expected} tables this skill referenced, in order.`,
  };
}

/* A binding is checked against the union of the selected targets, matching the
 * server's own column check. */
function selectedColumns(
  datasets: SkillTargetDataset[],
  datasetIds: string[],
): { name: string; bindable: boolean }[] {
  const merged = new Map<string, boolean>();
  for (const dataset of datasets) {
    if (!datasetIds.includes(dataset.dataset_id)) continue;
    for (const column of dataset.columns ?? []) {
      const bindable = column.bindable !== false;
      merged.set(column.name, (merged.get(column.name) ?? true) && bindable);
    }
  }
  return Array.from(merged, ([name, bindable]) => ({ name, bindable }));
}

function BindingForm({
  skill,
  datasets,
  datasetIds,
  bindings,
  onDatasetToggle,
  onBindingChange,
}: {
  skill: SkillSummary;
  datasets: SkillTargetDataset[];
  datasetIds: string[];
  bindings: Record<string, string>;
  onDatasetToggle: (datasetId: string, selected: boolean) => void;
  onBindingChange: (name: string, column: string) => void;
}) {
  const columns = selectedColumns(datasets, datasetIds);
  const unbindable = (skill.params ?? []).length
    ? columns.filter((column) => !column.bindable)
    : [];
  return (
    <div className="flex flex-wrap items-end gap-3">
      {/* One checkbox per table per skill: twelve tables across twelve skills
        * put 144 checkboxes on this page, so picking a skill meant scrolling
        * past the target lists of eleven others. The chosen tables stay
        * visible in the summary; the full list opens only to change them.
        * <details> rather than the shared Disclosure so the <legend> keeps
        * naming the fieldset for assistive tech. */}
      <fieldset className="flex flex-col gap-1 text-xs text-status-neutral">
        <legend>{`Replay on datasets — ${datasetIds.length} selected`}</legend>
        <details className="group">
          <summary className="w-fit cursor-pointer list-none rounded-base border border-border px-2 py-1 text-sm text-text hover:bg-surface">
            <span className="group-open:hidden">
              {datasetIds.length === 0
                ? "Choose tables"
                : datasets
                    .filter((dataset) => datasetIds.includes(dataset.dataset_id))
                    .map((dataset) => dataset.name)
                    .join(", ")}
            </span>
            <span className="hidden group-open:inline">Done choosing</span>
          </summary>
          <div className="mt-1.5 flex flex-wrap gap-3">
            {datasets.map((dataset) => (
              <label
                key={dataset.dataset_id}
                className="flex items-center gap-1.5 text-sm text-text"
              >
                <input
                  type="checkbox"
                  checked={datasetIds.includes(dataset.dataset_id)}
                  onChange={(event) =>
                    onDatasetToggle(dataset.dataset_id, event.target.checked)
                  }
                />
                {dataset.name}
              </label>
            ))}
          </div>
        </details>
      </fieldset>
      {(skill.params ?? []).map((param) => (
        <label
          key={param.name}
          className="flex flex-col gap-1 text-xs text-status-neutral"
          title={param.description ?? undefined}
        >
          {`{${param.name}} — ${param.role}`}
          <select
            value={bindings[param.name] ?? ""}
            onChange={(event) => onBindingChange(param.name, event.target.value)}
            className="rounded-base border border-border bg-surface px-2 py-1 text-sm text-text"
          >
            <option value="">Select a column…</option>
            {columns.map((column) => (
              <option
                key={column.name}
                value={column.name}
                disabled={!column.bindable}
              >
                {column.bindable
                  ? column.name
                  : `${column.name} — not a plain identifier`}
              </option>
            ))}
          </select>
        </label>
      ))}
      {unbindable.length > 0 && (
        <p className="w-full text-xs text-status-neutral">
          {unbindable.map((column) => column.name).join(", ")} cannot be bound:
          a bound column is written into the SQL, so its name must be letters,
          digits and underscores only. Rename it upstream to use it here.
        </p>
      )}
    </div>
  );
}

function ConfirmCard({
  prepared,
  pending,
  onConfirm,
  onCancel,
}: {
  prepared: SkillReplayPrepared;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const dialog = useDialogFocus(onCancel);
  const expires = new Date(prepared.expires_at);
  return (
    <Card
      ref={dialog.dialogRef}
      onKeyDown={dialog.onKeyDown}
      tone="warn"
      role="alertdialog"
      aria-label="Confirm skill replay"
      className="flex flex-col gap-2 p-3 text-sm"
    >
      <p className="font-medium">Replay this skill as a new derived session?</p>
      <p>{prepared.question}</p>
      <p className="text-xs text-status-neutral">
        Datasets: {(prepared.dataset_names ?? []).join(", ") || "—"}
      </p>
      <pre className="overflow-x-auto rounded-base bg-code-bg p-2 font-mono text-xs">
        {prepared.sql_preview}
      </pre>
      <p className="text-xs text-status-neutral">
        LLM mode: <span className="font-medium">none</span> — replay runs this
        SQL through the read-only gate, so it costs nothing. Relation names are
        rebound to the selected dataset(s) at execution.
      </p>
      {/* The approval behind this card expires; leaving that unsaid is how the
       * "The approval expired." error arrives as a surprise. */}
      {!Number.isNaN(expires.getTime()) && (
        <p className="text-xs text-status-neutral">
          {`This approval is good until ${expires.toLocaleTimeString()}; after that, prepare it again.`}
        </p>
      )}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onConfirm}
          disabled={pending}
          className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
        >
          {pending ? "Starting…" : "Confirm & replay"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={pending}
          className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
        >
          Cancel
        </button>
      </div>
    </Card>
  );
}

function shortSql(sql: string, limit = 60): string {
  const flat = sql.split(/\s+/).join(" ");
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1)}…`;
}

function SaveSkillForm({
  sessionId,
  plans,
}: {
  sessionId: string;
  plans: SkillPlanCandidate[];
}) {
  const queryClient = useQueryClient();
  const [artifactId, setArtifactId] = useState(plans[0]?.artifact_id ?? "");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const save = useMutation({
    mutationFn: (idempotencyKey: string) =>
      api.saveSkill(
        sessionId,
        {
          source_artifact_id: artifactId,
          name: name.trim(),
          description: description.trim(),
        },
        idempotencyKey,
      ),
    onSuccess: () => {
      setName("");
      setDescription("");
      void queryClient.invalidateQueries({ queryKey: queryKeys.skills(sessionId) });
    },
  });

  return (
    <Card as="section" className="flex flex-col gap-3 p-4">
      <SectionHeader
        level={3}
        title="Save an analysis as a skill"
        description="Freezes one validated plan from this session into a named query the whole project can replay."
      />
      {plans.length === 0 ? (
        <p className="text-xs text-status-neutral">
          No analysis plan in this session yet. Ask a question on the Chat page;
          each validated plan can then be frozen into a skill here.
        </p>
      ) : (
        <form
          className="flex flex-col gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            save.mutate(crypto.randomUUID());
          }}
        >
          <label className="flex flex-col gap-1 text-xs text-status-neutral">
            Analysis plan
            <select
              value={artifactId}
              onChange={(event) => setArtifactId(event.target.value)}
              className="rounded-base border border-border bg-surface px-2 py-1 text-sm text-text"
            >
              {plans.map((plan) => (
                <option key={plan.artifact_id} value={plan.artifact_id}>
                  {`${plan.question} — ${shortSql(plan.sql)}`}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-status-neutral">
            Skill name
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Revenue by region"
              maxLength={120}
              className="rounded-base border border-border bg-surface px-2 py-1 text-sm text-text"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-status-neutral">
            Description (optional)
            <input
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Total order amount grouped by customer region."
              maxLength={1000}
              className="rounded-base border border-border bg-surface px-2 py-1 text-sm text-text"
            />
          </label>
          <button
            type="submit"
            disabled={!artifactId || !name.trim() || save.isPending}
            className="self-start rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
          >
            {save.isPending ? "Saving…" : "Save as skill"}
          </button>
          {save.isError && <ErrorState error={save.error} />}
        </form>
      )}
    </Card>
  );
}

function DeleteSkillButton({
  projectId,
  sessionId,
  skill,
}: {
  projectId: string;
  sessionId: string;
  skill: SkillSummary;
}) {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);

  const remove = useMutation({
    mutationFn: (idempotencyKey: string) =>
      api.deleteSkill(projectId, skill.skill_id, idempotencyKey),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: queryKeys.skills(sessionId) }),
  });

  if (!confirming) {
    return (
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="rounded-base border border-border px-2 py-0.5 text-xs hover:bg-surface"
      >
        Delete
      </button>
    );
  }
  return (
    <span className="flex flex-wrap items-center justify-end gap-2 text-xs">
      <span className="text-status-neutral">Delete “{skill.name}”?</span>
      <button
        type="button"
        onClick={() => remove.mutate(crypto.randomUUID())}
        disabled={remove.isPending}
        className="rounded-base border border-status-critical/50 px-2 py-0.5 text-status-critical hover:bg-surface disabled:opacity-50"
      >
        {remove.isPending ? "Deleting…" : "Delete"}
      </button>
      <button
        type="button"
        onClick={() => setConfirming(false)}
        className="rounded-base border border-border px-2 py-0.5 hover:bg-surface"
      >
        Cancel
      </button>
      {remove.isError && (
        <span role="alert" className="w-full text-right text-status-critical">
          {remove.error instanceof Error
            ? remove.error.message
            : "Could not delete this skill."}
        </span>
      )}
    </span>
  );
}

function SkillCard({
  projectId,
  sessionId,
  skill,
  datasets,
}: {
  projectId: string;
  sessionId: string;
  skill: SkillSummary;
  datasets: SkillTargetDataset[];
}) {
  const { startTracking } = useJobActivity();
  const queryClient = useQueryClient();
  const [datasetIds, setDatasetIds] = useState<string[]>(
    datasets[0] ? [datasets[0].dataset_id] : [],
  );
  const [bindings, setBindings] = useState<Record<string, string>>({});
  /* One idempotency key per prepared approval: Confirm retries replay the same
   * key (and job), while a fresh prepare binds a fresh key. */
  const [executeKey, setExecuteKey] = useState("");
  const [startedReplay, setStartedReplay] =
    useState<SkillReplayStarted | null>(null);
  const replayButton = useRef<HTMLButtonElement>(null);
  const cancelReplay = () => {
    prepare.reset();
    requestAnimationFrame(() => replayButton.current?.focus());
  };

  const prepare = useMutation({
    mutationFn: () =>
      api.prepareSkillReplay(sessionId, skill.skill_id, {
        dataset_ids: datasetIds,
        bindings,
      }),
    onSuccess: () => {
      execute.reset();
      setStartedReplay(null);
      setExecuteKey(crypto.randomUUID());
    },
  });

  const importSeed = useMutation({
    mutationFn: (idempotencyKey: string) =>
      api.importSeedSkill(
        sessionId,
        skill.skill_id,
        {
          dataset_ids: datasetIds,
          bindings,
          name: skill.name,
        },
        idempotencyKey,
      ),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: queryKeys.skills(sessionId) }),
  });

  const execute = useMutation({
    mutationFn: (prepared: SkillReplayPrepared) =>
      api.executeSkillReplay(
        sessionId,
        skill.skill_id,
        {
          action_hash: prepared.action_hash,
          approval_token: prepared.approval_token,
        },
        executeKey,
      ),
    onSuccess: (started: SkillReplayStarted) => {
      setStartedReplay(started);
      prepare.reset();
      startTracking({
        jobId: started.job.job_id,
        sessionId: started.execution_session_id,
        sourceSessionId: sessionId,
        projectId,
        eventsUrl: started.job.events_url,
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.skills(sessionId) });
    },
  });

  const staleGuidance = staleApprovalGuidance(execute.error);
  const unbound = (skill.params ?? []).filter((param) => !bindings[param.name]);
  const arity = targetArity(skill);
  const canPrepare = arity.allowed(datasetIds.length) && unbound.length === 0;

  /* Rebuilt from the view's order so N targets map onto expected_datasets in a
   * predictable order rather than the order the boxes were clicked. */
  const toggleDataset = (datasetId: string, selected: boolean) =>
    setDatasetIds((current) => {
      const next = new Set(current);
      if (selected) next.add(datasetId);
      else next.delete(datasetId);
      return datasets
        .map((dataset) => dataset.dataset_id)
        .filter((id) => next.has(id));
    });

  return (
    <Card as="li" className="flex flex-col gap-2 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={isSaved(skill) ? "brand" : "llm"}>
          {isSaved(skill) ? "Saved skill" : "Built-in template"}
        </Badge>
        {skill.method && <Badge>{skill.method}</Badge>}
        <span className="text-sm font-medium">{skill.name}</span>
        <span className="ml-auto flex items-center gap-2">
          {skill.source_session_id && (
            <Link
              to={sessionSectionPath(projectId, skill.source_session_id, "artifacts")}
              className="font-mono text-xs text-primary underline-offset-2 hover:underline"
            >
              {`Source session ${skill.source_session_id}`}
            </Link>
          )}
          {/* Seeds ship with the package: only saved skills can be removed. */}
          {isSaved(skill) && (
            <DeleteSkillButton
              projectId={projectId}
              sessionId={sessionId}
              skill={skill}
            />
          )}
        </span>
      </div>
      <p className="text-sm">{skill.question}</p>
      {(skill.param_columns ?? []).length > 0 && (
        <p className="text-xs text-status-neutral">
          Reads: {(skill.param_columns ?? []).join(", ")}
        </p>
      )}
      {/* Summary → detail: the query is what the skill *is*, but a page of full
       * SELECTs buries the names and the controls that act on them. */}
      <Disclosure
        summary="SQL"
        meta={<span className="font-mono">{shortSql(skill.sql, 48)}</span>}
      >
        <pre className="overflow-x-auto rounded-base bg-code-bg p-2 font-mono text-xs">
          {skill.sql}
        </pre>
      </Disclosure>

      {datasets.length === 0 ? (
        <p className="text-xs text-status-neutral">
          This session has no dataset to replay on.
        </p>
      ) : prepare.data ? (
        staleGuidance ? (
          <div
            role="alert"
            className="flex flex-col gap-2 rounded-base border border-status-warn/50 p-3 text-sm"
          >
            <p className="font-medium text-status-warn">
              {staleGuidance.message}
            </p>
            <p className="text-status-neutral">{staleGuidance.hint}</p>
            <button
              type="button"
              onClick={() => prepare.mutate()}
              className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface"
            >
              Prepare again
            </button>
          </div>
        ) : (
          <ConfirmCard
            prepared={prepare.data}
            pending={execute.isPending}
            onConfirm={() => execute.mutate(prepare.data)}
            onCancel={cancelReplay}
          />
        )
      ) : (
        <div className="flex flex-col gap-2">
          <BindingForm
            skill={skill}
            datasets={datasets}
            datasetIds={datasetIds}
            bindings={bindings}
            onDatasetToggle={toggleDataset}
            onBindingChange={(name, column) =>
              setBindings((current) => ({ ...current, [name]: column }))
            }
          />
          <p className="text-xs text-status-neutral">{arity.hint}</p>
          <div className="flex flex-wrap items-center gap-2">
            <button
              ref={replayButton}
              type="button"
              onClick={() => prepare.mutate()}
              disabled={!canPrepare || prepare.isPending}
              className="rounded-base border border-border px-3 py-1.5 text-sm font-medium hover:bg-surface disabled:opacity-50"
            >
              {prepare.isPending ? "Preparing…" : "Replay"}
            </button>
            {/* Seeds only: replay is one-off, import keeps the bound analysis
             * in the project library where it outlives this session. */}
            {skill.source === "seed" && (
              <button
                type="button"
                onClick={() => importSeed.mutate(crypto.randomUUID())}
                disabled={!canPrepare || importSeed.isPending}
                className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-50"
              >
                {importSeed.isPending ? "Importing…" : "Import as skill"}
              </button>
            )}
          </div>
          {importSeed.isSuccess && (
            <p className="text-xs text-status-ok">
              {`Imported “${importSeed.data.name}” into the skill library.`}
            </p>
          )}
          {importSeed.isError && (
            <ErrorState
              error={importSeed.error}
              onRetry={() => importSeed.reset()}
            />
          )}
          {unbound.length > 0 && (
            <p className="text-xs text-status-neutral">
              Bind {unbound.map((param) => `{${param.name}}`).join(", ")} to a
              column first.
            </p>
          )}
        </div>
      )}

      {prepare.isError && (
        <ErrorState error={prepare.error} onRetry={() => prepare.mutate()} />
      )}
      {execute.isError && !staleGuidance && (
        <div
          role="alert"
          className="rounded-base border border-status-critical/40 p-3 text-sm text-status-critical"
        >
          {execute.error instanceof Error
            ? execute.error.message
            : "Failed to replay the skill."}
        </div>
      )}
      {startedReplay && (
        <div
          role="status"
          className="flex flex-wrap items-center gap-2 rounded-base border border-status-ok/40 bg-status-ok/5 p-3 text-sm"
        >
          <span className="font-medium text-status-ok">Replay started.</span>
          <Link
            to={sessionSectionPath(
              projectId,
              startedReplay.execution_session_id,
              "artifacts",
            )}
            className="text-primary underline-offset-2 hover:underline"
          >
            Open replay results
          </Link>
          <Link
            to={sessionSectionPath(projectId, sessionId, "artifacts")}
            className="text-status-neutral underline-offset-2 hover:underline"
          >
            Source session artifacts
          </Link>
        </div>
      )}
    </Card>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const [query, setQuery] = useRouteSearchParam("q");
  const skills = useSkills(sessionId);

  if (skills.isPending) {
    return <LoadingSkeleton lines={4} label="Loading skills" />;
  }
  if (skills.isError) {
    return (
      <div className="p-6">
        <ErrorState error={skills.error} onRetry={() => skills.refetch()} />
      </div>
    );
  }
  const pages = skills.data.pages;
  const items = pages.flatMap((page) => page.skills ?? []);
  const datasets = pages.flatMap((page) => page.datasets ?? []);
  const plans = pages.flatMap((page) => page.savable_plans ?? []);

  const needle = query.trim().toLowerCase();
  const matches = (skill: SkillSummary) =>
    !needle ||
    [
      skill.name,
      skill.question,
      skill.description ?? "",
      skill.method ?? "",
      ...(skill.param_columns ?? []),
    ].some((value) => value.toLowerCase().includes(needle));
  const filteredItems = items.filter(matches);
  const saved = filteredItems.filter(isSaved);
  const seeds = filteredItems.filter((skill) => !isSaved(skill));
  const datasetNames = datasets.map((dataset) => dataset.name).join(", ");

  const group = (
    title: string,
    description: string,
    skills: SkillSummary[],
  ) =>
    skills.length === 0 ? null : (
      <section className="flex flex-col gap-3">
        <SectionHeader
          title={
            <span className="flex items-baseline gap-2">
              {title}
              <span className="tabular font-normal text-status-neutral">
                {`(${formatCompact(skills.length)})`}
              </span>
            </span>
          }
          description={description}
        />
        <ul className="flex flex-col gap-3">
          {skills.map((skill) => (
            <SkillCard
              key={`${skill.source}:${skill.skill_id}`}
              projectId={projectId}
              sessionId={sessionId}
              skill={skill}
              datasets={datasets}
            />
          ))}
        </ul>
      </section>
    );

  return (
    <div className="mx-auto flex w-[95%] max-w-data flex-col gap-5 p-6">
      <SectionHeader
        level={1}
        title="Skills"
        description="Save and replay trusted analyses. Saved skills belong to the project; replays use this session's data and create a new result run."
      />
      <p className="text-xs text-status-neutral">
        {datasets.length === 0
          ? "This session has no dataset, so nothing here can be replayed yet."
          : `A replay binds to this session's data: ${datasetNames}.`}
      </p>
      {items.length > 0 && (
        <Card
          tone="quiet"
          className="flex min-w-0 flex-wrap items-center gap-2 px-3 py-2"
        >
          <label className="flex min-w-0 flex-1 items-center gap-2 text-sm">
            <span className="shrink-0 text-status-neutral">Find a skill</span>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Name, question, method, or field"
              className="min-w-0 flex-1 rounded-base border border-border bg-bg px-3 py-1.5 text-sm"
            />
          </label>
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="rounded-base border border-border bg-bg px-2.5 py-1.5 text-sm hover:bg-surface"
            >
              Clear
            </button>
          )}
          <span className="tabular text-xs text-status-neutral">
            {filteredItems.length} of {items.length}
          </span>
        </Card>
      )}
      <SaveSkillForm sessionId={sessionId} plans={plans} />
      {items.length === 0 ? (
        <EmptyState
          title="No skills available"
          description="Nothing is saved for this project yet and no seed template is loaded. Answer a question on the Chat page, then freeze that validated plan into a skill with the form above."
        />
      ) : filteredItems.length === 0 ? (
        <EmptyState
          title="No skills match"
          description="Try another name, question, method, or field."
        />
      ) : (
        <>
          {group(
            "Saved in this project",
            "Frozen from a validated analysis. They outlive this session, and only these can be deleted.",
            saved,
          )}
          {group(
            "Built-in templates",
            "Bind one to this session's columns to replay it once, or import it to keep the bound copy in the project.",
            seeds,
          )}
        </>
      )}
      {skills.hasNextPage && (
        <button
          type="button"
          onClick={() => void skills.fetchNextPage()}
          disabled={skills.isFetchingNextPage}
          className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-50"
        >
          {skills.isFetchingNextPage ? "Loading…" : "Load more skills"}
        </button>
      )}
    </div>
  );
}
