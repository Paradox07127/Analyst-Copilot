/* Knowledge slice (§7.5 slice H): view and edit the project semantic layer
 * from this session. Field-meaning edits save the whole list with the version the
 * page loaded (optimistic lock); a 409 version_conflict switches the editor to
 * a reload prompt instead of silently overwriting someone else's save. Join
 * whitelist review and LLM meaning proposals reuse the idempotent POSTs. */

import { useState } from "react";
import { Link, useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ApiError,
  type EntityNoteView,
  type FieldMeaningView,
  type JoinWhitelistEntryView,
  type MeaningProposalView,
  type MetricDefinitionView,
  type SemanticView,
  type VerifiedAnswerView,
  type VerifiedRelationView,
} from "../../api/client";
import {
  queryKeys,
  useDatasets,
  useDeleteVerifiedRelation,
  useSemantic,
} from "../../api/hooks";
import { sessionSectionPath } from "../../app/paths";
import { useRouteSearchParam } from "../../app/route-state";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Disclosure,
  Hint,
  Marquee,
  SectionHeader,
} from "../../components/ui";

const JOIN_STATUS_STYLE: Record<string, string> = {
  confirmed: "bg-status-ok/15 text-status-ok",
  auto_confirmed: "bg-primary/15 text-primary",
  proposed: "bg-status-warn/15 text-status-warn",
};

function Badge({ tone, children }: { tone?: string; children: string }) {
  return (
    <span
      className={`rounded-base px-1.5 py-0.5 text-[10px] font-medium uppercase ${
        tone ?? "bg-code-bg text-status-neutral"
      }`}
    >
      {children}
    </span>
  );
}

function isVersionConflict(error: unknown): boolean {
  return error instanceof ApiError && error.code === "version_conflict";
}

function ConflictBanner({ onReload }: { onReload: () => void }) {
  return (
    <div
      role="alert"
      className="flex flex-col gap-2 rounded-base border border-status-warn/50 p-3 text-sm"
    >
      <p className="font-medium text-status-warn">
        The semantic layer changed while you were editing.
      </p>
      <p className="text-status-neutral">
        Reload to pick up the latest version, then apply your edit again.
      </p>
      <button
        type="button"
        onClick={onReload}
        className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface"
      >
        Reload
      </button>
    </div>
  );
}

/* Field meanings — inline row editing, saved with expected_version. */
function FieldMeaningRow({
  field,
  editing,
  onEdit,
  onCancel,
  onSave,
  saving,
}: {
  field: FieldMeaningView;
  editing: boolean;
  onEdit: () => void;
  onCancel: () => void;
  onSave: (meaning: string, unit: string) => void;
  saving: boolean;
}) {
  const [meaning, setMeaning] = useState(field.meaning);
  const [unit, setUnit] = useState(field.unit ?? "");
  /* Rows keep a stable key across saves, so refresh the draft from the server
   * values every time editing starts instead of relying on a remount. */
  const [wasEditing, setWasEditing] = useState(editing);
  if (editing !== wasEditing) {
    setWasEditing(editing);
    if (editing) {
      setMeaning(field.meaning);
      setUnit(field.unit ?? "");
    }
  }
  const key = `${field.dataset}.${field.column}`;

  if (!editing) {
    return (
      <tr className="border-t border-border">
        <td className="px-2 py-1.5 font-mono text-xs">{key}</td>
        <td className="px-2 py-1.5 text-sm">{field.meaning}</td>
        <td className="px-2 py-1.5 text-sm">{field.unit ?? "—"}</td>
        <td className="px-2 py-1.5">
          <button
            type="button"
            onClick={onEdit}
            className="rounded-base border border-border px-2 py-0.5 text-xs hover:bg-surface"
          >
            Edit
          </button>
        </td>
      </tr>
    );
  }
  return (
    <tr className="border-t border-border">
      <td className="px-2 py-1.5 font-mono text-xs">{key}</td>
      <td className="px-2 py-1.5">
        <input
          aria-label={`Meaning of ${key}`}
          value={meaning}
          onChange={(event) => setMeaning(event.target.value)}
          className="w-full rounded-base border border-border bg-surface px-1.5 py-1 text-sm"
        />
      </td>
      <td className="px-2 py-1.5">
        <input
          aria-label={`Unit of ${key}`}
          value={unit}
          onChange={(event) => setUnit(event.target.value)}
          className="w-20 rounded-base border border-border bg-surface px-1.5 py-1 text-sm"
        />
      </td>
      <td className="flex gap-1 px-2 py-1.5">
        <button
          type="button"
          onClick={() => onSave(meaning, unit)}
          disabled={saving || meaning.trim() === ""}
          className="rounded-base bg-primary px-2 py-0.5 text-xs font-medium text-bg hover:opacity-90 disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={saving}
          className="rounded-base border border-border px-2 py-0.5 text-xs hover:bg-surface"
        >
          Cancel
        </button>
      </td>
    </tr>
  );
}

function FieldMeanings({
  sessionId,
  view,
  editingKey,
  setEditingKey,
}: {
  sessionId: string;
  view: SemanticView;
  editingKey: string | null;
  setEditingKey: (key: string | null) => void;
}) {
  const queryClient = useQueryClient();

  const save = useMutation({
    mutationFn: (fieldMeanings: FieldMeaningView[]) =>
      api.updateSemanticSeeds(sessionId, {
        expected_version: view.seeds_version,
        field_meanings: fieldMeanings,
      }),
    onSuccess: () => {
      setEditingKey(null);
      void queryClient.invalidateQueries({ queryKey: queryKeys.semantic(sessionId) });
    },
  });

  const fields = view.field_meanings ?? [];
  const conflict = isVersionConflict(save.error);

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">Field meanings</h2>
      {conflict && (
        <ConflictBanner
          onReload={() => {
            save.reset();
            setEditingKey(null);
            void queryClient.invalidateQueries({
              queryKey: queryKeys.semantic(sessionId),
            });
          }}
        />
      )}
      {save.isError && !conflict && (
        <ErrorState error={save.error} onRetry={() => save.reset()} />
      )}
      {fields.length === 0 ? (
        <EmptyState
          title="No field meanings yet"
          description="Accept a suggested meaning below, or add meanings from the analysis flow."
        />
      ) : (
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="text-xs text-status-neutral">
              <th className="px-2 py-1 font-medium">Column</th>
              <th className="px-2 py-1 font-medium">Meaning</th>
              <th className="px-2 py-1 font-medium">Unit</th>
              <th className="px-2 py-1" />
            </tr>
          </thead>
          <tbody>
            {fields.map((field) => {
              const key = `${field.dataset}.${field.column}`;
              return (
                <FieldMeaningRow
                  key={key}
                  field={field}
                  editing={editingKey === key}
                  onEdit={() => setEditingKey(key)}
                  onCancel={() => setEditingKey(null)}
                  saving={save.isPending}
                  onSave={(meaning, unit) =>
                    save.mutate(
                      fields.map((item) =>
                        item.dataset === field.dataset &&
                        item.column === field.column
                          ? {
                              ...item,
                              meaning,
                              unit: unit.trim() === "" ? null : unit.trim(),
                            }
                          : item,
                      ),
                    )
                  }
                />
              );
            })}
          </tbody>
        </table>
      )}
      <p className="text-xs text-status-neutral">
        Version {view.seeds_version} — edits are rejected if someone saved in
        between.
      </p>
    </section>
  );
}

/* Hand-edited seed classes — metric definitions, entity notes and verified
 * answers. They are structurally the same (a keyed list of short text records),
 * so one section component drives all three over flat string rows; the DTO
 * mapping lives with each caller. Fields the form never renders (verified_at)
 * ride along in the row untouched. */

type SeedRow = Record<string, string>;

interface SeedField {
  name: string;
  label: string;
  multiline?: boolean;
  optional?: boolean;
}

function blankRow(fields: SeedField[]): SeedRow {
  return Object.fromEntries(fields.map((field) => [field.name, ""]));
}

function isComplete(row: SeedRow, fields: SeedField[]): boolean {
  return fields.every(
    (field) => field.optional || (row[field.name] ?? "").trim() !== "",
  );
}

function SeedInputs({
  fields,
  draft,
  onChange,
  ariaLabel,
}: {
  fields: SeedField[];
  draft: SeedRow;
  onChange: (name: string, value: string) => void;
  ariaLabel: (field: SeedField) => string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {fields.map((field) => {
        const shared = {
          "aria-label": ariaLabel(field),
          value: draft[field.name] ?? "",
          className:
            "rounded-base border border-border bg-surface px-1.5 py-1 text-sm text-text",
        };
        return (
          <label
            key={field.name}
            className="flex flex-col gap-1 text-xs text-status-neutral"
          >
            {field.optional ? `${field.label} (optional)` : field.label}
            {field.multiline ? (
              <textarea
                {...shared}
                rows={2}
                onChange={(event) => onChange(field.name, event.target.value)}
              />
            ) : (
              <input
                {...shared}
                onChange={(event) => onChange(field.name, event.target.value)}
              />
            )}
          </label>
        );
      })}
    </div>
  );
}

function SeedListSection({
  noun,
  title,
  hint,
  emptyDescription,
  fields,
  rows,
  version,
  saving,
  meta,
  onCommit,
}: {
  noun: string;
  title: string;
  hint: string;
  emptyDescription: string;
  fields: SeedField[];
  rows: SeedRow[];
  version: number;
  saving: boolean;
  meta?: (row: SeedRow) => string | null;
  onCommit: (rows: SeedRow[]) => void;
}) {
  const [editing, setEditing] = useState<number | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);
  const [draft, setDraft] = useState<SeedRow>({});
  const [adding, setAdding] = useState<SeedRow>(() => blankRow(fields));
  /* A committed save bumps the version and refetches: drop every in-flight
   * draft so a stale one cannot be re-submitted over the new server state. */
  const [seenVersion, setSeenVersion] = useState(version);
  if (version !== seenVersion) {
    setSeenVersion(version);
    setEditing(null);
    setConfirming(null);
    setAdding(blankRow(fields));
  }

  const keyName = fields[0]?.name ?? "";
  const labelOf = (row: SeedRow) => row[keyName] ?? "";
  const rest = fields.slice(1);

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="text-xs text-status-neutral">{hint}</p>
      {rows.length === 0 ? (
        <EmptyState title={`No ${noun} yet`} description={emptyDescription} />
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((row, index) => {
            const key = labelOf(row);
            const metaLine = meta?.(row);
            return (
              <li
                key={key}
                className="flex flex-col gap-1.5 rounded-base border border-border p-3"
              >
                {editing === index ? (
                  <>
                    <SeedInputs
                      fields={fields}
                      draft={draft}
                      onChange={(name, value) =>
                        setDraft((current) => ({ ...current, [name]: value }))
                      }
                      ariaLabel={(field) => `${field.label} of ${key}`}
                    />
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          onCommit(
                            rows.map((item, at) => (at === index ? draft : item)),
                          )
                        }
                        disabled={saving || !isComplete(draft, fields)}
                        className="rounded-base bg-primary px-2 py-1 text-xs font-medium text-bg hover:opacity-90 disabled:opacity-50"
                      >
                        {saving ? "Saving…" : `Save ${noun}`}
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditing(null)}
                        disabled={saving}
                        className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
                      >
                        {`Cancel ${noun} edit`}
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-sm font-medium">{key}</p>
                    {rest.map((field) =>
                      (row[field.name] ?? "").trim() === "" ? null : (
                        <p
                          key={field.name}
                          className="text-sm text-status-neutral"
                        >
                          {row[field.name]}
                        </p>
                      ),
                    )}
                    {metaLine && (
                      <p className="text-xs text-status-neutral">{metaLine}</p>
                    )}
                    {confirming === index ? (
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <span className="text-status-neutral">{`Delete “${key}”?`}</span>
                        <button
                          type="button"
                          onClick={() =>
                            onCommit(rows.filter((_, at) => at !== index))
                          }
                          disabled={saving}
                          className="rounded-base border border-status-critical/50 px-2 py-1 text-status-critical hover:bg-surface disabled:opacity-50"
                        >
                          {`Confirm deleting ${noun}`}
                        </button>
                        <button
                          type="button"
                          onClick={() => setConfirming(null)}
                          className="rounded-base border border-border px-2 py-1 hover:bg-surface"
                        >
                          {`Keep ${noun}`}
                        </button>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setDraft({ ...row });
                            setConfirming(null);
                            setEditing(index);
                          }}
                          className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface"
                        >
                          {`Edit ${noun} ${key}`}
                        </button>
                        <button
                          type="button"
                          onClick={() => setConfirming(index)}
                          className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface"
                        >
                          {`Delete ${noun} ${key}`}
                        </button>
                      </div>
                    )}
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <div className="flex flex-col gap-2 rounded-base border border-dashed border-border p-3">
        <SeedInputs
          fields={fields}
          draft={adding}
          onChange={(name, value) =>
            setAdding((current) => ({ ...current, [name]: value }))
          }
          ariaLabel={(field) => `New ${noun} ${field.label.toLowerCase()}`}
        />
        <button
          type="button"
          onClick={() => onCommit([...rows, adding])}
          disabled={saving || !isComplete(adding, fields)}
          className="self-start rounded-base bg-primary px-2 py-1 text-xs font-medium text-bg hover:opacity-90 disabled:opacity-50"
        >
          {`Add ${noun}`}
        </button>
      </div>
    </section>
  );
}

const METRIC_FIELDS: SeedField[] = [
  { name: "name", label: "Name" },
  { name: "definition", label: "Definition", multiline: true },
  { name: "formula", label: "Formula", optional: true },
  { name: "caveats", label: "Caveats", multiline: true, optional: true },
];

const ENTITY_FIELDS: SeedField[] = [
  { name: "name", label: "Entity" },
  { name: "note", label: "Note", multiline: true },
];

const ANSWER_FIELDS: SeedField[] = [
  { name: "question", label: "Question" },
  { name: "answer", label: "Answer", multiline: true },
  { name: "evidence_note", label: "Evidence note", optional: true },
];

function text(value: string | null | undefined): string {
  return value ?? "";
}

function trimmedOrNull(value: string | undefined): string | null {
  const trimmed = (value ?? "").trim();
  return trimmed === "" ? null : trimmed;
}

function toMetric(row: SeedRow): MetricDefinitionView {
  return {
    name: (row["name"] ?? "").trim(),
    definition: (row["definition"] ?? "").trim(),
    formula: trimmedOrNull(row["formula"]),
    caveats: trimmedOrNull(row["caveats"]),
  };
}

function toEntityNote(row: SeedRow): EntityNoteView {
  return {
    name: (row["name"] ?? "").trim(),
    note: (row["note"] ?? "").trim(),
  };
}

function toVerifiedAnswer(row: SeedRow): VerifiedAnswerView {
  return {
    question: (row["question"] ?? "").trim(),
    answer: (row["answer"] ?? "").trim(),
    evidence_note: trimmedOrNull(row["evidence_note"]),
    /* Round-tripped so an edit keeps the original date; blank on a new answer
     * lets the server stamp it. */
    verified_at: trimmedOrNull(row["verified_at"]),
  };
}

/* One mutation for all three classes: they share seeds.json and its version,
 * so a save always carries the loaded field meanings plus the edited class. */
function SeedClasses({ sessionId, view }: { sessionId: string; view: SemanticView }) {
  const queryClient = useQueryClient();
  const save = useMutation({
    mutationFn: (patch: {
      metric_definitions?: MetricDefinitionView[];
      entity_notes?: EntityNoteView[];
      verified_answers?: VerifiedAnswerView[];
    }) =>
      api.updateSemanticSeeds(sessionId, {
        expected_version: view.seeds_version,
        field_meanings: view.field_meanings ?? [],
        ...patch,
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: queryKeys.semantic(sessionId) }),
  });
  const conflict = isVersionConflict(save.error);

  const metricRows: SeedRow[] = (view.metric_definitions ?? []).map((metric) => ({
    name: metric.name,
    definition: metric.definition,
    formula: text(metric.formula),
    caveats: text(metric.caveats),
  }));
  const entityRows: SeedRow[] = (view.entity_notes ?? []).map((note) => ({
    name: note.name,
    note: note.note,
  }));
  const answerRows: SeedRow[] = (view.verified_answers ?? []).map((answer) => ({
    question: answer.question,
    answer: answer.answer,
    evidence_note: text(answer.evidence_note),
    verified_at: text(answer.verified_at),
  }));

  return (
    <>
      {conflict && (
        <ConflictBanner
          onReload={() => {
            save.reset();
            void queryClient.invalidateQueries({
              queryKey: queryKeys.semantic(sessionId),
            });
          }}
        />
      )}
      {save.isError && !conflict && (
        <ErrorState error={save.error} onRetry={() => save.reset()} />
      )}
      <SeedListSection
        noun="metric"
        title="Metric definitions"
        hint="Pin how a metric is computed so results don't drift between sessions."
        emptyDescription="Add the definition, the formula and the caveats of a metric your analyses should agree on."
        fields={METRIC_FIELDS}
        rows={metricRows}
        version={view.seeds_version}
        saving={save.isPending}
        onCommit={(rows) =>
          save.mutate({ metric_definitions: rows.map(toMetric) })
        }
      />
      <SeedListSection
        noun="entity note"
        title="Entity notes"
        hint="What a business entity really is — its grain and its quirks — so the agent doesn't guess."
        emptyDescription="Note the grain of an entity, for example that a customer row is one billing account rather than one person."
        fields={ENTITY_FIELDS}
        rows={entityRows}
        version={view.seeds_version}
        saving={save.isPending}
        onCommit={(rows) => save.mutate({ entity_notes: rows.map(toEntityNote) })}
      />
      <SeedListSection
        noun="verified answer"
        title="Verified answers"
        hint="Questions you have answered and blessed; the agent reuses them as ground truth instead of recomputing."
        emptyDescription="Record an answer you trust, with the evidence it rests on."
        fields={ANSWER_FIELDS}
        rows={answerRows}
        version={view.seeds_version}
        saving={save.isPending}
        meta={(row) =>
          (row["verified_at"] ?? "") === ""
            ? null
            : `verified ${(row["verified_at"] ?? "").slice(0, 10)}`
        }
        onCommit={(rows) =>
          save.mutate({ verified_answers: rows.map(toVerifiedAnswer) })
        }
      />
    </>
  );
}

/* Join whitelist — confirm / revoke over the idempotent review endpoints. */
function JoinEntry({
  sessionId,
  entry,
}: {
  sessionId: string;
  entry: JoinWhitelistEntryView;
}) {
  const queryClient = useQueryClient();
  const invalidate = () =>
    void queryClient.invalidateQueries({ queryKey: queryKeys.semantic(sessionId) });
  const confirm = useMutation({
    mutationFn: () =>
      api.confirmSemanticJoin(sessionId, entry.label, entry.seeds_version),
    onSuccess: invalidate,
  });
  const revoke = useMutation({
    mutationFn: () =>
      api.revokeSemanticJoin(sessionId, entry.label, entry.seeds_version),
    onSuccess: invalidate,
  });
  const busy = confirm.isPending || revoke.isPending;
  /* Mirrors the server-side gate (409 join_not_confirmable otherwise). */
  const canConfirm =
    entry.status === "proposed" &&
    entry.validation_verified &&
    entry.freshness === "fresh" &&
    entry.cardinality !== "many_to_many";

  return (
    <li className="flex flex-col gap-1.5 rounded-base border border-border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs">{entry.label}</span>
        <Badge tone={JOIN_STATUS_STYLE[entry.status]}>
          {entry.status.replaceAll("_", " ")}
        </Badge>
        <Badge>{entry.cardinality.replaceAll("_", " ")}</Badge>
        {entry.freshness !== "fresh" && (
          <Badge tone="bg-status-warn/15 text-status-warn">
            {`validation ${entry.freshness}`}
          </Badge>
        )}
      </div>
      <p className="text-xs text-status-neutral">{entry.confidence_source}</p>
      <div className="flex items-center gap-2">
        {entry.status === "proposed" &&
          (canConfirm ? (
            <button
              type="button"
              onClick={() => confirm.mutate()}
              disabled={busy}
              className="rounded-base bg-primary px-2 py-1 text-xs font-medium text-bg hover:opacity-90 disabled:opacity-50"
            >
              {confirm.isPending ? "Confirming…" : "Confirm"}
            </button>
          ) : (
            <span className="text-xs text-status-neutral">
              {entry.cardinality === "many_to_many"
                ? "Many-to-many joins cannot be confirmed."
                : "Not confirmable yet — this join must pass validation against the current uploads, which runs during analysis."}
            </span>
          ))}
        {entry.status === "auto_confirmed" && (
          <button
            type="button"
            onClick={() => revoke.mutate()}
            disabled={busy}
            className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
          >
            {revoke.isPending ? "Revoking…" : "Revoke auto-confirmation"}
          </button>
        )}
      </div>
      {confirm.isError && (
        <ErrorState error={confirm.error} onRetry={() => confirm.reset()} />
      )}
      {revoke.isError && (
        <ErrorState error={revoke.error} onRetry={() => revoke.reset()} />
      )}
    </li>
  );
}

/* The semantic layer is project-scoped, so /semantic returns every proposal the
 * project ever drafted. Measured on the Olist run that is 452 proposals of
 * which 52 touch a table this session actually loaded — the other 400 come from
 * unrelated uploads (a Stack Overflow survey, a World Cup set, an LLM-usage
 * set) and were rendered inline with no way to tell them apart. */
function partitionProposals(
  proposals: MeaningProposalView[],
  runDatasetNames: ReadonlySet<string>,
): { inRun: MeaningProposalView[]; elsewhere: MeaningProposalView[] } {
  const inRun: MeaningProposalView[] = [];
  const elsewhere: MeaningProposalView[] = [];
  for (const proposal of proposals) {
    (runDatasetNames.has(proposal.dataset) ? inRun : elsewhere).push(proposal);
  }
  return { inRun, elsewhere };
}

function groupByDataset(
  proposals: MeaningProposalView[],
): { dataset: string; items: MeaningProposalView[] }[] {
  const groups = new Map<string, MeaningProposalView[]>();
  for (const proposal of proposals) {
    const bucket = groups.get(proposal.dataset);
    if (bucket) bucket.push(proposal);
    else groups.set(proposal.dataset, [proposal]);
  }
  return [...groups]
    .map(([dataset, items]) => ({ dataset, items }))
    .sort((a, b) => a.dataset.localeCompare(b.dataset));
}

/* Suggested meanings — machine drafts reviewed into field-meaning seeds. */
function ProposalCard({
  sessionId,
  proposal,
  version,
  editLocked,
}: {
  sessionId: string;
  proposal: MeaningProposalView;
  version: number;
  editLocked: boolean;
}) {
  const queryClient = useQueryClient();
  const invalidate = () =>
    void queryClient.invalidateQueries({ queryKey: queryKeys.semantic(sessionId) });
  /* null = accept the machine draft verbatim; an object overrides it. */
  const accept = useMutation({
    mutationFn: (override: { meaning: string; unit: string } | null) =>
      api.acceptMeaningProposal(sessionId, {
        dataset: proposal.dataset,
        column: proposal.column,
        expected_version: version,
        ...(override
          ? { meaning: override.meaning, unit: override.unit }
          : {}),
      }),
    onSuccess: invalidate,
  });
  const reject = useMutation({
    mutationFn: () =>
      api.rejectMeaningProposal(sessionId, {
        dataset: proposal.dataset,
        column: proposal.column,
        expected_version: version,
      }),
    onSuccess: invalidate,
  });
  const disabled = accept.isPending || reject.isPending || editLocked;
  const key = `${proposal.dataset}.${proposal.column}`;
  const [editing, setEditing] = useState(false);
  const [meaning, setMeaning] = useState(proposal.meaning);
  const [unit, setUnit] = useState(proposal.unit_guess ?? "");

  const startEditing = () => {
    setMeaning(proposal.meaning);
    setUnit(proposal.unit_guess ?? "");
    setEditing(true);
  };

  return (
    <li className="flex flex-col gap-1.5 rounded-base border border-border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs">{key}</span>
        <Badge
          tone={
            proposal.confidence === "verified"
              ? "bg-status-ok/15 text-status-ok"
              : "bg-status-warn/15 text-status-warn"
          }
        >
          {proposal.confidence === "verified" ? "verified" : "hypothesis"}
        </Badge>
        {proposal.source === "document" && (
          <Badge tone="bg-primary/15 text-primary">from document</Badge>
        )}
        {proposal.unit_guess && <Badge>{`unit ${proposal.unit_guess}`}</Badge>}
      </div>
      {editing ? (
        <div className="flex flex-col gap-1.5">
          <label className="flex flex-col gap-1 text-xs text-status-neutral">
            Meaning
            <input
              aria-label={`Proposed meaning of ${key}`}
              value={meaning}
              onChange={(event) => setMeaning(event.target.value)}
              className="rounded-base border border-border bg-surface px-1.5 py-1 text-sm text-text"
            />
          </label>
          <label className="flex w-32 flex-col gap-1 text-xs text-status-neutral">
            Unit
            <input
              aria-label={`Proposed unit of ${key}`}
              value={unit}
              onChange={(event) => setUnit(event.target.value)}
              className="rounded-base border border-border bg-surface px-1.5 py-1 text-sm text-text"
            />
          </label>
        </div>
      ) : (
        <p className="text-sm">{proposal.meaning}</p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {editing ? (
          <>
            <button
              type="button"
              onClick={() =>
                accept.mutate({ meaning: meaning.trim(), unit: unit.trim() })
              }
              disabled={disabled || meaning.trim() === ""}
              className="rounded-base bg-primary px-2 py-1 text-xs font-medium text-bg hover:opacity-90 disabled:opacity-50"
            >
              {accept.isPending ? "Accepting…" : "Accept edited"}
            </button>
            <button
              type="button"
              onClick={() => setEditing(false)}
              disabled={accept.isPending}
              className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              onClick={() => accept.mutate(null)}
              disabled={disabled}
              className="rounded-base bg-primary px-2 py-1 text-xs font-medium text-bg hover:opacity-90 disabled:opacity-50"
            >
              {accept.isPending ? "Accepting…" : "Accept"}
            </button>
            <button
              type="button"
              onClick={startEditing}
              disabled={disabled}
              className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
            >
              Edit &amp; accept
            </button>
            <button
              type="button"
              onClick={() => reject.mutate()}
              disabled={disabled}
              className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
            >
              {reject.isPending ? "Dismissing…" : "Dismiss"}
            </button>
          </>
        )}
      </div>
      {accept.isError && (
        <ErrorState error={accept.error} onRetry={() => accept.reset()} />
      )}
      {reject.isError && (
        <ErrorState error={reject.error} onRetry={() => reject.reset()} />
      )}
    </li>
  );
}

/* Verified relations — sunk automatically from Relationships review; this
 * page only deletes a wrong one (delete by content, not index: see
 * semantic_ui.py:216-239 for the reference layout). */
function RelationEntry({
  sessionId,
  relation,
  version,
}: {
  sessionId: string;
  relation: VerifiedRelationView;
  version: number;
}) {
  const [confirming, setConfirming] = useState(false);
  const deleteRelation = useDeleteVerifiedRelation(sessionId);
  const label = `${relation.left} → ${relation.right}`;

  return (
    <li className="flex flex-col gap-1.5 rounded-base border border-border p-3">
      <p className="text-sm font-medium">{label}</p>
      <p className="text-xs text-status-neutral">
        <Badge tone="bg-status-ok/15 text-status-ok">
          {relation.cardinality.replaceAll("_", " ")}
        </Badge>
        {` · confirmed by ${relation.confirmed_by}`}
        {relation.confirmed_at && ` · ${relation.confirmed_at.slice(0, 10)}`}
        {relation.source_session_id && ` · from session ${relation.source_session_id}`}
      </p>
      {confirming ? (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-status-neutral">{`Delete “${label}”?`}</span>
          <button
            type="button"
            onClick={() =>
              deleteRelation.mutate(
                {
                  left: relation.left,
                  right: relation.right,
                  expected_version: version,
                },
                { onSuccess: () => setConfirming(false) },
              )
            }
            disabled={deleteRelation.isPending}
            className="rounded-base border border-status-critical/50 px-2 py-1 text-status-critical hover:bg-surface disabled:opacity-50"
          >
            Confirm deleting relation
          </button>
          <button
            type="button"
            onClick={() => setConfirming(false)}
            className="rounded-base border border-border px-2 py-1 hover:bg-surface"
          >
            Keep relation
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setConfirming(true)}
          className="self-start rounded-base border border-border px-2 py-1 text-xs hover:bg-surface"
        >
          {`Delete relation ${label}`}
        </button>
      )}
      {deleteRelation.isError && (
        <ErrorState
          error={deleteRelation.error}
          onRetry={() => deleteRelation.reset()}
        />
      )}
    </li>
  );
}

function VerifiedRelations({
  sessionId,
  relations,
  version,
}: {
  sessionId: string;
  relations: VerifiedRelationView[];
  version: number;
}) {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">Verified relations</h2>
      {relations.length === 0 ? (
        <EmptyState
          title="No verified relations yet"
          description="Confirming a join on the Relationships page sinks it here automatically. Remove one below if it turns out to be wrong."
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {relations.map((relation) => (
            <RelationEntry
              key={`${relation.left}__${relation.right}`}
              sessionId={sessionId}
              relation={relation}
              version={version}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function ColumnRoles({ view }: { view: SemanticView }) {
  const roles = view.column_roles ?? [];
  if (roles.length === 0) return null;
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">Column roles (this session)</h2>
      <ul className="flex flex-wrap gap-1.5">
        {roles.map((role) => (
          <li
            key={`${role.dataset}.${role.column}`}
            className="rounded-base bg-code-bg px-2 py-0.5 text-xs"
          >
            <span className="font-mono">{`${role.dataset}.${role.column}`}</span>
            <span className="text-status-neutral">{` · ${role.role}`}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

type KnowledgeView = "meanings" | "definitions" | "joins";

const KNOWLEDGE_VIEWS: {
  id: KnowledgeView;
  label: string;
}[] = [
  { id: "meanings", label: "Meanings review" },
  { id: "definitions", label: "Business definitions" },
  { id: "joins", label: "Join policy" },
];

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const semantic = useSemantic(sessionId);
  const datasets = useDatasets(sessionId);
  const [viewParam, setViewParam] = useRouteSearchParam("view", "meanings");
  const activeView: KnowledgeView = KNOWLEDGE_VIEWS.some(
    (item) => item.id === viewParam,
  )
    ? (viewParam as KnowledgeView)
    : "meanings";
  /* Lifted here so an in-progress field edit can lock proposal review:
   * accepting/rejecting bumps seeds_version and refetches, which would
   * otherwise throw away the edit in progress. */
  const [editingKey, setEditingKey] = useState<string | null>(null);

  if (semantic.isPending) {
    return <LoadingSkeleton lines={4} label="Loading knowledge" />;
  }
  if (semantic.isError) {
    return (
      <div className="p-6">
        <ErrorState error={semantic.error} onRetry={() => semantic.refetch()} />
      </div>
    );
  }
  const pages = semantic.data.pages;
  const first = pages[0]!;
  const mergedView: SemanticView = {
    ...first,
    field_meanings: pages.flatMap((page) => page.field_meanings ?? []),
    metric_definitions: pages.flatMap(
      (page) => page.metric_definitions ?? [],
    ),
    entity_notes: pages.flatMap((page) => page.entity_notes ?? []),
    verified_answers: pages.flatMap((page) => page.verified_answers ?? []),
    verified_relations: pages.flatMap(
      (page) => page.verified_relations ?? [],
    ),
    column_roles: pages.flatMap((page) => page.column_roles ?? []),
    join_whitelist: pages.flatMap((page) => page.join_whitelist ?? []),
    proposals: pages.flatMap((page) => page.proposals ?? []),
    next_cursor: pages.at(-1)?.next_cursor,
  };
  const joins = mergedView.join_whitelist ?? [];
  const proposals = mergedView.proposals ?? [];
  const relations = mergedView.verified_relations ?? [];
  /* Datasets resolve after the semantic view; until they do, treat everything
   * as out of scope rather than mislabelling a foreign table as this session's. */
  const runDatasetNames = new Set(
    (datasets.data ?? []).map((dataset) => dataset.display_name),
  );
  const { inRun, elsewhere } = partitionProposals(proposals, runDatasetNames);

  return (
    <div className="mx-auto flex w-[95%] max-w-data min-w-0 flex-col gap-5 p-6">
      <SectionHeader
        level={1}
        title="Knowledge"
        description="Review and maintain the project-wide facts the agent is allowed to reuse across analyses."
        actions={<Badge tone="bg-primary/15 text-primary">project-wide</Badge>}
      />

      <div className="grid gap-2 rounded-base border border-border bg-surface p-3 sm:grid-cols-2">
        <div className="min-w-0">
          <p className="text-xs font-medium">Project knowledge</p>
          <p className="mt-0.5 text-xs text-status-neutral">
            Accepted meanings, definitions and join rules apply to every
            analysis in this project.
          </p>
        </div>
        <div className="min-w-0 border-t border-hairline pt-2 sm:border-l sm:border-t-0 sm:pl-3 sm:pt-0">
          <p className="text-xs font-medium">Current session lens</p>
          <p className="mt-0.5 text-xs text-status-neutral">
            {datasets.isPending
              ? "Resolving the datasets loaded by this session…"
              : `${runDatasetNames.size} loaded dataset${
                  runDatasetNames.size === 1 ? "" : "s"
                }; project-only suggestions stay separated.`}
          </p>
        </div>
      </div>

      <div
        role="tablist"
        aria-label="Knowledge tasks"
        className="grid grid-cols-3 gap-1 rounded-base border border-border bg-surface p-1 sm:flex sm:w-fit"
      >
        {KNOWLEDGE_VIEWS.map((item) => (
          <button
            key={item.id}
            id={`knowledge-${item.id}-tab`}
            type="button"
            role="tab"
            aria-selected={activeView === item.id}
            aria-controls={`knowledge-${item.id}-panel`}
            onClick={() => setViewParam(item.id)}
            className={`min-w-0 rounded-base px-2 py-1.5 text-xs font-medium sm:px-3 sm:text-sm ${
              activeView === item.id
                ? "bg-bg text-text shadow-sm"
                : "text-status-neutral hover:text-text"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      {activeView === "meanings" && (
        <section
          id="knowledge-meanings-panel"
          role="tabpanel"
          aria-labelledby="knowledge-meanings-tab"
          className="flex min-w-0 flex-col gap-6"
        >
          <SectionHeader
            title="Meanings review"
            level={2}
            description="Review machine-drafted field meanings, then maintain the accepted meaning and unit."
            actions={
              <Hint label="Why other datasets appear">
                The semantic layer belongs to the project, not to one run, so
                every table the project has ever loaded contributes proposals.
                Tables this session did not load are grouped separately below —
                accepting one still writes to the shared project layer.
              </Hint>
            }
          />
          {proposals.length === 0 ? (
            <EmptyState
              title="No meaning suggestions to review"
              description="Accepted field meanings remain available below."
            />
          ) : (
            <>
              {editingKey !== null && (
                <p className="text-xs text-status-warn">
                  Save or cancel the field-meaning edit below before reviewing
                  suggestions.
                </p>
              )}
              {inRun.length === 0 ? (
                <EmptyState
                  title="No suggestions for this session's tables"
                  description="Every pending suggestion belongs to a table this session did not load. They are grouped separately below."
                />
              ) : (
                groupByDataset(inRun).map((group) => (
                  <div key={group.dataset} className="flex flex-col gap-2">
                    <h3 className="flex items-baseline gap-2 text-sm font-medium">
                      <Marquee className="font-mono">
                        {group.dataset}
                      </Marquee>
                      <span className="tabular shrink-0 text-xs font-normal text-status-neutral">
                        {group.items.length}
                      </span>
                    </h3>
                    <ul className="flex flex-col gap-2">
                      {group.items.map((proposal) => (
                        <ProposalCard
                          key={`${proposal.dataset}.${proposal.column}`}
                          sessionId={sessionId}
                          proposal={proposal}
                          version={mergedView.seeds_version}
                          editLocked={editingKey !== null}
                        />
                      ))}
                    </ul>
                  </div>
                ))
              )}

              {elsewhere.length > 0 && (
                <Disclosure
                  summary={`From other tables in this project (${elsewhere.length})`}
                  meta={`${groupByDataset(elsewhere).length} tables this session did not load`}
                >
                  <div className="flex flex-col gap-3">
                    {groupByDataset(elsewhere).map((group) => (
                      <div key={group.dataset} className="flex flex-col gap-2">
                        <h3 className="flex items-baseline gap-2 text-sm font-medium">
                          <Marquee className="font-mono text-status-neutral">
                            {group.dataset}
                          </Marquee>
                          <span className="tabular shrink-0 text-xs font-normal text-status-neutral">
                            {group.items.length}
                          </span>
                        </h3>
                        <ul className="flex flex-col gap-2">
                          {group.items.map((proposal) => (
                            <ProposalCard
                              key={`${proposal.dataset}.${proposal.column}`}
                              sessionId={sessionId}
                              proposal={proposal}
                              version={mergedView.seeds_version}
                              editLocked={editingKey !== null}
                            />
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </Disclosure>
              )}
            </>
          )}

          {semantic.hasNextPage ? (
            <LoadAllKnowledge
              loading={semantic.isFetchingNextPage}
              onLoad={() => void semantic.fetchNextPage()}
            />
          ) : (
            <FieldMeanings
              sessionId={sessionId}
              view={mergedView}
              editingKey={editingKey}
              setEditingKey={setEditingKey}
            />
          )}
          <ColumnRoles view={mergedView} />
        </section>
      )}

      {activeView === "definitions" && (
        <section
          id="knowledge-definitions-panel"
          role="tabpanel"
          aria-labelledby="knowledge-definitions-tab"
          className="flex min-w-0 flex-col gap-6"
        >
          <SectionHeader
            title="Business definitions"
            level={2}
            description="Maintain stable metrics, entity grain and verified answers that every session in this project can reuse."
          />
          {semantic.hasNextPage ? (
            <LoadAllKnowledge
              loading={semantic.isFetchingNextPage}
              onLoad={() => void semantic.fetchNextPage()}
            />
          ) : (
            <SeedClasses sessionId={sessionId} view={mergedView} />
          )}
        </section>
      )}

      {activeView === "joins" && (
        <section
          id="knowledge-joins-panel"
          role="tabpanel"
          aria-labelledby="knowledge-joins-tab"
          className="flex min-w-0 flex-col gap-6"
        >
          <SectionHeader
            title="Project join policy"
            level={2}
            description="This is the shared ledger of joins available to cross-table analysis. Candidate discovery and full-table validation happen in Relationships."
            actions={
              <Link
                to={sessionSectionPath(projectId, sessionId, "relationships")}
                className="rounded-base border border-border px-3 py-1.5 text-sm font-medium hover:bg-surface"
              >
                Open Relationships
              </Link>
            }
          />

          <VerifiedRelations
            sessionId={sessionId}
            relations={relations}
            version={mergedView.seeds_version}
          />

          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold">Allowed joins</h2>
            <p className="text-xs text-status-neutral">
              Only confirmed entries can be used by cross-table questions and
              generated JOIN SQL. Verified proposals can be adopted here;
              discovery stays in Relationships.
            </p>
            {joins.length === 0 ? (
              <EmptyState
                title="No join rules yet"
                description="Open Relationships to discover and validate candidates before adopting them into this project policy."
              />
            ) : (
              <ul className="flex flex-col gap-2">
                {joins.map((entry) => (
                  <JoinEntry
                    key={entry.label}
                    sessionId={sessionId}
                    entry={entry}
                  />
                ))}
              </ul>
            )}
          </section>
        </section>
      )}
    </div>
  );
}

function LoadAllKnowledge({
  loading,
  onLoad,
}: {
  loading: boolean;
  onLoad: () => void;
}) {
  return (
    <section className="flex flex-col items-start gap-2 rounded-base border border-border p-3">
      <p className="text-xs text-status-neutral">
        More project knowledge is available. Load every page before editing so
        a full-list save cannot omit rows that are not currently visible.
      </p>
      <button
        type="button"
        onClick={onLoad}
        disabled={loading}
        className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-50"
      >
        {loading ? "Loading…" : "Load more knowledge"}
      </button>
    </section>
  );
}
