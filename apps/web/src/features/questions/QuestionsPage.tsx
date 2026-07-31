/* Questions is intentionally a single-question workflow:
 * review one card -> approve its prepared content -> execute it.
 * Editing an existing card and drafting one from free text live here too. */

import { useState } from "react";
import { Link, useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type QuestionCardEdit,
  type QuestionDraftPrepared,
  type QuestionExecutionPrepared,
  type QuestionExecutionStarted,
  type QuestionSummary,
} from "../../api/client";
import { queryKeys, useQuestions } from "../../api/hooks";
import {
  approvalGuidance,
  type ApprovalGuidance,
} from "../../api/stale-approval";
import { useJobActivity } from "../../app/job-activity";
import { sessionSectionPath } from "../../app/paths";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Dot,
  Hint,
  SectionHeader,
  StepChain,
  type Tone,
} from "../../components/ui";
import { useDialogFocus } from "../../components/use-dialog-focus";
import { DRAFT_STEPS, RUN_ONE_STEPS } from "./StepChain";

const OUTCOME_TONE: Record<string, Tone> = {
  answered: "ok",
  abstained: "warn",
  awaiting_approval: "brand",
  failed: "critical",
};

function priorityLabel(priority: number): string {
  if (priority >= 0.7) return "High";
  if (priority >= 0.4) return "Medium";
  return "Low";
}

/* Approval lifecycle errors get guided recovery, mirroring the cleaning page. */
function staleApprovalGuidance(error: unknown): ApprovalGuidance | null {
  return approvalGuidance(error, {
    approval_expired: {
      message: "The approval expired.",
      hint: "Approve the question again to request a fresh approval.",
    },
    approval_consumed: {
      message: "This approval was already used.",
      hint:
        "Its execution already ran — check the activity drawer or the " +
        "execution badge. Approving again runs the question once more.",
    },
    question_source_changed: {
      message: "The question changed since it was approved.",
      hint: "Approve the question again to review the fresh content.",
    },
    job_conflict: {
      message: "This request conflicts with an earlier execution.",
      hint:
        "The retry key was already used by a different job. Approve the " +
        "question again to start a fresh execution.",
    },
  });
}

/** The card's loudest chip: what came of this question, or that nothing did. */
function ExecutionBadge({
  projectId,
  question,
}: {
  projectId: string;
  question: QuestionSummary;
}) {
  const execution = question.execution;
  if (!execution) {
    return (
      <Badge caps>
        <Dot tone="neutral" />
        Not executed
      </Badge>
    );
  }
  const outcome = execution.outcome;
  const tone = OUTCOME_TONE[outcome] ?? "neutral";
  return (
    <span className="flex items-center gap-2">
      <Badge tone={tone} caps>
        <Dot tone={tone} />
        {outcome.replace("_", " ")}
      </Badge>
      <Link
        to={sessionSectionPath(projectId, execution.execution_session_id, "artifacts")}
        className="text-xs text-primary underline-offset-2 hover:underline"
      >
        {(execution.findings_count ?? 0) === 1
          ? "1 finding"
          : `${execution.findings_count ?? 0} findings`}
      </Link>
    </span>
  );
}

function ConfirmCard({
  prepared,
  pending,
  onConfirm,
  onCancel,
}: {
  prepared: QuestionExecutionPrepared;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { dialogRef, onKeyDown } = useDialogFocus(onCancel);
  return (
    <Card
      ref={dialogRef}
      onKeyDown={onKeyDown}
      tone="warn"
      role="alertdialog"
      aria-label="Confirm question execution"
      className="flex flex-col gap-2 p-3 text-sm"
    >
      <StepChain label="Run this question" steps={RUN_ONE_STEPS} current={1} />
      <p className="font-medium">Run this question as a new analysis session?</p>
      <p>{prepared.question}</p>
      {(prepared.target_datasets ?? []).length > 0 && (
        <p className="text-xs">
          <span className="mr-1.5 text-status-neutral">Datasets</span>
          {(prepared.target_datasets ?? []).join(", ")}
        </p>
      )}
      {prepared.sql_preview ? (
        <pre className="overflow-x-auto rounded-base bg-code-bg p-2 font-mono text-xs">
          {prepared.sql_preview}
        </pre>
      ) : prepared.uses_llm ? (
        <p className="text-xs text-status-warn">
          The agent chooses the analysis at execution time. It may inspect
          evidence, run several read-only queries, replay a compatible Skill,
          or use secured Python when a sandbox is available.
        </p>
      ) : null}
      <p className="text-xs">
        <span className="mr-1.5 text-status-neutral">LLM mode:</span>
        <span className="font-medium">{prepared.llm_mode}</span>
      </p>
      {prepared.llm_mode === "env" && (
        <p className="text-xs text-status-warn">
          Executes with the live model and may incur cost.
        </p>
      )}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onConfirm}
          disabled={pending}
          className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
        >
          {pending ? "Starting…" : "Confirm & execute"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={pending}
          className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-bg"
        >
          Cancel
        </button>
        <span className="text-xs text-status-neutral">
          Nothing has run yet.
        </span>
      </div>
    </Card>
  );
}

const EDIT_TEXT_FIELDS = [
  { field: "question_en", label: "Question", multiline: false },
  { field: "business_decision", label: "Business decision", multiline: true },
  { field: "value_hypothesis", label: "Value hypothesis", multiline: true },
  { field: "success_criterion", label: "Success criterion", multiline: true },
  { field: "data_signal", label: "Data signal", multiline: true },
  { field: "priority_rationale", label: "Priority rationale", multiline: true },
] as const;

const EDIT_LIST_FIELDS = [
  { field: "risks", label: "Risks (one per line)" },
  { field: "data_requirements", label: "Data requirements (one per line)" },
] as const;

function CardEditForm({
  sessionId,
  question,
  onDone,
}: {
  sessionId: string;
  question: QuestionSummary;
  onDone: () => void;
}) {
  const queryClient = useQueryClient();
  /* `question_en` is the card's own field name; the summary exposes it as
   * `question`, so the form maps it explicitly rather than by index. */
  const [draft, setDraft] = useState<Record<string, string>>(() => ({
    question_en: question.question,
    business_decision: question.business_decision ?? "",
    value_hypothesis: question.value_hypothesis ?? "",
    success_criterion: question.success_criterion ?? "",
    data_signal: question.data_signal ?? "",
    priority_rationale: question.priority_rationale ?? "",
    risks: (question.risks ?? []).join("\n"),
    data_requirements: (question.data_requirements ?? []).join("\n"),
  }));

  const save = useMutation({
    mutationFn: () => {
      const body: QuestionCardEdit = {
        expected_version: question.card_version,
      };
      for (const { field } of EDIT_TEXT_FIELDS) {
        (body as Record<string, unknown>)[field] = draft[field] ?? "";
      }
      for (const { field } of EDIT_LIST_FIELDS) {
        (body as Record<string, unknown>)[field] = (draft[field] ?? "")
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);
      }
      return api.editQuestionCard(sessionId, question.question_id, body);
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.questions(sessionId),
      });
      onDone();
    },
  });

  return (
    <form
      aria-label="Edit question card"
      className="flex flex-col gap-2 rounded-base border border-border p-3"
      onSubmit={(event) => {
        event.preventDefault();
        save.mutate();
      }}
    >
      <p className="flex items-center gap-1.5 text-xs text-status-neutral">
        Framing only — the approved dataset scope stays locked.
        <Hint label="Why framing only">
          The agent chooses its method and tools during execution, but the
          question and target datasets define its approved scope. Saving bumps
          the card version and invalidates approvals bound to the old framing.
        </Hint>
      </p>
      {EDIT_TEXT_FIELDS.map(({ field, label, multiline }) => (
        <label key={field} className="flex flex-col gap-1 text-xs">
          {label}
          {multiline ? (
            <textarea
              rows={2}
              value={draft[field] ?? ""}
              onChange={(event) =>
                setDraft((current) => ({ ...current, [field]: event.target.value }))
              }
              className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
            />
          ) : (
            <input
              value={draft[field] ?? ""}
              onChange={(event) =>
                setDraft((current) => ({ ...current, [field]: event.target.value }))
              }
              className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
            />
          )}
        </label>
      ))}
      {EDIT_LIST_FIELDS.map(({ field, label }) => (
        <label key={field} className="flex flex-col gap-1 text-xs">
          {label}
          <textarea
            rows={2}
            value={draft[field] ?? ""}
            onChange={(event) =>
              setDraft((current) => ({ ...current, [field]: event.target.value }))
            }
            className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
          />
        </label>
      ))}
      <div className="flex items-center gap-2">
        <button
          type="submit"
          disabled={save.isPending}
          className="rounded-base bg-primary px-3 py-1 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
        >
          {save.isPending ? "Saving…" : "Save card"}
        </button>
        <button
          type="button"
          onClick={onDone}
          className="rounded-base border border-border px-3 py-1 text-sm hover:bg-surface"
        >
          Cancel
        </button>
      </div>
      {save.isError && (
        <div
          role="alert"
          className="rounded-base border border-status-critical/40 p-2 text-xs text-status-critical"
        >
          {save.error instanceof Error ? save.error.message : "Could not save the card."}
        </div>
      )}
    </form>
  );
}

function DraftQuestionPanel({
  projectId,
  sessionId,
  hasSuggestions,
}: {
  projectId: string;
  sessionId: string;
  hasSuggestions: boolean;
}) {
  const { startTracking } = useJobActivity();
  const queryClient = useQueryClient();
  const [text, setText] = useState("");
  const [llmMode, setLlmMode] = useState<"env" | "offline">("env");
  const [draftKey, setDraftKey] = useState("");
  const [prepared, setPrepared] = useState<QuestionDraftPrepared | null>(null);
  const draftDialog = useDialogFocus(
    () => setPrepared(null),
    prepared !== null,
  );

  const prepare = useMutation({
    mutationFn: () => api.prepareQuestionDraft(sessionId, { question: text, llm: llmMode }),
    onSuccess: (data) => {
      submit.reset();
      setDraftKey(crypto.randomUUID());
      setPrepared(data);
    },
  });

  const submit = useMutation({
    mutationFn: (data: QuestionDraftPrepared) =>
      api.draftQuestionCard(
        sessionId,
        { action_hash: data.action_hash, approval_token: data.approval_token },
        draftKey,
      ),
    onSuccess: (started) => {
      setPrepared(null);
      setText("");
      startTracking({
        jobId: started.job.job_id,
        sessionId: started.execution_session_id,
        sourceSessionId: sessionId,
        projectId,
        eventsUrl: started.job.events_url,
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.questions(sessionId) });
    },
  });

  return (
    <Card
      tone={hasSuggestions ? "quiet" : "default"}
      className="flex flex-col gap-3 p-4"
    >
      <SectionHeader
        level={3}
        title={
          hasSuggestions
            ? "Ask your own question"
            : "Start with your own question"
        }
        description={
          hasSuggestions
            ? "Not seeing the right question? Draft one for review."
            : "Write what you want to learn. The agent will draft a reviewable question card."
        }
        actions={
          <Hint label="Drafting a card">
            Enter the question only. The agent drafts the rest of the card —
            decision context, scope, method, success criterion, risks — and the
            new card joins the list above for you to approve like any other.
          </Hint>
        }
      />
      <StepChain
        label="Draft a question card"
        steps={DRAFT_STEPS}
        current={prepared ? 1 : 0}
      />
      <label
        htmlFor="draft-question"
        className="flex flex-col gap-1.5 text-xs font-medium"
      >
        Your question
        <textarea
          id="draft-question"
          rows={3}
          value={text}
          placeholder="For example: which customer segments have the highest return rate?"
          onChange={(event) => setText(event.target.value)}
          className="rounded-base border border-border bg-bg px-3 py-2 text-sm font-normal"
        />
      </label>
      {prepared ? (
        <Card
          ref={draftDialog.dialogRef}
          onKeyDown={draftDialog.onKeyDown}
          tone="warn"
          role="alertdialog"
          aria-label="Confirm question drafting"
          className="flex flex-col gap-2 p-3 text-sm"
        >
          <p className="font-medium">Draft a card for this question?</p>
          <p>{prepared.question}</p>
          <p className="text-xs">
            <span className="mr-1.5 text-status-neutral">LLM mode:</span>
            <span className="font-medium">{prepared.llm_mode}</span>
          </p>
          <p className="text-xs text-status-neutral">
            {prepared.llm_mode === "offline"
              ? "Offline drafting appends the question with no score and no filled-in review fields."
              : "Calls the live model and may incur cost."}
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => submit.mutate(prepared)}
              disabled={submit.isPending}
              className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
            >
              {submit.isPending ? "Starting…" : "Confirm & draft"}
            </button>
            <button
              type="button"
              onClick={() => setPrepared(null)}
              className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-bg"
            >
              Cancel
            </button>
          </div>
        </Card>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!text.trim() || prepare.isPending}
            onClick={() => prepare.mutate()}
            className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
          >
            {prepare.isPending ? "Preparing…" : "Draft question card"}
          </button>
          <label className="flex items-center gap-1 text-xs text-status-neutral">
            Draft LLM mode
            <select
              value={llmMode}
              onChange={(event) =>
                setLlmMode(event.target.value as "env" | "offline")
              }
              className="rounded-base border border-border bg-bg px-1.5 py-1 text-xs"
            >
              <option value="env">env (live model)</option>
              <option value="offline">offline</option>
            </select>
          </label>
        </div>
      )}
      {prepare.isError && (
        <ErrorState error={prepare.error} onRetry={() => prepare.mutate()} />
      )}
      {submit.isError && (
        <div
          role="alert"
          className="rounded-base border border-status-critical/40 p-2 text-xs text-status-critical"
        >
          {submit.error instanceof Error
            ? submit.error.message
            : "Could not start drafting."}
        </div>
      )}
    </Card>
  );
}

/* Genre — origin, method, value category, card version — is the weakest tier on
 * the card, so it renders as one quiet line rather than four shouting chips. */
function GenreLine({ question }: { question: QuestionSummary }) {
  const terms = [
    question.origin,
    question.analysis_mode ?? null,
    question.value_category
      ? question.value_category.replaceAll("_", " ")
      : null,
    question.card_version > 1 ? `v${question.card_version}` : null,
  ].filter((term): term is string => Boolean(term));

  return (
    <p className="flex flex-wrap items-center gap-x-1.5 text-xs text-status-neutral">
      {terms.map((term, index) => (
        <span key={term} className="flex items-center gap-1.5">
          {index > 0 && (
            <span aria-hidden className="text-status-neutral/50">
              ·
            </span>
          )}
          <span>{term}</span>
        </span>
      ))}
    </p>
  );
}

function QuestionCard({
  projectId,
  sessionId,
  question,
}: {
  projectId: string;
  sessionId: string;
  question: QuestionSummary;
}) {
  const { startTracking } = useJobActivity();
  const queryClient = useQueryClient();
  /* One idempotency key per prepared approval: Confirm retries replay the
   * same key (and job), while a fresh prepare binds a fresh key. */
  const [executeKey, setExecuteKey] = useState("");
  const [llmMode, setLlmMode] = useState<"env" | "offline">("env");

  const prepare = useMutation({
    mutationFn: () =>
      api.prepareQuestion(sessionId, question.question_id, { llm: llmMode }),
    onSuccess: () => {
      execute.reset();
      setExecuteKey(crypto.randomUUID());
    },
  });

  const execute = useMutation({
    mutationFn: (prepared: QuestionExecutionPrepared) =>
      api.executeQuestion(
        sessionId,
        question.question_id,
        {
          action_hash: prepared.action_hash,
          approval_token: prepared.approval_token,
        },
        executeKey,
      ),
    onSuccess: (started: QuestionExecutionStarted) => {
      prepare.reset();
      startTracking({
        jobId: started.job.job_id,
        sessionId: started.execution_session_id,
        sourceSessionId: sessionId,
        projectId,
        eventsUrl: started.job.events_url,
      });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.questions(sessionId),
      });
    },
  });

  const staleGuidance = staleApprovalGuidance(execute.error);
  const [editing, setEditing] = useState(false);
  const highPriority = question.priority >= 0.7;

  return (
    <li>
      <Card tone="default" className="flex flex-col gap-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1.5 text-xs text-status-neutral">
            <Dot tone={highPriority ? "brand" : "neutral"} />
            {`${priorityLabel(question.priority)} priority`}
          </span>
          {question.exploratory && <Badge tone="warn">exploratory</Badge>}
          <span className="ml-auto">
            <ExecutionBadge projectId={projectId} question={question} />
          </span>
        </div>

        <p className="text-sm font-medium">{question.question}</p>

        {question.business_decision && (
          <p className="text-sm">
            <span className="mr-1.5 text-xs text-status-neutral">Decision</span>
            {question.business_decision}
          </p>
        )}
        {(question.risks ?? []).length > 0 && (
          <p className="text-sm">
            <span className="mr-1.5 text-xs text-status-neutral">Risks</span>
            {(question.risks ?? []).join(" | ")}
          </p>
        )}

        <GenreLine question={question} />

        {editing ? (
          <CardEditForm
            sessionId={sessionId}
            question={question}
            onDone={() => setEditing(false)}
          />
        ) : (
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="self-start text-xs text-primary underline-offset-2 hover:underline"
          >
            Edit card
          </button>
        )}

        {question.executable ? (
          prepare.data ? (
            staleGuidance ? (
              <Card
                tone="warn"
                role="alert"
                className="flex flex-col gap-2 p-3 text-sm"
              >
                <p className="font-medium text-status-warn">
                  {staleGuidance.message}
                </p>
                <p className="text-status-neutral">{staleGuidance.hint}</p>
                <button
                  type="button"
                  onClick={() => prepare.mutate()}
                  className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-bg"
                >
                  Approve again
                </button>
              </Card>
            ) : (
              <ConfirmCard
                prepared={prepare.data}
                pending={execute.isPending}
                onConfirm={() => execute.mutate(prepare.data)}
                onCancel={() => prepare.reset()}
              />
            )
          ) : (
            <Card tone="quiet" className="flex flex-col gap-2 p-3">
              <StepChain
                label="Run this question"
                steps={RUN_ONE_STEPS}
                current={0}
              />
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => prepare.mutate()}
                  disabled={prepare.isPending}
                  className="rounded-base border border-border bg-bg px-3 py-1.5 text-sm font-medium hover:bg-surface disabled:opacity-50"
                >
                  {prepare.isPending ? "Preparing…" : "Approve & run"}
                </button>
                <label className="flex items-center gap-1 text-xs text-status-neutral">
                  LLM mode
                  <select
                    value={llmMode}
                    onChange={(event) =>
                      setLlmMode(event.target.value as "env" | "offline")
                    }
                    className="rounded-base border border-border bg-bg px-1.5 py-1 text-xs"
                  >
                    <option value="env">env (autonomous agent)</option>
                    <option value="offline">offline (fixed SQL only)</option>
                  </select>
                </label>
                <span className="text-xs text-status-neutral">
                  Step 2 shows the approved scope and capabilities. Nothing
                  runs until you confirm there.
                </span>
              </div>
            </Card>
          )
        ) : (
          <Card tone="quiet" className="flex items-center gap-2 p-3">
            <Dot tone="warn" />
            <p className="text-xs text-status-neutral">
              Not executable: feasibility is{" "}
              {question.feasibility_status ?? "unknown"}.
            </p>
            <Hint label="Feasibility">
              The agent checked whether this session's data can answer the question
              and said no. Editing the card's framing does not change it —
              the missing data does.
            </Hint>
          </Card>
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
              : "Failed to execute the question."}
          </div>
        )}
      </Card>
    </li>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const questions = useQuestions(sessionId);

  if (questions.isPending) {
    return <LoadingSkeleton lines={4} label="Loading questions" />;
  }
  if (questions.isError) {
    return (
      <div className="p-6">
        <ErrorState
          error={questions.error}
          onRetry={() => questions.refetch()}
        />
      </div>
    );
  }
  const items = questions.data.questions ?? [];
  const hasSuggestions = items.length > 0;

  return (
    <div className="mx-auto flex w-[90%] max-w-data flex-col gap-5 p-6">
      <header>
        <SectionHeader
          level={1}
          title="Questions"
          description="Review a suggested question or ask your own, then approve what runs."
        />
      </header>

      {!hasSuggestions && (
        <DraftQuestionPanel
          projectId={projectId}
          sessionId={sessionId}
          hasSuggestions={false}
        />
      )}

      <section className="flex flex-col gap-3">
        <SectionHeader
          title="Suggested questions"
          description={
            hasSuggestions
              ? "Choose one question to review, approve, and execute."
              : "No questions were discovered for this session."
          }
          actions={
            hasSuggestions ? (
              <Badge>{`${items.length} suggested`}</Badge>
            ) : undefined
          }
        />
        {hasSuggestions ? (
          <ul aria-label="Question candidates" className="flex flex-col gap-3">
            {items.map((question) => (
              <QuestionCard
                key={question.question_id}
                projectId={projectId}
                sessionId={sessionId}
                question={question}
              />
            ))}
          </ul>
        ) : (
          <EmptyState
            title="No suggested questions yet"
            description="You can still draft and run your own question above."
          />
        )}
      </section>

      {hasSuggestions && (
        <DraftQuestionPanel
          projectId={projectId}
          sessionId={sessionId}
          hasSuggestions
        />
      )}
    </div>
  );
}
