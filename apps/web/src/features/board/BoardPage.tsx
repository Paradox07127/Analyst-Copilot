/* Investigation board slice (§5.4 / §10.3): a project-scoped kanban of leads.
 *
 * §5.4 requires every drag to have a keyboard alternative, an explicit drag
 * handle, cancel/undo, optimistic-update rollback, an `expected_version`
 * optimistic lock and a conflict prompt. Pointer drags go through dnd-kit;
 * the keyboard path is independent of dnd-kit's coordinate system on purpose —
 * grab a handle with Enter, move with the arrows, Enter commits, Escape
 * reverts to the pre-grab snapshot. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  closestCorners,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  api,
  ApiError,
  type BoardCard,
  type BoardColumn,
  type BoardView,
} from "../../api/client";
import { queryKeys, useBoard, useFindings, useQuestions } from "../../api/hooks";
import { ErrorState, LoadingSkeleton } from "../../components/async-states";
import {
  Badge,
  Card,
  Marquee,
  type Tone,
} from "../../components/ui";

export const BOARD_ID = "investigation";

/* The handle's own instructions live in the page header; point at them so a
 * screen reader reads them instead of dnd-kit's empty injected description. */
const DRAG_HELP_ID = "board-drag-help";

/* Long enough to notice a wrong drop, short enough that the button never
 * outlives the user's memory of what "last move" meant. */
const UNDO_TIMEOUT_MS = 30_000;

/* `finding: find_1` told the reader nothing they could act on. The kind is a
 * word, the id stays available for the person chasing the artifact. */
const REF_LABEL: Record<string, string> = {
  finding: "Finding",
  question: "Question",
};

const REF_TONE: Record<string, Tone> = {
  finding: "ok",
  question: "info",
};

/* Only the three columns the app itself creates; a server-defined column keeps
 * its own title and gets no invented meaning. */
const COLUMN_BLURB: Record<string, string> = {
  leads: "Worth a look. Nothing has been checked yet.",
  investigating: "Being worked on right now.",
  confirmed: "Checked against the data and backed by a finding.",
};

const DEFAULT_COLUMNS: BoardColumn[] = [
  { id: "leads", title: "Leads", card_ids: [] },
  { id: "investigating", title: "Investigating", card_ids: [] },
  { id: "confirmed", title: "Confirmed", card_ids: [] },
];

interface BoardState {
  version: number;
  columns: BoardColumn[];
  cards: BoardCard[];
}

interface KeyboardTransaction {
  cardId: string;
  baseVersion: number;
  baseColumns: BoardColumn[];
  draftColumns: BoardColumn[];
}

interface UndoLayout {
  columns: BoardColumn[];
}

function toState(board: BoardView): BoardState {
  const columns = board.columns ?? [];
  return {
    version: board.version ?? 0,
    columns: columns.length > 0 ? columns.map(cloneColumn) : DEFAULT_COLUMNS.map(cloneColumn),
    cards: (board.cards ?? []).map((card) => ({ ...card })),
  };
}

function cloneColumn(column: BoardColumn): BoardColumn {
  return { ...column, card_ids: [...(column.card_ids ?? [])] };
}

function columnIndexOfCard(columns: BoardColumn[], cardId: string): number {
  return columns.findIndex((column) => (column.card_ids ?? []).includes(cardId));
}

/* Single move primitive shared by pointer drops and arrow keys: remove the card
 * from wherever it is, then insert it at a clamped position in the target. */
function moveCard(
  state: BoardState,
  cardId: string,
  targetColumnIndex: number,
  targetIndex: number,
): BoardState {
  const columns = state.columns.map(cloneColumn);
  const from = columnIndexOfCard(columns, cardId);
  if (from < 0) return state;
  const to = Math.max(0, Math.min(targetColumnIndex, columns.length - 1));
  const fromColumn = columns[from];
  const toColumn = columns[to];
  if (!fromColumn || !toColumn) return state;
  fromColumn.card_ids = (fromColumn.card_ids ?? []).filter((id) => id !== cardId);
  const targetIds = [...(toColumn.card_ids ?? [])];
  const index = Math.max(0, Math.min(targetIndex, targetIds.length));
  targetIds.splice(index, 0, cardId);
  toColumn.card_ids = targetIds;
  return { ...state, columns };
}

function sameLayout(left: BoardState, right: BoardState): boolean {
  return (
    JSON.stringify(left.columns.map((c) => [c.id, c.card_ids])) ===
    JSON.stringify(right.columns.map((c) => [c.id, c.card_ids]))
  );
}

type HandleRegistry = Map<string, HTMLButtonElement>;

interface CardHandleProps {
  card: BoardCard;
  grabbed: boolean;
  handles: HandleRegistry;
  onGrabToggle: () => void;
  onCancel: () => void;
  onMove: (direction: "up" | "down" | "left" | "right") => void;
  onUpdate: (card: BoardCard) => void;
  onRemove: () => void;
}

function SortableCard({
  card,
  grabbed,
  handles,
  onGrabToggle,
  onCancel,
  onMove,
  onUpdate,
  onRemove,
}: CardHandleProps) {
  const [editing, setEditing] = useState(false);
  const [confirmingRemove, setConfirmingRemove] = useState(false);
  const [title, setTitle] = useState(card.title);
  const [note, setNote] = useState(card.note);
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: card.id, disabled: editing || confirmingRemove });

  useEffect(() => {
    if (editing) return;
    setTitle(card.title);
    setNote(card.note);
  }, [card.note, card.title, editing]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onGrabToggle();
      return;
    }
    if (event.key === "Escape" && grabbed) {
      event.preventDefault();
      onCancel();
      return;
    }
    if (!grabbed) return;
    const map: Record<string, "up" | "down" | "left" | "right"> = {
      ArrowUp: "up",
      ArrowDown: "down",
      ArrowLeft: "left",
      ArrowRight: "right",
    };
    const direction = map[event.key];
    if (direction) {
      event.preventDefault();
      onMove(direction);
    }
  };

  return (
    <li
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={`flex flex-col gap-1 rounded-base border p-2 text-sm ${
        grabbed ? "border-primary bg-primary/10" : "border-border bg-surface"
      } ${isDragging ? "opacity-60" : ""}`}
    >
      <div className="flex items-start gap-2">
        <button
          type="button"
          ref={(element) => {
            if (element) handles.set(card.id, element);
            else handles.delete(card.id);
          }}
          {...attributes}
          {...listeners}
          aria-label={`Move card: ${card.title}`}
          aria-describedby={DRAG_HELP_ID}
          aria-pressed={grabbed}
          onKeyDown={handleKeyDown}
          className="cursor-grab rounded-base border border-border px-1 text-xs text-status-neutral hover:bg-code-bg"
        >
          ⠿
        </button>
        <span className="min-w-0 flex-1 break-words">{card.title}</span>
        {!editing && !confirmingRemove && (
          <span className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={() => setEditing(true)}
              aria-label={`Edit card: ${card.title}`}
              className="rounded-base px-1.5 py-0.5 text-[11px] text-status-neutral hover:bg-bg hover:text-text"
            >
              Edit
            </button>
            <button
              type="button"
              onClick={() => setConfirmingRemove(true)}
              aria-label={`Remove card: ${card.title}`}
              className="rounded-base px-1.5 py-0.5 text-[11px] text-status-neutral hover:bg-status-critical/10 hover:text-status-critical"
            >
              Remove
            </button>
          </span>
        )}
      </div>
      {card.ref_type !== "none" && (
        <span className="flex flex-wrap items-center gap-1.5">
          <Badge tone={REF_TONE[card.ref_type] ?? "neutral"}>
            {REF_LABEL[card.ref_type] ?? card.ref_type}
          </Badge>
          <Marquee className="font-mono text-[10px] text-status-neutral" title={card.ref_id}>
            {card.ref_id}
          </Marquee>
        </span>
      )}
      {card.note && (
        <span className="text-xs text-status-neutral">{card.note}</span>
      )}
      {editing && (
        <form
          aria-label={`Edit ${card.title}`}
          className="mt-1 flex flex-col gap-2 border-t border-hairline pt-2"
          onSubmit={(event) => {
            event.preventDefault();
            const nextTitle = title.trim();
            if (!nextTitle) return;
            onUpdate({ ...card, title: nextTitle, note: note.trim() });
            setEditing(false);
          }}
        >
          <label className="flex flex-col gap-1 text-xs font-medium">
            Title
            <input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              maxLength={240}
              autoFocus
              className="rounded-base border border-border bg-bg px-2 py-1.5 text-sm font-normal"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs font-medium">
            Note
            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              rows={2}
              maxLength={1000}
              placeholder="Optional context or next step"
              className="resize-y rounded-base border border-border bg-bg px-2 py-1.5 text-sm font-normal"
            />
          </label>
          <span className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setTitle(card.title);
                setNote(card.note);
                setEditing(false);
              }}
              className="rounded-base px-2 py-1 text-xs text-status-neutral hover:bg-bg"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!title.trim()}
              className="rounded-base bg-primary px-2 py-1 text-xs font-medium text-bg disabled:opacity-50"
            >
              Save card
            </button>
          </span>
        </form>
      )}
      {confirmingRemove && (
        <div
          role="alert"
          className="mt-1 flex flex-wrap items-center justify-between gap-2 border-t border-status-critical/30 pt-2"
        >
          <span className="text-xs text-status-critical">
            Remove this card from the project board?
          </span>
          <span className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setConfirmingRemove(false)}
              className="rounded-base px-2 py-1 text-xs text-status-neutral hover:bg-bg"
            >
              Keep card
            </button>
            <button
              type="button"
              onClick={onRemove}
              aria-label={`Confirm remove ${card.title}`}
              className="rounded-base border border-status-critical/50 px-2 py-1 text-xs text-status-critical hover:bg-status-critical/10"
            >
              Remove
            </button>
          </span>
        </div>
      )}
    </li>
  );
}

function Column({
  column,
  cards,
  grabbedId,
  handles,
  onGrabToggle,
  onCancel,
  onMove,
  onUpdate,
  onRemove,
}: {
  column: BoardColumn;
  cards: BoardCard[];
  grabbedId: string | null;
  handles: HandleRegistry;
  onGrabToggle: (cardId: string) => void;
  onCancel: () => void;
  onMove: (cardId: string, direction: "up" | "down" | "left" | "right") => void;
  onUpdate: (card: BoardCard) => void;
  onRemove: (cardId: string) => void;
}) {
  const { setNodeRef, isOver } = useDroppable({ id: `column:${column.id}` });
  const cardIds = column.card_ids ?? [];
  const blurb = COLUMN_BLURB[column.id];
  return (
    <section
      ref={setNodeRef}
      aria-label={column.title}
      className={`flex min-w-0 flex-col gap-2 rounded-base border p-3 ${
        isOver ? "border-primary bg-primary/5" : "border-border"
      }`}
    >
      <h2 className="flex items-baseline gap-2 text-sm font-medium">
        {column.title}
        <span className="tabular text-status-neutral">({cardIds.length})</span>
      </h2>
      {blurb && <p className="text-xs text-status-neutral">{blurb}</p>}
      <SortableContext items={cardIds} strategy={verticalListSortingStrategy}>
        <ul className="flex flex-col gap-2">
          {cardIds.map((cardId) => {
            const card = cards.find((item) => item.id === cardId);
            if (!card) return null;
            return (
              <SortableCard
                key={card.id}
                card={card}
                grabbed={grabbedId === card.id}
                handles={handles}
                onGrabToggle={() => onGrabToggle(card.id)}
                onCancel={onCancel}
                onMove={(direction) => onMove(card.id, direction)}
                onUpdate={onUpdate}
                onRemove={() => onRemove(card.id)}
              />
            );
          })}
        </ul>
      </SortableContext>
      {/* Deliberately not an <li>: an empty column would otherwise report a
       * card to anything counting list items, here and in the tests. */}
      {cardIds.length === 0 && (
        <p className="rounded-base border border-dashed border-border px-2 py-3 text-center text-xs text-status-neutral">
          Nothing here. Drop a card, or move one with its handle.
        </p>
      )}
    </section>
  );
}

function AddCardForm({
  sessionId,
  onAdd,
}: {
  sessionId: string;
  onAdd: (card: BoardCard) => void;
}) {
  const findings = useFindings(sessionId);
  const questions = useQuestions(sessionId);
  const [selection, setSelection] = useState("");

  /* Two sources with different scopes end up in one picker — findings are the
   * project's (other sessions included), questions are this session's — so they are
   * grouped and labelled rather than flattened into one anonymous list. */
  const groups = useMemo(() => {
    const findingRows: { value: string; label: string; card: BoardCard }[] = [];
    const questionRows: { value: string; label: string; card: BoardCard }[] = [];
    for (const finding of findings.data?.findings ?? []) {
      findingRows.push({
        value: `finding:${finding.artifact_id}`,
        label: finding.from_current_session
          ? finding.question
          : `${finding.question} (from another session)`,
        card: {
          id: "",
          title: finding.question,
          ref_type: "finding",
          ref_id: finding.artifact_id,
          note: "",
        },
      });
    }
    for (const question of questions.data?.questions ?? []) {
      questionRows.push({
        value: `question:${question.question_id}`,
        label: question.question,
        card: {
          id: "",
          title: question.question,
          ref_type: "question",
          ref_id: question.question_id,
          note: "",
        },
      });
    }
    return [
      { key: "finding", label: "Findings in this project", rows: findingRows },
      { key: "question", label: "Questions raised in this session", rows: questionRows },
    ];
  }, [findings.data, questions.data]);

  const total = groups.reduce((sum, group) => sum + group.rows.length, 0);

  return (
    <form
      className="flex flex-wrap items-center gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        const option = groups
          .flatMap((group) => group.rows)
          .find((item) => item.value === selection);
        if (!option) return;
        onAdd({ ...option.card, id: crypto.randomUUID() });
        setSelection("");
      }}
    >
      <label className="flex items-center gap-1 text-xs text-status-neutral">
        Add card from
        <select
          aria-label="Add card from"
          value={selection}
          onChange={(event) => setSelection(event.target.value)}
          className="max-w-80 rounded-base border border-border bg-surface px-1.5 py-1 text-xs"
        >
          <option value="">Select a finding or question…</option>
          {groups.map((group) =>
            group.rows.length > 0 ? (
              <optgroup key={group.key} label={group.label}>
                {group.rows.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </optgroup>
            ) : null,
          )}
        </select>
      </label>
      <button
        type="submit"
        disabled={!selection}
        className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
      >
        Add card
      </button>
      {total === 0 && !findings.isPending && !questions.isPending && (
        <span className="text-xs text-status-neutral">
          Nothing to add yet — cards come from a finding this project recorded
          or a question this session raised.
        </span>
      )}
    </form>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const queryClient = useQueryClient();
  const board = useBoard(projectId, BOARD_ID);
  const serverState = useMemo(
    () => (board.data ? toState(board.data) : null),
    [board.data],
  );
  const [transaction, setTransaction] =
    useState<KeyboardTransaction | null>(null);
  const [pendingUpdate, setPendingUpdate] = useState<BoardState | null>(null);
  const [conflict, setConflict] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [undo, setUndo] = useState<UndoLayout | null>(null);
  const handles = useRef<HandleRegistry>(new Map());
  const grabbedId = transaction?.cardId ?? null;
  const state = useMemo(() => {
    if (!serverState) return null;
    if (pendingUpdate) return pendingUpdate;
    if (!transaction) return serverState;
    return {
      version: transaction.baseVersion,
      columns: transaction.draftColumns,
      cards: serverState.cards,
    };
  }, [pendingUpdate, serverState, transaction]);

  /* No KeyboardSensor: the arrow-key path below is independent of dnd-kit's
   * coordinate system, and a second keyboard handler would fight it. */
  const sensors = useSensors(useSensor(PointerSensor));

  useEffect(() => {
    if (!undo) return;
    const timer = setTimeout(() => setUndo(null), UNDO_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [undo]);

  /* A cross-column move remounts the card under a different list, which would
   * drop keyboard focus mid-grab; follow the card so the arrows keep working. */
  useEffect(() => {
    if (grabbedId) handles.current.get(grabbedId)?.focus();
  }, [grabbedId, state]);

  const save = useMutation({
    mutationFn: ({
      next,
      idempotencyKey,
    }: {
      next: BoardState;
      idempotencyKey: string;
    }) =>
      api.putBoard(projectId, BOARD_ID, {
        expected_version: next.version,
        columns: next.columns,
        cards: next.cards,
      }, idempotencyKey),
    /* A transport failure may mean the server committed but the response was
     * lost. Retry once with the SAME key so the durable backend record replays
     * the original result instead of turning the retry into a stale write. */
    retry: (failureCount, error) =>
      failureCount < 1 && !(error instanceof ApiError),
    retryDelay: 0,
    onSuccess: (saved) => {
      setConflict(false);
      queryClient.setQueryData(queryKeys.board(projectId, BOARD_ID), saved);
      setPendingUpdate(null);
    },
    onError: (error) => {
      setPendingUpdate(null);
      setConflict(error instanceof ApiError && error.code === "version_conflict");
    },
  });

  const commit = useCallback(
    (next: BoardState, onSaved?: () => void) => {
      setPendingUpdate(next);
      save.mutate({ next, idempotencyKey: crypto.randomUUID() }, {
        onSuccess: () => onSaved?.(),
      });
    },
    [save],
  );

  if (board.isError) {
    return (
      <div className="p-6">
        <ErrorState error={board.error} onRetry={() => board.refetch()} />
      </div>
    );
  }
  if (board.isPending || !state) {
    return <LoadingSkeleton lines={3} label="Loading board" />;
  }

  const draggedCard = activeId
    ? state.cards.find((card) => card.id === activeId)
    : undefined;

  const move = (
    cardId: string,
    direction: "up" | "down" | "left" | "right",
  ) => {
    const columnIndex = columnIndexOfCard(state.columns, cardId);
    if (columnIndex < 0) return;
    const column = state.columns[columnIndex];
    if (!column) return;
    const index = (column.card_ids ?? []).indexOf(cardId);
    const next =
      direction === "up" || direction === "down"
        ? moveCard(state, cardId, columnIndex, index + (direction === "up" ? -1 : 1))
        : moveCard(
            state,
            cardId,
            columnIndex + (direction === "left" ? -1 : 1),
            index,
          );
    setTransaction((current) =>
      current
        ? {
            ...current,
            draftColumns: next.columns.map(cloneColumn),
          }
        : current,
    );
    const landed = state.columns[columnIndexOfCard(next.columns, cardId)];
    setAnnouncement(
      `Card moved to ${landed?.title ?? "column"}. Press Enter to confirm, Escape to cancel.`,
    );
  };

  const grabToggle = (cardId: string) => {
    if (transaction?.cardId === cardId) {
      const before: BoardState = {
        version: transaction.baseVersion,
        columns: transaction.baseColumns,
        cards: state.cards,
      };
      setTransaction(null);
      if (!sameLayout(before, state)) {
        commit({ ...state, version: transaction.baseVersion }, () => {
          setUndo({ columns: transaction.baseColumns.map(cloneColumn) });
          setAnnouncement("Card position saved.");
        });
        setAnnouncement("Saving card position.");
      } else {
        setAnnouncement("Card position unchanged.");
      }
      return;
    }
    setTransaction({
      cardId,
      baseVersion: state.version,
      baseColumns: state.columns.map(cloneColumn),
      draftColumns: state.columns.map(cloneColumn),
    });
    setAnnouncement("Card grabbed. Use the arrow keys to move it, Enter to confirm.");
  };

  const cancelGrab = () => {
    setTransaction(null);
    setAnnouncement("Move cancelled.");
  };

  const onDragEnd = (event: DragEndEvent) => {
    setActiveId(null);
    const draggedId = String(event.active.id);
    const overId = event.over ? String(event.over.id) : null;
    if (!overId || overId === draggedId) return;
    const before = { ...state, columns: state.columns.map(cloneColumn) };
    const targetColumnIndex = overId.startsWith("column:")
      ? state.columns.findIndex((column) => `column:${column.id}` === overId)
      : columnIndexOfCard(state.columns, overId);
    if (targetColumnIndex < 0) return;
    const targetColumn = state.columns[targetColumnIndex];
    const targetIndex = overId.startsWith("column:")
      ? (targetColumn?.card_ids ?? []).length
      : (targetColumn?.card_ids ?? []).indexOf(overId);
    commit(moveCard(state, draggedId, targetColumnIndex, targetIndex), () => {
      setUndo({ columns: before.columns });
      setAnnouncement("Card position saved.");
    });
  };

  /* Reverse the last committed move as a new write against the CURRENT
   * version, so it goes through the same optimistic lock as any other edit. */
  const undoLastMove = () => {
    if (!undo) return;
    commit({ ...state, columns: undo.columns.map(cloneColumn) }, () => {
      setUndo(null);
      setAnnouncement("Last move undone.");
    });
  };

  const addCard = (card: BoardCard) => {
    const columns = state.columns.map(cloneColumn);
    const first = columns[0];
    if (!first) return;
    first.card_ids = [...(first.card_ids ?? []), card.id];
    commit({ ...state, columns, cards: [...state.cards, card] });
  };

  const updateCard = (card: BoardCard) => {
    commit({
      ...state,
      cards: state.cards.map((item) => (item.id === card.id ? card : item)),
    });
    setAnnouncement(`Saving changes to ${card.title}.`);
  };

  const removeCard = (cardId: string) => {
    const card = state.cards.find((item) => item.id === cardId);
    commit({
      ...state,
      columns: state.columns.map((column) => ({
        ...cloneColumn(column),
        card_ids: (column.card_ids ?? []).filter((id) => id !== cardId),
      })),
      cards: state.cards.filter((item) => item.id !== cardId),
    });
    setAnnouncement(`${card?.title ?? "Card"} removed from the board.`);
  };

  return (
    <div className="mx-auto flex w-[90%] max-w-data min-w-0 flex-col gap-4 p-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold">Investigation board</h1>
        {/* The board belongs to the project, not to the session in the URL — every
         * run of this project opens this same one. */}
        <p className="max-w-content text-sm text-status-neutral">
          One board per project: every session in this project opens these same
          cards. Findings come from the whole project, questions from the session
          you have open.
        </p>
        <p id={DRAG_HELP_ID} className="max-w-content text-sm text-status-neutral">
          Drag a card by its handle, or focus a handle and press Enter to grab
          it — arrow keys move it, Enter confirms, Escape cancels.
        </p>
        <p className="flex items-center gap-2 text-xs text-status-neutral">
          <span className="tabular">Version {state.version}</span>
          <span>·</span>
          <span>
            {save.isPending
              ? "Saving…"
              : "Every move saves as soon as you confirm it."}
          </span>
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <AddCardForm sessionId={sessionId} onAdd={addCard} />
        {undo && (
          <button
            type="button"
            onClick={undoLastMove}
            disabled={save.isPending}
            className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
          >
            Undo last move
          </button>
        )}
      </div>

      {conflict && (
        <Card
          tone="warn"
          role="alert"
          className="flex flex-col gap-2 p-3 text-sm"
        >
          <p className="font-medium text-status-warn">
            Someone else changed this board.
          </p>
          <p className="text-status-neutral">
            Your move was rolled back. Reload the board and try again.
          </p>
          <button
            type="button"
            onClick={() => {
              setConflict(false);
              void board.refetch();
            }}
            className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface"
          >
            Reload board
          </button>
        </Card>
      )}

      {save.isError && !conflict && (
        <p role="alert" className="text-sm text-status-critical">
          {save.error instanceof Error
            ? save.error.message
            : "Failed to save the board."}
        </p>
      )}

      <p aria-live="polite" className="sr-only">
        {announcement}
      </p>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCorners}
        onDragStart={(event: DragStartEvent) =>
          setActiveId(String(event.active.id))
        }
        onDragCancel={() => setActiveId(null)}
        onDragEnd={onDragEnd}
      >
        <div className="grid w-full grid-cols-[repeat(auto-fit,minmax(min(100%,16rem),1fr))] gap-3">
          {state.columns.map((column) => (
            <Column
              key={column.id}
              column={column}
              cards={state.cards}
              grabbedId={grabbedId}
              handles={handles.current}
              onGrabToggle={grabToggle}
              onCancel={cancelGrab}
              onMove={move}
              onUpdate={updateCard}
              onRemove={removeCard}
            />
          ))}
        </div>
        <DragOverlay>
          {draggedCard ? (
            <div
              data-testid="drag-preview"
              className="rounded-base border border-primary bg-surface p-2 text-sm shadow-overlay"
            >
              {draggedCard.title}
            </div>
          ) : null}
        </DragOverlay>
      </DndContext>
    </div>
  );
}
