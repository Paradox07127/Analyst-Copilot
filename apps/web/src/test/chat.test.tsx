import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { FakeEventSource } from "./fake-event-source";
import { renderAppAt } from "./render";

const RUN = "r1";

function frame(seq: number, type: string, data: Record<string, unknown>) {
  return { seq, session_id: RUN, message_id: "msg_1", type, data };
}

function pendingPlanFrame(seq: number) {
  return frame(seq, "plan.pending", {
    plan_id: "chatplan_1",
    action_hash: "a".repeat(64),
    approval_token: "c".repeat(32),
    question: "Total amount by region",
    method: "group_by",
    sql: "SELECT region, sum(amount) FROM orders GROUP BY region",
    dataset_names: ["orders.csv"],
    estimated_scan: "small",
  });
}

async function sendMessage(text = "which column has the most missing values?") {
  const user = userEvent.setup();
  renderAppAt(`/projects/p1/sessions/${RUN}/chat`);
  await screen.findByRole("heading", { name: "Chat" });
  /* fireEvent, not user.type: react-resizable-panels' document-level
   * pointerdown handler preventDefaults clicks at jsdom's all-zero rects,
   * so user-event's click never focuses the textarea (see launchpad.test). */
  fireEvent.change(screen.getByLabelText("Message"), { target: { value: text } });
  await user.click(screen.getByRole("button", { name: "Send" }));
  const source = await screen.findByText("Streaming…").then(() =>
    FakeEventSource.latest(),
  );
  return { user, source };
}

describe("Chat page", () => {
  it("renders the persisted transcript newest-last", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/chat/messages", () =>
        HttpResponse.json({
          session_id: RUN,
          total: 2,
          next_cursor: null,
          messages: [
            {
              seq: 0,
              role: "user",
              content: "hello there",
              status: "answer",
              sql: null,
              artifact_refs: [],
              created_at: null,
            },
            {
              seq: 1,
              role: "assistant",
              content: "orders.csv: 20 rows x 2 columns",
              status: "answer",
              sql: "select 1",
              artifact_refs: ["prof_1"],
              created_at: null,
            },
          ],
        }),
      ),
    );
    renderAppAt(`/projects/p1/sessions/${RUN}/chat`);
    expect(await screen.findByText("hello there")).toBeInTheDocument();
    expect(
      screen.getByText("orders.csv: 20 rows x 2 columns"),
    ).toBeInTheDocument();
    expect(screen.getByText("select 1")).toBeInTheDocument();

    const conversation = screen.getByRole("region", { name: "Conversation" });
    const composer = screen.getByRole("form", {
      name: "Ask about this session",
    });
    expect(conversation).not.toContainElement(composer);
  });

  it("keeps the reader's position while live updates arrive", async () => {
    const { user, source } = await sendMessage("summarize the current evidence");
    const conversation = screen.getByRole("region", { name: "Conversation" });
    const scrollTo = vi.fn();
    Object.defineProperties(conversation, {
      scrollHeight: { configurable: true, value: 1_200 },
      clientHeight: { configurable: true, value: 300 },
      scrollTop: { configurable: true, value: 180, writable: true },
      scrollTo: { configurable: true, value: scrollTo },
    });

    fireEvent.scroll(conversation);
    const jump = await screen.findByRole("button", { name: "Jump to latest" });
    scrollTo.mockClear();

    act(() =>
      source.emit(
        "progress",
        frame(1, "progress", { stage: "planning" }),
      ),
    );
    expect(screen.getByText("Planning the analysis…")).toBeInTheDocument();
    await waitFor(() => expect(scrollTo).not.toHaveBeenCalled());

    await user.click(jump);
    expect(scrollTo).toHaveBeenCalledWith({
      top: 1_200,
      behavior: "smooth",
    });
    expect(
      screen.queryByRole("button", { name: "Jump to latest" }),
    ).not.toBeInTheDocument();
  });

  it("preserves the visible position when older messages are prepended", async () => {
    server.use(
      http.get(
        "/api/v1/sessions/:sessionId/chat/messages",
        ({ request }) => {
          const cursor = new URL(request.url).searchParams.get("cursor");
          return HttpResponse.json({
            session_id: RUN,
            total: 2,
            next_cursor: cursor ? null : "older",
            messages: [
              {
                seq: cursor ? 0 : 1,
                role: cursor ? "user" : "assistant",
                content: cursor ? "older question" : "newest answer",
                status: "answer",
                sql: null,
                artifact_refs: [],
                created_at: null,
              },
            ],
          });
        },
      ),
    );
    renderAppAt(`/projects/p1/sessions/${RUN}/chat`);
    expect(await screen.findByText("newest answer")).toBeInTheDocument();

    const conversation = screen.getByRole("region", { name: "Conversation" });
    let scrollHeight = 600;
    Object.defineProperties(conversation, {
      scrollHeight: {
        configurable: true,
        get: () => scrollHeight,
      },
      clientHeight: { configurable: true, value: 300 },
      scrollTop: { configurable: true, value: 40, writable: true },
    });

    fireEvent.click(
      screen.getByRole("button", { name: "Load older messages" }),
    );
    scrollHeight = 1_000;

    expect(await screen.findByText("older question")).toBeInTheDocument();
    await waitFor(() => expect(conversation.scrollTop).toBe(440));
  });

  it("streams progress, tool calls and the final message", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts/sql_1", () =>
        HttpResponse.json({
          artifact_id: "sql_1",
          type: "SqlResult",
          project_id: "p1",
          session_id: RUN,
          created_at: null,
          payload: {
            sql: "select region, total from orders",
            columns: ["region", "total"],
            dtypes: { region: "string", total: "float64" },
            units: {},
            rows_preview: [
              { region: "North", total: 1200 },
              { region: "South", total: 800 },
            ],
            row_count: 2,
            truncated: false,
          },
          warnings: ["Totals exclude cancelled orders."],
        }),
      ),
    );
    const { source } = await sendMessage();
    expect(source.url).toContain("message_id=msg_1");

    act(() => source.emit("turn.started", frame(1, "turn.started", { stage: "loading_datasets" })));
    expect(screen.getByText("Loading this session's data…")).toBeInTheDocument();

    act(() => source.emit("progress", frame(2, "progress", { stage: "planning" })));
    expect(screen.getByText("Planning the analysis…")).toBeInTheDocument();

    act(() =>
      source.emit(
        "tool.call",
        frame(3, "tool.call", {
          trace_type: "tool_completed",
          name: "run_sql",
          summary: { sql: "select region from orders", row_count: 4 },
        }),
      ),
    );
    expect(screen.getByText("Ran tool")).toBeInTheDocument();
    expect(screen.getByText("select region from orders")).toBeInTheDocument();
    expect(screen.getByText("4 rows")).toBeInTheDocument();

    act(() =>
      source.emit(
        "message.completed",
        frame(4, "message.completed", {
          content: "Ran SQL analysis: 4 rows returned.",
          status: "answer",
          sql: "select region from orders",
          artifact_refs: ["sql_1"],
          validation: {
            status: "warn",
            findings: ["One segment has a small sample."],
          },
        }),
      ),
    );
    expect(
      await screen.findByRole("button", { name: "Send" }),
    ).toBeInTheDocument();
    const table = await screen.findByRole("table", { name: "SQL result sql_1" });
    expect(table).toHaveTextContent("North");
    expect(table).toHaveTextContent("1,200");
    expect(screen.getByText("Quick visualization")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Automatically inferred from the two-column preview.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: /Ran SQL analysis.*quick visualization/,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Validation: warn")).toBeInTheDocument();
    expect(screen.getByText("One segment has a small sample.")).toBeInTheDocument();
    expect(screen.getByText("Totals exclude cancelled orders.")).toBeInTheDocument();
    // Terminal frame ends the server stream; the client must not reconnect.
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });

  it("surfaces a turn failure", async () => {
    const { source } = await sendMessage();
    act(() =>
      source.emit(
        "turn.failed",
        frame(1, "turn.failed", {
          code: "datasets_unavailable",
          message: "This session's source data could not be reloaded.",
        }),
      ),
    );
    expect(
      await screen.findByText("This session's source data could not be reloaded."),
    ).toBeInTheDocument();
  });

  it("shows an approval card for a plan and does not run it unprompted", async () => {
    let approveCalls = 0;
    server.use(
      http.post("/api/v1/sessions/:sessionId/chat/plans/:planId/approve", () => {
        approveCalls += 1;
        return HttpResponse.json(
          {
            session_id: RUN,
            message_id: "msg_approved",
            stream_url: `/api/v1/sessions/${RUN}/chat/stream?message_id=msg_approved`,
          },
          { status: 202 },
        );
      }),
    );

    const { user, source } = await sendMessage("total amount by region");
    act(() => source.emit("plan.pending", pendingPlanFrame(1)));

    const card = await screen.findByRole("alertdialog", {
      name: "Approve analysis plan",
    });
    expect(card).toHaveTextContent("Total amount by region");
    expect(card).toHaveTextContent(
      "SELECT region, sum(amount) FROM orders GROUP BY region",
    );
    // Rendering the card must not have executed anything.
    expect(approveCalls).toBe(0);

    await user.click(screen.getByRole("button", { name: "Approve & run" }));
    expect(approveCalls).toBe(1);
    // Approving opens a fresh stream for the continuation turn.
    expect(FakeEventSource.latest().url).toContain("message_id=msg_approved");
  });

  it("rejects a pending plan without starting a turn", async () => {
    let rejectCalls = 0;
    server.use(
      http.post("/api/v1/sessions/:sessionId/chat/plans/:planId/reject", ({ params }) => {
        rejectCalls += 1;
        return HttpResponse.json({
          session_id: RUN,
          plan_id: String(params["planId"]),
          status: "rejected",
        });
      }),
    );

    const { user, source } = await sendMessage("total amount by region");
    act(() => source.emit("plan.pending", pendingPlanFrame(1)));
    await screen.findByRole("alertdialog", { name: "Approve analysis plan" });

    await user.click(screen.getByRole("button", { name: "Reject" }));
    expect(rejectCalls).toBe(1);
    expect(
      screen.queryByRole("alertdialog", { name: "Approve analysis plan" }),
    ).not.toBeInTheDocument();
  });

  it("recovers the approval card for a plan stranded by a lost stream", async () => {
    let pendingCalls = 0;
    let approveBody: Record<string, unknown> | null = null;
    server.use(
      http.get("/api/v1/sessions/:sessionId/chat/messages", () =>
        HttpResponse.json({
          session_id: RUN,
          total: 1,
          next_cursor: null,
          messages: [
            {
              seq: 0,
              role: "assistant",
              content: "This analysis plan requires approval before execution.",
              status: "awaiting_approval",
              sql: null,
              artifact_refs: [],
              created_at: null,
            },
          ],
        }),
      ),
      http.get("/api/v1/sessions/:sessionId/chat/pending-plans", () => {
        pendingCalls += 1;
        return HttpResponse.json({
          session_id: RUN,
          plans: [
            {
              plan_id: "chatplan_1",
              action_hash: "a".repeat(64),
              /* A re-issued token: the one the lost stream carried is dead. */
              approval_token: "reissued-token",
              expires_at: "2099-01-01T00:00:00Z",
              question: "Total amount by region",
              method: "group_by",
              sql: "SELECT region, sum(amount) FROM orders GROUP BY region",
              dataset_names: ["orders.csv"],
              estimated_scan: "small",
            },
          ],
        });
      }),
      http.post(
        "/api/v1/sessions/:sessionId/chat/plans/:planId/approve",
        async ({ request }) => {
          approveBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              session_id: RUN,
              message_id: "msg_recovered",
              stream_url: `/api/v1/sessions/${RUN}/chat/stream?message_id=msg_recovered`,
            },
            { status: 202 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(`/projects/p1/sessions/${RUN}/chat`);
    await screen.findByRole("heading", { name: "Chat" });

    const card = await screen.findByRole("alertdialog", {
      name: "Approve analysis plan",
    });
    expect(card).toHaveTextContent("Total amount by region");
    expect(pendingCalls).toBe(1);

    await user.click(screen.getByRole("button", { name: "Approve & run" }));
    expect(approveBody).toEqual({
      action_hash: "a".repeat(64),
      approval_token: "reissued-token",
    });
    expect(FakeEventSource.latest().url).toContain("message_id=msg_recovered");
  });

  it("does not ask for pending plans when the transcript is settled", async () => {
    let pendingCalls = 0;
    server.use(
      http.get("/api/v1/sessions/:sessionId/chat/messages", () =>
        HttpResponse.json({
          session_id: RUN,
          total: 1,
          next_cursor: null,
          messages: [
            {
              seq: 0,
              role: "assistant",
              content: "orders.csv: 20 rows",
              status: "answer",
              sql: null,
              artifact_refs: [],
              created_at: null,
            },
          ],
        }),
      ),
      http.get("/api/v1/sessions/:sessionId/chat/pending-plans", () => {
        pendingCalls += 1;
        return HttpResponse.json({ session_id: RUN, plans: [] });
      }),
    );

    renderAppAt(`/projects/p1/sessions/${RUN}/chat`);
    expect(await screen.findByText("orders.csv: 20 rows")).toBeInTheDocument();
    expect(
      screen.queryByRole("alertdialog", { name: "Approve analysis plan" }),
    ).not.toBeInTheDocument();
    expect(pendingCalls).toBe(0);
  });

  it("explains a busy run instead of dropping the message", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/chat/messages", () =>
        HttpResponse.json(
          {
            error: {
              code: "chat_busy",
              message: "Session r1 already has a chat turn in flight.",
            },
          },
          { status: 409 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(`/projects/p1/sessions/${RUN}/chat`);
    await screen.findByRole("heading", { name: "Chat" });
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "hello" },
    });
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(
      await screen.findByText(
        "A turn is already running for this session. Wait for it to finish.",
      ),
    ).toBeInTheDocument();
  });
});
