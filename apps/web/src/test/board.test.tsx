import { beforeAll, describe, expect, it } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { QueryClient } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router";
import type { BoardView } from "../api/client";
import { AppProviders } from "../app/providers";
import { routes } from "../app/router";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

const PROJECT = "p1";
const RUN = "r1";

type BoardColumns = NonNullable<BoardView["columns"]>;

/* jsdom ships no PointerEvent and no pointer capture, so dnd-kit's
 * PointerSensor never activates without these. */
class TestPointerEvent extends MouseEvent {
  isPrimary = true;
  pointerType = "mouse";
  pointerId: number;
  constructor(type: string, init: MouseEventInit & { pointerId?: number } = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
  }
}

beforeAll(() => {
  (globalThis as unknown as { PointerEvent: unknown }).PointerEvent =
    TestPointerEvent;
  Element.prototype.setPointerCapture = () => {};
  Element.prototype.releasePointerCapture = () => {};
  Element.prototype.hasPointerCapture = () => false;
});

/* Every rect is 0×0 in jsdom, so collision detection cannot tell the columns
 * apart until they are given distinct geometry. */
function stubRect(element: Element, left: number, width: number): void {
  element.getBoundingClientRect = () =>
    ({
      x: left,
      y: 0,
      left,
      top: 0,
      width,
      height: 300,
      right: left + width,
      bottom: 300,
      toJSON: () => ({}),
    }) as DOMRect;
}

function seededBoard(version = 3): BoardView {
  return {
    project_id: PROJECT,
    board_id: "investigation",
    version,
    columns: [
      { id: "leads", title: "Leads", card_ids: ["c1", "c2"] },
      { id: "investigating", title: "Investigating", card_ids: [] },
      { id: "confirmed", title: "Confirmed", card_ids: [] },
    ],
    cards: [
      {
        id: "c1",
        title: "Revenue dip in March",
        ref_type: "finding",
        ref_id: "find_1",
        note: "",
      },
      {
        id: "c2",
        title: "Duplicate customer rows",
        ref_type: "question",
        ref_id: "q_dupes",
        note: "",
      },
    ],
  };
}

function useSeededBoard(version = 3) {
  server.use(
    http.get("/api/v1/projects/:projectId/boards/:boardId", () =>
      HttpResponse.json(seededBoard(version)),
    ),
  );
}

function cardTitlesIn(columnName: string): string[] {
  const column = screen.getByRole("region", { name: columnName });
  return within(column)
    .queryAllByRole("listitem")
    .map((item) => item.textContent ?? "");
}

async function openBoard() {
  const user = userEvent.setup();
  renderAppAt(`/projects/${PROJECT}/sessions/${RUN}/board`);
  await screen.findByRole("heading", { name: "Investigation board" });
  return user;
}

async function openBoardWithClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  const router = createMemoryRouter(routes, {
    initialEntries: [`/projects/${PROJECT}/sessions/${RUN}/board`],
  });
  render(
    <AppProviders client={client}>
      <RouterProvider router={router} />
    </AppProviders>,
  );
  await screen.findByRole("heading", { name: "Investigation board" });
  return { client, user: userEvent.setup() };
}

describe("Investigation board", () => {
  it("renders columns and cards from the stored board", async () => {
    useSeededBoard();
    await openBoard();
    expect(await screen.findByText("Revenue dip in March")).toBeInTheDocument();
    expect(cardTitlesIn("Leads")).toHaveLength(2);
    expect(cardTitlesIn("Investigating")).toHaveLength(0);
    expect(screen.getByText(/Version 3/)).toBeInTheDocument();
  });

  /* The board is project state reached through a run URL; saying so is the
   * difference between "my card vanished" and "that was another run". */
  it("says the board belongs to the project, not the open run", async () => {
    useSeededBoard();
    await openBoard();
    expect(
      screen.getByText(/One board per project: every session in this project/),
    ).toBeInTheDocument();
  });

  it("names the kind of thing a card points at, keeping the id", async () => {
    useSeededBoard();
    await openBoard();
    const card = (await screen.findByText("Revenue dip in March")).closest("li");
    expect(card).not.toBeNull();
    expect(within(card!).getByText("Finding")).toBeInTheDocument();
    /* The raw id stays reachable for whoever has to chase the artifact. */
    expect(within(card!).getByText("find_1")).toBeInTheDocument();
    expect(within(card!).queryByText("finding: find_1")).not.toBeInTheDocument();
  });

  it("an empty column says what it is for without faking a card", async () => {
    useSeededBoard();
    await openBoard();
    const empty = screen.getByRole("region", { name: "Investigating" });
    expect(within(empty).getByText(/Nothing here/)).toBeInTheDocument();
    expect(within(empty).getByText(/Being worked on right now/)).toBeInTheDocument();
    expect(cardTitlesIn("Investigating")).toHaveLength(0);
  });

  it("moves a card across columns with the keyboard and saves on Enter", async () => {
    let putBody: Record<string, unknown> | null = null;
    useSeededBoard();
    server.use(
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          putBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            ...seededBoard(4),
            columns: putBody["columns"],
            cards: putBody["cards"],
          });
        },
      ),
    );

    const user = await openBoard();
    await screen.findByText("Revenue dip in March");

    const handle = screen.getByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    /* Tab focus → Enter grabs → arrow moves → Enter confirms (§5.4). */
    handle.focus();
    expect(handle).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(handle).toHaveAttribute("aria-pressed", "true");

    await user.keyboard("{ArrowRight}");
    expect(cardTitlesIn("Investigating")).toHaveLength(1);
    expect(cardTitlesIn("Leads")).toHaveLength(1);
    // Moving alone must not persist anything: Enter is the commit.
    expect(putBody).toBeNull();

    await user.keyboard("{Enter}");
    /* Re-query: the cross-column move remounted the card, so the original
     * node is detached — focus followed the card to the new handle. */
    const movedHandle = screen.getByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    expect(movedHandle).toHaveAttribute("aria-pressed", "false");
    expect(movedHandle).toHaveFocus();
    await screen.findByText(/Version 4/);

    expect(putBody).not.toBeNull();
    const body = putBody as unknown as {
      expected_version: number;
      columns: { id: string; card_ids: string[] }[];
    };
    expect(body.expected_version).toBe(3);
    expect(body.columns.find((c) => c.id === "investigating")?.card_ids).toEqual([
      "c1",
    ]);
  });

  it("reuses one Idempotency-Key when a lost Board PUT response is retried", async () => {
    const keys: Array<string | null> = [];
    let attempt = 0;
    useSeededBoard();
    server.use(
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          keys.push(request.headers.get("Idempotency-Key"));
          attempt += 1;
          if (attempt === 1) return HttpResponse.error();
          const body = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            ...seededBoard(4),
            columns: body["columns"],
            cards: body["cards"],
          });
        },
      ),
    );

    const user = await openBoard();
    const handle = await screen.findByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    handle.focus();
    await user.keyboard("{Enter}{ArrowRight}{Enter}");

    await screen.findByText(/Version 4/);
    expect(keys).toHaveLength(2);
    expect(keys[0]).toMatch(/^[0-9a-f-]{36}$/);
    expect(keys[1]).toBe(keys[0]);
  });

  it("reorders within a column with the arrow keys", async () => {
    useSeededBoard();
    server.use(
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          const body = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            ...seededBoard(4),
            columns: body["columns"],
            cards: body["cards"],
          });
        },
      ),
    );

    const user = await openBoard();
    await screen.findByText("Revenue dip in March");
    expect(cardTitlesIn("Leads")[0]).toContain("Revenue dip in March");

    const handle = screen.getByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    handle.focus();
    await user.keyboard("{Enter}{ArrowDown}");
    expect(cardTitlesIn("Leads")[0]).toContain("Duplicate customer rows");
    expect(cardTitlesIn("Leads")[1]).toContain("Revenue dip in March");
    await user.keyboard("{Enter}");
    await screen.findByText(/Version 4/);
  });

  it("Escape cancels a keyboard move and restores the original order", async () => {
    let putCalls = 0;
    useSeededBoard();
    server.use(
      http.put("/api/v1/projects/:projectId/boards/:boardId", () => {
        putCalls += 1;
        return HttpResponse.json(seededBoard(4));
      }),
    );

    const user = await openBoard();
    await screen.findByText("Revenue dip in March");
    const handle = screen.getByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    handle.focus();
    await user.keyboard("{Enter}{ArrowRight}");
    expect(cardTitlesIn("Investigating")).toHaveLength(1);

    await user.keyboard("{Escape}");
    expect(cardTitlesIn("Investigating")).toHaveLength(0);
    expect(cardTitlesIn("Leads")).toHaveLength(2);
    expect(putCalls).toBe(0);
  });

  it("keeps a keyboard transaction across refetch and cancels onto fresh server state", async () => {
    let refreshed = false;
    const fresh = seededBoard(8);
    fresh.columns = [
      { id: "leads", title: "Leads", card_ids: ["c2"] },
      { id: "investigating", title: "Investigating", card_ids: [] },
      { id: "confirmed", title: "Confirmed", card_ids: ["c1"] },
    ];
    server.use(
      http.get("/api/v1/projects/:projectId/boards/:boardId", () =>
        HttpResponse.json(refreshed ? fresh : seededBoard()),
      ),
    );
    const { client, user } = await openBoardWithClient();
    const handle = await screen.findByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    handle.focus();
    await user.keyboard("{Enter}{ArrowRight}");
    expect(cardTitlesIn("Investigating")).toHaveLength(1);

    refreshed = true;
    await client.refetchQueries();
    await screen.findByText(/Version 3/);
    // The draft remains visible and anchored to version 3 while the refetch is
    // held underneath it; no effect overwrites the keyboard transaction.
    expect(cardTitlesIn("Investigating")).toHaveLength(1);

    await user.keyboard("{Escape}");
    await screen.findByText(/Version 8/);
    expect(cardTitlesIn("Investigating")).toHaveLength(0);
    expect(cardTitlesIn("Confirmed")).toHaveLength(1);
    expect(cardTitlesIn("Leads")).toHaveLength(1);
  });

  it("submits the anchored base version after a mid-grab refetch and rolls back to fresh state on conflict", async () => {
    let refreshed = false;
    let expectedVersion: number | null = null;
    const fresh = seededBoard(8);
    fresh.columns = [
      { id: "leads", title: "Leads", card_ids: ["c2"] },
      { id: "investigating", title: "Investigating", card_ids: [] },
      { id: "confirmed", title: "Confirmed", card_ids: ["c1"] },
    ];
    server.use(
      http.get("/api/v1/projects/:projectId/boards/:boardId", () =>
        HttpResponse.json(refreshed ? fresh : seededBoard()),
      ),
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          const body = (await request.json()) as { expected_version: number };
          expectedVersion = body.expected_version;
          return HttpResponse.json(
            {
              error: {
                code: "version_conflict",
                message: "Board changed since it was loaded.",
              },
            },
            { status: 409 },
          );
        },
      ),
    );
    const { client, user } = await openBoardWithClient();
    const handle = await screen.findByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    handle.focus();
    await user.keyboard("{Enter}{ArrowRight}");

    refreshed = true;
    await client.refetchQueries();
    await user.keyboard("{Enter}");

    expect(
      await screen.findByText("Someone else changed this board."),
    ).toBeInTheDocument();
    expect(expectedVersion).toBe(3);
    expect(screen.getByText(/Version 8/)).toBeInTheDocument();
    expect(cardTitlesIn("Confirmed")).toHaveLength(1);
    expect(cardTitlesIn("Investigating")).toHaveLength(0);
    expect(
      screen.queryByRole("button", { name: "Undo last move" }),
    ).not.toBeInTheDocument();
  });

  it("rolls the move back and offers a reload on a version conflict", async () => {
    useSeededBoard();
    server.use(
      http.put("/api/v1/projects/:projectId/boards/:boardId", () =>
        HttpResponse.json(
          {
            error: {
              code: "version_conflict",
              message: "Board changed since it was loaded.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = await openBoard();
    await screen.findByText("Revenue dip in March");
    const handle = screen.getByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    handle.focus();
    await user.keyboard("{Enter}{ArrowRight}{Enter}");

    expect(
      await screen.findByText("Someone else changed this board."),
    ).toBeInTheDocument();
    // Optimistic move rolled back to the pre-move layout.
    expect(cardTitlesIn("Investigating")).toHaveLength(0);
    expect(cardTitlesIn("Leads")).toHaveLength(2);
    expect(
      screen.getByRole("button", { name: "Reload board" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Undo last move" }),
    ).not.toBeInTheDocument();
  });

  it("offers a retry when the board fails to load", async () => {
    let failing = true;
    server.use(
      http.get("/api/v1/projects/:projectId/boards/:boardId", () =>
        failing
          ? HttpResponse.json(
              {
                error: {
                  code: "project_not_found",
                  message: "Project p1 does not exist.",
                },
              },
              { status: 404 },
            )
          : HttpResponse.json(seededBoard()),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(`/projects/${PROJECT}/sessions/${RUN}/board`);

    expect(
      await screen.findByText("Request failed (project_not_found)"),
    ).toBeInTheDocument();
    expect(screen.getByText("Project p1 does not exist.")).toBeInTheDocument();
    const retry = screen.getByRole("button", { name: "Retry" });

    failing = false;
    await user.click(retry);
    expect(await screen.findByText("Revenue dip in March")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Retry" }),
    ).not.toBeInTheDocument();
  });

  it("previews the dragged card and highlights the column under it", async () => {
    useSeededBoard();
    await openBoard();
    await screen.findByText("Revenue dip in March");

    const investigating = screen.getByRole("region", { name: "Investigating" });
    stubRect(screen.getByRole("region", { name: "Leads" }), 0, 100);
    stubRect(investigating, 200, 100);
    stubRect(screen.getByRole("region", { name: "Confirmed" }), 400, 100);

    const handle = screen.getByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    fireEvent.pointerDown(handle, { button: 0, clientX: 10, clientY: 10 });

    /* §5.4 drag preview: the grabbed card follows the pointer. */
    const preview = await screen.findByTestId("drag-preview");
    expect(preview).toHaveTextContent("Revenue dip in March");

    fireEvent.pointerMove(document, { clientX: 240, clientY: 20 });
    /* §5.4 valid drop target: the column under the pointer says so. */
    await waitFor(() => {
      expect(investigating.className).toContain("border-primary");
    });

    fireEvent.pointerUp(document, { clientX: 240, clientY: 20 });
    await waitFor(() => {
      expect(screen.queryByTestId("drag-preview")).not.toBeInTheDocument();
    });
  });

  it("undoes the last committed move and then hides the button", async () => {
    const bodies: { expected_version: number; columns: BoardColumns }[] = [];
    useSeededBoard();
    server.use(
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          const body = (await request.json()) as {
            expected_version: number;
            columns: BoardColumns;
            cards: BoardView["cards"];
          };
          bodies.push({
            expected_version: body.expected_version,
            columns: body.columns,
          });
          return HttpResponse.json({
            ...seededBoard(body.expected_version + 1),
            columns: body.columns,
            cards: body.cards,
          });
        },
      ),
    );

    const user = await openBoard();
    await screen.findByText("Revenue dip in March");
    /* No undo before anything has been committed. */
    expect(
      screen.queryByRole("button", { name: "Undo last move" }),
    ).not.toBeInTheDocument();

    screen
      .getByRole("button", { name: "Move card: Revenue dip in March" })
      .focus();
    await user.keyboard("{Enter}{ArrowRight}{Enter}");
    await screen.findByText(/Version 4/);
    expect(cardTitlesIn("Investigating")).toHaveLength(1);

    await user.click(
      await screen.findByRole("button", { name: "Undo last move" }),
    );
    await screen.findByText(/Version 5/);

    expect(cardTitlesIn("Investigating")).toHaveLength(0);
    expect(cardTitlesIn("Leads")).toHaveLength(2);
    // The reversal is a normal write against the version the move produced.
    expect(bodies).toHaveLength(2);
    expect(bodies[1]?.expected_version).toBe(4);
    expect(bodies[1]?.columns.find((c) => c.id === "leads")?.card_ids).toEqual([
      "c1",
      "c2",
    ]);
    expect(
      screen.queryByRole("button", { name: "Undo last move" }),
    ).not.toBeInTheDocument();
  });

  it("points the drag handle at the board's own instructions", async () => {
    useSeededBoard();
    await openBoard();
    const handle = await screen.findByRole("button", {
      name: "Move card: Revenue dip in March",
    });
    const describedBy = handle.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy!)).toHaveTextContent(
      /arrow keys move it, Enter confirms, Escape cancels/,
    );
  });

  it("adds a card from an existing question", async () => {
    let putBody: { cards: { title: string; ref_type: string }[] } | null = null;
    server.use(
      http.get("/api/v1/projects/:projectId/boards/:boardId", () =>
        HttpResponse.json({
          project_id: PROJECT,
          board_id: "investigation",
          version: 0,
          columns: [],
          cards: [],
        } satisfies BoardView),
      ),
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          const body = (await request.json()) as {
            expected_version: number;
            columns: BoardView["columns"];
            cards: BoardView["cards"];
          };
          putBody = body as unknown as typeof putBody;
          return HttpResponse.json({
            project_id: PROJECT,
            board_id: "investigation",
            version: body.expected_version + 1,
            columns: body.columns,
            cards: body.cards,
          } satisfies BoardView);
        },
      ),
    );

    const user = await openBoard();
    /* Default columns appear for a board that was never written (version 0). */
    expect(screen.getByRole("region", { name: "Leads" })).toBeInTheDocument();

    await user.selectOptions(
      await screen.findByLabelText("Add card from"),
      "question:q_trend",
    );
    await user.click(screen.getByRole("button", { name: "Add card" }));

    /* Scoped to the column: the picker option now carries the bare question
     * text too (its optgroup says "Questions raised in this session"), so an
     * unscoped query would match the <option> as well as the new card. */
    expect(
      await within(screen.getByRole("region", { name: "Leads" })).findByText(
        "How is value trending over time?",
      ),
    ).toBeInTheDocument();
    const body = putBody as unknown as {
      expected_version: number;
      cards: { title: string; ref_type: string }[];
    };
    expect(body.expected_version).toBe(0);
    expect(body.cards[0]?.ref_type).toBe("question");
  });

  it("edits a card title and note through the stored board", async () => {
    let putBody: {
      expected_version: number;
      columns: BoardView["columns"];
      cards: NonNullable<BoardView["cards"]>;
    } | null = null;
    useSeededBoard();
    server.use(
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          const body = (await request.json()) as NonNullable<typeof putBody>;
          putBody = body;
          return HttpResponse.json({
            ...seededBoard(body.expected_version + 1),
            columns: body.columns,
            cards: body.cards,
          });
        },
      ),
    );

    const user = await openBoard();
    await user.click(
      await screen.findByRole("button", {
        name: "Edit card: Revenue dip in March",
      }),
    );
    const form = screen.getByRole("form", {
      name: "Edit Revenue dip in March",
    });
    await user.clear(within(form).getByRole("textbox", { name: "Title" }));
    await user.type(
      within(form).getByRole("textbox", { name: "Title" }),
      "Revenue dip needs validation",
    );
    await user.type(
      within(form).getByRole("textbox", { name: "Note" }),
      "Check the March campaign calendar.",
    );
    await user.click(within(form).getByRole("button", { name: "Save card" }));

    expect(
      await screen.findByText("Revenue dip needs validation"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Check the March campaign calendar."),
    ).toBeInTheDocument();
    const saved = putBody as unknown as {
      expected_version: number;
      cards: NonNullable<BoardView["cards"]>;
    };
    expect(saved.expected_version).toBe(3);
    expect(saved.cards.find((card) => card.id === "c1")).toMatchObject({
      title: "Revenue dip needs validation",
      note: "Check the March campaign calendar.",
      ref_id: "find_1",
    });
  });

  it("requires confirmation before removing a card", async () => {
    let putBody: {
      columns: NonNullable<BoardView["columns"]>;
      cards: NonNullable<BoardView["cards"]>;
    } | null = null;
    useSeededBoard();
    server.use(
      http.put(
        "/api/v1/projects/:projectId/boards/:boardId",
        async ({ request }) => {
          const body = (await request.json()) as {
            expected_version: number;
            columns: NonNullable<BoardView["columns"]>;
            cards: NonNullable<BoardView["cards"]>;
          };
          putBody = body;
          return HttpResponse.json({
            ...seededBoard(body.expected_version + 1),
            columns: body.columns,
            cards: body.cards,
          });
        },
      ),
    );

    const user = await openBoard();
    await user.click(
      await screen.findByRole("button", {
        name: "Remove card: Revenue dip in March",
      }),
    );
    const confirmation = screen.getByRole("alert");
    expect(confirmation).toHaveTextContent(
      "Remove this card from the project board?",
    );
    expect(screen.getByText("Revenue dip in March")).toBeInTheDocument();

    await user.click(
      within(confirmation).getByRole("button", {
        name: "Confirm remove Revenue dip in March",
      }),
    );

    await waitFor(() =>
      expect(screen.queryByText("Revenue dip in March")).not.toBeInTheDocument(),
    );
    const saved = putBody as unknown as {
      columns: NonNullable<BoardView["columns"]>;
      cards: NonNullable<BoardView["cards"]>;
    };
    expect(saved.cards.some((card) => card.id === "c1")).toBe(false);
    expect(
      saved.columns.some((column) => (column.card_ids ?? []).includes("c1")),
    ).toBe(false);
  });
});
