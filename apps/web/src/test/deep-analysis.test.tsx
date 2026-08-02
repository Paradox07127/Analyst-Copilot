import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

const PATH = "/projects/p1/sessions/r1/deep-analysis";

describe("Deep analysis page", () => {
  it("renders the three sections from persisted analysis artifacts", async () => {
    renderAppAt(PATH);

    expect(
      await screen.findByRole("heading", { name: "Deep analysis" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Review only")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Run a question" })).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/questions",
    );
    expect(screen.getByText("Analysis tables")).toBeInTheDocument();
    expect(screen.getByText("Statistical tests")).toBeInTheDocument();
    expect(screen.getByText("ML baseline model cards")).toBeInTheDocument();

    /* Stat test row keeps stored precision and carries the derived verdict. */
    expect(
      screen.getByRole("link", { name: "independent_t_test" }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/artifacts?artifact=stat_1",
    );
    expect(screen.getByText("<0.001")).toBeInTheDocument();
    expect(
      screen.getByText("Significant at alpha=0.05 (small effect)"),
    ).toBeInTheDocument();
    expect(screen.getByText("small")).toBeInTheDocument();

    /* Model card headline metric, leakage verdict and limitations. */
    expect(screen.getByText("churn · classification")).toBeInTheDocument();
    expect(screen.getByText("leakage mitigated")).toBeInTheDocument();
    expect(screen.getByText("0.83")).toBeInTheDocument();
    expect(
      screen.getByText("Limitations: Single split; no cross-validation."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Open evidence" }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/artifacts?artifact=card_1",
    );
  });

  /* These two used to drive native <details>. jsdom leaves the children of a
   * closed <details> in the accessibility tree, so "collapsed" was only ever
   * asserted through the `open` attribute — a control inside could be clicked
   * while invisible and the test would still pass. The Disclosure primitive
   * marks the collapsed panel aria-hidden, so a role query is now the guard:
   * queryByRole skips inaccessible subtrees, queryByText would not. */
  it("keeps analysis table rows out of the a11y tree until expanded", async () => {
    const user = userEvent.setup();
    renderAppAt(PATH);
    const toggle = await screen.findByRole("button", {
      name: /Numeric summary/,
    });

    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("table", { name: "Numeric summary" }),
    ).not.toBeInTheDocument();

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("table", { name: "Numeric summary" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Question: Baseline EDA (not tied to a selected question card)",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Per-column descriptive statistics."),
    ).toBeInTheDocument();
  });

  it("keeps trivial correlation pairs out of the table until asked for", async () => {
    const user = userEvent.setup();
    renderAppAt(PATH);
    await user.click(await screen.findByRole("button", { name: /Correlations/ }));

    /* One disclosure level, not two: the trivial pair is appended to the same
     * table rather than buried in a nested <details> the reader has to find. */
    const table = screen.getByRole("table", { name: "Correlations" });
    expect(within(table).getByText("total")).toBeInTheDocument();
    expect(within(table).queryByText("value_cents")).not.toBeInTheDocument();

    await user.click(
      screen.getByLabelText("Show 1 trivial/degenerate pair(s)"),
    );
    expect(within(table).getByText("value_cents")).toBeInTheDocument();
    expect(within(table).getByText("(trivial)")).toBeInTheDocument();
  });

  it("shows an empty state when the run produced no analysis artifacts", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/analysis", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          tables: [],
          stat_tests: [],
          model_cards: [],
        }),
      ),
    );
    renderAppAt(PATH);
    expect(
      await screen.findByText("No deterministic analysis artifacts"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: "Review suggested questions",
      }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/questions");
  });

  it("surfaces API errors with a retry", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/analysis", () =>
        HttpResponse.json(
          { error: { code: "session_not_found", message: "Session not found: r1" } },
          { status: 404 },
        ),
      ),
    );
    renderAppAt(PATH);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Request failed (session_not_found)",
    );
  });

  it("renders a forbidden state for 403 instead of an empty state", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/analysis", () =>
        HttpResponse.json(
          {
            error: {
              code: "forbidden",
              message: "You cannot view analysis for this session",
            },
          },
          { status: 403 },
        ),
      ),
    );
    renderAppAt(PATH);

    const forbidden = await screen.findByRole("alert");
    expect(forbidden).toHaveTextContent("Access forbidden");
    expect(forbidden).toHaveTextContent(
      "You cannot view analysis for this session",
    );
    expect(
      screen.queryByText("No deterministic analysis artifacts"),
    ).not.toBeInTheDocument();
  });
});
