import { describe, expect, it } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  filterPreviewRows,
  sortPreviewRows,
} from "../features/datasets/TablePreviewPage";
import { renderAppAt, renderAppWithRouterAt } from "./render";

describe("sortPreviewRows", () => {
  it("sorts numeric columns by value, not lexicographic order", () => {
    const rows = [[10], [2], [1]];
    expect(sortPreviewRows(rows, 0, "asc")).toEqual([[1], [2], [10]]);
    expect(sortPreviewRows(rows, 0, "desc")).toEqual([[10], [2], [1]]);
  });

  it("keeps null and undefined values last regardless of direction", () => {
    const rows = [[3], [null], [1], [undefined], [2]];
    expect(sortPreviewRows(rows, 0, "asc")).toEqual([
      [1],
      [2],
      [3],
      [null],
      [undefined],
    ]);
    expect(sortPreviewRows(rows, 0, "desc")).toEqual([
      [3],
      [2],
      [1],
      [null],
      [undefined],
    ]);
  });

  it("falls back to locale string comparison for non-numeric columns", () => {
    const rows = [["banana"], ["Apple"], ["cherry"]];
    expect(sortPreviewRows(rows, 0, "asc")).toEqual([
      ["Apple"],
      ["banana"],
      ["cherry"],
    ]);
  });
});

describe("filterPreviewRows", () => {
  it("matches case-insensitively across all columns", () => {
    const rows = [
      [1, "Alice"],
      [2, "Bob"],
      [3, "alice2"],
    ];
    expect(filterPreviewRows(rows, "ALICE")).toEqual([
      [1, "Alice"],
      [3, "alice2"],
    ]);
  });

  it("returns every row for an empty or whitespace-only query", () => {
    const rows = [
      [1, "a"],
      [2, "b"],
    ];
    expect(filterPreviewRows(rows, "   ")).toEqual(rows);
  });
});

describe("Table Preview toolbar (integration)", () => {
  it("keeps search collapsed by default and filters visible rows when opened", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/table/sample");
    await screen.findByText("row-0");
    expect(screen.queryByLabelText("Find in loaded rows")).not.toBeInTheDocument();
    const searchButton = screen.getByRole("button", { name: "Search loaded rows" });
    expect(searchButton).toHaveAttribute("aria-expanded", "false");
    await user.click(searchButton);
    expect(searchButton).toHaveAttribute("aria-expanded", "true");

    // fireEvent, not user.type: react-resizable-panels' document-level
    // pointerdown handler preventDefaults clicks at jsdom's all-zero rects,
    // so user-event never focuses the input (same workaround as chat.test).
    fireEvent.change(screen.getByLabelText("Find in loaded rows"), {
      target: { value: "row-1" },
    });
    // "row-1" matches row-1 and row-10..row-19: 11 of the 100 loaded rows.
    expect(screen.queryByText("row-0")).not.toBeInTheDocument();

    await user.click(searchButton);
    expect(searchButton).toHaveAttribute("aria-expanded", "false");
    expect(await screen.findByText("row-0")).toBeInTheDocument();
  });

  it("shows a clear empty state when the search matches nothing", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/table/sample");
    await screen.findByText("row-0");

    await user.click(screen.getByRole("button", { name: "Search loaded rows" }));

    fireEvent.change(screen.getByLabelText("Find in loaded rows"), {
      target: { value: "no-such-row" },
    });
    expect(
      await screen.findByText("No rows match your search"),
    ).toBeInTheDocument();
    expect(screen.queryByRole("columnheader")).not.toBeInTheDocument();
  });

  it("cycles a column header through ascending, descending, and unsorted", async () => {
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/table/sample",
    );
    await screen.findByText("row-0");

    const idHeaderButton = screen.getByRole("button", { name: "id" });
    const idHeaderCell = idHeaderButton.closest("th") as HTMLElement;
    expect(idHeaderCell).toHaveAttribute("aria-sort", "none");

    await user.click(idHeaderButton);
    expect(idHeaderCell).toHaveAttribute("aria-sort", "ascending");
    expect(router.state.location.search).toBe("?sort=id&dir=asc");

    await user.click(idHeaderButton);
    expect(idHeaderCell).toHaveAttribute("aria-sort", "descending");
    expect(router.state.location.search).toBe("?sort=id&dir=desc");
    expect(await screen.findByText("row-99")).toBeInTheDocument();

    await user.click(idHeaderButton);
    expect(idHeaderCell).toHaveAttribute("aria-sort", "none");
    expect(router.state.location.search).toBe("");
  });

  it("keeps table selection and icon search in one borderless toolbar", async () => {
    renderAppAt("/projects/p1/sessions/r1/table/sample");
    await screen.findByText("row-0");

    expect(screen.getByLabelText("Table")).toHaveValue("sample");
    expect(screen.getByRole("button", { name: "Search loaded rows" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export loaded rows" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "All tables" })).not.toBeInTheDocument();
    expect(screen.queryByText("Inspecting sample.csv")).not.toBeInTheDocument();
  });

  it("restores a page-scoped search and sort from the URL", async () => {
    renderAppAt(
      "/projects/p1/sessions/r1/table/sample?q=row-1&sort=id&dir=desc",
    );

    expect(await screen.findByDisplayValue("row-1")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "id" }).closest("th"),
    ).toHaveAttribute("aria-sort", "descending");
  });
});

describe("Table header distributions (integration)", () => {
  it("renders distributions inside their corresponding column header without a repeated card grid", async () => {
    renderAppAt("/projects/p1/sessions/r1/table/sample");
    const valueHeader = await screen.findByRole("columnheader", {
      name: "value float64",
    });

    expect(
      await within(valueHeader).findAllByTestId("header-distribution-bin"),
    ).toHaveLength(10);
    expect(screen.queryByTestId("distribution-card")).not.toBeInTheDocument();
    expect(screen.queryByText("Hide column distributions")).not.toBeInTheDocument();
  });

  it("highlights a hovered histogram bin and exposes its range, count, and share in the header", async () => {
    renderAppAt("/projects/p1/sessions/r1/table/sample");
    const valueHeader = await screen.findByRole("columnheader", {
      name: "value float64",
    });
    const bins = await within(valueHeader).findAllByTestId(
      "header-distribution-bin",
    );

    fireEvent.mouseEnter(bins[0]!);

    expect(
      within(valueHeader).getByText("0 – 1 · 4 records (1.6%)"),
    ).toBeInTheDocument();
    expect(bins[1]).toHaveClass("opacity-45");
  });

  it("renders and highlights categorical bars in their matching header", async () => {
    renderAppAt("/projects/p1/sessions/r1/table/sample");
    const nameHeader = await screen.findByRole("columnheader", {
      name: "name string",
    });
    const bars = await within(nameHeader).findAllByTestId(
      "header-distribution-category",
    );

    expect(bars).toHaveLength(3);
    fireEvent.mouseEnter(bars[0]!);
    expect(
      within(nameHeader).getByText("row-1 · 30 records (12.0%)"),
    ).toBeInTheDocument();
    expect(bars[1]).toHaveClass("opacity-45");
  });
});
