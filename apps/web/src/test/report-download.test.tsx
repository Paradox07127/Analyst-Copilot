import { describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import {
  filenameFromContentDisposition,
  saveBlob,
} from "../api/client";
import { server } from "./msw/server";
import { renderAppAt } from "./render";
import { objectUrls } from "./setup";

const REPORT_PATH = "/projects/p1/sessions/r1/report";

/* jsdom's Blob implements neither text() nor arrayBuffer(); FileReader is the
 * only reader it ships. */
function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

describe("Report downloads", () => {
  it("offers HTML, PDF and decision-report downloads", async () => {
    renderAppAt(REPORT_PATH);
    const exports = await screen.findByRole("group", {
      name: "Report exports",
    });
    expect(within(exports).getByText("Technical")).toBeInTheDocument();
    expect(within(exports).getByText("Decision story")).toBeInTheDocument();
    const html = await screen.findByRole("button", { name: "Download HTML" });
    await waitFor(() => expect(html).toBeEnabled());
    expect(screen.getByRole("button", { name: "Download PDF" })).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Download Decision report (MD)" }),
    ).toBeInTheDocument();
  });

  it("clicking HTML fetches the export and hands it to the browser", async () => {
    const user = userEvent.setup();
    renderAppAt(REPORT_PATH);
    const html = await screen.findByRole("button", { name: "Download HTML" });
    await waitFor(() => expect(html).toBeEnabled());
    await user.click(html);

    await waitFor(() => expect(objectUrls.created).toHaveLength(1));
    expect(await readBlob(objectUrls.created[0]!)).toContain("<!doctype html>");
    /* The object URL must be released, or every download leaks a blob. */
    expect(objectUrls.revoked).toHaveLength(1);
  });

  it("disables PDF and explains why when the host cannot render it", async () => {
    server.use(
      http.get("/api/v1/system/capabilities", () =>
        HttpResponse.json({
          pdf_export_available: false,
          pdf_export_hint:
            "PDF export needs WeasyPrint and its pango libraries. Install them with `uv sync --extra pdf`.",
        }),
      ),
    );
    renderAppAt(REPORT_PATH);
    expect(await screen.findByText(/uv sync --extra pdf/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download PDF" })).toBeDisabled();
    // HTML is unaffected by a missing PDF renderer.
    expect(screen.getByRole("button", { name: "Download HTML" })).toBeEnabled();
  });

  it("disables the decision-report download when the report is stale", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "available",
          artifact_id: "dreport_1",
          title: "Churn drivers",
          sections: [],
          limitations: [],
          investigation_gaps: [],
          candidate_decisions: [],
          evidence_refs: [],
          source_finding_artifact_ids: [],
          granted_evidence_artifact_ids: [],
          freshness: { status: "stale", reasons: ["Source data changed."] },
          export_available: false,
        }),
      ),
    );
    renderAppAt(REPORT_PATH);
    expect(
      await screen.findByText(/out of date with its source findings/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Download Decision report (MD)" }),
    ).toBeDisabled();
  });

  it("surfaces a typed server refusal instead of failing silently", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report/download", () =>
        HttpResponse.json(
          {
            error: {
              code: "report_export_unavailable",
              message: "PDF export needs WeasyPrint and its pango libraries.",
            },
          },
          { status: 503 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(REPORT_PATH);
    await user.click(await screen.findByRole("button", { name: "Download HTML" }));
    expect(
      await screen.findByText("PDF export needs WeasyPrint and its pango libraries."),
    ).toBeInTheDocument();
    expect(objectUrls.created).toHaveLength(0);
  });

  it("disables every report download when the run has no report", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "none",
          markdown: "",
          generated_at: null,
        }),
      ),
    );
    renderAppAt(REPORT_PATH);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Download HTML" })).toBeDisabled(),
    );
    expect(screen.getByRole("button", { name: "Download PDF" })).toBeDisabled();
  });
});

describe("download filename handling", () => {
  it("takes the server-chosen name from Content-Disposition", () => {
    expect(
      filenameFromContentDisposition('attachment; filename="run_1.html"'),
    ).toBe("run_1.html");
    expect(filenameFromContentDisposition("attachment; filename=run_1.pdf")).toBe(
      "run_1.pdf",
    );
  });

  it("never lets a header steer the local path", () => {
    expect(
      filenameFromContentDisposition(
        'attachment; filename="../../../etc/passwd"',
      ),
    ).toBe("passwd");
    expect(
      filenameFromContentDisposition('attachment; filename="..\\\\evil.html"'),
    ).toBe("evil.html");
    expect(filenameFromContentDisposition(null)).toBeNull();
    expect(filenameFromContentDisposition("attachment")).toBeNull();
  });

  it("saveBlob revokes the object URL even if the click throws", () => {
    const blob = new Blob(["x"], { type: "text/plain" });
    saveBlob(blob, "a.txt");
    expect(objectUrls.created).toHaveLength(1);
    expect(objectUrls.revoked).toHaveLength(1);
  });
});
