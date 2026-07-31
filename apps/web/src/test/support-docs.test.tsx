import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

const LAUNCHPAD = "/projects/p1/new-session";

function textFile(name: string, body = "orders: one row per order\n"): File {
  return new File([body], name, { type: "text/markdown" });
}

describe("Launchpad support documents", () => {
  it("uploads a document and lists it", async () => {
    const user = userEvent.setup();
    renderAppAt(LAUNCHPAD);

    const input = await screen.findByLabelText("Support documents (optional)");
    expect(input).toHaveAttribute("accept", ".md,.txt,.csv,.pdf");
    await user.upload(input, textFile("data_dictionary.md"));

    const section = screen.getByRole("region", { name: "Support documents" });
    expect(
      await within(section).findByRole("button", { name: "1 support doc" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("data_dictionary.md")).toBeInTheDocument();
  });

  it("accepts documents dropped anywhere in the section", async () => {
    renderAppAt(LAUNCHPAD);

    const section = await screen.findByRole("region", {
      name: "Support documents",
    });
    fireEvent.dragEnter(section, {
      dataTransfer: { files: [textFile("terms.txt")] },
    });
    expect(section).toHaveClass("bg-primary/5");

    fireEvent.drop(section, {
      dataTransfer: { files: [textFile("terms.txt")] },
    });
    expect(
      await within(section).findByRole("button", { name: "1 support doc" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("terms.txt")).toBeInTheDocument();
    expect(section).not.toHaveClass("bg-primary/5");
  });

  it("keeps project deletion out of the launch form and in the Inspector", async () => {
    const user = userEvent.setup();
    renderAppAt(LAUNCHPAD);

    const section = await screen.findByRole("region", {
      name: "Support documents",
    });
    await user.upload(
      within(section).getByLabelText("Support documents (optional)"),
      textFile("readme.txt"),
    );
    expect(
      await within(section).findByRole("button", { name: "1 support doc" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("readme.txt")).toBeInTheDocument();
    expect(
      within(section).queryByRole("button", { name: /Remove|Delete/ }),
    ).not.toBeInTheDocument();

    await user.click(
      await screen.findByRole("button", {
        name: "Delete support document readme.txt",
      }),
    );
    await waitFor(() =>
      expect(screen.queryByText("readme.txt")).not.toBeInTheDocument(),
    );
  });

  it("surfaces a typed refusal without breaking the launch form", async () => {
    server.use(
      http.post("/api/v1/projects/:projectId/support-docs", () =>
        HttpResponse.json(
          {
            error: {
              code: "support_doc_too_large",
              message: "Support document exceeds the 10000000 byte limit.",
            },
          },
          { status: 413 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(LAUNCHPAD);
    await user.upload(
      await screen.findByLabelText("Support documents (optional)"),
      textFile("huge.md"),
    );

    expect(
      await screen.findByText("Support document exceeds the 10000000 byte limit."),
    ).toBeInTheDocument();
    // The CSV upload + launch controls are untouched by a support-doc failure.
    expect(screen.getByLabelText("Data files (.csv)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeInTheDocument();
  });

  it("says what these documents are and are not used for", async () => {
    /* Support docs may shape how a column is read but must never be the source
     * of a reported number. Saying so is the point of the control. */
    const user = userEvent.setup();
    renderAppAt(LAUNCHPAD);

    await user.click(
      await screen.findByRole("button", { name: "What is Support documents?" }),
    );

    expect(
      screen.getByText(/Nothing in them can confirm a join or enter a report figure/),
    ).toBeInTheDocument();
  });
});

describe("Settings About: sandbox status", () => {
  async function openAbout() {
    const user = userEvent.setup();
    renderAppAt("/settings");
    await user.click(await screen.findByRole("tab", { name: "About" }));
  }

  it("shows the active sandbox backend", async () => {
    await openAbout();
    expect(await screen.findByText("Code sandbox")).toBeInTheDocument();
    expect(await screen.findByText("docker")).toBeInTheDocument();
    expect(
      screen.getByText(/open-ended Python analysis is available/),
    ).toBeInTheDocument();
  });

  it("warns, without blocking, when no safe sandbox resolved", async () => {
    server.use(
      http.get("/api/v1/system/sandbox", () =>
        HttpResponse.json({
          backend: "none",
          available: false,
          safe_for_untrusted_code: false,
          open_python_analysis_available: false,
          detail: "No safe sandbox backend is available.",
          message:
            "Open-ended Python analysis is unavailable: no safe sandbox backend resolved.",
        }),
      ),
    );
    await openAbout();
    expect(
      await screen.findByText(/Open-ended Python analysis is unavailable/),
    ).toBeInTheDocument();
    // The rest of the About panel still renders.
    expect(screen.getByText("0.2.0")).toBeInTheDocument();
  });

  it("toggles the developer inspector", async () => {
    const user = userEvent.setup();
    renderAppAt("/settings");
    await user.click(await screen.findByRole("tab", { name: "About" }));

    const toggle = await screen.findByRole("checkbox", {
      name: /Developer inspector/,
    });
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    await waitFor(() => expect(toggle).toBeChecked());
  });
});

describe("Settings Model & API: cost rates", () => {
  it("saves the per-1k token prices", async () => {
    const user = userEvent.setup();
    renderAppAt("/settings");
    /* Pricing only applies to a live provider; offline disables the inputs. */
    await user.selectOptions(
      await screen.findByLabelText("Provider"),
      "deepseek",
    );

    const prompt = await screen.findByLabelText("Cost / 1k prompt tokens (USD)");
    await waitFor(() => expect(prompt).toBeEnabled());
    const completion = screen.getByLabelText(
      "Cost / 1k completion tokens (USD)",
    );
    await user.clear(prompt);
    await user.type(prompt, "0.0014");
    await user.clear(completion);
    await user.type(completion, "0.0028");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(prompt).toHaveValue(0.0014));
    expect(completion).toHaveValue(0.0028);
  });
});
