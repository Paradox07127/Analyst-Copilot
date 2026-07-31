import { describe, expect, it } from "vitest";
import {
  JOB_KINDS,
  jobInvalidationKeys,
  type JobKind,
} from "../api/job-invalidation";

const context = {
  projectId: "project",
  sessionId: "derived",
  sourceSessionId: "source",
  resultSessionId: "result",
};

const base = [
  ["session", "derived"],
  ["sessions", "project"],
  ["projects"],
  ["workspace-usage"],
  ["session-metrics", "derived"],
  ["trace", "derived"],
];

const expected: Record<JobKind, unknown[][]> = {
  auto_eda: [
    ...base,
    ["datasets", "derived"],
    ["quality", "derived"],
    ["profiles", "derived"],
    ["charts", "derived"],
    ["report", "derived"],
    ["artifacts", "derived"],
    ["questions", "derived"],
    ["findings", "derived"],
    ["semantic", "derived"],
    ["analysis", "derived"],
    ["skills", "derived"],
    ["cleaning-log", "derived"],
    ["cleaning-raw", "derived"],
    ["session-debug", "derived"],
    ["decision-coverage", "derived"],
  ],
  question_exec: [
    ...base,
    ["questions", "source"],
    ["artifacts", "derived"],
    ["findings", "derived"],
  ],
  skill_replay: [
    ...base,
    ["skills", "source"],
    ["artifacts", "derived"],
    ["findings", "derived"],
  ],
  relationship_validate: [
    ...base,
    ["relationships", "source"],
    ["artifacts", "source"],
  ],
  relationship_discover: [
    ...base,
    ["relationships", "source"],
    ["artifacts", "source"],
  ],
  report_generate: [
    ...base,
    ["report", "source"],
    ["artifacts", "source"],
    ["session-metrics", "source"],
  ],
  session_fork: base,
  question_draft: [
    ...base,
    ["questions", "source"],
    ["artifacts", "source"],
  ],
  investigation_plan: [...base, ["investigations", "source"]],
  investigation_execute: [
    ...base,
    ["investigations", "source"],
    ["artifacts", "result"],
    ["findings", "result"],
  ],
  macro_loop: [
    ...base,
    ["investigations", "source"],
    ["artifacts", "result"],
    ["findings", "result"],
  ],
  synthesis_brief_create: [...base, ["decision-story", "source"]],
  decision_report_generate: [
    ...base,
    ["decision-report", "source"],
    ["artifacts", "result"],
  ],
  cleaning_preview: base,
  cleaning_apply: [
    ...base,
    ["cleaning-log", "source"],
    ["cleaning-raw", "source"],
  ],
  dataset_distributions: base,
  custom_chart: base,
};

describe("job terminal cache invalidation", () => {
  it.each(JOB_KINDS)("%s invalidates only its declared result keys", (kind) => {
    expect(jobInvalidationKeys(kind, context)).toEqual(expected[kind]);
  });

  it("never uses unrelated global questions/relationship/investigation roots", () => {
    for (const kind of JOB_KINDS) {
      const serialized = jobInvalidationKeys(kind, context).map((key) =>
        JSON.stringify(key),
      );
      expect(serialized).not.toContain(JSON.stringify(["questions"]));
      expect(serialized).not.toContain(JSON.stringify(["relationships"]));
      expect(serialized).not.toContain(JSON.stringify(["investigations"]));
    }
  });
});
