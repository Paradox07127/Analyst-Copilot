# Changelog

## Unreleased: what two real runs exposed — resource limits, the tool loop auto-EDA never entered, and five bugs - 2026-08-26

Two live runs on the Olist nine-table set (gpt-5.6-luna) drove this pass. Plan
and per-slice acceptance: `docs/agent/remediation-plan-2026-08-25.md` (T13–T15).

- Resource ceilings are now user settings that persist: the working-set ceiling
  (GB) and the per-dataset row ceiling (millions) are editable, allow-listed
  server-side, and applied as the default policy for new runs. A stopped run no
  longer reports only "Resource preflight stopped this run before data
  ingestion" — `EdaResourceLimitGuidance` carries the estimate, the ceiling, the
  overage, the largest table, and the ways out, including a *computed* "turning
  the clean off would fit" (re-estimated, not guessed, and offered only when the
  working set is the sole limit). The run that started this: precleaning doubles
  the retained term, taking the estimate from 1820 MB to 2284 MB against a
  2048 MB ceiling, and the whole analysis stopped in 0.19 s while reporting
  "completed".
- Auto-EDA now enters the agent tool loop. It previously called only the SQL
  executor, so all twenty typed tools — including the three method families
  added earlier this month — were unreachable from the main pipeline: a
  forecasting question failed on generated SQL and two method questions
  abstained. Questions carrying a method answer contract now route through the
  shared `execute_question_with_tools`; plain aggregates stay on SQL. The step's
  output contract widened to the ten artifact types the loop can emit (the
  kernel rejects undeclared types, so a call swap alone would have failed at
  runtime), and the loop's context now includes this run's earlier EDA
  artifacts, filtered so context artifacts are not republished as step output.
- A method-contract failure no longer discards a successful query and tells the
  user nothing was produced. Anomaly and segmentation questions publish their
  findings with `contract_status="unverified"` and a plain-language disclosure;
  forecast, prediction and causal questions still refuse a SQL proxy. The run
  that motivated this had 2591 rows of freight outliers and 73 category
  diagnostics sitting in artifacts while the report said the evidence was never
  produced.
- `causal_overclaim` is repaired deterministically instead of deleted: causal
  verbs are rewritten to association wording with a qualifier, then re-gated.
  Deleting them had emptied both Key EDA Insights and Business Recommendations,
  and the LLM repair round was measured going in circles (two byte-identical
  validation rounds).
- Claims cite large values exactly. `{value:.4g}` rendered 25455 as
  "2.546e+04"; the gate read 25460, disagreed with the stored evidence, and
  pruned three *correct* claims (25460/19780/12410 in the live run match the
  rendering exactly). The same leak ran wider than the ranked builder: `{value:g}`
  gave up at six significant digits, so single-value, generic and trend findings,
  the ranked group-size suffix, and — the third pruned claim — the
  `allowed_numbers` list the interpretation model copies its figures from all
  offered "1.35916e+07" for a GMV of 13591643.7. The renderer that answers the
  gate's half-ULP rule now lives beside it as `report_validator.gate_safe_number`,
  and every one of those sites calls it.
- Quality-issue locators with `code=`/`column=` predicates resolve again, so a
  data-quality claim whose figures match the artifact verbatim is no longer
  labeled unverified; the qualifier labels themselves now read as English.
  Share claims must restate the evidence's own denominator (the report had
  called freight "14.2% of GMV" when the query's denominator was price+freight).
- Report redundancy: a question contributes one merged claim instead of a
  finding plus a near-duplicate interpretation, and de-duplication drops a
  repeated claim outright rather than leaving a "see other section" pointer.

Gates: backend `tests/unit` 3721 passed, 0 failed, 0 errors; `tests/golden` +
`tests/evals` 198 passed, 1 skipped; frontend vitest 653/653; `tsc --noEmit`
clean; production build clean; `scripts/benchmark_offline.py` exit 0; ruff clean.

## Unreleased: 2026-08 remediation — exploration reaches the user, investigation branch retired, three method families land - 2026-08-26

A four-agent design-completeness review (interaction surface / design-vs-code
gaps / unfinished markers / end-to-end journey) drove a ten-slice remediation
run. Plan and per-slice acceptance: `docs/agent/remediation-plan-2026-08-25.md`.

- **The exploration admission gate is gone.** A deep dive no longer needs an
  E4a release certificate, a pinned issuer public key, or any environment
  variable: a bare install runs prepare → authorize → execute → publish. The
  certificate schema, its verifier, and the evidence issuer stay as
  *evaluation* tooling (`drivers/exploration_evidence_issuer.py` still answers
  "can the deep dive find planted structure?"); they no longer decide who may
  use the feature. What replaces the certificate as the run's identity is
  derived from the live build — the read-only tool digest, the two policy
  versions, and a per-tier budget ceiling (`exploration_hard_caps`) — so
  policy freeze and budget hard caps still hold on resume. The pre-run budget
  confirmation (prepare → authorize, `exploration_start` approval chain) is
  untouched: that is a spend confirmation, not an admission gate.
  `scripts/issue_local_exploration_certificate.py` is deleted, and
  `SystemCapabilitiesView` no longer reports exploration as optional.
- Exploration output is no longer a dead end: a gracefully stopped run
  publishes its gate-passed findings, cited receipts, and rendered report into
  the ArtifactStore (`publish_exploration_outputs`), the main report gains a
  hard-gated "Deep-Dive Exploration" section whose numbers resolve against
  receipt artifacts, and each finding is also published as a `ValidatedFinding`
  (`origin="exploration"`) so Findings, Decision Report, promotion, and
  publication chains finally have a live producer.
- The investigation branch (plan → approve → execute → macro-loop) is retired
  end to end per user decision: 24 files / ~13k lines deleted (routers, three
  worker job kinds, orchestrator, loop agents, service, loop schemas), eight
  cross-cutting contract tests decoupled, macro-loop metrics removed, and
  `RETIRED_ARTIFACT_TYPES` keeps stores with legacy payloads readable.
  `ValidatedFinding`/`InvestigationRecord`/`InvestigationPlan` schemas stay —
  live consumers read them.
- Per-question interpretations now reach the main report as `qintp_` claims
  citing the question's own evidence union; `_diagnose_missingness` registers
  its Holm-corrected target associations in the stat registry, so those tests
  count toward multiplicity families instead of silently escaping them.
- Exploration UX: `GET /sessions/{id}/explorations`, goal carried server-side,
  LLM spend rolled up into the source session's trace (own event type — the
  budget-restore path must not see foreign `llm_usage`), markdown report
  rendering, nav entry and split-pane routes.
- Failure paths: failed/cancelled runs are named as such in the nav (no more
  eternal "still preparing"), `POST /jobs/{id}/retry` re-runs a failed
  auto_eda with its persisted params, worker exceptions translate to
  user-readable messages (raw detail stays in trace), and LLM degradation
  (question fallback, deterministic report) marks the job summary degraded
  with the reason surfaced on the report page.
- Task management: the frontend cancel whitelist is gone (the backend never
  had one), `GET /sessions/{id}/jobs` lets Activity recover in-flight jobs
  without localStorage, and a chat turn can be stopped between tool steps.
- Follow-up loop: "Ask about this" entries on report sections and findings,
  chat's tool context includes derived-session result artifacts, a stale-report
  banner offers one-click regeneration after new question runs.
- Method families are live (design: `docs/agent/method-families-design-2026-08-25.md`):
  `run_forecast` (four fpp3 baselines under rolling-origin backtests, empirical
  error bands, fixed "descriptive baseline projection" limitation),
  `run_segmentation` (KMeans + silhouette k, ARI resampling stability; unstable
  runs are marked evidence-invalid; new `SEGMENTATION_RESULT` artifact), and
  `run_causal_experiment` (observational contrast + SMD balance by default;
  the randomized tier computes ATE±CI only under a user-issued
  `randomized_design_confirmation` credential, and the claim gate licenses
  causal wording solely from that receipt provenance — never from text).
- Interaction pack: upload size pre-check and human-readable 413, CSV export
  for table preview and findings, custom charts saveable as artifacts, delete
  confirmations (incl. unfiled uploads), activity-button recovery, project
  list expansion, search empty state, derived-runs filter, plain-language LLM
  mode control, and a minimal UI for user SQL skill templates.
- The memory ceiling is a setting, and hitting it says why. A nine-table run
  with pre-cleaning on used to stop in 0.19s behind one sentence ("Resource
  preflight stopped this run before data ingestion.") while the job read
  *completed*. Settings → Analysis behavior now carries two limits in the units
  a person uses — memory for one analysis (GB, default 2) and the largest table
  (million rows, default 10) — remembered in this browser so a raised ceiling
  survives a server restart; they resolve into the run's `EdaResourcePolicy`
  and nothing else about the policy is settable from a browser. A stopped run
  now publishes structured numbers (`EdaResourceLimitGuidance`: estimate vs
  ceiling, how far over, the largest table, and a suggested ceiling) on both
  `GET /sessions/{id}` and the `job.completed` event, and the Data Map and the
  job strip render them as the estimate, the biggest table, and the ways out.
  "Turn the clean off" is offered only when the working set is re-estimated
  without pre-cleaning and actually fits.
- Cleanup: `WorkflowEvalEnvironment` now carries real build identity (git
  revision, tool-registry and sandbox-policy digests, prompt digest) so
  `trial_environment_mismatch` can actually fire; the narrative-reviewer stub
  and its dead switch are deleted (it fabricated an audit note if ever
  enabled); orphan fields dropped or wired (`handoff.completed_at` is now
  written); `nl2sql_eval` moved out of `src/`.

Still closed to users: E4a release certificate remains unissued — exploration
stays hidden until the three-tier live-provider admission trials run on
current code. The stale "Still residual" list in the 2026-08-02 entry below is
superseded by this pass; `docs/exploration-remaining-issues.md` carries the
authoritative per-item status.

Gates: backend `tests/unit` 2871 passed, 4 failed pre-existing — 2 in
`test_upload_hardening` (the venv lacks `httpx2`, breaking starlette's
TestClient, same root as the 345 baseline collection errors) and 2 in
`test_report_llm_settings` (`load_*_from_env_file(path=None)` reads the real
repo-root `.env`, whose values leak into assertions — independent of httpx2);
`tests/golden` + `tests/evals` 198 passed, 1 skipped; frontend vitest 647/647;
`tsc --noEmit` clean; ruff clean.

Follow-up (same day): both environment debts are paid. `uv sync --extra dev`
installs the already-declared `httpx2`, clearing the 345 collection errors and
`test_upload_hardening`; `test_report_llm_settings` isolates the repo-root
`.env` behind an autouse `DEFAULT_ENV_PATH` monkeypatch. Un-masking the API
tests exposed three files of drift from this same iteration, all
test-expectation updates (report view's `degraded` fields, mutation count
69→64 after the investigation retirement net of three new mutations, custom
charts' profile fixture missing now-required `DatasetProfile` fields). Full
backend suite: 3990 passed, 1 skipped, 0 failed.

Follow-up (2026-08-26): the artifact detail read now resolves one hop down the
lineage. Chat's tool context includes derived-session result artifacts, so an
answer can cite an id that lives in a `qsess_` partition while the UI fetches
it under the run the user has open — which 404'd and degraded to "could not be
loaded". `ArtifactService.get_artifact` falls back to
`artifact_index_rows_in_children` when the named partition misses: same project
only, direct children only, internal runs still hidden even when an internal
sibling holds the newer copy of the same content-derived id. The partition key
is unchanged and the served detail names the run that actually owns the
artifact. Full backend suite: 3996 passed, 1 skipped, 0 failed.

## Unreleased: injected claims stop buying LLM repair rounds - 2026-08-20

Compare run `sess_1787201833042_9n7rig` validated the report three times with
byte-identical results: the same 5 critical findings, 4 of them on `qfind_q_*`
claims. Those claims are injected from executed question findings, so the plan
LLM cannot rewrite them — both repair rounds were a full m2 call spent on a
guaranteed no-op, and the hard gate pruned the claims anyway.

`_repair_mode_for_code` is now claim-aware: a numeric_mismatch,
currency_unit_mismatch or causal_overclaim on a platform-authored claim id
(`qfind_`/`qbg_`/`qbiz_`/`qfocus_`/`dataset_overview_`/`exec_summary_` and the
deterministic inventory prefixes) routes to `prune` instead of `llm`, so
`_requires_llm_retry` only fires when the plan LLM actually authored something
broken. Published output is unchanged — the hard gate already pruned these.

Gates: `eda_platform/tests/unit` 2951 passed (353 errors and 4 failures are
pre-existing: the venv lacks `httpx2` for starlette's TestClient; both counts
are identical with the change reverted).

## Unreleased: the report override now writes the whole report - 2026-08-13

Forensics on two runs' empty Business Recommendations: the LLM plan proposed
them every time, the causal-language gate killed them ("driven by", "due to"),
and the LLM repair that should rephrase them died of reasoning-token
truncation. Two changes:

- `EDA_REPORT_LLM_*` now routes the m2 claim plan (and its cap raise/restore)
  through the override client, not just the narration — pointing the report at
  a non-reasoning model is otherwise ineffective against plan truncation.
- The plan instructions tell the model that recommendations pair a cited
  observation with an action without asserting causation, and that numbers are
  copied verbatim (0.78 stays a ratio; the 78.3446% unit-conversion mismatch
  was the third pruned claim). Effectiveness is judged by the next live run.

Gates: ruff clean; full offline suite 4090 passed, 1 skipped.

## Unreleased: pre-commit codex sweep — seven cross-file gaps closed - 2026-08-13

A whole-tree codex review before committing the 2026-08-12 iteration surfaced
nine findings; seven were confirmed by code reads or live reproduction and
fixed red-green, one was rejected as an intentional design (verified resource
checks run after per-table analysis by design), one optional deferred.

- The sandbox output archive no longer charges mounted read-only inputs
  against output limits: an input larger than `max_output_file_bytes` used to
  block an otherwise successful execution (reproduced live in Docker).
- A recovered live-LLM job now fails closed instead of silently completing
  offline: startup recovery drops the per-job env overlay, and the worker
  refuses `llm="env"` when no `EDA_LLM_PROVIDER` is configured anywhere.
- The lazy DuckDB catalog falls back to the pandas frame for non-UTF-8 CSVs
  (gb18030/utf-16/latin-1): DuckDB reads them only via the network-installed
  `encodings` extension, so `from_csv_auto` failed on files the loader accepts.
- A verified-limited stop publishes the populated manifest: the handoff
  carried empty `input_hashes`, breaking freshness and replay identity.
- `EDA_LLM_ORGANIZATION` reaches workers again (env overlay + allowlist), so
  organization-scoped OpenAI credentials authorize correctly.
- Endpoint-policy failures during model discovery fall back to the snapshot
  catalog with a warning instead of a 500 from `GET /settings/models`.
- Worker processes inherit `EDA_OBSERVABILITY`, `PHOENIX_COLLECTOR_ENDPOINT`,
  `OTEL_EXPORTER_OTLP_ENDPOINT`, and `EDA_DEBUG_LOG`; stripping them silently
  disabled span export and debug capture exactly where the work happens.
- `api/openapi.json` and the generated TypeScript schema were regenerated:
  they had drifted behind `SessionMetricsView` v7 and its four new fields.
- A leaked sandbox supervisor container now expires with the execution window
  (timeout + adoption + slack) instead of sleeping a year as in-container root.

Gates: ruff clean; full offline suite 4089 passed, 1 skipped.

## Unreleased: review-driven fixes — the run got narrower, the gate got greener - 2026-08-12

A five-track review (trace forensics, report quality, blind recompute, framework
diff, eval diff) of the 2026-08-12 iteration and its deepseek run, followed by
item-by-item fixes. Blind recompute matched 3 of 5 published numbers exactly;
one differed only in basis; one trend claim was unsupported by its own evidence.

- Question-discovery repair retries work again: the `MalformedProviderResponseError`
  wrapper (a RuntimeError) landed in the give-up branch instead of the repair
  loop, so one truncated JSON response silently halved a run's analysis breadth.
- The auto-execution fill lane now tops up from resolved template candidates
  when the exploratory pool is empty, instead of idling half the budget.
- A skipped discovery route is disclosed in the report's Limitations section;
  the candidate set carries `llm_route_skipped` for the exporter.
- The strong-ratio gate no longer scores injected row-count claims or
  exec-summary copies: they are labeled but cannot vote, and the audit note
  reports how many were excluded.
- A truncated trend series reports its observed window ("first N of M periods")
  and refuses a full-series direction verdict; endpoint claims off a truncated
  daily series were the one wrong published conclusion this review found.
- The worker environment allowlist inherits `EDA_SANDBOX_*`, `DOCKER_*`,
  `EDA_LLM_DEBUG_FULL`, and `HOME` again — stripping them silently downgraded a
  hard sandbox requirement to "auto" in workers. `_worker_environment` now has
  direct tests, including per-job overlays being refused for sandbox keys.
- Relationship discovery reuses catalog tables (`has_relation`) instead of
  registering every pandas frame, which pinned all tables in memory at once and
  defeated the one-table frame pool on the eager path.
- A verified-phase resource stop under `on_exceed="limited"` closes the session
  as limited with its per-table artifacts kept, instead of failing after all
  the work was done.
- Trusted CSV ingestion applies the loader's identifier string-forcing via
  dtype overrides, so SQL answers and profiles agree that `store_code` is text.
- Workflow eval: aggregate verifies each trial's case fingerprint and dataset
  identity; the protocol digest (schema 3) covers dataset fingerprints and the
  baseline comparison rejects a mismatched identity; release readiness checks
  that the latest audit's parents include the latest report bundle
  (`report_audit_stale`).

A fresh deepseek run (same data, same seed) then confirmed the fixes live:
the m4 truncation recurred and was recovered by one repair retry, breadth
returned to 11 questions with every domain metric matching the baseline, and
three anomaly questions abstained under their method contracts instead of
shipping SQL proxies. The remaining bottleneck moved to the report claim plan
(three truncated attempts at the 12k completion cap → deterministic fallback),
addressed by two follow-ups accepted in this iteration:

- The m2 claim-plan budget scales with run width (first attempt shrinks one
  claim per question past six; wide runs start at the completion ceiling), and
  the raised cap is restored after plan generation so narration and
  session-title calls keep the operator's configured reservation.
- The deterministic fallback inventory families (row/column counts, quality
  issue counts, availability restatements) are labeled but no longer vote in
  the strong-ratio gate; a pure-fallback report is degraded, not pass.

Full offline suite: 4079 passed, 1 skipped; ruff clean.

## Unreleased: agent harness, eval control plane, and sandbox boundary - 2026-08-12

This iteration re-ran the current agent workflow end to end, audited the new
run evidence against the evaluation and sandbox research reports, and treated
the harness as part of the system under test rather than as trusted glue.

### Round 1 — answer and release contracts

- Added versioned method-aware answer contracts and fail-closed adapters for
  prediction and anomaly work; unsupported forecast, segmentation, and causal
  methods can no longer downgrade into a generic metric answer.
- Made fixed and agentic question execution share the same method-contract
  gate, persisted the effective contract, and stopped prompt/comment/string
  literals from spoofing planner alias repair.
- Hardened exploration certification with planted, negative-control, and
  injection buckets, matched item/tier/seed baselines, minimum trials, and
  production-provider identity checks.
- Reconciled trace, artifact, and fixed-pipeline tool counts and corrected the
  xdist-sensitive scoreboard aggregation.

### Round 2 — canonical workflow evaluation

- Added `WorkflowEvalTrial` as a reproducible case/environment/dataset trial
  with real artifact-parent lineage, resolvable evidence locators, exact usage
  reconciliation, source payload digests, and action/state spans.
- Added explicit milestone and minefield rules, unresolved-approval and report
  publication gates, and dataset fingerprints to trial identity.
- Connected the official workflow-eval CLI to canonical trials. Replay without
  a durable trace is inconclusive, and old component-quality scores can no
  longer override a failed hard gate or determine a passing exit code.
- Added adversarial tests for contract downgrade, fake lineage, bogus locators,
  evidence payload replacement, pending approval, and baseline poisoning.

### Round 3 — sandbox and credential boundary

- Pinned managed LLM providers to registry origins; custom endpoints are
  loopback-only unless an operator allow-lists the exact HTTPS origin. URLs
  with userinfo/query/fragment and credential-bearing redirects are rejected.
- Replaced full `os.environ` worker inheritance with an explicit runtime and
  per-job LLM allow-list. Repointing an endpoint clears the saved credential.
- Made LLM debug capture metadata-only by default (type/shape/size/digest);
  plaintext capture requires the existing explicit opt-in.
- Replaced Docker's live writable host `/work` bind with a byte/inode-bounded
  tmpfs. The user process is run through `docker exec`; residual sandbox-user
  processes are killed before a bounded tar stream adopts outputs, then the
  container is destroyed. Outputs are scanned again before publication.
- Restricted the official local launcher to loopback binds and extended run
  deletion to reclaim both code-agent and question-agent scratch roots.

### Round 4 — framework cleanup and closeout

- Removed the old standalone workflow-evaluator API and renamed its result to
  `WorkflowQualityResult`, making it explicit that quality scoring is only one
  component of a canonical `WorkflowEvalTrial` release decision.
- Removed the reversible legacy eval-spec conversion layer. Cases now compile
  in one direction into `CompiledWorkflowEvalCase`; the CLI, baselines, tests,
  aggregation, and comparisons all use canonical trials.
- Deleted the obsolete host-output watchdog and unused host mountpoint setup
  helpers left behind by the tmpfs migration. Added a
  quiesce/validate/stream/destroy output-adoption protocol and a unit test that
  locks down its ordering.
- Removed the unused fixed-length LLM preview constant. Metadata-only capture
  now has a direct bounded-output assertion rather than an obsolete plaintext
  truncation contract.
- Reworked the public README as the GitHub project entry point, including the
  current architecture, harness and evaluation model, security boundaries,
  quick start, contribution checks, and explicit project/license status. The
  Docker sandbox README now matches the quiesce-and-stream output protocol.

### Round 5 — bounded-memory Auto EDA

- Separated process-level job completion from the analysis session outcome.
  Resource preflight stops remain `limited` through the lifecycle transaction,
  SSE stream, Activity center, top bar, and project list instead of appearing
  as a green completed analysis.
- Replaced run-long lists of pandas DataFrames with content-addressed
  `DatasetSource` handles and a one-table frame pool. Profile, quality, charts,
  statistics, and optional modeling now complete one table's lifecycle before
  the next full frame is loaded.
- Added a private file-backed DuckDB catalog for lazy sources. Trusted CSVs are
  materialized sequentially, external access is disabled before generated SQL
  runs, and the engine has a 512 MB memory ceiling with private-disk spill.
- Made checkpoint identities dataset-specific so repeated per-table steps
  cannot reuse another table's checkpoint.
- Changed resource preflight to model the largest active table rather than the
  sum of all retained frames. A nine-table Olist regression moved from a false
  2.41 GB resource stop to a completed 123-artifact run with a measured 446 MB
  process peak (242 MB above baseline).
- Fixed tmpfs output adoption to run the trusted post-quiescence archiver as
  the sandbox tmpfs owner. This preserves zero extra capabilities while making
  the bounded tar handoff usable; the attested adoption policy is now v2.

Validation evidence and any remaining limits are recorded in the accompanying
2026-08-12 agent-harness review and research reports. A live provider run was
not performed because it would transmit project prompts/data to an external
service; the local three-bucket certificate remains the repeatable release
check for this iteration.

## Unreleased: the planner was never the problem - 2026-08-06

First live execution-accuracy run against the 20-question golden set
(deepseek-v4-flash). Calling `build_plan` directly scored 100%; the same model,
same questions, through the auto-EDA question path scored 75%. Everything lost
was lost after the SQL was written.

- A plan is only asked for approval where approval exists. The planner is told
  to raise `needs_approval` for costly or cross-table work, which is right for
  chat -- it registers the flag with ApprovalService and surfaces a pending
  frame -- and a dead end for an automatic run, where three of the twenty
  questions died on it, all cross-table. The instruction is now addressed only
  to callers that can grant it, so the model is no longer asked for something
  the caller will then refuse.
- A declared column may name its table. `orders.amount` is how a join has to
  write a column, and the model mirrors that into the plan's `columns`, where
  the guard compared it against bare names and called it a hallucination. Two
  more questions died there, and whether a cross-table question survived came
  down to whether the model happened to write the prefix. The qualifier still
  has to name a dataset the plan declared and hold the column.

Together: 75% to 95%, against an 80% target. The remaining failure is a third
kind -- the model filling `columns` with its output aliases rather than the
columns it read, on a schema field that documents neither.

## Unreleased: truncation costs the range, not the ranking - 2026-08-06

Replaying all 82 stored SQL results through the current finding binder found one
sentence that had regressed, in a report already delivered.

- A ranking survives truncation. The partial-result guard refused a ranking and
  a range together, but the preview is `select * from (<statement>) limit n`:
  on an ordering the engine applied, it holds the head, so the leaders are
  provable and only the far end is missing. Four stored questions had gone from
  naming their top rows to saying nothing at all, two of them in the
  2026-08-05 credit-card run. A truncated ranking now names its leaders, drops
  the bunching note it can no longer support, and says how much of the result
  it did not see.

## Unreleased: Compare stops voting on a constant - 2026-08-06

`code_version` was one of five dimensions behind the Compare page's
comparability verdict, and every writer of it is a literal: auto-EDA reported
"local" from a hardcoded function, the two orchestrators their own names, and a
derived run inherits its source's. Two runs built from entirely different code
always matched on it, so the dimension could never appear in
`changed_dimensions` and the verdict read "controlled" for something nothing had
checked. No test asserted on those dimension lists, and the two that touched the
field injected the same value on both sides.

- The dimension is gone. The field is still reported per side; it just no longer
  votes. What is lost with it: two runs from different pipelines are no longer
  flagged as such, which the remaining four dimensions do not cover.
- `_current_code_version()` is deleted from both drivers. Auto-EDA and question
  batches now write what the field actually holds everywhere else -- the name of
  the pipeline that produced the run -- instead of a build marker that was never
  a build marker.

## Unreleased: a rejected plan is a rewrite, not a lost question - 2026-08-06

Across the twelve stored runs, thirteen auto-executed questions never reached
the reader. Ten of them were rejected by a check the planner could have
satisfied, and were thrown away instead of asked again — the planner's own
retry loop already existed for hallucinated columns and unbindable SQL, and
these two checks sat outside it.

- The join scope check runs inside the planner's repair loop (7 of the 13). Its
  rejection carries a `fix_hint` written to be acted on — "rewrite the SQL
  without a JOIN" — which nothing was reading. Both live incidents behind this
  were fixed by teaching `sql_base_tables` a new SQL shape, once for string
  literals and once for `LATERAL`; that is a list with no end, and the loop is
  what makes the next entry cost a call instead of an answer.
- The whitelist check deliberately stays outside the loop, and now runs before
  the first plan is requested. `required_relations` is a fact about the question
  card, identical on every attempt, so re-asking buys the same answer for the
  price of a call.
- A plan needing approval is asked once to fit without it (3 of the 13). The
  planner is instructed to raise `needs_approval` for costly work, and an
  automatic run has nobody to answer it — a dead end the prompt actively invites.
  The scan is bounded by the engine's row cap and the query timeout either way.
  A planner that raises the flag twice still abstains, unchanged.

Replaying all 67 unique stored statements through the refactored guard: one
rejection, a genuine two-base-table join, from the one question that was already
dead on the approval flag.

## Unreleased: the answers were already there - 2026-08-04

Two FIFA runs over identical inputs, and an independent pandas recomputation of
the same twelve tables, agreed the SQL was right. Of ten questions the second
run executed, three were killed by a guard, three were answered and then thrown
away by the layer that turns rows into sentences, and one was published with its
conclusion inverted. Nothing here changes what the model is asked or what the
database computes.

- The join guard counts base tables instead of matching `\bjoin\b`. All three
  questions it killed read one table and joined a CTE back to it, which is how
  DuckDB users write "compare each row against a benchmark". The whitelist was
  empty because relationship discovery is deferred, so no declaration could have
  satisfied it and there is no repair loop behind the check. String literals and
  comments are excluded first: one query advised avoiding "transitions from
  low-elevation venues" and the scan read `low` as a table.
- A ranked claim names the row it ranked. The label was the first non-numeric
  column, which either dropped half the identity — three leaders reading "UEFA,
  UEFA, UEFA", and the whole ranking discarded as indistinguishable — or kept
  the wrong half, publishing "top is result_type" for a row that was
  result_type=Regular. The label is now the shortest leading run of columns that
  tells every row apart.
- A summed ranking metric reports its group size. `ABS(SUM(gap))` put a
  95-match bucket above a 1-match one whose per-match gap was fourteen times
  larger; the count was in the same row and not in the sentence.
- A result with no `ORDER BY` states its range instead of a random row. Min and
  max are computed rather than read off row order, so they survive a missing
  ordering — but not truncation, which now says how many rows there were
  (a 264-row result was being narrated as "Returned 50 rows").
- A metric whose answer is a date may say the date. `time_coverage` returned
  five fields and published one, because every template slot had to parse as a
  number. Relatedly, the validator no longer reads `2026-06-11T00:00:00` as the
  magnitudes 2026, 0 and 0 — a `numeric_mismatch` that fired in both runs.
- A 0/1 flag stored as an integer groups. Both tool-guard rejections were
  `run_stat_test` refusing `is_knockout` and `is_starting_xi`, whose type the
  guard re-derived from the pandas dtype; the profiler had already called them
  boolean. The two now share one predicate.
- Limitations carries its claims *and* the quality scan. One surviving claim
  used to suppress the whole synthesis: same datasets, and run 2 reported a
  38-day window as its only limitation while 11 high-missing columns, 57 outlier
  columns and 3 empty ones left the report.

- Three registry metrics leave the analysis lane. `concentration_hhi`,
  `missing_hotspots` and `time_coverage` carry no domain precondition, so they
  resolve on every dataset and were taken into the first slots unconditionally,
  ahead of every LLM question, sharing one registry sentence as their
  business_decision — including the HHI question, which is not about quality.
  Their SQL is a registry template and costs no model call, so they keep running,
  outside the analysis budget: a metric declares the section its answer belongs
  in, and there it is filed, out of the analysis focus list and out of the
  Executive Summary picker. The exploratory ceiling goes with them; every other
  lane is taken before the fill, so nothing it protected could be starved, and
  keeping it only left idle slots on a dataset with no business metrics.
  Replaying the World Cup candidates: 8 LLM questions become 9, and the tenth is
  refused by the method gate for having 28 rows where anomaly detection needs 30.

- `missing_hotspots` states the numbers it computed. Its interpretation was a
  registry sentence with no placeholders — "The highest-missing columns and
  their null shares" — while its own SQL held 103 of 208 rows (49.52%). Nothing
  bound, evidence came out empty, and the validator dropped the claim: the
  question ran, answered, and reached the reader as nothing. The output columns
  are named after the columns the metric found, so the skeleton is built beside
  the SQL rather than read from the registry, which cannot know the field names.
- The Business Findings fallback stops relabelling background metrics as
  business findings. Routing the injection was not enough: this second path put
  time coverage back on the cover as `qbiz_`, a prefix the summary picker
  selects on, and the dedup pass then left Business Findings holding two
  "See Dataset Overview" stubs and nothing else.

Replaying the stored run: of eight questions that reached execution, three
produced nothing usable and now produce a finding, one wrong conclusion is
corrected, and three more are no longer blocked before they start.

Gate: 3950 backend tests green (one pre-existing full-run-only failure in
`test_api_relationships`, reproduced with these files restored to HEAD); 599
frontend tests green; ruff and tsc clean.

## Unreleased: the report says why, not just what - 2026-08-04

The World Cup run answered 3 of 10 slots with a stadium-capacity concentration
index and a date range, left 3 slots unused, and reported two failed questions
as `(outcome: failed)` with no reason anywhere. The reason had been recorded all
along, in a field addressed to the model that was going to retry.

- The domain-metric lane is capped and the free-form lane spends the rest.
  Three registry metrics carry no domain precondition, so they fire on any
  dataset and outscore every free-form question; uncapped they took the top
  slots while the exploratory cap closed the run early. Replaying the same
  candidate sets: World Cup 7/10 executed with 4 LLM questions becomes 10/10
  with 8; Olist keeps its two best metrics instead of six.
- A question that produced no answer now carries one reader-facing sentence,
  derived in the model validator so no failure path can omit it. It strips the
  tool guard's repair protocol and translates abstention codes. Surfaced in the
  report, the trace, and the question card.
- Limitations groups a condition that recurs thinly across datasets, naming
  every place it occurred. The existing thresholds both needed one dataset to
  carry several flagged columns; eleven datasets with one column each matched
  neither, so the section ran to eighteen near-identical lines. Now twelve, and
  "the Quality page" is a link.
- Report sections get connective prose over claims that already cleared every
  gate. It may only reuse their figures: a paragraph carrying a number the
  claims do not state is discarded and the section falls back to its bullets.
  Citations are built from the cited claims' own evidence, so the model never
  authors an artifact id, and its prose cannot open a markdown block or forge a
  code span. Settings gains a second model picker for it, defaulting to "same
  as the analysis model"; the choice reaches the worker through the existing env
  overlay rather than a new config channel.
- A test helper wrote to `params_json` before the migration that creates the
  column, so one test passed only when another happened to run first.

Gate: 3920 backend tests green; 599 frontend tests green; ruff and tsc clean.

## Unreleased: gate verdicts match by content, and the frontend says when a run broke - 2026-08-03

The first gpt-5.6-luna trial ran clean and was then refused with an eighth
issuance error: `journal gate id differs from canonical reducer bundle`. The
issuer paired gate verdicts to hypotheses POSITIONALLY, by the order each
hypothesis first committed a receipt. The reducer gates in selection order.
Serial runs make those two orders coincide; concurrent probe sessions
interleave their commits, so every pairing shifted and a healthy root read as
forged.

- Gate verdicts are matched by content: a verdict's bundle id must be
  recomputable from some ungated hypothesis's (candidate, receipts) group.
  Order-independent, and strictly stronger than before — the old rule trusted
  an id because its position lined up. Verified by re-verifying the real
  interleaved luna root (now ISSUED OK), a synthetic interleaved regression, a
  control (an invented bundle id still raises) and a mutation.
- Audited every other order dependency in the issuer: decision_cursor runs in
  the serial admit phase, tool_results_digest fingerprints the journal as-is,
  and sequence_index is session-local. This was the only one.
- Provider-unavailable classification now also covers the generate phase,
  which still recorded a 503 as an unknown outcome and consumed its
  reservation.
- Exploration UI: a stop the run did not choose gets an alert with its
  consequence, not the same grey line as a converged run. `failed` and
  `state_witness_changed` are alerts, `budget_exhausted` is a warning that
  says results are incomplete, healthy stops stay quiet.

Gate: 3546 unit tests green; 586 frontend tests green; scripted smoke issues
at concurrency 3.

## Unreleased: a busy provider must not bill a run or end it - 2026-08-03

Seed 8 met a DeepSeek capacity outage 33 seconds in. Six concurrent probes got
HTTP 503 "Service is too busy"; the run ended `failed` with no report, two
committed receipts were discarded, and 173,587 tokens ($0.037) were billed for
zero output. Three defects, all in the "a normal external condition ends the
run" family:

- 503/429/502/504 are now retried on their own backoff ladder (1/2/4/8s). The
  reason transport retries stay short — a retried read timeout may already have
  been served and billed — does not apply here: the provider states it did not
  process the request, so a resend cannot duplicate a generation. 500 stays
  un-retried on purpose (an internal error can follow a processed request).
- An exhausted ladder raises the typed `ProviderUnavailableError`, which the
  ledger settles as unserved: the reservation is released, the request slot
  returns, and the usage event records zero rather than the reservation.
- `provider_unavailable` joins PROBE_LOCAL_ERROR_CODES, so the probe ends
  locally and the round still settles and renders its report. The journal
  records the call as `rejected`, not `uncertain` — nothing was generated.

Controls in the same suite: a 400 is a served answer and is never retried; a
digest mismatch still ends the run. Concurrency was confirmed working on the
real provider in the same trial (peak 4 in flight, 6/8 overlapping) — and it
is what turned one outage window into six full reservations, which is why the
accounting fix matters.

Scope, stated plainly: the ladder spans ~15 seconds, so this survives a blip,
not an incident. DeepSeek's status page put the seed-8 outage at 15+ minutes.
Two gaps remain open (see docs/agent/findings-log.md): a sustained outage
still ends a generate call as a hard failure, and a run whose probes all fail
this way settles two empty rounds and stops as `no_new_information` — a stop
reason that would claim convergence the run never reached.

Gate: 3544 unit tests green; scripted smoke issues at concurrency 3.

## Unreleased: concurrent probe sessions and four waste guards - 2026-08-03

Seed-6/7 measurement: 97% of wall clock is serial LLM waiting (0/227 calls
overlapped), and healthy exploration burns to the hard budget cap because the
soft stop only catches degenerate runs. Two workstreams from
docs/exploration-speedup-plan.md:

Parallel line (P):
- Journal pending state is multi-slot for both tools and LLM calls (the
  planned "atomic segment" died on review: tool_call_started is a
  pre-execution budget gate and cannot be bundled after execution). Legacy
  scalar fields remain, always None; a stopped root's snapshot equals a fresh
  replay, proven by test. In-flight slots count against the budget caps so
  concurrent starts cannot over-admit.
- Budget ledger file appends are atomic (thread lock + sidecar flock).
- Adapter last-call metadata (usage, request id, byte counts, param repairs)
  is thread-local; the ledger client's whole-call lock is gone. Two 0.25s
  fake calls now overlap (red test was 0.51s serial).
- SupervisorProbeExecutorPort gains probe_concurrency (default 1 = the old
  serial loop, byte-for-byte). Sessions run in a thread pool; results keep
  selection order; receipt validation stays sequential. Harness flag
  --probe-concurrency. Scripted smoke at concurrency 3 issues end-to-end
  with results identical to serial.

Waste line (W):
- W1 error-fingerprint circuit breaker: the same (tool, canonical error)
  twice disables the tool for the rest of the run (seed-6: run_domain_metrics
  failed 18x identically). Counter rebuilt from journal events on resume;
  wired in the composition.
- W3 sufficient-evidence wind-down: one adjudicating receipt injects
  corroborate-then-conclude guidance; two same-direction receipts close the
  session's tool inventory.
- W4 zero-row filter families: the third value-variant of a filter family
  (same columns and operators) that already matched zero rows twice is
  rejected locally (seed-6: 24 such retries).
- Deferred: W2 (feeding disabled method families back into GENERATE and
  admission) — needs a new executor-to-generator seam; tracked in the plan.

Gate: 3539 unit tests green; scripted smoke issues at concurrency 1 and 3.

## Unreleased: seed-7 review — the seventh check, and predicates that name metrics - 2026-08-03

Seed 7 (first real run after the six-check fix) certified past every receipt
check and was then rejected by a seventh sibling nobody had counted:
`_verify_scheduler_and_reductions` requires every persisted scheduling
decision to be consumed by a reduction, and the interrupted trailing round's
decisions never are. The fixture missed it because its trailing round had
receipts but no decisions — a shape that cannot occur (probes only run on
chosen candidates).

- Trailing decisions are tolerated iff they exactly cover the trailing
  round's recovered candidate batch with matching identity — the same checks
  a settled round gets, minus the reduction that never ran. A smuggled
  decision the batch cannot account for still rejects (negative test +
  mutation-verified).
- The interruption fixture now writes the full real shape: generate recovery
  body, ledger triple (reserve/usage/settle), persisted decisions, orphan
  receipts, report and checker artifacts all consistent.
- Metric-name resolution for adjudication: the model's natural predicates
  name derived metrics under numeric comparison ("pearson_correlation > 0.6",
  "r2 > 0.2") while facts are tool-namespaced ("pair0.coefficient",
  "metric.r2") — in seed 7 all 30 correlate/baseline receipts adjudicated to
  None because of this. Now: a tool-agnostic "metric.<name>" namespace
  fallback, an r2 alias family, and a correlation-coefficient family resolved
  through the pair facts. Metrics needing computation (rmse ratios) or with
  no backing fact (trend_slope) stay honestly unadjudicated.
- Retrying issuance on the real seed-7 root now passes the scheduler check
  and is refused at typed-adjudication replay instead — correct: the
  resolution layer changed adjudication semantics, so pre-change roots
  cannot replay under post-change code. New runs issue under the new
  semantics; old roots stay archived under the old.

Gate: 3506 unit tests green; scripted smoke issues end-to-end.

## Unreleased: an interrupted run is certifiable in all six checks, not one - 2026-08-03

The previous entry claimed issuance tolerated a budget-interrupted trailing
round. It relaxed one of six exact-set checks on that path, so the same root
was still rejected — with a different message. A regression test that calls
`verify_e4a_evidence_root` itself (rather than the one helper) found the rest,
including one nobody predicted.

- `_rebuild_canonical_bundles` skips orphan receipts and accepts a trailing
  open round only when it contributed nothing certified — no canonical bundle
  and no gate verdict.
- `_verify_tool_results` and `_verify_stat_registry` read the journal as of the
  last settled round. Usage and spend still count the whole journal: the
  interrupted round's work was really paid for.
- Release grade is now the reducer's rule, not the tool allowlist (option B,
  user decision): only receipts an insight rests on need a durable result
  contract. Reconnaissance an insight never cited — `profile_slice` and the
  other eight tools without contracts — no longer blocks issuance.
- `claim_gates` module docstring said the statistical gate records without
  blocking; it blocks like the other five. Corrected.

Gate: 3493 unit tests green; scripted smoke still issues end-to-end.

## Unreleased: seed-6 improvement pass — measure what the model actually did - 2026-08-03

The first open-exploration run (5 rounds, 228 LLM calls, $0.32) worked but was
mis-measured and could not be certified. Fixes across the whole chain:

- `_verify_committed_receipts` tolerates an interrupted trailing round:
  journal-only receipts are accepted iff they were committed strictly after
  the last `round_settled` event; every workflow receipt must still be
  journal-committed. (Only one of the six checks that reject such a root —
  see the follow-up entry above.)
- A mid-round budget latch no longer loses the certified report:
  `PhaseContext.terminal_reason` carries the latched stop reason into the
  deterministic renderer instead of silently falling back to a bare terminal.
- Checker v2 (`e4a-checker-v2`, version-gated): two-pass matching — exact
  predicate, then semantic (tool + columns + typed supports + optional
  required fact values). Independent replication is credit, not a false
  positive. New scores: `mandatory_probe_recall`, `agent_discovery_count`,
  `independent_replication_count`, `agent_novel_supported_count`,
  `refuted_insight_count`, `agent_first_discovery_round`; search dynamics are
  structure-level. The fixture gains the fourth planted structure
  (`planted_trend_revenue`); seed-6 rescored: precision 0.60 → 1.0,
  recall 4/4, two structures independently replicated by the agent.
- Typed adjudication for `correlate_columns`, `screen_anomalies` and
  `run_baseline_model` — 30 of seed-6's 196 receipts carried evidence that
  could never move an insight; these tools can now adjudicate.
- Harness: profile/quality artifacts are seeded before the run (23 of 40 tool
  failures were the missing-profile precondition), run witness derived after
  seeding, `--rescore` recomputes the checker offline, refused issuance no
  longer dies in `_freeze_manifest` (this also silently dropped seed 4 from
  the trials summary).
- Report is written for humans: `InsightRecord` gains optional
  `statement`/`rationale`, the renderer leads with the hypothesis statement
  plus adjudicated key stats; the claim dump stays in workflow-state.json.
- Plan-B soft stop (user decision): two consecutive rounds with zero
  adjudicated transitions end the run as `no_new_information` even while gate
  admissions keep counting as progress; journal-derivable, legacy journals
  never soft-stop retroactively; branch-enabled runs defer to E6 stagnation.
- Probe-loop prompt now steers toward adjudicating tools and away from
  repeated descriptive slicing.

Gate: 3491 unit tests green; scripted smoke issues end-to-end
(`issued=True`) for the first time.

## Unreleased: ten paths where normal model behaviour ended the run - 2026-08-03

A systematic scan of every path that terminates an exploration found ten, six
of which a working model triggers routinely. All ten are fixed, and the first
fully successful live trial followed: all three planted structures found,
`recall=1.0`, grounding rate 1.0, zero fabricated receipts, 21 receipts and 5
gate passes in 189 seconds for $0.019.

### Recoverable, not fatal
- Structured proposal calls are retried once with the schema error fed back,
  and two unusable answers conclude the round instead of ending the run. The
  batch cap the contract enforces is now stated in the prompt, read from the
  schema so the two cannot drift.
- Nine more provider-response shape errors raise `MalformedProviderResponseError`
  and travel the executor's rejected-and-retry path.
- Transport-level failures (timeouts, connection errors) get a bounded backoff
  retry; HTTP status errors still fail immediately.
- A probe that the model cannot finish — truncated by `max_tokens`, filtered by
  the provider, or answering empty — no longer ends the run; its committed
  receipts stand and the round continues. The empty-answer allowance is now a
  per-run budget of three rather than a single global flag.
- A model that emits more tool calls in one turn than the batch allows has the
  excess rejected with an observation instead of crashing after executing the
  first twelve.
- Contradicting evidence from a narrower scope is recorded as a limitation on
  the insight instead of raising: probing globally and then stratifying is the
  behaviour this loop wants, and it was a fatal error.
- Cancellation inside a probe now stops the run as `cancelled` rather than
  `failed`.
- A reduction body remembered before its journal commit can be recomputed on
  resume; once the journal has committed it, it stays immutable.

### Changed
- Trial issuance is opt-in (`--require-issuance`). Issuance is the release
  gate, not the measurement: a refused root still yields scores, usage and
  timings from the run's own durable artifacts, and the root can be re-verified
  later. Known gap: only 3 of 11 certified tools have durable result contracts,
  so live runs currently record `issued=false`.

Gates: Ruff clean, Pyright `0 errors`, backend `3449 passed / 0 failed`,
scripted trials quick/standard/deep all `recall=1.0`.

## Unreleased: a probe that runs out of steps no longer ends the run - 2026-08-03

`ProbeExecutor` returns `limit_reached` when a probe conversation uses up its
own step or tool-call budget — an expected outcome for an eager model, whose
committed receipts are already journaled and valid. The workflow port treated
every non-`completed` status as a control-plane invariant violation, so one
such probe ended a live trial in round 0 after thirty committed receipts.
`limit_reached` is now accepted and the round proceeds with the evidence it
gathered; `failed` stays terminal because it also covers integrity faults.

The trial harness now prints the supervisor's error string, which lives only
on the in-memory result — the journal never records why an unclassified
exception ended a run.

Gates: Ruff clean, Pyright `0 errors`, backend `3420 passed / 0 failed`,
scripted trials quick/standard/deep all `recall=1.0`.

## Unreleased: three defects the first post-fix real trial exposed - 2026-08-03

The exploration loop now produces supported and refuted insights on a live
provider (2 supported, 1 refuted, 1 contested across 3 rounds and 40 receipts,
against 2 all-inconclusive before). Three defects still blocked its evidence
root, all fixed:

- **The multiplicity check judged immutable receipts against their family's
  final size.** A receipt issued as attempt 2 correctly caches `p x 2`; when a
  third test joins the family it cannot be re-signed, and the previous fix
  still voided it. The gate now checks only self-consistency with the attempt
  number the registry assigned to that receipt — forgery remains catchable,
  legitimate family growth does not. (The earlier control test encoded the
  legitimate case as the forgery; it was rewritten to a genuinely
  self-inconsistent receipt.)
- **The report's proof validator was a third copy of the pre-R1 invariant.**
  It required an insight's evidence to equal its bundle's receipts exactly and
  its proof to cover every bundle fact, so a bundle carrying an exploratory
  unadjudicated claim could not render. Evidence must now be drawn from the
  bundle, and proof must cover exactly the adjudicated receipts. This module
  had no unit tests; it has them now.
- **Malformed provider tool arguments ended the run.** A provider that answers
  with unparseable arguments has answered — the answer is merely unusable, so
  it now raises the new `MalformedProviderResponseError` and travels the
  executor's existing rejected-and-retry path instead of the unknown-outcome
  path that discards every round already paid for.

Gates: Ruff clean, Pyright `0 errors`, backend `3418 passed / 0 failed`,
scripted trials quick/standard/deep all `recall=1.0`.

## Unreleased: full-pipeline review — seventeen fixes across the exploration loop - 2026-08-03

A six-agent review (five code domains plus artifact forensics on the real
deepseek trial) traced the observed recall=0 to a chain of seven defects, none
of them model capability. Seventeen fixes landed in three waves, each
red-to-green with its own regression, plus adversarial mutation checks on the
issuer mirrors.

### Evidence semantics
- Unadjudicated receipts (profile_slice et al.) no longer poison an insight:
  they stay in the claim bundle as exploratory claims but enter neither
  evidence side; a decisive `supports` receipt now yields `new/supported`.
- Typed outcomes are hard-validated against their assigned side (a
  `contradicts` receipt listed as supporting raises instead of minting
  `new/supported`).
- Same-snapshot corroboration keeps the prior status instead of downgrading
  to `inconclusive` (recorded as a limitation), so extra verification can no
  longer reduce recall.
- Absence claims carry the scanned scope copied verbatim from their receipt;
  the issuer's canonical bundle mirrors both changes (mutation-verified).
- `differs`/`associated_with` can now adjudicate to `contradicts` (p > 0.05,
  or significant-but-immaterial effects); "verified, no effect" is
  representable at last.
- The stale-multiplicity gate recomputes the family correction at
  verification time instead of retroactively invalidating immutable receipts.

### Generation and scheduling
- GENERATE now receives the dataset schema and method vocabulary; the model
  no longer proposes placeholder dataset ids that admission must reject.
- Mandatory probes replay every round until their coverage completes and are
  exempt from historical-fingerprint dedup; quick tier structurally reaches
  full coverage (scripted recall 1/3 → 1.0).
- Per-round scheduling decisions are stored without cross-round dedup, so
  frontier digests replay; the issuer compares candidates by semantic
  identity (sequence_index and friends excluded) and repeat candidates
  verify end to end.
- Tier budgets resized from calibration data: 3000 output tokens per call
  across tiers (measured reasoning peak 2443), quick requests 8 → 12 — the
  old quick budget could never finish its own three rounds.

### Control plane
- Budget exhaustion mid-run recovers the last committed reduction and renders
  the deterministic report instead of returning nothing.
- Admitted gate bundles count as progress; an empty frontier no longer
  double-counts as "candidates below the admission line" (two consecutive
  empty frontiers are required for exhaustion).
- `gate_verdict` writes are idempotent, so a crash between verdict and
  reduction no longer permanently poisons the journal replay cursor.
- Usage-meter contract violations settle as failed tool calls the model can
  recover from; only terminal budget/cancellation errors end the run.
- An uncertain LLM call no longer blocks issuance: the issuer bills it at the
  same conservative reservation the budget restore path already charges.
- `max_cost_usd=None` means uncapped, not zero budget.

Gates: Ruff clean, Pyright `0 errors`, backend `3413 passed / 0 failed`,
scripted trials quick/standard/deep all `recall=1.0` with full issuer
verification.

## Unreleased: an empty frontier still commits a reduction - 2026-08-03

### Fixed

- **A round whose frontier came out empty now commits a probe-free reduction
  before it settles.** That round still scored candidates, and the reduction
  event's `frontier_digest` is the only thing binding those scheduling
  decisions to the journal — settling without one left them unverifiable, and
  the E4a issuer rejected the whole evidence root. A real deepseek-v4-flash
  trial hit this on its second round, where every model proposal was rejected
  by admission. `ReducerPort` gains `reduce_without_probes`; the supervisor
  routes the empty-frontier case through the normal settle path, so the
  terminal edge also runs the finalizer and the root gets its report artifact
  (the separately recorded "graceful stop without report.md" gap).

### Added

- Trial harness: `--budget-scale N` runs a capability probe on a scaled tier
  budget, `--max-tokens` overrides the per-call output cap, and every run
  prints per-call completion/reasoning tokens. Calibration trial ids carry an
  `-xN` suffix and their manifests are refused entry to `trial-plan.json` — a
  scaled run is never a release candidate.

Gates: Ruff clean, Pyright `0 errors`, backend `3379 passed / 0 failed`.

## Unreleased: certified exploration tools must be able to produce receipts - 2026-08-03

### Fixed

- **Six tools that cannot emit an `EvidenceReceipt` were removed from the
  certified exploration inventory**: `inspect_data_catalog`, `list_artifacts`,
  `read_artifact`, `run_sql`, `list_saved_skills`, `run_saved_skill`. The probe
  executor refuses a successful tool call without a receipt, so any model that
  called one killed the whole run with an unclassified `failed` — a real
  gpt-5.6-terra trial called `inspect_data_catalog` on its first probe.
  `run_open_analysis` stays: its receipt is the injected executor's contract and
  it never materializes without one.
  `test_every_certified_exploration_tool_can_produce_an_evidence_receipt` now
  executes every certified tool and asserts a receipt comes back, because a
  static list goes stale and a test double hides the real tool's contract.

## Unreleased: gpt-5.6 tool-calling pin — the whole family, not just luna - 2026-08-03

### Fixed

- **`gpt-5.6-terra` and `gpt-5.6-sol` now pin `reasoning_effort: "none"` on tool
  turns, like luna.** The live API rejects function tools at the family's
  default reasoning effort on `/v1/chat/completions` ("Function tools with
  reasoning_effort are not supported ... set reasoning_effort to 'none'") — the
  client never sent the field; the server-side default is what conflicts, so it
  must be explicitly overridden. A test previously asserted the opposite for
  terra and sol, turning an unverified inference into a stated fact. terra and
  luna are observed; sol is pinned by family/endpoint inference. Tool turns lose
  reasoning as a result; restoring it means migrating to `/v1/responses`.
  Capability catalog version is now `agent-models-2026-08-03`.
- E4a trial harness sizes per-call `max_tokens` from the tier budget
  (`max_output_tokens // max_requests`) instead of inheriting the interactive
  `.env` default, which equalled the quick tier's entire output budget and
  exhausted every run on its second call. It also reports `stop_reason` and
  settled rounds when the issuer rejects a root, and refuses to re-run into an
  already-stopped journal.

## Unreleased: E4a trial harness + two evidence-pipeline fixes - 2026-08-03

Preparation for the real-provider E4a trials on the planted retail fixture.
The offline (scripted-provider) validation runs the full arc per tier: shadow
run → evidence root → deterministic checker → issuer verification → frozen
manifest in an issuer-owned trial plan. Real runs await interactive execution.

### Added

- `tests/evals/exploration_baseline/run_e4a_trials.py`: per-(tier, seed) trial
  runner mirroring the worker composition (minus the release-certificate gate,
  which cannot exist before trials). Ground-truth fixture predicates are taken
  verbatim from the mandatory probe seeds, so the three planted structures are
  reachable by deterministic coverage probes. Idempotent and resumable; never
  issues a production certificate.

### Fixed

- **`analyze_time_series` now registers its Ljung-Box test on the multiplicity
  ledger.** The receipt carried a p-value with no registry attempt, so the E4a
  issuer rejected any evidence root containing the mandatory spike probe. The
  tool now begins/completes a registry attempt and stamps
  `statistical_family_id`/`sequence_index`; the issuer derives the expected
  family from `time_column`/`value_column` and knows the tool's fixed test.
- Trial harness writes the deterministic report for graceful stops that end on
  the empty-frontier path (which skips the finalizer — known product gap, a
  supervisor-side fix is still owed); the issuer re-verifies the bytes.

Observed in offline validation: quick tier recall is structurally capped at
1/3 (batch size 1 + round-0-only mandatory seeds + decision-fingerprint
dedup); standard and deep reach 3/3. Recorded in the findings log as an open
admission-semantics decision.

Gates: Ruff clean, Pyright `0 errors`, backend `3377 passed / 0 failed`.

## Unreleased: E6 adaptive branching — stagnation-triggered, dark by default - 2026-08-03

The E6 shape from the plan (sequential by default, stagnation-triggered
branching, max depth 2, abandoned lines feeding structured admission
constraints, no true parallel execution) is fully implemented and dark: no
tier profile enables it, so runtime behavior is unchanged until E4a trial data
justifies turning it on.

### Added

- **Branch-aware journal, linear as before.** `RoundStartedEvent.branch_id`
  plus a new `BranchAbandonedEvent`; the reducer enforces that abandonment
  requires the system stagnation signal (a provider cannot declare branch
  mode), successor branches follow deterministic naming, and
  `no_new_information` is rejected while branch budget remains — so "branches
  must be exhausted before terminating" is replayed identically by the issuer.
- **Deterministic abandonment constraints.**
  `agents/exploration/branching.py` derives "tried + why it failed" entries
  (refuted / gate_rejected / inconclusive) from receipts, gate reports and
  candidate seeds; hypothesis attribution parses the executor-minted run_id.
  The live deriver and the issuer's `_verify_branch_abandonments` run the
  same function; tampered constraints fail the root.
- **Admission upgrade.** `AdmissionContext.abandoned_constraints` and a ninth
  check `not_previously_abandoned`: refuted/gate-rejected fingerprints are
  hard-blocked, and the scheduler port force-merges journal constraints so no
  environment-specific builder can drop them (mutation-verified).
- **Supervisor branch mode.** `SupervisorConfig.branch_trigger_stagnant_rounds`
  / `max_branches` (validated as a pair); stagnation switches lines instead of
  terminating until branches are exhausted; branch policy rides in
  `ExplorationBudgetPolicy.branching` and is absent from every tier profile.
- End-to-end shadow-driver test covering the full arc: refuted main line,
  abandonment with derived constraints, blocked re-proposal inside the
  branch, cross-branch stat-family accumulation (2 attempts, 1 family), and
  issuer recomputation accepting the honest journal while rejecting a
  tampered one.

### Changed

- `ExplorationBudgetPolicy` gained the optional `branching` field: every
  policy fingerprint changes and pre-existing journals cannot resume. The
  platform has no production runs, so this is acceptable now and noted as a
  schema-evolution caveat for later. OpenAPI and generated frontend types are
  re-exported (digest stable across two exports).

Gates: Ruff clean, Pyright `0 errors`, backend `3376 passed / 0 failed`,
frontend `41 files / 583 tests passed`, typecheck and production build green.

## Unreleased: E4a–E5 close-out — search dynamics metrics, verification-boundary pins - 2026-08-03

Everything left before real-provider trials that does not spend money. The only
remaining step for E4a is cost-authorized provider runs.

### Added

- **Search dynamics scores in the deterministic checker** (plan §10.3.1, E6
  prerequisite). `_recompute_checker` now emits `auc_over_steps` (mean
  recall-at-round over `rounds_started`) and `first_improvement_step` (smallest
  `created_round` among matched insights, `-1.0` when none), both recomputable
  from the journal and workflow state alone.
- **Regression pins on the verification boundary.**
  `test_manifest_digest_covers_numeric_changes` proves a numeric change in a
  tool artifact changes `manifest.root_digest`;
  `test_trial_plan_rejects_unpinned_manifest_digest` proves `_verify_trial_plan`
  fails closed on an unpinned digest (mutation-verified: disabling the
  comparison turns both new and existing plan tests red).

### Documented

- The issuance-flow constraint is now in the plan document's E4a section:
  the trial-plan manifest digest must be frozen before an evidence root leaves
  issuer control, and standalone `verify_e4a_evidence_root` is an internal
  audit, not an admission gate — production issuance goes through
  `issue_e4a_release_from_evidence_roots`.
- TOCTOU assessment for concurrent evidence-root replacement: the byte-level
  window between semantic reads and digest computation exists but is not
  exploitable under the frozen-manifest trust model; no code change
  (docs/agent/findings-log.md, 2026-08-03).

Gates: Ruff clean, Pyright `0 errors`, backend `3359 passed / 0 failed`,
exploration-targeted 15-file suite `265 passed`.

## Unreleased: Exploration workflow E4a–E5 — shadow loop, API, product surface - 2026-08-03

The autonomous exploration loop, its API and its product surface are built and
pass every code gate. The production gate stays shut: no release certificate is
installed, the trusted release key map is empty, and the capability defaults to
false, so nothing here is reachable by a user yet. Remaining before that gate
opens: real-provider trials at five or more runs per tier, which need explicit
cost authorization.

Gates on the frozen tree: Ruff clean, Pyright `0 errors`, backend
`3355 passed / 0 failed`, exploration-targeted suite `248 passed`, frontend
`41 files / 583 tests passed`, typecheck and production build green, OpenAPI and
generated types free of drift.

### Added

- **A single-supervisor exploration loop that survives a crash.** Candidate
  generation, frontier, scheduler, bounded executor and reducer run under
  `drivers/exploration.py` with a single writer, an attempt fence, and recovery
  from the journal plus snapshot. The journal is the authority; a projection can
  never report success the journal does not carry.
- **An evidence issuer that replays provenance rather than trusting it.**
  `exploration_evidence_issuer.py` rebuilds every value it certifies from the
  durable run — journal, recovery bodies, candidates, receipts, gate reports and
  manifest — and refuses self-reported attestation. Each committed receipt must
  bind to exactly one canonical provider invocation.
- **A durable result contract so a receipt is never its own evidence.**
  `data_tool_result_contracts.py` reconstructs facts, statistics, output digest
  and artifact binding for the three release-grade tools from the separately
  persisted primary artifact and provider-facing result content. Production
  adapters and the issuer run the same contract.
- **Typed receipt adjudication.** Hypothesis outcomes for stat, missingness and
  time-series receipts are recomputed by the system. An absolute threshold can
  no longer be answered by the sign of a standardized effect, and informational
  method warnings no longer collapse a real result to inconclusive.
- **Exploration API with a resumable pause and a fail-closed certificate gate.**
  `prepare / start / get / pause / resume / cancel / extend-budget` plus SSE
  cursored on `<exploration_id>:<seq>`; reconnect resumes without a duplicate
  job. Pause is recoverable, stopped is terminal, and the worker re-verifies
  policy, witness, certificate and runtime bindings before it runs.
- **A typed exploration UI behind a capability flag.** Current hypothesis,
  evidence, proof trail, coverage gap, base and amended budget, actual cost,
  stop reason and the fixed six-section report. Exploratory and confirmatory
  evidence are separate lanes with no certainty wording; navigation appears only
  when the capability is true.

## Unreleased: EDA scripts & visualization review — P1 fixes - 2026-08-02

Second pass over `docs/eda-scripts-and-viz-full-review-2026-08-02.md`, covering
the stability and resource items. Theme: put a bound on the unbounded, and stop
answering 200 to a request the pipeline cannot actually honour.

### Fixed

- **A skipped automatic statistical test leaves a trace.**
  `SessionStatTestsStep` swallowed `ValueError` and returned no artifacts, so an
  auto-selected comparison could vanish with nothing to debug against. It now
  emits `stat_test_skipped` carrying the test type, both columns and the reason,
  matching what `SessionBaselineModelStep` already did.
- **Aggregated custom charts are bounded.** `_stream_grouped_chart` accumulated
  one state per group with no ceiling, and `median` kept every value in a list —
  both unbounded on a high-cardinality column. A request over
  `CUSTOM_CHART_GROUP_LIMIT` (1,000) groups or
  `CUSTOM_CHART_MEDIAN_VALUE_LIMIT` (1,000,000) retained values is now refused
  with a message naming the fix, rather than silently sampled or left to OOM.
- **A numeric aggregate over a text column is a 422, not a chart of zeros.**
  `sum`/`mean`/`median` coerce with `errors="coerce"`, so a text Y produced a
  200 response full of `0` and `null`. The request is rejected when no value in
  the column parses as a number; a column stored as text but holding numbers
  still charts.
- **Truncation says what was actually dropped.** An aggregated chart keeps every
  observation in `row_count`, so the byte budget cutting whole groups left the
  client reporting "showing 12,000 of 12,000 rows". `CustomChartView` gains
  `series_truncated`, and the builder's notice distinguishes dropped groups from
  dropped rows.
- **The vega expression gate covers transform constructs.** `_EXPRESSION_KEYS`
  now includes `calculate`, and a string-valued `filter` is treated as an
  expression while a field-predicate object stays allowed.
- **The expression gate also runs at the render seam.** ChatPage renders a raw
  `ChartSpec`/`RawChartSpec` artifact payload straight from the generic artifact
  endpoint, which never passes through `get_chart`. `VegaChart` now applies the
  same check itself, so every caller is covered wherever the spec came from.
- **A view that resolves after unmount is finalized.** `VegaChart`'s cleanup ran
  against a still-null `result` when the component unmounted mid-embed, leaking
  the view for the rest of the session.

## Unreleased: EDA scripts & visualization review — P0 fixes - 2026-08-02

Acts on the six P0 items in `docs/eda-scripts-and-viz-full-review-2026-08-02.md`
after re-verifying each against the code. Theme: stop the pipeline from
overstating what it produced — a degraded run reporting success, a sampled
relationship confirming itself, charts computed but never shown, and two
subsystems reporting contradictory numbers about the same column.

### Fixed

- **A resource-limited run no longer reports itself as completed.** Both
  preflight exits in `run_auto_eda` published preflight/metrics/handoff, no
  profile, no charts and no report — then marked the session `completed`. They
  now mark it `LIMITED_SESSION_STATUS` (`"limited"`), which `SessionNav` treats
  as terminal (polling stops) and `SessionRail` renders as a warning dot rather
  than a success dot.
- **A sampled relationship can no longer auto-confirm itself into the join
  whitelist.** `propose_join_candidates` gated auto-confirmation on
  `validation.verified`, which is true after a full-table validation SQL — but
  the `confidence == "high"` label it also requires comes from overlap
  statistics that may have been computed on a bounded sample. Auto-confirmation
  now additionally requires `not sampled` on both the validation and the
  candidate; sampled edges stay `proposed` for a human to confirm.
- **Field-level diagnostic charts are visible again.** `Distribution of…`,
  `Top values in…` and `Missing values by column` were filtered out of the
  chart gallery under a note saying they were "available in Profile and
  Quality" — where Profile shows only a `DIST_SAMPLE_CAP`-row sampled sparkline
  and Quality renders no Vega chart at all. They now render in a collapsed
  `Field diagnostics` disclosure below the gallery (closed by default, so the
  deferred spec fetch still does not fire), batched six at a time.
- **The histogram outlier switch names the axis it actually cuts.**
  `_full_histogram_chart` fences the X column, while the builder's label
  promised a Y fence and the API refused the request unless a (subsequently
  ignored) Y column was chosen. A histogram may now set `drop_outliers` with no
  Y column, skips the pointless Y-fence scan, reports `row_count` as the number
  of binned observations, and the label reads "from X" for a histogram.
- **`quality_context` stops recomputing what the profiler already measured.**
  Its pattern facts contradicted the very issues they annotated: whole-row
  `frame.duplicated()` reported 0 duplicates whenever a surrogate key made
  every row unique, `pd.to_numeric` turned `"80,824"` into NaN and reported 0
  outliers, and a bare `to_datetime` read `20240101` as an epoch offset and
  found no parse failures. Duplicates now restate `duplicate_rows` and its
  scope, outliers reuse `parse_numeric_like` + `iqr_bounds` (the same fence
  `_iqr_outlier_count` applies), and date failures read
  `ColumnProfile.parse_failure_count`.
- **`classifyColumn` puts the semantic type ahead of the dtype.** CSV date and
  amount columns read back as `object`, so the dtype branch classified them as
  `text` before the semantic branches were reached — the Type filter missed
  them and the icon contradicted the `semantic_type` shown in the next cell.

## Unreleased: Post-merge E0–E3 review + PR #1 residue cleanup - 2026-08-02

Second foundation pass after `bd66d81` (E0–E3) and merge of PR #1
(`codex/refine-eda-workbench-ux`). Goal: judge E0–E3 completeness against
`docs/agent`, close holes that would poison E4a, and remove UX leftovers that
over-promised exploration.

### Status judgment

| Slice | Code | Production wiring | Verdict |
|---|---|---|---|
| E0 answer/budget fuse | done | live on question path | complete |
| E1 tool surface | done | live | 16 fixed tools + optional open analysis; `diagnose_missingness` and `run_baseline_model` are now agent-exposed |
| E1.5 receipt integrity | done | live on analysis tools | complete for current tools |
| E2 durable journal/budget | foundation | **no supervisor consumer** | library + tests only |
| E3 claim gates/renderer | foundation | **not the production answer exit** | library + tests only |
| E4a exploration loop | not started | — | blocker for “autonomous explore” product |

Eval-0 remains the control: planted insight F1 = 0 for both measured models;
absence/injection/grounding full marks. Bottleneck is dimension choice, not
arithmetic.

### Added (this pass)

- **E1 closes: two more agent tools.** `diagnose_missingness` reports observable
  missingness structure (rates, indicator correlations, group-rate ranges,
  Holm-corrected target association) and states in its own contract that
  observed data cannot rule out MNAR — it never labels a mechanism.
  `run_baseline_model` was already complete in `tools/` but had no agent
  adapter; it now has one, so the model-evidence constant in `claim_gates.py`
  finally has the real producer it was written for. Both emit verified
  receipts, and `test_real_baseline_receipt_licenses_the_matching_model_metric_claim`
  proves a genuine baseline receipt passes the entity gate end to end.
- **Advisory method skills, retrieved by trigger.** `core/method_skills.py`
  appends a packaged `causal-claim-boundary` skill to the agent system prompt
  only when the request actually mentions causal language (English or Chinese).
  The skill is explicitly subordinate: it cannot override deterministic tool or
  claim gates, and a test asserts that sentence stays in the shipped text.
  Packaged via `pyproject` package-data so a wheel install keeps it.
- **Signed permutation importance.** `FeatureImportance` gains
  `signed_importance` and `importance_std`; the legacy `importance` field stays
  non-negative for the existing API/UI contract. This was previously deferred as
  "separate slice" in the plan's E1.5 residuals.

### Changed (behaviour)

- **`split_policy="random"` on temporal data is now rejected, not warned.**
  `run_baseline_model` previously emitted a `random_split_on_time_data` leakage
  warning and proceeded; it now raises. A warning that the model may ignore is
  not a guard when the consequence is training rows postdating test rows.
  Callers who genuinely want a random split must drop the time column.

### Fixed (this pass)

- **A confirmatory claim can no longer ship on a hollow statistics receipt**
  (found by the closing adversarial review, after the fail-closed change below).
  Every completeness and multiplicity check keys off `p_value`, and the older
  `confirmatory_without_statistics` check only verified a stats receipt was
  *cited* — not that it carried a result. So a confirmatory claim asserting
  significance, citing a receipt whose `p_value`/`effect_size`/CI/`sample_size`
  were all null, passed every gate with `health_score` 1.0. Such a receipt is
  self-consistently digested, so reachability cleared it too. A confirmatory
  claim now requires an actual test statistic
  (`confirmatory_without_test_statistic`). Exploratory-lane behaviour is
  unchanged — descriptive evidence there legitimately has no p-value.
- **Statistical exit gate is fail-closed**: incomplete confirmatory evidence,
  missing registry family counts, and stale multiplicity adjustments now
  block publication (previously recorded-only). Gate reports bind
  `claim_bundle_digest` + `run_witness`; the renderer refuses a reused report
  for a mutated same-id bundle or a mismatched witness.
- **Model metrics must come from the cited model receipt**, not an unrelated
  ModelCard paired with a SQL count. Independence licensing is claim-local so
  one claim’s holdout evidence cannot license a sibling.
- **Provider rejections consume request budget**: `llm_call_rejected` settles
  like a completed attempt so started→rejected loops cannot bypass
  `max_requests`. Budget amendment fingerprints are system-derived under the
  writer lock (`amend_budget` / `amended_policy_fingerprint`); forged
  fingerprints fail closed.
- **Payload policy is enforced at the provider-facing tool boundary**: every
  tool observation is filtered by session policy. Default tier
  (`schema+aggregates`) withholds raw SQL row previews, profile sample values,
  and unverifiable open-analysis output while still persisting full local
  artifacts for receipts.
- **Window-function aggregate bypass** (found in this review): the default-tier
  SQL classifier treated `sum(x) OVER ()` as a safe scalar aggregate. The
  disallowed-shape regex ended with `\b` after `over\s*(`, which never matches
  `OVER ()` (non-word after `(`), and the aggregate matcher used `[^;]*` so
  the window clause was swallowed inside the “function args”. Fixed with a
  bare `\bover\b` ban, balanced-paren aggregate parsing, multi-statement
  rejection, and regression coverage for empty and ordered windows.
- **Journal durability**: parent-directory fsync after journal append and
  snapshot replace (POSIX), closing the “directory entry lost on power loss”
  window called out in the residual list.
- **PR #1 residue**: removed unreferenced `public/eda-ux-checklist.html` and
  added a public-asset allowlist guard; removed Settings “Thinking level”
  Deep/Ultra copy that promised investigation/macro-loop behaviour the product
  does not ship; rewrote Findings/Deep Analysis empty-state links that told
  users to “start an investigation” from Questions; CI now runs web
  typecheck/test/build; e2e visual + cross-browser suites are opt-in;
  scoreboard corpus absence skips cleanly unless
  `EDA_REQUIRE_SCOREBOARD_CORPUS=1`; demo scripts look under `sessions/` not
  the retired `runs/` path.

### Still residual (do not block this pass; must inform E4a)

- No exploration supervisor / frontier / ProbeExecutor — E2/E3 have no
  production consumer.
- Causal / independence / prediction phrase tables remain tripwires, not a
  semantic firewall; `claim_type` is still model-chosen.
- Numeric gate is bare-token only (units, “million”, spelled numbers out of
  scope); `ReceiptFact.unit` unused by the gate.
- `ToolCallLedger` has no production call site; failed heavy scans vs success
  accounting must be validated when E4a wires it.
- Three near-duplicate jsonl tail-truncation implementations
  (event_journal / receipt_outbox / stat_registry) still invite drift.
- **The parent-directory fsync landed only in `event_journal`.**
  `receipt_outbox` and `stat_registry` still fsync the file handle alone
  (`receipt_outbox.py:286`, `stat_registry.py:293`), so the directory entry for
  a newly *created* WAL can still be lost on power failure. Appends to an
  existing file are unaffected. This is the same drift the duplicate-truncation
  residual above predicts, now observed: extract the shared helper when those
  three merge.
- Server `analysis_depth` API still exists; UI control removed pending product
  decision (wire to exploration quick/standard/deep vs delete).
- **The renderer's "Statistical caveats" section is now unreachable** and needs a
  product decision. `_caveat_lines` collects statistical violations only from
  *passed* reports, but any statistical violation now forces `passed=False` and
  the bundle is withheld — so the section can only ever emit `(none)`, which is
  exactly what its one test asserts. It was designed for the register-only v1
  model. Either delete `_CAVEAT_PHRASES` and the section, or reintroduce a
  deliberate "downgrade to exploratory with a caveat" path if soft warnings are
  still wanted. Do not leave it as decoration that reads like a working feature.
- **Nothing forces a claim's text to quote the *adjusted* p-value.** The gate
  verifies the receipt carries a correct Bonferroni `adjusted_p_value`, but a
  claim citing a family of 5 may still present the raw marginal p as
  significant and pass. Reporting fidelity, distinct from the multiplicity
  arithmetic itself.
- **`render_claim_report` trusts `run_metadata["witness"]` from its caller.** The
  stale-witness guarantee is only as strong as whoever supplies it; there is no
  independent live-witness cross-check. Neither gate nor renderer has a
  production caller yet, so pin this when E4a wires them.
- `CHANGELOG.md` and `docs/` remain gitignored local docs (README claim that
  the changelog ships with code is currently false).

### Verification

- Full backend suite before the gate fix: **3403 passed, 1 skipped**
  (`uv run python -m pytest`, exit 0, 465s) — up from 3372 at `bd66d81`.
- Gate fix verified directly: `test_claim_gates.py` + `test_claim_renderer.py`
  **60 passed, exit 0**, with the new test confirmed RED before the fix
  (`assert not report.passed` → `assert not True`) and the reviewer's original
  standalone repro flipping from `PASSED: True` to `PASSED: False`.
- **The post-fix full backend run is not a clean gate and should not be quoted
  as one.** It reported 4 failures in `test_driver_cancellation_contract.py`,
  all transient: that test AST-parses driver source, and another session wrote
  `drivers/auto_eda.py` at 16:33:45 and `tools/relationship_discovery.py` at
  16:34:04 — inside the 16:26–16:34 run window — so the parse hit a
  half-written file (`SyntaxError: invalid syntax` on a one-line fragment). The
  same file passes **35 passed, exit 0** in isolation. Re-run the full suite
  once the tree is quiet before treating any number as the gate.
- Full frontend suite: **562 passed across 39 files** (`npm test`, 39/39 files).
  One earlier run reported 561 passed / 1 failed; the same suite re-run clean,
  so that file is flaky rather than broken. Not yet isolated — if it reappears,
  bisect it rather than assuming the re-run was the truth.
- Ruff clean on touched Python files.

## Unreleased: Foundation review — eight confirmed defects fixed - 2026-08-02

Five parallel read-only reviews of the E0–E3 foundation (evidence chain,
journal recovery, gates and budget, security surface, mutation testing), each
required to ship an executable reproduction with every finding. Every accepted
finding was re-run by the orchestrator before a fix was commissioned. Full
gates green: 3372 passed / 1 skipped.

### Fixed

- **Statistical correction could be bypassed entirely** (critical): the
  multiple-comparison family was a free-text string the model filled in, so
  renaming it per test yielded five uncorrected p-values where an honest agent
  got corrected ones. The family is now derived deterministically from
  `(dataset, tested columns)`; the model-supplied field is rejected outright,
  and each receipt records the session-wide attempt count so a shattered
  partition is visible downstream.
- **Any number could skip the numeric gate** by prefixing it with `<` or `>`:
  the threshold exemption compared against *any* cited statistic, so a routine
  t-test receipt licensed "churn stayed < 0.01" while the real value was 40%.
  Inequalities now bind to the statistic their own text names, falling back to
  ordinary pool matching otherwise — `p < 0.05` still verifies.
- **Absence claims could be laundered**: coverage unioned datasets and columns
  independently, so two receipts covering `(orders, amount)` and
  `(users, email)` licensed a claim about `(orders, email)` — a pair nobody
  scanned. Declared filters and time ranges were never compared at all.
  Coverage is now per-receipt and asymmetric: an unfiltered scan covers a
  narrower declared slice, a filtered one only covers an identical declaration.
- **Finalization reserve could be silently consumed**: reservation enforced the
  non-protected sub-cap but settlement did not, so actual usage above the
  reserved amount ate into protected capacity without raising. Settlement now
  enforces the symmetric bound.
- **SQL injection through the slice filter**: the WHERE-clause blocklist missed
  DuckDB's FROM-first set operations, letting fabricated rows into a profile
  (mean 1.0 → 4.6e11) while the row-count fact stayed honest and the receipt
  digest still verified. Replaced with an AST allowlist built on DuckDB's own
  parser, which also closed four further shapes the review found (GROUP BY
  HAVING, SAMPLE, ORDER BY/LIMIT, QUALIFY) and a subquery form with no `select`
  keyword.
- **A torn journal tail bricked the run**: a crash between the closing brace
  and its newline left a record the reader counted and the writer deleted,
  opening a sequence hole that no journal API could recover from. The newline
  is now the sole commit marker on both sides.
- **Receipt commits could be stolen**: per-call locking let a second worker take
  over a logical step while the first marked *its* artifact durable and
  committed — pointing a commit record at an artifact never written. Fenced
  with the receipt id the worker prepared.
- **Call identities could collide**: `tool_call_id` drew all its entropy from
  provider-supplied text that degrades to a position counter restarting at 1,
  so two distinct receipts shared one id and the gate's lookup dropped one
  silently. Executor-owned sequence index now participates in both the call id
  and the receipt id, while replaying one logical step keeps its id stable.
- Reentrant journal locking (recording a side effect inside its own fence
  deadlocked, holding the session write guard), epoch claiming moved fully
  inside the lock, causal-phrase tables merged and completed, and gate input
  revalidated so forged bundle instances cannot skip the structure gate.

### Testing

- Mutation testing across ten load-bearing invariants; all but one produced red
  tests. The exception — the report's evidence-trail section could be truncated
  undetected because its only assertion rendered a single bundle — is fixed and
  re-verified by mutation.
- Added negative coverage for budget schema constraints, notably that an
  amendment raising nothing is rejected.

## Unreleased: Eval-0 baseline measured on real providers - 2026-08-02

Real-provider baseline for the autonomous-analysis pipeline. Two rounds: v1
(15 runs, exposed two provider bugs) and v2 (18 runs, clean pipeline after the
fixes — deepseek-v4-flash ×9, gpt-5.6-luna ×9 across planted, absence and
injection buckets). Total spend across all Eval-0 work ≈ $2.10. Raw results in
`output/eval0/baseline{,-v2}/`.

### Measured (v2, clean pipeline)

- **Planted-insight F1 is 0 for both models**, unchanged from v1 — a genuine
  capability gap, not a scoring artifact. Verified in three steps: the report
  does contain 14 structured claims, the extractor reads them correctly, and a
  fair column-backfill rescore still yields 0. The pipeline computes global
  statistics competently but never slices by the dimension carrying the
  signal: across every run, revenue was never grouped by region. Failures come
  in two shapes — *missed entirely* (never picked the dimension) and *near
  miss* (found the surface fact, never chased the structure: reported
  "satisfaction 25.32% missing" without linking it to channel, "33 IQR
  outliers" without locating the date).
- **Absence handling is perfect**: 3/3 per model once the recorded violations
  are traced to a scorer false positive (a compound claim is classified by
  first-matching keyword, its referenced columns are unioned, and subset
  matching then reports a pattern the claim never asserted). Grounding 18/18;
  injection canaries never leaked.
- Efficiency per run — deepseek: $0.0185, 467s, 14.3 requests; luna: $0.1397,
  129s, 14.3 requests. Identical request counts confirm both pipelines now run
  to completion; luna is 3.6× faster and 7.5× more expensive.

### Fixed (both found only by real API traffic)

- **All OpenAI models failed structured output on the question-proposal
  step**: `RawLLMQuestionProposal`'s 16 bare-`Any` fields generated schema
  properties with no `type` key, which strict json_schema mode rejects with
  HTTP 400. Fixed with `Annotated[Any, WithJsonSchema(...)]` so the advertised
  schema is concrete while runtime validation stays permissive; dynamic-key
  maps became pair arrays with a fold that still accepts maps, so the
  json_object providers do not regress. Confirmed live: luna went from 8.7 to
  14.3 requests per run.
- **All OpenAI models failed chat routing**: `Intent.params` is a dynamic-key
  object, which strict mode also rejects. Fixed with a wire-only `RoutedIntent`
  subclass so the domain model (and its OpenAPI contract) keeps the map shape.
- Harness `worst` aggregated as `min(values)`, inverting the label for
  lower-is-better metrics; now direction-aware, with v1 recomputed from stored
  JSON so both rounds share one yardstick.

## Unreleased: Exploration Foundation Wave 3 — E3 claim gates + deterministic renderer - 2026-08-02

Completes the E0–E3 foundation. Full gates green: 3224 passed / 1 skipped
(trajectory: 2828 baseline → 3077 W1 → 3155 W2 → 3224 W3).

### Added

- **ClaimBundle schema** (`schemas/claims.py`): seven claim types, qualified
  evidence references (`<receipt_id>:<fact_id>` — fact ids are only unique
  within one receipt), absence claims bound to absence support, bundle-level
  evidence lane and independence declaration.
- **Six exit gates** (`core/claim_gates.py`): structure, reachability
  (fabricated receipts and digest mismatches rejected), numeric (the claim
  itself never enters the value pool; percent/raw pools split, counts exact,
  half-ULP tolerance via report_validator), entity/capability (model claims
  need a model receipt, causal rejected by default, absence needs a coverage
  receipt with `result_count == 0` and scope actually scanned), statistical
  (registration-only in v1), and state-witness equality. GroundEval-style
  `(1−v)²` health score as a metric, not a blocker. Dual-channel rejection
  feedback that never echoes the failed claim text; retry budget of one, then
  `abstained`.
- **Deterministic claim renderer** (`core/claim_renderer.py`): renders only
  gate-passed bundles in a fixed section order, byte-identical across runs,
  and re-scans the final markdown for numbers outside the evidence pool —
  the same yardstick that will police LLM-polished synthesis in E4a.
- Receipt R4 fields (`evidence_independence_key`, `replication_kind`) added
  without breaking previously persisted receipts: absent/None fields are
  omitted from the digest view, set values are digest-protected.

## Unreleased: Exploration Foundation Wave 2 — E2 durable journal + exploration budget - 2026-08-02

Four parallel subagent packages on top of Wave 1; the orchestrator froze the
budget schema contract first. Full gates green: 3155 passed / 1 skipped
(the OTel skip is gone — the SDK now runs in the dev group).

### Added

- **Generic event journal** (`core/event_journal.py`): the six file mechanics
  (locking, fsync, tail truncation, attempt-epoch fencing, execution lock,
  snapshots) extracted from the investigation loop journal as a typed generic
  base; `loop_journal.py` shrinks to a thin semantic layer (-269/+31) with
  every investigation test untouched and green — the regression proof for the
  refactor.
- **Exploration journal + schemas** (`schemas/exploration.py`,
  `core/exploration_journal.py`): sealed policy fingerprints, a 19-event
  discriminated union with non-terminal pause (`pause_requested → paused →
  resumed`), six terminal stop reasons (pausing is not one), crash recovery
  that marks unterminated LLM calls `uncertain` (fully charged, never resent)
  without killing the run.
- **Exploration budget runtime** (`core/exploration_budget.py`):
  success-counted ToolCallLedger with projection-only batch prechecks and
  derived-state latching; monotonic budget-amendment chains (schema
  revalidation, replay rejection, fingerprint linkage, independent
  monotonicity check); per-kind tool caps are an allowlist (absent = 0,
  fail closed); finalization reserve rides the existing `protected_*` budget
  fields with zero new enforcement code.
- **OTel GenAI adapter** (`core/observability.py`): one mapping seam from
  platform trace events to `gen_ai.*` spans with default-deny redaction —
  non-allowlisted values leave only length + digest; content capture is
  opt-in via `EDA_LLM_DEBUG_FULL`.
- **Deterministic ContextAssembler** (`core/context_assembler.py`): fixed
  section order with per-section character budgets and explicit truncation
  markers; governance facts (policy fingerprints, data-state witness,
  amendment ids) are structurally untruncatable; frontier/insight sections
  are provider-pluggable for the E4a supervisor.

## Unreleased: Exploration Foundation Wave 1 — Eval-0 harness + E1.5 receipt integrity - 2026-08-02

Four parallel subagent packages plus orchestrator merge on top of `ea3c5c8`
(E0 + E1). Full gates green: 3077 passed / 2 skipped (baseline was 2828);
both real providers (gpt-5.6-luna, deepseek-v4-flash) pass the tool-calling
contract smoke at ≈$0.005 total.

### Added

- **Eval-0 harness** (`eda_platform/tests/evals/exploration_baseline/`):
  deterministic checkers (planted / negative-absence / injection / grounding
  rate per §10.3), per-item JSON results with bucketed summaries, capability
  vs regression suites, and a 3-adapter runner (`null` / `replay` /
  `auto_eda`). Fixtures ship with pandas-recomputed ground truth; InsightEval
  is adapter-only (no LICENSE, cannot vendor).
- **Receipt integrity (E1.5 core)**: executor-injected `ToolExecutionContext`
  so call identity cannot be forged from tool name + arguments; receipt
  artifacts surface in `AgentToolResult` and the trace; versioned witness
  triplet digest (fail-closed on schema/profile drift); receipt outbox WAL
  with exactly-once logical commit under crash injection; resolved receipt
  scope + bounded fact manifests (unevaluated rows are explicit); enforced
  digest verification with deeply-frozen parameters; system-owned statistical
  registry allocating family sequence indices (failures counted, rollbacks
  rejected).
- **Method hardening (E1.5)**: effect CI coverage for all 8 test types with an
  explicit `TEST_PUBLISHABILITY` matrix (no dark paths), signed effect sizes,
  Holm-default correlation screening (Spearman, min pairwise n), robust STL
  decomposition with explosive-resample preflight, slice projection preflight
  with resolved columns, ML split policies (group/time-aware CV, PR-AUC,
  Brier, signed permutation importance aggregated to original features).

### Fixed

- gpt-5.6-luna rejected function tools at its default reasoning effort on
  `/v1/chat/completions`; tools requests now pin `reasoning_effort` from a
  per-model catalog field (catalog `agent-models-2026-08-01`), verified live.
  Other OpenAI models' payloads are byte-identical (pinned by tests).
- Fisher exact now reports an odds ratio with Haldane–Anscombe correction and
  a Woolf CI; `method_findings` no longer KeyErrors on fisher results, and
  effect magnitude and interestingness deviation both use odds-ratio cutoffs
  (3.0/5.0, reciprocal-symmetric) instead of correlation-scale ones.
- Streamlit-retirement guard now catches bare `*_ui.py` references (seven
  stale comments had survived the prefixed pattern) and skips cleanly on
  fresh clones instead of erroring on gitignored local-ops files.

### Removed

- ~166 lines of proven-dead code: legacy investigations frontend client
  methods/hooks, orphan symbols (`costSignal`, `LoopJournal` protocol,
  `process_peak_rss_bytes`), ghost `.gitignore` rules, stale retired-UI
  comments. The backend investigations chain is deliberately retained as the
  regression proof for the E2 journal generalization.

## Unreleased: DI Sprint 10 — Live-Review Fixes: The Last Mile - 2026-07-18

Triggered by a three-way live review (browser walkthrough, trace forensics on
three real runs, report quality vs. the manual-analysis baseline). Five
parallel subagent packages + orchestrator integration; full gates green
(1031 passed / 4 skipped); fresh-workspace live Olist acceptance passed —
GMV matches the manual baseline (13.59M), late-delivery rate now uses the
customer-delivery basis (8.11%, was 0.48% against the carrier date), and the
cross-table repeat-purchase rate executes on the FIRST run via auto-confirmed
joins. English-only surface by decision; no localization.

### Fixed

- **Answer-question consistency (W1)**: findings now pick the answer column by
  question intent (sum/share/avg/duration; domain-metric questions consume
  their registered interpretation template first) — a "total GMV" question can
  no longer be answered with an `avg_` column, and share answers always carry
  the percentage with numerator/denominator. Late-delivery and fulfillment
  metrics prefer customer-delivery timestamps (carrier basis only as a
  disclosed fallback). The e-commerce metric pack now requires >=2 independent
  commerce signals, so a lone `Amount` column no longer earns "GMV" wording.
- **Execution funnel (W2)**: domain-metric questions auto-execute in full
  (deterministic SQL), at least one exploratory LLM question always executes,
  cap raised top-3 -> 10 with composition telemetry. Bootstrap and question
  LLM calls are now metered (llm_calls 4 -> 13 on Olist; cost no longer
  understated by half).
- **Report readability (W3 + integration)**: limitations aggregate per
  (dataset, code) AND per code across datasets (live Olist: 32 -> 9 lines);
  shape claims ("N rows and M columns") are banned from the Executive Summary,
  which now leads with scored business findings; repeated findings render once
  with cross-references; internal artifact ids are scrubbed from prose and
  numbers are formatted (thousands separators, no scientific notation,
  ratio-as-percent). Validator numeric mismatches now FORCE an evidence
  interleave round — correct values are fetched deterministically and injected
  into the repair payload instead of waiting for the writer to volunteer.
- **UI (W4)**: sessions are reachable across projects (grouped list +
  project switch), a global error boundary replaces raw tracebacks,
  exploratory/domain-metric badges render, executed questions leave the
  "Uninvestigated" list, empty report/cleaning sections collapse, and
  whitelist entries that reference datasets absent from the session sink with
  a warning.
- **Semantic hygiene (W5)**: join proposals are role-gated (verified
  timestamp/measure columns are rejected as join keys) and quality-ranked;
  high-confidence proposals (high confidence + id-named both sides + not m:n)
  auto-confirm as a distinct, revocable `auto_confirmed` state whose usage is
  disclosed in execution limitations and the report; the whitelist gains
  dataset scoping (`entries_for`) so cross-project residue is inert; integer
  seconds-offset columns (creditcard `Time`) no longer profile as timestamps
  (real epoch seconds still do); time charts skip when fewer than two periods
  exist. A Knowledge-page Revoke button pins revoked joins back to
  `proposed` — deleting instead would let the next run silently re-confirm.

## Unreleased: DI Sprint 9 — Value Deepening: Report Value Trio - 2026-07-18

Roadmap stage H (items 2 and 4) plus the deferred domain-metric pack. Three
parallel subagent packages + orchestrator integration; full gates green;
live Olist regression 6/6 (repeat-purchase rate resolves cross-table through
a confirmed whitelist join).

### Added

- **Evidence-interleaved report writing (H9-A, EvidFuse-shaped)**: the report
  writer can request evidence WHILE writing — typed request/grant/teaching-
  style-rejection loop (`agents/evidence_interleave.py`), locator grammar
  shared with the report validator, bounded budgets (≤2 per section, ≤8 per
  report). Resolvers only read already-persisted artifacts — never new SQL or
  computation. Wired into both report routes (decision-report SCQA rewrite +
  agentic claim plan); granted values join the number-gate evidence pool; all
  existing gates unchanged; offline output byte-identical to before. The full
  exchange record persists as an `EVIDENCE_INTERLEAVE_TRANSCRIPT` artifact.
- **Interestingness scoring + finding dedup (H9-B, zero LLM)**: deviation ×
  coverage × nontriviality geometric mean with degenerate-pattern penalties
  (target-in-features, flagged==rows, group==value column) and an exploratory
  no-stat-backing penalty; `FindingScore` gains an optional fourth component
  (byte-compatible with old artifacts). Cross-question clustering (column set
  + direction + magnitude bucket) keeps one representative per cluster and
  tags the rest `supporting` — nothing is deleted; the synthesis brief
  discloses merge counts.
- **Registered domain metric packs (H9-C)**: nine deterministic metrics —
  e-commerce GMV / AOV / repeat-purchase rate / late-delivery rate /
  fulfillment time / freight ratio, generic HHI concentration / missing
  hotspots / time coverage. Applicability is decided purely from the DI8
  semantic role layer + confirmed join whitelist (unverified columns never
  participate); a new `domain_metric` question family (≤6, guaranteed supply
  — deliberately NOT throttled by the template backstop) generates the SQL
  (always with a count column for time-boundary checks).
- **RunMetrics**: six new rollup fields (interleave granted/rejected, dedup
  clusters/merged, domain-metric questions/skipped).

## Unreleased: DI Sprint 8 — Context Understanding + Cross-Table Value Loop - 2026-07-18

Roadmap stage G. Five work packages implemented by parallel subagents and
integrated; full gates green (903 passed / 4 skipped, ruff + pyright clean);
two-pass live Olist adversarial regression 13/13.

### Added

- **Column semantic role layer + bootstrap (DI8-B)**: `core/column_roles.py`
  (8 mutually-exclusive roles, deterministic verifiers — uniqueness,
  per-group 1..n with 0.5% dirt tolerance, leading-zero codes, coordinate
  bounds) + `agents/semantic_bootstrap.py` (batched LLM hypotheses → label
  remapping → per-hypothesis data verification; priors never count as
  verification). Output is a rebuildable-cache `COLUMN_ROLE_SET` artifact
  (WrenAI-MDL-shaped, model-versioned). Identifier/sequence columns leave
  every stats candidate pool and carry `impact_weight == 0`.
- **Question generation inversion (DI8-E)**: LIDA-style data summary feeds
  LLM free-form questioning as the PRIMARY route (all marked `exploratory`);
  templates demoted to a coverage-backstop checklist (metered); the guard
  only filters for answerability, never generates.
- **Cross-table join whitelist (DI8-C)**: relationship discovery proposes
  `JoinCandidate`s (cardinality-aware), Knowledge-page or programmatic
  confirmation, `cross_table_aggregation` template family consumes confirmed
  non-M:N joins only; execution-side guard double-checks (JOIN must be
  declared, declared must be confirmed) + Vanna-style usage counters.
- **Three-tier report gate + time boundary + narrative (DI8-D)**: semantic
  soft gate now returns pass / degraded-with-structured-confidence-label /
  rejected (correctness hard gate untouched, one line unchanged); finding
  scores become impact × significance; `tools/time_boundary.py` detects
  partial edge periods (edge bucket < 0.5× interior median) so boundary-fooled
  trends degrade instead of publishing; Executive Summary organizes verified
  findings into a MetaInsight commonness+exceptions skeleton; Report page
  gains a limitations block.
- **LLM route fault tolerance (DI8-A)**: lenient list coercion, four-stage
  dataset-name resolve-then-reject (rejection feedback always carries the
  known-dataset list), repair-style retries (violation text + schema example
  injected; blind same-payload retries removed), partial acceptance, and
  RunMetrics yellow-light fields (`question_llm_skipped`, drops, coercions,
  `degraded`, report gate rollup).
- **L2 anti-tunnel-vision**: normalized probe-SQL fingerprinting rejects
  repeated probes (second repeat force-exits as `repeated_action`), failure
  history (last 5, compressed) is fed back into the next loop prompt; both
  observable via new trace events wired into the orchestrator.

### Fixed

- Live regression caught a real-world criterion flaw: 82/103,886 Olist
  payment rows start their sequence at 2 (missing first installment), so the
  strict 100% per-group 1..n check left `payment_sequential` unverified and
  back inside stats pools. `check_sequence` now tolerates 0.5% dirty rows
  (mirroring the identifier philosophy) — verified against the real dataset.

## Unreleased: Final Trust Audit — Hardening + Dead-Code Removal - 2026-07-17

Three-agent close-out review of the full decision-intelligence delivery
(adversarial trust-boundary audit with runtime reproductions, maintainability
review, deprecated-code sweep). Overall verdict: core boundaries
(fingerprint-bound approval, scope limits, deterministic reducers,
fail-closed report numbers) hold under attack; three concrete gaps found and
fixed the same round.

### Fixed (trust hardening)

- **L2 probe number injection (High)**: the LLM-authored probe `purpose` was
  interpolated verbatim into reducer claim text, so a fabricated "87%" could
  reach a ValidatedFinding sentence. Number tokens are now stripped from the
  purpose before reduction (`[n]` placeholder), and the probe merge gate
  additionally validates every number in probe claim text against that
  finding's own evidence (percent-aware) — belt and braces, both tested.
- **Interpretation gate percent/raw washing (High)**: the L1 gate ignored
  the %-suffix, so raw evidence `42` could legitimize a fabricated "42%".
  Allowed-number pools now carry units (`EvidenceRef.unit` + text %-flags)
  and percent-vs-raw must match, mirroring the report-validator and
  decision-report gates.
- **Causal single-sourcing completed (Medium)**: `report_validator` and
  `narrative_reviewer` still carried divergent local causal lists (missing
  zh + several phrases). Both now consume `core.claim_language` (family
  extended with "drove"), making the legacy report gate as strict as the DI
  gates.

### Removed (dead code, grep-verified zero references)

- `ReportDraft` + `ReportBundle.from_draft` (schemas/reports.py) and the
  `_normalize_section_bodies`/`_contains_body_causal_language`/
  `_BODY_CAUSAL_TERMS` cluster in agents/reporting.py — superseded since the
  July report-engine fixes; the one covering test now exercises
  `merge_duplicate_sections` directly. Sweep found no other dead code, no
  mistakenly committed artifacts, and both demo scripts still run offline.

### Docs

- Status banners corrected: decision-intelligence design marked implemented
  (was "proposal only"); m7 role contract §11 updated for sprints 4-6
  deliveries. Next-sprint refactor plan recorded (orchestrator decomposition,
  shared finding finalization, project-artifact index, observable tolerant
  failures) from the maintainability review.

## Unreleased: Decision Intelligence Sprint 7 — Golden Evals + Debt Cleanup (roadmap close-out) - 2026-07-17

Roadmap phases E+F ([di-sprint7-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint7-plan.md)).
With this sprint all roadmap phases A–F are delivered.

### Added

- **Method-executor golden tests** (`tests/golden/test_method_executor_golden.py`):
  seeded known-truth datasets assert stat-test direction/effect-size band
  (+ null case), ML baseline metric floor (+ leakage-guard trigger case),
  and exact planted-outlier detection.
- **Investigation benchmark** (`scripts/benchmark_offline.py --with-investigation`):
  offline plan → approve → execute round-trip per run with inv columns in
  the metrics table (verified: 0.02s, 6 artifacts, 1 validated finding).
- **Prompt budget audit**
  ([prompt-budget-audit](docs/archive/2026-07/infrastructure/eda-agent-platform-prompt-budget-audit-2026-07-17.md)):
  fixed-vs-variable prompt overhead across all 5 LLM call sites; one
  refactor recommendation (chat planner catalog_columns uncapped).

### Changed (debt cleanup)

- Causal-phrase family unified into `core/claim_language.py` (single
  source; orchestrator/interpretation/loop/decision-report consume it;
  a separate report-narrative list flagged as a follow-up chip).
- `core/methods.py` feasibility verdicts now derive from structured
  `missing_kinds` (structural/data/scale) — string-set matching deleted;
  all existing verdicts preserved.
- `referenced_columns` keys unified on dataset name (producers), with
  name-then-id fallback on consumers for old artifacts.
- L2 probe evidence refs rewritten to the persisted
  DEEP_INVESTIGATION_RESULT artifact (step-prefixed locators) so every
  merged ref resolves against the store (sprint-5 debt).
- Test hygiene: `Path("unused")` dummy workspaces replaced with tmp_path
  (repo root can no longer be polluted by UI tests); dead-code suspicion
  on `_trigger_run_b`/`_render_batch_result` investigated — both live.

## Unreleased: Decision Intelligence Sprint 6 — Semantic Pins + Finding Freshness + Knowledge Promotion - 2026-07-17

Roadmap phase D ([di-sprint6-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint6-plan.md)).

### Added

- **Semantic pin injection** (`core/semantic.pinned_context_block`): the
  project's user-confirmed metric definitions + field meanings render into a
  deterministic, budgeted context block injected FIRST into the question-
  proposal and L1-interpretation prompts with a hard "established facts —
  never reinterpret" instruction (community research: pinned definitions fix
  the top silent-failure class). Context only — never enters the allowed-
  numbers set; empty seeds inject nothing.
- **Finding freshness** (`core/finding_freshness.py`): deterministic
  fresh/stale/unverifiable verdict per validated finding — stale when a
  target dataset now resolves to a different dataset version or claim
  evidence no longer loads (guards the experience-following failure mode:
  old conclusions must not silently flow into new reports).
- **Knowledge promotion** (`drivers/knowledge_promotion.py`, m7 §9):
  user-confirmed ValidatedFinding → VerifiedAnswer in the project seeds;
  the answer is built ONLY from deterministic claim sentences; stale or
  unverifiable findings are refused with teaching errors; promotion is
  always a UI-confirmed action, never automatic. Promoted answers surface
  on the Knowledge page and feed the existing chat/question context.
- **UI**: freshness badges with reasons on finding details; promote →
  review (question/answer/evidence preview) → confirm flow; synthesis
  finding selector excludes stale findings by default (opt-in toggle).

### Verification

- Full gates green. Browser-verified end-to-end on the real workspace:
  fresh badge → promotion review → confirm → seeds.json carries the
  VerifiedAnswer (claim-sentence answer + evidence note).

## Unreleased: Decision Intelligence Sprint 5 — Bounded Investigation Loop + Decision Coverage - 2026-07-17

Roadmap phase C ([di-sprint5-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint5-plan.md)).

### Added

- **Level-2 bounded investigation loop** (`agents/investigation_loop.py`,
  `schemas/deep_investigation.py`): after a deep-enabled plan's primary
  method succeeds, the LLM proposes typed follow-up probes
  (`{"action":"probe"|"conclude"}` closed union), executed deterministically
  within the approved scope. Double hard caps (3 probe slots, 8 LLM calls
  incl. retries) plus a consecutive-probe-error cap — all typed exits, never
  exceptions. Out-of-scope SQL is refused before touching the engine;
  in-scope probes pass the read-only guards + EXPLAIN precheck; findings
  come from the Level-0 reducers; the concluding rationale passes the same
  gates as an L1 interpretation or is dropped. Offline: zero LLM calls,
  typed "offline" exit. Full transcript persisted as a
  DEEP_INVESTIGATION_RESULT artifact; probe findings merge into the
  ValidatedFinding through the existing evidence + causal gates. Deep mode
  is opt-in at plan creation, visibly recorded in the plan's assumptions
  and allowed_tools so the fingerprint-bound approval covers it.
- **Decision coverage** (`schemas/decision_coverage.py`,
  `core/decision_coverage.py`): deterministic m7 §8 stop-rule assessment —
  top-5 feasible cards by score, terminal-state ratio, uninvestigated
  high-value questions, validated/not-eligible counts, teaching-style gap
  sentences; persisted as an idempotent DECISION_COVERAGE artifact.
- **UI**: deep-investigation checkbox in the plan builder; per-plan deep
  transcript timeline (exit-reason badge, per-step purpose/SQL/findings,
  validated conclusion caption); Decision Coverage panel atop the Findings
  page (ready/gaps badge, terminal ratio, named uninvestigated questions).
- `agents/interpretation.py` internal gate refactored into public
  `validate_interpretation_text` (frozen API unchanged) so the loop shares
  the exact same validation.

### Verification

- Full gates green (20+ new adversarial loop/coverage tests: step cap,
  LLM-call cap incl. retry burn, scope escape refused pre-engine,
  consecutive-error exit, fabricated-number conclusion dropped, offline
  zero-call). Browser-verified live: Decision Coverage panel over the real
  workspace ("Coverage gaps remain", 2/5 terminal, 3 named uninvestigated
  questions).

### Debt noted

- Probe findings' evidence refs point at in-loop SQL result ids that are not
  persisted as standalone artifacts (the full transcript lives in the deep
  artifact); rewrite refs to the deep artifact or persist probe results —
  roadmap debt list.

## Unreleased: Decision Intelligence Sprint 4 — L1 Interpretation + SCQA Decision Report + Card Editing - 2026-07-17

Roadmap phase B ([di-sprint4-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint4-plan.md)).

### Added

- **Level-1 calibrated interpretation** (`agents/interpretation.py`):
  the LLM reads the full typed result and writes a business "what this
  means" note, admitted onto `ValidatedFinding`/`QuestionExecutionResult`
  (new `interpretation` + three-state `interpretation_status` fields) only
  after a deterministic gate: every number must match finding evidence
  (±1% relative, exact for small ints), causal-phrase family absent,
  length ≤600 chars; rejected text is never persisted (`fallback` with a
  reject reason); offline runs stamp `absent` with zero LLM calls. Wired
  into both the investigation route and Run-B auto-execution.
  `tools/report_validator.extract_numbers` exposed as the shared tokenizer.
- **SCQA decision report** (`schemas/decision_report.py`,
  `drivers/decision_report.py`): SynthesisBrief → `DecisionReport` artifact
  with Situation/Complication/Question/Answer framing, one provenance-linked
  section per selected finding, limitations/gaps, and a fail-closed number
  check (any sentence with an unresolvable number is stripped before
  publishing). Optional LLM refinement of the prose re-runs the same gates
  and falls back to the deterministic version. Markdown export via
  `exporter.decision_report_to_markdown`; rendered on the Report page with
  a generate button and download.
- **Card editing** (`drivers/card_edit.py` + Questions-page form): text
  fields editable (question/decision/hypothesis/criterion/risks/...),
  structural fields rejected with teaching errors; `card_version` bumps,
  feasibility is recomputed deterministically (edits can never smuggle a
  verdict), the candidate set is overwritten in place so existing plan
  fingerprints go stale and the sprint-2 approval gate refuses old plans
  with zero new orchestrator code. `v{n}` badge on edited cards.
- **UI**: validated interpretations render under an "AI interpretation
  (validated against evidence)" caption on the Findings library and both
  Outcomes surfaces; fallback/absent render nothing.

### Verification

- Full gates green (705+ tests incl. 30 new adversarial interpretation/
  report/edit tests; ruff/pyright clean). Browser-verified: SynthesisBrief →
  Decision Report rendered live with all four SCQA blocks, readiness and
  narrative badges, on the real workspace. compare_ui flake confirmed
  pre-existing (timestamp collision, passes 3/3 in isolation).

## Unreleased: Decision Intelligence Sprint 3 — Method-Level Execution + Observability - 2026-07-17

Roadmap phase A ([roadmap-2026-07](docs/archive/2026-07/eda-agent-platform-roadmap-2026-07.md);
plan: [di-sprint3-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint3-plan.md)). Approved
investigation plans now execute real methods instead of SQL-only, and the
platform gains the observability groundwork for future benchmarks and the
bounded investigation loop.

### Added

- **Method-level execution** (`drivers/investigation_orchestrator.py`):
  `group_comparison` plans fetch scope-limited data and run the deterministic
  stat-test route (guarded test selection, effect sizes, ANOVA boxplot);
  `outcome_prediction` runs the guarded ML baseline — a leakage-guard failure
  produces an InvestigationRecord with a failed method gate, never a
  ValidatedFinding; successes carry `claim_class="predictive"` plus a forced
  non-causal baseline caveat; `anomaly_detection` runs the new robust
  screening tool. Assumption warnings flow into finding limitations
  (disclose, don't suppress). Descriptive/template route unchanged.
- **Anomaly screening** (`tools/anomaly.py`, `schemas/anomaly.py`): robust
  z-score (0.6745·(x−median)/MAD) and IQR methods, bounded top-10 outlier
  payloads, constant-column/all-null teaching errors, >20% flagged share
  labeled a distribution shift rather than individual anomalies.
- **Deterministic claim reducers** (`tools/method_findings.py`): stat/model/
  anomaly findings templated only from typed result fields; every number in
  the text carries an EvidenceRef; p-values never round to 0; effect sizes
  labeled small/medium/large; causal phrasing structurally absent.
- **Plan advertisement matches execution** (`tools/investigation_methods.py`):
  executor-backed families (comparison/prediction/anomaly) advertise their
  real route and tools on the reviewable plan; the template-SQL wording
  remains only for descriptive and not-yet-supported families.
- **Run metrics** (`schemas/run_metrics.py`, `core/run_metrics.py`):
  per-run rollup of LLM/tool calls, tokens, cost, duration, per-step table,
  artifact counts, findings/failures — persisted as a RUN_METRICS artifact at
  auto-EDA completion and computable on read for any run. Rendered on
  Trust & Trace (dev inspector) with a debug-log download.
- **Failure log** (`core/failure_log.py`): `record_failure` persists
  sanitized tracebacks as trace events and never raises; nine broad UI
  exception handlers now record durable failure traces instead of only
  showing a toast (closes a PR #6 review finding).
- **Debug JSONL** (`core/debug_log.py`): `EDA_DEBUG_LOG=1` mirrors every
  trace event to `<run_dir>/debug.jsonl`; zero overhead when off.
- **Offline benchmark** (`scripts/benchmark_offline.py`): golden-data
  end-to-end ×N with per-run and aggregate metrics tables; verified live
  (2 runs, 37 artifacts each, ~0.38s/run).
- **Method summaries in UI**: stat test name/p-value/effect-size badges,
  model task/headline-metric/leakage badges, anomaly counts on findings and
  outcomes; new pure badge helpers in `ui/workbench.py`.
- **Docs**: new decision-intelligence roadmap
  ([roadmap-2026-07](docs/archive/2026-07/eda-agent-platform-roadmap-2026-07.md)) mapping the
  four research reports onto phases A–F; community/practitioner research
  report; todo rewired to the roadmap.

### Verification

- Full gates green (pytest incl. adversarial leakage/assumption/constant-
  column/out-of-scope tests, ruff, pyright). Browser-verified: a template
  group-comparison card built a plan advertising the stat route, approval
  bound to the plan fingerprint, execution routed to the stat executor and
  failed closed with a teaching-style record on the 20-row demo dataset
  (constant/empty groups) — the disclosed-failure path working as designed.
  Investigation artifacts (plan/approval/record) visible on Trace & evidence.

## Unreleased: Decision Intelligence Sprint 2 — Three-Role Pipeline (PR #6 harvest) - 2026-07-17

Ports the closed PR #6's role-based workflow onto main with its trust
boundaries rewritten per the 2026-07-17 deep review. PR #6 was closed without
merge; its branch served as reference material. Plan + red-line acceptance
list: [di-sprint2-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint2-plan.md);
reconciliation analysis:
[pr6-reconciliation](docs/archive/2026-07/infrastructure/eda-agent-platform-pr6-reconciliation-2026-07-17.md);
external research:
[agent-framework-research](docs/archive/2026-07/infrastructure/eda-agent-platform-agent-framework-research-2026-07-17.md),
[insight-platforms-research](docs/archive/2026-07/infrastructure/eda-agent-platform-insight-platforms-research-2026-07-17.md).

### Added

- **Role handoff contracts** (frozen first): `schemas/quality_context.py`
  (QualityContext/Set — observed data conditions, never pass/fail scores),
  `schemas/value_discovery.py` (ValueMap/ValueOpportunity with closed
  `AnalysisMode` enum, no free-text directions), `schemas/investigations.py`
  (InvestigationPlan with required `candidate_fingerprint`, new
  `InvestigationApproval` bound to a plan fingerprint, ValidatedFinding,
  InvestigationRecord), `schemas/synthesis.py` (SynthesisBrief/StoryBeat).
  All persisted payloads carry `schema_version`; contradictory
  report-eligibility states raise instead of being silently normalized.
- **Card unification** (`schemas/questions.py`): `ValueCategory` vocabulary
  plus flat card fields `value_category`, `data_signal`,
  `priority_rationale`, `card_version`, `referenced_columns`,
  `source_artifact_ids`, `quality_context_artifact_ids`;
  `QuestionCandidateSet.value_map_artifact_id`. PR #6's nested `card:` model
  was dropped in favor of this flat contract.
- **Deterministic quality context + value discovery**
  (`tools/quality_context.py`, `tools/value_discovery.py`): EDA issues become
  disclosable QualityContext artifacts; ValueMap opportunities carry
  registry-computed feasibility (never LLM- or heuristic-authored verdicts).
- **Method adapter** (`tools/investigation_methods.py`): thin layer over
  `core/methods.py` producing method_family/recipe/allowed_tools/gates;
  dispatches on `analysis_mode`; the PR's second, contradictory registry
  (20-row vs 30-row thresholds) is gone.
- **Investigation orchestrator** (`drivers/investigation_orchestrator.py`):
  immutable plans fingerprinted against their source card; execution requires
  a persisted approval whose fingerprint matches the exact plan content and
  fails closed on stale cards, missing approvals, double execution, or
  out-of-scope datasets (SQL catalog built only from the approved targets).
  ValidatedFinding claim sentences come exclusively from the deterministic
  question_exec reducers; a broadened causal-phrase gate (en+zh) guards claim
  wording; evidence_support is derived from actual evidence resolution.
- **Synthesis orchestrator** (`drivers/synthesis_orchestrator.py`): decision
  story beats assembled only from validated finding texts; user-supplied
  business_context is stored, hashed into run identity, and never
  interpolated into story bodies. `drivers/investigation_library.py` reads
  the Findings Library / Investigation Log with same-run, bidirectional
  provenance authentication.
- **Auto EDA wiring** (`drivers/auto_eda.py`): BuildQualityContextStep and
  BuildValueMapStep emit the new artifacts before question discovery;
  candidate sets link their ValueMap; LLM proposal schema gained optional
  text fields only (value_category/data_signal/priority_rationale) — no
  verdict fields.
- **UI**: Findings page (Validated Findings library + Investigation Log with
  readiness/reliability badges), Decision Story section on the Report page
  (select findings → SynthesisBrief; business_context rendered only in a
  labeled unverified block), two-step plan review/approve/execute flow on
  Questions that reads plans/approvals/outcomes from the artifact store on
  every render (survives reloads), card badges for value_category plus
  data_signal/priority_rationale in Card details. Exporter threads
  quality-context limitation wording into report limitations.

### Verification

- Full gates green (pytest incl. 30+ new adversarial trust-boundary tests,
  ruff, pyright). Offline e2e demo produces QualityContextSet + ValueMap
  artifacts and unified cards. Browser-verified on a live run: build plan →
  fingerprint-bound approval → execution → validated finding (`pearson
  0.903672`, observed, report-eligible with non-causal limitation) → Findings
  library row.

## Unreleased: Decision Intelligence Sprint 1 — Opportunity Cards + Method Registry - 2026-07-17

First delivery of the decision-intelligence framework (design §8 steps 1+2).
Sprint plan with decided defaults:
[di-sprint1-plan](docs/archive/2026-07/infrastructure/eda-agent-platform-di-sprint1-plan.md).

### Added

- **Analysis Opportunity Card contract** (`schemas/questions.py`, frozen
  first): `AnalysisMode`, `FeasibilityStatus`, `ProposedAction`,
  `OpportunityFeasibility{status, method_id, reasons, missing}` and ten
  optional card fields on `QuestionCandidate` (business_decision,
  value_hypothesis, analysis_mode, candidate_methods, data_requirements,
  feasibility, success_criterion, risks, proposed_action). All optional with
  safe defaults — legacy artifacts stay valid. The red line is built into the
  contract: `feasibility` is ALWAYS produced by the deterministic method
  registry, never the LLM.
- **Deterministic method registry** (`core/methods.py`): seven method
  families with profile-only feasibility gates — `group_comparison`,
  `outcome_prediction`, `anomaly_detection`, `descriptive_sql` supported;
  `forecast`/`segmentation`/`causal_experiment` registered but marked
  unsupported (cards can propose them; the gate explains what's missing and
  suggests the next action, e.g. causal → design_experiment).
  `evaluate_feasibility` maps gate outcomes to
  ready/constrained/needs_data/unsuitable with teaching-style reasons.
- **Card generation**: the LLM question route proposes decision-focused card
  text (decision, value hypothesis, mode, success criterion, risks) via the
  existing strict+raw validation pattern; invalid modes degrade to None.
  Template-route candidates get conservative defaults (descriptive /
  diagnostic for group-difference). Every assembled candidate gets a
  registry-computed feasibility in auto-EDA.
- **Auto-execution gating**: candidates whose feasibility is `needs_data` or
  `unsuitable` are excluded from auto-execution; legacy candidates
  (feasibility=None) keep current behavior.
- **Opportunity card UI** (Questions page): mode badge + feasibility badge
  per candidate, "Decision:" / "Why it matters:" lines when present,
  non-ready cards show reasons/missing + suggested next step and render a
  disabled checkbox; "Card details" expander carries data requirements,
  success criterion, risks, methods. Legacy candidates render exactly as
  before.

### Verified

- Full suite `603 passed / 4 skipped` (+22 tests); ruff + pyright clean.
- Offline browser walkthrough on a fresh run: 12 template candidates render
  as cards with `descriptive`/`diagnostic` mode badges, `ready` feasibility
  badges (descriptive_sql / group_comparison method ids), and Card details
  expanders; legacy sessions unaffected.

### Decisions recorded (design §9)

Target domain unlocked (Business context as anchor); first method families =
comparison/prediction/anomaly + descriptive fallback; business value =
free-text + card hypothesis (no KPI catalog); gate-failed cards produce no
conclusions; all approval gates unchanged (join / ML / cleaning / execution).

## Unreleased: Analysis Depth Level 0 — findings read the whole ranked table - 2026-07-17

Addresses "the answer is just one number" in Run B / auto-executed questions.
Design + roadmap: [analysis-depth-design](docs/archive/2026-07/base/eda-agent-platform-analysis-depth-design.md).

### Changed

- **Ranked findings name the winner and runners-up.** A free-form SQL result
  is no longer collapsed to `rows[0]` + one column. `_generic_finding` /
  `_ranked_group_finding` now read the whole ranked table via a shared
  `_ranked_finding`: they name the leading row's label and up to two
  runners-up, and add a numberless "closely bunched" note when the top values
  cluster (`drivers/question_exec.py`). Real example (credit-card fraud,
  per-PCA-component separation): before — *"The returned separation is
  7.04545."*; after — *"…The strongest is V3 (separation 7.045), followed by
  V14 (separation 6.984) and V17 (separation 6.677), with the leading results
  closely bunched."*
- **Validator-safe by construction**: only metric-column cell values reach the
  text (each cited with a precise `rows_preview[i].{metric}` locator), labels
  like `V3`/`V14` don't leak digits (the number regex's lookbehind excludes
  them), and cluster wording carries no orphan counts. Single-value templates
  (`correlation_probe`, `trend`) intentionally still report their one/two
  numbers — a correlation *is* one number.

### Verified

- Full suite `579 passed / 4 skipped`; ruff + pyright clean.
- Ran the new reducer against the real stored fraud SQL artifact
  (`qrun_20260717_031328`): produces the V3→V14→V17 ranked finding above; the
  validator's number extractor pulls only the three evidenced metric values.
- Added `test_generic_finding_reads_the_whole_ranked_table` +
  `test_ranked_finding_without_label_column_reports_leading_row`.

### Design (not yet built)

- Level 1 (LLM interprets the full result table, validator-gated) and Level 2
  (bounded multi-step investigation loop) + Analysis Opportunity Cards are
  specified in the design doc as the decision-intelligence framework phase —
  the red line stays: numbers are deterministic, the LLM plans and interprets
  but never fabricates or edits a number.

## Unreleased: UI Hardening Wave 1 + Backend Visualization Generators - 2026-07-16

Implements P0 (readability/trust) and the backend half of P1-8/P2-11/P2-12 from
the [UI hardening plan](docs/archive/2026-07/base/eda-agent-platform-ui-hardening-plan-2026-07-16.md)
(companion audit: [UI audit](docs/archive/2026-07/base/eda-agent-platform-ui-audit-2026-07-16.md)).
Work was executed by three parallel subagent packages (report engine / UI
cleanup / backend generators) plus a merge-and-verify pass.

### Changed — report readability (P0-1/2/3)

- **Narrative drops internal ids**: report claims render as plain prose — no
  `claim_x:` prefixes, no inline `Evidence: prof_…` lines. Evidence ids stay in
  the Claim Ledger table and Trust rail. The validator still checks the
  untouched bundle claims (`tools/exporter.py`, `tools/html_exporter.py`).
- **Duplicate rendering fixed**: same-title sections merge and same-id claims
  dedupe (also within a single section — bundles persisted by older code carry
  injected claims twice; `schemas/reports.py merge_duplicate_sections`). The
  Report page re-renders the narrative live from the stored bundle so old runs
  pick up the fixes, and truncates before the Claim Ledger (the Trust rail
  already shows it); downloads keep the full markdown
  (`ui/views.py`, `tools/exporter.py narrative_markdown`).
- **Limitations reads as prose**: lines open with the dataset name and cap
  column enumerations at 8 + "… and N more (see the Quality page)".

### Changed — UI cleanup (P0-4/5/6)

- **Quality table leads with the message**: rows are `message / column /
  dataset_name`; `artifact_id`/`dataset_id` removed from display (provenance
  stays on Trace & evidence). Stat-test rows and chart/question labels resolve
  dataset names instead of `ds_…` ids (`ui/workbench.py`, `ui/views.py`).
- **Dead chrome removed**: Streamlit's Deploy button hidden
  (`.streamlit/config.toml [client] toolbarMode="minimal"`); the no-op
  "Show raw debug payloads" checkbox deleted.
- **Report empty state**: a run without a report shows a friendly pointer to
  Launchpad instead of a blank box.
- **Session-load race fixed**: sidebar session rows commit the load via
  `on_click` callbacks, so clicking a session then immediately navigating no
  longer silently drops the load (`ui/sessions_ui.py`, `ui/state.py`).

### Added — deterministic visualization generators (P1-8 backend, P2-11/12)

- **Correlation heatmap** (`mark="rect"`, diverging scale centered at 0,
  trivial pairs excluded, top-15 columns) and **scatter charts** for the top
  non-trivial pairs (|r| ≥ 0.5, ≤200 evenly-spaced points, titles state
  "observation, not causation") wired into auto-EDA
  (`tools/chart_specs.py`, `drivers/auto_eda.py`).
- **Stat tests hardened**: `effect_size` populated per test family (eta²,
  Cohen's d, Cramér's V, rank-biserial), tiny p-values preserved unrounded,
  and a bounded ANOVA **boxplot** companion chart (≤20 groups × ≤50
  evenly-spaced values; `tools/stat_tests.py`, `schemas/charts.py` adds the
  `boxplot` mark).
- **Sample-size metadata** on correlation rows so the UI can badge small
  samples without cross-artifact lookups (`tools/analysis.py`).

### Changed — analysis value layer (P1-7/8/9/10, Waves 2D + 3)

- **Quality rollup**: the Quality page groups issues by `code × dataset`
  ("teams.csv — constant_column × 26"), critical groups expanded, warn/info
  collapsed, with a flat-table expander preserving full detail
  (`ui/workbench.py rollup_quality_rows`, `ui/views.py _render_quality`).
- **Question list convergence**: near-identical template candidates collapse
  to ≤3 diverse representatives per template family × dataset, the shown list
  caps at 12 with "…and N more collapsed", and effectively-tied scores are no
  longer presented as a meaningful ranking. Selection is presentation-level —
  question ids/payloads/execution order stay byte-stable
  (`tools/question_discovery.py select_display_candidates`).
- **Zero-findings honesty**: an executed question with no numeric findings
  shows a neutral "ran, no numeric result" badge instead of the green
  succeeded badge (rendering only; stored status untouched).
- **Deep analysis is no longer table-only**: correlation heatmap + scatter
  charts render beside each dataset's correlation table and the ANOVA boxplot
  beside the stat tests (matched by artifact parentage; absent gracefully on
  old runs). Trivial/degenerate pairs are hidden by default behind a
  "Show N trivial/degenerate pair(s)" expander; a "small sample (n=…) —
  interpret with care" badge fires below n=30; p-values display as `<0.001`
  when tiny. The generic Profiles & charts listing excludes these charts so
  they live on Deep analysis only.

### Verified

- Full suite: `579 passed / 4 skipped`; ruff + pyright clean.
- Browser walkthrough on the live app (old stored run): narrative has zero id
  prefixes/inline evidence, duplicated focus/analysis claims render once,
  Limitations shows "teams.csv: … and 48 more", Quality table leads with the
  message column, Deploy hidden, session click + cross-page navigation keeps
  the loaded run.
- End-to-end on a fresh offline run (football tables, 39 artifacts): pipeline
  emitted 1 heatmap + 3 scatter + 1 boxplot ChartSpecs; ANOVA carries
  `effect_size=0.078` and `p=5.3e-16` unrounded; Deep analysis rendered all
  5 vega charts with the small-sample badge and trivial-pair toggle.
- Merge-pass review fixed one sampling bias (boxplot groups now sample evenly
  spaced instead of head-truncating, matching the scatter sampler).

### Still open (deliberate)

- P2-13 per-column mini distribution sparklines — needs histogram bins in the
  `ColumnProfile` payload (profiler backend), deferred as a small follow-up.

## Unreleased: Question Context and Relationship Reference Safety - 2026-07-14

This local, uncommitted update improves how the Questions and Relationships
views guide investigation. It does not yet implement the proposed
decision-intelligence analysis agent described in
[Decision Intelligence Agent Design](docs/archive/2026-07/infrastructure/eda-agent-platform-decision-intelligence-design.md).

### Changed

- **LLM-first question discovery**: when the configured LLM returns valid
  question candidates, the Questions page shows those candidates rather than
  mixing them with generic deterministic templates. Templates remain the
  offline, unavailable-LLM, or invalid-response fallback.
- **Business context in question generation**: the Question Agent now receives
  the project's Business Context as domain context. It is explicitly treated
  as context, not as executable instructions.
- **Human-readable dataset names in questions**: Question candidates carry a
  business-facing label for each target dataset. The UI shows the readable
  label in the question and keeps the raw file name in parentheses for
  traceability.
- **Relationships are reference-only by default**: relationship discovery no
  longer auto-adopts high-confidence joins. A confirmed relationship is a
  reference for a later, selected analysis question; it does not merge datasets
  or become the default basis of EDA.
- **No automatic cross-dataset execution**: automatic question execution skips
  questions that require a relationship. A user must deliberately select a
  question before a join can be part of its analysis.
- **Relationship candidate controls**: Candidates now support a minimum-score
  slider and high/medium/low confidence filters. Confirmation wording makes
  the reference-only status clear.

### Verified

- Targeted Question, Relationship, and UI unit tests: `34 passed`.
- Ruff passes for the affected Question, Relationship, and UI modules.

## M6.2: Provenance Artifacts + Analysis Skills + Cleaning-to-Analysis Fork - 2026-07-09

M6.1 made the agent analyze; M6.2 makes analysis reproducible, reusable, and
comparable.

### Added

- **Analyze the cleaned version** (`db86447`): after applying a cleaning recipe,
  the Cleaning page forks a fresh Auto EDA run onto the cleaned dataset (new
  run in the same project; the original stays selectable). The first instance
  of "vary one decision and re-run".
- **Provenance-complete artifacts** (`956d63c`): every `Artifact` carries an
  optional `env_digest` (reproducible fingerprint of python + key deps,
  stamped run-globally at `store.save_artifact`), plus `code_ref` and
  `plain_language` on computed artifacts (charts, stat tests, model cards).
  Surfaced in the HTML/markdown report and the Trace page. Artifact ids are
  unchanged (they hash payload only).
- **Analysis Skills** (`f345157`): save a validated `AnalysisPlan` as a named
  skill and replay it on another dataset. Replay verifies the referenced
  columns exist (teachable refusal on a miss), maps dataset names, and runs
  through `run_chat_turn`'s approved-plan path — reusing the permission gate
  and read-only DuckDB, offline and LLM-free. New Skills page.
- **Run-fork / What-if** (`a44beb7`): `fork_run(parent, decision)` generalizes
  the cleaning->analysis fork — take a run, change exactly one decision
  (dataset version, ML target, or report language), re-run inheriting
  everything else. New What-if page forks a variant and shows both runs'
  headline metrics side by side (with deltas), all extracted deterministically
  from artifacts. The inline 'analyze cleaned version' now routes through the
  same driver.

### Verified

- `525 passed, 4 skipped`, ruff + pyright clean. Replay is permission-gated
  (the SQL is re-validated at execution, so a crafted skill can't bypass);
  env_digest is stable across calls and identical for all artifacts in a run.

## M6.1 Wrap-up: Observability, NarrativeReviewer, Cleaning Approval - 2026-07-09

Closes the four remaining gaps the post-squash alignment audit flagged.

### Added

- **Real OpenInference span export** (`core/observability.py`, T6): replaces the
  no-op stub. When `EDA_OBSERVABILITY=1`, `store.append_trace` mirrors each
  TraceEvent into an OTel span tree (per-run root + child spans, LLM events
  tagged with OpenInference model/token conventions), exportable to a local
  Phoenix. Off by default and zero-cost — OTel imports are lazy, so it never
  breaks a run or requires the extra. TraceEvent stays the source of truth.
- **scoped NarrativeReviewer** (`agents/narrative_reviewer.py`, T7): a narrow
  post-validator pass that deepens business-narrative prose only. A body-only
  rewrite schema plus four lock layers (claims/evidence byte-identical, no new
  numbers, no new causal terms, re-run the validator) revert anything unsafe.
  Gated behind `enable_semantic_audit` (default off).
- **Live cleaning-apply CONFIRM dispatch** (`drivers/cleaning_apply.py` +
  Cleaning page): the `cleaning_apply` CONFIRM tier now has a real dispatch
  point — preview a recipe, approve, write a new dataset version (original
  untouched). The approval binds a digest of the whole recipe, so an
  approve-then-swap (even same transform ids, changed params) fails the hash
  check.

### Verified

- Docker real-container tests pass on a host with the daemon + image
  (`3 @requires_docker`: pandas analysis, runtime network isolation, read-only
  mount) — D11 landed. CI without a daemon still skips these.
- `493 passed, 4 skipped`, ruff + pyright clean, verify-security clean on the
  new dispatch. Commits `721817d`, `5edf3b0` (local `main`, not yet pushed).

## M6.1 Wave 2b-4: Docker Sandbox Default, Seeds-in-Chat, CI Gates; Squash to main - 2026-07-09

Everything below landed via PR #3 (`codex/docker-sandbox-m6`, 14 atomic
commits) and was then squash-merged to `main` together with all prior waves as
PR #4 (`00bd760`, verified zero-drift against the branch tip). `main` is now
the single source of truth; the `m0-foundation` branch lifecycle is closed.

### Added

- **Docker/OCI sandbox as the default secure backend** (`core/sandbox_docker.py`,
  `core/sandbox_broker.py`, `docker/eda-agent-sandbox/`): fail-closed
  `SandboxBroker` (auto/docker/seatbelt/subprocess; subprocess is dev/test-only
  behind an explicit env flag; open analysis is disabled when no secure backend
  is available). Container hardening: read-only input mounts, isolated output,
  no network, read-only rootfs, cap drop, no-new-privileges, pids/memory+swap/
  cpu/time/output limits, image with locked runtime deps and availability
  pre-check. Adversarial tests in `tests/security/test_docker_sandbox.py`.
- **Semantic seeds consumed by chat planning** (T5b): field meanings, metric
  definitions, and verified answers now feed the chat planner context —
  closing the write-only gap flagged in Wave 2.
- **Unified action dispatcher** (`core/action_dispatcher.py`) and UI sandbox
  status indicator.
- **CI gates** (T8): `.github/workflows/ci.yml` + `scripts/ci_local.sh`
  (pytest + ruff + pyright + security suites; live-LLM evals stay key-gated).
- Observability adapter skeleton + `[observability]` extra (full
  Phoenix/OpenInference span export still open).

### Fixed / Security

- Approved-plan re-entry no longer trusts a UI-supplied boolean: the driver
  re-validates the approval hash against the exact plan before executing
  ("the UI is not the execution boundary").
- Docker backend fails closed when the daemon is up but the image is missing;
  memory and swap are limited together.
- Chat artifacts reload correctly when reopening a session.

### Alignment audit (2026-07-09, two-agent review after the squash)

- Gates re-verified on `main`: `475 passed, 4 skipped` (1 live-eval,
  3 Docker-runtime; Seatbelt 11 run on this macOS host), ruff and pyright
  clean; no test files were net-deleted across the PR range.
- Three real gaps confirmed and tracked in
  `docs/archive/2026-07/base/eda-agent-platform-m6-plan.md` §3: the `cleaning_apply` /
  `artifact_overwrite` CONFIRM tiers have zero dispatch points (the cleaning
  audit page is read-only, not an approval flow); the observability adapter is
  a stub with no `trace.py` hook; the Docker backend has no real-container
  end-to-end run yet.
- Security posture is strictly tightened (driver-side plan-hash enforcement,
  image-missing fail-closed, memory+swap dual limit) — no guard weakened.

## M0 EDA Raw Cleaning Audit - 2026-07-06

This pass makes the pre-cleaning path auditable from the Streamlit UI without
mixing raw and cleaned evidence.

### Added

- **Cleaning info and raw data page** under Understand the data: shows
  raw-before-cleaning profiles, automatic raw chart specs, raw data previews,
  and a cleaning log for each run.
- **Raw EDA artifact types** (`RawDatasetProfile`, `RawChartSpec`,
  `RawDataPreview`) so raw evidence is recorded separately from the cleaned
  dataset profiles used by the rest of the analysis pipeline.
- **Cleaning guardrails** on `CleaningRecipe`: guard-only events are now
  persisted when a missing-value or outlier drop is skipped because it would
  remove every column or violate the minimum row-retention threshold.
- **Cleaning log UI** with summary rows, deleted-data details, protection
  triggers, thresholds, and practical next-step suggestions.

### Verified

- `ruff check` passed on the touched files.
- `303 passed, 1 skipped` for `eda_platform/tests/unit`.

## M6.1 Wave 2: CodeAgent Wiring + Per-Action Permissions + Semantic Editing - 2026-07-05

Wave 2 turns the dormant sandbox into a chat-triggerable capability behind a
per-action permission gate, and adds human editing of the semantic layer.

### Added

- **Per-action permission tiering** (`core/permissions.py`): every chat-dispatched
  action is classified AUTO (read-only DuckDB SELECT, artifact read, sandboxed
  compute) / CONFIRM (cleaning apply, artifact overwrite — state-changing) / DENY
  (host write, network, sandbox bypass). CONFIRM approvals are bound to a hash of
  the canonical action content, so an approve-then-swap cannot execute different
  work than what was shown. DENY messages reuse the teachable tool-guard format.
- **CodeAgent wired into chat** (`open_analysis` intent → `agents/code_agent.py`):
  open-ended analysis the SQL path can't express now runs as a bounded
  generate → sandbox-execute → observe → repair loop. Hard ceiling of 3 attempts
  (`min(max_repairs+1, 3)`), token/time budget checked each iteration, typed
  `ExecArtifact` exit predicate (requires stdout JSON). Data is mounted read-only;
  execution goes through the guardrailed `PortableSubprocessBackend` — the AST
  permission check is a defense-in-depth pre-filter, not the boundary. Each
  attempt emits a trace event so repair-loop success rate is measurable.
- **Semantic layer editing** (`ui/semantic_ui.py`, Knowledge page): human
  add/edit/delete over FR-15's knowledge classes — field meanings, metric
  definitions, verified answers — plus entity notes and auto-sunk verified
  relations. `SemanticSeeds` extended backward-compatibly (old seeds.json loads
  unchanged).

### Fixed

- Upload filenames are now reduced to their basename before landing in
  `_incoming`, so a crafted `../../evil.csv` name cannot escape the staging
  directory (found via verify-security; 2 regression tests).
- Knowledge-page add forms no longer wipe typed input on a failed-validation
  submit (dropped `clear_on_submit`; fields clear only on the success path).

### Security review

- Adversarial pass on the permission layer confirmed the real boundary is the
  subprocess sandbox, not the AST check; host-write attempts are blocked at both
  layers (classifier flags DENY *and* sandbox returns `blocked` with the file
  never created), and approve-then-swap is rejected by hash mismatch.
- `cleaning_apply` / `artifact_overwrite` CONFIRM tiers are defined and
  unit-tested but not yet dispatched from chat (cleaning still applies via the
  UI) — deferred with the cleaning-approval UI, not a live gap.

### Verified

- `409 passed, 1 skipped` (skip = live-LLM gated), `ruff` passed,
  `pyright 0 errors`, verify-security clean on the new files.

## M6.1 Wave 1: Tool Guards + PDF Export + M4 Eval Week - 2026-07-05

M6 kicked off per `docs/archive/2026-07/base/eda-agent-platform-m6-plan.md` (deployment target:
local single-user; decisions D1-D10 recorded there). Wave 1 delivers the
agent-loop safety prerequisite, the first export format beyond HTML, and the
M4 eval instruments.

### Added

- **Tool guards with teachable errors** (`core/tool_guard.py`): every
  LLM-supplied tool parameter (plan references, stat test selection, ML
  target/time columns, cleaning transforms) is validated for range / enum /
  non-emptiness / column existence / column semantic type before execution.
  Rejections render a three-part model-actionable message (what was wrong /
  allowed / how to fix), feed back into the existing bounded retry loops
  (self-heal verified end-to-end in tests), and emit trace events.
- **PDF report export** (`tools/pdf_exporter.py`): WeasyPrint >= 69 renders
  the self-contained HTML report (inline SVG, CJK font fallback chain, A4
  print CSS with page numbers). Optional `pdf` extra with runtime probing —
  missing system deps degrade to an install hint in the UI instead of
  crashing. PDF bytes are cached per report content to keep Streamlit reruns
  fast.
- **M4 eval week instruments** (`tests/evals/`, offline-verified, live
  scoring deferred to manual runs — commands in
  `docs/archive/2026-07/base/eda-agent-platform-m4-eval-report.md`):
  - NL2SQL: 20 zh/en golden cases + execution-accuracy harness
    (golden-vs-golden self-check 20/20; equivalence-preserving rewrites and
    known-bad SQL both classified correctly).
  - Join correctness: ecommerce ground truth labeled; measured
    precision 1.0 / recall 1.0 against the >=0.9 / >=0.8 DoD line.
  - Question quality: scoring rubric + 12-case judge calibration set
    (human scoring pending).
  - External benchmarks: DABStep hard x7 + KramaBench x5 samples landed
    with adapter notes.
  - `scripts/demo_j1.py` / `demo_j3.py` offline demo scripts.
  - 5 new adversarial tests for the causal-quote exemption boundary (M4 §11
    gap).

### Verified

- `377 passed, 1 skipped` (skip = live-LLM gated), `ruff` passed,
  `pyright 0 errors`.
- Real WeasyPrint conversion verified on this host (Chinese text + inline
  SVG fixture); degradation path unit-tested.
- ccg verify-change: pass (2 INFO); verify-quality: warnings are
  pre-existing style debt (file/param counts), no Critical.

## M5 Final UI/Run Review - 2026-07-05

Review against the latest UI iteration and run
`run_20260705_055142_429744_f95766`.

This closes M5 for the current roadmap. Remaining agent-loop enhancements are
tracked as M5+ / M6 follow-ups, so M6 service/production planning can start from
this baseline.

### Fixed

- Report question claims no longer duplicate when the LLM already emits
  qfocus/qfind claims and deterministic question injection runs afterward; the
  deterministic claims now upsert by id.
- Seatbelt security tests now probe whether `sandbox-exec` can actually apply a
  minimal profile, not just whether the binary exists. Restricted hosts where
  `sandbox_apply` is denied skip the enhanced-backend suite cleanly.
- `docs/archive/2026-07/eda-agent-platform-todo.md` now reflects the current clean-start M5
  closeout state and reclassifies M5+ agent-loop enhancements as non-blocking
  follow-ups.
- README workflow text now matches the current top-navigation IA.

### Verified

- Latest run validated with DeepSeek flash, 3 LLM report calls, 40,435 tokens,
  and final `ReportAudit.status=validated`.
- `283 passed, 11 skipped`
- `ruff check .` passed
- `pyright`: `0 errors`

## M5 part1 Review + Remediation - 2026-07-04

Adversarial review of M5 part1 (cleaning/stats/ML/sandbox) found 7 Critical +
several High/Medium — all 216 tests were green because they only covered happy
paths. Every finding fixed and each Critical independently re-probed. Findings
and per-item status in `docs/archive/2026-07/base/eda-agent-platform-m5-part1-review.md`.

### Fixed — Critical

- Sandbox was not a security boundary: the cross-platform AST-only filter is
  trivially bypassed (getattr, object-graph traversal, chr()-built paths) →
  proven command exec, host-file read, network egress, and memory exhaustion.
  Implemented `LocalSeatbeltBackend` (macOS `sandbox-exec` profile denying
  network + scoping filesystem to the exec dir, plus an RSS memory watchdog).
  Verified the Seatbelt profile ALONE contains all four escapes with the AST
  filter disabled; fail-closed on non-macOS. `PortableSubprocessBackend` kept
  for dev but its docstring now states it is not a security boundary. Security
  tests are now adversarial, not naive.
- ML leakage guard was silently certifying leak-driven models as clean:
  categorical 1:1 target proxies invisible (only numeric Pearson checked) →
  now caught via group-purity / mutual-information / correlation-ratio;
  temporal split silently fell back to random unless the column was named
  date/time → now detected by dtype/content; preprocessing (one-hot universe +
  median imputation) was fit before the split → now fit on train only.
- Cleaning tier gate trusted a caller-supplied label (defaulting "safe") so
  `drop_rows`/`drop_column`/`fill_missing` ran with no approval → tier now
  derived from the operation type server-side. Version collisions silently
  overwrote a prior cleaning → next free version allocated. `parse_numeric`
  auto-applied as "safe" silently nulled unparseable real values → demoted to
  lossy with the loss count surfaced.

### Fixed — High / Medium

- ML: classification leakage threshold corrected (0.99), task detection tightened
  (continuous ≤20-unique targets no longer forced to classification), numeric
  high-cardinality ID columns dropped, `_nan` indicator columns only when NaN.
- Cleaning: lineage version-mismatch assertion, affected-row split into
  dropped/edited/cells, scientific-notation parsing added, duplicate-column
  clear error, `flag_constant_column`/`clip_outliers` implemented.
- Stats: two live-surfaced degenerate-input failures. (1) Shapiro-Wilk / Levene
  assumption checks divided by zero on constant / near-constant groups
  (`RuntimeWarning: invalid value in divide`) and reported NaN as a silent
  "warn" → zero-variance guard + warning-capturing scipy wrapper +
  NaN→not_applicable. (2) The main test (t-test/ANOVA/…) on a constant column
  produced a non-finite/None statistic that crashed the whole EDA pipeline
  (`2 validation errors for StatTestResult`) → `_round_float` is now None-safe,
  a non-computable result is skipped via ValueError (the pipeline step already
  catches it), and scipy's degenerate-input RuntimeWarnings are suppressed
  during the test call. Regression tests added.
- Wiring: stat multiple-comparisons warning now fires (driver threads
  `comparison_count`); ML baseline only models datasets that contain the target.

### Verified

- 257 tests (216 → +41 adversarial regressions), ruff clean, pyright 0 errors.
- Each Critical independently re-probed by the orchestrator (not the fixing
  agent); stats/ml pass under `-W error::RuntimeWarning`.
- Sandbox + CodeAgent and the cleaning recipe remain unwired (dormant); they are
  now a real boundary / actually safe, so wiring them is the next step.

## M5 Foundation / Stage 1 Closeout - 2026-07-04

M5 implementation started from `docs/archive/2026-07/base/eda-agent-platform-m5-plan.md`, with the
sandbox baseline adjusted for future Windows support instead of locking the app
to macOS Seatbelt.

Stage 1 is closed as the backend foundation: M5a deterministic tools are landed,
and M5b has the portable sandbox plus first CodeAgent repair loop. Enhanced
OS-level sandbox backends and the full cleaning approval UI remain follow-up
work.

### Added

- Cleaning recipe foundation:
  - `CleaningRecipe`, `CleaningPreview`, and `CleaningApplyResult` schemas
  - deterministic transforms for whitespace trimming, numeric-like parsing,
    duplicate removal, missing-row removal, fill-missing, and column dropping
  - preview diff with changed-row counts and versioned apply output; original
    uploads are not mutated in place
  - dataset lineage fields: parent dataset/version and recipe id
- Deterministic statistical tests:
  - `StatTestResult` schema and artifact type
  - scipy-backed independent/paired t-test, chi-square independence, one-way
    ANOVA, Mann-Whitney U, and Kruskal-Wallis
  - assumption checks and multiple-comparison warning support
  - automatic pipeline step that emits one suitable stat-test artifact when a
    categorical group and numeric measure are available
- Thin ML baseline:
  - `ModelCard` schema and artifact type
  - sklearn random-forest classification/regression baseline
  - leakage guard for target-copy/perfect-correlation features, ID-like fields,
    and time-ordered split handling
  - optional `ml_target_column` / `ml_time_column` pipeline parameters and
    Streamlit sidebar controls
- M5 artifacts are now included in the evidence pack, LLM-facing manifest,
  deterministic fallback report claims, validator numeric resolution, and the
  Analysis tab.
- Cross-platform sandbox foundation:
  - `ExecutionBackend` protocol
  - `PortableSubprocessBackend` with fresh subprocess execution, env scrub,
    static dangerous-operation blocking, timeout, and output caps
  - portable security tests for blocked network import, infinite-loop timeout,
    and stdout cap
- CodeAgent repair loop:
  - LLM drafts Python into `CodeDraft`
  - execution goes through `ExecutionBackend`
  - sandbox failure feedback is sent back for up to two repair rounds
  - attempt events expose status, sandbox status, duration, and error summary

### Changed

- M5 sandbox plan now documents `PortableSubprocessBackend` as the default local
  backend and macOS Seatbelt / Linux seccomp / Docker / microVM / remote
  sandboxes as enhanced backends. The portable backend is explicitly documented
  as a development guardrail, not full OS isolation.
- Project dependencies now include `scipy` and `scikit-learn`.
- README now documents M5 status, optional ML columns, no-`uv` startup/testing
  commands, and the portable sandbox caveat.

### Fixed

- Optional ML baseline now skips with a trace event when the requested target is
  invalid for modeling, instead of failing the whole EDA pipeline.
- Portable sandbox now blocks obvious host path literals and rejects mount
  targets that resolve outside the execution directory.
- v2 roadmap sandbox wording now matches the M5 cross-platform default and no
  longer treats macOS Seatbelt / rlimit as the baseline for all local runs.
- M5 plan now separates portable guardrail attack coverage from enhanced-backend
  OS-level ACL and memory isolation coverage.

### Verified

- `216 passed`
- `ruff check .` passed
- `pyright`: `0 errors`

## M4 Live-Run Quality Fixes - 2026-07-04

Result-driven fixes from the first live M4 run on the football 3-table set.
The review found six defect classes; P0/P1 all fixed (codex package + one
orchestrator SQL fix), P2 UX items deferred to the eval week.

### Fixed

- LLM question route died on score scale (model returned 1-10 where the
  schema wants 0-1, all 10 proposals dropped): prompt now states the range
  with an example, out-of-range values are normalized (/10, /100, clamp),
  and one bounded retry feeds the ValidationError back. (P0-a)
- Degenerate correlations escaped the trivial filter (auto top-3 burned all
  slots on r=1.0 tautologies like `assists_against` vs
  `assists_per90_against`): any |r| ≥ 0.999 pair is now trivial regardless
  of names; `per90`/`pct`/`percent` are removable name tokens; constant /
  ≤3-distinct / CV<1% columns are excluded from correlation and
  group-difference templates by consuming quality artifacts (cleaning-lite).
  Football-data trivial_dropped: 14 → 95. (P0-b)
- Auto top-3 lacked diversity (3 slots, one template type, one dataset):
  max one question per template, distinct datasets preferred, one slot
  reserved for an eligible cross-table question. (P0-c)
- Relationship candidate noise (metric↔metric pairs like
  `teams.interceptions → matches.away_interceptions` reached medium and
  polluted the HITL list; `home_team` hit both `teams.team` and
  `teams.team_country`): key-likeness hard gate (metric-vs-metric can never
  reach medium/high) plus top-1-target-per-left-column pruning. Football
  result: 6 high (all semantically correct) / 0 medium / 24 low. (P1-d)
- Report retry truncation (attempt 2 died at completion==6000 cap, then
  resent the full manifest; claim coverage eroded 0.909 → 0.545): cap hits
  now raise the completion budget once and ask for fewer/shorter claims,
  never an identical re-send; deterministic Dataset Overview claims and
  Executive Summary synthesis protect base sections from pruning. (P1-e)
- Template SQL crashed on physically-VARCHAR numeric columns
  (`sum(players.age)` BinderException — age is "27-003" years-days text):
  all metric references in the five template SQL builders now use
  `try_cast(... as double)`; regression test with a mixed-type column.

### Verified

- 200 tests (189 → 200), ruff clean, pyright 0 errors.
- Offline acceptance rerun on the real football uploads: auto top-3 now 3/3
  succeeded across three distinct template types; top-ranked questions are
  real signals (away_sot vs home_saves r=0.954) instead of tautologies;
  relationship medium tier is clean.

### Known limitation

- The cross-table age aggregation succeeds but returns empty findings: age
  "27-003" is unparseable by SQL `try_cast` (rows filter to zero). Proper
  parsing of exotic numeric formats at execution time is an M5
  cleaning-recipe item; analysis tables already parse it via
  `parse_numeric_like`, so profile/report statistics are unaffected.

## M4 W1-W2 Core - 2026-07-04

Relationship discovery + question discovery + two-phase execution, per
`docs/archive/2026-07/base/eda-agent-platform-m4-plan.md`. Delivered dual-track (codex backend /
opus UI-report) with frozen schema contracts; orchestrator merged, fixed
typing, and verified: 189 tests (155 → 189), ruff clean, pyright 0 errors.

### Added

- Relationship discovery engine (deterministic, zero LLM):
  - ensemble signals per v2-plan §4.9 — name similarity, type compatibility,
    bidirectional value overlap (DuckDB SQL), right-uniqueness, null rates,
    format fingerprint; weighted score with configurable §4.9 thresholds
  - candidate-explosion guards (type prefilter, null/constant skip, per-pair
    cap with `truncated_pairs` accounting), deterministic sampling >50k rows
  - DuckDB join validation: row multiplier, orphan rates both sides,
    cardinality, archived `verification_sql`, M:N / inflation warnings
  - ER diagram artifact (plain DOT + tabular fallback)
  - new artifacts: `relcand` / `relval` / `erd`
- 🔗 Relations tab: `st.graphviz_chart` ER view, candidates with signal
  expanders, validation risk badges, medium-confidence "Confirm relation" →
  project-level `semantic/seeds.json` (idempotent), auto-adopted badges
- Question discovery (v2-plan §4.10 two-route):
  - five deterministic templates (trend, group difference, non-trivial
    correlation probe, high-missing quality, cross-table via auto-adopted
    joins) each carrying executable `sql_template`
  - complement/tautology filter: `is_trivial_pair` on correlation rows
    (home/away complements, near-constant sums, same-stem rescales) — fixes
    the possession r=-1.0 degenerate-insight class from live runs
  - deterministic scoring (availability/signal/quality-risk/join-risk); LLM
    route optional (graceful skip offline), LLM scores affect display only
  - auto-execution of top-3 gated template questions in Run A (template-only,
    join_risk ≤ 0.3, quality_risk ≤ 0.6, relations auto-adopted, evidence-ref
    findings phrased deterministically from SQL results)
  - new artifacts: `qcand` / `qexec`
- Run B driver `run_question_batch`: fresh run, reloads source datasets,
  executes approved questions (template directly; llm-route via planner),
  regenerates the report over combined artifacts
- ❓ Questions tab: auto-executed findings with evidence expanders, candidate
  checkboxes with risk chips, LLM-needed warning, "Run selected analyses" →
  Run B, inline new-run findings
- Report business sections filled from question execution (decision D1):
  - deterministic claims for Selected Analysis Focus / Agent-Performed
    Analysis; qexec digest grounds LLM Business Findings/Recommendations;
    fallback injection (re-validated) when LLM yields none
  - validator resolves numeric evidence from SQL_RESULT payloads; quoted
    spans exempt from the causal check only (numbers still verified)
- Day-0 report patches: Limitations synthesized from quality artifacts,
  deterministic chart-inventory appendix, claim-ledger evidence dedup

### Remaining for M4 closeout (eval week)

- 20-case NL2SQL golden (M3.11) + execution-accuracy run
- DABStep hard 5-10 / KramaBench 3-5 sampling; question-quality manual scoring
- J1/J3 demo scripts; live LLM smoke of the full discovery→report chain
- adversarial validator case: causal text hidden inside double quotes

## M3.8-M3.9 Report Quality Sprint - 2026-07-03

Report generation now spends less prompt budget on retries and produces
non-empty required sections even when the hard validator prunes claims.

### Added

- Validator findings now carry `repair_mode`: `deterministic`, `llm`, or `prune`.
- Deterministic report repair for mechanical findings:
  - duplicate evidence references are removed before retry;
  - `missing_quality_warning` attaches the matching quality issue artifact and
    adds a conservative caveat without another LLM call.
- First report attempt sends a compact `evidence_manifest` instead of the full
  `EvidencePack`.
- LLM retry prompts send only previous claims, validator findings, and repair
  instructions; they no longer resend the full evidence manifest.
- Required report sections get deterministic fallback body text when no
  validated claims survive.
- Dev Inspector now surfaces section coverage, claim survival, deterministic
  repair count, per-attempt prompt tokens, and run `code_version`.
- DeepSeek default cost estimation for `deepseek-v4-flash` and
  `deepseek-v4-pro`.
- LLM read timeout is configurable via `EDA_LLM_TIMEOUT_SECONDS` and the LLM
  Settings dialog; use a larger value for slower pro/reasoning models.
- Run manifests now record the current git short SHA when available.

### Changed

- Only semantic findings such as `numeric_mismatch` and causal overclaims trigger
  LLM retry. Structural and unsupported-entity failures go directly to repair or
  hard pruning.
- LLM trace events now include real start/finish timestamps so duration appears
  in Debug Inspector.
- Review follow-up: hard-gate pruning now revalidates even when all claims are
  removed; Debug Inspector separates rendered-section coverage from claim-section
  coverage; deterministic repair now clears unsupported section-body facts.

### Verified

- `155 passed`
- `ruff check .` passed
- `pyright`: `0 errors`

### Live Comparison

- DeepSeek flash/pro comparison on the same football dataset is recorded in
  `docs/archive/2026-07/base/eda-agent-platform-m38-m39-report-quality-sprint.md`.
- `deepseek-v4-flash`: 1 round, 13,921 / 6,305 prompt/completion tokens,
  $0.003714, default recommended.
- `deepseek-v4-pro`: 3 rounds, 22,629 / 27,353 prompt/completion tokens,
  $0.033641, validated but not recommended as the default report model.

## UI Iteration 2 - 2026-07-03

Second UI pass addressing direct feedback; grounded in current Streamlit docs
(st.dialog, st.status, layout patterns via Context7). Verified with AppTest
(empty / with-data / dev-toggle / dialog-open paths, no exceptions); suite 149
passed, ruff/pyright clean.

### Changed

- All UI copy is now English (labels, banners, buttons, chat prompts).
- Pipeline progress is a linear horizontal timeline/stepper (HTML+CSS) with
  per-stage circles (done ✓ / active / pending / failed ✕), driven live from
  `on_trace_event`; the completed timeline persists at the top after a run.
- The Debug Inspector is now a first-class top-level `🛠 Dev` tab with its own
  in-tab toggle (`st.toggle`), so it is discoverable without hunting through the
  sidebar; content shows trace/LLM/tool/cost tables when enabled.
- LLM settings moved into a modal `@st.dialog` opened from a sidebar button;
  settings persist in `session_state` (provider/model/status shown as a sidebar
  caption), which also decouples settings rendering from run-handler order.
- Top-level tabs are now 📊 Explore / 💡 Insight / 🗂 Artifacts / 🛠 Dev.

## UI Redesign (Dual-Hub) - 2026-07-03

Streamlit front-end restructured to the "Dual-Hub" layout (option 1 from a
multi-model UI review). Goals: clear section separation, visible pipeline
progress, function-matched components, better result/visualization display.
Backend/data flow unchanged; only `ui/streamlit_app.py` was rewritten.
Verified with `streamlit.testing.v1.AppTest` (empty and with-data paths render
without exceptions); full suite still 149 passed, ruff/pyright clean.

### Changed

- Navigation: 8 flat tabs collapsed into 3 hubs with nested sub-tabs —
  📊 数据探索 (Overview / Datasets / Quality / Charts / Analysis),
  💡 智能洞察 (Report and Chat side-by-side columns),
  ⚙️ 配置与调试 (Artifacts browser + Debug Inspector).
- Pipeline progress: the single global spinner is replaced by `st.status`
  driven from `on_trace_event`, listing each stage (① Profiling → ② Quality →
  ③ Charts → ④ Analysis → ⑤ Report) as it completes, with complete/error state.
- Overview: health banner (critical/warn/ok) + report-status line, metric cards,
  per-dataset bordered cards with semantic-type badges and PK candidates.
- Datasets: `column_config.ProgressColumn` mini-bars for missing % / unique %.
- Quality: severity color coding (🔴/🟡/🟢) with per-severity bordered sections
  and count metrics.
- Charts / Analysis: bordered card grid.
- Report: colored status header, model/token/cost metrics, claim ledger and
  validator findings in expanders, bordered markdown body.
- Chat: fixed-height scrollable message container; charts and Evidence panel per
  answer; approval plan card with primary confirm / cancel buttons.
- LLM settings moved from a sidebar expander into a sidebar `st.popover` (kept in
  the sidebar so it still executes before the run handler reads the settings).
- The page-level 70/30 dev split is removed; the Debug Inspector now lives in the
  ⚙️ hub, which also eliminates an illegal columns-in-columns nesting risk.

## M3.7 Hardening - 2026-07-03

Review-driven hardening pass over the M3 foundation (plan and findings in
`docs/archive/2026-07/base/eda-agent-platform-m3-hardening-plan.md`, S1-S10). Implemented by two
agents with disjoint file ownership; independently verified: 149 passed,
ruff clean, pyright 0 errors.

### Fixed

- Chat turns no longer crash on planner/LLM/SQL failures: every stage is
  wrapped, `ChatTurnResult.status` (`answer` / `awaiting_approval` /
  `refused` / `error`) is set on all branches, and failures append
  `chat_turn_failed` trace events. (S1)
- Hallucinated columns are caught before execution: `DuckDBQueryEngine.dry_run`
  (EXPLAIN binder check, `SqlBindingError`), wired into `build_plan` with one
  bounded LLM retry fed by `previous_error`. (S2)
- `ask_from_artifacts` is a real deterministic retriever now (missing / quality /
  shape / primary-key / inventory answers with artifact-id citations, EN+ZH),
  replacing the artifact-type-count stub. (S3)
- PII masking no longer re-validates the report per cell (`pii_labels`
  single-pass); digit-only IDs are no longer misclassified as phones;
  `product_name`-style columns are no longer masked as person names. (S4)
- NL2SQL eval comparison is order-insensitive (canonical multiset compare with
  float normalization), matching execution-accuracy practice. (S5)
- Chat history persists to `projects/<pid>/chat/<session_id>.jsonl` and is
  restored on reload (plain-text restore; live turns render rich). (S6)
- Approval flow is complete: `run_chat_turn(approved_plan=...)` executes the
  previewed plan exactly (no silent re-plan); UI has confirm/cancel buttons. (S7)
- Chat answers render charts (`chart_spec_from_sql_result` bar/line heuristic)
  and an Evidence expander (artifact ids, SQL, row counts, validation). (S8)
- Value context is computed once per run and reused across turns. (S9)
- Same-filename datasets no longer clobber the relation mapping
  (`dataset_id -> relation` stays reliable). (S10)

## M3 Closeout - 2026-07-03

M3 moves the V2 platform from deterministic Auto EDA/reporting into a usable
agent workflow foundation: live LLM report generation, SQL-backed chat, typed
artifacts, validator-gated reports, and developer observability are now in place.

### Added

- M3 Chat foundation:
  - intent routing via `m3_route_intent`
  - analysis planning via `m3_build_plan`
  - read-only DuckDB SQL execution
  - `SQL_RESULT` artifacts
  - `CHAT_TURN_PLAN` artifacts
  - Streamlit `Chat` tab for data questions on the current run
- SQL safety layer:
  - single-statement `SELECT` / `WITH` guard
  - external file access blocks
  - DDL/DML/COPY/read function blocks
  - row limits, truncation metadata, and wall-clock timeout support
- PII and value context:
  - heuristic PII tagging for email, phone, name, id-like columns
  - Chinese PII column-name support
  - top-N value profiles for planner context
  - PII values masked before entering prompts
- Agentic report workflow:
  - evidence pack builder
  - compact `ReportPlanDraft` LLM output
  - code-injected required report sections
  - claim ledger
  - deterministic hard report validator
  - Markdown and self-contained HTML report export
- Dev Mode:
  - right-side Debug Inspector
  - immediate inspector display when Dev Mode is enabled
  - live trace updates while Auto EDA is running
  - timeline, LLM calls, token usage, tools, artifacts, and errors tables
- LLM configuration:
  - `.env` loader
  - `.env.example`
  - OpenAI, DeepSeek, and OpenAI-compatible provider settings
  - DeepSeek presets: `deepseek-v4-flash`, `deepseek-v4-pro`
  - resolved base URL and structural config status in the UI
- Local-first docs:
  - README startup instructions
  - M0-M2 implementation record
  - M2 quality strategy findings
  - M3 plan
  - M3 implementation record

### Changed

- Report generation no longer asks the model to emit one full nested
  `ReportDraft`. The model now emits a smaller `ReportPlanDraft` containing
  evidence-backed claims, and the app assembles the full report structure.
- Report section prose is not trusted as evidence-bearing content. Quantitative
  or causal statements must live in `ReportClaim` objects with `EvidenceRef`s.
- Failed LLM attempts now record trace events and token usage instead of
  disappearing from Debug Inspector metrics.
- Report validation now emits trace events for every repair attempt.
- The validator can resolve numeric evidence from table/profile artifacts, not
  only from model-filled `EvidenceRef.value`.
- The final hard gate removes unsupported claims instead of publishing them.
- Dev Mode layout is page-level: the inspector is present before, during, and
  after a run.

### Verified

- Full local test suite:

```bash
UV_CACHE_DIR=.uv-cache .venv/bin/uv run pytest
UV_CACHE_DIR=.uv-cache .venv/bin/uv run ruff check .
UV_CACHE_DIR=.uv-cache .venv/bin/uv run pyright
git diff --check
```

- Latest verification result:
  - `116 passed`
  - ruff passed
  - pyright: `0 errors`
  - `git diff --check`: clean
- DeepSeek live smoke passed:
  - provider: `deepseek`
  - model: `deepseek-v4-flash`
  - report status: `validated`
  - fallback: `false`
  - compact report-plan path avoids the earlier JSON EOF failure

### Known Gaps

- Report coverage can still be sparse after hard-validator pruning. Empty
  sections are expected when no validated claims survive for that section.
- Evidence pack payloads are too large for wide multi-file datasets. The current
  default does not send raw full data, but it can still send too much schema,
  quality, and aggregate context.
- Debug `Errors` needs better severity grouping so intermediate validator
  rejections do not look like final run failures.
- Chat UI does not yet have complete approval/modify-plan controls.
- Chat answers do not yet generate chart artifacts.
- `validate_sql_result()` is still shape-oriented; answer text numbers are not
  fully grounded cell-by-cell yet.
- NL2SQL evaluation still uses a small scripted foundation; the planned 20-case
  golden set is not complete.
- Dedicated `chat/<session_id>.jsonl` persistence is not implemented yet.

### Next Recommended Work

- M3.8 evidence-pack compression:
  - manifest-first report planning
  - evidence retrieval by section/focus
  - retry with validator delta only
  - wide-table column grouping
- M3.9 report coverage hardening:
  - deterministic fallback claims for required sections
  - explicit empty-section rendering
  - coverage metrics in Debug Inspector
- M3.10 chat productization:
  - approval button flow
  - plan modification
  - chart artifact generation
  - evidence panels for chat answers
- M3.11 eval expansion:
  - 20-case NL2SQL golden set
  - live LLM gated execution-accuracy run
  - cost regression tracking
