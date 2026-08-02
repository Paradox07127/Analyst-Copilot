---
name: causal-claim-boundary
description: Assess whether an observational or quasi-experimental analysis can support causal language. Use when a question asks what caused an outcome, requests treatment or policy effects, proposes DiD, RDD, IV, propensity-score matching, or synthetic control, or when an analysis risks turning association or prediction into a causal claim.
---

# Causal Claim Boundary

Treat causal identification as a design argument, not a model metric. Use data
checks to find contradictions and weaknesses; never claim that observed data
proved an identifying assumption that depends on an unobserved counterfactual.

## Apply the boundary

1. Name the treatment, outcome, unit, assignment mechanism, intervention time,
   and target population. If any is missing, stop at associative language.
2. Choose a design only when the assignment mechanism matches it. Do not select
   a design because its estimator is available.
3. Record every identifying assumption below as `supported`, `violated`, or
   `unknown`, with an evidence artifact for `supported` or `violated`.
4. Treat any required `unknown` or `violated` assumption as blocking an
   unconditional causal claim. Pre-trend, balance, placebo, density, and
   first-stage checks can falsify or weaken a design; passing them does not prove
   the counterfactual assumption.
5. Report estimand, eligible population, time window, uncertainty, sensitivity
   checks, and limitations. Keep prediction metrics separate from treatment
   effects.
6. In the current EDA runtime, emit only observed/associative language: the
   deterministic claim gate rejects causal claims by policy. Produce a design
   readiness assessment for human review instead of bypassing that gate.

## Identification assumptions

| Design | Required assumptions | Checks that can weaken or falsify | Boundary |
|---|---|---|---|
| Difference-in-differences | Well-defined treatment; no anticipation; untreated potential outcomes would follow parallel trends; no treatment spillovers; comparison units remain valid | Event-study pre-trends, composition changes, placebo dates/outcomes, concurrent shocks, treatment-timing audit | Pre-trends do not prove post-treatment parallel counterfactual trends. With staggered timing, do not use an undiagnosed two-way fixed-effects coefficient when effects may vary by cohort or time. |
| Regression discontinuity | Treatment changes at a known cutoff; potential outcomes and other determinants are continuous at the cutoff; units cannot precisely manipulate assignment; bandwidth and functional form identify a local contrast | Running-variable density, covariate continuity, donut/placebo cutoffs, bandwidth and polynomial sensitivity, first stage for fuzzy RDD | The estimand is local to the cutoff. Smooth-looking covariates do not prove continuity of unobserved potential outcomes. |
| Instrumental variables | Instrument relevance; instrument is as-if randomly assigned conditional on the design; exclusion restriction; monotonicity for a LATE interpretation; no interference | First-stage strength, balance, negative controls, over-identification only when separately credible instruments exist, weak-IV robust inference | Relevance is testable; independence and exclusion are primarily design claims. Never translate a strong first stage into proof of instrument validity. |
| Propensity-score matching | Consistency; conditional exchangeability/no unmeasured confounding; positivity/common support; only pre-treatment covariates enter assignment adjustment; outcome model and matching choices are not selected on the result | Overlap, balance after matching, attrition, trimming sensitivity, negative controls, alternative propensity/outcome specifications | Balance covers observed covariates only. Matching alone never licenses “caused”; state the no-unmeasured-confounding assumption explicitly. |
| Synthetic control | Donor units are untreated and unaffected by spillovers; no anticipation; donor weights approximate the treated unit's untreated path; no coincident treated-unit shock; sufficient stable pre-period | Pre-treatment fit, leave-one-donor-out, in-space/in-time placebos, donor contamination, specification and pre-period sensitivity | Good pre-fit is necessary but not proof of the post-treatment counterfactual. Generalization beyond the treated unit and period requires a separate argument. |

## Language contract

- Without an identified design, write “is associated with,” “differs by,”
  “predicts,” or “was observed after.”
- With a plausible design whose assumptions remain partly untestable, write
  “the design estimates X under the stated assumptions” and enumerate them.
- Do not use “caused,” “impact,” “effect,” “attributable to,” “led to,” or
  “because of” in a publishable claim while the runtime causal gate is closed.
- Do not present statistical significance, feature importance, forecast
  accuracy, temporal ordering, or model calibration as causal identification.

## Required output

Return a compact design-readiness record containing:

- proposed design and estimand;
- treatment, outcome, unit, assignment mechanism, and time window;
- assumption ledger with status and evidence artifact IDs;
- falsification, placebo, overlap, or sensitivity checks performed;
- blocked causal phrases and an allowed associative rewrite;
- final status: `not_identified`, `design_plausible_human_review_required`, or
  `assumption_violated`.

## Sources

- Cunningham, *Causal Inference: The Mixtape*:
  https://mixtape.scunning.com/
- Facure, *Causal Inference for the Brave and True*:
  https://matheusfacure.github.io/python-causality-handbook/landing-page
- Goodman-Bacon, “Difference-in-Differences with Variation in Treatment
  Timing”: https://www.nber.org/papers/w25018
