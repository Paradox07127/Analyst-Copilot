# Blind question review rubric v1

Evaluate only the supplied question and dataset context. Scores are integers 1–5.
Unknown context must not be replaced by assumed facts or by imagined business value.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| answerability | Required data is absent or the question is not answerable | Partially answerable with material limitations | Available data and methods fully support a concrete answer |
| business_value | Tautology or unrelated to the stated goal | Useful monitoring with limited decision impact | Directly informs a specified decision |
| specificity | No identifiable entity, metric or scope | Entity is clear but metric or comparison is ambiguous | Entity, metric, scope and expected comparison are explicit |
| data_support | Evidence is absent or assumptions invalid | Material missingness, sample or join limitations | Data quality, grain and validated relationships support the analysis |

Use 2 or 4 for intermediate cases. Reject when answerability <= 2 or data_support <= 2;
other high scores do not cancel this veto. Distinguish an invalid question from insufficient
review context: when there is not enough information to grade, return insufficient_context.
Do not treat ID-like columns as measures, observational associations as causal effects,
or a query's successful execution as proof that it answers the question.

In the rationale identify the main evidence gap, duplicate analysis, or decision relevance
when the supplied context supports that judgment. Additional value beyond existing EDA is
important, but do not invent an EDA baseline when none is supplied.
