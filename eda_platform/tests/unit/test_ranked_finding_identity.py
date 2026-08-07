"""A ranked claim must name which row it ranked.

2026-08-04 FIFA run 2: the label was the first non-numeric column, so a result
whose rows are identified by two columns lost half its identity. Two failure
modes, both observed live:

- The dropped half made the leaders indistinguishable ("UEFA, UEFA, UEFA"), and
  the ranking was discarded rather than labelled properly.
- The kept half was coincidentally distinct but was the wrong column: a
  UNION-of-dimensions result was published as "top is result_type", naming the
  dimension instead of the value inside it.

Row shapes are the live ones from sess_1785876997709_wtkcwm.
"""

from __future__ import annotations

from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import _findings_for
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore


def _sql_artifact(rows: list[dict[str, object]], *, sql: str, truncated: bool = False) -> Artifact:
    columns = list(rows[0]) if rows else []
    payload = SqlResult(
        sql=sql,
        columns=columns,
        dtypes=dict.fromkeys(columns, "DOUBLE"),
        rows_preview=rows,
        row_count=len(rows),
        truncated=truncated,
    ).model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _candidate(question: str) -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_demo",
        question_en=question,
        origin="llm",
        target_datasets=["matches.csv"],
        exploratory=True,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.775,
        ),
    )


# Three GROUP BYs unioned into one dimension/value pair, ordered by a summed
# magnitude. The group key is the pair, not either column alone.
_XG_SQL = (
    "with base as (select 1) , aggregates as ("
    "select 'stage' as context_dimension, stage_context as context_value, "
    "count(*) as matches, abs(sum(gap)) as absolute_gap from base group by stage_context "
    "union all "
    "select 'result_type', result_context, count(*), abs(sum(gap)) "
    "from base group by result_context "
    "union all "
    "select 'venue_context', venue_context, count(*), abs(sum(gap)) "
    "from base group by venue_context) "
    "select context_dimension, context_value, matches, absolute_gap, "
    "average_gap_per_match from aggregates "
    "order by absolute_gap desc, matches desc"
)


def _xg(
    dimension: str, value: str, matches: int, gap: float, per_match: float
) -> dict[str, object]:
    return {
        "context_dimension": dimension,
        "context_value": value,
        "matches": matches,
        "absolute_gap": gap,
        "average_gap_per_match": per_match,
    }


# All 13 live rows: fewer would fall into the distribution-share branch instead
# of the ranked one, and the defect under test lives in the ranked branch.
_XG_ROWS: list[dict[str, object]] = [
    _xg("result_type", "Regular", 95, 28.42, -0.2992),
    _xg("stage", "1", 72, 23.66, -0.3286),
    _xg("venue_context", "neutral", 93, 21.1, -0.2269),
    _xg("venue_context", "home_host", 8, 7.4, -0.925),
    _xg("result_type", "AET", 5, 5.19, -1.038),
    _xg("stage", "6", 1, 4.24, -4.24),
    _xg("venue_context", "away_host", 3, 3.5, -1.1667),
    _xg("stage", "4", 4, 2.06, -0.515),
    _xg("result_type", "Penalties", 4, 1.61, 0.4025),
    _xg("stage", "5", 2, 0.73, -0.365),
    _xg("stage", "2", 16, 0.58, -0.0363),
    _xg("stage", "7", 1, 0.39, -0.39),
    _xg("stage", "3", 8, 0.34, -0.0425),
]

_CONFED_SQL = (
    "with group_strength as (select 1) "
    "select confederation, group_letter, team_count, avg_elo_rating, elo_range, "
    "concentration_mismatch_risk_score from group_strength "
    "order by concentration_mismatch_risk_score desc, elo_range desc"
)

_CONFED_ROWS: list[dict[str, object]] = [
    {
        "confederation": "UEFA",
        "group_letter": "I",
        "team_count": 2,
        "avg_elo_rating": 1937.5,
        "elo_range": 325,
        "concentration_mismatch_risk_score": 1.9124,
    },
    {
        "confederation": "UEFA",
        "group_letter": "B",
        "team_count": 2,
        "avg_elo_rating": 1752.5,
        "elo_range": 215,
        "concentration_mismatch_risk_score": 1.207,
    },
    {
        "confederation": "UEFA",
        "group_letter": "F",
        "team_count": 2,
        "avg_elo_rating": 1897.5,
        "elo_range": 165,
        "concentration_mismatch_risk_score": 1.1173,
    },
    {
        "confederation": "CONMEBOL",
        "group_letter": "J",
        "team_count": 2,
        "avg_elo_rating": 2150.0,
        "elo_range": 100,
        "concentration_mismatch_risk_score": 0.4,
    },
]


def _finding_text(candidate: QuestionCandidate, artifact: Artifact) -> str:
    findings = _findings_for(candidate, artifact)
    assert findings, "a result with rows must produce a finding"
    return findings[0].text


def test_a_union_of_dimensions_names_the_value_not_the_dimension() -> None:
    """The live defect: "top is result_type" hid which result type."""
    text = _finding_text(
        _candidate("Which match contexts show the largest scoring gaps?"),
        _sql_artifact(_XG_ROWS, sql=_XG_SQL),
    )
    assert "Regular" in text, text
    # The dimension name alone must never stand in for the row it labels.
    assert "top is result_type (" not in text, text


def test_a_repeated_first_column_widens_the_label_instead_of_giving_up() -> None:
    """Three UEFA groups in a row used to discard the whole ranking."""
    text = _finding_text(
        _candidate("Where are the largest competitive mismatch risks?"),
        _sql_artifact(_CONFED_ROWS, sql=_CONFED_SQL),
    )
    assert "no ranking basis" not in text, text
    assert "UEFA" in text and "I" in text, text


def test_a_truncated_ranking_discloses_that_it_is_only_the_leaders() -> None:
    """Widening the label must not become a licence to overstate a partial result.

    This asserted "refuses to rank" until 2026-08-06. The premise was wrong:
    `sql_runner` previews with `select * from (<statement>) limit n`, so on an
    ordering the engine applied, the preview is the head and the leaders hold.
    What truncation really costs is everything about the rest of the result, so
    that is what has to be said out loud.
    """
    text = _finding_text(
        _candidate("Which player profiles emerge?"),
        _sql_artifact(_CONFED_ROWS, sql=_CONFED_SQL, truncated=True),
    )
    assert "Ranked by" in text, text
    assert "the result was cut off" in text, text


def test_c_a_truncated_summary_reports_the_real_row_count() -> None:
    """ "Returned 50 rows" for a 264-row result is simply false.

    The preview length was being narrated as the result size, and the arbitrary
    first row of that preview was published as an "example".
    """
    rows = _CONFED_ROWS
    payload = SqlResult(
        sql="select confederation, group_letter from x",
        columns=list(rows[0]),
        dtypes=dict.fromkeys(rows[0], "DOUBLE"),
        rows_preview=rows,
        row_count=264,
        truncated=True,
    ).model_dump(mode="json")
    artifact = Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )
    text = _finding_text(_candidate("Which player profiles emerge?"), artifact)
    assert "264" in text, text
    assert "Returned 4 rows" not in text, text


def test_c_a_complete_unordered_result_describes_its_spread() -> None:
    """No ORDER BY still leaves order-independent facts on the table.

    Min and max over a complete result are computed, not read off row order, so
    they are claimable where a ranking is not.
    """
    text = _finding_text(
        _candidate("What contribution segments appear within each position?"),
        _sql_artifact(_CONFED_ROWS, sql="select confederation, avg_elo_rating from x"),
    )
    assert "no ranking basis" not in text, text
    # CONMEBOL/J holds the highest average Elo of the four groups.
    assert "2150" in text or "2,150" in text, text


def test_b_a_summed_ranking_metric_discloses_the_group_size() -> None:
    """ABS(SUM(...)) ranks by sample size as much as by effect.

    Regular (95 matches) outranks the final (1 match) on the summed gap while
    having the smaller per-match gap, so the count belongs in the sentence the
    reader sees.
    """
    text = _finding_text(
        _candidate("Which match contexts show the largest scoring gaps?"),
        _sql_artifact(_XG_ROWS, sql=_XG_SQL),
    )
    assert "95" in text, text


# The live team-ranking query: window functions put an ORDER BY inside each
# RANK() OVER (...), and the outer ORDER BY is the one that sorted the result.
_WINDOW_SQL = (
    "with scored as ("
    "select team_id, team_name, confederation, "
    "rank() over (order by fifa_ranking_pre_tournament asc) as fifa_rank_derived, "
    "rank() over (order by elo_rating desc) as elo_rank_derived, "
    "abs(1) as absolute_rank_discrepancy from teams) "
    "select team_id, team_name, confederation, fifa_rank_derived, elo_rank_derived, "
    "absolute_rank_discrepancy, "
    "count(*) over (partition by confederation) as confederation_team_count "
    "from scored order by absolute_rank_discrepancy desc, team_name asc"
)


def _team(team_id: int, name: str, confederation: str, gap: int) -> dict[str, object]:
    return {
        "team_id": team_id,
        "team_name": name,
        "confederation": confederation,
        "fifa_rank_derived": team_id,
        "elo_rank_derived": team_id + gap,
        "absolute_rank_discrepancy": gap,
        "confederation_team_count": 6,
    }


_TEAM_ROWS: list[dict[str, object]] = [
    _team(1, "Mexico", "CONCACAF", 9),
    _team(2, "Canada", "CONCACAF", 8),
    _team(3, "Japan", "AFC", 6),
    _team(48, "Panama", "CONCACAF", 1),
]


def test_d_a_window_function_does_not_hide_the_real_ordering() -> None:
    """2026-08-05 live run: three ORDER BY tokens, only one of them the result's.

    The parser bailed on the count, the ranking was discarded, and the fallback
    published "team_id ranges from 1 to 48, from Mexico to Panama" -- in the
    Executive Summary, as the report's opening finding.
    """
    text = _finding_text(
        _candidate("Which teams show the largest ranking discrepancies?"),
        _sql_artifact(_TEAM_ROWS, sql=_WINDOW_SQL),
    )
    assert "absolute_rank_discrepancy" in text, text
    assert "Mexico" in text, text
    assert "ranges from" not in text, text


def test_d_a_spread_never_measures_an_identifier() -> None:
    """`team_id` is a name, so its range says nothing; the safety net said it."""
    text = _finding_text(
        _candidate("Which teams show the largest ranking discrepancies?"),
        _sql_artifact(_TEAM_ROWS, sql="select team_id, absolute_rank_discrepancy from x"),
    )
    assert "team_id ranges" not in text, text


# The live Olist fraud-screen query: 2109 matching payments, ordered, preview
# capped at 50. `sql_runner` builds the preview as
# `select * from (<statement>) limit 51`, so a truncated preview is the *head*
# of the ordering, not a sample of it.
_TRUNCATED_SQL = (
    "SELECT op.order_id, op.payment_installments FROM olist_order_payments_dataset op "
    "ORDER BY op.payment_installments DESC"
)
_TRUNCATED_ROWS: list[dict[str, object]] = [
    {"order_id": f"o{index:03d}", "payment_installments": 24 - index} for index in range(50)
]


def test_a_truncated_ranking_still_names_its_leaders() -> None:
    """Cutting off the tail cannot change who is first.

    A 2026-08-06 replay of the stored runs caught this: four questions that had
    named their top rows went silent, because the partial-result guard refused a
    ranking and a range together. Truncation costs the range -- the smallest
    value is behind the cut -- and leaves the leaders provable.
    """
    artifact = _sql_artifact(_TRUNCATED_ROWS, sql=_TRUNCATED_SQL, truncated=True)
    artifact.payload["row_count"] = 2109

    finding = _findings_for(_candidate("Which payments look anomalous?"), artifact)[0]

    assert "Ranked by payment_installments descending" in finding.text
    assert "top is o000" in finding.text
    # ... and it must not let the reader take the leaders for the whole result.
    assert "2109" in finding.text


def test_a_truncated_ranking_does_not_describe_a_spread_it_cannot_see() -> None:
    """The bunching note is computed over the preview; say nothing rather than
    describe the whole result from its first fifty rows."""
    artifact = _sql_artifact(_TRUNCATED_ROWS, sql=_TRUNCATED_SQL, truncated=True)
    artifact.payload["row_count"] = 2109

    finding = _findings_for(_candidate("Which payments look anomalous?"), artifact)[0]

    assert "closely bunched" not in finding.text
    assert "ranges from" not in finding.text


def test_an_unordered_truncated_result_still_refuses_to_rank() -> None:
    """No ORDER BY means the preview's first row leads nothing."""
    artifact = _sql_artifact(
        _TRUNCATED_ROWS,
        sql="SELECT op.order_id, op.payment_installments FROM olist_order_payments_dataset op",
        truncated=True,
    )
    artifact.payload["row_count"] = 2109

    finding = _findings_for(_candidate("Which payments look anomalous?"), artifact)[0]

    assert "Ranked by" not in finding.text
    assert "only the first 50 were kept" in finding.text
