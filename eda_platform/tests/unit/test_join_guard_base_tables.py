"""The join guard must count tables, not the word "join".

2026-08-04 FIFA run 2: three of ten auto-executed questions died on
"SQL joins tables but the question declares no required_relations." Every one of
them read a single table and joined a CTE back to it -- the pattern DuckDB users
write for "compare each row against a benchmark". The whitelist was empty
because relationship discovery is deferred by default, so no declaration could
have satisfied the guard and there is no repair loop behind it.

The SQL below is the live text of those three, abridged to the join shape.
"""

from __future__ import annotations

from eda_platform.core.tool_guard import check_sql_joins_declared, sql_base_tables

# venues + a percentile CTE, cross-joined to band each venue's elevation.
_VENUE_SQL = """
WITH venue_stats AS (
    SELECT quantile_cont(elevation_meters, 0.33) AS elevation_p33
    FROM venues WHERE capacity IS NOT NULL
), ranked_venues AS (
    SELECT v.venue_id, v.capacity,
        CASE WHEN v.elevation_meters < s.elevation_p33 THEN 'low' ELSE 'high' END AS band
    FROM venues v CROSS JOIN venue_stats s
)
SELECT * FROM ranked_venues
"""

# match_team_stats aggregated, then compared against its own medians.
_TEAM_STYLE_SQL = """
WITH team_agg AS (
    SELECT team_id, AVG(possession_pct) AS avg_possession_pct
    FROM match_team_stats GROUP BY team_id
), benchmarks AS (
    SELECT quantile_cont(avg_possession_pct, 0.5) AS median_possession_pct FROM team_agg
)
SELECT t.team_id, t.avg_possession_pct
FROM team_agg t CROSS JOIN benchmarks b
"""

# match_lineups grouped twice and joined to itself for a per-player share.
_USAGE_SQL = """
WITH profile_stats AS (
    SELECT player_id, is_starting_xi, COUNT(*) AS matches_in_profile
    FROM match_lineups GROUP BY player_id, is_starting_xi
), player_totals AS (
    SELECT player_id, SUM(matches_in_profile) AS player_matches
    FROM profile_stats GROUP BY player_id
)
SELECT p.player_id, p.matches_in_profile, t.player_matches
FROM profile_stats p JOIN player_totals t ON p.player_id = t.player_id
"""

# The live venues query built prose guidance in a CASE arm. "transitions from
# low-elevation venues" reads as a table reference to anything scanning the raw
# text, and "join" inside a literal would trip the keyword test the same way.
_LITERAL_PROSE_SQL = """
WITH venue_stats AS (
    SELECT quantile_cont(elevation_meters, 0.67) AS elevation_p67 FROM venues
)
SELECT v.venue_id,
    CASE WHEN v.elevation_meters >= s.elevation_p67
        THEN 'Plan recovery; avoid abrupt transitions from low-elevation venues.'
        ELSE 'Standard assignment; teams join squads at the usual time.'
    END AS assignment_guidance
FROM venues v CROSS JOIN venue_stats s
"""

_SINGLE_TABLE_SQL = (_VENUE_SQL, _TEAM_STYLE_SQL, _USAGE_SQL, _LITERAL_PROSE_SQL)

# Two base tables: exactly what the guard exists to stop.
_CROSS_TABLE_SQL = """
SELECT m.match_id, v.stadium_name
FROM matches m JOIN venues v ON m.venue_id = v.venue_id
"""

_CROSS_TABLE_SUBQUERY_SQL = """
WITH recent AS (SELECT * FROM matches WHERE date > '2026-06-20')
SELECT r.match_id, v.stadium_name
FROM recent r JOIN venues v ON r.venue_id = v.venue_id
"""


def _violations(sql: str, *, relations: list[str] | None = None):
    return check_sql_joins_declared(
        "plan.sql",
        sql,
        required_relations=relations or [],
        confirmed_joins=[],
    )


def test_a_cte_join_over_one_table_needs_no_relation() -> None:
    for sql in _SINGLE_TABLE_SQL:
        assert _violations(sql) == [], sql


def test_a_second_base_table_is_still_refused() -> None:
    violations = _violations(_CROSS_TABLE_SQL)
    assert len(violations) == 1
    assert "required_relations" in violations[0].problem


def test_a_cte_does_not_launder_a_second_base_table() -> None:
    """Wrapping one side in a CTE must not buy an undeclared cross-table join."""
    assert len(_violations(_CROSS_TABLE_SUBQUERY_SQL)) == 1


def test_an_unconfirmed_relation_is_still_refused() -> None:
    """The whitelist check is independent of how many tables were counted."""
    violations = _violations(_VENUE_SQL, relations=["venues__matches"])
    assert len(violations) == 1
    assert "confirmed join" in violations[0].problem


# The live credit-card query: one table, pivoted to long form with a lateral
# VALUES list. `LATERAL` is a modifier, not a relation.
_LATERAL_SQL = """
WITH base AS (
    SELECT id, Class, Amount, V1, V2 FROM creditcard_2023
), signal_long AS (
    SELECT b.id, b.Class, s.signal_name, s.signal_value
    FROM base b
    CROSS JOIN LATERAL (VALUES
        ('Amount', CAST(b.Amount AS DOUBLE)),
        ('V1', CAST(b.V1 AS DOUBLE)),
        ('V2', CAST(b.V2 AS DOUBLE))
    ) AS s(signal_name, signal_value)
    WHERE s.signal_value IS NOT NULL
)
SELECT signal_name, corr(signal_value, Class) AS c FROM signal_long GROUP BY signal_name
"""

# Table functions read like table references to a scanner but name no relation.
_TABLE_FUNCTION_SQL = """
SELECT t.value, m.amount
FROM unnest([1, 2, 3]) AS t(value)
JOIN (SELECT 1 AS value, 2 AS amount) m ON m.value = t.value
"""


def test_a_lateral_modifier_is_not_a_table() -> None:
    """2026-08-05 credit-card run: a single-table query, refused.

    `CROSS JOIN LATERAL (VALUES ...)` pivots one table to long form. The scan
    read `LATERAL` as the second relation and the guard killed the question.
    """
    assert sql_base_tables(_LATERAL_SQL) == {"creditcard_2023"}
    assert _violations(_LATERAL_SQL) == []


def test_a_table_function_is_not_a_table() -> None:
    assert "unnest" not in sql_base_tables(_TABLE_FUNCTION_SQL)


def test_a_modifier_does_not_hide_the_table_behind_it() -> None:
    """Skipping the keyword must not skip the relation it modifies."""
    assert sql_base_tables("select * from only orders") == {"orders"}
