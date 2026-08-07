"""The replay harness must fail loud, not quiet.

Its whole value is that a sentence cannot change without someone reading it, so
the two ways it could betray that -- approving text nobody reviewed, and hiding
the change inside a fixed-width window -- are what these pin.
"""

from __future__ import annotations

from .binder_replay import ReplayCase, _from_divergence, drift


def _case(**overrides: object) -> ReplayCase:
    values: dict[str, object] = {
        "session": "sess_demo",
        "question_id": "q_demo",
        "question": "Which venues host the most matches?",
        "stored": ["top is Lusail (matches 8)"],
        "replayed": ["top is Lusail (matches 8, row_count 8)"],
    }
    values.update(overrides)
    return ReplayCase(**values)  # type: ignore[arg-type]


def test_a_case_the_golden_has_never_seen_is_reported() -> None:
    """A fresh checkout must re-report the whole history, not bless it.

    Defaulting an unseen key to "approved" would make the harness silent exactly
    when it has never checked anything.
    """
    assert drift([_case()], {}) == [_case()]


def test_a_case_matching_the_golden_is_quiet_even_though_history_differs() -> None:
    """`stored` is what was published once; the golden is what was last read."""
    reviewed = {"sess_demo:q_demo": ["top is Lusail (matches 8, row_count 8)"]}

    assert drift([_case()], reviewed) == []


def test_a_case_that_moved_past_the_golden_is_reported_again() -> None:
    reviewed = {"sess_demo:q_demo": ["something the binder no longer says"]}

    assert drift([_case()], reviewed) == [_case()]


def test_an_error_is_never_silenced_by_a_matching_golden() -> None:
    """A binder that now raises on stored rows is the loudest possible signal."""
    case = _case(replayed=[], error="ValueError: boom")
    reviewed = {"sess_demo:q_demo": []}

    assert drift([case], reviewed) == [case]


def test_verdicts_separate_a_lost_sentence_from_a_reworded_one() -> None:
    assert _case().verdict == "changed"
    assert _case(replayed=[]).verdict == "vanished"
    assert _case(stored=[]).verdict == "appeared"
    assert _case(replayed=_case().stored).verdict == "same"


def test_the_diff_window_opens_where_the_texts_part() -> None:
    """Findings repeat the question first, so a window off the front shows the
    same prose twice and hides the change underneath it."""
    shared = "Which venues host the most matches? Ranked by matches descending: "
    stored, replayed = _from_divergence(
        [shared + "top is Lusail"], [shared + "top is Al Bayt"], width=40
    )

    assert "Lusail" in stored[0]
    assert "Al Bayt" in replayed[0]
    assert stored[0].startswith("…")
