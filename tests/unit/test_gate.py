"""Tests for the regression gate's comparison. No database and no model.

The search itself is replaced: measured_pairs is swapped for a function that
returns prepared outcomes, so each test can say exactly what "search produces
now" and check the verdict.
"""

import pytest

from guide_search.config import Settings
from guide_search.evaluation import gate
from guide_search.evaluation.gate import GateError, committed_pairs, differing, run_gate
from guide_search.evaluation.questions import Question

MODES = ("vector", "keyword", "hybrid", "rerank")

SETTINGS = Settings(threshold=0.0, default_mode="hybrid", rerank_threshold=0.0)


def question(qid: str, expected: tuple[str, ...] = ("a-guide",), reviewed: bool = True) -> Question:
    return Question(
        id=qid,
        language="en",
        question=f"the text of {qid}",
        expected_slugs=expected,
        answer_headings={slug: ("",) for slug in expected},
        split="dev",
        reviewed=reviewed,
    )


QUESTIONS = [question("en-001"), question("en-002"), question("en-003"), question("na-001", ())]


def document(overrides: dict | None = None, threshold: float = 0.0, rerank_threshold: float = 0.0) -> dict:
    """Committed results in which every answerable question is found and nothing is refused."""
    outcomes = []
    for q in QUESTIONS:
        for method in (*MODES, "title_picking"):
            pair = (overrides or {}).get((method, q.id), (False, q.answerable))
            outcomes.append({"question_id": q.id, "method": method, "refused": pair[0], "hit_at_3": pair[1]})
    return {"threshold": threshold, "rerank_threshold": rerank_threshold, "outcomes": outcomes}


def measuring(overrides: dict | None = None):
    """A stand-in for measured_pairs: the committed picture, with the given changes."""

    def measured(conn, embedder, questions, settings, reranker):
        pairs = {(mode, q.id): (False, q.answerable) for mode in MODES for q in questions}
        pairs.update(overrides or {})
        return pairs

    return measured


def verdict(monkeypatch: pytest.MonkeyPatch, now: dict | None = None, committed: dict | None = None):
    monkeypatch.setattr(gate, "measured_pairs", measuring(now))
    return run_gate(None, None, QUESTIONS, document(committed), SETTINGS, None)


def test_the_gate_passes_when_search_produces_the_committed_results(monkeypatch: pytest.MonkeyPatch):
    passed, lines = verdict(monkeypatch)

    assert passed is True
    assert lines[-1] == "The gate passes."
    assert "vector: 4 questions, 0 differ from the committed results (ok)" in lines


def test_one_differing_question_in_a_mode_is_allowed_and_is_named(monkeypatch: pytest.MonkeyPatch):
    passed, lines = verdict(monkeypatch, now={("hybrid", "en-002"): (False, False)})

    assert passed is True
    assert "hybrid: 4 questions, 1 differ from the committed results (ok)" in lines
    assert any("en-002" in line and "no expected guide in the top three" in line for line in lines)


def test_two_differing_questions_in_one_mode_fail_the_gate(monkeypatch: pytest.MonkeyPatch):
    now = {("hybrid", "en-002"): (False, False), ("hybrid", "en-003"): (False, False)}

    passed, lines = verdict(monkeypatch, now=now)

    assert passed is False
    assert "hybrid: 4 questions, 2 differ from the committed results (FAIL)" in lines
    assert lines[-1].startswith("The gate fails")


def test_one_difference_in_each_of_two_modes_still_passes(monkeypatch: pytest.MonkeyPatch):
    now = {("vector", "en-001"): (False, False), ("keyword", "en-002"): (False, False)}

    passed, _ = verdict(monkeypatch, now=now)

    assert passed is True


def test_a_change_in_refusing_counts_as_a_difference(monkeypatch: pytest.MonkeyPatch):
    now = {("vector", "na-001"): (True, False), ("vector", "en-001"): (True, False)}

    passed, lines = verdict(monkeypatch, now=now)

    assert passed is False
    assert any("na-001" in line and "now refused" in line for line in lines)


def test_an_improvement_also_counts_as_a_difference(monkeypatch: pytest.MonkeyPatch):
    # Two questions that were misses in the committed results are now found.
    committed = {("keyword", "en-001"): (False, False), ("keyword", "en-002"): (False, False)}

    passed, lines = verdict(monkeypatch, committed=committed)

    assert passed is False  # the committed report no longer describes the code: rebuild it
    assert "keyword: 4 questions, 2 differ from the committed results (FAIL)" in lines


def test_title_picking_and_unreviewed_questions_are_not_gated():
    questions = [*QUESTIONS, question("ar-001", reviewed=False)]

    pairs = committed_pairs(document(), [q for q in questions if q.reviewed])

    assert {mode for mode, _ in pairs} == set(MODES)
    assert ("vector", "ar-001") not in pairs


def test_a_reviewed_question_with_no_committed_entry_stops_the_gate():
    extra = question("en-099")

    with pytest.raises(GateError, match="en-099"):
        committed_pairs(document(), [*QUESTIONS, extra])


def test_a_threshold_that_is_not_the_committed_one_stops_the_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gate, "measured_pairs", measuring())

    with pytest.raises(GateError, match="threshold"):
        run_gate(None, None, QUESTIONS, document(threshold=0.84), SETTINGS, None)


def test_a_rerank_threshold_that_is_not_the_committed_one_stops_the_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gate, "measured_pairs", measuring())

    with pytest.raises(GateError, match="rerank_threshold"):
        run_gate(None, None, QUESTIONS, document(rerank_threshold=0.5), SETTINGS, None)


def test_rerank_is_gated_like_the_other_modes(monkeypatch: pytest.MonkeyPatch):
    now = {("rerank", "en-001"): (False, False), ("rerank", "en-002"): (False, False)}

    passed, lines = verdict(monkeypatch, now=now)

    assert passed is False
    assert "rerank: 4 questions, 2 differ from the committed results (FAIL)" in lines


def test_differing_lists_only_the_questions_of_the_mode_asked_for():
    committed = {("vector", "en-001"): (False, True), ("hybrid", "en-001"): (False, True)}
    measured = {("vector", "en-001"): (False, False), ("hybrid", "en-001"): (False, True)}

    assert differing(committed, measured, "vector") == ["en-001"]
    assert differing(committed, measured, "hybrid") == []
