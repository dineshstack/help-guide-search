"""Tests for the agent evaluation: how results are sorted, stored and reported.

No model, no database. The questions and the stored lines are written by hand.
"""

import pytest

from guide_search.agent import AgentResult, CheckSummary
from guide_search.drafting import ModelCall
from guide_search.evaluation import agent_run
from guide_search.evaluation.agent_run import (
    ANSWERED_OTHER,
    ANSWERED_RIGHT,
    ANSWERED_UNANSWERABLE,
    DECLINED,
    current_lines,
    estimated_cost,
    instructions_hash,
    line_for,
    outcome_of,
    render_report,
    text_hash,
)
from guide_search.evaluation.questions import Question
from guide_search.ranking import GuideSummary

MODEL = "claude-opus-5-5"


def question(qid: str, expected: tuple[str, ...], split: str = "dev", reviewed: bool = True) -> Question:
    return Question(
        id=qid,
        language="en",
        question=f"the text of {qid}",
        expected_slugs=expected,
        answer_headings={slug: ("",) for slug in expected},
        split=split,
        reviewed=reviewed,
    )


def stored(q: Question, decision: str, cited: list[str], model: str = MODEL, **changes) -> dict:
    line = {
        "question_id": q.id,
        "question_hash": text_hash(q.question),
        "instructions_hash": instructions_hash(),
        "model": model,
        "effort": "medium",
        "split": q.split,
        "date": "2026-10-11",
        "decision": decision,
        "cited_slugs": cited,
        "hand_over_reason": None if decision != "hand_over" else "the_guides_do_not_answer_this",
        "attempts": 1,
        "failures": [],
        "model_calls": 1,
        "input_tokens": 1000,
        "output_tokens": 200,
        "seconds": 3.0,
    }
    line.update(changes)
    return line


RIGHT = question("en-001", ("share-a-link",))
TWO_HOMES = question("en-002", ("fix-links", "share-a-link"))
MISSED = question("en-003", ("count-stock",))
NO_GUIDE = question("na-001", ())
NO_GUIDE_2 = question("na-002", ())


def test_an_answer_citing_any_expected_guide_is_right():
    assert outcome_of(RIGHT, stored(RIGHT, "answer", ["share-a-link"])) == ANSWERED_RIGHT
    assert outcome_of(TWO_HOMES, stored(TWO_HOMES, "answer", ["other", "share-a-link"])) == ANSWERED_RIGHT


def test_an_answer_citing_only_other_guides_is_counted_apart():
    assert outcome_of(MISSED, stored(MISSED, "answer", ["share-a-link"])) == ANSWERED_OTHER


def test_declining_an_answerable_question_is_counted_as_declined():
    assert outcome_of(RIGHT, stored(RIGHT, "hand_over", [])) == DECLINED
    assert outcome_of(RIGHT, stored(RIGHT, "clarify", [])) == DECLINED


def test_for_an_unanswerable_question_declining_is_right_and_answering_is_wrong():
    assert outcome_of(NO_GUIDE, stored(NO_GUIDE, "hand_over", [])) == DECLINED
    assert outcome_of(NO_GUIDE, stored(NO_GUIDE, "clarify", [])) == DECLINED
    assert outcome_of(NO_GUIDE, stored(NO_GUIDE, "answer", ["share-a-link"])) == ANSWERED_UNANSWERABLE


def test_a_line_from_an_edited_question_or_older_instructions_is_not_counted():
    questions = [RIGHT, MISSED, NO_GUIDE]
    lines = [
        stored(RIGHT, "answer", ["share-a-link"]),
        stored(MISSED, "answer", ["count-stock"], question_hash=text_hash("an older wording")),
        stored(NO_GUIDE, "hand_over", [], instructions_hash="an older version"),
    ]

    kept = current_lines(lines, questions)

    # A line stored before reranking was added names no mode: it was a hybrid search.
    assert list(kept) == [(MODEL, "hybrid", "en-001")]


def test_the_latest_line_for_a_question_and_model_wins_and_unreviewed_questions_are_left_out():
    unreviewed = question("ar-001", ("share-a-link",), reviewed=False)
    lines = [
        stored(RIGHT, "hand_over", []),
        stored(RIGHT, "answer", ["share-a-link"]),
        stored(unreviewed, "answer", ["share-a-link"]),
    ]

    kept = current_lines(lines, [RIGHT, unreviewed])

    assert list(kept) == [(MODEL, "hybrid", "en-001")]
    assert kept[(MODEL, "hybrid", "en-001")]["decision"] == "answer"


def test_a_stored_line_sums_the_tokens_of_every_call_and_names_the_cited_slugs():
    def call(purpose: str) -> ModelCall:
        return ModelCall(purpose, MODEL, 900, 100, None, None, "end_turn", 2.5)

    result = AgentResult(
        question=RIGHT.question,
        decision="answer",
        reply="It lasts 48 hours.",
        sources=[GuideSummary(3, "share-a-link", "Share a link", "https://example.test/share-a-link")],
        hand_over_reason=None,
        check=CheckSummary(attempts=2, failures=["unsupported_number"]),
        usage=[call("draft"), call("rewrite")],
    )

    line = line_for(RIGHT, result, MODEL, "medium", "2026-10-11")

    assert line["cited_slugs"] == ["share-a-link"]
    assert (line["input_tokens"], line["output_tokens"], line["model_calls"]) == (1800, 200, 2)
    assert line["attempts"] == 2 and line["failures"] == ["unsupported_number"]
    assert line["seconds"] == 5.0
    assert "It lasts 48 hours." not in str(line)  # the reply text is not stored


def test_cost_is_tokens_times_the_listed_price():
    # 1,000,000 input at $4 and 100,000 output at $20 per million.
    assert estimated_cost("claude-opus-5-5", 1_000_000, 100_000) == 6.0
    assert estimated_cost("a-model-with-no-listed-price", 1000, 1000) is None


def test_the_report_counts_each_kind_per_model_and_lists_what_went_wrong():
    questions = [RIGHT, TWO_HOMES, MISSED, NO_GUIDE, NO_GUIDE_2]
    lines = [
        stored(RIGHT, "answer", ["share-a-link"]),
        stored(TWO_HOMES, "hand_over", []),
        stored(MISSED, "answer", ["share-a-link"], attempts=2, failures=["unsupported_number"]),
        stored(NO_GUIDE, "hand_over", []),
        stored(NO_GUIDE_2, "answer", ["share-a-link"]),
    ]

    report = render_report(questions, lines)
    row = next(line for line in report.splitlines() if line.startswith(f"| `{MODEL}`"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]

    assert cells[1] == "1 of 3 (33%)"  # answered, citing an expected guide
    assert cells[2] == "1 of 3 (33%)"  # answered, citing only other guides
    assert cells[3] == "1 of 3 (33%)"  # handed over or asked back
    assert cells[4] == "1 of 2 (50%)"  # unanswerable: declined
    assert cells[5] == "1 of 2 (50%)"  # unanswerable: answered
    assert cells[6] == "1 of 5 (20%)"  # needed a rewrite
    assert cells[7] == "1000 / 200"
    assert cells[8] == "$0.80"  # (1000 * 4 + 200 * 20) / 1e6 dollars per question, times 100
    # The three unwanted results are listed; the two wanted ones are not.
    for unwanted in (TWO_HOMES, MISSED, NO_GUIDE_2):
        assert f"`{unwanted.id}`" in report
    assert "`en-001`" not in report and "`na-001`" not in report


def test_a_model_that_was_not_run_on_every_question_gets_no_numbers():
    questions = [RIGHT, MISSED]
    lines = [stored(RIGHT, "answer", ["share-a-link"])]

    report = render_report(questions, lines)

    assert "not run on every question of this split (1 of 2)" in report


def test_the_report_states_its_limits_and_that_arabic_is_not_scored():
    report = render_report([RIGHT, question("ar-001", ("x",), reviewed=False)], [stored(RIGHT, "answer", ["share-a-link"])])

    assert "is not the same as answering correctly" in report
    assert "1 not reviewed and not scored" in report


def test_run_asks_only_the_reviewed_questions_of_the_split_that_have_no_current_line(tmp_path, monkeypatch):
    from guide_search.evaluation.questions import save_questions

    monkeypatch.chdir(tmp_path)
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "agent.toml").write_text('model = "claude-x"\neffort = "medium"\nmax_tokens = 4000\n')
    asked_before = question("en-010", ("share-a-link",))
    to_ask = question("en-011", ("share-a-link",))
    other_split = question("en-012", ("share-a-link",), split="test")
    unreviewed = question("ar-010", ("share-a-link",), reviewed=False)
    save_questions(agent_run.QUESTIONS_PATH, [asked_before, to_ask, other_split, unreviewed])
    agent_run.RESULTS_PATH.write_text(__import__("json").dumps(stored(asked_before, "answer", ["share-a-link"])) + "\n")
    asked: list[str] = []

    class StubService:
        def answer(self, text: str) -> AgentResult:
            asked.append(text)
            return AgentResult(text, "hand_over", "", [], "the_guides_do_not_answer_this", CheckSummary(1, []), [])

    code = agent_run.run("dev", MODEL, make_service=StubService, today="2026-10-11")

    assert code == 0
    assert asked == [to_ask.question]
    lines = agent_run.load_lines()
    assert [line["question_id"] for line in lines] == ["en-010", "en-011"]
    assert lines[1]["model"] == MODEL and lines[1]["decision"] == "hand_over"


def test_the_same_model_with_two_search_modes_gives_two_rows():
    questions = [RIGHT, MISSED]
    lines = [
        stored(RIGHT, "answer", ["share-a-link"]),
        stored(MISSED, "hand_over", []),
        stored(RIGHT, "answer", ["share-a-link"], mode="rerank"),
        stored(MISSED, "answer", ["count-stock"], mode="rerank"),
    ]

    report = render_report(questions, lines)

    hybrid = next(line for line in report.splitlines() if line.startswith(f"| `{MODEL}`, hybrid |"))
    rerank = next(line for line in report.splitlines() if line.startswith(f"| `{MODEL}`, rerank |"))
    assert "| 1 of 2 (50%) | 0 of 2 (0%) | 1 of 2 (50%) |" in hybrid
    assert "| 2 of 2 (100%) | 0 of 2 (0%) | 0 of 2 (0%) |" in rerank


def bar_questions(split: str) -> list[Question]:
    """20 answerable questions and 2 unanswerable ones in one split."""
    answerable = [question(f"en-{number:03}", ("share-a-link",), split=split) for number in range(20)]
    return answerable + [question("na-001", (), split=split), question("na-002", (), split=split)]


@pytest.mark.parametrize("declined, met", [(3, "yes"), (4, "no")])
def test_the_bar_allows_three_of_twenty_answerable_questions_declined(declined: int, met: str):
    questions = bar_questions("test")
    lines = []
    for number, q in enumerate(questions):
        if not q.answerable:
            lines.append(stored(q, "hand_over", [], mode="rerank"))
        elif number < declined:
            lines.append(stored(q, "hand_over", [], mode="rerank"))
        else:
            lines.append(stored(q, "answer", ["share-a-link"], mode="rerank"))

    report = render_report(questions, lines)

    bar = next(line for line in report.splitlines() if line.startswith(f"| `{MODEL}`, rerank | test |"))
    assert bar == f"| `{MODEL}`, rerank | test | {declined} of 20 ({declined * 5}%) | 3 | 2 of 2 (100%) | {met} |"


def test_the_bar_is_not_met_when_an_unanswerable_question_is_answered():
    questions = bar_questions("fresh")
    lines = [stored(q, "answer", ["share-a-link"], mode="rerank") for q in questions]

    report = render_report(questions, lines)

    bar = next(line for line in report.splitlines() if line.startswith(f"| `{MODEL}`, rerank | fresh |"))
    assert bar.endswith("| 0 of 2 (0%) | no |")


def test_a_report_without_a_rerank_run_has_no_bar_section():
    report = render_report([RIGHT], [stored(RIGHT, "answer", ["share-a-link"])])

    assert "## The bar for relying on search" not in report
