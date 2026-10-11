"""Tests for the answer agent's loop. No model, no database.

The model is replaced by a stand-in that hands out prepared drafts (or
prepared faults) one after another, and search by a function that returns a
prepared result. So each test can say exactly what the model "wrote" and
check what the agent did with it.
"""

import json
from pathlib import Path

import pytest

from guide_search import agent as agent_module
from guide_search.agent import (
    HAND_OVER_REPLY,
    Agent,
    AgentNotConfigured,
    AgentResult,
    CheckSummary,
    format_result,
)
from guide_search.checks import Draft, Failure
from guide_search.drafting import ModelCall, ModelFault
from guide_search.ranking import GuideHit, GuideSummary, PassageHit, SearchResult
from guide_search.store import DatabaseUnavailable

PAGE = "https://support.orions360.com/help/en/autoserv-pro/"

GUIDES = [
    GuideHit(
        guide_id=3,
        slug="share-a-link",
        title="Share a link",
        url=PAGE + "share-a-link",
        score=0.9,
        passages=[PassageHit(heading="When it ends", position=1, text="The link stops working after 48 hours.", similarity=0.9)],
    ),
    GuideHit(
        guide_id=7,
        slug="count-stock",
        title="Count stock",
        url=PAGE + "count-stock",
        score=0.5,
        passages=[PassageHit(heading="Units", position=1, text="Stock is counted in units.", similarity=0.5)],
    ),
]
FOUND = SearchResult(query="q", mode="hybrid", reason=None, best_similarity=0.9, guides=GUIDES)
NOTHING = SearchResult(query="q", mode="hybrid", reason="no_match", best_similarity=0.2, guides=[])

GOOD = Draft(decision="answer", reply="The link stops working after 48 hours.", cited_guide_ids=(3,))
BAD_NUMBER = Draft(decision="answer", reply="The link stops working after 72 hours.", cited_guide_ids=(3,))
CLARIFY = Draft(decision="clarify", reply="Which link do you mean?", cited_guide_ids=())
GIVE_UP = Draft(decision="hand_over", reply="", cited_guide_ids=())


def call_record(purpose: str, stop_reason: str = "end_turn") -> ModelCall:
    return ModelCall(
        purpose=purpose,
        model="claude-test-model",
        input_tokens=900,
        output_tokens=100,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        stop_reason=stop_reason,
        seconds=0.5,
    )


class ScriptedDrafter:
    """Stands in for the model: returns the prepared outcomes in order."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def draft(self, question, guides, previous, failure):
        self.calls.append({"question": question, "guides": guides, "previous": previous, "failure": failure})
        outcome = self.outcomes.pop(0)  # an IndexError here means a call the test did not expect
        purpose = "draft" if previous is None else "rewrite"
        if isinstance(outcome, ModelFault):
            raise outcome
        return outcome, call_record(purpose)


def searching(result: SearchResult):
    def search(question: str) -> SearchResult:
        return result

    return search


def run(drafter: ScriptedDrafter, found: SearchResult = FOUND, question: str = "how long does a link last"):
    return Agent(searching(found), drafter).answer(question)


# ---------------------------------------------------------------------------
# The four ways through the loop
# ---------------------------------------------------------------------------


def test_a_draft_that_passes_first_time_is_returned_with_its_sources():
    drafter = ScriptedDrafter(GOOD)

    result = run(drafter)

    assert result.decision == "answer"
    assert result.reply == "The link stops working after 48 hours."
    assert result.sources == [
        GuideSummary(guide_id=3, slug="share-a-link", title="Share a link", url=PAGE + "share-a-link")
    ]
    assert result.hand_over_reason is None
    assert result.check == CheckSummary(attempts=1, failures=[])
    assert [call.purpose for call in result.usage] == ["draft"]
    assert len(drafter.calls) == 1


def test_a_draft_that_fails_is_rewritten_once_and_the_model_is_told_the_reason():
    drafter = ScriptedDrafter(BAD_NUMBER, GOOD)

    result = run(drafter)

    assert result.decision == "answer"
    assert result.reply == "The link stops working after 48 hours."
    assert result.check == CheckSummary(attempts=2, failures=["unsupported_number"])
    assert [call.purpose for call in result.usage] == ["draft", "rewrite"]
    # The first call saw no earlier draft. The second saw the failed one and why.
    assert drafter.calls[0]["previous"] is None and drafter.calls[0]["failure"] is None
    assert drafter.calls[1]["previous"] == BAD_NUMBER
    assert isinstance(drafter.calls[1]["failure"], Failure)
    assert "72" in drafter.calls[1]["failure"].message


def test_a_draft_that_fails_twice_is_never_returned_and_there_is_no_third_call():
    # Only two outcomes are prepared: a third call would raise in the stand-in.
    drafter = ScriptedDrafter(BAD_NUMBER, BAD_NUMBER)

    result = run(drafter)

    assert result.decision == "hand_over"
    assert result.reply == HAND_OVER_REPLY
    assert "72" not in result.reply
    assert result.sources == []
    assert result.hand_over_reason == "check_failed_twice: unsupported_number"
    assert result.check == CheckSummary(attempts=2, failures=["unsupported_number", "unsupported_number"])
    assert len(drafter.calls) == 2
    assert len(result.usage) == 2


def test_when_the_model_hands_over_the_agent_hands_over_after_one_call():
    drafter = ScriptedDrafter(GIVE_UP)

    result = run(drafter)

    assert result.decision == "hand_over"
    assert result.reply == HAND_OVER_REPLY
    assert result.hand_over_reason == "the_guides_do_not_answer_this"
    assert result.check == CheckSummary(attempts=1, failures=[])
    assert len(drafter.calls) == 1


def test_a_clarifying_question_is_returned_with_no_sources():
    result = run(ScriptedDrafter(CLARIFY))

    assert result.decision == "clarify"
    assert result.reply == "Which link do you mean?"
    assert result.sources == []
    assert result.hand_over_reason is None


def test_a_clarifying_question_that_states_a_fact_is_sent_back_for_a_rewrite():
    with_fact = Draft(decision="clarify", reply="Do you mean the link that lasts 48 hours?", cited_guide_ids=())

    result = run(ScriptedDrafter(with_fact, CLARIFY))

    assert result.decision == "clarify"
    assert result.check.failures == ["clarify_states_a_fact"]


def test_a_rewrite_may_turn_into_a_hand_over():
    result = run(ScriptedDrafter(BAD_NUMBER, GIVE_UP))

    assert result.decision == "hand_over"
    assert result.hand_over_reason == "the_guides_do_not_answer_this"
    assert result.check == CheckSummary(attempts=2, failures=["unsupported_number"])


def test_only_the_cited_guides_are_given_as_sources():
    both = Draft(decision="answer", reply="The link stops working after 48 hours.", cited_guide_ids=(7, 3))

    result = run(ScriptedDrafter(both))

    # In the order search returned them.
    assert [source.guide_id for source in result.sources] == [3, 7]


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------


def test_a_model_fault_hands_over_and_keeps_the_record_of_the_call():
    refused = ModelFault("stopped_refusal", call_record("draft", stop_reason="refusal"))

    result = run(ScriptedDrafter(refused))

    assert result.decision == "hand_over"
    assert result.reply == HAND_OVER_REPLY
    assert result.hand_over_reason == "model_fault: stopped_refusal"
    assert result.check == CheckSummary(attempts=0, failures=[])
    assert [call.stop_reason for call in result.usage] == ["refusal"]


def test_a_model_fault_with_no_answer_from_the_api_hands_over_with_no_record():
    result = run(ScriptedDrafter(ModelFault("api_error_APIConnectionError")))

    assert result.decision == "hand_over"
    assert result.hand_over_reason == "model_fault: api_error_APIConnectionError"
    assert result.usage == []


def test_a_model_fault_during_the_rewrite_hands_over_and_never_returns_the_failed_draft():
    result = run(ScriptedDrafter(BAD_NUMBER, ModelFault("stopped_max_tokens", call_record("rewrite", "max_tokens"))))

    assert result.decision == "hand_over"
    assert "72" not in result.reply
    assert result.hand_over_reason == "model_fault: stopped_max_tokens"
    assert [call.purpose for call in result.usage] == ["draft", "rewrite"]


def test_a_database_fault_is_an_error_and_not_a_hand_over():
    drafter = ScriptedDrafter(GOOD)

    def broken_search(question: str) -> SearchResult:
        raise DatabaseUnavailable("could not connect")

    with pytest.raises(DatabaseUnavailable):
        Agent(broken_search, drafter).answer("how long does a link last")

    assert drafter.calls == []  # the model was never asked


# ---------------------------------------------------------------------------
# When search finds nothing
# ---------------------------------------------------------------------------


def test_with_no_passages_the_model_is_still_asked_and_may_hand_over():
    drafter = ScriptedDrafter(GIVE_UP)

    result = run(drafter, found=NOTHING)

    assert drafter.calls[0]["guides"] == []
    assert result.decision == "hand_over"


def test_with_no_passages_an_answer_cannot_pass_the_check():
    result = run(ScriptedDrafter(GOOD, GOOD), found=NOTHING)

    assert result.decision == "hand_over"
    assert result.hand_over_reason == "check_failed_twice: unknown_citation"


# ---------------------------------------------------------------------------
# No memory, the usage log, and the command line
# ---------------------------------------------------------------------------


def test_the_agent_keeps_nothing_from_one_question_to_the_next():
    drafter = ScriptedDrafter(BAD_NUMBER, GOOD, GOOD)
    agent = Agent(searching(FOUND), drafter)

    agent.answer("first question")
    second = agent.answer("second question")

    assert drafter.calls[2]["question"] == "second question"
    assert drafter.calls[2]["previous"] is None
    assert second.check == CheckSummary(attempts=1, failures=[])
    assert len(second.usage) == 1


def test_every_model_call_is_appended_to_the_usage_log_without_the_question(tmp_path: Path):
    log = tmp_path / "usage.jsonl"
    agent = Agent(searching(FOUND), ScriptedDrafter(BAD_NUMBER, GOOD, GOOD), usage_log=log)

    agent.answer("a question that must not be written down")
    agent.answer("another question")

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["purpose"] for line in lines] == ["draft", "rewrite", "draft"]
    assert all(line["model"] == "claude-test-model" for line in lines)
    assert all(line["input_tokens"] == 900 and line["output_tokens"] == 100 for line in lines)
    assert all("time" in line and line["decision"] == "answer" for line in lines)
    assert "must not be written down" not in log.read_text()


def test_without_a_usage_log_nothing_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    run(ScriptedDrafter(GOOD))

    assert list(tmp_path.iterdir()) == []


def test_the_printed_result_shows_the_decision_the_reply_the_sources_and_the_tokens():
    printed = format_result(run(ScriptedDrafter(BAD_NUMBER, GOOD)))

    assert "Decision: answer" in printed
    assert "The link stops working after 48 hours." in printed
    assert "Share a link" in printed and PAGE + "share-a-link" in printed
    assert "Drafts written: 2; failed the check: 1" in printed
    assert "900 tokens in" in printed


def test_the_printed_result_of_a_hand_over_gives_the_reason():
    printed = format_result(run(ScriptedDrafter(GIVE_UP)))

    assert "Decision: hand_over" in printed
    assert "the_guides_do_not_answer_this" in printed


class StubService:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.questions: list[str] = []

    def answer(self, question: str) -> AgentResult:
        self.questions.append(question)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_the_command_line_prints_the_result(capsys: pytest.CaptureFixture):
    service = StubService(run(ScriptedDrafter(GOOD)))

    code = agent_module.main(["  how long does a link last  "], make_service=lambda: service)

    assert code == 0
    assert service.questions == ["how long does a link last"]  # trimmed
    assert "Decision: answer" in capsys.readouterr().out


def test_the_command_line_refuses_an_empty_question(capsys: pytest.CaptureFixture):
    code = agent_module.main(["   "], make_service=lambda: StubService(None))

    assert code == 2
    assert "Not asked" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [DatabaseUnavailable("could not connect"), AgentNotConfigured("no model")],
)
def test_the_command_line_reports_a_fault_and_exits_with_an_error(capsys: pytest.CaptureFixture, error: Exception):
    code = agent_module.main(["a question"], make_service=lambda: StubService(error))

    captured = capsys.readouterr()
    assert code == 1
    assert "Could not answer" in captured.err
    assert captured.out == ""
