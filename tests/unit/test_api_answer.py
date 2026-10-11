"""Tests for POST /answer, with the service replaced by a stub.

The route itself is thin: it checks the question, asks the backend, and turns
the result into JSON. What matters is the status codes, and above all that a
fault is never dressed up as an answer or as a hand-over.
"""

from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from guide_search.agent import AgentNotConfigured, AgentResult, CheckSummary
from guide_search.api import create_app
from guide_search.drafting import ModelCall
from guide_search.ranking import GuideSummary
from guide_search.store import DatabaseUnavailable, IndexNotReady

PAGE = "https://support.orions360.com/help/en/autoserv-pro/"

ANSWERED = AgentResult(
    question="how long does a link last",
    decision="answer",
    reply="The link stops working after 48 hours.",
    sources=[GuideSummary(guide_id=3, slug="share-a-link", title="Share a link", url=PAGE + "share-a-link")],
    hand_over_reason=None,
    check=CheckSummary(attempts=1, failures=[]),
    usage=[
        ModelCall(
            purpose="draft",
            model="claude-test-model",
            input_tokens=900,
            output_tokens=100,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            stop_reason="end_turn",
            seconds=0.5,
        )
    ],
)
HANDED_OVER = AgentResult(
    question="can I pay by card machine",
    decision="hand_over",
    reply="I can't answer this from the help guides, so a person needs to help you with it.",
    sources=[],
    hand_over_reason="the_guides_do_not_answer_this",
    check=CheckSummary(attempts=1, failures=[]),
    usage=[],
)


class StubBackend:
    """Stands in for the service: returns the prepared result, or raises the prepared error."""

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.questions: list[str] = []

    def answer(self, question: str) -> AgentResult:
        self.questions.append(question)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def post(backend: StubBackend, body):
    with TestClient(create_app(backend)) as client:
        return client.post("/answer", json=body)


def test_an_answer_is_200_with_every_field_of_the_result():
    response = post(StubBackend(ANSWERED), {"question": "how long does a link last"})

    assert response.status_code == 200
    assert response.json() == asdict(ANSWERED)
    assert sorted(response.json()) == [
        "check",
        "decision",
        "hand_over_reason",
        "question",
        "reply",
        "sources",
        "usage",
    ]


def test_a_hand_over_is_a_normal_result_and_answers_200():
    response = post(StubBackend(HANDED_OVER), {"question": "can I pay by card machine"})

    assert response.status_code == 200
    assert response.json()["decision"] == "hand_over"
    assert response.json()["hand_over_reason"] == "the_guides_do_not_answer_this"
    assert response.json()["sources"] == []


def test_the_question_is_passed_on_trimmed():
    backend = StubBackend(ANSWERED)

    post(backend, {"question": "  how long does a link last \n"})

    assert backend.questions == ["how long does a link last"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_an_empty_or_over_long_question_is_422_and_is_not_asked(question: str):
    backend = StubBackend(ANSWERED)

    response = post(backend, {"question": question})

    assert response.status_code == 422
    assert backend.questions == []


@pytest.mark.parametrize("body", [{}, {"q": "how long"}, {"question": 5}, {"question": None}])
def test_a_body_without_a_text_question_is_422(body):
    backend = StubBackend(ANSWERED)

    assert post(backend, body).status_code == 422
    assert backend.questions == []


def test_a_question_of_exactly_500_characters_is_asked():
    backend = StubBackend(ANSWERED)

    assert post(backend, {"question": "x" * 500}).status_code == 200
    assert backend.questions == ["x" * 500]


@pytest.mark.parametrize(
    ("error", "name"),
    [
        (DatabaseUnavailable("could not connect"), "database_unavailable"),
        (IndexNotReady("no passages"), "index_not_ready"),
        (AgentNotConfigured("no model"), "agent_not_configured"),
    ],
)
def test_a_fault_is_503_and_never_an_answer_or_a_hand_over(error: Exception, name: str):
    response = post(StubBackend(error), {"question": "how long does a link last"})

    assert response.status_code == 503
    assert response.json() == {"error": name}


def test_answer_is_only_offered_as_post():
    with TestClient(create_app(StubBackend(ANSWERED))) as client:
        assert client.get("/answer").status_code == 405
