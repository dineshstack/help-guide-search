"""Tests for the model call. No test here calls the API.

The real client is replaced by a small stand-in whose messages.create()
remembers its arguments and returns a prepared response, or raises a prepared
error.
"""

from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from guide_search.checks import Draft, Failure
from guide_search.drafting import (
    DRAFT_SCHEMA,
    INSTRUCTIONS,
    AgentSettings,
    ClaudeDrafter,
    ModelFault,
    build_request,
    load_agent_settings,
    parse_draft,
)
from guide_search.ranking import GuideHit, PassageHit

REPOSITORY = Path(__file__).resolve().parents[2]
SETTINGS = AgentSettings(model="claude-test-model", effort="medium", max_tokens=4000)
PAGE = "https://support.orions360.com/help/en/autoserv-pro/"

GUIDES = [
    GuideHit(
        guide_id=3,
        slug="share-a-link",
        title="Share a link",
        url=PAGE + "share-a-link",
        score=0.9,
        passages=[
            PassageHit(heading="", position=0, text="A link can be sent to a customer.", similarity=0.9),
            PassageHit(heading="When it ends", position=2, text="The link stops working after 48 hours.", similarity=0.8),
        ],
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

GOOD_JSON = '{"decision": "answer", "reply": "It stops after 48 hours.", "cited_guide_ids": [3]}'


def response(text: str | None = GOOD_JSON, stop_reason: str = "end_turn", thinking_first: bool = False):
    """A stand-in for what the API returns."""
    content = []
    if thinking_first:
        content.append(SimpleNamespace(type="thinking", thinking=""))
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model="claude-test-model-served",
        usage=SimpleNamespace(
            input_tokens=900,
            output_tokens=120,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=0,
        ),
    )


class FakeClient:
    """Stands in for the API client: remembers each request, returns or raises what it was given."""

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_the_committed_agent_settings_load():
    settings = load_agent_settings(REPOSITORY / "agent.toml")

    assert settings.model.startswith("claude-")
    assert settings.effort in ("low", "medium", "high", "xhigh", "max")
    assert settings.max_tokens >= 1000


@pytest.mark.parametrize(
    "text",
    [
        'model = "claude-x"\neffort = "medium"\n',  # a setting is missing
        'model = "claude-x"\neffort = "medium"\nmax_tokens = 100\ntemperature = 0\n',  # one too many
        'model = ""\neffort = "medium"\nmax_tokens = 100\n',
        'model = "claude-x"\neffort = "extreme"\nmax_tokens = 100\n',
        'model = "claude-x"\neffort = "medium"\nmax_tokens = 0\n',
        'model = "claude-x"\neffort = "medium"\nmax_tokens = "many"\n',
    ],
)
def test_wrong_agent_settings_are_refused(tmp_path: Path, text: str):
    path = tmp_path / "agent.toml"
    path.write_text(text)

    with pytest.raises(ValueError):
        load_agent_settings(path)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_the_request_holds_only_the_stated_settings_and_nothing_else():
    request = build_request(SETTINGS, "how long does a link last", GUIDES, None, None)

    assert sorted(request) == ["max_tokens", "messages", "model", "output_config", "system"]
    assert request["model"] == "claude-test-model"
    assert request["max_tokens"] == 4000
    assert request["system"] == INSTRUCTIONS
    assert sorted(request["output_config"]) == ["effort", "format"]
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": DRAFT_SCHEMA}


def test_the_reply_shape_allows_the_three_decisions_and_no_extra_fields():
    assert DRAFT_SCHEMA["properties"]["decision"]["enum"] == ["answer", "clarify", "hand_over"]
    assert DRAFT_SCHEMA["required"] == ["decision", "reply", "cited_guide_ids"]
    assert DRAFT_SCHEMA["additionalProperties"] is False


def test_the_model_is_shown_the_question_and_every_passage_with_its_guide():
    request = build_request(SETTINGS, "how long does a link last", GUIDES, None, None)

    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    shown = request["messages"][0]["content"]
    assert "how long does a link last" in shown
    for guide in GUIDES:
        assert f"Guide id {guide.guide_id}: {guide.title}" in shown
        assert guide.url in shown
        for passage in guide.passages:
            assert passage.text in shown
    assert 'section "When it ends"' in shown
    assert 'section "(start of the guide)"' in shown


def test_a_first_draft_request_mentions_no_earlier_draft():
    shown = build_request(SETTINGS, "q", GUIDES, None, None)["messages"][0]["content"]

    assert "refused" not in shown


def test_a_rewrite_request_shows_the_failed_draft_and_the_reason():
    previous = Draft(decision="answer", reply="It stops after 72 hours.", cited_guide_ids=(3,))
    failure = Failure("unsupported_number", "The number 72 is not in the passages of the guides you cited.")

    shown = build_request(SETTINGS, "q", GUIDES, previous, failure)["messages"][0]["content"]

    assert "It stops after 72 hours." in shown
    assert "The number 72 is not in the passages of the guides you cited." in shown


def test_the_model_is_told_when_search_found_nothing():
    shown = build_request(SETTINGS, "q", [], None, None)["messages"][0]["content"]

    assert "The search found no passages." in shown


def test_the_instructions_name_the_three_decisions_and_forbid_guessing():
    for word in ('"answer"', '"clarify"', '"hand_over"'):
        assert word in INSTRUCTIONS
    assert "Do not guess" in INSTRUCTIONS


# ---------------------------------------------------------------------------
# Reading the reply
# ---------------------------------------------------------------------------


def test_a_reply_is_read_into_a_draft():
    draft = parse_draft('{"decision": "answer", "reply": "  Yes.  ", "cited_guide_ids": [3, 7, 3]}')

    assert draft == Draft(decision="answer", reply="Yes.", cited_guide_ids=(3, 7))


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        "[1, 2, 3]",
        '{"decision": "guess", "reply": "x", "cited_guide_ids": []}',
        '{"decision": "answer", "reply": 5, "cited_guide_ids": []}',
        '{"decision": "answer", "reply": "x", "cited_guide_ids": "3"}',
        '{"decision": "answer", "reply": "x", "cited_guide_ids": ["3"]}',
        '{"decision": "answer", "reply": "x", "cited_guide_ids": [true]}',
        '{"decision": "answer", "reply": "x"}',
    ],
)
def test_a_reply_of_the_wrong_shape_is_refused(text: str):
    with pytest.raises(ValueError):
        parse_draft(text)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def test_a_call_returns_the_draft_and_a_record_of_what_it_used():
    client = FakeClient(response())

    draft, call = ClaudeDrafter(client, SETTINGS).draft("how long does a link last", GUIDES, None, None)

    assert draft == Draft(decision="answer", reply="It stops after 48 hours.", cited_guide_ids=(3,))
    assert call.purpose == "draft"
    assert call.model == "claude-test-model-served"  # the model the API says answered
    assert (call.input_tokens, call.output_tokens) == (900, 120)
    assert call.cache_creation_input_tokens is None  # not reported stays None, never 0
    assert call.cache_read_input_tokens == 0
    assert call.stop_reason == "end_turn"
    assert call.seconds >= 0
    assert len(client.requests) == 1


def test_a_second_call_for_the_same_question_is_recorded_as_a_rewrite():
    previous = Draft(decision="answer", reply="It stops after 72 hours.", cited_guide_ids=(3,))
    failure = Failure("unsupported_number", "The number 72 is not in the passages.")

    _, call = ClaudeDrafter(FakeClient(response()), SETTINGS).draft("q", GUIDES, previous, failure)

    assert call.purpose == "rewrite"


def test_the_text_is_found_even_when_the_reply_starts_with_thinking():
    draft, _ = ClaudeDrafter(FakeClient(response(thinking_first=True)), SETTINGS).draft("q", GUIDES, None, None)

    assert draft.reply == "It stops after 48 hours."


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "pause_turn"])
def test_a_refused_or_cut_off_reply_is_a_fault_and_keeps_its_record(stop_reason: str):
    client = FakeClient(response(stop_reason=stop_reason))

    with pytest.raises(ModelFault) as raised:
        ClaudeDrafter(client, SETTINGS).draft("q", GUIDES, None, None)

    assert raised.value.reason == f"stopped_{stop_reason}"
    assert raised.value.call is not None
    assert raised.value.call.stop_reason == stop_reason
    assert raised.value.call.input_tokens == 900


def test_a_reply_that_cannot_be_read_is_a_fault():
    with pytest.raises(ModelFault) as raised:
        ClaudeDrafter(FakeClient(response(text="I think the answer is yes")), SETTINGS).draft("q", GUIDES, None, None)

    assert raised.value.reason == "unreadable_reply"
    assert raised.value.call is not None


def test_a_reply_with_no_text_is_a_fault():
    with pytest.raises(ModelFault) as raised:
        ClaudeDrafter(FakeClient(response(text=None)), SETTINGS).draft("q", GUIDES, None, None)

    assert raised.value.reason == "no_text_in_reply"


def test_an_api_error_is_a_fault_with_no_record():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(anthropic.APIConnectionError(request=request))

    with pytest.raises(ModelFault) as raised:
        ClaudeDrafter(client, SETTINGS).draft("q", GUIDES, None, None)

    assert raised.value.reason == "api_error_APIConnectionError"
    assert raised.value.call is None
