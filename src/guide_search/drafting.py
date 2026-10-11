"""Drafting: asking a Claude model for a draft, and recording what the call used.

The answer agent hands this file a question and the passages search found.
It asks the model for a draft in a fixed shape (a decision, a reply, the ids
of the guides it relied on) and returns it together with a record of the
call: which model answered, how many tokens it used and how long it took.

The agent sees the model only through the small "Drafter" interface below,
so the tests can use a stand-in that returns prepared drafts and never calls
the API.

Every setting of the request comes from agent.toml. Nothing is tuned here
out of sight.
"""

import json
import logging
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anthropic

from guide_search.checks import DECISIONS, Draft, Failure
from guide_search.ranking import GuideHit

logger = logging.getLogger(__name__)

AGENT_SETTINGS_PATH = Path("agent.toml")
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# How long one request may take before it is given up, in seconds. The SDK
# tries a failed request twice more by itself before it reports the failure.
TIMEOUT_SECONDS = 120.0

# Written for this project. The model is told what it may rely on, what each
# decision is for, and what the check will refuse.
INSTRUCTIONS = """\
You answer questions from people who use SAM, a workshop management product, \
using only passages from its help guides.

The user message gives you the question and the passages that a search found. \
Each passage belongs to a guide, shown with its id, title and web address. \
The passages are the closest matches the search found. They are not guaranteed \
to answer the question.

Choose one decision:

- "answer": the passages answer the question. Write the answer in plain words, \
in the language of the question, using only what the passages say. Put the id \
of every guide you relied on in cited_guide_ids.
- "clarify": the question is too vague to answer. Ask one short question back. \
Cite nothing, and state no fact, number or link.
- "hand_over": the passages do not answer the question. Do not guess and do \
not answer from general knowledge. A person will take over. Leave the reply \
empty and cite nothing.

Rules for an answer, which are checked by a program after you:

- Cite only guides that are in the user message.
- Every number you state must appear in the passages of the guides you cite.
- Every link you give must be the web address of a guide you cite, or a link \
that appears in its passages.
- Keep the reply under 1,200 characters.

If the passages cover only part of the question, answer that part and say \
what they do not cover. If they cover none of it, choose "hand_over".
"""

# The shape of the reply, enforced by the API.
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "reply": {"type": "string"},
        "cited_guide_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["decision", "reply", "cited_guide_ids"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AgentSettings:
    """The settings of the model request, read from agent.toml."""

    model: str
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class ModelCall:
    """What one call to the model used. One of these is kept for every call."""

    purpose: str  # "draft" or "rewrite"
    model: str  # the model that answered, as the API reports it
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    stop_reason: str | None
    seconds: float


class ModelFault(Exception):
    """The model could not be used for this draft.

    `reason` is a short name for what went wrong. `call` is the record of the
    call when the API did answer (for example with a refusal), and None when
    it did not answer at all.
    """

    def __init__(self, reason: str, call: ModelCall | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.call = call


class Drafter(Protocol):
    """What the agent needs from the model."""

    # `previous` and `failure` are both None for the first draft. For the one
    # rewrite they hold the draft that failed the check and the rule it broke.
    def draft(
        self,
        question: str,
        guides: list[GuideHit],
        previous: Draft | None,
        failure: Failure | None,
    ) -> tuple[Draft, ModelCall]: ...


def load_agent_settings(path: Path = AGENT_SETTINGS_PATH) -> AgentSettings:
    """Read agent.toml. Raises ValueError with a plain message when it is wrong."""
    with open(path, "rb") as file:
        values = tomllib.load(file)
    expected = {"model", "effort", "max_tokens"}
    if set(values) != expected:
        raise ValueError(f"{path} must hold exactly these settings: " + ", ".join(sorted(expected)))
    if not isinstance(values["model"], str) or values["model"].strip() == "":
        raise ValueError(f"{path}: model must be the id of a Claude model")
    if values["effort"] not in EFFORTS:
        raise ValueError(f"{path}: effort must be one of: " + ", ".join(EFFORTS))
    if isinstance(values["max_tokens"], bool) or not isinstance(values["max_tokens"], int) or values["max_tokens"] < 1:
        raise ValueError(f"{path}: max_tokens must be a whole number above zero")
    return AgentSettings(model=values["model"], effort=values["effort"], max_tokens=values["max_tokens"])


def passages_text(guides: list[GuideHit]) -> str:
    """Write the passages the way the model is shown them."""
    if not guides:
        return "(The search found no passages.)"
    blocks = []
    for guide in guides:
        lines = [f"Guide id {guide.guide_id}: {guide.title}", f"Web address: {guide.url}"]
        for passage in guide.passages:
            heading = passage.heading if passage.heading else "(start of the guide)"
            lines.append(f"\nPassage, section \"{heading}\":\n{passage.text}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def user_message(
    question: str,
    guides: list[GuideHit],
    previous: Draft | None,
    failure: Failure | None,
) -> str:
    """The one user message of a request: the question, the passages, and for a rewrite the reason."""
    parts = [f"Question:\n{question}", f"Passages the search found:\n\n{passages_text(guides)}"]
    if previous is not None and failure is not None:
        parts.append(
            "You already wrote the draft below, and the checking program refused it.\n\n"
            f"Your draft:\n{previous.reply}\n\n"
            f"Why it was refused: {failure.message}\n\n"
            "Write it again so that it passes. If you cannot, choose \"hand_over\"."
        )
    return "\n\n".join(parts)


def build_request(
    settings: AgentSettings,
    question: str,
    guides: list[GuideHit],
    previous: Draft | None,
    failure: Failure | None,
) -> dict:
    """The keyword arguments for client.messages.create."""
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": INSTRUCTIONS,
        "messages": [{"role": "user", "content": user_message(question, guides, previous, failure)}],
        # effort says how hard the model works; format makes the API hold the
        # reply to the schema, so it is always JSON of the expected shape.
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": DRAFT_SCHEMA},
        },
    }


def parse_draft(text: str) -> Draft:
    """Turn the model's JSON text into a Draft. Raises ValueError when it is not one."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("the reply is not JSON") from error
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    decision = data.get("decision")
    reply = data.get("reply")
    cited = data.get("cited_guide_ids")
    if decision not in DECISIONS or not isinstance(reply, str) or not isinstance(cited, list):
        raise ValueError("the reply does not have the expected fields")
    ids: list[int] = []
    for value in cited:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("a cited guide id is not a whole number")
        if value not in ids:
            ids.append(value)
    return Draft(decision=decision, reply=reply.strip(), cited_guide_ids=tuple(ids))


def real_client() -> anthropic.Anthropic:
    """The real API client. It reads its credentials from the environment."""
    return anthropic.Anthropic(timeout=TIMEOUT_SECONDS)


class ClaudeDrafter:
    """The real Drafter: one request to the Claude API for each draft."""

    def __init__(self, client, settings: AgentSettings) -> None:
        self.client = client
        self.settings = settings

    def draft(
        self,
        question: str,
        guides: list[GuideHit],
        previous: Draft | None,
        failure: Failure | None,
    ) -> tuple[Draft, ModelCall]:
        purpose = "draft" if previous is None else "rewrite"
        request = build_request(self.settings, question, guides, previous, failure)

        started = time.perf_counter()
        try:
            response = self.client.messages.create(**request)
        except anthropic.APIError as error:
            # The SDK has already retried what can be retried. Whatever is
            # left (no network, a wrong key, an unknown model, overload) means
            # no draft, and the agent hands over.
            logger.warning("model call failed: %s", type(error).__name__)
            raise ModelFault("api_error_" + type(error).__name__) from error
        seconds = time.perf_counter() - started

        usage = response.usage
        call = ModelCall(
            purpose=purpose,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            stop_reason=response.stop_reason,
            seconds=seconds,
        )

        # A refusal or a reply cut off at the token limit arrives as an
        # ordinary response. Neither is a usable draft.
        if response.stop_reason != "end_turn":
            raise ModelFault(f"stopped_{response.stop_reason}", call)

        # The reply may begin with the model's thinking, so the text block is
        # looked for by its type and not by its place.
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise ModelFault("no_text_in_reply", call)
        try:
            return parse_draft(text), call
        except ValueError:
            raise ModelFault("unreadable_reply", call) from None
