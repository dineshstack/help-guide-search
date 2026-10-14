"""The baseline that search has to beat: a model picks guides from a list of their titles.

For each question a Claude model is shown the id, slug and title of every
guide, and nothing else. It returns the slugs of up to three guides it would
read, most useful first, or an empty list when none looks relevant.

Each answer is stored as one line of `evaluation/title_picking.jsonl`, which
is committed. The report is built from that file, so nobody needs an API key
to rebuild it. Every line carries a hash of the question and a hash of the
whole request. If a question, a guide title or the instructions change, the
hashes stop matching and the stored line is no longer accepted.

Run it with:

    python -m guide_search.evaluation.title_picking --split dev
    python -m guide_search.evaluation.title_picking --split fresh   (reads the fresh question file)

This is the only part of the project that needs ANTHROPIC_API_KEY.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from guide_search.evaluation.questions import FRESH_PATH, QUESTIONS_PATH, SNAPSHOT_PATH, Question, load_questions
from guide_search.snapshot import Guide, load_snapshot

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_PICKS = 3

# One call and, if its reply cannot be used, one more.
ATTEMPTS = 2

RESULTS_PATH = Path("evaluation/title_picking.jsonl")

# The fields of one line of the results file.
RESULT_KEYS = (
    "question_id",
    "question_hash",
    "request_hash",
    "model",
    "date",
    "slugs",
    "stop_reason",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "seconds",
)

# What the model is told. The list of guides is added underneath.
# Changing this text changes the request hash, so every stored line goes out
# of date and the baseline has to be run again.
INSTRUCTIONS = """You are helping a customer of a workshop-management product find the right help guides.

At the end of this message is the full list of help guides, one per line, written as: id | slug | title
This list is all you can see. You cannot see the text of the guides.

The customer's question is the user message. It may be in English or in Arabic. The guides are in English.

Reply with the slugs of the guides you would read to answer the question.
Give at most three, the most useful first. Give fewer when fewer look relevant.

If no guide in the list looks relevant to the question, reply with an empty list.
Do not guess: include a guide only when its title or slug suggests that it covers what the customer is asking.

Guides:
"""


class BaselineError(RuntimeError):
    """The baseline could not get a usable answer, or its stored results cannot be trusted."""


# --- the request ------------------------------------------------------------


def build_request(guides: list[Guide], question: str) -> dict:
    """The keyword arguments for `client.messages.create` for one question."""
    guide_lines = [f"{guide.id} | {guide.slug} | {guide.title}" for guide in guides]
    schema = {
        "type": "object",
        "properties": {
            "slugs": {
                "type": "array",
                # Only a real slug can be returned. The schema cannot say
                # "at most three" or "no repeats"; parse_picks does that.
                "items": {"type": "string", "enum": [guide.slug for guide in guides]},
            }
        },
        "required": ["slugs"],
        "additionalProperties": False,
    }
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": INSTRUCTIONS + "\n".join(guide_lines),
        "messages": [{"role": "user", "content": question}],
        # Nothing else is set here. Every optional setting is left at the API's
        # default, so the baseline is not held back by a choice made in this file.
        "output_config": {
            # Structured output: the API makes the reply fit this schema.
            "format": {"type": "json_schema", "schema": schema},
        },
    }


def question_hash(question: str) -> str:
    """SHA-256 of the question text, as hex."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def request_hash(request: dict) -> str:
    """SHA-256 of the whole request: model, settings, instructions, guide list, schema and question."""
    # sort_keys so that the same request always gives the same text to hash.
    text = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- reading a reply --------------------------------------------------------


def returned_slugs(text: str) -> list:
    """The list the model returned, exactly as it returned it."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise BaselineError("the reply is not JSON") from error
    if not isinstance(data, dict) or not isinstance(data.get("slugs"), list):
        raise BaselineError('the reply is not of the form {"slugs": [...]}')
    return data["slugs"]


def parse_picks(text: str, valid_slugs: set[str]) -> list[str]:
    """Turn the reply text into the picks: real slugs only, no repeats, the first three."""
    picks: list[str] = []
    for slug in returned_slugs(text):
        if isinstance(slug, str) and slug in valid_slugs and slug not in picks:
            picks.append(slug)
    return picks[:MAX_PICKS]


def reply_text(response: Any) -> str:
    """The text of the first content block whose type is "text"."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise BaselineError("the reply has no text")


def read_reply(response: Any, valid_slugs: set[str]) -> list[str]:
    """The picks in one reply. Raises BaselineError when the reply cannot be used."""
    if response.stop_reason != "end_turn":
        # "max_tokens" means the reply was cut off and "refusal" that the
        # model declined. Either way its text is not a finished answer.
        raise BaselineError(f"the reply stopped with {response.stop_reason!r}, not 'end_turn'")
    text = reply_text(response)
    picks = parse_picks(text, valid_slugs)
    if not picks and returned_slugs(text):
        # The model named something, but nothing that is a guide. That is not
        # the model saying "no guide is relevant", so it is not stored as that.
        raise BaselineError("the reply names no slug that is a guide")
    return picks


def pick_once(
    client: Any,
    guides: list[Guide],
    question: Question,
    today: str,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Ask the model about one question and return one line for the results file.

    A reply that cannot be used is asked for once more. If the second reply
    is no better, this raises BaselineError and returns nothing, so a failed
    call can never be stored as "the model picked nothing".
    """
    request = build_request(guides, question.question)
    valid_slugs = {guide.slug for guide in guides}
    problem = ""
    for _ in range(ATTEMPTS):
        started = clock()
        response = client.messages.create(**request)
        seconds = clock() - started
        try:
            slugs = read_reply(response, valid_slugs)
        except BaselineError as error:
            problem = str(error)
            continue
        return {
            "question_id": question.id,
            "question_hash": question_hash(question.question),
            "request_hash": request_hash(request),
            "model": response.model,  # the model that answered, as the API reports it
            "date": today,
            "slugs": slugs,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            # The API may leave these two empty when nothing was cached.
            "cache_creation_input_tokens": response.usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens or 0,
            "seconds": round(seconds, 3),
        }
    raise BaselineError(f"question {question.id}: {problem} (asked {ATTEMPTS} times)")


# --- the results file -------------------------------------------------------


def load_results(path: str | Path) -> list[dict]:
    """Read the results file. A file that does not exist yet has no lines."""
    path = Path(path)
    if not path.exists():
        return []
    results: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise BaselineError(f"{path} line {number}: not valid JSON") from error
        if not isinstance(data, dict) or any(key not in data for key in RESULT_KEYS):
            raise BaselineError(f"{path} line {number}: not a results line")
        results.append(data)
    return results


def append_result(path: str | Path, line: dict) -> None:
    """Add one line to the end of the results file."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")


def current_lines(question: Question, results: list[dict], guides: list[Guide]) -> list[dict]:
    """The stored lines for this question that still match the question and the guides."""
    wanted_question = question_hash(question.question)
    wanted_request = request_hash(build_request(guides, question.question))
    return [
        line
        for line in results
        if line["question_id"] == question.id
        and line["question_hash"] == wanted_question
        and line["request_hash"] == wanted_request
    ]


def check_results(questions: list[Question], results: list[dict], guides: list[Guide]) -> dict[str, dict]:
    """Match every reviewed question to its stored line: question id -> line.

    Raises BaselineError, naming every question concerned, when a reviewed
    question has no line, only lines that are out of date, or more than one
    current line. Unreviewed questions are ignored, and so are lines for
    questions that are not in `questions`.
    """
    problems: list[str] = []
    checked: dict[str, dict] = {}
    for question in questions:
        if not question.reviewed:
            continue
        stored = [line for line in results if line["question_id"] == question.id]
        current = current_lines(question, stored, guides)
        if not stored:
            problems.append(f"{question.id}: no stored result")
        elif not current:
            problems.append(
                f"{question.id}: the stored result is out of date "
                "(the question, the guide list or the request has changed since it was recorded)"
            )
        elif len(current) > 1:
            problems.append(f"{question.id}: {len(current)} stored results where one is expected")
        else:
            checked[question.id] = current[0]
    if problems:
        raise BaselineError("\n".join(problems))
    return checked


# --- the command ------------------------------------------------------------


def real_client() -> Any:
    """The real API client. It reads ANTHROPIC_API_KEY from the environment by itself."""
    # Imported here, not at the top: the SDK is an optional install that only
    # this command needs, and the tests use a fake client instead.
    import anthropic

    return anthropic.Anthropic()


def main(argv: list[str] | None = None, make_client: Callable[[], Any] = real_client) -> int:
    """Send the questions that have no stored answer yet. Tests pass a `make_client` that returns a fake."""
    parser = argparse.ArgumentParser(description="Ask the baseline model which guides it would read.")
    parser.add_argument("--split", required=True, choices=["dev", "test", "fresh"])
    parser.add_argument(
        "--questions",
        help="the question file; by default the fresh file for --split fresh, the main file otherwise",
    )
    parser.add_argument("--snapshot", default=SNAPSHOT_PATH, help="the guide snapshot")
    parser.add_argument("--results", default=RESULTS_PATH, help="the file the answers are appended to")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. The baseline cannot run without it.", file=sys.stderr)
        return 1

    if args.questions is None:
        args.questions = FRESH_PATH if args.split == "fresh" else QUESTIONS_PATH
    guides = load_snapshot(args.snapshot)
    questions = [q for q in load_questions(args.questions) if q.reviewed and q.split == args.split]
    if not questions:
        print(f"No reviewed question has the split {args.split!r}. Nothing was sent.", file=sys.stderr)
        return 1

    results = load_results(args.results)
    waiting = [q for q in questions if not current_lines(q, results, guides)]
    stored = len(questions) - len(waiting)
    print(f"{len(questions)} reviewed {args.split} questions: {stored} already stored, {len(waiting)} to send")
    if not waiting:
        return 0

    client = make_client()
    today = date.today().isoformat()
    for question in waiting:
        try:
            line = pick_once(client, guides, question, today)
        except BaselineError as error:
            # The lines written so far stay in the file. Running the command
            # again carries on from this question.
            print(error, file=sys.stderr)
            return 1
        # Written straight away, so an answer that was paid for is never lost.
        append_result(args.results, line)
        print(f"{question.id}: {', '.join(line['slugs']) or '(no guide)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
