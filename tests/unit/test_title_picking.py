"""Tests for the title-picking baseline.

A fake client stands in for the API. It records what it was asked and hands
back prepared replies, so nothing here makes a network call or needs a key.
Guides, questions and replies are all invented.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from guide_search.evaluation.questions import Question, save_questions
from guide_search.evaluation.title_picking import (
    INSTRUCTIONS,
    MAX_PICKS,
    MAX_TOKENS,
    MODEL,
    BaselineError,
    build_request,
    check_results,
    load_results,
    main,
    parse_picks,
    pick_once,
    question_hash,
    request_hash,
)
from guide_search.snapshot import LOCALE, PRODUCT, PUBLIC_PAGE_PREFIX, SOURCE, Guide

ARABIC_QUESTION = "كيف أطبع فاتورة للعميل؟"
TODAY = "2026-10-12"


def make_guide(guide_id: int, slug: str, title: str) -> Guide:
    return Guide(
        id=guide_id,
        slug=slug,
        title=title,
        url=PUBLIC_PAGE_PREFIX + slug,
        updated_at="2026-01-01T00:00:00+00:00",
        body="Invented text for a test.",
    )


GUIDES = [
    make_guide(1, "invoices", "Create and send an invoice"),
    make_guide(2, "refunds", "Give a customer their money back"),
    make_guide(4, "bookings", "Book a job in the diary"),
    make_guide(7, "stock", "Keep track of parts in stock"),
]
VALID_SLUGS = {"invoices", "refunds", "bookings", "stock"}


def make_question(
    question_id: str,
    text: str = "How do I send an invoice?",
    split: str = "dev",
    reviewed: bool = True,
    expected: tuple[str, ...] = ("invoices",),
    language: str = "en",
) -> Question:
    return Question(
        id=question_id,
        language=language,
        question=text,
        expected_slugs=expected,
        answer_headings={slug: ("",) for slug in expected},
        split=split,
        reviewed=reviewed,
    )


# --- the fake client --------------------------------------------------------


class FakeMessages:
    """Stands in for `client.messages`: records each call and returns the next prepared reply."""

    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.replies.pop(0)


class FakeClient:
    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(replies)


def reply_with_text(text: str, stop_reason: str = "end_turn", model: str = MODEL) -> SimpleNamespace:
    """A reply shaped like the SDK's: content blocks, stop_reason, model and usage."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=18,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def reply(slugs: list[str], stop_reason: str = "end_turn", model: str = MODEL) -> SimpleNamespace:
    """A reply in which the model picked these slugs."""
    return reply_with_text(json.dumps({"slugs": slugs}), stop_reason, model)


def ticking_clock(*times: float) -> Callable[[], float]:
    """A clock that returns the given times, one per call."""
    remaining = list(times)

    def clock() -> float:
        return remaining.pop(0)

    return clock


def line_for(question: Question, guides: list[Guide] = GUIDES) -> dict:
    """The results line that a run would store for this question and these guides."""
    return pick_once(FakeClient([reply(["invoices"])]), guides, question, TODAY)


# --- the request ------------------------------------------------------------


def test_the_request_lists_every_guide_as_id_slug_title_after_the_instructions() -> None:
    request = build_request(GUIDES, "How do I send an invoice?")

    assert request["system"] == INSTRUCTIONS + (
        "1 | invoices | Create and send an invoice\n"
        "2 | refunds | Give a customer their money back\n"
        "4 | bookings | Book a job in the diary\n"
        "7 | stock | Keep track of parts in stock"
    )


def test_the_question_is_the_user_message_unchanged() -> None:
    request = build_request(GUIDES, ARABIC_QUESTION)

    assert request["messages"] == [{"role": "user", "content": ARABIC_QUESTION}]


def test_the_schema_only_allows_the_slugs_of_the_guides() -> None:
    request = build_request(GUIDES, "How do I send an invoice?")

    assert request["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "slugs": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["invoices", "refunds", "bookings", "stock"]},
                }
            },
            "required": ["slugs"],
            "additionalProperties": False,
        },
    }


def test_the_model_and_the_token_limit_are_the_agreed_ones_and_no_optional_setting_is_sent() -> None:
    request = build_request(GUIDES, "How do I send an invoice?")

    assert request["model"] == MODEL == "claude-sonnet-4-6"
    assert request["max_tokens"] == MAX_TOKENS == 1024
    # Only the shape of the reply is asked for. Every optional setting is left
    # at the API's default, so the baseline is not held back by a choice made here.
    assert set(request["output_config"]) == {"format"}


def test_the_request_sets_nothing_else() -> None:
    # In particular no temperature: the design leaves it alone.
    request = build_request(GUIDES, "How do I send an invoice?")

    assert set(request) == {"model", "max_tokens", "system", "messages", "output_config"}


# --- the two hashes ---------------------------------------------------------


def test_question_hash_is_the_sha256_of_the_text() -> None:
    # "abc" is the standard example; its SHA-256 is published.
    assert question_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_question_hash_works_for_arabic_and_differs_between_questions() -> None:
    assert len(question_hash(ARABIC_QUESTION)) == 64
    assert question_hash(ARABIC_QUESTION) != question_hash(ARABIC_QUESTION + " ")


def test_request_hash_is_the_same_for_the_same_guides_and_question() -> None:
    first = request_hash(build_request(GUIDES, ARABIC_QUESTION))
    second = request_hash(build_request(list(GUIDES), ARABIC_QUESTION))

    assert first == second
    assert len(first) == 64


def test_request_hash_changes_when_the_question_changes() -> None:
    assert request_hash(build_request(GUIDES, "How do I send an invoice?")) != request_hash(
        build_request(GUIDES, "How do I print an invoice?")
    )


def test_request_hash_changes_when_a_title_changes() -> None:
    retitled = [GUIDES[0], replace(GUIDES[1], title="Refund a payment"), GUIDES[2], GUIDES[3]]

    assert request_hash(build_request(retitled, "A question")) != request_hash(build_request(GUIDES, "A question"))


def test_request_hash_changes_when_a_guide_is_added_or_removed() -> None:
    original = request_hash(build_request(GUIDES, "A question"))
    one_fewer = request_hash(build_request(GUIDES[:-1], "A question"))
    one_more = request_hash(build_request(GUIDES + [make_guide(9, "reminders", "Send a reminder")], "A question"))

    assert len({original, one_fewer, one_more}) == 3


def test_request_hash_does_not_change_when_only_a_guide_body_changes() -> None:
    # The model is shown ids, slugs and titles, not the text of the guides.
    rewritten = [replace(guide, body="Different text.") for guide in GUIDES]

    assert request_hash(build_request(rewritten, "A question")) == request_hash(build_request(GUIDES, "A question"))


# --- reading the picks out of the reply text --------------------------------


def test_parse_picks_keeps_the_order_the_model_gave() -> None:
    assert parse_picks('{"slugs": ["stock", "invoices"]}', VALID_SLUGS) == ["stock", "invoices"]


def test_parse_picks_removes_repeats() -> None:
    assert parse_picks('{"slugs": ["invoices", "stock", "invoices"]}', VALID_SLUGS) == ["invoices", "stock"]


def test_parse_picks_keeps_the_first_three() -> None:
    text = '{"slugs": ["stock", "invoices", "refunds", "bookings"]}'

    assert parse_picks(text, VALID_SLUGS) == ["stock", "invoices", "refunds"]
    assert MAX_PICKS == 3


def test_parse_picks_removes_repeats_before_it_counts_to_three() -> None:
    text = '{"slugs": ["stock", "stock", "invoices", "refunds", "bookings"]}'

    assert parse_picks(text, VALID_SLUGS) == ["stock", "invoices", "refunds"]


def test_parse_picks_drops_a_slug_that_is_not_a_guide() -> None:
    assert parse_picks('{"slugs": ["no-such-guide", "refunds"]}', VALID_SLUGS) == ["refunds"]


def test_parse_picks_accepts_an_empty_list() -> None:
    assert parse_picks('{"slugs": []}', VALID_SLUGS) == []


@pytest.mark.parametrize(
    "text",
    ["I would read the invoices guide.", '["invoices"]', '{"guides": ["invoices"]}', '{"slugs": "invoices"}'],
)
def test_parse_picks_refuses_text_that_is_not_the_agreed_json(text: str) -> None:
    with pytest.raises(BaselineError):
        parse_picks(text, VALID_SLUGS)


# --- one call ---------------------------------------------------------------


def test_pick_once_sends_the_built_request_and_returns_one_results_line() -> None:
    question = make_question("en-001", "How do I give a customer their money back?")
    client = FakeClient([reply(["refunds", "invoices"], model="the-model-that-answered")])

    line = pick_once(client, GUIDES, question, TODAY, clock=ticking_clock(100.0, 101.25))

    request = build_request(GUIDES, question.question)
    assert client.messages.calls == [request]
    assert line == {
        "question_id": "en-001",
        "question_hash": question_hash(question.question),
        "request_hash": request_hash(request),
        "model": "the-model-that-answered",  # taken from the reply, not from the request
        "date": TODAY,
        "slugs": ["refunds", "invoices"],
        "stop_reason": "end_turn",
        "input_tokens": 1200,
        "output_tokens": 18,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "seconds": 1.25,  # 101.25 - 100.0
    }


def test_cache_token_counts_the_api_left_empty_are_stored_as_zero() -> None:
    answer = reply(["invoices"])
    answer.usage.cache_creation_input_tokens = None
    answer.usage.cache_read_input_tokens = None

    line = pick_once(FakeClient([answer]), GUIDES, make_question("en-001"), TODAY)

    assert line["cache_creation_input_tokens"] == 0
    assert line["cache_read_input_tokens"] == 0


def test_an_empty_list_is_stored_when_the_model_returned_one() -> None:
    line = pick_once(FakeClient([reply([])]), GUIDES, make_question("en-001"), TODAY)

    assert line["slugs"] == []


def test_an_empty_list_is_not_stored_when_the_model_named_only_things_that_are_not_guides() -> None:
    # Dropping the unknown slug would leave nothing, but the model did not say "no guide".
    client = FakeClient([reply(["no-such-guide"]), reply(["no-such-guide"])])

    with pytest.raises(BaselineError, match="en-001"):
        pick_once(client, GUIDES, make_question("en-001"), TODAY)


def test_a_reply_that_is_not_json_is_an_error_and_not_an_empty_pick() -> None:
    client = FakeClient([reply_with_text("I am not sure."), reply_with_text("I am not sure.")])

    with pytest.raises(BaselineError, match="en-001"):
        pick_once(client, GUIDES, make_question("en-001"), TODAY)


def test_a_reply_that_starts_with_a_non_text_block_is_still_read() -> None:
    answer = reply(["stock"])
    answer.content = [SimpleNamespace(type="thinking", thinking="..."), answer.content[0]]

    line = pick_once(FakeClient([answer]), GUIDES, make_question("en-001"), TODAY)

    assert line["slugs"] == ["stock"]


def test_a_reply_with_no_text_block_is_an_error() -> None:
    first, second = reply(["stock"]), reply(["stock"])
    first.content = []
    second.content = [SimpleNamespace(type="thinking", thinking="...")]

    with pytest.raises(BaselineError, match="en-001"):
        pick_once(FakeClient([first, second]), GUIDES, make_question("en-001"), TODAY)


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_a_cut_off_or_refused_reply_is_retried_once_and_then_raises(stop_reason: str) -> None:
    # The text of each reply looks fine. It must still not be used, because
    # the stop reason says the reply is not a finished answer.
    client = FakeClient(
        [
            reply(["stock"], stop_reason=stop_reason),
            reply(["stock"], stop_reason=stop_reason),
            reply(["stock"]),  # would succeed, but there is no third try
        ]
    )

    with pytest.raises(BaselineError, match="en-001") as raised:
        pick_once(client, GUIDES, make_question("en-001"), TODAY)

    assert stop_reason in str(raised.value)
    assert len(client.messages.calls) == 2


def test_a_retry_that_succeeds_gives_the_line_of_the_second_reply() -> None:
    client = FakeClient([reply(["stock"], stop_reason="max_tokens"), reply(["invoices"])])

    line = pick_once(client, GUIDES, make_question("en-001"), TODAY, clock=ticking_clock(10.0, 11.0, 20.0, 20.5))

    assert len(client.messages.calls) == 2
    assert line["slugs"] == ["invoices"]
    assert line["stop_reason"] == "end_turn"
    assert line["seconds"] == 0.5  # the second call only: 20.5 - 20.0


# --- checking stored results against the current questions and guides -------


def test_check_results_gives_the_line_of_every_reviewed_question() -> None:
    first, second = make_question("en-001"), make_question("en-002", "How do I move a booking?")
    lines = [line_for(first), line_for(second)]

    assert check_results([first, second], lines, GUIDES) == {"en-001": lines[0], "en-002": lines[1]}


def test_check_results_raises_when_a_reviewed_question_has_no_line() -> None:
    first, second = make_question("en-001"), make_question("en-002", "How do I move a booking?")

    with pytest.raises(BaselineError, match="en-002"):
        check_results([first, second], [line_for(first)], GUIDES)


def test_check_results_raises_when_a_reviewed_question_has_two_lines() -> None:
    question = make_question("en-001")

    with pytest.raises(BaselineError, match="en-001"):
        check_results([question], [line_for(question), line_for(question)], GUIDES)


def test_check_results_raises_when_the_question_text_has_changed() -> None:
    old = make_question("en-001", "How do I send an invoice?")
    new = replace(old, question="How do I email an invoice?")

    with pytest.raises(BaselineError, match="en-001"):
        check_results([new], [line_for(old)], GUIDES)


def test_check_results_raises_when_only_the_question_hash_does_not_match() -> None:
    question = make_question("en-001")
    edited = dict(line_for(question), question_hash="0" * 64)

    with pytest.raises(BaselineError, match="en-001"):
        check_results([question], [edited], GUIDES)


def test_check_results_raises_when_only_the_request_hash_does_not_match() -> None:
    question = make_question("en-001")
    edited = dict(line_for(question), request_hash="0" * 64)

    with pytest.raises(BaselineError, match="en-001"):
        check_results([question], [edited], GUIDES)


def test_check_results_raises_when_a_guide_title_has_changed_since_the_run() -> None:
    question = make_question("en-001")
    line = line_for(question, GUIDES)
    retitled = [replace(GUIDES[0], title="Invoices and receipts")] + GUIDES[1:]

    with pytest.raises(BaselineError, match="en-001"):
        check_results([question], [line], retitled)


def test_check_results_names_every_question_with_a_problem() -> None:
    first, second = make_question("en-001"), make_question("en-002", "How do I move a booking?")

    with pytest.raises(BaselineError) as raised:
        check_results([first, second], [], GUIDES)

    assert "en-001" in str(raised.value) and "en-002" in str(raised.value)


def test_check_results_ignores_unreviewed_questions() -> None:
    reviewed = make_question("en-001")
    unreviewed = make_question("ar-001", ARABIC_QUESTION, reviewed=False, language="ar")
    line = line_for(reviewed)

    # The unreviewed question has no line, and that is not a problem.
    assert check_results([reviewed, unreviewed], [line], GUIDES) == {"en-001": line}


def test_check_results_ignores_lines_of_questions_it_was_not_given() -> None:
    dev = make_question("en-001", split="dev")
    held_out = make_question("en-002", "How do I move a booking?", split="test")
    lines = [line_for(dev), line_for(held_out)]

    # Checking one split: the other split's lines are neither needed nor in the way.
    assert check_results([dev], lines, GUIDES) == {"en-001": lines[0]}
    assert check_results([held_out], [lines[1]], GUIDES) == {"en-002": lines[1]}


def test_check_results_uses_the_current_line_when_an_older_one_is_also_in_the_file() -> None:
    # After the guide list changed, the question was asked again and a second
    # line appended. The older line stays in the file as a record.
    question = make_question("en-001")
    older = line_for(question, GUIDES[:-1])
    current = line_for(question, GUIDES)

    assert check_results([question], [older, current], GUIDES) == {"en-001": current}


# --- the results file -------------------------------------------------------


def test_a_results_file_that_does_not_exist_yet_has_no_lines(tmp_path: Path) -> None:
    assert load_results(tmp_path / "title_picking.jsonl") == []


def test_load_results_names_a_line_that_is_not_a_results_line(tmp_path: Path) -> None:
    path = tmp_path / "title_picking.jsonl"
    good = json.dumps(line_for(make_question("en-001")))
    path.write_text(good + "\n" + '{"question_id": "en-002"}' + "\n", encoding="utf-8")

    with pytest.raises(BaselineError, match="line 2"):
        load_results(path)


# --- the command ------------------------------------------------------------


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests of the command need the key to be set. This is not a real key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")


class Files:
    """The three files the command reads and writes, in a temporary folder."""

    def __init__(self, folder: Path, questions: list[Question]) -> None:
        self.questions = folder / "questions.jsonl"
        self.snapshot = folder / "guides.json"
        self.results = folder / "title_picking.jsonl"
        save_questions(self.questions, questions)
        snapshot = {
            "source": SOURCE,
            "product": PRODUCT,
            "locale": LOCALE,
            "fetched_at": "2026-01-01",
            "guides": [asdict(guide) for guide in GUIDES],
        }
        self.snapshot.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(self, split: str, client: FakeClient) -> int:
        """Run the command with the fake client in place of a real one."""
        arguments = ["--split", split, "--questions", str(self.questions)]
        arguments += ["--snapshot", str(self.snapshot), "--results", str(self.results)]
        return main(arguments, make_client=lambda: client)

    def add_result(self, line: dict) -> None:
        with self.results.open("a", encoding="utf-8") as file:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")


def sent_questions(client: FakeClient) -> list[str]:
    """The question text of each call the fake client received, in order."""
    return [call["messages"][0]["content"] for call in client.messages.calls]


DEV_ONE = make_question("en-001", "How do I send an invoice?", split="dev")
DEV_NO_GUIDE = make_question("na-001", "Can it order parts for me?", split="dev", expected=())
DEV_UNREVIEWED = make_question("ar-001", ARABIC_QUESTION, split="dev", reviewed=False, language="ar")
TEST_ONE = make_question("en-002", "How do I move a booking?", split="test")
ALL_QUESTIONS = [DEV_ONE, TEST_ONE, DEV_UNREVIEWED, DEV_NO_GUIDE]


def test_the_command_sends_only_reviewed_questions_of_the_requested_split(
    tmp_path: Path, api_key: None
) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)
    client = FakeClient([reply(["invoices", "refunds"]), reply([])])

    status = files.run("dev", client)

    assert status == 0
    assert sent_questions(client) == [DEV_ONE.question, DEV_NO_GUIDE.question]


def test_the_command_reads_the_fresh_file_for_the_fresh_split(
    tmp_path: Path, api_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)
    fresh = make_question("fr-001", "Where do I see unpaid bills?", split="fresh")
    save_questions(tmp_path / "questions_fresh.jsonl", [fresh])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("guide_search.evaluation.title_picking.FRESH_PATH", Path("questions_fresh.jsonl"))
    client = FakeClient([reply(["invoices"])])

    # No --questions: the fresh split is read from the fresh file.
    arguments = ["--split", "fresh", "--snapshot", str(files.snapshot), "--results", str(files.results)]
    status = main(arguments, make_client=lambda: client)

    assert status == 0
    assert sent_questions(client) == [fresh.question]


def test_the_command_appends_one_line_for_each_question_it_sent(tmp_path: Path, api_key: None) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)

    files.run("dev", FakeClient([reply(["invoices", "refunds"]), reply([])]))

    lines = load_results(files.results)
    assert [(line["question_id"], line["slugs"]) for line in lines] == [
        ("en-001", ["invoices", "refunds"]),
        ("na-001", []),
    ]
    date.fromisoformat(lines[0]["date"])  # the date of the run; this raises unless it reads like 2026-10-12
    # The stored lines are accepted for the questions that were sent.
    assert check_results([DEV_ONE, DEV_NO_GUIDE], lines, GUIDES) == {"en-001": lines[0], "na-001": lines[1]}


def test_the_command_does_not_ask_again_for_a_question_that_has_a_current_line(
    tmp_path: Path, api_key: None
) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)
    stored = line_for(DEV_ONE)
    files.add_result(stored)
    client = FakeClient([reply([])])

    status = files.run("dev", client)

    assert status == 0
    assert sent_questions(client) == [DEV_NO_GUIDE.question]
    lines = load_results(files.results)
    assert lines[0] == stored  # the earlier line is still there, untouched
    assert [line["question_id"] for line in lines] == ["en-001", "na-001"]


def test_running_the_command_a_second_time_makes_no_calls(tmp_path: Path, api_key: None) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)
    files.run("dev", FakeClient([reply(["invoices"]), reply([])]))
    after_first_run = files.results.read_bytes()
    idle = FakeClient([])

    status = files.run("dev", idle)

    assert status == 0
    assert idle.messages.calls == []
    assert files.results.read_bytes() == after_first_run


def test_the_command_asks_again_when_the_stored_line_is_out_of_date(tmp_path: Path, api_key: None) -> None:
    files = Files(tmp_path, [DEV_ONE])
    files.add_result(line_for(DEV_ONE, GUIDES[:-1]))  # recorded when there was one guide fewer
    client = FakeClient([reply(["stock"])])

    status = files.run("dev", client)

    assert status == 0
    assert sent_questions(client) == [DEV_ONE.question]
    lines = load_results(files.results)
    assert len(lines) == 2
    assert check_results([DEV_ONE], lines, GUIDES) == {"en-001": lines[1]}


def test_the_command_refuses_to_start_without_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)
    client = FakeClient([reply(["invoices"]), reply([])])

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status_when_missing = files.run("dev", client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # what an unfilled .env file gives
    status_when_empty = files.run("dev", client)

    assert (status_when_missing, status_when_empty) == (1, 1)
    assert client.messages.calls == []
    assert not files.results.exists()
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_the_command_keeps_the_lines_already_written_when_a_question_fails(
    tmp_path: Path, api_key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    files = Files(tmp_path, ALL_QUESTIONS)
    refused = reply([], stop_reason="refusal")
    client = FakeClient([reply(["invoices"]), refused, refused])

    status = files.run("dev", client)

    assert status == 1
    assert [line["question_id"] for line in load_results(files.results)] == ["en-001"]
    assert "na-001" in capsys.readouterr().err


def test_the_command_fails_when_no_reviewed_question_is_in_the_split(
    tmp_path: Path, api_key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    files = Files(tmp_path, [DEV_UNREVIEWED, TEST_ONE])
    client = FakeClient([])

    status = files.run("dev", client)

    assert status == 1
    assert client.messages.calls == []
    assert "dev" in capsys.readouterr().err
