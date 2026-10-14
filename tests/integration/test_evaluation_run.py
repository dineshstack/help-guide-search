"""Tests for the evaluation runner, against a real Postgres.

They need a real Postgres (TEST_DATABASE_URL). The guides are the five
invented guides in tests/fixtures/five_guides.json and the embedder is the
fake one, so no model is downloaded and no API is called.

The questions are invented here, for the made-up kite workshop of the fixture:
five that the guides can answer, two that they cannot, and one Arabic question
that is not reviewed. The title-picking answers are written by hand in the
shape of the stored results file.

The fake embedder only counts shared words, so its scores are on another
scale than the real model's. The tests pass their own threshold, or choose one
with the project's own rule, and never use the one in settings.toml.

Two things about `final` shape these tests. It stops unless the threshold in
the settings file is the one the rule gives on dev, so its tests run
`threshold` first. And it stops unless the split of every question is the one
assign_split gives, so the ids below are chosen to fall the way the tests need.
"""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest

from guide_search.config import Settings, load_settings, save_settings
from guide_search.embedding import FakeEmbedder
from guide_search.evaluation.metrics import (
    choose_threshold,
    hit_at,
    passage_hit,
    summarise_answerable,
    summarise_unanswerable,
)
from guide_search.evaluation.questions import Question, save_questions
from guide_search.evaluation.run import (
    NEVER_REFUSE,
    WARM_UP_QUESTION,
    main,
    search_outcomes,
    title_picking_outcomes,
)
from guide_search.evaluation.title_picking import (
    MODEL,
    BaselineError,
    build_request,
    question_hash,
    request_hash,
)
from guide_search.ingest import ingest
from guide_search.reranking import FakeReranker
from guide_search.search import MODES
from guide_search.snapshot import PUBLIC_PAGE_PREFIX, Guide, load_snapshot

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"

# The tests' own "too weak to be a match" line for the fake embedder.
THRESHOLD = 0.4

SAIL = "mend-a-torn-sail"
SPAR = "replace-a-snapped-spar"
QUESTIONS_GUIDE = "workshop-questions"
REMINDER = "send-a-pickup-reminder"

# "How do I mend the torn sail?" No English passage contains these words.
ARABIC = "كيف أصلح الشراع الممزق؟"


def make_question(
    question_id: str,
    text: str,
    split: str,
    answer_headings: dict[str, tuple[str, ...]],
    language: str = "en",
    reviewed: bool = True,
) -> Question:
    """A question whose expected guides are the guides that have a section listed."""
    return Question(
        id=question_id,
        language=language,
        question=text,
        expected_slugs=tuple(answer_headings),
        answer_headings=answer_headings,
        split=split,
        reviewed=reviewed,
    )


# The commands check that each split is the one assign_split gives, so the
# split cannot be chosen freely here. For five English ids it puts en-001,
# en-004 and en-005 in dev and en-002 and en-003 in test; of two unanswerable
# ids the first is dev; a group of one is dev.

# --- dev ---
SUNDAY = make_question(
    "en-001", "is the workshop open on sunday", "dev",
    {QUESTIONS_GUIDE: ("Is the workshop open on Sunday?",)},
)
SMS = make_question(
    "en-004", "send a pickup reminder by sms", "dev",
    {REMINDER: ("Send the reminder by SMS",)},
)
# Either of two guides is an acceptable answer to this one.
SPAR_OR_SAIL = make_question(
    "en-005", "a snapped spar or a torn sail", "dev",
    {SPAR: ("Fit the new spar",), SAIL: ("Patch the tear",)},
)
# About the workshop, but nothing the guides cover.
BELLS = make_question("na-001", "do you sell bicycle bells", "dev", {})
# Not reviewed, so nothing about it may be scored.
UNREVIEWED = make_question("ar-001", ARABIC, "dev", {SAIL: ("Patch the tear",)}, language="ar", reviewed=False)

# --- test ---
CARD = make_question("en-002", "can I pay by card", "test", {QUESTIONS_GUIDE: ("Can I pay by card?",)})
# Only two words, so its best similarity is low: 0.32 with the fake embedder.
CODE = make_question("en-003", "pickup code", "test", {REMINDER: ("Send the reminder by SMS",)})
# Not covered either, but "counter" is a word of one passage.
COUNTER = make_question("na-002", "does the counter sell bicycle bells", "test", {})

DEV = [SUNDAY, SMS, SPAR_OR_SAIL, BELLS, UNREVIEWED]
TEST = [CARD, CODE, COUNTER]
ALL_QUESTIONS = DEV + TEST
REVIEWED_IDS = ["en-001", "en-004", "en-005", "na-001", "en-002", "en-003", "na-002"]

# The two passages keyword search returns for "pickup code", copied from the fixture.
SMS_PASSAGE = "Open the repair ticket and press Remind. The owner receives an SMS with the pickup code."
REMINDER_INTRO = "When a repair is ready, the owner is told to collect the kite."

# What the baseline model is supposed to have picked, written by hand.
PICKED = {
    "en-001": [QUESTIONS_GUIDE, REMINDER],
    "en-004": [SAIL],                # the wrong guide
    "en-005": [SAIL, SPAR],
    "na-001": [],                    # it picked nothing, which is right
    "en-002": [QUESTIONS_GUIDE],
    "en-003": [REMINDER],
    "na-002": [QUESTIONS_GUIDE],     # it answered when it should have refused
}


class RecordingEmbedder(FakeEmbedder):
    """The fake embedder, but it remembers every question it was asked to embed."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text)


class HalfSecondEmbedder(FakeEmbedder):
    """The fake embedder with a clock of its own: every question it embeds takes half a second."""

    def __init__(self) -> None:
        self.now = 0.0

    def embed_query(self, text: str) -> list[float]:
        self.now += 0.5
        return super().embed_query(text)

    def clock(self) -> float:
        return self.now


def fixture_guides() -> list[Guide]:
    return load_snapshot(FIXTURE)


def baseline_line(question: Question, slugs: list[str], seconds: float = 1.5) -> dict:
    """One line of the title-picking results file, as a real run would have stored it."""
    return {
        "question_id": question.id,
        "question_hash": question_hash(question.question),
        "request_hash": request_hash(build_request(fixture_guides(), question.question)),
        "model": MODEL,
        "date": "2026-01-02",
        "slugs": slugs,
        "stop_reason": "end_turn",
        "input_tokens": 900,
        "output_tokens": 12,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "seconds": seconds,
    }


def baseline_lines() -> list[dict]:
    """A stored answer for every reviewed question."""
    return [baseline_line(question, PICKED[question.id]) for question in ALL_QUESTIONS if question.reviewed]


@pytest.fixture
def indexed(conn: psycopg.Connection) -> psycopg.Connection:
    """The test database with the five fixture guides ingested."""
    ingest(conn, fixture_guides(), FakeEmbedder())
    return conn


# ---------------------------------------------------------------------------
# search_outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_puts_the_expected_guide_first_for_the_dev_questions_the_guides_answer(
    indexed: psycopg.Connection, mode: str
):
    outcomes = search_outcomes(indexed, FakeEmbedder(), DEV, mode, THRESHOLD, FakeReranker())

    for question in [SUNDAY, SMS, SPAR_OR_SAIL]:
        outcome = outcomes[question.id]
        assert outcome.question_id == question.id
        assert outcome.method == mode
        assert outcome.refused is False
        assert hit_at(question, outcome, 1) is True


def test_the_returned_slugs_are_recorded_best_first(indexed: psycopg.Connection):
    outcomes = search_outcomes(indexed, FakeEmbedder(), DEV, "vector", THRESHOLD)

    assert outcomes["en-001"].slugs[0] == QUESTIONS_GUIDE
    assert outcomes["en-004"].slugs[0] == REMINDER
    # With 16 passages in the table every guide is a candidate, so three come back.
    assert len(outcomes["en-001"].slugs) == 3


def test_a_question_with_two_expected_guides_is_a_hit_whichever_of_them_comes_first(
    indexed: psycopg.Connection,
):
    vector = search_outcomes(indexed, FakeEmbedder(), [SPAR_OR_SAIL], "vector", THRESHOLD)["en-005"]
    keyword = search_outcomes(indexed, FakeEmbedder(), [SPAR_OR_SAIL], "keyword", THRESHOLD)["en-005"]

    # The two modes disagree about the order, and both orders are a hit at 1.
    assert vector.slugs[0] == SAIL
    assert keyword.slugs[0] == SPAR
    assert hit_at(SPAR_OR_SAIL, vector, 1) is True
    assert hit_at(SPAR_OR_SAIL, keyword, 1) is True


def test_headings_and_chars_are_those_of_the_passages_returned(indexed: psycopg.Connection):
    outcome = search_outcomes(indexed, FakeEmbedder(), [CODE], "keyword", THRESHOLD)["en-003"]

    # "pickup" is in the title of one guide only, so only its passages match.
    # The passage that also holds "code" comes first. The other two each hold
    # "pickup" once, in the title, and the earlier one (the intro) is kept.
    assert outcome.refused is False
    assert outcome.slugs == (REMINDER,)
    assert outcome.headings == {REMINDER: ("Send the reminder by SMS", "")}
    # The passage texts only: no title, no heading, no URL.
    assert outcome.chars == len(SMS_PASSAGE) + len(REMINDER_INTRO)
    assert outcome.chars == 150
    assert passage_hit(CODE, outcome) is True


def test_vector_and_hybrid_record_the_best_similarity_and_keyword_records_none(
    indexed: psycopg.Connection,
):
    vector = search_outcomes(indexed, FakeEmbedder(), [SUNDAY], "vector", THRESHOLD)["en-001"]
    hybrid = search_outcomes(indexed, FakeEmbedder(), [SUNDAY], "hybrid", THRESHOLD)["en-001"]
    keyword = search_outcomes(indexed, FakeEmbedder(), [SUNDAY], "keyword", THRESHOLD)["en-001"]

    # Worked by hand for the fake embedder. The question has 6 different
    # words. The passage about Sunday, with its title and heading, holds all
    # of them: "is", "the", "workshop" and "sunday" twice, "open" once and
    # "on" three times, which adds up to 12. The lengths of the two word-count
    # vectors are sqrt(6) and sqrt(34).
    expected = 12 / math.sqrt(6 * 34)
    assert vector.best_similarity == pytest.approx(expected)
    assert hybrid.best_similarity == pytest.approx(expected)
    assert keyword.best_similarity is None


def test_a_refused_question_has_no_slugs_no_headings_and_no_text(indexed: psycopg.Connection):
    for mode in MODES:
        # The fake reranker's best score for this question is 0.4, so 0.5 refuses it.
        outcome = search_outcomes(indexed, FakeEmbedder(), [BELLS], mode, THRESHOLD, FakeReranker(), 0.5)["na-001"]

        assert outcome.refused is True
        assert outcome.slugs == ()
        assert outcome.headings == {}
        assert outcome.chars == 0


def test_a_refusal_by_the_threshold_still_records_the_best_similarity(indexed: psycopg.Connection):
    outcome = search_outcomes(indexed, FakeEmbedder(), [BELLS], "vector", THRESHOLD)["na-001"]

    # "do" and "you" are in one passage of 17 different words: 2 / sqrt(5 * 17).
    assert outcome.refused is True
    assert outcome.best_similarity == pytest.approx(2 / math.sqrt(5 * 17))
    assert outcome.best_similarity < THRESHOLD


def test_a_weak_answerable_question_is_a_wrong_refusal_for_vector_and_hybrid_but_not_for_keyword(
    indexed: psycopg.Connection,
):
    # summarise_answerable counts the two test questions the guides can answer.
    vector = summarise_answerable(TEST, search_outcomes(indexed, FakeEmbedder(), TEST, "vector", THRESHOLD))
    hybrid = summarise_answerable(TEST, search_outcomes(indexed, FakeEmbedder(), TEST, "hybrid", THRESHOLD))
    keyword = summarise_answerable(TEST, search_outcomes(indexed, FakeEmbedder(), TEST, "keyword", THRESHOLD))

    # "pickup code" scores 0.32, below the tests' threshold of 0.4.
    assert (vector.hit3, vector.wrong_refusals) == (1, 1)
    assert (hybrid.hit3, hybrid.wrong_refusals) == (1, 1)
    # Keyword search has no threshold and finds both.
    assert (keyword.hit3, keyword.wrong_refusals) == (2, 0)


def test_keyword_answers_an_uncovered_question_that_shares_a_word_with_a_passage(
    indexed: psycopg.Connection,
):
    vector = search_outcomes(indexed, FakeEmbedder(), TEST, "vector", THRESHOLD)
    keyword = search_outcomes(indexed, FakeEmbedder(), TEST, "keyword", THRESHOLD)

    # One test question is not covered by the guides. Vector search refuses it.
    assert summarise_unanswerable(TEST, vector).correct_refusals == 1
    # Keyword search answers it: "counter" is in the passage about paying by card.
    assert keyword["na-002"].refused is False
    assert keyword["na-002"].slugs == (QUESTIONS_GUIDE,)
    assert summarise_unanswerable(TEST, keyword).correct_refusals == 0


def test_with_the_threshold_at_zero_nothing_is_refused(indexed: psycopg.Connection):
    outcomes = search_outcomes(indexed, FakeEmbedder(), DEV + TEST, "hybrid", 0.0)

    assert [outcome.refused for outcome in outcomes.values()] == [False] * 7


def test_the_time_of_a_search_includes_embedding_the_question_and_leaves_out_the_warm_up(
    indexed: psycopg.Connection,
):
    # On this embedder's clock, embedding a question takes half a second and
    # nothing else takes any time.
    embedder = HalfSecondEmbedder()

    outcomes = search_outcomes(indexed, embedder, [SUNDAY, SMS], "hybrid", THRESHOLD, clock=embedder.clock)

    # Half a second each. The warm-up query was embedded too, before the
    # first question, and its half second is in neither time.
    assert outcomes["en-001"].seconds == 0.5
    assert outcomes["en-004"].seconds == 0.5


def test_a_keyword_search_embeds_nothing_so_it_takes_no_time_on_the_embedders_clock(
    indexed: psycopg.Connection,
):
    embedder = HalfSecondEmbedder()

    outcomes = search_outcomes(indexed, embedder, [SUNDAY], "keyword", THRESHOLD, clock=embedder.clock)

    assert outcomes["en-001"].seconds == 0.0


def test_one_warm_up_query_is_run_first_and_thrown_away(indexed: psycopg.Connection):
    embedder = RecordingEmbedder()

    outcomes = search_outcomes(indexed, embedder, [SUNDAY, SMS], "vector", THRESHOLD)

    assert embedder.queries == [WARM_UP_QUESTION, SUNDAY.question, SMS.question]
    assert list(outcomes) == ["en-001", "en-004"]


def test_an_unreviewed_question_is_not_searched_and_gets_no_outcome(indexed: psycopg.Connection):
    embedder = RecordingEmbedder()

    outcomes = search_outcomes(indexed, embedder, [SUNDAY, UNREVIEWED], "vector", THRESHOLD)

    assert list(outcomes) == ["en-001"]
    assert ARABIC not in embedder.queries


# ---------------------------------------------------------------------------
# title_picking_outcomes (no database is used)
# ---------------------------------------------------------------------------


def tiny_guide(guide_id: int, slug: str, body: str) -> Guide:
    return Guide(
        id=guide_id,
        slug=slug,
        title=f"Guide {guide_id}",
        url=PUBLIC_PAGE_PREFIX + slug,
        updated_at="2026-01-01T00:00:00+00:00",
        body=body,
    )


# Two guides whose bodies are exactly 300 and 100 characters long, heading
# lines and the "Related articles" section included.
TINY_GUIDES = [
    tiny_guide(1, "first-guide", "## Steps\n\n" + "a" * 290),
    tiny_guide(2, "second-guide", "b" * 60 + "\n\n## Related articles\n\n" + "c" * 17),
]


def test_title_picking_hands_over_whole_guides_so_chars_is_the_length_of_their_bodies():
    question = make_question("en-001", "an invented question", "dev", {"first-guide": ("Steps",)})
    # Only the two fields of a stored line that are read here.
    lines = {"en-001": {"slugs": ["second-guide", "first-guide"], "seconds": 2.25}}

    outcome = title_picking_outcomes([question], lines, TINY_GUIDES)["en-001"]

    assert [len(guide.body) for guide in TINY_GUIDES] == [300, 100]
    assert outcome.method == "title_picking"
    assert outcome.slugs == ("second-guide", "first-guide")   # in the stored order
    assert outcome.headings is None                            # the whole guide, not passages
    assert outcome.chars == 100 + 300
    assert outcome.seconds == 2.25                             # as stored when the baseline ran
    assert outcome.refused is False
    assert outcome.best_similarity is None
    # The whole guide was handed over, so finding the guide is a passage hit.
    assert passage_hit(question, outcome) is True


def test_an_empty_stored_pick_is_a_refusal():
    question = make_question("na-001", "an invented question", "dev", {})
    lines = {"na-001": {"slugs": [], "seconds": 1.0}}

    outcome = title_picking_outcomes([question], lines, TINY_GUIDES)["na-001"]

    assert outcome.refused is True
    assert outcome.slugs == ()
    assert outcome.chars == 0


def test_a_reviewed_question_without_a_stored_line_is_an_error_and_not_a_refusal():
    question = make_question("en-001", "an invented question", "dev", {"first-guide": ("Steps",)})

    with pytest.raises(BaselineError, match="en-001"):
        title_picking_outcomes([question], {}, TINY_GUIDES)


def test_title_picking_gives_no_outcome_for_an_unreviewed_question():
    reviewed = make_question("en-001", "an invented question", "dev", {"first-guide": ("Steps",)})
    unreviewed = replace(reviewed, id="en-002", reviewed=False)
    lines = {
        "en-001": {"slugs": ["first-guide"], "seconds": 1.0},
        "en-002": {"slugs": ["first-guide"], "seconds": 1.0},
    }

    outcomes = title_picking_outcomes([reviewed, unreviewed], lines, TINY_GUIDES)

    assert list(outcomes) == ["en-001"]


# ---------------------------------------------------------------------------
# The threshold, chosen from the best similarities
# ---------------------------------------------------------------------------


def test_the_threshold_chosen_from_the_dev_similarities_refuses_the_unanswerable_question(
    indexed: psycopg.Connection,
):
    # The scores the rule works on: a vector search with refusing switched
    # off, so the question the guides cannot answer has a best similarity too.
    unrefused = search_outcomes(indexed, FakeEmbedder(), DEV, "vector", NEVER_REFUSE)
    similarities = {question_id: outcome.best_similarity for question_id, outcome in unrefused.items()}
    reviewed_dev = [SUNDAY, SMS, SPAR_OR_SAIL, BELLS]
    assert list(similarities) == ["en-001", "en-004", "en-005", "na-001"]

    choice = choose_threshold([(similarities[q.id], q.answerable) for q in reviewed_dev])

    # The three answerable questions score above the unanswerable one, so one
    # threshold handles all four correctly, and it lies between the two groups.
    assert (choice.correct, choice.total) == (4, 4)
    lowest_answerable = min(similarities[q.id] for q in [SUNDAY, SMS, SPAR_OR_SAIL])
    assert similarities["na-001"] < choice.threshold < lowest_answerable

    for mode in ["vector", "hybrid"]:
        outcomes = search_outcomes(indexed, FakeEmbedder(), DEV, mode, choice.threshold, FakeReranker())
        assert outcomes["na-001"].refused is True
        assert [outcomes[q.id].refused for q in [SUNDAY, SMS, SPAR_OR_SAIL]] == [False, False, False]


# ---------------------------------------------------------------------------
# The three commands
# ---------------------------------------------------------------------------


# `final` cannot run without these two options. The other two commands accept
# them and do nothing with them.
FINAL_OPTIONS = ["--questions-commit", "abc1234", "--prompt-revisions", "0"]


class Files:
    """The files the command reads and writes, in a temporary folder."""

    def __init__(
        self,
        folder: Path,
        questions: list[Question] = ALL_QUESTIONS,
        lines: list[dict] | None = None,
        threshold: float = THRESHOLD,
    ) -> None:
        self.questions = folder / "questions.jsonl"
        self.baseline = folder / "title_picking.jsonl"
        self.settings = folder / "settings.toml"
        self.corrections = folder / "corrections.jsonl"
        # Neither file is written by default, so the repository's own files are never read.
        self.fresh = folder / "questions_fresh.jsonl"
        self.choice = folder / "reranker_choice.json"
        self.tokens = folder / "tokens.json"
        self.output = folder / "out"
        self.results = self.output / "results.json"
        self.report = self.output / "report.md"

        save_questions(self.questions, questions)
        # The default mode is not the usual one, to show that it is kept.
        save_settings(Settings(threshold=threshold, default_mode="keyword"), self.settings)
        if lines is None:
            lines = baseline_lines()
        text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        self.baseline.write_text(text, encoding="utf-8")

    def run(self, command: str, *more: str, embedder: FakeEmbedder | None = None) -> int:
        """Run the command on these files, with the fake embedder in place of the real model."""
        arguments = [command, "--snapshot", str(FIXTURE), "--questions", str(self.questions)]
        arguments += ["--baseline", str(self.baseline), "--settings", str(self.settings)]
        arguments += ["--corrections", str(self.corrections), "--output-dir", str(self.output)]
        arguments += ["--fresh", str(self.fresh), "--choice", str(self.choice), "--tokens", str(self.tokens)]
        arguments += list(more)
        return main(arguments, embedder=embedder or FakeEmbedder(), reranker=FakeReranker())

    def final(self, *more: str, embedder: FakeEmbedder | None = None) -> int:
        """Run `final`, telling it the two things it cannot run without."""
        return self.run("final", *FINAL_OPTIONS, *more, embedder=embedder)

    def read_results(self) -> dict:
        return json.loads(self.results.read_text(encoding="utf-8"))

    def entry(self, question_id: str, method: str) -> dict:
        """The one entry of results.json for this question and method."""
        found = [
            entry
            for entry in self.read_results()["outcomes"]
            if entry["question_id"] == question_id and entry["method"] == method
        ]
        assert len(found) == 1
        return found[0]


@pytest.fixture
def ready(indexed: psycopg.Connection, database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The index is built, and DATABASE_URL points the command at the test database."""
    monkeypatch.setenv("DATABASE_URL", database_url)


def files_after_threshold(
    folder: Path,
    questions: list[Question] = ALL_QUESTIONS,
    lines: list[dict] | None = None,
) -> Files:
    """The files as `threshold` leaves them: the settings file holds the value the rule gives.

    `final` stops with any other threshold, so its tests start from here. With
    the questions above the rule gives 0.362: halfway between "do you sell
    bicycle bells" (0.217) and "a snapped spar or a torn sail" (0.507).
    """
    files = Files(folder, questions=questions, lines=lines, threshold=0.0)
    assert files.run("threshold") == 0
    return files


# --- threshold ---


def test_threshold_writes_both_chosen_values_and_the_default_mode_to_the_settings_file(
    tmp_path: Path, ready: None
):
    files = Files(tmp_path, threshold=0.0)

    status = files.run("threshold")

    assert status == 0
    settings = load_settings(files.settings)
    # Between "do you sell bicycle bells" (0.22) and the weakest answerable
    # dev question, "a snapped spar or a torn sail" (0.51).
    assert 0.22 < settings.threshold < 0.50
    # The fake reranker's best scores: 1.0, 1.0 and 0.5 for the three
    # answerable dev questions, 0.4 for "do you sell bicycle bells". The rule
    # takes the midpoint between 0.4 and 0.5.
    assert settings.rerank_threshold == pytest.approx(0.45)
    # Both modes return an answering passage for all three answerable dev
    # questions and refuse the fourth. A tie keeps hybrid, replacing the
    # file's starting value.
    assert settings.default_mode == "hybrid"


def test_threshold_prints_the_candidates_the_gaps_and_the_three_ranges(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    files = Files(tmp_path, threshold=0.0)

    files.run("threshold")

    printed = capsys.readouterr().out
    assert "| 0.0000 | 3 of 4 |" in printed          # never refusing gets the unanswerable one wrong
    assert "| 4 of 4 | chosen |" in printed
    assert "Nearest dev score below it" in printed
    assert "Nearest dev score at or above it" in printed
    assert "| English, answerable | 3 |" in printed
    assert "| Arabic, answerable | 0 |" in printed    # the Arabic question is not reviewed
    assert "| Unanswerable | 1 |" in printed
    assert "## The rerank threshold" in printed
    assert "## The default mode" in printed
    assert "The rule's choice is hybrid." in printed
    assert str(files.settings) in printed


def test_threshold_looks_at_dev_questions_only(tmp_path: Path, ready: None):
    files = Files(tmp_path, threshold=0.0)
    embedder = RecordingEmbedder()

    files.run("threshold", embedder=embedder)

    # Each pass starts with the warm-up query, then searches the four reviewed
    # dev questions. Nothing else is searched: no test question, no unreviewed one.
    reviewed_dev = [SUNDAY.question, SMS.question, SPAR_OR_SAIL.question, BELLS.question]
    assert set(embedder.queries) == {WARM_UP_QUESTION, *reviewed_dev}
    assert embedder.queries[:5] == [WARM_UP_QUESTION, *reviewed_dev]


def test_threshold_writes_no_report_and_no_results(tmp_path: Path, ready: None):
    files = Files(tmp_path, threshold=0.0)

    files.run("threshold")

    assert not files.output.exists()


# --- dev ---


def test_dev_prints_the_dev_tables_and_the_dev_misses(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    files = Files(tmp_path)

    status = files.run("dev")

    printed = capsys.readouterr().out
    assert status == 0
    assert "## Results on dev" in printed
    # Three English dev questions. Title-picking picked the wrong guide for one.
    assert "| Title-picking | 2 of 3 (67%) | 2 of 3 (67%) |" in printed
    assert "| Hybrid | 3 of 3 (100%) | 3 of 3 (100%) |" in printed
    assert "### Misses on dev" in printed
    assert "`en-004` send a pickup reminder by sms" in printed


def test_dev_writes_nothing(tmp_path: Path, ready: None):
    files = Files(tmp_path)
    settings_before = files.settings.read_bytes()

    files.run("dev")

    assert not files.output.exists()
    assert files.settings.read_bytes() == settings_before


def test_dev_never_scores_a_test_question(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    # The baseline has answers for dev only, as it does before the reported run.
    dev_lines = [baseline_line(question, PICKED[question.id]) for question in DEV if question.reviewed]
    files = Files(tmp_path, lines=dev_lines)
    embedder = RecordingEmbedder()

    status = files.run("dev", embedder=embedder)

    printed = capsys.readouterr().out
    assert status == 0
    for question in TEST:
        assert question.question not in embedder.queries
        assert question.id not in printed
    assert "Results on test" not in printed


# --- final ---


def test_final_writes_the_report_and_the_results_and_names_both_files(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    # The order the commands are meant to be run in: `threshold` puts the
    # value the rule gives into the settings file, and only then `final`.
    files = Files(tmp_path, threshold=0.0)
    assert files.run("threshold") == 0

    status = files.final()

    assert status == 0
    assert files.report.is_file()
    assert files.results.is_file()
    last_line = capsys.readouterr().out.strip().split("\n")[-1]
    assert str(files.report) in last_line
    assert str(files.results) in last_line


def test_the_results_hold_the_threshold_the_snapshot_date_and_the_model(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.final()

    results = files.read_results()
    # The threshold that was used: the one `threshold` wrote to the settings file.
    assert results["threshold"] == load_settings(files.settings).threshold
    assert results["threshold"] == pytest.approx(0.3620, abs=0.0001)
    assert results["fetched_at"] == "2026-01-01"    # the fixture's fetched_at
    assert results["model_id"] == "fake"             # the fake embedder's model_id


def test_the_results_hold_one_entry_for_each_reviewed_question_and_method(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.final()

    entries = files.read_results()["outcomes"]
    pairs = [(entry["question_id"], entry["method"]) for entry in entries]
    methods = ["title_picking", "vector", "keyword", "hybrid", "rerank"]
    assert pairs == [(question_id, method) for question_id in REVIEWED_IDS for method in methods]
    for entry in entries:
        assert set(entry) == {
            "question_id", "split", "method", "refused", "slugs", "hit_at_3", "passage_hit", "best_similarity",
            "best_rerank_score",
        }


def test_an_unreviewed_question_is_not_in_the_results_and_not_in_the_report(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)
    embedder = RecordingEmbedder()

    files.final(embedder=embedder)

    assert "ar-001" not in files.results.read_text(encoding="utf-8")
    assert "ar-001" not in files.report.read_text(encoding="utf-8")
    assert ARABIC not in embedder.queries


def test_a_results_entry_says_what_the_method_did_with_the_question(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.final()

    # "pickup code": refused by the threshold in vector mode, found by keyword.
    refused = files.entry("en-003", "vector")
    assert refused["split"] == "test"
    assert refused["refused"] is True
    assert refused["slugs"] == []
    assert refused["hit_at_3"] is False
    assert refused["passage_hit"] is False
    assert refused["best_similarity"] == pytest.approx(0.3162, abs=0.0001)

    found = files.entry("en-003", "keyword")
    assert found["refused"] is False
    assert found["slugs"] == [REMINDER]
    assert found["hit_at_3"] is True
    assert found["passage_hit"] is True
    assert found["best_similarity"] is None


def test_the_results_tell_a_guide_hit_from_a_passage_hit(tmp_path: Path, ready: None):
    # The answer is in the section "Before you start". Keyword search finds
    # the right guide through the words "patch", "torn" and "sail", and hands
    # over two other sections of it.
    question = make_question(
        "en-001", "what do I do before I patch a torn sail", "dev", {SAIL: ("Before you start",)}
    )
    # With one dev question, and the guides can answer it, the rule gives 0:
    # no threshold does better than never refusing. So 0.0 is the right value here.
    files = Files(tmp_path, questions=[question], lines=[baseline_line(question, [SAIL])], threshold=0.0)
    assert files.run("threshold") == 0

    files.final()

    keyword = files.entry("en-001", "keyword")
    assert keyword["slugs"] == [SAIL]
    assert keyword["hit_at_3"] is True
    assert keyword["passage_hit"] is False
    # Title-picking hands over the whole guide, so the section is in it.
    assert files.entry("en-001", "title_picking")["passage_hit"] is True


def test_the_results_hold_the_stored_title_picking_answers(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.final()

    wrong_guide = files.entry("en-004", "title_picking")
    assert wrong_guide["slugs"] == [SAIL]
    assert wrong_guide["hit_at_3"] is False
    picked_nothing = files.entry("na-001", "title_picking")
    assert picked_nothing["refused"] is True
    assert picked_nothing["slugs"] == []
    right_guide = files.entry("en-001", "title_picking")
    assert right_guide["slugs"] == [QUESTIONS_GUIDE, REMINDER]
    assert right_guide["hit_at_3"] is True
    assert right_guide["passage_hit"] is True       # the whole guide was handed over
    assert right_guide["best_similarity"] is None


def test_the_report_is_built_from_what_the_methods_really_returned(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.final()

    report = files.report.read_text(encoding="utf-8")
    # The two English test questions: vector search finds one and refuses "pickup code".
    assert "| Vector | 1 of 2 (50%) | 1 of 2 (50%) | 1 of 2 (50%) | 0.500 | 1 of 2 (50%) |" in report
    assert "| Keyword | 2 of 2 (100%) | 2 of 2 (100%) | 2 of 2 (100%) | 1.000 | 0 of 2 (0%) |" in report
    # The one test question the guides cannot answer: keyword search answers it.
    assert "| Keyword | 0 of 1 (0%) | 1 of 1 (100%) |" in report
    assert "| Vector | 1 of 1 (100%) | 0 of 1 (0%) |" in report
    # Ranking only: with the threshold off, vector search finds "pickup code" too.
    assert "| Vector | English | 2 of 2 (100%) | 2 of 2 (100%) | 1.000 |" in report


def test_the_report_records_what_was_run(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.run("final", "--questions-commit", "abc1234", "--prompt-revisions", "1")

    report = files.report.read_text(encoding="utf-8")
    assert "the snapshot fetched on 2026-01-01, 5 guides, 16 passages in the index" in report
    assert "commit `abc1234`" in report
    assert "7 reviewed and scored (4 dev, 3 test); 1 not reviewed and not scored" in report
    assert "- Embedding model: `fake`" in report
    assert f"`{MODEL}`" in report
    assert "answers recorded on 2026-01-02" in report
    assert "revised 1 time " in report
    assert "- Threshold: 0.3620." in report
    assert "- Machine: " in report
    assert "This report covers English only" in report
    # The limits count the questions of this run: two of the seven scored ones.
    assert "2 of the 7 scored questions are unanswerable" in report


def test_final_without_questions_commit_stops_before_doing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # Nothing is set up here: no index, and not even DATABASE_URL. The message
    # is still the one about the option, so nothing else was tried first.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    files = Files(tmp_path)

    status = files.run("final", "--prompt-revisions", "0")

    error = capsys.readouterr().err
    assert status == 1
    assert "--questions-commit" in error
    assert "DATABASE_URL" not in error
    assert not files.output.exists()


def test_final_without_prompt_revisions_stops_before_doing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    files = Files(tmp_path)

    status = files.run("final", "--questions-commit", "abc1234")

    error = capsys.readouterr().err
    assert status == 1
    assert "--prompt-revisions" in error
    assert "DATABASE_URL" not in error
    assert not files.output.exists()


def test_the_first_final_run_finds_no_earlier_results_file_and_a_second_run_says_scored_again(
    tmp_path: Path, ready: None
):
    files = files_after_threshold(tmp_path)

    files.final()
    first = files.report.read_text(encoding="utf-8")
    files.final()
    second = files.report.read_text(encoding="utf-8")

    # The first report says what was seen, not "scored for the first time":
    # the command cannot know what was run somewhere else.
    assert "No results file from an earlier run was found in the output folder" in first
    assert "scored again" not in first
    assert "scored again" in second
    assert "No results file" not in second


def test_threshold_then_final_reports_that_the_committed_threshold_is_what_the_rule_gives(
    tmp_path: Path, ready: None
):
    files = Files(tmp_path, threshold=0.0)

    files.run("threshold")
    files.final()

    report = files.report.read_text(encoding="utf-8")
    assert "gives the same value" in report
    assert files.read_results()["threshold"] == load_settings(files.settings).threshold


def test_final_stops_before_any_test_question_is_searched_when_the_threshold_is_not_the_one_the_rule_gives(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    # The settings file still holds its starting value, 0.0: `threshold` was
    # never run. On these dev questions the rule gives 0.362.
    files = Files(tmp_path, threshold=0.0)
    embedder = RecordingEmbedder()

    status = files.final(embedder=embedder)

    assert status == 1
    # The file's starting values are all three wrong: 0.0, 0.0 and keyword.
    assert capsys.readouterr().err.strip() == (
        "settings.toml does not hold the values the rules give on dev "
        "(threshold, rerank_threshold, default_mode). Run `threshold` first. "
        "No test or fresh question was searched."
    )
    assert not files.output.exists()
    # The dev questions were searched: the rule works on their scores.
    assert SUNDAY.question in embedder.queries
    for question in TEST:
        assert question.question not in embedder.queries


def test_final_accepts_a_threshold_that_differs_from_the_rule_only_in_its_last_decimal_places(
    tmp_path: Path, ready: None
):
    # Embeddings differ slightly from one machine to another, so the rule can
    # give 0.3620 on one and 0.3621 on the next. Less than 0.0005 apart
    # counts as the same value. 0.0005 or more does not.
    files = files_after_threshold(tmp_path)
    chosen = load_settings(files.settings)

    save_settings(replace(chosen, threshold=chosen.threshold + 0.0004), files.settings)
    close_enough = files.final()
    save_settings(replace(chosen, threshold=chosen.threshold + 0.0006), files.settings)
    too_far = files.final()

    assert (close_enough, too_far) == (0, 1)


def test_final_holds_the_rerank_threshold_to_the_same_tolerance(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)
    chosen = load_settings(files.settings)

    save_settings(replace(chosen, rerank_threshold=chosen.rerank_threshold + 0.0004), files.settings)
    close_enough = files.final()
    save_settings(replace(chosen, rerank_threshold=chosen.rerank_threshold + 0.0006), files.settings)
    too_far = files.final()

    assert (close_enough, too_far) == (0, 1)


def test_final_stops_when_the_default_mode_is_not_the_one_the_rule_gives(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    files = files_after_threshold(tmp_path)
    save_settings(replace(load_settings(files.settings), default_mode="rerank"), files.settings)

    status = files.final()

    assert status == 1
    assert "(default_mode)" in capsys.readouterr().err
    assert not files.output.exists()


def test_final_does_not_change_the_settings_file(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)
    settings_before = files.settings.read_bytes()

    files.final()

    assert files.settings.read_bytes() == settings_before


def test_final_lists_a_logged_label_correction(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)
    correction = {
        "question_id": "en-005",
        "date": "2026-01-05",
        "old_expected_slugs": [SPAR],
        "new_expected_slugs": [SPAR, SAIL],
        "reason": "The sail guide answers it as well.",
    }
    files.corrections.write_text(json.dumps(correction) + "\n", encoding="utf-8")

    status = files.final()

    report = files.report.read_text(encoding="utf-8")
    assert status == 0
    assert f"`en-005` (2026-01-05): expected guides changed from `{SPAR}` to `{SPAR}`, `{SAIL}`" in report
    assert "The sail guide answers it as well." in report


def test_final_says_none_when_there_is_no_corrections_file(tmp_path: Path, ready: None):
    files = files_after_threshold(tmp_path)

    files.final()

    assert "None. No correction is logged in evaluation/corrections.jsonl" in files.report.read_text(
        encoding="utf-8"
    )


# --- when the run cannot be trusted, nothing is written ---


def test_final_refuses_when_a_stored_baseline_answer_is_missing(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    lines = [line for line in baseline_lines() if line["question_id"] != "en-002"]
    files = files_after_threshold(tmp_path, lines=lines)

    status = files.final()

    assert status == 1
    assert "en-002: no stored result" in capsys.readouterr().err
    assert not files.output.exists()


def test_final_refuses_when_a_stored_baseline_answer_is_for_another_wording_of_the_question(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    # The answer was recorded for a question that has since been reworded.
    old_wording = replace(CARD, question="can I pay with a card")
    lines = [line for line in baseline_lines() if line["question_id"] != "en-002"]
    lines.append(baseline_line(old_wording, [QUESTIONS_GUIDE]))
    files = files_after_threshold(tmp_path, lines=lines)

    status = files.final()

    assert status == 1
    assert "en-002: the stored result is out of date" in capsys.readouterr().err
    assert not files.output.exists()


@pytest.mark.parametrize("field", ["question_hash", "request_hash"])
def test_final_refuses_zero_prompt_revisions_when_the_baseline_file_holds_an_out_of_date_line(
    field: str, tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    # An older answer for en-001 is still in the file, recorded for a request
    # that is not the current one. The current answer is there too, so the
    # stored results can be used. But the question was asked twice, and the
    # report would say "asked once" and "not revised".
    older = dict(baseline_line(SUNDAY, [SAIL]), **{field: "0" * 64})
    files = files_after_threshold(tmp_path, lines=[older] + baseline_lines())
    embedder = RecordingEmbedder()

    status = files.final("--prompt-revisions", "0", embedder=embedder)

    error = capsys.readouterr().err
    assert status == 1
    assert "--prompt-revisions" in error
    assert "en-001" in error
    assert embedder.queries == []
    assert not files.output.exists()


def test_final_runs_with_an_out_of_date_line_when_the_revisions_are_counted(tmp_path: Path, ready: None):
    older = dict(baseline_line(SUNDAY, [SAIL]), request_hash="0" * 64)
    files = files_after_threshold(tmp_path, lines=[older] + baseline_lines())

    status = files.final("--prompt-revisions", "1")

    assert status == 0
    assert "revised 1 time " in files.report.read_text(encoding="utf-8")
    # The current answer is the one that is scored, not the older one.
    assert files.entry("en-001", "title_picking")["slugs"] == [QUESTIONS_GUIDE, REMINDER]


def test_an_out_of_date_line_for_a_question_that_is_not_scored_does_not_stop_final(
    tmp_path: Path, ready: None
):
    # ar-001 is not reviewed. It is not scored, so its lines are not looked at.
    stray = dict(baseline_line(UNREVIEWED, [SAIL]), request_hash="0" * 64)
    files = files_after_threshold(tmp_path, lines=baseline_lines() + [stray])

    assert files.final() == 0


def test_final_refuses_a_correction_that_does_not_match_the_question_file(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    files = files_after_threshold(tmp_path)
    correction = {
        "question_id": "en-005",
        "date": "2026-01-05",
        "old_expected_slugs": [SPAR],
        "new_expected_slugs": [SPAR],    # the question file lists two guides
        "reason": "An invented reason.",
    }
    files.corrections.write_text(json.dumps(correction) + "\n", encoding="utf-8")

    status = files.final()

    assert status == 1
    assert "en-005" in capsys.readouterr().err
    assert not files.output.exists()


def test_final_refuses_a_correction_for_a_question_that_does_not_exist(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    files = files_after_threshold(tmp_path)
    correction = {
        "question_id": "en-999",
        "date": "2026-01-05",
        "old_expected_slugs": [],
        "new_expected_slugs": [SPAR],
        "reason": "An invented reason.",
    }
    files.corrections.write_text(json.dumps(correction) + "\n", encoding="utf-8")

    status = files.final()

    assert status == 1
    assert "en-999" in capsys.readouterr().err
    assert not files.output.exists()


@pytest.mark.parametrize("command", ["threshold", "dev", "final"])
def test_an_empty_index_is_an_error_and_not_a_run_of_refusals(
    command: str,
    tmp_path: Path,
    conn: psycopg.Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    # `conn` gives an empty passages table: nothing has been ingested.
    monkeypatch.setenv("DATABASE_URL", database_url)
    files = Files(tmp_path)
    settings_before = files.settings.read_bytes()

    # The two options are for `final`. The other two commands do nothing with them.
    status = files.run(command, *FINAL_OPTIONS)

    assert status == 1
    assert "run the ingestion first" in capsys.readouterr().err
    assert not files.output.exists()
    assert files.settings.read_bytes() == settings_before


def test_an_index_that_does_not_match_the_snapshot_is_an_error(
    tmp_path: Path,
    conn: psycopg.Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    # Only four of the five guides are in the index: 13 passages, not 16.
    ingest(conn, fixture_guides()[:4], FakeEmbedder())
    monkeypatch.setenv("DATABASE_URL", database_url)
    files = Files(tmp_path)

    status = files.final()

    error = capsys.readouterr().err
    assert status == 1
    assert "13" in error
    assert "16" in error
    assert not files.output.exists()


def test_a_question_without_a_split_is_an_error(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    questions = [replace(question, split=None) for question in ALL_QUESTIONS]
    files = Files(tmp_path, questions=questions)

    status = files.final()

    assert status == 1
    assert "assign-split" in capsys.readouterr().err
    assert not files.output.exists()


@pytest.mark.parametrize("command", ["threshold", "dev", "final"])
def test_a_split_that_is_not_the_one_assign_split_gives_is_an_error(
    command: str, tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    # en-004 was moved by hand from dev to test. If that passed, a question
    # the threshold was chosen on could be scored as a test question.
    moved = [replace(question, split="test") if question.id == SMS.id else question for question in ALL_QUESTIONS]
    files = Files(tmp_path, questions=moved)
    settings_before = files.settings.read_bytes()
    embedder = RecordingEmbedder()

    status = files.run(command, *FINAL_OPTIONS, embedder=embedder)

    assert status == 1
    assert capsys.readouterr().err.strip() == "The split in the question file is not the one assign-split gives."
    assert embedder.queries == []
    assert not files.output.exists()
    assert files.settings.read_bytes() == settings_before


def test_a_question_that_expects_a_guide_the_snapshot_does_not_have_is_an_error(
    tmp_path: Path, ready: None, capsys: pytest.CaptureFixture[str]
):
    stray = make_question("en-006", "an invented question", "dev", {"no-such-guide": ("Anything",)})
    files = Files(tmp_path, questions=ALL_QUESTIONS + [stray])

    status = files.run("dev")

    assert status == 1
    assert "no-such-guide" in capsys.readouterr().err


def test_without_database_url_the_command_says_so_and_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    files = Files(tmp_path)

    status = files.final()

    assert status == 1
    assert "DATABASE_URL is not set" in capsys.readouterr().err
    assert not files.output.exists()


# --- small things ---


def test_importing_the_run_module_does_not_load_torch():
    # The real model is imported inside main(), so that importing this module
    # stays quick. A fresh Python process is used because another test in the
    # same run may already have loaded torch into this one.
    code = (
        "import sys\n"
        "import guide_search.evaluation.run\n"
        "print('torch' in sys.modules, 'sentence_transformers' in sys.modules)\n"
    )
    # The new process must find guide_search in the same places as this one.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", "False"]
