"""Tests for the evaluation report: the text that goes into evaluation/report.md.

Nothing here searches anything. Each test builds a small "context" by hand:
a few invented questions and what each method is supposed to have returned
for them. The numbers the report must then show are small enough to work out
by hand, and are written out in the comments.

The invented setting is a bicycle workshop with three help guides.
"""

import getpass
import os
import platform
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from guide_search.evaluation import report as report_module
from guide_search.evaluation.metrics import Outcome
from guide_search.evaluation.questions import Question
from guide_search.evaluation.report import (
    METHODS,
    container_limits,
    describe_machine,
    linux_processor_name,
    operating_system,
    rate,
    render_report,
)

PUNCTURE = "mend-a-puncture"
CHAIN = "oil-the-chain"
SADDLE = "adjust-the-saddle"

# "How do I mend the tyre?"
ARABIC_QUESTION = "كيف أصلح الإطار؟"

MACHINE = "Test processor, 4 cores, 8 GB memory, TestOS 1.0, Python 3.13.0"


def make_question(
    question_id: str,
    split: str,
    text: str = "An invented question",
    expected: tuple[str, ...] = (PUNCTURE,),
    language: str = "en",
    reviewed: bool = True,
) -> Question:
    return Question(
        id=question_id,
        language=language,
        question=text,
        expected_slugs=expected,
        # Every expected guide holds its answer in a section headed "How to".
        answer_headings={slug: ("How to",) for slug in expected},
        split=split,
        reviewed=reviewed,
    )


QUESTIONS = [
    make_question("en-t1", "test", "My tyre is flat, what now?", (PUNCTURE,)),
    make_question("en-t2", "test", "The chain squeaks", (CHAIN,)),
    make_question("na-t1", "test", "Do you sell helmets?", ()),
    make_question("en-d1", "dev", "The seat is too low", (SADDLE,)),
    make_question("en-d2", "dev", "How do I patch a tube?", (PUNCTURE,)),
    make_question("na-d1", "dev", "Can I hire a tandem?", ()),
    # Not reviewed, so it must not be counted anywhere.
    make_question("ar-d1", "dev", ARABIC_QUESTION, (PUNCTURE,), language="ar", reviewed=False),
]

# What each method returned for each question, best guide first.
# None means the method refused: it returned nothing.
RETURNED = {
    "title_picking": {
        "en-t1": (PUNCTURE,),
        "en-t2": (SADDLE, CHAIN),
        "na-t1": (SADDLE,),
        "en-d1": (SADDLE,),
        "en-d2": (CHAIN,),
        "na-d1": None,
    },
    "vector": {
        "en-t1": (PUNCTURE, CHAIN),
        "en-t2": None,
        "na-t1": None,
        "en-d1": (SADDLE,),
        "en-d2": (PUNCTURE,),
        "na-d1": None,
    },
    "keyword": {
        "en-t1": (CHAIN,),
        "en-t2": (CHAIN,),
        "na-t1": (PUNCTURE,),
        "en-d1": (SADDLE,),
        "en-d2": (PUNCTURE,),
        "na-d1": None,
    },
    "hybrid": {
        "en-t1": (PUNCTURE,),
        "en-t2": (CHAIN,),
        "na-t1": None,
        "en-d1": (SADDLE,),
        "en-d2": (PUNCTURE,),
        "na-d1": None,
    },
}

# The same two search modes with the threshold switched off: nothing is refused.
RANKING_ONLY = {
    "vector": {
        "en-t1": (PUNCTURE, CHAIN),
        "en-t2": (PUNCTURE, CHAIN),
        "na-t1": (CHAIN,),
        "en-d1": (SADDLE,),
        "en-d2": (PUNCTURE,),
        "na-d1": (CHAIN,),
    },
    "hybrid": {
        "en-t1": (PUNCTURE,),
        "en-t2": (CHAIN,),
        "na-t1": (CHAIN,),
        "en-d1": (SADDLE,),
        "en-d2": (PUNCTURE,),
        "na-d1": (CHAIN,),
    },
}

# The best similarity of each reviewed dev question. Sorted: 0.2, 0.7, 0.9.
# Candidates: 0.0 handles 2 of 3 (it never refuses), 0.45 handles 3 of 3,
# 0.8 handles 2 of 3. So the rule chooses 0.45, which is 0.25 away from the
# nearest score on each side.
DEV_SIMILARITIES = {"en-d1": 0.9, "en-d2": 0.7, "na-d1": 0.2}


def make_outcome(method: str, question_id: str, slugs: tuple[str, ...] | None, number: int) -> Outcome:
    """One outcome. `number` is the question's place in the list: 1, 2, 3, ..."""
    is_search = method != "title_picking"
    returned = slugs or ()
    return Outcome(
        question_id=question_id,
        method=method,
        refused=slugs is None,
        slugs=returned,
        # Search hands over passages, here always from the section "How to".
        # Title-picking hands over whole guides, which is written as None.
        headings={slug: ("How to",) for slug in returned} if is_search else None,
        # 100 characters for each guide from search, 1,000 for each whole guide.
        chars=len(returned) * (100 if is_search else 1000),
        # Search: 10, 40, 90, 160, 250, 360 milliseconds (the squares, so that
        # the median and the average differ). Title-picking: 1, 4, 9, ... seconds.
        seconds=number * number * (0.010 if is_search else 1.0),
        best_similarity=0.5 if is_search and method != "keyword" else None,
    )


def make_outcomes(returned: dict[str, dict[str, tuple[str, ...] | None]]) -> dict[str, dict[str, Outcome]]:
    outcomes: dict[str, dict[str, Outcome]] = {}
    for method, by_question in returned.items():
        outcomes[method] = {}
        for number, (question_id, slugs) in enumerate(by_question.items(), start=1):
            outcomes[method][question_id] = make_outcome(method, question_id, slugs, number)
    return outcomes


def make_context(**changes: object) -> dict:
    """The context of a complete, small run. `changes` replace single values."""
    outcomes = make_outcomes(RETURNED)
    # For one question, vector search found the right guide but handed over
    # the wrong section of it: a guide hit that is not a passage hit.
    outcomes["vector"]["en-t1"] = replace(
        outcomes["vector"]["en-t1"],
        headings={PUNCTURE: ("Troubleshooting",), CHAIN: ("How to",)},
    )
    context = {
        "snapshot_date": "2026-01-01",
        "guide_count": 3,
        "passage_count": 12,
        "questions_commit": "0123abcd",
        "model_id": "fake-model@rev1",
        "baseline_models": ["baseline-model-1"],
        "baseline_dates": ["2026-01-02"],
        "prompt_revisions": 0,
        "threshold": 0.45,
        "test_scored_before": False,
        "run_date": "2026-01-03",
        "machine": MACHINE,
        "questions": QUESTIONS,
        "outcomes": outcomes,
        "ranking_only": make_outcomes(RANKING_ONLY),
        "dev_similarities": DEV_SIMILARITIES,
        "corrections": [],
    }
    context.update(changes)
    return context


# --- reading the report back ------------------------------------------------


def section(text: str, heading: str) -> str:
    """The part of a markdown text under one heading, up to the next heading of the same or a higher level."""
    lines = text.split("\n")
    assert heading in lines, f"the report has no heading {heading!r}"
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for number in range(start + 1, len(lines)):
        marks = len(lines[number]) - len(lines[number].lstrip("#"))
        if 0 < marks <= level and lines[number][marks : marks + 1] == " ":
            end = number
            break
    return "\n".join(lines[start + 1 : end])


def row(text: str, *first_cells: str) -> list[str]:
    """The cells of the table row that starts with these cells."""
    for line in text.split("\n"):
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[: len(first_cells)] == list(first_cells):
            return cells
    raise AssertionError(f"no table row starts with {first_cells!r}")


def bullets(text: str) -> list[str]:
    """The top-level list items of a piece of markdown."""
    return [line for line in text.split("\n") if line.startswith("- ")]


ENGLISH = "### English questions the guides can answer"
ARABIC = "### Arabic questions the guides can answer"
UNANSWERABLE = "### Questions the guides cannot answer"
RANKING = "### Ranking only: the threshold switched off"


# --- rates ------------------------------------------------------------------


@pytest.mark.parametrize(
    "count, total, expected",
    [
        (14, 20, "14 of 20 (70%)"),
        (0, 5, "0 of 5 (0%)"),
        (5, 5, "5 of 5 (100%)"),
        (2, 3, "2 of 3 (67%)"),
        (1, 8, "1 of 8 (13%)"),  # 12.5 is rounded up, as people expect
    ],
)
def test_a_rate_is_shown_with_its_count(count: int, total: int, expected: str) -> None:
    assert rate(count, total) == expected


def test_a_rate_over_no_questions_has_no_percentage() -> None:
    assert rate(0, 0) == "0 of 0"


# --- the whole report -------------------------------------------------------


def test_the_report_has_every_section_in_order() -> None:
    report = render_report(make_context())

    headings = [line for line in report.split("\n") if line.startswith("## ")]

    assert headings == [
        "## What was run",
        "## Headline: hybrid search against title-picking",
        "## How to read the tables",
        "## Results on test",
        "## Results on dev",
        "## Time",
        "## The threshold",
        "## Misses",
        "## Label corrections",
        "## Limits",
    ]


def test_what_was_run_names_the_snapshot_the_questions_the_model_the_threshold_the_date_and_the_machine() -> None:
    what = section(render_report(make_context()), "## What was run")

    assert "2026-01-01" in what            # the snapshot date
    assert "3 guides" in what
    assert "12 passages" in what
    assert "`0123abcd`" in what            # the commit of the question file
    assert "`fake-model@rev1`" in what     # the embedding model and its revision
    assert "`baseline-model-1`" in what    # the model that picked from titles
    assert "0.4500" in what                # the threshold
    assert "2026-01-03" in what            # the run date
    assert MACHINE in what


def test_what_was_run_counts_scored_and_unreviewed_questions() -> None:
    what = section(render_report(make_context()), "## What was run")

    # Six of the seven questions are reviewed: three in dev and three in test.
    assert "6 reviewed and scored (3 dev, 3 test)" in what
    assert "1 not reviewed and not scored" in what


def test_a_run_with_no_earlier_results_file_says_only_that_none_was_found() -> None:
    what = section(render_report(make_context(test_scored_before=False)), "## What was run")

    # The code cannot know that `test` was never scored before. It only saw
    # that no results file was there, so that is all the report says.
    assert "No results file from an earlier run was found in the output folder" in what
    assert "scored again" not in what
    assert "first time" not in what


def test_a_later_run_says_that_test_was_scored_again() -> None:
    what = section(render_report(make_context(test_scored_before=True)), "## What was run")

    assert "scored again" in what
    assert "No results file" not in what


def test_what_was_run_says_how_often_the_baseline_prompt_was_revised() -> None:
    never = section(render_report(make_context(prompt_revisions=0)), "## What was run")
    twice = section(render_report(make_context(prompt_revisions=2)), "## What was run")

    assert "not revised" in never
    assert "revised 2 times" in twice


# --- the headline -----------------------------------------------------------

RIGHT = (PUNCTURE,)                # the expected guide, first
SECOND = (CHAIN, PUNCTURE)         # the expected guide, but in second place
WRONG = (CHAIN,)                   # the expected guide is missing


def headline_context(wins: int, losses: int) -> dict:
    """English test questions where hybrid alone is right `wins` times and title-picking alone `losses` times."""
    questions = []
    hybrid: dict[str, tuple[str, ...] | None] = {}
    title_picking: dict[str, tuple[str, ...] | None] = {}
    for number in range(1, wins + losses + 1):
        question = make_question(f"en-{number:03d}", "test")
        questions.append(question)
        hybrid_is_right = number <= wins
        hybrid[question.id] = RIGHT if hybrid_is_right else WRONG
        title_picking[question.id] = WRONG if hybrid_is_right else RIGHT
    # One more question that both get right. It must not enter the sign test.
    both = make_question("en-900", "test")
    questions.append(both)
    hybrid[both.id] = RIGHT
    title_picking[both.id] = RIGHT

    returned = {"title_picking": title_picking, "vector": hybrid, "keyword": hybrid, "hybrid": hybrid}
    return make_context(
        questions=questions,
        outcomes=make_outcomes(returned),
        ranking_only=make_outcomes({"vector": hybrid, "hybrid": hybrid}),
        dev_similarities={},
    )


def headline_sentence(report: str, measure: str) -> str:
    """The line of the headline that gives the verdict for hit@1 or hit@3."""
    headline = section(report, "## Headline: hybrid search against title-picking")
    for line in bullets(headline):
        if line.startswith(f"- {measure}:"):
            return line
    raise AssertionError(f"the headline has no sentence for {measure}")


def test_three_wins_and_one_loss_is_no_difference_that_this_set_can_show() -> None:
    report = render_report(headline_context(wins=3, losses=1))
    headline = section(report, "## Headline: hybrid search against title-picking")

    # Columns: measure, hybrid's hits, title-picking's hits, the two counts, p.
    # 3 against 1: p = 2 * (1 + 4) / 16 = 0.625.
    assert row(headline, "hit@1") == ["hit@1", "4 of 5 (80%)", "2 of 5 (40%)", "3", "1", "0.625"]
    assert "no difference that this set can show" in headline_sentence(report, "hit@1")
    assert "better" not in headline_sentence(report, "hit@1")
    assert "worse" not in headline_sentence(report, "hit@1")


def test_six_wins_and_no_loss_is_better() -> None:
    report = render_report(headline_context(wins=6, losses=0))
    headline = section(report, "## Headline: hybrid search against title-picking")

    # 6 against 0: p = 2 * 1 / 64 = 0.031.
    assert row(headline, "hit@1") == ["hit@1", "7 of 7 (100%)", "1 of 7 (14%)", "6", "0", "0.031"]
    assert "better" in headline_sentence(report, "hit@1")
    assert "no difference" not in headline_sentence(report, "hit@1")


def test_no_win_and_six_losses_is_worse() -> None:
    report = render_report(headline_context(wins=0, losses=6))

    assert "worse" in headline_sentence(report, "hit@1")
    assert "better" not in headline_sentence(report, "hit@1")


@pytest.mark.parametrize(
    "wins, losses, verdict",
    [
        # The design: "at least six wins with no losses, eight with one loss, or ten with two".
        (5, 0, "no difference that this set can show"),
        (6, 0, "better"),
        (7, 1, "no difference that this set can show"),
        (8, 1, "better"),
        (9, 2, "no difference that this set can show"),
        (10, 2, "better"),
        # The two cases nearest to the line: p = 0.057 and p = 0.049.
        (11, 3, "no difference that this set can show"),
        (13, 4, "better"),
        (2, 10, "worse"),
        (0, 0, "no difference that this set can show"),
    ],
)
def test_better_or_worse_is_said_only_when_the_sign_test_gives_p_below_0_05(
    wins: int, losses: int, verdict: str
) -> None:
    report = render_report(headline_context(wins, losses))

    assert verdict in headline_sentence(report, "hit@1")


def test_the_headline_judges_hit_at_1_and_hit_at_3_separately() -> None:
    # Hybrid puts the expected guide first six times. Title-picking finds it
    # too, but in second place. So hybrid wins 6 to 0 on hit@1, and on hit@3
    # nothing separates them.
    questions = [make_question(f"en-{number:03d}", "test") for number in range(1, 7)]
    hybrid = {question.id: RIGHT for question in questions}
    title_picking = {question.id: SECOND for question in questions}
    returned = {"title_picking": title_picking, "vector": hybrid, "keyword": hybrid, "hybrid": hybrid}
    report = render_report(
        make_context(
            questions=questions,
            outcomes=make_outcomes(returned),
            ranking_only=make_outcomes({"vector": hybrid, "hybrid": hybrid}),
            dev_similarities={},
        )
    )
    headline = section(report, "## Headline: hybrid search against title-picking")

    assert "better" in headline_sentence(report, "hit@1")
    assert row(headline, "hit@3") == ["hit@3", "6 of 6 (100%)", "6 of 6 (100%)", "0", "0", "1.000"]
    assert "no difference that this set can show" in headline_sentence(report, "hit@3")


def test_a_guide_that_hybrid_finds_in_second_place_wins_at_hit_at_3_but_not_at_hit_at_1() -> None:
    # Hybrid finds the expected guide six times, each time in second place.
    # Title-picking never finds it.
    questions = [make_question(f"en-{number:03d}", "test") for number in range(1, 7)]
    hybrid = {question.id: SECOND for question in questions}
    title_picking = {question.id: WRONG for question in questions}
    returned = {"title_picking": title_picking, "vector": hybrid, "keyword": hybrid, "hybrid": hybrid}
    report = render_report(
        make_context(
            questions=questions,
            outcomes=make_outcomes(returned),
            ranking_only=make_outcomes({"vector": hybrid, "hybrid": hybrid}),
            dev_similarities={},
        )
    )
    headline = section(report, "## Headline: hybrid search against title-picking")

    assert row(headline, "hit@1") == ["hit@1", "0 of 6 (0%)", "0 of 6 (0%)", "0", "0", "1.000"]
    assert row(headline, "hit@3") == ["hit@3", "6 of 6 (100%)", "0 of 6 (0%)", "6", "0", "0.031"]
    assert "no difference that this set can show" in headline_sentence(report, "hit@1")
    assert "better" in headline_sentence(report, "hit@3")


def test_the_headline_uses_english_test_questions_only() -> None:
    # In the small run hybrid is right on both English test questions and
    # title-picking puts the expected guide first for one of them. On dev,
    # title-picking misses one more question, which must not be counted here.
    headline = section(render_report(make_context()), "## Headline: hybrid search against title-picking")

    assert row(headline, "hit@1") == ["hit@1", "2 of 2 (100%)", "1 of 2 (50%)", "1", "0", "1.000"]


def test_the_headline_says_so_when_no_english_test_question_is_reviewed() -> None:
    dev_only = [question for question in QUESTIONS if question.split == "dev"]

    headline = section(render_report(make_context(questions=dev_only)), "## Headline: hybrid search against title-picking")

    assert "No reviewed English test question" in headline


# --- the tables -------------------------------------------------------------


def test_the_english_test_table_gives_every_rate_with_its_count() -> None:
    test = section(render_report(make_context()), "## Results on test")
    english = section(test, ENGLISH)

    # Columns: hit@1, hit@3, passage hit@3, MRR, wrong refusals, guides returned, text handed over.
    # Title-picking: first for en-t1, second for en-t2, so MRR is (1 + 1/2) / 2.
    # It returned 1 and 2 whole guides: 1.5 on average, 1,000 and 2,000 characters.
    assert row(english, "Title-picking") == [
        "Title-picking", "1 of 2 (50%)", "2 of 2 (100%)", "2 of 2 (100%)", "0.750", "0 of 2 (0%)", "1.5", "1,500",
    ]
    # Vector: right guide for en-t1 but the wrong section of it, and en-t2 refused.
    assert row(english, "Vector") == [
        "Vector", "1 of 2 (50%)", "1 of 2 (50%)", "0 of 2 (0%)", "0.500", "1 of 2 (50%)", "2.0", "200",
    ]
    # Keyword: wrong guide for en-t1, right guide for en-t2.
    assert row(english, "Keyword") == [
        "Keyword", "1 of 2 (50%)", "1 of 2 (50%)", "1 of 2 (50%)", "0.500", "0 of 2 (0%)", "1.0", "100",
    ]
    assert row(english, "Hybrid") == [
        "Hybrid", "2 of 2 (100%)", "2 of 2 (100%)", "2 of 2 (100%)", "1.000", "0 of 2 (0%)", "1.0", "100",
    ]


def test_a_method_that_refused_every_question_of_a_group_shows_a_dash_and_not_a_zero() -> None:
    # Here vector search refuses both English test questions. "Guides returned"
    # is an average and "text handed over" a median, both over the questions
    # that were not refused. Over no questions neither has a value.
    returned = {method: dict(by_question) for method, by_question in RETURNED.items()}
    returned["vector"]["en-t1"] = None
    returned["vector"]["en-t2"] = None
    report = render_report(make_context(outcomes=make_outcomes(returned)))

    english = section(section(report, "## Results on test"), ENGLISH)

    assert row(english, "Vector") == [
        "Vector", "0 of 2 (0%)", "0 of 2 (0%)", "0 of 2 (0%)", "0.000", "2 of 2 (100%)", "-", "-",
    ]
    # A method that answered at least one question still shows its numbers.
    assert row(english, "Hybrid")[-2:] == ["1.0", "100"]


def test_dev_and_test_are_never_pooled() -> None:
    report = render_report(make_context())
    dev_english = section(section(report, "## Results on dev"), ENGLISH)
    test_english = section(section(report, "## Results on test"), ENGLISH)

    # Title-picking is right on one of two dev questions and has the guide for
    # both test questions. Pooled, hit@3 would read "3 of 4".
    assert row(dev_english, "Title-picking")[2] == "1 of 2 (50%)"
    assert row(test_english, "Title-picking")[2] == "2 of 2 (100%)"
    assert "of 4" not in report


def test_the_unanswerable_table_gives_correct_refusals_and_answered_when_it_should_have_refused() -> None:
    test = section(render_report(make_context()), "## Results on test")
    unanswerable = section(test, UNANSWERABLE)

    # There is one such test question. Vector and hybrid refused it.
    assert row(unanswerable, "Title-picking") == ["Title-picking", "0 of 1 (0%)", "1 of 1 (100%)"]
    assert row(unanswerable, "Vector") == ["Vector", "1 of 1 (100%)", "0 of 1 (0%)"]
    assert row(unanswerable, "Keyword") == ["Keyword", "0 of 1 (0%)", "1 of 1 (100%)"]
    assert row(unanswerable, "Hybrid") == ["Hybrid", "1 of 1 (100%)", "0 of 1 (0%)"]


def test_the_ranking_only_table_shows_vector_and_hybrid_with_the_threshold_switched_off() -> None:
    test = section(render_report(make_context()), "## Results on test")
    ranking = section(test, RANKING)

    # With the threshold on, vector refused en-t2. With it off, the expected
    # guide comes back in second place: hit@1 1 of 2, hit@3 2 of 2, MRR 0.75.
    assert row(ranking, "Vector", "English") == ["Vector", "English", "1 of 2 (50%)", "2 of 2 (100%)", "0.750"]
    assert row(ranking, "Hybrid", "English") == ["Hybrid", "English", "2 of 2 (100%)", "2 of 2 (100%)", "1.000"]
    assert "Keyword" not in ranking
    assert "Title-picking" not in ranking


def test_each_split_has_its_own_ranking_only_table() -> None:
    dev = section(render_report(make_context()), "## Results on dev")

    assert row(section(dev, RANKING), "Vector", "English") == [
        "Vector", "English", "2 of 2 (100%)", "2 of 2 (100%)", "1.000",
    ]


def test_an_unreviewed_question_is_not_counted_in_any_table() -> None:
    dev = section(render_report(make_context()), "## Results on dev")

    # The only Arabic question is not reviewed, so there is nothing to show.
    assert "No reviewed question" in section(dev, ARABIC)
    assert "|" not in section(dev, ARABIC)


# --- Arabic -----------------------------------------------------------------


def test_the_report_says_it_covers_english_only_while_no_arabic_question_is_reviewed() -> None:
    report = render_report(make_context())

    assert "This report covers English only" in report


def context_with_reviewed_arabic() -> dict:
    """The small run again, with its Arabic dev question reviewed and scored."""
    questions = [replace(question, reviewed=True) if question.id == "ar-d1" else question for question in QUESTIONS]
    returned = {method: dict(by_question) for method, by_question in RETURNED.items()}
    returned["title_picking"]["ar-d1"] = (PUNCTURE,)
    returned["vector"]["ar-d1"] = (CHAIN, PUNCTURE)
    returned["keyword"]["ar-d1"] = None   # no English passage holds an Arabic word
    returned["hybrid"]["ar-d1"] = (CHAIN, PUNCTURE)
    ranking_only = {mode: dict(by_question) for mode, by_question in RANKING_ONLY.items()}
    ranking_only["vector"]["ar-d1"] = (CHAIN, PUNCTURE)
    ranking_only["hybrid"]["ar-d1"] = (CHAIN, PUNCTURE)
    return make_context(
        questions=questions,
        outcomes=make_outcomes(returned),
        ranking_only=make_outcomes(ranking_only),
        dev_similarities={**DEV_SIMILARITIES, "ar-d1": 0.6},
    )


def test_the_english_only_note_goes_away_once_an_arabic_question_is_reviewed() -> None:
    report = render_report(context_with_reviewed_arabic())

    assert "This report covers English only" not in report


def test_arabic_questions_get_their_own_table_and_are_not_mixed_with_english() -> None:
    report = render_report(context_with_reviewed_arabic())
    dev = section(report, "## Results on dev")

    # One Arabic dev question: keyword refused it, vector found its guide second.
    # Keyword answered no Arabic question, so its last two columns have no value.
    assert row(section(dev, ARABIC), "Keyword") == [
        "Keyword", "0 of 1 (0%)", "0 of 1 (0%)", "0 of 1 (0%)", "0.000", "1 of 1 (100%)", "-", "-",
    ]
    assert row(section(dev, ARABIC), "Vector")[:5] == [
        "Vector", "0 of 1 (0%)", "1 of 1 (100%)", "1 of 1 (100%)", "0.500",
    ]
    # The English dev numbers are still out of two questions.
    assert row(section(dev, ENGLISH), "Keyword")[1] == "2 of 2 (100%)"
    assert row(section(dev, RANKING), "Vector", "Arabic") == ["Vector", "Arabic", "0 of 1 (0%)", "1 of 1 (100%)", "0.500"]


# --- time -------------------------------------------------------------------


def test_time_is_the_median_and_the_maximum_over_all_questions_together() -> None:
    time_section = section(render_report(make_context()), "## Time")

    # Six questions, dev and test together. Search took 10, 40, 90, 160, 250
    # and 360 milliseconds: the median is halfway between 90 and 160.
    assert row(time_section, "Hybrid") == ["Hybrid", "6", "125 ms", "360 ms"]
    # Title-picking took 1, 4, 9, 16, 25 and 36 seconds.
    assert row(time_section, "Title-picking") == ["Title-picking", "6", "12,500 ms", "36,000 ms"]


def test_the_time_section_says_that_the_two_kinds_of_time_are_not_compared() -> None:
    time_section = section(render_report(make_context()), "## Time")

    assert "different kinds of time" in time_section
    assert "not compared" in time_section


# --- the threshold ----------------------------------------------------------


def test_the_threshold_section_lists_every_candidate_with_its_count() -> None:
    threshold = section(render_report(make_context()), "## The threshold")

    assert row(threshold, "0.0000") == ["0.0000", "2 of 3", ""]
    assert row(threshold, "0.4500") == ["0.4500", "3 of 3", "chosen"]
    assert row(threshold, "0.8000") == ["0.8000", "2 of 3", ""]


def test_the_threshold_section_gives_the_gap_to_the_nearest_dev_score_on_each_side() -> None:
    threshold = section(render_report(make_context()), "## The threshold")

    # The chosen 0.45 lies between the scores 0.2 and 0.7.
    assert "Nearest dev score below it: 0.2500 away" in threshold
    assert "Nearest dev score at or above it: 0.2500 away" in threshold


def test_the_threshold_section_gives_lowest_median_and_highest_similarity_for_each_group() -> None:
    threshold = section(render_report(make_context()), "## The threshold")

    # Columns: how many, lowest, median, highest.
    assert row(threshold, "English, answerable") == ["English, answerable", "2", "0.7000", "0.8000", "0.9000"]
    # No Arabic question is reviewed, so there is no similarity to show.
    assert row(threshold, "Arabic, answerable") == ["Arabic, answerable", "0", "-", "-", "-"]
    assert row(threshold, "Unanswerable") == ["Unanswerable", "1", "0.2000", "0.2000", "0.2000"]


def test_the_similarity_range_shows_the_median_and_not_the_average() -> None:
    # Three dev questions the guides can answer, scoring 0.5, 0.6 and 0.9.
    # The median is 0.6. The average would be 0.667.
    questions = [make_question(f"en-d{number}", "dev") for number in (1, 2, 3)]
    everywhere = {question.id: RIGHT for question in questions}
    context = make_context(
        questions=questions,
        outcomes=make_outcomes({method: everywhere for method in METHODS}),
        ranking_only=make_outcomes({"vector": everywhere, "hybrid": everywhere}),
        dev_similarities={"en-d1": 0.5, "en-d2": 0.9, "en-d3": 0.6},
        threshold=0.0,
    )

    threshold = section(render_report(context), "## The threshold")

    assert row(threshold, "English, answerable") == ["English, answerable", "3", "0.5000", "0.6000", "0.9000"]


def test_the_threshold_section_confirms_that_the_committed_value_is_what_the_rule_gives() -> None:
    threshold = section(render_report(make_context(threshold=0.45)), "## The threshold")

    assert "the same value" in threshold
    assert "not the value" not in threshold


def test_the_threshold_section_does_not_claim_the_same_value_when_the_two_differ() -> None:
    # `final` stops when the threshold is not the one the rule gives, so no
    # written report is in this state. Built by hand, the section still must
    # not say something it has not seen: here the run used 0.3 and the rule gives 0.45.
    threshold = section(render_report(make_context(threshold=0.3)), "## The threshold")

    assert "the same value" not in threshold
    assert "not the value" not in threshold
    assert "used in this run: 0.3000." in threshold
    assert "The rule's choice is 0.4500." in threshold


def test_a_difference_in_the_last_decimal_places_is_not_reported_as_another_threshold() -> None:
    # Embeddings differ slightly from one machine to another, so the same
    # rule can give 0.45 on one and 0.45001 on the next.
    threshold = section(render_report(make_context(threshold=0.45001)), "## The threshold")

    assert "the same value" in threshold


def test_the_report_says_that_search_never_refuses_when_zero_wins() -> None:
    # The unanswerable question scores above both answerable ones, so no
    # threshold beats "never refuse": 0.0 handles 2 of 3 and nothing does better.
    similarities = {"en-d1": 0.5, "en-d2": 0.6, "na-d1": 0.9}

    threshold = section(
        render_report(make_context(threshold=0.0, dev_similarities=similarities)), "## The threshold"
    )

    assert row(threshold, "0.0000") == ["0.0000", "2 of 3", "chosen"]
    assert "did not separate" in threshold
    assert "never refuse" in threshold
    # No score is below 0, and the lowest score, 0.5, is the nearest above it.
    assert "There is no dev score below it." in threshold
    assert "Nearest dev score at or above it: 0.5000 away" in threshold


def test_the_sentence_about_never_refusing_is_left_out_when_a_real_threshold_wins() -> None:
    threshold = section(render_report(make_context()), "## The threshold")

    assert "did not separate" not in threshold
    assert "never refuse" not in threshold


# --- misses -----------------------------------------------------------------


def test_a_miss_is_listed_with_the_question_what_was_expected_and_what_came_back() -> None:
    misses = section(render_report(make_context()), "## Misses")
    keyword = section(section(misses, "### Misses on test"), "#### Keyword")

    # Keyword search returned the chain guide for the flat-tyre question.
    assert "`en-t1`" in keyword
    assert "My tyre is flat, what now?" in keyword
    assert f"expected: `{PUNCTURE}`" in keyword
    assert f"returned: `{CHAIN}`" in keyword


def test_a_refused_answerable_question_is_listed_as_a_miss_that_returned_nothing() -> None:
    misses = section(render_report(make_context()), "## Misses")
    vector = section(section(misses, "### Misses on test"), "#### Vector")

    assert "`en-t2`" in vector
    assert "returned: nothing" in vector
    # en-t1 was a hit for vector search, so it is not listed.
    assert "`en-t1`" not in vector


def test_an_unanswerable_question_that_was_answered_is_listed_as_a_miss() -> None:
    misses = section(render_report(make_context()), "## Misses")
    title_picking = section(section(misses, "### Misses on test"), "#### Title-picking")

    assert "`na-t1`" in title_picking
    assert "Do you sell helmets?" in title_picking
    assert "expected: no guide" in title_picking
    assert f"returned: `{SADDLE}`" in title_picking


def test_an_expected_guide_in_second_place_is_not_a_miss() -> None:
    misses = section(render_report(make_context()), "## Misses")
    title_picking = section(section(misses, "### Misses on test"), "#### Title-picking")

    # Title-picking returned the chain guide second for en-t2: a hit at 3.
    assert "`en-t2`" not in title_picking


def test_a_method_without_misses_says_none() -> None:
    misses = section(render_report(make_context()), "## Misses")
    hybrid = section(section(misses, "### Misses on test"), "#### Hybrid")

    assert hybrid.strip() == "None."


def test_misses_are_listed_for_each_split_separately() -> None:
    misses = section(render_report(make_context()), "## Misses")
    dev = section(misses, "### Misses on dev")
    test = section(misses, "### Misses on test")

    # Title-picking missed en-d2, a dev question.
    assert "`en-d2`" in section(dev, "#### Title-picking")
    assert "`en-d2`" not in test
    assert "`en-t1`" not in dev


def test_an_unreviewed_question_is_never_listed_as_a_miss() -> None:
    report = render_report(make_context())

    assert "ar-d1" not in report
    assert ARABIC_QUESTION not in report


# --- label corrections ------------------------------------------------------


def test_the_report_says_none_when_no_label_was_corrected() -> None:
    corrections = section(render_report(make_context(corrections=[])), "## Label corrections")

    assert corrections.strip().startswith("None.")
    # What was looked at is the log of corrections, so the sentence is about the log.
    assert "No correction is logged in evaluation/corrections.jsonl" in corrections
    assert "has been changed" not in corrections


def test_a_label_correction_is_listed_with_old_and_new_guides_and_the_reason() -> None:
    correction = {
        "question_id": "en-t2",
        "date": "2026-01-04",
        "old_expected_slugs": [CHAIN],
        "new_expected_slugs": [CHAIN, PUNCTURE],
        "reason": "The puncture guide also explains how to oil the chain.",
    }

    corrections = section(render_report(make_context(corrections=[correction])), "## Label corrections")

    assert "`en-t2`" in corrections
    assert "2026-01-04" in corrections
    assert f"from `{CHAIN}` to `{CHAIN}`, `{PUNCTURE}`" in corrections
    assert "The puncture guide also explains how to oil the chain." in corrections
    assert "None." not in corrections


# --- limits -----------------------------------------------------------------


def test_the_limits_are_written_out() -> None:
    limits = section(render_report(make_context()), "## Limits")

    # The nine limits of the design, each as its own list item.
    assert len(bullets(limits)) == 9
    for phrase in [
        "The set is small",
        "about five unanswerable `dev` questions",
        "One threshold serves both languages",
        "Keyword mode has no threshold",
        "drafted by a model that has read the guides",
        "one reviewer's labels",
        "It was run once",
        "passage hit@3 is there to show the difference",
        "matched against English passages",
    ]:
        assert phrase in limits


def test_the_limits_count_the_unanswerable_questions_among_those_that_were_scored() -> None:
    # In the small run six questions are scored, and two of them are unanswerable.
    limits = section(render_report(make_context()), "## Limits")
    # With the Arabic question reviewed as well, seven are scored.
    limits_with_arabic = section(render_report(context_with_reviewed_arabic()), "## Limits")

    assert (
        "2 of the 6 scored questions are unanswerable because the set was built that way; "
        "the threshold depends on that mix."
    ) in limits
    assert "2 of the 7 scored questions are unanswerable" in limits_with_arabic
    assert "One question in six" not in limits


ARABIC_LIMITS_NOTE = (
    "The points above about Arabic results do not apply to this report yet: "
    "no Arabic question has been reviewed."
)


def test_the_limits_say_that_the_arabic_points_do_not_apply_while_no_arabic_question_is_reviewed() -> None:
    limits = section(render_report(make_context()), "## Limits").split("\n")

    # Directly under the list: the ninth limit, one empty line, then the note.
    last_limit = max(number for number, line in enumerate(limits) if line.startswith("- "))
    assert limits[last_limit + 1 : last_limit + 3] == ["", ARABIC_LIMITS_NOTE]


def test_the_note_about_the_arabic_limits_goes_away_once_an_arabic_question_is_reviewed() -> None:
    limits = section(render_report(context_with_reviewed_arabic()), "## Limits")

    assert "do not apply to this report yet" not in limits
    assert len(bullets(limits)) == 9


def test_the_report_ends_with_one_newline() -> None:
    report = render_report(make_context())

    assert report.endswith("\n")
    assert not report.endswith("\n\n")


def test_every_method_has_a_name_in_the_report() -> None:
    test = section(render_report(make_context()), "## Results on test")

    assert METHODS == ("title_picking", "vector", "keyword", "hybrid")
    for name in ["Title-picking", "Vector", "Keyword", "Hybrid"]:
        assert row(section(test, ENGLISH), name)[0] == name


# --- the machine ------------------------------------------------------------


def test_the_machine_description_has_no_host_name_user_name_or_path() -> None:
    description = describe_machine()

    assert socket.gethostname() not in description
    assert getpass.getuser() not in description
    assert str(Path.home()) not in description
    assert os.getcwd() not in description


def test_the_machine_description_gives_cores_memory_and_python_version() -> None:
    description = describe_machine()

    assert f"{os.cpu_count()} cores" in description
    assert "GB memory" in description
    assert f"Python {platform.python_version()}" in description


def test_on_linux_the_operating_system_is_given_without_the_kernel_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The report is public. "Linux" is enough for a reader; the kernel release
    # would say more about the machine than the report needs to.
    monkeypatch.setattr(report_module.sys, "platform", "linux")
    monkeypatch.setattr(report_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(report_module.platform, "release", lambda: "6.8.0-45-generic")

    assert operating_system() == "Linux"
    assert ", Linux, Python " in describe_machine()
    assert "6.8.0" not in describe_machine()


def test_on_macos_the_operating_system_is_given_with_its_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_module.sys, "platform", "darwin")
    monkeypatch.setattr(report_module.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))

    assert operating_system() == "macOS 15.1"


def test_the_machine_description_is_one_line() -> None:
    assert "\n" not in describe_machine()


def test_the_processor_name_is_read_from_the_model_name_line_on_linux(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Invented Processor 2.40GHz\nflags\t\t: fpu\n", encoding="utf-8")

    assert linux_processor_name(cpuinfo) == "Invented Processor 2.40GHz"


def test_a_linux_machine_that_gives_no_model_name_has_no_processor_name(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nBogoMIPS\t: 50.00\n", encoding="utf-8")

    assert linux_processor_name(cpuinfo) == ""
    assert linux_processor_name(tmp_path / "no-such-file") == ""


def test_the_machine_description_adds_the_limits_of_a_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_module, "container_limits", lambda: "2 CPUs and 3 GB memory")

    assert describe_machine().endswith("; this container is limited to 2 CPUs and 3 GB memory")


def test_the_machine_description_says_nothing_about_limits_outside_a_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_module, "container_limits", lambda: None)

    assert "limited" not in describe_machine()


def test_container_limits_are_read_from_the_two_files_docker_provides(tmp_path: Path) -> None:
    # 200000 of every 100000 microseconds: two processors. 3 GB in bytes.
    (tmp_path / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("3221225472\n", encoding="utf-8")

    assert container_limits(tmp_path) == "2 CPUs and 3 GB memory"


def test_no_container_limit_is_reported_when_the_files_say_max(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    assert container_limits(tmp_path) is None


def test_no_container_limit_is_reported_outside_a_container(tmp_path: Path) -> None:
    # Outside a container the two files are not there at all.
    assert container_limits(tmp_path) is None


def test_a_limit_on_only_one_of_the_two_is_still_reported(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("150000 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    assert container_limits(tmp_path) == "1.5 CPUs"
