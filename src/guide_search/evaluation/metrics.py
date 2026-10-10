"""The numbers the evaluation reports, worked out from questions and outcomes.

An Outcome is what one method did with one question: which guides it
returned, best first, or that it refused (returned nothing). Everything here
is plain arithmetic on lists that are already in memory. There is no
database, no model and no network, so each rule can be checked by hand.

Three things live here:

- the measures for one question (hit@1, hit@3, passage hit, reciprocal rank)
  and their totals over a list of questions;
- the rule that chooses the "too weak to be a match" threshold;
- the sign test that decides whether one method beat another.
"""

import math
import statistics
from dataclasses import dataclass

from guide_search.evaluation.questions import Question

# The measures look at the first three guides, because every method returns at most three.
TOP = 3


@dataclass(frozen=True)
class Outcome:
    question_id: str
    method: str                                  # "title_picking", "vector", "keyword" or "hybrid"
    refused: bool
    slugs: tuple[str, ...]                       # guides returned, best first (empty when refused)
    # slug -> headings of the passages returned; None means the whole guide was handed over
    headings: dict[str, tuple[str, ...]] | None
    chars: int                                   # text handed over
    seconds: float
    best_similarity: float | None


# --- one question -----------------------------------------------------------


def hit_at(question: Question, outcome: Outcome, k: int) -> bool:
    """True when one of the first k guides returned is an expected one."""
    if outcome.refused:
        return False
    return any(slug in question.expected_slugs for slug in outcome.slugs[:k])


def reciprocal_rank(question: Question, outcome: Outcome) -> float:
    """1 divided by the position of the first expected guide: 1, 1/2, 1/3, or 0 if absent."""
    if outcome.refused:
        return 0.0
    for position, slug in enumerate(outcome.slugs[:TOP], start=1):
        if slug in question.expected_slugs:
            return 1.0 / position
    return 0.0


def passage_hit(question: Question, outcome: Outcome) -> bool:
    """True when an expected guide came back through a section that holds the answer.

    Finding the right guide is not enough for search: the passages it hands
    over must come from a section listed in the question's answer_headings.
    """
    if not hit_at(question, outcome, TOP):
        return False
    if outcome.headings is None:
        # The whole guide was handed over, so the right section is in it.
        return True
    for slug in outcome.slugs[:TOP]:
        if slug not in question.expected_slugs:
            continue
        wanted = question.answer_headings.get(slug, ())
        returned = outcome.headings.get(slug, ())
        if any(heading in wanted for heading in returned):
            return True
    return False


# --- totals over a list of questions ----------------------------------------


@dataclass(frozen=True)
class AnswerableSummary:
    total: int
    hit1: int
    hit3: int
    passage_hit3: int
    mrr: float
    wrong_refusals: int
    guides_returned: float | None   # mean over questions that were not refused; None when there are none
    chars_median: float | None      # median over questions that were not refused; None when there are none


@dataclass(frozen=True)
class UnanswerableSummary:
    total: int
    correct_refusals: int


def outcome_for(question: Question, outcomes: dict[str, Outcome]) -> Outcome:
    """The outcome of one question. A missing one is an error, never counted as a refusal."""
    if question.id not in outcomes:
        raise KeyError(f"no outcome for question {question.id}")
    return outcomes[question.id]


def summarise_answerable(questions: list[Question], outcomes: dict[str, Outcome]) -> AnswerableSummary:
    """Totals over the answerable questions in the list. Unanswerable ones are left out."""
    hit1 = 0
    hit3 = 0
    passage_hit3 = 0
    rank_sum = 0.0
    total = 0
    answered: list[Outcome] = []  # the outcomes that were not refused
    for question in questions:
        if not question.answerable:
            continue
        outcome = outcome_for(question, outcomes)
        total += 1
        if hit_at(question, outcome, 1):
            hit1 += 1
        if hit_at(question, outcome, TOP):
            hit3 += 1
        if passage_hit(question, outcome):
            passage_hit3 += 1
        rank_sum += reciprocal_rank(question, outcome)
        if not outcome.refused:
            answered.append(outcome)

    # An average or a median over no questions has no value. None says so;
    # 0 would read as "it returned no guides", which is something else.
    guides_returned = None
    chars_median = None
    if answered:
        guides_returned = sum(len(outcome.slugs) for outcome in answered) / len(answered)
        chars_median = float(statistics.median(outcome.chars for outcome in answered))

    return AnswerableSummary(
        total=total,
        hit1=hit1,
        hit3=hit3,
        passage_hit3=passage_hit3,
        mrr=rank_sum / total if total else 0.0,
        wrong_refusals=total - len(answered),
        guides_returned=guides_returned,
        chars_median=chars_median,
    )


def summarise_unanswerable(questions: list[Question], outcomes: dict[str, Outcome]) -> UnanswerableSummary:
    """Totals over the unanswerable questions in the list. Answerable ones are left out."""
    total = 0
    correct_refusals = 0
    for question in questions:
        if question.answerable:
            continue
        total += 1
        if outcome_for(question, outcomes).refused:
            correct_refusals += 1
    return UnanswerableSummary(total=total, correct_refusals=correct_refusals)


# --- choosing the threshold -------------------------------------------------


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    correct: int
    total: int
    table: list[tuple[float, int]]   # every candidate and how many questions it handles correctly
    gap_below: float | None          # distance to the nearest score below the threshold
    gap_above: float | None          # distance to the nearest score at or above it


def threshold_candidates(points: list[tuple[float, bool]]) -> list[float]:
    """0.0 (never refuse) and the midpoint between each pair of neighbouring different scores."""
    scores = sorted({score for score, _ in points})
    candidates = {0.0}
    for lower, upper in zip(scores, scores[1:]):
        candidates.add((lower + upper) / 2)
    return sorted(candidates)


def count_correct(points: list[tuple[float, bool]], threshold: float) -> int:
    """How many questions this threshold handles correctly."""
    correct = 0
    for score, answerable in points:
        refused = score < threshold
        if answerable and not refused:
            correct += 1
        if not answerable and refused:
            correct += 1
    return correct


def choose_threshold(points: list[tuple[float, bool]]) -> ThresholdChoice:
    """Pick the threshold that handles the most questions correctly.

    Each point is one question: (its best similarity, whether the guides can
    answer it). A question is refused when its score is below the threshold.
    """
    table = [(candidate, count_correct(points, candidate)) for candidate in threshold_candidates(points)]

    # The table runs from the lowest candidate up, and a later candidate only
    # wins by being strictly better. So a tie keeps the lowest: when in doubt,
    # search answers rather than refuses.
    threshold, correct = table[0]
    for candidate, candidate_correct in table:
        if candidate_correct > correct:
            threshold, correct = candidate, candidate_correct

    below = [score for score, _ in points if score < threshold]
    at_or_above = [score for score, _ in points if score >= threshold]
    return ThresholdChoice(
        threshold=threshold,
        correct=correct,
        total=len(points),
        table=table,
        gap_below=threshold - max(below) if below else None,
        gap_above=min(at_or_above) - threshold if at_or_above else None,
    )


# --- did one method beat another? -------------------------------------------


def sign_test(wins: int, losses: int) -> float:
    """The exact two-sided sign test: the p-value for `wins` against `losses`.

    If the two methods were equally good, each question that only one of them
    got right would be a coin toss. The p-value is the chance that fair coin
    tosses come out at least this lopsided, in either direction. Questions
    that both got right, or both got wrong, are not part of the count.
    """
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses cannot be negative")
    tosses = wins + losses
    if tosses == 0:
        return 1.0
    smaller = min(wins, losses)
    # The number of ways to get `smaller` or fewer of one side, out of all 2**tosses outcomes.
    ways = sum(math.comb(tosses, count) for count in range(smaller + 1))
    # Doubled for the two directions. An even result would give more than 1, so cap it.
    return min(1.0, 2 * ways / 2**tosses)
