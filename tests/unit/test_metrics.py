"""Tests for the evaluation measures, the threshold rule and the sign test.

Every question and outcome here is invented and small enough to work out by
hand. The expected numbers are written out in the comments.
"""

import pytest

from guide_search.evaluation.metrics import (
    AnswerableSummary,
    Outcome,
    UnanswerableSummary,
    choose_default_mode,
    choose_threshold,
    hit_at,
    passage_hit,
    reciprocal_rank,
    sign_test,
    summarise_answerable,
    summarise_unanswerable,
)
from guide_search.evaluation.questions import Question


def make_question(
    question_id: str = "en-001",
    expected: tuple[str, ...] = ("invoices",),
    headings: dict[str, tuple[str, ...]] | None = None,
) -> Question:
    return Question(
        id=question_id,
        language="en",
        question="An invented question",
        expected_slugs=expected,
        answer_headings=headings or {},
        split="dev",
        reviewed=True,
    )


def returned(
    question_id: str = "en-001",
    slugs: tuple[str, ...] = (),
    headings: dict[str, tuple[str, ...]] | None = None,
    chars: int = 0,
    method: str = "hybrid",
) -> Outcome:
    """The outcome of a method that returned these guides, best first."""
    return Outcome(
        question_id=question_id,
        method=method,
        refused=False,
        slugs=slugs,
        headings=headings,
        chars=chars,
        seconds=0.01,
        best_similarity=0.85,
    )


def refused(question_id: str = "en-001") -> Outcome:
    """The outcome of a method that returned nothing."""
    return Outcome(
        question_id=question_id,
        method="hybrid",
        refused=True,
        slugs=(),
        headings={},
        chars=0,
        seconds=0.01,
        best_similarity=0.70,
    )


# --- hit@1, hit@3 and reciprocal rank ---------------------------------------


def test_the_expected_guide_in_first_place_is_a_hit_at_1_and_at_3_with_rank_1() -> None:
    question = make_question(expected=("invoices",))
    outcome = returned(slugs=("invoices", "refunds", "stock"))

    assert hit_at(question, outcome, 1) is True
    assert hit_at(question, outcome, 3) is True
    assert reciprocal_rank(question, outcome) == 1.0  # found at position 1: 1/1


def test_the_expected_guide_in_second_place_is_a_hit_at_3_only_with_rank_one_half() -> None:
    question = make_question(expected=("invoices",))
    outcome = returned(slugs=("refunds", "invoices", "stock"))

    assert hit_at(question, outcome, 1) is False
    assert hit_at(question, outcome, 3) is True
    assert reciprocal_rank(question, outcome) == 0.5  # found at position 2: 1/2


def test_the_expected_guide_in_third_place_has_rank_one_third() -> None:
    question = make_question(expected=("invoices",))
    outcome = returned(slugs=("refunds", "stock", "invoices"))

    assert hit_at(question, outcome, 3) is True
    assert reciprocal_rank(question, outcome) == pytest.approx(1 / 3)  # position 3


def test_no_expected_guide_among_those_returned_is_a_miss_with_rank_0() -> None:
    question = make_question(expected=("invoices",))
    outcome = returned(slugs=("refunds", "stock", "bookings"))

    assert hit_at(question, outcome, 1) is False
    assert hit_at(question, outcome, 3) is False
    assert reciprocal_rank(question, outcome) == 0.0


def test_with_several_expected_guides_finding_any_one_of_them_counts() -> None:
    question = make_question(expected=("invoices", "payments"))
    outcome = returned(slugs=("payments", "stock"))

    assert hit_at(question, outcome, 1) is True
    assert reciprocal_rank(question, outcome) == 1.0


def test_with_several_expected_guides_the_rank_is_that_of_the_first_one_found() -> None:
    question = make_question(expected=("invoices", "payments"))
    # "payments" is at position 2 and "invoices" at position 3; the earlier one counts.
    outcome = returned(slugs=("stock", "payments", "invoices"))

    assert hit_at(question, outcome, 1) is False
    assert hit_at(question, outcome, 3) is True
    assert reciprocal_rank(question, outcome) == 0.5


def test_a_method_that_returned_fewer_than_three_guides_is_scored_on_what_it_returned() -> None:
    question = make_question(expected=("invoices",))

    found_alone = returned(slugs=("invoices",))
    assert hit_at(question, found_alone, 1) is True
    assert hit_at(question, found_alone, 3) is True
    assert reciprocal_rank(question, found_alone) == 1.0

    found_second_of_two = returned(slugs=("refunds", "invoices"))
    assert hit_at(question, found_second_of_two, 3) is True
    assert reciprocal_rank(question, found_second_of_two) == 0.5

    missed_alone = returned(slugs=("refunds",))
    assert hit_at(question, missed_alone, 3) is False
    assert reciprocal_rank(question, missed_alone) == 0.0


def test_a_guide_in_fourth_place_counts_for_nothing() -> None:
    # No method returns more than three guides. If one ever did, hit@3 and
    # the reciprocal rank must still agree that only the first three count.
    question = make_question(expected=("invoices",))
    outcome = returned(slugs=("refunds", "stock", "bookings", "invoices"))

    assert hit_at(question, outcome, 3) is False
    assert reciprocal_rank(question, outcome) == 0.0


def test_a_refused_answerable_question_scores_0_on_everything() -> None:
    question = make_question(expected=("invoices",), headings={"invoices": ("Print it",)})
    outcome = refused()

    assert hit_at(question, outcome, 1) is False
    assert hit_at(question, outcome, 3) is False
    assert passage_hit(question, outcome) is False
    assert reciprocal_rank(question, outcome) == 0.0


def test_refused_is_the_field_that_decides() -> None:
    # `slugs` should be empty when `refused` is true. If the two ever
    # disagree, the measures go by `refused`, as the refusal counts do.
    question = make_question(expected=("invoices",), headings={"invoices": ("Print it",)})
    outcome = Outcome(
        question_id="en-001",
        method="hybrid",
        refused=True,
        slugs=("invoices",),
        headings={"invoices": ("Print it",)},
        chars=100,
        seconds=0.01,
        best_similarity=0.70,
    )

    assert hit_at(question, outcome, 1) is False
    assert passage_hit(question, outcome) is False
    assert reciprocal_rank(question, outcome) == 0.0


# --- passage hit ------------------------------------------------------------


def test_passage_hit_needs_a_returned_passage_from_a_listed_section() -> None:
    question = make_question(expected=("invoices",), headings={"invoices": ("Print it", "Send it")})
    outcome = returned(
        slugs=("refunds", "invoices"),
        headings={"refunds": ("",), "invoices": ("Troubleshooting", "Send it")},
    )

    assert passage_hit(question, outcome) is True  # "Send it" is one of the listed sections


def test_passage_hit_is_false_when_the_right_guide_came_back_through_the_wrong_section() -> None:
    question = make_question(expected=("invoices",), headings={"invoices": ("Print it",)})
    outcome = returned(slugs=("invoices",), headings={"invoices": ("Troubleshooting", "")})

    assert hit_at(question, outcome, 3) is True
    assert passage_hit(question, outcome) is False


def test_passage_hit_counts_the_intro_like_any_other_section() -> None:
    question = make_question(expected=("invoices",), headings={"invoices": ("",)})

    assert passage_hit(question, returned(slugs=("invoices",), headings={"invoices": ("",)})) is True
    assert passage_hit(question, returned(slugs=("invoices",), headings={"invoices": ("Print it",)})) is False


def test_a_listed_heading_only_counts_inside_its_own_guide() -> None:
    question = make_question(expected=("invoices",), headings={"invoices": ("Troubleshooting",)})
    # Another guide's "Troubleshooting" section came back, not the expected guide's.
    outcome = returned(
        slugs=("refunds", "invoices"),
        headings={"refunds": ("Troubleshooting",), "invoices": ("Print it",)},
    )

    assert passage_hit(question, outcome) is False


def test_passage_hit_with_several_expected_guides_needs_one_of_them_through_its_section() -> None:
    question = make_question(
        expected=("invoices", "payments"),
        headings={"invoices": ("Print it",), "payments": ("Take a card",)},
    )
    wrong_then_right = returned(
        slugs=("invoices", "payments"),
        headings={"invoices": ("Troubleshooting",), "payments": ("Take a card",)},
    )
    both_wrong = returned(
        slugs=("invoices", "payments"),
        headings={"invoices": ("Troubleshooting",), "payments": ("Troubleshooting",)},
    )

    assert passage_hit(question, wrong_then_right) is True
    assert passage_hit(question, both_wrong) is False


def test_passage_hit_equals_hit_at_3_when_the_whole_guide_was_handed_over() -> None:
    question = make_question(expected=("invoices",), headings={"invoices": ("Print it",)})
    # headings=None is how title-picking reports that it hands over whole guides.
    found = returned(slugs=("refunds", "invoices"), headings=None, method="title_picking")
    missed = returned(slugs=("refunds", "stock"), headings=None, method="title_picking")

    assert passage_hit(question, found) is True
    assert passage_hit(question, missed) is False


# --- summaries --------------------------------------------------------------


def test_summarise_answerable_adds_up_four_questions() -> None:
    questions = [
        make_question("en-001", expected=("invoices",), headings={"invoices": ("Send it",)}),
        make_question("en-002", expected=("refunds",), headings={"refunds": ("Give money back",)}),
        make_question("en-003", expected=("bookings",), headings={"bookings": ("",)}),
        make_question("en-004", expected=("stock",), headings={"stock": ("",)}),
    ]
    outcomes = {
        # First place, through the listed section: hit@1, hit@3, passage hit, rank 1.
        "en-001": returned(
            "en-001",
            slugs=("invoices", "refunds", "stock"),
            headings={"invoices": ("Send it",), "refunds": ("",), "stock": ("",)},
            chars=400,
        ),
        # Second place, through another section: hit@3 only, rank 1/2.
        "en-002": returned(
            "en-002",
            slugs=("stock", "refunds"),
            headings={"stock": ("",), "refunds": ("Troubleshooting",)},
            chars=900,
        ),
        # Not found: nothing, rank 0.
        "en-003": returned(
            "en-003",
            slugs=("invoices", "stock", "reminders"),
            headings={"invoices": ("",), "stock": ("",), "reminders": ("",)},
            chars=1000,
        ),
        # Refused: nothing, rank 0, and a wrong refusal.
        "en-004": refused("en-004"),
    }

    summary = summarise_answerable(questions, outcomes)

    assert summary == AnswerableSummary(
        total=4,
        hit1=1,
        hit3=2,
        passage_hit3=1,
        mrr=pytest.approx(0.375),  # (1 + 1/2 + 0 + 0) / 4
        wrong_refusals=1,
        guides_returned=pytest.approx(8 / 3),  # (3 + 2 + 3) / 3: the refused question is left out
        chars_median=900.0,  # the middle of 400, 900, 1000: the refused question is left out
    )


def test_the_median_of_an_even_number_of_questions_is_the_mean_of_the_middle_two() -> None:
    questions = [make_question("en-001"), make_question("en-002")]
    outcomes = {
        "en-001": returned("en-001", ("invoices",), {"invoices": ("",)}, 400),
        "en-002": returned("en-002", ("invoices", "stock"), {"invoices": ("",), "stock": ("",)}, 900),
    }

    summary = summarise_answerable(questions, outcomes)

    assert summary.chars_median == 650.0  # (400 + 900) / 2
    assert summary.guides_returned == 1.5  # (1 + 2) / 2


def test_summarise_answerable_with_no_questions_counts_nothing_and_has_no_average_or_median() -> None:
    # The counts are 0. The two measures over answered questions are None:
    # an average or a median over no questions has no value, and 0 would be a wrong one.
    assert summarise_answerable([], {}) == AnswerableSummary(
        total=0, hit1=0, hit3=0, passage_hit3=0, mrr=0.0, wrong_refusals=0, guides_returned=None, chars_median=None
    )


def test_when_every_question_is_refused_the_measures_over_answered_questions_have_no_value() -> None:
    questions = [make_question("en-001"), make_question("en-002")]
    outcomes = {"en-001": refused("en-001"), "en-002": refused("en-002")}

    summary = summarise_answerable(questions, outcomes)

    assert summary.total == 2
    assert summary.wrong_refusals == 2
    assert summary.mrr == 0.0
    # No question was answered, so there is nothing to average and no middle value.
    assert summary.guides_returned is None
    assert summary.chars_median is None


def test_summarise_answerable_leaves_out_unanswerable_questions() -> None:
    questions = [make_question("en-001"), make_question("na-001", expected=())]
    outcomes = {"en-001": returned("en-001", ("invoices",), {"invoices": ("",)}, 300), "na-001": refused("na-001")}

    summary = summarise_answerable(questions, outcomes)

    assert summary.total == 1
    assert summary.wrong_refusals == 0


def test_summarise_unanswerable_counts_the_questions_that_were_refused() -> None:
    questions = [
        make_question("na-001", expected=()),
        make_question("na-002", expected=()),
        make_question("na-003", expected=()),
        make_question("en-001"),  # answerable, so it is not counted here
    ]
    outcomes = {
        "na-001": refused("na-001"),
        "na-002": returned("na-002", ("stock",), {"stock": ("",)}, 500),  # answered when it should have refused
        "na-003": refused("na-003"),
        "en-001": refused("en-001"),
    }

    assert summarise_unanswerable(questions, outcomes) == UnanswerableSummary(total=3, correct_refusals=2)


def test_a_question_without_an_outcome_is_an_error_not_a_refusal() -> None:
    with pytest.raises(KeyError, match="en-001"):
        summarise_answerable([make_question("en-001")], {})
    with pytest.raises(KeyError, match="na-001"):
        summarise_unanswerable([make_question("na-001", expected=())], {})


# --- choosing the threshold -------------------------------------------------
# A point is (best similarity, answerable). A question is refused when its
# score is below the threshold. "Correct" means an answerable question was
# not refused, or an unanswerable one was.

ANSWERABLE = True
UNANSWERABLE = False


def test_the_candidates_are_zero_and_the_midpoints_between_neighbouring_scores() -> None:
    points = [(0.90, ANSWERABLE), (0.80, UNANSWERABLE), (0.84, ANSWERABLE)]

    choice = choose_threshold(points)

    # Sorted scores: 0.80, 0.84, 0.90. Midpoints: 0.82 and 0.87.
    assert [candidate for candidate, _ in choice.table] == pytest.approx([0.0, 0.82, 0.87])
    # 0.0 refuses nothing: 2 answerable right, 1 unanswerable wrong.
    # 0.82 refuses only the 0.80 question: all 3 right.
    # 0.87 also refuses the answerable 0.84 question: 2 right.
    assert [correct for _, correct in choice.table] == [2, 3, 2]
    assert choice.total == 3


def test_a_clean_separation_picks_the_midpoint_between_the_two_kinds() -> None:
    points = [
        (0.78, UNANSWERABLE),
        (0.80, UNANSWERABLE),  # the highest unanswerable score
        (0.84, ANSWERABLE),  # the lowest answerable score
        (0.88, ANSWERABLE),
        (0.90, ANSWERABLE),
    ]

    choice = choose_threshold(points)

    assert choice.threshold == pytest.approx(0.82)  # halfway between 0.80 and 0.84
    assert (choice.correct, choice.total) == (5, 5)
    assert choice.gap_below == pytest.approx(0.02)  # 0.82 - 0.80
    assert choice.gap_above == pytest.approx(0.02)  # 0.84 - 0.82


def test_when_the_two_kinds_overlap_the_candidate_with_the_most_correct_wins() -> None:
    points = [
        (0.70, UNANSWERABLE),
        (0.74, ANSWERABLE),
        (0.76, UNANSWERABLE),
        (0.78, UNANSWERABLE),
        (0.85, ANSWERABLE),
        (0.90, ANSWERABLE),
    ]

    choice = choose_threshold(points)

    # candidate  correct
    # 0.0        3   (the three answerable ones)
    # 0.72       4   (also refuses 0.70)
    # 0.75       3   (now refuses the answerable 0.74)
    # 0.77       4   (also refuses 0.76)
    # 0.815      5   (also refuses 0.78; only the answerable 0.74 is wrong)
    # 0.875      4   (now refuses the answerable 0.85 as well)
    assert [correct for _, correct in choice.table] == [3, 4, 3, 4, 5, 4]
    assert choice.threshold == pytest.approx(0.815)
    assert (choice.correct, choice.total) == (5, 6)
    assert choice.gap_below == pytest.approx(0.035)  # 0.815 - 0.78
    assert choice.gap_above == pytest.approx(0.035)  # 0.85 - 0.815


def test_a_tie_picks_the_lowest_candidate_so_that_search_answers_when_in_doubt() -> None:
    points = [
        (0.70, UNANSWERABLE),
        (0.75, ANSWERABLE),
        (0.80, UNANSWERABLE),
        (0.85, ANSWERABLE),
        (0.90, ANSWERABLE),
    ]

    choice = choose_threshold(points)

    # candidate  correct
    # 0.0        3
    # 0.725      4   (refuses 0.70)
    # 0.775      3   (refuses the answerable 0.75)
    # 0.825      4   (refuses 0.80 too)
    # 0.875      3
    assert [correct for _, correct in choice.table] == [3, 4, 3, 4, 3]
    assert choice.threshold == pytest.approx(0.725)  # 0.725 and 0.825 tie on 4; the lower one wins
    assert choice.correct == 4


def test_equal_scores_do_not_create_a_candidate_between_them() -> None:
    points = [(0.80, UNANSWERABLE), (0.80, ANSWERABLE), (0.90, ANSWERABLE)]

    choice = choose_threshold(points)

    # The only neighbouring pair of different scores is 0.80 and 0.90.
    assert [candidate for candidate, _ in choice.table] == pytest.approx([0.0, 0.85])
    # 0.0: 2 right (both answerable). 0.85: 2 right (0.80 unanswerable, 0.90 answerable). Tie: 0.0.
    assert choice.threshold == 0.0


def test_no_separation_at_all_picks_zero_which_never_refuses() -> None:
    # The unanswerable question scores above every answerable one, so no
    # threshold can refuse it without refusing them as well.
    points = [(0.70, ANSWERABLE), (0.72, ANSWERABLE), (0.90, UNANSWERABLE)]

    choice = choose_threshold(points)

    assert [correct for _, correct in choice.table] == [2, 1, 0]  # candidates 0.0, 0.71, 0.81
    assert choice.threshold == 0.0
    assert (choice.correct, choice.total) == (2, 3)
    assert choice.gap_below is None  # no score is below 0.0
    assert choice.gap_above == pytest.approx(0.70)  # the nearest score at or above 0.0


def test_a_score_equal_to_the_threshold_is_not_refused() -> None:
    # Search refuses only when the similarity is below the threshold, so a
    # threshold of 0.0 refuses nothing, not even a question that scored 0.0.
    choice = choose_threshold([(0.0, ANSWERABLE), (0.5, ANSWERABLE)])

    assert choice.threshold == 0.0
    assert (choice.correct, choice.total) == (2, 2)  # both answered, so both right
    assert choice.gap_above == 0.0  # "at or above": the lower score sits on the threshold


def test_only_answerable_questions_never_refuses() -> None:
    choice = choose_threshold([(0.80, ANSWERABLE), (0.90, ANSWERABLE)])

    assert choice.threshold == 0.0
    assert (choice.correct, choice.total) == (2, 2)


def test_the_gaps_are_measured_to_the_nearest_score_on_each_side() -> None:
    points = [(0.60, UNANSWERABLE), (0.70, UNANSWERABLE), (0.90, ANSWERABLE), (0.95, ANSWERABLE)]

    choice = choose_threshold(points)

    assert choice.threshold == pytest.approx(0.80)  # halfway between 0.70 and 0.90
    assert choice.gap_below == pytest.approx(0.10)  # down to 0.70, not to 0.60
    assert choice.gap_above == pytest.approx(0.10)  # up to 0.90, not to 0.95


def test_with_no_points_the_threshold_is_zero() -> None:
    choice = choose_threshold([])

    assert choice.threshold == 0.0
    assert (choice.correct, choice.total) == (0, 0)
    assert choice.table == [(0.0, 0)]
    assert choice.gap_below is None
    assert choice.gap_above is None


# --- the sign test ----------------------------------------------------------


def test_six_wins_and_no_losses_gives_p_of_one_in_thirty_two() -> None:
    assert sign_test(6, 0) == 0.03125  # 2 * (1/2)**6


def test_five_wins_and_no_losses_is_not_enough() -> None:
    assert sign_test(5, 0) == 0.0625  # 2 * (1/2)**5, above 0.05


def test_eight_wins_and_one_loss_is_below_five_percent() -> None:
    assert sign_test(8, 1) == pytest.approx(20 / 512)  # 2 * (1 + 9) / 2**9 = 0.0390625
    assert sign_test(8, 1) < 0.05


def test_seven_wins_and_one_loss_is_above_five_percent() -> None:
    assert sign_test(7, 1) == pytest.approx(18 / 256)  # 2 * (1 + 8) / 2**8 = 0.0703125
    assert sign_test(7, 1) > 0.05


def test_ten_wins_and_two_losses_is_below_five_percent_and_nine_and_two_is_not() -> None:
    assert sign_test(10, 2) == pytest.approx(158 / 4096)  # 2 * (1 + 12 + 66) / 2**12 = 0.0386
    assert sign_test(9, 2) == pytest.approx(134 / 2048)  # 2 * (1 + 11 + 55) / 2**11 = 0.0654


def test_no_wins_and_no_losses_gives_1() -> None:
    assert sign_test(0, 0) == 1.0


def test_an_even_result_gives_1_and_never_more() -> None:
    assert sign_test(3, 3) == 1.0
    assert sign_test(1, 1) == 1.0


def test_the_sign_test_is_symmetric() -> None:
    assert sign_test(8, 1) == sign_test(1, 8)
    assert sign_test(0, 6) == sign_test(6, 0)


def test_the_sign_test_refuses_negative_counts() -> None:
    with pytest.raises(ValueError):
        sign_test(-1, 3)


# --- the default mode -------------------------------------------------------

# Two answerable questions whose answer sits in the section "Steps" of
# "invoices", and one question the guides cannot answer.
DEFAULT_MODE_QUESTIONS = [
    make_question("en-001", headings={"invoices": ("Steps",)}),
    make_question("en-002", headings={"invoices": ("Steps",)}),
    make_question("na-001", expected=()),
]
ANSWERING = {"invoices": ("Steps",)}
ELSEWHERE = {"invoices": ("Troubleshooting",)}


def outcomes(en_001: dict, en_002: dict, na_001_refused: bool) -> dict[str, Outcome]:
    """Both answerable questions get the guide "invoices", with the passages given."""
    return {
        "en-001": returned("en-001", ("invoices",), en_001),
        "en-002": returned("en-002", ("invoices",), en_002),
        "na-001": refused("na-001") if na_001_refused else returned("na-001", ("invoices",), ELSEWHERE),
    }


def test_rerank_becomes_the_default_when_it_returns_more_answering_passages():
    hybrid = outcomes(ANSWERING, ELSEWHERE, na_001_refused=False)
    rerank = outcomes(ANSWERING, ANSWERING, na_001_refused=False)

    assert choose_default_mode(DEFAULT_MODE_QUESTIONS, hybrid, rerank) == "rerank"


def test_a_tie_keeps_hybrid():
    hybrid = outcomes(ANSWERING, ELSEWHERE, na_001_refused=False)
    rerank = outcomes(ELSEWHERE, ANSWERING, na_001_refused=False)

    assert choose_default_mode(DEFAULT_MODE_QUESTIONS, hybrid, rerank) == "hybrid"


def test_rerank_does_not_become_the_default_if_it_refuses_fewer_unanswerable_questions():
    hybrid = outcomes(ANSWERING, ELSEWHERE, na_001_refused=True)
    rerank = outcomes(ANSWERING, ANSWERING, na_001_refused=False)

    assert choose_default_mode(DEFAULT_MODE_QUESTIONS, hybrid, rerank) == "hybrid"


def test_the_right_guide_with_the_wrong_passage_does_not_count_for_the_default_mode():
    # Both modes find the right guide every time. Only the passages differ.
    hybrid = outcomes(ELSEWHERE, ELSEWHERE, na_001_refused=False)
    rerank = outcomes(ELSEWHERE, ANSWERING, na_001_refused=False)

    assert choose_default_mode(DEFAULT_MODE_QUESTIONS, hybrid, rerank) == "rerank"
