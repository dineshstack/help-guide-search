"""Tests for the ranking rules: pulling the words out of a question, rank
fusion, ordering, and grouping passages under their guides.

Everything tested here is plain Python. No database and no model are needed.
All guides and passages are invented.
"""

import pytest

from guide_search.ranking import (
    MAX_QUESTION_CHARS,
    Candidate,
    GuideHit,
    PassageHit,
    clean_question,
    group_by_guide,
    order_candidates,
    question_words,
    rrf_scores,
)


def candidate(
    guide_id: int,
    position: int,
    score: float,
    similarity: float | None = None,
) -> Candidate:
    """Build an invented candidate. The other fields follow from the two ids."""
    return Candidate(
        guide_id=guide_id,
        slug=f"guide-{guide_id}",
        title=f"Guide {guide_id}",
        url=f"https://example.test/help/guide-{guide_id}",
        heading=f"Section {position}",
        position=position,
        content=f"Text of guide {guide_id}, passage {position}.",
        score=score,
        similarity=similarity,
    )


def keys_of(candidates: list[Candidate]) -> list[tuple[int, int]]:
    """The (guide_id, position) of each candidate, to keep assertions short."""
    return [(item.guide_id, item.position) for item in candidates]


# ---------------------------------------------------------------------------
# question_words
# ---------------------------------------------------------------------------


def test_question_words_keeps_letters_and_digits():
    assert question_words("invoice 48 hours") == ["invoice", "48", "hours"]


def test_question_words_drops_punctuation():
    assert question_words("Can I export, then print? Yes!") == [
        "Can",
        "I",
        "export",
        "then",
        "print",
        "Yes",
    ]


def test_question_words_treats_quotes_as_separators():
    assert question_words('the "paid" stamp') == ["the", "paid", "stamp"]
    assert question_words("the “paid” stamp") == ["the", "paid", "stamp"]


def test_question_words_splits_at_an_apostrophe():
    question = "why can't I sign in"
    assert question_words(question) == ["why", "can", "t", "I", "sign", "in"]


def test_question_words_splits_at_a_hyphen():
    assert question_words("re-open a job card") == ["re", "open", "a", "job", "card"]


def test_question_words_splits_at_an_underscore():
    # The design says letters and digits only, and an underscore is neither.
    assert question_words("job_card status") == ["job", "card", "status"]


def test_question_words_leaves_no_keyword_query_syntax_behind():
    # & | ! ( ) : * < > all mean something in a Postgres keyword query.
    # None of them may survive, so a question can never be read as syntax.
    question = "brake & (pads | discs):* !oil <-> filter"
    assert question_words(question) == ["brake", "pads", "discs", "oil", "filter"]


# In the Arabic tests each word sits on a line of its own. An editor draws
# Arabic from right to left, and a line that mixes several Arabic words with
# brackets and commas is shown in an order that is hard to follow.


def test_question_words_keeps_arabic_words():
    # "How do I change the password?" The Arabic question mark is dropped.
    question = "كيف أغير كلمة المرور؟"
    assert question_words(question) == [
        "كيف",
        "أغير",
        "كلمة",
        "المرور",
    ]


def test_question_words_drops_the_arabic_comma():
    # "Yes, thank you."
    question = "نعم، شكرا"
    assert question_words(question) == [
        "نعم",
        "شكرا",
    ]


def test_question_words_keeps_a_latin_word_and_a_number_inside_an_arabic_question():
    # "Does it support WhatsApp within 48 hours?"
    question = "هل يدعم WhatsApp خلال 48 ساعة؟"
    assert question_words(question) == [
        "هل",
        "يدعم",
        "WhatsApp",
        "خلال",
        "48",
        "ساعة",
    ]


def test_question_words_removes_repeats_and_keeps_the_order_of_first_use():
    question = "tyre size and tyre pressure and size"
    assert question_words(question) == ["tyre", "size", "and", "pressure"]


def test_question_words_returns_an_empty_list_when_there_are_no_words():
    assert question_words("") == []
    assert question_words("   ") == []
    assert question_words("?!... -- ' \" _") == []


# ---------------------------------------------------------------------------
# rrf_scores
# ---------------------------------------------------------------------------


def test_rrf_rank_one_in_both_lists_scores_two_over_sixty_one():
    scores = rrf_scores([(1, 0)], [(1, 0)])
    assert scores == {(1, 0): pytest.approx(2 / 61)}


def test_rrf_rank_one_in_one_list_scores_one_over_sixty_one():
    assert rrf_scores([(1, 0)], []) == {(1, 0): pytest.approx(1 / 61)}
    assert rrf_scores([], [(1, 0)]) == {(1, 0): pytest.approx(1 / 61)}


def test_rrf_rank_twenty_in_both_lists_scores_two_over_eighty():
    found_by_both = (9, 0)
    vector_keys = [(1, position) for position in range(19)] + [found_by_both]
    keyword_keys = [(2, position) for position in range(19)] + [found_by_both]

    scores = rrf_scores(vector_keys, keyword_keys)

    assert scores[found_by_both] == pytest.approx(2 / 80)


def test_rrf_rank_twenty_in_both_lists_beats_rank_one_in_one_list():
    found_by_both = (9, 0)
    top_of_vector_only = (1, 0)
    top_of_keyword_only = (2, 0)
    vector_keys = [(1, position) for position in range(19)] + [found_by_both]
    keyword_keys = [(2, position) for position in range(19)] + [found_by_both]

    scores = rrf_scores(vector_keys, keyword_keys)

    assert scores[top_of_vector_only] == pytest.approx(1 / 61)
    assert scores[top_of_keyword_only] == pytest.approx(1 / 61)
    assert scores[found_by_both] > scores[top_of_vector_only]
    assert scores[found_by_both] > scores[top_of_keyword_only]


def test_rrf_key_in_one_list_only_gets_that_lists_term_alone():
    vector_keys = [(1, 0), (1, 1), (1, 2)]
    keyword_keys = [(2, 0), (1, 2)]

    scores = rrf_scores(vector_keys, keyword_keys)

    assert scores[(1, 1)] == pytest.approx(1 / 62)  # vector rank 2 only
    assert scores[(2, 0)] == pytest.approx(1 / 61)  # keyword rank 1 only
    assert scores[(1, 2)] == pytest.approx(1 / 63 + 1 / 62)  # vector 3, keyword 2


def test_rrf_scores_every_key_that_is_in_either_list_exactly_once():
    scores = rrf_scores([(1, 0), (1, 1)], [(1, 1), (2, 0)])
    assert set(scores) == {(1, 0), (1, 1), (2, 0)}


def test_rrf_with_an_empty_keyword_list_keeps_the_vector_order():
    # The keys are deliberately not in (guide_id, position) order, so only the
    # scores can reproduce the order of the vector list.
    vector_keys = [(5, 2), (1, 0), (3, 7), (1, 1), (4, 4)]

    scores = rrf_scores(vector_keys, [])

    by_score = sorted(scores, key=scores.get, reverse=True)
    assert by_score == vector_keys


def test_rrf_with_two_empty_lists_scores_nothing():
    assert rrf_scores([], []) == {}


def test_rrf_uses_the_k_it_is_given():
    # With k = 1, rank 1 in both lists is 1 / 2 + 1 / 2.
    assert rrf_scores([(1, 0)], [(1, 0)], k=1) == {(1, 0): pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# order_candidates
# ---------------------------------------------------------------------------


def test_order_candidates_sorts_by_score_high_to_low():
    candidates = [candidate(1, 0, 0.2), candidate(2, 0, 0.9), candidate(3, 0, 0.5)]
    assert keys_of(order_candidates(candidates)) == [(2, 0), (3, 0), (1, 0)]


def test_order_candidates_breaks_a_score_tie_by_guide_id():
    candidates = [candidate(8, 0, 0.5), candidate(3, 5, 0.5), candidate(5, 1, 0.5)]
    assert keys_of(order_candidates(candidates)) == [(3, 5), (5, 1), (8, 0)]


def test_order_candidates_breaks_a_tie_inside_one_guide_by_position():
    candidates = [candidate(4, 6, 0.5), candidate(4, 0, 0.5), candidate(4, 2, 0.5)]
    assert keys_of(order_candidates(candidates)) == [(4, 0), (4, 2), (4, 6)]


def test_order_candidates_puts_score_before_guide_id_and_position():
    candidates = [candidate(1, 0, 0.1), candidate(9, 9, 0.8)]
    assert keys_of(order_candidates(candidates)) == [(9, 9), (1, 0)]


def test_order_candidates_tie_between_a_vector_only_and_a_keyword_only_candidate():
    # Top of the vector list only, and top of the keyword list only: both score
    # 1 / 61. The lower guide id goes first, whatever the similarities are.
    scores = rrf_scores([(7, 0)], [(2, 3)])
    vector_only = candidate(7, 0, scores[(7, 0)], similarity=0.95)
    keyword_only = candidate(2, 3, scores[(2, 3)], similarity=0.10)

    ordered = order_candidates([vector_only, keyword_only])

    assert vector_only.score == keyword_only.score
    assert keys_of(ordered) == [(2, 3), (7, 0)]


def test_order_candidates_gives_the_same_order_whatever_order_it_is_given():
    candidates = [
        candidate(2, 1, 0.5),
        candidate(1, 4, 0.5),
        candidate(1, 0, 0.5),
        candidate(3, 0, 0.9),
    ]
    expected = [(3, 0), (1, 0), (1, 4), (2, 1)]

    assert keys_of(order_candidates(candidates)) == expected
    assert keys_of(order_candidates(list(reversed(candidates)))) == expected


def test_order_candidates_does_not_change_the_list_it_was_given():
    candidates = [candidate(1, 0, 0.2), candidate(2, 0, 0.9)]
    order_candidates(candidates)
    assert keys_of(candidates) == [(1, 0), (2, 0)]


# ---------------------------------------------------------------------------
# group_by_guide
# ---------------------------------------------------------------------------


def test_group_returns_at_most_three_guides():
    candidates = [
        candidate(1, 0, 0.5),
        candidate(2, 0, 0.9),
        candidate(3, 0, 0.7),
        candidate(4, 0, 0.8),
        candidate(5, 0, 0.6),
    ]
    guides = group_by_guide(candidates)
    assert [guide.guide_id for guide in guides] == [2, 4, 3]


def test_group_returns_at_most_two_passages_per_guide():
    candidates = [
        candidate(1, 0, 0.6),
        candidate(1, 1, 0.9),
        candidate(1, 2, 0.7),
        candidate(1, 3, 0.8),
    ]
    guides = group_by_guide(candidates)
    assert [passage.position for passage in guides[0].passages] == [1, 3]


def test_group_guide_score_is_the_score_of_its_best_candidate():
    candidates = [candidate(1, 0, 0.3), candidate(1, 1, 0.8), candidate(1, 2, 0.5)]
    guides = group_by_guide(candidates)
    assert guides[0].score == 0.8


def test_group_ranks_a_guide_by_its_best_passage_not_by_how_many_it_has():
    candidates = [
        candidate(1, 0, 0.50),
        candidate(1, 1, 0.49),
        candidate(1, 2, 0.48),
        candidate(2, 0, 0.60),
    ]
    guides = group_by_guide(candidates)
    assert [guide.guide_id for guide in guides] == [2, 1]


def test_group_breaks_a_guide_tie_by_guide_id():
    candidates = [candidate(9, 0, 0.5), candidate(4, 3, 0.5), candidate(6, 1, 0.5)]
    guides = group_by_guide(candidates)
    assert [guide.guide_id for guide in guides] == [4, 6, 9]


def test_group_orders_passages_best_first_then_by_position():
    # Positions 4 and 2 tie on score, so position 2 is kept and 4 is the one
    # that falls outside the limit of two.
    candidates = [candidate(1, 4, 0.5), candidate(1, 2, 0.5), candidate(1, 9, 0.8)]
    guides = group_by_guide(candidates)
    assert [passage.position for passage in guides[0].passages] == [9, 2]


def test_group_guide_with_one_candidate_returns_one_passage():
    guides = group_by_guide([candidate(1, 0, 0.9), candidate(2, 5, 0.4)])
    assert [len(guide.passages) for guide in guides] == [1, 1]


def test_group_fewer_than_three_guides_is_a_normal_result():
    candidates = [candidate(1, 0, 0.9), candidate(1, 1, 0.8), candidate(2, 0, 0.7)]
    guides = group_by_guide(candidates)
    assert [guide.guide_id for guide in guides] == [1, 2]


def test_group_empty_candidate_list_gives_an_empty_list():
    assert group_by_guide([]) == []


def test_group_passage_carries_heading_position_text_and_similarity():
    one = Candidate(
        guide_id=12,
        slug="print-a-job-card",
        title="Print a job card",
        url="https://example.test/help/print-a-job-card",
        heading="Choose a printer",
        position=2,
        content="Open the job card and press Print.",
        score=0.031,
        similarity=0.87,
    )
    guides = group_by_guide([one])
    assert guides[0].passages == [
        PassageHit(
            heading="Choose a printer",
            position=2,
            text="Open the job card and press Print.",
            similarity=0.87,
        )
    ]


def test_group_guide_carries_its_id_slug_title_url_and_score():
    one = Candidate(
        guide_id=12,
        slug="print-a-job-card",
        title="Print a job card",
        url="https://example.test/help/print-a-job-card",
        heading="",
        position=0,
        content="A job card lists the work agreed for one vehicle.",
        score=0.031,
        similarity=0.87,
    )
    assert group_by_guide([one]) == [
        GuideHit(
            guide_id=12,
            slug="print-a-job-card",
            title="Print a job card",
            url="https://example.test/help/print-a-job-card",
            score=0.031,
            passages=[
                PassageHit(
                    heading="",
                    position=0,
                    text="A job card lists the work agreed for one vehicle.",
                    similarity=0.87,
                )
            ],
        )
    ]


def test_group_keeps_a_missing_similarity_as_none():
    # Keyword mode never embeds the question, so there is no similarity.
    guides = group_by_guide([candidate(1, 0, 0.4, similarity=None)])
    assert guides[0].passages[0].similarity is None


def test_group_takes_candidates_in_any_order():
    candidates = [
        candidate(3, 0, 0.5),
        candidate(1, 2, 0.9),
        candidate(2, 0, 0.5),
        candidate(1, 0, 0.7),
        candidate(1, 1, 0.7),
        candidate(4, 0, 0.1),
    ]
    forwards = group_by_guide(candidates)
    backwards = group_by_guide(list(reversed(candidates)))

    assert forwards == backwards
    assert [guide.guide_id for guide in forwards] == [1, 2, 3]
    assert [passage.position for passage in forwards[0].passages] == [2, 0]


def test_group_uses_the_limits_it_is_given():
    candidates = [candidate(1, 0, 0.9), candidate(1, 1, 0.8), candidate(2, 0, 0.7)]
    guides = group_by_guide(candidates, max_guides=1, max_passages=1)
    assert [guide.guide_id for guide in guides] == [1]
    assert [passage.position for passage in guides[0].passages] == [0]


# ---------------------------------------------------------------------------
# clean_question: the one rule for which questions are accepted
# ---------------------------------------------------------------------------


def test_clean_question_removes_the_spaces_around_a_question():
    assert clean_question("  how do I mend a sail \n") == "how do I mend a sail"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_clean_question_refuses_a_question_with_nothing_in_it(question: str):
    with pytest.raises(ValueError):
        clean_question(question)


def test_clean_question_counts_the_length_after_trimming():
    assert clean_question(" " + "x" * MAX_QUESTION_CHARS + " ") == "x" * MAX_QUESTION_CHARS

    with pytest.raises(ValueError):
        clean_question("x" * (MAX_QUESTION_CHARS + 1))


def test_clean_question_counts_characters_not_bytes():
    arabic = "س" * MAX_QUESTION_CHARS

    assert clean_question(arabic) == arabic
