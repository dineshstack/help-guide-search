"""Tests for reranking.py: the fake reranker the other tests rely on.

The real reranker is tested in tests/model, because it needs the model.
"""

import pytest

from guide_search.reranking import FakeReranker


def test_a_text_scores_the_share_of_the_question_words_it_contains():
    scores = FakeReranker().score(
        "mend the torn sail",
        ["How to mend a torn sail", "Mend the frame", "Paint the hull"],
    )

    # "mend", "the", "torn", "sail": 3 of 4, 2 of 4 ("mend", "the"), 1 of 4 ("the").
    assert scores == [0.75, 0.5, 0.25]


def test_case_does_not_matter():
    assert FakeReranker().score("SAIL", ["sail"]) == [1.0]


def test_scores_come_back_in_the_order_the_texts_were_given():
    reranker = FakeReranker()
    texts = ["no match here", "sail", "a torn sail"]

    assert reranker.score("torn sail", texts) == [0.0, 0.5, 1.0]
    assert reranker.score("torn sail", list(reversed(texts))) == [1.0, 0.5, 0.0]


def test_a_question_without_words_scores_every_text_zero():
    assert FakeReranker().score("?!", ["sail", "spar"]) == [0.0, 0.0]


def test_no_texts_give_no_scores():
    assert FakeReranker().score("sail", []) == []


@pytest.mark.parametrize("text", ["", "sail", "mend the torn sail twice"])
def test_every_score_lies_between_0_and_1_like_the_real_models(text: str):
    [score] = FakeReranker().score("mend the torn sail", [text])

    assert 0.0 <= score <= 1.0
