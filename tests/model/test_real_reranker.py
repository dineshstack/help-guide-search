"""Tests that use the real reranker.

Run them with:  uv run pytest -m model -q

The first run downloads the model (about 2.3 GB) into the Hugging Face cache.
Later runs load it from there. If the model cannot be loaded these tests fail;
they are never skipped, so a green run always means the model really ran.
"""

from pathlib import Path

import pytest

from guide_search.chunking import chunk_guide
from guide_search.snapshot import load_snapshot

pytestmark = pytest.mark.model

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "guides.json"

# Invented for this file, written the way the reranker is given a passage:
# guide title, heading, then the text.
PASSWORD_PASSAGE = (
    "Sign in\nReset your password\n"
    "To reset your password, open the sign-in page and choose Forgot password. "
    "A reset link is sent to your email address."
)
INVOICE_PASSAGE = (
    "Invoices\nPrint an invoice\n"
    "To print an invoice, open it from the Invoices list and choose Print. "
    "You can also save it as a PDF file."
)


@pytest.fixture(scope="module")
def reranker():
    """Load the real model once for the whole file, because loading is slow."""
    # Imported here: importing cross_encoder loads torch.
    from guide_search.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


def test_the_pinned_model_loads_on_the_cpu(reranker):
    from guide_search.cross_encoder import MODEL_NAME, REVISION

    assert reranker.model_id == f"{MODEL_NAME}@{REVISION}"
    assert str(reranker.model.device) == "cpu"


@pytest.mark.parametrize(
    "question, matching, unrelated",
    [
        ("I forgot my password, how can I sign in again?", PASSWORD_PASSAGE, INVOICE_PASSAGE),
        ("How do I get a paper copy of an invoice?", INVOICE_PASSAGE, PASSWORD_PASSAGE),
        # "I forgot the password, how do I reset it?"
        ("نسيت كلمة المرور، كيف أعيد تعيينها؟", PASSWORD_PASSAGE, INVOICE_PASSAGE),
        # "How do I print an invoice?"
        ("كيف أطبع فاتورة؟", INVOICE_PASSAGE, PASSWORD_PASSAGE),
    ],
    ids=["english-password", "english-invoice", "arabic-password", "arabic-invoice"],
)
def test_a_question_scores_its_matching_passage_above_an_unrelated_one(reranker, question, matching, unrelated):
    matching_score, unrelated_score = reranker.score(question, [matching, unrelated])

    assert matching_score > unrelated_score


def test_scores_are_plain_floats_between_0_and_1_in_the_order_given(reranker):
    question = "How do I print an invoice?"

    scores = reranker.score(question, [PASSWORD_PASSAGE, INVOICE_PASSAGE])
    reversed_scores = reranker.score(question, [INVOICE_PASSAGE, PASSWORD_PASSAGE])

    assert all(type(score) is float and 0.0 <= score <= 1.0 for score in scores)
    assert scores == pytest.approx(list(reversed(reversed_scores)), abs=1e-4)


def test_no_texts_give_no_scores(reranker):
    assert reranker.score("anything", []) == []


def test_no_question_and_passage_in_the_snapshot_is_longer_than_the_input_limit(reranker):
    # A longest likely question (500 characters, the limit) is put with every
    # passage, so that no passage's end would be cut off.
    from guide_search.cross_encoder import MAX_LENGTH

    question = "word " * 100
    tokenizer = reranker.model.tokenizer
    too_long = []
    for guide in load_snapshot(SNAPSHOT_PATH):
        for passage in chunk_guide(guide):
            tokens = len(tokenizer(question.strip(), passage.search_text, truncation=False)["input_ids"])
            if tokens > MAX_LENGTH:
                too_long.append((passage.slug, passage.position, tokens))

    assert too_long == []
