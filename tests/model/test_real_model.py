"""Tests that use the real embedding model.

Run them with:  uv run pytest -m model -q

The first run downloads the model (about 0.5 GB) into the Hugging Face cache.
Later runs load it from there. If the model cannot be loaded these tests fail;
they are never skipped, so a green run always means the model really ran.

The two passages and the questions below are invented for this file.
"""

import math
from pathlib import Path

import pytest

from guide_search.chunking import chunk_guide
from guide_search.snapshot import load_snapshot

pytestmark = pytest.mark.model

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "guides.json"

PASSWORD_PASSAGE = (
    "To reset your password, open the sign-in page and choose Forgot password. "
    "A reset link is sent to your email address."
)
INVOICE_PASSAGE = (
    "To print an invoice, open it from the Invoices list and choose Print. "
    "You can also save it as a PDF file."
)

# The same text encoded twice can differ in its last decimal places, for
# example when it is encoded alone and then together with other texts.
CLOSE = 1e-4


@pytest.fixture(scope="module")
def embedder():
    """Load the real model once for the whole file, because loading is slow."""
    # Imported here and not at the top of the file: importing e5 loads torch,
    # which takes seconds, and pytest imports this file on every run, even on
    # runs that leave these tests out.
    from guide_search.e5 import E5Embedder

    return E5Embedder()


def similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors that already have length 1."""
    return sum(x * y for x, y in zip(a, b))


def test_pinned_model_loads_on_the_cpu_and_reads_at_most_512_tokens(embedder):
    assert embedder.model_id == "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert embedder.model.max_seq_length == 512
    assert embedder.model.device.type == "cpu"


def test_library_adds_no_prefix_of_its_own(embedder):
    # The prefixes come from this project's two functions. If the library also
    # added one, every text would carry two, and the "equals" tests below
    # could not tell.
    assert embedder.model.default_prompt_name is None


def test_vectors_are_plain_lists_of_384_floats_with_length_one(embedder):
    vectors = [embedder.embed_query("how do I reset my password")]
    vectors += embedder.embed_passages([PASSWORD_PASSAGE, INVOICE_PASSAGE])

    assert len(vectors) == 3
    for vector in vectors:
        assert type(vector) is list
        assert len(vector) == 384
        assert all(type(number) is float for number in vector)
        assert math.sqrt(sum(number * number for number in vector)) == pytest.approx(1.0, abs=1e-4)


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
def test_question_ranks_its_matching_passage_above_an_unrelated_one(embedder, question, matching, unrelated):
    question_vector = embedder.embed_query(question)
    matching_vector, unrelated_vector = embedder.embed_passages([matching, unrelated])

    assert similarity(question_vector, matching_vector) > similarity(question_vector, unrelated_vector)


def test_embed_query_is_the_models_encoding_of_query_prefix_plus_question(embedder):
    question = "how do I reset my password"

    vector = embedder.embed_query(question)
    with_prefix = embedder.model.encode("query: " + question, normalize_embeddings=True).tolist()
    without_prefix = embedder.model.encode(question, normalize_embeddings=True).tolist()

    assert vector == pytest.approx(with_prefix, abs=CLOSE)
    assert vector != pytest.approx(without_prefix, abs=1e-3)


def test_embed_passages_is_the_models_encoding_of_passage_prefix_plus_text_in_order(embedder):
    texts = [PASSWORD_PASSAGE, INVOICE_PASSAGE]

    vectors = embedder.embed_passages(texts)

    assert len(vectors) == 2
    for vector, text in zip(vectors, texts):
        with_prefix = embedder.model.encode("passage: " + text, normalize_embeddings=True).tolist()
        without_prefix = embedder.model.encode(text, normalize_embeddings=True).tolist()
        assert vector == pytest.approx(with_prefix, abs=CLOSE)
        assert vector != pytest.approx(without_prefix, abs=1e-3)


def test_token_count_includes_the_special_tokens_and_is_not_cut_off_at_the_limit(embedder):
    # The tokenizer wraps every text in a start and an end token, so even an
    # empty text counts 2.
    assert embedder.token_count("") == 2
    assert embedder.token_count("passage: invoice") > embedder.token_count("invoice")
    # 600 words are far more than 512 tokens. The count must say so, or the
    # next test could never fail.
    assert embedder.token_count("invoice " * 600) > 512


def test_no_passage_in_the_snapshot_is_longer_than_512_tokens(embedder):
    too_long = []
    checked = 0
    for guide in load_snapshot(SNAPSHOT_PATH):
        for passage in chunk_guide(guide):
            checked += 1
            tokens = embedder.token_count("passage: " + passage.search_text)
            if tokens > 512:
                too_long.append((passage.slug, passage.position, tokens))

    assert checked > 0
    assert too_long == []
