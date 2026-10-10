"""Tests for the prefix functions and the fake embedder.

Nothing here needs the real model, a database or the network. Two tests start
a second Python process, because they check things that cannot be seen from
inside the process that is already running the tests.
"""

import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import get_protocol_members

import pytest

from guide_search import embedding
from guide_search.embedding import Embedder, FakeEmbedder, passage_input, query_input

# The folder that holds the guide_search package, so a second Python process
# can import it too.
SRC_DIR = str(Path(embedding.__file__).resolve().parents[1])

# What the fake embedder must return for text that has no words in it.
NO_WORDS_VECTOR = [1.0] + [0.0] * 383


def similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors that already have length 1."""
    return sum(x * y for x, y in zip(a, b))


def run_python(code: str, hash_seed: str) -> str:
    """Run a short script in a new Python process and return what it printed."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = SRC_DIR
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


# ---------------------------------------------------------------- prefixes


def test_passage_input_puts_passage_colon_space_in_front():
    assert passage_input("Reset your password") == "passage: Reset your password"


def test_query_input_puts_query_colon_space_in_front():
    assert query_input("how do I print an invoice") == "query: how do I print an invoice"


def test_prefix_of_an_empty_text_is_the_prefix_alone_with_its_trailing_space():
    assert passage_input("") == "passage: "
    assert query_input("") == "query: "


def test_prefix_functions_do_not_trim_or_change_the_text():
    text = "  Two  spaces, CAPITALS,\na second line and Arabic: كلمة المرور  "

    assert passage_input(text) == "passage: " + text
    assert query_input(text) == "query: " + text


# ----------------------------------------------------- shape of the vectors


@pytest.mark.parametrize(
    "text",
    ["reset the staff password", "tyre tyre tyre brake", "one", "", "كلمة المرور"],
)
def test_fake_vector_has_384_numbers_and_length_one(text):
    vector = FakeEmbedder().embed_query(text)

    assert len(vector) == 384
    assert all(isinstance(number, float) for number in vector)
    assert math.sqrt(sum(number * number for number in vector)) == pytest.approx(1.0, abs=1e-9)


def test_one_word_puts_all_the_weight_on_that_words_fixed_position():
    # 280 was worked out by hand, outside this project's code: the first four
    # bytes of sha256("password") are 5e884898, and 0x5e884898 % 384 == 280.
    expected = [0.0] * 384
    expected[280] = 1.0

    assert FakeEmbedder().embed_query("password") == expected


def test_a_repeated_word_counts_once_for_every_time_it_appears():
    vector = FakeEmbedder().embed_query("tyre tyre brake")

    # Counts of 2 and 1, scaled to length 1: 2/sqrt(5) and 1/sqrt(5).
    weights = sorted(number for number in vector if number != 0.0)
    assert weights == pytest.approx([1 / math.sqrt(5), 2 / math.sqrt(5)])


# ------------------------------------------------------------ repeatability


def test_same_text_gives_the_same_vector_on_every_call():
    fake = FakeEmbedder()

    assert fake.embed_query("reset the staff password") == fake.embed_query("reset the staff password")


def test_same_text_gives_the_same_vector_in_a_fresh_object():
    first = FakeEmbedder().embed_query("reset the staff password")
    second = FakeEmbedder().embed_query("reset the staff password")

    assert first == second


def test_vector_does_not_depend_on_pythons_hash_randomisation():
    # Python's built-in hash() of a string changes with PYTHONHASHSEED. Two
    # processes with different seeds must still agree with each other and
    # with this process.
    code = (
        "import json\n"
        "from guide_search.embedding import FakeEmbedder\n"
        "print(json.dumps(FakeEmbedder().embed_query('reset the staff password')))\n"
    )

    with_seed_1 = json.loads(run_python(code, hash_seed="1"))
    with_seed_2 = json.loads(run_python(code, hash_seed="2"))

    assert with_seed_1 == with_seed_2
    assert with_seed_1 == FakeEmbedder().embed_query("reset the staff password")


# ------------------------------------------------------- what counts as a word


def test_case_does_not_matter():
    fake = FakeEmbedder()

    assert fake.embed_query("Reset The STAFF Password") == fake.embed_query("reset the staff password")


def test_punctuation_is_not_part_of_a_word():
    fake = FakeEmbedder()

    assert fake.embed_query("password?") == fake.embed_query("password")
    assert fake.embed_query("reset, the (staff) password!") == fake.embed_query("reset the staff password")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "?!", "... --- ..."])
def test_text_with_no_words_gives_the_fixed_vector(text):
    assert FakeEmbedder().embed_query(text) == NO_WORDS_VECTOR


def test_arabic_words_are_counted_like_any_other_word():
    vector = FakeEmbedder().embed_query("كلمة المرور")

    # Two different words, one count each, scaled to length 1.
    weights = [number for number in vector if number != 0.0]
    assert vector != NO_WORDS_VECTOR
    assert weights == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])


# ------------------------------------------------------------- similarity


def test_text_that_shares_words_scores_higher_than_text_that_shares_none():
    fake = FakeEmbedder()
    question = fake.embed_query("reset staff password")
    shares_words = fake.embed_query("how to reset a staff password")
    shares_none = fake.embed_query("print the invoice")

    assert similarity(question, shares_words) > similarity(question, shares_none)
    # Three shared words, between a text of three words and a text of six.
    assert similarity(question, shares_words) == pytest.approx(3 / math.sqrt(3 * 6))


def test_sharing_more_words_scores_higher_than_sharing_fewer():
    fake = FakeEmbedder()
    question = fake.embed_query("reset staff password")
    shares_three = fake.embed_query("reset staff password today")
    shares_one = fake.embed_query("print staff invoice today")

    assert similarity(question, shares_three) > similarity(question, shares_one) > 0.0


def test_identical_text_has_similarity_one():
    fake = FakeEmbedder()
    vector = fake.embed_query("reset the staff password")

    assert similarity(vector, vector) == pytest.approx(1.0)


# ------------------------------------------------------------ embed_passages


def test_embed_passages_returns_one_vector_per_text_in_the_same_order():
    fake = FakeEmbedder()
    texts = ["reset the staff password", "print the invoice", "", "book a tyre fitting"]

    vectors = fake.embed_passages(texts)

    assert len(vectors) == 4
    assert vectors == [fake.embed_query(text) for text in texts]


def test_embed_passages_of_no_texts_is_an_empty_list():
    assert FakeEmbedder().embed_passages([]) == []


def test_fake_embedder_adds_no_prefix_so_a_passage_and_a_question_with_the_same_text_match():
    fake = FakeEmbedder()

    # If a prefix were added, the words "passage" and "query" would be counted
    # and these two vectors would differ.
    assert fake.embed_passages(["print the invoice"])[0] == fake.embed_query("print the invoice")


# ------------------------------------------------- the interface and imports


def test_fake_embedder_has_everything_the_embedder_interface_asks_for():
    fake = FakeEmbedder()

    assert get_protocol_members(Embedder) == {"model_id", "embed_passages", "embed_query"}
    for name in get_protocol_members(Embedder):
        assert hasattr(fake, name), name
    assert fake.model_id == "fake"
    # Same parameter names, same types, same return type.
    assert inspect.signature(FakeEmbedder.embed_passages) == inspect.signature(Embedder.embed_passages)
    assert inspect.signature(FakeEmbedder.embed_query) == inspect.signature(Embedder.embed_query)


def test_importing_embedding_does_not_load_torch_or_sentence_transformers():
    # Checked in a new process: this one may already have loaded them for
    # another test.
    code = (
        "import sys\n"
        "import guide_search.embedding\n"
        "heavy = [name for name in ('torch', 'sentence_transformers') if name in sys.modules]\n"
        "print(heavy)\n"
    )

    assert run_python(code, hash_seed="0").strip() == "[]"
