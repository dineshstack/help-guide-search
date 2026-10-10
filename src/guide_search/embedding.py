"""Turning text into vectors: the interface, the two prefixes, and a fake.

An embedding turns a piece of text into a list of numbers (a vector), so that
texts with similar meaning get vectors that point in similar directions.

This file holds three things:

* the two prefix functions. The real model was trained with "passage: " in
  front of every passage and "query: " in front of every question, and gives
  worse results without them;
* `Embedder`, the small interface the rest of the code uses, so that nothing
  else needs to know which model is behind it;
* `FakeEmbedder`, used by the tests. It needs no download. It only counts
  words, so texts that share words get similar vectors.

The real model lives in `e5.py`. This file must stay cheap to import: it must
not import torch or sentence-transformers.
"""

import hashlib
import math
import re
from typing import Protocol

DIMENSIONS = 384
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def passage_input(text: str) -> str:
    """Return the text exactly as the real model must be given a passage."""
    return PASSAGE_PREFIX + text


def query_input(text: str) -> str:
    """Return the text exactly as the real model must be given a question."""
    return QUERY_PREFIX + text


class Embedder(Protocol):
    """What the rest of the code needs from an embedding model.

    A Protocol is a description, not a parent class. Any object with these
    names and these signatures can be used where an Embedder is expected.
    """

    model_id: str  # model name and revision; stored with every vector

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FakeEmbedder:
    """A stand-in for the real model, for tests only.

    It never adds a prefix: a prefix would only add the same word to every
    text. Passages and questions are treated in exactly the same way.
    """

    model_id = "fake"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [_word_count_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _word_count_vector(text)


def _word_position(word: str) -> int:
    """Return the position in the vector that this word always lands on."""
    # sha256 gives the same answer on every machine and every run. Python's
    # own hash() does not: for text it changes each time Python starts.
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % DIMENSIONS


def _word_count_vector(text: str) -> list[float]:
    """Count the words of a text into a vector, then scale it to length 1."""
    words = re.findall(r"\w+", text.lower())
    vector = [0.0] * DIMENSIONS

    if not words:
        # A vector of only zeros has no direction, so it cannot be scaled to
        # length 1 or compared. Text without words gets this fixed vector.
        vector[0] = 1.0
        return vector

    for word in words:
        vector[_word_position(word)] += 1.0

    length = math.sqrt(sum(number * number for number in vector))
    return [number / length for number in vector]
