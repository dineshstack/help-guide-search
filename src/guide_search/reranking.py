"""Reranking: putting the candidates of a search in a better order.

Vector search compares two vectors that were made separately, one for the
question and one for each passage. The model never sees the two together. A
reranker (a "cross-encoder") does: it reads the question and one passage at
the same time and gives a score for how well that passage answers that
question. That is too slow for all 335 passages, but fast enough for the few
dozen that hybrid search has already picked out.

This file holds two things:

* `Reranker`, the small interface the rest of the code uses, so that nothing
  else needs to know which model is behind it;
* `FakeReranker`, used by the tests. It needs no download.

The real model lives in `cross_encoder.py`. This file must stay cheap to
import: it must not import torch or sentence-transformers.
"""

import re
from typing import Protocol


class Reranker(Protocol):
    """What the rest of the code needs from a reranker.

    `score` returns one number for each text, in the order the texts were
    given. A higher number means a better answer to the question. The real
    model's scores lie between 0 and 1.
    """

    model_id: str  # model name and revision

    def score(self, question: str, texts: list[str]) -> list[float]: ...


class FakeReranker:
    """A stand-in for the real model, for tests only.

    A text scores the share of the question's words that it contains, so a
    test can say in advance which passage comes first. The scores lie between
    0 and 1, like the real model's.
    """

    model_id = "fake-reranker"

    def score(self, question: str, texts: list[str]) -> list[float]:
        words = set(re.findall(r"\w+", question.lower()))
        if not words:
            return [0.0 for _ in texts]
        scores = []
        for text in texts:
            found = words & set(re.findall(r"\w+", text.lower()))
            scores.append(len(found) / len(words))
        return scores
