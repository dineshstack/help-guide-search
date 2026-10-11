"""The ranking rules of search, as plain Python with no database.

This file answers four small questions:

- which words of a question are handed to keyword search (question_words);
- how the vector list and the keyword list are merged into one score
  (rrf_scores, "reciprocal rank fusion");
- in what order candidate passages are put (order_candidates);
- how candidate passages become the final answer of up to three guides with
  up to two passages each (group_by_guide).

It also defines the shapes that search hands back, so that the search code, the
HTTP routes, the MCP tools and the evaluation all use the same ones, and the
one rule for which questions are accepted (clean_question).

Nothing here talks to Postgres or to the embedding model, so every rule can be
tested with a few invented values.
"""

import re
from dataclasses import dataclass

# The constant in the fusion formula. It is the value from the paper that
# introduced the method and is not tuned here.
RRF_K = 60

# How many passages vector search and keyword search each contribute.
CANDIDATES = 20

# The size of a result: at most this many guides, each with at most this many
# passages.
MAX_GUIDES = 3
MAX_PASSAGES = 2

# The longest question that is accepted, counted after trimming.
MAX_QUESTION_CHARS = 500


@dataclass(frozen=True)
class Candidate:
    """One passage that a search found, before grouping by guide."""

    guide_id: int
    slug: str
    title: str
    url: str
    heading: str
    position: int
    content: str
    score: float  # the ordering value in the mode used
    similarity: float | None  # cosine similarity to the question; None in keyword mode


@dataclass(frozen=True)
class PassageHit:
    """One passage as it is shown in a result."""

    heading: str
    position: int
    text: str
    similarity: float | None


@dataclass(frozen=True)
class GuideHit:
    """One guide in a result, with its best passages."""

    guide_id: int
    slug: str
    title: str
    url: str
    score: float
    passages: list[PassageHit]


@dataclass(frozen=True)
class SearchResult:
    """The whole answer to one search."""

    query: str
    mode: str
    reason: str | None  # None, or "no_match"
    best_similarity: float | None
    guides: list[GuideHit]


@dataclass(frozen=True)
class GuideDetail:
    """One whole guide, as GET /guides/{slug} returns it."""

    guide_id: int
    slug: str
    title: str
    url: str
    passages: list[PassageHit]  # in position order, similarity None


@dataclass(frozen=True)
class GuideSummary:
    """One guide in the list of all guides: enough to name it and to link to it."""

    guide_id: int
    slug: str
    title: str
    url: str


@dataclass(frozen=True)
class GuideList:
    """Every guide in the index, in guide id order."""

    guides: list[GuideSummary]


def clean_question(question: str) -> str:
    """Return the question without the spaces around it, or raise ValueError.

    The HTTP route and the MCP tool both call this, so the two can never
    disagree about which questions are accepted.
    """
    trimmed = question.strip()
    if trimmed == "" or len(trimmed) > MAX_QUESTION_CHARS:
        raise ValueError(f"the question must be 1 to {MAX_QUESTION_CHARS} characters")
    return trimmed


def question_words(question: str) -> list[str]:
    """Return the words of a question: letters and digits only, no repeats.

    Keyword search gets these words one at a time. Because nothing but letters
    and digits survives, no part of a question can be read as query syntax.
    """
    # "\w" matches letters and digits in any script, so Arabic words are kept.
    # It also matches the underscore, which is neither, so underscores are
    # turned into spaces first.
    found = re.findall(r"\w+", question.replace("_", " "))

    words: list[str] = []
    for word in found:
        if word not in words:
            words.append(word)
    return words


def rrf_scores(
    vector_keys: list[tuple[int, int]],
    keyword_keys: list[tuple[int, int]],
    k: int = RRF_K,
) -> dict[tuple[int, int], float]:
    """Merge two ranked lists into one score for each passage.

    A key is (guide_id, position) and names one passage. Each list is best
    first. A passage gets 1 / (k + rank) from each list it is in, with ranks
    starting at 1, and nothing from a list it is missing from.
    """
    scores: dict[tuple[int, int], float] = {}
    for keys in (vector_keys, keyword_keys):
        for rank, key in enumerate(keys, start=1):
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
    return scores


def _best_first(candidate: Candidate) -> tuple[float, int, int]:
    """The sort key that puts candidates in their final order."""
    # Sorting goes from low to high. The score is negated so that the highest
    # score comes first, while guide_id and position still go low to high.
    return (-candidate.score, candidate.guide_id, candidate.position)


def order_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Return the candidates best first: by score, then guide_id, then position.

    The tie-break uses only values that belong to the guide text. A database
    row id is never used, because it changes when a guide is ingested again,
    and the same question must always give the same order.
    """
    return sorted(candidates, key=_best_first)


def _passage_hit(candidate: Candidate) -> PassageHit:
    """Keep the parts of a candidate that are shown under its guide."""
    return PassageHit(
        heading=candidate.heading,
        position=candidate.position,
        text=candidate.content,
        similarity=candidate.similarity,
    )


def group_by_guide(
    candidates: list[Candidate],
    max_guides: int = MAX_GUIDES,
    max_passages: int = MAX_PASSAGES,
) -> list[GuideHit]:
    """Turn candidate passages into guides, best guide first.

    A guide's score is the score of its best candidate. Guides are ordered by
    that score, then by guide_id. Each guide keeps its best candidates, best
    first, then by position. The candidates may be given in any order.
    """
    # After this sort, the first candidate met for a guide is that guide's best
    # one, and guides are first met in exactly the order they must be returned.
    # A dict remembers the order in which its keys were added, so collecting
    # the candidates under their guide id keeps both orders.
    by_guide: dict[int, list[Candidate]] = {}
    for candidate in order_candidates(candidates):
        if candidate.guide_id not in by_guide:
            by_guide[candidate.guide_id] = []
        by_guide[candidate.guide_id].append(candidate)

    guides: list[GuideHit] = []
    for members in list(by_guide.values())[:max_guides]:
        best = members[0]
        guides.append(
            GuideHit(
                guide_id=best.guide_id,
                slug=best.slug,
                title=best.title,
                url=best.url,
                score=best.score,
                passages=[_passage_hit(member) for member in members[:max_passages]],
            )
        )
    return guides
