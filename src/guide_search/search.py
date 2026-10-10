"""Search: find the passages that answer a question.

There are three modes.

- vector: turn the question into a vector and take the passages whose vectors
  point most nearly the same way. This finds text with the same meaning even
  when it shares no word with the question.
- keyword: Postgres full-text search. A passage matches when it contains any
  word of the question.
- hybrid: run both and merge the two lists (see rrf_scores in ranking.py).

Every mode ends the same way. The candidate passages (at most 20 from each
search, so up to 40 in hybrid mode) are grouped by guide, and the result is up
to 3 guides with up to 2 passages each.

Sometimes the honest result is "the guides do not cover this". Then the result
has no guides and the reason "no_match". That is not an error. An error is
when search could not look at all: the database cannot be reached, or holds no
passages. Those raise an exception and never come back as an empty result.
"""

from dataclasses import replace

import psycopg
from pgvector import Vector

from guide_search.embedding import Embedder
from guide_search.ranking import (
    CANDIDATES,
    Candidate,
    GuideDetail,
    PassageHit,
    SearchResult,
    group_by_guide,
    question_words,
    rrf_scores,
)
from guide_search.store import IndexNotReady, count_passages

MODES = ("vector", "keyword", "hybrid")
NO_MATCH = "no_match"

# "<=>" is pgvector's cosine distance: 0 when two vectors point the same way.
# Similarity is 1 minus that distance. Ties are broken by guide_id and position
# and never by the row id, which changes when a guide is ingested again.
VECTOR_SQL = """
    SELECT guide_id, slug, title, url, heading, position, content,
           1 - (embedding <=> %s) AS similarity
    FROM passages
    WHERE embedding_model = %s
    ORDER BY similarity DESC, guide_id, position
    LIMIT %s
"""

# {tsquery} is replaced by one "plainto_tsquery('english', %s)" for each word
# of the question, joined by "||" (or). The words themselves are only ever
# passed as %s parameters, so nothing in a question can be read as SQL or as
# query syntax. "@@" means "matches", and ts_rank_cd is in effect a count of
# the matching words. In keyword mode the vector parameter is NULL, and so is
# the similarity.
KEYWORD_SQL = """
    WITH question AS (SELECT {tsquery} AS query)
    SELECT guide_id, slug, title, url, heading, position, content,
           ts_rank_cd(tsv, query) AS rank,
           1 - (embedding <=> %s::vector) AS similarity
    FROM passages CROSS JOIN question
    WHERE embedding_model = %s AND tsv @@ query
    ORDER BY rank DESC, guide_id, position
    LIMIT %s
"""

GUIDE_SQL = """
    SELECT guide_id, slug, title, url, heading, position, content
    FROM passages
    WHERE slug = %s
    ORDER BY position
"""


def search(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    mode: str,
    threshold: float,
) -> SearchResult:
    """Search the passages stored for this embedder's model.

    Raises ValueError for an unknown mode and IndexNotReady when the table is
    missing or holds no passages for the model.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; the modes are: " + ", ".join(MODES))
    # An empty table would make every question look like "no match", so it is
    # ruled out before anything else is done.
    count_passages(conn, embedder.model_id)

    if mode == "keyword":
        # Keyword mode never embeds the question and has no threshold: it
        # refuses only when not one word of the question is in any passage.
        candidates = _keyword_candidates(conn, embedder.model_id, question, None)
        if not candidates:
            return _no_match(question, mode, None)
        return _found(question, mode, None, candidates)

    question_vector = Vector(embedder.embed_query(question))
    vector_list = _vector_candidates(conn, embedder.model_id, question_vector)
    if not vector_list:
        # Only possible if the table was emptied after the count above.
        raise IndexNotReady("the passages table was emptied while it was being searched")

    # Vector and hybrid share one rule for refusing: the best similarity must
    # reach the threshold. A fusion score cannot be used for this. It depends
    # only on rank positions, so the top passage scores about 1/61 whether it
    # is a perfect answer or nonsense.
    best_similarity = vector_list[0].similarity
    if best_similarity < threshold:
        return _no_match(question, mode, best_similarity)

    if mode == "vector":
        candidates = vector_list
    else:
        keyword_list = _keyword_candidates(conn, embedder.model_id, question, question_vector)
        candidates = _fuse(vector_list, keyword_list)
    return _found(question, mode, best_similarity, candidates)


def read_guide(conn: psycopg.Connection, slug: str) -> GuideDetail | None:
    """Return one guide's passages in reading order, or None if no guide has this slug."""
    rows = conn.execute(GUIDE_SQL, (slug,)).fetchall()
    if not rows:
        return None

    # Every row of a guide carries the same guide_id, slug, title and url.
    guide_id, guide_slug, title, url = rows[0][:4]
    passages = []
    for row in rows:
        heading, position, content = row[4:]
        passages.append(
            PassageHit(heading=heading, position=position, text=content, similarity=None)
        )
    return GuideDetail(
        guide_id=guide_id, slug=guide_slug, title=title, url=url, passages=passages
    )


def _found(
    question: str,
    mode: str,
    best_similarity: float | None,
    candidates: list[Candidate],
) -> SearchResult:
    """The result when there is something to return: the candidates, grouped by guide."""
    return SearchResult(
        query=question,
        mode=mode,
        reason=None,
        best_similarity=best_similarity,
        guides=group_by_guide(candidates),
    )


def _no_match(question: str, mode: str, best_similarity: float | None) -> SearchResult:
    """The result when the guides do not cover the question: no guides, and the reason."""
    return SearchResult(
        query=question,
        mode=mode,
        reason=NO_MATCH,
        best_similarity=best_similarity,
        guides=[],
    )


def _candidate(row: tuple, score: float, similarity: float | None) -> Candidate:
    """Make a Candidate from the first seven columns of a row and its two numbers."""
    guide_id, slug, title, url, heading, position, content = row[:7]
    return Candidate(
        guide_id=guide_id,
        slug=slug,
        title=title,
        url=url,
        heading=heading,
        position=position,
        content=content,
        score=score,
        similarity=similarity,
    )


def _vector_candidates(
    conn: psycopg.Connection, model_id: str, question_vector: Vector
) -> list[Candidate]:
    """The passages most similar to the question, best first."""
    rows = conn.execute(VECTOR_SQL, (question_vector, model_id, CANDIDATES)).fetchall()
    # The eighth column is the similarity. In this mode it is the score as well.
    return [_candidate(row, score=row[7], similarity=row[7]) for row in rows]


def _keyword_candidates(
    conn: psycopg.Connection,
    model_id: str,
    question: str,
    question_vector: Vector | None,
) -> list[Candidate]:
    """The passages that contain any word of the question, best first.

    `question_vector` is None in keyword mode. In hybrid mode it is given, so
    that a passage only this search finds still gets its similarity.
    """
    words = question_words(question)
    if not words:
        return []

    # One placeholder for each word, joined by "||". Only this fixed text is
    # put into the SQL; the words travel separately, as parameters.
    placeholders = ["plainto_tsquery('english', %s)"] * len(words)
    query = KEYWORD_SQL.format(tsquery=" || ".join(placeholders))
    rows = conn.execute(query, (*words, question_vector, model_id, CANDIDATES)).fetchall()
    # The eighth column is the keyword rank, the ninth the similarity (or NULL).
    return [_candidate(row, score=row[7], similarity=row[8]) for row in rows]


def _fuse(vector_list: list[Candidate], keyword_list: list[Candidate]) -> list[Candidate]:
    """Merge the two lists into one, each passage once, scored by rank fusion."""
    vector_keys = [(candidate.guide_id, candidate.position) for candidate in vector_list]
    keyword_keys = [(candidate.guide_id, candidate.position) for candidate in keyword_list]
    scores = rrf_scores(vector_keys, keyword_keys)

    # A passage found by both searches is in both lists. Keeping the
    # candidates in a dict by their key leaves one of each.
    fused: dict[tuple[int, int], Candidate] = {}
    for candidate in vector_list + keyword_list:
        key = (candidate.guide_id, candidate.position)
        # replace() copies the candidate with one field changed.
        fused[key] = replace(candidate, score=scores[key])
    return list(fused.values())
