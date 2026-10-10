"""Ingestion: put the guides into the database, ready to be searched.

    uv run --env-file .env python -m guide_search.ingest

It reads the snapshot, cuts every guide into passages, turns each passage
into a vector and stores the passages in the `passages` table.

It can be run again at any time. For each guide it compares the passages it
would store with the rows that are already there. A guide that is the same is
left alone, so nothing is embedded twice. A guide that differs in any way has
its rows deleted and written again. A guide that is no longer in the snapshot
has its rows deleted.

All of this happens in one transaction: either every change is kept, or, if
anything fails half-way, none is and the table is exactly as it was before.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import psycopg
from pgvector import Vector

from guide_search.chunking import Passage, chunk_guide
from guide_search.embedding import Embedder
from guide_search.snapshot import Guide, load_snapshot
from guide_search.store import DatabaseUnavailable, connect, ensure_schema

DEFAULT_SNAPSHOT = "data/guides.json"

# The stored fields that decide whether a guide is "the same" as before:
# everything in a row except its id and its vector. guide_id comes first so
# that the rows can be sorted into guides.
STORED_ROWS_SQL = """
    SELECT guide_id, slug, title, url, heading, position,
           content, search_text, content_hash, embedding_model
    FROM passages
    ORDER BY guide_id, position
"""

INSERT_SQL = """
    INSERT INTO passages (guide_id, slug, title, url, heading, position,
                          content, search_text, content_hash,
                          embedding_model, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class IngestReport:
    """What one run did. The first four count guides, the last counts passages."""

    unchanged: int
    replaced: int
    added: int
    removed: int
    embedded: int      # passages embedded in this run


def ingest(conn: psycopg.Connection, guides: list[Guide], embedder: Embedder) -> IngestReport:
    """Make the passages table match the given guides, and say what was done."""
    unchanged = 0
    replaced = 0
    added = 0
    removed = 0
    embedded = 0

    # Everything inside this block is one transaction. If any line raises,
    # Postgres undoes every change made in the block.
    with conn.transaction():
        ensure_schema(conn)

        # Chunk every guide before the first one is embedded. Embedding is the
        # slow part, and a guide that cannot be chunked should stop the run
        # before that work starts.
        passages_of = {guide.id: chunk_guide(guide) for guide in guides}
        stored = _stored_rows(conn)

        for guide in guides:
            passages = passages_of[guide.id]
            before = stored.get(guide.id, [])
            if before == _comparable_rows(passages, embedder.model_id):
                unchanged += 1
                continue
            _replace_guide(conn, guide.id, passages, embedder)
            embedded += len(passages)
            if before == []:
                added += 1
            else:
                replaced += 1

        for guide_id in stored:
            if guide_id not in passages_of:
                conn.execute("DELETE FROM passages WHERE guide_id = %s", (guide_id,))
                removed += 1

    return IngestReport(
        unchanged=unchanged,
        replaced=replaced,
        added=added,
        removed=removed,
        embedded=embedded,
    )


def _stored_rows(conn: psycopg.Connection) -> dict[int, list[tuple]]:
    """The rows now in the table, as guide id -> its rows in position order."""
    rows = conn.execute(STORED_ROWS_SQL).fetchall()
    stored: dict[int, list[tuple]] = {}
    for row in rows:
        guide_id = row[0]
        if guide_id not in stored:
            stored[guide_id] = []
        stored[guide_id].append(row[1:])
    return stored


def _comparable_rows(passages: list[Passage], model_id: str) -> list[tuple]:
    """The new passages, with the same fields in the same order as _stored_rows."""
    rows = []
    for passage in passages:
        rows.append(
            (
                passage.slug,
                passage.title,
                passage.url,
                passage.heading,
                passage.position,
                passage.content,
                passage.search_text,
                passage.content_hash,
                model_id,
            )
        )
    return rows


def _replace_guide(
    conn: psycopg.Connection, guide_id: int, passages: list[Passage], embedder: Embedder
) -> None:
    """Delete the guide's rows, then embed and store its new passages."""
    conn.execute("DELETE FROM passages WHERE guide_id = %s", (guide_id,))

    # What is embedded is search_text: the passage with its title and heading.
    vectors = embedder.embed_passages([passage.search_text for passage in passages])

    # strict=True raises if the embedder returned too few or too many vectors,
    # instead of quietly storing only some of the passages.
    for passage, vector in zip(passages, vectors, strict=True):
        conn.execute(
            INSERT_SQL,
            (
                passage.guide_id,
                passage.slug,
                passage.title,
                passage.url,
                passage.heading,
                passage.position,
                passage.content,
                passage.search_text,
                passage.content_hash,
                embedder.model_id,
                # Postgres needs a pgvector Vector here, not a plain list.
                Vector(vector),
            ),
        )


def format_report(report: IngestReport) -> str:
    """The two lines the command prints when it is done."""
    return (
        f"Guides: {report.added} added, {report.replaced} replaced, "
        f"{report.unchanged} unchanged, {report.removed} removed\n"
        f"Passages embedded: {report.embedded}"
    )


def main(argv: list[str] | None = None, embedder: Embedder | None = None) -> int:
    """Run ingestion from the command line. Returns 0, or 1 when it could not start.

    `embedder` is only given by the tests. Left out, the real model is used.
    """
    parser = argparse.ArgumentParser(
        description="Put the guides into the database, ready to be searched."
    )
    parser.add_argument(
        "--snapshot",
        default=DEFAULT_SNAPSHOT,
        help="the snapshot file (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    url = os.environ.get("DATABASE_URL", "").strip()
    if url == "":
        print(
            "DATABASE_URL is not set. Put it in .env and run: "
            "uv run --env-file .env python -m guide_search.ingest",
            file=sys.stderr,
        )
        return 1

    guides = load_snapshot(args.snapshot)

    # Connect before the model is loaded: a database that cannot be reached is
    # then reported at once, not after several seconds of loading.
    try:
        conn = connect(url)
    except DatabaseUnavailable as error:
        print(error, file=sys.stderr)
        return 1

    if embedder is None:
        # Imported here and not at the top, because this import loads torch,
        # which takes seconds. Importing this module must stay quick.
        from guide_search.e5 import E5Embedder

        embedder = E5Embedder()

    # "with conn" closes the connection when the block ends.
    with conn:
        report = ingest(conn, guides, embedder)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
