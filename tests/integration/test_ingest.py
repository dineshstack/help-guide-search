"""Tests for ingestion: putting guides into the database, again and again.

They need a real Postgres (TEST_DATABASE_URL). The guides are the five small
invented guides in tests/fixtures/five_guides.json, and the embedder is the
fake one, so no model is downloaded.

The thing to prove is that ingestion can be run any number of times: a run
after no change does nothing, a run after a change touches only the guides
that changed, and a run that fails leaves everything as it was.
"""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest

from guide_search.chunking import chunk_guide
from guide_search.embedding import FakeEmbedder
from guide_search.ingest import IngestReport, format_report, ingest, main
from guide_search.snapshot import PUBLIC_PAGE_PREFIX, Guide, load_snapshot

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"

# How many passages the five guides give: 4 + 1 + 3 + 5 + 3.
PASSAGES_IN_FIXTURE = 16

# Nothing listens on port 1, so a connection there is refused at once.
NOTHING_LISTENS_HERE = "postgresql://nobody@127.0.0.1:1/nothing_test"


class CountingEmbedder(FakeEmbedder):
    """The fake embedder, but it remembers how many texts it was given."""

    def __init__(self) -> None:
        self.texts_embedded = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return super().embed_passages(texts)


class OtherModelEmbedder(FakeEmbedder):
    """The fake embedder under another name, as if the model had been changed."""

    model_id = "fake-2"


class BreaksAtEmbedder(FakeEmbedder):
    """The fake embedder, but it fails when it meets a passage of one guide."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        for text in texts:
            # Every search_text starts with the title of its guide.
            if text.startswith(self.title):
                raise RuntimeError("the embedder broke")
        return super().embed_passages(texts)


def fixture_guides() -> list[Guide]:
    """The five invented guides: ids 2, 3, 5, 8 and 13, in that order."""
    return load_snapshot(FIXTURE)


def stored_rows(conn: psycopg.Connection) -> list[tuple]:
    """Every stored row except its vector, in a fixed order. The row id is first."""
    return conn.execute(
        """
        SELECT id, guide_id, slug, title, url, heading, position,
               content, search_text, content_hash, embedding_model
        FROM passages
        ORDER BY guide_id, position
        """
    ).fetchall()


def row_ids(conn: psycopg.Connection, guide_id: int) -> list[int]:
    """The row ids of one guide. A guide that is replaced gets new ones."""
    rows = conn.execute(
        "SELECT id FROM passages WHERE guide_id = %s ORDER BY position", (guide_id,)
    ).fetchall()
    return [row[0] for row in rows]


def swap_first_two_sections(body: str) -> str:
    """The same body with its first two headed sections in the other order."""
    parts = body.split("\n\n## ")  # the intro, then one item for each section
    parts[1], parts[2] = parts[2], parts[1]
    return "\n\n## ".join(parts)


# ---------------------------------------------------------------------------
# The first run
# ---------------------------------------------------------------------------


def test_first_ingest_reports_every_guide_as_added(conn: psycopg.Connection):
    report = ingest(conn, fixture_guides(), FakeEmbedder())

    assert report == IngestReport(
        unchanged=0, replaced=0, added=5, removed=0, embedded=PASSAGES_IN_FIXTURE
    )


def test_first_ingest_stores_exactly_the_passages_that_chunking_gives(conn: psycopg.Connection):
    guides = fixture_guides()

    ingest(conn, guides, FakeEmbedder())

    expected = []
    for guide in guides:
        for passage in chunk_guide(guide):
            expected.append(
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
                    "fake",
                )
            )
    without_row_id = [row[1:] for row in stored_rows(conn)]
    assert without_row_id == expected
    assert len(without_row_id) == PASSAGES_IN_FIXTURE


def test_the_stored_vector_is_the_embedding_of_search_text(conn: psycopg.Connection):
    ingest(conn, fixture_guides(), FakeEmbedder())

    rows = conn.execute("SELECT search_text, embedding FROM passages").fetchall()

    for search_text, embedding in rows:
        expected = FakeEmbedder().embed_passages([search_text])[0]
        # Postgres keeps a vector as 4-byte numbers, so the last digits differ.
        assert embedding.to_list() == pytest.approx(expected, abs=1e-6)


def test_ingest_creates_the_table_when_it_is_missing(conn: psycopg.Connection):
    conn.execute("DROP TABLE passages")

    ingest(conn, fixture_guides(), FakeEmbedder())

    assert len(stored_rows(conn)) == PASSAGES_IN_FIXTURE


def test_related_articles_section_is_never_stored(conn: psycopg.Connection):
    ingest(conn, fixture_guides(), FakeEmbedder())

    rows = stored_rows(conn)
    headings = [row[5] for row in rows]
    search_texts = [row[8] for row in rows]

    assert "Related articles" not in headings
    # "gazebo" occurs in the fixture only inside a Related articles section.
    assert not any("gazebo" in text for text in search_texts)


# ---------------------------------------------------------------------------
# Running it again
# ---------------------------------------------------------------------------


def test_second_ingest_reports_every_guide_as_unchanged_and_embeds_nothing(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    before = stored_rows(conn)
    embedder = CountingEmbedder()

    report = ingest(conn, guides, embedder)

    assert report == IngestReport(unchanged=5, replaced=0, added=0, removed=0, embedded=0)
    assert embedder.texts_embedded == 0
    # Not one row was rewritten: even the row ids are the same.
    assert stored_rows(conn) == before


def test_a_guide_whose_text_changes_is_replaced_and_the_others_are_left_alone(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    ids_before = {guide.id: row_ids(conn, guide.id) for guide in guides}
    spar = guides[1]
    changed = replace(spar, body=spar.body.replace("until it clicks", "until it locks"))

    report = ingest(conn, [guides[0], changed] + guides[2:], FakeEmbedder())

    assert report == IngestReport(unchanged=4, replaced=1, added=0, removed=0, embedded=1)
    content = conn.execute("SELECT content FROM passages WHERE guide_id = %s", (spar.id,)).fetchone()[0]
    assert "until it locks" in content
    # The changed guide has new rows. Every other guide kept its rows.
    assert row_ids(conn, spar.id) != ids_before[spar.id]
    for guide in guides:
        if guide.id != spar.id:
            assert row_ids(conn, guide.id) == ids_before[guide.id]


def test_a_guide_whose_slug_and_url_change_but_not_its_text_is_replaced(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    sail = guides[0]
    hashes_before = conn.execute(
        "SELECT content_hash FROM passages WHERE guide_id = %s ORDER BY position", (sail.id,)
    ).fetchall()
    moved = replace(sail, slug="patch-a-sail", url=PUBLIC_PAGE_PREFIX + "patch-a-sail")

    report = ingest(conn, [moved] + guides[1:], FakeEmbedder())

    assert report == IngestReport(unchanged=4, replaced=1, added=0, removed=0, embedded=4)
    rows = conn.execute(
        "SELECT slug, url, content_hash FROM passages WHERE guide_id = %s ORDER BY position", (sail.id,)
    ).fetchall()
    assert [row[0] for row in rows] == ["patch-a-sail"] * 4
    assert [row[1] for row in rows] == [PUBLIC_PAGE_PREFIX + "patch-a-sail"] * 4
    # The text did not change, so the hashes did not either.
    assert [(row[2],) for row in rows] == hashes_before
    old_slug = conn.execute("SELECT count(*) FROM passages WHERE slug = %s", (sail.slug,)).fetchone()[0]
    assert old_slug == 0


@pytest.mark.parametrize("kind", ["title", "heading", "section order"])
def test_a_change_of_title_heading_or_section_order_replaces_the_guide(conn: psycopg.Connection, kind: str):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    sail = guides[0]
    if kind == "title":
        changed = replace(sail, title="Repair a ripped sail")
    elif kind == "heading":
        changed = replace(sail, body=sail.body.replace("## Before you start", "## Before you begin"))
    else:
        changed = replace(sail, body=swap_first_two_sections(sail.body))

    report = ingest(conn, [changed] + guides[1:], FakeEmbedder())

    assert report == IngestReport(unchanged=4, replaced=1, added=0, removed=0, embedded=4)


def test_a_change_of_updated_at_alone_leaves_the_guide_unchanged(conn: psycopg.Connection):
    # updated_at is not stored, so a new date with the same text is no change.
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    touched = replace(guides[0], updated_at="2026-06-30T12:00:00+00:00")

    report = ingest(conn, [touched] + guides[1:], FakeEmbedder())

    assert report == IngestReport(unchanged=5, replaced=0, added=0, removed=0, embedded=0)


def test_a_guide_that_loses_a_section_ends_with_one_row_fewer(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    sail = guides[0]
    assert len(row_ids(conn, sail.id)) == 4
    section = "## Before you start\n\nDry the sail and brush off any sand. Lay it flat on the bench.\n\n"
    assert section in sail.body
    shorter = replace(sail, body=sail.body.replace(section, ""))

    ingest(conn, [shorter] + guides[1:], FakeEmbedder())

    rows = conn.execute(
        "SELECT position, heading FROM passages WHERE guide_id = %s ORDER BY position", (sail.id,)
    ).fetchall()
    assert rows == [(0, ""), (1, "Patch the tear"), (2, "Troubleshooting")]


def test_a_guide_removed_from_the_snapshot_disappears(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    questions = guides[3]
    without_it = guides[:3] + guides[4:]

    report = ingest(conn, without_it, FakeEmbedder())

    assert report == IngestReport(unchanged=4, replaced=0, added=0, removed=1, embedded=0)
    assert row_ids(conn, questions.id) == []
    assert len(stored_rows(conn)) == PASSAGES_IN_FIXTURE - 5


def test_a_guide_new_in_the_snapshot_is_added(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides[:4], FakeEmbedder())

    report = ingest(conn, guides, FakeEmbedder())

    assert report == IngestReport(unchanged=4, replaced=0, added=1, removed=0, embedded=3)
    assert len(stored_rows(conn)) == PASSAGES_IN_FIXTURE


def test_a_different_embedding_model_replaces_every_guide(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())

    report = ingest(conn, guides, OtherModelEmbedder())

    assert report == IngestReport(
        unchanged=0, replaced=5, added=0, removed=0, embedded=PASSAGES_IN_FIXTURE
    )
    models = conn.execute("SELECT DISTINCT embedding_model FROM passages").fetchall()
    assert models == [("fake-2",)]


# ---------------------------------------------------------------------------
# A run that fails
# ---------------------------------------------------------------------------


def test_an_error_half_way_leaves_the_previous_rows_untouched(conn: psycopg.Connection):
    guides = fixture_guides()
    ingest(conn, guides, FakeEmbedder())
    before = stored_rows(conn)
    # A new title changes every passage of a guide, so every guide must be replaced.
    retitled = [replace(guide, title=guide.title + " (new)") for guide in guides]
    embedder = BreaksAtEmbedder(retitled[2].title)

    with pytest.raises(RuntimeError):
        ingest(conn, retitled, embedder)

    # Two guides had already been replaced when the third one failed...
    assert embedder.calls == 3
    # ...and still nothing changed, not even a row id.
    assert stored_rows(conn) == before


# ---------------------------------------------------------------------------
# The command: python -m guide_search.ingest
# ---------------------------------------------------------------------------


def test_format_report_gives_the_two_lines_that_the_command_prints():
    report = IngestReport(unchanged=57, replaced=2, added=1, removed=0, embedded=14)

    assert format_report(report) == (
        "Guides: 1 added, 2 replaced, 57 unchanged, 0 removed\n"
        "Passages embedded: 14"
    )


def test_main_ingests_the_snapshot_and_prints_the_report(
    conn: psycopg.Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    # main reads DATABASE_URL. Here it is given the address of the test database.
    monkeypatch.setenv("DATABASE_URL", database_url)

    exit_code = main(["--snapshot", str(FIXTURE)], embedder=FakeEmbedder())

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "Guides: 5 added, 0 replaced, 0 unchanged, 0 removed\n"
        "Passages embedded: 16\n"
    )
    assert len(stored_rows(conn)) == PASSAGES_IN_FIXTURE


def test_main_without_database_url_says_so_and_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    exit_code = main(["--snapshot", str(FIXTURE)], embedder=FakeEmbedder())

    assert exit_code == 1
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_main_says_so_and_returns_1_when_the_database_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("DATABASE_URL", NOTHING_LISTENS_HERE)

    exit_code = main(["--snapshot", str(FIXTURE)], embedder=FakeEmbedder())

    assert exit_code == 1
    assert "could not connect to the database" in capsys.readouterr().err


def test_importing_the_ingest_module_does_not_load_torch():
    # The real model is imported inside main(), so that importing this module
    # stays quick. A fresh Python process is used because another test in the
    # same run may already have loaded torch into this one.
    code = (
        "import sys\n"
        "import guide_search.ingest\n"
        "print('torch' in sys.modules, 'sentence_transformers' in sys.modules)\n"
    )
    # The new process must find guide_search in the same places as this one.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", "False"]
