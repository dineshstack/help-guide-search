"""Tests for store.py: connecting to Postgres and the passages table.

They need a real Postgres with the pgvector extension, at TEST_DATABASE_URL.
The `conn` fixture (see conftest.py) hands each test a connection to the test
database with the table in place and empty.

All values are invented.
"""

import psycopg
import pytest
from pgvector import Vector

from guide_search.store import (
    DatabaseUnavailable,
    IndexNotReady,
    connect,
    count_passages,
    ensure_schema,
)

pytestmark = pytest.mark.integration

# Nothing listens on port 1, so a connection there is refused at once.
NOTHING_LISTENS_HERE = "postgresql://nobody@127.0.0.1:1/nothing_test"

# What keyword search sends for a question made only of words that Postgres
# drops as too common to search for.
ONLY_COMMON_WORDS = "SELECT plainto_tsquery('english', 'the') || plainto_tsquery('english', 'of')"


def unit_vector(index: int) -> Vector:
    """A vector of 384 numbers: 1.0 at one position and 0.0 everywhere else."""
    numbers = [0.0] * 384
    numbers[index] = 1.0
    return Vector(numbers)


def record_notices(connection: psycopg.Connection) -> list[str]:
    """Collect every notice Postgres sends on this connection into a list."""
    notices: list[str] = []

    def remember(notice: psycopg.errors.Diagnostic) -> None:
        notices.append(notice.message_primary)

    connection.add_notice_handler(remember)
    return notices


def insert_passage(
    conn: psycopg.Connection,
    guide_id: int,
    position: int,
    search_text: str = "Print a ticket\nPress Print.",
    model: str = "fake",
) -> None:
    """Insert one invented passage. The tsv column is left to Postgres."""
    conn.execute(
        """
        INSERT INTO passages (guide_id, slug, title, url, heading, position,
                              content, search_text, content_hash,
                              embedding_model, embedding)
        VALUES (%s, 'print-a-ticket', 'Print a ticket',
                'https://example.test/print-a-ticket', '', %s,
                'Press Print.', %s, 'not-a-real-hash', %s, %s)
        """,
        (guide_id, position, search_text, model, unit_vector(0)),
    )


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


def test_connect_to_an_address_where_nothing_listens_raises_database_unavailable():
    with pytest.raises(DatabaseUnavailable):
        connect(NOTHING_LISTENS_HERE)


def test_a_question_of_only_common_words_sends_no_notice(database_url: str):
    # First show the problem on a plain connection, so that this test cannot
    # pass merely because Postgres stopped sending these notices.
    plain = psycopg.connect(database_url, autocommit=True)
    plain_notices = record_notices(plain)
    plain.execute(ONLY_COMMON_WORDS)
    plain.close()
    assert plain_notices != []

    # A connection from connect() must stay silent for the same query.
    connection = connect(database_url)
    notices = record_notices(connection)
    connection.execute(ONLY_COMMON_WORDS)
    connection.close()
    assert notices == []


def test_a_change_made_outside_a_transaction_block_is_kept(conn: psycopg.Connection, database_url: str):
    # connect() gives a connection that saves each statement at once, unless
    # the statement is inside "with conn.transaction():". A second connection
    # can only see the row if the first one really saved it.
    insert_passage(conn, guide_id=1, position=0)

    other = connect(database_url)
    count = other.execute("SELECT count(*) FROM passages").fetchone()[0]
    other.close()

    assert count == 1


def test_a_failed_transaction_block_leaves_no_trace(conn: psycopg.Connection):
    with pytest.raises(RuntimeError):
        with conn.transaction():
            insert_passage(conn, guide_id=1, position=0)
            raise RuntimeError("stop half-way")

    assert conn.execute("SELECT count(*) FROM passages").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


def test_ensure_schema_twice_is_harmless_and_keeps_the_rows(conn: psycopg.Connection):
    # The conn fixture has already called ensure_schema once.
    insert_passage(conn, guide_id=1, position=0)

    ensure_schema(conn)
    ensure_schema(conn)

    assert conn.execute("SELECT count(*) FROM passages").fetchone()[0] == 1


def test_passages_table_has_the_columns_of_the_design(conn: psycopg.Connection):
    # format_type gives a column's type as it is written in CREATE TABLE.
    rows = conn.execute(
        """
        SELECT attname, format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'passages'::regclass AND attnum > 0 AND NOT attisdropped
        ORDER BY attnum
        """
    ).fetchall()

    assert rows == [
        ("id", "bigint"),
        ("guide_id", "integer"),
        ("slug", "text"),
        ("title", "text"),
        ("url", "text"),
        ("heading", "text"),
        ("position", "integer"),
        ("content", "text"),
        ("search_text", "text"),
        ("content_hash", "text"),
        ("embedding_model", "text"),
        ("embedding", "vector(384)"),
        ("tsv", "tsvector"),
    ]


def test_tsv_column_is_filled_by_postgres_with_english_stems(conn: psycopg.Connection):
    insert_passage(conn, guide_id=1, position=0, search_text="Printing the invoices")

    tsv = conn.execute("SELECT tsv::text FROM passages").fetchone()[0]

    # "the" is dropped as a common word, and the other two are cut to stems.
    assert tsv == "'invoic':3 'print':1"


def test_tsv_column_has_the_keyword_index(conn: psycopg.Connection):
    definition = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'passages_tsv_idx'"
    ).fetchone()[0]

    assert "USING gin (tsv)" in definition


def test_two_passages_cannot_share_a_guide_and_a_position(conn: psycopg.Connection):
    insert_passage(conn, guide_id=1, position=0)

    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_passage(conn, guide_id=1, position=0)


def test_a_vector_goes_in_as_a_pgvector_vector_and_comes_back_with_384_numbers(
    conn: psycopg.Connection,
):
    conn.execute(
        """
        INSERT INTO passages (guide_id, slug, title, url, heading, position,
                              content, search_text, content_hash,
                              embedding_model, embedding)
        VALUES (1, 's', 't', 'u', '', 0, 'c', 'c', 'h', 'fake', %s)
        """,
        (unit_vector(7),),
    )

    stored = conn.execute("SELECT embedding FROM passages").fetchone()[0]

    assert isinstance(stored, Vector)
    assert stored.to_list() == unit_vector(7).to_list()
    assert len(stored.to_list()) == 384


def test_a_vector_of_the_wrong_size_is_refused(conn: psycopg.Connection):
    with pytest.raises(psycopg.errors.DataException):
        conn.execute(
            """
            INSERT INTO passages (guide_id, slug, title, url, heading, position,
                                  content, search_text, content_hash,
                                  embedding_model, embedding)
            VALUES (1, 's', 't', 'u', '', 0, 'c', 'c', 'h', 'fake', %s)
            """,
            (Vector([1.0, 0.0, 0.0]),),
        )


def test_a_new_database_without_the_extension_can_be_connected_to_and_set_up(
    conn: psycopg.Connection, database_url: str
):
    # Put the test database back to how a brand-new database looks.
    conn.execute("DROP TABLE passages")
    conn.execute("DROP EXTENSION vector")

    fresh = connect(database_url)  # must not fail although vectors are unknown
    ensure_schema(fresh)
    insert_passage(fresh, guide_id=1, position=0)
    stored = fresh.execute("SELECT embedding FROM passages").fetchone()[0]
    fresh.close()

    # The same connection that created the extension can read vectors.
    assert isinstance(stored, Vector)


# ---------------------------------------------------------------------------
# count_passages
# ---------------------------------------------------------------------------


def test_count_passages_raises_index_not_ready_when_the_table_is_missing(conn: psycopg.Connection):
    conn.execute("DROP TABLE passages")

    with pytest.raises(IndexNotReady):
        count_passages(conn, "fake")


def test_count_passages_raises_index_not_ready_when_the_table_is_empty(conn: psycopg.Connection):
    with pytest.raises(IndexNotReady):
        count_passages(conn, "fake")


def test_count_passages_raises_index_not_ready_when_only_another_model_has_rows(
    conn: psycopg.Connection,
):
    insert_passage(conn, guide_id=1, position=0, model="some-other-model")

    with pytest.raises(IndexNotReady):
        count_passages(conn, "fake")


def test_count_passages_returns_the_number_of_rows_of_that_model(conn: psycopg.Connection):
    insert_passage(conn, guide_id=1, position=0, model="fake")
    insert_passage(conn, guide_id=1, position=1, model="fake")
    insert_passage(conn, guide_id=2, position=0, model="some-other-model")

    assert count_passages(conn, "fake") == 2
