"""The read-only database account, proved on a real Postgres.

The service connects as an account that may read the passages table and
nothing else. These tests make such an account (under a name of their own, so
they never touch the real one), give it the same right ingestion gives the
real account, and check what it can and cannot do.
"""

import secrets
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from guide_search.config import Settings
from guide_search.embedding import FakeEmbedder
from guide_search.ingest import ingest
from guide_search.search import search
from guide_search.service import Service
from guide_search.snapshot import load_snapshot
from guide_search.store import READER_ROLE, connect, grant_read_access

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"
TEST_ROLE = "guides_reader_for_tests"
SETTINGS = Settings(threshold=0.3, default_mode="hybrid")


def drop_role(conn: psycopg.Connection, role: str) -> None:
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
    if exists:
        # A role cannot be dropped while it still holds a right on something.
        conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
        conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.fixture
def reader_url(conn: psycopg.Connection, database_url: str) -> Iterator[str]:
    """A fresh login-only account, and the address to connect as it. Dropped afterwards."""
    ingest(conn, load_snapshot(FIXTURE), FakeEmbedder())
    drop_role(conn, TEST_ROLE)  # left over from a run that was cut short
    password = secrets.token_hex(12)
    conn.execute(
        sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(TEST_ROLE), sql.Literal(password))
    )
    address = conninfo_to_dict(database_url)
    address.update(user=TEST_ROLE, password=password)
    yield make_conninfo(**address)
    drop_role(conn, TEST_ROLE)


def test_the_real_account_name_is_not_the_one_these_tests_use():
    assert READER_ROLE == "guides_reader"
    assert TEST_ROLE != READER_ROLE


def test_an_account_that_was_given_nothing_cannot_read_the_passages(reader_url: str):
    with connect(reader_url) as reader:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            reader.execute("SELECT count(*) FROM passages")


def test_after_the_grant_the_account_can_read_and_search(conn: psycopg.Connection, reader_url: str):
    assert grant_read_access(conn, TEST_ROLE) is True

    with connect(reader_url) as reader:
        rows = reader.execute("SELECT count(*) FROM passages").fetchone()[0]
        found = search(reader, FakeEmbedder(), "mend a torn sail", "hybrid", 0.3)

    assert rows > 0
    assert found.guides[0].slug == "mend-a-torn-sail"


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM passages",
        "UPDATE passages SET title = 'changed'",
        "INSERT INTO passages (guide_id) VALUES (1)",
        "TRUNCATE passages",
        "DROP TABLE passages",
        "CREATE TABLE should_not_exist (n integer)",
    ],
)
def test_the_account_is_refused_every_change(conn: psycopg.Connection, reader_url: str, statement: str):
    grant_read_access(conn, TEST_ROLE)
    before = conn.execute("SELECT count(*) FROM passages").fetchone()[0]

    # An ordinary connection, not a read-only one: the refusal here comes from
    # what the account is allowed to do, not from the session setting.
    with connect(reader_url) as reader:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            reader.execute(statement)

    assert conn.execute("SELECT count(*) FROM passages").fetchone()[0] == before


def test_the_whole_service_works_when_it_connects_as_the_account(conn: psycopg.Connection, reader_url: str):
    grant_read_access(conn, TEST_ROLE)
    service = Service(reader_url, FakeEmbedder(), SETTINGS)

    service.health()
    found = service.search("mend a torn sail", None)
    listing = service.guides()
    guide = service.guide("mend-a-torn-sail")

    assert found.guides[0].slug == "mend-a-torn-sail"
    assert len(listing.guides) == 5
    assert guide is not None and guide.passages


def test_the_grant_does_nothing_when_the_account_does_not_exist(conn: psycopg.Connection):
    assert grant_read_access(conn, "an_account_that_does_not_exist") is False
