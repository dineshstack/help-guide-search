"""Shared setup for the integration tests: where the test database is.

pytest reads a file with this name by itself and offers its fixtures to every
test file in the same folder. A fixture is a function whose result a test asks
for by naming it as a parameter.

Two rules are enforced here, so that no single test has to remember them:

- The tests read TEST_DATABASE_URL and never DATABASE_URL, and the database
  name must end in "_test". The tests empty the passages table, so they must
  never be pointed at the real index.
- When the database is missing the tests fail. They are never skipped, so a
  green run always means the tests really ran.
"""

import os
from collections.abc import Iterator

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from guide_search.store import connect, ensure_schema


@pytest.fixture
def database_url() -> str:
    """The address of the test database, checked before any test uses it."""
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if url == "":
        pytest.fail(
            "TEST_DATABASE_URL is not set. The integration tests need a real "
            "Postgres. Put TEST_DATABASE_URL in .env and run: "
            "uv run --env-file .env pytest -m integration",
            pytrace=False,
        )

    # conninfo_to_dict takes the address apart. It never contacts the database.
    name = conninfo_to_dict(url).get("dbname") or ""
    if not name.endswith("_test"):
        pytest.fail(
            f"TEST_DATABASE_URL points at the database {name!r}. The tests "
            "empty the passages table, so they only run against a database "
            "whose name ends in '_test'.",
            pytrace=False,
        )
    return url


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection]:
    """A connection to the test database, with an empty passages table."""
    connection = connect(database_url)
    ensure_schema(connection)
    connection.execute("DELETE FROM passages")
    # Everything after "yield" runs when the test is over, passed or failed.
    yield connection
    connection.close()
