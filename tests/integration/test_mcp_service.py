"""The MCP tools on the real service and a real Postgres.

test_mcp_server.py tests the tools with the service replaced by a stub. Here
nothing is replaced except the model: the tools call the real Service, which
searches the test database (TEST_DATABASE_URL) holding the five invented
guides of tests/fixtures/five_guides.json.

Two more things are proved here because they need a real database: a
read-only connection really is refused every write, and the server really
works when an assistant starts it as a child process and talks to it over
standard input and output.
"""

import asyncio
import os
import sys
from pathlib import Path

import psycopg
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from guide_search import service as service_module
from guide_search.config import Settings
from guide_search.embedding import FakeEmbedder
from guide_search.ingest import ingest
from guide_search.mcp_server import NOT_BUILT, UNREACHABLE, build_server
from guide_search.service import Service
from guide_search.snapshot import load_snapshot
from guide_search.store import connect

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"
STDIO_SERVER = Path(__file__).resolve().parent / "stdio_server.py"

# The tests' own settings. The threshold suits the fake embedder's scores.
SETTINGS = Settings(threshold=0.3, default_mode="hybrid")

# Nothing listens on port 1, so a connection there is refused at once.
NOTHING_LISTENS_HERE = "postgresql://nobody@127.0.0.1:1/nothing_test"

EVERY_TOOL = [
    ("search_guides", {"question": "mend a torn sail"}),
    ("read_guide", {"slug": "mend-a-torn-sail"}),
    ("list_guides", {}),
]


def call(service: Service, name: str, arguments: dict | None = None):
    """Call one tool as an assistant would. A tool error is returned, not raised."""

    async def run():
        async with Client(build_server(service)) as client:
            return await client.call_tool(name, arguments or {}, raise_on_error=False)

    return asyncio.run(run())


@pytest.fixture
def service(conn: psycopg.Connection, database_url: str) -> Service:
    """The real service on the test database, with the five guides ingested."""
    ingest(conn, load_snapshot(FIXTURE), FakeEmbedder())
    return Service(database_url, FakeEmbedder(), SETTINGS)


# ---------------------------------------------------------------------------
# The three tools on real data
# ---------------------------------------------------------------------------


def test_search_guides_finds_the_guide_the_question_is_about(service: Service):
    result = call(service, "search_guides", {"question": "mend a torn sail"})

    assert result.is_error is False
    found = result.structured_content
    assert found["reason"] is None
    assert found["mode"] == "hybrid"
    assert found["guides"][0]["slug"] == "mend-a-torn-sail"
    assert found["guides"][0]["passages"][0]["text"] != ""


def test_list_guides_lists_every_guide_once_in_id_order(service: Service):
    result = call(service, "list_guides")

    expected = [
        {"guide_id": guide.id, "slug": guide.slug, "title": guide.title, "url": guide.url}
        for guide in load_snapshot(FIXTURE)
    ]
    assert result.is_error is False
    assert result.structured_content == {"guides": expected}
    assert len(expected) == 5


def test_read_guide_returns_the_passages_in_reading_order(service: Service):
    result = call(service, "read_guide", {"slug": "mend-a-torn-sail"})

    assert result.is_error is False
    guide = result.structured_content
    positions = [passage["position"] for passage in guide["passages"]]
    assert positions == sorted(positions)
    assert positions[0] == 0
    assert all(passage["heading"] != "Related articles" for passage in guide["passages"])


def test_read_guide_with_an_unknown_slug_is_an_error(service: Service):
    result = call(service, "read_guide", {"slug": "no-such-guide"})

    assert result.is_error is True
    assert "list_guides" in result.content[0].text


@pytest.mark.parametrize(("name", "arguments"), EVERY_TOOL)
def test_every_tool_reports_a_fault_on_an_empty_table(
    conn: psycopg.Connection, database_url: str, name: str, arguments: dict
):
    # The "conn" fixture has emptied the table and nothing was ingested.
    empty = Service(database_url, FakeEmbedder(), SETTINGS)

    result = call(empty, name, arguments)

    assert result.is_error is True
    assert result.content[0].text == NOT_BUILT


@pytest.mark.parametrize(("name", "arguments"), EVERY_TOOL)
def test_every_tool_reports_a_fault_when_the_database_cannot_be_reached(name: str, arguments: dict):
    unreachable = Service(NOTHING_LISTENS_HERE, FakeEmbedder(), SETTINGS)

    result = call(unreachable, name, arguments)

    assert result.is_error is True
    assert result.content[0].text == UNREACHABLE


# ---------------------------------------------------------------------------
# Read-only, enforced by Postgres
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM passages",
        "UPDATE passages SET title = 'changed'",
        "DROP TABLE passages",
        "CREATE TABLE should_not_exist (n integer)",
    ],
)
def test_a_read_only_connection_is_refused_every_write(service: Service, database_url: str, statement: str):
    with connect(database_url, read_only=True) as read_only:
        before = read_only.execute("SELECT count(*) FROM passages").fetchone()[0]

        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            read_only.execute(statement)

        # Reading still works, and nothing was changed.
        assert read_only.execute("SELECT count(*) FROM passages").fetchone()[0] == before
        assert before > 0


def test_an_ordinary_connection_can_still_write(conn: psycopg.Connection, database_url: str):
    # Ingestion needs to write, so read-only must be something that is asked for.
    with connect(database_url) as ordinary:
        ordinary.execute("CREATE TEMPORARY TABLE scratch (n integer)")
        ordinary.execute("INSERT INTO scratch VALUES (1)")
        assert ordinary.execute("SELECT count(*) FROM scratch").fetchone()[0] == 1


def test_the_service_opens_every_connection_read_only(
    service: Service, monkeypatch: pytest.MonkeyPatch
):
    asked_for: list[bool] = []
    real_connect = service_module.connect

    def remembering_connect(url: str, read_only: bool = False) -> psycopg.Connection:
        asked_for.append(read_only)
        return real_connect(url, read_only=read_only)

    monkeypatch.setattr(service_module, "connect", remembering_connect)

    service.search("mend a torn sail", None)
    service.guide("mend-a-torn-sail")
    service.guides()
    service.health()

    assert asked_for == [True, True, True, True]


# ---------------------------------------------------------------------------
# The server as an assistant would start it
# ---------------------------------------------------------------------------


def test_the_server_works_as_a_child_process_over_stdio(service: Service, database_url: str):
    """Start the server the way an assistant does and call it over stdio.

    If the server wrote anything but protocol messages to its standard output,
    the client could not read the answers and this test would fail.
    """
    environment = dict(
        os.environ,
        TEST_DATABASE_URL=database_url,
        PYTHONPATH=os.pathsep.join(sys.path),
    )
    transport = StdioTransport(
        command=sys.executable, args=[str(STDIO_SERVER)], env=environment
    )

    async def run():
        async with Client(transport) as client:
            tools = await client.list_tools()
            listing = await client.call_tool("list_guides", {})
            found = await client.call_tool("search_guides", {"question": "mend a torn sail"})
            return tools, listing, found

    tools, listing, found = asyncio.run(run())

    assert sorted(tool.name for tool in tools) == ["list_guides", "read_guide", "search_guides"]
    assert len(listing.structured_content["guides"]) == 5
    assert found.structured_content["guides"][0]["slug"] == "mend-a-torn-sail"
