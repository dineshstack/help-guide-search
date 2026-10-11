"""Tests for the MCP tools, with the service replaced by a stub.

The tools are called "in process": the test builds the server, connects
FastMCP's own client straight to it, and calls a tool the way an assistant
would. No child process, no database and no model are involved.

The client is asynchronous, so each call is wrapped in asyncio.run(), which
runs one such call to its end. That keeps the tests themselves ordinary
functions.
"""

import asyncio
import contextlib
import io
import os
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest
from fastmcp import Client

from guide_search import mcp_server
from guide_search.config import Settings
from guide_search.embedding import FakeEmbedder
from guide_search.mcp_server import NOT_BUILT, UNREACHABLE, build_server
from guide_search.ranking import (
    GuideDetail,
    GuideHit,
    GuideList,
    GuideSummary,
    PassageHit,
    SearchResult,
)
from guide_search.service import Service
from guide_search.store import DatabaseUnavailable, IndexNotReady

REPOSITORY = Path(__file__).resolve().parents[2]
PAGE = "https://support.orions360.com/help/en/autoserv-pro/"

FOUND = SearchResult(
    query="how do I mend a sail",
    mode="hybrid",
    reason=None,
    best_similarity=0.91,
    guides=[
        GuideHit(
            guide_id=3,
            slug="mend-a-sail",
            title="Mend a sail",
            url=PAGE + "mend-a-sail",
            score=0.0328,
            passages=[PassageHit(heading="Patch it", position=1, text="Use a patch.", similarity=0.91)],
        )
    ],
)
NOTHING = SearchResult(query="what is the weather", mode="hybrid", reason="no_match", best_similarity=0.4, guides=[])
DETAIL = GuideDetail(
    guide_id=3,
    slug="mend-a-sail",
    title="Mend a sail",
    url=PAGE + "mend-a-sail",
    passages=[
        PassageHit(heading="", position=0, text="A torn sail can be mended.", similarity=None),
        PassageHit(heading="Patch it", position=1, text="Use a patch.", similarity=None),
    ],
)
LISTING = GuideList(
    guides=[
        GuideSummary(guide_id=2, slug="fly-a-kite", title="Fly a kite", url=PAGE + "fly-a-kite"),
        GuideSummary(guide_id=3, slug="mend-a-sail", title="Mend a sail", url=PAGE + "mend-a-sail"),
    ]
)

# One call for each tool, used by the tests that must hold for all three.
EVERY_TOOL = [
    ("search_guides", {"question": "how do I mend a sail"}),
    ("read_guide", {"slug": "mend-a-sail"}),
    ("list_guides", {}),
]

# An error text of the kind a database driver produces. None of it may reach
# the assistant.
PRIVATE_DETAIL = "connection to server at 10.1.2.3, port 5432 failed for user admin"


class StubBackend:
    """Stands in for the service: returns what it is told to, or raises it."""

    def __init__(self) -> None:
        self.result: SearchResult = FOUND
        self.detail: GuideDetail | None = DETAIL
        self.listing: GuideList = LISTING
        self.error: Exception | None = None
        self.searches: list[tuple[str, str | None]] = []
        self.slugs: list[str] = []

    def search(self, question: str, mode: str | None) -> SearchResult:
        self.searches.append((question, mode))
        if self.error is not None:
            raise self.error
        return self.result

    def guide(self, slug: str) -> GuideDetail | None:
        self.slugs.append(slug)
        if self.error is not None:
            raise self.error
        return self.detail

    def guides(self) -> GuideList:
        if self.error is not None:
            raise self.error
        return self.listing


def call(backend: StubBackend, name: str, arguments: dict | None = None):
    """Call one tool as an assistant would. A tool error is returned, not raised."""

    async def run():
        async with Client(build_server(backend)) as client:
            return await client.call_tool(name, arguments or {}, raise_on_error=False)

    return asyncio.run(run())


def list_tools(backend: StubBackend) -> list:
    async def run():
        async with Client(build_server(backend)) as client:
            return await client.list_tools()

    return asyncio.run(run())


def message(result) -> str:
    """The text the assistant is shown for a failed call."""
    return result.content[0].text


# ---------------------------------------------------------------------------
# What the assistant is offered
# ---------------------------------------------------------------------------


def test_there_are_exactly_three_tools():
    names = sorted(tool.name for tool in list_tools(StubBackend()))

    assert names == ["list_guides", "read_guide", "search_guides"]


def test_every_tool_is_marked_read_only():
    for tool in list_tools(StubBackend()):
        assert tool.annotations.read_only_hint is True, tool.name
        assert tool.annotations.idempotent_hint is True, tool.name
        assert tool.annotations.open_world_hint is False, tool.name


def test_every_tool_has_a_description_written_for_the_assistant():
    for tool in list_tools(StubBackend()):
        assert tool.description is not None, tool.name
        assert len(tool.description) > 80, tool.name


def test_search_guides_needs_a_question_and_takes_an_optional_mode():
    search = next(tool for tool in list_tools(StubBackend()) if tool.name == "search_guides")

    assert search.input_schema["required"] == ["question"]
    assert sorted(search.input_schema["properties"]) == ["mode", "question"]


def test_the_server_says_what_it_is_for():
    server = build_server(StubBackend())

    assert server.name == "help-guide-search"
    assert "Read-only" in server.instructions


# ---------------------------------------------------------------------------
# search_guides
# ---------------------------------------------------------------------------


def test_search_guides_returns_the_same_fields_as_the_http_route():
    result = call(StubBackend(), "search_guides", {"question": "how do I mend a sail"})

    assert result.is_error is False
    # asdict() is exactly what GET /search turns into JSON.
    assert result.structured_content == asdict(FOUND)


def test_search_guides_without_a_mode_lets_the_service_choose():
    backend = StubBackend()

    call(backend, "search_guides", {"question": "how do I mend a sail"})

    assert backend.searches == [("how do I mend a sail", None)]


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_guides_passes_on_the_mode_that_is_asked_for(mode: str):
    backend = StubBackend()

    call(backend, "search_guides", {"question": "how do I mend a sail", "mode": mode})

    assert backend.searches == [("how do I mend a sail", mode)]


def test_search_guides_trims_the_question():
    backend = StubBackend()

    call(backend, "search_guides", {"question": "  how do I mend a sail \n"})

    assert backend.searches == [("how do I mend a sail", None)]


def test_no_match_is_a_normal_result_and_not_an_error():
    backend = StubBackend()
    backend.result = NOTHING

    result = call(backend, "search_guides", {"question": "what is the weather"})

    assert result.is_error is False
    assert result.structured_content["guides"] == []
    assert result.structured_content["reason"] == "no_match"


@pytest.mark.parametrize("question", ["", "   ", "\n\t", "x" * 501])
def test_an_empty_or_over_long_question_is_an_error_and_is_not_searched(question: str):
    backend = StubBackend()

    result = call(backend, "search_guides", {"question": question})

    assert result.is_error is True
    assert "1 to 500 characters" in message(result)
    assert backend.searches == []


def test_a_question_of_exactly_500_characters_is_searched():
    backend = StubBackend()

    result = call(backend, "search_guides", {"question": "x" * 500})

    assert result.is_error is False
    assert backend.searches == [("x" * 500, None)]


def test_an_unknown_mode_is_an_error_that_names_the_modes():
    backend = StubBackend()

    result = call(backend, "search_guides", {"question": "how do I mend a sail", "mode": "fuzzy"})

    assert result.is_error is True
    assert "vector, keyword, hybrid" in message(result)
    assert backend.searches == []


# ---------------------------------------------------------------------------
# read_guide and list_guides
# ---------------------------------------------------------------------------


def test_read_guide_returns_the_guide_and_its_passages_in_order():
    backend = StubBackend()

    result = call(backend, "read_guide", {"slug": "mend-a-sail"})

    assert result.is_error is False
    assert result.structured_content == asdict(DETAIL)
    assert [passage["position"] for passage in result.structured_content["passages"]] == [0, 1]
    assert backend.slugs == ["mend-a-sail"]


def test_an_unknown_slug_is_an_error_that_points_to_list_guides():
    backend = StubBackend()
    backend.detail = None

    result = call(backend, "read_guide", {"slug": "no-such-guide"})

    assert result.is_error is True
    assert "no-such-guide" in message(result)
    assert "list_guides" in message(result)


def test_list_guides_returns_id_slug_title_and_page_address_in_id_order():
    result = call(StubBackend(), "list_guides")

    assert result.is_error is False
    assert result.structured_content == asdict(LISTING)
    assert [guide["guide_id"] for guide in result.structured_content["guides"]] == [2, 3]
    assert sorted(result.structured_content["guides"][0]) == ["guide_id", "slug", "title", "url"]


# ---------------------------------------------------------------------------
# Faults: "we could not look" must not look like "nothing was found"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "arguments"), EVERY_TOOL)
def test_an_unreachable_database_is_an_error_in_plain_words(name: str, arguments: dict):
    backend = StubBackend()
    backend.error = DatabaseUnavailable(PRIVATE_DETAIL)

    result = call(backend, name, arguments)

    assert result.is_error is True
    assert message(result) == UNREACHABLE
    assert result.structured_content is None


@pytest.mark.parametrize(("name", "arguments"), EVERY_TOOL)
def test_an_index_that_is_not_built_is_an_error_in_plain_words(name: str, arguments: dict):
    backend = StubBackend()
    backend.error = IndexNotReady(PRIVATE_DETAIL)

    result = call(backend, name, arguments)

    assert result.is_error is True
    assert message(result) == NOT_BUILT
    assert result.structured_content is None


def test_the_two_fault_messages_say_it_is_a_fault_and_not_an_empty_result():
    for text in (UNREACHABLE, NOT_BUILT):
        assert "fault, not an empty result" in text


@pytest.mark.parametrize(("name", "arguments"), EVERY_TOOL)
def test_an_unexpected_error_reaches_the_assistant_without_its_detail(name: str, arguments: dict):
    backend = StubBackend()
    backend.error = RuntimeError(PRIVATE_DETAIL)

    result = call(backend, name, arguments)

    assert result.is_error is True
    for private in ("10.1.2.3", "5432", "admin"):
        assert private not in message(result)


# ---------------------------------------------------------------------------
# The process the assistant starts
# ---------------------------------------------------------------------------


def test_calling_the_tools_writes_nothing_to_standard_output():
    """Standard output carries the protocol, so the tools must stay silent on it."""
    backend = StubBackend()
    captured = io.StringIO()

    with contextlib.redirect_stdout(captured):
        for name, arguments in EVERY_TOOL:
            call(backend, name, arguments)
        backend.error = DatabaseUnavailable(PRIVATE_DETAIL)
        call(backend, "list_guides")

    assert captured.getvalue() == ""


def test_importing_the_module_does_not_load_the_model():
    """Loading the model takes seconds, so it must wait until main() runs."""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    code = "import sys, guide_search.mcp_server; print('torch' in sys.modules)"

    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=True
    )

    assert done.stdout.strip() == "False"


def test_main_builds_the_real_service_and_serves_over_stdio(monkeypatch: pytest.MonkeyPatch):
    # A stand-in for the module that loads the real model, so main() can run
    # here without it.
    stand_in = types.ModuleType("guide_search.e5")
    stand_in.E5Embedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "guide_search.e5", stand_in)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")
    monkeypatch.chdir(REPOSITORY)  # settings.toml is read from the project folder
    seen: dict = {}

    class RecordingServer:
        def run(self, **options) -> None:
            seen["options"] = options

    def recording_build_server(backend):
        seen["backend"] = backend
        return RecordingServer()

    monkeypatch.setattr(mcp_server, "build_server", recording_build_server)

    mcp_server.main()

    assert isinstance(seen["backend"], Service)
    assert seen["backend"].database_url == "postgresql://nobody@127.0.0.1:1/nothing"
    assert isinstance(seen["backend"].embedder, FakeEmbedder)
    assert isinstance(seen["backend"].settings, Settings)
    # No transport is named, which means stdio, and the banner is off.
    assert seen["options"] == {"show_banner": False}
