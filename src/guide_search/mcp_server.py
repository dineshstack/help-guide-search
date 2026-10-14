"""The MCP server: the guide search offered to AI assistants as three tools.

MCP (the Model Context Protocol) is a standard way to offer tools to an AI
assistant. An assistant that speaks MCP starts this file as a child process
and calls the tools itself:

- search_guides   the same search as GET /search
- read_guide      one whole guide, by its slug
- list_guides     every guide's id, slug, title and public page address

To start it by hand, from the project folder:

    uv run --env-file .env python -m guide_search.mcp_server

This file is a thin wrapper. It adds names, descriptions and error wording and
nothing else. The work is done by the same "backend" the HTTP routes use (the
Service in service.py), so the tools and the routes cannot give different
answers. The tests hand over a small stub instead.

Two rules shape the code:

- All three tools only read. They are marked read-only for the assistant, and
  the service opens its database connections read-only, so Postgres itself
  refuses a write.
- "We could not look" must never look like "we looked and found nothing". A
  database fault is reported to the assistant as an error, in plain words and
  with no address, user name or database message in it.

The assistant talks to this process through its standard input and output, so
nothing else may be written to standard output. Logs go to standard error.
"""

import logging
from typing import Protocol

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from guide_search.ranking import GuideDetail, GuideList, SearchResult, clean_question
from guide_search.search import MODES
from guide_search.store import DatabaseUnavailable, IndexNotReady

# Python's logging writes to standard error unless told otherwise, which is
# where this server's own messages belong.
logger = logging.getLogger(__name__)

SERVER_NAME = "help-guide-search"

INSTRUCTIONS = (
    "Read-only access to the public help guides of SAM, a workshop management "
    "product. search_guides finds the passages that best match a question. "
    "read_guide returns one whole guide. list_guides lists every guide. "
    "Nothing here can change anything."
)

# Hints for the assistant. They are advice only: nothing checks them. What
# actually prevents a write is the read-only database connection.
READ_ONLY = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

# What the assistant is told when the database is the problem. Deliberately
# free of any detail about the database itself.
UNREACHABLE = (
    "The guide index cannot be reached right now. This is a fault, not an "
    "empty result: the guides were not searched. Try again later."
)
NOT_BUILT = (
    "The guide index has not been built yet. This is a fault, not an empty "
    "result: there was nothing to search."
)


class Backend(Protocol):
    """What the tools need from the code behind them. Service provides it."""

    def search(self, question: str, mode: str | None) -> SearchResult: ...

    def guide(self, slug: str) -> GuideDetail | None: ...

    def guides(self) -> GuideList: ...


def database_fault(error: Exception) -> ToolError:
    """Turn one of the two database faults into the error the assistant sees."""
    # The real message may name a host, a port or a user. It is kept for the
    # person running the server and is not passed on.
    logger.warning("database fault: %s", error)
    if isinstance(error, IndexNotReady):
        return ToolError(NOT_BUILT)
    return ToolError(UNREACHABLE)


def build_server(backend: Backend) -> FastMCP:
    """Build the MCP server around the given backend."""
    # mask_error_details: if something unexpected goes wrong inside a tool, the
    # assistant is told only that the call failed. The detail goes to the log.
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS, mask_error_details=True)

    # The text under each function is what the assistant reads to decide when
    # to call the tool, so it is written for the assistant.

    @server.tool(annotations=READ_ONLY)
    def search_guides(question: str, mode: str | None = None) -> SearchResult:
        """Search the SAM help guides for the passages that best match a question.

        Use this first for any question about how to do something in SAM. Pass
        the question in the user's own words, in English or Arabic, in at most
        500 characters.

        The result lists up to three guides, best first. Each has its title,
        the address of its public page, and up to two passages of its text.
        A returned guide is the closest match found, not a promise that it
        answers the question: read the passages before you answer, and say so
        when they do not cover it. If the result has no guides and the reason
        "no_match", nothing close enough was found.

        mode is optional. Leave it out for the server's default, or pass
        "rerank" (meaning and keywords, then each passage checked against the
        question), "hybrid" (meaning and keywords together), "vector" (meaning
        only) or "keyword" (keywords only).
        """
        try:
            cleaned = clean_question(question)
        except ValueError:
            raise ToolError(
                "The question must be 1 to 500 characters long, not counting spaces around it."
            ) from None
        if mode is not None and mode not in MODES:
            raise ToolError("mode must be left out or be one of: " + ", ".join(MODES) + ".")
        try:
            return backend.search(cleaned, mode)
        except (DatabaseUnavailable, IndexNotReady) as error:
            raise database_fault(error) from None

    @server.tool(annotations=READ_ONLY)
    def read_guide(slug: str) -> GuideDetail:
        """Read one whole help guide: its title, public page address and passages in order.

        Use it when the passages from search_guides are not enough to answer.
        slug is the guide's short name as it appears in a search result or in
        list_guides, for example "sign-in-and-passwords".
        """
        try:
            detail = backend.guide(slug)
        except (DatabaseUnavailable, IndexNotReady) as error:
            raise database_fault(error) from None
        if detail is None:
            raise ToolError(
                f"No guide has the slug {slug!r}. Call list_guides to see the slugs that exist."
            )
        return detail

    @server.tool(annotations=READ_ONLY)
    def list_guides() -> GuideList:
        """List every help guide: id, slug, title and public page address, in id order.

        Use it to see what the help guides cover, or to find the slug to pass
        to read_guide.
        """
        try:
            return backend.guides()
        except (DatabaseUnavailable, IndexNotReady) as error:
            raise database_fault(error) from None

    return server


def main() -> None:
    """Start the server with the real model and talk to the assistant over stdio."""
    # Imported here and not at the top, because loading the model imports
    # torch, which takes seconds. Importing this module must stay quick.
    from guide_search import config
    from guide_search.cross_encoder import CrossEncoderReranker
    from guide_search.e5 import E5Embedder
    from guide_search.service import Service

    service = Service(
        config.database_url(),
        E5Embedder(),
        config.load_settings(),
        reranker=CrossEncoderReranker(),
    )
    # stdio is the default transport. The banner is switched off so that
    # nothing but the protocol's own messages can end up on standard output.
    build_server(service).run(show_banner=False)


if __name__ == "__main__":
    main()
