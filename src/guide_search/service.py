"""The service: the real code behind the three HTTP routes and the MCP tools.

api.py knows about HTTP and nothing else. It asks a "backend" for answers.
The Service class in this file is the real backend: it holds the database
address, the embedding model and the settings, and answers each request with
the search code.

To run the web server:

    uv run --env-file .env uvicorn guide_search.service:build --factory

`--factory` tells uvicorn that `build` is a function to call, not the app
itself. The model is loaded once, inside build(), when the server starts.
"""

import os
import threading
from pathlib import Path

import psycopg
from fastapi import FastAPI

from guide_search import config
from guide_search.agent import DEFAULT_USAGE_LOG, Agent, AgentNotConfigured, AgentResult
from guide_search.api import create_app
from guide_search.drafting import Drafter
from guide_search.embedding import Embedder
from guide_search.ranking import GuideDetail, GuideList, SearchResult
from guide_search.search import list_guides, read_guide
from guide_search.search import search as search_passages
from guide_search.store import DatabaseUnavailable, connect, count_passages


class Service:
    """Answers the three routes. This is the real `Backend` of api.py."""

    def __init__(
        self,
        database_url: str,
        embedder: Embedder,
        settings: config.Settings,
        drafter: Drafter | None = None,
        usage_log: Path | None = None,
    ) -> None:
        self.database_url = database_url
        self.embedder = embedder
        self.settings = settings
        # The answer agent is optional. Without a model for it, search, the
        # guide routes and the MCP tools all still work.
        self.agent: Agent | None = None
        if drafter is not None:
            self.agent = Agent(self._search_for_agent, drafter, usage_log)
        # The web server runs requests that arrive together in separate
        # threads, and they all share the one embedder. A model is not safe to
        # use from two threads at once, so searches take turns.
        self.search_lock = threading.Lock()

    # Every method opens its own connection and closes it again: "with
    # connect(...)" closes the connection when the block ends, also after an
    # error. If the database cannot be reached, connect raises
    # DatabaseUnavailable, which the routes turn into a 503. A database that
    # goes away in the middle of a request raises psycopg.OperationalError
    # instead; each method turns that into DatabaseUnavailable too, so the
    # caller never sees a bare server error for it.
    #
    # Every connection is opened read-only: the service never writes, and
    # Postgres then refuses a write even if a mistake in this code tried one.

    def search(self, question: str, mode: str | None) -> SearchResult:
        if mode is None:
            mode = self.settings.default_mode
        with self.search_lock:
            try:
                with connect(self.database_url, read_only=True) as conn:
                    return search_passages(conn, self.embedder, question, mode, self.settings.threshold)
            except psycopg.OperationalError as error:
                raise DatabaseUnavailable(f"the database stopped answering: {error}") from error

    def guide(self, slug: str) -> GuideDetail | None:
        try:
            with connect(self.database_url, read_only=True) as conn:
                # An empty table has no guide for any slug. That must be reported
                # as "the index is not ready" and not as "no such guide", so the
                # table is checked first. count_passages raises IndexNotReady.
                count_passages(conn, self.embedder.model_id)
                return read_guide(conn, slug)
        except psycopg.OperationalError as error:
            raise DatabaseUnavailable(f"the database stopped answering: {error}") from error

    def _search_for_agent(self, question: str) -> SearchResult:
        # The agent searches exactly as GET /search does with no mode given.
        return self.search(question, None)

    def answer(self, question: str) -> AgentResult:
        if self.agent is None:
            raise AgentNotConfigured(
                "The answer agent has no model to use. Set ANTHROPIC_API_KEY and start again."
            )
        return self.agent.answer(question)

    def guides(self) -> GuideList:
        try:
            with connect(self.database_url, read_only=True) as conn:
                # As in guide(): an empty table is "not ready", not "no guides".
                count_passages(conn, self.embedder.model_id)
                return GuideList(guides=list_guides(conn))
        except psycopg.OperationalError as error:
            raise DatabaseUnavailable(f"the database stopped answering: {error}") from error

    def health(self) -> None:
        try:
            with connect(self.database_url, read_only=True) as conn:
                count_passages(conn, self.embedder.model_id)
        except psycopg.OperationalError as error:
            raise DatabaseUnavailable(f"the database stopped answering: {error}") from error


def real_service() -> Service:
    """Build the service with the real embedding model, and the answer agent if it has a key."""
    # Imported here and not at the top, because this import loads torch, which
    # takes seconds. Importing this module must stay quick.
    from guide_search.e5 import E5Embedder

    drafter = None
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        from guide_search.drafting import ClaudeDrafter, load_agent_settings, real_client

        drafter = ClaudeDrafter(real_client(), load_agent_settings())
    # USAGE_LOG lets a container, whose program folder cannot be written to,
    # put the token records somewhere that can.
    usage_log = Path(os.environ.get("USAGE_LOG", "").strip() or DEFAULT_USAGE_LOG)
    return Service(config.database_url(), E5Embedder(), config.load_settings(), drafter, usage_log)


def build() -> FastAPI:
    """Build the web application with the real model. uvicorn calls this once."""
    return create_app(real_service())
