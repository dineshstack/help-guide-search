"""The HTTP routes: three read-only addresses that answer with JSON.

- GET /search?q=...&mode=...   the search result for a question
- GET /guides/{slug}           one guide's title, URL and passages, in order
- GET /health                  whether search is ready to answer

This file knows about HTTP and nothing else. It opens no database connection
and loads no model. The real work is done by a "backend" object that is handed
to create_app(). The real backend is built in service.py. The tests hand over
a small stub instead, which is why the routes can be tested with no database.

One rule matters more than the rest. When the database cannot be reached, or
holds no passages, every route answers 503. It never answers 200 with an empty
result, because "we could not look" and "we looked and found nothing" must not
look the same to the caller.
"""

from dataclasses import asdict
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from guide_search.ranking import MAX_QUESTION_CHARS, GuideDetail, SearchResult, clean_question
from guide_search.search import MODES
from guide_search.store import DatabaseUnavailable, IndexNotReady


class Backend(Protocol):
    """What the routes need from the code behind them."""

    def search(self, question: str, mode: str | None) -> SearchResult: ...

    def guide(self, slug: str) -> GuideDetail | None: ...

    # Returns nothing when healthy. Raises DatabaseUnavailable or IndexNotReady
    # when not.
    def health(self) -> None: ...


def service_unavailable(request: Request, error: Exception) -> JSONResponse:
    """Answer 503 and name which of the two database problems it was."""
    # FastAPI hands every error handler the request as well. It is not needed
    # here, because the answer depends only on the kind of error.
    if isinstance(error, IndexNotReady):
        name = "index_not_ready"
    else:
        name = "database_unavailable"
    return JSONResponse(status_code=503, content={"error": name})


def create_app(backend: Backend) -> FastAPI:
    """Build the web application around the given backend."""
    app = FastAPI(title="Guide search")

    # One handler serves every route. A route does not catch these two errors
    # itself: it lets them rise, and FastAPI calls the handler.
    app.add_exception_handler(DatabaseUnavailable, service_unavailable)
    app.add_exception_handler(IndexNotReady, service_unavailable)

    # The routes are plain "def", not "async def". Searching waits for the
    # database and the model, and FastAPI runs a plain function in a worker
    # thread, so one slow search does not hold up every other request.

    @app.get("/search")
    def search_route(q: str, mode: str | None = None) -> dict[str, Any]:
        # The rule for a valid question lives in one place, clean_question,
        # which the MCP tool uses as well.
        try:
            question = clean_question(q)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"q must be 1 to {MAX_QUESTION_CHARS} characters",
            ) from None
        # No mode is fine: the backend then uses the default from the settings.
        if mode is not None and mode not in MODES:
            raise HTTPException(
                status_code=422,
                detail="mode must be one of: " + ", ".join(MODES),
            )
        return asdict(backend.search(question, mode))

    @app.get("/guides/{slug}")
    def guide_route(slug: str) -> dict[str, Any]:
        # None means the database answered and has no such guide. A database
        # that could not answer raises instead, and becomes 503, not 404.
        detail = backend.guide(slug)
        if detail is None:
            raise HTTPException(status_code=404, detail="no guide has this slug")
        return asdict(detail)

    @app.get("/health")
    def health_route() -> dict[str, str]:
        # health() returns nothing when all is well and raises when it is not.
        backend.health()
        return {"status": "ok"}

    return app
