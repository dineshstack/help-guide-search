"""Tests for the three HTTP routes.

The real search is replaced by a small stub, so no database and no model are
needed. What is tested is the HTTP layer alone: the JSON that goes out, which
requests are turned away with 422, and that a database problem becomes 503 and
is never dressed up as an empty result.

All guides and passages are invented.
"""

import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from guide_search.api import create_app
from guide_search.ranking import GuideDetail, GuideHit, PassageHit, SearchResult
from guide_search.store import DatabaseUnavailable, IndexNotReady


def example_result() -> SearchResult:
    """A search result with one guide and one passage."""
    return SearchResult(
        query="how do I print a job card",
        mode="hybrid",
        reason=None,
        best_similarity=0.88,
        guides=[
            GuideHit(
                guide_id=12,
                slug="print-a-job-card",
                title="Print a job card",
                url="https://example.test/help/print-a-job-card",
                score=0.0328,
                passages=[
                    PassageHit(
                        heading="Choose a printer",
                        position=2,
                        text="Open the job card and press Print.",
                        similarity=0.88,
                    )
                ],
            )
        ],
    )


def passage(heading: str, position: int, text: str) -> PassageHit:
    """A passage with no similarity, as in a whole guide or a keyword result."""
    return PassageHit(heading=heading, position=position, text=text, similarity=None)


def example_guide() -> GuideDetail:
    """A whole guide with three passages, as the backend would return it."""
    return GuideDetail(
        guide_id=12,
        slug="print-a-job-card",
        title="Print a job card",
        url="https://example.test/help/print-a-job-card",
        passages=[
            passage("", 0, "A job card lists the work."),
            passage("Before you start", 1, "Add a printer."),
            passage("Choose a printer", 2, "Press Print."),
        ],
    )


class StubBackend:
    """Stands in for the real search.

    It returns what the test gave it, or raises the error the test gave it, and
    remembers how it was called.
    """

    def __init__(
        self,
        result: SearchResult | None = None,
        guide: GuideDetail | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else example_result()
        self.guide_detail = guide
        self.error = error
        self.search_calls: list[tuple[str, str | None]] = []
        self.guide_calls: list[str] = []

    def search(self, question: str, mode: str | None) -> SearchResult:
        self.search_calls.append((question, mode))
        if self.error is not None:
            raise self.error
        return self.result

    def guide(self, slug: str) -> GuideDetail | None:
        self.guide_calls.append(slug)
        if self.error is not None:
            raise self.error
        return self.guide_detail

    def health(self) -> None:
        if self.error is not None:
            raise self.error


def get(backend: StubBackend, path: str, params: dict[str, str] | None = None):
    """Send one GET request to an app built around the given backend."""
    # The "with" block starts the app and shuts it down again, which also stops
    # the background thread the test client uses.
    with TestClient(create_app(backend)) as client:
        return client.get(path, params=params)


# ---------------------------------------------------------------------------
# GET /search: what comes back
# ---------------------------------------------------------------------------


def test_search_returns_200_with_the_documented_json_shape():
    response = get(StubBackend(), "/search", {"q": "how do I print a job card"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "how do I print a job card",
        "mode": "hybrid",
        "reason": None,
        "best_similarity": 0.88,
        "guides": [
            {
                "guide_id": 12,
                "slug": "print-a-job-card",
                "title": "Print a job card",
                "url": "https://example.test/help/print-a-job-card",
                "score": 0.0328,
                "passages": [
                    {
                        "heading": "Choose a printer",
                        "position": 2,
                        "text": "Open the job card and press Print.",
                        "similarity": 0.88,
                        "rerank_score": None,
                    }
                ],
            }
        ],
        "best_rerank_score": None,
    }


def test_search_no_match_returns_200_with_no_guides_and_the_reason():
    no_match = SearchResult(
        query="do you sell tyres",
        mode="hybrid",
        reason="no_match",
        best_similarity=0.41,
        guides=[],
    )

    response = get(StubBackend(result=no_match), "/search", {"q": "do you sell tyres"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "do you sell tyres",
        "mode": "hybrid",
        "reason": "no_match",
        "best_similarity": 0.41,
        "guides": [],
        "best_rerank_score": None,
    }


def test_search_keyword_result_has_null_similarities():
    # Keyword mode never embeds the question, so there is no similarity to give.
    keyword_result = SearchResult(
        query="print",
        mode="keyword",
        reason=None,
        best_similarity=None,
        guides=[
            GuideHit(
                guide_id=12,
                slug="print-a-job-card",
                title="Print a job card",
                url="https://example.test/help/print-a-job-card",
                score=0.2,
                passages=[passage("", 0, "Press Print.")],
            )
        ],
    )

    backend = StubBackend(result=keyword_result)
    response = get(backend, "/search", {"q": "print", "mode": "keyword"})

    body = response.json()
    assert response.status_code == 200
    assert body["best_similarity"] is None
    assert body["guides"][0]["passages"][0]["similarity"] is None


# ---------------------------------------------------------------------------
# GET /search: requests that are turned away
# ---------------------------------------------------------------------------


def test_search_without_q_is_422():
    backend = StubBackend()
    response = get(backend, "/search")
    assert response.status_code == 422
    assert backend.search_calls == []


@pytest.mark.parametrize(
    "blank",
    ["", " ", "     ", "\t\n"],
    ids=["empty", "one space", "spaces", "tab and newline"],
)
def test_search_with_a_blank_q_is_422(blank: str):
    backend = StubBackend()
    response = get(backend, "/search", {"q": blank})
    assert response.status_code == 422
    assert backend.search_calls == []


def test_search_with_q_of_501_characters_is_422():
    backend = StubBackend()
    response = get(backend, "/search", {"q": "a" * 501})
    assert response.status_code == 422
    assert backend.search_calls == []


def test_search_with_q_of_500_characters_is_accepted():
    backend = StubBackend()
    response = get(backend, "/search", {"q": "a" * 500})
    assert response.status_code == 200
    assert backend.search_calls == [("a" * 500, None)]


def test_search_measures_q_after_trimming():
    # 500 characters with spaces around them is still 500 characters.
    backend = StubBackend()
    response = get(backend, "/search", {"q": "   " + "a" * 500 + "   "})
    assert response.status_code == 200
    assert backend.search_calls == [("a" * 500, None)]


def test_search_counts_characters_not_bytes():
    # An Arabic letter is one character but two bytes. 500 of them must pass.
    backend = StubBackend()
    response = get(backend, "/search", {"q": "س" * 500})
    assert response.status_code == 200
    assert backend.search_calls == [("س" * 500, None)]


@pytest.mark.parametrize("mode", ["fuzzy", "HYBRID", "vector,keyword"])
def test_search_with_an_unknown_mode_is_422(mode: str):
    backend = StubBackend()
    response = get(backend, "/search", {"q": "print a job card", "mode": mode})
    assert response.status_code == 422
    assert backend.search_calls == []


# ---------------------------------------------------------------------------
# GET /search: what is handed to the backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_passes_each_known_mode_to_the_backend(mode: str):
    backend = StubBackend()
    response = get(backend, "/search", {"q": "print a job card", "mode": mode})
    assert response.status_code == 200
    assert backend.search_calls == [("print a job card", mode)]


def test_search_without_mode_passes_none_to_the_backend():
    # None means "use the default mode", which the backend knows and the
    # routes do not.
    backend = StubBackend()
    get(backend, "/search", {"q": "print a job card"})
    assert backend.search_calls == [("print a job card", None)]


def test_search_passes_q_to_the_backend_trimmed():
    backend = StubBackend()
    get(backend, "/search", {"q": "  print a job card \n"})
    assert backend.search_calls == [("print a job card", None)]


# ---------------------------------------------------------------------------
# GET /search: a database problem is an error, not "no match"
# ---------------------------------------------------------------------------


def test_search_database_unavailable_is_503_and_not_an_empty_result():
    backend = StubBackend(error=DatabaseUnavailable("cannot connect"))

    response = get(backend, "/search", {"q": "print a job card"})

    assert response.status_code == 503
    assert response.json() == {"error": "database_unavailable"}
    assert "guides" not in response.json()


def test_search_index_not_ready_is_503():
    backend = StubBackend(error=IndexNotReady("no passages"))

    response = get(backend, "/search", {"q": "print a job card"})

    assert response.status_code == 503
    assert response.json() == {"error": "index_not_ready"}


# ---------------------------------------------------------------------------
# GET /guides/{slug}
# ---------------------------------------------------------------------------


def test_guide_returns_its_title_url_and_passages_in_order():
    backend = StubBackend(guide=example_guide())

    response = get(backend, "/guides/print-a-job-card")

    assert response.status_code == 200
    assert response.json() == {
        "guide_id": 12,
        "slug": "print-a-job-card",
        "title": "Print a job card",
        "url": "https://example.test/help/print-a-job-card",
        "passages": [
            {
                "heading": "",
                "position": 0,
                "text": "A job card lists the work.",
                "similarity": None,
                "rerank_score": None,
            },
            {
                "heading": "Before you start",
                "position": 1,
                "text": "Add a printer.",
                "similarity": None,
                "rerank_score": None,
            },
            {
                "heading": "Choose a printer",
                "position": 2,
                "text": "Press Print.",
                "similarity": None,
                "rerank_score": None,
            },
        ],
    }


def test_guide_asks_the_backend_for_the_slug_in_the_address():
    backend = StubBackend(guide=example_guide())
    get(backend, "/guides/print-a-job-card")
    assert backend.guide_calls == ["print-a-job-card"]


def test_guide_with_an_unknown_slug_is_404():
    # The backend answered, and its answer was "there is no such guide".
    backend = StubBackend(guide=None)
    response = get(backend, "/guides/no-such-guide")
    assert response.status_code == 404


def test_guide_database_unavailable_is_503_and_not_404():
    backend = StubBackend(error=DatabaseUnavailable("cannot connect"))

    response = get(backend, "/guides/print-a-job-card")

    assert response.status_code == 503
    assert response.json() == {"error": "database_unavailable"}


def test_guide_index_not_ready_is_503_and_not_404():
    backend = StubBackend(error=IndexNotReady("no passages"))

    response = get(backend, "/guides/print-a-job-card")

    assert response.status_code == 503
    assert response.json() == {"error": "index_not_ready"}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_is_200_with_status_ok():
    response = get(StubBackend(), "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_503_when_the_database_is_unavailable():
    backend = StubBackend(error=DatabaseUnavailable("cannot connect"))
    response = get(backend, "/health")
    assert response.status_code == 503
    assert response.json() == {"error": "database_unavailable"}


def test_health_is_503_when_the_index_is_not_ready():
    backend = StubBackend(error=IndexNotReady("no passages"))
    response = get(backend, "/health")
    assert response.status_code == 503
    assert response.json() == {"error": "index_not_ready"}


# ---------------------------------------------------------------------------
# The routes only read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/search?q=print", "/guides/print-a-job-card", "/health"])
def test_every_route_refuses_a_request_that_is_not_a_get(path: str):
    backend = StubBackend(guide=example_guide())

    with TestClient(create_app(backend)) as client:
        response = client.post(path)

    assert response.status_code == 405
    assert backend.search_calls == []
    assert backend.guide_calls == []


# ---------------------------------------------------------------------------
# Importing the routes must stay cheap
# ---------------------------------------------------------------------------


def test_importing_the_api_module_does_not_load_torch():
    # torch takes seconds to import and is only needed to embed text. A fresh
    # Python process is used because another test in the same run may already
    # have loaded torch into this one.
    code = (
        "import sys\n"
        "import guide_search.api\n"
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
