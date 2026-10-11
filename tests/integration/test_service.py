"""Tests for the real service behind the HTTP routes.

test_api.py tests the routes with the search replaced by a stub. Here nothing
is replaced except the model: the real routes call the real Service, which
searches a real Postgres (TEST_DATABASE_URL) holding the five invented guides
of tests/fixtures/five_guides.json. The embedder is the fake one.
"""

import os
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from guide_search import service as service_module
from guide_search.api import create_app
from guide_search.config import Settings, save_settings
from guide_search.embedding import FakeEmbedder
from guide_search.ingest import ingest
from guide_search.ranking import SearchResult
from guide_search.service import Service, build
from guide_search.snapshot import PUBLIC_PAGE_PREFIX, load_snapshot

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"

# The tests' own settings. The threshold suits the fake embedder's scores.
SETTINGS = Settings(threshold=0.3, default_mode="hybrid")

# Nothing listens on port 1, so a connection there is refused at once.
NOTHING_LISTENS_HERE = "postgresql://nobody@127.0.0.1:1/nothing_test"

# One address for each of the three routes.
EVERY_ROUTE = ["/search?q=mend+a+torn+sail", "/guides/mend-a-torn-sail", "/health"]


class OtherModelEmbedder(FakeEmbedder):
    """The fake embedder under another name, as if the model had been changed."""

    model_id = "fake-2"


class SlowEmbedder(FakeEmbedder):
    """The fake embedder, but slow, and it notices when two questions overlap."""

    def __init__(self) -> None:
        self.busy = False
        self.overlapped = False

    def embed_query(self, text: str) -> list[float]:
        if self.busy:
            self.overlapped = True
        self.busy = True
        time.sleep(0.05)
        self.busy = False
        return super().embed_query(text)


def ingest_fixture(conn: psycopg.Connection) -> None:
    ingest(conn, load_snapshot(FIXTURE), FakeEmbedder())


@pytest.fixture
def client(conn: psycopg.Connection, database_url: str) -> Iterator[TestClient]:
    """The real routes on a real Service, with the five guides ingested."""
    ingest_fixture(conn)
    service = Service(database_url, FakeEmbedder(), SETTINGS)
    # The "with" block starts the app and shuts it down again.
    with TestClient(create_app(service)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(conn: psycopg.Connection, database_url: str) -> Iterator[TestClient]:
    """The same, but nothing has been ingested: the table is there and empty."""
    service = Service(database_url, FakeEmbedder(), SETTINGS)
    with TestClient(create_app(service)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------


def test_search_returns_the_guide_the_question_is_about(client: TestClient):
    response = client.get("/search", params={"q": "replace a snapped spar"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "replace a snapped spar"
    assert body["reason"] is None
    first = body["guides"][0]
    assert first["guide_id"] == 3
    assert first["slug"] == "replace-a-snapped-spar"
    assert first["title"] == "Replace a snapped spar"
    assert first["url"] == PUBLIC_PAGE_PREFIX + "replace-a-snapped-spar"
    assert first["passages"][0]["heading"] == "Fit the new spar"
    assert first["passages"][0]["position"] == 0
    assert first["passages"][0]["text"].startswith("Slide the snapped spar")
    assert first["passages"][0]["similarity"] == body["best_similarity"]


def test_search_without_a_mode_uses_the_default_mode_of_the_settings(client: TestClient):
    response = client.get("/search", params={"q": "replace a snapped spar"})

    assert response.json()["mode"] == "hybrid"


def test_search_uses_the_mode_that_is_asked_for(client: TestClient):
    response = client.get("/search", params={"q": "replace a snapped spar", "mode": "keyword"})

    body = response.json()
    assert body["mode"] == "keyword"
    assert body["best_similarity"] is None
    assert body["guides"][0]["slug"] == "replace-a-snapped-spar"


def test_search_that_finds_nothing_good_enough_is_200_with_the_reason_no_match(client: TestClient):
    response = client.get("/search", params={"q": "do you sell bicycle bells"})

    assert response.status_code == 200
    body = response.json()
    assert body["reason"] == "no_match"
    assert body["guides"] == []
    assert 0 < body["best_similarity"] < SETTINGS.threshold


def test_search_refuses_by_the_threshold_of_the_settings(conn: psycopg.Connection, database_url: str):
    # The same question is answered with the tests' threshold of 0.3 (see the
    # first test). A service whose settings say 0.99 must refuse it.
    ingest_fixture(conn)
    strict = Service(database_url, FakeEmbedder(), Settings(threshold=0.99, default_mode="hybrid"))

    result = strict.search("replace a snapped spar", None)

    assert result.reason == "no_match"


def test_an_arabic_question_arrives_as_it_was_sent(client: TestClient):
    # "How can I send an SMS to the customer?"
    question = "كيف يمكنني إرسال SMS للعميل؟"

    response = client.get("/search", params={"q": question, "mode": "keyword"})

    body = response.json()
    assert body["query"] == question
    assert body["guides"][0]["slug"] == "send-a-pickup-reminder"


# ---------------------------------------------------------------------------
# GET /guides/{slug}
# ---------------------------------------------------------------------------


def test_guide_returns_the_title_the_url_and_the_passages_in_order(client: TestClient):
    response = client.get("/guides/mend-a-torn-sail")

    assert response.status_code == 200
    body = response.json()
    assert body["guide_id"] == 2
    assert body["slug"] == "mend-a-torn-sail"
    assert body["title"] == "Mend a torn sail"
    assert body["url"] == PUBLIC_PAGE_PREFIX + "mend-a-torn-sail"
    assert [(p["position"], p["heading"]) for p in body["passages"]] == [
        (0, ""),
        (1, "Before you start"),
        (2, "Patch the tear"),
        (3, "Troubleshooting"),
    ]
    assert [p["similarity"] for p in body["passages"]] == [None, None, None, None]


def test_guide_with_an_unknown_slug_is_404(client: TestClient):
    response = client.get("/guides/no-such-guide")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /health, and what every route does when search cannot look
# ---------------------------------------------------------------------------


def test_health_is_200_after_ingestion(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", EVERY_ROUTE)
def test_every_route_is_503_on_an_empty_table(empty_client: TestClient, path: str):
    # For /guides this matters most: an empty table must not look like
    # "there is no such guide" (404).
    response = empty_client.get(path)

    assert response.status_code == 503
    assert response.json() == {"error": "index_not_ready"}


@pytest.mark.parametrize("path", EVERY_ROUTE)
def test_every_route_is_503_when_the_table_holds_only_another_models_passages(
    conn: psycopg.Connection, database_url: str, path: str
):
    ingest_fixture(conn)  # stored under the model id "fake"
    service = Service(database_url, OtherModelEmbedder(), SETTINGS)

    with TestClient(create_app(service)) as test_client:
        response = test_client.get(path)

    assert response.status_code == 503
    assert response.json() == {"error": "index_not_ready"}


@pytest.mark.parametrize("path", EVERY_ROUTE)
def test_every_route_is_503_when_the_database_cannot_be_reached(path: str):
    service = Service(NOTHING_LISTENS_HERE, FakeEmbedder(), SETTINGS)

    with TestClient(create_app(service)) as test_client:
        response = test_client.get(path)

    assert response.status_code == 503
    assert response.json() == {"error": "database_unavailable"}


@pytest.mark.parametrize("path", EVERY_ROUTE)
def test_every_route_is_503_when_the_database_fails_in_the_middle_of_a_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
):
    """The connection opened fine, then the database went away."""

    def database_goes_away(*args, **kwargs):
        raise psycopg.OperationalError("server closed the connection unexpectedly")

    monkeypatch.setattr(service_module, "search_passages", database_goes_away)
    monkeypatch.setattr(service_module, "count_passages", database_goes_away)

    response = client.get(path)

    assert response.status_code == 503
    assert response.json() == {"error": "database_unavailable"}


# ---------------------------------------------------------------------------
# How the service uses the database and the embedder
# ---------------------------------------------------------------------------


def test_every_call_opens_one_connection_and_closes_it(
    conn: psycopg.Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
):
    ingest_fixture(conn)
    opened: list[psycopg.Connection] = []
    real_connect = service_module.connect

    def remembering_connect(url: str, read_only: bool = False) -> psycopg.Connection:
        connection = real_connect(url, read_only=read_only)
        opened.append(connection)
        return connection

    # monkeypatch swaps the function for the length of this test only.
    monkeypatch.setattr(service_module, "connect", remembering_connect)
    service = Service(database_url, FakeEmbedder(), SETTINGS)

    service.search("mend a torn sail", None)
    service.guide("mend-a-torn-sail")
    service.health()

    assert len(opened) == 3
    assert [connection.closed for connection in opened] == [True, True, True]


def test_the_embedder_is_never_used_by_two_searches_at_the_same_time(
    conn: psycopg.Connection, database_url: str
):
    ingest_fixture(conn)
    embedder = SlowEmbedder()
    service = Service(database_url, embedder, SETTINGS)

    def one_search(number: int) -> SearchResult:
        return service.search("mend a torn sail", "vector")

    # Four searches are started together, each in its own thread, the way the
    # web server runs requests that arrive together.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one_search, range(4)))

    assert [result.guides[0].slug for result in results] == ["mend-a-torn-sail"] * 4
    assert embedder.overlapped is False


# ---------------------------------------------------------------------------
# build(), the function the web server calls
# ---------------------------------------------------------------------------


def test_build_puts_the_database_address_the_settings_file_and_the_routes_together(
    conn: psycopg.Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    ingest_fixture(conn)
    # build() imports the real model from guide_search.e5. Put a stand-in
    # module under that name, so that the fake embedder is what it finds.
    stand_in = types.ModuleType("guide_search.e5")
    stand_in.E5Embedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "guide_search.e5", stand_in)
    # build() reads DATABASE_URL, and settings.toml in the current folder.
    monkeypatch.setenv("DATABASE_URL", database_url)
    # Without a key the answer agent is left out, which is what this test is about.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    save_settings(Settings(threshold=0.3, default_mode="keyword"), tmp_path / "settings.toml")
    monkeypatch.chdir(tmp_path)

    with TestClient(build()) as test_client:
        health = test_client.get("/health")
        found = test_client.get("/search", params={"q": "replace a snapped spar"})

    assert health.status_code == 200
    assert found.json()["mode"] == "keyword"
    assert found.json()["guides"][0]["slug"] == "replace-a-snapped-spar"


def test_importing_the_service_module_does_not_load_torch():
    # The real model is imported inside build(). A fresh Python process is
    # used because another test in the same run may already have loaded torch.
    code = (
        "import sys\n"
        "import guide_search.service\n"
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
