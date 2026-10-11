"""The answer agent on the real service and a real Postgres.

The model is replaced by a stand-in that writes a prepared draft, and the
embedding model by the fake embedder. Everything else is real: the routes, the
service, the search and the test database with the five invented guides.
"""

from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from guide_search.api import create_app
from guide_search.checks import Draft
from guide_search.config import Settings
from guide_search.drafting import ModelCall
from guide_search.embedding import FakeEmbedder
from guide_search.ingest import ingest
from guide_search.service import Service
from guide_search.snapshot import load_snapshot

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"
SETTINGS = Settings(threshold=0.3, default_mode="hybrid")
NOTHING_LISTENS_HERE = "postgresql://nobody@127.0.0.1:1/nothing_test"
QUESTION = {"question": "mend a torn sail"}


class CitesTheFirstGuide:
    """Stands in for the model: answers by citing the first guide search returned."""

    def __init__(self) -> None:
        self.seen_guides: list[list[int]] = []

    def draft(self, question, guides, previous, failure):
        self.seen_guides.append([guide.guide_id for guide in guides])
        ids = (guides[0].guide_id,) if guides else ()
        draft = Draft(decision="answer", reply="Patch the tear and let it dry.", cited_guide_ids=ids)
        call = ModelCall(
            purpose="draft" if previous is None else "rewrite",
            model="stand-in",
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            stop_reason="end_turn",
            seconds=0.0,
        )
        return draft, call


def ingest_fixture(conn: psycopg.Connection) -> None:
    ingest(conn, load_snapshot(FIXTURE), FakeEmbedder())


def test_answer_searches_the_real_index_and_returns_the_cited_guide_as_its_source(
    conn: psycopg.Connection, database_url: str, tmp_path: Path
):
    ingest_fixture(conn)
    drafter = CitesTheFirstGuide()
    log = tmp_path / "usage.jsonl"
    service = Service(database_url, FakeEmbedder(), SETTINGS, drafter, log)

    with TestClient(create_app(service)) as client:
        response = client.post("/answer", json=QUESTION)

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "answer"
    assert result["sources"][0]["slug"] == "mend-a-torn-sail"
    assert result["check"] == {"attempts": 1, "failures": []}
    assert [call["purpose"] for call in result["usage"]] == ["draft"]
    # The model was shown what GET /search returns for the same question.
    with TestClient(create_app(service)) as client:
        searched = client.get("/search", params={"q": "mend a torn sail"}).json()
    assert drafter.seen_guides[0] == [guide["guide_id"] for guide in searched["guides"]]
    assert len(log.read_text().splitlines()) == 1


def test_answer_on_an_empty_table_is_503_and_the_model_is_never_asked(
    conn: psycopg.Connection, database_url: str
):
    drafter = CitesTheFirstGuide()
    service = Service(database_url, FakeEmbedder(), SETTINGS, drafter)

    with TestClient(create_app(service)) as client:
        response = client.post("/answer", json=QUESTION)

    assert response.status_code == 503
    assert response.json() == {"error": "index_not_ready"}
    assert drafter.seen_guides == []


def test_answer_with_an_unreachable_database_is_503_and_not_a_hand_over():
    drafter = CitesTheFirstGuide()
    service = Service(NOTHING_LISTENS_HERE, FakeEmbedder(), SETTINGS, drafter)

    with TestClient(create_app(service)) as client:
        response = client.post("/answer", json=QUESTION)

    assert response.status_code == 503
    assert response.json() == {"error": "database_unavailable"}
    assert drafter.seen_guides == []


def test_answer_without_a_model_is_503_while_search_still_works(
    conn: psycopg.Connection, database_url: str
):
    ingest_fixture(conn)
    service = Service(database_url, FakeEmbedder(), SETTINGS)  # no drafter

    with TestClient(create_app(service)) as client:
        answered = client.post("/answer", json=QUESTION)
        searched = client.get("/search", params={"q": "mend a torn sail"})

    assert answered.status_code == 503
    assert answered.json() == {"error": "agent_not_configured"}
    assert searched.status_code == 200
