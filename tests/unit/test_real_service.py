"""Tests for how the real service is put together from the environment.

The embedding model and the reranker are replaced by stand-ins, and no
database is contacted: building the service only stores the address.
"""

import sys
import types
from pathlib import Path

import pytest

from guide_search.config import ConfigError, Settings
from guide_search.embedding import FakeEmbedder
from guide_search.reranking import FakeReranker
from guide_search.service import Service, real_service

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    stand_in = types.ModuleType("guide_search.e5")
    stand_in.E5Embedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "guide_search.e5", stand_in)
    reranker_stand_in = types.ModuleType("guide_search.cross_encoder")
    reranker_stand_in.CrossEncoderReranker = FakeReranker
    monkeypatch.setitem(sys.modules, "guide_search.cross_encoder", reranker_stand_in)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("USAGE_LOG", raising=False)
    monkeypatch.chdir(REPOSITORY)  # settings.toml and agent.toml are read from the project folder
    return monkeypatch


def test_without_a_key_the_service_has_no_answer_agent(environment: pytest.MonkeyPatch):
    assert real_service().agent is None


def test_with_a_key_the_service_has_an_answer_agent_that_logs_to_the_default_file(environment: pytest.MonkeyPatch):
    environment.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    service = real_service()

    assert service.agent is not None
    assert service.agent.usage_log == Path("usage.jsonl")


def test_the_usage_log_can_be_put_elsewhere(environment: pytest.MonkeyPatch, tmp_path: Path):
    environment.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    environment.setenv("USAGE_LOG", str(tmp_path / "records.jsonl"))

    assert real_service().agent.usage_log == tmp_path / "records.jsonl"


def test_the_real_service_has_a_reranker(environment: pytest.MonkeyPatch):
    assert isinstance(real_service().reranker, FakeReranker)


def test_a_service_whose_default_mode_is_rerank_cannot_be_built_without_a_reranker():
    settings = Settings(threshold=0.0, default_mode="rerank", rerank_threshold=0.0)

    with pytest.raises(ConfigError):
        Service("postgresql://nobody@127.0.0.1:1/nothing", FakeEmbedder(), settings)

    # With one it can.
    Service("postgresql://nobody@127.0.0.1:1/nothing", FakeEmbedder(), settings, reranker=FakeReranker())
