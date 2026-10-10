"""Tests for config.py: the settings file and the database address.

Nothing here needs a database. Settings files are written to a temporary
folder that pytest provides (`tmp_path`), except for one test that reads the
settings.toml committed at the top of the repository.
"""

from pathlib import Path

import pytest

from guide_search.config import (
    ConfigError,
    Settings,
    database_url,
    load_settings,
    save_settings,
)
from guide_search.search import MODES

# The settings.toml at the top of the repository, two folders above tests/unit.
COMMITTED_SETTINGS = Path(__file__).resolve().parents[2] / "settings.toml"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def test_settings_come_back_unchanged_after_save_and_load(tmp_path: Path):
    path = tmp_path / "settings.toml"
    # A threshold as the evaluation produces it: a midpoint, not rounded.
    settings = Settings(threshold=0.8500000000000001, default_mode="vector")

    save_settings(settings, path)

    assert load_settings(path) == settings


def test_the_committed_settings_file_loads():
    settings = load_settings(COMMITTED_SETTINGS)

    assert isinstance(settings.threshold, float)
    assert settings.default_mode in MODES


def test_saving_what_the_committed_file_holds_gives_the_same_file(tmp_path: Path):
    # So the first save by the evaluation changes the threshold line and
    # nothing else: the comment line and the layout stay as they are.
    copy = tmp_path / "settings.toml"

    save_settings(load_settings(COMMITTED_SETTINGS), copy)

    assert copy.read_text(encoding="utf-8") == COMMITTED_SETTINGS.read_text(encoding="utf-8")


def test_a_threshold_written_as_a_whole_number_is_read_as_a_float(tmp_path: Path):
    path = write(tmp_path / "settings.toml", 'threshold = 0\ndefault_mode = "hybrid"\n')

    settings = load_settings(path)

    assert settings == Settings(threshold=0.0, default_mode="hybrid")
    assert isinstance(settings.threshold, float)


# ---------------------------------------------------------------------------
# A settings file that cannot be used
# ---------------------------------------------------------------------------


def test_a_missing_settings_file_raises_config_error_naming_the_file(tmp_path: Path):
    path = tmp_path / "no-such-settings.toml"

    with pytest.raises(ConfigError) as caught:
        load_settings(path)

    assert "no-such-settings.toml" in str(caught.value)


@pytest.mark.parametrize(
    "text",
    [
        "threshold = = 0.5\n",
        'default_mode = "hybrid"\n',
        "threshold = 0.5\n",
        'threshold = 0.5\ndefault_mode = "hybrid"\nextra = 1\n',
        'threshold = "high"\ndefault_mode = "hybrid"\n',
        'threshold = true\ndefault_mode = "hybrid"\n',
        'threshold = 0.5\ndefault_mode = "fuzzy"\n',
        "threshold = 0.5\ndefault_mode = 3\n",
    ],
    ids=[
        "not TOML",
        "no threshold",
        "no default_mode",
        "an extra value",
        "threshold is text",
        "threshold is true",
        "unknown mode",
        "mode is a number",
    ],
)
def test_a_settings_file_that_breaks_a_rule_raises_config_error(tmp_path: Path, text: str):
    path = write(tmp_path / "settings.toml", text)

    with pytest.raises(ConfigError):
        load_settings(path)


def test_settings_with_an_unknown_mode_are_not_saved(tmp_path: Path):
    path = tmp_path / "settings.toml"

    with pytest.raises(ConfigError):
        save_settings(Settings(threshold=0.5, default_mode="fuzzy"), path)

    assert not path.exists()


# ---------------------------------------------------------------------------
# The database address
# ---------------------------------------------------------------------------


def test_database_url_returns_the_value_of_the_environment_variable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://someone@db.example.test/guides")

    assert database_url() == "postgresql://someone@db.example.test/guides"


def test_a_missing_database_url_raises_config_error_with_a_plain_message(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ConfigError) as caught:
        database_url()

    message = str(caught.value)
    assert "DATABASE_URL is not set" in message
    assert ".env" in message


def test_a_blank_database_url_counts_as_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "   ")

    with pytest.raises(ConfigError):
        database_url()
