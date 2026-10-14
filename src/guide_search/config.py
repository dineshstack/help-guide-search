"""Settings: the values search needs that are not written in the code.

There are two kinds, and they are kept in two places.

- Values that are the same on every machine: the two "too weak to be a
  match" thresholds and the default search mode. They live in settings.toml,
  which is committed. The evaluation writes that file when it chooses them on
  the dev questions. Search, the HTTP routes and the evaluation all read it,
  so they always agree.
- The one value that differs between machines: where Postgres is. It comes
  from the environment variable DATABASE_URL and is never committed.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from guide_search.search import MODES

# Relative, so it is looked up in the folder a command is started from: the
# top of the repository.
SETTINGS_PATH = Path("settings.toml")

# The first line of the file, so that a reader of the file knows not to edit it.
FIRST_LINE = "# Written by the evaluation when it chooses these values on dev. Not edited by hand."


class ConfigError(RuntimeError):
    """A setting is missing or cannot be used. The message says which, in plain words."""


@dataclass(frozen=True)
class Settings:
    """The three values in settings.toml."""

    threshold: float      # in vector and hybrid mode, a best similarity below this is "no_match"
    default_mode: str     # the search mode used when a request names none
    # In rerank mode, a best reranker score below this is "no_match". The
    # default refuses nothing, because the reranker's scores are never below 0.
    rerank_threshold: float = 0.0


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Read the settings file. Raises ConfigError if it is missing or breaks a rule."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(
            f"The settings file {path} was not found. "
            "Start the command from the top folder of the repository."
        ) from error

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"The settings file {path} is not valid TOML: {error}") from error

    if set(data) != {"threshold", "default_mode", "rerank_threshold"}:
        raise ConfigError(
            f"The settings file {path} must hold exactly three values: "
            "threshold, default_mode and rerank_threshold."
        )
    settings = Settings(
        threshold=data["threshold"],
        default_mode=data["default_mode"],
        rerank_threshold=data["rerank_threshold"],
    )
    _check(settings)
    # TOML reads "0" as a whole number. The rest of the code expects a float.
    return Settings(
        threshold=float(settings.threshold),
        default_mode=settings.default_mode,
        rerank_threshold=float(settings.rerank_threshold),
    )


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    """Write the settings file. Raises ConfigError, and writes nothing, for unusable settings."""
    _check(settings)
    # repr() of a float gives every digit that is needed to read the very same
    # number back, so the threshold is not rounded on its way through the file.
    lines = [
        FIRST_LINE,
        f"threshold = {float(settings.threshold)!r}",
        f'default_mode = "{settings.default_mode}"',
        f"rerank_threshold = {float(settings.rerank_threshold)!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def database_url() -> str:
    """Return where Postgres is, from the environment variable DATABASE_URL."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url == "":
        raise ConfigError(
            "DATABASE_URL is not set. Put it in the file .env and start the "
            "command with: uv run --env-file .env ..."
        )
    return url


def _check(settings: Settings) -> None:
    """Raise ConfigError unless both thresholds are numbers and the mode is a known one."""
    for name in ("threshold", "rerank_threshold"):
        value = getattr(settings, name)
        # True and False count as numbers in Python, so they are ruled out by name.
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not is_number:
            raise ConfigError(f"{name} must be a number, not {value!r}.")
    if settings.default_mode not in MODES:
        raise ConfigError(
            f"default_mode must be one of {', '.join(MODES)}, not {settings.default_mode!r}."
        )
