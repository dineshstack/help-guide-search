"""The guide snapshot (data/guides.json): its shape, and the check that a file has it.

The snapshot is the project's only input: one JSON file that holds every help
guide. Two places handle it, the fetch script that writes the file and the
ingestion that reads it, and both call `validate_snapshot` in this module, so
the rules are written down once.

The rules, in short:

- the file has exactly five keys, and every guide has exactly six;
- ids are whole numbers above zero and each is greater than the one before,
  so the guides are in id order and no id repeats (gaps are fine);
- no two guides share a slug or a title;
- a slug is lower-case words joined by hyphens, and `url` is the public page
  of that slug;
- `updated_at` is a timestamp with a time zone;
- no value is blank.

A snapshot that breaks any rule is rejected as a whole. The error lists every
problem that was found, not only the first one, so that one run shows
everything that is wrong.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SOURCE = "https://support.orions360.com/api/kb"
PRODUCT = "autoserv-pro"
LOCALE = "en"
PUBLIC_PAGE_PREFIX = "https://support.orions360.com/help/en/autoserv-pro/"
TOP_LEVEL_KEYS = ("source", "product", "locale", "fetched_at", "guides")
GUIDE_KEYS = ("id", "slug", "title", "url", "updated_at", "body")

# Lower-case words (letters and digits) joined by single hyphens,
# for example "open-the-shop" or "vat-48-hours".
SLUG_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


@dataclass(frozen=True)
class Guide:
    """One help guide, exactly as the snapshot holds it."""

    id: int
    slug: str
    title: str
    url: str
    updated_at: str
    body: str


class SnapshotError(ValueError):
    """Carries every problem found. `problems` is a list of plain sentences."""

    problems: list[str]

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = ["The snapshot is not valid:"]
        for problem in problems:
            lines.append(f"- {problem}")
        super().__init__("\n".join(lines))


def validate_snapshot(data: object) -> list[Guide]:
    """Check a snapshot that has been read from JSON and return its guides.

    Raises one SnapshotError that lists every problem, if there is any.
    """
    if not isinstance(data, dict):
        raise SnapshotError(["The snapshot must be a JSON object."])

    problems = _key_problems("The snapshot", data, TOP_LEVEL_KEYS)
    for key in ("source", "product", "locale", "fetched_at"):
        if key in data and not _is_text(data[key]):
            problems.append(f"The snapshot: '{key}' must be text and must not be blank.")
    if "guides" in data:
        problems.extend(_guide_list_problems(data["guides"]))

    if problems:
        raise SnapshotError(problems)
    return [_to_guide(raw) for raw in data["guides"]]


def load_snapshot(path: str | Path) -> list[Guide]:
    """Read a snapshot file, check it, and return its guides."""
    # The encoding is named because the default differs between machines,
    # and the guides contain characters outside plain ASCII.
    text = Path(path).read_text(encoding="utf-8")
    return validate_snapshot(json.loads(text))


# The functions below are helpers for this file only. The leading underscore
# is the usual Python way of saying "not meant to be called from outside".


def _to_guide(raw: dict) -> Guide:
    return Guide(
        id=raw["id"],
        slug=raw["slug"],
        title=raw["title"],
        url=raw["url"],
        updated_at=raw["updated_at"],
        body=raw["body"],
    )


def _key_problems(label: str, found: dict, expected: tuple[str, ...]) -> list[str]:
    """One problem for each key that is missing and each key that should not be there."""
    problems: list[str] = []
    for key in expected:
        if key not in found:
            problems.append(f"{label} has no '{key}' key.")
    for key in found:
        if key not in expected:
            problems.append(f"{label} has a key that is not allowed: '{key}'.")
    return problems


def _guide_list_problems(raw_guides: object) -> list[str]:
    """Problems in the list of guides: each guide alone, then the guides together."""
    if not isinstance(raw_guides, list) or len(raw_guides) == 0:
        return ["'guides' must be a list with at least one guide in it."]

    problems: list[str] = []
    # A guide is named by its place in the list, counting from 1,
    # because a broken guide may have no id or slug to name it by.
    for number, raw in enumerate(raw_guides, start=1):
        problems.extend(_guide_problems(f"Guide {number}", raw))
    problems.extend(_id_order_problems(raw_guides))
    problems.extend(_duplicate_problems(raw_guides, "slug"))
    problems.extend(_duplicate_problems(raw_guides, "title"))
    return problems


def _guide_problems(label: str, raw: object) -> list[str]:
    """Problems that can be seen by looking at one guide alone."""
    if not isinstance(raw, dict):
        return [f"{label} must be a JSON object."]

    problems = _key_problems(label, raw, GUIDE_KEYS)
    has_every_key = all(key in raw for key in GUIDE_KEYS)
    if not has_every_key:
        # The checks below read every key, so they cannot run on this guide.
        return problems

    if not _is_positive_int(raw["id"]):
        problems.append(f"{label}: 'id' must be a whole number greater than zero.")
    for key in ("slug", "title", "url", "updated_at", "body"):
        if not _is_text(raw[key]):
            problems.append(f"{label}: '{key}' must be text and must not be blank.")

    # A value of the wrong kind was reported above. The rules below only make
    # sense for text, so each one is skipped when its value is not text.
    slug = raw["slug"]
    url = raw["url"]
    updated_at = raw["updated_at"]
    # fullmatch, not match: the slug must fit the pattern from its first
    # character to its last, with nothing after it, not even a line break.
    if _is_text(slug) and SLUG_PATTERN.fullmatch(slug) is None:
        problems.append(f"{label}: the slug {slug!r} is not lower-case words joined by hyphens.")
    if _is_text(slug) and _is_text(url) and url != PUBLIC_PAGE_PREFIX + slug:
        expected = PUBLIC_PAGE_PREFIX + slug
        problems.append(f"{label}: 'url' is {url!r}, but the public page of the slug is {expected!r}.")
    if _is_text(updated_at):
        problems.extend(_timestamp_problems(label, updated_at))
    # Chunking splits the body at line breaks. A carriage return (the extra
    # character in Windows line endings) would stay inside the passages, so a
    # body that has one is refused here instead of being stored oddly.
    if _is_text(raw["body"]) and "\r" in raw["body"]:
        problems.append(f"{label}: 'body' contains a carriage return; line breaks must be plain '\\n'.")
    return problems


def _timestamp_problems(label: str, updated_at: str) -> list[str]:
    try:
        moment = datetime.fromisoformat(updated_at)
    except ValueError:
        return [f"{label}: 'updated_at' is {updated_at!r}, which is not a timestamp."]
    # Without a time zone the same text means a different moment on every machine.
    if moment.tzinfo is None:
        return [f"{label}: 'updated_at' is {updated_at!r}, which has no time zone."]
    return []


def _id_order_problems(raw_guides: list) -> list[str]:
    """Each id must be greater than the one before it.

    That one rule gives two things: the file is in id order, and no id is used
    twice. Gaps are fine, because the help centre's ids are not consecutive.
    """
    problems: list[str] = []
    previous: int | None = None
    for number, raw in enumerate(raw_guides, start=1):
        if not isinstance(raw, dict) or not _is_positive_int(raw.get("id")):
            continue  # already reported by _guide_problems
        guide_id = raw["id"]
        if previous is not None and guide_id <= previous:
            problems.append(
                f"Guide {number}: id {guide_id} does not come after id {previous}. "
                "Guides must be in id order and no id may repeat."
            )
        previous = guide_id
    return problems


def _duplicate_problems(raw_guides: list, key: str) -> list[str]:
    """One problem for each guide whose slug (or title) an earlier guide already has."""
    problems: list[str] = []
    first_use: dict[str, int] = {}
    for number, raw in enumerate(raw_guides, start=1):
        if not isinstance(raw, dict) or not _is_text(raw.get(key)):
            continue  # already reported by _guide_problems
        value = raw[key]
        if value in first_use:
            earlier = first_use[value]
            problems.append(f"Guide {number}: the {key} {value!r} is already used by guide {earlier}.")
        else:
            first_use[value] = number
    return problems


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_positive_int(value: object) -> bool:
    # In Python True and False count as whole numbers (1 and 0),
    # so they have to be ruled out by name.
    if isinstance(value, bool):
        return False
    return isinstance(value, int) and value > 0
