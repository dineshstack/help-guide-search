"""Fetches the public help guides and writes the snapshot file, data/guides.json.

This is the only code in the project that talks to the help centre, and the
only code that writes data/guides.json. It asks for the catalogue (the list of
guides) and then for each guide in turn: one request at a time, with a pause
before each.

Nothing is written unless every guide arrived, none of them is a stand-in from
another language, and the finished snapshot passes validation.

Run it from the repository root with: python scripts/fetch_guides.py
"""

import argparse
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from guide_search.snapshot import LOCALE, PRODUCT, PUBLIC_PAGE_PREFIX, SOURCE, validate_snapshot

CATALOGUE_URL = SOURCE + "/catalogue"
ARTICLE_URL = SOURCE + "/articles/{locale}/{product}/{slug}"
USER_AGENT = "help-guide-search/0.1 (builds a snapshot of the public help guides)"
PAUSE_SECONDS = 1.5

# A catalogue page holds 30 guides unless told otherwise. 200 puts them all on one page.
PER_PAGE = 200
TIMEOUT_SECONDS = 30.0
DEFAULT_OUTPUT = "data/guides.json"


class FetchError(RuntimeError):
    """The help centre gave an answer that must not end up in the snapshot."""


def normalise_timestamp(value: str) -> str:
    """Rewrite a timestamp as UTC in one fixed form: "2026-09-13T18:18:28+00:00".

    The help centre writes the same moment in more than one way. Keeping one
    form means the snapshot only changes when a guide really changed.
    """
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise FetchError(f"{value!r} is not a timestamp.") from error
    if moment.tzinfo is None:
        raise FetchError(f"The timestamp {value!r} has no time zone.")
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def guide_from_article(payload: dict, expected_slug: str) -> dict:
    """Turn the help centre's answer for one guide into the six keys the snapshot keeps."""
    # Asked for a language it does not have, the help centre sends the guide in
    # another language and marks the answer as a fallback. That is not the
    # guide that was asked for, so it is refused.
    if payload["is_fallback"]:
        raise FetchError(f"The guide '{expected_slug}' came back as a language fallback.")
    article = payload["article"]
    if article["locale"] != LOCALE:
        found = article["locale"]
        raise FetchError(f"The guide '{expected_slug}' came back in '{found}', not in '{LOCALE}'.")
    if article["slug"] != expected_slug:
        found = article["slug"]
        raise FetchError(f"Asked for the guide '{expected_slug}' and got '{found}'.")
    return {
        "id": article["id"],
        "slug": article["slug"],
        "title": article["title"],
        "url": PUBLIC_PAGE_PREFIX + article["slug"],
        "updated_at": normalise_timestamp(article["updated_at"]),
        # The text is copied exactly as received. It is never trimmed or tidied.
        "body": article["body"],
    }


def fetch_snapshot(client, today: str, sleep=time.sleep) -> dict:
    """Fetch the catalogue and every guide in it, and return the finished snapshot.

    `client` is anything with a `get(url, params=..., headers=...)` method, such
    as an httpx.Client. `today` becomes the snapshot's "fetched_at". `sleep` is
    the function used to wait; the tests pass one that does not wait.
    """
    guides = []
    for slug in _catalogue_slugs(client):
        # The catalogue was the first request, so pausing before each guide
        # puts one pause between every two requests.
        sleep(PAUSE_SECONDS)
        url = ARTICLE_URL.format(locale=LOCALE, product=PRODUCT, slug=slug)
        guides.append(guide_from_article(_get_json(client, url), slug))
    guides.sort(key=_guide_id)

    snapshot = {
        "source": SOURCE,
        "product": PRODUCT,
        "locale": LOCALE,
        "fetched_at": today,
        "guides": guides,
    }
    # Raises SnapshotError, listing every problem, if the result breaks a rule.
    validate_snapshot(snapshot)
    return snapshot


def open_client() -> httpx.Client:
    """The real HTTP client. A function of its own, so a test can put a fake in its place."""
    return httpx.Client(timeout=TIMEOUT_SECONDS)


def write_snapshot(snapshot: dict, path: Path) -> None:
    """Write the snapshot as indented JSON in UTF-8, ending with a line break."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" makes the file the same, byte for byte, on every system.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        # ensure_ascii=False writes "é" as "é" and not as "é",
        # so the guide text in the file can be read as it is.
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Fetch the guides and write the snapshot file. Returns 0 when the file was written."""
    parser = argparse.ArgumentParser(description="Fetch the public help guides into one file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="the file to write")
    args = parser.parse_args(argv)

    print(f"Fetching the catalogue, then each guide, {PAUSE_SECONDS} seconds apart.")
    print("This takes a minute or two.")
    client = open_client()
    try:
        snapshot = fetch_snapshot(client, date.today().isoformat())
    finally:
        client.close()

    # This line is reached only when every guide arrived and the snapshot is valid.
    write_snapshot(snapshot, Path(args.output))
    print(f"Wrote {len(snapshot['guides'])} guides to {args.output}")
    return 0


def _catalogue_slugs(client) -> list[str]:
    """Ask for the list of guides and return their slugs, in the catalogue's order."""
    params = {"product": PRODUCT, "locale": LOCALE, "per_page": PER_PAGE}
    catalogue = _get_json(client, CATALOGUE_URL, params)
    entries = catalogue["data"]
    meta = catalogue["meta"]
    # Both checks guard against quietly getting only part of the list.
    if meta["last_page"] != 1:
        pages = meta["last_page"]
        raise FetchError(f"The catalogue has {pages} pages. Every guide was expected on page 1.")
    if len(entries) != meta["total"]:
        total = meta["total"]
        raise FetchError(f"The catalogue lists {len(entries)} guides but reports a total of {total}.")
    return [entry["slug"] for entry in entries]


def _get_json(client, url: str, params: dict | None = None) -> dict:
    """Make one GET request. Any answer other than a plain success stops the fetch."""
    response = client.get(url, params=params, headers={"User-Agent": USER_AGENT})
    if response.status_code == 429:
        raise FetchError("The help centre says it is being asked too often (HTTP 429). Stopped.")
    if response.status_code != 200:
        raise FetchError(f"{url} answered with HTTP {response.status_code}, not 200.")
    return response.json()


def _guide_id(guide: dict) -> int:
    return guide["id"]
