"""Tests for fetching the guides from the help centre.

Nothing here touches the network. A fake client stands in for the help centre
and answers from a table that each test prepares, and a fake `sleep` stands in
for waiting. Every guide in this file is invented.
"""

import json
from datetime import date

import httpx
import pytest

from guide_search import fetch
from guide_search.fetch import (
    ARTICLE_URL,
    CATALOGUE_URL,
    USER_AGENT,
    FetchError,
    fetch_snapshot,
    guide_from_article,
    main,
    normalise_timestamp,
)
from guide_search.snapshot import (
    GUIDE_KEYS,
    PUBLIC_PAGE_PREFIX,
    Guide,
    SnapshotError,
    load_snapshot,
    validate_snapshot,
)

BODY = "A short intro.\n\n## First step\n\nDo the first thing."


class FakeResponse:
    """The two things the fetch code reads from a response."""

    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> dict | None:
        return self.payload


class FakeClient:
    """Stands in for httpx.Client.

    It answers from a table of address -> response and keeps a record of what
    it was asked, so a test can check the requests afterwards.
    """

    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[dict] = []
        self.events: list[tuple] = []
        self.closed = False

    def get(self, url, params=None, headers=None) -> FakeResponse:
        self.requests.append({"url": url, "params": params, "headers": headers})
        self.events.append(("get", url))
        # An address the test did not prepare answers "not found", as a server would.
        return self.responses.get(url, FakeResponse(404))

    def close(self) -> None:
        self.closed = True


def make_article(guide_id: int, slug: str, title: str, body: str = BODY) -> dict:
    """One guide in the shape the help centre returns it."""
    return {
        "id": guide_id,
        "slug": slug,
        "title": title,
        "body": body,
        "locale": "en",
        "product_slug": "autoserv-pro",
        "type": "guide",
        "published_at": "2026-01-01T08:00:00.000000Z",
        "updated_at": "2026-09-13T18:18:28.000000Z",
    }


def article_payload(article: dict, is_fallback: bool = False) -> dict:
    """The whole answer to a request for one guide."""
    return {
        "requested_locale": "en",
        "is_fallback": is_fallback,
        "article": article,
        "alternates": [],
    }


def catalogue_payload(articles: list[dict], last_page: int = 1, total: int | None = None) -> dict:
    """The whole answer to a request for the catalogue: the guides without their text."""
    entries = []
    for article in articles:
        entry = dict(article)
        del entry["body"]
        entry["updated_at"] = "2026-09-13T18:18:28+00:00"
        entries.append(entry)
    if total is None:
        total = len(entries)
    return {
        "data": entries,
        "meta": {"current_page": 1, "last_page": last_page, "total": total},
    }


def article_url(slug: str) -> str:
    return ARTICLE_URL.format(locale="en", product="autoserv-pro", slug=slug)


def help_centre(articles: list[dict]) -> FakeClient:
    """A fake help centre that lists these guides in its catalogue and serves each one."""
    responses = {CATALOGUE_URL: FakeResponse(200, catalogue_payload(articles))}
    for article in articles:
        responses[article_url(article["slug"])] = FakeResponse(200, article_payload(article))
    return FakeClient(responses)


def three_articles() -> list[dict]:
    """Three guides, listed out of id order on purpose."""
    return [
        make_article(61, "count-the-stock", "Count the stock"),
        make_article(2, "open-the-shop", "Open the shop"),
        make_article(7, "close-the-till", "Close the till"),
    ]


def no_sleep(seconds: float) -> None:
    """Passed as `sleep` so that a test does not wait."""


def use_fake_client_in_main(monkeypatch, client: FakeClient) -> None:
    """Make `main` talk to the fake client, and pause for no time at all."""

    def open_fake_client() -> FakeClient:
        return client

    monkeypatch.setattr(fetch, "open_client", open_fake_client)
    monkeypatch.setattr(fetch, "PAUSE_SECONDS", 0)


# --- the catalogue ------------------------------------------------------------


def test_catalogue_is_requested_with_per_page_200():
    client = help_centre(three_articles())

    fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    first = client.requests[0]
    assert first["url"] == "https://support.orions360.com/api/kb/catalogue"
    assert first["params"] == {"product": "autoserv-pro", "locale": "en", "per_page": 200}


def test_catalogue_with_a_second_page_is_an_error():
    articles = three_articles()
    client = help_centre(articles)
    client.responses[CATALOGUE_URL] = FakeResponse(200, catalogue_payload(articles, last_page=2))

    with pytest.raises(FetchError, match="page"):
        fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    # It stops at the catalogue. No guide is requested.
    assert len(client.requests) == 1


def test_catalogue_count_that_differs_from_the_reported_total_is_an_error():
    articles = three_articles()
    client = help_centre(articles)
    client.responses[CATALOGUE_URL] = FakeResponse(200, catalogue_payload(articles, total=4))

    with pytest.raises(FetchError, match="total"):
        fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    assert len(client.requests) == 1


# --- one guide ----------------------------------------------------------------


def test_guide_from_article_gives_exactly_the_six_snapshot_keys():
    payload = article_payload(make_article(2, "open-the-shop", "Open the shop"))

    guide = guide_from_article(payload, "open-the-shop")

    assert guide == {
        "id": 2,
        "slug": "open-the-shop",
        "title": "Open the shop",
        "url": "https://support.orions360.com/help/en/autoserv-pro/open-the-shop",
        "updated_at": "2026-09-13T18:18:28+00:00",
        "body": BODY,
    }
    assert tuple(guide) == GUIDE_KEYS


def test_guide_text_is_copied_unchanged():
    body = "  Say مرحبا at the door.\n\n## Prices\n\nA coffee is 5 €.\n\n"
    payload = article_payload(make_article(2, "open-the-shop", "Café opening hours", body))

    guide = guide_from_article(payload, "open-the-shop")

    assert guide["title"] == "Café opening hours"
    assert guide["body"] == body


def test_article_marked_as_fallback_is_an_error():
    payload = article_payload(make_article(2, "open-the-shop", "Open the shop"), is_fallback=True)

    with pytest.raises(FetchError, match="fallback"):
        guide_from_article(payload, "open-the-shop")


def test_article_in_the_wrong_locale_is_an_error():
    article = make_article(2, "open-the-shop", "Open the shop")
    article["locale"] = "ar"

    with pytest.raises(FetchError, match="'ar'"):
        guide_from_article(article_payload(article), "open-the-shop")


def test_article_with_the_wrong_slug_is_an_error():
    payload = article_payload(make_article(7, "close-the-till", "Close the till"))

    with pytest.raises(FetchError, match="close-the-till"):
        guide_from_article(payload, "open-the-shop")


# --- timestamps ---------------------------------------------------------------


def test_both_observed_timestamp_forms_normalise_to_the_same_string():
    assert normalise_timestamp("2026-09-13T18:18:28.000000Z") == "2026-09-13T18:18:28+00:00"
    assert normalise_timestamp("2026-09-13T18:18:28+00:00") == "2026-09-13T18:18:28+00:00"


def test_timestamp_in_another_time_zone_is_moved_to_utc():
    assert normalise_timestamp("2026-09-13T21:18:28+03:00") == "2026-09-13T18:18:28+00:00"


def test_timestamp_loses_its_microseconds():
    assert normalise_timestamp("2026-09-13T18:18:28.750000Z") == "2026-09-13T18:18:28+00:00"


def test_timestamp_without_a_time_zone_is_an_error():
    with pytest.raises(FetchError, match="time zone"):
        normalise_timestamp("2026-09-13T18:18:28")


@pytest.mark.parametrize("value", ["last Tuesday", "", None])
def test_value_that_is_not_a_timestamp_is_an_error(value):
    with pytest.raises(FetchError, match="not a timestamp"):
        normalise_timestamp(value)


# --- the whole fetch ----------------------------------------------------------


def test_guides_come_out_sorted_by_id_with_the_public_page_url():
    snapshot = fetch_snapshot(help_centre(three_articles()), "2026-10-10", sleep=no_sleep)

    assert [guide["id"] for guide in snapshot["guides"]] == [2, 7, 61]
    assert [guide["url"] for guide in snapshot["guides"]] == [
        "https://support.orions360.com/help/en/autoserv-pro/open-the-shop",
        "https://support.orions360.com/help/en/autoserv-pro/close-the-till",
        "https://support.orions360.com/help/en/autoserv-pro/count-the-stock",
    ]


def test_sleep_is_called_between_requests_with_one_and_a_half_seconds():
    client = help_centre(three_articles())

    def recording_sleep(seconds: float) -> None:
        client.events.append(("sleep", seconds))

    fetch_snapshot(client, "2026-10-10", sleep=recording_sleep)

    # One request for the catalogue, then one per guide in the catalogue's order,
    # with a pause between every two requests and none before the first or after the last.
    assert client.events == [
        ("get", CATALOGUE_URL),
        ("sleep", 1.5),
        ("get", "https://support.orions360.com/api/kb/articles/en/autoserv-pro/count-the-stock"),
        ("sleep", 1.5),
        ("get", "https://support.orions360.com/api/kb/articles/en/autoserv-pro/open-the-shop"),
        ("sleep", 1.5),
        ("get", "https://support.orions360.com/api/kb/articles/en/autoserv-pro/close-the-till"),
    ]


def test_every_request_sends_a_user_agent_that_names_the_project():
    client = help_centre(three_articles())

    fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    assert "guide-search" in USER_AGENT
    assert len(client.requests) == 4
    for request in client.requests:
        assert request["headers"] == {"User-Agent": USER_AGENT}


def test_result_passes_validate_snapshot():
    snapshot = fetch_snapshot(help_centre(three_articles()), "2026-10-10", sleep=no_sleep)

    assert snapshot["source"] == "https://support.orions360.com/api/kb"
    assert snapshot["product"] == "autoserv-pro"
    assert snapshot["locale"] == "en"
    assert snapshot["fetched_at"] == "2026-10-10"
    guides = validate_snapshot(snapshot)
    assert guides[0] == Guide(
        id=2,
        slug="open-the-shop",
        title="Open the shop",
        url=PUBLIC_PAGE_PREFIX + "open-the-shop",
        updated_at="2026-09-13T18:18:28+00:00",
        body=BODY,
    )
    assert len(guides) == 3


def test_result_is_validated_before_it_is_returned():
    # Each guide is fine on its own. Only the snapshot as a whole breaks a rule:
    # two guides share a title.
    articles = [
        make_article(1, "open-the-shop", "Open the shop"),
        make_article(2, "close-the-till", "Open the shop"),
    ]

    with pytest.raises(SnapshotError):
        fetch_snapshot(help_centre(articles), "2026-10-10", sleep=no_sleep)


def test_fetch_stops_at_a_guide_that_is_a_fallback():
    articles = three_articles()
    client = help_centre(articles)
    fallback = article_payload(articles[1], is_fallback=True)
    client.responses[article_url("open-the-shop")] = FakeResponse(200, fallback)

    with pytest.raises(FetchError, match="open-the-shop"):
        fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    # Catalogue, first guide, the fallback. The third guide is never requested.
    assert len(client.requests) == 3


def test_the_real_httpx_client_sends_the_page_size_and_the_user_agent():
    # The same fetch through the real client class. httpx.MockTransport hands
    # each request to the function below, so nothing reaches the network.
    prepared = help_centre(three_articles()).responses
    seen: list[tuple[str, str]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["user-agent"]))
        address = str(request.url).split("?")[0]
        response = prepared[address]
        return httpx.Response(response.status_code, json=response.payload)

    client = httpx.Client(transport=httpx.MockTransport(answer))
    snapshot = fetch_snapshot(client, "2026-10-10", sleep=no_sleep)
    client.close()

    catalogue_address = (
        "https://support.orions360.com/api/kb/catalogue"
        "?product=autoserv-pro&locale=en&per_page=200"
    )
    assert seen[0] == (catalogue_address, USER_AGENT)
    assert len(seen) == 4
    assert [guide["id"] for guide in snapshot["guides"]] == [2, 7, 61]


# --- HTTP errors --------------------------------------------------------------


def test_http_429_on_the_catalogue_is_an_error():
    client = help_centre(three_articles())
    client.responses[CATALOGUE_URL] = FakeResponse(429)

    with pytest.raises(FetchError, match="asked too often"):
        fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    assert len(client.requests) == 1


def test_http_429_on_a_guide_stops_the_fetch_at_once():
    client = help_centre(three_articles())
    client.responses[article_url("count-the-stock")] = FakeResponse(429)

    with pytest.raises(FetchError, match="asked too often"):
        fetch_snapshot(client, "2026-10-10", sleep=no_sleep)

    # The catalogue and the refused guide. Nothing is asked after the refusal.
    assert len(client.requests) == 2


@pytest.mark.parametrize("status_code", [301, 404, 500])
def test_any_other_status_than_200_is_an_error(status_code):
    client = help_centre(three_articles())
    client.responses[article_url("open-the-shop")] = FakeResponse(status_code)

    with pytest.raises(FetchError, match=str(status_code)):
        fetch_snapshot(client, "2026-10-10", sleep=no_sleep)


# --- writing the file ---------------------------------------------------------


def test_main_writes_the_file_with_a_trailing_newline_and_non_ascii_kept(
    tmp_path, monkeypatch, capsys
):
    client = help_centre(
        [
            make_article(2, "open-the-shop", "Café opening hours"),
            make_article(1, "close-the-till", "Close the till", "Say مرحبا.\n\n## Prices\n\n5 €."),
        ]
    )
    use_fake_client_in_main(monkeypatch, client)
    # The folder does not exist yet, as on the first run in a new clone.
    output = tmp_path / "data" / "guides.json"

    exit_code = main(["--output", str(output)])

    assert exit_code == 0
    written = output.read_bytes()
    assert written.endswith(b"\n}\n")
    text = written.decode("utf-8")
    assert "Café opening hours" in text
    assert "مرحبا" in text
    assert "€" in text
    assert "\\u" not in text
    # Indented by two spaces, so the file can be read and compared by eye.
    assert text.startswith('{\n  "source": "https://support.orions360.com/api/kb",\n')
    date.fromisoformat(json.loads(text)["fetched_at"])
    assert [guide.id for guide in load_snapshot(output)] == [1, 2]
    assert capsys.readouterr().out.splitlines()[-1] == f"Wrote 2 guides to {output}"
    assert client.closed


def test_main_leaves_the_existing_file_alone_when_the_fetch_fails(tmp_path, monkeypatch):
    articles = three_articles()
    client = help_centre(articles)
    fallback = article_payload(articles[2], is_fallback=True)
    client.responses[article_url("close-the-till")] = FakeResponse(200, fallback)
    use_fake_client_in_main(monkeypatch, client)
    output = tmp_path / "guides.json"
    output.write_text("the earlier snapshot\n", encoding="utf-8")

    with pytest.raises(FetchError):
        main(["--output", str(output)])

    assert output.read_text(encoding="utf-8") == "the earlier snapshot\n"
    assert client.closed
