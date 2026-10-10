"""Tests for snapshot validation: the rules that data/guides.json must follow.

Every guide in this file is invented. Nothing is copied from the real guides.
Most tests start from one valid snapshot, break exactly one rule, and check
that exactly one problem is reported.
"""

import json

import pytest

from guide_search.snapshot import (
    LOCALE,
    PRODUCT,
    PUBLIC_PAGE_PREFIX,
    SOURCE,
    Guide,
    SnapshotError,
    load_snapshot,
    validate_snapshot,
)

BODY = "A short intro.\n\n## First step\n\nDo the first thing."


def make_guide(guide_id: int, slug: str, title: str) -> dict:
    """One valid guide, as it appears in the snapshot file."""
    return {
        "id": guide_id,
        "slug": slug,
        "title": title,
        "url": PUBLIC_PAGE_PREFIX + slug,
        "updated_at": "2026-01-05T09:30:00+00:00",
        "body": BODY,
    }


def make_snapshot() -> dict:
    """A valid snapshot with two guides."""
    return {
        "source": SOURCE,
        "product": PRODUCT,
        "locale": LOCALE,
        "fetched_at": "2026-01-06",
        "guides": [
            make_guide(1, "open-the-shop", "Open the shop"),
            make_guide(2, "close-the-till", "Close the till"),
        ],
    }


def problems_in(data: object) -> list[str]:
    """Validate something that must be rejected and return the problems it reports."""
    with pytest.raises(SnapshotError) as caught:
        validate_snapshot(data)
    return caught.value.problems


# --- a valid snapshot ---------------------------------------------------------


def test_valid_snapshot_loads_into_guide_objects():
    guides = validate_snapshot(make_snapshot())

    assert guides == [
        Guide(
            id=1,
            slug="open-the-shop",
            title="Open the shop",
            url="https://support.orions360.com/help/en/autoserv-pro/open-the-shop",
            updated_at="2026-01-05T09:30:00+00:00",
            body=BODY,
        ),
        Guide(
            id=2,
            slug="close-the-till",
            title="Close the till",
            url="https://support.orions360.com/help/en/autoserv-pro/close-the-till",
            updated_at="2026-01-05T09:30:00+00:00",
            body=BODY,
        ),
    ]


def test_ids_with_gaps_are_accepted():
    snapshot = make_snapshot()
    snapshot["guides"] = [
        make_guide(1, "open-the-shop", "Open the shop"),
        make_guide(59, "close-the-till", "Close the till"),
        make_guide(61, "count-the-stock", "Count the stock"),
    ]

    guides = validate_snapshot(snapshot)

    assert [guide.id for guide in guides] == [1, 59, 61]


@pytest.mark.parametrize("slug", ["shop", "vat-48-hours", "a1-b2-c3"])
def test_slug_of_lower_case_words_and_digits_is_accepted(slug):
    snapshot = make_snapshot()
    snapshot["guides"][0] = make_guide(1, slug, "Open the shop")

    guides = validate_snapshot(snapshot)

    assert guides[0].slug == slug


@pytest.mark.parametrize(
    "updated_at",
    [
        "2026-01-05T09:30:00+00:00",
        "2026-01-05T09:30:00.000000Z",
        "2026-01-05T12:30:00+03:00",
    ],
)
def test_updated_at_with_a_time_zone_is_accepted_and_kept_as_written(updated_at):
    snapshot = make_snapshot()
    snapshot["guides"][0]["updated_at"] = updated_at

    guides = validate_snapshot(snapshot)

    assert guides[0].updated_at == updated_at


# --- the keys -----------------------------------------------------------------


def test_extra_top_level_key_fails():
    snapshot = make_snapshot()
    snapshot["note"] = "not part of the format"

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'note'" in problems[0]


def test_missing_top_level_key_fails():
    snapshot = make_snapshot()
    del snapshot["fetched_at"]

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'fetched_at'" in problems[0]


def test_extra_guide_key_fails():
    snapshot = make_snapshot()
    # The help-centre API returns this field too. The snapshot must not carry it.
    snapshot["guides"][1]["published_at"] = "2026-01-01T00:00:00+00:00"

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "Guide 2" in problems[0]
    assert "'published_at'" in problems[0]


def test_missing_guide_key_fails():
    snapshot = make_snapshot()
    del snapshot["guides"][0]["updated_at"]

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "Guide 1" in problems[0]
    assert "'updated_at'" in problems[0]


# --- the container ------------------------------------------------------------


@pytest.mark.parametrize("data", [[], "guides", None, 3])
def test_snapshot_that_is_not_an_object_fails(data):
    assert len(problems_in(data)) == 1


@pytest.mark.parametrize("guides", [[], {}, "none", None])
def test_guides_that_are_not_a_non_empty_list_fail(guides):
    snapshot = make_snapshot()
    snapshot["guides"] = guides

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'guides'" in problems[0]


def test_guide_that_is_not_an_object_fails():
    snapshot = make_snapshot()
    snapshot["guides"][1] = "close-the-till"

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "Guide 2" in problems[0]


@pytest.mark.parametrize("key", ["source", "product", "locale", "fetched_at"])
def test_top_level_value_that_is_not_text_fails(key):
    snapshot = make_snapshot()
    snapshot[key] = None

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert f"'{key}'" in problems[0]


# --- ids, slugs and titles ----------------------------------------------------


@pytest.mark.parametrize("bad_id", [0, -3, "1", 1.5, True, None])
def test_id_that_is_not_a_positive_whole_number_fails(bad_id):
    snapshot = make_snapshot()
    snapshot["guides"][0]["id"] = bad_id

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'id'" in problems[0]


@pytest.mark.parametrize("ids", [(2, 1), (5, 5)])
def test_ids_that_do_not_increase_fail(ids):
    snapshot = make_snapshot()
    snapshot["guides"] = [
        make_guide(ids[0], "open-the-shop", "Open the shop"),
        make_guide(ids[1], "close-the-till", "Close the till"),
    ]

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "Guide 2" in problems[0]
    assert "id" in problems[0]


def test_duplicate_slug_fails():
    snapshot = make_snapshot()
    snapshot["guides"][1] = make_guide(2, "open-the-shop", "Close the till")

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'open-the-shop'" in problems[0]


def test_duplicate_title_fails():
    snapshot = make_snapshot()
    snapshot["guides"][1] = make_guide(2, "close-the-till", "Open the shop")

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'Open the shop'" in problems[0]


@pytest.mark.parametrize(
    "slug",
    [
        "Open-The-Shop",
        "open_the_shop",
        "open the shop",
        "-open-the-shop",
        "open-the-shop-",
        "open--the-shop",
        "open-the-shop\n",
    ],
)
def test_slug_that_is_not_lower_case_words_joined_by_hyphens_fails(slug):
    snapshot = make_snapshot()
    # make_guide builds the url from the slug, so the slug is the only thing wrong.
    snapshot["guides"][0] = make_guide(1, slug, "Open the shop")

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "slug" in problems[0]


# --- url, updated_at and body -------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        PUBLIC_PAGE_PREFIX + "close-the-till",
        PUBLIC_PAGE_PREFIX + "open-the-shop/",
        "https://example.invalid/help/en/autoserv-pro/open-the-shop",
        "open-the-shop",
    ],
)
def test_url_that_is_not_the_public_page_of_the_slug_fails(url):
    snapshot = make_snapshot()
    snapshot["guides"][0]["url"] = url

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'url'" in problems[0]


@pytest.mark.parametrize("updated_at", ["2026-01-05T09:30:00", "2026-01-05"])
def test_updated_at_without_a_time_zone_fails(updated_at):
    snapshot = make_snapshot()
    snapshot["guides"][0]["updated_at"] = updated_at

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "time zone" in problems[0]


def test_updated_at_that_is_not_a_timestamp_fails():
    snapshot = make_snapshot()
    snapshot["guides"][0]["updated_at"] = "last Tuesday"

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'updated_at'" in problems[0]


@pytest.mark.parametrize("body", ["", "   ", "\n\t\n"])
def test_blank_body_fails(body):
    snapshot = make_snapshot()
    snapshot["guides"][0]["body"] = body

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "'body'" in problems[0]


def test_body_with_windows_line_endings_fails():
    snapshot = make_snapshot()
    snapshot["guides"][0]["body"] = "First line.\r\n\r\n## Heading\r\n\r\nText."

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert "carriage return" in problems[0]


@pytest.mark.parametrize("key", ["slug", "title", "url", "updated_at", "body"])
def test_guide_value_that_is_not_text_fails(key):
    snapshot = make_snapshot()
    snapshot["guides"][0][key] = 7

    problems = problems_in(snapshot)

    assert len(problems) == 1
    assert f"'{key}'" in problems[0]


# --- reporting ----------------------------------------------------------------


def test_several_problems_are_all_reported_in_one_error():
    snapshot = make_snapshot()
    snapshot["note"] = "not part of the format"
    snapshot["guides"][0] = make_guide(1, "Open_The_Shop", "Open the shop")
    snapshot["guides"][1]["body"] = "  "
    snapshot["guides"][1]["updated_at"] = "2026-01-05T09:30:00"

    with pytest.raises(SnapshotError) as caught:
        validate_snapshot(snapshot)

    problems = caught.value.problems
    assert len(problems) == 4
    assert any("'note'" in problem for problem in problems)
    assert any("Guide 1" in problem and "slug" in problem for problem in problems)
    assert any("Guide 2" in problem and "'body'" in problem for problem in problems)
    assert any("Guide 2" in problem and "time zone" in problem for problem in problems)
    # The error's own text shows every problem, so a person running a script sees them all.
    for problem in problems:
        assert problem in str(caught.value)


# --- reading the file ---------------------------------------------------------


def test_load_snapshot_reads_a_utf8_file(tmp_path):
    snapshot = make_snapshot()
    snapshot["guides"][0]["title"] = "Café opening hours"
    snapshot["guides"][0]["body"] = "Say مرحبا at the door.\n\n## Prices\n\nA coffee is 5 €."
    path = tmp_path / "guides.json"
    path.write_bytes(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))

    guides = load_snapshot(path)

    assert guides[0].title == "Café opening hours"
    assert guides[0].body == "Say مرحبا at the door.\n\n## Prices\n\nA coffee is 5 €."
    assert [guide.id for guide in guides] == [1, 2]


def test_load_snapshot_accepts_the_path_as_text(tmp_path):
    path = tmp_path / "guides.json"
    path.write_text(json.dumps(make_snapshot()), encoding="utf-8")

    guides = load_snapshot(str(path))

    assert [guide.slug for guide in guides] == ["open-the-shop", "close-the-till"]


def test_load_snapshot_rejects_a_file_that_breaks_a_rule(tmp_path):
    snapshot = make_snapshot()
    snapshot["guides"] = []
    path = tmp_path / "guides.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(SnapshotError):
        load_snapshot(path)
