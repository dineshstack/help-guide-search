"""Tests for chunking: how one guide is cut into passages (design section 4.2).

Every guide body in this file is invented for the test.

Most tests of split_long pass a small limit instead of the real 1,200, so the
whole input fits on a few lines and the expected pieces can be read at a
glance. The rule is the same at any limit.
"""

import hashlib
import time

import pytest

from guide_search.chunking import (
    ChunkingError,
    build_search_text,
    chunk_guide,
    split_long,
    split_sections,
)
from guide_search.snapshot import Guide


def make_guide(body: str) -> Guide:
    """An invented guide with the given body."""
    return Guide(
        id=7,
        slug="invented-guide",
        title="An invented guide",
        url="https://example.com/help/invented-guide",
        updated_at="2026-01-01T00:00:00+00:00",
        body=body,
    )


def headings_of(body: str) -> list[str]:
    """The heading of every passage the body gives, in order."""
    return [passage.heading for passage in chunk_guide(make_guide(body))]


# ---------------------------------------------------------------------------
# split_sections: cutting a body at its "## " lines
# ---------------------------------------------------------------------------


def test_text_before_the_first_heading_is_a_section_with_an_empty_heading():
    body = "Opening words.\n\n## First\n\nFirst text.\n\n## Second\n\nSecond text."

    assert split_sections(body) == [
        ("", "Opening words."),
        ("First", "First text."),
        ("Second", "Second text."),
    ]


def test_a_body_that_starts_with_a_heading_has_no_intro_section():
    body = "## First\n\nFirst text."

    assert split_sections(body) == [("First", "First text.")]


def test_a_section_with_no_text_is_dropped():
    body = "## Nothing\n\n## Only spaces\n   \n\n## Full\n\nSome text."

    assert split_sections(body) == [("Full", "Some text.")]


def test_section_text_is_trimmed_and_does_not_include_the_heading_line():
    body = "## Steps\n\n\n  First line.\nSecond line.  \n\n\n## Next\n\nMore."

    assert split_sections(body)[0] == ("Steps", "First line.\nSecond line.")


def test_only_a_line_that_starts_with_two_hashes_and_a_space_is_a_heading():
    body = "## Real\n\n### Three hashes\n##No space\nA line with ## inside it."

    assert split_sections(body) == [
        ("Real", "### Three hashes\n##No space\nA line with ## inside it."),
    ]


# ---------------------------------------------------------------------------
# chunk_guide: which sections become passages
# ---------------------------------------------------------------------------


def test_the_intro_becomes_a_passage_with_an_empty_heading():
    passages = chunk_guide(make_guide("Opening words.\n\n## First\n\nFirst text."))

    assert passages[0].heading == ""
    assert passages[0].content == "Opening words."


def test_a_guide_with_no_intro_has_no_intro_passage():
    body = "## First\n\nFirst text.\n\n## Second\n\nSecond text."

    assert headings_of(body) == ["First", "Second"]


def test_a_section_with_no_text_gives_no_passage():
    body = "## Nothing\n\n## Full\n\nSome text."

    assert headings_of(body) == ["Full"]


def test_the_section_headed_exactly_related_articles_is_left_out():
    body = (
        "## Steps\n\nDo the thing.\n\n"
        "## Related articles\n\n- [Another guide](/another-guide)"
    )

    assert headings_of(body) == ["Steps"]


def test_a_heading_that_only_begins_with_related_articles_is_kept():
    body = (
        "## Steps\n\nDo the thing.\n\n"
        "## Related articles and more\n\nUseful text."
    )

    assert headings_of(body) == ["Steps", "Related articles and more"]


def test_a_heading_that_differs_from_related_articles_only_in_capitals_is_kept():
    # The rule is "exactly", capitals included. A differently written heading
    # may hold real text, so it is indexed rather than silently dropped.
    body = (
        "## Steps\n\nDo the thing.\n\n"
        "## Related Articles\n\nUseful text."
    )

    assert headings_of(body) == ["Steps", "Related Articles"]


# ---------------------------------------------------------------------------
# chunk_guide: the 1,200 character limit
# ---------------------------------------------------------------------------


def test_a_section_of_exactly_1200_characters_is_one_passage():
    text = "a" * 600 + "\n" + "b" * 599
    assert len(text) == 1200

    passages = chunk_guide(make_guide("## Steps\n\n" + text))

    assert [passage.content for passage in passages] == [text]


def test_a_section_of_1201_characters_is_split():
    first_line = "a" * 600
    second_line = "b" * 600
    text = first_line + "\n" + second_line
    assert len(text) == 1201

    passages = chunk_guide(make_guide("## Steps\n\n" + text))

    assert [passage.content for passage in passages] == [first_line, second_line]


def test_every_piece_of_a_split_section_keeps_the_section_heading():
    body = "## Steps\n\n" + "a" * 600 + "\n" + "b" * 600

    assert headings_of(body) == ["Steps", "Steps"]


def test_a_line_of_exactly_1200_characters_is_allowed():
    body = "## Steps\n\nA short line.\n" + "x" * 1200

    passages = chunk_guide(make_guide(body))

    assert [passage.content for passage in passages] == ["A short line.", "x" * 1200]


def test_a_line_over_1200_characters_stops_with_an_error_naming_the_guide():
    body = "## Steps\n\nA short line.\n" + "x" * 1201

    with pytest.raises(ChunkingError, match="invented-guide"):
        chunk_guide(make_guide(body))


# ---------------------------------------------------------------------------
# chunk_guide: what each passage keeps
# ---------------------------------------------------------------------------


def test_positions_count_up_across_the_whole_guide():
    # An empty section, a split section and a left-out section must not
    # leave a gap in the numbering or restart it.
    body = (
        "Opening words.\n\n"
        "## Nothing\n\n"
        "## Long\n\n" + "a" * 600 + "\n" + "b" * 600 + "\n\n"
        "## Related articles\n\n- [Another guide](/another-guide)\n\n"
        "## Last\n\nClosing words."
    )

    passages = chunk_guide(make_guide(body))

    assert [(passage.position, passage.heading) for passage in passages] == [
        (0, ""),
        (1, "Long"),
        (2, "Long"),
        (3, "Last"),
    ]


def test_a_passage_keeps_the_id_slug_title_and_url_of_its_guide():
    passage = chunk_guide(make_guide("## Steps\n\nDo the thing."))[0]

    assert passage.guide_id == 7
    assert passage.slug == "invented-guide"
    assert passage.title == "An invented guide"
    assert passage.url == "https://example.com/help/invented-guide"


def test_search_text_is_title_heading_and_content_joined_by_newlines():
    search_text = build_search_text("A title", "A heading", "Some content.")

    assert search_text == "A title\nA heading\nSome content."


def test_search_text_for_the_intro_has_no_heading_line():
    search_text = build_search_text("A title", "", "Some content.")

    assert search_text == "A title\nSome content."


def test_every_passage_carries_its_own_search_text():
    body = "Opening words.\n\n## Steps\n\nDo the thing."

    passages = chunk_guide(make_guide(body))

    assert [passage.search_text for passage in passages] == [
        "An invented guide\nOpening words.",
        "An invented guide\nSteps\nDo the thing.",
    ]


def test_content_hash_is_the_sha256_of_search_text():
    # The euro sign is there so the test would notice a wrong text encoding.
    passage = chunk_guide(make_guide("## Prices\n\nA wash costs 5 €."))[0]

    expected = hashlib.sha256(
        "An invented guide\nPrices\nA wash costs 5 €.".encode("utf-8")
    ).hexdigest()
    assert passage.content_hash == expected


# ---------------------------------------------------------------------------
# split_long: where a long section is cut
# ---------------------------------------------------------------------------


def test_text_within_the_limit_comes_back_as_one_piece():
    text = "one\n\ntwo"
    assert len(text) == 8

    assert split_long(text, limit=8) == [text]


def test_a_long_section_of_short_paragraphs_is_cut_at_a_blank_line():
    text = "aaaa aaaa\n\nbbbb bbbb\n\ncccc cccc\n\ndddd dddd"

    pieces = split_long(text, limit=25)

    assert pieces == ["aaaa aaaa\n\nbbbb bbbb", "cccc cccc\n\ndddd dddd"]


def test_a_paragraph_that_fits_is_never_cut_inside():
    # Cutting the second paragraph in half would give more even pieces,
    # but a paragraph within the limit always stays whole.
    first = "aaaa aaaa"
    second = "bbbb bbbb\ncccc cccc\ndddd dddd\neeee eeee"

    pieces = split_long(first + "\n\n" + second, limit=40)

    assert pieces == [first, second]


def test_a_paragraph_over_the_limit_is_cut_at_a_line_boundary():
    text = "- item aa\n- item bb"

    pieces = split_long(text, limit=15)

    assert pieces == ["- item aa", "- item bb"]


def test_a_short_paragraph_shares_a_piece_with_the_first_lines_of_a_long_list():
    text = (
        "Do these:\n\n"
        "- item aa\n- item bb\n- item cc\n- item dd\n- item ee\n- item ff"
    )

    pieces = split_long(text, limit=40)

    assert pieces == [
        "Do these:\n\n- item aa\n- item bb",
        "- item cc\n- item dd\n- item ee\n- item ff",
    ]


def test_separators_count_toward_the_length_of_a_piece():
    # Two paragraphs are 10 characters of text, which would fit in 11.
    # With the blank line between them they are 12, so each stands alone.
    text = "aaaaa\n\nbbbbb\n\nccccc"

    pieces = split_long(text, limit=11)

    assert pieces == ["aaaaa", "bbbbb", "ccccc"]


def test_the_smallest_number_of_pieces_is_used():
    # One paragraph per piece would make every piece shorter, but two of
    # these paragraphs fit in 10 characters, so two pieces are enough.
    text = "aaaa\n\nbbbb\n\ncccc"

    pieces = split_long(text, limit=10)

    assert len(pieces) == 2


def test_the_longest_piece_is_as_short_as_possible():
    # Filling the first piece as full as it goes would take three lines
    # and leave "- item dd" as a stray last line. Two and two is better.
    text = "- item aa\n- item bb\n- item cc\n- item dd"

    pieces = split_long(text, limit=30)

    assert pieces == ["- item aa\n- item bb", "- item cc\n- item dd"]


def test_a_shorter_longest_piece_beats_an_earlier_cut():
    # Cutting after the first line is earlier, but it leaves a second piece
    # of 15 characters. Cutting after the second gives 14 and 9.
    text = "- wheels\n- oil\n- gearbox"

    pieces = split_long(text, limit=20)

    assert pieces == ["- wheels\n- oil", "- gearbox"]


def test_the_earlier_cut_wins_a_tie():
    # Cutting after the first line or after the second both give two
    # pieces whose longest is 16 characters.
    text = "aaaaaaaaaa\nbbbbb\ncccccccccc"

    pieces = split_long(text, limit=16)

    assert pieces == ["aaaaaaaaaa", "bbbbb\ncccccccccc"]


def test_pieces_do_not_overlap_and_rejoin_to_the_section_text():
    # The first cut falls on a blank line and the second on a line break.
    text = (
        "Read this first,\nthen go on.\n\n"
        "- item aa\n- item bb\n- item cc\n- item dd"
    )

    pieces = split_long(text, limit=30)

    assert len(pieces) == 3
    assert pieces[0] + "\n\n" + pieces[1] + "\n" + pieces[2] == text


def test_an_unusual_separator_inside_a_piece_is_kept_as_it_was():
    # Two blank lines between paragraphs, where one is usual.
    text = "aaaa\n\n\nbbbb\n\ncccc\n\n\ndddd"

    pieces = split_long(text, limit=12)

    assert pieces == ["aaaa\n\n\nbbbb", "cccc\n\n\ndddd"]


def test_split_long_refuses_a_single_line_over_the_limit():
    text = "A short line.\n" + "x" * 31

    with pytest.raises(ChunkingError):
        split_long(text, limit=30)


def test_two_hundred_units_are_split_into_ten_pieces_in_under_a_second():
    # Twenty of these lines, joined by line breaks, are 1,199 characters.
    line = "- " + "x" * 57
    text = "\n".join([line] * 200)

    started = time.perf_counter()
    pieces = split_long(text)
    seconds = time.perf_counter() - started

    assert len(pieces) == 10
    assert seconds < 1.0
