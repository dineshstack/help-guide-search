"""Facts about the committed snapshot, data/guides.json.

These numbers describe the snapshot of 10 October 2026. The design quotes
them (sections 2, 4.1 and 4.2) and some of its choices rest on them.

They are updated on purpose when the snapshot is refreshed. A failure here
right after a refresh is not a bug in the code: it means the guides changed.
Measure the new figures, then change this file and the design together.
"""

from pathlib import Path

import pytest

from guide_search.chunking import (
    MAX_CHARS,
    Passage,
    chunk_guide,
    split_long,
    split_sections,
)
from guide_search.snapshot import Guide, load_snapshot

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "guides.json"


@pytest.fixture(scope="module")
def guides() -> list[Guide]:
    return load_snapshot(SNAPSHOT_PATH)


@pytest.fixture(scope="module")
def passages_per_guide(guides: list[Guide]) -> list[list[Passage]]:
    return [chunk_guide(guide) for guide in guides]


@pytest.fixture(scope="module")
def passages(passages_per_guide: list[list[Passage]]) -> list[Passage]:
    return [passage for group in passages_per_guide for passage in group]


def headings_in(guide: Guide) -> list[str]:
    """The text of every "## " line in a guide, read straight from the body."""
    return [
        line[3:].strip()
        for line in guide.body.split("\n")
        if line.startswith("## ")
    ]


# ---------------------------------------------------------------------------
# The guides themselves (design sections 2 and 4.1)
# ---------------------------------------------------------------------------


def test_there_are_60_guides(guides):
    assert len(guides) == 60


def test_the_ids_are_1_to_59_and_61(guides):
    assert [guide.id for guide in guides] == list(range(1, 60)) + [61]


def test_bodies_are_between_1747_and_4198_characters(guides):
    lengths = [len(guide.body) for guide in guides]

    assert (min(lengths), max(lengths)) == (1747, 4198)


def test_there_are_332_headings_in_total(guides):
    assert sum(len(headings_in(guide)) for guide in guides) == 332


def test_every_guide_has_between_3_and_23_headings(guides):
    counts = [len(headings_in(guide)) for guide in guides]

    assert (min(counts), max(counts)) == (3, 23)


def test_two_hashes_is_the_only_heading_level(guides):
    # Chunking cuts only at "## ". A heading of another level would be
    # treated as ordinary text without anyone noticing.
    other_levels = [
        line
        for guide in guides
        for line in guide.body.split("\n")
        if line.startswith("#") and not line.startswith("## ")
    ]

    assert other_levels == []


def test_59_guides_have_text_before_the_first_heading(guides):
    with_intro = 0
    for guide in guides:
        first_heading, first_text = split_sections(guide.body)[0]
        if first_heading == "":
            with_intro += 1

    assert with_intro == 59


def test_related_articles_is_the_last_section_of_every_guide(guides):
    last_headings = {headings_in(guide)[-1] for guide in guides}

    assert last_headings == {"Related articles"}


# ---------------------------------------------------------------------------
# What chunking makes of them (design section 4.2)
# ---------------------------------------------------------------------------


def test_chunking_gives_335_passages(passages):
    assert len(passages) == 335


def test_every_guide_gives_between_4_and_22_passages(passages_per_guide):
    counts = [len(group) for group in passages_per_guide]

    assert (min(counts), max(counts)) == (4, 22)


def test_the_shortest_passage_is_71_characters_and_the_longest_1143(passages):
    lengths = [len(passage.content) for passage in passages]

    assert (min(lengths), max(lengths)) == (71, 1143)


def test_every_passage_has_between_1_and_1200_characters(passages):
    outside = [
        (passage.slug, passage.position, len(passage.content))
        for passage in passages
        if not 1 <= len(passage.content) <= 1200
    ]

    assert outside == []


def test_exactly_four_sections_are_split_and_each_into_two_pieces(guides):
    piece_counts = [
        len(split_long(text))
        for guide in guides
        for heading, text in split_sections(guide.body)
        if heading != "Related articles" and len(text) > MAX_CHARS
    ]

    assert piece_counts == [2, 2, 2, 2]


def test_no_passage_is_headed_related_articles(passages):
    assert "Related articles" not in {passage.heading for passage in passages}
