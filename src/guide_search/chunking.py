"""Chunking: cut one guide into several passages.

A whole guide is too long to search as one block, so each guide is cut into
passages and the passages are what gets searched.

The rule, in short (the design, section 4.2, has the reasons):

1. Cut the body at lines that start with "## ". The text before the first
   such line is the intro, a section with an empty heading.
2. Leave out the section headed exactly "Related articles".
3. A section of 1,200 characters or fewer is one passage.
4. A longer section is cut into the fewest pieces that fit, at blank lines
   where it can and at line breaks inside a paragraph that is itself too long.
5. A single line over 1,200 characters is an error. An over-long passage is
   never produced.

This file only works on text. It does not read files or talk to a database.
"""

import hashlib
import re
from dataclasses import dataclass

from guide_search.snapshot import Guide

# The embedding model reads at most 512 tokens and silently ignores the rest.
# 1,200 characters, plus the title and heading, is expected to stay inside that.
MAX_CHARS = 1200

# This section is a list of links to other guides and holds no answer text.
RELATED_HEADING = "Related articles"

_HEADING_PREFIX = "## "

# What separates two paragraphs: a line break, any blank lines, a line break.
# A line that holds only spaces counts as blank. The round brackets make
# re.split hand the separators back too, so they are kept exactly as they were.
_PARAGRAPH_BREAK = re.compile(r"(\n\s*\n)")


@dataclass(frozen=True)
class Passage:
    """One searchable piece of a guide."""

    guide_id: int
    slug: str
    title: str
    url: str
    heading: str        # "" for the intro
    position: int       # 0, 1, 2, ... within the guide
    content: str        # passage text, as shown to a reader
    search_text: str    # title, heading and content: what is actually searched
    content_hash: str   # sha256 of search_text, so a change to it can be seen


class ChunkingError(ValueError):
    """A guide cannot be cut into passages of at most MAX_CHARS characters."""


@dataclass(frozen=True)
class _Unit:
    """The smallest thing a long section is cut into: a paragraph or one line."""

    text: str
    separator: str      # what came after it in the section; "" after the last unit


def split_sections(body: str) -> list[tuple[str, str]]:
    """Cut a guide body into (heading, text) pairs, in the order they appear.

    The heading is given without its "## ". The text is trimmed. The intro has
    the heading "". A section with no text is dropped.
    """
    sections: list[tuple[str, str]] = []
    heading = ""                # until the first "## " line we are in the intro
    lines: list[str] = []
    for line in body.split("\n"):
        if line.startswith(_HEADING_PREFIX):
            _add_section(sections, heading, lines)
            heading = line[len(_HEADING_PREFIX):].strip()
            lines = []
        else:
            lines.append(line)
    _add_section(sections, heading, lines)
    return sections


def _add_section(
    sections: list[tuple[str, str]], heading: str, lines: list[str]
) -> None:
    """Add the section to the list, unless it has no text."""
    text = "\n".join(lines).strip()
    if text:
        sections.append((heading, text))


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Cut a section's text into pieces of at most `limit` characters.

    The text must already be trimmed, as split_sections returns it. Text that
    fits comes back as a single piece. Pieces do not overlap: joined again
    with the separators that stood between them, they give back the text.
    """
    if len(text) <= limit:
        return [text]
    units = _split_units(text, limit)
    # Step 1: how few pieces can it be? Filling every piece as full as it
    # goes never uses more pieces than any other way of cutting.
    fewest = len(_pack_from_end(units, limit))
    # Step 2: how short can the longest of that many pieces be? Packing to
    # that size instead of to the limit spreads the text evenly, so no piece
    # is one stray line.
    cap = _smallest_cap(units, fewest, limit)
    # Step 3: pack to that size. Filling from the end puts every cut as
    # early as it can go, which is the rule for a tie.
    return _pack_from_end(units, cap)


def _split_units(text: str, limit: int) -> list[_Unit]:
    """Cut text into paragraphs, and any over-long paragraph into its lines."""
    # re.split gives: paragraph, separator, paragraph, separator, ..., paragraph.
    parts = _PARAGRAPH_BREAK.split(text)
    paragraphs = parts[0::2]            # every second item, from the first
    separators = parts[1::2] + [""]     # nothing comes after the last paragraph
    units: list[_Unit] = []
    for paragraph, separator in zip(paragraphs, separators):
        if len(paragraph) <= limit:
            units.append(_Unit(paragraph, separator))
        else:
            units.extend(_line_units(paragraph, separator, limit))
    return units


def _line_units(paragraph: str, separator: str, limit: int) -> list[_Unit]:
    """Turn an over-long paragraph (in practice a long list) into its lines."""
    lines = paragraph.split("\n")
    for line in lines:
        if len(line) > limit:
            raise ChunkingError(
                f"a single line is {len(line)} characters long and the limit "
                f"is {limit}; a line is never cut"
            )
    units = [_Unit(line, "\n") for line in lines[:-1]]
    # The last line is followed by whatever followed the whole paragraph.
    units.append(_Unit(lines[-1], separator))
    return units


def _pack_from_end(units: list[_Unit], cap: int) -> list[str]:
    """Pack the units into pieces of at most `cap` characters.

    It starts with the last unit and works backwards, adding units to the
    front of the current piece until the next one would not fit. No unit may
    be longer than `cap`.
    """
    pieces: list[str] = []
    piece = units[-1].text
    for unit in reversed(units[:-1]):
        longer = unit.text + unit.separator + piece
        if len(longer) <= cap:
            piece = longer
        else:
            # The separator after this unit now lies between two pieces,
            # so it belongs to neither.
            pieces.append(piece)
            piece = unit.text
    pieces.append(piece)
    pieces.reverse()        # they were collected last piece first
    return pieces


def _smallest_cap(units: list[_Unit], pieces_wanted: int, limit: int) -> int:
    """The smallest piece size that still needs only `pieces_wanted` pieces.

    A larger size never needs more pieces, so the answer can be found by
    halving the range instead of trying every size: guess the middle, and
    keep the half that can still hold the answer.
    """
    # A piece holds at least one whole unit, so no smaller size can work.
    low = max(len(unit.text) for unit in units)
    # The limit is known to work: it is the size pieces_wanted was counted at.
    high = limit
    while low < high:
        middle = (low + high) // 2
        if len(_pack_from_end(units, middle)) <= pieces_wanted:
            high = middle           # middle works; the answer is middle or smaller
        else:
            low = middle + 1        # middle is too small
    return low


def build_search_text(title: str, heading: str, content: str) -> str:
    """The text that is embedded and keyword-indexed for one passage.

    A passage on its own is often too short to mean much, so the guide title
    and the section heading always go in front of it. The intro has no
    heading, so it gets no heading line.
    """
    if heading == "":
        return f"{title}\n{content}"
    return f"{title}\n{heading}\n{content}"


def chunk_guide(guide: Guide) -> list[Passage]:
    """Turn one guide into its passages, in reading order."""
    passages: list[Passage] = []
    for heading, text in split_sections(guide.body):
        if heading == RELATED_HEADING:
            continue
        try:
            pieces = split_long(text)
        except ChunkingError as error:
            # split_long does not know which guide it is working on, so the
            # guide and the section are named here.
            where = f"section '{heading}'" if heading else "the intro"
            message = f"guide '{guide.slug}', {where}: {error}"
            raise ChunkingError(message) from error
        for content in pieces:
            search_text = build_search_text(guide.title, heading, content)
            passages.append(
                Passage(
                    guide_id=guide.id,
                    slug=guide.slug,
                    title=guide.title,
                    url=guide.url,
                    heading=heading,
                    position=len(passages),     # counts on across sections
                    content=content,
                    search_text=search_text,
                    content_hash=_sha256(search_text),
                )
            )
    return passages


def _sha256(text: str) -> str:
    """The sha256 of the text, as 64 hexadecimal characters."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
