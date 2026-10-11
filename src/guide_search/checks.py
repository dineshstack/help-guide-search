"""The check: plain rules a draft must pass before anyone sees it.

The answer agent asks a model for a draft. This file decides, with ordinary
code and no model, whether that draft may be returned. The rules:

- an "answer" must cite at least one guide, and only guides that search
  returned; every number and every link in it must appear in the passages of
  the guides it cites;
- a "clarify" (a question back to the user) must cite nothing and state no
  number and no link;
- a "hand_over" always passes: it goes straight to a person.

A "cited passage" is a passage that search returned for a guide the draft
cites. The rest of that guide does not count, because the model never saw it.

What the check does NOT do, stated plainly: it does not test whether each
sentence is supported by the guides. A fluent wrong answer with no number and
no link in it passes, if it cites a guide that was retrieved. A number written
as a word in the reply ("eight") is not checked either.

Nothing here talks to a model or a database, so every rule can be tested with
a few invented values.
"""

import re
from dataclasses import dataclass

from guide_search.ranking import GuideHit

DECISIONS = ("answer", "clarify", "hand_over")

# The longest reply that is accepted.
MAX_REPLY_CHARS = 1500

# Links inside a guide are written without the site ("/help/en/..."). They
# are completed with this address before they are compared.
HELP_CENTRE = "https://support.orions360.com"

# Arabic-Indic and Eastern Arabic digits, read as ordinary digits.
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# A passage may write a number as a word where the reply uses digits.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}  # fmt: skip

# A web address: "http://...", "https://..." or "www....".
LINK_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]\"']+", re.IGNORECASE)

# A link written the Markdown way inside a guide: "[text](target)".
MARKDOWN_TARGET = re.compile(r"\]\(([^)\s]+)\)")

# Digits that start a number: not glued to a letter or digit before them, so
# the "360" inside a name like "Orions360" is not read as a number. A comma
# counts as part of the number only when it separates thousands ("1,000").
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?")

# "1. " or "2) " at the start of a line only numbers a step of a list.
LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+")


@dataclass(frozen=True)
class Draft:
    """What the model wrote: its decision, the reply, and the guides it relied on."""

    decision: str
    reply: str
    cited_guide_ids: tuple[int, ...]


@dataclass(frozen=True)
class Failure:
    """The first rule a draft broke."""

    code: str  # a short name, kept in the result
    message: str  # plain words, given to the model for its one rewrite


def _normalise_number(number: str) -> str:
    """Write a number one way, so "1,000" equals "1000" and "0.50" equals "0.5"."""
    plain = number.replace(",", "")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    plain = plain.lstrip("0")
    if plain == "" or plain.startswith("."):
        plain = "0" + plain
    return plain


def numbers_in(text: str) -> list[str]:
    """Return the numbers stated in a text, in order, each written one way.

    Left out on purpose: digits that only number the steps of a list, and
    digits inside a web address.
    """
    readable = text.translate(ARABIC_DIGITS)
    readable = LINK_PATTERN.sub(" ", readable)
    readable = LIST_MARKER.sub("", readable)
    return [_normalise_number(found) for found in NUMBER_PATTERN.findall(readable)]


def _normalise_link(link: str) -> str:
    """Write a link one way: with "https://", lower-case site, no trailing slash or dot."""
    cleaned = link.rstrip(".,;:!?")
    if cleaned.lower().startswith("www."):
        cleaned = "https://" + cleaned
    if cleaned.startswith("/"):
        cleaned = HELP_CENTRE + cleaned
    scheme, separator, rest = cleaned.partition("://")
    site, slash, path = rest.partition("/")
    return (scheme.lower() + separator + site.lower() + slash + path).rstrip("/")


def links_in(text: str) -> list[str]:
    """Return the web addresses written in a text, in order, each written one way."""
    return [_normalise_link(found) for found in LINK_PATTERN.findall(text)]


def _cited_guides(draft: Draft, guides: list[GuideHit]) -> list[GuideHit]:
    return [guide for guide in guides if guide.guide_id in draft.cited_guide_ids]


def supported_numbers(guides: list[GuideHit]) -> set[str]:
    """Every number the passages of these guides state, as digits or as a word."""
    supported: set[str] = set()
    for guide in guides:
        for passage in guide.passages:
            text = passage.heading + "\n" + passage.text
            supported.update(numbers_in(text))
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in NUMBER_WORDS:
                    supported.add(NUMBER_WORDS[word])
    return supported


def supported_links(guides: list[GuideHit]) -> set[str]:
    """The public page of each of these guides, and every link inside their passages."""
    supported: set[str] = set()
    for guide in guides:
        supported.add(_normalise_link(guide.url))
        for passage in guide.passages:
            supported.update(links_in(passage.text))
            for target in MARKDOWN_TARGET.findall(passage.text):
                supported.add(_normalise_link(target))
    return supported


def check_draft(draft: Draft, guides: list[GuideHit]) -> Failure | None:
    """Return the first rule the draft breaks, or None when it may be returned.

    `guides` is what search returned for the question.
    """
    if draft.decision not in DECISIONS:
        return Failure("unknown_decision", "The decision must be answer, clarify or hand_over.")
    if draft.decision == "hand_over":
        return None

    if draft.reply.strip() == "":
        return Failure("empty_reply", "The reply is empty.")
    if len(draft.reply) > MAX_REPLY_CHARS:
        return Failure(
            "reply_too_long",
            f"The reply is {len(draft.reply)} characters long. Keep it under {MAX_REPLY_CHARS}.",
        )

    if draft.decision == "clarify":
        return _check_clarify(draft)
    return _check_answer(draft, guides)


def _check_clarify(draft: Draft) -> Failure | None:
    """A question back to the user states no facts, so it cites and quotes nothing."""
    if draft.cited_guide_ids:
        return Failure("clarify_cites_a_guide", "A clarifying question must not cite a guide.")
    if numbers_in(draft.reply) or links_in(draft.reply):
        return Failure(
            "clarify_states_a_fact",
            "A clarifying question must not contain a number or a link.",
        )
    return None


def _check_answer(draft: Draft, guides: list[GuideHit]) -> Failure | None:
    if not draft.cited_guide_ids:
        return Failure("no_citation", "An answer must cite at least one of the guides it was given.")

    retrieved = {guide.guide_id for guide in guides}
    unknown = [guide_id for guide_id in draft.cited_guide_ids if guide_id not in retrieved]
    if unknown:
        return Failure(
            "unknown_citation",
            f"Guide {unknown[0]} was not among the guides you were given. Cite only those.",
        )

    cited = _cited_guides(draft, guides)
    known_numbers = supported_numbers(cited)
    for number in numbers_in(draft.reply):
        if number not in known_numbers:
            return Failure(
                "unsupported_number",
                f"The number {number} is not in the passages of the guides you cited.",
            )

    known_links = supported_links(cited)
    for link in links_in(draft.reply):
        if link not in known_links:
            return Failure(
                "unsupported_link",
                f"The link {link} is not the page of a guide you cited and is not in its passages.",
            )
    return None
