"""Tests for the check that every draft must pass.

Each test builds a small invented search result and a draft by hand, and says
whether the draft may be returned. There is no model and no database here.
"""

import pytest

from guide_search.checks import (
    MAX_REPLY_CHARS,
    Draft,
    check_draft,
    links_in,
    numbers_in,
)
from guide_search.ranking import GuideHit, PassageHit

PAGE = "https://support.orions360.com/help/en/autoserv-pro/"


def guide(guide_id: int, slug: str, *texts: str) -> GuideHit:
    passages = [
        PassageHit(heading=f"Section {number}", position=number, text=text, similarity=0.9)
        for number, text in enumerate(texts)
    ]
    return GuideHit(
        guide_id=guide_id, slug=slug, title=slug.replace("-", " "), url=PAGE + slug, score=0.9, passages=passages
    )


# What search returned, for most tests: two guides.
LINKS = guide(
    3,
    "share-a-link",
    "The link stops working after 48 hours. Send a new one from the customer page.",
    "A photo may be up to 8 MB. Upload at most ten images. See [Pricing](/help/en/autoserv-pro/plans) "
    "or https://app.example.com/start for the rest.",
)
STOCK = guide(7, "count-stock", "Stock is counted in units of 1,000. The smallest step is 0.50.")
GUIDES = [LINKS, STOCK]


def answer(reply: str, *cited: int) -> Draft:
    return Draft(decision="answer", reply=reply, cited_guide_ids=tuple(cited))


def code_of(draft: Draft, guides: list[GuideHit] = GUIDES) -> str | None:
    failure = check_draft(draft, guides)
    return None if failure is None else failure.code


# ---------------------------------------------------------------------------
# Reading numbers and links out of a text
# ---------------------------------------------------------------------------


def test_numbers_are_read_with_separators_and_decimals_written_one_way():
    assert numbers_in("Up to 1,000 items, at 0.50 each, for 14-day trials and 8MB files.") == [
        "1000",
        "0.5",
        "14",
        "8",
    ]


def test_arabic_digits_are_read_as_ordinary_digits():
    assert numbers_in("ينتهي الرابط بعد ٤٨ ساعة") == ["48"]
    assert numbers_in("۴۸") == ["48"]


def test_digits_that_number_the_steps_of_a_list_are_not_numbers():
    reply = "Do this:\n1. Open the page.\n2) Press Send.\n  3. Wait 48 hours."

    assert numbers_in(reply) == ["48"]


def test_digits_inside_a_name_or_a_link_are_not_numbers():
    assert numbers_in("Orions360 has a page at https://example.com/v2/page9") == []


def test_a_list_of_numbers_is_read_one_by_one():
    assert numbers_in("Terms of 15, 30, 45 or 60 days.") == ["15", "30", "45", "60"]


def test_links_are_read_and_written_one_way():
    assert links_in("See https://Example.com/Guide/ and www.example.com/x.") == [
        "https://example.com/Guide",
        "https://www.example.com/x",
    ]


def test_a_link_in_brackets_is_read_without_the_bracket():
    assert links_in("Read [the guide](https://example.com/guide).") == ["https://example.com/guide"]


# ---------------------------------------------------------------------------
# An answer
# ---------------------------------------------------------------------------


def test_an_answer_that_cites_a_retrieved_guide_and_quotes_its_numbers_passes():
    assert code_of(answer("The link stops working after 48 hours.", 3)) is None


def test_an_answer_with_no_citation_fails():
    assert code_of(answer("The link stops working after 48 hours.")) == "no_citation"


def test_an_answer_that_cites_a_guide_search_did_not_return_fails():
    failure = check_draft(answer("The link stops working after 48 hours.", 3, 99), GUIDES)

    assert failure is not None
    assert failure.code == "unknown_citation"
    assert "99" in failure.message


def test_a_number_that_is_not_in_the_cited_passages_fails_and_is_named():
    failure = check_draft(answer("The link stops working after 72 hours.", 3), GUIDES)

    assert failure is not None
    assert failure.code == "unsupported_number"
    assert "72" in failure.message


def test_a_number_from_a_retrieved_guide_that_is_not_cited_does_not_count():
    # 1,000 is in guide 7, but the answer cites only guide 3.
    assert code_of(answer("Stock is counted in units of 1,000.", 3)) == "unsupported_number"
    assert code_of(answer("Stock is counted in units of 1,000.", 7)) is None


def test_a_number_written_as_a_word_in_the_passage_supports_digits_in_the_reply():
    # The passage says "ten images".
    assert code_of(answer("You can upload at most 10 images.", 3)) is None


def test_numbers_match_whatever_their_separators():
    assert code_of(answer("Units of 1000, in steps of 0.5.", 7)) is None


def test_arabic_digits_in_a_reply_are_checked_like_any_others():
    assert code_of(answer("ينتهي الرابط بعد ٤٨ ساعة.", 3)) is None
    assert code_of(answer("ينتهي الرابط بعد ٧٢ ساعة.", 3)) == "unsupported_number"


def test_step_numbers_in_a_reply_are_not_checked():
    reply = "1. Open the customer page.\n2. Send a new link.\n3. It works for 48 hours."

    assert code_of(answer(reply, 3)) is None


def test_the_page_of_a_cited_guide_is_a_supported_link():
    assert code_of(answer(f"See {PAGE}share-a-link for the steps.", 3)) is None


def test_the_page_of_a_guide_that_is_not_cited_is_not_a_supported_link():
    assert code_of(answer(f"See {PAGE}count-stock for the steps.", 3)) == "unsupported_link"


def test_a_link_inside_a_cited_passage_is_supported():
    assert code_of(answer("Start at https://app.example.com/start.", 3)) is None


def test_a_link_written_without_the_site_inside_a_passage_is_supported_in_full():
    # The passage has [Pricing](/help/en/autoserv-pro/plans).
    assert code_of(answer(f"Prices are at {PAGE}plans.", 3)) is None


def test_an_invented_link_fails_and_is_named():
    failure = check_draft(answer("See https://example.org/made-up for more.", 3), GUIDES)

    assert failure is not None
    assert failure.code == "unsupported_link"
    assert "example.org/made-up" in failure.message


def test_an_empty_answer_fails():
    assert code_of(answer("   ", 3)) == "empty_reply"


def test_an_answer_longer_than_the_limit_fails():
    assert code_of(answer("x" * (MAX_REPLY_CHARS + 1), 3)) == "reply_too_long"
    assert code_of(answer("x" * MAX_REPLY_CHARS, 3)) is None


def test_an_answer_with_nothing_to_cite_cannot_pass():
    # Search returned nothing at all.
    assert code_of(answer("The link stops working after a while.", 3), []) == "unknown_citation"
    assert code_of(answer("The link stops working after a while."), []) == "no_citation"


def test_the_first_broken_rule_is_the_one_reported():
    # No citation, and an unsupported number: the citation rule comes first.
    assert code_of(answer("It lasts 72 hours.")) == "no_citation"


# ---------------------------------------------------------------------------
# A clarifying question and a hand-over
# ---------------------------------------------------------------------------


def test_a_plain_clarifying_question_passes():
    draft = Draft(decision="clarify", reply="Which page are you on when this happens?", cited_guide_ids=())

    assert code_of(draft) is None


def test_a_clarifying_question_that_cites_a_guide_fails():
    draft = Draft(decision="clarify", reply="Which page are you on?", cited_guide_ids=(3,))

    assert code_of(draft) == "clarify_cites_a_guide"


@pytest.mark.parametrize(
    "reply",
    ["Is it the link that lasts 48 hours?", "Did you try https://example.com/reset first?"],
)
def test_a_clarifying_question_with_a_number_or_a_link_fails(reply: str):
    draft = Draft(decision="clarify", reply=reply, cited_guide_ids=())

    assert code_of(draft) == "clarify_states_a_fact"


def test_a_hand_over_always_passes():
    draft = Draft(decision="hand_over", reply="", cited_guide_ids=())

    assert code_of(draft) is None


def test_a_decision_that_is_none_of_the_three_fails():
    draft = Draft(decision="guess", reply="Maybe.", cited_guide_ids=())

    assert code_of(draft) == "unknown_decision"
