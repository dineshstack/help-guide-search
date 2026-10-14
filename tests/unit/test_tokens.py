"""Tests for the token counts: the texts that are counted, and counting each text once.

No API is called: the counter is a stand-in that counts words.
"""

import json
from pathlib import Path

from guide_search.evaluation.tokens import CachedCounter, text_hash, title_picking_text
from guide_search.snapshot import Guide


def guide(number: int, slug: str, body: str) -> Guide:
    return Guide(
        id=number,
        slug=slug,
        title=slug.replace("-", " ").capitalize(),
        url=f"https://example.test/help/{slug}",
        updated_at="2026-01-01",
        body=body,
    )


GUIDES = [guide(1, "mend-a-sail", "Patch the tear."), guide(2, "fit-a-spar", "Slide the spar in.")]


class WordCounter:
    """Counts words, and remembers every text it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, model: str, text: str) -> int:
        self.texts.append(text)
        return len(text.split())


def test_the_title_picking_text_is_the_list_of_every_guide_then_each_picked_guide_whole():
    text = title_picking_text(GUIDES, ["fit-a-spar"])

    assert text == (
        "1 | mend-a-sail | Mend a sail\n"
        "2 | fit-a-spar | Fit a spar\n"
        "\n---\n\n"
        "Guide id 2: Fit a spar\n"
        "Web address: https://example.test/help/fit-a-spar\n"
        "\n"
        "Slide the spar in."
    )


def test_picking_nothing_still_means_reading_the_list_of_titles():
    assert title_picking_text(GUIDES, []) == "1 | mend-a-sail | Mend a sail\n2 | fit-a-spar | Fit a spar"


def test_each_text_is_counted_once(tmp_path: Path):
    words = WordCounter()
    counter = CachedCounter(tmp_path / "counts.json", "model-a", words)

    assert counter.tokens("one two three") == 3
    assert counter.tokens("one two three") == 3
    assert words.texts == ["one two three"]
    assert counter.sent == 1


def test_saved_counts_are_used_again_without_counting(tmp_path: Path):
    path = tmp_path / "counts.json"
    first = CachedCounter(path, "model-a", WordCounter())
    first.tokens("one two three")
    first.save()

    words = WordCounter()
    again = CachedCounter(path, "model-a", words)

    assert again.tokens("one two three") == 3
    assert words.texts == []
    assert json.loads(path.read_text())["counts"] == {text_hash("one two three"): 3}


def test_counts_made_for_another_model_are_not_used(tmp_path: Path):
    path = tmp_path / "counts.json"
    first = CachedCounter(path, "model-a", WordCounter())
    first.tokens("one two three")
    first.save()

    words = WordCounter()
    other = CachedCounter(path, "model-b", words)
    other.tokens("one two three")

    assert words.texts == ["one two three"]


# --- matching the counts to a run's results ---------------------------------


def outcome(question_id: str, slugs: tuple[str, ...]):
    from guide_search.evaluation.metrics import Outcome

    return Outcome(question_id, "rerank", slugs == (), slugs, {}, 0, 0.1, None)


def entry(question_id: str, slugs: list[str], tokens: int = 500) -> dict:
    return {"question_id": question_id, "method": "rerank", "slugs": slugs, "refused": slugs == [], "tokens": tokens}


def test_counts_made_from_the_same_results_are_all_kept():
    from guide_search.evaluation.run import matching_tokens

    tokens = {"model": "m", "date": "d", "entries": [entry("en-001", ["a", "b"]), entry("na-001", [], 0)]}
    outcomes = {"rerank": {"en-001": outcome("en-001", ("a", "b")), "na-001": outcome("na-001", ())}}

    kept = matching_tokens(tokens, outcomes)

    assert kept["entries"] == tokens["entries"]
    assert kept["left_out"] == 0


def test_a_count_made_from_another_result_is_left_out_and_counted():
    from guide_search.evaluation.run import matching_tokens

    # On the other machine a near-tie went the other way: b came before a.
    tokens = {"model": "m", "date": "d", "entries": [entry("en-001", ["b", "a"]), entry("en-002", ["c"])]}
    outcomes = {"rerank": {"en-001": outcome("en-001", ("a", "b")), "en-002": outcome("en-002", ("c",))}}

    kept = matching_tokens(tokens, outcomes)

    assert [item["question_id"] for item in kept["entries"]] == ["en-002"]
    assert kept["left_out"] == 1


def test_no_token_file_gives_no_counts():
    from guide_search.evaluation.run import matching_tokens

    assert matching_tokens(None, {}) is None
