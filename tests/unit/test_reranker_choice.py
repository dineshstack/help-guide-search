"""Tests for the rule that chooses the reranker. No model and no database."""

from guide_search.evaluation.reranker_choice import CHOSEN, REMOTE_CODE, TOO_SLOW, choose, load_note


def entry(name: str, passages: int, seconds: float) -> dict:
    return {"name": name, "loaded": True, "note": None, "passage_hit3": passages, "median_seconds": seconds}


def test_the_most_answering_passages_wins():
    entries = [entry("small", 10, 0.2), entry("large", 12, 0.8)]

    assert choose(entries) == "large"
    assert entries[1]["note"] == CHOSEN
    assert entries[0]["note"] is None


def test_a_tie_goes_to_the_faster_model():
    entries = [entry("slower", 12, 0.6), entry("faster", 12, 0.3)]

    assert choose(entries) == "faster"


def test_a_model_slower_than_the_limit_is_out_however_good():
    entries = [entry("small", 10, 0.2), entry("large", 15, 10.5)]

    assert choose(entries) == "small"
    assert entries[1]["note"] == TOO_SLOW


def test_exactly_the_limit_is_not_too_slow():
    assert choose([entry("borderline", 10, 10.0)]) == "borderline"


def test_a_model_that_could_not_be_loaded_is_skipped():
    entries = [{"name": "odd", "loaded": False, "note": "it could not be loaded"}, entry("small", 3, 0.2)]

    assert choose(entries) == "small"
    assert entries[0]["note"] == "it could not be loaded"


def test_no_model_is_chosen_when_none_qualifies():
    assert choose([entry("large", 15, 30.0)]) is None


def test_a_model_that_needs_its_own_code_is_named_as_such():
    error = ValueError("Some/repo You can inspect it.\nPlease pass the argument `trust_remote_code=True`.")

    assert load_note(error) == REMOTE_CODE


def test_the_note_on_any_other_failed_load_is_one_short_line():
    error = OSError("the download failed\nmore detail\n" + "x" * 500)

    note = load_note(error)

    assert note == "it could not be loaded (OSError: the download failed)"
