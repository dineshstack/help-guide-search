"""Tests for search against a real Postgres, in all four modes.

They need a real Postgres (TEST_DATABASE_URL). Most tests search the five
invented guides in tests/fixtures/five_guides.json. A few build their own tiny
guides, where the point of the test needs more than 20 passages or an exact
tie.

The embedder and the reranker are the fake ones. They only count words, so
their scores are on a different scale from the real models': the tests pass
their own thresholds and never the ones in settings.toml.
"""

from pathlib import Path

import psycopg
import pytest

from guide_search.embedding import FakeEmbedder
from guide_search.ingest import ingest
from guide_search.ranking import GuideDetail, PassageHit, SearchResult
from guide_search.reranking import FakeReranker
from guide_search.search import MODES, NO_MATCH, read_guide, search
from guide_search.snapshot import PUBLIC_PAGE_PREFIX, Guide, load_snapshot
from guide_search.store import IndexNotReady

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "five_guides.json"

# The tests' own "too weak to be a match" line for the fake embedder.
THRESHOLD = 0.3

# One question for each fixture guide, built from that guide's own words.
QUESTION_AND_GUIDE = [
    ("mend a torn sail", "mend-a-torn-sail"),
    ("replace a snapped spar", "replace-a-snapped-spar"),
    ("how to read the stock list", "flying-line-in-stock"),
    ("is the workshop open on sunday", "workshop-questions"),
    ("send a pickup reminder by sms", "send-a-pickup-reminder"),
]

# About the made-up workshop, but nothing the guides cover. It shares two
# everyday words ("do", "you") with one passage, so its best score is small
# but not zero.
NOT_COVERED = "do you sell bicycle bells"

# Only words that Postgres drops as too common to search for.
ONLY_COMMON_WORDS = "how do I do it"

# "How do I mend the torn sail?" No English passage contains these words.
ARABIC = "كيف أصلح الشراع الممزق؟"

# "How can I send an SMS to the customer?" One word is in Latin letters.
ARABIC_WITH_SMS = "كيف يمكنني إرسال SMS للعميل؟"


class QueryCountingEmbedder(FakeEmbedder):
    """The fake embedder, but it counts the questions it is asked to embed."""

    def __init__(self) -> None:
        self.queries = 0

    def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        return super().embed_query(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("search must embed a question with embed_query")


class OtherModelEmbedder(FakeEmbedder):
    """The fake embedder under another name, as if the model had been changed."""

    model_id = "fake-2"


def fixture_guides() -> list[Guide]:
    """The five invented guides: ids 2, 3, 5, 8 and 13, in that order."""
    return load_snapshot(FIXTURE)


def make_guide(guide_id: int, slug: str, title: str, body: str) -> Guide:
    """A tiny guide for a test that needs its own data."""
    return Guide(
        id=guide_id,
        slug=slug,
        title=title,
        url=PUBLIC_PAGE_PREFIX + slug,
        updated_at="2026-01-01T00:00:00+00:00",
        body=body,
    )


def fake_similarity(question: str, search_text: str) -> float:
    """What the fake embedder makes of a question and a passage, worked out here."""
    embedder = FakeEmbedder()
    question_vector = embedder.embed_query(question)
    passage_vector = embedder.embed_passages([search_text])[0]
    return sum(a * b for a, b in zip(question_vector, passage_vector))


def best_fake_similarity(conn: psycopg.Connection, question: str) -> float:
    """The highest similarity of the question to any stored passage."""
    rows = conn.execute("SELECT search_text FROM passages").fetchall()
    return max(fake_similarity(question, row[0]) for row in rows)


def slugs(result: SearchResult) -> list[str]:
    return [guide.slug for guide in result.guides]


def all_passages(result: SearchResult) -> list[PassageHit]:
    return [passage for guide in result.guides for passage in guide.passages]


@pytest.fixture
def indexed(conn: psycopg.Connection) -> psycopg.Connection:
    """The test database with the five fixture guides ingested."""
    ingest(conn, fixture_guides(), FakeEmbedder())
    return conn


# ---------------------------------------------------------------------------
# Every mode finds the guide a question is about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("question, slug", QUESTION_AND_GUIDE)
def test_each_mode_puts_the_guide_the_question_is_about_first(
    indexed: psycopg.Connection, mode: str, question: str, slug: str
):
    result = search(indexed, FakeEmbedder(), question, mode, THRESHOLD, FakeReranker())

    assert result.query == question
    assert result.mode == mode
    assert result.reason is None
    assert result.guides[0].slug == slug


def test_every_passage_sits_under_its_own_guide_with_heading_and_position(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), "replace a snapped spar", "vector", THRESHOLD)

    guide = result.guides[0]
    assert guide.guide_id == 3
    assert guide.slug == "replace-a-snapped-spar"
    assert guide.title == "Replace a snapped spar"
    assert guide.url == PUBLIC_PAGE_PREFIX + "replace-a-snapped-spar"
    passage = guide.passages[0]
    assert passage.heading == "Fit the new spar"
    assert passage.position == 0
    # The text is the passage alone, without the title and heading lines.
    assert passage.text.startswith("Slide the snapped spar out of its sleeve.")


# ---------------------------------------------------------------------------
# Grouping: up to 3 guides, up to 2 passages each
# ---------------------------------------------------------------------------


def test_a_result_has_at_most_three_guides_and_two_passages_each(indexed: psycopg.Connection):
    # The table holds 16 passages, so in vector mode all five guides are candidates.
    question = "replace a snapped spar and mend a torn sail"

    result = search(indexed, FakeEmbedder(), question, "vector", THRESHOLD)

    assert len(result.guides) == 3
    assert slugs(result)[:2] == ["mend-a-torn-sail", "replace-a-snapped-spar"]
    # The sail guide has four passages in the table; only its best two are returned.
    assert len(result.guides[0].passages) == 2
    # The spar guide has a single passage, so it returns one.
    assert len(result.guides[1].passages) == 1
    assert len(result.guides[2].passages) <= 2


def test_a_guide_with_one_matching_passage_returns_one_and_is_a_normal_result(indexed: psycopg.Connection):
    # "lockers" is in one passage of the five-passage questions guide.
    result = search(indexed, FakeEmbedder(), "lockers", "keyword", THRESHOLD)

    assert result.reason is None
    assert slugs(result) == ["workshop-questions"]
    assert [passage.heading for passage in result.guides[0].passages] == ["Can I leave my kite overnight?"]


def test_the_passages_of_a_guide_come_best_first_not_in_reading_order(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), "mend a torn sail", "vector", THRESHOLD)

    passages = result.guides[0].passages
    assert [passage.position for passage in passages] == [3, 1]
    assert passages[0].similarity > passages[1].similarity


@pytest.mark.parametrize("mode", MODES)
def test_only_the_top_20_passages_are_candidates(conn: psycopg.Connection, mode: str):
    # One guide with a strong and a weak passage, and 21 other guides whose
    # single passage sits between the two. The weak passage is 23rd, outside
    # the top 20, so its guide returns one passage although it has two.
    target = make_guide(
        1,
        "target",
        "Target",
        "## Strong\n\nglue glue glue\n\n"
        "## Weak\n\nglue and a dozen more words that have no bearing on the matter",
    )
    fillers = [make_guide(n, f"filler-{n}", f"Filler {n}", "glue glue pot") for n in range(2, 23)]
    ingest(conn, [target] + fillers, FakeEmbedder())

    result = search(conn, FakeEmbedder(), "glue", mode, 0.0, FakeReranker())

    assert result.guides[0].slug == "target"
    assert [passage.heading for passage in result.guides[0].passages] == ["Strong"]


# ---------------------------------------------------------------------------
# Ordering is repeatable: ties go by guide id, then position, never row id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_two_guides_made_of_the_same_words_come_in_guide_id_order_whatever_was_ingested_first(
    conn: psycopg.Connection, mode: str
):
    # The two guides are made of the same words, so every search scores them alike.
    guide_4 = make_guide(4, "blue-kite", "Blue kite", "String and tail.")
    guide_7 = make_guide(7, "kite-blue", "Kite blue", "Tail and string.")

    ingest(conn, [guide_7, guide_4], FakeEmbedder())  # guide 7 gets the lower row ids
    first = search(conn, FakeEmbedder(), "blue kite tail", mode, THRESHOLD, FakeReranker())
    conn.execute("DELETE FROM passages")
    ingest(conn, [guide_4, guide_7], FakeEmbedder())  # now guide 4 does
    second = search(conn, FakeEmbedder(), "blue kite tail", mode, THRESHOLD, FakeReranker())

    assert [guide.guide_id for guide in first.guides] == [4, 7]
    assert second == first
    if mode != "hybrid":
        # The two scores are exactly equal: the tie-break alone put guide 4
        # first. (Hybrid has already turned the two tied lists into ranks 1
        # and 2 by the same tie-break, so its fusion scores differ.)
        assert first.guides[0].score == first.guides[1].score


def test_passages_that_all_score_the_same_come_in_guide_id_then_position_order(conn: psycopg.Connection):
    # Ingested last guide first, so the row ids run against the guide ids.
    ingest(conn, list(reversed(fixture_guides())), FakeEmbedder())

    # The fake embedder scores an Arabic question 0 against every passage.
    # A threshold of 0 never refuses, so the tie-break alone decides the result.
    result = search(conn, FakeEmbedder(), ARABIC, "vector", 0.0)

    assert result.reason is None
    assert result.best_similarity == 0.0
    assert [guide.guide_id for guide in result.guides] == [2, 3, 5]
    assert [[p.position for p in guide.passages] for guide in result.guides] == [[0, 1], [0], [0, 1]]


@pytest.mark.parametrize("mode", MODES)
def test_results_do_not_depend_on_the_order_in_which_guides_were_ingested(
    conn: psycopg.Connection, mode: str
):
    questions = [question for question, slug in QUESTION_AND_GUIDE] + ["troubleshooting", "kite"]
    guides = fixture_guides()

    ingest(conn, guides, FakeEmbedder())
    first = [search(conn, FakeEmbedder(), question, mode, 0.0, FakeReranker()) for question in questions]
    conn.execute("DELETE FROM passages")
    ingest(conn, list(reversed(guides)), FakeEmbedder())
    second = [search(conn, FakeEmbedder(), question, mode, 0.0, FakeReranker()) for question in questions]

    assert second == first


@pytest.mark.parametrize("mode", MODES)
def test_a_tie_at_the_edge_of_the_top_20_is_cut_by_guide_id_not_by_row_id(
    conn: psycopg.Connection, mode: str
):
    # 25 guides that every search scores alike, so only 20 can be candidates.
    # They are ingested last guide first: by row id the first 20 would be
    # guides 25 down to 6, and the result would start at guide 6.
    guides = [make_guide(n, f"filler-{n}", f"Filler {n}", "glue pot") for n in range(1, 26)]
    ingest(conn, list(reversed(guides)), FakeEmbedder())

    result = search(conn, FakeEmbedder(), "glue", mode, THRESHOLD, FakeReranker())

    assert [guide.guide_id for guide in result.guides] == [1, 2, 3]


def test_keyword_tie_between_two_guides_goes_to_the_lower_guide_id(indexed: psycopg.Connection):
    # "Troubleshooting" is a heading in guide 2 and in guide 13, once in each.
    result = search(indexed, FakeEmbedder(), "troubleshooting", "keyword", THRESHOLD)

    assert [guide.guide_id for guide in result.guides] == [2, 13]
    assert result.guides[0].score == result.guides[1].score


# ---------------------------------------------------------------------------
# No match
# ---------------------------------------------------------------------------


def test_vector_mode_is_no_match_below_the_threshold_and_reports_the_best_similarity(
    indexed: psycopg.Connection,
):
    result = search(indexed, FakeEmbedder(), NOT_COVERED, "vector", THRESHOLD)

    assert result.reason == NO_MATCH
    assert result.guides == []
    assert 0 < result.best_similarity < THRESHOLD
    assert result.best_similarity == pytest.approx(best_fake_similarity(indexed, NOT_COVERED), abs=1e-6)


def test_a_similarity_equal_to_the_threshold_is_a_match(indexed: psycopg.Connection):
    # Refusing means "below the threshold". Exactly on it is still a match.
    best = search(indexed, FakeEmbedder(), NOT_COVERED, "vector", 0.0).best_similarity

    on_the_line = search(indexed, FakeEmbedder(), NOT_COVERED, "vector", best)
    just_above = search(indexed, FakeEmbedder(), NOT_COVERED, "vector", best + 0.000001)

    assert on_the_line.reason is None
    assert just_above.reason == NO_MATCH


def test_keyword_mode_is_no_match_when_no_word_of_the_question_is_in_any_passage(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), "bicycle helmet prices", "keyword", THRESHOLD)

    assert result == SearchResult(
        query="bicycle helmet prices",
        mode="keyword",
        reason=NO_MATCH,
        best_similarity=None,
        guides=[],
    )


def test_keyword_mode_has_no_threshold(indexed: psycopg.Connection):
    # One matching word is enough, however high the threshold is set.
    result = search(indexed, FakeEmbedder(), "lockers", "keyword", 0.99)

    assert slugs(result) == ["workshop-questions"]


def test_hybrid_refuses_on_the_vector_similarity_even_when_a_keyword_matches(indexed: psycopg.Connection):
    # One word of six ("spar") is in a guide: enough for keyword search,
    # too little for the similarity to reach the threshold.
    question = "glider wing spar cracked near tip"

    keyword = search(indexed, FakeEmbedder(), question, "keyword", THRESHOLD)
    hybrid = search(indexed, FakeEmbedder(), question, "hybrid", THRESHOLD)

    assert slugs(keyword) == ["replace-a-snapped-spar"]
    assert hybrid.reason == NO_MATCH
    assert hybrid.guides == []
    assert 0 < hybrid.best_similarity < THRESHOLD


def test_hybrid_and_vector_refuse_the_same_questions(indexed: psycopg.Connection):
    vector = search(indexed, FakeEmbedder(), NOT_COVERED, "vector", THRESHOLD)
    hybrid = search(indexed, FakeEmbedder(), NOT_COVERED, "hybrid", THRESHOLD)

    assert hybrid.reason == NO_MATCH
    assert hybrid.best_similarity == vector.best_similarity


def test_hybrid_gives_the_same_guides_and_passages_as_vector_when_no_keyword_matches(
    indexed: psycopg.Connection,
):
    # Keyword search finds nothing for this question, so only the vector list is left.
    keyword = search(indexed, FakeEmbedder(), ONLY_COMMON_WORDS, "keyword", 0.0)
    vector = search(indexed, FakeEmbedder(), ONLY_COMMON_WORDS, "vector", 0.0)
    hybrid = search(indexed, FakeEmbedder(), ONLY_COMMON_WORDS, "hybrid", 0.0)

    assert keyword.reason == NO_MATCH
    assert len(vector.guides) == 3
    # Only the scores differ: a fusion score in one, a similarity in the other.
    assert slugs(hybrid) == slugs(vector)
    assert [guide.passages for guide in hybrid.guides] == [guide.passages for guide in vector.guides]
    assert hybrid.best_similarity == vector.best_similarity


# ---------------------------------------------------------------------------
# Keyword search: any word of the question, never read as query syntax
# ---------------------------------------------------------------------------


def test_keyword_full_sentence_matches_on_any_of_its_words(indexed: psycopg.Connection):
    # No passage holds every word of this sentence ("weekend" is in none),
    # so a search that required all the words would find nothing.
    question = "How do I mend a torn sail on my kite before the weekend?"

    result = search(indexed, FakeEmbedder(), question, "keyword", THRESHOLD)

    assert result.reason is None
    assert result.guides[0].slug == "mend-a-torn-sail"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("question", ["???", "... --- !!!", "\U0001F642\U0001F642"])
def test_a_question_with_no_words_at_all_is_answered_without_an_error(
    indexed: psycopg.Connection, mode: str, question: str
):
    result = search(indexed, FakeEmbedder(), question, mode, THRESHOLD, FakeReranker(), THRESHOLD)

    # What this test shows for every mode is: no error. Keyword mode always
    # gives no_match here, because there is no word to look for. In vector,
    # hybrid and rerank mode the thresholds decide, as for any other question.
    # It is a no_match here only because the fake embedder and the fake
    # reranker score wordless text 0, which is below these tests' own
    # threshold of 0.3.
    assert result.reason == NO_MATCH
    assert result.guides == []


@pytest.mark.parametrize("mode", MODES)
def test_a_question_of_several_hundred_words_is_searched_without_an_error(indexed: psycopg.Connection, mode: str):
    # The HTTP route caps a question at 500 characters; search() itself has
    # no cap, so it must cope with a very long one. Threshold 0: never refuse.
    question = " ".join(f"filler{number}" for number in range(300)) + " sail"

    result = search(indexed, FakeEmbedder(), question, mode, 0.0, FakeReranker())

    assert result.guides != []
    if mode == "keyword":
        # Only one of the 301 words is in any guide, and it is found.
        assert slugs(result)[0] == "mend-a-torn-sail"


@pytest.mark.parametrize("mode", MODES)
def test_a_question_with_a_control_character_is_searched_as_its_words(indexed: psycopg.Connection, mode: str):
    # Postgres refuses text that contains a NUL character. Only the words of
    # a question are ever sent to it, so the character never gets there.
    result = search(indexed, FakeEmbedder(), "torn\x00sail\x07", mode, 0.0, FakeReranker())

    assert slugs(result)[0] == "mend-a-torn-sail"


def test_keyword_question_of_only_common_words_is_no_match_and_no_error(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), ONLY_COMMON_WORDS, "keyword", THRESHOLD)

    assert result.reason == NO_MATCH
    assert result.guides == []


def test_keyword_question_with_an_apostrophe_and_a_question_mark(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), "The spar doesn't fit, what's wrong?", "keyword", THRESHOLD)

    assert slugs(result) == ["replace-a-snapped-spar"]


def test_keyword_question_is_never_read_as_query_syntax(indexed: psycopg.Connection):
    # In Postgres query syntax "&" means and, "|" or, "!" not and ":*" prefix.
    # Here they must mean nothing: "!kite" has to match passages WITH "kite".
    question = "sail & (spar | !kite):* '); DROP TABLE passages; --"

    result = search(indexed, FakeEmbedder(), question, "keyword", THRESHOLD)

    # The third guide is found through the word "kite" alone.
    assert slugs(result) == ["mend-a-torn-sail", "replace-a-snapped-spar", "workshop-questions"]
    assert indexed.execute("SELECT count(*) FROM passages").fetchone()[0] == 16


def test_keyword_matches_other_forms_of_a_word(indexed: psycopg.Connection):
    # The guide says "snapped spar"; the stems are the same.
    result = search(indexed, FakeEmbedder(), "snapping spars", "keyword", THRESHOLD)

    assert slugs(result) == ["replace-a-snapped-spar"]


def test_keyword_all_arabic_question_is_no_match(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), ARABIC, "keyword", THRESHOLD)

    assert result.reason == NO_MATCH
    assert result.guides == []


def test_keyword_arabic_question_with_one_latin_word_returns_the_passages_with_that_word(
    indexed: psycopg.Connection,
):
    result = search(indexed, FakeEmbedder(), ARABIC_WITH_SMS, "keyword", THRESHOLD)

    assert slugs(result) == ["send-a-pickup-reminder"]
    passages = result.guides[0].passages
    # Position 1 has "SMS" twice (heading and text), position 2 once.
    assert [passage.position for passage in passages] == [1, 2]
    for passage in passages:
        assert "SMS" in passage.heading + " " + passage.text


def test_related_articles_text_is_not_searched(indexed: psycopg.Connection):
    # "gazebo" occurs in the fixture only inside a Related articles section.
    result = search(indexed, FakeEmbedder(), "gazebo", "keyword", THRESHOLD)

    assert result.reason == NO_MATCH


# ---------------------------------------------------------------------------
# similarity, best_similarity and score
# ---------------------------------------------------------------------------


def test_keyword_mode_never_embeds_the_question_and_gives_no_similarity(indexed: psycopg.Connection):
    embedder = QueryCountingEmbedder()

    result = search(indexed, embedder, "send a pickup reminder by sms", "keyword", THRESHOLD)

    assert embedder.queries == 0
    assert result.best_similarity is None
    assert len(all_passages(result)) > 0
    for passage in all_passages(result):
        assert passage.similarity is None


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_vector_and_hybrid_embed_the_question_once_and_give_every_passage_its_similarity(
    indexed: psycopg.Connection, mode: str
):
    question = "send a pickup reminder by sms"
    embedder = QueryCountingEmbedder()

    result = search(indexed, embedder, question, mode, THRESHOLD)

    assert embedder.queries == 1
    assert len(all_passages(result)) > 0
    for guide in result.guides:
        for passage in guide.passages:
            search_text = indexed.execute(
                "SELECT search_text FROM passages WHERE guide_id = %s AND position = %s",
                (guide.guide_id, passage.position),
            ).fetchone()[0]
            assert passage.similarity == pytest.approx(fake_similarity(question, search_text), abs=1e-6)
    assert result.best_similarity == pytest.approx(best_fake_similarity(indexed, question), abs=1e-6)


def test_hybrid_gives_a_similarity_to_a_passage_that_only_keyword_search_found(conn: psycopg.Connection):
    # 21 guides share the three everyday words of the question, which the fake
    # embedder counts and keyword search ignores. One guide has the word
    # "zebra" among many others: keyword search finds only that guide, and by
    # similarity it comes 22nd, outside the top 20.
    question = "where is the zebra"
    zebra_text = "zebra " + " ".join(f"word{n}" for n in range(12))
    zebra = make_guide(1, "zebra-notes", "Notes", "## Stripes\n\n" + zebra_text)
    fillers = [
        make_guide(n, f"filler-{n}", f"Filler {n}", "where is the where is the")
        for n in range(2, 23)
    ]
    ingest(conn, [zebra] + fillers, FakeEmbedder())

    vector = search(conn, FakeEmbedder(), question, "vector", THRESHOLD)
    keyword = search(conn, FakeEmbedder(), question, "keyword", THRESHOLD)
    hybrid = search(conn, FakeEmbedder(), question, "hybrid", THRESHOLD)

    assert "zebra-notes" not in slugs(vector)
    assert slugs(keyword) == ["zebra-notes"]
    # Top of the keyword list ties with the top of the vector list (1/61 each),
    # and the lower guide id wins, so the keyword-only guide comes first.
    assert hybrid.guides[0].slug == "zebra-notes"
    zebra_similarity = fake_similarity(question, "Notes\nStripes\n" + zebra_text)
    assert hybrid.guides[0].passages[0].similarity == pytest.approx(zebra_similarity, abs=1e-6)
    # best_similarity is the best of the vector list, which is what was
    # compared with the threshold. It is not the first guide's similarity.
    assert hybrid.best_similarity == vector.best_similarity
    assert hybrid.best_similarity > THRESHOLD > zebra_similarity


def test_vector_score_is_the_similarity_of_the_guides_best_passage(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), "mend a torn sail", "vector", THRESHOLD)

    for guide in result.guides:
        assert guide.score == guide.passages[0].similarity
    assert result.best_similarity == result.guides[0].score


def test_keyword_score_is_the_ts_rank_cd_of_the_guides_best_passage(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), ARABIC_WITH_SMS, "keyword", THRESHOLD)

    expected = indexed.execute(
        """
        SELECT ts_rank_cd(tsv, plainto_tsquery('english', 'SMS'))
        FROM passages
        WHERE slug = 'send-a-pickup-reminder' AND position = 1
        """
    ).fetchone()[0]
    assert result.guides[0].score == expected
    # With one word to look for, the rank is 0.1 for each time it occurs.
    assert expected == pytest.approx(0.2)


def test_hybrid_score_is_the_fusion_score(indexed: psycopg.Connection):
    # The spar guide's only passage is first in the vector list and first in
    # the keyword list, so it gets 1 / (60 + 1) from each.
    result = search(indexed, FakeEmbedder(), "replace a snapped spar", "hybrid", THRESHOLD)

    assert result.guides[0].slug == "replace-a-snapped-spar"
    assert result.guides[0].score == pytest.approx(2 / 61)


# ---------------------------------------------------------------------------
# Errors are not "no match"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_an_empty_table_raises_index_not_ready_in_every_mode(conn: psycopg.Connection, mode: str):
    with pytest.raises(IndexNotReady):
        search(conn, FakeEmbedder(), "mend a torn sail", mode, THRESHOLD, FakeReranker())


@pytest.mark.parametrize("mode", MODES)
def test_a_table_filled_by_another_model_raises_index_not_ready(indexed: psycopg.Connection, mode: str):
    with pytest.raises(IndexNotReady):
        search(indexed, OtherModelEmbedder(), "mend a torn sail", mode, THRESHOLD, FakeReranker())


@pytest.mark.parametrize("mode", MODES)
def test_passages_stored_for_another_model_are_never_returned(indexed: psycopg.Connection, mode: str):
    indexed.execute("UPDATE passages SET embedding_model = 'fake-2' WHERE slug = 'replace-a-snapped-spar'")

    result = search(indexed, FakeEmbedder(), "replace a snapped spar", mode, 0.0, FakeReranker())

    assert "replace-a-snapped-spar" not in slugs(result)


def test_an_unknown_mode_raises_value_error(indexed: psycopg.Connection):
    with pytest.raises(ValueError):
        search(indexed, FakeEmbedder(), "mend a torn sail", "fuzzy", THRESHOLD)


# ---------------------------------------------------------------------------
# read_guide
# ---------------------------------------------------------------------------


def test_read_guide_returns_the_passages_in_order_without_related_articles(indexed: psycopg.Connection):
    detail = read_guide(indexed, "send-a-pickup-reminder")

    assert detail == GuideDetail(
        guide_id=13,
        slug="send-a-pickup-reminder",
        title="Send a pickup reminder",
        url=PUBLIC_PAGE_PREFIX + "send-a-pickup-reminder",
        passages=[
            PassageHit(
                heading="",
                position=0,
                text="When a repair is ready, the owner is told to collect the kite.",
                similarity=None,
            ),
            PassageHit(
                heading="Send the reminder by SMS",
                position=1,
                text=(
                    "Open the repair ticket and press Remind. "
                    "The owner receives an SMS with the pickup code."
                ),
                similarity=None,
            ),
            PassageHit(
                heading="Troubleshooting",
                position=2,
                text=(
                    "If the SMS does not arrive, check the phone number "
                    "on the repair ticket and press Remind again."
                ),
                similarity=None,
            ),
        ],
    )


def test_read_guide_returns_both_pieces_of_a_section_that_was_cut_in_two(indexed: psycopg.Connection):
    detail = read_guide(indexed, "flying-line-in-stock")

    assert [(p.position, p.heading) for p in detail.passages] == [
        (0, "How to read the list"),
        (1, "Stock list"),
        (2, "Stock list"),
    ]


def test_read_guide_returns_none_for_an_unknown_slug(indexed: psycopg.Connection):
    assert read_guide(indexed, "no-such-guide") is None


# ---------------------------------------------------------------------------
# Rerank mode
# ---------------------------------------------------------------------------


class OneSectionReranker:
    """A reranker that scores one section 0.9 and every other passage 0.1. It records what it was given."""

    model_id = "one-section"

    def __init__(self, heading: str) -> None:
        self.heading = heading
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, texts: list[str]) -> list[float]:
        self.calls.append((question, texts))
        # A passage's text is its title, its heading and its content, one per line.
        return [0.9 if text.split("\n")[1] == self.heading else 0.1 for text in texts]


def test_rerank_puts_first_the_passage_the_reranker_scores_highest(indexed: psycopg.Connection):
    hybrid = search(indexed, FakeEmbedder(), "mend a torn sail", "hybrid", THRESHOLD)
    reranked = search(
        indexed, FakeEmbedder(), "mend a torn sail", "rerank", THRESHOLD, OneSectionReranker("Fit the new spar")
    )

    # Hybrid search puts the sail guide first. The reranker prefers a section
    # of the spar guide, so rerank search puts that guide first.
    assert slugs(hybrid)[0] == "mend-a-torn-sail"
    assert slugs(reranked)[0] == "replace-a-snapped-spar"
    assert reranked.guides[0].passages[0].heading == "Fit the new spar"
    assert reranked.guides[0].score == 0.9
    assert reranked.best_rerank_score == 0.9


def test_every_passage_of_a_rerank_result_carries_its_reranker_score(indexed: psycopg.Connection):
    result = search(indexed, FakeEmbedder(), "mend a torn sail", "rerank", THRESHOLD, FakeReranker())

    scores = [passage.rerank_score for passage in all_passages(result)]
    assert None not in scores
    assert result.best_rerank_score == max(scores)
    # The best similarity is still the vector search's, as in the other modes.
    assert result.best_similarity == pytest.approx(best_fake_similarity(indexed, "mend a torn sail"))


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_the_other_modes_carry_no_reranker_score(indexed: psycopg.Connection, mode: str):
    result = search(indexed, FakeEmbedder(), "mend a torn sail", mode, THRESHOLD, FakeReranker())

    assert result.best_rerank_score is None
    assert {passage.rerank_score for passage in all_passages(result)} == {None}


def test_the_reranker_reads_each_candidate_once_as_title_heading_and_text(indexed: psycopg.Connection):
    reranker = OneSectionReranker("Patch the tear")

    search(indexed, FakeEmbedder(), "mend a torn sail", "rerank", THRESHOLD, reranker)

    [(question, texts)] = reranker.calls
    assert question == "mend a torn sail"
    # The same text that was embedded and keyword-indexed for each passage.
    stored = {row[0] for row in indexed.execute("SELECT search_text FROM passages").fetchall()}
    assert set(texts) <= stored
    assert len(texts) == len(set(texts))


def test_rerank_refuses_when_the_best_reranker_score_is_below_its_threshold(indexed: psycopg.Connection):
    # The fake reranker's best score for this question is 0.4: "do" and "you"
    # are 2 of its 5 words, and one passage has both.
    result = search(indexed, FakeEmbedder(), NOT_COVERED, "rerank", 0.0, FakeReranker(), 0.5)

    assert result.reason == NO_MATCH
    assert result.guides == []
    assert result.best_rerank_score == pytest.approx(0.4)
    assert result.best_similarity is not None


def test_rerank_mode_has_its_own_threshold_and_ignores_the_vector_one(indexed: psycopg.Connection):
    # A similarity threshold of 0.99 refuses this question in hybrid mode.
    hybrid = search(indexed, FakeEmbedder(), "mend a torn sail", "hybrid", 0.99)
    reranked = search(indexed, FakeEmbedder(), "mend a torn sail", "rerank", 0.99, FakeReranker(), 0.0)

    assert hybrid.reason == NO_MATCH
    assert reranked.reason is None
    assert slugs(reranked)[0] == "mend-a-torn-sail"


def test_rerank_mode_without_a_reranker_is_an_error(indexed: psycopg.Connection):
    with pytest.raises(ValueError, match="reranker"):
        search(indexed, FakeEmbedder(), "mend a torn sail", "rerank", THRESHOLD)
