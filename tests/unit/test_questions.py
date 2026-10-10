"""Tests for the question file: reading and writing it, checking it, and the dev/test split.

Everything here is invented. Nothing touches a database, a model or the network.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from guide_search.evaluation.questions import (
    SPLIT_SEED,
    Question,
    QuestionError,
    assign_split,
    guide_headings,
    load_questions,
    main,
    save_questions,
    validate_questions,
)
from guide_search.snapshot import LOCALE, PRODUCT, PUBLIC_PAGE_PREFIX, SOURCE, Guide

ARABIC_QUESTION = "كيف أطبع فاتورة للعميل؟"

# The sections each invented guide has. "" is the text before the first heading.
HEADINGS = {
    "invoices": {"", "Print an invoice", "Troubleshooting"},
    "refunds": {"", "Give money back"},
    "bookings": {"Move a booking"},
}


def make_question(
    question_id: str,
    language: str = "en",
    text: str = "How do I print an invoice?",
    expected: tuple[str, ...] = ("invoices",),
    headings: dict[str, tuple[str, ...]] | None = None,
    split: str | None = None,
    reviewed: bool = False,
) -> Question:
    """A valid question by default; each test changes the one thing it is about."""
    if headings is None:
        headings = {"invoices": ("Print an invoice",)} if expected == ("invoices",) else {}
    return Question(
        id=question_id,
        language=language,
        question=text,
        expected_slugs=expected,
        answer_headings=headings,
        split=split,
        reviewed=reviewed,
    )


def unanswerable(question_id: str, language: str = "en") -> Question:
    return make_question(question_id, language=language, text="Can it order parts for me?", expected=())


def problems_for(questions: list[Question]) -> list[str]:
    """Run the validation and return the problems it reports."""
    with pytest.raises(QuestionError) as raised:
        validate_questions(questions, HEADINGS)
    return raised.value.problems


def splits(questions: list[Question]) -> dict[str, str | None]:
    return {question.id: question.split for question in questions}


def numbered(prefix: str, count: int) -> list[Question]:
    """`count` answerable questions with ids like en-001; the prefix is also the language."""
    return [make_question(f"{prefix}-{number:03d}", language=prefix) for number in range(1, count + 1)]


# --- reading and writing the file -------------------------------------------


def test_saving_then_loading_gives_back_the_same_questions(tmp_path: Path) -> None:
    questions = [
        make_question("en-001", split="dev", reviewed=True),
        make_question(
            "ar-001",
            language="ar",
            text=ARABIC_QUESTION,
            expected=("invoices", "refunds"),
            headings={"invoices": ("", "Print an invoice"), "refunds": ("Give money back",)},
        ),
        unanswerable("na-001"),
    ]
    path = tmp_path / "questions.jsonl"

    save_questions(path, questions)

    assert load_questions(path) == questions


def test_loading_then_saving_leaves_a_file_with_an_arabic_question_unchanged(tmp_path: Path) -> None:
    original = (
        '{"id": "en-001", "language": "en", "question": "How do I print an invoice?", '
        '"expected_slugs": ["invoices"], "answer_headings": {"invoices": ["Print an invoice"]}, '
        '"split": "dev", "reviewed": true}\n'
        '{"id": "ar-001", "language": "ar", "question": "' + ARABIC_QUESTION + '", '
        '"expected_slugs": ["invoices"], "answer_headings": {"invoices": [""]}, '
        '"split": null, "reviewed": false}\n'
    )
    source = tmp_path / "questions.jsonl"
    source.write_bytes(original.encode("utf-8"))
    copy = tmp_path / "copy.jsonl"

    save_questions(copy, load_questions(source))

    assert copy.read_bytes() == original.encode("utf-8")


def test_the_file_has_one_json_object_per_line_and_arabic_is_written_as_itself(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"

    save_questions(path, [make_question("en-001"), make_question("ar-001", language="ar", text=ARABIC_QUESTION)])

    text = path.read_bytes().decode("utf-8")
    lines = text.split("\n")
    assert lines[-1] == ""  # the file ends with a newline
    assert [json.loads(line)["id"] for line in lines[:-1]] == ["en-001", "ar-001"]
    assert ARABIC_QUESTION in text  # not turned into \uXXXX escapes
    assert list(json.loads(lines[0])) == [
        "id",
        "language",
        "question",
        "expected_slugs",
        "answer_headings",
        "split",
        "reviewed",
    ]


def test_a_unicode_line_separator_inside_a_question_does_not_break_its_line(tmp_path: Path) -> None:
    # U+2028 counts as a line break for str.splitlines() but not for JSON,
    # and pasted text sometimes contains one.
    question = make_question("en-001", text="First part second part")
    path = tmp_path / "questions.jsonl"

    save_questions(path, [question])

    assert load_questions(path) == [question]


def test_a_line_without_split_is_read_as_not_yet_assigned(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id": "en-001", "language": "en", "question": "How do I print an invoice?", '
        '"expected_slugs": [], "answer_headings": {}, "reviewed": false}\n',
        encoding="utf-8",
    )

    assert load_questions(path)[0].split is None


def test_loading_names_the_line_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    save_questions(path, [make_question("en-001")])
    with path.open("a", encoding="utf-8") as file:
        file.write("this is not JSON\n")

    with pytest.raises(QuestionError) as raised:
        load_questions(path)

    assert len(raised.value.problems) == 1
    assert raised.value.problems[0].startswith("line 2:")


def test_loading_reports_a_missing_field_an_unknown_field_and_a_wrong_type_together(tmp_path: Path) -> None:
    good = asdict(make_question("en-001"))
    missing = {key: value for key, value in good.items() if key != "reviewed"}
    unknown = dict(good, note="written by hand")
    wrong_type = dict(good, expected_slugs="invoices")
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in (missing, unknown, wrong_type)), encoding="utf-8")

    with pytest.raises(QuestionError) as raised:
        load_questions(path)

    problems = raised.value.problems
    assert len(problems) == 3
    assert problems[0].startswith("line 1:") and "reviewed" in problems[0]
    assert problems[1].startswith("line 2:") and "note" in problems[1]
    assert problems[2].startswith("line 3:") and "expected_slugs" in problems[2]


# --- answerable and group ---------------------------------------------------


def test_group_is_the_language_for_a_question_the_guides_can_answer() -> None:
    assert make_question("en-001").group == "en"
    assert make_question("ar-001", language="ar").group == "ar"
    assert make_question("en-001").answerable is True


def test_group_is_unanswerable_when_no_guide_is_expected() -> None:
    assert unanswerable("na-001").group == "unanswerable"
    assert unanswerable("na-001").answerable is False


# --- validation -------------------------------------------------------------


def test_a_good_set_of_questions_has_no_problems() -> None:
    questions = [
        make_question("en-001", reviewed=True, split="dev"),
        make_question(
            "ar-001",
            language="ar",
            text=ARABIC_QUESTION,
            expected=("invoices", "refunds"),
            headings={"invoices": ("",), "refunds": ("Give money back",)},
            reviewed=True,
            split="test",
        ),
        unanswerable("na-001"),
    ]

    validate_questions(questions, HEADINGS)  # does not raise


def test_an_id_used_twice_is_a_problem() -> None:
    problems = problems_for([make_question("en-001"), make_question("en-001")])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "more than once" in problems[0]


def test_a_language_other_than_english_or_arabic_is_a_problem() -> None:
    problems = problems_for([make_question("en-001", language="fr")])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "'fr'" in problems[0]


def test_a_blank_question_is_a_problem() -> None:
    problems = problems_for([make_question("en-001", text="   ")])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "blank" in problems[0]


def test_an_expected_guide_that_is_not_in_the_snapshot_is_a_problem() -> None:
    problems = problems_for([make_question("en-001", expected=("no-such-guide",), headings={})])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "'no-such-guide'" in problems[0]


def test_answer_headings_for_a_guide_that_is_not_expected_is_a_problem() -> None:
    question = make_question("en-001", headings={"invoices": ("",), "refunds": ("Give money back",)})

    problems = problems_for([question])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "'refunds'" in problems[0]


def test_a_heading_the_guide_does_not_have_is_a_problem() -> None:
    problems = problems_for([make_question("en-001", headings={"invoices": ("Print a receipt",)})])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "'Print a receipt'" in problems[0]


def test_the_intro_is_only_accepted_for_a_guide_that_has_one() -> None:
    # "bookings" has no text before its first heading, so "" is not one of its sections.
    question = make_question("en-001", expected=("bookings",), headings={"bookings": ("",)})

    problems = problems_for([question])

    assert len(problems) == 1
    assert "'bookings'" in problems[0]


def test_a_reviewed_question_must_list_a_section_for_every_expected_guide() -> None:
    question = make_question(
        "en-001", expected=("invoices", "refunds"), headings={"invoices": ("",)}, reviewed=True
    )

    problems = problems_for([question])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "'refunds'" in problems[0]


def test_an_unreviewed_question_may_still_lack_a_section() -> None:
    question = make_question("en-001", expected=("invoices", "refunds"), headings={"invoices": ("",)})

    validate_questions([question], HEADINGS)  # does not raise


def test_an_unanswerable_question_that_is_not_english_is_a_problem() -> None:
    problems = problems_for([unanswerable("na-001", language="ar")])

    assert len(problems) == 1
    assert "na-001" in problems[0] and "English" in problems[0]


def test_a_split_other_than_dev_or_test_is_a_problem() -> None:
    problems = problems_for([make_question("en-001", split="train")])

    assert len(problems) == 1
    assert "en-001" in problems[0] and "'train'" in problems[0]


def test_every_problem_is_reported_in_one_error() -> None:
    questions = [
        make_question("en-001", text=""),
        make_question("en-002", language="de"),
        unanswerable("na-001", language="ar"),
        make_question("en-003", split="holdout"),
    ]

    problems = problems_for(questions)

    assert len(problems) == 4
    assert [problem.split(":")[0] for problem in problems] == ["en-001", "en-002", "na-001", "en-003"]


# --- the dev/test split -----------------------------------------------------


def test_each_group_gets_a_dev_half_rounded_up_and_a_test_half() -> None:
    questions = numbered("en", 5) + numbered("ar", 4) + [unanswerable(f"na-00{n}") for n in (1, 2, 3)]

    result = assign_split(questions)

    def count(group: str, split: str) -> int:
        return len([q for q in result if q.group == group and q.split == split])

    assert (count("en", "dev"), count("en", "test")) == (3, 2)  # 5 questions: 3 and 2
    assert (count("ar", "dev"), count("ar", "test")) == (2, 2)  # 4 questions: 2 and 2
    assert (count("unanswerable", "dev"), count("unanswerable", "test")) == (2, 1)  # 3 questions: 2 and 1


def test_a_group_of_one_goes_to_dev() -> None:
    assert assign_split([make_question("en-001")])[0].split == "dev"


def test_the_split_is_the_same_on_every_run_with_the_same_seed() -> None:
    questions = numbered("en", 5)

    first = assign_split(questions)
    second = assign_split(questions)

    assert splits(first) == splits(second)
    # The exact answer for the project's seed, worked out once. If this ever
    # fails after a Python upgrade, the shuffle itself has changed and the
    # committed split could no longer be reproduced.
    assert SPLIT_SEED == 20261010
    assert splits(first) == {
        "en-001": "dev",
        "en-002": "test",
        "en-003": "test",
        "en-004": "dev",
        "en-005": "dev",
    }


def test_the_split_does_not_depend_on_the_order_of_the_input() -> None:
    questions = numbered("en", 9) + numbered("ar", 4)

    forwards = assign_split(questions)
    backwards = assign_split(list(reversed(questions)))

    assert splits(forwards) == splits(backwards)


def test_one_group_is_split_without_looking_at_the_others() -> None:
    english = numbered("en", 9)

    alone = assign_split(english)
    with_arabic = assign_split(english + numbered("ar", 6))

    assert splits(alone) == {q.id: q.split for q in with_arabic if q.group == "en"}


def test_unreviewed_questions_are_split_together_with_reviewed_ones() -> None:
    unreviewed = numbered("en", 6)
    half_reviewed = [replace(q, reviewed=(number % 2 == 0)) for number, q in enumerate(unreviewed)]

    result = assign_split(half_reviewed)

    assert all(q.split in ("dev", "test") for q in result)
    # Being reviewed plays no part: the same ids land in the same halves.
    assert splits(result) == splits(assign_split(unreviewed))


def test_the_split_keeps_every_other_field_and_the_order() -> None:
    questions = [
        make_question("en-002", reviewed=True),
        make_question("ar-001", language="ar", text=ARABIC_QUESTION, headings={"invoices": ("",)}),
        unanswerable("na-001"),
        make_question("en-001", expected=("invoices", "refunds"), headings={"refunds": ("Give money back",)}),
    ]

    result = assign_split(questions)

    assert [replace(q, split=None) for q in result] == questions


def test_assigning_again_changes_nothing() -> None:
    once = assign_split(numbered("en", 7))

    assert assign_split(once) == once


def test_the_split_refuses_two_questions_with_the_same_id() -> None:
    # With a repeated id, "sort by id" has no single answer, so the result
    # would depend on the order of the file.
    with pytest.raises(QuestionError) as raised:
        assign_split([make_question("en-001"), make_question("en-001")])

    assert "en-001" in raised.value.problems[0]


# --- guide_headings and the command line ------------------------------------


def make_guide(guide_id: int, slug: str, title: str, body: str) -> Guide:
    return Guide(
        id=guide_id,
        slug=slug,
        title=title,
        url=PUBLIC_PAGE_PREFIX + slug,
        updated_at="2026-01-01T00:00:00+00:00",
        body=body,
    )


GUIDES = [
    make_guide(
        1,
        "invoices",
        "Create and print an invoice",
        "An invoice lists the work done.\n\n## Print an invoice\n\nOpen the invoice and choose Print.\n\n"
        "## Related articles\n\n- [Refunds](/help/en/refunds)\n",
    ),
    make_guide(2, "bookings", "Book a job", "## Move a booking\n\nDrag the booking to another day.\n"),
]


def write_snapshot(path: Path) -> None:
    snapshot = {
        "source": SOURCE,
        "product": PRODUCT,
        "locale": LOCALE,
        "fetched_at": "2026-01-01",
        "guides": [asdict(guide) for guide in GUIDES],
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_guide_headings_lists_the_sections_that_became_passages() -> None:
    # "Related articles" is left out of the index, so it is not a section an answer can sit in.
    assert guide_headings(GUIDES) == {
        "invoices": {"", "Print an invoice"},
        "bookings": {"Move a booking"},
    }


def test_assign_split_command_rewrites_the_file_with_splits_assigned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = numbered("en", 3) + numbered("ar", 2) + [unanswerable("na-001")]
    path = tmp_path / "questions.jsonl"
    save_questions(path, questions)

    status = main(["assign-split", "--path", str(path)])

    assert status == 0
    assert load_questions(path) == assign_split(questions)
    assert capsys.readouterr().out == "en: 2 dev, 1 test; ar: 1 dev, 1 test; unanswerable: 1 dev, 0 test\n"


def test_running_the_assign_split_command_a_second_time_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "questions.jsonl"
    save_questions(path, numbered("en", 5) + [unanswerable("na-001")])
    main(["assign-split", "--path", str(path)])
    after_first_run = path.read_bytes()
    first_output = capsys.readouterr().out

    status = main(["assign-split", "--path", str(path)])

    assert status == 0
    assert path.read_bytes() == after_first_run
    assert capsys.readouterr().out == first_output


def test_assign_split_command_stops_without_saving_when_it_would_move_a_question(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Five questions get their split, and then a sixth is added to the file.
    # Six questions fall differently from five: dealt again, en-001 would move
    # from dev to test. A question the threshold was chosen on would then be
    # scored as a test question, so the command must refuse.
    path = tmp_path / "questions.jsonl"
    save_questions(path, numbered("en", 5))
    main(["assign-split", "--path", str(path)])
    assert splits(load_questions(path))["en-001"] == "dev"
    save_questions(path, load_questions(path) + [make_question("en-006")])
    before = path.read_bytes()
    capsys.readouterr()  # forget what the first run printed

    status = main(["assign-split", "--path", str(path)])

    assert status == 1
    assert path.read_bytes() == before
    output = capsys.readouterr().out
    assert "en-001" in output
    assert "Nothing was saved" in output


def test_validate_command_says_when_there_are_no_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions_path = tmp_path / "questions.jsonl"
    snapshot_path = tmp_path / "guides.json"
    save_questions(questions_path, [make_question("en-001"), unanswerable("na-001")])
    write_snapshot(snapshot_path)

    status = main(["validate", "--path", str(questions_path), "--snapshot", str(snapshot_path)])

    assert status == 0
    assert capsys.readouterr().out == "2 questions, no problems\n"


def test_validate_command_lists_the_problems_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions_path = tmp_path / "questions.jsonl"
    snapshot_path = tmp_path / "guides.json"
    bad = make_question("en-001", headings={"invoices": ("Related articles",)})
    save_questions(questions_path, [bad, make_question("en-002", language="fr")])
    write_snapshot(snapshot_path)

    status = main(["validate", "--path", str(questions_path), "--snapshot", str(snapshot_path)])

    assert status == 1
    output = capsys.readouterr().out
    assert "en-001" in output and "'Related articles'" in output
    assert "en-002" in output and "'fr'" in output
    assert "2 problems" in output
