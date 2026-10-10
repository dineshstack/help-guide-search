"""The evaluation questions: reading and writing the file, checking it, and the dev/test split.

The questions live in `evaluation/questions.jsonl`, one JSON object per line.
A question the guides can answer lists the guides that would be an acceptable
answer (`expected_slugs`) and the sections that hold the answer
(`answer_headings`). A question the guides cannot answer lists no guide.

This file can also be run:

    python -m guide_search.evaluation.questions validate       check the file against the guides
    python -m guide_search.evaluation.questions assign-split   give every question "dev" or "test"

The split is assigned once. Running `assign-split` again on the same questions
changes nothing. If it would move a question that already has a split, it
stops and saves nothing.
"""

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

from guide_search.chunking import chunk_guide
from guide_search.snapshot import Guide, load_snapshot

QUESTIONS_PATH = Path("evaluation/questions.jsonl")
SNAPSHOT_PATH = Path("data/guides.json")

# The seed for the shuffle that decides the split. It is fixed so that anyone
# who runs the split again gets the same halves.
SPLIT_SEED = 20261010

LANGUAGES = ("en", "ar")
SPLITS = ("dev", "test")
GROUPS = ("en", "ar", "unanswerable")

# The fields of one line of the file, in the order they are written.
FIELDS = ("id", "language", "question", "expected_slugs", "answer_headings", "split", "reviewed")


@dataclass(frozen=True)
class Question:
    id: str
    language: str                                # "en" or "ar"
    question: str
    expected_slugs: tuple[str, ...]              # empty for an unanswerable question
    answer_headings: dict[str, tuple[str, ...]]  # slug -> headings holding the answer; "" is the intro
    split: str | None                            # "dev", "test" or None before assignment
    reviewed: bool

    @property
    def answerable(self) -> bool:
        """True when at least one guide answers the question."""
        return len(self.expected_slugs) > 0

    @property
    def group(self) -> str:
        """The group the question is counted in: "en", "ar" or "unanswerable"."""
        if not self.answerable:
            return "unanswerable"
        return self.language


class QuestionError(ValueError):
    """Carries every problem found, so they can all be fixed in one go.

    `problems` is a list of plain sentences.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


# --- reading and writing the file -------------------------------------------


def load_questions(path: str | Path) -> list[Question]:
    """Read the question file. Raises QuestionError naming every bad line."""
    text = Path(path).read_text(encoding="utf-8")
    questions: list[Question] = []
    problems: list[str] = []
    # split("\n") and not splitlines(): splitlines() also breaks a line at
    # characters such as U+2028, which may appear inside a question.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {number}: not valid JSON ({error.msg})")
            continue
        line_problems = shape_problems(data)
        if line_problems:
            problems.extend(f"line {number}: {problem}" for problem in line_problems)
        else:
            questions.append(question_from_dict(data))
    if problems:
        raise QuestionError(problems)
    return questions


def save_questions(path: str | Path, questions: list[Question]) -> None:
    """Write the question file: one JSON object per line, UTF-8, Arabic kept as Arabic."""
    lines = [json.dumps(question_to_dict(question), ensure_ascii=False) + "\n" for question in questions]
    # newline="\n" so the file has the same bytes on every operating system.
    Path(path).write_text("".join(lines), encoding="utf-8", newline="\n")


def shape_problems(data: object) -> list[str]:
    """What is wrong with the shape of one line: its fields and their types."""
    if not isinstance(data, dict):
        return ["not a JSON object"]
    problems: list[str] = []
    for key in data:
        if key not in FIELDS:
            # Rejected rather than ignored: the file is rewritten when the
            # split is assigned, and an unknown field would be lost then.
            problems.append(f"unknown field {key!r}")
    for key in FIELDS:
        # "split" may be left out before the split has been assigned.
        if key not in data and key != "split":
            problems.append(f"missing field {key!r}")
    if problems:
        return problems

    if not isinstance(data["id"], str) or not data["id"]:
        problems.append("id must be a non-empty string")
    if not isinstance(data["language"], str):
        problems.append("language must be a string")
    if not isinstance(data["question"], str):
        problems.append("question must be a string")
    if not is_list_of_strings(data["expected_slugs"]):
        problems.append("expected_slugs must be a list of strings")
    headings = data["answer_headings"]
    if not isinstance(headings, dict) or not all(is_list_of_strings(value) for value in headings.values()):
        problems.append("answer_headings must map each slug to a list of headings")
    split = data.get("split")
    if split is not None and not isinstance(split, str):
        problems.append("split must be a string or null")
    if not isinstance(data["reviewed"], bool):
        problems.append("reviewed must be true or false")
    return problems


def is_list_of_strings(value: object) -> bool:
    """True for a list whose items are all strings. An empty list counts."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def question_from_dict(data: dict) -> Question:
    """Build a Question from one line that has already passed shape_problems."""
    return Question(
        id=data["id"],
        language=data["language"],
        question=data["question"],
        expected_slugs=tuple(data["expected_slugs"]),
        answer_headings={slug: tuple(headings) for slug, headings in data["answer_headings"].items()},
        split=data.get("split"),
        reviewed=data["reviewed"],
    )


def question_to_dict(question: Question) -> dict:
    """One line of the file, with the fields in the order of FIELDS."""
    return {
        "id": question.id,
        "language": question.language,
        "question": question.question,
        "expected_slugs": list(question.expected_slugs),
        "answer_headings": {slug: list(headings) for slug, headings in question.answer_headings.items()},
        "split": question.split,
        "reviewed": question.reviewed,
    }


# --- checking the questions against the guides ------------------------------


def guide_headings(guides: list[Guide]) -> dict[str, set[str]]:
    """For every guide, by slug, the headings of its passages ("" is the intro).

    These come from chunking, so a section that is left out of the index
    ("Related articles", or a heading with no text) is not listed: search can
    never return it, so an answer cannot be said to sit there.
    """
    return {guide.slug: {passage.heading for passage in chunk_guide(guide)} for guide in guides}


def validate_questions(questions: list[Question], headings_by_slug: dict[str, set[str]]) -> None:
    """Check every question. Raises one QuestionError listing all the problems."""
    problems = duplicate_id_problems(questions)
    for question in questions:
        problems.extend(f"{question.id}: {problem}" for problem in question_problems(question, headings_by_slug))
    if problems:
        raise QuestionError(problems)


def duplicate_id_problems(questions: list[Question]) -> list[str]:
    """One problem for each id that is used by more than one question."""
    seen: set[str] = set()
    reported: set[str] = set()
    problems: list[str] = []
    for question in questions:
        if question.id in seen and question.id not in reported:
            problems.append(f"{question.id}: this id is used more than once")
            reported.add(question.id)
        seen.add(question.id)
    return problems


def question_problems(question: Question, headings_by_slug: dict[str, set[str]]) -> list[str]:
    """What is wrong with one question, as plain sentences without its id."""
    problems: list[str] = []
    if question.language not in LANGUAGES:
        problems.append(f"language is {question.language!r}; it must be 'en' or 'ar'")
    if not question.question.strip():
        problems.append("the question is blank")
    if not question.answerable and question.language != "en":
        problems.append("a question the guides cannot answer must be in English")
    if question.split is not None and question.split not in SPLITS:
        problems.append(f"split is {question.split!r}; it must be 'dev' or 'test', or not set yet")

    for slug in question.expected_slugs:
        if slug not in headings_by_slug:
            problems.append(f"expected guide {slug!r} is not in the snapshot")
        if question.reviewed and not question.answer_headings.get(slug):
            problems.append(f"reviewed, but no section is listed for expected guide {slug!r}")

    for slug, headings in question.answer_headings.items():
        if slug not in question.expected_slugs:
            problems.append(f"answer_headings names {slug!r}, which is not an expected guide")
        elif slug in headings_by_slug:
            for heading in headings:
                if heading not in headings_by_slug[slug]:
                    problems.append(missing_section_problem(slug, heading))
    return problems


def missing_section_problem(slug: str, heading: str) -> str:
    """The sentence for a listed heading that the guide does not have."""
    if heading == "":
        return f"guide {slug!r} has no text before its first heading, so \"\" is not one of its sections"
    return f"guide {slug!r} has no section headed {heading!r}"


# --- the dev/test split -----------------------------------------------------


def assign_split(questions: list[Question], seed: int = SPLIT_SEED) -> list[Question]:
    """Give every question, reviewed or not, the split "dev" or "test".

    Within each group the ids are sorted, then shuffled with the fixed seed,
    and the first half (rounded up) is dev. Sorting first means the order of
    the file plays no part. The questions come back in the order they were
    given, with only `split` changed.
    """
    # With a repeated id there is no single sorted order, so the answer
    # would depend on the order of the file after all.
    problems = duplicate_id_problems(questions)
    if problems:
        raise QuestionError(problems)

    split_by_id: dict[str, str] = {}
    for group in sorted({question.group for question in questions}):
        ids = sorted(question.id for question in questions if question.group == group)
        # A new generator for every group, so that one group's halves do not
        # depend on how many questions another group has.
        random.Random(seed).shuffle(ids)
        dev_count = (len(ids) + 1) // 2  # half, rounded up
        for position, question_id in enumerate(ids):
            split_by_id[question_id] = "dev" if position < dev_count else "test"
    return [replace(question, split=split_by_id[question.id]) for question in questions]


def moved_questions(before: list[Question], after: list[Question]) -> list[str]:
    """The ids of the questions that already had a split and would now get the other one.

    `after` is what assign_split returned for `before`, so the two lists hold
    the same questions in the same order.
    """
    moved: list[str] = []
    for old, new in zip(before, after):
        if old.split is not None and old.split != new.split:
            moved.append(old.id)
    return moved


def split_summary(questions: list[Question]) -> str:
    """For example: "en: 20 dev, 20 test; ar: 5 dev, 5 test; unanswerable: 5 dev, 5 test"."""
    parts: list[str] = []
    for group in GROUPS:
        dev = len([q for q in questions if q.group == group and q.split == "dev"])
        test = len([q for q in questions if q.group == group and q.split == "test"])
        parts.append(f"{group}: {dev} dev, {test} test")
    return "; ".join(parts)


# --- the command line -------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run `validate` or `assign-split`. Returns 0, or 1 after printing the problems found."""
    parser = argparse.ArgumentParser(description="Check the evaluation questions, or assign the dev/test split.")
    parser.add_argument("command", choices=["validate", "assign-split"])
    parser.add_argument("--path", default=QUESTIONS_PATH, help="the question file")
    parser.add_argument("--snapshot", default=SNAPSHOT_PATH, help="the guide snapshot (used by validate)")
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.path)
        if args.command == "validate":
            validate_questions(questions, guide_headings(load_snapshot(args.snapshot)))
            print(f"{len(questions)} questions, no problems")
        else:
            assigned = assign_split(questions)
            moved = moved_questions(questions, assigned)
            if moved:
                # The split is dealt once. Dealing it again could move a
                # question the threshold was chosen on into `test`, so the
                # command stops here and leaves the file as it is.
                print("These questions already have a split, and assigning it again would move them:")
                print(", ".join(moved))
                print("That happens when questions were added or removed, or a split was edited by hand.")
                print(f"Nothing was saved to {args.path}")
                return 1
            # Running the command again on the same questions gets here: nothing moves.
            save_questions(args.path, assigned)
            print(split_summary(assigned))
    except QuestionError as error:
        for problem in error.problems:
            print(problem)
        count = len(error.problems)
        print(f"{count} problem{'' if count == 1 else 's'} in {args.path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
