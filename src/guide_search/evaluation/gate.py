"""The regression gate: does search still produce the committed results?

Continuous integration runs this after every push:

    uv run python -m guide_search.evaluation.gate

It runs the four search modes on every reviewed question of both question
files with the committed thresholds. For each question and mode it works out
two things: was the question refused, and was an expected guide among the top
three. It compares them with what evaluation/results.json holds for the
reported run, and fails when more than one question differs in any one mode.

One question of slack is allowed because embeddings differ in their last
decimal places from one machine to another, which can flip a near-tie.

The gate chooses nothing and scores nothing new. It needs the database with
the index built, the embedding model and the reranker, and no API key: the title-picking
baseline and the answer agent are not gated, because they need paid model
calls and vary from run to run.
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg

from guide_search import config
from guide_search.embedding import Embedder
from guide_search.evaluation.metrics import hit_at
from guide_search.evaluation.questions import FRESH_PATH, Question, load_questions
from guide_search.evaluation.run import search_outcomes
from guide_search.reranking import Reranker
from guide_search.store import connect

GATED_MODES = ("vector", "keyword", "hybrid", "rerank")

# How many questions may differ in one mode before the gate fails.
SLACK = 1

# Two thresholds closer than this are the same threshold.
SAME_THRESHOLD = 0.0005

QUESTIONS_PATH = Path("evaluation/questions.jsonl")
RESULTS_PATH = Path("evaluation/results.json")

# What the gate compares for one question in one mode: (refused, expected guide in the top three).
Pair = tuple[bool, bool]


class GateError(RuntimeError):
    """The gate cannot be run as things are. This is a setup problem, not a regression."""


def committed_pairs(document: dict, questions: list[Question]) -> dict[tuple[str, str], Pair]:
    """What the committed results say, for the gated modes and the reviewed questions."""
    wanted = {question.id for question in questions}
    pairs: dict[tuple[str, str], Pair] = {}
    for entry in document["outcomes"]:
        if entry["method"] in GATED_MODES and entry["question_id"] in wanted:
            pairs[(entry["method"], entry["question_id"])] = (entry["refused"], entry["hit_at_3"])
    missing = [(mode, question.id) for mode in GATED_MODES for question in questions if (mode, question.id) not in pairs]
    if missing:
        mode, question_id = missing[0]
        raise GateError(
            f"The committed results have no {mode} entry for {question_id} "
            f"({len(missing)} entries are missing). Rebuild the report before the gate can run."
        )
    return pairs


def measured_pairs(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    settings: config.Settings,
    reranker: Reranker,
) -> dict[tuple[str, str], Pair]:
    """What search produces now, in the same form."""
    pairs: dict[tuple[str, str], Pair] = {}
    by_id = {question.id: question for question in questions}
    for mode in GATED_MODES:
        outcomes = search_outcomes(
            conn, embedder, questions, mode, settings.threshold, reranker, settings.rerank_threshold
        )
        for question_id, outcome in outcomes.items():
            pairs[(mode, question_id)] = (outcome.refused, hit_at(by_id[question_id], outcome, 3))
    return pairs


def differing(committed: dict[tuple[str, str], Pair], measured: dict[tuple[str, str], Pair], mode: str) -> list[str]:
    """The questions whose outcome in this mode is not the committed one."""
    return sorted(
        question_id
        for (entry_mode, question_id), pair in committed.items()
        if entry_mode == mode and measured.get((entry_mode, question_id)) != pair
    )


def describe(pair: Pair | None) -> str:
    if pair is None:
        return "not measured"
    refused, hit = pair
    if refused:
        return "refused"
    return "an expected guide in the top three" if hit else "no expected guide in the top three"


def run_gate(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    document: dict,
    settings: config.Settings,
    reranker: Reranker,
) -> tuple[bool, list[str]]:
    """Return whether the gate passes, and the lines to print."""
    for name in ("threshold", "rerank_threshold"):
        if abs(document[name] - getattr(settings, name)) >= SAME_THRESHOLD:
            raise GateError(
                f"The committed results were produced with {name} {document[name]}, "
                f"but the settings file holds {getattr(settings, name)}. "
                "Rebuild the report before the gate can run."
            )
    reviewed = [question for question in questions if question.reviewed]
    committed = committed_pairs(document, reviewed)
    measured = measured_pairs(conn, embedder, reviewed, settings, reranker)

    passed = True
    lines = []
    for mode in GATED_MODES:
        different = differing(committed, measured, mode)
        verdict = "ok" if len(different) <= SLACK else "FAIL"
        lines.append(f"{mode}: {len(reviewed)} questions, {len(different)} differ from the committed results ({verdict})")
        for question_id in different:
            lines.append(
                f"  {question_id}: committed {describe(committed[(mode, question_id)])}, "
                f"now {describe(measured.get((mode, question_id)))}"
            )
        if len(different) > SLACK:
            passed = False
    lines.append(
        "The gate passes." if passed else f"The gate fails: more than {SLACK} question differs in at least one mode."
    )
    return passed, lines


def main(
    argv: list[str] | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Check that search still produces the committed results.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--fresh", type=Path, default=FRESH_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--settings", type=Path, default=config.SETTINGS_PATH)
    arguments = parser.parse_args(argv)

    questions = load_questions(arguments.questions)
    if arguments.fresh.exists():
        questions += load_questions(arguments.fresh)
    document = json.loads(arguments.results.read_text(encoding="utf-8"))
    settings = config.load_settings(arguments.settings)
    # Imported here and not at the top, because these imports load torch.
    if embedder is None:
        from guide_search.e5 import E5Embedder

        embedder = E5Embedder()
    if reranker is None:
        from guide_search.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()

    try:
        with connect(config.database_url(), read_only=True) as conn:
            passed, lines = run_gate(conn, embedder, questions, document, settings, reranker)
    except GateError as error:
        print(f"The gate could not run: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
