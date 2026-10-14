"""The evaluation runner: put the questions to every method and write down what came back.

    uv run --env-file .env python -m guide_search.evaluation.run threshold
    uv run --env-file .env python -m guide_search.evaluation.run dev
    uv run --env-file .env python -m guide_search.evaluation.run final --questions-commit COMMIT --prompt-revisions N

- threshold  chooses on the dev questions the two "too weak to be a match"
             thresholds (one on vector similarity, one on the reranker's
             score) and the default search mode, writes them to
             settings.toml and prints how they were chosen.
- dev        prints the dev tables for all five methods. It writes nothing.
- final      scores dev, test and the fresh questions with the values in
             settings.toml, and writes evaluation/results.json and
             evaluation/report.md. It has to be told two things that the
             report states and no file holds: the git commit of the question
             files, and how often the baseline prompt was revised after its
             first run (0 if never).

Three rules are kept here in code:

- only questions marked `reviewed: true` are scored;
- only `final` ever searches a test or fresh question, so nothing can be
  chosen on them. Before it searches one, it checks that settings.toml holds
  the values the rules give on dev;
- when something cannot be trusted (a missing or out-of-date baseline answer,
  an empty index, an index that does not match the snapshot, a split that is
  not the one the script assigns) the command stops with a message and
  writes nothing.

The search methods need the index in Postgres (DATABASE_URL), the embedding
model and the reranker. Title-picking needs none of them: its answers were
stored when the baseline was run, and are read from
evaluation/title_picking.jsonl.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import psycopg

from guide_search.chunking import chunk_guide
from guide_search.config import (
    SETTINGS_PATH,
    ConfigError,
    Settings,
    database_url,
    load_settings,
    save_settings,
)
from guide_search.embedding import Embedder
from guide_search.evaluation.metrics import (
    Outcome,
    choose_default_mode,
    choose_threshold,
    hit_at,
    passage_hit,
)
from guide_search.evaluation.questions import (
    FRESH_PATH,
    QUESTIONS_PATH,
    SNAPSHOT_PATH,
    Question,
    QuestionError,
    assign_split,
    guide_headings,
    load_questions,
    validate_questions,
)
from guide_search.evaluation.report import (
    METHODS,
    RANKING_ONLY_MODES,
    THRESHOLD_TOLERANCE,
    default_mode_lines,
    describe_machine,
    miss_lines,
    render_report,
    rerank_threshold_lines,
    split_lines,
    threshold_lines,
)
from guide_search.evaluation.title_picking import RESULTS_PATH as BASELINE_PATH
from guide_search.evaluation.title_picking import BaselineError, check_results, current_lines, load_results
from guide_search.ranking import SearchResult
from guide_search.reranking import Reranker
from guide_search.search import MODES, NO_MATCH, search
from guide_search.snapshot import Guide, load_snapshot
from guide_search.store import DatabaseUnavailable, IndexNotReady, connect, count_passages

OUTPUT_DIR = Path("evaluation")
RESULTS_FILE = "results.json"
REPORT_FILE = "report.md"

# Written by guide_search.evaluation.reranker_choice and by
# guide_search.evaluation.tokens. Both are optional: the report has a section
# for each only when its file is there.
CHOICE_PATH = Path("evaluation/reranker_choice.json")
TOKENS_PATH = Path("evaluation/tokens.json")

# The log of label corrections. The file is optional: without it there are none.
CORRECTIONS_PATH = Path("evaluation/corrections.jsonl")
CORRECTION_KEYS = ("question_id", "date", "old_expected_slugs", "new_expected_slugs", "reason")

# The first search of a run is slower than the rest: the model and the
# database both have things to set up. One search is therefore run and thrown
# away before any search is timed. Its wording does not matter.
WARM_UP_QUESTION = "warm-up question"

# The threshold that switches refusing off. Search refuses when the best
# score is below the threshold. The real model's similarities lie far above 0,
# the fake embedder's are never below it, and no reranker score is below it,
# so with this value vector, hybrid and rerank search always answer.
NEVER_REFUSE = 0.0


class EvaluationError(RuntimeError):
    """The evaluation cannot be run as things are. The message says why, in plain words."""


# --- what each method did with each question --------------------------------


def search_outcomes(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    mode: str,
    threshold: float,
    reranker: Reranker | None = None,
    rerank_threshold: float = NEVER_REFUSE,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Outcome]:
    """Search for every reviewed question in one mode. Returns question id -> Outcome.

    `reranker` and `rerank_threshold` are only used in rerank mode. `clock`
    is only given by the tests, which need times they can predict.
    """
    search(conn, embedder, WARM_UP_QUESTION, mode, threshold, reranker, rerank_threshold)

    outcomes: dict[str, Outcome] = {}
    for question in questions:
        if not question.reviewed:
            continue
        # The time is taken around the whole search, embedding the question
        # and reranking included.
        started = clock()
        result = search(conn, embedder, question.question, mode, threshold, reranker, rerank_threshold)
        seconds = clock() - started
        outcomes[question.id] = _search_outcome(question.id, result, seconds)
    return outcomes


def _search_outcome(question_id: str, result: SearchResult, seconds: float) -> Outcome:
    """Keep the parts of one search result that the measures need."""
    headings: dict[str, tuple[str, ...]] = {}
    chars = 0
    for guide in result.guides:
        headings[guide.slug] = tuple(passage.heading for passage in guide.passages)
        # Text handed over: the passage texts only, without title, heading or URL.
        chars += sum(len(passage.text) for passage in guide.passages)
    return Outcome(
        question_id=question_id,
        method=result.mode,
        refused=result.reason == NO_MATCH,
        slugs=tuple(guide.slug for guide in result.guides),
        headings=headings,
        chars=chars,
        seconds=seconds,
        best_similarity=result.best_similarity,
        best_rerank_score=result.best_rerank_score,
    )


def title_picking_outcomes(
    questions: list[Question],
    lines: dict[str, dict],
    guides: list[Guide],
) -> dict[str, Outcome]:
    """Turn the stored title-picking answers into outcomes. Returns question id -> Outcome.

    `lines` is what check_results returns: question id -> the stored line.
    """
    body_length = {guide.slug: len(guide.body) for guide in guides}

    outcomes: dict[str, Outcome] = {}
    for question in questions:
        if not question.reviewed:
            continue
        if question.id not in lines:
            # A missing answer must never be counted as "the model picked nothing".
            raise BaselineError(f"{question.id}: no stored title-picking result")
        line = lines[question.id]
        slugs = tuple(line["slugs"])
        outcomes[question.id] = Outcome(
            question_id=question.id,
            method="title_picking",
            refused=len(slugs) == 0,
            slugs=slugs,
            # None means: the whole guide is handed over, not chosen passages.
            headings=None,
            chars=sum(body_length[slug] for slug in slugs),
            # The model's response time, recorded when the baseline was run.
            seconds=line["seconds"],
            best_similarity=None,
        )
    return outcomes


# --- reading what the run needs ---------------------------------------------


def reviewed(questions: list[Question], split: str) -> list[Question]:
    """The reviewed questions of one split. Only these are ever scored."""
    return [question for question in questions if question.reviewed and question.split == split]


def check_questions(questions: list[Question], guides: list[Guide]) -> None:
    """Stop unless the question file fits the guides and holds the split the script assigns."""
    validate_questions(questions, guide_headings(guides))
    if any(question.split is None for question in questions):
        raise EvaluationError(
            "The questions have no dev/test split yet. "
            "Run: python -m guide_search.evaluation.questions assign-split"
        )
    # Nobody edits `split` by hand. A question moved from dev to test would be
    # scored as a test question although the threshold was chosen on it. The
    # script always deals the same split, so the file can be checked against it.
    if assign_split(questions) != questions:
        raise EvaluationError("The split in the question file is not the one assign-split gives.")


def check_fresh(fresh: list[Question], questions: list[Question], guides: list[Guide]) -> None:
    """Stop unless the fresh file fits the guides, is all "fresh", and shares no id with the other file."""
    validate_questions(fresh, guide_headings(guides))
    if any(question.split != "fresh" for question in fresh):
        raise EvaluationError('Every question in the fresh file must have the split "fresh".')
    shared = sorted({question.id for question in fresh} & {question.id for question in questions})
    if shared:
        raise EvaluationError("The fresh file uses ids that the question file uses too: " + ", ".join(shared))


def load_optional_json(path: Path) -> dict | None:
    """The content of a JSON file, or None when there is no such file."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def matching_tokens(tokens: dict | None, outcomes: dict[str, dict[str, Outcome]]) -> dict | None:
    """The token counts, keeping only those made from the same result as this run's.

    The counts are made by a separate command, which searches again, perhaps
    on another machine. Search gives the same result for the same question,
    but the last decimal places of a score can differ between machines and
    turn a near-tie the other way. A count whose guides are not the ones this
    run returned is left out, and `left_out` says how many were.
    """
    if tokens is None:
        return None
    kept = []
    for entry in tokens["entries"]:
        outcome = outcomes.get(entry["method"], {}).get(entry["question_id"])
        if outcome is not None and list(outcome.slugs) == entry["slugs"] and outcome.refused == entry["refused"]:
            kept.append(entry)
    left_out = len([entry for entry in tokens["entries"] if entry not in kept])
    return {**tokens, "entries": kept, "left_out": left_out}


def check_index(conn: psycopg.Connection, embedder: Embedder, guides: list[Guide]) -> int:
    """Return how many passages the index holds. Stop if it was not built from this snapshot."""
    # count_passages raises IndexNotReady when the table is missing or empty.
    stored = count_passages(conn, embedder.model_id)
    expected = sum(len(chunk_guide(guide)) for guide in guides)
    if stored != expected:
        raise EvaluationError(
            f"The index holds {stored} passages, but the snapshot gives {expected}. "
            "Run the ingestion again, so that the index matches the snapshot."
        )
    return stored


def snapshot_date(path: str | Path) -> str:
    """The date the snapshot was fetched on. load_snapshot returns the guides only."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["fetched_at"]


def load_corrections(path: Path, questions: list[Question]) -> list[dict]:
    """Read the log of label corrections: one JSON object per line.

    An expected guide may be changed after the questions are committed, but
    only when the change is logged here, so that the report can list it.
    """
    if not path.exists():
        return []
    expected_now = {question.id: sorted(question.expected_slugs) for question in questions}

    corrections: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            correction = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"{path} line {number}: not valid JSON") from error
        if not isinstance(correction, dict) or set(correction) != set(CORRECTION_KEYS):
            raise EvaluationError(
                f"{path} line {number}: a correction has exactly these fields: " + ", ".join(CORRECTION_KEYS)
            )
        if correction["question_id"] not in expected_now:
            raise EvaluationError(f"{path} line {number}: there is no question {correction['question_id']!r}")
        corrections.append(correction)

    # A dict keeps one value per key, so this leaves the last correction of each question.
    latest = {correction["question_id"]: correction for correction in corrections}
    for question_id, correction in latest.items():
        if sorted(correction["new_expected_slugs"]) != expected_now[question_id]:
            raise EvaluationError(
                f"{path}: the last correction of {question_id} does not give the expected guides "
                "that the question file has now."
            )
    return corrections


# --- what `final` checks before it scores ------------------------------------


def check_final_options(args: argparse.Namespace) -> None:
    """Stop unless `final` was told the two things that the report states and no file holds."""
    if args.questions_commit is None:
        raise EvaluationError(
            "final needs --questions-commit: the git commit of the question file, for the report. "
            "Find it with: git log -n 1 --format=%H -- evaluation/questions.jsonl"
        )
    if args.prompt_revisions is None:
        raise EvaluationError(
            "final needs --prompt-revisions: how often the baseline prompt was revised "
            "after its first run, for the report. Pass 0 if it never was."
        )


def check_prompt_revisions(
    questions: list[Question],
    stored: list[dict],
    guides: list[Guide],
    revisions: int,
) -> None:
    """Stop when --prompt-revisions is 0 but the baseline file shows an earlier request.

    A stored line is out of date when its question hash or its request hash is
    not that of the request as it would be sent now. Such a line for a scored
    question means the question was asked more than once. The report would
    then be wrong to say "asked once" and "not revised".
    """
    if revisions != 0:
        return
    asked_before: list[str] = []
    for question in questions:
        own = [line for line in stored if line["question_id"] == question.id]
        # Every line of this question that is not a current one is out of date.
        if len(current_lines(question, own, guides)) < len(own):
            asked_before.append(question.id)
    if asked_before:
        raise EvaluationError(
            "--prompt-revisions is 0, but the baseline file holds an out-of-date answer for: "
            + ", ".join(asked_before)
            + ". Such an answer means the question was sent more than once, so the report cannot say "
            "that the prompt was never revised. Pass the number of revisions with --prompt-revisions."
        )


def check_settings(
    dev: list[Question],
    similarities: dict[str, float],
    rerank_scores: dict[str, float],
    outcomes: dict[str, dict[str, Outcome]],
    settings: Settings,
) -> None:
    """Stop unless the settings file holds the values the rules give on dev.

    `similarities` holds the best vector similarity of every dev question, and
    `rerank_scores` the reranker's best score. `outcomes` are the dev outcomes
    with the settings' thresholds. This runs before any test or fresh question
    is searched, so a value that was never chosen, or chosen for other guides,
    cannot use up the scoring of those questions.
    """
    wrong = []
    choice = choose_threshold([(similarities[question.id], question.answerable) for question in dev])
    if abs(settings.threshold - choice.threshold) >= THRESHOLD_TOLERANCE:
        wrong.append("threshold")
    choice = choose_threshold([(rerank_scores[question.id], question.answerable) for question in dev])
    if abs(settings.rerank_threshold - choice.threshold) >= THRESHOLD_TOLERANCE:
        wrong.append("rerank_threshold")
    if choose_default_mode(dev, outcomes["hybrid"], outcomes["rerank"]) != settings.default_mode:
        wrong.append("default_mode")
    if wrong:
        raise EvaluationError(
            "settings.toml does not hold the values the rules give on dev ("
            + ", ".join(wrong)
            + "). Run `threshold` first. No test or fresh question was searched."
        )


# --- running the methods ----------------------------------------------------


def run_methods(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    guides: list[Guide],
    settings: Settings,
    lines: dict[str, dict],
    reranker: Reranker,
) -> dict:
    """Score the given questions with all five methods.

    `lines` is what check_results returned: the stored title-picking answer of
    each question. Returns the part of the report's context that holds what
    the methods did.
    """
    outcomes = {"title_picking": title_picking_outcomes(questions, lines, guides)}
    for mode in MODES:
        outcomes[mode] = search_outcomes(
            conn, embedder, questions, mode, settings.threshold, reranker, settings.rerank_threshold
        )

    # "Ranking only": the same search with refusing switched off, so that a
    # weak result can be traced to the ranking or to the refusing. When a
    # mode's threshold is already the one that refuses nothing, the search
    # would be the very same, so it is not run a second time.
    ranking_only = {}
    for mode in RANKING_ONLY_MODES:
        used = settings.rerank_threshold if mode == "rerank" else settings.threshold
        if used == NEVER_REFUSE:
            ranking_only[mode] = outcomes[mode]
        else:
            ranking_only[mode] = search_outcomes(conn, embedder, questions, mode, NEVER_REFUSE, reranker, NEVER_REFUSE)

    return {"outcomes": outcomes, "ranking_only": ranking_only}


def best_scores(outcomes: dict[str, Outcome], field: str) -> dict[str, float]:
    """Question id -> one of the two best scores ("best_similarity" or "best_rerank_score")."""
    return {question_id: getattr(outcome, field) for question_id, outcome in outcomes.items()}


def results_document(context: dict) -> dict:
    """The content of results.json: what every method did with every scored question."""
    entries = []
    for question in context["questions"]:
        if not question.reviewed:
            continue
        for method in METHODS:
            outcome = context["outcomes"][method][question.id]
            entries.append(
                {
                    "question_id": question.id,
                    "split": question.split,
                    "method": method,
                    "refused": outcome.refused,
                    "slugs": list(outcome.slugs),
                    "hit_at_3": hit_at(question, outcome, 3),
                    "passage_hit": passage_hit(question, outcome),
                    "best_similarity": outcome.best_similarity,
                    "best_rerank_score": outcome.best_rerank_score,
                }
            )
    return {
        "threshold": context["threshold"],
        "rerank_threshold": context["rerank_threshold"],
        "default_mode": context["default_mode"],
        "fetched_at": context["snapshot_date"],
        "model_id": context["model_id"],
        "reranker_id": context["reranker_id"],
        "outcomes": entries,
    }


# --- the three commands -----------------------------------------------------


def run_threshold(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    settings_path: Path,
    reranker: Reranker,
) -> None:
    """Choose the two thresholds and the default mode on the reviewed dev questions, and save them."""
    dev = reviewed(questions, "dev")
    if not dev:
        raise EvaluationError("No dev question is reviewed yet, so there is nothing to choose a threshold from.")

    # Searches with refusing switched off: the best score of each question is
    # what each threshold is chosen from.
    similarities = best_scores(search_outcomes(conn, embedder, dev, "vector", NEVER_REFUSE), "best_similarity")
    reranked = search_outcomes(conn, embedder, dev, "rerank", NEVER_REFUSE, reranker, NEVER_REFUSE)
    rerank_scores = best_scores(reranked, "best_rerank_score")
    threshold = choose_threshold([(similarities[q.id], q.answerable) for q in dev]).threshold
    rerank_threshold = choose_threshold([(rerank_scores[q.id], q.answerable) for q in dev]).threshold

    # The default mode is chosen on the two modes as they will run: each with its threshold.
    outcomes = {"hybrid": search_outcomes(conn, embedder, dev, "hybrid", threshold)}
    if rerank_threshold == NEVER_REFUSE:
        outcomes["rerank"] = reranked
    else:
        outcomes["rerank"] = search_outcomes(conn, embedder, dev, "rerank", threshold, reranker, rerank_threshold)
    default_mode = choose_default_mode(dev, outcomes["hybrid"], outcomes["rerank"])

    save_settings(Settings(threshold, default_mode, rerank_threshold), settings_path)

    context = {
        "questions": questions,
        "dev_similarities": similarities,
        "dev_rerank_scores": rerank_scores,
        "threshold": threshold,
        "rerank_threshold": rerank_threshold,
        "default_mode": default_mode,
        "outcomes": outcomes,
    }
    print("\n".join(threshold_lines(context) + rerank_threshold_lines(context) + default_mode_lines(context)))
    print(f"Wrote threshold {threshold!r}, rerank_threshold {rerank_threshold!r} "
          f"and default_mode {default_mode!r} to {settings_path}")


def run_dev(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    guides: list[Guide],
    settings: Settings,
    baseline_path: Path,
    reranker: Reranker,
) -> None:
    """Print the dev tables and the dev misses. No other question is touched."""
    dev = reviewed(questions, "dev")
    if not dev:
        raise EvaluationError("No dev question is reviewed yet, so there is nothing to score.")

    # The stored title-picking answers are checked first. That is quick, and a
    # missing or out-of-date answer stops the run before any search is made.
    lines = check_results(dev, load_results(baseline_path), guides)

    context = {"questions": questions, "threshold": settings.threshold}
    context.update(run_methods(conn, embedder, dev, guides, settings, lines, reranker))

    print(f"Thresholds used: {settings.threshold!r} and, for rerank, {settings.rerank_threshold!r} "
          "(from the settings file)")
    print()
    print("\n".join(split_lines(context, "dev") + miss_lines(context, "dev")))


def run_final(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[Question],
    guides: list[Guide],
    settings: Settings,
    passage_count: int,
    args: argparse.Namespace,
    reranker: Reranker,
) -> None:
    """Score dev, check the settings, score test and fresh, then write results.json and report.md."""
    dev = reviewed(questions, "dev")
    test = reviewed(questions, "test")
    fresh = reviewed(questions, "fresh")
    if not dev and not test and not fresh:
        raise EvaluationError("No question is reviewed yet, so there is nothing to score.")

    output_dir = Path(args.output_dir)
    results_path = output_dir / RESULTS_FILE
    report_path = output_dir / REPORT_FILE

    # The stored title-picking answers are checked first. That is quick, and a
    # missing or out-of-date answer stops the run before any search is made.
    stored = load_results(Path(args.baseline))
    lines = check_results(dev + test + fresh, stored, guides)
    check_prompt_revisions(dev + test + fresh, stored, guides, args.prompt_revisions)

    # All the command can see is whether a results file from an earlier run is
    # in the output folder, and which splits it holds. The report says that
    # and no more.
    earlier = load_optional_json(results_path)
    earlier_splits = {entry["split"] for entry in earlier["outcomes"]} if earlier else set()
    context = {
        "snapshot_date": snapshot_date(args.snapshot),
        "guide_count": len(guides),
        "passage_count": passage_count,
        "questions_commit": args.questions_commit,
        "model_id": embedder.model_id,
        "reranker_id": reranker.model_id,
        "baseline_models": sorted({line["model"] for line in lines.values()}),
        "baseline_dates": sorted({line["date"] for line in lines.values()}),
        "prompt_revisions": args.prompt_revisions,
        "threshold": settings.threshold,
        "rerank_threshold": settings.rerank_threshold,
        "default_mode": settings.default_mode,
        "test_scored_before": earlier is not None,
        "fresh_scored_before": "fresh" in earlier_splits,
        "run_date": date.today().isoformat(),
        "machine": describe_machine(),
        "questions": questions,
        "corrections": load_corrections(Path(args.corrections), questions),
        "reranker_choice": load_optional_json(Path(args.choice)),
    }

    # dev is scored first. Its "ranking only" searches give the best scores
    # of every dev question, which is what the threshold rules work on. So
    # the settings can be checked before any other question is searched, and
    # no dev question has to be searched an extra time for it.
    scored = run_methods(conn, embedder, dev, guides, settings, lines, reranker)
    similarities = best_scores(scored["ranking_only"]["vector"], "best_similarity")
    rerank_scores = best_scores(scored["ranking_only"]["rerank"], "best_rerank_score")
    check_settings(dev, similarities, rerank_scores, scored["outcomes"], settings)

    # Only now are test and fresh scored. Their outcomes are added to those of dev.
    for later in (test, fresh):
        later_scored = run_methods(conn, embedder, later, guides, settings, lines, reranker)
        for method in METHODS:
            scored["outcomes"][method].update(later_scored["outcomes"][method])
        for mode in RANKING_ONLY_MODES:
            scored["ranking_only"][mode].update(later_scored["ranking_only"][mode])

    context.update(scored)
    # The threshold sections of the report show how the rules work out on this run's dev scores.
    context["dev_similarities"] = similarities
    context["dev_rerank_scores"] = rerank_scores

    context["tokens"] = matching_tokens(load_optional_json(Path(args.tokens)), context["outcomes"])

    # Both texts are built before either file is written, so that a failure
    # cannot leave a new report beside old results.
    report = render_report(context)
    results = json.dumps(results_document(context), ensure_ascii=False, indent=2) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    results_path.write_text(results, encoding="utf-8")
    print(f"Wrote {report_path} and {results_path}")


def load_all_questions(questions_path: str | Path, fresh_path: str | Path, guides: list[Guide]) -> list[Question]:
    """The questions of both files, each file checked. The fresh file is optional."""
    questions = load_questions(questions_path)
    check_questions(questions, guides)
    fresh: list[Question] = []
    if Path(fresh_path).exists():
        fresh = load_questions(fresh_path)
        check_fresh(fresh, questions, guides)
    return questions + fresh


def main(
    argv: list[str] | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> int:
    """Run `threshold`, `dev` or `final`. Returns 0, or 1 after saying why it could not run.

    `embedder` and `reranker` are only given by the tests. Left out, the real
    models are used.
    """
    parser = argparse.ArgumentParser(description="Score the search methods on the evaluation questions.")
    parser.add_argument("command", choices=["threshold", "dev", "final"])
    parser.add_argument("--questions", default=QUESTIONS_PATH, help="the question file")
    parser.add_argument("--fresh", default=FRESH_PATH, help="the fresh question file, if there is one")
    parser.add_argument("--snapshot", default=SNAPSHOT_PATH, help="the guide snapshot")
    parser.add_argument("--baseline", default=BASELINE_PATH, help="the stored title-picking answers")
    parser.add_argument("--settings", default=SETTINGS_PATH, help="the settings file that holds the thresholds")
    parser.add_argument("--corrections", default=CORRECTIONS_PATH, help="the log of label corrections, if there is one")
    parser.add_argument("--choice", default=CHOICE_PATH, help="the record of the reranker choice, if there is one")
    parser.add_argument("--tokens", default=TOKENS_PATH, help="the token counts, if there are any")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="where final writes results.json and report.md")
    # The next two have no default on purpose. `final` stops without them,
    # because the report states both and no file holds either.
    parser.add_argument(
        "--questions-commit",
        help="final needs it: the git commit of the question files, for the report",
    )
    parser.add_argument(
        "--prompt-revisions",
        type=int,
        help="final needs it: how often the baseline prompt was revised after its first run (0 if never)",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "final":
            # Checked before anything else is read or opened.
            check_final_options(args)
        url = database_url()
        guides = load_snapshot(args.snapshot)
        questions = load_all_questions(args.questions, args.fresh, guides)
        settings = load_settings(Path(args.settings))

        # Connect before the models are loaded: a database that cannot be
        # reached is then reported at once, not after seconds of loading.
        conn = connect(url)
        # Imported here and not at the top, because these imports load torch,
        # which takes seconds. Importing this module must stay quick.
        if embedder is None:
            from guide_search.e5 import E5Embedder

            embedder = E5Embedder()
        if reranker is None:
            from guide_search.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()

        # "with conn" closes the connection when the block ends.
        with conn:
            passage_count = check_index(conn, embedder, guides)
            if args.command == "threshold":
                run_threshold(conn, embedder, questions, Path(args.settings), reranker)
            elif args.command == "dev":
                run_dev(conn, embedder, questions, guides, settings, Path(args.baseline), reranker)
            else:
                run_final(conn, embedder, questions, guides, settings, passage_count, args, reranker)
    except (
        EvaluationError,
        BaselineError,
        ConfigError,
        QuestionError,
        DatabaseUnavailable,
        IndexNotReady,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
