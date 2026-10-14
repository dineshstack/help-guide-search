"""Choose the reranker: run every candidate on the dev questions.

    uv run --env-file .env python -m guide_search.evaluation.reranker_choice

For each candidate in cross_encoder.CANDIDATES it loads the model and runs
rerank search, with refusing switched off, on every reviewed dev question. It
writes what each candidate did to evaluation/reranker_choice.json and prints
the table that the report shows.

The rule was set before the run:

- the winner returns a passage that holds the answer for the most answerable
  dev questions;
- a tie goes to the faster model;
- a model slower than one second per question is out.

No candidate met that last part. The only one that did better than hybrid
search took about 8 seconds per question. After seeing that, the owner raised
the limit to 10 seconds. Nothing else in the rule changed, and the report says
so (CHOICE_NOTE in report.py).

Torch is limited to two threads, the CPUs the API container gets, so the
times are closer to what the container would see. They are still measured on
the machine this runs on, and the record names it.

Only `dev` questions are ever searched here. The chosen model is then pinned
by hand in cross_encoder.py.
"""

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

from guide_search.config import ConfigError, database_url
from guide_search.embedding import Embedder
from guide_search.evaluation.metrics import Outcome, summarise_answerable
from guide_search.evaluation.questions import QUESTIONS_PATH, Question, load_questions
from guide_search.evaluation.report import describe_machine, reranker_choice_lines
from guide_search.evaluation.run import CHOICE_PATH, NEVER_REFUSE, reviewed, search_outcomes
from guide_search.store import DatabaseUnavailable, IndexNotReady, connect, count_passages

THREADS = 2

# Seconds per question, as the median over the dev questions. It was 1.0 when
# the rule was set; see the top of this file.
SLOWEST = 10.0

TOO_SLOW = "slower than the limit"
CHOSEN = "chosen"


def entry_from(questions: list[Question], outcomes: dict[str, Outcome]) -> dict:
    """What one candidate did, in the form the record keeps."""
    summary = summarise_answerable(questions, outcomes)
    seconds = [outcome.seconds for outcome in outcomes.values()]
    return {
        "loaded": True,
        "note": None,
        "hit1": summary.hit1,
        "hit3": summary.hit3,
        "passage_hit3": summary.passage_hit3,
        "total": summary.total,
        "median_seconds": statistics.median(seconds),
        "max_seconds": max(seconds),
    }


def choose(entries: list[dict]) -> str | None:
    """Apply the rule. Marks each entry's note and returns the chosen name, or None."""
    eligible = []
    for entry in entries:
        if not entry["loaded"]:
            continue
        if entry["median_seconds"] > SLOWEST:
            entry["note"] = TOO_SLOW
        else:
            eligible.append(entry)
    if not eligible:
        return None
    most = max(entry["passage_hit3"] for entry in eligible)
    tied = [entry for entry in eligible if entry["passage_hit3"] == most]
    winner = min(tied, key=lambda entry: entry["median_seconds"])
    winner["note"] = CHOSEN
    return winner["name"]


REMOTE_CODE = "it cannot be loaded without running code from its own repository"


def load_note(error: Exception) -> str:
    """Why a candidate could not be loaded, in one short line."""
    message = str(error).strip()
    # The library names this setting when a model needs code of its own.
    if "trust_remote_code" in message:
        return REMOTE_CODE
    first_line = message.split("\n")[0]
    return f"it could not be loaded ({type(error).__name__}: {first_line[:160]})"


def run_choice(conn, embedder: Embedder, questions: list[Question], load_reranker) -> dict:
    """Run every candidate. `load_reranker(name)` returns a reranker or raises."""
    from guide_search.cross_encoder import CANDIDATES

    dev = reviewed(questions, "dev")
    entries = []
    for name, revision in CANDIDATES.items():
        print(f"{name}: loading", file=sys.stderr)
        try:
            reranker = load_reranker(name)
        except Exception as error:  # noqa: BLE001 - any failure to load rules the candidate out
            entries.append({"name": name, "revision": revision, "loaded": False, "note": load_note(error)})
            continue
        print(f"{name}: searching {len(dev)} dev questions", file=sys.stderr)
        outcomes = search_outcomes(conn, embedder, dev, "rerank", NEVER_REFUSE, reranker, NEVER_REFUSE)
        entries.append({"name": name, "revision": revision, **entry_from(dev, outcomes)})

    hybrid = entry_from(dev, search_outcomes(conn, embedder, dev, "hybrid", NEVER_REFUSE))
    chosen = choose(entries)
    return {
        "date": date.today().isoformat(),
        "machine": describe_machine(),
        "threads": THREADS,
        "limit_seconds": SLOWEST,
        "answerable": hybrid["total"],
        "candidates": entries,
        "hybrid": hybrid,
        "chosen": chosen,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every reranker candidate on the dev questions.")
    parser.add_argument("--questions", default=QUESTIONS_PATH, help="the question file")
    parser.add_argument("--output", default=CHOICE_PATH, help="where the record is written")
    args = parser.parse_args(argv)

    # Imported here and not at the top, because these imports load torch.
    import torch

    from guide_search.cross_encoder import CrossEncoderReranker
    from guide_search.e5 import E5Embedder

    torch.set_num_threads(THREADS)
    try:
        conn = connect(database_url(), read_only=True)
        embedder = E5Embedder()
        with conn:
            count_passages(conn, embedder.model_id)
            record = run_choice(conn, embedder, load_questions(args.questions), CrossEncoderReranker)
    except (ConfigError, DatabaseUnavailable, IndexNotReady) as error:
        print(error, file=sys.stderr)
        return 1

    Path(args.output).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(reranker_choice_lines({"reranker_choice": record})))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
