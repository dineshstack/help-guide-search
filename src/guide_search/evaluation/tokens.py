"""Count the tokens of guide text a model must read per question, for every method.

    uv run --env-file .env python -m guide_search.evaluation.tokens

The report counts the text handed over in characters. A model is charged by
the token, so this command turns that text into tokens. For every reviewed
question of both question files, and every method:

- search: the search is run again with the committed settings, and the text
  is the returned passages, written exactly as the answer agent is shown them
  (with each guide's title and web address);
- title-picking: the text is the list of every guide's id, slug and title,
  then each picked guide whole. That is what an assistant reads when it picks
  from titles and then opens the guides it picked.

The question and the instructions are left out: they are the same for every
method.

Each text is counted with the API's token counter, for the answer agent's
model (agent.toml). Counting is free, but it needs ANTHROPIC_API_KEY. Every
count is kept in evaluation/token_counts.json under a hash of its text, so a
text is sent once, and running this again sends only texts it has not seen.

It writes evaluation/tokens.json, which `run final` reads and checks against
its own search results.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import psycopg

from guide_search import config
from guide_search.drafting import passages_text
from guide_search.embedding import Embedder
from guide_search.evaluation.questions import FRESH_PATH, QUESTIONS_PATH, SNAPSHOT_PATH, Question
from guide_search.evaluation.run import TOKENS_PATH, EvaluationError, load_all_questions
from guide_search.evaluation.title_picking import RESULTS_PATH as BASELINE_PATH
from guide_search.evaluation.title_picking import BaselineError, check_results, load_results
from guide_search.reranking import Reranker
from guide_search.search import MODES, NO_MATCH, search
from guide_search.snapshot import Guide, load_snapshot
from guide_search.store import DatabaseUnavailable, IndexNotReady, connect

COUNTS_PATH = Path("evaluation/token_counts.json")

# Counts a text: (model, text) -> tokens.
Counter = Callable[[str, str], int]


def title_picking_text(guides: list[Guide], slugs: list[str]) -> str:
    """The list of every guide, as the baseline shows it, then each picked guide whole."""
    by_slug = {guide.slug: guide for guide in guides}
    parts = ["\n".join(f"{guide.id} | {guide.slug} | {guide.title}" for guide in guides)]
    for slug in slugs:
        guide = by_slug[slug]
        parts.append(f"Guide id {guide.id}: {guide.title}\nWeb address: {guide.url}\n\n{guide.body}")
    return "\n\n---\n\n".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedCounter:
    """Counts each text once. Earlier counts come from the counts file."""

    def __init__(self, path: Path, model: str, count: Counter) -> None:
        self.path = path
        self.model = model
        self.count = count
        self.counts: dict[str, int] = {}
        self.sent = 0
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            # Counts made for another model are not used: its tokens differ.
            if stored["model"] == model:
                self.counts = stored["counts"]

    def tokens(self, text: str) -> int:
        key = text_hash(text)
        if key not in self.counts:
            self.counts[key] = self.count(self.model, text)
            self.sent += 1
        return self.counts[key]

    def save(self) -> None:
        document = {"model": self.model, "counts": dict(sorted(self.counts.items()))}
        self.path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


def count_entries(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker,
    questions: list[Question],
    guides: list[Guide],
    settings: config.Settings,
    picks: dict[str, dict],
    counter: CachedCounter,
) -> list[dict]:
    """One entry per reviewed question and method."""
    entries = []
    for question in questions:
        if not question.reviewed:
            continue
        slugs = picks[question.id]["slugs"]
        entries.append(
            {
                "question_id": question.id,
                "split": question.split,
                "method": "title_picking",
                "slugs": slugs,
                "refused": len(slugs) == 0,
                "tokens": counter.tokens(title_picking_text(guides, slugs)),
            }
        )
        for mode in MODES:
            result = search(
                conn, embedder, question.question, mode, settings.threshold, reranker, settings.rerank_threshold
            )
            refused = result.reason == NO_MATCH
            entries.append(
                {
                    "question_id": question.id,
                    "split": question.split,
                    "method": mode,
                    "slugs": [guide.slug for guide in result.guides],
                    "refused": refused,
                    # A refused search hands nothing over.
                    "tokens": 0 if refused else counter.tokens(passages_text(result.guides)),
                }
            )
    return entries


def api_counter(client) -> Counter:
    """A Counter that asks the API's token counter."""

    def count(model: str, text: str) -> int:
        response = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
        return response.input_tokens

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count the tokens of guide text a model must read per question.")
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--fresh", default=FRESH_PATH)
    parser.add_argument("--snapshot", default=SNAPSHOT_PATH)
    parser.add_argument("--baseline", default=BASELINE_PATH)
    parser.add_argument("--counts", type=Path, default=COUNTS_PATH)
    parser.add_argument("--output", type=Path, default=TOKENS_PATH)
    args = parser.parse_args(argv)

    from guide_search.drafting import load_agent_settings, real_client

    try:
        guides = load_snapshot(args.snapshot)
        questions = load_all_questions(args.questions, args.fresh, guides)
        scored = [question for question in questions if question.reviewed]
        picks = check_results(scored, load_results(args.baseline), guides)
        settings = config.load_settings()
        model = load_agent_settings().model
        conn = connect(config.database_url(), read_only=True)
    except (BaselineError, EvaluationError, config.ConfigError, DatabaseUnavailable, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    # Imported here and not at the top, because these imports load torch.
    from guide_search.cross_encoder import CrossEncoderReranker
    from guide_search.e5 import E5Embedder

    counter = CachedCounter(args.counts, model, api_counter(real_client()))
    try:
        with conn:
            entries = count_entries(
                conn, E5Embedder(), CrossEncoderReranker(), questions, guides, settings, picks, counter
            )
    except IndexNotReady as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        # Whatever was counted is kept, also after an error, so it is not sent again.
        counter.save()

    document = {"model": model, "date": date.today().isoformat(), "entries": entries}
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(entries)} counts, {counter.sent} texts sent to the token counter. Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
