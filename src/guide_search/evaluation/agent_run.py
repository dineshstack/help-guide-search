"""Measure the answer agent on the reviewed questions.

Stage 1 found that search never refuses: it returns its three closest guides
even for a question the guides do not cover. This script measures whether the
agent, which reads the passages, declines those questions, and how often it
answers the others from the right guide.

    uv run --env-file .env python -m guide_search.evaluation.agent_run run --split dev --model claude-opus-5-5
    uv run --env-file .env python -m guide_search.evaluation.agent_run run --split test --model claude-opus-5-5 --mode rerank
    uv run python -m guide_search.evaluation.agent_run report

"run" asks the agent every reviewed question of one split with one model and
one search mode (the default mode unless --mode names another), and stores
one line per question in evaluation/agent_results.jsonl. It needs the
database, the models and an API key, and it costs money. A question that
already has a current line for that model and mode is not asked again.

"report" rebuilds evaluation/agent_report.md from the stored lines. It needs
no key and no database.

The rules of stage 1 apply: the instructions are adjusted on "dev" only, and
"test" is scored after they are fixed. A stored line records a hash of the
instructions it was produced with, and a line from older instructions is not
counted.

What this does not measure: whether the text of an answer is correct. "Cites
an expected guide" is the nearest thing that code can check.
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from guide_search.agent import AgentResult
from guide_search.drafting import DRAFT_SCHEMA, INSTRUCTIONS
from guide_search.evaluation.questions import Question, load_questions
from guide_search.search import MODES

QUESTIONS_PATH = Path("evaluation/questions.jsonl")
FRESH_PATH = Path("evaluation/questions_fresh.jsonl")
RESULTS_PATH = Path("evaluation/agent_results.jsonl")
REPORT_PATH = Path("evaluation/agent_report.md")

# Lines stored before reranking was added name no search mode. They were all
# searched in hybrid mode, the default then.
MODE_OF_OLD_LINES = "hybrid"

# The splits in the order the report shows them, each with its note.
SPLIT_NOTES = (
    ("test", "The `test` questions played no part in writing the instructions."),
    (
        "fresh",
        "The fresh questions were written after the instructions were fixed and after every other "
        "result here was known. Nothing was adjusted on them.",
    ),
    ("dev", "These are the questions the instructions may be adjusted on."),
)

# US dollars per million tokens (input, output), from the public price list on
# 11 October 2026. Used only to turn token counts into an estimated cost.
PRICES = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# What happened to one question, as the report counts it.
ANSWERED_RIGHT = "answered, citing an expected guide"
ANSWERED_OTHER = "answered, citing only other guides"
DECLINED = "handed over or asked back"
ANSWERED_UNANSWERABLE = "answered"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instructions_hash() -> str:
    """One value that changes whenever the instructions or the reply shape change."""
    return text_hash(INSTRUCTIONS + json.dumps(DRAFT_SCHEMA, sort_keys=True))


def line_for(
    question: Question, result: AgentResult, model: str, effort: str, today: str, mode: str = MODE_OF_OLD_LINES
) -> dict:
    """The stored line for one question."""
    return {
        "question_id": question.id,
        "question_hash": text_hash(question.question),
        "instructions_hash": instructions_hash(),
        "model": model,
        "mode": mode,
        "effort": effort,
        "split": question.split,
        "date": today,
        "decision": result.decision,
        "cited_slugs": [source.slug for source in result.sources],
        "hand_over_reason": result.hand_over_reason,
        "attempts": result.check.attempts,
        "failures": result.check.failures,
        "model_calls": len(result.usage),
        "input_tokens": sum(call.input_tokens or 0 for call in result.usage),
        "output_tokens": sum(call.output_tokens or 0 for call in result.usage),
        "seconds": round(sum(call.seconds for call in result.usage), 2),
    }


def line_mode(line: dict) -> str:
    return line.get("mode", MODE_OF_OLD_LINES)


def load_all_questions() -> list[Question]:
    """The questions of both question files. The fresh file is optional."""
    questions = load_questions(QUESTIONS_PATH)
    if FRESH_PATH.exists():
        questions += load_questions(FRESH_PATH)
    return questions


def load_lines(path: Path = RESULTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_current(line: dict, question: Question) -> bool:
    """A line counts only if the question and the instructions are still what they were."""
    return (
        line["question_hash"] == text_hash(question.question)
        and line["instructions_hash"] == instructions_hash()
        and line["split"] == question.split
    )


# A run is one model with one search mode.
Run = tuple[str, str]


def current_lines(lines: list[dict], questions: list[Question]) -> dict[tuple[str, str, str], dict]:
    """The latest current line for each (model, mode, question id)."""
    by_id = {question.id: question for question in questions}
    kept: dict[tuple[str, str, str], dict] = {}
    for line in lines:
        question = by_id.get(line["question_id"])
        if question is not None and question.reviewed and is_current(line, question):
            kept[(line["model"], line_mode(line), line["question_id"])] = line
    return kept


def run_label(run: Run) -> str:
    model, mode = run
    return f"`{model}`, {mode}"


def outcome_of(question: Question, line: dict) -> str:
    """Sort one result into the kind the report counts."""
    answered = line["decision"] == "answer"
    if not question.answerable:
        return ANSWERED_UNANSWERABLE if answered else DECLINED
    if not answered:
        return DECLINED
    cites_expected = any(slug in question.expected_slugs for slug in line["cited_slugs"])
    return ANSWERED_RIGHT if cites_expected else ANSWERED_OTHER


def estimated_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """US dollars for these tokens, or None when the model has no listed price."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def count(part: int, whole: int) -> str:
    if whole == 0:
        return "-"
    return f"{part} of {whole} ({round(100 * part / whole)}%)"


def split_table(questions: list[Question], kept: dict[tuple[str, str, str], dict], runs: list[Run], split: str) -> list[str]:
    """The table of one split: one row per run."""
    scored = [q for q in questions if q.reviewed and q.split == split]
    if not scored:
        return ["No reviewed question in this split."]
    answerable = [q for q in scored if q.answerable]
    unanswerable = [q for q in scored if not q.answerable]
    rows = [
        "| Model, search mode | Answered, citing an expected guide | Answered, citing only other guides | Handed over or asked back | Unanswerable: declined | Unanswerable: answered | Needed a rewrite | Tokens per question (in / out) | Cost per 100 questions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        model, mode = run
        lines = [kept[(model, mode, q.id)] for q in scored if (model, mode, q.id) in kept]
        if not lines:
            continue
        if len(lines) != len(scored):
            rows.append(f"| {run_label(run)} | not run on every question of this split ({len(lines)} of {len(scored)}) | | | | | | | |")
            continue
        outcomes = {q.id: outcome_of(q, kept[(model, mode, q.id)]) for q in scored}
        right = sum(outcomes[q.id] == ANSWERED_RIGHT for q in answerable)
        other = sum(outcomes[q.id] == ANSWERED_OTHER for q in answerable)
        declined = sum(outcomes[q.id] == DECLINED for q in answerable)
        refused = sum(outcomes[q.id] == DECLINED for q in unanswerable)
        rewrites = sum(line["attempts"] > 1 for line in lines)
        tokens_in = sum(line["input_tokens"] for line in lines)
        tokens_out = sum(line["output_tokens"] for line in lines)
        cost = estimated_cost(model, tokens_in, tokens_out)
        cost_text = "-" if cost is None else f"${100 * cost / len(lines):.2f}"
        rows.append(
            f"| {run_label(run)} | {count(right, len(answerable))} | {count(other, len(answerable))} | "
            f"{count(declined, len(answerable))} | {count(refused, len(unanswerable))} | "
            f"{count(len(unanswerable) - refused, len(unanswerable))} | {count(rewrites, len(lines))} | "
            f"{round(tokens_in / len(lines))} / {round(tokens_out / len(lines))} | {cost_text} |"
        )
    return rows


def wrong_lines(questions: list[Question], kept: dict[tuple[str, str, str], dict], runs: list[Run], split: str) -> list[str]:
    """Every result that is not what one would want, per run."""
    lines: list[str] = []
    for run in runs:
        model, mode = run
        if not any((model, mode, q.id) in kept for q in questions if q.reviewed and q.split == split):
            continue
        entries = []
        for question in questions:
            line = kept.get((model, mode, question.id))
            if line is None or not question.reviewed or question.split != split:
                continue
            outcome = outcome_of(question, line)
            if outcome == ANSWERED_RIGHT or (outcome == DECLINED and not question.answerable):
                continue
            expected = ", ".join(f"`{slug}`" for slug in question.expected_slugs) or "no guide"
            cited = ", ".join(f"`{slug}`" for slug in line["cited_slugs"]) or "nothing"
            reason = f" ({line['hand_over_reason']})" if line["hand_over_reason"] else ""
            entries.append(
                f"- `{question.id}` {question.question}\n"
                f"  - expected: {expected}; the agent: {line['decision']}{reason}, citing {cited}"
            )
        lines.append(f"#### {run_label(run)}")
        lines.append("")
        lines.extend(entries or ["- none"])
        lines.append("")
    return lines


def bar_lines(questions: list[Question], kept: dict[tuple[str, str, str], dict], runs: list[Run]) -> list[str]:
    """Whether each rerank run met the agent's half of the bar, on test and on the fresh questions."""
    out = [
        "## The bar for relying on search",
        "",
        "Set before rerank was run on `test` or on the fresh questions: with rerank search, the agent "
        "declines at most 3 of 20 answerable questions, and still declines every question the guides "
        "cannot answer. The other half of the bar, about the passages search returns, is in `report.md`.",
        "",
        "| Model, search mode | Questions | Answerable declined | At most | Unanswerable declined | Met |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = 0
    for run in runs:
        model, mode = run
        if mode != "rerank":
            continue
        for split in ("test", "fresh"):
            scored = [q for q in questions if q.reviewed and q.split == split]
            if not scored or any((model, mode, q.id) not in kept for q in scored):
                continue
            answerable = [q for q in scored if q.answerable]
            unanswerable = [q for q in scored if not q.answerable]
            declined = sum(outcome_of(q, kept[(model, mode, q.id)]) == DECLINED for q in answerable)
            refused = sum(outcome_of(q, kept[(model, mode, q.id)]) == DECLINED for q in unanswerable)
            # 3 of 20, scaled to the number of answerable questions and rounded down.
            allowed = 3 * len(answerable) // 20
            met = declined <= allowed and refused == len(unanswerable)
            out.append(
                f"| {run_label(run)} | {split} | {count(declined, len(answerable))} | {allowed} | "
                f"{count(refused, len(unanswerable))} | {'yes' if met else 'no'} |"
            )
            rows += 1
    if rows == 0:
        return out[:4] + ["No rerank run covers every question of `test` or of the fresh questions yet.", ""]
    return out + [""]


def render_report(questions: list[Question], lines: list[dict]) -> str:
    """The whole of evaluation/agent_report.md."""
    kept = current_lines(lines, questions)
    # Dearest model first; within a model, the modes in their usual order.
    runs = sorted(
        {(model, mode) for model, mode, _ in kept},
        key=lambda run: (-PRICES.get(run[0], (0, 0))[0], run[0], MODES.index(run[1])),
    )
    models = sorted({model for model, _ in runs}, key=lambda m: -PRICES.get(m, (0, 0))[0])
    reviewed = [q for q in questions if q.reviewed]
    unreviewed = [q for q in questions if not q.reviewed]
    unreviewed_note = " (the Arabic ones)" if unreviewed and all(q.language == "ar" for q in unreviewed) else ""
    efforts = sorted({line["effort"] for line in kept.values()})
    dates = sorted({line["date"] for line in kept.values()})
    out = [
        "# Answer agent: evaluation report",
        "",
        "Stage 1 found that search never refuses: it returns its three closest guides even for a "
        "question the guides do not cover. This report measures the answer agent, which reads the "
        "passages search found: how often it answers from an expected guide, and how often it "
        "declines the questions the guides cannot answer. The numbers are reported as measured.",
        "",
        "## What was run",
        "",
        f"- Questions: {len(reviewed)} reviewed and scored; "
        f"{len(questions) - len(reviewed)} not reviewed and not scored{unreviewed_note}.",
        f"- Models: {', '.join(f'`{m}`' for m in models) or 'none yet'}; effort: {', '.join(efforts) or '-'}; "
        "every other request setting at its default.",
        f"- Each question was asked once per model. Run dates: {', '.join(dates) or '-'}.",
        "- Search: the mode named in each row, with the committed thresholds. The agent sees up to 3 guides with up to 2 passages each.",
        f"- Instructions: the version with hash `{instructions_hash()[:12]}`. Every result counted here was produced by this version; "
        "any adjustment to it is made on `dev` only.",
        "",
        "## How to read the tables",
        "",
        "- **Answered, citing an expected guide**: the agent answered and at least one guide it cited is one a reviewer listed for the question.",
        "- **Answered, citing only other guides**: the agent answered from guides the reviewer did not list. The answer may still be right; nobody judged its text.",
        "- **Handed over or asked back**: the agent declined a question the guides can answer.",
        "- **Unanswerable: declined**: the correct outcome for a question no guide covers. **Unanswerable: answered** is the wrong one.",
        "- **Needed a rewrite**: the first draft failed the check and the model was asked once more.",
        "- **Cost** is estimated from the token counts and the public prices of 11 October 2026.",
        "",
    ]
    if any(mode == "rerank" for _, mode in runs):
        out += bar_lines(questions, kept, runs)
    for split, note in SPLIT_NOTES:
        if split == "fresh" and not any(q.reviewed and q.split == "fresh" for q in questions):
            continue
        out += [f"## Results on {split}", "", note, ""]
        out += split_table(questions, kept, runs, split)
        out += ["", f"### Results on {split} that are not what one would want", ""]
        out += wrong_lines(questions, kept, runs, split)
    out += [
        "## Limits",
        "",
        "- Citing an expected guide is not the same as answering correctly. Judging the text of an answer needs a person or another model, and was not done.",
        "- The set is small: about 20 answerable and 5 unanswerable questions per split. One unanswerable question is 20 points.",
        "- Each question was asked once. A model's output varies from run to run, so a second run could move a result by a question or two.",
        "- The agent only sees what search returned. When search misses the right guide, the best the agent can do is decline.",
        "- The expected guides are one reviewer's labels.",
        "- Only English questions are scored. The Arabic ones wait for a reviewer who reads Arabic.",
        "",
    ]
    return "\n".join(out)


def run(
    split: str,
    model: str,
    make_service=None,
    today: str | None = None,
    mode: str = MODE_OF_OLD_LINES,
) -> int:
    """Ask the agent every reviewed question of one split that has no current line yet for this model and mode."""
    # Imported here: building the service loads the models.
    from dataclasses import replace

    from guide_search import config
    from guide_search.drafting import AgentSettings, ClaudeDrafter, load_agent_settings, real_client

    settings = load_agent_settings()
    settings = AgentSettings(model=model, effort=settings.effort, max_tokens=settings.max_tokens)
    questions = [q for q in load_all_questions() if q.reviewed and q.split == split]
    done = current_lines(load_lines(), questions)
    todo = [q for q in questions if (model, mode, q.id) not in done]
    print(f"{model}, {mode}, {split}: {len(questions)} reviewed questions, {len(todo)} still to ask")
    if not todo:
        return 0

    if make_service is None:
        from guide_search.cross_encoder import CrossEncoderReranker
        from guide_search.e5 import E5Embedder
        from guide_search.service import Service

        def make_service():
            # The agent searches in the service's default mode, so the mode
            # asked for is made the default here.
            search_settings = replace(config.load_settings(), default_mode=mode)
            return Service(
                config.database_url(),
                E5Embedder(),
                search_settings,
                ClaudeDrafter(real_client(), settings),
                reranker=CrossEncoderReranker(),
            )

    service = make_service()
    today = today or date.today().isoformat()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as file:
        for question in todo:
            result = service.answer(question.question)
            line = line_for(question, result, model, settings.effort, today, mode)
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
            file.flush()
            print(f"  {question.id}: {result.decision} {line['cited_slugs']}")
    return 0


def report() -> int:
    questions = load_all_questions()
    REPORT_PATH.write_text(render_report(questions, load_lines()), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the answer agent on the reviewed questions.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="ask the agent the questions of one split (costs money)")
    run_parser.add_argument("--split", choices=["dev", "test", "fresh"], required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--mode", choices=MODES, help="the search mode; the default mode if left out")
    commands.add_parser("report", help="rebuild the report from the stored results (free)")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        mode = arguments.mode
        if mode is None:
            from guide_search import config

            mode = config.load_settings().default_mode
        return run(arguments.split, arguments.model, mode=mode)
    return report()


if __name__ == "__main__":
    sys.exit(main())
