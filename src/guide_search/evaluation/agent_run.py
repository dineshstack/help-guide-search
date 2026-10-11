"""Measure the answer agent on the reviewed questions.

Stage 1 found that search never refuses: it returns its three closest guides
even for a question the guides do not cover. This script measures whether the
agent, which reads the passages, declines those questions, and how often it
answers the others from the right guide.

    uv run --env-file .env python -m guide_search.evaluation.agent_run run --split dev --model claude-opus-5-5
    uv run python -m guide_search.evaluation.agent_run report

"run" asks the agent every reviewed question of one split with one model and
stores one line per question in evaluation/agent_results.jsonl. It needs the
database, the embedding model and an API key, and it costs money. A question
that already has a current line is not asked again.

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

QUESTIONS_PATH = Path("evaluation/questions.jsonl")
RESULTS_PATH = Path("evaluation/agent_results.jsonl")
REPORT_PATH = Path("evaluation/agent_report.md")

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


def line_for(question: Question, result: AgentResult, model: str, effort: str, today: str) -> dict:
    """The stored line for one question."""
    return {
        "question_id": question.id,
        "question_hash": text_hash(question.question),
        "instructions_hash": instructions_hash(),
        "model": model,
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


def current_lines(lines: list[dict], questions: list[Question]) -> dict[tuple[str, str], dict]:
    """The latest current line for each (model, question id)."""
    by_id = {question.id: question for question in questions}
    kept: dict[tuple[str, str], dict] = {}
    for line in lines:
        question = by_id.get(line["question_id"])
        if question is not None and question.reviewed and is_current(line, question):
            kept[(line["model"], line["question_id"])] = line
    return kept


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


def split_table(questions: list[Question], kept: dict[tuple[str, str], dict], models: list[str], split: str) -> list[str]:
    """The table of one split: one row per model."""
    scored = [q for q in questions if q.reviewed and q.split == split]
    if not scored:
        return ["No reviewed question in this split."]
    answerable = [q for q in scored if q.answerable]
    unanswerable = [q for q in scored if not q.answerable]
    rows = [
        "| Model | Answered, citing an expected guide | Answered, citing only other guides | Handed over or asked back | Unanswerable: declined | Unanswerable: answered | Needed a rewrite | Tokens per question (in / out) | Cost per 100 questions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        lines = [kept[(model, q.id)] for q in scored if (model, q.id) in kept]
        if len(lines) != len(scored):
            rows.append(f"| `{model}` | not run on every question of this split ({len(lines)} of {len(scored)}) | | | | | | | |")
            continue
        outcomes = {q.id: outcome_of(q, kept[(model, q.id)]) for q in scored}
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
            f"| `{model}` | {count(right, len(answerable))} | {count(other, len(answerable))} | "
            f"{count(declined, len(answerable))} | {count(refused, len(unanswerable))} | "
            f"{count(len(unanswerable) - refused, len(unanswerable))} | {count(rewrites, len(lines))} | "
            f"{round(tokens_in / len(lines))} / {round(tokens_out / len(lines))} | {cost_text} |"
        )
    return rows


def wrong_lines(questions: list[Question], kept: dict[tuple[str, str], dict], models: list[str], split: str) -> list[str]:
    """Every result that is not what one would want, per model."""
    lines: list[str] = []
    for model in models:
        entries = []
        for question in questions:
            line = kept.get((model, question.id))
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
        lines.append(f"#### `{model}`")
        lines.append("")
        lines.extend(entries or ["- none"])
        lines.append("")
    return lines


def render_report(questions: list[Question], lines: list[dict]) -> str:
    """The whole of evaluation/agent_report.md."""
    kept = current_lines(lines, questions)
    models = sorted({model for model, _ in kept}, key=lambda m: -PRICES.get(m, (0, 0))[0])
    reviewed = [q for q in questions if q.reviewed]
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
        f"{len(questions) - len(reviewed)} not reviewed and not scored (the Arabic ones).",
        f"- Models: {', '.join(f'`{m}`' for m in models) or 'none yet'}; effort: {', '.join(efforts) or '-'}; "
        "every other request setting at its default.",
        f"- Each question was asked once per model. Run dates: {', '.join(dates) or '-'}.",
        "- Search: the default mode with the committed threshold. The agent sees up to 3 guides with up to 2 passages each.",
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
    for split, note in (("test", "The `test` questions played no part in writing the instructions."), ("dev", "These are the questions the instructions may be adjusted on.")):
        out += [f"## Results on {split}", "", note, ""]
        out += split_table(questions, kept, models, split)
        out += ["", f"### Results on {split} that are not what one would want", ""]
        out += wrong_lines(questions, kept, models, split)
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


def run(split: str, model: str, make_service=None, today: str | None = None) -> int:
    """Ask the agent every reviewed question of one split that has no current line yet."""
    # Imported here: building the service loads the embedding model.
    from guide_search import config
    from guide_search.drafting import AgentSettings, ClaudeDrafter, load_agent_settings, real_client

    settings = load_agent_settings()
    settings = AgentSettings(model=model, effort=settings.effort, max_tokens=settings.max_tokens)
    questions = [q for q in load_questions(QUESTIONS_PATH) if q.reviewed and q.split == split]
    done = current_lines(load_lines(), questions)
    todo = [q for q in questions if (model, q.id) not in done]
    print(f"{model}, {split}: {len(questions)} reviewed questions, {len(todo)} still to ask")
    if not todo:
        return 0

    if make_service is None:
        from guide_search.e5 import E5Embedder
        from guide_search.service import Service

        def make_service():
            return Service(config.database_url(), E5Embedder(), config.load_settings(), ClaudeDrafter(real_client(), settings))

    service = make_service()
    today = today or date.today().isoformat()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as file:
        for question in todo:
            result = service.answer(question.question)
            line = line_for(question, result, model, settings.effort, today)
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
            file.flush()
            print(f"  {question.id}: {result.decision} {line['cited_slugs']}")
    return 0


def report() -> int:
    questions = load_questions(QUESTIONS_PATH)
    REPORT_PATH.write_text(render_report(questions, load_lines()), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the answer agent on the reviewed questions.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="ask the agent the questions of one split (costs money)")
    run_parser.add_argument("--split", choices=["dev", "test"], required=True)
    run_parser.add_argument("--model", required=True)
    commands.add_parser("report", help="rebuild the report from the stored results (free)")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        return run(arguments.split, arguments.model)
    return report()


if __name__ == "__main__":
    sys.exit(main())
