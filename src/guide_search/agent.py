"""The answer agent: search, draft, check, and at most one rewrite.

One question goes in. One result comes out, and it is one of three things:
an answer taken from the help guides with its sources, a question back to the
user, or a hand-over to a person.

    search ──> draft ──> check ──┬── passes ───────────────────> done
                 ^               ├── hand_over, or fails again ─> hand over
                 │               └── fails, first time ──┐
                 └──────── rewrite, told the reason ─────┘

- search is the search of this project, with its default mode.
- draft asks a Claude model (drafting.py) for a draft.
- check is plain code (checks.py).

The loop is written with LangGraph: each step is a small function, and the
arrows above are "edges" that say which step runs next. A step receives the
"state" (everything known so far) and returns the part of it that changed.

Rules that hold whatever the model does:

- At most two model calls per question: the draft, and one rewrite.
- A draft that fails the check twice is never returned. Nor is a draft from a
  call that failed, was refused or was cut off. All of these hand over.
- A database fault is not a hand-over. It is raised as an error, because "we
  could not look" must never look like an answer.

To ask one question from the command line:

    uv run --env-file .env python -m guide_search.agent "how do I reset my password"
"""

import argparse
import json
import operator
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from guide_search.checks import Draft, Failure, check_draft
from guide_search.drafting import Drafter, ModelCall, ModelFault
from guide_search.ranking import GuideHit, GuideSummary, SearchResult

# The most model calls for one question: the draft and one rewrite.
MAX_MODEL_CALLS = 2

# What the user is told on a hand-over. It is fixed text, because nothing the
# model wrote in a failed or refused draft has passed the check.
HAND_OVER_REPLY = "I can't answer this from the help guides, so a person needs to help you with it."

# Why a question was handed over. One of these three, sometimes with a detail.
MODEL_CHOSE = "the_guides_do_not_answer_this"
CHECK_FAILED = "check_failed_twice"
MODEL_FAULT = "model_fault"

DEFAULT_USAGE_LOG = Path("usage.jsonl")


class AgentNotConfigured(Exception):
    """The answer agent was asked for, but no model is set up for it."""


@dataclass(frozen=True)
class CheckSummary:
    attempts: int  # how many drafts were written: 1 or 2 (0 if the model never answered)
    failures: list[str]  # the rule each failed draft broke, in order


@dataclass(frozen=True)
class AgentResult:
    """The whole result for one question."""

    question: str
    decision: str  # "answer", "clarify" or "hand_over"
    reply: str
    sources: list[GuideSummary]  # the guides an answer cites
    hand_over_reason: str | None
    check: CheckSummary
    usage: list[ModelCall]  # one record for every model call made


class State(TypedDict, total=False):
    """Everything the steps share. Each step returns only the keys it changes."""

    question: str
    guides: list[GuideHit]
    draft: Draft | None
    failure: Failure | None
    fault: str | None
    attempts: int
    # "Annotated[..., operator.add]" tells LangGraph to add what a step returns
    # to the list, instead of replacing the list. So every call stays recorded.
    failures: Annotated[list[str], operator.add]
    usage: Annotated[list[ModelCall], operator.add]


def build_graph(search: Callable[[str], SearchResult], drafter: Drafter):
    """Wire the steps together. `search` takes a question and returns a search result."""

    def search_step(state: State) -> State:
        # A database fault raised here is not caught: it ends the run as an error.
        return {"guides": search(state["question"]).guides}

    def draft_step(state: State) -> State:
        # On the second visit there is a failed draft and its reason to show the model.
        failure = state.get("failure")
        previous = state.get("draft") if failure is not None else None
        try:
            draft, call = drafter.draft(state["question"], state["guides"], previous, failure)
        except ModelFault as fault:
            calls = [fault.call] if fault.call is not None else []
            return {"fault": fault.reason, "usage": calls}
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1, "usage": [call]}

    def check_step(state: State) -> State:
        failure = check_draft(state["draft"], state["guides"])
        if failure is None:
            return {"failure": None}
        return {"failure": failure, "failures": [failure.code]}

    def after_draft(state: State) -> str:
        # No draft to check when the model could not be used.
        return "stop" if state.get("fault") else "check"

    def after_check(state: State) -> str:
        if state["failure"] is None:
            return "stop"
        # One rewrite only. This counter, not a library setting, is what
        # guarantees there is never a third call.
        return "rewrite" if state["attempts"] < MAX_MODEL_CALLS else "stop"

    graph = StateGraph(State)
    graph.add_node("search", search_step)
    graph.add_node("draft", draft_step)
    graph.add_node("check", check_step)
    graph.add_edge(START, "search")
    graph.add_edge("search", "draft")
    graph.add_conditional_edges("draft", after_draft, {"check": "check", "stop": END})
    graph.add_conditional_edges("check", after_check, {"rewrite": "draft", "stop": END})
    return graph.compile()


def result_from(state: State) -> AgentResult:
    """Turn the final state into the result. This is where a hand-over is decided."""
    question = state["question"]
    usage = state.get("usage", [])
    check = CheckSummary(attempts=state.get("attempts", 0), failures=state.get("failures", []))

    def hand_over(reason: str) -> AgentResult:
        return AgentResult(
            question=question,
            decision="hand_over",
            reply=HAND_OVER_REPLY,
            sources=[],
            hand_over_reason=reason,
            check=check,
            usage=usage,
        )

    if state.get("fault"):
        return hand_over(f"{MODEL_FAULT}: {state['fault']}")
    failure = state.get("failure")
    if failure is not None:
        return hand_over(f"{CHECK_FAILED}: {failure.code}")
    draft = state["draft"]
    if draft.decision == "hand_over":
        return hand_over(MODEL_CHOSE)

    cited = [guide for guide in state["guides"] if guide.guide_id in draft.cited_guide_ids]
    sources = [
        GuideSummary(guide_id=guide.guide_id, slug=guide.slug, title=guide.title, url=guide.url)
        for guide in cited
    ]
    return AgentResult(
        question=question,
        decision=draft.decision,
        reply=draft.reply,
        sources=sources,
        hand_over_reason=None,
        check=check,
        usage=usage,
    )


class Agent:
    """Answers one question at a time. It keeps no memory between questions."""

    def __init__(
        self,
        search: Callable[[str], SearchResult],
        drafter: Drafter,
        usage_log: Path | None = None,
    ) -> None:
        self.graph = build_graph(search, drafter)
        self.usage_log = usage_log

    def answer(self, question: str) -> AgentResult:
        # recursion_limit is only a backstop. The loop is bounded by the
        # attempts counter in after_check.
        state = self.graph.invoke({"question": question}, config={"recursion_limit": 12})
        result = result_from(state)
        self._record_usage(result)
        return result

    def _record_usage(self, result: AgentResult) -> None:
        """Append one line per model call to the usage log. The question is not written."""
        if self.usage_log is None or not result.usage:
            return
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with open(self.usage_log, "a", encoding="utf-8") as file:
            for call in result.usage:
                record = {"time": now, "decision": result.decision, **asdict(call)}
                file.write(json.dumps(record) + "\n")


def format_result(result: AgentResult) -> str:
    """The result as the command line prints it."""
    lines = [f"Decision: {result.decision}"]
    if result.hand_over_reason:
        lines.append(f"Reason:   {result.hand_over_reason}")
    lines.append("")
    lines.append(result.reply)
    if result.sources:
        lines.append("")
        lines.append("Sources:")
        lines.extend(f"- {source.title}  {source.url}" for source in result.sources)
    lines.append("")
    lines.append(f"Drafts written: {result.check.attempts}; failed the check: {len(result.check.failures)}")
    for call in result.usage:
        lines.append(
            f"- {call.purpose}: {call.model}, {call.input_tokens} tokens in, "
            f"{call.output_tokens} out, {call.seconds:.1f} s"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None, make_service=None) -> int:
    parser = argparse.ArgumentParser(description="Answer one question from the help guides.")
    parser.add_argument("question")
    arguments = parser.parse_args(argv)

    # Imported here and not at the top: building the real service loads the
    # embedding model, which takes seconds.
    from guide_search.ranking import clean_question
    from guide_search.store import DatabaseUnavailable, IndexNotReady

    try:
        question = clean_question(arguments.question)
    except ValueError as error:
        print(f"Not asked: {error}.", file=sys.stderr)
        return 2

    if make_service is None:
        from guide_search.service import real_service as make_service
    try:
        result = make_service().answer(question)
    except (DatabaseUnavailable, IndexNotReady, AgentNotConfigured) as error:
        print(f"Could not answer: {error}", file=sys.stderr)
        return 1
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
