"""The evaluation report: the text of evaluation/report.md.

`render_report(context)` turns what the methods did into one markdown text.
It searches nothing and reads no file. Everything it needs is handed to it in
`context`, a plain dictionary that run.py fills in:

    snapshot_date       the snapshot's fetched_at
    guide_count         how many guides the snapshot holds
    passage_count       how many passages the index holds
    questions_commit    the git commit of the question file
    model_id            the embedding model and its revision
    baseline_models     the model ids stored with the title-picking answers
    baseline_dates      the dates those answers were recorded on
    prompt_revisions    how often the baseline prompt was revised after its first run
    threshold           the threshold this run used (from settings.toml)
    test_scored_before  True when a results file from an earlier run was in the output folder
    run_date            the date of this run
    machine             describe_machine()
    questions           every question in both question files, reviewed or not
    outcomes            method -> question id -> Outcome, with the thresholds
    ranking_only        search mode -> question id -> Outcome, with the thresholds at 0
    dev_similarities    question id -> best similarity, for the reviewed dev questions
    corrections         the logged label corrections, each a dict

and, from the round that added reranking:

    reranker_id          the reranker and its revision
    rerank_threshold     the reranker threshold this run used (from settings.toml)
    default_mode         the default search mode (from settings.toml)
    fresh_scored_before  True when the earlier results file already held fresh questions
    dev_rerank_scores    question id -> the reranker's best score, for the reviewed dev questions
    reranker_choice      the record of how the reranker was chosen, or None
    tokens               the token counts, or None

A report of a run without rerank (the tests build such contexts) leaves out
what belongs to rerank. Only reviewed questions are counted, and splits are
never added together: every table is for one split.

The smaller functions each write one section and return it as a list of
lines. run.py also uses three of them on their own, to print the threshold
table and the dev tables.
"""

import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

from guide_search.evaluation.metrics import (
    Outcome,
    choose_default_mode,
    choose_threshold,
    hit_at,
    sign_test,
    summarise_answerable,
    summarise_unanswerable,
)
from guide_search.evaluation.questions import Question

# The five methods, in the order of every table.
METHODS = ("title_picking", "vector", "keyword", "hybrid", "rerank")
METHOD_NAMES = {
    "title_picking": "Title-picking",
    "vector": "Vector",
    "keyword": "Keyword",
    "hybrid": "Hybrid",
    "rerank": "Rerank",
}

# The three search modes that have a threshold. They are run a second time
# with the threshold switched off.
RANKING_ONLY_MODES = ("vector", "hybrid", "rerank")

# `test` comes first in the report, because its numbers are the ones reported,
# then the fresh questions, which nothing had touched.
SPLITS = ("test", "fresh", "dev")

# Search is good enough to rely on when a returned passage holds the answer
# for at least this share of the answerable questions: 18 of 20.
BAR_SHARE = 0.9

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}

# The headline says "better" or "worse" only when the sign test gives a
# p-value below this.
SIGNIFICANCE = 0.05

# Embeddings differ in their last decimal places from one machine to another.
# Two thresholds closer together than this count as the same value. run.py
# uses the same number to decide whether `final` may go on to the test questions.
THRESHOLD_TOLERANCE = 0.0005

GIGABYTE = 2**30

# Inside a Docker container, Linux keeps the container's limits in this folder.
CONTAINER_LIMITS_FOLDER = Path("/sys/fs/cgroup")

# The notes of a run without rerank.
SPLIT_NOTES = {
    "test": (
        "The `test` questions played no part in any choice. "
        "The threshold was frozen before they were scored."
    ),
    "dev": (
        "The threshold was chosen on these questions. The refusal numbers of vector and hybrid "
        "search are therefore better here than they would be on new questions."
    ),
}

# The notes of a run with rerank.
RERANK_SPLIT_NOTES = {
    "test": (
        "No value was chosen on the `test` questions: the thresholds and the default mode were "
        "frozen before they were scored."
    ),
    "fresh": (
        "The `fresh` questions were written and reviewed after every result above them in this "
        "file was known from earlier runs, and before any rerank search was run on them. No value "
        "was chosen on them. Unlike `test`, whose earlier results were known when reranking was "
        "added, nothing had been measured on them."
    ),
    "dev": (
        "The thresholds, the default mode and the reranker were chosen on these questions. The "
        "refusal numbers of vector, hybrid and rerank search are therefore better here than they "
        "would be on new questions."
    ),
}

MEASURES = [
    "- **hit@1**: the first guide returned is an expected one.",
    "- **hit@3**: an expected guide is among those returned.",
    "- **Passage hit@3**: hit@3, and one of that guide's returned passages comes from a section "
    "that holds the answer. For title-picking it equals hit@3, because the whole guide is handed over.",
    "- **MRR**: 1 divided by the position of the first expected guide (1, 1/2, 1/3, or 0 if it is "
    "absent), averaged over the questions.",
    "- **Wrong refusals**: questions the guides can answer for which nothing was returned. "
    "Such a question scores 0 on every measure above.",
    "- **Guides returned**: the average number of guides returned per question that was not refused. "
    "A method may return fewer than three.",
    "- **Text handed over**: characters, as the median over the questions that were not refused. "
    "For search it is the total length of the returned passage texts. For title-picking it is the "
    "total length of the picked guides' whole bodies. Titles, headings and URLs are not counted.",
    "- **Correct refusals**: questions the guides cannot answer for which nothing was returned. "
    "The rest were answered when they should have been refused.",
]

HEADLINE_NOTE = (
    "This is the comparison that was named before any method was run: the English `test` questions "
    "the guides can answer, on hit@1 and on hit@3. The two counts are the questions that only one "
    "of the two methods got right. Questions that both got right, or both got wrong, do not enter "
    'the test. "Better" or "worse" is said only when an exact two-sided sign test on the two counts '
    "gives p below 0.05. All other numbers in this report are descriptive."
)

TIME_NOTE = (
    "Per question, over all scored questions together. Search is timed on the machine named above, "
    "around the search function, including embedding the question and, for rerank, scoring the "
    "candidates, after one discarded warm-up query. Title-picking time is the model's response time over the network, recorded when the "
    "baseline was run. These are different kinds of time. They are shown side by side and are not compared."
)

THRESHOLD_NOTE = (
    "Vector and hybrid search return no match when the best similarity of a question is below the "
    "threshold. It is chosen on the reviewed `dev` questions only. The candidates are 0, which "
    "refuses nothing, and the midpoint between each pair of neighbouring best similarities. A "
    "candidate handles a question correctly when it lets through a question the guides can answer, "
    "or refuses one they cannot. The candidate with the highest count is used, and the lowest of "
    "them on a tie. The same value serves vector and hybrid search, and both languages."
)

ZERO_NOTE = (
    "The rule chose 0. On these questions similarity did not separate the questions the guides "
    "cannot answer from the ones they can. With a threshold of 0, vector and hybrid search never refuse."
)

RERANK_THRESHOLD_NOTE = (
    "Rerank search returns no match when the reranker's best score for a question is below this "
    "threshold. The reranker reads the question and each passage together, so its score is a direct "
    "judgement of whether the passage answers the question, which a similarity is not. The rule is "
    "the one above, applied to the reranker's best scores on the reviewed `dev` questions. Its "
    "scores lie between 0 and 1."
)

RERANK_ZERO_NOTE = (
    "The rule chose 0. On these questions the reranker's score did not separate the questions the "
    "guides cannot answer from the ones they can either. With a threshold of 0, rerank search never refuses."
)

DEFAULT_MODE_NOTE = (
    "The default mode is what a request that names no mode gets, the answer agent included. "
    "Rerank becomes the default only when, on the reviewed `dev` questions and with each mode's "
    "threshold, it returns a passage that holds the answer for more answerable questions than "
    "hybrid does, and refuses at least as many of the unanswerable ones. Otherwise hybrid stays "
    "the default, because it needs no reranker."
)

BAR_NOTE = (
    "The bar was set before rerank was run on `test` or on the fresh questions: a returned passage "
    "must hold the answer for at least 18 of 20 answerable questions (90%). Only then can an "
    "assistant rely on the passages instead of reading a list of titles and then whole guides. "
    "The second half of the bar, that the answer agent declines at most 3 of 20 answerable "
    "questions while still declining the unanswerable ones, is in `agent_report.md`."
)

TOKENS_NOTE = (
    "The guide text a model must read to answer one question, in tokens, counted with the API's "
    "token counter for the model named above. For search it is the returned passages, written as "
    "the answer agent is shown them, with their guide titles and web addresses. For title-picking "
    "it is the list of all guide titles plus the whole picked guides. The question and the "
    "instructions are left out: they are the same for every method. Each count includes the few "
    "tokens the API adds to any message. A refused question counts 0 and is left out of the median. "
    "Fewer tokens are only worth having when the right passage is among them: read this beside "
    "passage hit@3."
)

CHOICE_NOTE = (
    "Every candidate reranked the candidates of hybrid search for the reviewed `dev` questions, "
    "with refusing switched off. The rule, set before the first run: the most questions with a "
    "returned passage that holds the answer; a tie goes to the faster model; a model slower than "
    "one second per question is out. **No candidate met the time limit.** The only one that did "
    "better than hybrid search took about 8 seconds per question on two threads. After seeing "
    "that, the owner raised the limit to {limit:g} seconds; nothing else in the rule changed. The "
    "table is from the run after that change. The candidates are open models with a licence that "
    "allows commercial use, trained on Arabic as well as English, that run on a CPU and load "
    "without running code from the model's own repository."
)

MISSES_NOTE = (
    "A miss is a question the guides can answer for which no expected guide was among those "
    "returned, or a question the guides cannot answer that was answered anyway."
)

# The second limit says how many of the scored questions are unanswerable.
# Those two numbers are counted when the report is written (see limit_lines),
# because the first report covers fewer questions than the full set.
LIMITS = [
    "- The set is small. In `test`, one English question moves hit@1 or hit@3 by about 5 points "
    "(1 of about 20). One Arabic question, or one unanswerable question, moves its result by about "
    "20 points (1 of about 5).",
    "- The threshold is set from about five unanswerable `dev` questions and checked on about five "
    "more. The refusal numbers show whether refusing works at all, not how often. {unanswerable} of "
    "the {scored} scored questions are unanswerable because the set was built that way; the "
    "threshold depends on that mix.",
    "- One threshold serves both languages and is fitted mostly on English questions. Arabic wrong "
    "refusals show what that costs. Refusing Arabic questions that the guides do not cover is not measured.",
    "- Keyword mode has no threshold, so its refusal numbers follow from question wording and "
    "language, not from tuning. Vector and hybrid share one refusal rule, so their refusal numbers "
    "are the same by construction.",
    "- The questions are drafted by a model that has read the guides. They may echo guide wording "
    "more than real customers would, which flatters keyword and vector search. The drafting rules "
    "and the review reduce this; they do not remove it.",
    "- The expected guides are one reviewer's labels, and the reviewer also built the search. "
    "A missing acceptable guide turns a right result into a miss.",
    "- Title-picking here is one choice from id, slug and title. An assistant that reads a guide "
    "before choosing the next may do better, so beating this baseline is not the same as beating "
    "such an assistant. The slug is shown as well as the title, and many slugs contain words the "
    "title does not, so the baseline has slightly more than titles to go on. Its model is not the "
    "newest available; a newer one might pick better. It was run once, and a second run could move "
    "its numbers by a question or two.",
    "- A guide-level hit for search does not mean the returned passages hold the answer; passage "
    "hit@3 is there to show the difference. Text handed over is only like for like when read beside it.",
    "- Arabic questions are matched against English passages. Published benchmarks for this "
    "embedding model show that this is clearly weaker than English to English, so lower Arabic "
    "results are expected. The baseline's model reads Arabic directly and has no such handicap.",
]

# The limits that belong to rerank. They are added only when rerank was run.
RERANK_LIMITS = [
    "- The reranker was chosen on the `dev` questions from three candidates, so its `dev` "
    "numbers flatter it a little. The `test` questions had been scored before, and their earlier "
    "results were known when reranking was added. The fresh questions are the only ones nothing "
    "had touched, and there are 25 of them.",
    "- Rerank can only reorder what hybrid search found: a passage that neither vector nor "
    "keyword search returns among its first 20 never reaches the reranker.",
]

# Some limits are about Arabic results. While no Arabic question is reviewed
# the report has no such results, and this line under the list says so.
ARABIC_LIMITS_NOTE = (
    "The points above about Arabic results do not apply to this report yet: "
    "no Arabic question has been reviewed."
)


# --- the machine ------------------------------------------------------------


def describe_machine() -> str:
    """The machine the run happened on, in one line.

    Processor, cores, memory, operating system and Python version. Never a
    host name, a user name or a file path: the report is public.
    """
    parts = [
        processor_name(),
        f"{os.cpu_count()} cores",
        f"{memory_gb()} GB memory",
        operating_system(),
        f"Python {platform.python_version()}",
    ]
    description = ", ".join(parts)
    limits = container_limits()
    if limits is not None:
        # The numbers above are those of the whole machine. A container may
        # be allowed less, and that is what the timings depend on.
        description += f"; this container is limited to {limits}"
    return description


def processor_name() -> str:
    """The processor's name, or its family when the system gives no name."""
    name = ""
    if sys.platform == "darwin":
        name = macos_processor_name()
    elif sys.platform == "linux":
        name = linux_processor_name(Path("/proc/cpuinfo"))
    # platform.machine() is the family only, for example "arm64" or "x86_64".
    return name or platform.machine() or "unknown processor"


def macos_processor_name() -> str:
    """Ask macOS for the processor's name. Returns "" when it does not say."""
    try:
        done = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if done.returncode != 0:
        return ""
    return done.stdout.strip()


def linux_processor_name(cpuinfo: Path) -> str:
    """Read the processor's name from Linux's own description. Returns "" when there is none."""
    try:
        text = cpuinfo.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.split("\n"):
        # The line looks like "model name : Some Processor 2.40GHz".
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return ""


def memory_gb() -> int:
    """The machine's memory in gigabytes, rounded to a whole number."""
    # The system counts memory in pages: the number of pages times the size of one.
    total_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return round(total_bytes / GIGABYTE)


def operating_system() -> str:
    """For example "macOS 15.1", or just "Linux"."""
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]}"
    # On Linux the name alone. The kernel release (platform.release()) is left
    # out: the report is public, and it says more about the machine than a
    # reader needs. platform.node() and platform.uname() are not used
    # anywhere either: they hold the host name.
    return platform.system()


def container_limits(folder: Path = CONTAINER_LIMITS_FOLDER) -> str | None:
    """The processor and memory limits of this container, or None when there are none.

    Docker tells Linux the limits, and Linux shows them to the container in
    two small files. Outside a container the files are not there.
    """
    parts: list[str] = []

    # "200000 100000" means 200000 of every 100000 microseconds: two
    # processors. "max 100000" means no limit.
    cpu = first_line(folder / "cpu.max").split()
    if len(cpu) == 2 and cpu[0] != "max":
        cpus = int(cpu[0]) / int(cpu[1])
        parts.append("1 CPU" if cpus == 1 else f"{cpus:g} CPUs")

    # A number of bytes, or "max" for no limit.
    memory = first_line(folder / "memory.max")
    if memory.isdigit():
        parts.append(f"{round(int(memory) / GIGABYTE, 1):g} GB memory")

    if not parts:
        return None
    return " and ".join(parts)


def first_line(path: Path) -> str:
    """The first line of a small text file, or "" when the file cannot be read."""
    try:
        return path.read_text(encoding="utf-8").split("\n")[0].strip()
    except OSError:
        return ""


# --- small pieces of text ---------------------------------------------------


def rate(count: int, total: int) -> str:
    """A rate with its count, for example "14 of 20 (70%)"."""
    if total == 0:
        return "0 of 0"
    # Adding 0.5 before cutting off the decimals rounds a half up: 12.5 becomes 13.
    percent = int(100 * count / total + 0.5)
    return f"{count} of {total} ({percent}%)"


def number_or_dash(value: float | None, layout: str) -> str:
    """A number in the given layout, or "-" when there is no number to show.

    A measure over no questions is None: when a method refused every question
    of a group, there is no "guides returned" and no "text handed over".
    """
    if value is None:
        return "-"
    return format(value, layout)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table, as lines."""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def slug_list(slugs: list[str] | tuple[str, ...]) -> str:
    """Slugs as they are shown in the report: `one`, `two`."""
    if not slugs:
        return "no guide"
    return ", ".join(f"`{slug}`" for slug in slugs)


def methods_in(context: dict) -> list[str]:
    """The methods this run has outcomes for, in table order."""
    return [method for method in METHODS if method in context["outcomes"]]


def has_rerank(context: dict) -> bool:
    return "rerank" in context["outcomes"]


def splits_in(context: dict) -> list[str]:
    """The splits that get a section: test and dev always, fresh once a fresh question is reviewed."""
    return [split for split in SPLITS if split != "fresh" or scored(context, "fresh")]


def name_list(names: list[str]) -> str:
    """"A", "A and b", "A, b and c": the first name as it is, the others in lower case."""
    words = [names[0]] + [name.lower() for name in names[1:]]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def scored(context: dict, split: str, group: str | None = None) -> list[Question]:
    """The reviewed questions of one split, or of one group within it."""
    questions = []
    for question in context["questions"]:
        if not question.reviewed or question.split != split:
            continue
        if group is None or question.group == group:
            questions.append(question)
    return questions


# --- what was run -----------------------------------------------------------


def what_was_run_lines(context: dict) -> list[str]:
    questions = context["questions"]
    reviewed = [question for question in questions if question.reviewed]
    dev = len([question for question in reviewed if question.split == "dev"])
    test = len([question for question in reviewed if question.split == "test"])
    fresh = len([question for question in reviewed if question.split == "fresh"])
    unreviewed = len(questions) - len(reviewed)

    models = ", ".join(f"`{model}`" for model in context["baseline_models"])
    dates = ", ".join(context["baseline_dates"])

    revisions = context["prompt_revisions"]
    if revisions == 0:
        prompt = "its prompt was not revised after the first run"
    elif revisions == 1:
        prompt = "its prompt was revised 1 time after the first run, looking at `dev` results only"
    else:
        prompt = f"its prompt was revised {revisions} times after the first run, looking at `dev` results only"

    if context["test_scored_before"]:
        test_note = (
            "scored again. A results file from an earlier run was already there, "
            "so this is not the first time they were scored."
        )
    else:
        # Not "scored for the first time": the code cannot know that. It only
        # saw that no results file was there.
        test_note = "scored in this run. No results file from an earlier run was found in the output folder."

    if fresh:
        files = "the question files"
        counts = f"{dev} dev, {test} test, {fresh} fresh"
    else:
        files = "the question file"
        counts = f"{dev} dev, {test} test"

    lines = [
        "## What was run",
        "",
        f"- Guides: the snapshot fetched on {context['snapshot_date']}, "
        f"{context['guide_count']} guides, {context['passage_count']} passages in the index",
        f"- Questions: {files} at commit `{context['questions_commit']}`; "
        f"{len(reviewed)} reviewed and scored ({counts}); "
        f"{unreviewed} not reviewed and not scored",
        f"- Embedding model: `{context['model_id']}`",
    ]
    if has_rerank(context):
        lines.append(f"- Reranker: `{context['reranker_id']}`")
    lines += [
        f"- Title-picking model: {models}; each question was asked once; "
        f"answers recorded on {dates}; {prompt}",
        f"- Threshold: {context['threshold']:.4f}. Below this best similarity, "
        "vector and hybrid search return no match.",
    ]
    if has_rerank(context):
        lines += [
            f"- Rerank threshold: {context['rerank_threshold']:.4f}. Below this best reranker score, "
            "rerank search returns no match.",
            f"- Default mode: {context['default_mode']}",
        ]
    lines.append(f"- Test questions: {test_note}")
    if fresh:
        if context["fresh_scored_before"]:
            fresh_note = "scored again. The results file from an earlier run already held them."
        else:
            fresh_note = "scored in this run. No results file from an earlier run held them."
        lines.append(f"- Fresh questions: {fresh_note}")
    lines += [
        f"- Run date: {context['run_date']}",
        f"- Machine: {context['machine']}",
        "",
    ]

    arabic_reviewed = [question for question in reviewed if question.language == "ar"]
    if not arabic_reviewed:
        lines += [
            "**This report covers English only.** No Arabic question has been reviewed yet, "
            "so none is scored. When a reader of Arabic has reviewed them, the threshold is "
            "chosen again on `dev` and this report is rebuilt.",
            "",
        ]
    return lines


# --- the headline -----------------------------------------------------------


def wins_and_losses(
    questions: list[Question],
    ours: dict[str, Outcome],
    theirs: dict[str, Outcome],
    k: int,
) -> tuple[int, int]:
    """How many questions only we got right, and how many only they did."""
    wins = 0
    losses = 0
    for question in questions:
        our_hit = hit_at(question, ours[question.id], k)
        their_hit = hit_at(question, theirs[question.id], k)
        if our_hit and not their_hit:
            wins += 1
        if their_hit and not our_hit:
            losses += 1
    return wins, losses


def verdict(wins: int, losses: int, mode: str = "hybrid") -> str:
    """The sentence for one measure: better, worse, or no difference that this set can show."""
    if sign_test(wins, losses) >= SIGNIFICANCE:
        return f"**no difference that this set can show** between {mode} search and title-picking."
    if wins > losses:
        return f"{mode} search is **better** than title-picking."
    return f"{mode} search is **worse** than title-picking."


def comparison_lines(context: dict, mode: str, split: str) -> list[str]:
    """One search mode against title-picking, on the English questions of one split: a table and two sentences."""
    questions = scored(context, split, "en")
    ours = context["outcomes"][mode]
    title_picking = context["outcomes"]["title_picking"]
    name = METHOD_NAMES[mode]
    rows = []
    sentences = []
    for k in (1, 3):
        wins, losses = wins_and_losses(questions, ours, title_picking, k)
        our_hits = len([q for q in questions if hit_at(q, ours[q.id], k)])
        title_picking_hits = len([q for q in questions if hit_at(q, title_picking[q.id], k)])
        rows.append(
            [
                f"hit@{k}",
                rate(our_hits, len(questions)),
                rate(title_picking_hits, len(questions)),
                str(wins),
                str(losses),
                f"{sign_test(wins, losses):.3f}",
            ]
        )
        sentences.append(f"- hit@{k}: {verdict(wins, losses, mode)}")

    header = [
        "Measure",
        name,
        "Title-picking",
        f"{name} right, title-picking wrong",
        f"Title-picking right, {mode} wrong",
        "p (two-sided sign test)",
    ]
    return table(header, rows) + [""] + sentences


def headline_lines(context: dict) -> list[str]:
    lines = ["## Headline: hybrid search against title-picking", ""]
    if not scored(context, "test", "en"):
        return lines + ["No reviewed English test question has been scored, so there is no headline.", ""]
    return lines + comparison_lines(context, "hybrid", "test") + ["", HEADLINE_NOTE, ""]


def rerank_comparison_lines(context: dict) -> list[str]:
    """Rerank against title-picking, on test and on the fresh questions."""
    lines = [
        "## Rerank search against title-picking",
        "",
        "The comparison of the headline, for rerank search, on `test` and on the fresh questions. "
        "Unlike the headline, these comparisons were not named before the first run, so read them "
        "as descriptive.",
        "",
    ]
    for split in ("test", "fresh"):
        if scored(context, split, "en"):
            lines += [f"### On {split}", ""] + comparison_lines(context, "rerank", split) + [""]
    return lines


def bar_lines(context: dict) -> list[str]:
    """Whether rerank search met the bar on test and on the fresh questions."""
    lines = ["## Is search good enough to rely on?", "", BAR_NOTE, ""]
    rows = []
    sentences = []
    for split in ("test", "fresh"):
        questions = scored(context, split, "en")
        if not questions:
            continue
        for mode in ("hybrid", "rerank"):
            summary = summarise_answerable(questions, context["outcomes"][mode])
            # 9 * total / 10 is exact whenever it is a whole number, so ceil cannot round 18 up to 19.
            needed = math.ceil(9 * summary.total / 10)
            met = summary.passage_hit3 >= needed
            rows.append(
                [
                    split,
                    METHOD_NAMES[mode],
                    rate(summary.passage_hit3, summary.total),
                    f"{needed} of {summary.total}",
                    "yes" if met else "no",
                ]
            )
            if mode == "rerank":
                if met:
                    sentences.append(f"- On {split}, rerank search **meets** the bar.")
                else:
                    short = needed - summary.passage_hit3
                    sentences.append(
                        f"- On {split}, rerank search **falls short** of the bar by {short} "
                        f"question{'s' if short != 1 else ''}."
                    )
    if not rows:
        return lines + ["No reviewed English test or fresh question has been scored.", ""]
    header = ["Questions", "Method", "A returned passage holds the answer", "Bar", "Met"]
    return lines + table(header, rows) + [""] + sentences + [""]


# --- the tables of one split ------------------------------------------------


def measure_lines() -> list[str]:
    return [
        "## How to read the tables",
        "",
        *MEASURES,
        "",
        "Every rate is shown with its count. `dev` and `test` are never added together, "
        "because the threshold was fitted on `dev`.",
        "",
    ]


def answerable_lines(context: dict, split: str, language: str) -> list[str]:
    lines = [f"### {LANGUAGE_NAMES[language]} questions the guides can answer", ""]
    questions = scored(context, split, language)
    if not questions:
        return lines + ["No reviewed question in this group.", ""]

    rows = []
    for method in methods_in(context):
        summary = summarise_answerable(questions, context["outcomes"][method])
        rows.append(
            [
                METHOD_NAMES[method],
                rate(summary.hit1, summary.total),
                rate(summary.hit3, summary.total),
                rate(summary.passage_hit3, summary.total),
                f"{summary.mrr:.3f}",
                rate(summary.wrong_refusals, summary.total),
                number_or_dash(summary.guides_returned, ".1f"),
                number_or_dash(summary.chars_median, ",.0f"),
            ]
        )
    header = [
        "Method",
        "hit@1",
        "hit@3",
        "Passage hit@3",
        "MRR",
        "Wrong refusals",
        "Guides returned",
        "Text handed over (characters)",
    ]
    return lines + table(header, rows) + [""]


def unanswerable_lines(context: dict, split: str) -> list[str]:
    lines = ["### Questions the guides cannot answer", ""]
    questions = scored(context, split, "unanswerable")
    if not questions:
        return lines + ["No reviewed question in this group.", ""]

    rows = []
    for method in methods_in(context):
        summary = summarise_unanswerable(questions, context["outcomes"][method])
        answered = summary.total - summary.correct_refusals
        rows.append(
            [
                METHOD_NAMES[method],
                rate(summary.correct_refusals, summary.total),
                rate(answered, summary.total),
            ]
        )
    header = ["Method", "Correct refusals", "Answered when it should have refused"]
    return lines + table(header, rows) + [""]


def ranking_only_lines(context: dict, split: str) -> list[str]:
    modes = [mode for mode in RANKING_ONLY_MODES if mode in context["ranking_only"]]
    lines = [
        "### Ranking only: the threshold switched off",
        "",
        f"{name_list([METHOD_NAMES[mode] for mode in modes])} search, run again with the threshold at 0, "
        "so that nothing is refused. When a result above is weak, this shows whether the ranking or "
        "the refusing is the cause.",
        "",
    ]
    rows = []
    for mode in modes:
        for language in LANGUAGE_NAMES:
            questions = scored(context, split, language)
            if not questions:
                continue
            summary = summarise_answerable(questions, context["ranking_only"][mode])
            rows.append(
                [
                    METHOD_NAMES[mode],
                    LANGUAGE_NAMES[language],
                    rate(summary.hit1, summary.total),
                    rate(summary.hit3, summary.total),
                    f"{summary.mrr:.3f}",
                ]
            )
    if not rows:
        return lines + ["No reviewed question in this split.", ""]
    return lines + table(["Method", "Questions", "hit@1", "hit@3", "MRR"], rows) + [""]


def split_lines(context: dict, split: str) -> list[str]:
    """Every table of one split. The other split is not looked at."""
    notes = RERANK_SPLIT_NOTES if has_rerank(context) else SPLIT_NOTES
    lines = [f"## Results on {split}", "", notes[split], ""]
    for language in LANGUAGE_NAMES:
        lines += answerable_lines(context, split, language)
    lines += unanswerable_lines(context, split)
    lines += ranking_only_lines(context, split)
    return lines


# --- time -------------------------------------------------------------------


def milliseconds(seconds: float) -> str:
    """A time as whole milliseconds, for example "35 ms" or "1,840 ms"."""
    return f"{seconds * 1000:,.0f} ms"


def time_lines(context: dict) -> list[str]:
    rows = []
    for method in methods_in(context):
        seconds = [outcome.seconds for outcome in context["outcomes"][method].values()]
        if seconds:
            rows.append(
                [
                    METHOD_NAMES[method],
                    str(len(seconds)),
                    milliseconds(statistics.median(seconds)),
                    milliseconds(max(seconds)),
                ]
            )
    header = ["Method", "Questions", "Median", "Maximum"]
    return ["## Time", ""] + table(header, rows) + ["", TIME_NOTE, ""]


# --- the threshold ----------------------------------------------------------


def gap_line(side: str, gap: float | None) -> str:
    if gap is None:
        return f"- There is no dev score {side}."
    return f"- Nearest dev score {side}: {gap:.4f} away"


def threshold_lines(context: dict) -> list[str]:
    """The threshold section. It needs only `questions`, `dev_similarities` and `threshold`."""
    return threshold_section(
        context,
        heading="## The threshold",
        note=THRESHOLD_NOTE,
        zero_note=ZERO_NOTE,
        scores_key="dev_similarities",
        used=context["threshold"],
        setting="Threshold",
        score_name="best similarity",
    )


def rerank_threshold_lines(context: dict) -> list[str]:
    """The rerank threshold section. It needs only `questions`, `dev_rerank_scores` and `rerank_threshold`."""
    return threshold_section(
        context,
        heading="## The rerank threshold",
        note=RERANK_THRESHOLD_NOTE,
        zero_note=RERANK_ZERO_NOTE,
        scores_key="dev_rerank_scores",
        used=context["rerank_threshold"],
        setting="Rerank threshold",
        score_name="reranker's best score",
    )


def threshold_section(
    context: dict,
    heading: str,
    note: str,
    zero_note: str,
    scores_key: str,
    used: float,
    setting: str,
    score_name: str,
) -> list[str]:
    """One threshold: the rule's candidates on dev, its choice, the gaps, and the scores by group."""
    lines = [heading, "", note, ""]
    dev = scored(context, "dev")
    if not dev:
        return lines + ["No reviewed dev question has been scored, so there is nothing to choose from.", ""]

    scores = context[scores_key]
    choice = choose_threshold([(scores[question.id], question.answerable) for question in dev])
    same = abs(used - choice.threshold) < THRESHOLD_TOLERANCE

    lines.append(f"{setting} in `settings.toml`, used in this run: {used:.4f}.")
    # `final` in run.py stops when the two differ, so a written report always
    # has this sentence. It is still checked here, so that this function
    # never says something it has not looked at.
    if same:
        lines.append("The rule, applied to this run's dev scores, gives the same value.")
    lines.append("")

    rows = []
    for candidate, correct in choice.table:
        mark = "chosen" if candidate == choice.threshold else ""
        rows.append([f"{candidate:.4f}", f"{correct} of {choice.total}", mark])
    lines += table(["Candidate", "Dev questions handled correctly", ""], rows)
    lines.append("")

    lines += [
        f"The rule's choice is {choice.threshold:.4f}.",
        "",
        gap_line("below it", choice.gap_below),
        gap_line("at or above it", choice.gap_above),
        "",
        f"The {score_name} of the dev questions, by group:",
        "",
    ]

    rows = []
    groups = [("en", "English, answerable"), ("ar", "Arabic, answerable"), ("unanswerable", "Unanswerable")]
    for group, name in groups:
        group_scores = [scores[question.id] for question in dev if question.group == group]
        if group_scores:
            rows.append(
                [
                    name,
                    str(len(group_scores)),
                    f"{min(group_scores):.4f}",
                    f"{statistics.median(group_scores):.4f}",
                    f"{max(group_scores):.4f}",
                ]
            )
        else:
            rows.append([name, "0", "-", "-", "-"])
    lines += table(["Dev questions", "How many", "Lowest", "Median", "Highest"], rows)
    lines.append("")

    if same and choice.threshold == 0.0:
        lines += [zero_note, ""]
    return lines


# --- the default mode -------------------------------------------------------


def default_mode_lines(context: dict) -> list[str]:
    """How the default mode was chosen. It needs `questions`, `outcomes` (hybrid and rerank) and `default_mode`."""
    lines = ["## The default mode", "", DEFAULT_MODE_NOTE, ""]
    dev = scored(context, "dev")
    if not dev:
        return lines + ["No reviewed dev question has been scored, so there is nothing to choose from.", ""]

    rows = []
    for mode in ("hybrid", "rerank"):
        outcomes = context["outcomes"][mode]
        answerable = summarise_answerable(dev, outcomes)
        unanswerable = summarise_unanswerable(dev, outcomes)
        rows.append(
            [
                METHOD_NAMES[mode],
                rate(answerable.passage_hit3, answerable.total),
                rate(unanswerable.correct_refusals, unanswerable.total),
            ]
        )
    lines += table(["Dev questions", "A returned passage holds the answer", "Correct refusals"], rows)
    choice = choose_default_mode(dev, context["outcomes"]["hybrid"], context["outcomes"]["rerank"])
    lines += ["", f"The rule's choice is {choice}. `settings.toml` holds {context['default_mode']}.", ""]
    return lines


# --- the reranker -----------------------------------------------------------


def choice_row(name: str, entry: dict) -> list[str]:
    """One row of the reranker choice: a candidate, or hybrid search for comparison."""
    if not entry["loaded"]:
        return [name, f"not run: {entry['note']}", "", "", "", "", ""]
    return [
        name,
        rate(entry["hit1"], entry["total"]),
        rate(entry["hit3"], entry["total"]),
        rate(entry["passage_hit3"], entry["total"]),
        milliseconds(entry["median_seconds"]),
        milliseconds(entry["max_seconds"]),
        entry["note"] or "",
    ]


def reranker_choice_lines(context: dict) -> list[str]:
    """How the reranker was chosen, from the record that reranker_choice.py wrote."""
    choice = context.get("reranker_choice")
    if not choice:
        return []
    lines = [
        "## The reranker",
        "",
        CHOICE_NOTE.format(limit=choice["limit_seconds"]),
        "",
        f"Run on {choice['date']}, on {choice['machine']}, with torch limited to {choice['threads']} "
        f"threads, over the {choice['answerable']} answerable reviewed `dev` questions. A time is the "
        "whole search, the round trips to the database included. The times of the reported run are "
        "in the Time section.",
        "",
    ]
    rows = [choice_row(f"`{entry['name']}`", entry) for entry in choice["candidates"]]
    rows.append(choice_row("Hybrid search, no reranker", choice["hybrid"]))
    header = ["Candidate", "hit@1", "hit@3", "Passage hit@3", "Median time", "Maximum time", ""]
    lines += table(header, rows) + [""]
    if choice["chosen"]:
        lines.append(f"Chosen: `{choice['chosen']}`.")
    else:
        lines.append("No candidate met the rule.")
    return lines + [""]


# --- tokens -----------------------------------------------------------------


def token_lines(context: dict) -> list[str]:
    """The guide text a model must read per question, in tokens, from the record tokens.py wrote."""
    tokens = context.get("tokens")
    if not tokens:
        return []
    lines = [
        "## Tokens a model must read",
        "",
        f"Counted for `{tokens['model']}` on {tokens['date']}.",
        "",
    ]
    rows = []
    for method in methods_in(context):
        counts = [entry["tokens"] for entry in tokens["entries"] if entry["method"] == method and not entry["refused"]]
        if counts:
            rows.append(
                [
                    METHOD_NAMES[method],
                    str(len(counts)),
                    f"{statistics.median(counts):,.0f}",
                    f"{max(counts):,}",
                ]
            )
    lines += table(["Method", "Questions", "Median tokens", "Maximum tokens"], rows)
    lines += ["", TOKENS_NOTE, ""]
    left_out = tokens.get("left_out", 0)
    if left_out:
        lines += [
            f"{left_out} count{'s were' if left_out != 1 else ' was'} left out: made from a search result "
            "other than this run's, as can happen when a near-tie turns the other way on another machine.",
            "",
        ]
    return lines


# --- misses -----------------------------------------------------------------


def is_miss(question: Question, outcome: Outcome) -> bool:
    if question.answerable:
        return not hit_at(question, outcome, 3)
    # A question the guides cannot answer is missed when it was answered.
    return not outcome.refused


def miss_lines(context: dict, split: str) -> list[str]:
    """Every miss of one split, method by method."""
    lines = [f"### Misses on {split}", ""]
    questions = scored(context, split)
    for method in methods_in(context):
        lines += [f"#### {METHOD_NAMES[method]}", ""]
        outcomes = context["outcomes"][method]
        missed = [question for question in questions if is_miss(question, outcomes[question.id])]
        if not missed:
            lines += ["None.", ""]
            continue
        for question in missed:
            outcome = outcomes[question.id]
            expected = slug_list(question.expected_slugs)
            if not question.answerable:
                expected += " (the guides do not cover this)"
            returned = "nothing" if outcome.refused else slug_list(outcome.slugs)
            lines += [
                f"- `{question.id}` {question.question}",
                f"  - expected: {expected}",
                f"  - returned: {returned}",
            ]
        lines.append("")
    return lines


# --- label corrections and limits -------------------------------------------


def correction_lines(context: dict) -> list[str]:
    lines = ["## Label corrections", ""]
    corrections = context["corrections"]
    if not corrections:
        # What was looked at is the log, so the sentence is about the log.
        return lines + ["None. No correction is logged in evaluation/corrections.jsonl.", ""]
    for correction in corrections:
        lines.append(
            f"- `{correction['question_id']}` ({correction['date']}): expected guides changed "
            f"from {slug_list(correction['old_expected_slugs'])} "
            f"to {slug_list(correction['new_expected_slugs'])}. Reason: {correction['reason']}"
        )
    return lines + ["", "Every method was scored again with the corrected labels.", ""]


def limit_lines(context: dict) -> list[str]:
    """The limits. Fixed text, except for one count and one note that depend on what was scored."""
    reviewed = [question for question in context["questions"] if question.reviewed]
    unanswerable = [question for question in reviewed if not question.answerable]

    lines = ["## Limits", ""]
    for limit in LIMITS:
        # Only the second limit has the two places to fill in. format() gives
        # the other eight back unchanged.
        lines.append(limit.format(unanswerable=len(unanswerable), scored=len(reviewed)))
    if has_rerank(context):
        lines += RERANK_LIMITS

    arabic_reviewed = [question for question in reviewed if question.language == "ar"]
    if not arabic_reviewed:
        # An empty line first: without it markdown would read the note as part of the last limit.
        lines += ["", ARABIC_LIMITS_NOTE]
    return lines


# --- the whole report -------------------------------------------------------


def render_report(context: dict) -> str:
    """The whole of evaluation/report.md, as one text."""
    intro = (
        "This report answers one question: is search better than letting a model pick guides "
        "from a list of their titles?"
    )
    if has_rerank(context):
        intro += (
            " Since reranking was added, it answers a second: is search good enough that an "
            "assistant can rely on the passages it returns?"
        )
    lines = [
        "# Evaluation report",
        "",
        intro + " The numbers are reported as measured, whichever way they come out.",
        "",
    ]
    lines += what_was_run_lines(context)
    lines += headline_lines(context)
    if has_rerank(context):
        lines += rerank_comparison_lines(context)
        lines += bar_lines(context)
    lines += measure_lines()
    for split in splits_in(context):
        lines += split_lines(context, split)
    lines += time_lines(context)
    lines += token_lines(context)
    lines += threshold_lines(context)
    if has_rerank(context):
        lines += rerank_threshold_lines(context)
        lines += default_mode_lines(context)
    lines += reranker_choice_lines(context)
    lines += ["## Misses", "", MISSES_NOTE, ""]
    for split in splits_in(context):
        lines += miss_lines(context, split)
    lines += correction_lines(context)
    lines += limit_lines(context)
    return "\n".join(lines) + "\n"
