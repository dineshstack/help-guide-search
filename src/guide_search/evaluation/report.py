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
    questions           every question in the question file, reviewed or not
    outcomes            method -> question id -> Outcome, with the threshold
    ranking_only        "vector" and "hybrid" -> question id -> Outcome, with the threshold at 0
    dev_similarities    question id -> best similarity, for the reviewed dev questions
    corrections         the logged label corrections, each a dict

Only reviewed questions are counted, and `dev` and `test` are never added
together: every table is for one split.

The smaller functions each write one section and return it as a list of
lines. run.py also uses three of them on their own, to print the threshold
table and the dev tables.
"""

import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

from guide_search.evaluation.metrics import (
    Outcome,
    choose_threshold,
    hit_at,
    sign_test,
    summarise_answerable,
    summarise_unanswerable,
)
from guide_search.evaluation.questions import Question

# The four methods, in the order of every table.
METHODS = ("title_picking", "vector", "keyword", "hybrid")
METHOD_NAMES = {
    "title_picking": "Title-picking",
    "vector": "Vector",
    "keyword": "Keyword",
    "hybrid": "Hybrid",
}

# The two search modes that have a threshold. They are run a second time with
# the threshold switched off.
RANKING_ONLY_MODES = ("vector", "hybrid")

# `test` comes first in the report, because its numbers are the ones reported.
SPLITS = ("test", "dev")

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
    "around the search function, including embedding the question, after one discarded warm-up "
    "query. Title-picking time is the model's response time over the network, recorded when the "
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

    lines = [
        "## What was run",
        "",
        f"- Guides: the snapshot fetched on {context['snapshot_date']}, "
        f"{context['guide_count']} guides, {context['passage_count']} passages in the index",
        f"- Questions: the question file at commit `{context['questions_commit']}`; "
        f"{len(reviewed)} reviewed and scored ({dev} dev, {test} test); "
        f"{unreviewed} not reviewed and not scored",
        f"- Embedding model: `{context['model_id']}`",
        f"- Title-picking model: {models}; each question was asked once; "
        f"answers recorded on {dates}; {prompt}",
        f"- Threshold: {context['threshold']:.4f}. Below this best similarity, "
        "vector and hybrid search return no match.",
        f"- Test questions: {test_note}",
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


def verdict(wins: int, losses: int) -> str:
    """The sentence for one measure: better, worse, or no difference that this set can show."""
    if sign_test(wins, losses) >= SIGNIFICANCE:
        return "**no difference that this set can show** between hybrid search and title-picking."
    if wins > losses:
        return "hybrid search is **better** than title-picking."
    return "hybrid search is **worse** than title-picking."


def headline_lines(context: dict) -> list[str]:
    lines = ["## Headline: hybrid search against title-picking", ""]
    questions = scored(context, "test", "en")
    if not questions:
        return lines + ["No reviewed English test question has been scored, so there is no headline.", ""]

    hybrid = context["outcomes"]["hybrid"]
    title_picking = context["outcomes"]["title_picking"]
    rows = []
    sentences = []
    for k in (1, 3):
        wins, losses = wins_and_losses(questions, hybrid, title_picking, k)
        hybrid_hits = len([q for q in questions if hit_at(q, hybrid[q.id], k)])
        title_picking_hits = len([q for q in questions if hit_at(q, title_picking[q.id], k)])
        rows.append(
            [
                f"hit@{k}",
                rate(hybrid_hits, len(questions)),
                rate(title_picking_hits, len(questions)),
                str(wins),
                str(losses),
                f"{sign_test(wins, losses):.3f}",
            ]
        )
        sentences.append(f"- hit@{k}: {verdict(wins, losses)}")

    header = [
        "Measure",
        "Hybrid",
        "Title-picking",
        "Hybrid right, title-picking wrong",
        "Title-picking right, hybrid wrong",
        "p (two-sided sign test)",
    ]
    return lines + table(header, rows) + [""] + sentences + ["", HEADLINE_NOTE, ""]


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
    for method in METHODS:
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
    for method in METHODS:
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
    lines = [
        "### Ranking only: the threshold switched off",
        "",
        "Vector and hybrid search, run again with the threshold at 0, so that nothing is refused. "
        "When a result above is weak, this shows whether the ranking or the refusing is the cause.",
        "",
    ]
    rows = []
    for mode in RANKING_ONLY_MODES:
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
    lines = [f"## Results on {split}", "", SPLIT_NOTES[split], ""]
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
    for method in METHODS:
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
    lines = ["## The threshold", "", THRESHOLD_NOTE, ""]
    dev = scored(context, "dev")
    if not dev:
        return lines + ["No reviewed dev question has been scored, so there is nothing to choose from.", ""]

    similarities = context["dev_similarities"]
    choice = choose_threshold([(similarities[question.id], question.answerable) for question in dev])
    used = context["threshold"]
    same = abs(used - choice.threshold) < THRESHOLD_TOLERANCE

    lines.append(f"Threshold in `settings.toml`, used in this run: {used:.4f}.")
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
        "The best similarity of the dev questions, by group:",
        "",
    ]

    rows = []
    groups = [("en", "English, answerable"), ("ar", "Arabic, answerable"), ("unanswerable", "Unanswerable")]
    for group, name in groups:
        scores = [similarities[question.id] for question in dev if question.group == group]
        if scores:
            rows.append(
                [
                    name,
                    str(len(scores)),
                    f"{min(scores):.4f}",
                    f"{statistics.median(scores):.4f}",
                    f"{max(scores):.4f}",
                ]
            )
        else:
            rows.append([name, "0", "-", "-", "-"])
    lines += table(["Dev questions", "How many", "Lowest", "Median", "Highest"], rows)
    lines.append("")

    if same and choice.threshold == 0.0:
        lines += [ZERO_NOTE, ""]
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
    for method in METHODS:
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

    arabic_reviewed = [question for question in reviewed if question.language == "ar"]
    if not arabic_reviewed:
        # An empty line first: without it markdown would read the note as part of the last limit.
        lines += ["", ARABIC_LIMITS_NOTE]
    return lines


# --- the whole report -------------------------------------------------------


def render_report(context: dict) -> str:
    """The whole of evaluation/report.md, as one text."""
    lines = [
        "# Evaluation report",
        "",
        "This report answers one question: is search better than letting a model pick guides "
        "from a list of their titles? The numbers are reported as measured, whichever way they come out.",
        "",
    ]
    lines += what_was_run_lines(context)
    lines += headline_lines(context)
    lines += measure_lines()
    for split in SPLITS:
        lines += split_lines(context, split)
    lines += time_lines(context)
    lines += threshold_lines(context)
    lines += ["## Misses", "", MISSES_NOTE, ""]
    for split in SPLITS:
        lines += miss_lines(context, split)
    lines += correction_lines(context)
    lines += limit_lines(context)
    return "\n".join(lines) + "\n"
