# help-guide-search

A small Python service that answers questions from the 60 public help guides of SAM, a workshop management product. It finds the passages that answer a question, offers that search to AI assistants as tools, and has an answering agent whose replies are checked by code before they go out.

It also measures itself, and reports what it measured whether or not the result is flattering.

It reads the public guides and nothing else. It is not connected to any live system, holds no customer data, and serves no users.

```
                       ┌──────────────── one Service ────────────────┐
HTTP API  ─────────────┤                                             │
MCP tools (assistants) ┤  search: vector + keyword, fused, grouped   ├──> Postgres + pgvector
Answer agent ──────────┤  agent: search → Claude draft → code check  │     335 passages, read-only
                       └─────────────────────────────────────────────┘
```

## What was measured

Two evaluations, on 50 reviewed questions written the way a workshop's staff would ask them: 40 the guides can answer and 10 they cannot. Half were used to make choices (`dev`), half were held back (`test`). The numbers below are from `test`.

### Search against letting a model pick from titles

The simplest thing an assistant can do is show a model the 60 guide titles and let it pick. Search has to beat that to be worth having. At this size, it did not.

| Method | Right guide first | Right guide in top 3 | Text handed on | Time per question |
| --- | --- | --- | --- | --- |
| Title-picking (Claude Sonnet 4.6) | 17 of 20 | 20 of 20 | 6,864 characters | 1,582 ms |
| Hybrid search | 10 of 20 | 14 of 20 | 2,632 characters | 35 ms |
| Vector search | 8 of 20 | 13 of 20 | 2,195 characters | 34 ms |
| Keyword search | 4 of 20 | 9 of 20 | 2,784 characters | 3 ms |

- **Title-picking found the right guide more often.** On top-3 the difference is larger than chance explains (6 questions only title-picking got right, none the other way). On first place, 17 against 10 is not a difference a set this small can prove.
- **Hybrid is the best of the three search modes.**
- **Search hands on about 40% of the text and needs no model call.** The two times are different kinds of time (local computation against a network call) and are not compared.
- **Nothing refused.** All four methods returned guides for all 5 questions the guides cannot answer. No similarity threshold separated them from real questions.

Full report: [evaluation/report.md](evaluation/report.md).

### The answer agent

The agent reads the passages search found, and may decline. Four Claude models, same questions:

| Model | Answered from an expected guide | Declined an answerable question | Declined the 5 unanswerable ones | Cost per 100 questions |
| --- | --- | --- | --- | --- |
| Fable 5.1 | 13 of 20 | 7 of 20 | 5 of 5 | $2.78 |
| Opus 5.5 | 13 of 20 | 7 of 20 | 5 of 5 | $1.12 |
| Sonnet 5.5 | 13 of 20 | 7 of 20 | 5 of 5 | $0.50 |
| Haiku 5.5 | 14 of 20 | 6 of 20 | 5 of 5 | $0.03 |

- **The agent declines what the guides do not cover.** Search could not; every model here declined all 5 on `test` (3 or 4 of 5 on `dev`).
- **It never answered from a wrong guide on `test`.** When search missed the right guide, it declined instead of guessing. That is why about a third of answerable questions were handed over: the agent is limited by what search finds.
- **No visible gain from a larger model on this set.** With 25 questions, a difference of one is noise.

Full report: [evaluation/agent_report.md](evaluation/agent_report.md).

### Limits

- The set is small. One answerable question is 5 points; one unanswerable question is 20.
- "Cites an expected guide" is not "the answer is correct". Nobody judged the text of the answers.
- The questions were drafted by a model that had read the guides, then reviewed by a person.
- Only English is scored. The guides are English, and the Arabic questions wait for a reviewer who reads Arabic.
- 60 guides fit in one prompt. Title-picking cannot scale past what fits; that case was not measured.

## Run it

It needs [uv](https://docs.astral.sh/uv/) for Python, and Postgres 16 with pgvector.

### With Docker Compose

```bash
cp .env.example .env        # then set POSTGRES_PASSWORD and READER_PASSWORD to two random values
docker compose up -d --wait db
docker compose --profile tools run --rm tools python -m guide_search.ingest
docker compose up -d --wait api
curl 'http://127.0.0.1:8765/search?q=how+do+I+reset+my+password'
```

The API listens on `127.0.0.1` only. It connects to the database as an account that can only read, and runs as an ordinary user. The embedding model is inside the image, so nothing is downloaded at start.

### Without Docker

```bash
uv python install 3.13
uv sync --group dev --group localdb
uv run python scripts/local_postgres.py start      # prints two lines to put in .env
uv run --env-file .env python -m guide_search.ingest
uv run --env-file .env uvicorn guide_search.service:build --factory --port 8000
```

`scripts/local_postgres.py` starts a Postgres 16 that comes inside a Python package. It is a convenience for development; the reported results come from the pinned Docker image.

## Use it

### HTTP

| Route | Returns |
| --- | --- |
| `GET /search?q=...&mode=hybrid` | Up to 3 guides with up to 2 passages each, or an empty list with reason `no_match` |
| `GET /guides/{slug}` | One guide's passages in order |
| `GET /health` | `200` when the index is built and the model is loaded |
| `POST /answer` with `{"question": "..."}` | An answer with its sources, a question back, or a hand-over |

A database fault is always `503`. It is never an empty result and never a hand-over, because "we could not look" must not look like "we looked and found nothing".

### AI assistants (MCP)

Three read-only tools through the Model Context Protocol: `search_guides`, `read_guide`, `list_guides`. The assistant starts the server itself, as a child process:

```bash
uv run --directory /path/to/help-guide-search --env-file .env python -m guide_search.mcp_server
```

### The answer agent

```bash
uv run --env-file .env python -m guide_search.agent "how do I reset my password"
```

It needs `ANTHROPIC_API_KEY` in `.env`. A Claude model writes a draft; plain code then checks it: an answer must cite guides that search actually returned, and every number and link in it must appear in the passages of those guides. A draft that fails gets one rewrite, told why. A second failure, a refusal or an API error hands over to a person. The model and its settings are in `agent.toml`, and token use is recorded for every call.

## Tests

```bash
uv run pytest                                             # 640 unit tests: no database, no model, no network
uv run --env-file .env pytest -m integration              # 243 tests against a real Postgres
uv run pytest -m model                                    # 11 tests with the real embedding model
uv run --env-file .env python -m guide_search.evaluation.gate
```

Tests that need something fail when it is missing. They never skip, so a green run means the tests ran.

**The gate.** On every push, GitHub Actions runs all of the above, builds both Docker images and starts the API. The gate re-runs the three search modes on the evaluation questions and fails if more than one question in a mode no longer gives the committed result. The baseline and the agent are not gated: they need paid model calls and vary from run to run.

## Where things are

| File | What it does |
| --- | --- |
| `data/guides.json` | The 60 guides, as fetched by `scripts/fetch_guides.py` |
| `src/guide_search/snapshot.py` | The rules a guides file must meet |
| `src/guide_search/chunking.py` | A guide becomes passages |
| `src/guide_search/embedding.py`, `e5.py` | Text becomes a vector; the fake embedder for tests |
| `src/guide_search/store.py`, `ingest.py` | The one table, and building the index |
| `src/guide_search/ranking.py`, `search.py` | Rank fusion, grouping, and the three search modes |
| `src/guide_search/service.py` | The one object behind every way in |
| `src/guide_search/api.py` | The HTTP routes |
| `src/guide_search/mcp_server.py` | The MCP tools |
| `src/guide_search/agent.py`, `drafting.py`, `checks.py` | The agent's loop, the model call, the check |
| `src/guide_search/evaluation/` | Questions, measures, the baseline, the reports, the gate |
| `evaluation/` | The questions, the stored results and the two reports |

To understand the code, start with [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md): one real question followed from arrival to answer, with the real values at each step. The reasons for each choice are in [docs/design.md](docs/design.md) and the three stage designs beside it.

## The guide text

The guides are kept in this repository as one file, `data/guides.json`, so that every run uses the same input.

- The guide text belongs to its publisher and is included here with permission.
- It is copied unchanged, and each guide keeps the address of its public page.
- The MIT licence in `LICENSE` covers the code in this repository. It does not cover the guide text.
