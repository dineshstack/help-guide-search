# Stage 4 design: packaging

Status: draft for review, 11 October 2026. Nothing in this stage is built yet. Stages 1 to 3 are finished.

## 1. What this stage adds

No new behaviour. This stage makes what exists easy to run, hard to break, and easy to understand:

| Part | What it is |
| --- | --- |
| Finished Docker files | The service as a long-running container beside Postgres |
| Continuous integration | Every push runs the tests, and the evaluation as a gate |
| README | What the project is, how to run it, and what was measured |
| Walkthrough | One question followed through the code, for someone learning it |

## 2. Docker

Today the image is a place to run one command at a time. It becomes two images built from one `Dockerfile`:

| Image | Holds | Used for |
| --- | --- | --- |
| `runtime` | The code, its libraries and the embedding model. No tests, no test libraries. Runs as an ordinary user, not as root. | The HTTP API, the MCP server, the agent's command line, ingestion |
| `tools` | Everything in `runtime`, plus the tests and their libraries | The integration tests and the evaluation |

- **The model is built into the image**, at its pinned revision. A container then starts without downloading anything and without calling out to the model's host. The image grows by about 0.5 GB.
- **Compose gets an `api` service**: the HTTP API, started with `docker compose up -d api`, with a health check that calls `GET /health`. It waits for the database to be healthy first. Its port is published on `127.0.0.1` only.
- **The MCP server and the agent's command line run in the same image**, one command each, documented in the README.
- The one-off commands of stage 1 keep working, now through the `tools` image.

## 3. A database account that can only read (new)

Stage 2 made the service's connections read-only, and said that a separate account that may only read "belongs to stage 4". This is that.

| Account | May | Used by |
| --- | --- | --- |
| `postgres` (the owner) | Everything | Ingestion, the integration tests |
| `guides_reader` | Read the `passages` table. Nothing else. | The API, the MCP server, the agent |

So even a mistake in this project's code, or a stolen copy of the service's database address, cannot change or delete the index. The account is created by a script when the database is first set up, and ingestion grants it the right to read the table.

Cost: a second password in `.env`, and on a machine where the database already exists, the database is created again once and the index rebuilt (about a minute).

## 4. Continuous integration

One GitHub Actions workflow, run on every push to `main` and on every pull request.

| Job | Needs | Does |
| --- | --- | --- |
| Unit tests | Nothing | `pytest` |
| Integration tests | The pinned Postgres image as a service | `pytest -m integration` |
| Evaluation gate | The pinned Postgres image, the embedding model (cached between runs) | Builds the index from the committed guides, then runs the gate |

**The gate.** A new command, `python -m guide_search.evaluation.gate`, runs the three search modes on every reviewed question with the committed threshold. For each question and mode it works out two things: was it refused, and was an expected guide in the top three. It compares them with the committed `evaluation/results.json`. The build fails if more than one question differs in any one mode, and the output names the questions that differ.

- One question of slack is allowed because embeddings differ in their last decimal places between machines.
- The gate does not choose a threshold and does not score anything new. It only checks that the code still produces the committed results.
- Not gated: the title-picking baseline and the answer agent. Both need a paid model call and vary from run to run. Their results are committed and their reports rebuild without a key.

No secret is needed in CI. The workflow has no API key and never calls a model.

## 5. README

Rewritten around what a visitor needs, in this order:

1. What it is, in three sentences, and a diagram of the parts.
2. **What was measured**, as two short tables with the limits beside them:
   - search against title-picking (title-picking won at this size; hybrid is the best search mode; nothing refuses);
   - the answer agent (declines what the guides do not cover; never answered from a wrong guide on `test`; bounded by what search finds; no visible gain from a larger model).
3. How to run it: with Docker Compose, and without Docker.
4. How to use it: the HTTP API, the MCP tools, the agent.
5. How it is tested, and what the gate checks.
6. Where things are in the code.
7. The notice about the guide text, and the licence.

The results are stated as measured, including the ones that do not flatter the project.

## 6. Walkthrough

`docs/WALKTHROUGH.md` follows one real question from the moment it arrives to the moment a result is returned, naming the file and function at each step and showing the real values along the way: the cleaned question, the vector and keyword candidates with their scores, how they are fused and grouped, what the model is shown, the draft, the check, the result.

It then steps back to cover what happened before the question arrived: how a guide became passages, vectors and rows.

It is written for someone who reads Python but is new to embeddings, Postgres full-text search, rank fusion, MCP and LangGraph. Each of those is explained in a few sentences where it first appears. The values shown are produced by running the code, not written by hand.

## 7. What needs the owner

- **Pushing the workflow file starts CI.** The repository is private, so runs use the account's included Actions minutes; one run is expected to take under ten minutes.
- **One run on the Docker machine** to prove the finished image: recreate the database (for the new account), build, ingest, start the `api` service, call `/health` and `/search`, then stop it. The commands are given one at a time, as before.
- **Making the repository public** is the owner's step and is not part of this stage.

## 8. Out of scope

Hosting or deploying the service anywhere; a public address; automatic deployment; monitoring and alerting; publishing an image to a registry.
