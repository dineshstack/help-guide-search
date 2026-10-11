# Stage 2 design: MCP server

Status: draft for review, 11 October 2026. Nothing in this stage is built yet. It builds on stage 1 (`docs/design.md`), which is finished.

## 1. What this stage adds

The Model Context Protocol (MCP) is a standard way to offer tools to AI assistants. An assistant that speaks MCP can start this project's server and call its tools itself, without any code written for that assistant.

This stage adds one small server that offers the stage 1 search as three tools. It adds no new search behaviour: the tools call the same code as the HTTP API.

## 2. The three tools

| Tool | Input | Returns |
| --- | --- | --- |
| `search_guides` | `question` (text, 1 to 500 characters after trimming); `mode` (optional: `vector`, `keyword` or `hybrid`) | Exactly what `GET /search` returns: the question, the mode used, up to 3 guides with up to 2 passages each, or an empty list with reason `no_match` |
| `read_guide` | `slug` | The guide's id, slug, title and public page address, and its passages in order |
| `list_guides` | nothing | Id, slug, title and public page address of every guide, in id order |

- When `mode` is left out, the default from the settings file is used, as in the HTTP API.
- Each tool has a short description written for an assistant: what it is for, when to use it, and what comes back.
- Results are structured (named fields), not text the assistant has to pick apart.

## 3. How it is built

```
assistant ── stdio ──> mcp_server.py ──> Service (stage 1) ──> Postgres
                        (thin wrapper)    search / guide / list
```

- **One new file**, `src/guide_search/mcp_server.py`, of about a hundred lines. It holds the three tool functions and a `main()` that starts the server.
- **The tools call the stage 1 `Service`**, the same object the HTTP routes use. The MCP layer adds nothing but names, descriptions and error wording.
- **Small additions to stage 1 code:**
  - a `list_guides` query, and a `Service` method for it;
  - the rule for a valid question (trimmed, 1 to 500 characters) moves into one shared function, so the HTTP route and the tool cannot drift apart;
  - `Service` opens its database connections read-only (section 4).
- **Built with FastMCP**, the standalone `fastmcp` package, pinned to an exact version (4.1.0). Its own documentation says small releases may change behaviour, which is the reason for the exact pin.

## 4. Read-only, and enforced

All three tools are marked read-only for the assistant. Those marks are only hints: an assistant may use them to decide what is safe to call, but nothing checks them.

So the promise is also enforced where it counts. Every connection the service opens is switched to read-only with one statement (`SET default_transaction_read_only = on`). After that, Postgres itself refuses any insert, update, delete or table change on that connection. This was checked on Postgres 16 before writing this design.

This covers the HTTP API as well, since it uses the same service. Ingestion is not affected; it uses its own connection.

Limit, stated plainly: this stops mistakes in this project's own code. It is not a wall against someone who holds the database password. A separate database account that may only read would be that wall, and belongs to stage 4.

## 5. Errors

The stage 1 rule holds: "we could not look" must never look like "we looked and found nothing".

| Situation | What the assistant gets |
| --- | --- |
| Nothing good enough was found | A normal result with an empty list and reason `no_match` |
| Empty or over-long question, unknown mode | An error that says which rule was broken |
| Unknown slug | An error that names the slug and says to call `list_guides` |
| Database cannot be reached, or the index is not built | An error that says so in plain words, and says it is a fault, not an empty result |
| Anything unexpected | An error with no internal detail. The detail goes to the server's own log. |

Error messages sent to the assistant never contain an address, a user name or a database message.

## 6. Running it

- **Transport: stdio only.** The assistant starts the server as a child process on the same machine and talks to it through its standard input and output. There is no network port.
- **Command:** `python -m guide_search.mcp_server`, run from the project folder with the same environment as the HTTP API (`DATABASE_URL`, and `settings.toml` for the threshold and default mode). The exact line to give an assistant is written into the README once it has been tried.
- **Start-up** loads the embedding model, which takes a few seconds, then waits for requests.
- **Standard output belongs to the protocol.** The server writes nothing else there; progress bars and logs go to standard error. A test checks this.
- Postgres must be reachable at `DATABASE_URL`, with the index already built.

## 7. Tests

Written first, as in stage 1.

| Kind | Needs | Covers |
| --- | --- | --- |
| Unit | Nothing (the tools are called in-process, with the service replaced by a stub) | Exactly three tools, each marked read-only and described; `search_guides` returns the same fields as `GET /search`; no mode means the default; `no_match` is a normal result; each input error; unknown slug; both database faults arrive as errors with plain wording and no internal detail; an unexpected error is hidden; `list_guides` is in id order; nothing is written to standard output; importing the module does not load the model |
| Integration | A real Postgres | The real service with the fake embedder and the five-guide fixture, called through the MCP client; a write attempted on a service connection is refused by Postgres |
| stdio | A real Postgres | One test starts the server as a real child process and calls a tool over stdio, to prove the wiring end to end |

## 8. Out of scope for this stage

Network transport, authentication and hosting; MCP resources and prompts (tools only); any new search behaviour; the answering agent (stage 3).
