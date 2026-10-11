# Walkthrough: one question, from arrival to answer

This follows one real question through the code. Every value shown was produced by running the code on the committed guides on 11 October 2026; none is written by hand.

It is for someone who reads Python but is new to embeddings, Postgres full-text search, rank fusion, MCP and LangGraph. Each idea is explained where it first appears.

The question:

> How can I take part payment for an invoice?

It arrives at the answer agent, either from the command line or as `POST /answer`:

```bash
uv run --env-file .env python -m guide_search.agent "How can I take part payment for an invoice?"
```

## The path in one picture

```
question
   │  clean_question            ranking.py     trim, 1 to 500 characters
   ▼
Service.answer                  service.py     the one object behind every way in
   ▼
Agent.answer                    agent.py       the loop: search → draft → check
   │
   ├─ search                    search.py
   │     ├─ embed the question  e5.py          text → 384 numbers
   │     ├─ vector search       search.py      closest passages by meaning
   │     ├─ keyword search      search.py      passages sharing words
   │     ├─ fuse the two lists  ranking.py     reciprocal rank fusion
   │     └─ group by guide      ranking.py     3 guides, up to 2 passages each
   │
   ├─ draft                     drafting.py    Claude writes a structured draft
   └─ check                     checks.py      plain code decides if it may go out
   ▼
result: decision, reply, sources, token records
```

## 1. The question is cleaned

`clean_question` in `ranking.py` trims the text and refuses an empty or over-long one. The HTTP route, the MCP tool and the command line all call this one function, so they cannot disagree about what is accepted.

```
"  How can I take part payment for an invoice?  "   →   "How can I take part payment for an invoice?"
```

## 2. One object serves every way in

`Service` in `service.py` is the real code behind the HTTP routes, the MCP tools and the agent. `Service.answer` hands the question to the agent, and gives the agent its own `search` method to search with. So the agent searches exactly as `GET /search` does.

Every database connection the service opens is read-only, and in Docker it connects as an account that may only read. Either one alone would stop a write.

## 3. The agent's loop

`agent.py` describes the loop with LangGraph. LangGraph is a library for writing a program as named steps ("nodes") and rules for which step runs next ("edges"). Each step is a small function that receives the state (everything known so far) and returns what changed.

```
search ──> draft ──> check ──┬── passes ───────────────────> done
             ^               ├── hand_over, or fails again ─> hand over
             │               └── fails, first time ──┐
             └──────── rewrite, told the reason ─────┘
```

The rule "one rewrite, never a third model call" is a counter in the state, checked in `after_check`. It does not rely on any library setting.

## 4. Search

### 4.1 The question becomes a vector

An embedding turns a text into a list of numbers (a vector) so that texts with similar meaning get vectors that point in similar directions. The model here is `intfloat/multilingual-e5-small`, pinned to one exact revision in `e5.py`.

That model was trained with a prefix on every text: `"query: "` before a question, `"passage: "` before a passage. `embedding.py` adds them in two one-line functions:

```
"query: How can I take part payment for an invoice?"
```

Out come 384 numbers, scaled so the vector has length 1. The first three are `0.0351, -0.0446, -0.0537`. This took 152 ms on a laptop CPU.

### 4.2 Vector search: closest by meaning

The 60 guides were cut into 335 passages earlier (section 8), and each passage has its own vector stored in Postgres, in a column of the `pgvector` extension's `vector` type.

Similarity between two vectors is their cosine similarity: 1 means the same direction, 0 means unrelated. In pgvector `<=>` is cosine *distance*, so similarity is `1 - (embedding <=> question)`. With 335 rows Postgres simply compares the question with every passage; no approximate index is needed.

`_vector_candidates` in `search.py` takes the 20 closest. The top five:

| Rank | Guide | Section | Similarity |
| --- | --- | --- | --- |
| 1 | record-invoice-payments | Record a payment | 0.9003 |
| 2 | frequently-asked-questions | Can I record a partial payment? | 0.8805 |
| 3 | record-invoice-payments | Choose the correct method | 0.8785 |
| 4 | record-invoice-payments | (text before the first heading) | 0.8773 |
| 5 | print-and-share-documents | Print a receipt or share an invoice | 0.8685 |

Notice how close together the scores are. That is a property of this model: even unrelated text scores around 0.77. It is why a similarity threshold could not separate questions the guides do not cover (see the evaluation report).

### 4.3 Keyword search: shared words

Postgres has full-text search built in. Each passage has a second column, `tsv`, which Postgres fills in by itself: the passage's words with common words removed and endings cut off ("invoices" and "invoice" both become `invoic`). That cut-down form is called a stem.

`question_words` in `ranking.py` pulls the words out of the question, letters and digits only:

```
['How', 'can', 'I', 'take', 'part', 'payment', 'for', 'an', 'invoice']
```

Each word is handed to Postgres separately, as a value and never as part of the query text, so nothing a person types can be read as query syntax. Postgres drops the common words and joins the rest with "or":

```
'take' | 'part' | 'payment' | 'invoic'
```

A passage matches if it contains *any* of these. Requiring all of them would return nothing for most full-sentence questions. Passages are ranked by `ts_rank_cd`, which here is in effect a count of matching words. The top five:

| Rank | Guide | Section | Rank value |
| --- | --- | --- | --- |
| 1 | record-invoice-payments | Record a payment | 1.30 |
| 2 | record-invoice-payments | Choose the correct method | 1.10 |
| 3 | advances-and-bulk-payments | Record one payment across invoices | 1.10 |
| 4 | create-and-issue-an-invoice | Prepare the invoice | 0.90 |
| 5 | invoice-and-payment-problems | The save timed out | 0.90 |

Ranks 2 and 3 tie at 1.10. Ties are always broken by guide id, then position within the guide, so results are the same on every machine.

### 4.4 Fusing the two lists

The two lists use numbers that cannot be compared: a similarity near 0.9 and a word count near 1. Reciprocal rank fusion ignores the numbers and uses only the positions:

```
score(passage) = 1 / (60 + rank in vector list) + 1 / (60 + rank in keyword list)
```

A passage missing from one list gets nothing from that list. `rrf_scores` in `ranking.py` is those few lines. The fused top, from 29 distinct passages:

| Rank | Guide | Section | Score | Vector rank | Keyword rank |
| --- | --- | --- | --- | --- | --- |
| 1 | record-invoice-payments | Record a payment | 0.0328 | 1 | 1 |
| 2 | record-invoice-payments | Choose the correct method | 0.0320 | 3 | 2 |
| 3 | print-and-share-documents | Print a receipt or share an invoice | 0.0303 | 5 | 7 |
| 4 | advances-and-bulk-payments | Record one payment across invoices | 0.0302 | 10 | 3 |
| 5 | create-and-issue-an-invoice | Prepare the invoice | 0.0301 | 9 | 4 |

First place in both lists gives `1/61 + 1/61 = 0.0328`. A passage that both searches found beats one that only one search ranked highly: the FAQ passage, second by meaning, is not in this top five because keyword search ranked it low.

### 4.5 Grouping by guide

`group_by_guide` in `ranking.py` turns passages into the final shape: at most three guides, each with at most two of its passages. A guide's score is the score of its best passage.

| Guide | Score | Passages returned |
| --- | --- | --- |
| record-invoice-payments | 0.0328 | "Record a payment" (577 characters), "Choose the correct method" (637) |
| print-and-share-documents | 0.0303 | "Print a receipt or share an invoice" (507) |
| advances-and-bulk-payments | 0.0302 | "Record one payment across invoices" (519), "Record an advance" (732) |

That is 2,972 characters of passage text. The whole of those three guides would be about 6,500.

### 4.6 Refusing

Search may return nothing, with the reason `no_match`, when the best *vector* similarity is below a threshold. The fusion score cannot be used for this: its top value is about 0.033 whether the passage is a perfect answer or nonsense.

Here the best similarity is 0.9003 and the committed threshold is 0.0, so nothing is refused. The threshold is 0.0 because the evaluation found no value that separated the questions the guides cannot answer. Declining those is the agent's job, in the next step.

A database that cannot be reached, or an index that is empty, is never reported as `no_match`. It raises an error, and the HTTP route answers `503`.

## 5. The draft

`drafting.py` builds one request to the Claude API. The model is given instructions written for this project, the question, and the passages with each guide's id, title and web address. The whole user message is 3,748 characters.

The reply is requested as structured output: the API is given a JSON schema and holds the reply to it. So the draft always has three fields:

| Field | Value for this question |
| --- | --- |
| `decision` | `answer` |
| `cited_guide_ids` | `[27, 29]` |
| `reply` | Six numbered steps for recording a payment, then: "Enter only the amount actually received. It must be at least 0.01 and must not exceed the balance due. A partial payment leaves the unpaid remainder on the invoice..." |

The call used 1,976 input tokens and 363 output tokens with `claude-opus-5-5` and took 4.3 seconds. A record of that is kept with the result.

The model had three choices: `answer`, `clarify` (ask one question back) or `hand_over` (the passages do not answer this). For "Can our card machine be hooked up so a card payment shows on the bill by itself?" it chose `hand_over`, because no guide covers that.

## 6. The check

`checks.py` decides, with plain code and no model, whether the draft may be returned. For an answer:

| Rule | This draft |
| --- | --- |
| It cites at least one guide | Yes: 27 and 29 |
| Every cited guide is one that search returned | Yes: both are in the result above |
| Every number in the reply is in the passages of the cited guides | The reply states one number, `0.01`. The passage "Record a payment" says the amount must be at least 0.01. Supported. |
| Every link in the reply is a cited guide's page, or in its passages | The reply has no link |
| The reply is not empty and not over 1,500 characters | Yes |

The digits `1.` to `6.` at the start of the reply's lines are not treated as numbers: they only number the steps.

The draft passes, so there is no rewrite. Had the reply said "at least 0.05", the check would have failed with the reason *"The number 0.05 is not in the passages of the guides you cited"*. The model would have been shown its draft and that sentence, once. A second failure would have become a hand-over.

What the check does not do: it does not test whether each sentence is supported by the guides. It catches invented numbers, invented links and invented sources.

## 7. The result

```json
{
  "question": "How can I take part payment for an invoice?",
  "decision": "answer",
  "reply": "To record a part payment on an invoice: ...",
  "sources": [
    {"guide_id": 27, "slug": "record-invoice-payments", "title": "Record a full or partial invoice payment", "url": "https://support.orions360.com/help/en/autoserv-pro/record-invoice-payments"},
    {"guide_id": 29, "slug": "print-and-share-documents", "title": "Print a job card, invoice or payment receipt", "url": "https://support.orions360.com/help/en/autoserv-pro/print-and-share-documents"}
  ],
  "hand_over_reason": null,
  "check": {"attempts": 1, "failures": []},
  "usage": [{"purpose": "draft", "model": "claude-opus-5-5", "input_tokens": 1976, "output_tokens": 363, "stop_reason": "end_turn", "seconds": 4.3}]
}
```

Every source is a guide that search returned and the model cited, with the address of its public page. (The token record is shortened here; it also holds the two cache counts.)

## 8. Before the question arrived: how the index was built

`python -m guide_search.ingest` prepares everything search needs, from one file.

| Step | File | What happens |
| --- | --- | --- |
| Snapshot | `data/guides.json`, `snapshot.py` | The 60 guides are kept in the repository as one file, fetched from the public help centre by `scripts/fetch_guides.py`. `validate_snapshot` refuses a file that breaks a rule, listing every problem. |
| Chunking | `chunking.py` | Each guide is split at its `##` headings. A section over 1,200 characters is cut into the fewest pieces that fit, at paragraph or line boundaries. The "Related articles" section is left out. 60 guides become 335 passages. |
| Search text | `chunking.py` | Each passage is embedded and keyword-indexed as *guide title + heading + text*. Many passages are too short to mean anything alone: "No. Stop or finish the running task or package before starting another." needs its heading, "Can I run two technician timers?" |
| Embedding | `e5.py` | `"passage: "` + search text → 384 numbers |
| Storing | `store.py`, `ingest.py` | One table, `passages`, with the vector and the keyword column. Ingestion runs in one transaction and replaces only the guides that changed, so running it twice changes nothing the second time. |

## 9. The same search, offered to an assistant

`mcp_server.py` offers the search to AI assistants through the Model Context Protocol, a standard way to give an assistant tools. The assistant starts the server as a child process and calls `search_guides`, `read_guide` or `list_guides`. Each tool calls the same `Service` as everything above, so step 4 is identical.

## 10. How this is tested

| Kind | How many | What it needs |
| --- | --- | --- |
| Unit | 640 | Nothing. The model is replaced by a fake that builds vectors from words, and the database by small stubs. |
| Integration | 243 | A real Postgres. Five invented guides are ingested and searched. |
| Real model | 11 | The embedding model download |

On every push, GitHub Actions runs all three, then the regression gate: it runs the three search modes on the evaluation questions again and fails if more than one question in a mode no longer gives the committed result.

## Where to read next

- `docs/design.md` for why each choice was made.
- `evaluation/report.md` and `evaluation/agent_report.md` for what was measured.
- `tests/unit/test_checks.py` and `tests/unit/test_ranking.py`: the shortest way to see the rules, one small example each.
