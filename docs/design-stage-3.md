# Stage 3 design: answer agent

Status: draft for review, 11 October 2026. Nothing in this stage is built yet. It builds on stages 1 and 2, which are finished.

## 1. What this stage adds

A small agent that takes one question and returns one of three things: an answer taken from the help guides with its sources, a question back to the user, or a hand-over to a person.

It exists because of what stage 1 measured: search always returns its three closest guides, even for a question the guides do not cover. Deciding "these passages do not answer this" needs a reader. Here the reader is a Claude model, and plain code then checks what it wrote before anything is returned.

## 2. The flow

```
search ──> draft ──> check ──┬── passes ───────────────────────> done
             ^               ├── hand_over, or fails again ────> hand over
             │               └── fails, first time ──┐
             └──────── rewrite, told the reason ─────┘
```

| Step | What happens | Who does it |
| --- | --- | --- |
| search | The stage 1 search in its default mode: up to 3 guides with up to 2 passages each | Code |
| draft | The model reads the question and the passages and returns a structured draft | Claude |
| check | The draft is tested against fixed rules (section 4) | Code |
| rewrite | The model is shown its draft and the reason it failed, and writes it again. Once only. | Claude |
| hand over | No answer is given; the result says a person is needed, and why | Code |

So there are at most two model calls per question. The loop is written with LangGraph, a library for describing a program as named steps and the rules for which step runs next, so that the flow above is explicit in the code and each step can be tested alone.

## 3. The draft

The model's reply is requested as structured output, so it always has this shape:

| Field | Meaning |
| --- | --- |
| `decision` | `answer`, `clarify` or `hand_over` |
| `reply` | The text for the user, in the language of the question |
| `cited_guide_ids` | The ids of the guides the reply relies on |

The model is told, in instructions written for this project:

- answer only from the passages it was given, and name the guides it used;
- choose `hand_over` when the passages do not answer the question, instead of guessing;
- choose `clarify`, with one short question, when the question is too vague to answer;
- never state a number or a link that is not in the passages.

## 4. The check

Plain code, no model. A "cited passage" is a passage that search returned for a guide the draft cites, not the rest of that guide.

| Draft | Passes only if |
| --- | --- |
| `answer` | it cites at least one guide; every cited guide is one that search returned; every number in the reply is supported; every link in the reply is supported; the reply is not empty and not over 1,500 characters |
| `clarify` | it cites nothing, and contains no number and no link |
| `hand_over` | always (it goes straight to hand over) |

- **A number is supported** when it appears in a cited passage, as digits or as the matching word ("8" and "eight"). Thousands separators are ignored and Arabic digits are read as ordinary digits. Digits that only number the steps of a list in the reply are not checked.
- **A link is supported** when it is the public page address of a cited guide, or appears in a cited passage.
- The first rule that fails is the reason given to the model for its one rewrite, in plain words ("the number 48 is not in the passages of the guides you cited").

Limits, stated plainly:

- The check does not test whether each sentence is supported by the guides. A fluent wrong answer with no number and no link in it passes, if it cites a guide that was retrieved.
- A number written as a word in the reply is not checked.

## 5. Model calls, faults and cost

- **Model:** a Claude model called through the Anthropic Python SDK directly. The model id is a setting. The proposed default is `claude-haiku-5-5`, the cheapest current model. As in stage 1, the id and the request are checked against the live API with a free call before the first paid one.
- **Every setting of the request is written in the settings file** and shown in the result of an evaluation run. Nothing is tuned out of sight.
- **When the model cannot be used** (an API error after the SDK's own retries, a refusal, a reply that was cut off), the result is a hand-over with that reason. An unchecked answer is never returned.
- **When the database cannot be used**, the result is an error, as in stages 1 and 2: `503` from the HTTP route. It is not turned into a hand-over, because "we could not look" is a fault the caller must see.
- **Token use is recorded for every model call:** what the call was for (draft or rewrite), the model that served it, input and output tokens, cache-write and cache-read tokens, the stop reason and the time taken. The records are part of every result, and are also appended to a local file, one line per call, which git ignores.

## 6. How it is used

| Way in | Form |
| --- | --- |
| Command line | `python -m guide_search.agent "how do I reset my password"` prints the decision, the reply, the sources and the token counts |
| HTTP | `POST /answer` with `{"question": "..."}` |

The result, in both cases:

```json
{
  "question": "how do I reset my password",
  "decision": "answer",
  "reply": "<text for the user>",
  "sources": [{"guide_id": 3, "slug": "sign-in-and-passwords", "title": "...", "url": "..."}],
  "hand_over_reason": null,
  "check": {"attempts": 1, "failures": []},
  "usage": [{"purpose": "draft", "model": "...", "input_tokens": 0, "output_tokens": 0}]
}
```

A hand-over is only a returned decision. Nothing is sent to anyone.

One question in, one result out. The agent keeps no memory of earlier questions: after a `clarify`, the caller sends a new, fuller question.

## 7. Tests

Written first. No test calls the model: the model is replaced by a stand-in that returns prepared drafts.

| Kind | Needs | Covers |
| --- | --- | --- |
| Unit | Nothing | Every rule of the check, one test each, with numbers in digits, words, with separators and in Arabic digits, list numbering, links to guide pages and links inside passages; the whole flow with the stand-in model: passes first time, fails then passes, fails twice, the model hands over, a clarify, a model fault, no passages; exactly one rewrite and never a third call; the token records; the `POST /answer` route and its status codes; the command line |
| Integration | A real Postgres | The real search with the fake embedder and the stand-in model, end to end through `POST /answer` |

## 8. Proposed addition: measure the agent

Stage 1 left a question open: can anything refuse the questions the guides do not cover? This stage can answer it with the questions that already exist, for a few US cents.

A second evaluation command runs the agent on the reviewed questions and reports, per split:

| Questions | Measure |
| --- | --- |
| The guides can answer | Answered and cites an expected guide; answered but cites only other guides; handed over or asked back |
| The guides cannot answer | Handed over or asked back (correct); answered (wrong) |
| All | How often the check failed and a rewrite was needed; tokens per question |

The rules of stage 1 apply: the instructions are adjusted on `dev` only, `test` is scored after they are fixed, results are stored so the report rebuilds without a key, and the numbers are reported as measured.

Limit: "cites an expected guide" is not the same as "the answer is correct". Judging the text of an answer needs a person or another model, and is not done here.

## 9. Out of scope for this stage

Conversations of more than one turn; sending anything to anyone; streaming; checking each sentence against the guides; rerankers; the model calling tools by itself (the flow is fixed, and search is always the first step).

## 10. Changes to earlier code

- `anthropic` becomes a normal dependency (it was only needed for the baseline), and `langgraph` is added.
- The HTTP API gets the one new route; the service gets one new method.
