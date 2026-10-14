# Design: reranking

Status: approved and built, 14 October 2026. Section 9 lists what changed during the build.

## 1. Why

The evaluation says where search is weak. On the 20 English `test` questions:

| | Hybrid search | Title-picking |
| --- | --- | --- |
| Right guide first | 10 | 17 |
| Right guide in the top 3 | 14 | 20 |
| A returned passage holds the answer | 12 | 20 |
| Text handed on | 2,632 characters | 6,864 characters |

The answer agent declined 6 or 7 answerable questions, mostly because search had not returned the passage that answers them. Search is cheap: it hands on about 40% of the text and needs no model call. But that saving is only worth having if the right passage arrives.

The goal of this round: **make search good enough that an assistant can rely on the returned passages instead of reading a list of all titles and then whole guides.** Reranking is the most direct way to try, and the existing evaluation measures it with no new tooling.

## 2. What reranking is

Today's vector search compares two vectors that were made separately: one for the question, one for each passage, computed in advance. That is fast, but the model never sees the question and the passage together.

A reranker (a "cross-encoder") reads the question and one passage together and gives a score for how well that passage answers that question. It is too slow to run on all 335 passages, but fast enough for the few dozen that search has already picked out. So the two are combined: search finds candidates quickly, the reranker puts them in a better order.

## 3. What changes

A fourth search mode, `rerank`:

```
question ──> hybrid search: up to 40 candidate passages (as today)
         ──> reranker: score every (question, passage) pair
         ──> order by that score ──> group by guide: 3 guides, 2 passages each (as today)
```

- The reranker reads the same text that is embedded: guide title, heading and passage text.
- The three existing modes do not change, so the stage 1 results stay comparable and the gate keeps checking them.
- `rerank` becomes the default mode only if it wins on `dev` (section 6).
- Each passage in a result also carries its reranker score.

## 4. Which model

Rules for a candidate:

- an open licence that allows commercial use;
- runs on a CPU, inside the API container's 2 CPUs;
- covers Arabic as well as English;
- loads without running code from the model's repository.

| Candidate | Licence | Size | Notes |
| --- | --- | --- | --- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0 | 471 MB | Small and fast; trained on 14 languages including Arabic |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 2.3 GB | Widely used multilingual reranker; about five times the size, so slower |
| `Alibaba-NLP/gte-multilingual-reranker-base` | Apache-2.0 | 612 MB | 74 languages; kept only if it loads without remote code |

Not considered: Jina's multilingual reranker (non-commercial licence), and hosted rerank services, which would send every question to another company and charge per call.

**How the model is chosen:** all candidates run on the `dev` questions. The winner has the most questions with a returned passage that holds the answer. Ties go to the faster model, and a model slower than one second per question on 2 CPUs is out. The chosen model is pinned to one revision and built into the Docker image, as the embedding model is.

## 5. Can the reranker refuse?

Stage 1 found that vector similarity cannot tell a question the guides do not cover from one they do. A reranker's score is a direct judgement of "does this passage answer this question", so it may separate them better.

This is measured, not assumed. The stage 1 threshold rule is applied on `dev` to the reranker's best score. If a threshold separates the unanswerable questions, `rerank` mode refuses below it. If not, it refuses nothing, and the report says so.

## 6. Measuring it

The same questions, the same measures, and three additions:

1. **Search:** `rerank` is scored beside the four existing methods. The measures are first place, top 3, passage holds the answer, refusals, text handed on, and time per question; the reranker adds time, and that is reported.
2. **The agent:** re-run with `rerank` as its search mode (a few US cents). The number to watch is how many answerable questions it still declines.
3. **Tokens a model must read per question:** for title-picking, the list of titles plus the whole picked guides; for search, the returned passages. Counted with the API's free token counter on the real texts. This turns "hands on less text" into a token figure.

**Keeping the result honest.** The 25 `test` questions have been scored before, so this round scores them again and the report says so. I recommend also adding **25 fresh English questions** (20 the guides answer, 5 they do not), drafted and reviewed under the same rules before any `rerank` run, and reported separately as a set nothing has touched. That needs about 15 minutes of review from the owner.

**When search is good enough to rely on** (proposed; the owner sets these before the run): on `test` and on the fresh set,

- a returned passage holds the answer for at least 18 of 20 answerable questions; and
- the agent declines at most 3 of 20 answerable questions, while still declining the unanswerable ones.

If `rerank` meets both, an assistant can use passages instead of whole guides. If not, the report says how far off it is, and title-picking stays the better method.

## 7. Costs

- **Time:** the reranker adds time per question. It is measured and must stay under one second.
- **Image size:** grows by the chosen model, from 0.5 GB to 2.3 GB.
- **Tests:** unit tests use a fake reranker with known scores, so they need no download. One real-model test checks the chosen reranker loads, and ranks a matching passage above an unrelated one in English and in Arabic.

## 8. Out of scope

Hosted rerank services; using a Claude model as the reranker; training or fine-tuning any model; changing the three existing search modes.

## 9. What changed during the build

- **The time limit.** No candidate met "slower than one second per question is out". On the `dev` questions, with two threads:

  | Candidate | A returned passage holds the answer | Median time |
  | --- | --- | --- |
  | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 13 of 20 | about 1.1 to 1.4 s |
  | `BAAI/bge-reranker-v2-m3` | 19 of 20 | about 7 to 8 s |
  | `Alibaba-NLP/gte-multilingual-reranker-base` | not run: it needs code from its own repository | |
  | Hybrid search, no reranker | 14 of 20 | |

  Reranking fewer candidates was also measured: bge on the top 20 gave 18 of 20 in about 4.4 s, and on the top 10 gave 17 of 20 in about 1.9 s. More threads barely helped. The owner chose to raise the limit to 10 seconds and keep all candidates. Nothing else in the rule changed, and the report says so.
- **The cost of that choice.** About 7 to 8 seconds of two CPUs per search, 2.3 GB more in the image, and the API and tools containers' memory limit raised from 3 GB to 5 GB.
- **The rerank threshold.** The rule chose 0.0025 on `dev`: it refuses 4 of the 5 unanswerable questions and wrongly refuses 2 answerable ones, and the nearest `dev` scores lie only 0.0006 away on either side. It is used as chosen; `test` and the fresh questions show how it holds.
- **The default mode.** By the rule in section 3, rerank became the default: on `dev` it returned a passage that holds the answer for 18 of 20 answerable questions against 14 for hybrid, and refused 4 of the 5 unanswerable ones against none.
- **The fresh questions.** 25, as recommended: 20 answerable and 5 unanswerable, in `evaluation/questions_fresh.jsonl`, reviewed by the owner before any search was run on them.
