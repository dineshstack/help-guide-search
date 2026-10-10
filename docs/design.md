# SAM guide search: design

Status: draft for review, 10 October 2026. Nothing is built yet.

## 1. What this is

SAM (Smart Auto Manager) is workshop management software. Its help centre has 60 public guides.
This project is a small standalone Python service that:

1. finds the passages in those guides that answer a question, by meaning as well as by keyword;
2. measures how well it does that, against a simpler method, on a fixed set of questions;
3. offers the search to AI assistants as tools, and uses it in a small answering agent that is told to answer only from the guides and whose citations, numbers and links are checked in code.

It reads the public guides and nothing else. It is not connected to any live system, holds no customer data, and is not deployed anywhere.
Every prompt, tool description and fixed reply is written for this project and kept in this repository.

The evaluation answers one question: **is search better than letting a model pick guides from a list of their titles?**
The result is reported as measured, whichever way it comes out.

## 2. The guides

Measured on 10 October 2026 on a copy of the 60 guides. Ids, titles and update times were checked against the live catalogue that day. The figures are checked again by a test when the snapshot is built.

| Fact | Value |
| --- | --- |
| Guides | 60, English only. Asking the help centre for Arabic returns the English guide, marked as a fallback. |
| Body | Markdown, 1,747 to 4,198 characters each, about 135,000 characters in total |
| Headings | Only `##`. 332 headings in total, 3 to 23 per guide |
| Text before the first heading | Present in 59 guides. In two of them it is the main content. |
| "Related articles" | The last section of every guide. A list of links, no answer text. |
| Repeated headings | "What happens next" and "Troubleshooting" each appear in 47 guides |
| FAQ guide | One guide holds 22 short answers. The question is in the heading; the answer is often just "Yes" or "No" plus a clause. |
| Not present | Tables, images, code blocks, raw HTML, headings other than `##` |

Three things follow from this and shape the design:

- A passage on its own is often too short to mean anything. About half are under 250 characters. So the guide title and the section heading are always attached to a passage before it is searched.
- "Related articles" sections name *other* guides. Indexing them would make a guide match questions about its neighbours. They are left out of the index.
- The guides are English and some questions will be Arabic. Search by meaning has to work across languages. Keyword search cannot.

## 3. Build order

Each stage works on its own and is reviewed before the next begins.

| Stage | What it adds |
| --- | --- |
| 1 | Ingestion, search (three modes), an HTTP API, and the evaluation |
| 2 | An MCP server with three read-only tools, so AI assistants can use the search |
| 3 | An answer agent that searches, drafts, checks its own draft in code, and hands over when the draft fails that check twice |
| 4 | Packaging: finished Docker files, continuous integration with the evaluation as a regression gate (a check that fails the build if results get worse), a README with measured results, and a code walkthrough |

Stage 1 is designed in detail below. Stages 2 to 4 are outlined; each gets its own short design before it is built.

## 4. Stage 1: ingestion, search and evaluation

```
data/guides.json ──> chunking ──> embeddings ──> Postgres (passages table)
                                                        │
question ──> search (vector | keyword | hybrid) <───────┘
                 │
                 └──> up to 3 guides, up to 2 passages each, or "no_match"
```

### 4.1 Snapshot

The guides are kept in the repository as `data/guides.json`, so every run uses the same input.

```json
{
  "source": "https://support.orions360.com/api/kb",
  "product": "autoserv-pro",
  "locale": "en",
  "fetched_at": "<date of fetch>",
  "guides": [
    {
      "id": 3,
      "slug": "sign-in-and-passwords",
      "title": "Sign in or reset your staff password",
      "url": "https://support.orions360.com/help/en/autoserv-pro/sign-in-and-passwords",
      "updated_at": "<ISO 8601 timestamp, UTC>",
      "body": "<Markdown>"
    }
  ]
}
```

Guides are stored in order of `id`. The file holds only what the public help-centre API returns, plus the public page address.

The guide text belongs to its publisher and is included with permission. It is copied unchanged, and each guide keeps the address of its public page. The code is under the MIT licence. That licence covers the code and not the guide text, and the README says so from the first commit that contains `data/guides.json`.

`data/guides.json` is only ever written by `scripts/fetch_guides.py`, from the public API. It is never copied or converted from another file. The script:

- reads the catalogue with an explicit page size (the default page holds only 30 guides) and checks the count against the total the API reports;
- fetches each guide, and rejects any response marked as a language fallback;
- waits 1.5 seconds between requests, stops with an error if the server says it is being asked too often, and sends a user agent that names this project;
- validates the result before writing anything.

Validation lives in one module and is used by both the fetch script and ingestion. A snapshot is valid when:

- the file has exactly the five top-level keys shown above, and every guide has exactly the six keys shown above. Any other key fails;
- ids, slugs and titles are unique;
- the slug is lower-case words joined by hyphens, and the `url` is the public page address for that slug;
- `updated_at` is a timestamp with a time zone;
- the body is not empty and uses plain line breaks (no carriage returns).

Ids are not assumed to be consecutive (today they are 1 to 59 and 61).

### 4.2 Chunking

Chunking turns one guide into several passages.

1. Split the body at lines that start with `## `. The text before the first heading is the "intro" and is treated as a section with an empty heading.
2. Leave out the section whose heading is exactly "Related articles".
3. Trim each section. The `## ` line is not part of the text; the heading is kept separately. A section with no text gives no passage. A section of 1,200 characters or fewer is one passage.
4. A longer section is split into pieces of at most 1,200 characters:
   - Its units are its paragraphs (separated by blank lines). A paragraph that is itself over 1,200 characters, which happens with long lists, contributes its lines as units instead.
   - Units stay in order and keep their original separators, which count toward the length.
   - Use the smallest number of pieces that fits. Among the ways to do that, use the one whose longest piece is shortest, so no piece is a stray last line. If still tied, cut earlier.
   - Pieces do not overlap.
5. If a single line is longer than 1,200 characters, chunking stops with an error naming the guide. It never produces an over-long passage.

Each passage keeps: guide id, slug, title, public URL, heading, position within the guide (0, 1, 2, ...), its text, and a content hash.

The text that is embedded and keyword-indexed is called `search_text`: the guide title, the heading and the passage text, joined by newlines. For an intro the heading line is left out. The text shown to a reader is the passage text alone.

Why these choices:

- **1,200 characters.** The embedding model reads at most 512 tokens (word pieces) and silently ignores the rest. 1,200 characters plus title and heading is expected to stay inside that, and a test with the real model checks that no passage is cut off.
- **No overlap.** Overlap stops a sentence losing its context at a cut. Here only four sections are cut, all inside long lists whose items stand alone, and every piece carries the title and heading. Overlap would only store the same text twice.

On today's guides this rule gives **335 passages**, 4 to 22 per guide, between 71 and 1,143 characters. Four sections are split, each into two pieces. A test fixes these numbers and asserts that every passage has between 1 and 1,200 characters.

### 4.3 Embeddings

An embedding turns a piece of text into a list of numbers (a vector) so that texts with similar meaning have vectors that point in similar directions. Cosine similarity measures how closely two vectors point the same way: 1 is the same direction, 0 is unrelated. It is the "similarity" used in the rest of this document.

| Setting | Value |
| --- | --- |
| Model | `intfloat/multilingual-e5-small`, loaded with `sentence-transformers` |
| Revision | Pinned to commit `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Size | 384 numbers per vector; about 0.5 GB to download |
| Input limit | 512 tokens including the prefix. Longer input is cut off without warning. |
| Device | CPU, set explicitly. Left alone, the library would use the GPU on an Apple laptop and the CPU elsewhere, and results would differ slightly between machines. |
| Prefixes | `"passage: "` in front of every passage and `"query: "` in front of every question. The model was trained with them and is worse without. |
| Normalised | Every vector is scaled to length 1, so similarity depends only on direction |

This model is used because it runs on a laptop CPU, has a permissive licence (MIT), and was trained on about 100 languages including English and Arabic, so a question in one can match a passage in the other.

The prefixes are added by two small functions in this project, `passage_input(text)` and `query_input(text)`, as ordinary string concatenation. They are called inside the real embedder. It is one visible line each, covered by a unit test.

The rest of the code sees the model only through a small interface:

```python
class Embedder(Protocol):
    model_id: str   # model name and revision; stored with every vector

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

There are two implementations.

- The real one loads the model.
- The fake one, for tests, needs no download. It builds a vector from the words in the text: each lower-cased word adds 1 at a position chosen by a fixed hash of the word (not Python's built-in `hash()`, which changes between runs), and the result is scaled to length 1. Texts that share words score higher. Texts with no shared words score 0 unless two different words happen to land on the same position, which is common with only 384 positions; such a score stays small. So a test can state which passage must come first, and a no-match test asserts "below the test's own threshold", never "exactly 0". Text with no words maps to one fixed vector.

One property of the real model matters later: its similarity scores are bunched together. Its documentation says they mostly fall between 0.7 and 1.0, so even unrelated texts score high and good matches sit in a narrow band above them. Where unrelated text falls for these guides, and for Arabic questions, is not known until it is measured. The "too weak to be a match" threshold is therefore measured (section 4.7), not guessed.

### 4.4 Storage

Postgres 16 with the `pgvector` extension, running in Docker from the image `pgvector/pgvector:0.8.7-pg16`, pinned by tag and digest in `compose.yaml`.

Postgres is used rather than a separate vector database because it also does the keyword search and holds the passage text. One table and one service cover everything.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS passages (
  id              bigserial PRIMARY KEY,
  guide_id        integer NOT NULL,      -- the help centre's own id
  slug            text    NOT NULL,
  title           text    NOT NULL,
  url             text    NOT NULL,
  heading         text    NOT NULL,      -- '' for the intro
  position        integer NOT NULL,      -- order within the guide
  content         text    NOT NULL,      -- passage text, as shown to a reader
  search_text     text    NOT NULL,      -- title, heading and content
  content_hash    text    NOT NULL,      -- sha256 of search_text
  embedding_model text    NOT NULL,      -- the embedder's model_id
  embedding       vector(384) NOT NULL,
  tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
  UNIQUE (guide_id, position)
);

CREATE INDEX IF NOT EXISTS passages_tsv_idx ON passages USING GIN (tsv);
```

- `embedding` is the vector column. There is no approximate vector index: with 335 rows Postgres compares the question with every passage exactly, which is quick and exactly right at this size.
- `tsv` is the keyword-search column. A `tsvector` is Postgres's prepared form of a text for keyword search: its word stems and where they occur. Postgres fills it in by itself from `search_text`. The GIN index is a lookup from each stem to the rows that contain it.

**Ingestion is re-runnable.** In one transaction it:

1. creates the table and index if they do not exist;
2. loads and validates the snapshot and chunks every guide;
3. for each guide, compares the complete ordered list of new passages with the stored rows, on every stored field except the row id and the vector. If they are the same the guide is skipped. Otherwise that guide's rows are deleted and the new ones are embedded and inserted;
4. deletes rows whose guide is no longer in the snapshot;
5. prints what it did: guides unchanged, replaced, added, removed.

Because it is one transaction, a failure half-way leaves the previous index untouched. Running it twice in a row changes nothing the second time. A change of title, slug, URL, heading, text, section order or embedding model replaces the guide.

### 4.5 Search

Input: a question and a mode. The mode is `vector`, `keyword` or `hybrid`. The default is configuration; it is `hybrid` until the evaluation shows another mode is better.

**Vector.** Embed the question, then order passages by similarity. In pgvector, `<=>` is cosine distance (0 for the same direction), so similarity is `1 - (embedding <=> question_vector)`.

**Keyword.** Postgres full-text search with the English configuration, which removes common words and reduces the rest to stems (a stem is a word with its ending removed, so "invoices" and "invoicing" count as the same word).

- A passage matches if it contains *any* word of the question. Requiring *all* the words, which is what Postgres does with plain text by default, is expected to return nothing for most full-sentence questions.
- The query is built without letting the question be read as query syntax: the code pulls the words out of the question (letters and digits only), passes each one to `plainto_tsquery('english', word)` as a parameter, and joins the results with the `||` (or) operator. If no word is left after common words are removed, the keyword list is empty.
- Passages are ranked by `ts_rank_cd` with its default options. With any-word matching this is in effect a count of matching words, title and heading included. It does not know rare words from common ones and it favours long passages. That weakness is measured, not tuned away.
- A question written only in Arabic matches nothing in this mode: Postgres accepts the Arabic words, and no English passage contains them. A Latin-script word or a number inside an Arabic question ("WhatsApp", "VAT", "48") is matched like any other word, so such a question returns passages chosen on that word alone.

**Hybrid.** Take the top 20 passages from vector search and the top 20 from keyword search and combine them with reciprocal rank fusion. Ranks start at 1.

```
score(passage) = 1 / (60 + rank in vector list) + 1 / (60 + rank in keyword list)
```

A passage missing from one list gets nothing from that list. The 60 makes neighbouring ranks nearly equal in value, so a passage found by both searches beats one that tops only one list. It is the value from the paper that introduced the method, which found results change little around it, so it is not tuned here. Twenty per list is a round number and is not tuned either.

If keyword search finds nothing, hybrid gives the same result as vector. For Arabic questions hybrid is therefore expected to equal vector at best, and to trail it when a Latin-script word pulls in unrelated passages.

**Ordering is repeatable.** In every list (vector, keyword, fused, guides) ties are broken by guide id and then position. The row id is never used for ordering, because it changes when a guide is re-ingested.

**Grouping by guide.** In every mode the candidates are a short list: the top 20 by similarity (vector), the top 20 by keyword rank (keyword, possibly fewer), or the fused list (hybrid). Candidates are grouped by guide, and a guide's score is the score of its best candidate. The result is up to 3 guides, each with up to 2 of its candidate passages, best first. Fewer than 3 guides, or a guide with one passage, is a normal result. Three guides because the baseline also returns up to three; at most two passages per guide to keep the text handed to an answering model small.

**No match.** Sometimes the honest result is "the guides do not cover this".

| Mode | The result is empty, with reason `no_match`, when... |
| --- | --- |
| vector | the best passage's similarity is below the threshold |
| keyword | no passage contains any word of the question |
| hybrid | the best *vector* similarity is below the same threshold |

- Hybrid uses the vector similarity for this decision because a fusion score depends only on rank positions: the top passage scores about 1/61 whether it is a perfect answer or nonsense.
- In hybrid the decision is made once per question. The guides are then ordered by fusion, so the first guide can be one that only keyword search found.
- Keyword mode has no threshold. It refuses only when not one word matches, so a question about the product that the guides do not cover is rarely refused.
- The threshold is chosen by the evaluation and committed (section 4.9).

**Errors are not "no match".** If the database cannot be reached, or holds no passages for the embedding model in use, search raises an error. It never returns an empty result, because "we could not look" and "we looked and found nothing" must not look the same to the caller.

### 4.6 HTTP API

FastAPI, three read-only routes.

| Route | Returns |
| --- | --- |
| `GET /search?q=...&mode=hybrid` | The search result. `422` if `q` is blank or over 500 characters, or `mode` is not one of the three. |
| `GET /guides/{slug}` | The guide's title and URL and its passages in order. `404` if the database answered and has no such guide. |
| `GET /health` | `200` when the model is loaded and the table holds passages for it. |

Every route returns `503` when the database is unavailable or the index is empty, through one error handler.

A search result has this shape (the numbers are made-up examples):

```json
{
  "query": "how do I change my password",
  "mode": "hybrid",
  "reason": null,
  "best_similarity": 0.88,
  "guides": [
    {
      "guide_id": 3,
      "slug": "sign-in-and-passwords",
      "title": "Sign in or reset your staff password",
      "url": "https://support.orions360.com/help/en/autoserv-pro/sign-in-and-passwords",
      "score": 0.0328,
      "passages": [
        { "heading": "<section heading>", "position": 2, "text": "<passage text>", "similarity": 0.88 }
      ]
    }
  ]
}
```

- Every passage sits under its guide's id, slug, title and URL and carries its own heading and position, so anything shown can be traced to its source.
- `score` is the guide's ordering value in the mode used: similarity in vector mode, `ts_rank_cd` in keyword mode, the fusion score in hybrid mode. It is comparable only within one response.
- `similarity` is the passage's cosine similarity to the question. It is given for every returned passage in vector and hybrid modes, and is `null` in keyword mode, where the question is never embedded.
- `best_similarity` is the value that was compared with the threshold (`null` in keyword mode).
- When nothing is good enough, `guides` is empty and `reason` is `"no_match"`.

### 4.7 Evaluation

**Questions.** `evaluation/questions.jsonl`, one question per line:

```json
{"id": "en-001", "language": "en", "question": "...", "expected_slugs": ["sign-in-and-passwords"], "answer_headings": {"sign-in-and-passwords": ["<section heading>"]}, "split": "dev", "reviewed": false}
```

- About 60 questions in three groups: about 40 English and about 10 Arabic that the guides can answer, and about 10 that they cannot.
- They are drafted with an AI model that has read the guides, each answerable question from a different guide, phrased as a customer would ask and without copying phrases from the guide's title, headings or text. The project owner then reviews each one. **Only questions marked `reviewed: true` are counted.**
- Arabic questions are written independently, not as translations of the English ones, and are reviewed by someone who reads Arabic. They are drafted and committed with the rest and stay `reviewed: false` until that review happens. Until then the report covers English only and says so. When they are reviewed, the threshold is chosen again on `dev` and the report is rebuilt.
- No question is copied or adapted from a real customer message.
- `expected_slugs` lists every guide that would be an acceptable answer. Some facts appear in more than one guide, and a problem can be covered both by a topic guide and by a troubleshooting guide. Finding any listed guide counts.
- `answer_headings` records, for each expected guide, the section or sections that hold the answer (`""` for the intro).
- The unanswerable questions have an empty `expected_slugs`. They are all English, and all near misses: about the product, but not covered by any guide. Before sign-off each one is checked against the guide text to confirm that.

**Order of work**, so that results cannot shape the questions:

1. Draft and review the questions.
2. Commit `questions.jsonl`.
3. A script assigns the split: within each group the questions, reviewed or not, are shuffled with a fixed seed and half go to `dev`, half to `test`. Nobody edits `split` by hand, and the evaluation refuses a file whose split is not the one the script gives.
4. Only then is any method run.

After step 2 no question is added, removed or un-reviewed. Marking the Arabic questions as reviewed later is allowed; nothing about them is scored before that. A label may be corrected, but only as a logged correction, a line in `evaluation/corrections.jsonl`: the report lists each one (question, old and new expected guides, reason) and every method is re-scored.

`dev` is used to choose the threshold. `test` is kept aside so that the headline numbers come from questions that played no part in any choice. Half each, because both jobs need as many questions as a small set can give.

**Methods compared.**

| Method | How it finds guides |
| --- | --- |
| Title-picking (the baseline) | A Claude model is shown the id, slug and title of all 60 guides plus the question. It returns up to 3 slugs it would read, most useful first, or none. |
| Vector | Section 4.5 |
| Keyword | Section 4.5 |
| Hybrid | Section 4.5 |

Title-picking is the simplest retrieval a model-driven assistant can do, and it is the method search has to beat to be worth having.

- The model is fixed at `claude-sonnet-4-6`. Its id is stored with every result. Every optional setting of the request is left at the API's default.
- The answer is requested as structured output: the API is given a JSON schema, here a list whose items must be one of the 60 slugs, and constrains the reply to fit it. The schema cannot limit the list to three or forbid repeats, so code removes repeats and keeps the first three.
- A call that fails, is cut off or is refused is an error. It is repeated, and is never stored as "the model picked nothing". An empty list is stored only when the model returned one.
- Results are stored in a committed file, `evaluation/title_picking.jsonl`. Each line holds the question id, a hash of the question text, a hash of the full request (prompt, guide list, schema), the model, the date, the picked slugs, token counts and response time. The report can then be rebuilt by anyone without an API key, and still after the model is retired. The report script refuses to run if a line no longer matches the current questions or guides.
- A model's output varies from run to run, and the old setting for reducing that (temperature) never guaranteed identical output and is rejected by newer models. It is left alone. The baseline is run once, to keep the cost down, and the report says so.
- The baseline's prompt and settings are committed before its first run. If the prompt is revised, only `dev` results may be looked at, and the report says how many revisions there were. That number is given to the report command, which has no default for it.

**What is measured**, for every method.

| Measure | Meaning |
| --- | --- |
| hit@1 | The first guide returned is an expected one |
| hit@3 | An expected guide is among those returned |
| Passage hit@3 | hit@3, and one of that guide's returned passages comes from a section listed in `answer_headings`. For title-picking it equals hit@3, because the whole guide is handed over. |
| MRR | 1 divided by the position of the first expected guide (1, 1/2, 1/3, or 0 if absent), averaged over questions |
| Wrong refusals | Answerable questions where nothing was returned |
| Correct refusals | Unanswerable questions where nothing was returned. The rest are "answered when it should have refused". |
| Guides returned | Average number of guides returned per question that was not refused. Shown as `-` when every question was refused. |
| Text handed over | For search, the total length of the returned passage texts. For title-picking, the total length of the picked guides' whole bodies. Titles, headings and URLs are not counted. Median over questions that were not refused; `-` when every question was refused. |
| Time | Median and maximum per question, over all questions together |

How each is calculated:

- Every measure is calculated for `dev` and for `test` separately. The two are never pooled, because the threshold was fitted on `dev`.
- Within a split, hit@1, hit@3, passage hit@3, MRR and wrong refusals are calculated over answerable questions only, separately for English and Arabic. A refused answerable question scores 0 on all of them. A method may return fewer than three guides; the measures use what was returned.
- Correct refusals are calculated over the unanswerable questions, which are all English.
- Every rate is shown with its count ("14 of 20").
- Search is timed around the search function, including embedding the question, after one discarded warm-up query. Title-picking time is the model's response time, recorded when the baseline was run. These are different kinds of time (local computation against a network call to a model) and are shown side by side, not compared.
- The report also gives hit@1, hit@3 and MRR for vector and hybrid with the threshold switched off, labelled "ranking only" and also given per split, so that a weak result can be traced to ranking or to refusing.
- Every miss is listed with the question, what was expected and what came back.

**Choosing the threshold.** Using only the `dev` questions:

1. For each question, record the best vector similarity.
2. The candidate thresholds are 0 (never refuse) and the midpoint between each pair of neighbouring recorded values.
3. For each candidate, count the questions handled correctly: an answerable question is not refused, and an unanswerable one is.
4. Use the candidate with the highest count. If several tie, use the lowest, so that when in doubt search answers rather than refuses.

The same value serves vector and hybrid, which share one refusal rule, and both languages. The report prints the candidates and their counts, the gap between the chosen threshold and the nearest `dev` score on each side, and the lowest, median and highest best similarity for English answerable, Arabic answerable and unanswerable questions. If 0 wins, the report says that similarity did not separate unanswerable questions and that search never refuses.

**Keeping `test` honest.** The threshold is frozen before `test` is scored, and the evaluation script scores `test` only when asked to. When asked, it first checks that the committed threshold is the one the rule gives on `dev`, and stops before any `test` question is searched if it is not. Nothing is ever chosen on `test`: any later change to chunking, search or the threshold is decided on `dev` only. The per-question outcomes of the reported run are committed as `evaluation/results.json`. If search changes after that, the report is rebuilt and, because a results file from an earlier run is there, says that `test` was scored again. Code cannot enforce this; the git history makes it checkable.

**The headline.** The comparison named in advance is hybrid against title-picking on the English `test` questions, on hit@1 and on hit@3. For each, the report gives two counts: questions hybrid got right and title-picking got wrong, and the reverse. It says "better" or "worse" only if an exact two-sided sign test on those two counts gives p below 0.05. Questions that both methods got right, or both got wrong, do not enter the test. That needs at least six wins with no losses, eight with one loss, or ten with two. Otherwise it says "no difference that this set can show". All other numbers are descriptive.

**The report.** A script writes `evaluation/report.md` and `evaluation/results.json` from the questions, the baseline results file and a run of the three search modes against a database ingested from empty. It records the snapshot date, the commit of the question file, the model revision, the threshold, the run date, and a description of the machine (processor, cores, memory, operating system, Python version; never a host name, user name or file path; on Linux only the word "Linux", no kernel version). The commit of the question file is given to the command.

**Limits to state in the report.**

- The set is small. In `test`, one English question moves hit@1 or hit@3 by about 5 points (1 of about 20). One Arabic question, or one unanswerable question, moves its result by about 20 points (1 of about 5).
- The threshold is set from about five unanswerable `dev` questions and checked on about five more. The refusal numbers show whether refusing works at all, not how often. N of the M scored questions are unanswerable because the set was built that way; the threshold depends on that mix. (The report fills in the two numbers.)
- One threshold serves both languages and is fitted mostly on English questions. Arabic wrong refusals show what that costs. Refusing Arabic questions that the guides do not cover is not measured.
- Keyword mode has no threshold, so its refusal numbers follow from question wording and language, not from tuning. Vector and hybrid share one refusal rule, so their refusal numbers are the same by construction.
- The questions are drafted by a model that has read the guides. They may echo guide wording more than real customers would, which flatters keyword and vector search. The drafting rules and the review reduce this; they do not remove it.
- The expected guides are one reviewer's labels, and the reviewer also built the search. A missing acceptable guide turns a right result into a miss.
- Title-picking here is one choice from id, slug and title. An assistant that reads a guide before choosing the next may do better, so beating this baseline is not the same as beating such an assistant. The slug is shown as well as the title, and many slugs contain words the title does not, so the baseline has slightly more than titles to go on. Its model is not the newest available; a newer one might pick better. It was run once, and a second run could move its numbers by a question or two.
- A guide-level hit for search does not mean the returned passages hold the answer; passage hit@3 is there to show the difference. Text handed over is only like for like when read beside it.
- Arabic questions are matched against English passages. Published benchmarks for this embedding model show that this is clearly weaker than English to English, so lower Arabic results are expected. The baseline's model reads Arabic directly and has no such handicap.

While no Arabic question is reviewed, the report adds one line under these limits: the points about Arabic results do not apply to it yet.

### 4.8 Tests

pytest, with each test written before the code it tests. A marker is a label on a test, such as `@pytest.mark.integration`; `pytest -m integration` runs only the tests with that label.

| Kind | Needs | Covers |
| --- | --- | --- |
| Unit | Nothing | Chunking; rank fusion; snapshot validation; the prefix functions; the metric calculations and the threshold rule; the HTTP routes with search replaced by a stub (`422`, `404`, `503` rather than an empty result when search raises a database error, health `200` and `503`) |
| Integration | A real Postgres | Storing and searching, on a fixture of five small guides with the fake embedder: ingest; re-run changes nothing; a guide whose text changes is replaced; a guide whose slug and URL change but whose text does not is replaced; a guide that loses a section ends with one row fewer; a removed guide disappears; each search mode; grouping; `no_match`; an empty table is an error; a database outage is an error. Keyword cases: a full-sentence question, a question of only common words, a question with an apostrophe and a question mark, an all-Arabic question, an Arabic question with one Latin word. |
| Real model | The model download | One short file: the pinned model loads; vectors have 384 numbers and length 1; a question ranks its matching passage above an unrelated one; `embed_query(q)` equals the model's encoding of `"query: " + q` and differs from its encoding of `q`; no passage in the snapshot is longer than 512 tokens |

- Plain `pytest` runs the unit tests. Integration and real-model tests are selected by marker, and fail rather than skip if what they need is missing, so a green run always means the tests actually ran.
- Integration tests read `TEST_DATABASE_URL`, never `DATABASE_URL`. It points at a separate database that the tests are free to empty, so they can never wipe the real index.
- Search tests pass their own threshold, never the calibrated one, because the fake embedder's scores are on a different scale.

### 4.9 Running it

| Piece | Choice |
| --- | --- |
| Python | 3.13 everywhere: Docker, continuous integration and local work. It is the newest version every dependency declares support for. |
| Dependencies | Managed and locked with `uv` (`pyproject.toml` and `uv.lock`) |
| torch | On Linux, the CPU-only build, set up in `pyproject.toml`. Without that, Linux would install a much larger GPU build. macOS uses the ordinary one. |
| Database | `compose.yaml` defines a `db` service from the pinned pgvector image, with a health check and a data volume |
| App image | `Dockerfile` builds an image used for one-off commands: ingestion, the integration tests and the evaluation (`docker compose run app ...`) |
| Postgres without Docker | Optional. `scripts/local_postgres.py` starts a throwaway Postgres with pgvector from a Python package, for working on a machine that has no Docker. It is Postgres 16 like the pinned image, but not the same build, so reported results always come from the pinned image. |
| Model files | Kept in a cache directory, which in Compose is a named volume, so a new container does not download the model again. After the first download the model's files are read from that cache. The library still makes two small metadata requests each time it loads; setting `HF_HUB_OFFLINE=1` stops them. |
| Settings that differ between machines | Environment variables. `DATABASE_URL` says where Postgres is; `TEST_DATABASE_URL` is for the integration tests. An example file is committed; real values never are. |
| Settings that are the same everywhere | The threshold and the default mode live in one committed settings file. It is written when the threshold is chosen (section 4.7) and read by search, the API, the evaluation and the stage 4 gate. |

Everything that needs the database reads its address from the environment, so the code does not care where Postgres runs. The Docker files need a machine with a Docker-compatible engine that provides `docker compose`.

An API key is needed for one thing only: running the title-picking baseline. Search, the tests and rebuilding the report need none.

Stage 4 adds what a finished service needs: the API as a long-running service with its own health check, the model built into the image, a non-root user, and entry points for the MCP server and the agent.

## 5. Stage 2: MCP server (outline)

The Model Context Protocol (MCP) is a standard way to offer tools to AI assistants. This stage wraps the stage 1 search as three tools:

| Tool | Does |
| --- | --- |
| `search_guides` | The same search as `GET /search` |
| `read_guide` | One guide's passages, by slug |
| `list_guides` | Id, slug and title of every guide |

- Built with FastMCP, the standalone `fastmcp` package, pinned to an exact version.
- It runs over stdio: the assistant starts the server as a child process on the same machine and exchanges messages through its standard input and output, so there is no network port.
- All three tools are read-only. They are marked as such for the assistant, and they only ever read from the database.
- The tool logic is plain functions shared with the HTTP API. The MCP layer is a thin wrapper.
- Tests call the tools in-process, with no separate server to start.

## 6. Stage 3: answer agent (outline)

A small agent that answers a question from the guides or declines. It is built with LangGraph, a library for writing a program as named steps (nodes) with rules for which step runs next (edges). It is used so that the loop below is explicit in code and each step can be tested alone.

```
search ──> draft ──> check ──┬── passes ───────────────────────────> done
                      ^      ├── hand_over, or fails again ────────> hand over
                      │      └── fails, first time ──> rewrite ──┐
                      └──────────────────────────────────────────┘
```

- **Draft.** Claude returns a structured draft: a decision (`answer`, `clarify` or `hand_over`), the reply text, and the ids of the guides it relied on. If search returned `no_match` the draft can only clarify or hand over, because there is nothing to cite.
- **Check.** Plain code, no model.
  - A cited passage is a passage that search returned for a guide the draft cites, not the rest of that guide.
  - An `answer` passes only if it cites at least one guide, cites only guides that search returned, and every number and every link in it is supported.
  - A number is supported if it appears in a cited passage, as digits or as the matching number word. Digits that only number the steps of a list in the reply are not checked.
  - A link is supported if it is the address of a cited guide, or appears in a cited passage once relative links are resolved against the help-centre address.
  - A `clarify` passes if it cites nothing and contains no number and no link.
  - The check does not test whether each sentence is supported by the guides.
- **Rewrite.** A failed draft gets one rewrite, and the model is told the reason. A second failure becomes a hand-over.
- **Hand over.** In this project a hand-over is only a returned decision: the agent gives no answer and says a person is needed. Nothing is sent to anyone.
- **Token use** is recorded for every model call: input and output tokens, cache-write and cache-read tokens (the API can store a repeated opening of a prompt and reuse it at a lower price, and it reports both counts), the model that served the call, and what the call was for.
- The nodes call the Anthropic Python SDK directly. The model id is configuration, and current ids and API usage are checked before this stage is written.
- Offered as a command-line tool and as `POST /answer`.

## 7. Stage 4: packaging (outline)

- Finished `Dockerfile` and Docker Compose file (service plus Postgres).
- GitHub Actions: unit tests; integration tests against a Postgres service container; and the evaluation as a regression gate. The gate runs the three search modes with the committed threshold and compares per-question outcomes (refused or not; expected guide in the top 3 or not) with the committed `evaluation/results.json`. It fails if more than one question in a mode differs, and prints the questions that differ. One question of slack is allowed because embeddings differ in their last decimal places from one machine to another. Similarity values, times and title-picking results are not compared.
- A README that states the measured results and repeats the guide-text notice from section 4.1.
- `docs/WALKTHROUGH.md`, which follows one question through the code for someone learning it.

## 8. Out of scope

Messaging channels, customer data, production databases, multi-agent systems, fine-tuning, a user interface, rerankers and hosting.
