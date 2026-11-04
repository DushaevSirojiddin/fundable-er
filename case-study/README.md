# Case-study pack

You get a database of organizations (companies and investors) and people, plus incoming records that carry only a
name and some text. For each record, decide whether it is already in the database or new, and find links for the new
ones. Then we run your code on a holdout input you have not seen.

| file (per type: `organizations`, `people`) | what it is |
|---|---|
| `existing_<type>.csv` | The database: `id, name, description` and the record's links. |
| `input_<type>.csv` | Records to resolve: `input_id, name` plus a `description` (organizations) or a `source_article` (people: a ~1,500-character excerpt of a funding-round article that mentions them). |
| `submission_template_<type>.csv` | Every `input_id` with blank answers. Fill it in as `submission_<type>.csv`. |
| `answer_key_<type>.csv` | The answers, in submission format. |

## For each input row

1. **Match.** If the entity is in `existing_<type>.csv`, put its `id` in `match_id`; otherwise `NEW`.
2. **Dedupe.** The same new entity can appear more than once in an input file. Link its rows with `dup_of`, the
   `input_id` of another row in the group. Rows are grouped by following these links, so any linking that joins the
   whole group scores the same. The key leaves the group's first row blank and points the others at it. Matched rows
   need no `dup_of`.
3. **Links.** For a `NEW` entity, fill `domain, linkedin, crunchbase, x` (people have no `domain`), formatted as in
   `existing_<type>.csv`: full URLs, a bare domain. Each link is **the entity's own current canonical page** (for a
   renamed entity, the current brand's), never a parent's, acquirer's or founder's. If it has none, or you are not
   sure, leave it blank. Fill them on every row of a `NEW` group, twins included. Links on matched rows are ignored.

## Deliver

The time box is 8 hours.

- **Code** with one entry command that reads a database folder and an input folder and writes both
  `submission_<type>.csv` files, e.g. `uv run solve.py --db <dir> --input <dir> --out <dir>`.
  - A `pyproject.toml` with `requires-python = ">=3.12"` and your dependencies. We run it with `uv run`.
  - Keys come only from the env vars below. A run over this input must finish within 10 minutes.
  - Retry with backoff on rate limits and server errors (429, 5xx); one failed call must not stop the run.
  - Cache every search and LLM response from the first call; you may ship caches or precomputed embeddings. Track
    your own spend from the responses' usage fields.
  - The holdout folder uses the same file names (`input_<type>.csv`) and columns and runs against the same
    `existing_*.csv`, but has different `input_id`s; don't rely on their prefixes or row order.
- **Your submissions** for this input.
- **A write-up**, two pages at most:
  - 1 page: Directory map and explanation of process
  - 1 page: Production system design - how you would run this on a continuous stream of inbound name + text records for
    organizations and people at volume, covering matching and dedupe, enrichment, cost and latency, human review,
    and how corrections feed back in.

Using AI is heavily encouraged, but you are responsible for every line of code and for explaining how it works.

## Search and models

| service | env var | budget |
|---|---|---|
| [OpenRouter](https://openrouter.ai) | `OPENROUTER_API_KEY` | $20 on the key we send you |
| [Serper](https://serper.dev) (Google results) | `SERPER_API_KEY` | 2,500 queries: sign up for the free tier yourself (no card) |

On OpenRouter, prefer OpenAI models, or another model with high rate limits such as open-weight DeepSeek, Qwen or
Kimi. OpenRouter's web search is allowed and comes out of the same $20. On the holdout run we set our own keys in the
same env vars.

## Rules

- **Allowed:** Serper, OpenRouter (web search included), and plain HTTP fetches of public pages such as an
  entity's own website.
- **Not allowed:** logged-in or automated scraping of LinkedIn, Crunchbase or X; paid enrichment APIs; any other
  search or LLM provider; labelling rows by hand.
- **The answer key** is for grading yourself and studying your errors. Don't put its values into code, prompts or
  caches.

## Scoring

**Decision, per row:** +1 for the right existing `id`, or `NEW` with exactly the right `dup_of` group. −1 for
anything else:

- **duplicate:** `NEW` for an entity in the database, or a duplicate in input left unlinked;
- **wrong merge:** the wrong `id`, an `id` for a new entity, or rows grouped that aren't one entity;
- **invalid:** a blank or missing row, an unknown `match_id`, or a `dup_of` that isn't an `input_id`.

**Links**, only on rows that are new and that you called `NEW`. With k the number of links the entity has in the key
(at least 1): each correct link +1/k, each wrong link (or a link where the entity has none) −1/k, a blank 0, capped
at ±1 per row. Before comparing, case, `https://`, `www.`, a trailing `/`, query strings and `#fragments` are
ignored, URL-encoding is decoded, and `twitter.com` counts as `x.com`.

**Score** = points ÷ (one per row + one per new row). It can go below zero.

*Example.* A new entity with 4 links in the key: `NEW` with 3 right links and 1 wrong scores +1 + (3 − 1)/4 = +1.5
of 2. Matching it to an existing record scores −1, and its links are not scored.

## Grade yourself

```
python3 grade.py path/to/your_submissions/
```

It needs only Python 3.9+, prints per-type scores and error counts, and writes `grade_report.md` (every error by
cause, yours vs correct) and `grade_errors.csv` (one error per row, handy to give an LLM) to the folder, or `--out DIR`.
