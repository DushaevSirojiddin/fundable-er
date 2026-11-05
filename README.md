# Fundable case study: entity resolution and link enrichment

For each input organization or person, the solver decides whether it is already in the database or `NEW`, links duplicate `NEW` rows to each other, and finds the website, LinkedIn, Crunchbase and X links of new entities.

The design and the production plan are in **[writeup.pdf](writeup.pdf)** (source: [WRITEUP.md](WRITEUP.md)).

## Run

```bash
cp .env.example .env        # set OPENROUTER_API_KEY and SERPER_API_KEY
uv run solve.py --db case-study --input case-study --out submissions
python3 case-study/grade.py submissions
```

- `--db` is a folder with `existing_{organizations,people}.csv`, and `--input` is a folder with `input_{organizations,people}.csv`. The solver reads nothing else; in particular it never reads the answer key.
- Output: `submission_organizations.csv`, `submission_people.csv` and `run_report.json`. The report holds every decision with its confidence, reason, candidates and link options, plus timings, cache hits, retries, failures and spend.
- `cache/cache.sqlite` holds every search, LLM and fetch response behind the shipped submission, including homepage fetches that failed for good. With it, the visible set reruns in under a second at $0, makes no live calls and reproduces the CSVs exactly. On new input (the holdout) the cache misses and calls go live.

| Run | Time | OpenRouter | Serper queries |
|---|---|---|---|
| Visible set, shipped cache | < 1 s | $0 | 0 |
| Cold cache, 160 rows | ~170 s | ~$0.25 | ~560 |

The limit is 10 minutes. A soft deadline at 540 s makes the run finish with what it has, and it always writes valid CSVs.

## Results (visible set, `submissions/`)

| | score | decisions | link precision / recall |
|---|---|---|---|
| organizations | 97.1 % | 107/107 | domain 98/94, LinkedIn 100/98, Crunchbase 95/86, X 97/97 |
| people | 92.4 % | 52/53 | LinkedIn 100/96, Crunchbase 100/93, X 100/73 |
| **overall** | **95.5 %** | 159/160 | |

**Holdout simulation.** `scripts/holdout_sim.py` builds a copy of the input with new ids and shuffled rows. A run of the final code on it with an empty cache and 10 % of API calls failing on purpose (`FAULT_RATE=0.1`) scored **95.6 %** (organizations 95.6 %, people 95.7 %, 160/160 decisions) in 172 s for $0.25 and 559 Serper queries. Every injected OpenRouter and Serper failure was recovered by retry.

LLM output is not fully deterministic, so cold runs vary by a row or two. Earlier cold runs scored 96.3 % and 93.8 %. The 93.8 % run merged a same-name investor with a different firm. That led to the last escalation rule below. Replaying that run's evidence with the rule scored 95.1 %.

```bash
python3 scripts/holdout_sim.py case-study scratch/holdout_sim
CACHE_PATH=scratch/empty.sqlite FAULT_RATE=0.1 uv run solve.py --db scratch/holdout_sim --input scratch/holdout_sim --out scratch/hs_out
python3 case-study/grade.py scratch/hs_out --key scratch/holdout_sim --holdout
```

## Configuration (environment variables)

| var | default | meaning |
|---|---|---|
| `OPENROUTER_API_KEY`, `SERPER_API_KEY` | required | the only credentials; read from the environment (a local `.env` is loaded if present) |
| `MODEL_FAST` / `MODEL_STRONG` | `openai/gpt-5.6-luna` / `openai/gpt-5.4` | bulk model, and the model used for escalations |
| `LLM_BUDGET_USD` | 5 | hard spend cap per run; beyond it the LLM steps fall back to rules |
| `DEADLINE_SECONDS` | 540 | soft deadline |
| `LINK_THRESHOLD` / `X_SEARCH_THRESHOLD` | 0.6 / 0.95 | minimum picker confidence for a link, and the stricter bar for X handles seen only in search |
| `CACHE_PATH` | `cache/cache.sqlite` | response cache |
| `FAULT_RATE` | 0 | fraction of live calls to fail on purpose, for resilience testing |

`uv run solve.py ... --prune-cache` drops cache entries the run did not use. `--no-cache` bypasses the cache entirely.

## Spend

- **Development, total:** $1.50 of the $20 OpenRouter budget and about 2,400 Serper queries.
- **One cold run of 160 rows:** about $0.25 and 560 queries.
- Spend is tracked from OpenRouter's `usage.cost` and written to `run_report.json`.

## Layout

```
solve.py              entry point (argparse, orchestration, CSV/report writing)
er/normalize.py       name keys, URL canonicalizers matching the grader's rules, DB link index, union-find
er/clients.py         SQLite cache, spend ledger, retrying HTTP caller (429/5xx/timeouts, Retry-After, deadline)
er/llm.py             OpenRouter chat (strict JSON schema) and embeddings
er/search.py          Serper search; plain-HTTP fetch of an entity's own homepage
er/candidates.py      blocking: name/nickname/slug keys, fuzzy, embeddings, hard link keys, website conflicts
er/resolve.py         evidence gathering, LLM adjudication with escalation, dedupe inside the input
er/links.py           link search and selection for NEW entities
er/prompts.py         prompts and JSON schemas
tests/                pytest (uv run pytest)
scripts/holdout_sim.py
case-study/           the pack as received (grade.py, inputs, DB, answer keys; only grade.py reads the keys)
```

## Rules compliance

- **Allowed sources only:** OpenRouter (OpenAI models), Serper, and plain HTTP fetches of an entity's own homepage. LinkedIn, Crunchbase and X are never fetched; their URLs are only read from search results or from links on the entity's own site.
- **No answer-key data in code, prompts or caches.** The key was used only through `grade.py`, to study error classes.
- **Keys come only from environment variables.** `.env` is gitignored.

## With more time

- People X recall (73 %). Many handles are not indexed for name + site queries. The next step would be to look for handles on personal sites and company team pages.
- A small labelled set of hard namesake cases, to calibrate the escalation band with data rather than by hand.
- Repeated sampling at the strong tier, to reduce run-to-run variance on borderline people matches.
