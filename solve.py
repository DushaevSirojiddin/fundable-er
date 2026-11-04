"""Resolve input organizations and people against the database and write the submission CSVs.

    uv run solve.py --db case-study --input case-study --out submissions

Reads only existing_<type>.csv from --db and input_<type>.csv from --input. Keys come from OPENROUTER_API_KEY and
SERPER_API_KEY (a local .env is loaded if present). Writes submission_<type>.csv and run_report.json to --out.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from er.candidates import Blocker
from er.clients import Cache, Caller
from er.config import Settings
from er.links import enrich_group
from er.llm import LLM
from er.model import Row
from er.normalize import link_fields
from er.resolve import adjudicate, dedupe, fallback_decision, gather_evidence
from er.search import Search

TYPES = ["organizations", "people"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def log(s: Settings, msg: str) -> None:
    print(f"[{s.elapsed():6.1f}s] {msg}", file=sys.stderr, flush=True)


async def gather_limited(coros, s: Settings):
    """Run coroutines concurrently; any exception degrades only its own item."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            log(s, f"warning: task failed: {type(r).__name__}: {r}")
    return results


async def resolve_type(entity: str, db: list[dict], inputs: list[dict], llm: LLM, search: Search, s: Settings):
    text_col = "description" if entity == "organizations" else "source_article"
    rows = [Row(r["input_id"], r["name"], r.get(text_col, ""), entity) for r in inputs]

    # 1. evidence (people: extract article context first, then search)
    await gather_limited([gather_evidence(search, llm, r) for r in rows], s)
    log(s, f"{entity}: evidence gathered for {len(rows)} rows")

    # 2. blocking
    db_emb = in_emb = None
    if entity == "organizations":
        db_emb = await llm.embed([r["description"][:2000] or r["name"] for r in db])
        in_emb = await llm.embed([r.text[:2000] or r.name for r in rows])
        if db_emb is None or in_emb is None:
            db_emb = in_emb = None
            log(s, "warning: embeddings unavailable, name/link blocking only")
    blocker = Blocker(db, entity, db_emb)
    for i, r in enumerate(rows):
        r.cands = blocker.candidates(r, in_emb[i] if in_emb is not None else None)

    # 3. adjudication
    await gather_limited([adjudicate(llm, r) for r in rows], s)
    for r in rows:
        if not r.match_id:
            fallback_decision(r)
    log(s, f"{entity}: matched {sum(r.match_id != 'NEW' for r in rows)}, NEW {sum(r.match_id == 'NEW' for r in rows)}")

    # 4. dedupe twins inside the input
    dedupe_log = await dedupe(llm, rows, in_emb)
    log(s, f"{entity}: dedupe checked {len(dedupe_log)} pairs, {sum(p['same'] for p in dedupe_log)} confirmed")

    # 5. links for each NEW group, once per group
    groups: dict[str, list[Row]] = {}
    for r in rows:
        if r.match_id == "NEW":
            groups.setdefault(r.dup_of or r.iid, []).append(r)
    if s.time_left() > 20:
        await gather_limited([enrich_group(search, llm, g, blocker.index) for g in groups.values()], s)
    log(s, f"{entity}: links filled for {sum(bool(r.links) for r in rows if r.match_id == 'NEW')} NEW rows")
    return rows, dedupe_log


def write_submission(path: Path, entity: str, rows: list[Row]) -> None:
    fields = ["input_id", "match_id", "dup_of"] + link_fields(entity)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in rows:
            new = r.match_id == "NEW"
            w.writerow({"input_id": r.iid, "match_id": r.match_id or "NEW", "dup_of": r.dup_of if new else "",
                        **{f: (r.links.get(f, "") if new else "") for f in link_fields(entity)}})


async def main_async(args) -> int:
    s = Settings.from_env()
    if args.deadline:
        s.deadline_s = args.deadline
    if not s.openrouter_key:
        log(s, "warning: OPENROUTER_API_KEY not set - rules-only fallback")
    if not s.serper_key:
        log(s, "warning: SERPER_API_KEY not set - no web evidence, links stay blank")
    cache = None if args.no_cache else Cache(s.cache_path)
    caller = Caller(s, cache)
    llm, search = LLM(caller), Search(caller)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"settings": {"model_fast": s.model_fast, "model_strong": s.model_strong, "embed_model": s.embed_model,
                           "link_threshold": s.link_threshold, "deadline_s": s.deadline_s}, "types": {}}

    async def run_one(entity: str) -> None:
        db = read_csv(Path(args.db) / f"existing_{entity}.csv")
        inputs = read_csv(Path(args.input) / f"input_{entity}.csv")
        t0 = time.monotonic()
        try:
            rows, dlog = await resolve_type(entity, db, inputs, llm, search, s)
        except Exception as e:                       # last line of defence: always write a valid file
            log(s, f"error: {entity} pipeline failed ({type(e).__name__}: {e}); writing NEW for all rows")
            rows, dlog = [Row(r["input_id"], r["name"], "", entity, match_id="NEW") for r in inputs], []
        write_submission(out / f"submission_{entity}.csv", entity, rows)
        report["types"][entity] = {
            "seconds": round(time.monotonic() - t0, 1),
            "rows": len(rows), "matched": sum(r.match_id != "NEW" for r in rows),
            "new": sum(r.match_id == "NEW" for r in rows),
            "decided_by": {k: sum(r.decided_by == k for r in rows) for k in sorted({r.decided_by for r in rows})},
            "dedupe_pairs": dlog,
            "decisions": [{"input_id": r.iid, "name": r.name, "match_id": r.match_id, "dup_of": r.dup_of,
                           "confidence": round(r.confidence, 3), "decided_by": r.decided_by,
                           "current_name": r.current_name, "reason": r.reason,
                           "candidates": [c.id for c in r.cands], "links": r.links,
                           "link_confidence": r.link_conf, "link_options": r.link_options} for r in rows],
        }

    try:
        await asyncio.gather(*(run_one(e) for e in args.types))
    finally:
        await caller.aclose()
        if cache:
            if args.prune_cache:
                log(s, f"pruned {cache.prune()} stale cache entries")
            cache.close()
    report["seconds_total"] = round(s.elapsed(), 1)
    report["spend"] = caller.ledger.as_dict()
    (out / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    sp = report["spend"]
    log(s, f"done in {report['seconds_total']}s | OpenRouter live ${sp['openrouter_cost_usd_live']:.4f} "
           f"(uncached ${sp['openrouter_cost_usd_if_uncached']:.4f}) | Serper live {sp['serper_queries_live']} | "
           f"cache hits {sp['cache_hits']} | failures {sp['failures']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, help="folder with existing_organizations.csv and existing_people.csv")
    ap.add_argument("--input", required=True, help="folder with input_organizations.csv and input_people.csv")
    ap.add_argument("--out", required=True, help="where to write submission_*.csv and run_report.json")
    ap.add_argument("--types", nargs="+", default=TYPES, choices=TYPES)
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the response cache")
    ap.add_argument("--deadline", type=float, help="soft deadline in seconds (default 540)")
    ap.add_argument("--prune-cache", action="store_true", help="drop cache entries this run did not use")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
