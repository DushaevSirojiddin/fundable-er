"""Link enrichment for NEW entities: targeted searches + homepage links -> LLM picks among real URLs only."""
from __future__ import annotations

import asyncio

from . import prompts as P
from .llm import LLM
from .model import Row
from .normalize import LinkIndex, canon_domain, canon_link, compare_key, link_fields
from .resolve import render_input
from .search import Evidence, Search


def site_queries(row: Row) -> list[str]:
    n = row.search_name
    if row.entity == "organizations":
        return [f"{n} site:linkedin.com/company", f"{n} site:crunchbase.com/organization",
                f"{n} (site:x.com OR site:twitter.com)"]
    org = row.org_hint
    return [f'"{n}" {org} site:linkedin.com/in', f'"{n}" site:crunchbase.com/person',
            f'"{n}" {org} (site:x.com OR site:twitter.com)']


def candidate_domains(ev: Evidence, limit: int = 3) -> list[str]:
    out = []
    for u in ([ev.kg_website] if ev.kg_website else []) + [h.link for h in ev.hits if "site:" not in h.query]:
        d = canon_domain(u)
        if d and d not in out:
            out.append(d)
    return out[:limit]


async def enrich_group(search: Search, llm: LLM, rows: list[Row], index: LinkIndex) -> None:
    """rows = one NEW group (twins). Picks links once and copies them to every row."""
    lead = max(rows, key=lambda r: len(r.text))
    ev = Evidence()
    for r in rows:
        ev.extend(r.evidence)
    results = await asyncio.gather(*(search.google(q) for q in site_queries(lead)))
    for r in results:
        ev.extend(r)

    homepages = []
    if lead.entity == "organizations":
        pages = await asyncio.gather(*(search.homepage(d) for d in candidate_domains(ev)))
        homepages = [(d, p) for d, p in zip(candidate_domains(ev), pages) if p]

    fields = link_fields(lead.entity)
    options: dict[str, list[tuple[str, str]]] = {f: [] for f in fields}   # field -> [(url, context)]

    def offer(field: str, url: str, context: str):
        c = canon_link(field, url, lead.entity)
        if not c or index.lookup(field, c):     # a link owned by an existing DB entity cannot be this NEW entity's
            return
        if all(compare_key(field, c) != compare_key(field, u) for u, _ in options[field]) and len(options[field]) < 8:
            options[field].append((c, context))

    if ev.kg_website and "domain" in options:
        offer("domain", ev.kg_website, f"Google knowledge panel website ({ev.kg_title})")
    for h in ev.hits:
        for f in fields:
            offer(f, h.link, f'search result: "{h.title}" - {h.snippet[:160]}')
    for d, p in homepages:
        if "domain" in options:
            offer("domain", p["final_host"] or d,
                  f'fetched homepage of {d}: final host {p["final_host"]}; title "{p["title"]}"; {p["description"][:160]}')
        for link in p["links"]:
            for f in fields:
                if f != "domain":
                    offer(f, link, f"linked from the homepage of {d} (title: {p['title'][:80]})")

    if not any(options.values()):
        return
    kind = "organization" if lead.entity == "organizations" else "person"
    blocks = []
    for f in fields:
        lines = [f"  {i}. {u}  <- {ctx}" for i, (u, ctx) in enumerate(options[f])] or ["  (no options)"]
        blocks.append(f"{f} options:\n" + "\n".join(lines))
    names = ", ".join(sorted({r.name for r in rows} | ({lead.current_name} if lead.current_name else set())))
    user = f"{render_input(lead)}\nAll names seen for this entity: {names}\n\n" + "\n\n".join(blocks)
    out = await llm.json(P.LINK_SYSTEM.format(kind=kind), user, P.link_schema(fields), "pick_links")
    if not out:
        return
    picked, conf = {}, {}
    for f in fields:
        choice, c = out[f]["choice"], float(out[f]["confidence"])
        if 0 <= choice < len(options[f]):
            conf[f] = c
            if c >= llm.s.link_threshold:
                url = options[f][choice][0]
                picked[f] = canon_domain(url) if f == "domain" else url
    for r in rows:
        r.links, r.link_conf = dict(picked), dict(conf)
