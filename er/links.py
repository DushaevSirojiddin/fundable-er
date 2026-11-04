"""Link enrichment for NEW entities: targeted searches + homepage links -> LLM picks among real URLs only."""
from __future__ import annotations

import asyncio

from . import prompts as P
from .llm import LLM
from .model import Row
from .normalize import LinkIndex, canon_domain, canon_link, compare_key, link_fields
from .resolve import render_input
from .search import Evidence, Search


def person_q(name: str) -> str:
    """First name loose (nicknames: Fred/Freddy), last name exact."""
    toks = name.split()
    return f'{" ".join(toks[:-1])} "{toks[-1]}"' if len(toks) > 1 else f'"{name}"'


def site_queries(row: Row, anchor: str) -> list[str]:
    n = row.search_name
    if row.entity == "organizations":
        qs = [f"{n} site:linkedin.com/company", f"{n} site:crunchbase.com/organization",
              f"{n} (site:x.com OR site:twitter.com)"]
        if anchor:   # profile pages show the website, so the domain disambiguates namesakes
            qs += [f'{n} "{anchor}" site:linkedin.com/company', f'{n} "{anchor}" site:crunchbase.com/organization',
                   f'{n} "{anchor}" (site:x.com OR site:twitter.com)']
        return qs
    pn, org = person_q(row.name), row.org_hint
    return [f"{pn} {org} site:linkedin.com/in", f"{pn} {org} site:crunchbase.com/person",
            f"{pn} site:crunchbase.com/person", f"{pn} {org} (site:x.com OR site:twitter.com)",
            f"{pn} (site:x.com OR site:twitter.com)"]


def anchor_domain(rows: list[Row]) -> str:
    """Best verified website for the group: a fetched homepage that names the entity, search-found first."""
    ranked = []
    for r in rows:
        for d, p in r.sites.items():
            if p.get("name_match"):
                ranked.append((p["source"].startswith("model"), canon_domain(p.get("final_host") or d) or d))
    return sorted(ranked)[0][1] if ranked else ""


async def enrich_group(search: Search, llm: LLM, rows: list[Row], index: LinkIndex) -> None:
    """rows = one NEW group (twins). Picks links once and copies them to every row."""
    lead = max(rows, key=lambda r: len(r.text))
    ev = Evidence()
    for r in rows:
        ev.extend(r.evidence)
    anchor = anchor_domain(rows) if lead.entity == "organizations" else ""
    results = await asyncio.gather(*(search.google(q) for q in site_queries(lead, anchor)))
    for r in results:
        ev.extend(r)

    sites: dict[str, dict] = {}
    for r in rows:
        sites.update(r.sites)

    fields = link_fields(lead.entity)
    options: dict[str, list[tuple[str, str]]] = {f: [] for f in fields}   # field -> [(url, context)]

    def offer(field: str, url: str, context: str):
        c = canon_link(field, url, lead.entity)
        if not c or index.lookup(field, c):     # a link owned by an existing DB entity cannot be this NEW entity's
            return
        if all(compare_key(field, c) != compare_key(field, u) for u, _ in options[field]) and len(options[field]) < 10:
            options[field].append((c, context))

    # the entity's own (fetched) websites first: verified ones, then the rest
    for d, p in sorted(sites.items(), key=lambda kv: not kv[1].get("name_match")):
        tag = f'fetched homepage {d} ({p["source"]}): final host {p.get("final_host", "")}; title "{p.get("title", "")[:100]}"; {p.get("description", "")[:140]}'
        if "domain" in options:
            offer("domain", p.get("final_host") or d, tag)
        for link in p.get("links", []):
            for f in fields:
                if f != "domain":
                    offer(f, link, f'linked from the homepage of {d} (title "{p.get("title", "")[:80]}")')
    if ev.kg_website and "domain" in options:
        offer("domain", ev.kg_website, f"Google knowledge panel website ({ev.kg_title})")
    for h in ev.hits:
        for f in fields:
            offer(f, h.link, f'search result for [{h.query}]: "{h.title}" - {h.snippet[:160]}')

    for r in rows:
        r.link_options = {f: [u for u, _ in options[f]] for f in fields}
    if not any(options.values()):
        return
    kind = "organization" if lead.entity == "organizations" else "person"
    blocks = []
    for f in fields:
        lines = [f"  {i}. {u}  <- {ctx}" for i, (u, ctx) in enumerate(options[f])] or ["  (no options)"]
        blocks.append(f"{f} options:\n" + "\n".join(lines))
    names = ", ".join(dict.fromkeys(n for r in rows for n in r.all_names))
    user = f"{render_input(lead)}\nAll names seen for this entity: {names}\n\n" + "\n\n".join(blocks)
    out = await llm.json(P.LINK_SYSTEM.format(kind=kind), user, P.link_schema(fields), "pick_links")
    if not out:
        return
    picked, conf = {}, {}
    for f in fields:
        choice, c = out[f]["choice"], float(out[f]["confidence"])
        if 0 <= choice < len(options[f]):
            conf[f] = c
            url, context = options[f][choice]
            # calibration: X handles seen only in search results were wrong most of the time below 0.95,
            # handles linked from the entity's own website are reliable
            bar = llm.s.link_threshold
            if f == "x" and not context.startswith("linked from the homepage"):
                bar = max(bar, llm.s.x_search_threshold)
            if c >= bar:
                picked[f] = canon_domain(url) if f == "domain" else url
    for r in rows:
        r.links, r.link_conf = dict(picked), dict(conf)
