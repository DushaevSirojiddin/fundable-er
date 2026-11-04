"""Matching (input row -> DB id or NEW) and dedupe of NEW rows within the input."""
from __future__ import annotations

import asyncio
import re

import numpy as np
from rapidfuzz import fuzz

from . import prompts as P
from .llm import LLM
from .model import Cand, Row
from .normalize import canon_domain, compact_key, fold, group_ids, name_key, name_tokens, person_key
from .search import Search, render_hits


def render_sites(row: Row) -> str:
    if not row.sites:
        return ""
    lines = [f"- {d} ({p['source']}): final host {p.get('final_host', '')}; title \"{p.get('title', '')[:120]}\"; "
             f"{p.get('description', '')[:160]}" for d, p in row.sites.items()]
    return "\n\nWebsites fetched for the input (check the title fits the input before trusting a site):\n" + "\n".join(lines)

STOP = {"a", "an", "the", "and", "of", "for", "with", "that", "which", "is", "its", "in", "on", "to", "by", "from",
        "whose", "company", "platform", "startup", "firm", "provider", "based", "developing", "builds", "offers"}


def org_query(row: Row) -> str:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", row.text) if w.lower() not in STOP][:6]
    return f"{row.name} {' '.join(words)}".strip()


def person_query(row: Row) -> str:
    org = row.org_hint
    return f'"{row.name}" {org}'.strip()


def article_window(row: Row, width: int = 1400) -> str:
    return row.text[:width]


# ---------------------------------------------------------------------------- evidence


async def person_context(llm: LLM, row: Row) -> None:
    user = f"Person: {row.name}\n\nExcerpt:\n{article_window(row, 1600)}"
    out = await llm.json(P.PERSON_CONTEXT_SYSTEM, user, P.PERSON_CONTEXT_SCHEMA, "person_context")
    row.context = out or {}


GENERIC_TOKENS = {"capital", "ventures", "venture", "partners", "partner", "group", "holdings", "holding", "the",
                  "fund", "funds", "management", "investments", "investment", "labs", "lab", "technologies",
                  "technology", "tech", "global", "international", "company", "and", "of", "inc", "ltd", "llc", "co",
                  "vc", "ai", "io", "app", "hq", "bio", "health", "solutions", "systems", "services"}


def distinctive_tokens(names: list[str]) -> set[str]:
    return {t for n in names for t in name_tokens(n) if t not in GENERIC_TOKENS and len(t) >= 3}


def page_names_entity(page: dict, names: list[str]) -> bool:
    hay = re.sub(r"[^a-z0-9]", "", fold(" ".join([page.get("final_host", ""), page.get("title", ""),
                                                   page.get("description", "")[:200]])))
    toks = distinctive_tokens(names) or {compact_key(n) for n in names if compact_key(n)}
    return any(t in hay for t in toks) or any(compact_key(n) and compact_key(n) in hay for n in names)


async def check_sites(search: Search, row: Row, domains: list[tuple[str, str]], limit: int = 5) -> None:
    """Fetch candidate homepages (plain HTTP, the entity's own site) and record whether they name the entity."""
    todo, seen = [], set(row.sites)
    for d, src in domains:
        d = canon_domain(d)
        if d and d not in seen:
            seen.add(d)
            todo.append((d, src))
    todo = todo[:limit]
    pages = await asyncio.gather(*(search.homepage(d) for d, _ in todo))
    for (d, src), page in zip(todo, pages):
        if page:
            row.sites[d] = {**page, "source": src, "name_match": page_names_entity(page, row.all_names)}


def result_domains(row: Row, limit: int = 3) -> list[tuple[str, str]]:
    out = []
    if row.evidence.kg_website:
        out.append((row.evidence.kg_website, "Google knowledge panel"))
    for pos, h in enumerate(row.evidence.hits, 1):
        if canon_domain(h.link) and "site:" not in h.query:
            out.append((h.link, f"search result #{pos}"))
    return out[:limit + 1]


async def org_profile(llm: LLM, row: Row) -> None:
    out = await llm.json(P.ORG_PROFILE_SYSTEM, f"name: {row.name}\ndescription: {row.text}",
                         P.ORG_PROFILE_SCHEMA, "org_profile")
    row.profile = out or {}


async def gather_evidence(search: Search, llm: LLM, row: Row) -> None:
    if row.entity == "people":
        await person_context(llm, row)
        ev = await search.google(person_query(row))
        if ev is not None and not ev.hits and row.org_hint:
            ev = await search.google(f'"{row.name}"')
        if ev:
            row.evidence.extend(ev)
        return
    await org_profile(llm, row)
    ev = await search.google(row.profile.get("search_query") or org_query(row))
    if ev:
        row.evidence.extend(ev)
    guesses = [(d, "model's guess (unverified)") for d in row.profile.get("official_domains", [])[:3]]
    await check_sites(search, row, guesses + result_domains(row))


# ---------------------------------------------------------------------------- matching


def render_cands(cands: list[Cand]) -> str:
    out = []
    for c in cands:
        links = " | ".join(f"{k}: {v}" for k, v in c.links.items() if v)
        out.append(f"[{c.id}] {c.name}\n   links: {links or '-'}\n   description: {c.description[:700]}\n"
                   f"   why it is a candidate: {'; '.join(c.signals)}")
    return "\n".join(out)


def render_input(row: Row) -> str:
    if row.entity == "organizations":
        return f"INPUT organization\nname: {row.name}\ndescription: {row.text}"
    ctx = row.context
    facts = ""
    if ctx:
        facts = (f"\nfrom the article: role={ctx.get('role', '')}; organization={ctx.get('organization', '')}; "
                 f"other organizations={', '.join(ctx.get('other_organizations', []))}; "
                 f"location={ctx.get('location', '')}")
    return f"INPUT person\nname: {row.name}{facts}\narticle excerpt:\n{article_window(row)}"


def name_related(name: str, c: Cand) -> bool:
    return (fuzz.WRatio(name_key(name), name_key(c.name)) >= 80
            or any(s.startswith(("input name matches", "same name", "similar name", "name contains"))
                   for s in c.signals))


def exact_name(row: Row, c: Cand) -> bool:
    if row.entity == "people":
        return person_key(c.name) == person_key(row.name)
    return compact_key(c.name) == compact_key(row.name)


def fallback_decision(row: Row) -> None:
    """Used when the LLM is unavailable: deterministic keys only, conservative."""
    hard = [c for c in row.cands if c.hard_key and fuzz.WRatio(name_key(row.name), name_key(c.name)) >= 85]
    exact = [c for c in row.cands if exact_name(row, c)]
    if len(hard) == 1:
        row.match_id, row.confidence, row.decided_by = hard[0].id, 0.8, "rule:hard-key"
    elif len(exact) == 1:
        row.match_id, row.confidence, row.decided_by = exact[0].id, 0.6, "rule:unique-exact-name"
    else:
        row.match_id, row.confidence, row.decided_by = "NEW", 0.5, "rule:no-unique-candidate"


async def adjudicate(llm: LLM, row: Row) -> None:
    if not row.cands:
        row.match_id, row.confidence, row.decided_by, row.reason = "NEW", 0.9, "rule:no-candidates", ""
        return
    system = P.MATCH_ORG_SYSTEM if row.entity == "organizations" else P.MATCH_PERSON_SYSTEM
    user = (f"{render_input(row)}\n\nWeb search results for the input:\n{render_hits(row.evidence)}{render_sites(row)}\n\n"
            f"CANDIDATES from the database:\n{render_cands(row.cands)}\n\n"
            "Answer with the matching candidate id, or NEW.")
    by_id = {c.id: c for c in row.cands}
    for strong in (False, True):
        out = await llm.json(system, user, P.MATCH_SCHEMA, "match", strong=strong)
        if not out:
            continue
        mid = out["match_id"].strip()
        mid = "NEW" if mid.upper() == "NEW" or mid not in by_id else mid
        row.match_id, row.confidence = mid, float(out["confidence"])
        row.current_name, row.reason = out["current_name"].strip(), out["reason"]
        row.decided_by = "llm:strong" if strong else "llm:fast"
        # the two riskiest calls always get a second opinion from the strong model:
        #  - a match across unrelated names with no hard key (possible competitor merge)
        #  - NEW although a candidate has the same name (possible missed match)
        if mid != "NEW":
            risky = not by_id[mid].hard_key and not name_related(row.name, by_id[mid])
        else:
            risky = any(exact_name(row, c) for c in row.cands)
        if row.confidence >= llm.s.escalate_below and not (risky and not strong):
            return
    if not row.decided_by:
        fallback_decision(row)


# ---------------------------------------------------------------------------- dedupe


def top_domain(row: Row) -> str:
    if row.evidence.kg_website:
        return canon_domain(row.evidence.kg_website)
    for h in row.evidence.hits[:3]:
        d = canon_domain(h.link)
        if d:
            return d
    return ""


def pair_candidates(rows: list[Row], emb: np.ndarray | None) -> list[tuple[int, int, str]]:
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            why = ""
            if a.entity == "people":
                if person_key(a.name) == person_key(b.name) or fuzz.WRatio(name_key(a.name), name_key(b.name)) >= 92:
                    why = "similar names"
            else:
                if compact_key(a.name) == compact_key(b.name) or fuzz.WRatio(name_key(a.name), name_key(b.name)) >= 90:
                    why = "similar names"
                elif emb is not None and float(emb[i] @ emb[j]) >= 0.80:
                    why = f"similar descriptions ({float(emb[i] @ emb[j]):.2f})"
                elif top_domain(a) and top_domain(a) == top_domain(b):
                    why = f"same top web domain {top_domain(a)}"
            if why:
                pairs.append((i, j, why))
    return pairs


async def same_entity(llm: LLM, a: Row, b: Row, why: str) -> bool:
    kind = "organization" if a.entity == "organizations" else "person"

    def show(r: Row, tag: str) -> str:
        return f"{tag}\n{render_input(r)}\nTop web results:\n{render_hits(r.evidence, limit=3, snippet_len=160)}"

    out = await llm.json(P.SAME_SYSTEM.format(kind=kind), f"{show(a, 'RECORD A')}\n\n{show(b, 'RECORD B')}\n\n"
                         f"(paired because: {why})", P.SAME_SCHEMA, "same_entity")
    if out is None:   # fallback: identical name key and same top domain
        return name_key(a.name) == name_key(b.name) and bool(top_domain(a)) and top_domain(a) == top_domain(b)
    return bool(out["same"]) and float(out["confidence"]) >= 0.6


async def dedupe(llm: LLM, rows: list[Row], emb: np.ndarray | None) -> list[dict]:
    """Links twin rows. Twins of a matched row inherit its match; NEW-only groups get dup_of = first row."""
    pairs = pair_candidates(rows, emb)
    verdicts = await asyncio.gather(*(same_entity(llm, rows[i], rows[j], why) for i, j, why in pairs))
    edges = [(rows[i].iid, rows[j].iid) for (i, j, _), ok in zip(pairs, verdicts) if ok]
    log = [{"a": rows[i].iid, "b": rows[j].iid, "why": why, "same": ok} for (i, j, why), ok in zip(pairs, verdicts)]
    by_iid = {r.iid: r for r in rows}
    for rep, members in group_ids([r.iid for r in rows], edges).items():
        if len(members) < 2:
            continue
        matched = {by_iid[m].match_id for m in members if by_iid[m].match_id != "NEW"}
        if len(matched) == 1:
            mid = matched.pop()
            for m in members:
                if by_iid[m].match_id == "NEW":
                    by_iid[m].match_id, by_iid[m].decided_by = mid, by_iid[m].decided_by + "+twin-of-matched"
        elif not matched:
            for m in members[1:]:
                by_iid[m].dup_of = rep
        # conflicting matches inside one group: keep each row's own decision
    return log
