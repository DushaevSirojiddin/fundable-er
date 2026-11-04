"""Blocking: cheap candidate generation so the LLM only ever compares a handful of plausible DB records.

Signals (any one is enough to become a candidate):
  * name: exact / compact / nickname-aware key, rapidfuzz similarity, short-name containment
  * description embedding similarity (orgs) - catches renames where the names share nothing
  * hard keys: a URL in the input's web results equals a DB record's domain / LinkedIn / Crunchbase / X
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz

from .model import Cand, Row
from .normalize import LinkIndex, compact_key, link_fields, name_key, name_tokens, person_key


def slug_keys(r: dict, entity: str) -> set[str]:
    """Name-like keys hidden in a record's links. Crunchbase/LinkedIn slugs and X handles often keep a
    former name (a renamed company keeps its old slug), so they double as an alias list."""
    out = set()
    for f in ("crunchbase", "linkedin", "x"):
        v = (r.get(f) or "").rstrip("/")
        if not v:
            continue
        slug = v.rsplit("/", 1)[-1]
        slug = re.sub(r"-(?:[0-9a-f]{4}|\d{1,3})$", "", slug)          # crunchbase disambiguation suffixes
        k = name_key(re.sub(r"[-_.]+", " ", slug))
        if len(k.replace(" ", "")) >= 4:
            out.add(k)
    if entity == "organizations" and r.get("domain"):
        label = r["domain"].split(".")[0]
        if len(label) >= 4:
            out.add(name_key(label))
    out.discard(name_key(r["name"]))
    return out


class Blocker:
    def __init__(self, db: list[dict], entity: str, db_emb: np.ndarray | None):
        self.db = db
        self.by_id = {r["id"]: r for r in db}
        self.entity = entity
        self.emb = db_emb
        self.index = LinkIndex(db, entity)
        self.keys = [name_key(r["name"]) for r in db]
        self.compact = [compact_key(r["name"]) for r in db]
        self.pkeys = [person_key(r["name"]) for r in db]
        self.toks = [name_tokens(r["name"]) for r in db]
        self.slugs = [slug_keys(r, entity) for r in db]

    def candidates(self, row: Row, row_emb: np.ndarray | None, cap: int = 8) -> list[Cand]:
        found: dict[str, Cand] = {}

        def add(i: int | str, score: float, signal: str, hard: bool = False):
            r = self.db[i] if isinstance(i, int) else self.by_id[i]
            c = found.get(r["id"])
            if not c:
                c = found[r["id"]] = Cand(r["id"], r["name"], r["description"],
                                          {f: r.get(f, "") for f in link_fields(self.entity)})
            c.score = max(c.score, score)
            if signal not in c.signals:
                c.signals.append(signal)
            c.hard_key = c.hard_key or hard

        k, ck, toks = name_key(row.name), compact_key(row.name), name_tokens(row.name)
        pk = person_key(row.name)
        for i in range(len(self.db)):
            if not self.keys[i]:
                continue
            if self.entity == "people":
                if pk and pk == self.pkeys[i]:
                    add(i, 1.0, "same name")
                    continue
                last_ok = toks and self.toks[i] and toks[-1] == self.toks[i][-1]
                ratio = fuzz.WRatio(k, self.keys[i])
                if ratio >= 90 or (last_ok and fuzz.ratio(toks[0], self.toks[i][0]) >= 75):
                    add(i, ratio / 100 * 0.9, f"similar name ({ratio:.0f})")
            else:
                if k == self.keys[i] or ck == self.compact[i]:
                    add(i, 1.0, "same name key")
                    continue
                ratio = fuzz.WRatio(k, self.keys[i])
                if ratio >= 86:
                    add(i, ratio / 100 * 0.9, f"similar name ({ratio:.0f})")
                elif toks and self.toks[i] and toks[0] == self.toks[i][0] and len(toks[0]) >= 3 and (
                        set(toks) <= set(self.toks[i]) or set(self.toks[i]) <= set(toks)):
                    add(i, 0.6, "name contains the other")
            if len(ck) >= 4:
                for sk in self.slugs[i]:
                    if ck == sk.replace(" ", "") or (len(toks) > 1 and fuzz.token_set_ratio(k, sk) >= 95):
                        add(i, 0.85, f"input name matches this record's profile slug/handle '{sk}' "
                                     "(slugs often keep a former name)")
                        break

        if self.emb is not None and row_emb is not None:
            sims = self.emb @ row_emb
            for i in np.argsort(-sims)[:6]:
                if sims[i] >= 0.42:
                    add(int(i), float(sims[i]), f"description similarity {sims[i]:.2f}")

        # aliases the profile step recalled (former / current names)
        for alias in row.all_names[1:]:
            ak, ack = name_key(alias), compact_key(alias)
            for i in range(len(self.db)):
                if ak and (ak == self.keys[i] or ack == self.compact[i] or ack in {s.replace(" ", "") for s in self.slugs[i]}):
                    add(i, 0.9, f"the input is also known as '{alias}' (model recall), which is this record's name/slug")

        for pos, url in enumerate(row.evidence.urls()[:10], 1):
            for f in link_fields(self.entity):
                for db_id in self.index.lookup(f, url):
                    add(db_id, 1.0 - pos * 0.02, f"web result #{pos} for the input is this record's {f} ({url})",
                        hard=True)

        # fetched homepages: their host and their own social links
        for d, page in row.sites.items():
            urls = [("domain", page.get("final_host") or d), ("domain", d)] + [
                (f, u) for u in page.get("links", []) for f in ("linkedin", "crunchbase", "x")]
            for f, u in urls:
                for db_id in self.index.lookup(f, u):
                    trusted = page.get("name_match") and not page["source"].startswith("model")
                    add(db_id, 0.95, f"the input's fetched website {d} ({page['source']}; title "
                                     f"\"{page.get('title', '')[:80]}\") {'is' if f == 'domain' else 'links to'} this "
                                     f"record's {f}", hard=bool(trusted))

        # conflict: the input's verified website differs from a same-name candidate's website
        own = {page.get("final_host") or d for d, page in row.sites.items() if page.get("name_match")} | set(row.sites)
        verified = [d for d, p in row.sites.items() if p.get("name_match") and not p["source"].startswith("model")]
        for c in found.values():
            cd = (c.links.get("domain") or "").lower()
            if verified and cd and not c.hard_key and not any(
                    cd == o or o.endswith("." + cd) or cd.split(".")[0] == o.split(".")[0] for o in own):
                c.signals.append(f"CONFLICT: the input's own website appears to be {verified[0]}, "
                                 f"while this record's website is {cd}")

        ranked = sorted(found.values(), key=lambda c: (-c.hard_key, -c.score))
        return ranked[:cap]
