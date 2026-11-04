"""Evidence gathering: Serper (Google) search and plain-HTTP fetches of an entity's own homepage.

No LinkedIn / Crunchbase / X pages are ever fetched; their URLs are only read from search results or from links
on the entity's own website.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape

import httpx

from .clients import Caller
from .normalize import canon_domain, host_of

SERPER_URL = "https://google.serper.dev/search"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


@dataclass
class Hit:
    title: str
    link: str
    snippet: str
    query: str = ""


@dataclass
class Evidence:
    hits: list[Hit] = field(default_factory=list)
    kg_website: str = ""          # Google knowledge-graph website, when present
    kg_title: str = ""
    kg_desc: str = ""

    def extend(self, other: "Evidence | None") -> None:
        if not other:
            return
        seen = {h.link for h in self.hits}
        self.hits += [h for h in other.hits if h.link not in seen]
        self.kg_website = self.kg_website or other.kg_website
        self.kg_title = self.kg_title or other.kg_title
        self.kg_desc = self.kg_desc or other.kg_desc

    def urls(self) -> list[str]:
        out = [h.link for h in self.hits]
        if self.kg_website:
            out.append(self.kg_website)
        return out


class Search:
    def __init__(self, caller: Caller):
        self.c = caller
        self.s = caller.s

    @property
    def available(self) -> bool:
        return bool(self.s.serper_key)

    async def google(self, q: str, num: int = 10) -> Evidence | None:
        if not self.available:
            return None
        body = {"q": q, "num": num}

        def parse(r: httpx.Response):
            d = r.json()
            if "organic" not in d:
                return None
            kg = d.get("knowledgeGraph") or {}
            return {"organic": [{k: o.get(k, "") for k in ("title", "link", "snippet")} for o in d["organic"]],
                    "kg": {k: kg.get(k, "") for k in ("title", "website", "description")}}

        out = await self.c.call("serper", body, lambda cl: cl.post(
            SERPER_URL, headers={"X-API-KEY": self.s.serper_key, "Content-Type": "application/json"}, json=body), parse)
        if out is None:
            return None
        ev = Evidence(hits=[Hit(o["title"], o["link"], o["snippet"], q) for o in out["organic"] if o.get("link")])
        ev.kg_website, ev.kg_title, ev.kg_desc = out["kg"]["website"], out["kg"]["title"], out["kg"]["description"]
        return ev

    async def homepage(self, domain: str) -> dict | None:
        """GET https://<domain>; returns final host, <title>, meta description and social/profile links found."""
        domain = canon_domain(domain)
        if not domain:
            return None
        req = {"url": f"https://{domain}"}

        def parse(r: httpx.Response):
            html = r.text[:600_000]
            title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)', html, re.I)
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
            links = sorted({unescape(h) for h in hrefs
                            if re.search(r"(linkedin\.com/(company|in|school)/|crunchbase\.com/(organization|person)/"
                                         r"|(?<![a-z])(x|twitter)\.com/)", h, re.I)})
            return {"final_host": host_of(str(r.url)), "title": unescape(title.group(1).strip())[:200] if title else "",
                    "description": unescape(desc.group(1).strip())[:300] if desc else "", "links": links[:40]}

        return await self.c.call("fetch", req, lambda cl: cl.get(req["url"], headers={"User-Agent": UA}, timeout=12.0),
                                 parse, max_attempts=2)


def render_hits(ev: Evidence | None, limit: int = 8, snippet_len: int = 220) -> str:
    if not ev or not ev.hits:
        return "(no search results)"
    lines = []
    if ev.kg_title or ev.kg_website:
        lines.append(f"Google knowledge panel: {ev.kg_title} | website: {ev.kg_website} | {ev.kg_desc[:200]}")
    for h in ev.hits[:limit]:
        lines.append(f"- {h.title} | {h.link} | {h.snippet[:snippet_len]}")
    return "\n".join(lines)
