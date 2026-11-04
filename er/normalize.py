"""Deterministic keys: name normalization, URL canonicalization and the DB link index.

Everything here is pure and side-effect free so it can be unit-tested and reused by every stage.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from urllib.parse import unquote, urlsplit

from unidecode import unidecode

# ---------------------------------------------------------------------------- names

LEGAL_SUFFIXES = {
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "lp", "plc", "gmbh", "ag", "sa", "sas", "sarl", "bv",
    "nv", "oy", "ab", "as", "aps", "spa", "srl", "pty", "pte", "co", "corp", "corporation", "company", "kk",
}
NICKNAMES = {
    "andy": "andrew", "drew": "andrew", "bob": "robert", "rob": "robert", "bobby": "robert", "bill": "william",
    "will": "william", "billy": "william", "mike": "michael", "mick": "michael", "chris": "christopher",
    "kate": "katherine", "katie": "katherine", "kathy": "katherine", "cathy": "catherine", "tom": "thomas",
    "tommy": "thomas", "jim": "james", "jimmy": "james", "dave": "david", "dan": "daniel", "danny": "daniel",
    "alex": "alexander", "sam": "samuel", "ben": "benjamin", "matt": "matthew", "nick": "nicholas",
    "tony": "anthony", "joe": "joseph", "steve": "steven", "stephen": "steven", "liz": "elizabeth",
    "beth": "elizabeth", "jen": "jennifer", "jenny": "jennifer", "ed": "edward", "eddie": "edward",
    "ted": "edward", "rick": "richard", "rich": "richard", "dick": "richard", "greg": "gregory",
    "jeff": "jeffrey", "geoff": "geoffrey", "jon": "jonathan", "johnny": "john", "pat": "patrick",
    "patty": "patricia", "pete": "peter", "phil": "philip", "ron": "ronald", "sue": "susan", "tim": "timothy",
    "vince": "vincent", "zach": "zachary", "zack": "zachary", "max": "maximilian", "fred": "frederick",
    "charlie": "charles", "chuck": "charles", "abby": "abigail", "meg": "margaret", "maggie": "margaret",
}


def fold(s: str) -> str:
    """Lower-case ASCII fold: 'Ørsted' -> 'orsted', 'Köhler' -> 'kohler'."""
    return unidecode(unicodedata.normalize("NFKC", s or "")).lower().strip()


def name_tokens(s: str) -> list[str]:
    s = fold(s).replace("&", " and ")
    s = re.sub(r"\.(com|io|ai|co|vc|xyz|app|gg|org|net)\b", r" \1", s)  # 'circle.co' -> 'circle co'
    return re.findall(r"[a-z0-9]+", s)


def name_key(s: str) -> str:
    """Canonical org/person name key: folded, punctuation-free, trailing legal suffixes removed."""
    toks = name_tokens(s)
    while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def compact_key(s: str) -> str:
    return name_key(s).replace(" ", "")


def person_key(s: str) -> str:
    """First + last token with nicknames expanded: 'Andy  Schwartz' -> 'andrew schwartz'."""
    toks = name_tokens(s)
    if not toks:
        return ""
    first = NICKNAMES.get(toks[0], toks[0])
    return f"{first} {toks[-1]}" if len(toks) > 1 else first


# ---------------------------------------------------------------------------- URLs

AGGREGATOR_DOMAINS = {
    "linkedin.com", "crunchbase.com", "x.com", "twitter.com", "facebook.com", "instagram.com", "youtube.com",
    "wikipedia.org", "bloomberg.com", "pitchbook.com", "techcrunch.com", "medium.com", "github.com",
    "cbinsights.com", "tracxn.com", "zoominfo.com", "dnb.com", "globenewswire.com", "prnewswire.com",
    "businesswire.com", "reuters.com", "forbes.com", "wellfound.com", "angel.co", "apollo.io",
    "rocketreach.co", "theorg.com", "glassdoor.com", "indeed.com", "owler.com", "craft.co", "f6s.com",
    "producthunt.com", "google.com", "apple.com", "tiktok.com", "threads.net", "substack.com", "eu-startups.com",
    "sifted.eu", "dealroom.co", "signalhire.com", "contactout.com", "leadiq.com", "cbinsights.com", "yahoo.com",
    "wsj.com", "ft.com", "venturebeat.com", "finsmes.com", "builtin.com", "golden.com", "pitchbook.com",
    "ycombinator.com", "startupblink.com", "seedtable.com", "vcsheet.com", "openvc.app", "pappers.fr",
    "northdata.com", "opencorporates.com", "companieshouse.gov.uk", "find-and-update.company-information.service.gov.uk",
    "bizapedia.com", "zaubacorp.com", "tofler.in", "economictimes.indiatimes.com", "inc42.com", "yourstory.com",
    "entrackr.com", "vccircle.com", "axios.com", "cnbc.com", "businessinsider.com", "news.ycombinator.com",
    "slideshare.net", "scribd.com", "issuu.com", "amazon.com", "play.google.com", "apps.apple.com", "reddit.com",
    "quora.com", "wikidata.org", "britannica.com", "marketscreener.com", "macrotrends.net", "stockanalysis.com",
    "sec.gov", "gov.uk", "europa.eu", "lusha.com", "datanyze.com", "clay.com", "growjo.com", "getlatka.com",
    "similarweb.com", "semrush.com", "linktr.ee", "about.me", "angellist.com", "venturebeat.com",
}
X_RESERVED = {
    "home", "i", "intent", "share", "search", "hashtag", "explore", "settings", "login", "signup", "privacy",
    "tos", "status", "messages", "notifications", "compose", "about", "help", "jobs", "en", "web", "download",
}


def _split(url: str):
    url = (url or "").strip()
    if not url:
        return None
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        url = "https://" + url
    try:
        return urlsplit(url)
    except ValueError:
        return None


def host_of(url: str) -> str:
    p = _split(url)
    if not p or not p.hostname:
        return ""
    h = p.hostname.lower().rstrip(".")
    return h[4:] if h.startswith("www.") else h


def canon_domain(url: str) -> str:
    """Bare host (no scheme/www/path); '' for aggregator or social hosts."""
    h = host_of(url)
    if not h or "." not in h:
        return ""
    if any(h == d or h.endswith("." + d) for d in AGGREGATOR_DOMAINS):
        return ""
    return h


def _path(url: str) -> tuple[str, str]:
    p = _split(url)
    if not p:
        return "", ""
    return host_of(url), unquote(p.path or "")


def canon_linkedin(url: str, kind: str) -> str:
    """kind 'company' or 'in' -> 'https://www.linkedin.com/<kind>/<slug>' or ''."""
    h, path = _path(url)
    if not (h == "linkedin.com" or h.endswith(".linkedin.com")):
        return ""
    m = re.match(rf"^/{kind}/([^/]+)", path)
    if not m:
        return ""
    slug = m.group(1).strip()
    return f"https://www.linkedin.com/{kind}/{slug}" if slug else ""


def canon_crunchbase(url: str, kind: str) -> str:
    """kind 'organization' or 'person'."""
    h, path = _path(url)
    if not (h == "crunchbase.com" or h.endswith(".crunchbase.com")):
        return ""
    m = re.match(rf"^/{kind}/([^/]+)", path)
    if not m:
        return ""
    return f"https://www.crunchbase.com/{kind}/{m.group(1).strip()}"


def canon_x(url: str) -> str:
    h, path = _path(url)
    if h not in {"x.com", "twitter.com", "mobile.twitter.com", "mobile.x.com"}:
        return ""
    m = re.match(r"^/@?([A-Za-z0-9_]{1,15})/?$", path) or re.match(r"^/@?([A-Za-z0-9_]{1,15})/(?!status)", path)
    if not m or m.group(1).lower() in X_RESERVED:
        return ""
    return f"https://x.com/{m.group(1)}"


def canon_link(field: str, url: str, entity: str) -> str:
    """Canonicalize a URL for an output field; entity is 'organizations' or 'people'."""
    org = entity == "organizations"
    if field == "domain":
        return canon_domain(url)
    if field == "linkedin":
        return canon_linkedin(url, "company" if org else "in")
    if field == "crunchbase":
        return canon_crunchbase(url, "organization" if org else "person")
    if field == "x":
        return canon_x(url)
    raise ValueError(field)


def compare_key(field: str, v: str) -> str:
    """Same normalization the grader applies before comparing links."""
    v = unicodedata.normalize("NFC", unquote((v or "").strip()))
    if not v:
        return ""
    v = v.lower()
    v = re.sub(r"^[a-z][a-z0-9+.-]*://", "", v)
    v = re.split(r"[?#]", v)[0]
    v = re.sub(r"^www\.", "", v)
    v = re.sub(r"^[a-z]{2}\.linkedin\.com/", "linkedin.com/", v)
    v = re.sub(r"^(mobile\.)?twitter\.com/", "x.com/", v)
    return v.rstrip("/")


def link_fields(entity: str) -> list[str]:
    return ["domain", "linkedin", "crunchbase", "x"] if entity == "organizations" else ["linkedin", "crunchbase", "x"]


class LinkIndex:
    """field -> compare_key(link) -> {db ids}. A hit is a deterministic match key."""

    def __init__(self, rows: list[dict], entity: str):
        self.entity = entity
        self.idx: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for r in rows:
            for f in link_fields(entity):
                k = compare_key(f, r.get(f, ""))
                if k:
                    self.idx[f][k].add(r["id"])

    def lookup(self, field: str, url: str) -> set[str]:
        c = canon_link(field, url, self.entity)
        return set(self.idx[field].get(compare_key(field, c), ())) if c else set()


# ---------------------------------------------------------------------------- union-find


def group_ids(ids: list[str], edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Union-find over ids; returns representative -> members in input order (rep = first member)."""
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return {members[0]: members for members in groups.values()}
