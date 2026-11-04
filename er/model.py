"""In-memory records passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

from .search import Evidence


@dataclass
class Cand:
    id: str
    name: str
    description: str
    links: dict
    score: float = 0.0
    signals: list[str] = field(default_factory=list)
    hard_key: bool = False


@dataclass
class Row:
    iid: str
    name: str
    text: str                      # organization description, or the person's source article
    entity: str                    # 'organizations' | 'people'
    context: dict = field(default_factory=dict)    # people: facts extracted from the article
    profile: dict = field(default_factory=dict)    # orgs: current name, aliases, guessed domains, search query
    sites: dict = field(default_factory=dict)      # domain -> fetched homepage info (+ source, name_match)
    evidence: Evidence = field(default_factory=Evidence)
    cands: list[Cand] = field(default_factory=list)
    match_id: str = ""             # db id or NEW
    confidence: float = 0.0
    current_name: str = ""
    reason: str = ""
    decided_by: str = ""
    dup_of: str = ""
    links: dict = field(default_factory=dict)
    link_conf: dict = field(default_factory=dict)
    link_options: dict = field(default_factory=dict)

    @property
    def search_name(self) -> str:
        return self.current_name or self.profile.get("current_name") or self.name

    @property
    def all_names(self) -> list[str]:
        names = [self.name, self.current_name, self.profile.get("current_name", "")] + list(
            self.profile.get("other_names", []))
        return [n for i, n in enumerate(names) if n and n not in names[:i]]

    def verified_sites(self) -> list[str]:
        """Fetched homepages whose title/host names this entity, best source first."""
        return [d for d, p in self.sites.items() if p.get("name_match")]

    @property
    def org_hint(self) -> str:
        return (self.context.get("organization") or "").strip()
