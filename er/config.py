"""Runtime settings. Secrets come only from environment variables (a local .env is loaded if present)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    openrouter_key: str = ""
    serper_key: str = ""
    model_fast: str = "openai/gpt-5.6-luna"     # bulk: adjudication, dedupe, link picking
    model_strong: str = "openai/gpt-5.4"        # escalation for low-confidence decisions
    embed_model: str = "openai/text-embedding-3-small"
    llm_budget_usd: float = 5.0                 # hard cap per run; past it LLM steps fall back to rules
    deadline_s: float = 540.0                   # soft deadline; the README limit is 600 s
    escalate_below: float = 0.7
    link_threshold: float = 0.6                 # emit a link only if the picker's confidence >= this
    x_search_threshold: float = 0.95            # stricter bar for X handles found only via search
    cache_path: Path = ROOT / "cache" / "cache.sqlite"
    conc_llm: int = 16
    conc_search: int = 8
    conc_fetch: int = 16
    fault_rate: float = 0.0                     # FAULT_RATE: fraction of live calls failed on purpose (testing)
    started: float = field(default_factory=time.monotonic)

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        s = cls(
            openrouter_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
            serper_key=os.environ.get("SERPER_API_KEY", "").strip(),
        )
        s.model_fast = os.environ.get("MODEL_FAST", s.model_fast)
        s.model_strong = os.environ.get("MODEL_STRONG", s.model_strong)
        s.llm_budget_usd = _f("LLM_BUDGET_USD", s.llm_budget_usd)
        s.deadline_s = _f("DEADLINE_SECONDS", s.deadline_s)
        s.link_threshold = _f("LINK_THRESHOLD", s.link_threshold)
        s.fault_rate = _f("FAULT_RATE", 0.0)
        s.x_search_threshold = _f("X_SEARCH_THRESHOLD", s.x_search_threshold)
        if os.environ.get("CACHE_PATH"):
            s.cache_path = Path(os.environ["CACHE_PATH"])
        return s

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def time_left(self) -> float:
        return self.deadline_s - self.elapsed()
