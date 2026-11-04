"""Shared I/O layer: response cache, spend ledger, and an HTTP caller with retry/backoff.

Every external call goes through `Caller.call`, which
  1. returns a cached response if this exact request was made before (cache key = sha256 of provider + request,
     never the auth header),
  2. otherwise waits on the provider's semaphore, retries 429/5xx/timeouts with exponential backoff + jitter
     (honouring Retry-After), and stops retrying when the run's deadline is near,
  3. records calls, cache hits, failures and cost in the ledger,
  4. returns None on final failure so one bad call degrades one row instead of stopping the run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .config import Settings

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class Cache:
    """SQLite key/value store of raw responses. Written on first success; shipped with the repo."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, provider TEXT, request TEXT, response TEXT, ts REAL)")
        self.db.commit()
        self._pending = 0

    @staticmethod
    def key(provider: str, request: Any) -> str:
        blob = json.dumps([provider, request], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> Any | None:
        row = self.db.execute("SELECT response FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, provider: str, request: Any, response: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?,?,?)",
                        (k, provider, json.dumps(request, ensure_ascii=False), json.dumps(response, ensure_ascii=False),
                         time.time()))
        self._pending += 1
        if self._pending >= 25:
            self.flush()

    def flush(self) -> None:
        self.db.commit()
        self._pending = 0

    def close(self) -> None:
        self.flush()
        self.db.close()


@dataclass
class Ledger:
    calls: Counter = field(default_factory=Counter)        # live network calls that succeeded
    cache_hits: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    retries: Counter = field(default_factory=Counter)
    cost_usd: float = 0.0                                  # spend of live calls this run (OpenRouter usage.cost)
    cached_cost_usd: float = 0.0                           # what cached responses originally cost
    tokens: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {"calls": dict(self.calls), "cache_hits": dict(self.cache_hits), "failures": dict(self.failures),
                "retries": dict(self.retries), "openrouter_cost_usd_live": round(self.cost_usd, 5),
                "openrouter_cost_usd_if_uncached": round(self.cost_usd + self.cached_cost_usd, 5),
                "serper_queries_live": self.calls.get("serper", 0), "tokens": dict(self.tokens)}


class Caller:
    def __init__(self, settings: Settings, cache: Cache | None):
        self.s = settings
        self.cache = cache
        self.ledger = Ledger()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0), follow_redirects=True,
                                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        self.sems = {"openrouter": asyncio.Semaphore(settings.conc_llm),
                     "serper": asyncio.Semaphore(settings.conc_search),
                     "fetch": asyncio.Semaphore(settings.conc_fetch)}

    async def aclose(self) -> None:
        await self.client.aclose()
        if self.cache:
            self.cache.flush()

    async def call(self, provider: str, request: Any, send: Callable[[httpx.AsyncClient], Awaitable[httpx.Response]],
                   parse: Callable[[httpx.Response], Any], *, max_attempts: int = 5, use_cache: bool = True,
                   cost_of: Callable[[Any], float] | None = None) -> Any | None:
        """Cached, retried call. `parse` turns a 2xx response into the value that is cached and returned."""
        k = Cache.key(provider, request) if (self.cache and use_cache) else None
        if k:
            hit = self.cache.get(k)
            if hit is not None:
                self.ledger.cache_hits[provider] += 1
                if cost_of:
                    self.ledger.cached_cost_usd += cost_of(hit) or 0.0
                return hit
        if self.s.time_left() <= 0:
            self.ledger.failures[provider + ":deadline"] += 1
            return None
        sem = self.sems.get(provider) or self.sems["fetch"]
        for attempt in range(1, max_attempts + 1):
            wait = None
            try:
                if self.s.fault_rate and random.random() < self.s.fault_rate:   # resilience testing only
                    raise httpx.ConnectError("injected fault")
                async with sem:
                    resp = await send(self.client)
                if resp.status_code < 300:
                    value = parse(resp)
                    if value is None:            # e.g. 200 with an error body: treat as retryable
                        wait = self._backoff(attempt)
                    else:
                        self.ledger.calls[provider] += 1
                        if cost_of:
                            self.ledger.cost_usd += cost_of(value) or 0.0
                        if k:
                            self.cache.put(k, provider, request, value)
                        return value
                elif resp.status_code in RETRY_STATUS:
                    wait = self._retry_after(resp) or self._backoff(attempt)
                else:                            # 4xx that retrying will not fix (bad key, bad request)
                    self.ledger.failures[f"{provider}:{resp.status_code}"] += 1
                    return None
            except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError, ValueError, KeyError):
                wait = self._backoff(attempt)
            if attempt == max_attempts or self.s.time_left() < wait + 5:
                break
            self.ledger.retries[provider] += 1
            await asyncio.sleep(wait)
        self.ledger.failures[provider] += 1
        return None

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(20.0, 0.75 * 2 ** (attempt - 1)) * (0.5 + random.random())

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        try:
            return min(30.0, float(resp.headers.get("retry-after", "")))
        except ValueError:
            return None
