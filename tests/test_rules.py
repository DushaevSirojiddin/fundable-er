"""Escalation rule and failure caching. No network: the HTTP layer is replaced by fake send functions."""
import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from er.clients import Cache, Caller  # noqa: E402
from er.config import Settings  # noqa: E402
from er.model import Cand, Row  # noqa: E402
from er.resolve import is_risky  # noqa: E402

CONFLICT = "CONFLICT: the input's own website appears to be apex-robotics.com, while this record's website is apex.io"


def _row(name="Apex Capital", cands=()):
    return Row(iid="i1", name=name, text="", entity="organizations", cands=list(cands))


def test_same_name_match_is_safe_without_conflict():
    c = Cand("o1", "APEX CAPITAL", "", {"domain": "apexcapital.com"})
    assert not is_risky(_row(cands=[c]), "o1", {"o1": c})


def test_same_name_match_with_website_conflict_is_risky():
    c = Cand("o1", "APEX CAPITAL", "", {"domain": "apex.io"}, signals=[CONFLICT])
    assert is_risky(_row(cands=[c]), "o1", {"o1": c})


def _site(name_match=True, source="model's guess (unverified)", final_host=""):
    return {"name_match": name_match, "source": source, "final_host": final_host}


def test_match_is_risky_when_a_fetched_site_naming_the_input_is_another_website():
    c = Cand("o1", "APEX CAPITAL", "", {"domain": "apex.capital"})
    row = _row(cands=[c])
    row.sites = {"apexcapital.com": _site(), "jobs-board.io": _site(name_match=False)}
    assert is_risky(row, "o1", {"o1": c})


def test_match_is_safe_when_the_fetched_site_is_the_candidates_own():
    c = Cand("o1", "APEX CAPITAL", "", {"domain": "apexcapital.com"})
    row = _row(cands=[c])
    row.sites = {"apexcapital.de": _site(), "en.apexcapital.com": _site(),
                 "apex-old.com": _site(final_host="www.apexcapital.com"), "news.io": _site(name_match=False)}
    assert not is_risky(row, "o1", {"o1": c})


def test_hard_key_match_is_never_risky():
    c = Cand("o1", "Nova Labs", "", {}, signals=[CONFLICT], hard_key=True)
    assert not is_risky(_row(cands=[c]), "o1", {"o1": c})


def test_match_across_unrelated_names_is_risky():
    c = Cand("o1", "Summit Partners", "", {})
    assert is_risky(_row(cands=[c]), "o1", {"o1": c})


def test_new_despite_exact_name_candidate_is_risky():
    c = Cand("o1", "Apex Capital Ltd", "", {})
    assert is_risky(_row(cands=[c]), "NEW", {"o1": c})
    other = Cand("o2", "Summit Partners", "", {})
    assert not is_risky(_row(cands=[other]), "NEW", {"o2": other})


def _run_twice(tmp_path, send, cache_failures):
    """Make the same call twice through one cache; return (results, how many times send ran)."""
    calls = {"n": 0}

    async def counted(client):
        calls["n"] += 1
        return send()

    async def go():
        caller = Caller(Settings(), Cache(tmp_path / "c.sqlite"))
        out = [await caller.call("fetch", {"url": "https://example.com"}, counted, lambda r: {"ok": True},
                                 max_attempts=1, cache_failures=cache_failures) for _ in range(2)]
        await caller.aclose()
        return out

    return asyncio.run(go()), calls["n"]


def test_failed_fetch_is_cached_when_asked(tmp_path):
    out, n = _run_twice(tmp_path, lambda: httpx.Response(403), cache_failures=True)
    assert out == [None, None] and n == 1          # second call answered from the cache


def test_exhausted_retries_are_cached_when_asked(tmp_path):
    def boom():
        raise httpx.ConnectError("down")
    out, n = _run_twice(tmp_path, boom, cache_failures=True)
    assert out == [None, None] and n == 1


def test_failures_are_not_cached_by_default(tmp_path):
    out, n = _run_twice(tmp_path, lambda: httpx.Response(403), cache_failures=False)
    assert out == [None, None] and n == 2


def test_success_is_cached(tmp_path):
    out, n = _run_twice(tmp_path, lambda: httpx.Response(200), cache_failures=True)
    assert out == [{"ok": True}, {"ok": True}] and n == 1
