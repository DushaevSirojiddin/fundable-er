"""OpenRouter: strict-JSON chat completions and embeddings, on top of the cached/retried Caller."""
from __future__ import annotations

import base64
import json

import httpx
import numpy as np

from .clients import Caller

URL = "https://openrouter.ai/api/v1"


class LLM:
    def __init__(self, caller: Caller):
        self.c = caller
        self.s = caller.s

    @property
    def available(self) -> bool:
        return bool(self.s.openrouter_key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.openrouter_key}", "X-Title": "fundable-er case study"}

    async def json(self, system: str, user: str, schema: dict, name: str, *, strong: bool = False) -> dict | None:
        """Chat completion constrained to `schema`. Returns the parsed object or None."""
        if not self.available or self.c.ledger.cost_usd >= self.s.llm_budget_usd:
            self.c.ledger.failures["openrouter:skipped"] += 1
            return None
        model = self.s.model_strong if strong else self.s.model_fast
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            "usage": {"include": True},
            "seed": 7,
        }
        if "gpt-4" in model:
            body["temperature"] = 0
        else:
            body["reasoning"] = {"effort": "medium" if strong else "low"}

        def parse(r: httpx.Response):
            d = r.json()
            if "error" in d or not d.get("choices"):
                return None
            content = d["choices"][0]["message"].get("content") or ""
            obj = json.loads(content)                      # JSONDecodeError -> retried by Caller
            u = d.get("usage") or {}
            return {"obj": obj, "cost": u.get("cost") or 0.0, "model": d.get("model", model),
                    "tokens": [u.get("prompt_tokens", 0), u.get("completion_tokens", 0)]}

        out = await self.c.call("openrouter", body,
                                lambda cl: cl.post(f"{URL}/chat/completions", headers=self._headers(), json=body),
                                parse, cost_of=lambda v: v.get("cost", 0.0))
        if out is None:
            return None
        self.c.ledger.tokens["prompt"] += out["tokens"][0]
        self.c.ledger.tokens["completion"] += out["tokens"][1]
        return out["obj"]

    async def embed(self, texts: list[str]) -> np.ndarray | None:
        """Unit-normalized embeddings, one cached request per batch of up to 64 texts."""
        if not self.available:
            return None
        vecs: list[np.ndarray | None] = [None] * len(texts)
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            body = {"model": self.s.embed_model, "input": batch}

            def parse(r: httpx.Response):
                d = r.json()
                if "data" not in d:
                    return None
                rows = sorted(d["data"], key=lambda x: x["index"])
                # cache as float16 base64 to keep the shipped cache small
                packed = [base64.b64encode(np.asarray(x["embedding"], dtype=np.float16).tobytes()).decode()
                          for x in rows]
                return {"vecs": packed, "cost": (d.get("usage") or {}).get("cost") or 0.0}

            out = await self.c.call("openrouter", body,
                                    lambda cl, b=body: cl.post(f"{URL}/embeddings", headers=self._headers(), json=b),
                                    parse, cost_of=lambda v: v.get("cost", 0.0))
            if out is None:
                return None
            for i, p in enumerate(out["vecs"]):
                vecs[start + i] = np.frombuffer(base64.b64decode(p), dtype=np.float16).astype(np.float32)
        m = np.vstack(vecs)
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
