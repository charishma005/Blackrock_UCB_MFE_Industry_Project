"""Anthropic LLM client — the thing BaseInvestorAgent.judge() calls.

Matches the interface agents/base.py already expects:
    raw = self.llm.complete(system=..., user=...)   # returns a JSON string

Retry logic mirrors the pattern in virattt/ai-hedge-fund's src/utils/llm.py:
  - retry up to `max_retries` times on any API error
  - if the model wraps JSON in prose or code fences, strip and extract it
  - final failure raises, so BaseInvestorAgent.judge() can catch it and
    fall back to a neutral InvestorSignal (never crash the whole run)

Usage:
    from src.backtest.anthropic_client import AnthropicClient
    client = AnthropicClient(model="claude-opus-4-8")
    agent = RayDalioAgent(llm_client=client, macro_data=macro, prices=prices)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import anthropic


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        temperature: float = 0.0,
        cache_dir: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        # temperature 0 by default so a backtest is reproducible run-to-run;
        # without pinning it, the same rebalance date returns different signals
        # each run and you can't attribute a change in results to a code change.
        self.temperature = temperature
        # optional on-disk response cache keyed by (model, temperature,
        # max_tokens, system, user). A backtest re-queries the SAME prompt on
        # every rerun (same as-of date -> same facts -> same prompt), so a
        # persistent cache turns reruns free AND deterministic. Delete the dir
        # to force a clean re-query.
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (or pass api_key=...)")
        self._client = anthropic.Anthropic(api_key=key)

    def _cache_path(self, system: str, user: str) -> str | None:
        if not self.cache_dir:
            return None
        payload = json.dumps(
            [self.model, self.temperature, self.max_tokens, system, user],
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def complete(self, system: str, user: str) -> str:
        """Returns a raw JSON string (parsed by the caller into a Pydantic model).

        Raises on final failure — callers (BaseInvestorAgent.judge) should
        catch and fall back to neutral rather than letting one bad call
        crash an entire backtest.
        """
        cache_path = self._cache_path(system, user)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                return fh.read()

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                )
                result = _extract_json(text)
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as fh:
                        fh.write(result)
                return result
            except Exception as e:  # noqa: BLE001 — deliberately broad, this is a retry boundary
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence before/after.
    Strip fences first, then fall back to a brace-matching scan.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # first balanced {...} block
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")
        depth = 0
        end = None
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            raise ValueError(f"Unbalanced JSON in LLM output: {text[:200]!r}")
        candidate = text[start : end + 1]

    json.loads(candidate)  # validate before returning
    return candidate