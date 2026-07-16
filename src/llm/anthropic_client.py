"""Anthropic LLM client — the thing BaseInvestorAgent.judge() calls.

Matches the interface agents/base.py already expects:
    raw = self.llm.complete(system=..., user=...)   # returns a JSON string

Retry logic mirrors the pattern in virattt/ai-hedge-fund's src/utils/llm.py:
  - retry up to `max_retries` times on any API error
  - if the model wraps JSON in prose or code fences, strip and extract it
  - final failure raises, so BaseInvestorAgent.judge() can catch it and
    fall back to a neutral InvestorSignal (never crash the whole run)

Usage:
    from src.llm.anthropic_client import AnthropicClient
    client = AnthropicClient(model="claude-opus-4-8")
    agent = RayDalioAgent(llm_client=client, macro_data=macro, prices=prices)
"""
from __future__ import annotations

import json
import os
import re
import time

import anthropic

# Errors that will NEVER succeed on retry — a bad key, a wrong model id, or a
# malformed request. Retrying these just sleeps through `max_retries` on every
# one of hundreds of calls (that is what made an invalid-key run crawl), so we
# re-raise immediately instead.
_NON_RETRYABLE = (
    anthropic.AuthenticationError,    # 401 — invalid/missing API key
    anthropic.PermissionDeniedError,  # 403 — key lacks access
    anthropic.BadRequestError,        # 400 — bad params
    anthropic.NotFoundError,          # 404 — wrong model id
)


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (or pass api_key=...)")
        self._client = anthropic.Anthropic(api_key=key)
        # Audit trail — accumulate across the run so every launch reports how many
        # calls it made, how many tokens it burned, and the estimated cost.
        self.calls = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str) -> str:
        """Returns a raw JSON string (parsed by the caller into a Pydantic model).

        Raises on final failure — callers (BaseInvestorAgent.judge) should
        catch and fall back to neutral rather than letting one bad call
        crash an entire backtest.
        """
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                u = getattr(resp, "usage", None)
                if u is not None:
                    self.calls += 1
                    self.input_tokens += getattr(u, "input_tokens", 0) or 0
                    self.output_tokens += getattr(u, "output_tokens", 0) or 0
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                )
                return _extract_json(text)
            except _NON_RETRYABLE:
                raise  # fail fast — retrying a bad key / model / request never helps
            except Exception as e:  # noqa: BLE001 — transient (429 / 5xx / network / parse): retry
                last_err = e
                if attempt < self.max_retries:
                    self.retries += 1
                    time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def validate(self) -> None:
        """Cheap preflight — one 1-token call to confirm the key and model work
        before a long run. Raises the underlying anthropic error on failure so
        the CLI can stop immediately instead of failing on every call."""
        self._client.messages.create(
            model=self.model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )

    # USD per 1M tokens (input, output); prefix match, falls back to Haiku rates.
    _PRICES = {
        "claude-fable-5": (10.0, 50.0),
        "claude-opus-4-8": (5.0, 25.0),
        "claude-sonnet-5": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
    }

    def usage_summary(self) -> dict:
        """Auditable per-run token + cost tally (see estimate before you launch)."""
        p_in, p_out = 1.0, 5.0  # default to Haiku
        for prefix, (i, o) in self._PRICES.items():
            if self.model.startswith(prefix):
                p_in, p_out = i, o
                break
        return {
            "model": self.model,
            "calls": self.calls,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "est_cost_usd": round(self.input_tokens / 1e6 * p_in
                                  + self.output_tokens / 1e6 * p_out, 4),
        }


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
