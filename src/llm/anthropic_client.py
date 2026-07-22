"""Anthropic LLM client — the thing ``LLMAnalyst.form_view`` calls.

The analyst's only requirement of a model is:
    raw = self.llm.complete(system=..., user=..., tool=SUBMIT_VIEW_TOOL)  # → JSON str

Behaviour that earns its keep on a long run:
  - non-retryable errors (bad key / wrong model) fail fast instead of sleeping
    through `max_retries` on every one of hundreds of calls
  - a forced tool call is the portable structured-output path (works on Haiku,
    Sonnet and Opus alike); prose/fence-wrapped JSON is stripped as a fallback
  - final failure raises, so ``LLMAnalyst`` can catch it and emit a degraded
    view rather than crashing the run

Usage:
    from src.llm.anthropic_client import AnthropicClient
    client = AnthropicClient(model="claude-haiku-4-5-20251001")
    analyst = LLMAnalyst.from_persona("inflation", llm=client, text_selector=sel)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

import anthropic

# Errors that will NEVER succeed on retry — a bad key, a wrong model id, or a
# malformed request. Retrying these just sleeps through `max_retries` on every
# one of hundreds of calls (that is what made an invalid-key run crawl), so we
# re-raise immediately instead.
# BadRequestError (400) is handled explicitly in complete() — a prefill-unsupported
# model is recovered rather than failed — so it is not in this blanket tuple.
_NON_RETRYABLE = (
    anthropic.AuthenticationError,    # 401 — invalid/missing API key
    anthropic.PermissionDeniedError,  # 403 — key lacks access
    anthropic.NotFoundError,          # 404 — wrong model id
)


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        temperature: float | None = None,
        cache_dir: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        # Opt-in extras folded in from the backtest fork (see the merge proposal).
        # temperature=None means "omit the param" so the analyst path is byte-for-
        # byte unchanged; a backtest passes 0.0 to pin reproducibility. cache_dir
        # turns reruns of an identical prompt free AND deterministic — the same
        # as-of date yields the same facts, the same prompt, the same cache hit.
        self.temperature = temperature
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (or pass api_key=...)")
        self._client = anthropic.Anthropic(api_key=key)
        # Audit trail — accumulate across the run so every launch reports how many
        # calls it made, how many tokens it burned, and the estimated cost.
        # ``_lock`` guards the tally so a threaded caller can't drop counts.
        self._lock = threading.Lock()
        self.calls = 0
        self.retries = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _cache_path(self, system: str, user: str, tool_name: str | None) -> str | None:
        if not self.cache_dir:
            return None
        payload = json.dumps(
            [self.model, self.temperature, self.max_tokens, system, user, tool_name],
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def complete(self, system: str, user: str, prefill: str | None = None,
                 tool: dict | None = None) -> str:
        """Return a JSON string (parsed by the caller).

        ``tool`` is the reliable, model-agnostic way to get structured output: a
        tool schema plus a forced ``tool_choice`` makes the model emit a validated
        object, and the SDK hands it back as a dict, which is serialized here so the
        caller's parsing path is unchanged. This exists because neither cheaper
        mechanism is portable — instructions alone let Haiku *and* Sonnet answer in
        prose/Markdown (parsed as nothing, burned as retries), while the ``prefill``
        opening-brace trick that fixes that is rejected outright by Sonnet 5 and
        Opus 4.8. Forcing a tool works on all three.

        ``prefill`` seeds the assistant turn with e.g. ``"{"``; kept for models that
        support it and for callers not using a tool. When both are given, ``tool``
        wins and prefill is ignored.

        Raises on final failure — ``LLMAnalyst`` catches it and emits a degraded
        view rather than letting one bad call crash an entire run.
        """
        tool_name = tool["name"] if tool is not None else None
        cache_path = self._cache_path(system, user, tool_name)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                result = fh.read()
            with self._lock:
                self.cached_calls += 1
            return result

        last_err: Exception | None = None
        # Prefill forces JSON on models that support it (Haiku), but Sonnet 5 and
        # Opus 4.8 reject an assistant-terminated conversation with a 400. So it is
        # best-effort: if a model refuses the prefill we drop it and fall back to
        # instruction + brace extraction, which the stronger models follow reliably.
        use_prefill = bool(prefill) and tool is None
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            messages: list[dict] = [{"role": "user", "content": user}]
            if use_prefill:
                messages.append({"role": "assistant", "content": prefill})
            kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                          system=system, messages=messages)
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if tool is not None:
                kwargs["tools"] = [tool]
                kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}
            try:
                resp = self._client.messages.create(**kwargs)
                u = getattr(resp, "usage", None)
                if u is not None:
                    with self._lock:
                        self.calls += 1
                        self.input_tokens += getattr(u, "input_tokens", 0) or 0
                        self.output_tokens += getattr(u, "output_tokens", 0) or 0
                if tool is not None:
                    for block in resp.content:
                        if getattr(block, "type", None) == "tool_use":
                            result = json.dumps(block.input)  # already a validated dict
                            return self._cache_write(cache_path, result)
                    raise ValueError("forced tool_choice returned no tool_use block")
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                )
                if use_prefill:
                    text = prefill + text   # the reply continues the seed, so restore it
                return self._cache_write(cache_path, _extract_json(text))
            except anthropic.BadRequestError as e:
                # A model that cannot take a prefill: strip it and retry, without
                # spending a real attempt on a fixable configuration mismatch.
                if use_prefill and "prefill" in str(e).lower():
                    use_prefill = False
                    attempt -= 1
                    continue
                raise  # any other 400 is a genuine bad request — never retryable
            except _NON_RETRYABLE:
                raise  # fail fast — retrying a bad key / model never helps
            except Exception as e:  # noqa: BLE001 — transient (429 / 5xx / network / parse): retry
                last_err = e
                if attempt < self.max_retries:
                    with self._lock:
                        self.retries += 1
                    time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def _cache_write(self, cache_path: str | None, result: str) -> str:
        """Persist a successful response (when caching is on), then return it."""
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as fh:
                fh.write(result)
        return result

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
            "cached_calls": self.cached_calls,
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

    # strict=False permits literal newlines and control characters inside string
    # values. Models write multi-paragraph prose into a JSON field, which is invalid
    # strict JSON; parsing it strictly failed ~40-55% of analyst calls and silently
    # burned them as retries. Tolerating it is preferable to forbidding paragraphs.
    json.loads(candidate, strict=False)  # validate before returning
    return candidate
