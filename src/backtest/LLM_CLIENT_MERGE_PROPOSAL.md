# Proposal: unify the two `AnthropicClient` forks (for Elias)

There are currently **two** `AnthropicClient` implementations:

| | `src/llm/anthropic_client.py` (Elias / analysts) | `src/backtest/anthropic_client.py` (backtest, moved from NW) |
|---|---|---|
| Structured output | **forced tool-calling** (`SUBMIT_VIEW_TOOL`) + prose/fence fallback | prose/fence extraction only |
| Determinism | — | **`temperature` pinned to 0** |
| Caching | — | **on-disk `cache_dir`** keyed by (model, temp, max_tokens, system, user) → free/deterministic reruns |
| Robustness | **`_NON_RETRYABLE` fast-fail** (bad key/model), prefill recovery | broad retry loop |
| Auditing | **`validate()`, `usage_summary()`** (token + cost tally) | — |
| JSON parsing | `json.loads(strict=False)` (tolerates multi-paragraph strings) | `json.loads` (strict) |

They share the same core call: `complete(system, user) -> JSON string`.

## Why they weren't merged in this change
The backtest needs `cache_dir`/`temperature`; Elias's client needs tool-calling +
audit. Merging means editing `src/llm/anthropic_client.py`, which is Elias's file —
so it was left untouched and the backtest keeps its own fork under
`src/backtest/`. This note is the hand-off.

## Proposed single client (superset)
Adopt Elias's client as the base and add the two backtest features as **opt-in**
constructor args so nothing in the analyst path changes:

- add `temperature: float = 0.0` and `cache_dir: str | None = None` params
- wrap `complete()` so that, when `cache_dir` is set, it reads/writes a cache
  file keyed by `(model, temperature, max_tokens, system, user, tool_name)`
- keep tool-calling, `_NON_RETRYABLE` fast-fail, `validate()`, `usage_summary()`,
  and `strict=False` parsing exactly as-is

Then both callers import `from src.llm.anthropic_client import AnthropicClient`
and `src/backtest/anthropic_client.py` is deleted.

## Migration once merged
1. Elias adds `temperature` + `cache_dir` to `src/llm/anthropic_client.py`.
2. In `src/run_backtest.py` change the import back to `src.llm.anthropic_client`.
3. Delete `src/backtest/anthropic_client.py` and this file.
