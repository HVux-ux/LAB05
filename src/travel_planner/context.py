"""Context-window budgeting: pack only the relevant slice of data into the prompt.

The problem
-----------
An LLM can only attend to what fits in its context window at once (roughly
8k-128k tokens for the locally runnable models in Ollama; a token is about
4 characters of English). Two naive designs both fail:

* **Dump everything.** With a 50,000-row Kaggle dataset that is millions of
  tokens: it does not fit at all, and even if it did, cost and latency grow
  linearly while accuracy falls ("lost in the middle" [1]).
* **Summarise everything.** A pre-computed summary cannot answer "find me a
  cheap vegetarian place near the river in Hoi An" — the detail is already gone.

The design used here
--------------------
Preprocess with pandas first, so that the model sees a **retrieved subset**:

    1. filter   hard constraints (city, category, price, rating, month)
    2. rank     by a relevance score
    3. pack     rows into a token budget, cheapest-to-drop last
    4. render   as compact JSON, not prose (fewer tokens, no invented fields)

The model then does what it is actually good at — composing a plan from a small
set of facts — and is instructed to use only those facts.

[1] Liu et al., "Lost in the Middle: How Language Models Use Long Contexts",
    TACL 2024. https://arxiv.org/abs/2307.03172
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

#: Fields worth spending prompt tokens on. Anything not listed (ids, lat/long,
#: address strings) is dropped from the prompt: it costs tokens and the model
#: cannot use it to write a better itinerary.
PROMPT_FIELDS: tuple[str, ...] = (
    "name", "city", "category", "price_usd", "rating", "duration_hours", "tags", "description",
)

#: Truncate descriptions to this many characters, keeping prompts predictable.
MAX_DESCRIPTION_CHARS = 150


# --------------------------------------------------------------------- tokens
def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 characters per token for English text.

    Deliberately approximate. The real tokenizer (tiktoken / the model's own)
    would be more precise but adds a dependency and is model-specific; for
    budgeting purposes a consistent over-estimate is safer than an exact one
    that lets the prompt overflow.
    """
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    # +4 tokens per message for the chat-format overhead, following the usual
    # OpenAI-style rule of thumb. Close enough for budgeting.
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


# ------------------------------------------------------------------ rendering
def _shorten(text: Any, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def row_to_compact(record: dict[str, Any]) -> dict[str, Any]:
    """One dataset row -> the smallest dict that still describes it well."""
    out: dict[str, Any] = {}
    for key in PROMPT_FIELDS:
        value = record.get(key)
        if value is None or value == "" or value == []:
            continue
        if key == "description":
            value = _shorten(value)
        elif isinstance(value, float):
            value = round(value, 2)
        elif isinstance(value, list):
            value = value[:6]
        out[key] = value
    return out


def render_items(records: Iterable[dict[str, Any]]) -> str:
    """Render rows as one compact JSON object per line.

    JSONL in the prompt beats prose: it is unambiguous, dense, and the model can
    copy real names and prices out of it without paraphrasing them into fiction.
    """
    import json

    return "\n".join(json.dumps(row_to_compact(r), ensure_ascii=False) for r in records)


# ------------------------------------------------------------------- packing
@dataclass
class ContextPack:
    """The result of packing rows into a budget, with its own accounting."""

    text: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    budget: int = 0
    dropped_rows: int = 0
    dropped_fields: bool = False

    @property
    def used_ratio(self) -> float:
        return (self.tokens / self.budget) if self.budget else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "rows_included": len(self.rows),
            "rows_dropped": self.dropped_rows,
            "tokens": self.tokens,
            "budget": self.budget,
            "used_pct": round(self.used_ratio * 100, 1),
        }


def pack_rows(
    records: list[dict[str, Any]],
    budget_tokens: int,
    *,
    min_rows: int = 5,
) -> ContextPack:
    """Greedily add rows until the budget is reached.

    ``records`` is expected to be pre-sorted by relevance, so the first rows
    added are the ones we would most want to keep. If the budget is so small
    that even ``min_rows`` do not fit, descriptions are trimmed next; if it is
    still too small we return what fits and report the overflow honestly rather
    than silently truncating mid-JSON.
    """
    if budget_tokens <= 0:
        return ContextPack(text="", budget=budget_tokens, dropped_rows=len(records))

    accepted: list[dict[str, Any]] = []
    lines: list[str] = []
    used = 0

    for record in records:
        line = render_items([record])
        cost = estimate_tokens(line) + 1
        if used + cost > budget_tokens and len(accepted) >= min_rows:
            break
        if used + cost > budget_tokens and len(accepted) < min_rows:
            # Budget is very tight: strip the description and try once more.
            stripped = {k: v for k, v in record.items() if k != "description"}
            line = render_items([stripped])
            cost = estimate_tokens(line) + 1
            if used + cost > budget_tokens:
                break
        accepted.append(record)
        lines.append(line)
        used += cost

    return ContextPack(
        text="\n".join(lines),
        rows=accepted,
        tokens=used,
        budget=budget_tokens,
        dropped_rows=max(len(records) - len(accepted), 0),
    )


def build_context(
    frame: pd.DataFrame,
    budget_tokens: int,
    *,
    include_stats: bool = True,
    stats: dict[str, Any] | None = None,
) -> ContextPack:
    """Turn a filtered frame into a token-budgeted context block."""
    records = frame.to_dict(orient="records")
    pack = pack_rows(records, budget_tokens)

    if include_stats and stats:
        items = stats.get("items", stats.get("items_in_dataset", 0))
        cities = stats.get("cities", [])
        city_count = len(cities) if isinstance(cities, (list, tuple)) else cities
        header = (
            "DATASET SUMMARY (computed with pandas, do not recompute): "
            f"{items} rows in the dataset, {stats.get('items_selected', items)} in this "
            f"selection, {city_count} cities, average rating {stats.get('avg_rating', 0)}."
        )
        pack.text = f"{header}\nITEMS (one JSON object per row):\n{pack.text}"
        pack.tokens += estimate_tokens(header)

    return pack


# ----------------------------------------------------------------- budgeting UI
def budget_for_model(model: str, context_window: int | None = None) -> int:
    """Pick a prompt budget from a model's context window.

    Rule of thumb: reserve ~40% of the window for the *response*, the system
    prompt and the user's request, and give the data the rest. If the window is
    unknown, fall back to a conservative 3000-token data budget, which every
    model downloadable through Ollama can handle.
    """
    if context_window is None:
        known = {
            "llama3.2": 131072,
            "llama3.1": 131072,
            "llama3": 8192,
            "gemma3": 131072,
            "gemma2": 8192,
            "qwen2.5": 32768,
            "qwen3": 40960,
            "mistral": 32768,
            "phi3": 4096,
        }
        for prefix, window in known.items():
            if model.lower().startswith(prefix):
                context_window = window
                break
        else:
            context_window = 8192  # safe assumption

    data_budget = int(context_window * 0.45)
    return max(500, min(data_budget, 6000))
