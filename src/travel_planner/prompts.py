"""Prompt templates.

Two rules shape these prompts:

1. **Facts come from the context block, never from the model's memory.** Models
   happily invent a hotel that does not exist; instructing them to use only the
   supplied rows, and to say when the rows are insufficient, is what keeps the
   output checkable.
2. **The model composes, pandas computes.** Totals, averages and rankings are
   already calculated and passed in, so the model is never asked to do
   arithmetic over a table.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a careful Vietnam travel planner.

Rules you must follow:
- Use ONLY the places listed in the CONTEXT section. Never invent a name, price, address or rating.
- If the context does not contain something the traveller needs, say so explicitly under "Gaps".
- Prices are in USD per person; convert to VND only when asked.
- Do not recompute averages or totals: they are already computed and given to you.
- Keep to the requested number of days and the daily budget.
- Output Markdown only. Do not wrap the whole answer in a code fence.

Structure your answer exactly as:
## Overview
Two or three sentences describing the shape of the trip.
## Day-by-day plan
One "### Day N - City" section per day. Each day: a morning / afternoon / evening
list, with the real place names from the context and a running cost.
## Where to stay
One option from the accommodation rows, with its price.
## Budget
A short table: lodging, food, activities, total, and the total per day.
## Gaps
Bullet points for anything requested that the dataset could not cover.
"""


def build_user_prompt(
    *,
    request: str,
    days: int,
    travellers: int,
    budget_usd: float | None,
    style: str,
    month: int | None,
    context_block: str,
    stats: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Assemble the user message: the request, then the retrieved context."""
    constraints = {
        "days": days,
        "travellers": travellers,
        "daily_budget_usd": budget_usd,
        "style": style,
        "month": month,
    }
    lines = [
        "TRAVELLER REQUEST",
        request.strip() or "A balanced first trip to Vietnam.",
        "",
        "CONSTRAINTS",
        json.dumps({k: v for k, v in constraints.items() if v is not None}, ensure_ascii=False),
    ]

    if notes:
        lines += ["", "PRE-COMPUTED BY PANDAS (trust these numbers)", *[f"- {n}" for n in notes]]

    if stats:
        lines += [
            "",
            "DATASET STATS",
            json.dumps(stats, ensure_ascii=False),
        ]

    lines += [
        "",
        "CONTEXT - the only places you may use",
        context_block or "(empty: nothing in the dataset matched the constraints)",
        "",
        "Write the itinerary now, following the required structure.",
    ]
    return "\n".join(lines)


def build_repair_prompt(answer: str) -> str:
    """A second pass used when the model ignored the required sections."""
    return (
        "Your previous answer did not follow the required section structure. "
        "Rewrite it with exactly these headings: Overview, Day-by-day plan, Where to stay, "
        "Budget, Gaps. Keep the same places and do not add new ones.\n\n"
        f"PREVIOUS ANSWER:\n{answer}"
    )
