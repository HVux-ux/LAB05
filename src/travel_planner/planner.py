"""Orchestration: user request -> pandas selection -> context pack -> LLM -> itinerary.

This is the module that implements the Exercise A answer in code. The flow is
deliberately explicit and inspectable at every step, because the interesting
question ("how do you fit a dataset into a context window?") is answered by the
intermediate artefacts, not just the final prose:

    Filters  ->  DataFrame      (what the user asked for)
             ->  ContextPack    (what the model is allowed to see, and its cost)
             ->  messages       (the exact prompt)
             ->  answer         (Markdown itinerary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from .context import ContextPack, budget_for_model, build_context, estimate_messages_tokens
from .dataset import Filters, TravelDataset
from .llm import LLMConfig, LLMError, MockLLM, OllamaClient, build_messages


@dataclass
class PlanRequest:
    """Everything the user chose in the UI, in one place."""

    text: str = ""
    cities: Sequence[str] = ()
    region: str | None = None
    categories: Sequence[str] = ()
    days: int = 3
    travellers: int = 2
    daily_budget_usd: float | None = None
    style: str = "mid"
    month: int | None = None
    min_rating: float | None = None
    tags: Sequence[str] = ()
    max_items: int = 40

    def to_filters(self) -> Filters:
        # No month filter while selecting: a place that is merely off-season is
        # still a candidate; the month is passed to the model as a hint instead.
        return Filters(
            cities=self.cities,
            region=self.region,
            categories=self.categories,
            min_rating=self.min_rating,
            tags=self.tags,
            search=self.text if len(self.text.split()) <= 6 else "",
        )


@dataclass
class PlanResult:
    answer: str
    pack: ContextPack
    selection: pd.DataFrame
    stats: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    model: str = "mock"
    budget: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def debug(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "context": self.pack.summary(),
            "dataset_stats": self.stats,
            "pandas_notes": self.notes,
            "budget": self.budget,
            "selected_items": len(self.selection),
        }


def select_items(dataset: TravelDataset, request: PlanRequest) -> pd.DataFrame:
    """Filter, then make sure every category needed for a plan survives.

    A pure "best rated" ranking tends to return only landmarks. A usable plan
    also needs somewhere to sleep and something to eat, so the selection is
    topped up per category before packing.
    """
    filters = request.to_filters()
    selected = dataset.search(
        cities=filters.cities,
        region=filters.region,
        categories=filters.categories,
        min_rating=filters.min_rating,
        tags=filters.tags,
        text=filters.search,
    )

    if selected.empty:
        # Progressively relax: drop tags, then rating, then city restriction.
        selected = dataset.search(
            cities=filters.cities, categories=filters.categories, region=filters.region
        )
    if selected.empty:
        selected = dataset.search(region=filters.region)

    balance = {"landmark": 0.35, "experience": 0.25, "food": 0.2, "accommodation": 0.15, "transport": 0.05}
    quota = {cat: max(1, int(request.max_items * share)) for cat, share in balance.items()}

    parts = []
    for category, limit in quota.items():
        subset = selected[selected["category"] == category]
        if not subset.empty:
            parts.append(subset.head(limit))
    if parts:
        combined = pd.concat(parts)
        # Keep relevance order: re-sort the balanced set by rating then reviews.
        selected = combined.drop_duplicates(subset=["id"]).sort_values(
            ["rating", "reviews"], ascending=False
        )
    return selected.head(request.max_items).reset_index(drop=True)


def run_plan(
    dataset: TravelDataset,
    request: PlanRequest,
    *,
    llm: OllamaClient | MockLLM | None = None,
    config: LLMConfig | None = None,
    context_window: int | None = None,
) -> PlanResult:
    """Build a plan. Raises LLMError if the model cannot be reached."""
    config = config or (llm.config if llm else LLMConfig())
    llm = llm or OllamaClient(config)

    selection = select_items(dataset, request)
    budget_tokens = budget_for_model(config.model, context_window)

    stats = {
        "items_in_dataset": dataset.size,
        "items_selected": int(len(selection)),
        "cities": sorted(selection["city"].unique().tolist()) if not selection.empty else [],
        "avg_rating": round(float(selection["rating"].mean()), 2) if not selection.empty else 0.0,
        "avg_price_usd": round(float(selection["price_usd"].mean()), 2) if not selection.empty else 0.0,
    }

    notes: list[str] = []
    if not selection.empty:
        summary = dataset.city_summary(cities=request.cities or None)
        if not summary.empty:
            top = summary.head(4)
            notes.append(
                "city_summary (pandas groupby): "
                + "; ".join(
                    f"{row.city}: {row.items} items, avg rating {row.avg_rating}"
                    for row in top.itertuples()
                )
            )
        est = dataset.budget_estimate(
            cities=request.cities or stats["cities"] or dataset.cities,
            days=request.days,
            style=request.style,
        )
        notes.append(
            f"budget_estimate ({request.style}): ${est['nightly_usd']}/night, "
            f"lodging ${est['lodging_usd']} for {est['nights']} nights, "
            f"food ${est['food_usd']} for {request.days} days"
        )
    else:
        notes.append("No rows matched, so no statistics were computed.")

    pack = build_context(selection, budget_tokens, stats=stats)

    messages = build_messages(
        request=request.text,
        context_block=pack.text,
        days=request.days,
        travellers=request.travellers,
        budget_usd=request.daily_budget_usd,
        style=request.style,
        month=request.month,
        stats=stats,
        notes=notes,
    )
    prompt_tokens = estimate_messages_tokens(messages)

    answer = llm.chat(messages)

    return PlanResult(
        answer=answer,
        pack=pack,
        selection=selection,
        stats=stats,
        notes=notes,
        prompt_tokens=prompt_tokens,
        model=config.model if not isinstance(llm, MockLLM) else "mock",
        budget={"context_window": context_window, "data_budget_tokens": budget_tokens},
    )


def compare_prompting_strategies(
    dataset: TravelDataset, request: PlanRequest
) -> list[dict[str, Any]]:
    """Quantify the three obvious designs, for the report and the app's explainer.

    This turns the Exercise A "hint" into a measurement: what would it cost to
    put the whole dataset in the prompt, versus a summary, versus retrieval?
    """
    from .context import estimate_tokens, render_items

    all_tokens = estimate_tokens(render_items(dataset.frame.to_dict(orient="records")))
    selection = select_items(dataset, request)
    retrieved_tokens = estimate_tokens(render_items(selection.to_dict(orient="records")))
    # A fixed-size summary is roughly constant regardless of dataset size.
    summary_tokens = 220

    rows = [
        {
            "strategy": "Whole dataset in prompt",
            "tokens": all_tokens,
            "scales_with_data": "yes, linearly",
            "fits_8k_model": all_tokens <= 3600,
            "grounded": "yes",
            "weakness": "does not fit; slow; detail gets ignored in the middle",
        },
        {
            "strategy": "Pre-computed summary",
            "tokens": summary_tokens,
            "scales_with_data": "no, constant",
            "fits_8k_model": True,
            "grounded": "partly",
            "weakness": "cannot answer specific questions; averages hide the options",
        },
        {
            "strategy": "Filter + retrieve + pack (this app)",
            "tokens": retrieved_tokens,
            "scales_with_data": "no, capped by budget",
            "fits_8k_model": retrieved_tokens <= 3600,
            "grounded": "yes",
            "weakness": "recall depends on the filters; needs good ranking",
        },
    ]
    return rows
