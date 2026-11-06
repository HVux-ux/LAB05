"""Ollama client, plus an offline mock generator for demos and tests.

Ollama exposes a local HTTP API on ``http://localhost:11434``; nothing leaves the
machine. This module talks to it with plain ``requests`` so the app carries no
framework dependency and the failure modes (server down, model not pulled) can be
reported in plain language instead of a stack trace.

``MockLLM`` produces a deterministic, structure-compliant itinerary from the same
context block. It exists so the pipeline is testable without a multi-gigabyte
model download, and so the app can be demonstrated anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from .context import estimate_messages_tokens
from .prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_HOST = "http://localhost:11434"

#: Approximate exchange rate used only for display conversions.
VND_PER_USD = 25000.0


class LLMError(RuntimeError):
    """Any failure talking to the model, with a message meant for the UI."""


@dataclass
class LLMConfig:
    model: str = "gemma3:4b"
    host: str = DEFAULT_HOST
    temperature: float = 0.4
    num_ctx: int = 8192
    timeout: int = 180


# =========================================================== Ollama (live mode)
class OllamaClient:
    """Minimal Ollama chat client."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    # ------------------------------------------------------------------ health
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.config.host}. Start it with `ollama serve`, "
                "or run the app in mock mode (the default)."
            ) from exc
        return sorted(m["name"] for m in response.json().get("models", []))

    def ensure_model(self, model: str | None = None) -> None:
        model = model or self.config.model
        available = self.list_models()
        base = model.split(":")[0]
        if not any(m == model or m.split(":")[0] == base for m in available):
            raise LLMError(
                f"Model {model!r} is not installed. Run `ollama pull {model}` first. "
                f"Installed: {', '.join(available) or 'none'}"
            )

    # -------------------------------------------------------------------- chat
    def chat(self, messages: list[dict[str, str]], *, stream: bool = False):
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        try:
            response = requests.post(
                f"{self.config.host}/api/chat", json=payload,
                timeout=self.config.timeout, stream=stream,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Lost the connection to Ollama at {self.config.host}. Is `ollama serve` running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                f"Ollama did not answer within {self.config.timeout}s. "
                "A smaller model or a smaller context budget will be faster."
            ) from exc
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if not stream:
            return response.json().get("message", {}).get("content", "")
        return self._iter_stream(response)

    def _iter_stream(self, response) -> Iterator[str]:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise LLMError(str(chunk["error"]))
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break


# ============================================================ Mock (offline mode)
class MockLLM:
    """Deterministic stand-in that follows the same output contract.

    It reads the JSONL context block, allocates days across the cities that have
    the most to offer, and emits the required sections. Enough to verify the whole
    pipeline end to end (filters -> context pack -> prompt -> answer) with no model.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.supports_streaming = False

    # ------------------------------------------------------------------ health
    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["mock"]

    def ensure_model(self, model: str | None = None) -> None:
        return None

    # ----------------------------------------------------------------- parsing
    @staticmethod
    def _parse_items(context_block: str) -> list[dict[str, Any]]:
        items = []
        for line in context_block.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    @staticmethod
    def _parse_constraints(prompt: str) -> dict[str, Any]:
        match = re.search(r"CONSTRAINTS\n(\{[^\n]*\})", prompt)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _split_by_city(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
        order: list[str] = []
        buckets: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            city = item.get("city") or "Unknown"
            if city not in buckets:
                buckets[city] = []
                order.append(city)
            buckets[city].append(item)
        return [(city, buckets[city]) for city in order]

    @staticmethod
    def _allocate_days(
        groups: list[tuple[str, list[dict[str, Any]]]], days: int
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Spread ``days`` over cities, weighted by how much each one offers.

        With more cities than days, only the first ``days`` are visited: a plan
        that teleports between five cities in three days is not a plan.
        """
        groups = [(c, rows) for c, rows in groups if rows]
        if not groups:
            return []
        if len(groups) > days:
            return [(c, rows) for c, rows in groups[:days]]

        weight = {c: max(len(rows), 1) for c, rows in groups}
        total_weight = sum(weight.values())
        allocation = {c: max(1, round(days * weight[c] / total_weight)) for c, _ in groups}

        while sum(allocation.values()) > days:
            donor = max(allocation, key=lambda c: allocation[c] if allocation[c] > 1 else 0)
            if allocation[donor] <= 1:
                break
            allocation[donor] -= 1
        busiest = max(allocation, key=lambda c: weight[c])
        while sum(allocation.values()) < days:
            allocation[busiest] += 1

        plan: list[tuple[str, list[dict[str, Any]]]] = []
        for city, rows in groups:
            plan.extend([(city, rows)] * allocation[city])
        return plan

    # -------------------------------------------------------------- generation
    def generate(self, prompt: str, days: int = 3) -> str:
        items = self._parse_items(prompt)
        constraints = self._parse_constraints(prompt)
        days = max(1, int(constraints.get("days", days) or days))
        style = {"budget": "budget", "mid": "mid-range", "luxury": "luxury"}.get(
            str(constraints.get("style") or "mid"), "mid-range"
        )
        style_key = constraints.get("style") or "mid"

        stays = [i for i in items if i.get("category") == "accommodation"]
        sights = [i for i in items if i.get("category") in ("landmark", "experience", "food")]
        out: list[str] = ["## Overview"]

        if not items:
            return "\n".join(out + [
                "The dataset returned no rows for these constraints, so no plan can be grounded. "
                "Widen the price ceiling, the month or the city selection.",
                "", "## Day-by-day plan", "Nothing to schedule.", "",
                "## Where to stay", "No accommodation rows matched.", "",
                "## Budget", "| item | usd |", "| --- | --- |", "| total | 0.00 |", "",
                "## Gaps", "- The filters matched zero rows in the dataset.",
            ])

        groups = self._split_by_city(sights)
        day_plan = self._allocate_days(groups, days)
        visited = list(dict.fromkeys(city for city, _ in day_plan))

        out.append(
            f"{days} days covering {', '.join(visited) or 'the selected area'}, built only from "
            f"the {len(items)} dataset rows that matched the constraints. Each day keeps one "
            "anchor sight in the morning, something nearby in the afternoon, and dinner from the "
            "food rows for that city."
        )

        cursor: dict[str, int] = {city: 0 for city, _ in groups}
        used: set[str] = set()
        activity_total = 0.0

        def take(city: str, rows: list[dict], wanted: tuple[str, ...],
                 *, allow_repeat: bool = False) -> dict | None:
            """Next unused row of the wanted kinds; optionally allow a revisit."""
            for offset in range(len(rows)):
                index = (cursor[city] + offset) % len(rows)
                candidate = rows[index]
                if candidate.get("category") not in wanted:
                    continue
                if candidate.get("name") in used:
                    if allow_repeat:
                        cursor[city] = index + 1
                        return candidate
                    continue
                used.add(candidate["name"])
                cursor[city] = index + 1
                return candidate
            return None

        def line_for(row: dict) -> str:
            return (
                f"- {row.get('name')} ({row.get('duration_hours', '?')} h, "
                f"${row.get('price_usd', 0)})"
            )

        out += ["", "## Day-by-day plan"]
        for number, (city, rows) in enumerate(day_plan, start=1):
            out += ["", f"### Day {number} - {city}", "", "**Morning**"]
            anchor = (
                take(city, rows, ("landmark", "experience"))
                or take(city, rows, ("food",))
                or take(city, rows, ("landmark", "experience", "food"), allow_repeat=True)
            )
            if anchor:
                activity_total += float(anchor.get("price_usd") or 0)
                out.append(line_for(anchor))
            else:
                out.append("- A slow morning in the neighbourhood.")

            out += ["", "**Afternoon**"]
            second = take(city, rows, ("experience", "landmark")) or take(
                city, rows, ("experience", "landmark", "food"), allow_repeat=True
            )
            if second:
                activity_total += float(second.get("price_usd") or 0)
                out.append(line_for(second))
            else:
                out.append("- Walk the old quarter and stop for coffee.")

            out += ["", "**Evening**"]
            dinner = take(city, rows, ("food",), allow_repeat=True)
            out.append(
                f"- Dinner at {dinner.get('name')} (${dinner.get('price_usd', 0)})"
                if dinner else "- Street food near the accommodation."
            )

        out += ["", "## Where to stay"]
        if stays:
            ranked = sorted(stays, key=lambda s: float(s.get("price_usd") or 0))
            pick = {"budget": ranked[0], "luxury": ranked[-1]}.get(
                style_key, ranked[len(ranked) // 2]
            )
            nightly = float(pick.get("price_usd") or 0)
            out.append(
                f"- **{pick.get('name')}** in {pick.get('city')} - ${nightly:.2f} per night, "
                f"rated {pick.get('rating', 'n/a')}. Chosen from {len(stays)} accommodation rows "
                f"for a {style} trip."
            )
        else:
            nightly = 0.0
            out.append("- No accommodation row matched these filters.")

        food_rows = [i for i in items if i.get("category") == "food"]
        per_meal = (
            sum(float(i.get("price_usd") or 0) for i in food_rows) / len(food_rows)
            if food_rows else 8.0
        )
        nights = max(days - 1, 1)
        lodging = nightly * nights
        food_est = per_meal * 3 * days
        total = lodging + food_est + activity_total
        out += [
            "", "## Budget", "",
            "| item | usd |", "| --- | --- |",
            f"| lodging ({nights} nights x ${nightly:.2f}) | {lodging:.2f} |",
            f"| food ({days} days x 3 meals x ${per_meal:.2f}) | {food_est:.2f} |",
            f"| activities | {activity_total:.2f} |",
            f"| **total** | **{total:.2f}** |",
            f"| per day | {total / days:.2f} |",
        ]

        out += ["", "## Gaps"]
        if not stays:
            out.append("- No accommodation found in the selected cities.")
        if len(used) < len(sights):
            out.append(
                f"- {len(sights) - len(used)} retrieved rows were not scheduled: the dataset "
                "holds more options than these days can fit."
            )
        budget = constraints.get("daily_budget_usd")
        if budget and total / days > float(budget):
            out.append(
                f"- The plan costs ${total / days:.2f}/day, above the ${budget} requested; "
                "drop a paid activity or choose a cheaper stay."
            )
        month = constraints.get("month")
        if month:
            out.append(
                f"- Month {month} was used as a hint only: confirm opening hours and expect "
                "regional weather differences."
            )
        out.append("- Mock mode: wording is templated, not composed. Run Ollama for real prose.")
        return "\n".join(out)

    def chat(self, messages: list[dict[str, str]], *, stream: bool = False) -> str:
        prompt = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        return self.generate(prompt)


# ==================================================================== messaging
def build_messages(
    *,
    request: str,
    context_block: str,
    days: int,
    travellers: int,
    budget_usd: float | None,
    style: str,
    month: int | None,
    stats: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> list[dict[str, str]]:
    """Assemble the chat payload: a fixed system prompt plus one user message."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                request=request,
                days=days,
                travellers=travellers,
                budget_usd=budget_usd,
                style=style,
                month=month,
                context_block=context_block,
                stats=stats,
                notes=notes,
            ),
        },
    ]
