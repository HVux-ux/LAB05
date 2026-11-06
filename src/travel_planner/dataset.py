"""Filtering, sorting and statistics over the travel dataset (the pandas layer).

This module is the only place that touches pandas directly. Keeping it here
means the UI and the prompt builder both work from the same, tested behaviour,
and the dataset can be swapped (sample CSV -> Kaggle JSON) without changing
either.

Typical use::

    ds = load_dataset("data/sample/travel_vn.csv")
    picks = ds.search(city="Hoi An", category="food", max_price_usd=10)
    stats = ds.city_summary()
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from .adapter import MappingProfile, load_mapped
from .loader import CANONICAL_FIELDS, DatasetError, read_any, normalise

DEFAULT_SAMPLE_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "sample"


@dataclass
class Filters:
    """A declarative description of what the user wants. All fields optional."""

    cities: Sequence[str] = ()
    region: str | None = None
    categories: Sequence[str] = ()
    max_price_usd: float | None = None
    min_rating: float | None = None
    month: int | None = None
    max_duration_hours: float | None = None
    tags: Sequence[str] = ()
    search: str = ""

    def describe(self) -> dict[str, Any]:
        """Only the constraints that are actually set — used in prompts/debug."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value not in (None, "", (), []):
                out[key] = list(value) if isinstance(value, (list, tuple)) else value
        return out


@dataclass
class TravelDataset:
    """A normalised travel dataset plus the query helpers used by the app."""

    frame: pd.DataFrame
    source_format: str = "unknown"
    filters: Filters = field(default_factory=Filters)

    # ------------------------------------------------------------------ basics
    @property
    def size(self) -> int:
        return len(self.frame)

    @property
    def cities(self) -> list[str]:
        return sorted(self.frame["city"].dropna().unique().tolist())

    @property
    def categories(self) -> list[str]:
        return sorted(self.frame["category"].dropna().unique().tolist())

    @property
    def regions(self) -> list[str]:
        return sorted(self.frame["region"].dropna().unique().tolist())

    @property
    def tags(self) -> list[str]:
        all_tags: set[str] = set()
        for tag_list in self.frame["tags"]:
            all_tags.update(tag_list if isinstance(tag_list, list) else [])
        return sorted(all_tags)

    # -------------------------------------------------------------- filtering
    def search(
        self,
        *,
        cities: Iterable[str] | None = None,
        region: str | None = None,
        categories: Iterable[str] | None = None,
        max_price_usd: float | None = None,
        min_rating: float | None = None,
        month: int | None = None,
        max_duration_hours: float | None = None,
        tags: Iterable[str] | None = None,
        text: str = "",
        free_text: bool = True,
    ) -> pd.DataFrame:
        """Filter the frame and return a *copy* sorted by usefulness.

        Sorting here is deliberate: the rows that survive are later packed into
        a token budget, and the ones that should survive truncation first are
        the best-rated, most-reviewed, cheapest ones.
        """
        frame = self.frame
        mask = pd.Series(True, index=frame.index)

        if cities:
            cities = [c for c in cities if c]
            if cities:
                mask &= frame["city"].isin(cities)
        if region and region != "any":
            mask &= frame["region"].str.lower() == region.lower()
        if categories:
            categories = [c for c in categories if c]
            if categories:
                mask &= frame["category"].isin(categories)
        if max_price_usd is not None:
            mask &= frame["price_usd"] <= float(max_price_usd)
        if min_rating is not None:
            mask &= frame["rating"] >= float(min_rating)
        if max_duration_hours is not None:
            mask &= frame["duration_hours"] <= float(max_duration_hours)
        if month is not None:
            mask &= frame["best_months"].map(
                lambda months: month in months if isinstance(months, list) else True
            )
        if tags:
            wanted = {t.lower() for t in tags if t}
            if wanted:
                mask &= frame["tags"].map(
                    lambda item_tags: bool(
                        wanted & {str(t).lower() for t in item_tags}
                    ) if isinstance(item_tags, list) else False
                )

        selected = frame[mask]

        if text and free_text:
            needle = text.strip()
            if needle:
                haystack = (
                    selected["name"].fillna("") + " "
                    + selected["description"].fillna("") + " "
                    + selected["city"].fillna("") + " "
                    + selected["subcategory"].fillna("") + " "
                    + selected["tags"].map(lambda t: " ".join(t) if isinstance(t, list) else "")
                )
                selected = selected[haystack.str.contains(needle, case=False, regex=False)]

        # Sort by a simple "how good is this row" score, then by rating.
        scored = selected.assign(
            _score=(
                selected["rating"] * 20
                + selected["reviews"].clip(upper=25000) / 5000
                - selected["price_usd"] / 25
            )
        )
        return scored.sort_values(["_score", "rating"], ascending=False).drop(columns="_score")

    def describe_filters(self, filters: Filters) -> dict[str, Any]:
        return filters.describe()

    # ------------------------------------------------------------ statistics
    def city_summary(self, cities: Iterable[str] | None = None) -> pd.DataFrame:
        """Per-city counts, average rating and cost — the pandas `groupby` demo.

        This is the kind of aggregation the LLM should never be asked to do:
        pandas is exact, free and instant, whereas an LLM estimating a mean from
        40 rows in its context window is just guessing.
        """
        frame = self.frame
        if cities:
            frame = frame[frame["city"].isin(list(cities))]
        if frame.empty:
            return pd.DataFrame(
                columns=["city", "region", "items", "attractions", "avg_rating", "avg_price_usd"]
            )

        grouped = frame.groupby(["city", "region"], as_index=False).agg(
            items=("id", "count"),
            attractions=("category", lambda s: int((s == "landmark").sum())),
            avg_rating=("rating", "mean"),
            avg_price_usd=("price_usd", "mean"),
            total_reviews=("reviews", "sum"),
        )
        grouped["avg_rating"] = grouped["avg_rating"].round(2)
        grouped["avg_price_usd"] = grouped["avg_price_usd"].round(2)
        return grouped.sort_values("items", ascending=False).reset_index(drop=True)

    def category_summary(self) -> pd.DataFrame:
        grouped = self.frame.groupby("category", as_index=False).agg(
            items=("id", "count"),
            avg_rating=("rating", "mean"),
            min_price_usd=("price_usd", "min"),
            max_price_usd=("price_usd", "max"),
        )
        grouped["avg_rating"] = grouped["avg_rating"].round(2)
        return grouped.sort_values("items", ascending=False).reset_index(drop=True)

    def budget_estimate(self, cities: Iterable[str], days: int, style: str = "mid") -> dict[str, float]:
        """Rough per-person budget for a set of cities.

        Uses the accommodation and food rows for those cities as the price
        source, with the style ('budget' | 'mid' | 'luxury') picking a quantile
        rather than the cheapest possible option.
        """
        frame = self.frame[self.frame["city"].isin(list(cities))]
        quantile = {"budget": 0.25, "mid": 0.5, "luxury": 0.9}.get(style, 0.5)

        stays = frame[frame["category"] == "accommodation"]["price_usd"]
        meals = frame[frame["category"] == "food"]["price_usd"]

        nightly = float(stays.quantile(quantile)) if not stays.empty else 40.0
        per_meal = float(meals.quantile(quantile)) if not meals.empty else 8.0

        return {
            "nights": max(days - 1, 1),
            "nightly_usd": round(nightly, 2),
            "lodging_usd": round(nightly * max(days - 1, 1), 2),
            "food_usd": round(per_meal * 3 * days, 2),
            "activities_usd": 0.0,  # filled in by the planner from selected rows
            "days": days,
        }

    def stats(self) -> dict[str, Any]:
        """Headline numbers for the dataset itself (shown in the app sidebar)."""
        return {
            "items": int(len(self.frame)),
            "cities": int(self.frame["city"].nunique()),
            "categories": int(self.frame["category"].nunique()),
            "avg_rating": round(float(self.frame["rating"].mean()), 2) if len(self.frame) else 0.0,
            "free_items": int((self.frame["price_usd"] == 0).sum()),
            "format": self.source_format,
        }

    # ------------------------------------------------------------------ export
    def to_records(self, frame: pd.DataFrame | None = None) -> list[dict]:
        frame = self.frame if frame is None else frame
        columns = [c for c in CANONICAL_FIELDS if c in frame.columns]
        return frame[columns].to_dict(orient="records")


# --------------------------------------------------------------------- loading
def load_dataset(
    path: str | pathlib.Path | None = None,
    *,
    profile: MappingProfile | str | pathlib.Path | None = None,
) -> TravelDataset:
    """Load a dataset, defaulting to the bundled CSV sample.

    Passing ``profile`` (a MappingProfile or a path to one) enables the adapter
    path for third-party files with unfamiliar column names.
    """
    if path is None:
        path = DEFAULT_SAMPLE_DIR / "travel_vn.csv"
    path = pathlib.Path(path)

    if profile is not None:
        resolved = (
            MappingProfile.load(profile)
            if isinstance(profile, (str, pathlib.Path))
            else profile
        )
        frame = load_mapped(path, resolved)
        return TravelDataset(frame=frame, source_format=f"mapped:{resolved.name}")

    rows, fmt = read_any(path)
    frame = normalise(rows, source=path.name)
    if frame.empty:
        raise DatasetError(f"{path}: no usable rows after normalisation")
    return TravelDataset(frame=frame, source_format=fmt)
