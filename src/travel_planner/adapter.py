"""Adapter layer: map an arbitrary dataset (e.g. a Kaggle download) to our schema.

The sample data already uses the canonical column names, but a real Kaggle CSV
almost never does. Instead of asking the user to hand-edit a spreadsheet, a
small JSON *mapping profile* declares how the source columns map onto ours::

    {
      "name": "kaggle-vietnam-tourism",
      "columns": {
        "id":          "place_id",
        "name":        "place_name",
        "city":        "province",
        "category":    "type",
        "price_usd":   "avg_price",
        "rating":      "rating_5",
        "latitude":    "lat",
        "longitude":   "lng"
      },
      "constants": {"region": "unknown"},
      "city_aliases": {"Ha Noi": "Hanoi", "TP. Ho Chi Minh": "Ho Chi Minh City"},
      "category_aliases": {"restaurant": "food", "hotel": "accommodation"}
    }

Anything not declared in ``columns`` falls back to the built-in synonyms in
``loader.normalise``, so a partial profile is fine.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd

from .loader import normalise, read_any

DEFAULT_PROFILE_DIR = pathlib.Path(__file__).resolve().parents[2] / "config"


class MappingProfile:
    """A declared correspondence between source columns and canonical fields."""

    def __init__(self, data: dict[str, Any]):
        self.name: str = data.get("name", "unnamed")
        self.columns: dict[str, str] = data.get("columns", {}) or {}
        self.constants: dict[str, Any] = data.get("constants", {}) or {}
        self.city_aliases: dict[str, str] = {
            k.lower(): v for k, v in (data.get("city_aliases", {}) or {}).items()
        }
        self.category_aliases: dict[str, str] = {
            k.lower(): v for k, v in (data.get("category_aliases", {}) or {}).items()
        }

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "MappingProfile":
        path = pathlib.Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid mapping profile ({exc})") from exc
        return cls(data)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename source columns to canonical names and apply alias tables."""
        rename = {src: dst for dst, src in self.columns.items() if src in frame.columns}
        frame = frame.rename(columns=rename)

        for field, value in self.constants.items():
            frame[field] = value

        if self.city_aliases and "city" in frame.columns:
            frame["city"] = frame["city"].map(
                lambda c: self.city_aliases.get(str(c).strip().lower(), c)
            )
        if self.category_aliases and "category" in frame.columns:
            frame["category"] = frame["category"].map(
                lambda c: self.category_aliases.get(str(c).strip().lower(), str(c).strip().lower())
            )
        return frame


def load_mapped(path: str | pathlib.Path, profile: MappingProfile | None = None) -> pd.DataFrame:
    """Read any supported file and return it in the canonical schema."""
    rows, fmt = read_any(path)
    frame = pd.DataFrame.from_records(rows)
    if profile is not None:
        frame = profile.apply(frame)
    return normalise(frame.to_dict(orient="records"), source=f"{pathlib.Path(path).name}:{fmt}")


def list_profiles(directory: str | pathlib.Path = DEFAULT_PROFILE_DIR) -> list[pathlib.Path]:
    directory = pathlib.Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.mapping.json"))


def autodetect_profile(path: str | pathlib.Path) -> MappingProfile | None:
    """Return the first bundled profile whose declared columns exist in the file.

    This is a convenience for the UI: drop in a supported Kaggle file and it
    loads without configuration. It is a heuristic, not magic — a bad guess is
    visible immediately in the preview table.
    """
    try:
        rows, _ = read_any(path)
    except Exception:
        return None
    if not rows:
        return None
    available = {str(c).lower() for c in rows[0]}

    best, best_score = None, 0
    for profile_path in list_profiles():
        profile = MappingProfile.load(profile_path)
        declared = [c for c in profile.columns.values() if str(c).lower() in available]
        if len(declared) > best_score:
            best, best_score = profile, len(declared)

    # Require at least a name column plus one more before trusting the profile.
    return best if best_score >= 2 else None
