"""Read travel datasets from CSV, JSON or JSONL into one normalised frame.

Each of the three formats has a different shape, and each needs a different
strategy:

===========  ==========================================  =====================
Format       Physical shape                              How it is read
===========  ==========================================  =====================
CSV          flat rows, no nesting, no types             ``pd.read_csv``
             -> lists must be jammed into one cell
JSON         one document, often an object wrapping an   ``json.load`` then
             array; values keep their real types          locate the array
JSONL        no wrapper at all: one object per line, so  line-by-line
             it can be streamed and appended to           ``json.loads``
===========  ==========================================  =====================

The output of this module is always the same canonical schema (see
``CANONICAL_FIELDS``) so nothing downstream has to care which file it came from.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable

import pandas as pd

#: The one schema the whole application speaks.
CANONICAL_FIELDS: list[str] = [
    "id", "name", "city", "region", "category", "subcategory", "description",
    "tags", "price_vnd", "price_usd", "rating", "reviews", "duration_hours",
    "best_months", "address", "latitude", "longitude", "source",
]

#: Columns that hold a list of values rather than a scalar.
LIST_FIELDS: tuple[str, ...] = ("tags", "best_months")

#: Numeric columns, with the fallback used when a value is missing or garbage.
NUMERIC_FIELDS: dict[str, float] = {
    "price_vnd": 0.0,
    "price_usd": 0.0,
    "rating": 0.0,
    "reviews": 0.0,
    "duration_hours": 0.0,
    "latitude": 0.0,
    "longitude": 0.0,
}

#: Values inside a CSV list cell can be separated by any of these.
LIST_SEPARATORS = ("|", ";", ",")

#: Canonical categories, and the words third-party datasets use for them.
CATEGORY_ALIASES: dict[str, str] = {
    "landmark": "landmark", "attraction": "landmark", "attractions": "landmark",
    "sightseeing": "landmark", "tourist attraction": "landmark", "monument": "landmark",
    "museum": "landmark", "temple": "landmark", "pagoda": "landmark", "viewpoint": "landmark",
    "food": "food", "restaurant": "food", "cafe": "food", "café": "food",
    "food & drink": "food", "street food": "food",
    "accommodation": "accommodation", "hotel": "accommodation", "hostel": "accommodation",
    "resort": "accommodation", "homestay": "accommodation", "guesthouse": "accommodation",
    "experience": "experience", "activity": "experience", "tour": "experience",
    "tourist attraction activity": "experience", "entertainment": "experience",
    "transport": "transport", "transportation": "transport",
    "airport transfer": "transport", "transfer": "transport",
}


def canonical_category(value: Any) -> str:
    """Map a free-text category onto the one of five canonical values."""
    text = str(value or "").strip().lower()
    if text in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[text]
    for alias, canonical in CATEGORY_ALIASES.items():
        if alias and alias in text:
            return canonical
    return "experience"


class DatasetError(RuntimeError):
    """Raised when a file cannot be turned into the canonical schema."""


# ---------------------------------------------------------------- list helpers
def split_list_value(value: Any, *, separators: Iterable[str] = LIST_SEPARATORS,
                     cast: type | None = None) -> list:
    """Normalise a list-ish cell into a real Python list.

    Handles all the shapes the formats throw at us::

        "a|b|c"      -> ["a", "b", "c"]        (our CSV)
        "a, b"       -> ["a", "b"]             (a Kaggle CSV)
        ["a", "b"]   -> ["a", "b"]             (JSON / JSONL)
        "3;4;5"      -> [3, 4, 5]  with cast=int
        ""/None/NaN  -> []
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        # Strip a JSON-looking wrapper, e.g. '["a","b"]' stored in a CSV cell.
        if text.startswith("[") and text.endswith("]"):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    parts = loaded
                else:
                    parts = [loaded]
            except json.JSONDecodeError:
                parts = [text]
        else:
            parts = [text]
            for sep in separators:
                if sep in text:
                    parts = text.split(sep)
                    break

    cleaned: list = []
    for part in parts:
        item = part.strip() if isinstance(part, str) else part
        if item == "" or item is None:
            continue
        if cast is not None:
            try:
                item = cast(item)
            except (TypeError, ValueError):
                continue
        cleaned.append(item)
    return cleaned


def _to_number(value: Any, fallback: float = 0.0) -> float:
    """Coerce to float, tolerating '1,200', '$58', '4.6/5' and empty cells."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "").replace("₫", "")
    if "/" in text:
        text = text.split("/")[0].strip()
    try:
        return float(text)
    except ValueError:
        return fallback


# ---------------------------------------------------------------- file readers
def _locate_records(document: Any) -> list[dict]:
    """Find the row array inside a decoded JSON document.

    Datasets on Kaggle are posted in many wrapper shapes: a bare array, or an
    object with "data"/"items"/"records"/"results", or an object keyed by city.
    Rather than guess one, look for the first array-of-objects we can find.
    """
    if isinstance(document, list):
        return [row for row in document if isinstance(row, dict)]

    if isinstance(document, dict):
        for key in ("items", "data", "records", "results", "rows", "places"):
            candidate = document.get(key)
            if isinstance(candidate, list) and any(isinstance(r, dict) for r in candidate):
                return [row for row in candidate if isinstance(row, dict)]
        # Fallback: object-of-objects, e.g. {"hanoi": {...}, "hue": {...}}
        values = [v for v in document.values() if isinstance(v, (dict, list))]
        if values and all(isinstance(v, dict) for v in values):
            return list(document.values())

    raise DatasetError("could not find an array of records in this JSON document")


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    """CSV: everything arrives as text; lists are packed into one cell."""
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # pandas raises several unrelated types
        raise DatasetError(f"{path}: unreadable CSV ({exc})") from exc
    return frame.to_dict(orient="records")


def read_json_rows(path: pathlib.Path) -> list[dict]:
    """JSON: one whole document in memory, so it must be parsed as a unit."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON ({exc})") from exc
    return _locate_records(document)


def read_jsonl_rows(path: pathlib.Path) -> list[dict]:
    """JSONL: one independent object per line, read line by line.

    A malformed line is skipped rather than aborting the whole file — that is
    the operational advantage of JSONL for logs and scraped data.
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSONL line ({exc})") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


READERS = {
    ".csv": read_csv_rows,
    ".json": read_json_rows,
    ".jsonl": read_jsonl_rows,
    # .ndjson is the same format under a different name
    ".ndjson": read_jsonl_rows,
}


def read_any(path: str | pathlib.Path) -> tuple[list[dict], str]:
    """Read any supported file, returning ``(rows, format_name)``."""
    path = pathlib.Path(path)
    if not path.exists():
        raise DatasetError(f"no such file: {path}")
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise DatasetError(
            f"unsupported format {path.suffix!r}; expected one of {sorted(READERS)}"
        )
    return reader(path), path.suffix.lower().lstrip(".")


# ------------------------------------------------------------- normalisation
def normalise(rows: Iterable[dict], *, source: str = "unknown") -> pd.DataFrame:
    """Force arbitrary records into the canonical schema.

    Missing columns are filled with sensible defaults instead of failing, so a
    partial third-party dataset still produces a usable frame.
    """
    records: list[dict] = []
    for index, row in enumerate(rows):
        lowered = {str(k).strip().lower(): v for k, v in row.items()}

        def pick(*names, default=None):
            for name in names:
                if name in lowered and lowered[name] not in (None, ""):
                    return lowered[name]
            return default

        record = {
            "id": str(pick("id", "item_id", "place_id", default=f"ROW-{index:04d}")),
            "name": str(pick("name", "title", "place_name", "hotel_name", default="Unnamed")),
            "city": str(pick("city", "location", "province", "destination", default="Unknown")),
            "region": str(pick("region", "area", default="unknown")).lower(),
            "category": canonical_category(pick("category", "type", default="landmark")),
            "subcategory": str(pick("subcategory", "sub_type", default="")),
            "description": str(pick("description", "details", "summary", "review_text", default="")),
            "tags": split_list_value(pick("tags", "amenities", "highlights", default="")),
            # Currency-specific columns win. A bare "price" is ambiguous, so its
            # magnitude decides: 60,000 is dong, 60 is dollars.
            "price_vnd": _to_number(pick("price_vnd", "cost_vnd", "price_vnd_vnd")),
            "price_usd": _to_number(pick("price_usd", "cost_usd", "price_dollar", "price_usd_usd")),
            "rating": _to_number(pick("rating", "score", "stars")),
            "reviews": _to_number(pick("reviews", "review_count", "num_reviews")),
            "duration_hours": _to_number(pick("duration_hours", "duration", "hours")),
            "best_months": split_list_value(
                pick("best_months", "months", "best_time", default=""), cast=int
            ),
            "address": str(pick("address", "location_address", default="")),
            "latitude": _to_number(pick("latitude", "lat")),
            "longitude": _to_number(pick("longitude", "lng", "lon", "long")),
            "source": source,
        }
        records.append(record)

    # Some datasets carry only one ambiguous "price"/"cost" column. Resolve it
    # once per row, by magnitude, rather than guessing a currency for the file.
    ambiguous = [
        _to_number(row.get("price") or row.get("cost") or row.get("avg_price")
                   or row.get("average_price") or row.get("price_range_min"))
        for row in rows
    ]
    if any(ambiguous):
        for index, value in enumerate(ambiguous):
            if not value:
                continue
            if records[index]["price_vnd"] or records[index]["price_usd"]:
                continue
            if value >= 1000:  # dong amounts are in the thousands and up
                records[index]["price_vnd"] = value
            else:
                records[index]["price_usd"] = value

    frame = pd.DataFrame.from_records(records, columns=CANONICAL_FIELDS)

    # If a dataset only carries one currency, derive the other for comparison.
    rate = 25000.0
    if frame["price_usd"].sum() == 0 and frame["price_vnd"].sum() > 0:
        frame["price_usd"] = (frame["price_vnd"] / rate).round(2)
    elif frame["price_vnd"].sum() == 0 and frame["price_usd"].sum() > 0:
        frame["price_vnd"] = (frame["price_usd"] * rate).round(0)

    for column, fallback in NUMERIC_FIELDS.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fallback)

    frame["description"] = frame["description"].fillna("")
    frame["region"] = frame["region"].fillna("unknown")

    # Drop exact duplicate places (a common problem in scraped datasets).
    frame = frame.drop_duplicates(subset=["name", "city"]).reset_index(drop=True)
    return frame
