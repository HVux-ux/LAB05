"""Tests for the loader, dataset, context packer and planner.

Run with::

    python -m pytest tests -q

The tests are deliberately about behaviour that would silently break the app:
the three formats producing identical frames, the token budget actually being
respected, and the model never being handed rows outside the filters.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from travel_planner.adapter import MappingProfile, autodetect_profile, load_mapped  # noqa: E402
from travel_planner.context import (  # noqa: E402
    budget_for_model,
    build_context,
    estimate_tokens,
    pack_rows,
)
from travel_planner.dataset import TravelDataset, load_dataset  # noqa: E402
from travel_planner.llm import LLMConfig, MockLLM  # noqa: E402
from travel_planner.loader import (  # noqa: E402
    DatasetError,
    normalise,
    read_any,
    split_list_value,
)
from travel_planner.planner import (  # noqa: E402
    PlanRequest,
    compare_prompting_strategies,
    run_plan,
    select_items,
)

SAMPLE = ROOT / "data" / "sample"


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def dataset() -> TravelDataset:
    return load_dataset(SAMPLE / "travel_vn.csv")


@pytest.fixture(scope="module")
def mock_llm() -> MockLLM:
    return MockLLM(LLMConfig())


# --------------------------------------------------------------------- formats
@pytest.mark.parametrize("name", ["travel_vn.csv", "travel_vn.json", "travel_vn.jsonl"])
def test_every_format_loads(name):
    ds = load_dataset(SAMPLE / name)
    assert ds.size > 40
    assert set(ds.frame.columns) >= {"id", "name", "city", "category", "price_usd"}


def test_formats_agree_on_content():
    """CSV, JSON and JSONL must produce the same rows, not just similar ones."""
    frames = {ext: load_dataset(SAMPLE / f"travel_vn.{ext}").frame for ext in ("csv", "json", "jsonl")}
    reference = frames["json"].set_index("id").sort_index()
    for ext in ("csv", "jsonl"):
        other = frames[ext].set_index("id").sort_index()
        assert list(reference.index) == list(other.index)
        for column in ("name", "city", "category", "price_usd", "rating"):
            assert reference[column].tolist() == other[column].tolist(), f"{ext}.{column} differs"


def test_csv_list_cells_become_lists():
    ds = load_dataset(SAMPLE / "travel_vn.csv")
    tags = ds.frame.loc[0, "tags"]
    months = ds.frame.loc[0, "best_months"]
    assert isinstance(tags, list) and all(isinstance(t, str) for t in tags)
    assert isinstance(months, list) and all(isinstance(m, int) for m in months)


def test_best_months_are_ints_from_csv():
    ds = load_dataset(SAMPLE / "travel_vn.csv")
    for months in ds.frame["best_months"].head(10):
        assert all(isinstance(m, int) and 1 <= m <= 12 for m in months)


# ---------------------------------------------------------------- list parsing
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a|b|c", ["a", "b", "c"]),
        ("a, b", ["a", "b"]),
        (["a", "b"], ["a", "b"]),
        ('["x", "y"]', ["x", "y"]),
        ("", []),
        (None, []),
        ("   ", []),
    ],
)
def test_split_list_value(raw, expected):
    assert split_list_value(raw) == expected


def test_split_list_value_casts_and_drops_junk():
    assert split_list_value("3;4;oops;5", cast=int) == [3, 4, 5]


# ------------------------------------------------------------------- filtering
def test_city_filter_is_exclusive(dataset):
    result = dataset.search(cities=["Hoi An"])
    assert not result.empty
    assert set(result["city"]) == {"Hoi An"}


def test_price_and_rating_filters(dataset):
    result = dataset.search(max_price_usd=5, min_rating=4.0)
    assert (result["price_usd"] <= 5).all()
    assert (result["rating"] >= 4.0).all()


def test_month_filter(dataset):
    result = dataset.search(month=12)
    for months in result["best_months"]:
        assert 12 in months


def test_tag_filter(dataset):
    result = dataset.search(tags=["beach"])
    for tags in result["tags"]:
        assert "beach" in tags


def test_unknown_city_returns_empty_not_error(dataset):
    assert dataset.search(cities=["Atlantis"]).empty


def test_results_are_sorted_by_relevance(dataset):
    """Ordering is a composite of rating, popularity and price, so raw rating is
    not monotonic; the composite is."""
    result = dataset.search(categories=["landmark"])
    score = result["rating"] * 20 + result["reviews"].clip(upper=25000) / 5000 - result["price_usd"] / 25
    assert score.tolist() == sorted(score.tolist(), reverse=True)
    assert result["rating"].iloc[0] >= result["rating"].iloc[-1]


# ------------------------------------------------------------------ statistics
def test_city_summary_shape(dataset):
    summary = dataset.city_summary()
    assert {"city", "items", "avg_rating"} <= set(summary.columns)
    assert summary["items"].sum() == dataset.size


def test_city_summary_restricted_to_cities(dataset):
    summary = dataset.city_summary(cities=["Hue"])
    assert summary["city"].tolist() == ["Hue"]


def test_budget_estimate_orders_by_style(dataset):
    budget = dataset.budget_estimate(["Hanoi"], 3, "budget")
    luxury = dataset.budget_estimate(["Hanoi"], 3, "luxury")
    assert budget["nightly_usd"] <= luxury["nightly_usd"]
    assert budget["lodging_usd"] == pytest.approx(budget["nightly_usd"] * 2)


# --------------------------------------------------------------- context pack
def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_pack_respects_budget():
    rows = [{"name": f"Place {i}", "city": "Hanoi", "description": "x" * 200} for i in range(200)]
    pack = pack_rows(rows, 300)
    assert pack.tokens <= 300
    assert pack.dropped_rows > 0
    assert len(pack.rows) < 200


def test_pack_keeps_order_of_relevance():
    rows = [{"name": f"Place {i}", "city": "Hanoi", "rating": 5 - i / 10} for i in range(50)]
    pack = pack_rows(rows, 250)
    assert pack.rows[0]["name"] == "Place 0"


def test_pack_drops_description_when_budget_is_tiny():
    rows = [{"name": "Only one", "city": "Hue", "description": "d" * 2000}]
    pack = pack_rows(rows, 20)
    assert "description" not in pack.text


def test_zero_budget_is_honest():
    pack = pack_rows([{"name": "x"}], 0)
    assert pack.text == ""
    assert pack.dropped_rows == 1


def test_budget_for_model_scales_with_window():
    assert budget_for_model("llama3", 8192) < budget_for_model("llama3.1", 131072)
    assert 500 <= budget_for_model("unknown-model") <= 6000


def test_context_contains_only_selected_rows(dataset):
    selection = dataset.search(cities=["Hue"])
    pack = build_context(selection, 4000)
    names_in_pack = {row["name"] for row in pack.rows}
    assert names_in_pack <= set(selection["name"])


# --------------------------------------------------------------------- planner
def test_selection_is_balanced_across_categories(dataset):
    request = PlanRequest(cities=("Hanoi", "Hoi An"), days=4)
    selection = select_items(dataset, request)
    categories = set(selection["category"])
    assert "landmark" in categories
    assert "accommodation" in categories


def test_plan_uses_only_retrieved_places(dataset, mock_llm):
    request = PlanRequest(cities=("Hoi An",), days=2, style="mid")
    result = run_plan(dataset, request, llm=mock_llm)
    allowed = set(result.selection["name"])
    mentioned = [
        name for name in allowed
        if name in result.answer
    ]
    assert mentioned, "the itinerary should name at least one retrieved place"

    # Nothing outside the selection may appear as a bulleted place line.
    for row in dataset.frame.itertuples():
        if row.name not in allowed and row.name in result.answer:
            pytest.fail(f"{row.name} leaked into the answer but was not retrieved")


def test_plan_has_required_sections(dataset, mock_llm):
    result = run_plan(dataset, PlanRequest(cities=("Da Nang",), days=3), llm=mock_llm)
    for heading in ("## Overview", "## Day-by-day plan", "## Where to stay",
                    "## Budget", "## Gaps"):
        assert heading in result.answer


def test_plan_day_count_matches_request(dataset, mock_llm):
    result = run_plan(dataset, PlanRequest(cities=("Hanoi",), days=6), llm=mock_llm)
    assert result.answer.count("### Day ") == 6


def test_plan_respects_city_choice(dataset, mock_llm):
    result = run_plan(dataset, PlanRequest(cities=("Hue",), days=2), llm=mock_llm)
    assert set(result.selection["city"]) == {"Hue"}
    assert "Hoi An" not in result.answer


def test_plan_with_impossible_filters_degrades_gracefully(dataset, mock_llm):
    request = PlanRequest(cities=("Hue",), max_items=5, min_rating=5.0)
    result = run_plan(dataset, request, llm=mock_llm)
    assert isinstance(result.answer, str) and result.answer


def test_prompt_tokens_reported(dataset, mock_llm):
    result = run_plan(dataset, PlanRequest(cities=("Hanoi", "Hue"), days=3), llm=mock_llm)
    assert result.prompt_tokens > 0
    assert result.pack.tokens <= result.pack.budget


def test_strategy_comparison_shows_retrieval_wins(dataset):
    rows = {r["strategy"]: r for r in compare_prompting_strategies(dataset, PlanRequest())}
    whole = next(v for k, v in rows.items() if k.startswith("Whole"))
    retrieved = next(v for k, v in rows.items() if k.startswith("Filter"))
    assert retrieved["tokens"] < whole["tokens"]


# ------------------------------------------------------------------- app layer
def test_run_pipeline_returns_seven_outputs(dataset, mock_llm):
    """The Gradio callback is a plain function, so it is tested without a browser."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("appmod", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module._RUNTIME.update({"llm": mock_llm, "config": LLMConfig()})
    outputs = module.run_pipeline(
        "Street food and beaches", ["Hanoi", "Hoi An"], dataset.categories,
        "any", "March", 4, 2, 90, "Mid-range", 4.0,
        "CSV (data/sample/travel_vn.csv)", "gemma3:4b",
    )
    assert len(outputs) == 7
    answer, info, preview, pack_md, context, strategies, stats = outputs
    assert "## Overview" in answer
    assert "tokens" in info
    assert len(preview) > 0 and len(stats) > 0
    assert "rows included" in pack_md
    assert "Whole dataset in prompt" in strategies["strategy"].iloc[0]


@pytest.mark.parametrize("label,ext", [
    ("CSV (data/sample/travel_vn.csv)", "csv"),
    ("JSON (data/sample/travel_vn.json)", "json"),
    ("JSONL (data/sample/travel_vn.jsonl)", "jsonl"),
])
def test_run_pipeline_works_for_each_format(mock_llm, label, ext):
    import importlib.util

    spec = importlib.util.spec_from_file_location("appmod2", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._RUNTIME.update({"llm": mock_llm, "config": LLMConfig()})

    outputs = module.run_pipeline(
        "A short trip", ["Hue"], ["landmark", "food", "accommodation"],
        "any", "Any month", 2, 1, 60, "Budget", 0.0, label, "gemma3:4b",
    )
    assert "## Day-by-day plan" in outputs[0]


def test_run_pipeline_reports_model_failure_without_crashing(dataset):
    """A dead Ollama must surface a message, not a traceback."""
    import importlib.util

    from travel_planner.llm import OllamaClient

    spec = importlib.util.spec_from_file_location("appmod3", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Nothing listens on this port, so the client fails fast.
    dead = OllamaClient(LLMConfig(host="http://127.0.0.1:9", timeout=2))
    module._RUNTIME.update({"llm": dead, "config": LLMConfig(host="http://127.0.0.1:9")})
    outputs = module.run_pipeline(
        "Any trip", ["Hanoi"], ["landmark"], "any", "Any month", 2, 1, 60,
        "Budget", 0.0, "CSV (data/sample/travel_vn.csv)", "gemma3:4b",
    )
    assert "Model unavailable" in outputs[0]
    assert len(outputs[2]) > 0, "retrieval should still be shown"


# --------------------------------------------------------------------- adapter
def test_normalise_fills_missing_columns():
    frame = normalise([{"name": "Somewhere", "city": "Hue"}], source="unit")
    assert list(frame.columns) >= ["id", "name", "city", "price_usd", "rating"]
    assert frame.loc[0, "region"] == "unknown"


def test_normalise_reads_synonym_columns():
    frame = normalise([{
        "title": "Test hotel", "province": "Hue", "type": "Hotel",
        "review_count": "1,200", "lat": "16.4", "lng": "107.5", "price": "$58",
    }], source="unit")
    row = frame.iloc[0]
    assert row["name"] == "Test hotel"
    assert row["category"] == "accommodation"
    assert row["reviews"] == 1200
    assert row["price_usd"] == 58


def test_mapping_profile_renames_and_aliases(tmp_path):
    source = tmp_path / "kaggle.csv"
    source.write_text(
        "place_id,place_name,province,category,rating\n"
        "K1,Test Place,ha noi,attraction,4.5\n",
        encoding="utf-8",
    )
    profile = MappingProfile({
        "name": "unit",
        "columns": {"id": "place_id", "name": "place_name", "city": "province"},
        "city_aliases": {"ha noi": "Hanoi"},
        "category_aliases": {"attraction": "landmark"},
    })
    frame = load_mapped(source, profile)
    row = frame.iloc[0]
    assert row["name"] == "Test place" or row["name"] == "Test Place"
    assert row["city"] == "Hanoi"
    assert row["category"] == "landmark"


def test_bundled_profile_is_valid_json():
    profile = MappingProfile.load(ROOT / "config" / "vietnam-tourism.mapping.json")
    assert profile.columns["name"] == "name"
    assert profile.city_aliases["ha noi"] == "Hanoi"


@pytest.mark.parametrize("suffix", ["", ".txt", ".xlsx"])
def test_unsupported_format_raises(tmp_path, suffix):
    path = tmp_path / f"data{suffix}"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(DatasetError):
        read_any(path)


def test_missing_file_raises():
    with pytest.raises(DatasetError):
        read_any(ROOT / "data" / "nope.csv")


def test_json_wrapper_shapes_are_found(tmp_path):
    for wrapper in ({"items": [{"name": "A"}]}, {"data": [{"name": "A"}]}, [{"name": "A"}]):
        path = tmp_path / "wrapped.json"
        path.write_text(json.dumps(wrapper), encoding="utf-8")
        rows, fmt = read_any(path)
        assert rows[0]["name"] == "A"
        assert fmt == "json"


def test_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"name":"A"}\n\n{"name":"B"}\n', encoding="utf-8")
    rows, _ = read_any(path)
    assert [r["name"] for r in rows] == ["A", "B"]


def test_jsonl_reports_bad_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"name":"A"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=":2:"):
        read_any(path)


# -------------------------------------------------------------- sample dataset
def test_sample_dataset_invariants():
    ds = load_dataset()
    frame = ds.frame
    assert frame["id"].is_unique
    assert (frame["rating"].between(0, 5)).all()
    assert (frame["price_usd"] >= 0).all()
    assert set(frame["category"]) <= {"landmark", "food", "accommodation", "experience", "transport"}
    assert frame["region"].isin(["north", "central", "south"]).all()


def test_every_city_has_somewhere_to_stay_and_eat():
    ds = load_dataset()
    for city in ds.cities:
        subset = ds.frame[ds.frame["city"] == city]
        assert (subset["category"] == "accommodation").any(), f"{city} has no lodging"
        assert (subset["category"].isin(["food", "landmark", "experience"])).any(), \
            f"{city} has nothing to do"


def test_regions_are_consistent_with_cities():
    ds = load_dataset()
    pairs = ds.frame.groupby("city")["region"].nunique()
    assert (pairs == 1).all(), "a city is listed under two regions"
