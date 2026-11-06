#!/usr/bin/env python3
"""Gradio front-end for the Vietnam trip planner.

Run it with:

    python app.py                          # mock mode, no model needed
    python app.py --live                   # use Ollama on localhost:11434
    python app.py --live --model qwen2.5:7b
    python app.py --share                  # temporary public link

The UI is intentionally thin: every piece of logic lives in
``src/travel_planner`` so it can be tested without a browser.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import gradio as gr
import pandas as pd

from travel_planner.dataset import DEFAULT_SAMPLE_DIR, TravelDataset, load_dataset
from travel_planner.llm import LLMConfig, LLMError, MockLLM, OllamaClient
from travel_planner.planner import (
    PlanRequest,
    compare_prompting_strategies,
    run_plan,
    select_items,
)

MONTHS = {
    "Any month": None, "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}
STYLES = {"Budget": "budget", "Mid-range": "mid", "Luxury": "luxury"}

DATA_FILES = {
    "CSV (data/sample/travel_vn.csv)": DEFAULT_SAMPLE_DIR / "travel_vn.csv",
    "JSON (data/sample/travel_vn.json)": DEFAULT_SAMPLE_DIR / "travel_vn.json",
    "JSONL (data/sample/travel_vn.jsonl)": DEFAULT_SAMPLE_DIR / "travel_vn.jsonl",
}


# --------------------------------------------------------------------- helpers
def format_stats(ds: TravelDataset) -> str:
    s = ds.stats()
    return (
        f"**{s['items']} rows** · {s['cities']} cities · {s['categories']} categories · "
        f"avg rating {s['avg_rating']} · {s['free_items']} free · read as `{s['format']}`"
    )


def make_choices(ds: TravelDataset) -> tuple[list[str], list[str]]:
    return ds.cities, ds.categories


def _format_pack(pack) -> str:
    rows = "\n".join(f"| {k.replace('_', ' ')} | {v} |" for k, v in pack.summary().items())
    return "| metric | value |\n| --- | --- |\n" + rows


def _preview_selection(dataset: TravelDataset, request: PlanRequest) -> pd.DataFrame:
    """The retrieval half of the pipeline, without touching the model."""
    frame = select_items(dataset, request).head(60)
    return frame.assign(
        tags=lambda f: f["tags"].map(lambda t: ", ".join(t) if isinstance(t, list) else "")
    )[["name", "city", "category", "price_usd", "rating", "duration_hours", "tags"]]


# ------------------------------------------------------------------- pipeline
#: Active runtime (LLM client and config) for this process. Set by build_app().
#: A module-level registry rather than gr.State: State components are not
#: addressable through the Gradio API client, which would make the app
#: undriveable from tests, scripts or any non-browser client.
_RUNTIME: dict[str, object] = {}


def run_pipeline(request_text, chosen_cities, chosen_categories, region, month_label,
                 days, travellers, budget, style_label, min_rating, path_label, model_name):
    """The single Gradio callback: request -> selection -> context -> answer.

    Reads the active LLM from ``_RUNTIME`` so this stays a plain function that a
    test can call directly.
    """
    llm = _RUNTIME["llm"]
    config = _RUNTIME["config"]

    path = DATA_FILES.get(path_label, DEFAULT_SAMPLE_DIR / "travel_vn.csv")
    active = load_dataset(path)

    active_llm = llm
    if not isinstance(llm, MockLLM) and model_name and model_name != config.model:
        active_llm = OllamaClient(LLMConfig(**{**config.__dict__, "model": model_name}))

    request = PlanRequest(
        text=request_text or "",
        cities=tuple(chosen_cities or ()),
        region=None if region in (None, "any") else region,
        categories=tuple(chosen_categories or ()),
        days=int(days),
        travellers=int(travellers),
        daily_budget_usd=float(budget) if budget is not None else None,
        style=STYLES.get(style_label, "mid"),
        month=MONTHS.get(month_label),
        min_rating=float(min_rating) or None,
    )

    try:
        result = run_plan(active, request, llm=active_llm)
    except LLMError as exc:
        return (
            f"### Model unavailable\n\n{exc}\n\n"
            "The retrieval step still ran: see the *Retrieved data* tab.",
            "", _preview_selection(active, request), "", "",
            pd.DataFrame(), active.city_summary(),
        )

    summary = result.pack.summary()
    info = (
        f"**{summary['rows_included']} of {result.stats['items_selected']} selected rows** "
        f"fit the budget · {summary['used_pct']}% of {result.pack.budget} data tokens · "
        f"prompt ≈ {result.prompt_tokens} tokens · model `{result.model}`"
    )

    strategies = pd.DataFrame(compare_prompting_strategies(active, request))[
        ["strategy", "tokens", "scales_with_data", "fits_8k_model", "grounded", "weakness"]
    ]

    return (
        result.answer,
        info,
        _preview_selection(active, request),
        _format_pack(result.pack),
        result.pack.text,
        strategies,
        active.city_summary(),
    )


# ------------------------------------------------------------------- UI wiring
def build_app(dataset: TravelDataset, llm, config: LLMConfig) -> gr.Blocks:
    cities, categories = make_choices(dataset)

    _RUNTIME["llm"] = llm
    _RUNTIME["config"] = config

    with gr.Blocks(title="Vietnam Trip Planner") as demo:
        gr.Markdown(
            "# Vietnam Trip Planner\n"
            "An itinerary composed by a **locally running** language model, grounded in a "
            "travel dataset. Pandas filters and ranks the data first; only the matching rows "
            "are packed into the model's context window."
        )

        with gr.Row():
            with gr.Column(scale=3):
                request_box = gr.Textbox(
                    label="What kind of trip do you want?",
                    placeholder=("Two relaxed weeks: street food, one boat trip, "
                                 "and not too many early starts."),
                    lines=2,
                )
                with gr.Row():
                    city_box = gr.Dropdown(cities, value=cities[:3], multiselect=True,
                                           label="Cities")
                    category_box = gr.Dropdown(categories, value=categories, multiselect=True,
                                               label="Include categories")
                with gr.Row():
                    region_box = gr.Dropdown(["any", "north", "central", "south"], value="any",
                                             label="Region")
                    month_box = gr.Dropdown(list(MONTHS), value="Any month", label="Travel month")
                with gr.Row():
                    days_slider = gr.Slider(1, 21, value=4, step=1, label="Days")
                    travellers_slider = gr.Slider(1, 8, value=2, step=1, label="Travellers")
                with gr.Row():
                    budget_slider = gr.Slider(15, 400, value=90, step=5,
                                              label="Daily budget per person (USD)")
                    style_box = gr.Dropdown(list(STYLES), value="Mid-range", label="Style")
                    rating_slider = gr.Slider(0.0, 5.0, value=0.0, step=0.1,
                                              label="Minimum rating")

                with gr.Accordion("Dataset and model", open=False):
                    data_choice = gr.Dropdown(list(DATA_FILES), value=list(DATA_FILES)[0],
                                              label="Data file (same data, three formats)")
                    model_box = gr.Dropdown([config.model], value=config.model, label="Ollama model")
                    refresh_btn = gr.Button("Refresh model list", size="sm")
                    dataset_info = gr.Markdown(format_stats(dataset))

                generate_btn = gr.Button("Plan my trip", variant="primary")

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Itinerary"):
                        output = gr.Markdown("Your plan will appear here.")
                        run_info = gr.Markdown("")
                    with gr.Tab("Retrieved data"):
                        gr.Markdown(
                            "The rows below are the **only** places the model was allowed to use."
                        )
                        table = gr.Dataframe(
                            headers=["name", "city", "category", "price_usd", "rating",
                                     "duration_hours", "tags"],
                            interactive=False, wrap=True,
                        )
                    with gr.Tab("Prompt and token budget"):
                        gr.Markdown(
                            "How the dataset fits into the context window: filtering happens "
                            "in pandas, then rows are packed until the token budget is full."
                        )
                        context_md = gr.Markdown("")
                        context_box = gr.Textbox(label="Context block sent to the model",
                                                 lines=18, max_lines=30)
                    with gr.Tab("Strategy comparison"):
                        gr.Markdown(
                            "Three ways to get a dataset into a prompt, measured on the current "
                            "selection."
                        )
                        strategy_table = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Statistics"):
                        gr.Markdown("Computed with pandas, not by the model.")
                        stats_table = gr.Dataframe(interactive=False)

        # ---------------------------------------------------------- callbacks
        def switch_dataset(path_label: str):
            ds = load_dataset(DATA_FILES[path_label])
            new_cities, new_categories = make_choices(ds)
            return (
                gr.update(choices=new_cities, value=new_cities[:3]),
                gr.update(choices=new_categories, value=new_categories),
                format_stats(ds),
                ds.city_summary(),
            )

        def refresh_models():
            if isinstance(llm, MockLLM):
                return gr.update(choices=["mock"], value="mock")
            try:
                models = llm.list_models() or ["(none installed)"]
                return gr.update(choices=models, value=models[0])
            except LLMError as exc:
                return gr.update(choices=[str(exc)], value=str(exc))

        generate_btn.click(
            run_pipeline,
            inputs=[request_box, city_box, category_box, region_box, month_box,
                    days_slider, travellers_slider, budget_slider, style_box, rating_slider,
                    data_choice, model_box],
            outputs=[output, run_info, table, context_md, context_box, strategy_table, stats_table],
        )
        refresh_btn.click(refresh_models, outputs=[model_box])
        data_choice.change(
            switch_dataset, inputs=[data_choice],
            outputs=[city_box, category_box, dataset_info, stats_table],
        )

        demo.load(lambda: dataset.city_summary(), outputs=[stats_table])

    return demo


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnam trip planner UI")
    parser.add_argument("--live", action="store_true",
                        help="use Ollama instead of the deterministic mock")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model tag")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    dataset = load_dataset()
    config = LLMConfig(model=args.model, host=args.host, num_ctx=args.num_ctx)

    if args.live:
        llm = OllamaClient(config)
        if not llm.is_available():
            print(f"! Ollama is not reachable at {args.host}; falling back to mock mode.")
            print("  Start it with `ollama serve` and pull a model, then rerun with --live.")
            llm = MockLLM(config)
        else:
            try:
                llm.ensure_model(args.model)
            except LLMError as exc:
                print(f"! {exc}")
                print("  Falling back to mock mode.")
                llm = MockLLM(config)
    else:
        llm = MockLLM(config)
        print("Running in mock mode (templated itineraries, no model download).")

    demo = build_app(dataset, llm, config)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
                theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
