# AGENT.md

Project context for AI coding agents (and humans) working in this repository.
Read this before making changes. It states what the project *is*, what has been
decided, and which changes are welcome versus which break the design.

---

## 1. What this repository is

A Lab 05 submission with two deliverables that must stay consistent with each
other:

1. **A written research brief** — [`docs/exercise-a-report.md`](docs/exercise-a-report.md)
   answers five research questions (Mermaid, CSV/JSON/JSONL + Pandas, Kaggle
   datasets, the LLM context window, and a retrieval design).
2. **A working application** that implements what the brief argues for: a Vietnam
   trip planner where **pandas decides what the model may see** and a **local
   Ollama model** composes the itinerary.

The report is not decoration. If you change how the app retrieves or packs data,
the measurements and diagrams in the report become wrong and must be updated.

**Stack:** Python 3.11, pandas, Gradio, requests, Ollama (local LLM). Tests with
pytest. No cloud services, no API keys, no network calls at runtime except to
`localhost:11434`.

---

## 2. Hard rules

These are project decisions, not preferences. Do not silently violate them.

| Rule | Why |
|---|---|
| **No third-party AI vendor logos or icons** (Gemini, ChatGPT/OpenAI, Claude, Copilot, etc.) anywhere — code, UI, docs, or generated assets | The app runs a local model. Vendor branding misrepresents what it does and implies data leaves the machine, which it does not |
| **No vendor SDKs** — no `openai`, `google-generativeai`, `anthropic`, `langchain` | The whole point is a readable, dependency-light local pipeline |
| **The model composes; pandas computes** | LLMs are unreliable at arithmetic over tables. Averages, counts and budgets come from pandas and are passed in as facts |
| **Only retrieved rows may appear in an answer** | Grounding is the core claim of the project. `tests/test_pipeline.py` enforces it |
| **Every format must stay loadable** | CSV, JSON and JSONL are all part of the exercise. `test_formats_agree_on_content` must keep passing |
| **Tests must pass without a model or a network** | Use `MockLLM`. Never make a test depend on Ollama |
| **Data files stay small** | `data/sample/` is curated and committed; `data/raw/` is git-ignored. Do not commit a Kaggle download |
| **No secrets in the repo** | Kaggle credentials live in `~/.kaggle/kaggle.json`, never here |

---

## 3. Architecture, and where to put code

```
app.py                      Gradio UI + run_pipeline() callback. Thin: no domain logic
src/travel_planner/
    __init__.py             public exports
    loader.py               read CSV/JSON/JSONL → one canonical DataFrame
    adapter.py              map foreign columns (Kaggle) → canonical schema
    dataset.py              TravelDataset: filter, rank, statistics   [ONLY pandas layer]
    context.py              token estimation, packing, budget_for_model
    prompts.py              system + user prompt templates
    llm.py                  OllamaClient + MockLLM
    planner.py              orchestration: PlanRequest → PlanResult
scripts/
    build_sample_data.py    single source of truth → renders csv/json/jsonl
    download_kaggle.py      fetch a real dataset into data/raw/ (git-ignored)
config/
    vietnam-tourism.mapping.json   column/city/category alias profile
data/sample/                committed sample dataset (csv, json, jsonl, cities.json)
docs/                       exercise-a-report.md, architecture.md
tests/test_pipeline.py      52 tests, all offline
```

**Where new code belongs:**

| You are adding… | Put it in | Not in |
|---|---|---|
| A new filter (e.g. "walking distance") | `dataset.search()` | `app.py` |
| A new data source | a reader in `loader.py` + a profile in `config/` | `dataset.py` |
| A new prompt section | `prompts.py` | inline strings in `planner.py` |
| A new model provider | a class next to `OllamaClient` with the same `chat()` signature | `planner.py` |
| A UI control | `app.py` only, then pass the value into `PlanRequest` | the core modules |

**Layering invariant:** `dataset.py` is the only module that touches pandas
directly; `llm.py` is the only module that performs network I/O. Preserve both.

---

## 4. The data contract

One row per *place*. The canonical schema lives in `loader.CANONICAL_FIELDS`:

| Field | Type | Notes |
|---|---|---|
| `id` | str | Unique. `test_sample_dataset_invariants` asserts uniqueness |
| `name` | str | The real place name; the model must copy it verbatim |
| `city` | str | Must be consistent with `region` for the same city |
| `region` | str | `north` \| `central` \| `south` \| `unknown` |
| `category` | str | **Exactly one of five**: `landmark`, `food`, `accommodation`, `experience`, `transport` |
| `subcategory` | str | Free text, e.g. `street food`, `cable car` |
| `description` | str | 1–2 sentences. Truncated to 150 chars when packed into a prompt |
| `tags` | list[str] | Lowercase; used by the tag filter |
| `price_vnd` | float | Per person |
| `price_usd` | float | Per person; derived at ~25,000 VND/USD if the source only has one currency |
| `rating` | float | 0–5 |
| `reviews` | float | Review count |
| `duration_hours` | float | 0 for lodging |
| `best_months` | list[int] | 1–12 |
| `address` | str | Dropped from prompts (costs tokens, cannot improve prose) |
| `latitude`, `longitude` | float | Dropped from prompts |
| `source` | str | File name or `kaggle` |

**Invariants enforced by tests:** ids unique; ratings within 0–5; prices ≥ 0;
exactly the five categories; every city has at least one accommodation row and
something to do; a city has exactly one region.

**Changing the dataset?** Edit `scripts/build_sample_data.py`, then run
`python scripts/build_sample_data.py`. Never hand-edit the three rendered files:
they are generated, and hand edits make the formats disagree.

---

## 5. Commands

```bash
# setup (a venv is required: the system Python is externally managed)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# run: mock mode, no model needed
.venv/bin/python app.py

# run: live with Ollama
ollama serve && ollama pull gemma3:4b
.venv/bin/python app.py --live --model gemma3:4b

# tests (offline, ~1s)
.venv/bin/python -m pytest tests -q

# regenerate the sample data in all three formats
.venv/bin/python scripts/build_sample_data.py

# fetch a real Kaggle dataset into data/raw/
.venv/bin/python scripts/download_kaggle.py --dataset owner/dataset-slug
```

The app must bind `0.0.0.0` (Gradio's `server_name`), not `127.0.0.1`, so the
sandbox preview proxy can reach it.

---

## 6. How the context budget works

Two numbers matter:

- `LLMConfig.num_ctx` — the context window actually given to Ollama. Default
  8,192 because that is what small local models handle reliably.
- `context.budget_for_model()` — the **data** budget: 45% of the window, clamped
  to 500–6,000 tokens. The rest is reserved for the system prompt, the request,
  statistics, and roughly 3,000 tokens of answer.

`pack_rows()` fills the budget greedily from an already-ranked list, so the rows
that get dropped are the least relevant. If even `min_rows` do not fit,
descriptions are stripped; if they still do not fit, the pack reports
`dropped_rows` rather than truncating mid-JSON.

**Do not raise the budget casually.** Bigger prompts are slower on local models
and facts in the middle of a long context get ignored ("lost in the middle"). If
a plan is missing options, the fix is better filtering or ranking, not a bigger
prompt.

---

## 7. Known pitfalls

1. **`num_ctx` defaults are small.** Ollama does not use the model's maximum
   context unless told to. Always pass `num_ctx`.
2. **CSV has no types.** `best_months` arrives as `"1|2|3"`; every field needs
   coercion, and `split_list_value` handles `|`, `;`, `,` and JSON-looking cells.
3. **An ambiguous `price` column.** `normalise()` resolves it per row by
   magnitude (≥1,000 ⇒ dong) rather than guessing a currency for the file.
4. **Category drift.** A Kaggle file says `hotel` or `restaurant`. `canonical_category()`
   maps it; without that, every filter silently returns nothing.
5. **Duplicate rows** are common in scraped data; `normalise()` drops rows
   duplicated on `(name, city)`.
6. **Do not call `pd.isna()` on a list cell.** It returns an array and raises in
   a boolean context. `split_list_value` already handles this.
7. **Gradio 6 API:** `theme` belongs on `launch()`, not on `Blocks()`; a
   `gr.State` is used to pass the LLM into the callback so the callback stays
   testable at module level.
8. **`MockLLM` must stay deterministic.** Tests depend on exact day counts and
   section headings. If you change its output, update the tests in the same
   commit.
9. **Never print a Kaggle token.** If a user shares `kaggle.json` contents, do
   not echo or store them.

---

## 8. Change checklist

Before you consider a change finished:

- [ ] `python -m pytest tests -q` passes, offline
- [ ] Both modes still start: `python app.py` and `python app.py --live` (the
      live path must fail *gracefully* when Ollama is absent)
- [ ] `python scripts/build_sample_data.py` runs and the data tests pass
- [ ] No vendor AI branding, logos, names or SDKs introduced
- [ ] If retrieval, budgeting or measured numbers changed, update
      `docs/exercise-a-report.md` (the token table and §5) and any Mermaid
      diagram in `docs/architecture.md` that describes the flow
- [ ] If the schema changed, update §4 here **and** the adapter profile
- [ ] New logic has a test; new behaviour is described in the report
