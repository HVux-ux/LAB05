# Architecture

How a request becomes an itinerary, and why the code is split the way it is.

## Module map

```mermaid
flowchart LR
    subgraph ui["UI"]
        APP[app.py<br/>Gradio]
    end
    subgraph core["src/travel_planner"]
        L[loader.py<br/>CSV / JSON / JSONL → DataFrame]
        A[adapter.py<br/>foreign columns → schema]
        D[dataset.py<br/>filter · sort · statistics]
        C[context.py<br/>token budget · packing]
        PR[prompts.py<br/>templates]
        P[planner.py<br/>orchestration]
        LL[llm.py<br/>Ollama · Mock]
    end
    subgraph data["data"]
        S[(data/sample<br/>csv · json · jsonl)]
        R[(data/raw<br/>Kaggle, git-ignored)]
    end
    S --> L
    R --> A
    A --> L
    L --> D
    D --> C
    D --> P
    C --> P
    PR --> P
    P --> LL
    LL --> APP
    P --> APP
    APP -.->|constraints| P
```

**The rule that keeps this maintainable:** pandas lives only in `dataset.py`, and
the model is reached only through `llm.py`. Everything else is pure functions
over plain dicts. That is what lets the whole pipeline run with `MockLLM` in
tests, with no network and no model download.

## The planning pipeline

```mermaid
flowchart TD
    START([User request]) --> REQ[PlanRequest<br/>cities · days · budget · style · month]
    REQ --> FILT[dataset.search<br/>hard constraints + text match]
    FILT --> EMPTY{Any rows?}
    EMPTY -->|no| RELAX[Relax in a stated order:<br/>tags → rating → city]
    RELAX --> FILT
    EMPTY -->|yes| BAL[Balance by category<br/>35% sights · 25% experiences<br/>20% food · 15% stays · 5% transport]
    BAL --> RANK[Rank by relevance score]
    RANK --> STATS[City summary + budget estimate<br/>computed in pandas]
    STATS --> BUDGET[Data budget = 45% of num_ctx<br/>capped at 6000 tokens]
    BUDGET --> PACK[pack_rows<br/>greedy fill, drop description if tight]
    PACK --> RENDER[Render as JSONL]
    RENDER --> MSG[System prompt + user prompt]
    MSG --> MODEL{Ollama reachable?}
    MODEL -->|no| ERR[LLMError surfaced in the UI<br/>retrieval still displayed]
    MODEL -->|yes| ANSWER[Markdown itinerary<br/>Overview · Days · Stay · Budget · Gaps]
    ANSWER --> OUT([PlanResult<br/>answer + pack + selection + debug])
```

## Sequence of a plan request

```mermaid
sequenceDiagram
    autonumber
    actor U as Traveller
    participant G as Gradio app
    participant P as planner
    participant D as TravelDataset
    participant X as context
    participant O as Ollama
    U->>G: cities, days, budget, month, free text
    G->>P: PlanRequest
    P->>D: search(cities, categories, min_rating, tags)
    D-->>P: ranked DataFrame
    P->>D: city_summary(), budget_estimate()
    D-->>P: exact statistics
    P->>X: rows + budget_for_model()
    X-->>P: ContextPack(rows, tokens, dropped)
    P->>P: build_messages()
    P->>O: POST /api/chat (stream: false)
    O-->>P: Markdown
    P-->>G: PlanResult
    G-->>U: itinerary + retrieved rows + prompt + token accounting
```

## The context-window design in one picture

```mermaid
flowchart LR
    subgraph full["Whole dataset"]
        F1[54 rows<br/>≈4,300 tokens]
        F2[540 rows<br/>≈43,000 tokens]
        F3[5,400 rows<br/>≈429,000 tokens]
    end
    subgraph window["num_ctx = 8,192"]
        W1[system prompt ≈120]
        W2[constraints + stats ≈250]
        W3[DATA ≈1,200]
        W4[room to answer ≈3,000]
    end
    F1 -.->|"too big already"| window
    F2 -.->|"cannot fit"| window
    F3 -.->|"cannot fit"| window
    RETR[Filter → rank → pack] -->|"takes only what fits"| W3
    classDef out fill:#fee,stroke:#c55;
    class F1,F2,F3 out;
```

Data flow in the opposite direction, for the person reading the answer:

```mermaid
flowchart TD
    A[Answer: a named place] --> B{In the retrieved rows?}
    B -->|yes| C[Traceable to the dataset]
    B -->|no| D[Model invented it →<br/>prompt violation]
    C --> E[Price and rating visible<br/>in the Retrieved data tab]
    E --> F[Budget table checked against<br/>pandas budget_estimate]
    classDef bad fill:#fee,stroke:#c55;
    class D bad;
```

## Data model

```mermaid
erDiagram
    CITY ||--o{ ITEM : contains
    CITY {
        string city PK
        string region "north | central | south"
        string airport
        int_list best_months
    }
    ITEM {
        string id PK
        string name
        string city FK
        string category "landmark | food | accommodation | experience | transport"
        string description
        string_list tags
        float price_usd
        float rating
        int reviews
        float duration_hours
        int_list best_months
    }
    CITY ||--o{ PLAN_DAY : schedules
    PLAN_DAY {
        int day
        string city FK
        string morning
        string afternoon
        string evening
    }
```

`cities.json` is deliberately a *different* JSON shape from `travel_vn.json` —
an object with a `cities` array, rather than an object with an `items` array.
Having two shapes in the sample data forces the loader to handle both instead of
hard-coding one, which is what a real Kaggle file will demand.

## Where each format is exercised

| Format | File | Read by | What it demonstrates |
|---|---|---|---|
| CSV | `data/sample/travel_vn.csv` | `read_csv_rows` | Flat table; lists packed as `a|b|c` and split on read |
| JSON | `data/sample/travel_vn.json` | `read_json_rows` | Wrapped array; native types and list fields |
| JSONL | `data/sample/travel_vn.jsonl` | `read_jsonl_rows` | Line-by-line streaming; bad line reports its number |
| JSON (different shape) | `data/sample/cities.json` | `_locate_records` | Object-of-array wrapper; shape autodetection |
| Any | `data/raw/**` | `adapter.load_mapped` | Column renaming + city/category aliases |

## Failure behaviour

Deliberate choices, because a demo that dies is worse than one that explains:

| Situation | Behaviour |
|---|---|
| Ollama not running | `LLMError` with instructions; the UI still shows the retrieved rows |
| Model not pulled | Error names the model and the `ollama pull` command |
| Filters match nothing | Constraints relax in a fixed order; if still empty, the app says so |
| Budget too small for even one row | Descriptions are dropped; if still too small, the pack reports `dropped_rows` |
| Malformed JSONL line | Error reports the line number instead of failing silently |
| Unknown file extension | `DatasetError` listing the supported formats |
