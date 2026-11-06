"""Travel Planner — plan Vietnam trips with a local LLM (Ollama).

Package layout
--------------
loader.py    read CSV / JSON / JSONL into one normalised pandas DataFrame
adapter.py   map a third-party (e.g. Kaggle) dataset onto the canonical schema
dataset.py   filtering, sorting and statistics over the dataset
context.py   pack the *relevant slice* of data into a token budget
llm.py       Ollama client plus an offline mock generator
prompts.py   prompt templates
planner.py   orchestration: request -> selection -> context -> LLM -> itinerary
"""

from .dataset import TravelDataset, Filters, load_dataset
from .planner import PlanRequest, PlanResult, run_plan

__all__ = [
    "TravelDataset",
    "Filters",
    "load_dataset",
    "PlanRequest",
    "PlanResult",
    "run_plan",
]
__version__ = "1.0.0"
