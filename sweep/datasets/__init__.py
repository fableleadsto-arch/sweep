"""Dataset loading and exploration for the sweep terminal.

Grounds discussion and reasoning in the data sources Sweep can learn from —
the reasoning QA families (LogiQA, GSM8K, MMLU, CommonsenseQA, bAbI, CoT),
molecular biology (PDB via biotite, ProteinNet, ESM) and large web corpora
(The Pile, FineWeb, Wikipedia).

Data caches under ``~/.sweep/datasets`` (override with ``SWEEP_DATASETS_DIR``).
Online sources are downloaded on first use and then served from cache, so
discussion itself never calls an external API.
"""

from __future__ import annotations

from .registry import (
    DATASETS,
    dataset_dir,
    dataset_status,
    load_dataset,
    list_datasets,
)
from .qa import ask_dataset, evaluate_sample
from .molecular import protein_stats

__all__ = [
    "DATASETS",
    "ask_dataset",
    "dataset_dir",
    "dataset_status",
    "evaluate_sample",
    "list_datasets",
    "load_dataset",
    "protein_stats",
]