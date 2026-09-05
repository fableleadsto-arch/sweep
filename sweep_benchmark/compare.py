"""Public comparison with fairness checks (Phases 12-13).

Only published numbers with sources are used.  Each comparison row records
fairness attributes; when the Sweep run and the public result do not share
dataset/metric/hardware, the row is marked NOT DIRECTLY COMPARABLE.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = "benchmark/results"


def load_cpu_summary() -> dict:
    try:
        return json.loads(Path(f"{RESULTS}/summary.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


# Published references (all values come from the cited public sources).
PUBLIC_REFERENCES = [
    {
        "model": "GPT-4o",
        "organization": "OpenAI",
        "benchmark": "MMLU (5-shot)",
        "dataset": "MMLU",
        "accuracy": 0.887,
        "metric": "accuracy",
        "publication_date": "2024-05",
        "source": "https://openai.com/index/hello-gpt-4o/",
        "hardware": "not reported (API)",
        "notes": "Published in OpenAI's GPT-4o system card.",
    },
    {
        "model": "Claude 3.5 Sonnet",
        "organization": "Anthropic",
        "benchmark": "MMLU",
        "dataset": "MMLU",
        "accuracy": 0.887,
        "metric": "accuracy",
        "publication_date": "2024-06",
        "source": "https://www.anthropic.com/news/claude-3-5-sonnet",
        "hardware": "not reported (API)",
        "notes": "Published by Anthropic.",
    },
    {
        "model": "Gemini 1.5 Pro",
        "organization": "Google",
        "benchmark": "MMLU",
        "dataset": "MMLU",
        "accuracy": 0.853,
        "metric": "accuracy",
        "publication_date": "2024-02",
        "source": "https://storage.googleapis.com/deepmind-media/gemini/gemini_1_5_pro_report.pdf",
        "hardware": "not reported",
        "notes": "Gemini 1.5 Pro technical report.",
    },
    {
        "model": "GPT-4",
        "organization": "OpenAI",
        "benchmark": "MMLU",
        "dataset": "MMLU",
        "accuracy": 0.864,
        "metric": "accuracy",
        "publication_date": "2023-03",
        "source": "https://arxiv.org/abs/2303.08774",
        "hardware": "not reported",
        "notes": "GPT-4 technical report.",
    },
    {
        "model": "DeepSeek-V3",
        "organization": "DeepSeek",
        "benchmark": "MMLU (CoT)",
        "dataset": "MMLU",
        "accuracy": 0.884,
        "metric": "accuracy",
        "publication_date": "2024-12",
        "source": "https://arxiv.org/abs/2412.19437",
        "hardware": "not reported",
        "notes": "DeepSeek-V3 technical report.",
    },
    {
        "model": "LLaMA-3-70B",
        "organization": "Meta",
        "benchmark": "MMLU",
        "dataset": "MMLU",
        "accuracy": 0.82,
        "metric": "accuracy",
        "publication_date": "2024-04",
        "source": "https://ai.meta.com/blog/meta-llama-3/",
        "hardware": "not reported",
        "notes": "Meta LLaMA-3 blog.",
    },
    {
        "model": "BERT-large (fine-tuned)",
        "organization": "Google/community",
        "benchmark": "MNLI (GLUE)",
        "dataset": "MNLI",
        "accuracy": 0.867,
        "metric": "accuracy",
        "publication_date": "2018-10",
        "source": "https://arxiv.org/abs/1810.04805",
        "hardware": "GPU (TPU in paper)",
        "notes": "NLI task on which Sweep's evidence classifier is analogous.",
    },
    {
        "model": "distilbert-base",
        "organization": "HuggingFace",
        "benchmark": "MNLI",
        "dataset": "MNLI",
        "accuracy": 0.824,
        "metric": "accuracy",
        "publication_date": "2019-10",
        "source": "https://arxiv.org/abs/1910.01108",
        "hardware": "GPU",
        "notes": "DistilBERT paper.",
    },
]


def build_comparison() -> dict:
    summary = load_cpu_summary()
    sweep_acc = summary.get("accuracy")
    sweep_lat = (summary.get("latency_ms") or {}).get("median_p50")

    rows = []
    for ref in PUBLIC_REFERENCES:
        # Fairness: Sweep was evaluated on a private synthetic benchmark, not MMLU/MNLI.
        same_dataset = False
        comparable = "NOT DIRECTLY COMPARABLE"
        rows.append({
            "model": ref["model"],
            "organization": ref["organization"],
            "benchmark": ref["benchmark"],
            "accuracy": ref["accuracy"],
            "accuracy_pct": round(ref["accuracy"] * 100, 1),
            "dataset": ref["dataset"],
            "publication_date": ref["publication_date"],
            "source": ref["source"],
            "hardware": ref["hardware"],
            "notes": ref["notes"],
            "sweep_measured_accuracy": sweep_acc,
            "sweep_measured_median_latency_ms": sweep_lat,
            "comparable": comparable,
            "fairness": {
                "same_benchmark": False, "same_dataset": same_dataset,
                "same_metric": False, "same_hardware_class": False,
                "same_batch_size": False, "same_precision": False,
                "same_inference_conditions": False,
                "reason": ("Sweep evaluated on a private synthetic offline benchmark "
                           "(this repo), not on the public dataset of the reference."),
            },
        })

    return {
        "sweep": {
            "benchmark": "private synthetic Sweep suite (this repo)",
            "accuracy": sweep_acc,
            "median_latency_ms": sweep_lat,
            "hardware": "CPU-only, no GPU",
        },
        "rows": rows,
        "methodology": ("Sweep scores are measured locally on this machine. Reference "
                        "scores are published numbers from the cited sources. No row "
                        "shares the same dataset, so NONE are directly comparable; "
                        "differences in accuracy should not be read as superiority."),
    }


if __name__ == "__main__":
    print(json.dumps(build_comparison(), indent=1))
