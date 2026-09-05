# SWEEP Competitive Matrix

*Date: 2026-09-05. Rules followed: no claim is stated as fact without a source; unverified claims are labeled; every row carries source + test date + confidence. No system here was evaluated on Sweep's private dataset except Sweep itself, so all cross-system comparisons are **NOT DIRECTLY COMPARABLE**.*

## 1. Scope

Sweep's measured surface today is: offline deterministic reasoning (logic, evidence classification, contradiction detection, abstention). It is **not** an LLM and currently has no general-knowledge chat capability; comparisons to LLMs are context, not scoreboards. Relevant competitor categories for the full platform vision: research agents, deep-research systems, OSINT/investigation platforms, evidence/verification tooling.

## 2. Sweep — measured (local, reproducible)

Source: `benchmark/REPORT.md` (2026-09, Git SHA 5375b35f, suite SHA-256 e16e29d7, CPU-only, offline). Confidence: HIGH (local, reproducible, documented).

| Dimension | Value |
|---|---|
| Overall accuracy | 65.3% (436/668) — historical tree; see update below |
| Basic logic | 94.7% |
| Multi-step | 80.0% |
| Adversarial | 81.8% |
| Evidence reasoning | 66.7% (degenerate artifact — fixed 2026-09-05) |
| Ambiguity/abstention | 0.0% (0/108) — historical; fixed 2026-09-05 |
| Calibration (ECE) | 0.1746 |
| Median latency | 50.4 ms (p95 89.7 ms) |
| Throughput | 26.1 cases/s |
| Peak RAM | 1805.6 MB |
| Cost per query | $0 (local CPU) |
| Web research / multimodal / citation accuracy | NOT TESTED (no suite exists) |

**Update (2026-09-05, current tree measured fresh-process):** overall **668/668 = 100.0%**, p50 0.45–0.60 ms, 192–230 cases/s, peak RAM ~1322 MB — before and after this session's artifact/abstention fixes. **Interpretation rule:** 100% on a suite that was generated post-hoc and tuned against means the suite is SATURATED, not that the system is perfect. It supports no comparative claim against any other system (no other system has run this suite) and cannot measure further improvement. The only supportable differentiators remain latency, $0 marginal cost, determinism, and CPU-only operation. A hidden evaluation set is required before any "better than X" statement (SWEEP_ROADMAP.md P1 #7).

## 3. Public LLM context (published self-reports; NOT DIRECTLY COMPARABLE)

From `benchmark/REPORT.md` §14 (sources verified at report time; confidence MEDIUM — vendor self-reported):

| System | Benchmark | Score | Source |
|---|---|---|---|
| GPT-4o | MMLU 5-shot | 88.7% | openai.com/index/hello-gpt-4o |
| Claude 3.5 Sonnet | MMLU | 88.7% | anthopic.com/news/claude-3-5-sonnet |
| Gemini 1.5 Pro | MMLU | 85.3% | Gemini 1.5 tech report (Google storage) |
| GPT-4 | MMLU | 86.4% | arxiv.org/abs/2303.08774 |
| DeepSeek-V3 | MMLU CoT | 88.4% | arxiv.org/abs/2412.19437 |
| LLaMA-3-70B | MMLU | 82.0% | ai.meta.com/blog/meta-llama-3 |

Landscape update (2026-09, from leaderboard aggregators; confidence LOW-MEDIUM — aggregator data, not primary reports, not independently verified here):

- MMLU/GSM8K/HumanEval are saturated (top models cluster 88–99%); GPQA Diamond and MMLU-Pro are the differentiating benchmarks (iternal.ai selection guide; artificialanalysis.ai/evaluations/mmlu-pro).
- Aggregators currently list frontier reasoning models at 90%+ GPQA Diamond (llm-stats.com, retrieved 2026-09-05; NOT VERIFIED against primary reports).

**Do not compare the numbers in §2 against §3.** Different datasets, modes, and model classes. Where Sweep can honestly differentiate today: latency (ms vs seconds), cost ($0 vs per-token), CPU-only operation, deterministic reproducibility. Where it is objectively behind: general knowledge, robustness, calibration, abstention.

## 4. Competitor categories (capability comparison, not scores)

Confidence for unscored rows: qualitative, from public product documentation — treat as leads to verify, not verdicts.

| Category | Examples | What they demonstrate publicly | Sweep today |
|---|---|---|---|
| Deep-research agents | ChatGPT Deep Research, Gemini Deep Research, Perplexity Pro, open-source GPT-Researcher | multi-step web research, cited reports, minutes-scale latency | Web layer exists (search/crawl/extract) but research loop + citation engine NOT TESTED |
| Search agents / APIs | Perplexity API, Exa, Tavily, You.com | retrieval quality, API access | Sweep has 6 HTML parsers + 4 API providers; retrieval quality NOT MEASURED |
| OSINT/investigation | Maltego, Bellingcat toolkit, SpiderFoot, ShadowDragon | transforms, entity linking, investigation graphs | Entity/Evidence data contracts exist; no ER engine, no source-dependency graph |
| Knowledge graphs / verification | Google Knowledge Graph (closed), Wikidata (open), claim-review systems | structured entities, provenance | evidence_graph module exists in neural mesh; no provenance store, no snapshots |
| Verification / fact-check tooling | ClaimReview ecosystem, Full Fact tooling (public writeups) | claim extraction + check-worthiness | Fact-checking gate NOT IMPLEMENTED |

## 5. Honest scorecard (populated only from actual tests)

Legend: values from measured tests only. Blank cells = NOT TESTED — never invent.

| Dimension | SWEEP | Frontier LLMs | Deep-research agents | OSINT platforms |
|---|---|---|---|---|
| Deterministic logic (offline) | 94.7% | NOT TESTED on our suite | — | — |
| Contradiction detection (pairs) | 100% P/R | NOT TESTED on our suite | — | — |
| Evidence classification | 66.7% → fix in progress | NOT TESTED on our suite | — | — |
| Abstention on ambiguity | 0.0% | NOT TESTED on our suite | — | — |
| Web research quality | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Multimodal | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Latency (per query, offline) | 50 ms p50 | NOT MEASURED here | NOT MEASURED here | — |
| Cost per query | $0 | >$0 | >$0 | >$0 |

**Building the shared testbeds (web research suite, citation accuracy suite) is the prerequisite for any future "better than X" statement.** Until then the only supportable claims are the measured local ones above.
