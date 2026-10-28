# Processing cost and unit economics

2026-10-28. Status: **Partially implemented**. Local document-foundation timing
and process-memory smoke measurements exist; monetary economics and customer
capacity remain unmeasured. No cloud charges were incurred for this milestone.

## Reproducible baseline

Run `.venv/Scripts/python.exe scripts/benchmark_documents.py --runs 3`. It creates
three small fixed synthetic files, launches fresh source Python processes, measures
first/repeated extraction and verifies exact references and source preservation.
It prints raw samples and code/fixture hashes. No customer files, downloaded
datasets, model service or optional orchestration framework is used.

The benchmark's `cold_process_seconds` includes interpreter startup and script
overhead. `first_reader_seconds` includes importing the document reader;
`repeated_reader_seconds` is a second extraction in the same process. CPU time
covers reader import and both reads plus reference validation. OS peak working
set/RSS measures this child process, including imports, not just Python allocations.
The parent fixture generator and the whole native application are excluded. File
system caches are not flushed; cold process is not guaranteed cold storage.

Raw measured samples and environment are in
[the baseline artifact](evaluations/document-foundation-2026-10-28.json).
Hardware: i5-1335U, 10 physical/12 logical cores, about 15.72 GiB reported RAM.
Python 3.13.7 on Windows; pypdf 6.19.0. Only three launches per format and tiny
69-character documents were measured. This establishes a smoke baseline, not
large-document performance, commercial accuracy or p95 latency. Chat cold-start
delays are separate and remain unresolved.

| Format | Fresh process wall time (3 samples) | Peak child process memory |
| --- | --- | --- |
| txt | 0.280 s, 0.225 s, 0.227 s | 47.4–48.0 MiB |
| docx | 0.234 s, 0.221 s, 0.229 s | 47.9–48.1 MiB |
| pdf | 0.448 s, 0.369 s, 0.376 s | 56.7–56.9 MiB |

## Missing measurements and metering design

Before pilot: measure 30+ samples per relevant workload with sizes/pages/formats,
scan fraction and both failures/successes. Measure UI idle memory, entire process
tree peak RAM, OCR/model residency, cancellation, queued latency and retention
growth. Shared-host throughput must be measured across authenticated users on
actual supported hardware; fair scheduling alone is not capacity or cost evidence.

Future local usage events should record workspace-scoped operation ID, type,
duration, CPU/peak-memory scope, pages/bytes, parser/model revision and outcome,
without raw document text or secret values. Retrying a job records a separate
attempt; billing semantics must prevent accidental double charging. Model calls
need measured tokens/provider charges where available; do not estimate tokens
from character counts and present them as billed amounts.

## Economic model (not measured values)

Cost per completed workflow includes compute (local energy/hardware allocation or
authorized cloud billing), storage/retention, integration fees, failure retries,
support and human review. Gross margin requires actual collected revenue and
attributable cost. Local processing is not free: electricity, equipment and support
are not measured here. Prices, revenue, conversion and margin remain unknown.

Future cloud use requires explicit routing permission, per-user/workspace budget,
pre-admission cost checks, quotas and cancellation policy. Budget uncertainty must
be shown; a remote timeout may still incur charges. No cloud backend, billing SDK,
telemetry service or automatic peer sharing is enabled by this document.
