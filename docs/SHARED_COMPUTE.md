# Optional shared compute host

Status: Python scheduling foundation, not a deployable multi-user service.
The installed desktop remains standalone and local. No listener, pairing flow,
remote inference route or automatic peer participation is enabled by this work.

The intended product is one trusted host running heavy computation for lightweight
desktop clients. Additional clients should not need model weights or training
frameworks. Desktop actions, local file grants and approvals stay on the requesting
device. Files or conversation content are sent to a host only through an explicit
paired-host flow that is still to be implemented.

## Implemented foundation

`companion.compute.FairComputePool` wraps an existing `ComputeWorker`, including
the real `LocalSubprocessWorker`. It gives the wrapper exclusive ownership of that
worker; callers must not submit to the underlying worker simultaneously.

- One running process by default; configurable total and per-owner concurrency.
- At most 32 outstanding jobs overall and eight per owner by default. Outstanding
  includes running work. Rejected submissions raise `QueueFullError`; they are
  not silently dropped or launched beyond the limit.
- FIFO order for each owner, round-robin dispatch between eligible owners.
  This is fairness by job dispatch, not equal CPU time or multi-machine balancing.
  Running jobs are not preempted when another user arrives.
- Host-generated IDs and a copied JSON specification, limited to 64 KiB.
- Status and cancellation require the same owner identity used at admission.
  Unknown, expired and foreign job IDs have the same not-found behavior.
- Cancellation of queued work prevents execution. A running slot is held until
  the worker reports termination. Local cancellation terminates, waits, then
  escalates to killing the owned process if necessary.
- A five-minute job timeout by default; the slot stays occupied if cancellation
  cannot be confirmed. Shutdown closes admission and stops owned work.
- At most 64 completed status records in memory by default. Idle polling threads
  exit and restart only when new work arrives. Local status reads are capped at
  1 MiB and exited process handles are removed from tracking.
- Importing the pool and querying the CPU backend version loads no numerical or
  model frameworks and starts no threads. NumPy is imported only for execution;
  an installation without it correctly reports the CPU backend as unavailable.

The local worker rejects unsafe paths/reserved Windows names in job IDs and will
not overwrite an existing job specification. It waits for actual process exit
before returning a terminal result, detects crashes without valid results, and
does not invent cancellation records for nonexistent or finished jobs. Child
status updates replace the file atomically, with a bounded retry for transient
Windows sharing locks. Requested task kinds now reach the
backend correctly instead of falling through to generic capability execution.

## Python usage for trusted callers

```python
import time

from companion.compute import FairComputePool
from companion.compute.workers import LocalSubprocessWorker, WorkerJob

pool = FairComputePool(LocalSubprocessWorker(".relayhub/workers"))
try:
    # A future service must derive this from authenticated credentials.
    owner = "example-local-owner"
    job_id = pool.submit(owner, WorkerJob(
        kind="capability",
        backend="cpu",
        spec={"capability": "math", "data": "1 2 3 4"},
    ))
    deadline = time.monotonic() + 60
    status = pool.status(owner, job_id)
    while status.state in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.1)
        status = pool.status(owner, job_id)
    print(status.to_dict())  # inspect state/error before using result
finally:
    pool.shutdown()  # also cancels work if the caller's deadline expires
```

## Required before exposing a shared host

1. Pairing and credential lifecycle with authenticated user/organization identity.
   Never take the pool's owner argument from a client-supplied owner field. The
   legacy service-wide bearer token is not multi-user authentication.
2. An authenticated transport, explicit host trust and revocation, request size
   limits, and a narrow supported-operation contract. Do not expose arbitrary
   `WorkerJob` specifications or legacy training/execution capabilities remotely.
3. Per-user workspaces, file-transfer grants, access checks, result retention and
   deletion. The scheduler's owner checks are not an OS sandbox: local worker
   processes currently run with the host account's permissions. Disk job files
   are retained separately; memory eviction does not delete them.
4. Durable job ownership, restart recovery, quotas and disconnect/cancellation
   behavior. This pool is in-memory and does not resume jobs after a host crash.
5. Desktop selection of standalone versus paired-host mode, without requiring
   local model installation in the latter. Keep native actions on the client.
6. Real multi-device inference/load tests, latency and memory measurements,
   streaming backpressure and deployment-specific CPU/GPU limits. Process-count
   caps are not OS memory limits, nor do they manage arbitrary descendant trees.

The `RemoteWorker` remains a legacy transport example; this milestone does not
validate it as a secure shared-host client or connect it to the desktop.

## Validation (2026-10-22)

Compute pool, existing compute and API tests: **83 passed**. These include real
CPU subprocess jobs for two owners, actual process cancellation/crash handling,
concurrent admission, owner checks, round-robin order, resource limits, idle
thread shutdown, timeout handling and both transient and persistent file locks.
An isolated interpreter verifies framework-free imports. These are single-host
tests, not multi-device/network acceptance tests. Ruff checks passed.

A wider legacy capability run reported five failures because `sympy`,
`matplotlib`, `networkx`, `cv2`, `spacy` and `nltk` are absent from this environment.
Three other optional-framework checks were skipped. No dependency was installed
or test suppressed to turn those unavailable capabilities into a passing claim.
