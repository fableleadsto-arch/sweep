# Optional shared compute host

Status: experimental Python text host with device credentials and a real local
inference route, explicitly launched from the operator CLI. It is not integrated
with the installed desktop. Nothing listens on import or desktop startup; the
installed app remains standalone. Native host selection/pairing is still pending.

The intended product is one trusted host running heavy computation for lightweight
desktop clients. Additional clients should not need model weights or training
frameworks. Desktop actions, local file grants and approvals stay on the requesting
device. Native clients must obtain explicit permission to send conversation text
to a paired host; that desktop flow is still to be implemented. The current API
accepts text only and has no file, browser, shell or desktop-action routes.

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

## Authenticated text host

`companion.hosting` is a separate application; it does not mount the legacy brain
service or its execution, training and installation endpoints. Host operators
create organizations/users locally. Single-use, random 256-bit invitations expire
after ten minutes. Redemption transactionally issues a device token lasting 30
days; only SHA-256 digests of high-entropy invitations/tokens are stored in SQLite.
These are random bearer secrets, not user passwords. Device tokens are shown only
in the pairing response. Administrative device lists never return token digests.

Each authenticated request derives its organization/user identity from the
credential database. Multiple devices belonging to one user share the same queue
quota. Devices can revoke themselves; the local operator can revoke a device or
disable a user and all pending invites. Every subsequent API request checks the
database. Revocation does not retroactively cancel already admitted work; those
jobs can be cancelled by another authorized device for that user or host shutdown.

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/v1/health` | Minimal public service/scope status |
| POST | `/v1/pair` | Redeem `invitation` and `device_name`; return device credential |
| GET | `/v1/me` | Return only the authenticated device's principal |
| DELETE | `/v1/device` | Revoke the calling device |
| POST | `/v1/chat` | Queue `{messages:[{role,content},...]}`; return job ID |
| GET | `/v1/jobs/{id}` | Owner-scoped state, bounded preview or completed result |
| POST | `/v1/jobs/{id}/cancel` | Cancel only the authenticated user's job |

Authentication uses `Authorization: Bearer <device token>`. Conversations contain
at most 12 messages / 10,000 characters: user messages at most 6,000, assistant
history at most 2,000, with a nonblank final user message. System prompts, model
selection, tools, file paths and identity fields are forbidden. Results contain
plain text with `scope: text_only`; they never dispatch computer actions.

The worker reuses desktop downloaded-model verification and bounded local
streaming. Host-selected settings permit only loopback inference and reject cloud
aliases before sending text. Child processes load no model weights themselves;
the existing local engine performs inference. Normal completion/cancellation
removes each job's temporary input/status files; shutdown removes the owned temp
directory. Host inference settings and personal desktop history are not modified.

HTTP is accepted only for an actual loopback peer and loopback Host header.
Non-loopback serving requires TLS certificate/key files and explicit allowed host
names. The runner disables proxy-header trust, uses one service worker, and limits
concurrent connections. Browser Origin headers, query parameters, duplicate
authorization headers and encoded request bodies are rejected. Bodies are capped
at 32 KiB with a five-second read deadline. Validation responses do not echo input;
credential/result responses use `Cache-Control: no-store`. Global per-minute limits
are 30 pairing attempts and 1,200 credential checks, including result polling.
Clients should poll no more than once a second while queued/idle. Access logging
is disabled to avoid recording credentials accidentally placed in rejected URLs.

The transport guard uses [ASGI middleware](https://fastapi.tiangolo.com/advanced/middleware/).
Do not deploy this prototype behind a proxy that rewrites untrusted transport
metadata. Client certificate/host trust must be verified; no skip-verification
option is provided by the host workflow.

## Operator entry point

Run from the Python checkout/environment. These commands create host-specific
identity data; they do not change desktop chat history or start a browser.

```powershell
$sweepHostState = Join-Path $env:LOCALAPPDATA "Sweep/host/identities.sqlite"
$sweepOrg = .venv\Scripts\python.exe -m companion.hosting --state $sweepHostState organization "Personal"
$sweepUser = .venv\Scripts\python.exe -m companion.hosting --state $sweepHostState user $sweepOrg "My account"
.venv\Scripts\python.exe -m companion.hosting --state $sweepHostState invite $sweepUser
.venv\Scripts\python.exe -m companion.hosting --state $sweepHostState serve --config-root "$env:LOCALAPPDATA/Sweep"
```

The invitation is a secret for the intended device; share it through a trusted
channel. The final command stays in the foreground and listens on `127.0.0.1:8787`.
`devices USER_ID`, `revoke DEVICE_ID` and `disable USER_ID` manage credentials.
Use a host-private state directory: SQLite file mode is 0600 on POSIX; Windows
inherits the parent directory ACL. This is not isolation from the host operator.

Network serving additionally requires `--bind`, `--certfile`, `--keyfile` and
`--allow-host`. No network listener or permanent host configuration was installed
during validation. The installed desktop does not yet consume these endpoints.

## Required for desktop delivery and broader hosting

1. Native pairing/settings with explicit host certificate trust, protected client
   credential storage, revocation and standalone/host selection. The additional
   client must not need local model installation; native actions stay on it.
2. Client lifecycle, cancellation/disconnect behavior and bounded preview polling;
   keep remote text responses separate from local approved-action execution.
3. Real multi-device TLS/inference load tests and CPU/GPU memory measurements.
   Current tests use one machine. Process/request limits do not enforce engine
   RAM; killing a child closes its HTTP request without killing the shared engine
   or proving the engine has immediately stopped generation.
4. Durable jobs/restart recovery, a timed result-retention policy and cleanup of
   temp residue after host crashes. Results currently live in bounded memory;
   identity records persist, jobs do not. Generic subprocess worker files still
   have their original retention behavior; host text workers clean their own.
5. Per-user workspaces, file-transfer grants and OS isolation before any file or
   non-text task is offered remotely. Current worker processes run under the host
   account and the text host does not expose arbitrary worker specifications.

The `RemoteWorker` remains a legacy transport example; this milestone does not
use it as the host client or connect it to the desktop.

## Validation (2026-10-22)

Compute pool, existing compute and API tests: **83 passed**. These include real
CPU subprocess jobs for two owners, actual process cancellation/crash handling,
concurrent admission, owner checks, round-robin order, resource limits, idle
thread shutdown, timeout handling and both transient and persistent file locks.
An isolated interpreter verifies framework-free imports. These are single-host
tests, not multi-device/network acceptance tests. Ruff checks passed.

After adding identity, transport and real text-worker integration: **154 passed**
across host identity/API/worker/CLI, pool, compute and legacy API tests (37.20 s).
This includes SQLite replay/concurrent-redemption tests, expiry/revocation,
cross-user denial, client-field spoofing, unsafe transport/body limits, real child
processes against a controlled loopback engine, cloud-alias denial, cancellation
and removal of temporary input files. One upstream test-client deprecation warning
remains; Ruff checks passed.

A separate actual-model check ran a temporary loopback TCP host for two synthetic
users. Both paired and received replies; the second job was observed queued.
Foreign reads/cancellation returned 404, revoked credentials returned 401, and
personal inference settings remained byte-for-byte unchanged. First reply:
31.59 s to first preview / 32.55 s total; second reply: 35.31 s including queueing
(about 2.8 s after the first completed). Cold latency remains a significant gap.
No job files/processes or test listener remained after normal shutdown. This is
not a multi-machine test or a general latency guarantee.

A wider legacy capability run reported five failures because `sympy`,
`matplotlib`, `networkx`, `cv2`, `spacy` and `nltk` are absent from this environment.
Three other optional-framework checks were skipped. No dependency was installed
or test suppressed to turn those unavailable capabilities into a passing claim.
