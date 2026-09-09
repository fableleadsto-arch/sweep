from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

VERSION = "device-host/0.1.0"
_ALLOWED_URL_SCHEMES = ("http://", "https://")
_SAFE_APP_NAME = re.compile(r"^[A-Za-z0-9 ._+-]{1,64}$")
_SAFE_QUERY = re.compile(r"^[A-Za-z0-9 _.,:+\-/]{1,200}$")


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PermissionMode(str, Enum):
    ASK_EVERY_TIME = "ask_every_time"
    ALLOW_ONCE = "allow_once"
    ALLOW_FOR_SESSION = "allow_for_session"
    ALWAYS_ALLOW = "always_allow"
    DENY = "deny"
    PERMANENTLY_DENY = "permanently_deny"


class ActionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    DENIED = "denied"
    UNSUPPORTED = "unsupported"
    DISABLED = "disabled"


@dataclass(frozen=True)
class DeviceActionRequest:
    action: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False


@dataclass
class DeviceActionResult:
    success: bool
    action: str
    target: str = ""
    status: str = ActionStatus.COMPLETED.value
    message: str = ""
    error: dict[str, str] | None = None
    timestamp: str = ""
    execution_time_ms: int = 0
    risk_level: str = RiskLevel.LOW.value
    requires_confirmation: bool = False
    approval_request_id: str | None = None
    permission_decision: str = PermissionMode.ASK_EVERY_TIME.value
    dry_run: bool = False
    simulation: bool = False
    observation: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    description: str
    risk_level: RiskLevel
    requires_confirmation: bool
    timeout_seconds: int
    supported: bool = True


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    fingerprint: str
    request: DeviceActionRequest
    risk_level: RiskLevel
    created_at: float
    expires_at: float
    message: str


@dataclass
class DeviceHostConfig:
    enabled: bool = False
    simulation_mode: bool = False
    dry_run_default: bool = False
    permissions_file: str = ".relayhub/device-host-permissions.json"
    audit_file: str = ".relayhub/device-host-audit.jsonl"
    allowed_roots: list[str] = field(default_factory=list)
    path_aliases: dict[str, str] = field(default_factory=dict)
    app_allowlist: dict[str, list[str]] = field(default_factory=dict)
    max_directory_entries: int = 100
    max_search_results: int = 50
    approval_ttl_seconds: int = 300

    def normalized_aliases(self) -> dict[str, Path]:
        aliases = dict(self.path_aliases)
        for root in self.allowed_roots:
            resolved = Path(root).expanduser().resolve(strict=False)
            aliases.setdefault(resolved.name.lower(), str(resolved))
        return {k.lower(): Path(v).expanduser().resolve(strict=False) for k, v in aliases.items()}


class LocalOSAdapter:
    def open_url(self, url: str) -> dict[str, Any]:
        ok = bool(webbrowser.open(url, new=0, autoraise=False))
        return {"opened": ok}

    def open_application(self, executable: str, args: list[str] | None = None) -> dict[str, Any]:
        subprocess.Popen([executable, *(args or [])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"pid_available": True}

    def open_path(self, path: Path) -> dict[str, Any]:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"path": str(path)}

    def system_info(self) -> dict[str, Any]:
        uname = platform.uname()
        return {
            "system": uname.system,
            "release": uname.release,
            "machine": uname.machine,
            "python": platform.python_version(),
        }


class SimulatedDeviceAdapter(LocalOSAdapter):
    def __init__(self) -> None:
        self.opened_urls: list[str] = []
        self.opened_paths: list[str] = []
        self.opened_apps: list[str] = []

    def open_url(self, url: str) -> dict[str, Any]:
        self.opened_urls.append(url)
        return {"opened": True, "url": url}

    def open_application(self, executable: str, args: list[str] | None = None) -> dict[str, Any]:
        self.opened_apps.append(executable)
        return {"opened": True, "executable": executable, "args": args or []}

    def open_path(self, path: Path) -> dict[str, Any]:
        self.opened_paths.append(str(path))
        return {"opened": True, "path": str(path)}

    def system_info(self) -> dict[str, Any]:
        info = super().system_info()
        info["simulated"] = True
        return info


class PermissionStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._rules: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            self._rules = {str(k): str(v) for k, v in data.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._rules, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, fingerprint: str) -> str | None:
        return self._rules.get(fingerprint)

    def set(self, fingerprint: str, mode: PermissionMode) -> None:
        self._rules[fingerprint] = mode.value
        self.save()


class DeviceHost:
    def __init__(self, config: DeviceHostConfig, adapter: LocalOSAdapter | None = None) -> None:
        self.config = config
        self.adapter = adapter or (SimulatedDeviceAdapter() if config.simulation_mode else LocalOSAdapter())
        self.permissions = PermissionStore(config.permissions_file)
        self._session_rules: dict[str, str] = {}
        self._pending: dict[str, ApprovalRequest] = {}
        self._aliases = config.normalized_aliases()
        self._specs = self._build_specs()

    def _build_specs(self) -> dict[str, ActionSpec]:
        return {
            "open_application": ActionSpec("open_application", "Launch an allowlisted application", RiskLevel.LOW, True, 20),
            "open_url": ActionSpec("open_url", "Open a web URL in the default browser", RiskLevel.LOW, True, 20),
            "open_file": ActionSpec("open_file", "Open a local file under an allowed root", RiskLevel.MEDIUM, True, 20),
            "open_folder": ActionSpec("open_folder", "Open a local folder under an allowed root", RiskLevel.LOW, True, 20),
            "create_folder": ActionSpec("create_folder", "Create a folder under an allowed root", RiskLevel.MEDIUM, True, 20),
            "list_directory": ActionSpec("list_directory", "List directory entries under an allowed root", RiskLevel.LOW, False, 10),
            "search_local_files": ActionSpec("search_local_files", "Search filenames under an allowed root", RiskLevel.MEDIUM, True, 20),
            "get_system_information": ActionSpec("get_system_information", "Return basic host information", RiskLevel.LOW, False, 5),
        }

    def capabilities(self) -> list[str]:
        return sorted(self._specs)

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.config.enabled,
            "enabled": self.config.enabled,
            "version": VERSION,
            "platform": platform.system().lower() or os.name,
            "dry_run_default": self.config.dry_run_default,
            "simulation_mode": self.config.simulation_mode,
            "capabilities": self.capabilities(),
            "pending_approvals": len(self._pending),
            "security": {
                "arbitrary_shell_execution": False,
                "allowlisted_actions_only": True,
                "confirmation_required_without_rule": True,
            },
        }

    def execute(self, request: DeviceActionRequest) -> DeviceActionResult:
        spec = self._specs.get(request.action)
        if not self.config.enabled:
            return self._result(False, request, ActionStatus.DISABLED, PermissionMode.DENY, RiskLevel.LOW, "Device Host is disabled.", code="DEVICE_HOST_DISABLED")
        if spec is None:
            return self._result(False, request, ActionStatus.UNSUPPORTED, PermissionMode.DENY, RiskLevel.HIGH, f"Action '{request.action}' is not supported.", code="ACTION_UNSUPPORTED")
        try:
            normalized = self._normalize_request(request, spec)
        except ValueError as exc:
            return self._result(False, request, ActionStatus.FAILED, PermissionMode.DENY, spec.risk_level, str(exc), code="INVALID_REQUEST")
        fingerprint = self._fingerprint(normalized)
        rule = self._session_rules.get(fingerprint) or self.permissions.get(fingerprint)
        if rule in (PermissionMode.PERMANENTLY_DENY.value, PermissionMode.DENY.value):
            return self._result(False, normalized, ActionStatus.DENIED, PermissionMode(rule), spec.risk_level, "Action denied by policy.", code="PERMISSION_DENIED")
        if spec.requires_confirmation and rule not in (PermissionMode.ALLOW_FOR_SESSION.value, PermissionMode.ALWAYS_ALLOW.value):
            approval = self._queue_approval(normalized, spec, fingerprint)
            return self._result(False, normalized, ActionStatus.REQUIRES_CONFIRMATION, PermissionMode.ASK_EVERY_TIME, spec.risk_level, approval.message, approval_id=approval.approval_id, requires_confirmation=True, code="CONFIRMATION_REQUIRED")
        if not spec.requires_confirmation and rule == PermissionMode.PERMANENTLY_DENY.value:
            return self._result(False, normalized, ActionStatus.DENIED, PermissionMode.PERMANENTLY_DENY, spec.risk_level, "Action denied by policy.", code="PERMISSION_DENIED")
        return self._dispatch(normalized, spec, PermissionMode(rule or PermissionMode.ALLOW_FOR_SESSION.value))

    def approve(self, approval_id: str, decision: PermissionMode) -> DeviceActionResult:
        approval = self._pending.get(approval_id)
        if approval is None or approval.expires_at < time.time():
            self._pending.pop(approval_id, None)
            request = DeviceActionRequest("approval")
            return self._result(False, request, ActionStatus.FAILED, PermissionMode.DENY, RiskLevel.LOW, "Approval request is missing or expired.", code="APPROVAL_NOT_FOUND")
        self._pending.pop(approval_id, None)
        if decision == PermissionMode.ALLOW_FOR_SESSION:
            self._session_rules[approval.fingerprint] = decision.value
        elif decision == PermissionMode.ALWAYS_ALLOW:
            self.permissions.set(approval.fingerprint, decision)
        elif decision == PermissionMode.PERMANENTLY_DENY:
            self.permissions.set(approval.fingerprint, decision)
            return self._result(False, approval.request, ActionStatus.DENIED, decision, approval.risk_level, "Action permanently denied.", code="PERMISSION_DENIED")
        elif decision == PermissionMode.DENY:
            return self._result(False, approval.request, ActionStatus.DENIED, decision, approval.risk_level, "Action denied.", code="PERMISSION_DENIED")
        elif decision != PermissionMode.ALLOW_ONCE:
            return self._result(False, approval.request, ActionStatus.FAILED, PermissionMode.DENY, approval.risk_level, "Unsupported approval decision.", code="INVALID_APPROVAL")
        return self._dispatch(approval.request, self._specs[approval.request.action], decision)

    def audit_history(self) -> list[dict[str, Any]]:
        path = Path(self.config.audit_file)
        if not path.exists():
            return []
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return lines

    def _queue_approval(self, request: DeviceActionRequest, spec: ActionSpec, fingerprint: str) -> ApprovalRequest:
        approval = ApprovalRequest(
            approval_id=f"approval_{uuid.uuid4().hex[:12]}",
            fingerprint=fingerprint,
            request=request,
            risk_level=spec.risk_level,
            created_at=time.time(),
            expires_at=time.time() + self.config.approval_ttl_seconds,
            message=f"Allow SWEEP to perform {request.action} on {request.target or 'this target'}?",
        )
        self._pending[approval.approval_id] = approval
        return approval

    def _dispatch(self, request: DeviceActionRequest, spec: ActionSpec, decision: PermissionMode) -> DeviceActionResult:
        started = time.perf_counter()
        dry_run = self.config.dry_run_default or request.dry_run
        try:
            if request.action == "get_system_information":
                data = self.adapter.system_info()
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, "Collected system information.", data=data, observation="System information returned.", dry_run=dry_run, started=started)
            if request.action == "list_directory":
                path = self._resolve_allowed_path(request.target)
                entries = self._list_directory(path)
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Listed {len(entries)} entries.", data={"entries": entries, "path": str(path)}, observation=f"Folder contains {len(entries)} entries.", dry_run=dry_run, started=started)
            if request.action == "search_local_files":
                root = self._resolve_allowed_path(request.target)
                entries = self._search_files(root, str(request.parameters.get("query", "")))
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Found {len(entries)} matches.", data={"matches": entries, "path": str(root)}, observation=f"Search returned {len(entries)} matches.", dry_run=dry_run, started=started)
            if request.action == "create_folder":
                path = self._resolve_allowed_path(request.target)
                if dry_run:
                    return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Would create folder {path}", data={"path": str(path)}, observation="No changes were made.", dry_run=True, started=started)
                path.mkdir(parents=True, exist_ok=True)
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Created folder {path}", data={"path": str(path)}, observation="Folder created.", dry_run=False, started=started)
            if request.action == "open_url":
                url = self._validate_url(request.target)
                if dry_run:
                    return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Would open {url}", data={"url": url}, observation="No browser action taken.", dry_run=True, started=started)
                data = self.adapter.open_url(url)
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Opened {url}", data=data, observation="Browser open requested.", dry_run=False, started=started)
            if request.action == "open_application":
                executable = self._resolve_application(request.target)
                if dry_run:
                    return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Would open application {request.target}", data={"executable": executable}, observation="No application launched.", dry_run=True, started=started)
                data = self.adapter.open_application(executable)
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Opened application {request.target}", data=data, observation="Application launch requested.", dry_run=False, started=started)
            if request.action in {"open_folder", "open_file"}:
                path = self._resolve_allowed_path(request.target)
                if request.action == "open_folder" and not path.is_dir():
                    raise ValueError("Requested folder does not exist.")
                if request.action == "open_file" and not path.is_file():
                    raise ValueError("Requested file does not exist.")
                if dry_run:
                    return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Would open {path}", data={"path": str(path)}, observation="No application launch requested.", dry_run=True, started=started)
                data = self.adapter.open_path(path)
                return self._result(True, request, ActionStatus.COMPLETED, decision, spec.risk_level, f"Opened {path}", data=data, observation="Open request sent to the OS.", dry_run=False, started=started)
            raise ValueError("Action handler not implemented.")
        except ValueError as exc:
            return self._result(False, request, ActionStatus.FAILED, decision, spec.risk_level, str(exc), code="ACTION_FAILED", started=started)

    def _normalize_request(self, request: DeviceActionRequest, spec: ActionSpec) -> DeviceActionRequest:
        parameters = dict(request.parameters)
        if not isinstance(parameters, dict):
            raise ValueError("Action parameters must be an object.")
        if request.action in {"open_application", "open_url", "open_file", "open_folder", "create_folder", "list_directory", "search_local_files"} and not request.target.strip():
            raise ValueError("Action target is required.")
        if request.action == "open_application":
            target = request.target.strip().lower()
            if not _SAFE_APP_NAME.fullmatch(target):
                raise ValueError("Application name contains invalid characters.")
            parameters = {}
            return DeviceActionRequest(request.action, target, parameters, bool(request.dry_run))
        if request.action == "open_url":
            return DeviceActionRequest(request.action, self._validate_url(request.target), {}, bool(request.dry_run))
        if request.action in {"open_file", "open_folder", "create_folder", "list_directory"}:
            self._resolve_allowed_path(request.target)
        if request.action == "search_local_files":
            self._resolve_allowed_path(request.target)
            query = str(parameters.get("query", "")).strip()
            if not _SAFE_QUERY.fullmatch(query):
                raise ValueError("Search query contains invalid characters.")
            parameters = {"query": query}
        return DeviceActionRequest(request.action, request.target.strip(), parameters, bool(request.dry_run))

    def _validate_url(self, url: str) -> str:
        value = url.strip()
        if not value.startswith(_ALLOWED_URL_SCHEMES):
            raise ValueError("Only http and https URLs are allowed.")
        return value

    def _fingerprint(self, request: DeviceActionRequest) -> str:
        payload = json.dumps({"action": request.action, "target": request.target, "parameters": request.parameters}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_allowed_path(self, target: str) -> Path:
        raw = target.strip()
        if raw.startswith("alias:"):
            rest = raw[6:]
            alias, _, tail = rest.partition("/")
            base = self._aliases.get(alias.lower())
            if base is None:
                raise ValueError("Unknown path alias.")
            candidate = (base / tail).expanduser().resolve(strict=False)
            if not (candidate == base or base in candidate.parents):
                raise ValueError("Path escapes the requested alias root.")
            return candidate
        candidate = Path(raw).expanduser().resolve(strict=False)
        allowed = list(self._aliases.values()) + [Path(p).expanduser().resolve(strict=False) for p in self.config.allowed_roots]
        if not allowed:
            raise ValueError("No allowed roots are configured.")
        if not any(candidate == root or root in candidate.parents for root in allowed):
            raise ValueError("Path is outside allowed roots.")
        return candidate

    def _resolve_application(self, target: str) -> str:
        candidates = self.config.app_allowlist.get(target.lower())
        if not candidates:
            raise ValueError("Application is not allowlisted.")
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        if self.config.simulation_mode:
            return candidates[0]
        raise ValueError("Application could not be located.")

    def _list_directory(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_dir():
            raise ValueError("Requested directory does not exist.")
        entries = []
        for item in sorted(path.iterdir(), key=lambda p: p.name.lower())[: self.config.max_directory_entries]:
            entries.append({"name": item.name, "is_dir": item.is_dir(), "size": item.stat().st_size if item.is_file() else None})
        return entries

    def _search_files(self, root: Path, query: str) -> list[dict[str, Any]]:
        if not root.is_dir():
            raise ValueError("Requested search root does not exist.")
        matches: list[dict[str, Any]] = []
        lowered = query.lower()
        for item in root.rglob("*"):
            if len(matches) >= self.config.max_search_results:
                break
            if lowered in item.name.lower():
                matches.append({"name": item.name, "path": str(item), "is_dir": item.is_dir()})
        return matches

    def _result(self, success: bool, request: DeviceActionRequest, status: ActionStatus, decision: PermissionMode, risk: RiskLevel, message: str, *, code: str | None = None, approval_id: str | None = None, requires_confirmation: bool = False, data: dict[str, Any] | None = None, observation: str = "", dry_run: bool = False, started: float | None = None) -> DeviceActionResult:
        result = DeviceActionResult(
            success=success,
            action=request.action,
            target=request.target,
            status=status.value,
            message=message,
            error=None if code is None else {"code": code, "message": message},
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            execution_time_ms=0 if started is None else int((time.perf_counter() - started) * 1000),
            risk_level=risk.value,
            requires_confirmation=requires_confirmation,
            approval_request_id=approval_id,
            permission_decision=decision.value,
            dry_run=dry_run,
            simulation=self.config.simulation_mode,
            observation=observation,
            data=data or {},
        )
        self._append_audit(result)
        return result

    def _append_audit(self, result: DeviceActionResult) -> None:
        path = Path(self.config.audit_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": result.timestamp,
            "action": result.action,
            "target": result.target,
            "status": result.status,
            "duration_ms": result.execution_time_ms,
            "permission_decision": result.permission_decision,
            "error_code": (result.error or {}).get("code"),
            "dry_run": result.dry_run,
            "simulation": result.simulation,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
