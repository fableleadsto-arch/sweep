from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_ALIAS_WORDS = {
    "downloads": "downloads",
    "download": "downloads",
    "documents": "documents",
    "document": "documents",
    "desktop": "desktop",
    "home": "home",
    "temp": "temp",
    "tmp": "temp",
}

_APP_NAMES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "chromium": "chrome",
    "vscode": "vscode",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "discord": "discord",
    "terminal": "terminal",
}

_DOMAIN = re.compile(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/\S*)?$")


@dataclass(frozen=True)
class PlannedDeviceAction:
    action: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceIntentPlan:
    recognized: bool
    actions: list[PlannedDeviceAction] = field(default_factory=list)
    clarification: str = ""
    confidence: float = 0.0


def _normalize_url(value: str) -> str:
    value = value.strip().rstrip(".")
    if value.startswith(("http://", "https://")):
        return value
    if _DOMAIN.match(value):
        return "https://" + value
    return value


def _alias_target(name: str, remainder: str = "") -> str:
    alias = _ALIAS_WORDS[name.lower()]
    return f"alias:{alias}/{remainder}".rstrip("/")


def plan_device_actions(message: str) -> DeviceIntentPlan:
    text = " ".join(message.strip().split())
    lower = text.lower()
    if not text:
        return DeviceIntentPlan(False)

    match = re.match(r"^(?:open|launch)\s+(.+?)\s+and\s+(?:go\s+to|open)\s+(\S.+?)\.?$", lower)
    if match:
        app_name = _APP_NAMES.get(match.group(1).strip())
        url = _normalize_url(match.group(2))
        if app_name and url.startswith(("http://", "https://")):
            return DeviceIntentPlan(
                True,
                actions=[
                    PlannedDeviceAction("open_application", app_name),
                    PlannedDeviceAction("open_url", url),
                ],
                confidence=0.92,
            )

    match = re.match(r"^(?:open|launch)\s+(.+?)\.?$", lower)
    if match:
        target = match.group(1).strip()
        app_name = _APP_NAMES.get(target)
        if app_name:
            return DeviceIntentPlan(True, [PlannedDeviceAction("open_application", app_name)], confidence=0.93)
        alias_match = re.match(r"(?:my\s+)?(downloads|documents|desktop|home|temp|tmp)(?:\s+folder)?$", target)
        if alias_match:
            return DeviceIntentPlan(True, [PlannedDeviceAction("open_folder", _alias_target(alias_match.group(1)))], confidence=0.95)
        url = _normalize_url(target)
        if url.startswith(("http://", "https://")):
            return DeviceIntentPlan(True, [PlannedDeviceAction("open_url", url)], confidence=0.9)

    match = re.match(r"^(?:go\s+to|open)\s+(https?://\S+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/\S*)?)\.?$", lower)
    if match:
        return DeviceIntentPlan(True, [PlannedDeviceAction("open_url", _normalize_url(match.group(1)))], confidence=0.89)

    match = re.match(r"^(?:show\s+me\s+)?(?:the\s+)?contents\s+of\s+(?:my\s+)?(downloads|documents|desktop|home|temp|tmp)(?:\s+folder)?\.?$", lower)
    if match:
        return DeviceIntentPlan(True, [PlannedDeviceAction("list_directory", _alias_target(match.group(1)))], confidence=0.88)

    match = re.match(r"^search\s+(?:my\s+)?local\s+files\s+for\s+(.+?)(?:\s+in\s+(downloads|documents|desktop|home|temp|tmp))?\.?$", lower)
    if match:
        query = match.group(1).strip().strip('"')
        location = match.group(2) or "home"
        return DeviceIntentPlan(
            True,
            [PlannedDeviceAction("search_local_files", _alias_target(location), {"query": query})],
            confidence=0.87,
        )

    match = re.match(r"^create\s+(?:a\s+)?folder\s+called\s+(.+?)(?:\s+in\s+(downloads|documents|desktop|home|temp|tmp))?\.?$", lower)
    if match:
        name = match.group(1).strip().strip('"')
        location = match.group(2)
        if not location:
            return DeviceIntentPlan(True, clarification="Folder location is required before execution.", confidence=0.78)
        return DeviceIntentPlan(
            True,
            [PlannedDeviceAction("create_folder", _alias_target(location, name))],
            confidence=0.86,
        )

    return DeviceIntentPlan(False)
