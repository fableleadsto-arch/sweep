"""Bluetooth integrations via Bleak (cross-platform BLE).

BlueZ itself is the Linux kernel Bluetooth stack and cannot run on
Windows; on Linux hosts Sweep's BLE path still goes through Bleak,
which uses BlueZ underneath.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sweep_core.integrations import _module_available


def availability() -> dict[str, Any]:
    return {
        "bleak": {"available": _module_available("bleak")},
        "bluez": {
            "available": False,
            "reason": "Linux kernel stack; consumed indirectly through Bleak on Linux",
        },
    }


async def scan_devices(timeout: float = 5.0) -> list[dict[str, Any]]:
    """Scan for BLE advertisements and return discovered devices."""
    from bleak import BleakScanner

    devices = await BleakScanner.discover(timeout=timeout)
    return [
        {
            "address": device.address,
            "name": device.name,
            "rssi": details.get("rssi") if isinstance(details := advertisement, dict) else None,
        }
        for device, advertisement in devices
    ]


def scan_devices_sync(timeout: float = 5.0) -> list[dict[str, Any]]:
    return asyncio.run(scan_devices(timeout))
