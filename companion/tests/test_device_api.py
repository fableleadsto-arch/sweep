from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from test_device_host import build_host


class _StubRouter:
    def __init__(self, prefix: str = "", tags=None):
        self.prefix = prefix
        self.tags = tags or []
        self.routes = {}

    def _register(self, method: str, path: str):
        def decorator(fn):
            self.routes[(method, self.prefix + path)] = fn
            return fn
        return decorator

    def get(self, path: str):
        return self._register("GET", path)

    def post(self, path: str):
        return self._register("POST", path)


def _stub_depends(fn):
    return fn


class DeviceApiTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.host = build_host(self.root)
        fake_fastapi = types.SimpleNamespace(APIRouter=_StubRouter, Depends=_stub_depends)
        with patch.dict(sys.modules, {"fastapi": fake_fastapi}):
            from companion.device_api import (
                DeviceApproveRequest,
                DeviceExecuteRequest,
                DevicePlanRequest,
                create_device_router,
            )
        self.DeviceApproveRequest = DeviceApproveRequest
        self.DeviceExecuteRequest = DeviceExecuteRequest
        self.DevicePlanRequest = DevicePlanRequest
        self.router = create_device_router(lambda: self.host)

    def _route(self, method: str, path: str):
        return self.router.routes[(method, path)]

    def test_status_and_capabilities(self):
        status = self._route("GET", "/api/brain/device/status")(self.host)
        self.assertTrue(status["enabled"])
        caps = self._route("GET", "/api/brain/device/capabilities")(self.host)
        self.assertIn("open_url", caps["capabilities"])

    def test_plan_endpoint(self):
        body = self._route("POST", "/api/brain/device/plan")(self.DevicePlanRequest(message="Open Chrome and go to example.com"))
        self.assertTrue(body["recognized"])
        self.assertEqual([a["action"] for a in body["actions"]], ["open_application", "open_url"])

    def test_execute_then_approve(self):
        pending = self._route("POST", "/api/brain/device/execute")(
            self.DeviceExecuteRequest(action="open_url", target="https://example.com"), self.host
        )
        approval_id = pending["approval_request_id"]
        approved = self._route("POST", "/api/brain/device/approve")(
            self.DeviceApproveRequest(approval_id=approval_id, decision="allow_once"), self.host
        )
        self.assertTrue(approved["success"])

    def test_execute_rejects_path_traversal_before_approval(self):
        resp = self._route("POST", "/api/brain/device/execute")(
            self.DeviceExecuteRequest(action="open_folder", target="alias:downloads/../Secrets"), self.host
        )
        self.assertFalse(resp["success"])
        self.assertEqual(resp["error"]["code"], "INVALID_REQUEST")
        self.assertIsNone(resp["approval_request_id"])

    def test_audit_endpoint(self):
        self._route("POST", "/api/brain/device/execute")(
            self.DeviceExecuteRequest(action="get_system_information"), self.host
        )
        resp = self._route("GET", "/api/brain/device/audit")(self.host)
        self.assertTrue(resp["events"])


if __name__ == "__main__":
    unittest.main()
