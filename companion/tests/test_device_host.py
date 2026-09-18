from __future__ import annotations

import json
import unittest
from pathlib import Path

from companion.device_host import DeviceActionRequest, DeviceHost, DeviceHostConfig, PermissionMode


def build_host(tmp_path: Path, *, enabled: bool = True, simulation: bool = True, dry_run_default: bool = False) -> DeviceHost:
    root = tmp_path / "home"
    downloads = root / "Downloads"
    documents = root / "Documents"
    downloads.mkdir(parents=True, exist_ok=True)
    documents.mkdir(parents=True, exist_ok=True)
    return DeviceHost(
        DeviceHostConfig(
            enabled=enabled,
            simulation_mode=simulation,
            dry_run_default=dry_run_default,
            permissions_file=str(tmp_path / "permissions.json"),
            audit_file=str(tmp_path / "audit.jsonl"),
            allowed_roots=[str(root)],
            path_aliases={"home": str(root), "downloads": str(downloads), "documents": str(documents)},
            app_allowlist={
                "chrome": ["google-chrome", "chromium", "chrome"],
                "vscode": ["code"],
                "terminal": ["x-terminal-emulator", "gnome-terminal", "xterm"],
            },
        )
    )


class DeviceHostTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_host_starts_disabled_by_default(self):
        host = build_host(self.root, enabled=False)
        result = host.execute(DeviceActionRequest("get_system_information"))
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.error["code"], "DEVICE_HOST_DISABLED")

    def test_capabilities_report_and_security(self):
        host = build_host(self.root)
        status = host.status()
        self.assertTrue(status["enabled"])
        self.assertIn("open_application", status["capabilities"])
        self.assertFalse(status["security"]["arbitrary_shell_execution"])

    def test_open_application_requires_confirmation_first(self):
        host = build_host(self.root)
        result = host.execute(DeviceActionRequest("open_application", "chrome"))
        self.assertEqual(result.status, "requires_confirmation")
        self.assertTrue(result.approval_request_id)

    def test_allow_once_executes_but_does_not_persist(self):
        host = build_host(self.root)
        pending = host.execute(DeviceActionRequest("open_application", "chrome"))
        approved = host.approve(pending.approval_request_id, PermissionMode.ALLOW_ONCE)
        self.assertTrue(approved.success)
        again = host.execute(DeviceActionRequest("open_application", "chrome"))
        self.assertEqual(again.status, "requires_confirmation")

    def test_always_allow_persists_across_host_restart(self):
        host = build_host(self.root)
        pending = host.execute(DeviceActionRequest("open_url", "https://example.com"))
        approved = host.approve(pending.approval_request_id, PermissionMode.ALWAYS_ALLOW)
        self.assertTrue(approved.success)
        host2 = build_host(self.root)
        direct = host2.execute(DeviceActionRequest("open_url", "https://example.com"))
        self.assertTrue(direct.success)
        self.assertFalse(direct.requires_confirmation)

    def test_permanent_deny_persists(self):
        host = build_host(self.root)
        pending = host.execute(DeviceActionRequest("open_url", "https://example.com"))
        denied = host.approve(pending.approval_request_id, PermissionMode.PERMANENTLY_DENY)
        self.assertEqual(denied.status, "denied")
        host2 = build_host(self.root)
        blocked = host2.execute(DeviceActionRequest("open_url", "https://example.com"))
        self.assertEqual(blocked.status, "denied")

    def test_dry_run_create_folder_makes_no_change(self):
        host = build_host(self.root, dry_run_default=True)
        target = "alias:downloads/Research"
        pending = host.execute(DeviceActionRequest("create_folder", target))
        approved = host.approve(pending.approval_request_id, PermissionMode.ALLOW_ONCE)
        self.assertTrue(approved.success)
        self.assertTrue(approved.dry_run)
        self.assertFalse((self.root / "home" / "Downloads" / "Research").exists())

    def test_real_create_folder_under_allowed_root(self):
        host = build_host(self.root, simulation=False)
        target = "alias:downloads/Research"
        pending = host.execute(DeviceActionRequest("create_folder", target))
        result = host.approve(pending.approval_request_id, PermissionMode.ALLOW_ONCE)
        self.assertTrue(result.success)
        self.assertTrue((self.root / "home" / "Downloads" / "Research").exists())

    def test_path_traversal_is_rejected(self):
        host = build_host(self.root)
        result = host.execute(DeviceActionRequest("open_folder", "alias:downloads/../Secrets"))
        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_REQUEST")
        self.assertIsNone(result.approval_request_id)

    def test_javascript_url_is_rejected(self):
        host = build_host(self.root)
        result = host.execute(DeviceActionRequest("open_url", "javascript:alert(1)"))
        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_REQUEST")

    def test_command_injection_app_name_is_rejected(self):
        host = build_host(self.root)
        result = host.execute(DeviceActionRequest("open_application", "chrome; rm -rf /"))
        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_REQUEST")

    def test_list_directory_and_search(self):
        host = build_host(self.root)
        d = self.root / "home" / "Downloads"
        (d / "budget-2026.txt").write_text("x", encoding="utf-8")
        (d / "notes.txt").write_text("y", encoding="utf-8")
        listed = host.execute(DeviceActionRequest("list_directory", "alias:downloads"))
        self.assertTrue(listed.success)
        self.assertTrue({e["name"] for e in listed.data["entries"]} >= {"budget-2026.txt", "notes.txt"})
        pending = host.execute(DeviceActionRequest("search_local_files", "alias:downloads", {"query": "budget"}))
        found = host.approve(pending.approval_request_id, PermissionMode.ALLOW_ONCE)
        self.assertTrue(found.success)
        self.assertEqual(found.data["matches"][0]["name"], "budget-2026.txt")

    def test_system_information_is_available_without_confirmation(self):
        host = build_host(self.root)
        result = host.execute(DeviceActionRequest("get_system_information"))
        self.assertTrue(result.success)
        self.assertIn("system", result.data)

    def test_audit_log_records_without_file_contents(self):
        host = build_host(self.root)
        host.execute(DeviceActionRequest("get_system_information"))
        entries = host.audit_history()
        self.assertTrue(entries)
        self.assertEqual(entries[0]["action"], "get_system_information")
        self.assertNotIn("contents", json.dumps(entries[0]).lower())

    def test_expired_approval_request_is_rejected(self):
        host = build_host(self.root)
        host.config.approval_ttl_seconds = -1
        pending = host.execute(DeviceActionRequest("open_url", "https://example.com"))
        result = host.approve(pending.approval_request_id, PermissionMode.ALLOW_ONCE)
        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "APPROVAL_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
