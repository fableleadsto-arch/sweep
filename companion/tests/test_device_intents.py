from __future__ import annotations

import unittest

from companion.device_intents import plan_device_actions


class DeviceIntentTests(unittest.TestCase):
    def test_open_application(self):
        plan = plan_device_actions("Open Chrome.")
        self.assertTrue(plan.recognized)
        self.assertEqual([a.action for a in plan.actions], ["open_application"])
        self.assertEqual(plan.actions[0].target, "chrome")

    def test_open_application_and_url(self):
        plan = plan_device_actions("Open Chrome and go to example.com")
        self.assertTrue(plan.recognized)
        self.assertEqual([a.action for a in plan.actions], ["open_application", "open_url"])
        self.assertEqual(plan.actions[1].target, "https://example.com")

    def test_open_alias_folder(self):
        plan = plan_device_actions("Open my Downloads folder")
        self.assertTrue(plan.recognized)
        self.assertEqual(plan.actions[0].action, "open_folder")
        self.assertEqual(plan.actions[0].target, "alias:downloads")

    def test_open_url_direct(self):
        plan = plan_device_actions("Go to https://example.com/path")
        self.assertTrue(plan.recognized)
        self.assertEqual(plan.actions[0].action, "open_url")
        self.assertEqual(plan.actions[0].target, "https://example.com/path")

    def test_search_local_files(self):
        plan = plan_device_actions("search local files for budget in downloads")
        self.assertTrue(plan.recognized)
        self.assertEqual(plan.actions[0].action, "search_local_files")
        self.assertEqual(plan.actions[0].target, "alias:downloads")
        self.assertEqual(plan.actions[0].parameters["query"], "budget")

    def test_create_folder_requires_location(self):
        plan = plan_device_actions("create a folder called Research")
        self.assertTrue(plan.recognized)
        self.assertEqual(plan.actions, [])
        self.assertIn("location", plan.clarification.lower())

    def test_unknown_request_not_recognized(self):
        plan = plan_device_actions("please reflect philosophically on software")
        self.assertFalse(plan.recognized)


if __name__ == "__main__":
    unittest.main()
