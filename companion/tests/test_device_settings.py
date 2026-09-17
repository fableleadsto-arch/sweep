import os,unittest
from unittest.mock import patch
from companion.device_settings import DeviceRouteSettings
class DeviceSettingsTests(unittest.TestCase):
 def test_secure_defaults(self):
  with patch.dict(os.environ,{},clear=True):
   s=DeviceRouteSettings.from_env();self.assertFalse(s.enabled);self.assertFalse(s.simulation_mode);self.assertTrue(s.dry_run_default);self.assertEqual(s.allowed_roots,[])
 def test_explicit_json_configuration(self):
  env={"COMPANION_DEVICE_HOST_ENABLED":"true","COMPANION_DEVICE_HOST_SIMULATION":"1","COMPANION_DEVICE_HOST_DRY_RUN":"false","COMPANION_DEVICE_HOST_ALLOWED_ROOTS":"[\"/tmp\"]","COMPANION_DEVICE_HOST_APP_ALLOWLIST":"{\"chrome\":[\"chromium\"]}"}
  with patch.dict(os.environ,env,clear=True):
   s=DeviceRouteSettings.from_env();self.assertTrue(s.enabled);self.assertTrue(s.simulation_mode);self.assertFalse(s.dry_run_default);self.assertEqual(s.app_allowlist["chrome"],["chromium"])
 def test_invalid_json_fails_closed(self):
  with patch.dict(os.environ,{"COMPANION_DEVICE_HOST_ALLOWED_ROOTS":"{}"},clear=True):
   with self.assertRaises(ValueError):DeviceRouteSettings.from_env()
