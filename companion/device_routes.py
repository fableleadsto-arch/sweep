"""Mount the permissioned Device Host routes into the live companion app."""
from __future__ import annotations
from functools import lru_cache
from fastapi import Depends
from .device_api import create_device_router
from .device_host import DeviceHost,DeviceHostConfig
from .device_settings import DeviceRouteSettings
from .main import app,require_token
@lru_cache(maxsize=1)
def get_device_host():
 s=DeviceRouteSettings.from_env();return DeviceHost(DeviceHostConfig(enabled=s.enabled,simulation_mode=s.simulation_mode,dry_run_default=s.dry_run_default,permissions_file=s.permissions_file,audit_file=s.audit_file,allowed_roots=s.allowed_roots,path_aliases=s.path_aliases,app_allowlist=s.app_allowlist))
def _install():
 prefix="/api/brain/device"
 if any(getattr(route,"path","").startswith(prefix) for route in app.routes):return
 app.include_router(create_device_router(get_device_host),dependencies=[Depends(require_token)])
_install()
