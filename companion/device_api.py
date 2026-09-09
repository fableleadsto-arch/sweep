from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .device_host import DeviceActionRequest, DeviceHost, PermissionMode
from .device_intents import plan_device_actions


class DevicePlanRequest(BaseModel):
    message: str = Field(min_length=1)


class DeviceExecuteRequest(BaseModel):
    action: str = Field(min_length=1)
    target: str = ""
    parameters: dict = Field(default_factory=dict)
    dry_run: bool = False


class DeviceApproveRequest(BaseModel):
    approval_id: str = Field(min_length=1)
    decision: PermissionMode


def create_device_router(host_factory: Callable[[], DeviceHost]) -> APIRouter:
    router = APIRouter(prefix="/api/brain/device", tags=["device"])

    def get_host() -> DeviceHost:
        return host_factory()

    @router.get("/status")
    def status(host: DeviceHost = Depends(get_host)) -> dict:
        return host.status()

    @router.get("/capabilities")
    def capabilities(host: DeviceHost = Depends(get_host)) -> dict:
        return {"capabilities": host.capabilities()}

    @router.post("/plan")
    def plan(request: DevicePlanRequest) -> dict:
        plan = plan_device_actions(request.message)
        return {
            "recognized": plan.recognized,
            "actions": [a.__dict__ for a in plan.actions],
            "clarification": plan.clarification,
            "confidence": plan.confidence,
        }

    @router.post("/execute")
    def execute(request: DeviceExecuteRequest, host: DeviceHost = Depends(get_host)) -> dict:
        result = host.execute(DeviceActionRequest(request.action, request.target, dict(request.parameters), request.dry_run))
        return result.to_dict()

    @router.post("/approve")
    def approve(request: DeviceApproveRequest, host: DeviceHost = Depends(get_host)) -> dict:
        result = host.approve(request.approval_id, request.decision)
        return result.to_dict()

    @router.get("/audit")
    def audit(host: DeviceHost = Depends(get_host)) -> dict:
        return {"events": host.audit_history()}

    return router
