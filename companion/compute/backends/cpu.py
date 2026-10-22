"""CPU backend — the core numerical executor.

Runs capability tasks through the light NumPy/SciPy/sklearn tool chain and
executes smoke tests natively. NumPy loads only when executing numerical work.
"""

from __future__ import annotations

from ..base import BackendKind, ComputeBackend, ComputeJobResult, ComputeTask, ComputeTaskKind
from ..capability_detector import detect


class CPUBackend(ComputeBackend):
    id = "cpu"
    label = "CPU (core)"
    kind = BackendKind.CPU
    required_libraries: tuple[str, ...] = ("numpy",)
    preferred_for = ("math", "data-analysis", "plot", "ml", "symbolic", "simulation", "graph", "nlp")
    devices_import_free = True
    runner_capabilities = {"math", "simulation", "data-analysis", "plot", "ml", "symbolic", "graph", "vision", "nlp"}

    @property
    def available(self) -> bool:
        return not self.missing_libraries

    def supports(self, task: ComputeTask) -> bool:
        if task.kind == ComputeTaskKind.CAPABILITY:
            return task.capability in self.runner_capabilities
        return task.kind == ComputeTaskKind.SMOKE

    def devices(self):
        from ..base import DeviceInfo

        env = detect()
        return [
            DeviceInfo(
                name="CPU",
                kind="cpu",
                index=0,
                memory_total_mb=env.memory.total_mb,
                memory_free_mb=env.memory.available_mb,
                arch=env.cpu.arch,
            )
        ]

    def run_smoke(self, task: ComputeTask) -> ComputeJobResult:
        import numpy as np

        a = np.arange(12.0).reshape(3, 4)
        b = (a @ a.T).sum().item()
        return ComputeJobResult(
            ok=True,
            backend=self.id,
            device="cpu",
            result={"matrix_sum": round(float(b), 4)},
            summary="CPU smoke test passed (NumPy matmul).",
            libraries_used=["numpy"],
        )

    def run_capability(self, task: ComputeTask) -> ComputeJobResult:
        """Route light computational tasks to the matching tool runner."""
        from ...tools import data, graph, ml, nlp, numeric, symbolic, vision

        runners = {
            "math": numeric.run_numeric,
            "simulation": numeric.run_simulation,
            "data-analysis": data.run_data_analysis,
            "plot": data.run_plot,
            "ml": ml.run_ml,
            "symbolic": symbolic.run_symbolic,
            "graph": graph.run_graph,
            "vision": vision.run_vision,
            "nlp": nlp.run_nlp,
        }
        runner = runners.get(task.capability)
        if runner is None:
            raise NotImplementedError(f"cpu backend has no runner for capability '{task.capability}'")
        outcome = runner(task.payload)
        result = outcome.get("result")
        if result is None or result is False:
            return ComputeJobResult(
                ok=False,
                backend=self.id,
                device="cpu",
                error=outcome.get("summary", "No result"),
                summary=str(outcome.get("summary", "No result")),
            )
        return ComputeJobResult(
            ok=True,
            backend=self.id,
            device="cpu",
            result=result,
            summary=outcome.get("summary", "Done."),
            libraries_used=outcome.get("libraries_used", []),
        )
