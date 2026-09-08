"""Regression tests of complete production API/cortex modules, not model accuracy.

Run: python -m unittest discover -s sweep_neural_mesh/tests -p test_api_failure_contract.py -v

Modules are loaded from adjacent source files without the package __init__ (which
imports the wider mesh). This exercises actual module code but NOT package boot,
heavy model loading, the full application, or OS actions. Dependency doubles are
explicit fixtures. Test loading does not make a partial snapshot a full checkout.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class FailureContractTests(unittest.TestCase):
    def setUp(self):
        self.original_path = list(sys.path)
        self.package = "_sweep_failure_contract_test"
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_path))
        package = types.ModuleType(self.package)
        package.__path__ = [str(ROOT)]
        sys.modules[self.package] = package
        self.cortex = self.load(self.package + ".cortex_integration", "cortex_integration.py")
        # Compatibility alias also lets the original pre-patch API run as written.
        sys.modules["cortex_integration"] = self.cortex
        self.api_module = self.load(self.package + ".sweep_api", "sweep_api.py")
        self.api = self.api_module.SweepAPI()

    @staticmethod
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def fixture_pipeline(self, initialized=True):
        pipeline = Mock()
        pipeline.initialize.return_value = initialized
        pipeline.infer.return_value = self.cortex.InferenceResult(
            answer="FIXTURE ONLY", confidence=0.0, method="fixture", components_used=[]
        )
        pipeline._trained_model = None
        pipeline._seq2seq_model = None
        pipeline._logic_engine = None
        self.cortex.get_pipeline = Mock(return_value=pipeline)
        return pipeline

    def test_failed_initialization_does_not_mark_ready(self):
        self.fixture_pipeline(False)
        self.api._ensure_init()
        self.assertFalse(self.api._initialized)
        self.assertIsNone(self.api._pipeline)

    def test_failed_initialization_never_invokes_inference(self):
        pipeline = self.fixture_pipeline(False)
        result = self.api.query("fixture question")
        pipeline.infer.assert_not_called()
        self.assertEqual(result.method, "error")

    def test_truthy_nonboolean_initialization_is_not_success(self):
        for value in (1, "ready", {"ready": True}, None):
            with self.subTest(value=value):
                self.api = self.api_module.SweepAPI()
                self.fixture_pipeline(value)
                self.api._ensure_init()
                self.assertFalse(self.api._initialized)
                self.assertIsNone(self.api._pipeline)

    def test_initialization_exception_clears_pipeline(self):
        pipeline = self.fixture_pipeline()
        pipeline.initialize.side_effect = RuntimeError("fixture diagnostic")
        self.api._ensure_init()
        self.assertIsNone(self.api._pipeline)
        self.assertFalse(self.api._initialized)

    def test_initialization_logs_do_not_expose_exception_message(self):
        pipeline = self.fixture_pipeline()
        pipeline.initialize.side_effect = RuntimeError("SENSITIVE_FIXTURE_NOT_A_SECRET")
        with self.assertLogs("sweep.api", level="ERROR") as captured:
            self.api._ensure_init()
        self.assertNotIn("SENSITIVE_FIXTURE_NOT_A_SECRET", "\n".join(captured.output))

    def test_success_preserves_query_arguments_and_zero_score(self):
        pipeline = self.fixture_pipeline()
        result = self.api.query("q", ["e"], "context")
        pipeline.infer.assert_called_once_with("q", evidence=["e"], context="context")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.evidence_used, ["e"])

    def test_success_initializes_only_once(self):
        pipeline = self.fixture_pipeline()
        self.api.query("one")
        self.api.query("two")
        pipeline.initialize.assert_called_once()

    def test_failed_initialization_can_retry(self):
        pipeline = self.fixture_pipeline()
        pipeline.initialize.side_effect = [False, True]
        self.api._ensure_init()
        self.assertFalse(self.api._initialized)
        self.api._ensure_init()
        self.assertTrue(self.api._initialized)
        self.assertEqual(pipeline.initialize.call_count, 2)

    def test_package_import_ignores_unrelated_top_level_module(self):
        pipeline = self.fixture_pipeline()
        shadow = types.ModuleType("cortex_integration")
        shadow.get_pipeline = Mock(side_effect=RuntimeError("unrelated module"))
        sys.modules["cortex_integration"] = shadow
        self.api._ensure_init()
        self.assertIs(self.api._pipeline, pipeline)
        shadow.get_pipeline.assert_not_called()

    def test_initialization_does_not_modify_sys_path(self):
        self.fixture_pipeline()
        before = list(sys.path)
        self.api._ensure_init()
        self.assertEqual(before, sys.path)

    def test_standalone_module_import_remains_supported(self):
        pipeline = self.fixture_pipeline()
        module = self.load("_standalone_sweep_api_test", "sweep_api.py")
        api = module.SweepAPI()
        api._ensure_init()
        self.assertIs(api._pipeline, pipeline)

    def test_failed_status_does_not_claim_models_available(self):
        self.fixture_pipeline(False)
        self.assertEqual(self.api.status(), {
            "initialized": False, "pipeline": False, "trained_model": False,
            "seq2seq": False, "logic_engines": False,
        })

    def no_provider_pipeline(self):
        pipeline = self.cortex.SweepInferencePipeline()
        # Deliberately skip heavyweight initialization. All real provider fields
        # remain absent. This is a no-provider control-flow test, not inference.
        pipeline._initialized = True
        return pipeline

    def test_real_cortex_without_providers_does_not_echo_question(self):
        result = self.no_provider_pipeline().infer("Open Chrome.")
        self.assertNotEqual(result.answer, "Open Chrome.")
        self.assertEqual(result.method, "unavailable")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.components_used, [])

    def test_actual_api_and_cortex_propagate_unavailable(self):
        pipeline = self.no_provider_pipeline()
        self.cortex.get_pipeline = lambda: pipeline
        result = self.api.query("Open Chrome.")
        self.assertEqual(result.method, "unavailable")
        self.assertEqual(result.confidence, 0.0)
        self.assertNotEqual(result.answer, result.query)

    def test_existing_logic_result_priority_is_preserved(self):
        pipeline = self.no_provider_pipeline()
        logic = types.SimpleNamespace(conclusion="supported", reasoning="fixture")
        pipeline._logic_engine = Mock()
        pipeline._logic_engine.analyze.return_value = logic
        logic.confidence = 0.4
        pipeline._rag = Mock()
        result = pipeline.infer("fixture", ["fixture evidence"])
        self.assertEqual(result.method, "logic_engine")
        self.assertEqual(result.confidence, 0.4)
        pipeline._rag.query.assert_not_called()

    def test_existing_rag_result_path_is_preserved(self):
        pipeline = self.no_provider_pipeline()
        pipeline._rag = Mock()
        pipeline._rag.query.return_value = types.SimpleNamespace(
            answer="RAG FIXTURE", sources=["fixture"], latency_ms=1, confidence=0.2
        )
        result = pipeline.infer("fixture")
        self.assertEqual(result.answer, "RAG FIXTURE")
        self.assertEqual(result.method, "rag")
        self.assertEqual(result.confidence, 0.2)


if __name__ == "__main__":
    unittest.main()
