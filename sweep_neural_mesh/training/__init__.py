"""
Sweep Neural Mesh — Training and Improvement System.

§1: No general learning rule. Training is explicit.
§2: Verified examples enter primary dataset.
§3: Mechanics cannot be degraded.
§14: Dataset pipeline (JSONL, CSV, splits, contamination).
§16: Hard negative training.
§17: Adversarial testing.
§20: Ablation testing.
§22: Hardware adaptation.
§23: Training safety & audit.
§28: Failure analysis.
§29: Generalization testing.
§34: Autonomous training mode.
"""

from sweep_neural_mesh.training.domains import DomainScore, ExpertiseTracker
from sweep_neural_mesh.training.task_generator import TaskGenerator, Task
from sweep_neural_mesh.training.solver import Solver, Candidate, SolveResult
from sweep_neural_mesh.training.critique import Critique, CritiqueResult
from sweep_neural_mesh.training.verifier import Verifier
from sweep_neural_mesh.training.experience import Experience, ExperienceMemory
from sweep_neural_mesh.training.curriculum import CurriculumManager, CurriculumState
from sweep_neural_mesh.training.calibration import ConfidenceCalibrator
from sweep_neural_mesh.training.versioning import VersionManager, ModelVersion
from sweep_neural_mesh.training.dashboard import Dashboard
from sweep_neural_mesh.training.trainer import Trainer, TrainingConfig, TrainingResult
from sweep_neural_mesh.training.hard_negatives import HardNegativeGenerator, HardNegative
from sweep_neural_mesh.training.adversarial import AdversarialTestSuite, AdversarialTask, AdversarialResult
from sweep_neural_mesh.training.ablation import AblationStudy, AblationResult, AblationConfig
from sweep_neural_mesh.training.hardware import HardwareDetector, HardwareProfile, InferenceConfig
from sweep_neural_mesh.training.failure_analysis import FailureAnalyzer, FailureRecord, FAILURE_CATEGORIES
from sweep_neural_mesh.training.generalization import GeneralizationTester, GeneralizationTask, GeneralizationResult
from sweep_neural_mesh.training.dataset_pipeline import DatasetPipeline, DatasetEntry, DatasetStats
from sweep_neural_mesh.training.dataset_sources import DatasetManifest, ImportReport, OpenSourceDatasetRegistry
from sweep_neural_mesh.training.safety import SafetyManager, DataLicense, AuditEntry

__all__ = [
    "DomainScore", "ExpertiseTracker",
    "TaskGenerator", "Task",
    "Solver", "Candidate", "SolveResult",
    "Critique", "CritiqueResult",
    "Verifier",
    "Experience", "ExperienceMemory",
    "CurriculumManager", "CurriculumState",
    "ConfidenceCalibrator",
    "VersionManager", "ModelVersion",
    "Dashboard",
    "Trainer", "TrainingConfig", "TrainingResult",
    "HardNegativeGenerator", "HardNegative",
    "AdversarialTestSuite", "AdversarialTask", "AdversarialResult",
    "AblationStudy", "AblationResult", "AblationConfig",
    "HardwareDetector", "HardwareProfile", "InferenceConfig",
    "FailureAnalyzer", "FailureRecord", "FAILURE_CATEGORIES",
    "GeneralizationTester", "GeneralizationTask", "GeneralizationResult",
    "DatasetPipeline", "DatasetEntry", "DatasetStats",
    "DatasetManifest", "ImportReport", "OpenSourceDatasetRegistry",
    "SafetyManager", "DataLicense", "AuditEntry",
]
