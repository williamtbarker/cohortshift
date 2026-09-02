"""CohortShift public API."""

from cohortshift.audit import AuditFinding, audit_frame
from cohortshift.config import EvaluationConfig
from cohortshift.evaluate import EvaluationError, EvaluationResult, evaluate_csv, evaluate_frame
from cohortshift.split import TemporalFold, temporal_folds

__all__ = [
    "AuditFinding",
    "EvaluationConfig",
    "EvaluationError",
    "EvaluationResult",
    "TemporalFold",
    "audit_frame",
    "evaluate_csv",
    "evaluate_frame",
    "temporal_folds",
]

__version__ = "0.1.0"
