# Experiment infrastructure: runner, comparison, analysis

from code.experiment.boundary import (
    BoundaryResult,
    ThreePointValidation,
    run_boundary_condition,
    run_three_point_validation,
)
from code.experiment.comparison import (
    get_conditions,
    run_full_comparison,
    save_comparison_results,
)
from code.experiment.convergence import (
    ConvergenceResult,
    detect_convergence,
    run_convergence_analysis,
)
from code.experiment.reproducibility import (
    get_git_hash,
    get_hardware_info,
    get_library_versions,
    set_seeds,
)
from code.experiment.runner import (
    ComparisonResult,
    ConditionResult,
    CouplingConfig,
    ExperimentRunner,
)
from code.experiment.spatial_coherence_test import (
    SpatialCoherenceResult,
    run_spatial_coherence_test,
)
from code.experiment.temporal import (
    TemporalQualityResult,
    run_temporal_quality_tracking,
)

__all__ = [
    "BoundaryResult",
    "ComparisonResult",
    "ConditionResult",
    "ConvergenceResult",
    "CouplingConfig",
    "ExperimentRunner",
    "SpatialCoherenceResult",
    "TemporalQualityResult",
    "ThreePointValidation",
    "detect_convergence",
    "get_conditions",
    "get_git_hash",
    "get_hardware_info",
    "get_library_versions",
    "run_boundary_condition",
    "run_convergence_analysis",
    "run_full_comparison",
    "run_spatial_coherence_test",
    "run_temporal_quality_tracking",
    "run_three_point_validation",
    "save_comparison_results",
    "set_seeds",
]
