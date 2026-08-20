"""Public implementation of the COLREGs structured-reasoning framework."""

from .benchmark import BenchmarkRelease

from .contracts import LoadedContract, load_contract
from .completion import CompletionValidator
from .datasets import build_all_training_adapters, observable_scene_spec
from .model_binding import build_model_manifest, verify_model_manifest
from .runs import RunDirectory
from .training import (
    build_training_comparability,
    load_training_dataset,
    validate_training_smoke_suite,
)

__all__ = [
    "BenchmarkRelease",
    "CompletionValidator",
    "LoadedContract",
    "RunDirectory",
    "build_all_training_adapters",
    "build_model_manifest",
    "build_training_comparability",
    "load_training_dataset",
    "load_contract",
    "observable_scene_spec",
    "validate_training_smoke_suite",
    "verify_model_manifest",
]
