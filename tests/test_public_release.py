from __future__ import annotations

import unittest
from pathlib import Path

from colregs_framework.benchmark import BenchmarkRelease
from colregs_framework.scoring import score_predictions
from colregs_framework.semantic_projection import (
    assert_expert_review_complete,
    load_mapping,
    load_ontology,
)
from colregs_framework.symbolic_engine import run_symbolic_sample


UPLOAD_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = UPLOAD_ROOT / "COLREGs-Bench-final"


class PublicReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release = BenchmarkRelease(BENCHMARK_ROOT)

    def test_release_counts_and_images(self) -> None:
        report = self.release.validate(check_images=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["total_samples"], 1700)
        self.assertEqual(report["referenced_images"], 3400)

    def test_perfect_core_three_predictions_score_one(self) -> None:
        gold = self.release.references("test", "expert_corrected_core3")[:8]
        predictions = [
            {"sample_id": row["sample_id"], "prediction": row["labels"]}
            for row in gold
        ]
        bundle = score_predictions(
            gold,
            predictions,
            run_id="unit_test",
            output_scope="core_three_v1",
            bootstrap_replicates=100,
        )
        self.assertEqual(bundle.metrics["primary_mean_field_f1"], 1.0)

    def test_common_semantic_mappings_are_final(self) -> None:
        root = BENCHMARK_ROOT / "evaluation" / "common_semantic"
        ontology = load_ontology(root / "ontology.json")
        for name in ("benchmark_mapping.json", "symbolic_engine_mapping.json"):
            mapping = load_mapping(root / name, ontology)
            assert_expert_review_complete(mapping)

    def test_symbolic_engine_smoke(self) -> None:
        row = self.release.inputs("test")[0]
        rules = BENCHMARK_ROOT / "rules"
        result = run_symbolic_sample(
            row,
            rules_path=rules / "colregs_atomic.yaml",
            params_path=rules / "operational_params.yaml",
            priorities_path=rules / "priorities.yaml",
        )
        self.assertEqual(result["sample_id"], row["sample_id"])
        self.assertEqual(result["input_boundary"], "inputs_only_v1")


if __name__ == "__main__":
    unittest.main()
