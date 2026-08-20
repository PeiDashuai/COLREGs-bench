from pathlib import Path

from colregs_framework import BenchmarkRelease


benchmark = BenchmarkRelease(Path("../COLREGs-Bench-final"))
print(benchmark.validate(check_images=False))
print(benchmark.joined("test", "expert_corrected_core3")[0]["sample_id"])
