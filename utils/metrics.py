"""
utils/metrics.py
=================
Pipeline timing metrics and benchmark reporting.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    detection_ms: float = 0.0
    recognition_ms: float = 0.0
    solving_ms: float = 0.0
    interaction_ms: float = 0.0
    success: bool = False
    error: Optional[str] = None

    @property
    def total_ms(self) -> float:
        return self.detection_ms + self.recognition_ms + self.solving_ms + self.interaction_ms

    def to_dict(self) -> dict:
        return {
            "detection_ms": round(self.detection_ms, 1),
            "recognition_ms": round(self.recognition_ms, 1),
            "solving_ms": round(self.solving_ms, 1),
            "interaction_ms": round(self.interaction_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "success": self.success,
            "error": self.error,
        }


class BenchmarkReport:
    """
    Formats and saves comparison results across recognition methods.

    Example output:
    ┌─────────────┬────────────────┬──────────────┬────────────┬─────────┐
    │ Method      │ Recognition ms │ Total ms     │ Accuracy % │ Success │
    ├─────────────┼────────────────┼──────────────┼────────────┼─────────┤
    │ easyocr     │ 312.4          │ 485.2        │ 98.8       │ ✓       │
    │ tesseract   │  98.1          │ 271.0        │ 94.2       │ ✓       │
    │ template    │  24.3          │ 197.1        │ 89.5       │ ✓       │
    │ cnn         │  18.7          │ 191.5        │ 99.1       │ ✓       │
    └─────────────┴────────────────┴──────────────┴────────────┴─────────┘
    """

    HEADERS = ["method", "detection_ms", "recognition_ms", "solving_ms",
               "interaction_ms", "total_ms", "success"]

    def __init__(self, results: dict):
        self.results = results

    def print_table(self):
        print("\n" + "=" * 72)
        print("BENCHMARK RESULTS — Digit Recognition Method Comparison")
        print("=" * 72)
        fmt = "{:<14} {:>16} {:>14} {:>12} {:>10}"
        print(fmt.format("Method", "Recognition (ms)", "Total (ms)", "Speedup×", "Status"))
        print("-" * 72)

        # Find baseline (easyocr)
        baseline_rec = self.results.get("easyocr", {}).get("recognition_ms", 1)

        for method, data in self.results.items():
            if not data.get("success"):
                print(fmt.format(method, "N/A", "N/A", "N/A", "✗ FAILED"))
                continue
            rec = data["recognition_ms"]
            tot = data["total_ms"]
            speedup = baseline_rec / rec if rec > 0 else 0
            status = "✓"
            print(fmt.format(
                method,
                f"{rec:.1f}",
                f"{tot:.1f}",
                f"{speedup:.1f}×",
                status
            ))

        print("=" * 72)
        self._print_summary()

    def _print_summary(self):
        valid = {k: v for k, v in self.results.items() if v.get("success")}
        if not valid:
            return
        fastest = min(valid, key=lambda k: valid[k]["recognition_ms"])
        most_stable = min(valid, key=lambda k: valid[k]["total_ms"])
        print(f"\n→ Fastest recognition: {fastest} "
              f"({valid[fastest]['recognition_ms']:.1f} ms)")
        print(f"→ Lowest total latency: {most_stable} "
              f"({valid[most_stable]['total_ms']:.1f} ms)\n")

    def save_csv(self, output_path: str = "benchmark_results.csv"):
        path = Path(output_path)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writeheader()
            for method, data in self.results.items():
                row = {"method": method}
                row.update({k: data.get(k, "") for k in self.HEADERS[1:]})
                writer.writerow(row)
        logger.info(f"Benchmark results saved to {path}")
