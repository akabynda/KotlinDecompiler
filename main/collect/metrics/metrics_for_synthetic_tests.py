from pathlib import Path
from typing import Dict, List

import pandas as pd

from main.collect.metrics.metrics_collector import MetricsCollector


class SyntheticMetricsCalculator:
    """
    Calculates synthetic metrics for Kotlin decompiled code pairs
    and saves the results to a CSV file.
    """

    def __init__(self, test_root: Path, output_csv: Path) -> None:
        """
        Initialize the calculator.

        Args:
            test_root (Path): Path to the directory with test cases.
            output_csv (Path): Path to save the resulting CSV.
        """
        self.metrics_collector = MetricsCollector()
        self.test_root: Path = test_root
        self.output_csv: Path = output_csv
        self.p_uni, self.p_bi, self.p_left = self.metrics_collector.load_lm()

    def build_rows(self) -> List[Dict[str, float]]:
        """
        Build a list of metric rows for each test pair.

        Returns:
            List[Dict[str, float]]: List of metric dictionaries.
        """
        tests = self.metrics_collector.collect_tests(self.test_root)
        pairs = self.metrics_collector.build_pairs(tests)
        rows: List[Dict[str, float]] = []

        for test, category, dec_code, orig_code in pairs:
            print(f"Processing: {test} [{category}]")
            row: Dict[str, float] = {"Test": test, "Category": category}
            row.update(self.metrics_collector.structural(dec_code))
            row.update(self.metrics_collector.entropy_metrics(orig_code, dec_code))
            row.update(
                self.metrics_collector.lm_metrics(src=dec_code, p_uni=self.p_uni, p_left=self.p_left, p_bi=self.p_bi))
            rows.append(row)

        return rows

    def run(self) -> None:
        """
        Run the full metric calculation pipeline and save results.
        """
        rows = self.build_rows()
        df = pd.DataFrame(rows).sort_values(["Test", "Category"]).reset_index(drop=True)
        df.to_csv(self.output_csv, index=False)
        print(f"Metrics saved to {self.output_csv} (rows: {len(df)})")


def main() -> None:
    """
    Entry point for running the synthetic metrics calculation.
    """
    test_root = Path(input("Path to the folder with tests: ").strip()).expanduser()
    output_csv = Path("full_synthetic_metrics.csv")
    calculator = SyntheticMetricsCalculator(test_root, output_csv)
    calculator.run()


if __name__ == "__main__":
    main()
