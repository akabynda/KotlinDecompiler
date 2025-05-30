import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.main.collect.metrics.metrics_collector import MetricsCollector


class ModelMetricsCalculator:
    """
    Orchestrates the computation of metrics for Kotlin test cases and saves results to a CSV.
    """

    def __init__(self, workers: Optional[int] = None) -> None:
        """
        Initialize the metrics processor.

        Args:
            workers (Optional[int]): Number of parallel worker processes.
                                      Defaults to CPU count minus 6.
        """
        self.collector = MetricsCollector()
        self.workers: int = workers if workers is not None else max(1, cpu_count() - 6)
        self.p_uni, self.p_bi, self.p_left = self.collector.load_lm()

    def compute_row(self, args: Tuple[str, str, str, str, List[str]]) -> List[Any]:
        """
        Compute metrics for a single test case.

        Args:
            args (tuple): (kt_path, field, code, orig_code, metric_list).

        Returns:
            list: List with test path, field, and metric values.
        """
        kt_path, field, code, orig_code, metric_list = args
        s = self.collector.structural(code)
        lm = self.collector.lm_metrics(self.p_uni, self.p_bi, self.p_left, code)
        ent = self.collector.entropy_metrics(orig_code, code)

        row: List[Any] = [kt_path, field]
        for metric in metric_list:
            row.append(s.get(metric, lm.get(metric, ent.get(metric, 0.0))))
        return row

    def prepare_tasks(
            self, jsonl_file: Path, allowed_paths_file: Path, output_csv: Path
    ) -> Tuple[List[Tuple[str, str, str, str, List[str]]], List[str], Set[Tuple[str, str]]]:
        """
        Prepare the list of tasks to process and gather existing entries.

        Args:
            jsonl_file (Path): Path to JSONL file.
            allowed_paths_file (Path): Path to JSON file with allowed paths.
            output_csv (Path): Path to output CSV.

        Returns:
            tuple: (tasks, metric list, existing entries).
        """
        with allowed_paths_file.open("r", encoding="utf-8") as f:
            allowed_paths: Dict[str, Set[str]] = json.load(f)

        with jsonl_file.open("r", encoding="utf-8") as f:
            first_line = f.readline()
        if not first_line:
            raise ValueError("JSONL file is empty.")

        first_record = json.loads(first_line)
        metric_names: Set[str] = set()

        for field, code in first_record.items():
            if field in ("kt_path", "classes") or code is None:
                continue
            orig_code = first_record.get("kt_source")
            s = self.collector.structural(code)
            lm = self.collector.lm_metrics(self.p_uni, self.p_bi, self.p_left, code)
            ent = self.collector.entropy_metrics(orig_code, code) if orig_code else {}
            for name in list(s) + list(lm) + list(ent):
                if not name.startswith("detekt_"):
                    metric_names.add(name)

        metric_list: List[str] = sorted(metric_names)

        existing: Set[Tuple[str, str]] = set()
        if output_csv.exists():
            with output_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        existing.add((row[0], row[1]))

        tasks: List[Tuple[str, str, str, str, List[str]]] = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                kt_path = rec.get("kt_path")
                for field, code in rec.items():
                    if field in ("kt_path", "classes") or code is None:
                        continue
                    if field not in allowed_paths:
                        continue
                    if kt_path not in allowed_paths[field]:
                        continue
                    key = (kt_path, field)
                    if key in existing:
                        continue
                    orig_code = rec.get("kt_source")
                    tasks.append((kt_path, field, code, orig_code, metric_list))

        return tasks, metric_list, existing

    def process_metrics(
            self, jsonl_file: Path, output_csv: Path, allowed_paths_file: Path
    ) -> None:
        """
        Compute metrics in parallel and write them to a CSV file.

        Args:
            jsonl_file (Path): Path to the merged JSONL file.
            output_csv (Path): Path to save the CSV output.
            allowed_paths_file (Path): JSON file with allowed paths.
        """
        tasks, metric_list, existing = self.prepare_tasks(jsonl_file, allowed_paths_file, output_csv)

        if not output_csv.exists():
            with output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["kt_path", "model"] + metric_list)

        print(f"Total tasks to process: {len(tasks)}")

        with ProcessPoolExecutor(max_workers=self.workers) as executor, \
                output_csv.open("a", newline="", encoding="utf-8", buffering=1) as f:
            writer = csv.writer(f)
            futures = {executor.submit(self.compute_row, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                f.flush()

        print(f"Metrics processing complete. Results saved to {output_csv}.")


def main() -> None:
    """
    Entry point for running metrics processing.
    """
    jsonl_path = Path(input("Path to merged .jsonl file: ").strip())
    allowed_paths_path = Path(input("Path to allowed_paths JSON file: ").strip())
    output_csv = Path(input("Path to output CSV: ").strip())

    model_calculator = ModelMetricsCalculator()
    model_calculator.process_metrics(jsonl_path, output_csv, allowed_paths_path)


if __name__ == "__main__":
    main()
