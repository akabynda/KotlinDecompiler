import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def count_success() -> None:
    dataset = input("Path to dataset: ").strip()
    data_root = Path(dataset)

    fields = [
        p.name for p in data_root.iterdir()
        if p.is_dir() and (p / "originals").is_dir()
    ]

    results = []
    compiled_info: dict[str, list[str]] = {}

    for field in fields:
        orig_root = data_root / field / "originals"
        bytecode_root = data_root / field / "bytecode"

        if bytecode_root.exists():
            repo_dirs = [d for d in bytecode_root.iterdir() if d.is_dir()]
            count = len(repo_dirs)

            paths: list[str] = []
            for repo in repo_dirs:
                orig_repo_dir = orig_root / repo.name
                if orig_repo_dir.exists():
                    for kt_file in orig_repo_dir.rglob("*.kt"):
                        rel = kt_file.relative_to(orig_root)
                        paths.append(rel.as_posix())
            compiled_info[field] = paths
        else:
            count = 0
            compiled_info[field] = []

        results.append((field, count))

    results.sort(key=lambda x: x[1], reverse=True)

    counts_csv = data_root / "bytecode_repo_counts.csv"
    with counts_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "repo_count"])
        writer.writerows(results)
    print(f"Repository counts saved to {counts_csv}")

    compiled_json = data_root / "compiled_repos.json"
    with compiled_json.open("w", encoding="utf-8") as file:
        json.dump(compiled_info, file, ensure_ascii=False, indent=2)
    print(f"Compiled kt_path lists saved to {compiled_json}")

    fields_sorted = [x[0] for x in results]
    counts_sorted = [x[1] for x in results]

    plt.figure(figsize=(12, 6))
    plt.bar(fields_sorted, counts_sorted)
    plt.xticks(rotation=45, ha="right")
    plt.title("Compiled Repositories per Model")
    plt.xlabel("Model")
    plt.ylabel("Percentage")
    plt.tight_layout()

    plot_path = data_root / "bytecode_repo_counts.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    count_success()
