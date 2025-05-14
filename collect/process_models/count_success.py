import csv
from pathlib import Path

import matplotlib.pyplot as plt


def count_success() -> None:
    dataset = input("Path to dataset: ").strip()
    data_root = Path(dataset)
    fields = [
        p.name for p in data_root.iterdir()
        if p.is_dir() and (p / "originals").is_dir() and p.name != "kt_source"
    ]
    results = []
    for field in fields:
        bytecode_dir = data_root / field / "bytecode"
        if bytecode_dir.exists():
            repo_dirs = [d for d in bytecode_dir.iterdir() if d.is_dir()]
            count = len(repo_dirs)
        else:
            count = 0
        results.append((field, count))

    results.sort(key=lambda x: x[1], reverse=True)

    counts_csv = data_root / 'bytecode_repo_counts.csv'
    with counts_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['field', 'repo_count'])
        writer.writerows(results)

    print(f"Repository counts saved to {counts_csv}")

    fields_sorted = [x[0] for x in results]
    counts_sorted = [x[1] for x in results]

    plt.figure(figsize=(12, 6))
    plt.bar(fields_sorted, counts_sorted)
    plt.xticks(rotation=45, ha='right')
    plt.title("Compiled Repositories per Model")
    plt.xlabel("Model")
    plt.ylabel("Percentage")
    plt.tight_layout()

    plot_path = data_root / "bytecode_repo_counts.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    count_success()
