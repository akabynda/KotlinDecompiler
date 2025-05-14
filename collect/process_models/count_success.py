import csv
from pathlib import Path

def count_success() -> None:
    dataset = input("Path to dataset: ").strip()
    data_root = Path(dataset)
    fields = [p.name for p in data_root.iterdir() if p.is_dir() and (p / "originals").is_dir()]

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

if __name__ == "__main__":
    count_success()
