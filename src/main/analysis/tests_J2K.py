import csv
from pathlib import Path
from typing import List, Dict


def main() -> None:
    """
    Analyze decompiler outputs in test directories and write results
    to a CSV table and a summary text file.
    """
    root = Path(input("Root: ").strip()).expanduser()
    out_table = Path("results-J2K-raw.csv")
    out_summary = Path("summary-J2K-raw.txt")
    decompilers: List[str] = ["CFR", "JDGUI", "Fernflower"]

    tests = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)
    passed: Dict[str, int] = {d: 0 for d in decompilers}
    rows: List[Dict[str, str]] = []

    for test_dir in tests:
        row: Dict[str, str] = {"Test": test_dir.name}
        for decompiler in decompilers:
            decompiler_dir = test_dir / decompiler
            has_j2k_class = (
                any("J2K" in cls.name for cls in decompiler_dir.glob("*.class"))
                if decompiler_dir.is_dir()
                else False
            )
            row[decompiler] = "+" if has_j2k_class else "-"
            if has_j2k_class:
                passed[decompiler] += 1
        rows.append(row)

    with out_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Test"] + decompilers)
        writer.writeheader()
        writer.writerows(rows)

    total_tests = len(tests)
    with out_summary.open("w", encoding="utf-8") as f:
        for decompiler in decompilers:
            pct = (passed[decompiler] / total_tests * 100) if total_tests else 0
            f.write(f"{decompiler}: {passed[decompiler]}/{total_tests} — {pct:.2f}%\n")

    print(f"→ {out_table}, {out_summary}")


if __name__ == "__main__":
    main()
