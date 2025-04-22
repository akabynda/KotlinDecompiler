import csv
from pathlib import Path


def main():
    root = Path(input("Root: ").strip()).expanduser()
    out_tbl = Path("results-ChatGPT.csv")
    out_sum = Path("summary-ChatGPT.txt")
    decomp = ["CFR", "JDGUI", "Fernflower", "Bytecode"]

    tests = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)
    passed = {d: 0 for d in decomp}
    rows = []

    for td in tests:
        row = {"Test": td.name}
        for d in decomp:
            ok = any("ChatGPT" in cls.name for cls in (td / d).glob("*.class")) if (td / d).is_dir() else False
            row[d] = "+" if ok else "-"
            if ok: passed[d] += 1
        rows.append(row)

    with out_tbl.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Test"] + decomp);
        w.writeheader();
        w.writerows(rows)

    total = len(tests)
    with out_sum.open("w", encoding="utf-8") as f:
        for d in decomp:
            pct = passed[d] / total * 100 if total else 0
            f.write(f"{d}: {passed[d]}/{total} — {pct:.2f}%\n")

    print(f"→ {out_tbl}, {out_sum}")


if __name__ == "__main__":
    main()
