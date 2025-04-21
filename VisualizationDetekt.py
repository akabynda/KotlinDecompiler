import re
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Статистика по detekt_report.txt")
    parser.add_argument("report_path", type=Path, help="detekt_report.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("stats_output"))
    args = parser.parse_args()

    report_path = args.report_path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prefix = Path("/Users/akabynda/KotlinDecompiler/src/syntheticExamples")
    pattern = re.compile(r'^(?P<issue>\w+)\s+-.*\s+at\s+(?P<path>/.*?):\d+:\d+')
    known_decompilers = {"JDGUI", "CFR", "Fernflower", "Bytecode"}
    known_converters  = {"J2K", "ChatGPT"}

    records = []
    with report_path.open(encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            issue = m.group("issue")
            full_path = Path(m.group("path"))
            try:
                rel = full_path.relative_to(base_prefix)
            except ValueError:
                continue
            parts = rel.parts

            decompiler = parts[1] if len(parts) >= 2 and parts[1] in known_decompilers else "original"
            converter = "original"
            if decompiler != "original" and len(parts) >= 3:
                for conv in known_converters:
                    if conv in parts[2]:
                        converter = conv
                        break

            if decompiler != "original" and converter == "original":
                continue

            records.append({"issue": issue, "decompiler": decompiler, "converter": converter})

    df = pd.DataFrame(records)
    counts = df.groupby(["decompiler", "converter", "issue"]) \
               .size() \
               .reset_index(name="count")

    counts["Category"] = counts.apply(
        lambda r: "Original" if (r.decompiler=="original" and r.converter=="original")
                  else f"{r.decompiler}{r.converter}",
        axis=1
    )

    pivot = counts.pivot_table(
        index="Category", columns="issue", values="count", fill_value=0
    )

    (output_dir / "issue_counts_by_category.csv").write_text(pivot.to_csv())

    for issue in pivot.columns:
        plt.figure(figsize=(10, 4))
        pivot[issue].plot(kind="bar")
        plt.title(f"Количество «{issue}» по категориям")
        plt.ylabel("Count")
        plt.xlabel("Category")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{issue}_by_category.png")
        plt.close()

    print(f"Готово! CSV и графики — в папке {output_dir}")

if __name__ == "__main__":
    main()
