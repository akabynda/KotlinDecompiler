import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_BASE_PREFIX = Path("/Users/akabynda/KotlinDecompiler/src/syntheticExamples")
_DECOMPILERS = {"JDGUI", "CFR", "Fernflower", "Bytecode"}
_CONVERTERS = {"J2K", "ChatGPT"}
_PATTERN = re.compile(r'^(?P<issue>\w+)\s+-.*\s+at\s+(?P<path>/.*?):\d+:\d+')


def main():
    rpt = Path(input("Путь к detekt_report.txt: ").strip()).expanduser()
    out = Path(input("Папка для CSV/PNG [stats_output]: ").strip() or "stats_output")
    out.mkdir(parents=True, exist_ok=True)

    rec = []
    with rpt.open(encoding="utf-8") as f:
        for line in f:
            m = _PATTERN.match(line)
            if not m:
                continue
            issue = m["issue"]
            try:
                rel = Path(m["path"]).relative_to(_BASE_PREFIX)
            except ValueError:
                continue
            parts = rel.parts
            decomp = parts[1] if len(parts) > 1 and parts[1] in _DECOMPILERS else "original"
            conv = "original"
            if decomp != "original" and len(parts) > 2:
                for c in _CONVERTERS:
                    if c in parts[2]:
                        conv = c;
                        break
            if decomp != "original" and conv == "original":
                continue
            rec.append({"issue": issue, "decompiler": decomp, "converter": conv})

    df = pd.DataFrame(rec)
    cnt = (df.groupby(["decompiler", "converter", "issue"])
           .size()
           .reset_index(name="count"))
    cnt["Category"] = cnt.apply(
        lambda r: "Original" if r.decompiler == r.converter == "original"
        else f"{r.decompiler}{r.converter}", axis=1)

    pv = cnt.pivot_table(index="Category", columns="issue", values="count", fill_value=0)
    (out / "detekt_metrics_category_summary.csv").write_text(pv.to_csv())

    for issue in pv.columns:
        plt.figure(figsize=(10, 4))
        pv[issue].plot(kind="bar")
        plt.title(f"{issue} by category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out / f"{issue}_by_category.png")
        plt.close()

    print(f"Готово — результаты в {out}")


if __name__ == "__main__":
    main()
