from pathlib import Path

import pandas as pd

from collect.metrics.common import structural, entropy_metrics, collect_tests, \
    build_pairs, parse_detekt, lm_metrics, load_lm

test_root = Path(input("Путь к папке с тестами: ").strip()).expanduser()
report_path = Path(input("Путь к detekt_report.txt: ").strip()).expanduser()
out_csv = Path("full_metrics.csv")

tests = collect_tests(test_root)
pairs = build_pairs(tests)
p_uni, p_bi, p_left = load_lm("kstack-clean+kexercises")
detekt_df = parse_detekt(report_path, test_root)


def build_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for test, cat, dec_code, orig_code in pairs:
        print(test, cat)
        row: dict[str, float] = {"Test": test, "Category": cat}
        row.update(structural(dec_code))
        row.update(entropy_metrics(orig_code, dec_code))
        row.update(lm_metrics(src=dec_code, p_uni=p_uni, p_left=p_left, p_bi=p_bi))
        if cat in detekt_df.index:
            for issue, val in detekt_df.loc[cat].items():
                row[f"detekt_{issue}"] = val
        rows.append(row)
    return rows


df = pd.DataFrame(build_rows()).sort_values(["Test", "Category"]).reset_index(drop=True)
df.to_csv(out_csv, index=False)
print(f"{out_csv} (строк: {len(df)})")
