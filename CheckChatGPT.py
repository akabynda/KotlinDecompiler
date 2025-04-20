import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Проверка прохождения тестов через циклы декомпилятор→конвертер"
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Путь к корневой папке с папками-тестами",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("results-ChatGPT.csv"),
        help="Имя выходного CSV-файла с подробными результатами",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("summary-ChatGPT.txt"),
        help="Имя выходного файла с процентом успеха для каждого декомпилятора",
    )
    args = parser.parse_args()

    decompilers = ["CFR", "JDGUI", "Fernflower", "Bytecode"]

    # Список всех тестовых директорий
    test_dirs = [d for d in args.root.iterdir() if d.is_dir()]
    test_dirs.sort(key=lambda d: d.name)

    results = []
    passed_counts = {dec: 0 for dec in decompilers}
    total = len(test_dirs)

    for td in test_dirs:
        row = {"Test": td.name}
        for dec in decompilers:
            dec_dir = td / dec
            ok = False
            if dec_dir.is_dir():
                # ищем .class‑файл с ChatGPT в имени
                for cls in dec_dir.glob("*.class"):
                    if "ChatGPT" in cls.name:
                        ok = True
                        break
            row[dec] = "+" if ok else "-"
            if ok:
                passed_counts[dec] += 1
        results.append(row)

    # Сохраняем детальную таблицу
    with open(args.output_table, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Test"] + decompilers)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Сохраняем сводный отчёт
    with open(args.output_summary, "w", encoding="utf-8") as f:
        for dec in decompilers:
            count = passed_counts[dec]
            pct = (count / total * 100) if total else 0.0
            f.write(f"{dec}: {count}/{total} тестов (+) — {pct:.2f}%\n")

    print(f"Детали сохранены в {args.output_table}, сводка — в {args.output_summary}")

if __name__ == "__main__":
    main()
