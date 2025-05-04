import csv
from pathlib import Path
from typing import Dict, List, Any


def append_metrics(csv_path: Path, metrics: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf8') as f:
            rows = list(csv.reader(f))
        header = rows[0] if rows else []
        data = rows[1] if len(rows) > 1 else []
    else:
        header, data = [], []

    for k, v in metrics.items():
        header.append(k)
        data.append(str(v))

    with open(csv_path, 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(data)


def write_summary(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    header = ['File']
    for row in rows:
        for key in row:
            if key != 'File' and key not in header:
                header.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([row.get(col, '') for col in header])


def write_category_summary(csv_path: Path,
                           data: Dict[str, Dict[str, List[float]]]) -> None:
    if not data:
        return

    header = ["Category"]
    for cat_metrics in data.values():
        for metric in cat_metrics:
            if metric not in header:
                header.append(metric)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for category, m_dict in data.items():
            row = [category]
            for metric in header[1:]:
                values = m_dict.get(metric, [])
                avg = sum(values) / len(values) if values else ""
                row.append(avg)
            w.writerow(row)
