import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

from collect.metrics.common import structural, lm_metrics, load_lm


def compute_row(args):
    kt_path, field, code, metric_list = args
    s = structural(str(code))
    lm = lm_metrics(src=str(code), p_uni=P_UNI, p_bi=P_BI, p_left=P_LEFT)
    row = [kt_path, field]
    for m in metric_list:
        row.append(s.get(m, lm.get(m, 0.0)))
    return row


def init_worker(p_uni, p_bi, p_left):
    global P_UNI, P_BI, P_LEFT
    P_UNI, P_BI, P_LEFT = p_uni, p_bi, p_left


def metrics_for_models(jsonl_file: Path, output_csv: Path, workers: int = None) -> None:
    p_uni, p_bi, p_left = load_lm()

    with jsonl_file.open('r', encoding='utf-8') as infile:
        first_line = infile.readline()
    if not first_line:
        print("Empty JSONL file.")
        return

    first = json.loads(first_line)
    metric_names = set()
    for field, code in first.items():
        if field in ('kt_path', 'classes') or code is None:
            continue
        s = structural(str(code))
        lm = lm_metrics(src=str(code), p_uni=p_uni, p_bi=p_bi, p_left=p_left)
        for name in list(s) + list(lm):
            if name.startswith('detekt_'):
                continue
            metric_names.add(name)

    metric_list = sorted(metric_names)

    existing = set()
    if output_csv.exists():
        with output_csv.open('r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if len(r) >= 2:
                    existing.add((r[0], r[1]))

    if not output_csv.exists():
        with output_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['kt_path', 'model'] + metric_list)

    tasks = []
    with jsonl_file.open('r', encoding='utf-8') as infile:
        for line in infile:
            rec = json.loads(line)
            kt_path = rec.get('kt_path')
            for field, code in rec.items():
                if field in ('kt_path', 'classes') or code is None:
                    continue
                key = (kt_path, field)
                if key in existing:
                    continue
                tasks.append((kt_path, field, code, metric_list))

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(p_uni, p_bi, p_left)) as executor, \
            output_csv.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        futures = {executor.submit(compute_row, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            writer.writerow(row)


if __name__ == "__main__":
    jsonl_path = Path(input("Path to merged .jsonl file: ").strip())
    out_csv = Path("./KExercises_metrics_for_models.csv")
    metrics_for_models(jsonl_path, out_csv, workers=cpu_count() - 1)
    print(f"Metrics streaming complete. Results in {out_csv}")
