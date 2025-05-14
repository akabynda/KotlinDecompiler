import json
from pathlib import Path
from typing import Iterable, Dict, Any

def load_jsonl(jsonl_file: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_file.open('r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def save_fields_by_column(
        jsonl_path: Path,
        output_root: Path
) -> None:
    records = list(load_jsonl(jsonl_path))
    if not records:
        print(f"No records found in {jsonl_path}")
        return

    all_fields = set().union(*(r.keys() for r in records))
    all_fields.discard('classes')
    all_fields.discard('kt_path')

    for field in all_fields:
        (output_root / field).mkdir(parents=True, exist_ok=True)

    for rec in records:
        kt_path = rec.get('kt_path')
        if not kt_path:
            continue
        for field in all_fields:
            if field not in rec:
                continue
            content = rec[field]

            if content in ("", None, [], {}):
                continue

            dest_file = output_root / field / "originals" / kt_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(content, str):
                text = content
            else:
                text = json.dumps(content, ensure_ascii=False)

            dest_file.write_text(text, encoding='utf-8')

    print(f"All fields saved under {output_root}")


def main() -> None:
    jsonl_file = Path(input("Введите путь к объединённому файлу (.jsonl): "))
    output_dir = Path(input("Введите корневую директорию для сохранения полей: "))
    save_fields_by_column(jsonl_file, output_dir)


if __name__ == '__main__':
    main()
