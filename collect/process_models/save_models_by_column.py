import json
from pathlib import Path
from typing import Any, Dict, Iterable


class JSONLProcessor:
    """
    Processes a JSONL file to extract and save content grouped by model.
    """

    @staticmethod
    def load_jsonl(jsonl_file: Path) -> Iterable[Dict[str, Any]]:
        """
        Loads a JSONL file line by line.

        Args:
            jsonl_file (Path): Path to the JSONL file.

        Yields:
            dict: JSON-decoded dictionary for each line.
        """
        with jsonl_file.open('r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    def save_models_by_column(self, jsonl_path: Path, output_root: Path) -> None:
        """
        Processes the JSONL file and saves data for each model into separate folders.

        Args:
            jsonl_path (Path): Path to the JSONL file.
            output_root (Path): Root directory to store separated model files.
        """
        records: list[Dict[str, Any]] = list(self.load_jsonl(jsonl_path))
        if not records:
            print(f"No records found in {jsonl_path}")
            return

        # Identify model names
        all_models: set[str] = set().union(*(r.keys() for r in records))
        all_models.discard('classes')
        all_models.discard('kt_path')

        # Create directories for each model
        for model in all_models:
            (output_root / model).mkdir(parents=True, exist_ok=True)

        # Save data for each model
        for rec in records:
            kt_path: str | None = rec.get('kt_path')
            if not kt_path:
                continue
            for model in all_models:
                if model not in rec:
                    continue
                content: Any = rec[model]
                if content in ("", None, [], {}):
                    continue

                dest_file: Path = output_root / model / "originals" / kt_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                text: str
                if isinstance(content, str):
                    text = content
                else:
                    text = json.dumps(content, ensure_ascii=False)

                dest_file.write_text(text, encoding='utf-8')

        print(f"All models saved under {output_root}")


def main() -> None:
    """
    Entry point for the JSONL processor.
    """
    jsonl_file = Path(input("Enter the path to the merged JSONL file: "))
    output_dir = Path(input("Enter the root directory for saving models: "))

    processor = JSONLProcessor()
    processor.save_models_by_column(jsonl_file, output_dir)


if __name__ == '__main__':
    main()
