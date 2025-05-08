import sys
from pathlib import Path
from typing import Iterable, Dict, Any

from datasets import load_dataset


def get_exercises_dataset(
        split: str = "train",
        streaming: bool = True
) -> Iterable[Dict[str, Any]]:
    dataset = load_dataset(
        "JetBrains/KExercises",
        split=split,
        streaming=streaming
    )
    for example in dataset:
        yield example


def save_exercises(
        dataset: Iterable[Dict[str, Any]],
        originals_root: Path
) -> None:
    originals_root.mkdir(parents=True, exist_ok=True)
    for idx, example in enumerate(dataset):
        name = f"{idx}"
        folder = originals_root / name
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"solution_{idx}.kt"
        file_path.write_text(example.get("problem", "").strip() + "\n" + example.get("solution", "").strip(),
                             encoding="utf-8")


def main() -> None:
    output_root = Path("./kexercises/originals")
    split = "train"
    streaming_flag = True

    ds = get_exercises_dataset(split, streaming_flag)
    save_exercises(ds, output_root)

    print("Saved", output_root)


if __name__ == "__main__":
    sys.exit(main())
