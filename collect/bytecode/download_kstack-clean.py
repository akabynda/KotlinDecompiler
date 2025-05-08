import sys
from pathlib import Path
from typing import Iterable, Dict, Any

from datasets import load_dataset


def get_kotlin_dataset(
        split: str,
        streaming: bool = True
) -> Iterable[Dict[str, Any]]:
    dataset = load_dataset(
        "JetBrains/KStack-clean",
        split=split,
        streaming=streaming
    )
    for example in dataset:
        if example.get("path", "").endswith(".kt"):
            yield example


def save_kotlin_sources(
        dataset: Iterable[Dict[str, Any]],
        originals_root: Path
) -> None:
    for example in dataset:
        owner: str = example["owner"]
        repo: str = example["name"]
        sha: str = example["commit_sha"][:7]
        rel_dir = originals_root / f"{owner}__{repo}__{sha}"
        dst_file = rel_dir / example["path"]
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_text(example["content"], encoding="utf-8")


def main() -> None:
    output_root = Path("./kstack-clean")
    originals_root = output_root / "originals"
    originals_root.mkdir(parents=True, exist_ok=True)

    dataset_split = "train"
    streaming_flag = True

    kotlin_ds = get_kotlin_dataset(dataset_split, streaming_flag)
    save_kotlin_sources(kotlin_ds, originals_root)

    print(f"Saved all .kt files to {originals_root}")


if __name__ == "__main__":
    sys.exit(main())
