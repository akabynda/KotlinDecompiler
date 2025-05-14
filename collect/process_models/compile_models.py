from multiprocessing import cpu_count
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from collect.bytecode.kotlin_to_bytecode import find_repositories, compile_task


def main() -> None:
    dataset = input("Path to dataset: ").strip()
    data_root = Path(dataset)
    fields = [p.name for p in data_root.iterdir() if p.is_dir() and (p / "originals").is_dir()]
    for field in fields:
        src = data_root / field / "originals"
        dst = data_root / field / "bytecode"
        dst.mkdir(parents=True, exist_ok=True)

        repos = find_repositories(src)
        print(f"Found {len(repos)} repositories for field '{field}'.")

        tasks = [
            (repo, dst)
            for repo in repos
            if not (dst / repo.name).exists()
        ]

        print(f"Found {len(tasks)} tasks to compile for field '{field}'.")

        process_map(
            compile_task,
            tasks,
            max_workers=cpu_count() - 1,
            chunksize=1,
            desc="Compiling"
        )

        print(f"Compilation finished for field '{field}'.")
        print(f"See compile_errors.log in {dst} for errors.")


if __name__ == "__main__":
    main()
