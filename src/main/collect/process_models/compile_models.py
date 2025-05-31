from multiprocessing import cpu_count
from pathlib import Path
from typing import List

from tqdm.contrib.concurrent import process_map

from src.main.collect.bytecode.kotlin_bytecode_compiler import KotlinBytecodeCompiler


class CompileModels:
    """
    Orchestrates the compilation of Kotlin bytecode for multiple models.
    """

    def __init__(self, dataset_root: Path) -> None:
        """
        Initialize the manager.

        Args:
            dataset_root (Path): Path to the dataset containing models with 'originals' directories.
        """
        self.dataset_root: Path = dataset_root

    def find_models(self) -> List[str]:
        """
        Find dataset models that contain 'originals' directories.

        Returns:
            List of model names.
        """
        return [
            p.name
            for p in self.dataset_root.iterdir()
            if p.is_dir() and (p / "originals").is_dir()
        ]

    def compile_model(self, model: str) -> None:
        """
        Compile bytecode for a specific dataset model.

        Args:
            model (str): model name.
        """
        src = self.dataset_root / model / "originals"
        dst = self.dataset_root / model / "bytecode"
        dst.mkdir(parents=True, exist_ok=True)

        repos: List[Path] = KotlinBytecodeCompiler.find_repositories(src)
        print(f"Found {len(repos)} repositories for model '{model}'.")

        tasks: List[tuple[Path, Path]] = [
            (repo, dst) for repo in repos if not (dst / repo.name).exists()
        ]

        print(f"Found {len(tasks)} tasks to compile for model '{model}'.")

        process_map(
            KotlinBytecodeCompiler.compile_task,
            tasks,
            max_workers=cpu_count() - 1,
            chunksize=1,
            desc=f"Compiling ({model})",
        )

        print(f"Compilation finished for model '{model}'.")
        print(f"See compile_errors.log in {dst} for errors.")

    def run(self) -> None:
        """
        Start the compilation process for all models.
        """
        models: List[str] = self.find_models()
        for model in models:
            self.compile_model(model)


def main() -> None:
    """
    Entry point for bytecode compilation.
    """
    dataset_path = Path(input("Path to dataset: ").strip())
    manager = CompileModels(dataset_path)
    manager.run()


if __name__ == "__main__":
    main()
