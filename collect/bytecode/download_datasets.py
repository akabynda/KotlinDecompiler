import sys

from collect.bytecode.download_kexercises import KExercisesDownloader
from collect.bytecode.download_kstack_clean import KStackCleanDownloader


def download() -> None:
    """
    Entry point for processing and saving JetBrains dataset.
    """
    KExercisesDownloader().process()
    KStackCleanDownloader().process()


if __name__ == "__main__":
    sys.exit(download())
