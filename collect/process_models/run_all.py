import subprocess
import sys

from utils.clear_hf_cache import clear_hf_cache
from shared import Config

if __name__ == "__main__":
    clear_hf_cache()

    python_executable = sys.executable

    CFG = Config()
    CFG.out_dir.mkdir(exist_ok=True)

    for model in CFG.model_names:
        print(f"\n=== Running model: {model} ===")
        result = subprocess.run(
            [python_executable, "process_model.py", model],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            print(f"Error processing {model}, return code: {result.returncode}")

        clear_hf_cache()
