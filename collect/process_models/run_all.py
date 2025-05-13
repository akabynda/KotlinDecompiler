import subprocess
import sys
from shared import Config

import os
import shutil
from pathlib import Path


def clear_hf_cache():
    hf_cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))

    if hf_cache_dir.exists():
        try:
            shutil.rmtree(hf_cache_dir)
            print(f"Cleared Hugging Face cache at {hf_cache_dir}")
        except Exception as e:
            print(f"Failed to clear Hugging Face cache: {e}")
    else:
        print(f"No Hugging Face cache found at {hf_cache_dir}")


if __name__ == "__main__":
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
