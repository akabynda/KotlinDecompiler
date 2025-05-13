import os
from pathlib import Path
import shutil


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
