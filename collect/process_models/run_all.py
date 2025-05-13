import subprocess
import sys
from shared import Config

if __name__ == "__main__":
    python_executable = sys.executable

    CFG = Config()
    CFG.out_dir.mkdir(exist_ok=True)

    for model in CFG.model_names:
        print(f"\n=== Running model: {model} ===")
        result = subprocess.run(
            [python_executable, "process_model.py", model],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error processing {model}, return code: {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
