import subprocess
from shared import Config

if __name__ == "__main__":
    CFG = Config()
    CFG.out_dir.mkdir(exist_ok=True)

    for model in CFG.model_names:
        print(f"\n=== Running model: {model} ===")
        result = subprocess.run(["python", "process_model.py", model])
        if result.returncode != 0:
            print(f"Error processing {model}, return code: {result.returncode}")
