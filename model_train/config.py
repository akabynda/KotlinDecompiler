from pathlib import Path

MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_NAME = MODEL.split("/")[-1]
RAW_DS_PATH = "KExercises-KStack-clean-bytecode"
STUDY_NAME = f"{MODEL_NAME}-{RAW_DS_PATH}-4bit-lora"
RUNS_DIR = Path(STUDY_NAME) / "runs"
DB_URI = f"sqlite:///{STUDY_NAME}.db"
TRAIN_SUBSET_SIZE = 256
VAL_SUBSET_SIZE = TRAIN_SUBSET_SIZE // 20
GLOBAL_SEED = 228
METRIC_TIMEOUT = 30
