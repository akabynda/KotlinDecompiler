from pathlib import Path

MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_NAME = MODEL.split("/")[-1]
DATASET = "akabynda/KExercises-KStack-clean-bytecode"
STUDY_NAME = f"{MODEL_NAME}-{DATASET.split("/")[-1]}-4bit-lora"
RUNS_DIR = Path(STUDY_NAME) / "runs"
DB_URI = f"sqlite:///{STUDY_NAME}.db"
TRAIN_SUBSET_SIZE = 256
VAL_SUBSET_SIZE = TRAIN_SUBSET_SIZE // 20
METRIC_TIMEOUT = 30
TRAIN_EPOCHS = 4
GRAD_ACC = 32
SEQ_LEN_PERCENTILE = 95
LORA_CFG = dict(r=32, lora_alpha=128, lora_dropout=0.05, bias="lora_only", target_modules="all-linear")
CLIP_NORM = 0.7
WEIGHT_DECAY = 0.05
LEARNING_RATE = 1e-4
WARMUP = 0.05
