"""
Configuration settings for training and evaluation.
"""

from pathlib import Path
from typing import Dict, Any

# Model and dataset information
MODEL: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_NAME: str = MODEL.split("/")[-1]
DATASET: str = "akabynda/KExercises-KStack-clean-bytecode"

# Study details
STUDY_NAME: str = f"{MODEL_NAME}-{DATASET.split('/')[-1]}-4bit-lora"
RUNS_DIR: Path = Path(STUDY_NAME) / "runs"
DB_URI: str = f"sqlite:///{STUDY_NAME}.db"

# Dataset sizes
TRAIN_SUBSET_SIZE: int = 256
VAL_SUBSET_SIZE: int = TRAIN_SUBSET_SIZE // 20

# Evaluation
METRIC_TIMEOUT: int = 30

# Training hyperparameters
TRAIN_EPOCHS: int = 4
GRAD_ACC: int = 32
SEQ_LEN_PERCENTILE: int = 95

# LoRA (Low-Rank Adaptation) configuration
LORA_CFG: Dict[str, Any] = {
    "r": 16,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": "all-linear",
    "init_lora_weights": "gaussian",
}

# Optimizer parameters
CLIP_NORM: float = 0.7
WEIGHT_DECAY: float = 0.05
LEARNING_RATE: float = 1e-4
WARMUP: float = 0.1
