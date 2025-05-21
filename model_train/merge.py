from peft import PeftModel
from transformers import AutoModelForCausalLM

from model_train.config import MODEL, RUNS_DIR

base_model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", trust_remote_code=True)

model = PeftModel.from_pretrained(base_model, RUNS_DIR / "full_finetune" / "model")

merged_model = model.merge_and_unload()

merged_model.save_pretrained(RUNS_DIR / "full_finetune" / "merged_model")
