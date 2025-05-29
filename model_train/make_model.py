import torch
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from model_train.config import MODEL


def make_model(r: int, alpha: int, dropout: float) -> PeftModel:
    """
    Loads and prepares a LoRA model with 4-bit quantization.

    Args:
        r (int): LoRA rank.
        alpha (int): LoRA alpha parameter.
        dropout (float): LoRA dropout rate.

    Returns:
        PeftModel: The prepared PEFT model with LoRA.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        trust_remote_code=True,
        device_map="auto",
        use_cache=False
    )

    base_model = prepare_model_for_kbit_training(base_model)

    return get_peft_model(
        base_model,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules="all-linear",
            init_lora_weights="gaussian"
        )
    )
