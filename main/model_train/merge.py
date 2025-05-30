from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

from main.model_train.config import MODEL, RUNS_DIR


class ModelMerger:
    """
    Merges a base model with its LoRA adapter and saves the merged model.
    """

    def __init__(self, model_path: Path, output_dir: Path) -> None:
        """
        Initialize the ModelMerger.

        Args:
            model_path (Path): Path to the directory with the LoRA adapter.
            output_dir (Path): Directory to save the merged model.
        """
        self.model_path: Path = model_path
        self.output_dir: Path = output_dir

    def merge_and_save(self) -> None:
        """
        Merge the base model with the LoRA adapter and save the result.
        """
        print("Loading base model ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            trust_remote_code=True
        )

        print("Loading PEFT adapter ...")
        model = PeftModel.from_pretrained(base_model, self.model_path)

        print("Merging and unloading LoRA adapter ...")
        merged_model = model.merge_and_unload()

        print(f"Saving merged model to {self.output_dir} ...")
        merged_model.save_pretrained(self.output_dir)
        print("Merge and save complete.")


def main() -> None:
    """
    Entry point for merging and saving the model.
    """
    model_path = RUNS_DIR / "full_finetune" / "model"
    output_dir = RUNS_DIR / "full_finetune" / "merged_model"

    merger = ModelMerger(model_path, output_dir)
    merger.merge_and_save()


if __name__ == "__main__":
    main()
