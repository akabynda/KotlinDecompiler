from pathlib import Path

import pandas as pd
from datasets import load_dataset

from collect.process_models.shared import Config

CFG = Config()


def merge_all_jsonl_with_hf(input_dir: str, output_file: str):
    input_path = Path(input_dir)

    ds = load_dataset(CFG.dataset_name, split=CFG.split)
    hf_df = pd.DataFrame(ds)

    merged_df = hf_df.copy()

    for jsonl_file in input_path.glob("*.jsonl"):
        local_df = pd.read_json(jsonl_file, lines=True)
        local_df = local_df.drop(
            columns=[col for col in local_df.columns if col in merged_df.columns and col != 'kt_path'])
        merged_df = pd.merge(
            merged_df,
            local_df,
            on="kt_path",
            how="left",
            suffixes=("", f"_{jsonl_file.stem}")
        )

    local_columns = [col for col in merged_df.columns if col != "kt_path" and col not in hf_df.columns]
    merged_df = merged_df.dropna(subset=local_columns, how='all')

    merged_df.to_json(
        output_file,
        orient="records",
        lines=True,
        force_ascii=False
    )
    print(f"Merged dataset saved to: {output_file}")


if __name__ == "__main__":
    input_dir = input("Введите путь к директории с .jsonl файлами: ")
    output_file = input_dir + ".jsonl"
    merge_all_jsonl_with_hf(input_dir, output_file)
