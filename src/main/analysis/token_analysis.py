from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer


class TokenAnalysis:
    """
    Performs token counting and statistical analysis for Kotlin source code
    and its compiled JVM bytecode.
    """

    def __init__(self, dataset_name: str, tokenizer_name: str) -> None:
        """
        Initialize the TokenAnalysis with a dataset and tokenizer.
        """
        self.dataset: Dataset = load_dataset(dataset_name, split="train")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text using the tokenizer.
        """
        return len(self.tokenizer(text).input_ids)

    def analyze(self) -> pd.DataFrame:
        """
        Analyze the dataset by computing token counts for Kotlin source
        and JVM bytecode, and calculate ratios.
        """
        records: List[Dict[str, int]] = []
        for row in self.dataset:
            kt_tokens = self.count_tokens(row["kt_source"])
            bytecode = "\n".join(cls["javap"] for cls in row["classes"])
            bc_tokens = self.count_tokens(bytecode)

            records.append({"kt_tokens": kt_tokens, "bc_tokens": bc_tokens})

        df = pd.DataFrame(records)
        df["ratio_bc_to_kt"] = df["bc_tokens"] / df["kt_tokens"]
        return df

    @staticmethod
    def compute_statistics(df: pd.DataFrame) -> None:
        """
        Compute and print Pearson and Spearman correlation coefficients,
        as well as mean and median ratios.
        """
        pearson_r, pearson_p = pearsonr(df["kt_tokens"], df["bc_tokens"])
        spearman_r, spearman_p = spearmanr(df["kt_tokens"], df["bc_tokens"])

        print(f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})")
        print(f"Spearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})")

        mean_ratio = df["ratio_bc_to_kt"].mean()
        median_ratio = df["ratio_bc_to_kt"].median()
        print(f"Mean ratio (bytecode/source): {mean_ratio:.2f}")
        print(f"Median ratio: {median_ratio:.2f}")

    @staticmethod
    def plot_data(df: pd.DataFrame) -> None:
        """
        Plot a scatter plot comparing source and bytecode token counts.
        """
        plt.scatter(df["kt_tokens"], df["bc_tokens"], alpha=0.4, s=20)
        plt.title("Kotlin source vs. JVM bytecode (token count)")
        plt.xlabel("Tokens in Kotlin source")
        plt.ylabel("Tokens in bytecode (javap)")
        plt.grid(True)
        plt.show()


def main() -> None:
    analysis = TokenAnalysis(
        dataset_name="akabynda/KExercises-bytecode",
        tokenizer_name="JetBrains/deepseek-coder-1.3B-kexer"
    )

    df = analysis.analyze()
    print(df.head())

    analysis.compute_statistics(df)
    analysis.plot_data(df)


if __name__ == "__main__":
    main()
