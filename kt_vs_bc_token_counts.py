import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

ds = load_dataset("akabynda/KExercises-bytecode", split="train")

tokenizer = AutoTokenizer.from_pretrained("JetBrains/deepseek-coder-1.3B-kexer")


def count_tokens(text: str) -> int:
    return len(tokenizer(text).input_ids)


records = []
for row in ds:
    kt_tokens = count_tokens(row["kt_source"])

    bytecode = "\n".join(cls["javap"] for cls in row["classes"])
    bc_tokens = count_tokens(bytecode)

    records.append({"kt_tokens": kt_tokens, "bc_tokens": bc_tokens})

df = pd.DataFrame(records)
print(df.head())

pearson_r, pearson_p = pearsonr(df.kt_tokens, df.bc_tokens)
spearman_r, spearman_p = spearmanr(df.kt_tokens, df.bc_tokens)
print(f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})")
print(f"Spearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})")

df["ratio_bc_to_kt"] = df["bc_tokens"] / df["kt_tokens"]

mean_ratio = df["ratio_bc_to_kt"].mean()
median_ratio = df["ratio_bc_to_kt"].median()

print(f"Среднее соотношение (байткод / исходник): {mean_ratio:.2f}")
print(f"Медианное соотношение: {median_ratio:.2f}")

plt.scatter(df.kt_tokens, df.bc_tokens, alpha=0.4, s=20)
plt.title("Kotlin source vs. JVM bytecode (token count)")
plt.xlabel("Tokens in kt_source")
plt.ylabel("Tokens in bytecode (javap)")
plt.grid(True)
plt.show()
