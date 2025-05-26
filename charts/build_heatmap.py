from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FEATURES = ['CondE', 'Conditional Complexity', 'Halstead Distinct Operators', 'JSD', 'KL', 'LM_CE', 'LM_CondE', 'LM_KL']



def save_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    selected_metrics = [m for m in FEATURES if m in summary.columns]
    if not selected_metrics:
        print("No selected metrics found in summary, skipping heatmap.")
        return

    summary.index = [i.replace("KExercises-KStack-clean-bytecode-4bit-lora", "Finetuned") for i in summary.index]

    subset = summary[selected_metrics]

    rename_dict = {
        'Conditional Complexity': 'CondComp',
        'Halstead Distinct Operators': 'HalstDO'
    }

    subset = subset.rename(columns=rename_dict)

    plt.figure(figsize=(12, 8))
    sns.heatmap(subset, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Average value'})
    plt.title("Heatmap of Selected Metrics by Model")
    plt.ylabel("Model")
    plt.xlabel("Metric")
    plt.tight_layout()

    heatmap_path = out_dir / "heatmap_selected_metrics.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {heatmap_path.absolute()}")
