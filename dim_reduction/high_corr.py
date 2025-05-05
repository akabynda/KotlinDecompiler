from pathlib import Path
from typing import List

import pandas as pd

from dim_reduction.common import numeric_df, safe_corr_df, safe_corr_df_interactive

if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv:")))

    num = numeric_df(df)
    removed: List[str] = []
    num = safe_corr_df_interactive(num, removed)
    if removed:
        print("Окончательно удалены как избыточные:", len(removed), removed)

    print("Остались:", len(num.columns), num.columns.tolist())
