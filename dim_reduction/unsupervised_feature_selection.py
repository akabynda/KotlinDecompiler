from pathlib import Path
from typing import List

import pandas as pd

from dim_reduction.common import numeric_df, drop_low_variance, safe_corr_df

if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv: ")))

    removed: List[str] = []

    num_df = numeric_df(df, removed)

    low_var_df = drop_low_variance(num_df, removed)

    reduced_df = safe_corr_df(low_var_df, removed)

    if removed:
        print("Окончательно удалены как избыточные:", len(removed), removed)

    print("Остались:", len(reduced_df.columns), reduced_df.columns.tolist())
