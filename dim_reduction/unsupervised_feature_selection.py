from pathlib import Path
from typing import List

import pandas as pd

from dim_reduction.common import numeric_df, safe_corr_df_interactive, drop_low_variance

if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv: ")))

    removed: List[str] = []

    num_df = numeric_df(df, removed)

    low_var_df = drop_low_variance(num_df, removed)

    reduced_df = safe_corr_df_interactive(low_var_df, removed)

    if removed:
        print("Окончательно удалены как избыточные:", len(removed), removed)

    print("Остались:", len(reduced_df.columns), reduced_df.columns.tolist())

"""Остались: 16 ['Abrupt Control Flow', 'Halstead Difficulty', 'Halstead Volume', 'Labeled Blocks', 'Pivovarsky N(G)', 'CE', 'PPL', 'CondE', 'LM_CE', 'LM_PPL', 'detekt_FunctionParameterNaming', 'detekt_MagicNumber', 'detekt_MaxLineLength', 'detekt_LoopWithTooManyJumpStatements', 'detekt_ClassNaming', 'detekt_InvalidPackageDeclaration']"""
