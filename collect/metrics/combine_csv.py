from pathlib import Path

import pandas as pd

df1 = pd.read_csv(Path(input("First file: ")))
df2 = pd.read_csv(Path(input("Second file: ")))
output = Path(input("Output: "))

merged_df = pd.concat([df1, df2], ignore_index=True)

merged_df.to_csv(output, index=False)
