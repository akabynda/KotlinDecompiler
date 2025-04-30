from pathlib import Path

import pandas as pd

path = Path(input()).expanduser().resolve()
df = pd.read_csv(path)
df.sort_values("R", ascending=False).to_csv(path, index=False)
