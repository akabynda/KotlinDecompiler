from typing import List

GLOBAL_SEED = 228
FEATURES: List[str] = [
    'CondE',
    'Conditional Complexity',
    'Halstead Distinct Operators',
    'JSD',
    'KL',
    'LM_CE',
    'LM_CondE',
    'LM_KL'
]
COV_EPS: float = 1e-3
