from tree_sitter import Node

from main.metrics import metric
from .halstead_common import counts


@metric("Halstead Total Operators")
def halstead_N1(root: Node):
    N1, _, _, _ = counts(root)
    return N1
