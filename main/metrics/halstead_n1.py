from tree_sitter import Node

from main.metrics import metric
from .halstead_common import counts


@metric("Halstead Distinct Operators")
def halstead_n1(root: Node):
    _, _, n1, _ = counts(root)
    return n1
