from tree_sitter import Node

from main.metrics import metric
from .halstead_common import counts


@metric("Halstead Distinct Operands")
def halstead_n2(root: Node):
    _, _, _, n2 = counts(root)
    return n2
