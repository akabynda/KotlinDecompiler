from tree_sitter import Node

from metrics import metric
from .halstead_common import counts


@metric("Halstead Total Operands")
def halstead_N2(root: Node):
    _, N2, _, _ = counts(root)
    return N2
