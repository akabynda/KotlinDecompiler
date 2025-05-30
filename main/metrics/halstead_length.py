from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Length")
def halstead_length(root: Node):
    return derived(root)["Halstead Length"]
