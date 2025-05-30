from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Time")
def halstead_time(root: Node):
    return derived(root)["Halstead Time"]
