from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Effort")
def halstead_effort(root: Node):
    return derived(root)["Halstead Effort"]
