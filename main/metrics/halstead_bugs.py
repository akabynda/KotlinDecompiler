from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Bugs")
def halstead_bugs(root: Node):
    return derived(root)["Halstead Bugs"]
