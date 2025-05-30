from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Volume")
def halstead_volume(root: Node):
    return derived(root)["Halstead Volume"]
