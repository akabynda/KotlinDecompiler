from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Difficulty")
def halstead_difficulty(root: Node):
    return derived(root)["Halstead Difficulty"]
