from tree_sitter import Node

from main.metrics import metric
from .halstead_common import derived


@metric("Halstead Vocabulary")
def halstead_vocabulary(root: Node):
    return derived(root)["Halstead Vocabulary"]
