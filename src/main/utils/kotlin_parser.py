from functools import lru_cache

from tree_sitter_languages import get_language, get_parser


@lru_cache
def _get_parser():
    language = get_language("kotlin")
    return get_parser("kotlin")


def parse(code: str):
    parser = _get_parser()
    return parser.parse(code.encode("utf8")).root_node
