"""
This module dynamically imports all submodules of the current package, except for __init__.py.
This ensures that any metric functions decorated with @metric in submodules
are automatically registered and available for usage.
"""

import importlib
import pathlib
import pkgutil
from typing import Callable, Dict

registry: Dict[str, Callable] = {}


def metric(name: str):
    def wrapper(fn: Callable):
        registry[name] = fn
        return fn

    return wrapper


_pkg_path = pathlib.Path(__file__).parent
for m in pkgutil.iter_modules([str(_pkg_path)]):
    if m.name != "__init__":
        importlib.import_module(f"{__name__}.{m.name}")
