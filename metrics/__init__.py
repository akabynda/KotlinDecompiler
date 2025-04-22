"""
Любая функция, помеченная декоратором @metric, автоматически
попадает в общий реестр и будет вызвана в analyse_tests.py
"""
from typing import Callable, Dict

registry: Dict[str, Callable] = {}


def metric(name: str):
    def wrapper(fn: Callable):
        registry[name] = fn
        return fn

    return wrapper


import importlib
import pkgutil
import pathlib

_pkg_path = pathlib.Path(__file__).parent
for m in pkgutil.iter_modules([str(_pkg_path)]):
    if m.name != '__init__':
        importlib.import_module(f"{__name__}.{m.name}")
