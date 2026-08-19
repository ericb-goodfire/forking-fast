"""Import the pristine baseline/ tree (otrecon + runners) under isolated
module names so tests can compare optimized vs reference implementations
in one process."""
from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "baseline")
_PREFIXES = ("otrecon", "src")
_CACHE = None


def _matches(name: str) -> bool:
    return name in _PREFIXES or name.startswith(
        tuple(p + "." for p in _PREFIXES))


def load_baseline() -> SimpleNamespace:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    saved = {}
    for k in list(sys.modules):
        if _matches(k):
            saved[k] = sys.modules.pop(k)
    sys.path.insert(0, BASE)
    try:
        models = importlib.import_module("otrecon.models")
        cv = importlib.import_module("otrecon.cv")
        data = importlib.import_module("otrecon.data")
        run_scale = importlib.import_module("src.run_scale")
        run_ablation = importlib.import_module("src.run_ablation")
    finally:
        for k in list(sys.modules):
            if _matches(k):
                del sys.modules[k]
        sys.modules.update(saved)
        while BASE in sys.path:
            sys.path.remove(BASE)
    _CACHE = SimpleNamespace(models=models, cv=cv, data=data,
                             run_scale=run_scale, run_ablation=run_ablation)
    return _CACHE
