"""Shape-sweep loader. Each problem declares its canonical shape list in shapes.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_shapes(problem_dir: Path) -> list[dict]:
    """Load `SHAPES: list[dict]` from problems/<X>/shapes.py."""
    shapes_py = problem_dir / "shapes.py"
    if not shapes_py.exists():
        raise FileNotFoundError(f"{shapes_py} not found")
    spec = importlib.util.spec_from_file_location("shapes", shapes_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "SHAPES"):
        raise AttributeError(f"{shapes_py} must define SHAPES: list[dict]")
    return list(mod.SHAPES)
