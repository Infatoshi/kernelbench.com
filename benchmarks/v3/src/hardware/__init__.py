"""Hardware target registry for KernelBench benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


class HardwareTarget:
    """Base class for hardware-specific benchmark configurations."""
    name: str = "base"
    display_name: str = "Base"
    gpu_sku: str = "UNKNOWN"
    vram_gb: int = 0
    problem_dirs: List[str] = []
    exclude_problems: List[str] = []
    is_metal: bool = False

    def max_turns(self, level: int) -> int:
        return {1: 10, 2: 12, 3: 15, 4: 15}.get(level, 15)

    def max_time(self, level: int) -> int:
        """Wall clock timeout in seconds for the agent loop."""
        return {1: 900, 2: 1800, 3: 2700, 4: 2700}.get(level, 2700)

    def find_problems(self, project_root: Path) -> List[Tuple[int, Path]]:
        problems: List[Tuple[int, Path]] = []
        problems_dir = project_root / "problems"
        for dir_name in self.problem_dirs:
            d = problems_dir / dir_name
            if not d.exists():
                continue
            level = _extract_level(dir_name)
            for f in sorted(d.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                if f.name in self.exclude_problems:
                    continue
                problems.append((level, f))
        return problems

    def create_sandbox(self, problem_code: str):
        raise NotImplementedError


def _extract_level(dir_name: str) -> int:
    """Extract numeric level from directory name like 'level2' or 'metal_level3'."""
    import re
    m = re.search(r"level(\d+)", dir_name)
    return int(m.group(1)) if m else 1


TARGETS: Dict[str, HardwareTarget] = {}


def register(name: str):
    def decorator(cls):
        TARGETS[name] = cls()
        return cls
    return decorator


def get_target(name: str) -> HardwareTarget:
    if name not in TARGETS:
        available = ", ".join(sorted(TARGETS.keys()))
        raise ValueError(f"Unknown hardware target '{name}'. Available: {available}")
    return TARGETS[name]


def list_targets() -> List[str]:
    return sorted(TARGETS.keys())


from src.hardware import rtx3090, h100, b200, m4max  # noqa: E402, F401
