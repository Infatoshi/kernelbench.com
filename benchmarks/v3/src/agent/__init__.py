"""
Agentic Kernel Optimization Module

Provides an agentic evaluation framework where the model can:
- Execute bash commands in a sandbox
- Read/write files
- Introspect models to understand their structure
- Iterate on solutions with real feedback
"""

__all__ = [
    "ModalSandbox",
    "ModalSandboxConfig",
    "GPUType",
    "create_modal_sandbox",
    "LocalSandbox",
    "LocalSandboxConfig",
    "create_local_sandbox",
    "MetalSandbox",
    "MetalSandboxConfig",
    "create_metal_sandbox",
]

_LAZY_IMPORTS = {
    "ModalSandbox": ".modal_sandbox",
    "ModalSandboxConfig": ".modal_sandbox",
    "GPUType": ".modal_sandbox",
    "create_modal_sandbox": ".modal_sandbox",
    "LocalSandbox": ".local_sandbox",
    "LocalSandboxConfig": ".local_sandbox",
    "create_local_sandbox": ".local_sandbox",
    "MetalSandbox": ".metal_sandbox",
    "MetalSandboxConfig": ".metal_sandbox",
    "create_metal_sandbox": ".metal_sandbox",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
