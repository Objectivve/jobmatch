"""Graph modules: prompts, typed state and the compiled workflow.

Exports are lazy so importing `jobmatch.tools` first cannot create a cycle.
"""

from typing import Any

_EXPORTS = {
    "build_graph",
    "build_user_message",
    "create_checkpointer",
    "interrupt_payload",
    "make_config",
    "resume_run",
    "start_run",
    "status_from_result",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from . import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
