from .log import CommandLog
from .state import State, apply_patch, diff, parse_assignments, schema
from .steer import Hooks, Steer

__version__ = "0.1.0"
__all__ = [
    "CommandLog",
    "Feeder",
    "Hooks",
    "Live",
    "State",
    "Steer",
    "apply_patch",
    "builtin",
    "diff",
    "due",
    "parse_assignments",
    "schema",
    "set_lr",
    "train_mode",
]
_LAZY = {"Feeder", "Live", "builtin", "due", "set_lr", "train_mode"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import hooks

        return getattr(hooks, name)
    raise AttributeError(name)
