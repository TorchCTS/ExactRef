"""Exact answers whatever the caller's floating-point environment (internal).

A thread may flush subnormal numbers to zero: ``torch.set_flush_denormal(True)``
sets DAZ and FTZ in x86's MXCSR, or FZ in ARM's FPCR. Float arithmetic follows
that setting, so the public calls that compute are wrapped in ``ieee``: the
operations and their backwards, ``any_of``, ``asarray``, ``convert``,
``round_to_tf32``, and the methods of results and algorithms. When the calling
thread flushes, the wrapper saves the thread's environment, resets it to the
default, runs the call and restores the environment; a save, reset or restore
that fails raises ``ReferenceUnavailableError``. The save, reset and restore are
the C library's ``fegetenv`` and ``fesetenv``, called by exactref's C
(``csrc/system.c``). The element codec (a format's ``encode``, ``decode`` and
``to_scalar``, ``float()`` and ``complex()`` of a value, and ``Array``'s
``scalar``, ``scalars`` and ``tolist``) needs no wrapper: it moves a float's
bits with ``struct`` (an array's elements in the C, by ``memcpy``) and encodes
with the C's integer arithmetic, which flushing does not reach.
"""

from __future__ import annotations

import functools
import struct
import sys
from typing import Callable, ParamSpec, TypeVar

from . import _native
from .errors import ReferenceUnavailableError

_SMALLEST = b"\x01\x00\x00\x00"  # the smallest float32 subnormal, little-endian


def flushing() -> bool:
    """Whether the calling thread flushes subnormal inputs or results to zero."""
    return struct.unpack("<f", _SMALLEST)[0] == 0.0 or sys.float_info.min / 2 == 0.0


def _unavailable(cause: str) -> ReferenceUnavailableError:
    return ReferenceUnavailableError(
        f"the calling thread flushes subnormal numbers to zero, and ExactRef cannot compute exactly that way: {cause}; "
        "turn flushing off around ExactRef calls (for example torch.set_flush_denormal(False))"
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def ieee(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``function`` in the default floating-point environment when the calling thread
    flushes subnormals, and restore the thread's environment afterwards."""

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not flushing():
            return function(*args, **kwargs)
        saved = _native.fenv_save()
        if saved is None:
            raise _unavailable("saving the environment failed")
        try:
            if not _native.fenv_reset() or flushing():
                raise _unavailable("resetting the environment did not stop it")
            result = function(*args, **kwargs)
        finally:
            restored = _native.fenv_restore(saved)
        # reached only when the call returned, so a failed restore never hides the call's own error
        if not restored:
            raise ReferenceUnavailableError("restoring the calling thread's floating-point environment after an ExactRef call failed: "
                                            "the thread now runs in the default environment, without flushing")
        return result

    return wrapper
