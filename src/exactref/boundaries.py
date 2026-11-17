"""Rounding boundaries of operations.

A *boundary* is a place where every correct implementation turns an exact
intermediate value into a finite-precision number, because the value must be
materialized before the next step can use it (the row sum of a softmax, say).
Each operation declares its boundaries. Kinds:

- ``"round"``: one rounding of a computed value.
- ``"accumulate"``: a sum of terms (normal accumulation by default, any
  order with ``standard="any_order"``).
- ``"function"``: the result of a library function such as ``exp``, which
  may be off by a few units in the last place.
- ``"backward"``: for matrix functions, the computed result is the exact
  function of a slightly perturbed input (a backward error).
- ``Uses``: a nested operation whose boundaries appear under this name,
  plus an ``output`` boundary for its own result.

Full names are dotted paths such as ``"values.accumulate"``.

``Result.tolerance()`` budgets precision loss at these boundaries: one
rounding in the compute format at each, accumulation in the compute format
(normal accumulation, or any order with ``standard="any_order"``), library
functions within a number of ulps, and the final rounding to the output
format. Roundings that only some implementations make belong to
named algorithms instead.

``RoundTo`` and ``Accumulation`` are the settings at a boundary:
``required_settings`` gives the ones a tolerance requires, named algorithms
replace some of them, and the attention kernels read them (internal).

Each operation registers its ``OpSpec`` (its boundaries, guarantee and limitations)
in ``OPS`` with ``register``; ``leaf_boundaries`` and ``Result.boundaries`` read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .accumulation import MMA, get_method
from .formats import FloatFormat, Rounding, float32

KINDS = ("round", "accumulate", "function", "backward")


@dataclass(frozen=True)
class Boundary:
    """One of an operation's rounding boundaries, where every correct implementation turns an exact
    intermediate value into a finite-precision number: its ``kind`` (``"round"``, one rounding of a
    computed value; ``"accumulate"``, a sum of terms; ``"function"``, a library function's result
    within some ulps; ``"backward"``, a matrix function's backward error), a ``description`` of the
    value, and for an accumulation what it ``applies_to``: ``"sum"`` or ``"dot"``."""

    kind: str
    description: str
    applies_to: str | None = None  # "sum" or "dot", for accumulations

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"boundary kind must be one of {KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class Uses:
    """A nested operation whose boundaries appear under this boundary's name."""

    op: str
    description: str


@dataclass(frozen=True)
class RoundTo:
    """Setting for a rounding or function boundary: the format, the rounding
    direction, and the error allowed there in units in the last place. The
    default 1/2 means one correct rounding in that direction: half an ulp to
    nearest, one ulp for a directed rounding. With ``flush``, a result below
    2**emin may also be off by up to 2**emin at this boundary, as with the
    flush-to-zero option (Lemma 1's flushing case)."""

    format: FloatFormat
    rounding: Rounding = Rounding.NEAREST_EVEN
    ulps: Fraction = Fraction(1, 2)
    flush: bool = False


@dataclass(frozen=True)
class Accumulation:
    """Setting for an accumulation boundary: a method and an accumulator format (with the
    ``MMA`` description for the "mma" method)."""

    method: str
    format: FloatFormat
    block_size: int | None = None
    rounding: Rounding = Rounding.NEAREST_EVEN
    mma: MMA | None = None

    def __post_init__(self) -> None:
        get_method(self.method)
        if (self.block_size is not None) != (self.method == "blocked"):
            raise ValueError("block_size is required for, and only for, the 'blocked' method")
        if self.block_size is not None and type(self.block_size) is not int:
            raise TypeError("block_size must be an int")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size must be positive")
        if (self.mma is not None) != (self.method == "mma"):
            raise ValueError("mma is required for, and only for, the 'mma' method")


@dataclass(frozen=True)
class OpSpec:
    """What the code implements for one operation.

    ``guarantee`` is what the rounded outputs satisfy; ``limitations`` lists
    its declared gaps relative to the mathematical operation.
    """

    key: str
    boundaries: dict
    guarantee: str
    limitations: tuple[str, ...] = ()


# A backward's limitation where the forward also takes integers: a gradient is a real or complex quantity
NO_INTEGER_GRADIENT = "floating-point and complex inputs only (integers have no gradient)"

OPS: dict[str, OpSpec] = {}

_NESTED_OUTPUT = Boundary("round", "rounding of this nested operation's result")


def register(spec: OpSpec) -> None:
    """Register an operation. A backward is registered as ``"<op>.backward"``,
    under its forward's key."""
    OPS[spec.key] = spec


def leaf_boundaries(op: str, *, nested: bool = False) -> dict[str, Boundary]:
    """Every boundary of an operation, keyed by full dotted path."""
    out: dict[str, Boundary] = {}

    def walk(boundaries: dict, prefix: str) -> None:
        for name, boundary in boundaries.items():
            path = prefix + name
            if isinstance(boundary, dict):
                walk(boundary, path + ".")
            elif isinstance(boundary, Uses):
                for sub_path, sub in leaf_boundaries(boundary.op, nested=True).items():
                    out[f"{path}.{sub_path}"] = sub
            else:
                out[path] = boundary

    walk(OPS[op].boundaries, "")
    if nested:
        out["output"] = _NESTED_OUTPUT
    return out


def required_settings(op: str, compute: FloatFormat, function_ulps: Fraction) -> dict[str, RoundTo | Accumulation]:
    """The settings of the required boundaries: every boundary in ``compute``,
    accumulations as any order (whose bound also covers normal accumulation),
    library functions within ``function_ulps``."""
    settings = {}
    for path, boundary in leaf_boundaries(op).items():
        if boundary.kind == "accumulate":
            settings[path] = Accumulation("any_order", compute)
        elif boundary.kind == "function":
            settings[path] = RoundTo(compute, ulps=function_ulps)
        elif boundary.kind == "round":
            settings[path] = RoundTo(compute)
    return settings


def default_compute(fmt: FloatFormat) -> FloatFormat:
    """Implementations widen narrow floats: below float32 precision they compute in float32."""
    return float32 if fmt.precision < float32.precision else fmt
