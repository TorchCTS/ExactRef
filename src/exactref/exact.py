"""Certified exact values in packed storage.

An ``ExactArray`` holds, for each output element, a midpoint and a radius: the
true value lies within the radius of the midpoint. No Python object exists per
element until one is read. There are two fixed-width streams:

- Midpoints, 24 bytes each: a 128-bit unsigned significand ``m``, then a
  signed 64-bit word ``2 * e + sign``, for ``(-1)**sign * m * 2**e``.
- Flags, one byte each: a 2-bit kind (finite, NaN, infinite, near zero) and a
  6-bit radius code. Code 0 means radius zero; code k means a radius of
  ``2**(k - 63)`` units in the last place of the midpoint, that unit being
  ``2**(top - 128)`` for ``2**(top - 1) <= |midpoint| < 2**top``. So the radius
  is at most one unit in the midpoint's 128th bit.

A near-zero element stores an absolute radius ``r`` in its midpoint slot: the
true value lies in ``[-r, r]``. A complex element holds two entries, real
then imaginary.

Stored magnitudes lie between ``2**-LIMIT`` and ``2**LIMIT``; the computations
that produce them have no such limit. A finite nonzero value beyond that range
is stored the way MPFR reports one: as an infinity (too big) or a zero (too
small) with the value's sign, marked out of range by radius code 1 on the
infinite or near-zero kind. Every format ExactRef accepts lies inside the
range, so such an element still determines its rounding to any of them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from . import _native
from .array import _normalize_shape, flat_position
from .values import _KINDS, FloatValue

EXACT_PRECISION = _native.ENTRY_BITS + 32  # bits; enough for exact entries in one pass
_WIDTH = _native.ENTRY_WIDTH


def start_precision(fmt) -> int:
    """The working precision a refinement (Ziv's strategy) starts from for outputs in ``fmt``:
    32 bits beyond the format, and enough for exact entries in one pass."""
    return max(fmt.precision + 32, EXACT_PRECISION)


@dataclass(frozen=True)
class Enclosure:
    """A certified value: the true value lies in ``[midpoint - radius, midpoint + radius]``.

    ``radius`` is 0 when ``midpoint`` is the exact value, NaN or infinite.
    With ``out_of_range``, the true value is finite and nonzero with
    ``midpoint``'s sign, and its magnitude is at least ``2**LIMIT`` (``midpoint``
    infinite) or below ``2**-LIMIT`` (``midpoint`` zero).
    """

    midpoint: FloatValue
    radius: Fraction
    out_of_range: bool = False


class ExactArray:
    """Certified exact values of one output: per element a midpoint and a radius, the true
    value within the radius of the midpoint, packed so that no Python object exists per
    element until one is read."""

    __slots__ = ("_shape", "_components", "_midpoints", "_flags")

    def __init__(self, shape: tuple[int, ...], components: int, entries: list) -> None:
        """Build from ``(flag, midpoint bytes)`` entries in row-major order, ``components`` per element."""
        self._shape = tuple(shape)
        self._components = components
        if len(entries) != math.prod(self._shape) * components:
            raise ValueError(f"shape {self._shape} needs {math.prod(self._shape) * components} entries, got {len(entries)}")
        self._flags = bytes(flag for flag, _ in entries)
        self._midpoints = b"".join(midpoint for _, midpoint in entries)

    @classmethod
    def _from_storage(cls, shape: tuple, components: int, flags: bytes, midpoints: bytes) -> ExactArray:
        """Built from the packed storage itself: a flag byte and a midpoint per entry."""
        out = cls.__new__(cls)
        out._shape, out._components, out._flags, out._midpoints = tuple(shape), components, flags, midpoints
        return out

    def _storage(self) -> tuple[bytes, bytes]:
        """The packed storage itself, as ``_from_storage`` takes it: the flag bytes and the midpoints."""
        return self._flags, self._midpoints

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return math.prod(self._shape)

    def value(self, index=()) -> Enclosure | tuple[Enclosure, Enclosure]:
        """The enclosure of one element; a ``(real, imaginary)`` pair for complex outputs."""
        return self._element(flat_position(index, self._shape))

    def values(self) -> list[Enclosure | tuple[Enclosure, Enclosure]]:
        """Every element's enclosure in row-major order."""
        return [self._element(position) for position in range(self.size)]

    def reshape(self, shape) -> ExactArray:
        """The same elements in row-major order with a new shape of the same size."""
        shape = _normalize_shape(shape)
        if math.prod(shape) != self.size:
            raise ValueError(f"cannot reshape {self._shape} to {shape}")
        return ExactArray._from_storage(shape, self._components, self._flags, self._midpoints)

    def take(self, positions: Sequence[int], shape) -> ExactArray:
        """The elements at row-major ``positions``, as an array of ``shape``; a negative
        position counts from the end, as in ``torch.take``."""
        shape = _normalize_shape(shape)
        c = self._components
        if positions:
            size, low = self.size, min(positions)
            if low < -size or max(positions) >= size:
                raise IndexError(f"a position is out of range for {size} elements")
            if low < 0:
                positions = [p % size for p in positions]
        if len(positions) != math.prod(shape):
            raise ValueError(f"{len(positions)} positions do not fill shape {shape}")
        rows = [p * c + j for p in positions for j in range(c)]
        return ExactArray._from_storage(shape, c, bytes(self._flags[i] for i in rows),
                                        b"".join(self._midpoints[i * _WIDTH : (i + 1) * _WIDTH] for i in rows))

    def paired(self, shape) -> ExactArray:
        """A real array's entries read in pairs (real part, imaginary part) as a complex array
        of ``shape``."""
        shape = _normalize_shape(shape)
        if self._components != 1 or 2 * math.prod(shape) != self.size:
            raise ValueError(f"cannot pair {self._shape} into complex {shape}")
        return ExactArray._from_storage(shape, 2, self._flags, self._midpoints)

    def _element(self, position: int):
        parts = [self._decode(position * self._components + c) for c in range(self._components)]
        return parts[0] if self._components == 1 else tuple(parts)

    def _decode(self, i: int) -> Enclosure:
        kind, negative, m, e, rm, re, out_of_range = _native.entry_read(self._flags[i], self._midpoints[i * _WIDTH : (i + 1) * _WIDTH])
        radius = Fraction(rm) * Fraction(2) ** re if rm else Fraction(0)
        return Enclosure(FloatValue(_KINDS[kind], negative, m, e), radius, out_of_range)

    def __repr__(self) -> str:
        return f"ExactArray(shape={self._shape})"


# Entries built one value at a time, for the tests: operations build an ExactArray from the kernels' bytes
# (``ExactArray._from_storage``).
def exact_entry(value) -> tuple[int, bytes]:
    """The entry of an exactly known value: a ``FloatValue``, an int or a ``Fraction``.

    A value longer than 128 bits is rounded to nearest, with a radius of half a unit.
    """
    if isinstance(value, FloatValue):
        return _native.exact_entry(_KINDS.index(value.kind), value.negative, value.significand, value.exponent)
    q = Fraction(value)
    return _native.ratio_entry(q < 0, abs(q.numerator), q.denominator)


def scaled_entry(m: int, e: int) -> tuple[int, bytes]:
    """``exact_entry`` of the value ``m * 2**e`` for an integer ``m``, with the same bytes."""
    return _native.exact_entry(0, m < 0, abs(m), e)


def interval_entry(lo, hi) -> tuple[int, bytes] | None:
    """The entry of a value known to lie in the dyadic interval ``[lo, hi]``, or None if it is too wide.

    An interval that contains zero gives a near-zero entry, and one wholly
    beyond the stored range an out-of-range entry; one that crosses the range's
    edge is too wide. The ends are compared by their exponents before any
    alignment, so ends far apart never build huge integers.
    """
    return _native.interval_entry(lo, hi)
