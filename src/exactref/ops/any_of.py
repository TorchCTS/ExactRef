"""``any_of``: the center's result, covering an implementation that computes one of several expressions."""

from __future__ import annotations

from .. import _native
from .._fenv import ieee
from ..array import Array
from ..formats import _part_format, complex128, float64, uint8
from ..result import Bounds, Result


@ieee
def any_of(center: Result, *alternatives: Result) -> Result:
    """The center's result, with its bounds widened to cover an implementation that computes
    any one of ``alternatives`` instead.

    Per element (each part of a complex element), and for every set of tolerance arguments
    (including the standard): the finite range is the smallest range holding the center's
    and every alternative's, the infinities and NaN are those any of them allows, and the
    tolerance is the largest distance from the center's ``rounded`` to an end of those
    ranges, rounded upward. A center part that is not finite has a NaN tolerance, as in
    every tolerance; a part that has no bound in the center or in an alternative has an
    infinite one. ``rounded``, ``exact``, ``guarantee``, ``normal()`` and ``backward()`` are
    the center's, so ``normal()`` is the center expression's normal answer. Every result
    must have one unnamed output (a backward's gradient as ``result[name]``), with the
    same shape and dtype (``proofs/any_of.md``).
    """
    results = (center,) + alternatives
    for r in results:
        if not isinstance(r, Result):
            raise TypeError(f"any_of takes results, got {type(r).__name__}")
        r._single("passed")
    first = center.rounded
    for r in alternatives:
        other = r.rounded
        if other.dtype != first.dtype:
            raise TypeError(f"any_of needs one dtype, got {first.dtype.name} and {other.dtype.name}")
        if other.shape != first.shape:
            raise ValueError(f"any_of needs one shape, got {first.shape} and {other.shape}")

    def bounds(compute, ulps, flush, standard, workers=None):
        return _widest(first, [r._bounds_of(compute, ulps, flush, standard, workers) for r in results])

    return center._with_bounds(bounds)


def _widest(center: Array, found: list[Bounds]) -> Bounds:
    """``any_of``'s bounds around ``center`` (the center's ``rounded``) from the center's and
    the alternatives' bounds."""
    dtype = center.dtype
    component = _part_format(dtype)

    def data(array):
        return bytes(array.buffer), array.byteorder == "big"

    arrays = [(*data(b.tolerance), *data(b.low), *data(b.high), bytes(b.flags.buffer)) for b in found]
    tolerance, low, high, flags = _native.widest(component._layout, *data(center), float64._layout, arrays, dtype.components)
    shape = center.shape
    return Bounds(Array(complex128 if dtype.kind == "complex" else float64, shape, tolerance), Array(dtype, shape, low), Array(dtype, shape, high),
                  Array(uint8, shape, flags))
