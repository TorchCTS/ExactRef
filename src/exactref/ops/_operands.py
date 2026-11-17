"""Operation inputs: arrays, or earlier results' values with their tolerances (internal)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from .. import _native
from ..array import Array, as_parts_layout, interleave_elements, row_major_strides, take_elements
from ..convert import asarray
from ..formats import (
    ComplexFormat,
    FloatFormat,
    Format,
    _part_format,
    bfloat16,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint64,
)
from ..result import Bounds, BoundsCallback, NormalCallback, Result, exact_bounds
from ..values import ComplexValue, FloatValue
from ._shapes import broadcast_index, iter_indices


class Operand:
    """An operation's input: an array, or an earlier result's ``rounded`` values, which the
    implementation under test may have stored anywhere that result's bounds allow."""

    __slots__ = ("array", "_bounds", "_normal")

    def __init__(self, array: Array, bounds: BoundsCallback[Bounds] | None = None, normal: NormalCallback[Array] | None = None) -> None:
        self.array = array
        self._bounds = bounds  # laid out as array's elements
        self._normal = normal

    @property
    def uncertain(self) -> bool:
        """Whether this is an earlier result's values (stored anywhere its bounds allow)."""
        return self._bounds is not None

    def bounds(self, compute, ulps, flush: bool, standard: str, workers=None) -> Bounds | None:
        """The earlier result's bounds, laid out as ``array``'s elements; None when every value
        is exact."""
        return None if self._bounds is None else self._bounds(compute, ulps, flush, standard, workers)

    def radius(self, compute, ulps, flush: bool, standard: str, workers=None) -> Array | None:
        """The most a stored value can differ from ``array``, per element and part: the
        earlier result's tolerance (float64, or complex128 with one radius per part; inf or
        nan where there is no bound). None when every value is exact."""
        found = self.bounds(compute, ulps, flush, standard, workers)
        if found is None:
            return None
        radius = found.tolerance
        return radius if any(radius.buffer) else None

    def marks(self, compute, ulps, flush: bool, standard: str, workers=None) -> list | None:
        """Per part, the non-finite values a stored value may be, and where it may be any
        finite value though the reference is not finite (``Bounds.marks``); None when
        there are none."""
        found = self.bounds(compute, ulps, flush, standard, workers)
        return None if found is None else found.marks(self.array)

    def normal(self, compute, flush: bool, workers=None) -> Array:
        """The values normal accumulation stores: the earlier result's ``normal()``."""
        return self.array if self._normal is None else self._normal(compute, flush, workers)

    def take(self, positions: Sequence[int], shape: tuple, conjugate: bool = False) -> Operand:
        """The elements at row-major ``positions`` (complex conjugates with ``conjugate``)
        as an operand of ``shape``."""

        def pick(array):
            return take_elements(array, positions, shape, conjugate)

        def pick_bounds(found):
            # Conjugation keeps each part's magnitude, so the radius bytes move unchanged.
            found = found.take(positions, shape)
            return found.conjugated() if conjugate else found

        return self._mapped(pick, pick_bounds)

    def converted(self, fmt) -> Operand:
        """This operand in ``fmt``, which must hold every value exactly (a widening, as
        type promotion makes); the radii and normal values carry over."""
        if self.array.dtype == fmt:
            return self
        target = _part_format(fmt)

        def convert(array):
            # each part encoded in the wider format exactly; a NaN keeps only its sign, as encode gives it
            source = _part_format(array.dtype)
            data = _native.recode(bytes(array.buffer), array.byteorder == "big", source._layout, target._layout)
            return Array(fmt, array.shape, data)

        return self._mapped(convert, lambda found: Bounds(found.tolerance, convert(found.low), convert(found.high), found.flags))

    def as_real(self) -> Operand:
        """A complex operand's parts as a real operand of shape ``shape + (2,)``, each part
        with its own radius and flags."""
        fmt = self.array.dtype.component

        def parts(array):
            return as_parts_layout(array, fmt, array.shape + (2,))

        return self._mapped(parts, lambda found: found.parts())

    def _mapped(self, values, bounds) -> Operand:
        """This operand with ``values`` applied to its array and to its normal values, and
        ``bounds`` to its bounds."""
        source, normal = self._bounds, self._normal

        def mapped_bounds(compute, ulps, flush, standard, workers=None):
            return bounds(source(compute, ulps, flush, standard, workers))

        def mapped_normal(compute, flush, workers=None):
            return values(normal(compute, flush, workers))

        return Operand(values(self.array), None if source is None else mapped_bounds, None if normal is None else mapped_normal)

def grad_operand(op: str, grad, shape: tuple, fmt) -> Operand:
    """A backward's output gradient ``grad`` as an operand, for inputs of format ``fmt``: the
    inputs must be floating-point or complex, and ``grad`` of their kind with ``shape``."""
    if fmt.kind not in ("float", "complex"):
        raise TypeError(f"{op} backward needs floating-point or complex inputs, got {fmt.name}")
    G = operand(grad)
    if G.array.shape != shape:
        raise ValueError(f"grad must have the output's shape {shape}, got {G.array.shape}")
    if G.array.dtype.kind != fmt.kind:
        raise TypeError(f"grad must be of the inputs' kind, got {G.array.dtype.name}")
    return G


def broadcast_to(x: Operand, shape: tuple) -> Operand:
    """``x`` broadcast to ``shape`` (NumPy rules)."""
    if x.array.shape == shape:
        return x
    if x.array.size == 1:  # a scalar or a one-element array: every position is its element's
        return x.take([0] * math.prod(shape), shape)
    strides = row_major_strides(x.array.shape)
    positions = [sum(i * s for i, s in zip(broadcast_index(index, x.array.shape), strides)) for index in iter_indices(shape)]
    return x.take(positions, shape)


def stacked(operands: list, shape: tuple) -> Operand:
    """Operands of one dtype and size, interleaved element by element (element ``i`` of each
    in turn), as one operand of ``shape``."""
    if len({(x.array.dtype, x.array.size) for x in operands}) != 1:
        raise RuntimeError(f"internal error: stacked operands of {', '.join(f'{x.array.dtype.name} {x.array.shape}' for x in operands)}")
    def stacked_bounds(compute, ulps, flush, standard, workers=None):
        found = [exact_bounds(x.array) if x._bounds is None else x._bounds(compute, ulps, flush, standard, workers) for x in operands]
        return Bounds.interleaved(found, shape)

    def stacked_normal(compute, flush, workers=None):
        return interleave_elements([x.normal(compute, flush, workers) for x in operands], shape)

    bounds = stacked_bounds if any(x._bounds is not None for x in operands) else None
    normal = stacked_normal if any(x._normal is not None for x in operands) else None
    return Operand(interleave_elements([x.array for x in operands], shape), bounds, normal)


_STANDARD = {
    "float": [float16, bfloat16, float32, float64],
    "complex": [complex64, complex128],
    "int": [int8, uint8, int16, int32, int64],
}


def holds(f, g) -> bool:
    """Whether format ``f`` represents every value of format ``g`` (of the same kind)."""
    if f == g:
        return True
    if f.kind != g.kind:
        return False
    if f.kind == "complex":
        return holds(f.component, g.component)
    return _native.holds(f._layout, g._layout)


def promoted_format(formats: list[Format]) -> Format:
    """The format type promotion gives operands of these formats (one kind): one of them that
    holds the others, else the narrowest standard format that holds them all (as PyTorch:
    bfloat16 with float16 gives float32)."""
    kinds = {f.kind for f in formats}
    if len(kinds) != 1:
        raise TypeError(f"operands must be of one kind, got {', '.join(f.name for f in formats)}")
    found = holding_format(formats)
    if found is None:
        raise TypeError(f"no standard format holds {', '.join(f.name for f in formats)}")
    return found


def holding_format(formats: list[Format]) -> Format | None:
    """One of ``formats`` (one kind) that holds the others, else the narrowest standard format
    that holds them all; None where none does."""
    return next((f for f in list(formats) + _STANDARD.get(formats[0].kind, []) if all(holds(f, g) for g in formats)), None)


def exact_scalar(value) -> int | FloatValue | ComplexValue:
    """An exact scalar (alpha, beta or a scalar operand) as an int, ``FloatValue`` or
    ``ComplexValue``. Accepted: int, float, complex, a ``Fraction`` whose denominator is a
    power of two, a ``FloatValue`` or ``ComplexValue``, or an array of shape ()."""
    if isinstance(value, Array):
        if value.shape != ():
            raise ValueError(f"a scalar array has shape (), got {value.shape}")
        return value.value(())
    if isinstance(value, (FloatValue, ComplexValue)):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return FloatValue.from_float(value)
    if isinstance(value, complex):
        return ComplexValue.from_complex(value)
    if isinstance(value, Fraction):
        return _float_value(value)
    raise TypeError(f"a scalar must be an int, float, complex, Fraction, FloatValue or array of shape (), got {type(value).__name__}")


def is_unit(a) -> bool:
    """Whether ``a`` is 1 or -1, so ``a * y`` is exact."""
    if isinstance(a, ComplexValue):
        return a.imag.is_zero and is_unit(a.real)
    if isinstance(a, FloatValue):
        return a.is_finite and abs(a.to_fraction()) == 1
    return abs(a) == 1


def _float_value(q: Fraction) -> FloatValue:
    d = q.denominator
    if d & (d - 1):
        raise ValueError(f"{q} is not a binary floating-point value (its denominator is not a power of two)")
    if not q:
        return FloatValue.zero()
    m, e = abs(q.numerator), -(d.bit_length() - 1)
    shift = (m & -m).bit_length() - 1
    return FloatValue("finite", q < 0, m >> shift, e + shift)


def scalar_operand(values: list, kind: str, shape: tuple, like) -> Operand:
    """Exact scalars (``exact_scalar`` values) as an array operand of ``shape`` (row-major)
    of kind ``kind``, in the narrowest format that holds them: ``like`` (the data's format)
    when it does, else a standard format (for integers int8 to int64, then uint64), else a
    wider binary format made for them."""
    if kind == "int":
        values = [_as_int(v) for v in values]
        fmt = next((f for f in [like] + _STANDARD["int"] + [uint64] if f.kind == "int" and all(f.min_value <= v <= f.max_value for v in values)), None)
        if fmt is None:
            outside = next(v for v in values if not -(2**63) <= v <= 2**64 - 1)
            raise OverflowError(f"an integer scalar must lie in [-2**63, 2**64 - 1], got {outside}")
        return Operand(Array.from_values(fmt, shape, values))
    if kind == "complex":
        values = [v if isinstance(v, ComplexValue) else ComplexValue(_as_float(v), FloatValue.zero()) for v in values]
        parts = [p for v in values for p in (v.real, v.imag)]
        component = _holding(parts, like.component if like.kind == "complex" else None)
        fmt = next((f for f in _STANDARD["complex"] if f.component == component), None) or ComplexFormat(f"complex_{component.name}", component)
        return Operand(Array.from_values(fmt, shape, values))
    values = [_as_float(v) for v in values]
    return Operand(Array.from_values(_holding(values, like if like.kind == "float" else None), shape, values))


def repeated_operand(value, kind: str, count: int, like) -> Operand:
    """``count`` copies of one exact scalar as a column (``count`` x 1) in the format
    ``scalar_operand`` gives it, made from one element's bytes."""
    one = scalar_operand([value], kind, (1, 1), like).array
    return Operand(Array(one.dtype, (count, 1), bytes(one.buffer) * count, byteorder=one.byteorder))


def _as_int(v) -> int:
    if isinstance(v, int):
        return v
    raise TypeError(f"integer operands take integer scalars, got {v}")


def _as_float(v) -> FloatValue:
    if isinstance(v, FloatValue):
        return v
    if isinstance(v, int):
        return _float_value(Fraction(v))
    raise TypeError(f"real operands take real scalars, got {v}")


def _holding(values: list, like) -> FloatFormat:
    """The narrowest binary format representing every value: ``like``, a standard format, or one made for them."""
    for f in ([like] if like is not None else []) + _STANDARD["float"]:
        if all(_exact_in(f, v) for v in values):
            return f
    finite = [v for v in values if v.is_finite and not v.is_zero]
    bits = max((v.significand.bit_length() for v in finite), default=1)
    low = min((v.exponent for v in finite), default=0)
    high = max((v.exponent + v.significand.bit_length() for v in finite), default=0)
    e, m = _native.binary_holding(bits, low, high)
    return FloatFormat(f"binary_e{e}m{m}", e, m)


def _exact_in(fmt: FloatFormat, v: FloatValue) -> bool:
    if not v.is_finite:
        return v.is_nan or fmt.special == "ieee"
    return fmt.decode(fmt.encode(v)) == v  # rounds to nearest: equal only when v is representable


def operand(x) -> Operand:
    """An input as an ``Operand``: an array (any source ``asarray`` takes) or a result with one output."""
    if isinstance(x, Operand):
        return x
    if not isinstance(x, Result):
        return Operand(asarray(x))
    rounded = x._single("passed")

    def bounds(compute, ulps, flush, standard, workers=None):
        return x._bounds_of(compute, ulps, flush, standard, workers)

    def normal(compute, flush, workers=None):
        return x.normal(compute, flush_to_zero=flush, workers=workers)

    return Operand(rounded, bounds, normal)


def unflagged_radius(x: Operand, compute, ulps, flush: bool, standard: str, workers=None) -> Array | None:
    """``x.radius``, with inf where a stored value may be non-finite (``marks``): for the
    operations whose exp or division can turn such a value back into a finite one, which
    then give no bound there (attention, matrix_exp)."""
    radius, marks = x.radius(compute, ulps, flush, standard, workers), x.marks(compute, ulps, flush, standard, workers)
    if marks is None:
        return radius
    dtype = complex128 if x.array.dtype.kind == "complex" else float64
    found = [0] * len(marks) if radius is None else radius.encodings()
    if dtype is complex128:
        found = [part for pair in found for part in pair] if radius is not None else found
    infinity = float64.encode(math.inf)
    encodings = [infinity if mark else r for r, mark in zip(found, marks)]
    if dtype is complex128:
        encodings = list(zip(encodings[0::2], encodings[1::2]))
    return Array.from_encodings(dtype, x.array.shape, encodings)
