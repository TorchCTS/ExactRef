"""The object every operation definition returns, and the bounds behind its tolerance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Callable, TypeVar

from . import _native
from ._fenv import ieee
from .array import Array, _normalize_shape, as_parts_layout, interleave_elements, row_major_strides, take_elements
from .boundaries import Boundary, leaf_boundaries
from .convert import asarray
from .exact import ExactArray
from .formats import _INT_OVERFLOW_CODES, Format, _part_format, bool_, complex128, conjugated, flip_imaginary_signs, float64, get_format, uint8


class Outputs(Mapping):
    """Named outputs of one operation, as a read-only mapping: ``outputs.grad_A`` or
    ``outputs["grad_A"]``, and ``dict(outputs)``, ``outputs.items()`` and ``len(outputs)``. The names
    are attention's ``output`` and ``logsumexp``, and a backward's ``grad_`` plus its input's name,
    such as ``grad_A`` or ``grad_query``."""

    __slots__ = ("_items",)

    def __init__(self, items: dict) -> None:
        self._items = dict(items)

    def __getattr__(self, name: str):
        items = object.__getattribute__(self, "_items")
        if name in items:
            return items[name]
        raise AttributeError(f"no output {name!r}; outputs are {', '.join(items)}")

    def __getitem__(self, name: str):
        if name not in self._items:
            raise KeyError(f"no output {name!r}; outputs are {', '.join(self._items)}")
        return self._items[name]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        raise ValueError("named outputs have no truth value; check each, as all(x.all() for x in outputs.values())")

    def __repr__(self) -> str:
        return f"Outputs({', '.join(f'{name}={value!r}' for name, value in self._items.items())})"


# The non-finite outputs of one real value; an imaginary part's flags are shifted by IMAG.
PLUS_INF, MINUS_INF, NAN = _native.FLAG_PLUS_INF, _native.FLAG_MINUS_INF, _native.FLAG_NAN
IMAG = 3
ALL = PLUS_INF | MINUS_INF | NAN
# In ``possible`` only: an input reached may have been stored as any finite value while the
# reference used an infinity, so the output may be any finite value too.
ANY = _native.FLAG_ANY
_MAY_BE_FLAGS = {"+inf": PLUS_INF, "-inf": MINUS_INF, "nan": NAN}


_CONJUGATED_FLAGS = bytes(f & ~((PLUS_INF | MINUS_INF) << IMAG) | (f >> 1 & PLUS_INF << IMAG) | (f << 1 & MINUS_INF << IMAG) for f in range(256))


class Bounds:
    """One output's tolerance, finite range and non-finite flags, elements in row-major order:
    ``tolerance`` (float64, complex128 for complex outputs: one bound per part), ``low`` and
    ``high`` (the output's dtype) and ``flags`` (uint8; ``PLUS_INF``, ``MINUS_INF`` and ``NAN``
    for the real or only part, shifted by ``IMAG`` for the imaginary part)."""

    __slots__ = ("tolerance", "low", "high", "flags")

    def __init__(self, tolerance: Array, low: Array, high: Array, flags: Array) -> None:
        self.tolerance, self.low, self.high, self.flags = tolerance, low, high, flags

    def _arrays(self) -> tuple:
        return self.tolerance, self.low, self.high, self.flags

    def reshape(self, shape) -> Bounds:
        return Bounds(*(a.reshape(shape) for a in self._arrays()))

    def take(self, positions: Sequence[int], shape: tuple) -> Bounds:
        """The elements at row-major ``positions``, in ``shape``."""

        return Bounds(*(take_elements(a, positions, shape) for a in self._arrays()))

    def conjugated(self) -> Bounds:
        """The bounds of the complex conjugates: each imaginary part's range negated and its
        infinities' signs swapped; the tolerance does not change. Real bounds are unchanged."""
        if self.low.dtype.kind != "complex":
            return self
        dtype, shape = self.low.dtype, self.low.shape
        flags = Array(uint8, shape, bytes(self.flags.buffer).translate(_CONJUGATED_FLAGS))
        if self.low.byteorder == self.high.byteorder == "little":
            size = dtype.itemsize
            low, high = bytes(self.low.buffer), bytes(self.high.buffer)
            low_c, high_c = bytearray(low), bytearray(high)
            for b in range(size // 2, size):  # each imaginary part: the other end's, negated
                low_c[b::size], high_c[b::size] = high[b::size], low[b::size]
            flip_imaginary_signs(low_c, size)
            flip_imaginary_signs(high_c, size)
            return Bounds(self.tolerance, Array(dtype, shape, bytes(low_c)), Array(dtype, shape, bytes(high_c)), flags)
        low, high = self.low.encodings(), self.high.encodings()
        low_c = conjugated([(lo[0], hi[1]) for lo, hi in zip(low, high)], dtype)
        high_c = conjugated([(hi[0], lo[1]) for lo, hi in zip(low, high)], dtype)
        return Bounds(self.tolerance, Array.from_encodings(dtype, shape, low_c), Array.from_encodings(dtype, shape, high_c), flags)

    def parts(self) -> Bounds:
        """Complex bounds as the bounds of their parts, a real array of shape ``shape + (2,)``."""
        shape = self.low.shape + (2,)
        component = self.low.dtype.component

        flags = bytes(x for f in self.flags.buffer for x in (f & ALL, f >> IMAG & ALL))
        return Bounds(as_parts_layout(self.tolerance, float64, shape), as_parts_layout(self.low, component, shape),
                      as_parts_layout(self.high, component, shape), Array(uint8, shape, flags))

    @staticmethod
    def interleaved(found: list, shape: tuple) -> Bounds:
        """Several equal-length bounds with their elements in turn, in ``shape``."""
        return Bounds(*(interleave_elements([b._arrays()[n] for b in found], shape) for n in range(4)))

    def marks(self, values: Array) -> list | None:
        """Per part (real and imaginary parts in turn for complex arrays), the non-finite values a
        stored value may be (``PLUS_INF``, ``MINUS_INF``, ``NAN``), with ``ANY`` where the value
        (``values``, the reference) is not finite but finite values may have been stored, or
        where the tolerance is inf (nothing bounds the stored value); None when there are none."""
        dtype = self.low.dtype
        if dtype.kind == "int":
            return None
        component = _part_format(dtype)
        marks = _native.marks(component._layout, float64._layout, *_data(values), *_data(self.low), *_data(self.tolerance),
                              bytes(self.flags.buffer), dtype.components)
        return marks if any(marks) else None

    def paired(self, out: Format, shape: tuple) -> Bounds:
        """These bounds of real values, real and imaginary parts in turn, as a complex output's."""

        flags = bytes(self.flags.buffer)
        return Bounds(as_parts_layout(self.tolerance, complex128, shape), as_parts_layout(self.low, out, shape),
                      as_parts_layout(self.high, out, shape), Array(uint8, shape, bytes(f | g << IMAG for f, g in zip(flags[0::2], flags[1::2]))))

    def may_be(self, kind: str) -> Array:
        """Where a correct implementation may output ``kind``; a last axis of 2 (real part,
        imaginary part) for complex outputs."""
        flag = check_kind(kind)
        flags = bytes(self.flags.buffer)  # one flag byte per element
        real = flags.translate(_flag_table(flag, 0))
        if self.low.dtype.kind != "complex":
            return Array(bool_, self.low.shape, real)
        found = bytearray(2 * len(flags))
        found[0::2], found[1::2] = real, flags.translate(_flag_table(flag, IMAG))
        return Array(bool_, self.low.shape + (2,), bytes(found))

    def covers(self, output) -> Array:
        """Where a correct implementation can produce ``output``'s value: a finite value within
        the range, or a non-finite value its flag allows; a last axis of 2 for complex outputs."""
        array = checked_output(output, self.low.dtype, self.low.shape)
        dtype = array.dtype
        component = _part_format(dtype)
        found = _native.covers(component._layout, float64._layout, *_data(array), *_data(self.low), *_data(self.high),
                               bytes(self.flags.buffer), dtype.components)
        return Array(bool_, array.shape + (2,) if dtype.kind == "complex" else array.shape, bytes(found))


def _flag_table(flag: int, shift: int) -> bytes:
    """For ``bytes.translate``: per flag byte, 1 where ``flag`` is set after a right shift by ``shift``, else 0."""
    return bytes(f >> shift & flag != 0 for f in range(256))


def checked_output(output, dtype: Format, shape: tuple) -> Array:
    """An implementation's output as an ``Array``, which must have the result's ``dtype`` and
    ``shape``; ``covers`` checks it before computing any bound."""
    array = asarray(output)
    if array.dtype != dtype:
        raise TypeError(f"output must have the result's dtype {dtype.name}, got {array.dtype.name}")
    if array.shape != shape:
        raise ValueError(f"output must have the result's shape {shape}, got {array.shape}")
    return array


def _data(array: Array) -> tuple[bytes, bool]:
    """An array's bytes and whether they are big-endian, as the C reads them."""
    return bytes(array.buffer), array.byteorder == "big"


_T = TypeVar("_T")
# The bounds behind a tolerance, for tolerance arguments (compute, ulps, flush, standard, workers): a Result's per
# output (dict[str, Bounds]), an Operand's for its one array (Bounds)
BoundsCallback = Callable[[Format | str | None, int | Fraction | None, bool, str, int | None], _T]
# The normal answer for (compute, flush, workers): a Result's per output (dict[str, Array]), an Operand's (Array)
NormalCallback = Callable[[Format | str | None, bool, int | None], _T]


def exact_bounds(values: Array) -> Bounds:
    """The bounds of exact values (integer outputs, or arrays passed as inputs): tolerance 0,
    the range the values themselves, no flags."""
    shape = values.shape
    return Bounds(Array.zeros(complex128 if values.dtype.kind == "complex" else float64, shape), values, values, Array.zeros(uint8, shape))


def integer_bounds(values: Array, overflow: str, intervals) -> Bounds:
    """The bounds of an integer output ``values`` (the reference) whose exact values lie in
    ``intervals``: per element in row-major order ``(lo, hi)``, integers. Each exact value
    becomes an output as ``encode`` makes it with ``overflow``: saturated, wrapped (a range
    crossing a wrap boundary covers every value), or raising, so only the values inside the
    format are outputs."""
    fmt = values.dtype
    code = _INT_OVERFLOW_CODES[overflow]
    tolerance, low, high = [], [], []
    for bits, (lo, hi) in zip(values.encodings(), intervals):
        tol, lo, hi = _native.int_bounds(fmt.itemsize, fmt.signed, code, float64._layout, lo, hi, bits)
        tolerance.append(tol)
        low.append(lo)
        high.append(hi)
    shape = values.shape
    return Bounds(Array.from_encodings(float64, shape, tolerance), Array.from_encodings(fmt, shape, low), Array.from_encodings(fmt, shape, high),
                  Array.zeros(uint8, shape))


def check_standard(standard: str) -> str:
    if standard not in ("normal", "any_order"):
        raise ValueError(f"standard must be 'normal' or 'any_order', got {standard!r}")
    return standard


def check_ulps(ulps) -> Fraction | None:
    """``ulps`` as a Fraction, at least 1/2 as everywhere; None stays None (each operation's default)."""
    if ulps is None:
        return None
    value = Fraction(ulps)
    if value < Fraction(1, 2):
        raise ValueError("ulps must be at least 1/2")
    return value


def check_kind(kind: str) -> int:
    """``may_be``'s kind as its flag bit."""
    flag = _MAY_BE_FLAGS.get(kind)
    if flag is None:
        raise ValueError(f"kind must be one of {', '.join(_MAY_BE_FLAGS)}, got {kind!r}")
    return flag


def named_outputs(output, names) -> dict[str, object]:
    """``{name: output}``, a dict or ``Outputs`` (any mapping), as a dict, each name one of ``names``."""
    if not isinstance(output, Mapping):
        raise TypeError(f"a result with named outputs ({', '.join(names)}) takes {{name: output}}, got {type(output).__name__}")
    found = {name: output[name] for name in output}
    for name in found:
        if name not in names:
            raise KeyError(f"no output {name!r}; outputs are {', '.join(names)}")
    return found


class Result:
    """What an operation definition returns.

    - ``rounded``: the exact answer rounded once to the output format.
    - ``exact``: the exact answer as certified enclosures (``ExactArray``).
    - ``tolerance()``: how far normal accumulation's finite output can be
      from ``rounded`` on these inputs, a proven bound;
      ``tolerance(standard="any_order")`` covers every accumulation order
      that does not overflow.
    - ``finite_range()``, ``may_be()`` and ``covers()``: the lowest and
      highest finite output, the infinities and NaN an implementation may
      produce, and the judgment of an implementation's output.
    - ``normal()``: the output of normal accumulation in index order on
      these inputs, computed when asked for.

    For one output these are the arrays themselves; several outputs, and a
    backward's gradients whatever their number, are named: attention's
    ``output`` and ``logsumexp``, and a backward's ``grad_`` plus its input's
    name, for example ``result.rounded.grad_A``. ``result[name]`` is a result
    holding just that output. A one-output result's output is named by its
    operation: matmul's ``C``, gemm's ``D``, multiply's and add's ``output``,
    sum's ``s``, matrix_exp's ``E`` and matrix_exp_frechet's ``L``. ``guarantee`` states what ``rounded``
    satisfies, and ``boundaries`` lists the operation's required rounding
    boundaries. ``backward(grad)`` runs the operation's backward from the
    same inputs.

    A result with one unnamed output (a forward's; pass a backward's gradient
    as ``result[name]``) can be passed in place of an array to any operation,
    backward or algorithm, except as a tensor-core algorithm's float32 ``A``
    or ``B`` (its stored values need not be tf32 values) and as a held-scalar
    algorithm's scalar of shape () (the chain ``multiply(s, 1, dtype=...)``
    holds it), which raise ValueError. The
    implementation under test stored that output, so each stored value is one
    this result's bounds allow: within its
    tolerance of ``rounded`` (part by part for complex values), or an infinity
    or NaN that ``may_be()`` allows. The later result's ``rounded`` and
    ``exact`` are the exact answer on ``rounded``, rounded once; its bounds
    cover every value these bounds allow, taken with the same arguments: an
    infinite radius leaves no bound in the outputs it reaches, and a stored
    infinity or NaN passes on as their flags (``proofs/foundations.md``,
    Results as inputs); its ``normal()`` works on this ``normal()``; and an
    algorithm's ``run()`` uses ``rounded``.
    """

    # Operations build results from results through four internal methods, their interface to this class: _single (the
    # one output's rounded), _bounds_of (its bounds for tolerance arguments), _with_bounds (this result with other
    # bounds) and _view (this result in another shape).

    def __init__(
        self,
        op: str,
        guarantee: str,
        rounded: dict,
        exact: dict,
        tolerance: BoundsCallback[dict[str, Bounds]],
        backward: Callable[..., Result] | None = None,
        normal: NormalCallback[dict[str, Array]] | None = None,
        named: bool | None = None,
    ) -> None:
        self.op = op
        self.guarantee = guarantee
        self._rounded = dict(rounded)
        self._exact = dict(exact)
        self._tolerance = tolerance
        self._bounds: dict = {}
        self._backward = backward
        self._normal = normal
        self._normals: dict = {}
        self._named = len(self._rounded) > 1 if named is None else named  # outputs by name (Outputs), not the one array

    @property
    def rounded(self) -> Array | Outputs:
        return self._unwrap(self._rounded)

    @property
    def exact(self) -> ExactArray | Outputs:
        return self._unwrap(self._exact)

    @property
    def boundaries(self) -> dict[str, Boundary]:
        return leaf_boundaries(self.op)

    @ieee
    def tolerance(
        self, compute: Format | str | None = None, *, ulps: int | Fraction | None = None, flush_to_zero: bool = False,
        standard: str = "normal", workers: int | None = None,
    ) -> Array | Outputs:
        """Worst-case ``|implementation output - rounded|`` per element.

        The bound covers every implementation that loses precision only at
        the operation's required boundaries:

        - at most one rounding to ``compute`` at each required rounding boundary,
        - accumulation in ``compute``: with ``standard="normal"`` (the
          default), normal accumulation, which adds in index order or in 2 or
          4 SIMD-style lanes, rounds each product and then adds it, and starts
          a new partial sum where adding the next value would overflow; with
          ``standard="any_order"``, any order (any summation tree, with or
          without fused multiply-add) that does not overflow,
        - library functions within ``ulps`` units in the last place of
          ``compute``, and for matrix functions a backward error measured in
          ``ulps`` (see each operation),
        - the final rounding to the output format.

        An output within the normal tolerance is no worse than normal
        accumulation can be. Beyond it but within the any-order tolerance, it
        is technically correct but worse than normal accumulation can be (for
        example from a poor accumulation order). Beyond the any-order
        tolerance, no implementation that rounds only there without
        overflowing could produce it. Matrix functions bound a backward error
        instead, the same under ``standard="normal"`` and ``"any_order"``.

        ``compute`` defaults to float32 for inputs narrower than float32 and to
        the input format otherwise. With ``flush_to_zero``, the implementation
        flushes subnormal values of the compute format to zero, as
        ``torch.set_flush_denormal(True)`` does: such a value may be read as
        zero, and a result that would be one may be produced as zero.

        The result is a float64 array rounded upward, counting only finite
        output values the output format can represent. For complex outputs it
        is complex128 with one bound per part: its real part bounds
        ``|Re(output - rounded)|`` and its imaginary part
        ``|Im(output - rounded)|``. A finite reference gets a finite tolerance
        wherever its finite outputs can be bounded; ``inf`` marks elements with
        no finite bound, and ``nan`` elements whose reference is not finite.
        ``finite_range()`` gives the lowest and highest finite output,
        ``may_be()`` the infinities and NaN an implementation may produce, and
        ``covers()`` judges an implementation's output. Results are kept, so
        repeating a call with the same arguments returns at once. ``workers``
        is the number of threads the bounds are computed on (default: the
        physical cores, as PyTorch picks its own); it changes only the time.
        Each operation's bound is proven in ``proofs/``.
        """
        return self._unwrap({name: b.tolerance for name, b in self._bounds_dict(compute, ulps, flush_to_zero, standard, workers).items()})

    @ieee
    def finite_range(
        self, compute: Format | str | None = None, *, ulps: int | Fraction | None = None, flush_to_zero: bool = False,
        standard: str = "normal", workers: int | None = None,
    ) -> tuple[Array, Array] | Outputs:
        """The lowest and highest finite output of a correct implementation, per element, as
        ``(low, high)`` arrays of the output's dtype; the arguments are ``tolerance()``'s.

        They lie within ``tolerance()`` of ``rounded``, and can be closer on one side. Where
        the reference is an infinity only because a finite exact value rounds past the
        largest finite value, they hold the finite outputs that can still occur. Where no
        finite output can occur, both are NaN, so ``low <= output <= high`` is false.
        """
        return self._unwrap({name: (b.low, b.high) for name, b in self._bounds_dict(compute, ulps, flush_to_zero, standard, workers).items()})

    @ieee
    def may_be(
        self, kind: str, compute: Format | str | None = None, *, ulps: int | Fraction | None = None, flush_to_zero: bool = False,
        standard: str = "normal", workers: int | None = None,
    ) -> Array | Outputs:
        """Where a correct implementation may output ``kind`` (``"+inf"``, ``"-inf"`` or
        ``"nan"``), as a bool array; for complex outputs it has a last axis of 2 (real part,
        imaginary part). The other arguments are ``tolerance()``'s. A reference that is
        not finite counts, and a format without infinity gives NaN past its largest value only
        with ``overflow="nan"`` or a directed rounding away from zero (else its largest value).
        It can be true where no correct implementation overflows (an overflow check uses
        the error bound), never false where one can.
        """
        check_kind(kind)
        return self._unwrap({name: b.may_be(kind) for name, b in self._bounds_dict(compute, ulps, flush_to_zero, standard, workers).items()})

    @ieee
    def covers(
        self, output, compute: Format | str | None = None, *, ulps: int | Fraction | None = None, flush_to_zero: bool = False,
        standard: str = "normal", workers: int | None = None,
    ) -> Array | Outputs:
        """Where a correct implementation can produce ``output``, as a bool array: a finite
        value within ``finite_range()``, or an infinity or NaN that ``may_be()`` allows.

        ``output`` is an implementation's output (anything ``asarray`` takes) with this
        result's dtype and shape; for complex outputs the array has a last axis of 2 (real
        part, imaginary part). ``assert result.covers(output).all()`` checks an
        implementation. The other arguments are ``tolerance()``'s; the standard, the compute
        format and flushing describe the implementations judged correct. A result with
        named outputs takes ``{name: output}`` (a dict, or ``Outputs`` such as an
        implementation's outputs read back) for any of them and returns ``Outputs`` of
        those, as ``AttentionAlgorithm.covers`` does.
        """
        if not self._named and isinstance(output, Mapping):
            raise TypeError("this result has one unnamed output: pass the output array itself, not a mapping")
        outputs = named_outputs(output, self._rounded) if self._named else {name: output for name in self._rounded}
        outputs = {name: checked_output(array, self._rounded[name].dtype, self._rounded[name].shape) for name, array in outputs.items()}
        found = self._bounds_dict(compute, ulps, flush_to_zero, standard, workers)
        covered = {name: found[name].covers(array) for name, array in outputs.items()}
        return Outputs(covered) if self._named else next(iter(covered.values()))

    def _bounds_of(self, compute, ulps, flush_to_zero: bool, standard: str, workers: int | None = None):
        """The bounds of this result's one output (``Bounds``) for these tolerance
        arguments; operations use it to build results from results."""
        (found,) = self._bounds_dict(compute, ulps, flush_to_zero, standard, workers).values()
        return found

    def _bounds_dict(self, compute, ulps, flush_to_zero: bool, standard: str, workers: int | None = None) -> dict:
        """Per output, its bounds (``Bounds``) for these tolerance arguments, kept
        (``workers`` changes only the time, so it is no part of the key)."""
        key = (None if compute is None else get_format(compute), check_ulps(ulps), flush_to_zero, check_standard(standard))
        if key not in self._bounds:
            self._bounds[key] = self._tolerance(compute, ulps, flush_to_zero, standard, workers)
        return self._bounds[key]

    @ieee
    def normal(self, compute: Format | str | None = None, *, flush_to_zero: bool = False, workers: int | None = None) -> Array | Outputs:
        """The output of normal accumulation in index order on these inputs.

        Each product is rounded to ``compute`` and added in index order, each
        sum rounded to ``compute`` (to nearest, ties to even); where adding the
        next value would overflow, it starts a new partial sum, and the partial
        sums are then combined the same way. The result is rounded to the
        output format as ``rounded`` is. ``compute`` and ``flush_to_zero`` are
        as in ``tolerance()``; each operation says what else its normal answer
        fixes. It is computed only when called, and kept.
        """
        return self._unwrap(self._normal_dict(compute, flush_to_zero, workers))

    def _normal_dict(self, compute, flush_to_zero: bool, workers: int | None = None) -> dict:
        if self._normal is None:
            raise NotImplementedError(f"{self.op} has no normal answer")
        key = (None if compute is None else get_format(compute), flush_to_zero)
        if key not in self._normals:
            self._normals[key] = self._normal(compute, flush_to_zero, workers)
        return self._normals[key]

    def __getitem__(self, name: str) -> Result:
        """A result holding just the output ``name``: its ``rounded``, ``exact``, bounds
        (``tolerance()`` and the rest) and ``normal()``."""
        if name not in self._rounded:
            raise KeyError(f"no output {name!r}; outputs are {', '.join(self._rounded)}")

        def tolerance(compute, ulps, flush, standard, workers=None):
            return {name: self._bounds_dict(compute, ulps, flush, standard, workers)[name]}

        def normal(compute, flush, workers=None):
            return {name: self._normal_dict(compute, flush, workers)[name]}

        return Result(self.op, self.guarantee, {name: self._rounded[name]}, {name: self._exact[name]}, tolerance, _view_backward,
                      None if self._normal is None else normal)

    def _with_bounds(self, bounds: BoundsCallback[Bounds]) -> Result:
        """This one-output result with other bounds: ``bounds(compute, ulps, flush, standard,
        workers)`` returns the output's ``Bounds``; operations use it to build results from
        results."""
        (name,) = self._rounded

        def by_name(compute, ulps, flush, standard, workers=None):
            return {name: bounds(compute, ulps, flush, standard, workers)}

        return Result(self.op, self.guarantee, self._rounded, self._exact, by_name, self._backward, self._normal)

    def reshape(self, shape) -> Result:
        """This result's one unnamed output with its elements in row-major order in a new ``shape``.

        Moving elements rounds nothing: ``rounded``, ``exact``, ``tolerance()`` and
        ``normal()`` are this result's, moved the same way, so the view can be passed
        to a later operation in place of an array. A view has no backward of its own:
        call ``backward`` on this result, with the gradient reshaped back."""
        size = self._single().size
        shape = _normalize_shape(shape)
        if math.prod(shape) != size:
            raise ValueError(f"cannot reshape {self._single().shape} to {shape}")
        return self._view(None, shape, _view_backward)

    def transpose(self, i: int, j: int) -> Result:
        """This result's one unnamed output with dimensions ``i`` and ``j`` swapped, as
        ``reshape`` describes for views; a 0-d result's transpose is the result itself, its
        backward included, as PyTorch transposes a 0-d tensor's dimension 0 with itself."""
        shape = self._single().shape
        ndim = max(len(shape), 1)  # a 0-d result has dimension 0 (or -1), as in PyTorch
        for d in (i, j):
            if not -ndim <= d < ndim:
                raise ValueError(f"dimension {d} is out of range for {len(shape)} dimensions")
        if not shape:
            return self  # PyTorch transposes a 0-d tensor's dimension 0 with itself
        axes = list(range(len(shape)))
        axes[i], axes[j] = axes[j], axes[i]
        new_shape = tuple(shape[a] for a in axes)
        strides = row_major_strides(shape)
        # the source position of each element of new_shape, row-major: built one axis at a time
        positions = [0]
        for t, size in enumerate(new_shape):
            offsets = [k * strides[axes[t]] for k in range(size)]
            positions = [p + d for p in positions for d in offsets]
        return self._view(positions, new_shape, _view_backward)

    def _unwrap(self, items: dict):
        """Per output ``items`` as this result gives them: ``Outputs`` by name, or the one value."""
        return Outputs(items) if self._named else next(iter(items.values()))

    def _single(self, use: str = "viewed") -> Array:
        """The one output's ``rounded``; a result with named outputs raises, as it is ``use``
        one output at a time. Operations use it to take a result in place of an array."""
        if self._named:
            raise TypeError(f"a result with named outputs ({', '.join(self._rounded)}) is {use} one output at a time, as result[name]")
        return self.rounded

    def _view(self, positions: list | None, shape: tuple, backward: Callable[..., Result] | None) -> Result:
        """This one-output result with the elements at row-major ``positions`` in ``shape``, or
        with every element in order, reshaped, for None, and ``backward`` as its backward;
        operations use it to give a result its output's shape."""
        (name,) = self._rounded

        def moved(array):
            return array.reshape(shape) if positions is None else take_elements(array, positions, shape)

        def moved_parts(x):  # Bounds or ExactArray
            return x.reshape(shape) if positions is None else x.take(positions, shape)

        def tolerance(compute, ulps, flush, standard, workers=None):
            return {name: moved_parts(self._bounds_dict(compute, ulps, flush, standard, workers)[name])}

        def normal(compute, flush, workers=None):
            return {name: moved(self._normal_dict(compute, flush, workers)[name])}

        return Result(self.op, self.guarantee, {name: moved(self._rounded[name])}, {name: moved_parts(self._exact[name])},
                      tolerance, backward, None if self._normal is None else normal)

    @ieee
    def backward(self, grad, **options) -> Result:
        """The operation's backward for these inputs and the output gradient ``grad``.

        The same as calling the operation's standalone ``.backward`` with this
        result's inputs; ``options`` are its keyword options (such as ``dtype``
        and ``rounding``), with the same defaults.
        """
        if self._backward is None:
            raise NotImplementedError(f"{self.op} has no backward")
        return self._backward(grad, **options)

    def __dlpack_device__(self):
        """A result is not an array: ``asarray`` (which reads arrays through DLPack) raises here,
        and so does any other DLPack reader."""
        raise TypeError("a result is not exact data; pass result.rounded to use a result's values as exact")

    def __dlpack__(self, *args, **kwargs):
        """As ``__dlpack_device__``."""
        raise TypeError("a result is not exact data; pass result.rounded to use a result's values as exact")

    def __repr__(self) -> str:
        return f"Result({self.op}, guarantee={self.guarantee!r}, rounded={self.rounded!r})"


def _view_backward(grad, **options):
    """The backward of a view of a result (``reshape``, ``transpose`` or ``result[name]``), which has none of its own."""
    raise NotImplementedError("a view of a result (reshape, transpose or result[name]) has no backward; call backward on the "
                              "original result, with the gradient shaped back")
