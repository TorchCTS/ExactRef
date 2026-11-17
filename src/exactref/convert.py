"""Importing data into arrays, and converting arrays between formats (tf32 included)."""

from __future__ import annotations

import sys
from fractions import Fraction

from . import _native, formats
from ._dlpack import from_dlpack
from ._fenv import ieee
from .array import Array, MutableArray, _nonzero
from .formats import (FLOAT_OVERFLOW, INT_OVERFLOW, Format, Rounding, UnsupportedFormatError, _check_overflow, _check_rounding, _float_codes,
                      _layout, _part_format, get_format)
from .values import ComplexValue, FloatValue

_SIGNED = {1: formats.int8, 2: formats.int16, 4: formats.int32, 8: formats.int64}
_UNSIGNED = {1: formats.uint8, 2: formats.uint16, 4: formats.uint32, 8: formats.uint64}
_FLOATS = {("e", 2): formats.float16, ("f", 4): formats.float32, ("d", 8): formats.float64}
_COMPLEX = {("Ze", 4): formats.complex32, ("Zf", 8): formats.complex64, ("Zd", 16): formats.complex128}
_SCALARS = (bool, int, float, complex, Fraction, FloatValue, ComplexValue)
# TF32: float32's exponent with the top 10 of its 23 fraction bits.
_TF32 = _layout(8, 10)


@ieee
def asarray(obj, dtype: Format | str | None = None, *, rounding: Rounding | None = None, overflow: str | None = None) -> Array:
    """Import data as an immutable ``Array``.

    Accepted sources:

    - ``Array`` (returned as is) and ``MutableArray`` (frozen copy).
    - Python scalars and nested lists or tuples. Without ``dtype``, bools
      become ``bool``, ints ``int64``, floats ``float64`` and complex numbers
      ``complex128``; a value the inferred format cannot hold exactly raises.
      ``Fraction``, ``FloatValue`` and ``ComplexValue`` need an explicit ``dtype``.
    - Objects with the buffer protocol whose element format is known: NumPy
      arrays, ``array.array``, ``memoryview``, and ``bytes`` (as ``uint8``).
    - CPU tensors with ``__dlpack__`` (PyTorch, JAX, NumPy), including bfloat16
      and float8. A PyTorch tensor's lazy conjugate and negative views are
      applied first, and one that requires grad is detached.

    Data that carries its own format is imported exactly. When ``dtype``
    names a different format, values are converted with ``rounding`` and
    ``overflow``. Following PyTorch, converting array data to an integer
    dtype wraps on overflow by default (like ``tensor.to``), while Python
    values that do not fit raise (like ``torch.tensor``); other defaults are
    each format's own: in a format without infinity, a value past the largest
    rounds within the format's values (``FloatFormat.encode``). Converting to
    ``bool`` reads every nonzero value, NaN included, as True, as PyTorch does.
    Converting a float that is not a whole number to an integer dtype needs
    ``rounding`` (ValueError without it). To reinterpret raw bytes instead, use
    ``Array(dtype, shape, buffer)``.
    """
    target = None if dtype is None else get_format(dtype)
    # the modes, checked before any value is read
    if rounding is not None:
        _check_rounding(rounding)
    if overflow is not None and target is not None and target.kind != "bool":
        _check_overflow(overflow, INT_OVERFLOW if target.kind == "int" else FLOAT_OVERFLOW, target)
    options = {}
    if rounding is not None:
        options["rounding"] = rounding
    if overflow is not None:
        options["overflow"] = overflow

    if isinstance(obj, MutableArray):
        obj = obj.freeze()
    if isinstance(obj, Array):
        return obj if target in (None, obj.dtype) else convert(obj, target, **options)
    if isinstance(obj, (list, tuple)) or isinstance(obj, _SCALARS):
        return _from_python(obj, target, options)
    obj = _resolved(obj)
    imported = _from_buffer(obj)
    if imported is None and hasattr(obj, "__dlpack__"):
        imported = from_dlpack(obj)
    if imported is None:
        raise TypeError(f"cannot import {type(obj).__name__} as an array")
    return imported if target in (None, imported.dtype) else convert(imported, target, **options)


@ieee
def convert(array: Array, dtype: Format | str, **options) -> Array:
    """Return ``array``'s values encoded in another format (rounding as needed).

    Integer targets wrap on overflow unless ``overflow`` says otherwise, as PyTorch's casts do.
    """
    dtype = get_format(dtype)
    if dtype.kind == "int":
        options.setdefault("overflow", "wrap")
    if (array.dtype.kind, dtype.kind) in _RECODED and array.byteorder == "little" and _default_modes(dtype, options):
        data = _native.recode(bytes(array.buffer), False, _part_format(array.dtype)._layout, _part_format(dtype)._layout)
        return Array(dtype, array.shape, data)
    values = array.values()
    if dtype.kind == "bool":
        values = [_nonzero(v) for v in values]  # as PyTorch's casts: nonzero, NaN included, is True
    return Array.from_values(dtype, array.shape, values, byteorder=array.byteorder, **options)


# Conversions the C re-encodes in one pass, by source and target kind, at the default modes: integer targets wrap,
# float and complex ones round to nearest even with IEEE overflow (csrc/reading.c's part_write)
_RECODED = {("int", "int"), ("int", "float"), ("float", "float"), ("complex", "complex")}


def _default_modes(dtype: Format, options: dict) -> bool:
    """Whether ``options`` are the default modes of a conversion to ``dtype``, the ones ``_native.recode``
    applies (an integer target's rounding has no effect on integer values)."""
    if dtype.kind == "int":
        return options.get("overflow") == "wrap"
    return options.get("rounding", Rounding.NEAREST_EVEN) is Rounding.NEAREST_EVEN and options.get("overflow", "ieee") == "ieee"


@ieee
def round_to_tf32(x, rounding: Rounding, *, overflow: str = "ieee") -> Array:
    """Round float32 data to TF32 values, as NVIDIA's tensor cores read float32 inputs.

    ``x`` is float32 data in any form ``asarray`` accepts; Python values become float64
    there, so pass them as ``er.round_to_tf32(er.asarray(values, "float32"), rounding)``.
    The result is a float32 ``Array`` whose every value is a TF32 value: 11 significant bits
    in float32's exponent range, with subnormals on the grid of 2**-136 (NVIDIA leaves the
    handling of subnormal inputs in its mma instructions unspecified). NaN stays NaN, and
    infinities and signed zeros stay.

    ``rounding`` is required, since kernels differ: PTX's ``cvt.rna.tf32.f32`` rounds to
    nearest with ties away from zero (``NEAREST_AWAY``), ``cvt.rn`` and ``cvt.rz`` (sm_90
    and later) round to nearest even and toward zero, and ``wgmma`` truncates the low 13
    bits of float32 inputs (``TOWARD_ZERO``). Past the largest TF32 value,
    (2 - 2**-10) * 2**127, ``overflow`` applies as for a format: "ieee" gives infinity
    where the rounding is to nearest or outward and the largest value otherwise,
    "saturate" always gives the largest value (as PTX's ``.satfinite``), and "nan" acts as
    "ieee", since TF32 has infinities.
    """
    float32 = formats.float32
    codes = _float_codes(float32, rounding, overflow)
    array = asarray(x)
    if array.dtype is not float32:
        raise TypeError(f"round_to_tf32 takes float32 data, got {array.dtype.name}")
    data = _native.float_narrow(bytes(array.buffer), array.byteorder == "big", float32._layout, _TF32, *codes)
    return Array._new(float32, array.shape, "little", data, checked=True)


def _resolved(obj):
    """A tensor ready for export: one that requires grad detached, and lazy conjugate and
    negative views applied (PyTorch's DLPack export ignores the negative bit, which would
    flip signs). Attributes only, so nothing here needs PyTorch."""
    if getattr(obj, "requires_grad", False) and hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "resolve_conj") and hasattr(obj, "resolve_neg"):
        obj = obj.resolve_conj().resolve_neg()
    return obj


def _from_buffer(obj) -> Array | None:
    try:
        view = memoryview(obj)
    except TypeError:
        return None
    with view:
        dtype, byteorder = _buffer_format(view.format, view.itemsize)
        if dtype is None:
            if hasattr(obj, "__dlpack__"):
                return None
            raise UnsupportedFormatError(
                f"buffer element format {view.format!r} is not supported; "
                "use Array(dtype, shape, buffer) to reinterpret raw bytes"
            )
        shape = view.shape
        return Array(dtype, shape, view.tobytes(), byteorder=byteorder)


def _buffer_format(code: str, itemsize: int) -> tuple[Format | None, str]:
    """Map a struct-style buffer format to a dtype and byte order."""
    byteorder = sys.byteorder
    if code[:1] in ("@", "=", "<", ">", "!"):
        byteorder = {"<": "little", ">": "big", "!": "big"}.get(code[0], sys.byteorder)
        code = code[1:]
    if code == "?" and itemsize == 1:
        return formats.bool_, byteorder
    if len(code) == 1 and code in "bhilqn":
        return _SIGNED.get(itemsize), byteorder
    if len(code) == 1 and code in "BHILQN":
        return _UNSIGNED.get(itemsize), byteorder
    return _FLOATS.get((code, itemsize)) or _COMPLEX.get((code, itemsize)), byteorder


def _from_python(obj, target: Format | None, options: dict) -> Array:
    shape, flat = _flatten(obj)
    if target is None:
        target = _infer(flat)
        encoded = [target.encode(v) for v in flat]
        decoded = [target.decode(bits) for bits in encoded]
        for value, exact in zip(flat, decoded):
            if not _same_value(value, exact):
                raise ValueError(f"{value!r} is not exactly representable in the inferred {target.name}; pass dtype=")
        return Array.from_encodings(target, shape, encoded)
    if target.kind == "bool":
        flat = [_nonzero(v) for v in flat]
    return Array.from_values(target, shape, flat, **options)


def _flatten(obj) -> tuple[tuple[int, ...], list]:
    if not isinstance(obj, (list, tuple)):
        return (), [obj]
    if not obj:
        return (0,), []
    parts = [_flatten(item) for item in obj]
    inner = parts[0][0]
    if any(shape != inner for shape, _ in parts):
        raise ValueError("nested sequences must be rectangular")
    return (len(obj),) + inner, [value for _, values in parts for value in values]


def _infer(flat: list) -> Format:
    if not flat:
        return formats.float64
    kinds = set()
    for value in flat:
        if isinstance(value, bool):
            kinds.add("bool")
        elif isinstance(value, int):
            kinds.add("int")
        elif isinstance(value, float):
            kinds.add("float")
        elif isinstance(value, complex):
            kinds.add("complex")
        elif isinstance(value, (Fraction, FloatValue, ComplexValue)):
            raise TypeError(f"{type(value).__name__} values need an explicit dtype")
        else:
            raise TypeError(f"cannot import {type(value).__name__} values")
    if kinds == {"bool"}:
        return formats.bool_
    if "bool" in kinds:
        raise TypeError("cannot mix bools with numbers without an explicit dtype")
    if "complex" in kinds:
        return formats.complex128
    if "float" in kinds:
        return formats.float64
    return formats.int64


def _same_value(original, decoded) -> bool:
    if isinstance(original, (bool, int)) and not isinstance(decoded, (FloatValue, ComplexValue)):
        return original == decoded
    if isinstance(original, complex):
        return decoded == ComplexValue.from_complex(original)
    if isinstance(decoded, ComplexValue):
        return decoded.imag.is_zero and _same_value(original, decoded.real)
    if isinstance(original, float):
        return decoded == FloatValue.from_float(original)
    return decoded.is_finite and decoded.to_fraction() == original

