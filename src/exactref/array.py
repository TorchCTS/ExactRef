"""Array holders: one flat byte buffer plus dtype, shape and byte order.

Elements are stored contiguously in row-major order, so element ``index``
starts at byte ``itemsize * sum(i * stride for i, stride in zip(index, strides))``.
No Python object exists per element until one is read.
"""

from __future__ import annotations

import math
import operator
import sys
from array import array as _pyarray
from collections.abc import Sequence
from typing import TypeVar

from . import _native
from .formats import BoolFormat, ComplexFormat, FloatFormat, Format, IntFormat, conjugated, flip_imaginary_signs, get_format
from .values import ComplexValue, FloatValue


def flat_position(index, shape: tuple) -> int:
    """The row-major position of ``index`` (an int, or a tuple of one per axis) in ``shape``,
    with range checks."""
    if not isinstance(index, tuple):
        index = (index,)
    if len(index) != len(shape):
        raise IndexError(f"shape {shape} needs {len(shape)} indices, got {len(index)}")
    position = 0
    for axis, (i, n) in enumerate(zip(index, shape)):
        i = operator.index(i)
        if not 0 <= i < n:
            raise IndexError(f"index {i} is out of range for axis {axis} with size {n}")
        position = position * n + i
    return position


def row_major_strides(shape) -> tuple[int, ...]:
    """Element strides of a contiguous row-major ``shape``."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)

_BYTEORDERS = ("little", "big")

# Unsigned array-module typecodes by item size, for fast bulk packing.
_TYPECODES: dict[int, str] = {}
for _code in "BHILQ":
    _TYPECODES.setdefault(_pyarray(_code).itemsize, _code)


def _normalize_shape(shape) -> tuple[int, ...]:
    try:
        dims = (operator.index(shape),)
    except TypeError:
        dims = tuple(operator.index(d) for d in shape)
    for d in dims:
        if d < 0:
            raise ValueError(f"shape {dims} has a negative dimension")
    return dims


def _metadata(dtype, shape, byteorder) -> tuple[Format, tuple[int, ...], str]:
    if byteorder not in _BYTEORDERS:
        raise ValueError(f"byteorder must be 'little' or 'big', got {byteorder!r}")
    return get_format(dtype), _normalize_shape(shape), byteorder


def _check_count(shape: tuple[int, ...], count: int, what: str) -> None:
    expected = math.prod(shape)
    if count != expected:
        raise ValueError(f"shape {shape} needs {expected} {what}, got {count}")


def _pack(dtype: Format, encodings: list, byteorder: str) -> bytes:
    if dtype.components > 1:
        encodings = [part for bits in encodings for part in dtype.check_encoding(bits)]
    itemsize = dtype.itemsize // dtype.components
    try:
        code = _TYPECODES.get(itemsize)
        if code is None:
            return b"".join(operator.index(bits).to_bytes(itemsize, byteorder) for bits in encodings)
        packed = _pyarray(code, encodings)
    except OverflowError as error:
        raise ValueError(f"an encoding does not fit in a {8 * itemsize}-bit {dtype.name} element") from error
    if byteorder != sys.byteorder:
        packed.byteswap()
    return packed.tobytes()


def _unpack(data, dtype: Format, byteorder: str) -> list:
    itemsize = dtype.itemsize // dtype.components
    code = _TYPECODES.get(itemsize)
    if code is None:
        parts = [int.from_bytes(data[i : i + itemsize], byteorder) for i in range(0, len(data), itemsize)]
    else:
        unpacked = _pyarray(code)
        unpacked.frombytes(data)
        if byteorder != sys.byteorder:
            unpacked.byteswap()
        parts = unpacked.tolist()
    if dtype.components > 1:
        return list(zip(parts[0::2], parts[1::2]))
    return parts


def _read_layout(dtype: Format):
    """The layout the C reads ``dtype``'s elements by (csrc/reading.c): a float or integer format's, a complex
    format's component's, or None for bool's bytes; NotImplemented for any other Format, which its own codec reads."""
    if isinstance(dtype, ComplexFormat):
        return dtype.component._layout
    if isinstance(dtype, (FloatFormat, IntFormat)):
        return dtype._layout
    return None if isinstance(dtype, BoolFormat) else NotImplemented


def _nonzero(value) -> bool:
    """An element's exact value read as true or false: nonzero, as in PyTorch (NaN is true, a
    zero of either sign false), without rounding to a Python scalar first."""
    if isinstance(value, ComplexValue):
        return not (value.real.is_zero and value.imag.is_zero)
    if isinstance(value, FloatValue):
        return not value.is_zero
    return bool(value)


_A = TypeVar("_A", bound="_ArrayBase")  # an _ArrayBase subclass, as a classmethod builds it (typing.Self needs 3.11)


class _ArrayBase:
    """Read access shared by ``Array`` and ``MutableArray``."""

    __slots__ = ("_dtype", "_shape", "_byteorder", "_data", "_size")
    _storage: type

    def __init__(self, dtype: Format | str, shape, buffer, *, byteorder: str = "little") -> None:
        """Hold ``buffer`` as elements of ``dtype`` in row-major ``shape``.

        The buffer must hold exactly ``itemsize * prod(shape)`` bytes. The array
        owns its storage: any buffer it could not keep immutably is copied.
        """
        dtype, shape, byteorder = _metadata(dtype, shape, byteorder)
        self._setup(dtype, shape, byteorder, self._own(buffer), checked=False)

    @staticmethod
    def _own(buffer):
        raise NotImplementedError

    @classmethod
    def _new(cls, dtype: Format, shape: tuple[int, ...], byteorder: str, data, *, checked: bool = False):
        self = object.__new__(cls)
        self._setup(dtype, shape, byteorder, data, checked=checked)
        return self

    def _setup(self, dtype: Format, shape: tuple[int, ...], byteorder: str, data, *, checked: bool) -> None:
        if not checked:
            size = math.prod(shape)
            if len(data) != size * dtype.itemsize:
                raise ValueError(
                    f"shape {shape} of {dtype.name} needs {size * dtype.itemsize} bytes, got {len(data)}"
                )
            dtype.check_buffer(data)
        self._dtype = dtype
        self._shape = shape
        self._byteorder = byteorder
        self._data = data
        self._size = math.prod(shape)

    @classmethod
    def zeros(cls: type[_A], dtype: Format | str, shape, *, byteorder: str = "little") -> _A:
        """Return an array whose every byte is zero (``+0``, ``0`` or ``False``)."""
        dtype, shape, byteorder = _metadata(dtype, shape, byteorder)
        return cls._new(dtype, shape, byteorder, cls._storage(math.prod(shape) * dtype.itemsize))

    @classmethod
    def from_encodings(cls: type[_A], dtype: Format | str, shape, encodings, *, byteorder: str = "little") -> _A:
        """Build an array from raw encodings in row-major order."""
        dtype, shape, byteorder = _metadata(dtype, shape, byteorder)
        encodings = list(encodings)
        _check_count(shape, len(encodings), "encodings")
        return cls._new(dtype, shape, byteorder, cls._storage(_pack(dtype, encodings, byteorder)))

    @classmethod
    def from_values(cls: type[_A], dtype: Format | str, shape, values, *, byteorder: str = "little", **options) -> _A:
        """Build an array by encoding values in row-major order.

        ``options`` (such as ``rounding``) are passed to ``dtype.encode``.
        """
        dtype, shape, byteorder = _metadata(dtype, shape, byteorder)
        encodings = [dtype.encode(value, **options) for value in values]
        _check_count(shape, len(encodings), "values")
        return cls._new(dtype, shape, byteorder, cls._storage(_pack(dtype, encodings, byteorder)))

    @property
    def dtype(self) -> Format:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def byteorder(self) -> str:
        return self._byteorder

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        """The number of elements."""
        return self._size

    @property
    def nbytes(self) -> int:
        return len(self._data)

    def byte_offset(self, index=()) -> int:
        """Return the byte at which element ``index`` starts."""
        return flat_position(index, self._shape) * self._dtype.itemsize

    def encoding(self, index=()) -> int | tuple[int, int]:
        """Return the raw encoding of one element: an unsigned int, or a pair for complex."""
        start = self.byte_offset(index)
        dtype = self._dtype
        if dtype.components > 1:
            part = dtype.itemsize // dtype.components
            return tuple(
                int.from_bytes(self._data[offset : offset + part], self._byteorder)
                for offset in range(start, start + dtype.itemsize, part)
            )
        return int.from_bytes(self._data[start : start + dtype.itemsize], self._byteorder)

    def value(self, index=()) -> FloatValue | ComplexValue | int | bool:
        """Return the exact value of one element (a ``FloatValue``, ``ComplexValue``, ``int`` or ``bool``)."""
        return self._dtype.decode(self.encoding(index))

    def scalar(self, index=()) -> float | complex | int | bool:
        """Return one element as an ordinary Python ``float``, ``complex``, ``int`` or ``bool``."""
        return self._dtype.to_scalar(self.encoding(index))

    def encodings(self) -> list[int] | list[tuple[int, int]]:
        """Return every raw encoding in row-major order."""
        return _unpack(self._data, self._dtype, self._byteorder)

    def values(self) -> list[FloatValue | ComplexValue | int | bool]:
        """Return every exact value in row-major order."""
        decode = self._dtype.decode
        return [decode(bits) for bits in self.encodings()]

    def scalars(self) -> list[float | complex | int | bool]:
        """Return every element as a Python scalar in row-major order."""
        layout = _read_layout(self._dtype)
        if layout is not NotImplemented:
            found = _native.scalars(bytes(self._data), self._byteorder == "big", layout, self._dtype.components)
            if found is not NotImplemented:  # a float format where a C double is not binary64
                return found
        to_scalar = self._dtype.to_scalar
        return [to_scalar(bits) for bits in self.encodings()]

    def tolist(self) -> list | float | complex | int | bool:
        """Return the elements as nested lists of Python scalars (a bare scalar for shape ``()``)."""
        flat = self.scalars()
        if not self._shape:
            return flat[0]

        def nest(start: int, shape: tuple):
            if len(shape) == 1:
                return flat[start : start + shape[0]]
            step = math.prod(shape[1:])
            return [nest(start + i * step, shape[1:]) for i in range(shape[0])]

        return nest(0, self._shape)

    def all(self) -> bool:
        """Whether every element is true: nonzero, as in PyTorch (NaN is true, a zero of
        either sign false). True for an empty array."""
        if self._dtype.kind == "bool":
            return 0 not in self._data
        return self._nonzero_elements(True)

    def any(self) -> bool:
        """Whether some element is true (nonzero, as ``all`` reads it). False for an empty array."""
        if self._dtype.kind == "bool":
            return any(self._data)
        return self._nonzero_elements(False)

    def _nonzero_elements(self, every: bool) -> bool:
        """Whether every element (``every``) or some element is nonzero, read from the encodings by the C, or for a
        format the C does not read from the values."""
        layout = _read_layout(self._dtype)
        if layout is NotImplemented:
            return (all if every else any)(map(_nonzero, self.values()))
        return _native.nonzero(bytes(self._data), self._byteorder == "big", layout, self._dtype.components, every)

    def __bool__(self) -> bool:
        """A one-element array's truth (as ``all`` reads it); any other array has none and raises
        ValueError, as in PyTorch: check a result with ``assert result.covers(out).all()``."""
        if self._size != 1:
            raise ValueError(f"the truth value of an array of {self._size} elements is ambiguous; use all() or any()")
        return self.any()

    def identical(self, other: _ArrayBase) -> bool:
        """Whether ``other`` has the same dtype, shape and encodings (byte order ignored)."""
        if not isinstance(other, _ArrayBase):
            raise TypeError(f"expected an array, got {type(other).__name__}")
        if self._dtype != other._dtype or self._shape != other._shape:
            return False
        if self._byteorder == other._byteorder:
            return self._data == other._data
        return self.encodings() == other.encodings()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._dtype.name}, shape={self._shape}, byteorder={self._byteorder!r})"


class Array(_ArrayBase):
    """An immutable array, suitable as a reference input.

    Its storage is a ``bytes`` object that no caller can modify, so values
    cannot change during a calculation.
    """

    __slots__ = ()
    _storage = bytes

    @staticmethod
    def _own(buffer) -> bytes:
        if type(buffer) is bytes:
            return buffer
        return memoryview(buffer).tobytes()

    @property
    def buffer(self) -> bytes:
        """The underlying bytes, for bulk routines."""
        return self._data

    def reshape(self, shape) -> Array:
        """Return an array with the same storage and element count but a new shape."""
        shape = _normalize_shape(shape)
        if math.prod(shape) != self._size:
            raise ValueError(f"cannot reshape {self._size} elements of shape {self._shape} into {shape}")
        return Array._new(self._dtype, shape, self._byteorder, self._data, checked=True)

    def mutable_copy(self) -> MutableArray:
        """Return a mutable array with independent storage."""
        return MutableArray._new(self._dtype, self._shape, self._byteorder, bytearray(self._data), checked=True)


class MutableArray(_ArrayBase):
    """An array whose elements can be set, for data being built or edited.

    Writes go through the codec (``set_value``) or checked bulk methods. Call
    ``freeze()`` to obtain an immutable ``Array``.
    """

    __slots__ = ()
    _storage = bytearray

    @staticmethod
    def _own(buffer) -> bytearray:
        return bytearray(memoryview(buffer).tobytes())

    @property
    def buffer(self) -> memoryview:
        """A read-only view of the current bytes, for bulk routines."""
        return memoryview(self._data).toreadonly()

    def set_encoding(self, index, bits) -> None:
        """Store a raw encoding (an int, or a pair for complex) at one element."""
        start = self.byte_offset(index)
        data = _pack(self._dtype, [bits], self._byteorder)
        self._dtype.check_buffer(data)
        self._data[start : start + self._dtype.itemsize] = data

    def set_value(self, index, value, **options) -> None:
        """Encode a value and store it at one element.

        ``options`` (such as ``rounding``) are passed to ``dtype.encode``.
        """
        self.set_encoding(index, self._dtype.encode(value, **options))

    def set_encodings(self, encodings) -> None:
        """Replace every element with raw encodings in row-major order."""
        encodings = list(encodings)
        _check_count(self._shape, len(encodings), "encodings")
        self._replace(_pack(self._dtype, encodings, self._byteorder))

    def set_values(self, values, **options) -> None:
        """Replace every element by encoding values in row-major order."""
        encode = self._dtype.encode
        encodings = [encode(value, **options) for value in values]
        _check_count(self._shape, len(encodings), "values")
        self._replace(_pack(self._dtype, encodings, self._byteorder))

    def set_bytes(self, data) -> None:
        """Replace the whole buffer with bytes of exactly the same length."""
        data = memoryview(data).tobytes()
        if len(data) != len(self._data):
            raise ValueError(f"expected {len(self._data)} bytes, got {len(data)}")
        self._replace(data)

    def _replace(self, data: bytes) -> None:
        self._dtype.check_buffer(data)
        self._data[:] = data

    def freeze(self) -> Array:
        """Return an immutable copy. This array remains usable."""
        return Array._new(self._dtype, self._shape, self._byteorder, bytes(self._data), checked=True)

    def copy(self) -> MutableArray:
        """Return a mutable array with independent storage."""
        return MutableArray._new(self._dtype, self._shape, self._byteorder, bytearray(self._data), checked=True)


def as_parts_layout(array: Array, dtype, shape) -> Array:
    """A complex array as the real array of its parts in turn (``dtype`` the component
    format, ``shape`` ending in 2), or such a real array as the complex one: the same bytes
    for a little-endian array, otherwise re-encoded little-endian."""
    if array.byteorder == "little":
        return Array(dtype, shape, array.buffer)
    encodings = array.encodings()
    if array.dtype.kind == "complex":
        return Array.from_encodings(dtype, shape, [x for pair in encodings for x in pair])
    return Array.from_encodings(dtype, shape, list(zip(encodings[0::2], encodings[1::2])))


def take_elements(array: Array, positions: Sequence[int], shape, conjugate: bool = False) -> Array:
    """The elements of ``array`` at row-major ``positions`` as a little-endian array of
    ``shape``, their bytes copied (complex conjugates with ``conjugate``: each imaginary
    sign bit flipped). ``positions`` of ``range(array.size)`` on a little-endian array,
    unconjugated, shares its bytes instead: pass a range, not a list, for every element."""
    dtype = array.dtype
    if positions == range(array.size) and array.byteorder == "little" and not (conjugate and dtype.kind == "complex"):
        return Array(dtype, shape, array.buffer)  # every element in order: the same bytes
    width = next(w for w in (8, 4, 2, 1) if dtype.itemsize % w == 0)
    if array.byteorder != "little" or width not in _TYPECODES:
        encodings = conjugated(array.encodings(), dtype) if conjugate else array.encodings()
        return Array.from_encodings(dtype, shape, [encodings[p] for p in positions])
    words = dtype.itemsize // width
    flat = memoryview(array.buffer).cast(_TYPECODES[width])
    index = positions if words == 1 else [p * words + i for p in positions for i in range(words)]
    if len(index) > 1:
        data = _pyarray(_TYPECODES[width], operator.itemgetter(*index)(flat)).tobytes()
    else:
        data = _pyarray(_TYPECODES[width], [flat[i] for i in index]).tobytes()
    if conjugate and dtype.kind == "complex":
        flipped = bytearray(data)
        flip_imaginary_signs(flipped, dtype.itemsize)
        data = bytes(flipped)
    return Array(dtype, shape, data)


def interleave_elements(arrays: list, shape) -> Array:
    """Arrays of one dtype and size with their elements in turn (element ``i`` of each), as a
    little-endian array of ``shape``, their bytes copied."""
    dtype, size, count = arrays[0].dtype, arrays[0].size, len(arrays)
    if all(a.byteorder == "little" for a in arrays):
        joined = Array(dtype, (count * size,), b"".join(a.buffer for a in arrays))
        return take_elements(joined, [j * size + i for i in range(size) for j in range(count)], shape)
    encodings = [a.encodings() for a in arrays]
    return Array.from_encodings(dtype, shape, [e[i] for i in range(size) for e in encodings])
