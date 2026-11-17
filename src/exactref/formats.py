"""Element formats: codecs between raw encodings and values.

A format describes one element. It converts between an unsigned integer
encoding (``0x3F80`` for BF16 1.0) and a value, and it knows nothing about
byte order or array layout; the array holder handles those. A complex
element's encoding is a ``(real_bits, imag_bits)`` pair.
"""

from __future__ import annotations

import abc
import enum
import operator
from dataclasses import dataclass, field
from fractions import Fraction

from . import _native
from .values import _KINDS, ComplexValue, FloatValue

# Exact values are stored with magnitudes between 2**-LIMIT and 2**LIMIT (exact.py); every
# format accepted here lies inside that range.
LIMIT = _native.LIMIT


class UnsupportedFormatError(ValueError):
    """Raised for an unknown format name or a format this library cannot hold."""


class Rounding(enum.Enum):
    """IEEE 754 rounding-direction attributes."""

    NEAREST_EVEN = "nearest_even"
    NEAREST_AWAY = "nearest_away"
    TOWARD_ZERO = "toward_zero"
    TOWARD_POSITIVE = "toward_positive"
    TOWARD_NEGATIVE = "toward_negative"

    @property
    def is_nearest(self) -> bool:
        """Whether this mode rounds to nearest (ties to even or away)."""
        return self is Rounding.NEAREST_EVEN or self is Rounding.NEAREST_AWAY


FLOAT_OVERFLOW = ("ieee", "saturate", "nan")
INT_OVERFLOW = ("error", "wrap", "saturate")

# The codes of rounding modes, overflow modes, special-value schemes and value kinds in the C
# (csrc/formats.c): their positions in these orders.
_ROUNDING_CODES = {rounding: code for code, rounding in enumerate(Rounding)}
_FLOAT_OVERFLOW_CODES = {mode: code for code, mode in enumerate(FLOAT_OVERFLOW)}
_INT_OVERFLOW_CODES = {mode: code for code, mode in enumerate(INT_OVERFLOW)}
_KIND_CODES = {kind: code for code, kind in enumerate(_KINDS)}
_ZERO, _INFINITY, _NAN, _MAX_FINITE = range(4)  # the C's special encodings


class Format(abc.ABC):
    """How one element's encoding maps to a value.

    Subclasses provide ``name``, ``itemsize`` (bytes per element), ``kind``
    (``"float"``, ``"int"``, ``"bool"`` or ``"complex"``) and ``components``
    (encodings per element: 1, or 2 for complex).
    """

    name: str
    itemsize: int
    kind = "abstract"
    components = 1

    @abc.abstractmethod
    def decode(self, bits):
        """Return the exact value of an encoding."""

    @abc.abstractmethod
    def encode(self, value):
        """Return the encoding of a value, or raise if it cannot be encoded."""

    @abc.abstractmethod
    def to_scalar(self, bits):
        """Return an ordinary Python scalar for an encoding."""

    def check_encoding(self, bits) -> int | tuple[int, int]:
        """Return ``bits`` as an int (a pair of ints for a complex format), or raise if it is not
        a valid encoding."""
        bits = operator.index(bits)
        if not 0 <= bits < 1 << (8 * self.itemsize):
            raise ValueError(f"{bits:#x} is not a {8 * self.itemsize}-bit {self.name} encoding")
        return bits

    def check_buffer(self, data: bytes) -> None:
        """Raise if a buffer holds an invalid encoding. Every bit pattern is valid by default."""


def _check_rounding(rounding) -> None:
    if not isinstance(rounding, Rounding):
        raise TypeError(f"rounding must be a Rounding, got {rounding!r}")


def _check_overflow(overflow, allowed: tuple[str, ...], fmt: Format) -> None:
    if overflow not in allowed:
        raise ValueError(f"overflow for {fmt.name} must be one of {allowed}, got {overflow!r}")


_SPECIALS = ("ieee", "fn", "fnuz")


def _layout(exponent_bits: int, mantissa_bits: int, special: str = "ieee", bias: int | None = None, nan_encodings=None) -> tuple:
    """A float format's definition as the C reads it (a bias of None is the default)."""
    nan = nan_encodings or (None, None)
    return exponent_bits, mantissa_bits, _SPECIALS.index(special), bias, nan[0], nan[1]


def _float_codes(fmt: Format, rounding, overflow: str) -> tuple[int, int]:
    try:
        return _ROUNDING_CODES[rounding], _FLOAT_OVERFLOW_CODES[overflow]
    except (KeyError, TypeError):
        _check_rounding(rounding)
        _check_overflow(overflow, FLOAT_OVERFLOW, fmt)
        raise


@dataclass(frozen=True)
class FloatFormat(Format):
    """A signed binary floating-point format.

    ``special`` selects how non-finite values are encoded:

    - ``"ieee"``: an all-ones exponent encodes infinity (zero mantissa) or NaN.
    - ``"fn"``: finite only; all-ones exponent and mantissa encode NaN
      (as in ``float8_e4m3fn``).
    - ``"fnuz"``: finite only, no negative zero; the negative-zero encoding is
      the only NaN (as in ``float8_e4m3fnuz``).

    ``bias`` defaults to ``2**(e-1) - 1``, or ``2**(e-1)`` for ``"fnuz"``.
    ``nan_encodings`` gives the encodings of positive and negative NaN when
    they differ from the quiet NaN with the value's sign. Elements take a
    whole number of bytes, at most 8.
    """

    name: str
    exponent_bits: int
    mantissa_bits: int
    special: str = "ieee"
    bias: int | None = None  # None: the default bias, which the format then holds
    nan_encodings: tuple[int, int] | None = None
    # derived in __post_init__
    _emin: int = field(init=False, repr=False, compare=False)
    _emax: int = field(init=False, repr=False, compare=False)
    _sign_bit: int = field(init=False, repr=False, compare=False)
    _layout: tuple = field(init=False, repr=False, compare=False)  # the format as the C takes it

    kind = "float"

    def __post_init__(self) -> None:
        e, m = self.exponent_bits, self.mantissa_bits
        if self.special not in _SPECIALS:
            raise UnsupportedFormatError(f"special must be one of {_SPECIALS}, got {self.special!r}")
        derived = _native.float_layout(e, m, _SPECIALS.index(self.special), self.bias)
        if derived == "bits":
            raise UnsupportedFormatError(f"{self.name}: need at least 2 exponent bits and 1 mantissa bit")
        if derived == "bytes":
            raise UnsupportedFormatError(f"{self.name}: {1 + e + m}-bit elements are not a whole number of bytes")
        if derived == "size":
            raise UnsupportedFormatError(f"{self.name}: {1 + e + m}-bit elements are wider than 64 bits")
        # Exact values beyond 2**+-LIMIT are stored only as too big or too small, so every
        # value of the format, and the points where rounding to it overflows or underflows,
        # must lie inside that range.
        if derived == "range":
            raise UnsupportedFormatError(f"{self.name}: its range exceeds what exact values hold (2**-{LIMIT} to 2**{LIMIT})")
        bias, itemsize, emin, emax, sign_bit = derived
        for attribute, value in (("bias", bias), ("itemsize", itemsize), ("_emin", emin), ("_emax", emax), ("_sign_bit", sign_bit),
                                 ("_layout", _layout(e, m, self.special, bias, self.nan_encodings))):
            object.__setattr__(self, attribute, value)
        if self.nan_encodings is not None and not all(self.decode(bits).is_nan for bits in self.nan_encodings):
            raise UnsupportedFormatError(f"{self.name}: nan_encodings must both decode as NaN")

    @property
    def precision(self) -> int:
        """Significand bits, including the implicit leading bit."""
        return self.mantissa_bits + 1

    @property
    def emin(self) -> int:
        """Exponent of the smallest normal value (``2**emin``)."""
        return self._emin

    @property
    def emax(self) -> int:
        """Exponent of the largest finite binade (``2**emax <= max_finite``)."""
        return self._emax

    @property
    def has_infinity(self) -> bool:
        return self.special == "ieee"

    @property
    def has_negative_zero(self) -> bool:
        return self.special != "fnuz"

    @property
    def max_finite(self) -> FloatValue:
        return self.decode(_native.float_special(self._layout, _MAX_FINITE, False))

    def decode(self, bits: int) -> FloatValue:
        kind, negative, significand, exponent = _native.float_decode(self._layout, self.check_encoding(bits))
        return FloatValue(_KINDS[kind], negative, significand, exponent)

    def encode(
        self,
        value: FloatValue | Fraction | int | float,
        *,
        rounding: Rounding = Rounding.NEAREST_EVEN,
        overflow: str = "ieee",
    ) -> int:
        """Round a value to this format and return its encoding.

        ``overflow="ieee"`` (the default) follows IEEE 754 in a format with
        infinity: a result rounding past the largest finite value becomes
        infinity, or the largest finite value when the rounding direction points
        toward zero. A format without infinity has no value past its largest, so
        the result rounds within the values it has: the largest finite value, or
        NaN when a directed rounding points away from zero, where the format has
        no value. In a format without infinity, ``overflow="nan"`` instead gives
        NaN wherever IEEE 754 would give infinity (to nearest too), as kernels
        that keep the IEEE rule in such a format do; in a format with infinity it
        changes nothing. ``overflow="saturate"`` replaces every
        infinity this format cannot hold, or that overflow would produce, with
        the largest finite value. An infinite value gives NaN in a format
        without infinity, except with ``"saturate"``.

        NaN encodes as this format's canonical NaN: the quiet NaN with the
        value's sign, or the encoding ``nan_encodings`` gives (PyTorch's
        bfloat16 NaN is always ``0x7FC0``; its float8_e5m2 NaN has every
        mantissa bit set).
        """
        rounding_code, overflow_code = _float_codes(self, rounding, overflow)
        if isinstance(value, float):
            value = FloatValue.from_float(value)
        if isinstance(value, FloatValue):
            kind, negative, m, e = _KIND_CODES[value.kind], value.negative, value.significand, value.exponent
            return _native.float_encode(self._layout, kind, negative, m, e, rounding_code, overflow_code)
        if isinstance(value, int):
            return _native.float_encode(self._layout, 0, value < 0, abs(value), 0, rounding_code, overflow_code)
        if isinstance(value, Fraction):
            negative, a, b = value < 0, abs(value.numerator), value.denominator
            return _native.float_encode_ratio(self._layout, negative, a, b, rounding_code, overflow_code)
        raise TypeError(f"cannot encode {type(value).__name__} as {self.name}")

    def to_scalar(self, bits: int) -> float:
        """Return the nearest Python float (exact for formats no wider than float64)."""
        return float(self.decode(bits))

    def scaled_value(self, bits: int) -> tuple[int, int] | None:
        """The value of an encoding as ``(m, e)`` meaning ``m * 2**e``, or None when it is
        not finite (the same value as ``decode``, without a value object)."""
        kind, negative, m, e = _native.float_decode(self._layout, bits)
        if kind:
            return None
        return (-m if negative else m), e

    def encode_scaled(self, m: int, e: int, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str = "ieee") -> int:
        """``encode(m * 2**e)`` for an integer ``m``, with the same result (+0 for ``m`` 0)."""
        rounding_code, overflow_code = _float_codes(self, rounding, overflow)
        return _native.float_encode(self._layout, 0, m < 0, abs(m), e, rounding_code, overflow_code)


def _exact_rational(value, fmt: Format) -> Fraction:
    """Return the exact rational value of a finite scalar, or raise."""
    if isinstance(value, float):
        value = FloatValue.from_float(value)
    if isinstance(value, FloatValue):
        if not value.is_finite:
            raise ValueError(f"cannot convert {value.kind} to {fmt.name}")
        return value.to_fraction()
    if isinstance(value, Fraction):
        return value
    try:
        return Fraction(operator.index(value))
    except TypeError:
        raise TypeError(f"cannot encode {type(value).__name__} as {fmt.name}") from None


@dataclass(frozen=True)
class IntFormat(Format):
    """A two's-complement (``signed``) or unsigned integer format of ``itemsize`` bytes, 1 to 8."""

    name: str
    itemsize: int
    signed: bool
    # derived in __post_init__
    _layout: tuple = field(init=False, repr=False, compare=False)  # (itemsize, signed), as the C takes it
    min_value: int = field(init=False, repr=False, compare=False)
    max_value: int = field(init=False, repr=False, compare=False)

    kind = "int"

    def __post_init__(self) -> None:
        if self.itemsize < 1:
            raise UnsupportedFormatError(f"{self.name}: itemsize must be at least 1")
        found = _native.int_range(self.itemsize, self.signed)
        if found == "size":
            raise UnsupportedFormatError(f"{self.name}: {8 * self.itemsize}-bit elements are wider than 64 bits")
        lo, hi = found
        object.__setattr__(self, "_layout", (self.itemsize, self.signed))
        object.__setattr__(self, "min_value", lo)
        object.__setattr__(self, "max_value", hi)

    def decode(self, bits: int) -> int:
        return _native.int_decode(self.check_encoding(bits), self.itemsize, self.signed)

    def encode(self, value, *, rounding: Rounding | None = None, overflow: str = "error") -> int:
        """Return the encoding of an integer.

        A value that is not an integer is accepted only with an explicit
        ``rounding``. NaN and infinities always raise. ``overflow`` is
        ``"error"`` (raise ``OverflowError``), ``"wrap"`` (two's-complement
        modular wrap) or ``"saturate"`` (clamp to the range).
        """
        _check_overflow(overflow, INT_OVERFLOW, self)
        if rounding is not None:
            _check_rounding(rounding)
        q = _exact_rational(value, self)
        if q.denominator == 1:
            n = q.numerator
        elif rounding is None:
            raise ValueError(f"{q} is not an integer; pass rounding= to convert it to {self.name}")
        else:
            n = _native.round_integer(q.numerator, q.denominator, _ROUNDING_CODES[rounding])
        bits = _native.int_encode(n, self.itemsize, self.signed, _INT_OVERFLOW_CODES[overflow])
        if bits is None:
            raise OverflowError(f"{n} is outside the {self.name} range [{self.min_value}, {self.max_value}]")
        return bits

    def to_scalar(self, bits: int) -> int:
        return self.decode(bits)


@dataclass(frozen=True)
class BoolFormat(Format):
    """One byte holding 0 (False) or 1 (True). Other byte values are invalid."""

    name: str = "bool"
    itemsize: int = 1

    kind = "bool"

    def check_encoding(self, bits: int) -> int:
        bits = operator.index(bits)
        if bits not in (0, 1):
            raise ValueError(f"{bits:#x} is not a bool encoding (0 or 1)")
        return bits

    def check_buffer(self, data: bytes) -> None:
        if data.translate(None, b"\x00\x01"):
            raise ValueError("bool buffer contains a byte other than 0 or 1")

    def decode(self, bits: int) -> bool:
        return self.check_encoding(bits) == 1

    def encode(self, value: bool, *, rounding: Rounding | None = None, overflow: str | None = None) -> int:
        """``value``'s encoding. ``rounding`` and ``overflow`` are taken as every format's encode takes
        them, and unused: True and False are exact."""
        if type(value) is not bool:
            raise TypeError(f"bool encodes only True or False, got {type(value).__name__}")
        return int(value)

    def to_scalar(self, bits: int) -> bool:
        return self.decode(bits)


@dataclass(frozen=True)
class ComplexFormat(Format):
    """A complex format: real then imaginary part, each in ``component`` format."""

    name: str
    component: FloatFormat

    kind = "complex"
    components = 2

    def __post_init__(self) -> None:
        if not isinstance(self.component, FloatFormat):
            raise UnsupportedFormatError(f"{self.name}: component must be a FloatFormat")
        object.__setattr__(self, "itemsize", 2 * self.component.itemsize)

    def check_encoding(self, bits) -> tuple[int, int]:
        if not isinstance(bits, tuple):
            raise TypeError(f"a {self.name} encoding is a (real_bits, imag_bits) tuple, got {bits!r}")
        if len(bits) != 2:
            raise ValueError(f"a {self.name} encoding is a (real_bits, imag_bits) tuple, got {bits!r}")
        return self.component.check_encoding(bits[0]), self.component.check_encoding(bits[1])

    def decode(self, bits) -> ComplexValue:
        real, imag = self.check_encoding(bits)
        return ComplexValue(self.component.decode(real), self.component.decode(imag))

    def encode(self, value, *, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str = "ieee") -> tuple[int, int]:
        """Encode each part with the component format. A real value gets a ``+0`` imaginary part."""
        if isinstance(value, complex):
            value = ComplexValue.from_complex(value)
        if isinstance(value, ComplexValue):
            real, imag = value.real, value.imag
        else:
            real, imag = value, FloatValue.zero()
        encode = self.component.encode
        return encode(real, rounding=rounding, overflow=overflow), encode(imag, rounding=rounding, overflow=overflow)

    def to_scalar(self, bits) -> complex:
        return complex(self.decode(bits))


def conjugated(encodings: list[tuple[int, int]], fmt: Format) -> list[tuple[int, int]]:
    """Exact complex conjugates of complex encodings (each imaginary sign bit flipped); real ones unchanged."""
    if fmt.kind != "complex":
        return encodings
    sign = fmt.component._sign_bit
    return [(re, im ^ sign) for re, im in encodings]


_SIGN_FLIP = bytes(b ^ 0x80 for b in range(256))


def flip_imaginary_signs(data: bytearray, itemsize: int) -> None:
    """``conjugated`` on little-endian complex elements of ``itemsize`` bytes, in place: the
    sign bit of every imaginary part is the top bit of each element's top byte."""
    data[itemsize - 1 :: itemsize] = data[itemsize - 1 :: itemsize].translate(_SIGN_FLIP)


bool_ = BoolFormat()
int8 = IntFormat("int8", 1, True)
int16 = IntFormat("int16", 2, True)
int32 = IntFormat("int32", 4, True)
int64 = IntFormat("int64", 8, True)
uint8 = IntFormat("uint8", 1, False)
uint16 = IntFormat("uint16", 2, False)
uint32 = IntFormat("uint32", 4, False)
uint64 = IntFormat("uint64", 8, False)
float16 = FloatFormat("float16", 5, 10)
bfloat16 = FloatFormat("bfloat16", 8, 7, nan_encodings=(0x7FC0, 0x7FC0))
float32 = FloatFormat("float32", 8, 23)
float64 = FloatFormat("float64", 11, 52)
float8_e5m2 = FloatFormat("float8_e5m2", 5, 2, nan_encodings=(0x7F, 0xFF))
float8_e4m3 = FloatFormat("float8_e4m3", 4, 3)
float8_e3m4 = FloatFormat("float8_e3m4", 3, 4)
float8_e4m3fn = FloatFormat("float8_e4m3fn", 4, 3, special="fn")
float8_e5m2fnuz = FloatFormat("float8_e5m2fnuz", 5, 2, special="fnuz")
float8_e4m3fnuz = FloatFormat("float8_e4m3fnuz", 4, 3, special="fnuz")
float8_e4m3b11fnuz = FloatFormat("float8_e4m3b11fnuz", 4, 3, special="fnuz", bias=11)
complex32 = ComplexFormat("complex32", float16)
complex64 = ComplexFormat("complex64", float32)
complex128 = ComplexFormat("complex128", float64)

FORMATS: dict[str, Format] = {
    f.name: f
    for f in (
        bool_,
        int8, int16, int32, int64,
        uint8, uint16, uint32, uint64,
        float16, bfloat16, float32, float64,
        float8_e5m2, float8_e4m3, float8_e3m4, float8_e4m3fn,
        float8_e5m2fnuz, float8_e4m3fnuz, float8_e4m3b11fnuz,
        complex32, complex64, complex128,
    )
}


def _part_format(fmt: Format) -> Format:
    """The format of one part of an element: a complex format's component, else the format itself."""
    return fmt.component if fmt.kind == "complex" else fmt


def get_format(dtype: Format | str) -> Format:
    """Return a format instance, looking up names in ``FORMATS``."""
    if isinstance(dtype, Format):
        return dtype
    if isinstance(dtype, str):
        try:
            return FORMATS[dtype]
        except KeyError:
            supported = ", ".join(FORMATS)
            raise UnsupportedFormatError(f"unsupported format {dtype!r}; supported: {supported}") from None
    raise TypeError(f"expected a Format or format name, got {type(dtype).__name__}")
