"""Exact representations of decoded element values."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from fractions import Fraction


_KINDS = ("finite", "infinite", "nan")


@dataclass(frozen=True, slots=True)
class FloatValue:
    """The exact value of one binary floating-point encoding.

    A finite value equals ``(-1)**negative * significand * 2**exponent``. The pair
    is normalized (``significand`` is odd, or zero with ``exponent == 0``), so
    equal values have equal fields. Unlike ``Fraction``, a ``FloatValue`` keeps
    signed zeros, infinities and NaN.

    Equality is structural, not IEEE comparison: a NaN equals another NaN of the
    same sign, and ``-0`` differs from ``+0``. NaN payloads are not represented;
    read raw encodings when a payload must be preserved.
    """

    kind: str
    negative: bool
    significand: int = 0
    exponent: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if type(self.negative) is not bool:
            raise TypeError("negative must be a bool")
        if type(self.significand) is not int or type(self.exponent) is not int:
            raise TypeError("significand and exponent must be ints")
        if self.kind != "finite":
            if self.significand or self.exponent:
                raise ValueError(f"a {self.kind} value has no significand or exponent")
            return
        if self.significand < 0:
            raise ValueError("significand must be non-negative; use negative for the sign")
        if self.significand == 0:
            object.__setattr__(self, "exponent", 0)
            return
        trailing = (self.significand & -self.significand).bit_length() - 1
        if trailing:
            object.__setattr__(self, "significand", self.significand >> trailing)
            object.__setattr__(self, "exponent", self.exponent + trailing)

    @classmethod
    def zero(cls, negative: bool = False) -> FloatValue:
        return cls("finite", negative)

    @classmethod
    def infinity(cls, negative: bool = False) -> FloatValue:
        return cls("infinite", negative)

    @classmethod
    def nan(cls, negative: bool = False) -> FloatValue:
        return cls("nan", negative)

    @classmethod
    def from_float(cls, x: float) -> FloatValue:
        """Return the exact value of a Python float, including its sign: its bytes decoded as
        float64, with no float arithmetic, so a thread that flushes subnormals to zero gets the
        value it passed."""
        if not isinstance(x, float):
            raise TypeError(f"expected a float, got {type(x).__name__}")
        from .formats import float64  # formats imports this module

        return float64.decode(int.from_bytes(struct.pack("<d", x), "little"))

    @property
    def is_finite(self) -> bool:
        return self.kind == "finite"

    @property
    def is_infinite(self) -> bool:
        return self.kind == "infinite"

    @property
    def is_nan(self) -> bool:
        return self.kind == "nan"

    @property
    def is_zero(self) -> bool:
        return self.kind == "finite" and self.significand == 0

    def to_fraction(self) -> Fraction:
        """Return the exact rational value. The sign of zero is not preserved."""
        if self.kind != "finite":
            raise ValueError(f"a {self.kind} value has no rational value")
        if self.exponent >= 0:
            magnitude = Fraction(self.significand << self.exponent)
        else:
            magnitude = Fraction(self.significand, 1 << -self.exponent)
        return -magnitude if self.negative else magnitude

    def __float__(self) -> float:
        """Return the nearest Python float (ties to even), keeping the sign of zero: the float64
        encoding, moved into a float by its bits."""
        from .formats import float64  # formats imports this module

        return struct.unpack("<d", float64.encode(self).to_bytes(8, "little"))[0]


@dataclass(frozen=True, slots=True)
class ComplexValue:
    """The exact value of one complex encoding: a pair of ``FloatValue`` parts."""

    real: FloatValue
    imag: FloatValue

    def __post_init__(self) -> None:
        if not isinstance(self.real, FloatValue) or not isinstance(self.imag, FloatValue):
            raise TypeError("real and imag must be FloatValue instances")

    @classmethod
    def from_complex(cls, z: complex) -> ComplexValue:
        """Return the exact value of a Python complex, including signed zeros."""
        if not isinstance(z, complex):
            raise TypeError(f"expected a complex, got {type(z).__name__}")
        return cls(FloatValue.from_float(z.real), FloatValue.from_float(z.imag))

    @property
    def is_finite(self) -> bool:
        return self.real.is_finite and self.imag.is_finite

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))
