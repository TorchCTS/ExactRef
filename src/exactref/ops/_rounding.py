"""Output and compute formats, encoding options, the bounds' working precision, and the checks
of formats, ulps and an accumulator's rounding (internal)."""

from __future__ import annotations

from fractions import Fraction

from ..boundaries import default_compute
from ..formats import FLOAT_OVERFLOW, INT_OVERFLOW, FloatFormat, Format, Rounding, _check_overflow, _check_rounding, _part_format, get_format
from ..result import check_ulps


def output_format(dtype, input_format: Format) -> Format:
    out = input_format if dtype is None else get_format(dtype)
    if out.kind != input_format.kind:
        raise TypeError(f"output dtype {out.name} must be the same kind as the input dtype {input_format.name}")
    return out


def encode_options(fmt: Format, rounding: Rounding, overflow: str | None) -> dict[str, Rounding | str]:
    """Output encoding options, checked; integer results wrap by default, as in PyTorch."""
    _check_rounding(rounding)
    if fmt.kind == "int":
        overflow = overflow or "wrap"
        _check_overflow(overflow, INT_OVERFLOW, fmt)
        return {"overflow": overflow}
    overflow = overflow or "ieee"
    _check_overflow(overflow, FLOAT_OVERFLOW, fmt)
    return {"rounding": rounding, "overflow": overflow}


def bound_precision(*formats: FloatFormat) -> int:
    """The working precision of a tolerance's bounds over these formats (its rounding points'
    and its output's): twice the widest one's precision plus 64, at least 128."""
    return max(128, 2 * max(f.precision for f in formats) + 64)


def compute_format(compute, input_format: Format) -> FloatFormat:
    """The format of required boundaries: given, or the default for the input format."""
    if compute is not None:
        fmt = get_format(compute)
        if not isinstance(fmt, FloatFormat):
            raise TypeError(f"compute must be a floating-point format, got {fmt.name}")
        return fmt
    base = _part_format(input_format)
    return default_compute(base)


def tolerance_ulps(ulps, default) -> Fraction:
    """``ulps``, or the operation's ``default`` when None, checked as ``check_ulps`` checks it."""
    return check_ulps(default if ulps is None else ulps)


def require_real_float(fmt: Format, what: str) -> FloatFormat:
    if not isinstance(fmt, FloatFormat):
        raise TypeError(f"{what} supports real floating-point formats, not {fmt.name}")
    return fmt


def nearest_accumulator(rounding: Rounding) -> Rounding:
    """``rounding``, which an accumulator needs to be to nearest (ties to even or away): a
    directed rounding saturates past the largest value, which no tolerance here bounds
    usefully (``proofs/foundations.md``, The normal standard, Units)."""
    if not isinstance(rounding, Rounding):
        raise TypeError(f"accumulator_rounding must be a Rounding, got {rounding!r}")
    if not rounding.is_nearest:
        raise ValueError(f"an accumulator rounds to nearest (NEAREST_EVEN or NEAREST_AWAY), not {rounding.name}")
    return rounding
