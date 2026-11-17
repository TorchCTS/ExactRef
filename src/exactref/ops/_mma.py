"""Tensor-core kernels (internal): what the tensor-core algorithms share, ``matmul_mma``,
``gemm_mma`` and ``AttentionAlgorithm``'s ``mma`` method (``proofs/foundations.md``, Lemma T)."""

from __future__ import annotations

import functools
from fractions import Fraction

from .. import _native
from ..accumulation import MMA, MMAStep
from ..convert import _TF32
from ..formats import Rounding, _float_codes, bfloat16, float16, float32, float64, get_format
from ._rounding import require_real_float

_INPUTS = (float16, bfloat16, float32)  # float32 inputs must be tf32 values


def check_unflushed(flush_to_zero: bool) -> None:
    if flush_to_zero:
        raise ValueError("tensor-core steps keep subnormal values: flush_to_zero does not apply")


def check_step(step) -> MMAStep:
    if not isinstance(step, MMAStep):
        raise TypeError(f"step must be an MMAStep, got {type(step).__name__}")
    return step


def check_reduction(reduction, name: str):
    """The reduction format (None, or a real floating-point format)."""
    return None if reduction is None else require_real_float(get_format(reduction), f"{name}'s reduction")


def describe(prefix: str, step: MMAStep, reduction) -> tuple[str, str]:
    """A tensor-core algorithm's name and description: ``prefix`` (``matmul_mma``, ``gemm_mma``), the step's products and
    fraction bits, and the reduction format split-K partial sums are rounded to (None for none)."""
    reduced = f"_{reduction.name}" if reduction is not None else ""
    partials = f", split-K partial sums rounded to {reduction.name}" if reduction is not None else ""
    return (f"{prefix}{step.products}_f{step.fraction_bits}{reduced}",
            f"tensor-core steps of {step.products} products, {step.fraction_bits} fraction bits, exponent floor {step.exponent_floor}{partials}")


def check_inputs(name: str, operands: list, formats: list | None = None) -> None:
    """The operands' format is one a tensor-core step multiplies exactly: float16, bfloat16, or
    float32 holding tf32 values (low 13 bits zero, as ``round_to_tf32`` leaves them). An operand
    whose own format (``formats``, where it was converted to the operands' shared one) is float32
    must be an array: an earlier result's stored values need not be tf32 values."""
    fmt = operands[0].array.dtype
    if fmt not in _INPUTS:
        raise TypeError(f"{name} takes float16, bfloat16 or float32 (tf32 values), got {fmt.name}")
    if fmt is float32:
        for x, own in zip(operands, formats or [x.array.dtype for x in operands]):
            if x.uncertain and own is float32:
                raise ValueError(f"{name} does not take an earlier float32 result as A or B: its stored values need not be tf32 "
                                 "values, and a kernel's rounding of them to tf32 is not in the bound; pass a float32 array of tf32 values")
            # tf32 values are those that narrowing to tf32 leaves alone, so both directions agree
            data, big = bytes(x.array.buffer), x.array.byteorder == "big"
            down, up = (_native.float_narrow(data, big, float32._layout, _TF32, *_float_codes(float32, rounding, "ieee"))
                        for rounding in (Rounding.TOWARD_ZERO, Rounding.TOWARD_POSITIVE))
            if down != up:
                raise ValueError(f"{name}'s float32 operands must be tf32 values: round them with round_to_tf32")


def extent(x, radius) -> tuple[bool, Fraction]:
    """Whether operand ``x`` may hold a nonzero subnormal value (an earlier result may), and the
    largest |value| (plus its radius) among its finite elements (the C's ``mma_extent``)."""
    radii = None if radius is None else bytes(radius.buffer)
    sub, (m, e) = _native.mma_extent(bytes(x.buffer), x.byteorder == "big", x.dtype._layout, float64._layout, radii,
                                     radius is not None and radius.byteorder == "big")
    return sub, Fraction(m) * Fraction(2) ** e


def _dyadic(q: Fraction) -> tuple[int, int]:
    """A nonnegative dyadic Fraction as ``(m, e)``."""
    return q.numerator, 1 - q.denominator.bit_length()


def slack(fmt, x: tuple, y: tuple) -> Fraction:
    """Lemma T's sigma for products of two factors in ``fmt``, each given by its ``extent``:
    whether it may be subnormal, and its largest magnitude (the C's ``mma_slack``)."""
    (sub_x, max_x), (sub_y, max_y) = x, y
    m, e = _native.mma_slack(fmt.emin, sub_x, _dyadic(max_x), sub_y, _dyadic(max_y))
    return Fraction(m) * Fraction(2) ** e


@functools.lru_cache(maxsize=256)
def mma_terms(mma: MMA, n: int, w: int, extra: int = 0, steps: int | None = None, alpha: tuple[int, int] | None = None
              ) -> tuple[tuple[int, int], tuple[int, int]]:
    """Lemma T (``proofs/foundations.md``) for a sum of K = ``n`` products (``mma.count`` where
    set) by tensor-core steps, rounded up to ``w`` bits (``csrc/mma.c``): ``(G, H)`` with

        G = (1 + rho)**(K + r) (1 + tau)**r (1 + 2**-24)**extra (1 + u_R) - 1,
        H = (G + 1) (r_all h + s eta_R + extra 2**-150),

    where r bounds the steps on a product's path (``steps``, else the method's depth), r_all the
    steps in all, s the pieces, and ``extra`` the float32 roundings more on every path (gemm:
    alpha's multiplication and beta c's join; online softmax: its rescalings and joins). With
    ``alpha`` (a dyadic ``(m, e)``), H for gemm's alpha P: times max(1, |alpha|)."""
    step, reduction = mma.step, mma.reduction
    sigma = _dyadic(Fraction(mma.slack))
    K = n if mma.count is None else mma.count
    return _native.mma_terms(step.products, step.fraction_bits, step.exponent_floor, step.split, None if reduction is None else reduction._layout,
                             sigma, K, extra, -1 if steps is None else steps, w, alpha)
