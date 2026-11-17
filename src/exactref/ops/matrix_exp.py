"""Matrix exponential ``exp(A) = sum_k A**k / k!``."""

from __future__ import annotations

import functools
import inspect
import math
import textwrap
from fractions import Fraction

from .. import _native
from .._fenv import ieee
from ..array import Array
from ..boundaries import Boundary, OpSpec, Uses, register
from ..exact import ExactArray, start_precision
from ..errors import ReferenceUnavailableError
from ..formats import Format, Rounding, _float_codes, _part_format, float64, uint8
from ..result import Bounds, Result
from ._operands import Operand, operand, unflagged_radius
from ._rounding import compute_format, encode_options, output_format, tolerance_ulps
from ._shapes import unravel

_MAX_PRECISION = 1 << 13
_LIMITATION = (
    f"an irrational part whose rounding needs more than {_MAX_PRECISION} bits of precision (one very close to a "
    "rounding boundary, the entries of an ill-conditioned matrix, or a part far below the largest entries of its row or "
    "column, such as e**-20000 next to 1/20000 in exp([[-20000, 1], [0, 0]]), even where it rounds to 0) raises "
    "ReferenceUnavailableError, and so does a part still undecided at the most precision the reference's products hold "
    f"for the matrix's size, less than {_MAX_PRECISION} bits from about 6600 x 6600 real or 3300 x 3300 complex "
    "matrices; so does a matrix with ||A||_inf of 2**60 or more (or just below, where its bound reaches 2**60) that is "
    "not diagonal, whose exponential's exponents are past the reference's range"
)
_BLOCK = (
    "the sizes and the norm above are those of the block matrix [[A, E'], [0, A]], 2n x 2n, E' = E scaled by a power "
    "of two to within a factor of 2 of max(||A||_inf, 1), or, where ||A||_inf reaches 2**57, only scaled down, below 2**57"
)

_BACKWARD_BLOCK = _BLOCK.replace("[[A, E'], [0, A]]", "[[A^H, grad'], [0, A^H]]").replace("E' = E", "grad' = grad")

register(
    OpSpec(
        "matrix_exp",
        boundaries={
            "result": Boundary(
                "backward",
                "the computed exponential is exp(A + dA) with ||dA||_inf <= ulps * (n + 16) * u * max(||A||_inf, 1) "
                "(u: unit roundoff of the compute format), held in the compute format",
            ),
        },
        guarantee="correctly_rounded",
        limitations=(_LIMITATION,),
    )
)
register(
    OpSpec(
        "matrix_exp_frechet",
        boundaries={"L": Uses("matrix_exp", "exp of the block matrix [[A, E], [0, A]], whose top-right block is L")},
        guarantee="correctly_rounded",
        limitations=("expA is a consumer mapping, not an output: exp(A) is matrix_exp(A), a separate call", _LIMITATION, _BLOCK),
    )
)
register(
    OpSpec(
        "matrix_exp.backward",
        boundaries={"grad_A": Uses("matrix_exp", "exp of the block matrix [[A^H, grad], [0, A^H]], whose top-right block is grad_A")},
        guarantee="correctly_rounded",
        limitations=(_LIMITATION, _BACKWARD_BLOCK),
    )
)

_DEFAULT_ULPS = 1


@ieee
def matrix_exp(A, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
               overflow: str | None = None, workers: int | None = None) -> Result:
    """Exponential of each square matrix in ``A`` (shape ``(..., n, n)``), real or complex.

    Every entry is the exact value correctly rounded to ``dtype``, which must be of
    the input's kind (convert across kinds with ``asarray(x, d)`` first). Entries
    are rounded from certified enclosures (exactref's interval matrix
    exponential), refined until both ends round alike. Each part of
    an entry is rational or transcendental (Lindemann-Weierstrass). A
    transcendental part is never on a rounding boundary, so refining it ends;
    a rational part can be, so parts still unrounded after the first
    refinements are tested for being rational with exact arithmetic, and
    rational parts are rounded from their exact values. An exact zero is +0,
    or -0 when rounding toward -inf. Any non-finite entry makes the whole
    matrix NaN.

    ``tolerance()`` uses a backward-error model, because no internal boundary
    is shared by all algorithms (Taylor or Pade, any degree, any number of
    squarings): the computed result is ``exp(A + dA)`` with
    ``||dA||_inf <= ulps * (n + 16) * u * max(||A||_inf, 1)`` (``ulps``
    defaults to 1, ``u`` is the compute format's unit roundoff; n and 16
    count the roundings of an algorithm's inner products and its others, and
    the floor of 1 covers the rounding of exp(A)'s identity part, which every
    algorithm makes), rounded to the compute format and then to ``dtype``.
    This is an assumption, measured on random Gaussian, symmetric, non-normal
    and skew-symmetric matrices (n = 2 to 64, ``||A||_inf`` 0.03 to 512):
    Higham's 2005 scaling and squaring with its degree-13 Pade approximant,
    and PyTorch's ``matrix_exp``, stay within half of it. Outside it lie:
    near a multiple of the identity with eigenvalues near +-5.37 * 2**s, that
    Pade formula's own rounding (up to about 3 times) and PyTorch's float32
    results (up to about 6 times, for negative eigenvalues); PyTorch's own
    results at small norms (float64 near 0.03, float32 up to 1); and
    ``scipy.linalg.expm`` 1.18.1 for some float64 matrices from
    ``||A||_inf`` 4. There the reference matches mpmath at high precision.
    ``A`` may also be an earlier result, as ``Result``
    describes: the model then holds for every A within its tolerance.
    """
    source = operand(A)
    A = source.array
    _check_square("matrix_exp", A)
    out = output_format(dtype, A.dtype)
    options = encode_options(out, rounding, overflow)
    enc, flags, mids = _kernel_reference(A, None, out, options, workers)
    E = Array(out, A.shape, enc)
    exact = ExactArray._from_storage(A.shape, out.components, flags, mids)

    def tolerance(compute, ulps, flush, standard, workers=None):
        radius = unflagged_radius(source, compute, ulps, flush, standard, workers)
        return {"E": _kernel_bounds(A, None, radius, None, compute_format(compute, A.dtype), tolerance_ulps(ulps, _DEFAULT_ULPS), flush, out, rounding,
                                options["overflow"], enc, workers)}

    def backward(grad, **options):
        return _matrix_exp_backward(source, grad, **options)

    return Result("matrix_exp", "correctly_rounded", {"E": E}, {"E": exact}, tolerance, backward)


@ieee
def matrix_exp_frechet(A, E, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                       workers: int | None = None) -> Result:
    """Frechet derivative of the matrix exponential at ``A`` in the direction ``E``.

    ``L = d/dt exp(A + t E)`` at ``t = 0``, for each square matrix in ``A`` and
    ``E`` (shape ``(..., n, n)``, the same for both), real or complex. ``L`` is
    the top-right block of ``exp([[A, E], [0, A]])``, so this is ``matrix_exp``
    of that 2n x 2n block matrix, restricted to the block (a proxy): every
    entry is correctly rounded once to ``dtype`` (of the inputs' kind, as in
    ``matrix_exp``), and ``exact`` and ``tolerance()`` are
    ``matrix_exp``'s for those entries. The tolerance is ``matrix_exp``'s
    backward-error model on the block matrix, which is how PyTorch computes
    ``matrix_exp``'s backward. ``A`` and ``E`` may also be earlier results.
    """
    return _frechet("matrix_exp_frechet", "L", operand(A), operand(E), dtype, rounding, overflow, workers)


@ieee
def _matrix_exp_backward(A, grad, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                         workers: int | None = None) -> Result:
    """Gradient of ``matrix_exp(A)`` for the output gradient ``grad``.

    ``grad_A = matrix_exp_frechet(A^H, grad)`` (``^H``: conjugate transpose),
    which is what PyTorch computes. ``grad`` must have A's shape and dtype.
    ``A`` and ``grad`` may also be earlier results.
    """
    A, G = operand(A), operand(grad)
    _check_pair("matrix_exp.backward", A.array, G.array)
    encode_options(output_format(dtype, A.array.dtype), rounding, overflow)  # the arguments, checked before A is transposed
    n, size = A.array.shape[-1], A.array.size
    transposed = [start + j * n + i for start in range(0, size, max(n * n, 1)) for i in range(n) for j in range(n)]
    A_h = A.take(transposed, A.array.shape, conjugate=True)
    return _frechet("matrix_exp.backward", "grad_A", A_h, G, dtype, rounding, overflow, workers, named=True)


matrix_exp.backward = _matrix_exp_backward


def _with_limits(function, *limits: str) -> None:
    """Append the reference's limits, as the registry states them, to ``function``'s docstring (cleaned first, as
    Python 3.13 and later do when compiling it)."""
    text = "; ".join(limits)
    function.__doc__ = inspect.cleandoc(function.__doc__) + "\n\n" + textwrap.fill(f"Limits: {text}.", 92)


_with_limits(matrix_exp, _LIMITATION)
_with_limits(matrix_exp_frechet, _LIMITATION, _BLOCK)
_with_limits(_matrix_exp_backward, _LIMITATION, _BACKWARD_BLOCK)


def _check_square(op: str, A: Array) -> None:
    """``op``'s input: floating-point or complex square matrices, (..., n, n)."""
    if A.dtype.kind not in ("float", "complex"):
        raise TypeError(f"{op} needs floating-point or complex input, got {A.dtype.name}")
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"{op} needs square matrices (..., n, n), got {A.shape}")


def _check_pair(op: str, A: Array, E: Array) -> None:
    _check_square(op, A)
    if E.dtype != A.dtype:
        raise TypeError(f"{op}: both matrices must have dtype {A.dtype.name}, got {E.dtype.name}")
    if E.shape != A.shape:
        raise ValueError(f"{op}: both matrices must have shape {A.shape}, got {E.shape}")


def _frechet(op: str, name: str, A_source: Operand, E_source: Operand, dtype, rounding: Rounding, overflow, workers: int | None,
             named: bool = False) -> Result:
    """The top-right block of ``matrix_exp([[A, E], [0, A]])`` for each matrix, as output ``name``."""
    A, E = A_source.array, E_source.array
    _check_pair(op, A, E)
    out = output_format(dtype, A.dtype)
    options = encode_options(out, rounding, overflow)
    enc, flags, mids = _kernel_reference(A, E, out, options, workers)
    rounded = Array(out, A.shape, enc)
    exact = ExactArray._from_storage(A.shape, out.components, flags, mids)

    def tolerance(compute, ulps, flush, standard, workers=None):
        ra = unflagged_radius(A_source, compute, ulps, flush, standard, workers)
        re = unflagged_radius(E_source, compute, ulps, flush, standard, workers)
        return {name: _kernel_bounds(A, E, ra, re, compute_format(compute, A.dtype), tolerance_ulps(ulps, _DEFAULT_ULPS), flush, out, rounding,
                                 options["overflow"], enc, workers)}

    return Result(op, "correctly_rounded", {name: rounded}, {name: exact}, tolerance, named=named)


# ---------------------------------------------------------------- the reference


def _kernel_reference(A: Array, E: Array | None, out, options: dict, workers: int | None = None) -> tuple[bytes, bytes, bytes]:
    """Each wanted entry part's encoding, entry flag and entry midpoint as bytes, row-major, by the
    reference kernel (``csrc/matrix_exp.c``, ``proofs/matrix_exp.md``, The reference output):
    exp(A) for each matrix of A, or with E the top-right block of exp([[A, E], [0, A]])."""
    n, parts = A.shape[-1], 2 if A.dtype.kind == "complex" else 1
    component, fmt = _part_format(out), _part_format(A.dtype)
    x = _native.matrix_exp_reference(bytes(A.buffer), fmt._layout, A.byteorder == "big", math.prod(A.shape[:-2]), n, parts,
                                     None if E is None else bytes(E.buffer), E is not None and E.byteorder == "big", component._layout,
                                     *_float_codes(component, options["rounding"], options["overflow"]), start_precision(component), _MAX_PRECISION)
    while tiles := _native.matrix_exp_reference_next(x):
        _native.run([functools.partial(_native.matrix_exp_reference_tile, x, t) for t in range(tiles)], workers)
    enc, flags, mids, status = _native.matrix_exp_reference_result(x)
    for slot, code in enumerate(status):
        if code:
            raise _reference_error(code, unravel(slot // parts, A.shape), n, E is not None)
    return enc, flags, mids


def _reference_error(code: int, index: tuple[int, ...], n: int, block: bool) -> ReferenceUnavailableError:
    """The error for the reference kernel's status ``code`` (2: past the range, 3: past the products'
    precision, otherwise past _MAX_PRECISION) at entry ``index`` of exp(A) for n x n matrices, or
    with ``block`` of the top-right block of exp([[A, E], [0, A]])."""
    batch = f" (batch index {index[:-2]})" if len(index) > 2 else ""
    if code == 2:
        what = "the block matrix [[A, E'], [0, A]] (E' = E scaled by a power of two)" if block else "a matrix"
        return ReferenceUnavailableError(f"matrix_exp of {what}{batch} with ||.||_inf of 2**60 or more (or just below, where its bound reaches "
                                         "2**60) that is not diagonal: its exponential's exponents are past the reference's range")
    if code == 3:
        what = f"the {2 * n} x {2 * n} block matrix [[A, E'], [0, A]]" if block else f"a {n} x {n} matrix"
        return ReferenceUnavailableError(f"matrix_exp of {what}{batch}: an entry needs more precision than the reference kernel's "
                                         "products hold at this size")
    return ReferenceUnavailableError(f"matrix_exp entry {index} needs more than {_MAX_PRECISION} bits to round (it lies very close to "
                                     "a rounding boundary, or far below the largest entries of its row or column)")


def _kernel_bounds(A: Array, E: Array | None, radius_a, radius_e, compute, allowance: Fraction, flush: bool, out, rounding, overflow: str, reference: bytes,
                   workers: int | None = None) -> Bounds:
    """The tolerance's bounds by the kernel (``csrc/matrix_exp.c``, ``proofs/matrix_exp.md``, The
    tolerance): for each stored matrix of A (with E, the top-right block of exp([[A, E], [0, A]])),
    its entries widened by the radii (float64 or complex128 Arrays, or None) when given, the box's
    and the ball's enclosures held in ``compute`` and rounded to ``out``; ``reference`` holds the
    reference's encodings."""
    n, complex_input = A.shape[-1], A.dtype.kind == "complex"
    component, fmt = _part_format(out), _part_format(A.dtype)

    def radii(r):
        return (None, False) if r is None else (bytes(r.buffer), r.byteorder == "big")

    ra, rab = radii(radius_a)
    re, reb = radii(radius_e)
    x = _native.matrix_exp_tolerance(bytes(A.buffer), fmt._layout, A.byteorder == "big", math.prod(A.shape[:-2]), n, 2 if complex_input else 1,
                                     None if E is None else bytes(E.buffer), E is not None and E.byteorder == "big", ra, rab, re, reb, float64._layout,
                                     compute._layout, flush, (allowance.numerator, allowance.denominator), component._layout,
                                     *_float_codes(component, rounding, overflow), reference)
    while tiles := _native.matrix_exp_tolerance_next(x):
        _native.run([functools.partial(_native.matrix_exp_tolerance_tile, x, t) for t in range(tiles)], workers)
    tol, low, high, flags = _native.matrix_exp_tolerance_result(x)
    if complex_input:
        parts = len(flags)
        return Bounds(Array(float64, (parts,), tol), Array(component, (parts,), low), Array(component, (parts,), high), Array(uint8, (parts,), flags)).paired(out, A.shape)
    return Bounds(Array(float64, A.shape, tol), Array(out, A.shape, low), Array(out, A.shape, high), Array(uint8, A.shape, flags))
