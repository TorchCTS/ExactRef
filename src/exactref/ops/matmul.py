"""Matrix product ``C = A @ B``, and matmul algorithms."""

from __future__ import annotations

from fractions import Fraction

from .. import _native
from .._fenv import ieee
from ..accumulation import METHODS, MMA, MMAStep, get_method
from . import _mma
from ._mma import mma_terms
from ..algorithm import Algorithm, ErrorModel
from ..array import Array
from ..boundaries import NO_INTEGER_GRADIENT, Accumulation, Boundary, OpSpec, Uses, register
from ..formats import Format, Rounding, _float_codes, float32, get_format
from ..result import Bounds, Result, check_kind, check_standard, checked_output
from ._operands import Operand, grad_operand, operand
from ._rounding import bound_precision, encode_options, nearest_accumulator, output_format, require_real_float
from ._product_core import MatmulShape, backward_operands, combined, kernel_bounds, kernel_run, product

register(
    OpSpec(
        "matmul",
        boundaries={
            "accumulate": Boundary("accumulate", "summation of the k products of one output element", applies_to="dot"),
        },
        guarantee="correctly_rounded",
        limitations=(
            "matmul algorithms (run, tolerance) support real floating-point inputs only",
            "structure and offs are consumer mappings, not parameters: storage forms are converted to ordinary "
            "matrices, and grouped (offs) matmuls are split into ordinary ones",
        ),
    )
)


register(
    OpSpec(
        "matmul.backward",
        boundaries={
            "grad_A": Uses("matmul", "grad @ B^H, with broadcast batch copies folded into the contraction"),
            "grad_B": Uses("matmul", "A^H @ grad, with broadcast batch copies folded into the contraction"),
        },
        guarantee="correctly_rounded",
        limitations=(NO_INTEGER_GRADIENT,),
    )
)


def _operands(A, B) -> tuple[Operand, Operand]:
    A, B = operand(A), operand(B)
    a, b = A.array, B.array
    if a.dtype != b.dtype:
        raise TypeError(f"matmul operands must share a dtype, got {a.dtype.name} and {b.dtype.name}")
    if a.dtype.kind == "bool":
        raise TypeError("matmul does not support bool operands")
    return A, B


@ieee
def matmul(A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
           overflow: str | None = None, workers: int | None = None) -> Result:
    """Matrix product with NumPy batching and 1-D promotion.

    Each output element is the exact sum of exact products, rounded once to
    ``dtype`` (default: the input dtype); ``exact`` holds the exact sums.
    ``A`` and ``B`` must share a dtype, as in ``torch.matmul``, and ``dtype`` must
    be of their kind; convert one with ``asarray(x, d)`` first.

    ``tolerance()`` budgets the required boundary, accumulation in the compute
    format, then the rounding to ``dtype``. By default it covers normal
    accumulation: in index order or in 2 or 4 SIMD-style lanes, each product
    rounded and then added, starting a new partial sum where adding the next
    value would overflow (fused multiply-adds are covered too).
    ``standard="any_order"`` covers every order (any tree of additions, with or
    without fused multiply-add or rounding each product first) that does not
    overflow. ``ulps`` changes nothing here (matmul calls no library function),
    but must be at least 1/2, as everywhere. ``normal()`` is the output of
    normal accumulation in index order; for complex inputs each part is the
    sum of its 2k real products, taken in index order.

    Non-finite inputs follow IEEE rules applied to the exact sum: NaN if any
    product is NaN (including inf * 0) or infinities of both signs occur,
    otherwise the infinity. An exact zero sum is +0, or -0 when rounding
    toward -inf; an empty sum (k = 0) is +0. Integer results wrap on overflow
    unless ``overflow`` says otherwise. For complex inputs each part follows
    these rules on its own real terms (``a_re b_re`` and ``-a_im b_im`` for the
    real part, ``a_re b_im`` and ``a_im b_re`` for the imaginary part):
    ``(inf + 0j) @ (2 + 0j)`` is ``inf + nanj``, as in PyTorch's batched product.

    ``A`` and ``B`` may also be earlier results, as ``Result`` describes.
    """
    A, B = _operands(A, B)

    def backward(grad, **options):
        return _matmul_backward(A, B, grad, **options)

    return product("matmul", A, B, output_format(dtype, A.array.dtype), rounding, overflow, A.array.dtype, backward=backward, workers=workers)


@ieee
def _matmul_backward(A, B, grad, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                     workers: int | None = None) -> Result:
    """Gradients of ``matmul(A, B)`` for the output gradient ``grad``.

    ``grad_A = grad @ B^H`` and ``grad_B = A^H @ grad`` (``^H``: conjugate
    transpose, as PyTorch uses for complex inputs), each summed over the batch
    copies that broadcasting made of its operand, with that operand's shape.
    Each gradient is one matmul: the broadcast copies are folded into its
    contraction, so every element is one exact sum rounded once, and
    ``exact``, ``tolerance()`` and ``normal()`` are that matmul's. PyTorch
    folds the same way except when both operands keep batch dimensions that
    broadcast against each other; then it rounds each batch's product to the
    dtype before summing the copies, an extra rounding this tolerance does
    not budget. ``grad`` must have the output's shape and the inputs' dtype.
    ``A``, ``B`` and ``grad`` may also be earlier results.
    """
    A, B = _operands(A, B)
    fmt = A.array.dtype
    sizes = MatmulShape(A.array, B.array)
    G = grad_operand("matmul", grad, sizes.out_shape, fmt)
    if G.array.dtype != fmt:
        raise TypeError(f"grad must have the inputs' dtype {fmt.name}, got {G.array.dtype.name}")
    encode_options(output_format(dtype, fmt), rounding, overflow)  # the arguments, checked before the operands are gathered
    options = {"dtype": dtype, "rounding": rounding, "overflow": overflow, "workers": workers}
    (left, right), (left_b, right_b) = backward_operands(A, B, G, sizes)
    grad_A = matmul(left, right, **options)
    grad_B = matmul(left_b, right_b, **options)
    return combined("matmul.backward", {"grad_A": grad_A.reshape(A.array.shape), "grad_B": grad_B.reshape(B.array.shape)})


matmul.backward = _matmul_backward


class _Bounded(Algorithm):
    """A matmul algorithm whose ``_bounds(A, B, dtype, rounding, overflow, standard,
    flush_to_zero, workers)`` gives the output's bounds; these methods read them, as ``Result``'s do."""

    @ieee
    def tolerance(
        self, A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None, standard: str = "normal",
        flush_to_zero: bool = False, workers: int | None = None,
    ) -> Array:
        """Worst-case ``|output - matmul(A, B).rounded|`` per element over this algorithm's
        finite outputs, as ``Result.tolerance`` describes.

        ``A`` and ``B`` may be earlier results; their tolerances are then taken with
        ``standard`` and ``flush_to_zero`` (which change nothing for arrays). ``overflow`` is the
        implementation's overflow mode, as in ``matmul``, and ``workers`` the number of threads, as
        in ``Result.tolerance``."""
        return self._bounds(A, B, dtype, rounding, overflow, standard, flush_to_zero, workers).tolerance

    @ieee
    def finite_range(
        self, A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None, standard: str = "normal",
        flush_to_zero: bool = False, workers: int | None = None,
    ) -> tuple[Array, Array]:
        """The lowest and highest finite output of this algorithm per element, as
        ``Result.finite_range`` describes; the arguments are ``tolerance``'s."""
        found = self._bounds(A, B, dtype, rounding, overflow, standard, flush_to_zero, workers)
        return found.low, found.high

    @ieee
    def may_be(
        self, kind: str, A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
        standard: str = "normal", flush_to_zero: bool = False, workers: int | None = None,
    ) -> Array:
        """Where this algorithm may output ``kind`` (``"+inf"``, ``"-inf"`` or ``"nan"``), as
        ``Result.may_be`` describes; the other arguments are ``tolerance``'s."""
        check_kind(kind)
        return self._bounds(A, B, dtype, rounding, overflow, standard, flush_to_zero, workers).may_be(kind)

    @ieee
    def covers(
        self, output, A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
        standard: str = "normal", flush_to_zero: bool = False, workers: int | None = None,
    ) -> Array:
        """Where this algorithm can produce ``output``, as ``Result.covers`` describes; the
        other arguments are ``tolerance``'s."""
        A, B, sizes, out = self._prepare(A, B, dtype)
        output = checked_output(output, out, sizes.out_shape)
        return self._bounds(A, B, dtype, rounding, overflow, standard, flush_to_zero, workers).covers(output)

    def _prepare(self, A, B, dtype):
        return _algorithm_operands(self.name, A, B, dtype)


class MatmulAlgorithm(_Bounded):
    """One fully specified matmul: each output element's k products, exact (fused
    multiply-add) or rounded to the accumulator format, summed by one accumulation
    method in that format, rounding to nearest (``accumulator_rounding``: ties to even or
    away), then rounded to the output format (``rounding`` in ``run`` and the queries). Real
    floating-point inputs only."""

    op = "matmul"

    def __init__(self, method: str, format: Format | str, *, fma: bool = True, block_size: int | None = None,
                 accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown matmul accumulation method {method!r}; known: {', '.join(METHODS)} (tensor cores: matmul_mma)")
        fmt = require_real_float(get_format(format), "a matmul algorithm")
        rounding = nearest_accumulator(accumulator_rounding)
        self.accumulation = Accumulation(method, fmt, block_size, rounding)
        self.rounded_products = not fma and method != "exact"
        self.name = "matmul_exact_accumulation" if method == "exact" else f"matmul_{method}"
        products = f"products rounded to {fmt.name}" if self.rounded_products else "exact products"
        blocks = f", in blocks of {block_size}" if block_size else ""
        self.description = f"{get_method(method).label} of {products}, accumulated in {fmt.name}{blocks}"

    @ieee
    def run(self, A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
            workers: int | None = None) -> Array:
        """This algorithm's output, emulated exactly (on the ``rounded`` values of operands that
        are earlier results)."""
        A, B, sizes, out = self._prepare(A, B, dtype)
        options = encode_options(out, rounding, overflow)
        acc = self.accumulation
        return kernel_run(sizes, A.array, B.array, out, options, acc.format, acc.rounding, get_method(acc.method).code, acc.block_size or 0,
                           not self.rounded_products, workers)

    def _bounds(self, A, B, dtype, rounding: Rounding, overflow, standard: str, flush_to_zero: bool, workers: int | None = None) -> Bounds:
        """Lemma S over the method's own summation tree, or the exact sum, per element."""
        A, B, sizes, out = self._prepare(A, B, dtype)
        options = encode_options(out, rounding, overflow)
        radii = _algorithm_radii(A, B, standard, flush_to_zero, workers)
        ma, mb = _algorithm_marks(A, B, standard, flush_to_zero, workers)
        A, B = A.array, B.array
        rounded = matmul(A, B, dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)._single()
        acc = self.accumulation
        # Lemma S over the method's own summation tree (proofs/matmul.md, Algorithms); for exact accumulation, the exact
        # sum rounded once to the accumulator format (Lemma 1), held within its largest value where it may overflow
        order = (get_method(acc.method).code, acc.block_size or 0)
        return kernel_bounds(sizes, A, B, out, acc.format, rounding, rounded, (ma, mb), overflow=options["overflow"],
                             order=order, radii=radii, workers=workers)

    def error_model(self, k: int) -> ErrorModel:
        """The bound on the accumulated sum of k products, before the output rounding: the
        summation bound's growth factor and constant (``proofs/foundations.md``, Lemma 2), rounded
        up to 128 bits. Raises ValueError where depth times u reaches 1, where that bound is not
        finite."""
        return self._error_model(k, self.name)

    def _error_model(self, k: int, name: str) -> ErrorModel:
        """``error_model``, its error naming the algorithm ``name``."""
        acc = self.accumulation
        method, fmt = get_method(acc.method), acc.format
        depth, roundings = method.depth(k, block_size=acc.block_size), method.roundings(k, block_size=acc.block_size)
        terms = _native.growth_terms(fmt._layout, _float_codes(fmt, Rounding.NEAREST_EVEN, "ieee")[0], depth, roundings, 128, False)
        if terms is None:
            raise ValueError(f"{name} has no finite error model for k = {k} in {fmt.name}")
        (gm, ge), (cm, ce), _ = terms
        return ErrorModel(Fraction(gm) * Fraction(2) ** ge, Fraction(cm) * Fraction(2) ** ce, "sum over k of |A[i,k] * B[k,j]|")


def _algorithm_operands(name: str, A, B, dtype) -> tuple:
    """An algorithm's operands (real floating-point), sizes and output format."""
    A, B = _operands(A, B)
    require_real_float(A.array.dtype, name)
    out = require_real_float(output_format(dtype, A.array.dtype), name)
    return A, B, MatmulShape(A.array, B.array), out


def _algorithm_marks(A: Operand, B: Operand, standard: str, flush: bool, workers: int | None = None) -> tuple:
    """The operands' marks (``Operand.marks``) when they are earlier results, taken as
    ``_algorithm_radii`` takes their radii."""
    return A.marks(None, None, flush, standard, workers), B.marks(None, None, flush, standard, workers)


def _algorithm_radii(A: Operand, B: Operand, standard: str, flush: bool, workers: int | None = None) -> tuple:
    """The operands' radii when they are earlier results, from their tolerances with
    ``standard`` and ``flush`` (and their default compute format and ulps)."""
    check_standard(standard)
    return A.radius(None, None, flush, standard, workers), B.radius(None, None, flush, standard, workers)


def matmul_sequential(format: Format | str, *, fma: bool = True, accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> MatmulAlgorithm:
    """Products added one at a time in index order, in ``format``."""
    return MatmulAlgorithm("sequential", format, fma=fma, accumulator_rounding=accumulator_rounding)


def matmul_pairwise(format: Format | str, *, fma: bool = True, accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> MatmulAlgorithm:
    """Products summed as a balanced binary tree, in ``format``."""
    return MatmulAlgorithm("pairwise", format, fma=fma, accumulator_rounding=accumulator_rounding)


def matmul_blocked(format: Format | str, block_size: int, *, fma: bool = True,
                   accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> MatmulAlgorithm:
    """Products summed sequentially in blocks of ``block_size``, then the block sums sequentially, in ``format``."""
    return MatmulAlgorithm("blocked", format, fma=fma, block_size=block_size, accumulator_rounding=accumulator_rounding)


def matmul_exact_accumulation(format: Format | str, *, accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> MatmulAlgorithm:
    """Products summed exactly (a long accumulator), then rounded once to ``format``."""
    return MatmulAlgorithm("exact", format, accumulator_rounding=accumulator_rounding)


class MatmulSlots16(_Bounded):
    """The 16-slot matmul (``proofs/matmul.md``, The 16-slot algorithm): per output element,
    each product rounded to the accumulator format (to nearest, ``accumulator_rounding``: ties to
    even or away), the finite products split by magnitude into 16
    slots of one sixteenth of the format's precision each (in binades below the largest
    finite product; anything smaller goes to the last slot), each slot adding its values in
    increasing magnitude, alternating positive and negative values, and the slot sums then
    added from the smallest slot up; products that are not finite join that sum last, by
    IEEE rules. Real floating-point inputs only."""

    op = "matmul"
    name = "matmul_slots16"
    SLOTS = 16

    def __init__(self, format: Format | str, *, accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> None:
        self.format = require_real_float(get_format(format), "a matmul algorithm")
        self.accumulator_rounding = nearest_accumulator(accumulator_rounding)
        self.description = f"16 magnitude slots of products rounded to {self.format.name}, alternating signs, smallest slot first"

    @ieee
    def run(self, A, B, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
            workers: int | None = None) -> Array:
        """This algorithm's output, emulated exactly (on the ``rounded`` values of operands that
        are earlier results)."""
        A, B, sizes, out = self._prepare(A, B, dtype)
        options = encode_options(out, rounding, overflow)
        return kernel_run(sizes, A.array, B.array, out, options, self.format, self.accumulator_rounding, 5, 0, False, workers)

    def _bounds(self, A, B, dtype, rounding: Rounding, overflow, standard: str, flush_to_zero: bool, workers: int | None = None) -> Bounds:
        """Lemma S over each element's slots, with the exact sums of the nodes' terms. Where
        an element's terms are uncertain, its slots are not known, so it gets Lemma S for
        any tree of its terms."""
        A, B, sizes, out = self._prepare(A, B, dtype)
        options = encode_options(out, rounding, overflow)
        radii = _algorithm_radii(A, B, standard, flush_to_zero, workers)
        marks = _algorithm_marks(A, B, standard, flush_to_zero, workers)
        A, B = A.array, B.array
        rounded = matmul(A, B, dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)._single()
        # order kind 5, its block the accumulator's rounding to nearest
        acc_rounding, _ = _float_codes(self.format, self.accumulator_rounding, "ieee")
        return kernel_bounds(sizes, A, B, out, self.format, rounding, rounded, marks, overflow=options["overflow"],
                             order=(5, acc_rounding), radii=radii, workers=workers)

    def error_model(self, k: int) -> ErrorModel:
        """The bound on the accumulated sum of k products, before the output rounding: in the
        worst case one slot holds every product, a chain of k - 1 additions after k product
        roundings (as ``matmul_sequential`` with rounded products). Raises ValueError where that chain's
        depth times u reaches 1, where the summation bound (``proofs/foundations.md``, Lemma 2) is
        not finite."""
        sequential = MatmulAlgorithm("sequential", self.format, fma=False, accumulator_rounding=self.accumulator_rounding)
        return sequential._error_model(k, self.name)


class MatmulMMA(_Bounded):
    """A tensor-core matmul (``proofs/matmul.md``, Algorithms; ``proofs/foundations.md``, Lemma T):
    each output element's products added by chains of tensor-core steps with ``step``'s
    parameters, the chains' results optionally rounded to nearest in ``reduction`` (split-K
    partial sums in a lower precision) and added in float32 in any order, then rounded to the
    output format. Inputs: float16, bfloat16, or float32 holding tf32 values (``round_to_tf32``,
    rounding as the kernel converts); a float32 input must be an array, not an earlier result,
    whose stored values need not be tf32 values (float16 and bfloat16 earlier results are
    taken). It has no ``run()``: kernels split and order the steps as they choose.
    ``standard="any_order"`` gives the same bound as the normal standard;
    ``flush_to_zero=True`` raises ValueError, as tensor-core steps keep subnormal values."""

    op = "matmul"

    def __init__(self, step: MMAStep, *, reduction: Format | str | None = None) -> None:
        self.step = _mma.check_step(step)
        self.reduction = _mma.check_reduction(reduction, "matmul_mma")
        self.name, self.description = _mma.describe("matmul_mma", self.step, self.reduction)

    def _bounds(self, A, B, dtype, rounding: Rounding, overflow, standard: str, flush_to_zero: bool, workers: int | None = None) -> Bounds:
        """Lemma T per element, from the exact sums and masses."""
        _mma.check_unflushed(flush_to_zero)
        A, B, sizes, out = self._prepare(A, B, dtype)
        options = encode_options(out, rounding, overflow)
        _mma.check_inputs(self.name, [A, B])
        radii = _algorithm_radii(A, B, standard, False, workers)
        ma, mb = _algorithm_marks(A, B, standard, False, workers)
        A, B = A.array, B.array
        mma = MMA(self.step, self.reduction, _mma.slack(A.dtype, _mma.extent(A, radii[0]), _mma.extent(B, radii[1])))
        reference_array = matmul(A, B, dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)._single()
        w = bound_precision(out)
        terms = mma_terms(mma, sizes.k, w)
        reduction = None if self.reduction is None else self.reduction._layout
        return kernel_bounds(sizes, A, B, out, float32, rounding, reference_array, (ma, mb), standard="any_order",
                             overflow=options["overflow"], w=w, mma=(terms[0], terms[1], reduction),
                             radii=radii, workers=workers)


def matmul_mma(step: MMAStep, *, reduction: Format | str | None = None) -> MatmulMMA:
    """A tensor-core matmul with ``step``'s parameters (``MatmulMMA``)."""
    return MatmulMMA(step, reduction=reduction)


def matmul_slots16(format: Format | str, *, accumulator_rounding: Rounding = Rounding.NEAREST_EVEN) -> MatmulSlots16:
    """Products rounded to ``format`` and summed in 16 magnitude slots (``MatmulSlots16``)."""
    return MatmulSlots16(format, accumulator_rounding=accumulator_rounding)
