"""``gemm``: ``D = beta C + alpha (A @ B)`` (category ``linalg.products``).

The exact value is matmul's exact sums scaled by alpha, plus beta C; the tolerance is
Lemma G of ``proofs/gemm.md``, from the per-batch quantities matmul's tolerance uses."""

from __future__ import annotations

import math
from fractions import Fraction

from .._fenv import ieee
from ..accumulation import MMA, MMAStep
from ..algorithm import Algorithm
from . import _mma
from ._mma import mma_terms
from ..array import Array
from ..boundaries import NO_INTEGER_GRADIENT, Boundary, OpSpec, Uses, register
from ..formats import Format, Rounding, float32
from ..result import ALL, ANY, Bounds, Result, check_kind, check_standard, checked_output, exact_bounds
from ..values import ComplexValue, FloatValue
from ._operands import Operand, broadcast_to, exact_scalar, grad_operand, holds, is_unit, operand, promoted_format, repeated_operand, scalar_operand
from ._rounding import bound_precision, compute_format, encode_options, output_format, require_real_float
from ._shapes import broadcast_shapes
from ._product_core import (
    _TERMS_HELD,
    MatmulShape,
    _terms,
    backward_operands,
    batch_starts,
    combined,
    conjugate_scalar,
    gradient,
    kernel_bounds,
    kernel_epilogue,
    kernel_int_bounds,
    kernel_normal,
    kernel_reference,
)
from .elementwise import promoted

register(
    OpSpec(
        "gemm",
        boundaries={
            "accumulate": Boundary("accumulate", "summation of the k products and beta C of one output element", applies_to="dot"),
        },
        guarantee="correctly_rounded",
        limitations=(
            "trans_a, trans_b and triangle are consumer mappings, not parameters: the consumer passes op(A) and op(B) "
            "(an earlier result through Result.transpose) and compares the triangle it computes",
            "array operands must be of one kind (integer, floating-point or complex); other mixes are converted by the consumer",
        ),
    )
)
register(
    OpSpec(
        "gemm.backward",
        boundaries={
            "grad_A": Uses("gemm", "conj(alpha) grad @ B^H, with batch copies folded into the contraction"),
            "grad_B": Uses("gemm", "conj(alpha) A^H @ grad, with batch copies folded into the contraction"),
            "grad_C": Uses("matmul", "grad * conj(beta), summed over C's broadcast copies"),
        },
        guarantee="correctly_rounded",
        limitations=(NO_INTEGER_GRADIENT,),
    )
)


class _Problem:
    """gemm's inputs: A, B and C as read (``sources``, operands; ``given``, their arrays) and
    promoted (``elementwise.promoted``, which checks the output format ``out`` and the modes
    first), alpha and beta exact in the operands' kind, A and B before any folding
    (``unfolded``), and the output's ``shape``; ``operands`` folds and broadcasts them for the
    forward."""

    def __init__(self, A, B, C, alpha, beta, reduce_batch: bool, dtype, rounding: Rounding, overflow) -> None:
        self.sources = [operand(x) for x in (A, B, C)]
        self.given = [X.array for X in self.sources]
        (A, B, C), fmt, self.out = promoted("gemm", self.sources, dtype, rounding, overflow)
        self.fmt = fmt
        self.alpha, self.beta = _scalar(alpha, fmt, "alpha"), _scalar(beta, fmt, "beta")
        self.sizes = MatmulShape(A.array, B.array)
        self.unfolded, self._C = (A, B), C
        self.reduce = reduce_batch and bool(self.sizes.batch)
        # folding the batches into the contraction leaves the matrices' dimensions
        self.shape = self.sizes.out_shape[len(self.sizes.batch):] if self.reduce else self.sizes.out_shape
        if broadcast_shapes(C.array.shape, self.shape) != self.shape:
            raise ValueError(f"C of shape {C.array.shape} does not broadcast to the output's shape {self.shape}")

    def operands(self) -> tuple[Operand, Operand, Operand]:
        """A and B with their batches folded into the contraction when ``reduce_batch``, and C
        broadcast to the output's shape, as the forward reads them."""
        A, B = _folded(*self.unfolded, self.sizes) if self.reduce else self.unfolded
        return A, B, broadcast_to(self._C, self.shape)


def _scalar(value, fmt, what: str):
    """alpha or beta taken exactly, as an int, ``FloatValue`` or ``ComplexValue`` of ``fmt``'s kind."""
    v = scalar_operand([exact_scalar(value)], fmt.kind, (), fmt).array.value(())
    parts = (v.real, v.imag) if isinstance(v, ComplexValue) else (v,)
    if any(isinstance(p, FloatValue) and not p.is_finite for p in parts):
        raise ValueError(f"{what} must be finite, got {value}")
    return v


def _folded(A: Operand, B: Operand, sizes: MatmulShape) -> tuple:
    """A and B with the output's batches folded into the contraction: row i of A's matrices
    side by side, the rows of B's matrices stacked, in batch order."""
    m, k, n = sizes.m, sizes.k, sizes.n
    starts = batch_starts(sizes)
    width = len(starts) * k
    a = [sa + i * k + q for i in range(m) for sa, _ in starts for q in range(k)]
    b = [sb + q * n + j for _, sb in starts for q in range(k) for j in range(n)]
    a_shape = (m, width) if A.array.ndim > 1 else (width,)
    b_shape = (width, n) if B.array.ndim > 1 else (width,)
    return A.take(a, a_shape), B.take(b, b_shape)


@ieee
def gemm(A, B, C, *, alpha=1, beta=1, reduce_batch: bool = False, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
         overflow: str | None = None, workers: int | None = None) -> Result:
    """``D = beta C + alpha (A @ B)``, with matmul's batching and 1-D promotion and C
    broadcast to the output's shape (PyTorch's addmm, addmv, addr, baddbmm).

    Each output element is the exact value rounded once to ``dtype`` (default: the
    operands' promoted format, as ``add`` promotes); ``dtype`` must be of the inputs'
    kind; convert across kinds with ``asarray(x, d)`` first. ``alpha`` and ``beta`` are
    taken exactly as given and must be finite (ValueError otherwise). With ``beta = 0`` C is not read, and with ``alpha = 0``
    neither A nor B is, as in PyTorch and BLAS. With ``reduce_batch=True`` the output's
    batch dimensions are summed into the contraction, so each element is one exact sum
    over all batches (addbmm's exact form; PyTorch's addbmm rounds after each batch,
    which is a loop of gemm calls on earlier results). ``addr(C, v, w)`` is gemm with v
    as a column (m, 1) and w as a row (1, n).

    ``tolerance()`` budgets the products rounded or fused in the compute format, alpha
    applied once on every product's way (to an operand, a product, a partial sum or the
    whole sum), beta C rounded once (not at all with beta 1 or -1, or i or -i, a wider C of
    shape () taken by value included, where the compute format holds the other operands'
    format) and added at any point (first, as PyTorch's CPU kernels do, or last), the
    additions in normal accumulation, and the final
    rounding; ``standard="any_order"`` allows any order of additions. With an alpha whose
    imaginary part is nonzero the normal tolerance is the any-order one. ``normal()``
    adds the products in index order, then computes
    ``fl(fl(alpha * sum) + fl(beta * C))`` in the compute format (the order of a plain loop,
    and of Netlib's reference BLAS when A is transposed; ``beta * C`` exact with beta 1, -1,
    i or -i). Integer results wrap on overflow unless
    ``overflow`` says otherwise. A non-finite input follows IEEE rules on the exact expression; for complex values in
    each part separately, as PyTorch's batched products do, with ``alpha`` and ``beta`` taken as
    exact coefficients: a zero part of a nonzero ``alpha`` or ``beta`` adds no term. ``A``, ``B`` and ``C`` may also be earlier results.
    """
    problem = _Problem(A, B, C, alpha, beta, reduce_batch, dtype, rounding, overflow)

    def backward(grad, **options):
        return _gemm_backward(*problem.sources, grad, alpha=alpha, beta=beta, reduce_batch=reduce_batch, **options)

    return _gemm("gemm", *problem.operands(), problem.alpha, problem.beta, problem.out, rounding, overflow, problem.fmt, "D", backward, workers)


class _Scalars:
    """alpha and beta as the product kernel takes them: signed dyadics ``(m, e)``, or for complex
    values ``(re, im)`` pairs of them, whose channels the kernel makes (``proofs/gemm.md``,
    Channels); and whether each term is read: a coefficient that is 0 leaves its operand unread,
    as in PyTorch's addmm (gemm's own rule, not the coefficient rule)."""

    def __init__(self, alpha, beta, kind: str) -> None:
        self.complex = kind == "complex"
        self.alpha, self.beta = ((_scaled(v.real), _scaled(v.imag)) if self.complex else _scaled(v) for v in (alpha, beta))
        self.has_P, self.has_C = (any(m for m, _ in (v if self.complex else (v,))) for v in (self.alpha, self.beta))


def _scaled(v) -> tuple[int, int]:
    """An int or finite ``FloatValue`` as ``(m, e)`` with value ``m * 2**e``."""
    if isinstance(v, int):
        return v, 0
    return (-v.significand if v.negative else v.significand), v.exponent


def _gemm(op: str, A: Operand, B: Operand, C: Operand, alpha, beta, out, rounding: Rounding, overflow, compute_base, name: str,
          backward=None, workers: int | None = None) -> Result:
    """gemm's result for promoted operands, with C broadcast to the output's shape."""
    a, b = A.array, B.array
    sizes = MatmulShape(a, b)
    scale = _Scalars(alpha, beta, a.dtype.kind)
    options = encode_options(out, rounding, overflow)
    D, exact_array = kernel_reference(sizes, a, b, out, options, False, _epilogue(scale, C.array), workers=workers)

    def tolerance(compute, ulps, flush, standard, workers=None) -> dict[str, Bounds]:
        radii = (A.radius(compute, ulps, flush, standard, workers), B.radius(compute, ulps, flush, standard, workers)) if scale.has_P else (None, None)
        c_radius = C.radius(compute, ulps, flush, standard, workers) if scale.has_C else None
        if out.kind == "int":
            if radii == (None, None) and c_radius is None:
                return {name: exact_bounds(D)}
            return {name: kernel_int_bounds(sizes, a, b, D, options["overflow"], radii, workers, _epilogue(scale, C.array, c_radius=c_radius))}
        fmt = compute_format(compute, compute_base)
        ab_marks = (A.marks(compute, ulps, flush, standard, workers), B.marks(compute, ulps, flush, standard, workers)) if scale.has_P else (None, None)
        c_marks = C.marks(compute, ulps, flush, standard, workers) if scale.has_C else None
        epilogue = _epilogue(scale, C.array, _c_terms(fmt, a, b, C.array), c_radius, c_marks, fmt if flush else None)
        return {name: kernel_bounds(sizes, a, b, out, fmt, rounding, D, ab_marks, flush=flush, standard=standard, overflow=options["overflow"],
                                    epilogue=epilogue, radii=radii, workers=workers)}

    def normal(compute, flush, workers=None) -> dict[str, Array]:
        if out.kind == "int":
            return {name: D}
        # P's parts as matmul's normal answer before its rounding, then fl(fl(alpha P) + fl(beta c))
        # per part from the channels in the compute format (beta c exact with a unit beta), then
        # rounded to out (proofs/gemm.md, The normal answer)
        fmt = compute_format(compute, compute_base)
        x, y = A.normal(compute, flush, workers), B.normal(compute, flush, workers)
        c = C.normal(compute, flush, workers) if scale.has_C else None
        return {name: kernel_normal(sizes, x, y, out, fmt, options, flush, False, workers, _epilogue(scale, c, _c_terms(fmt, a, b, C.array)))}

    guarantee = "exact" if out.kind == "int" else "correctly_rounded"
    return Result(op, guarantee, {name: D}, {name: exact_array}, tolerance, backward, normal)


def _epilogue(scale: _Scalars, C: Array, c_terms: int = _TERMS_HELD, c_radius: Array | None = None, c_marks: list | None = None,
              flush_format=None) -> tuple:
    """The product kernel's epilogue for gemm (``kernel_epilogue``)."""
    return kernel_epilogue(scale.alpha, scale.beta, C if scale.has_C else None, c_terms, c_radius, c_marks, flush_format)


def _c_terms(compute, A: Array, B: Array, C: Array) -> int:
    """How beta c enters the sum with a unit beta (``_terms``): exact where the compute format holds A's and B's
    formats, a value of it where it holds C's (a wider C of shape (), taken by value, need not be), else rounded."""
    return _terms(compute, C.dtype, True, promoted_format([A.dtype, B.dtype]))


@ieee
def _gemm_backward(A, B, C, grad, *, alpha=1, beta=1, reduce_batch: bool = False, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
                   overflow: str | None = None, workers: int | None = None) -> Result:
    """Gradients of ``gemm(A, B, C, alpha=alpha, beta=beta)`` for the output gradient ``grad``.

    ``grad_A = conj(alpha) grad @ B^H``, ``grad_B = conj(alpha) A^H @ grad`` and
    ``grad_C = conj(beta) grad``, each summed over the copies that broadcasting (and
    ``reduce_batch``) made of its operand, in that operand's shape and dtype. grad_A and
    grad_B are gemm calls with ``beta = 0`` on matmul's backward operands, so each element
    is one exact value rounded once; PyTorch's product followed by its scaling is covered.
    grad_C is one product per element, as ``add``'s gradients: its default compute format is
    grad's, and it takes their zero signs, including where a complex ``beta`` other than 1
    differs from PyTorch. ``grad`` must have
    the output's shape. ``A``, ``B``, ``C`` and ``grad`` may also be earlier results.
    """
    problem = _Problem(A, B, C, alpha, beta, reduce_batch, dtype, rounding, overflow)
    fmt = problem.fmt
    G = grad_operand("gemm", grad, problem.shape, fmt)
    base = promoted_format([G.array.dtype, fmt])
    sizes = problem.sizes
    full = G
    if sizes.out_shape != problem.shape:  # reduce_batch: every batch's gradient is grad
        size = G.array.size
        full = G.take([p % size if size else 0 for p in range(math.prod(sizes.out_shape))], sizes.out_shape)
    (left, right), (left_b, right_b) = backward_operands(*problem.unfolded, full, sizes)
    conj_alpha, conj_beta = conjugate_scalar(problem.alpha), conjugate_scalar(problem.beta)
    A_given, B_given, C_given = problem.given
    gradients = {}
    for name, (X, Y), given in (("grad_A", (left, right), A_given), ("grad_B", (left_b, right_b), B_given)):
        inner = _gemm("gemm.backward", X, Y, _zeros(X, Y), conj_alpha, _zero(fmt), output_format(dtype, given.dtype), rounding, overflow,
                      base, "g", workers=workers)
        gradients[name] = inner.reshape(given.shape)

    def column(gather, width):
        return repeated_operand(conj_beta, G.array.dtype.kind, width, G.array.dtype)

    gradients["grad_C"] = gradient("gemm.backward", G, column, C_given, problem.shape, G.array.dtype, dtype, rounding, overflow, workers,
                                   exact_terms=is_unit(conj_beta), coefficients=True)
    return combined("gemm.backward", gradients)


gemm.backward = _gemm_backward


def _zero(fmt):
    return ComplexValue(FloatValue.zero(), FloatValue.zero()) if fmt.kind == "complex" else FloatValue.zero()


def _zeros(X: Operand, Y: Operand) -> Operand:
    """A C of zero bytes for a product with ``beta = 0`` (only its dtype is read)."""
    return Operand(Array.zeros(X.array.dtype, MatmulShape(X.array, Y.array).out_shape))


class GemmMMA(Algorithm):
    """A tensor-core gemm (``proofs/gemm.md``, Tensor cores; ``proofs/foundations.md``, Lemma T):
    ``matmul_mma``'s products, then alpha applied once and beta C added in a float32 epilogue
    rounding to nearest, then the rounding to the output format. Real inputs: A and B float16,
    bfloat16, or float32 holding tf32 values (the tensor cores multiply them in their own
    promoted format, whatever C's is), with float32 A and B arrays, not earlier results; C in
    a format float32 holds, as the epilogue holds it, and it may be any earlier result;
    alpha and beta must be float32 values; each batch's products on their
    own (it takes no ``reduce_batch``). No ``run()``; ``standard="any_order"`` gives the same bound as the normal standard;
    ``flush_to_zero=True`` raises ValueError, as tensor-core steps keep subnormal values."""

    op = "gemm"

    def __init__(self, step: MMAStep, *, reduction: Format | str | None = None) -> None:
        self.step = _mma.check_step(step)
        self.reduction = _mma.check_reduction(reduction, "gemm_mma")
        self.name, description = _mma.describe("gemm_mma", self.step, self.reduction)
        self.description = description + ", and a float32 epilogue"

    def _problem(self, A, B, C, alpha, beta, dtype, rounding: Rounding, overflow, standard: str, flush_to_zero: bool) -> _Problem:
        """gemm's problem for these inputs, after the checks that read no operand; each query builds it once."""
        check_standard(standard)
        _mma.check_unflushed(flush_to_zero)
        return _Problem(A, B, C, alpha, beta, False, dtype, rounding, overflow)

    def _bounds(self, problem: _Problem, rounding: Rounding, overflow, standard: str, workers: int | None = None):
        """Lemma T's gemm bound per element."""
        fmt = require_real_float(problem.fmt, self.name)
        # the tensor cores multiply A and B in their own promoted format, whatever C's is: it decides the inputs' checks
        # and Lemma T's slack (gemm.md, Tensor cores)
        own = [X.array.dtype for X in problem.sources[:2]]
        ab = promoted_format(own)
        inputs = [X.converted(ab) for X in problem.sources[:2]]
        _mma.check_inputs(self.name, inputs, own)
        c_format = problem.sources[2].array.dtype
        if not holds(float32, c_format):
            raise TypeError(f"{self.name} holds C in its float32 epilogue: C must be in a format float32 holds, got {c_format.name}")
        out = require_real_float(problem.out, self.name)
        options = encode_options(out, rounding, overflow)
        (am, ae), (bm, be) = _scaled(problem.alpha), _scaled(problem.beta)
        for name, (v, x) in (("alpha", (am, ae)), ("beta", (bm, be))):
            if not _in_float32(v, x):
                raise ValueError(f"{self.name} needs {name} to be a float32 value, as the kernel holds it")
        sizes = problem.sizes
        X, Y, Z = problem.operands()
        has_P, has_C = am != 0 and sizes.k > 0, bm != 0
        radii = (X.radius(None, None, False, standard, workers), Y.radius(None, None, False, standard, workers)) if has_P else (None, None)
        c_radius = Z.radius(None, None, False, standard, workers) if has_C else None
        marks_a, marks_b = (X.marks(None, None, False, standard, workers), Y.marks(None, None, False, standard, workers)) if has_P else (None, None)
        marks_c = Z.marks(None, None, False, standard, workers) if has_C else None
        a, b, c = X.array, Y.array, Z.array
        mma = MMA(self.step, self.reduction, _mma.slack(ab, *(_mma.extent(X.array, r) for X, r in zip(inputs, radii))))
        reference = _gemm("gemm", X, Y, Z, problem.alpha, problem.beta, out, rounding, overflow, fmt, "D", workers=workers)._single()
        w = bound_precision(out)
        G, H = mma_terms(mma, sizes.k, w, extra=2, alpha=(am, ae))
        product_terms = mma_terms(mma, sizes.k, w)
        # Lemma T with the epilogue: the products' stage in its own units, the epilogue's in the output's
        c_marks = None if marks_c is None else [ALL | mark & ANY if mark else 0 for mark in marks_c]
        epilogue = kernel_epilogue((am, ae), (bm, be), c if has_C else None, _TERMS_HELD, c_radius, c_marks)
        reduction = None if self.reduction is None else self.reduction._layout
        return kernel_bounds(sizes, a, b, out, float32, rounding, reference, (marks_a, marks_b), standard="any_order",
                             overflow=options["overflow"], epilogue=epilogue, w=w,
                             mma=(product_terms[0], product_terms[1], reduction, G, H), radii=radii, workers=workers)

    @ieee
    def tolerance(self, A, B, C, *, alpha=1, beta=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                  standard: str = "normal", flush_to_zero: bool = False, workers: int | None = None) -> Array:
        """Worst-case ``|output - gemm(A, B, C, alpha=alpha, beta=beta).rounded|`` per element over
        this algorithm's finite outputs, as ``Result.tolerance`` describes; ``overflow`` is the
        implementation's overflow mode, as in ``gemm``."""
        problem = self._problem(A, B, C, alpha, beta, dtype, rounding, overflow, standard, flush_to_zero)
        return self._bounds(problem, rounding, overflow, standard, workers).tolerance

    @ieee
    def finite_range(self, A, B, C, *, alpha=1, beta=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                     standard: str = "normal", flush_to_zero: bool = False, workers: int | None = None) -> tuple[Array, Array]:
        """The lowest and highest finite output per element, as ``Result.finite_range`` describes."""
        problem = self._problem(A, B, C, alpha, beta, dtype, rounding, overflow, standard, flush_to_zero)
        found = self._bounds(problem, rounding, overflow, standard, workers)
        return found.low, found.high

    @ieee
    def may_be(self, kind: str, A, B, C, *, alpha=1, beta=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
               overflow: str | None = None, standard: str = "normal", flush_to_zero: bool = False, workers: int | None = None) -> Array:
        """Where this algorithm may output ``kind`` (``"+inf"``, ``"-inf"`` or ``"nan"``), as
        ``Result.may_be`` describes."""
        check_kind(kind)
        problem = self._problem(A, B, C, alpha, beta, dtype, rounding, overflow, standard, flush_to_zero)
        return self._bounds(problem, rounding, overflow, standard, workers).may_be(kind)

    @ieee
    def covers(self, output, A, B, C, *, alpha=1, beta=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
               overflow: str | None = None, standard: str = "normal", flush_to_zero: bool = False, workers: int | None = None) -> Array:
        """Where this algorithm can produce ``output``, as ``Result.covers`` describes."""
        problem = self._problem(A, B, C, alpha, beta, dtype, rounding, overflow, standard, flush_to_zero)
        output = checked_output(output, problem.out, problem.shape)
        return self._bounds(problem, rounding, overflow, standard, workers).covers(output)


def _in_float32(m: int, e: int) -> bool:
    """Whether ``m * 2**e`` is a float32 value: float32's own nearest value to it (``csrc/formats.c``) is itself."""
    v = float32.decode(float32.encode_scaled(m, e))
    return v.is_finite and v.to_fraction() == Fraction(m) * Fraction(2) ** e


def gemm_mma(step: MMAStep, *, reduction: Format | str | None = None) -> GemmMMA:
    """A tensor-core gemm with ``step``'s parameters (``GemmMMA``)."""
    return GemmMMA(step, reduction=reduction)
