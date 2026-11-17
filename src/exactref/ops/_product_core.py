"""matmul's product, the core every product-based operation calls (matmul, multiply, add, sum, gemm and their backwards)."""

from __future__ import annotations

import functools
import math
import operator
from array import array as _pyarray

from .. import _native
from ..array import Array
from ..exact import ExactArray
from ..formats import _INT_OVERFLOW_CODES, FloatFormat, Rounding, _float_codes, _part_format, complex128, float64, uint8
from ..result import Bounds, Result, exact_bounds
from ..values import ComplexValue, FloatValue
from ._operands import Operand, holds
from ._rounding import bound_precision, compute_format, encode_options, output_format
from ._shapes import broadcast_index, broadcast_shapes, iter_indices, unravel


class MatmulShape:
    """Shapes of a matmul after 1-D promotion and batch broadcasting."""

    def __init__(self, A: Array, B: Array) -> None:
        if A.ndim == 0 or B.ndim == 0:
            raise ValueError("matmul operands must have at least one dimension")
        self.a_shape = (1,) + A.shape if A.ndim == 1 else A.shape
        self.b_shape = B.shape + (1,) if B.ndim == 1 else B.shape
        self.m, self.k = self.a_shape[-2:]
        k2, self.n = self.b_shape[-2:]
        if self.k != k2:
            raise ValueError(f"matmul inner dimensions differ: {A.shape} @ {B.shape}")
        self.batch = broadcast_shapes(self.a_shape[:-2], self.b_shape[:-2])
        self.out_shape = self.batch + ((self.m,) if A.ndim > 1 else ()) + ((self.n,) if B.ndim > 1 else ())

    @staticmethod
    def _start(batch_index, shape) -> int:
        """Offset of the first element of the matrix that ``batch_index`` selects in a row-major ``shape``."""
        offset = 0
        for i, n in zip(broadcast_index(batch_index, shape[:-2]), shape[:-2]):
            offset = offset * n + i
        return offset * shape[-2] * shape[-1]



def product(op: str, A: Operand, B: Operand, out, rounding: Rounding, overflow, compute_base, *, name: str = "C", workers: int | None = None,
            backward=None, exact_terms: bool = False, unrounded_terms=None, coefficients: bool = False, few_terms: bool = False,
            elementwise: bool = False, shape: tuple | None = None, reference: tuple | None = None) -> Result:
    """``matmul``'s result for operands of one kind but possibly different formats: ``out``
    is the output format, ``compute_base`` the format whose default compute format applies,
    and ``exact_terms`` says that every product is a value of A's format, so it is not
    rounded where the compute format holds that format (``proofs/foundations.md``, Lemma U);
    ``unrounded_terms``, the format of the operation's array operands, says that every product
    is an exact value the model does not round on its own where the compute format holds that
    format, so a scalar taken by value need not be one of the compute format's values (``add``'s
    terms with a unit alpha, ``proofs/add.md``). A complex output
    whose terms include a non-finite part takes IEEE rules in each part separately, as
    PyTorch's elementwise and batched complex products do; with ``coefficients``, B holds exact coefficients, and
    a zero part of a nonzero coefficient adds no term (``proofs/matmul.md``). With
    ``few_terms`` (one-row, one-column products, as elementwise operations make), an element
    with exact inputs and at most two nonzero terms per part, without flush, gets the exact
    set of values the implementations compute (``proofs/foundations.md``, Lemma F). With
    ``elementwise``, no element is a reduction (each is one expression, as ``multiply``'s
    and ``add``'s are), so an exact zero takes the sign IEEE arithmetic gives it instead of
    the sum rule (``proofs/matmul.md``). The result is called ``op`` and its output
    ``name``; ``backward`` is its backward. ``shape`` is the operation's output shape where
    it reshapes the product's, for an error's element index. ``reference``, where given, is
    the ``(rounded, exact)`` pair of another expression of the product's size, which the
    result takes in place of the product's own, so the bounds of the product's model are
    measured from it (a held-scalar algorithm's: the operation's reference on the scalars as
    given, ``proofs/multiply.md``, Held scalars). The operations built on matmul are proxies
    of this call."""
    a, b = A.array, B.array
    sizes = MatmulShape(a, b)
    options = encode_options(out, rounding, overflow)
    if reference is None:
        C, exact_array = kernel_reference(sizes, a, b, out, options, elementwise, coefficients=coefficients, workers=workers, shape=shape)
    else:
        C, exact_array = (x.reshape(sizes.out_shape) for x in reference)

    def tolerance(compute, ulps, flush, standard, workers=None) -> dict[str, Bounds]:
        radii = (A.radius(compute, ulps, flush, standard, workers), B.radius(compute, ulps, flush, standard, workers))
        if out.kind == "int":
            if radii == (None, None):
                return {name: exact_bounds(C)}
            return {name: kernel_int_bounds(sizes, a, b, C, options["overflow"], radii, workers)}
        fmt = compute_format(compute, compute_base)
        marks = (A.marks(compute, ulps, flush, standard, workers), B.marks(compute, ulps, flush, standard, workers))
        exact = _terms(fmt, a.dtype, exact_terms, unrounded_terms)
        # the few-term rule where it applies, else the general bound, element by element; a
        # coefficient product's uncertain inputs (in A only) reach the parts the coefficient rule
        # lets them (product_reach)
        few = few_terms and not flush and sizes.m == 1 and sizes.n == 1
        reach = None
        if coefficients and (radii != (None, None) or marks != (None, None)):
            reach, marks = (b"" if marks[0] is None else bytes(marks[0])), (None, None)
        return {name: kernel_bounds(sizes, a, b, out, fmt, rounding, C, marks, flush=flush, standard=standard, exact_terms=exact,
                                    overflow=options["overflow"], few=few, radii=radii, coefficients=coefficients, workers=workers, reach=reach)}

    def normal(compute, flush, workers=None):
        if out.kind == "int":
            return {name: C}
        fmt = compute_format(compute, compute_base)
        x, y = A.normal(compute, flush, workers), B.normal(compute, flush, workers)
        terms = _terms(fmt, a.dtype, exact_terms, unrounded_terms)
        return {name: kernel_normal(sizes, x, y, out, fmt, options, flush, coefficients, workers, exact_terms=terms)}

    guarantee = "exact" if out.kind == "int" else "correctly_rounded"
    return Result(op, guarantee, {name: C}, {name: exact_array}, tolerance, backward, normal)


# How the product kernel takes a sum's terms (csrc/exactref.h, TERMS_*): each rounded to the compute format on its own,
# each a value of the compute format, or each an exact value the model does not round, which the compute format need
# not hold
_TERMS_ROUNDED, _TERMS_HELD, _TERMS_EXACT = 0, 1, 2


def _terms(compute, dtype, exact_terms: bool, unrounded_terms) -> int:
    """How the product kernel takes the terms of A's format ``dtype`` in the ``compute`` format (``product``'s
    ``exact_terms`` and ``unrounded_terms``): a compute format narrower than the array operands' rounds every term."""
    if not exact_terms and unrounded_terms is None:
        return _TERMS_ROUNDED
    if holds(compute, _part_format(dtype)):
        return _TERMS_HELD
    if unrounded_terms is not None and holds(compute, _part_format(unrounded_terms)):
        return _TERMS_EXACT
    return _TERMS_ROUNDED


_TILE = 1 << 16  # output elements per call into the product kernel, spread over the workers


def _bases(shape: tuple, sizes: MatmulShape, side: int):
    """How the output batches read an operand's matrices (its batch dimensions ``shape``) for
    the product kernel: a count c where batch t reads matrix t mod c (the operand's batch
    dimensions, past any leading ones, are the output's last ones), else per batch the row
    where its matrix starts, as native int64 bytes."""
    count = math.prod(shape)
    dims = tuple(shape)
    while dims and dims[0] == 1:
        dims = dims[1:]
    batch = sizes.batch
    if count <= 1 or tuple(batch[len(batch) - len(dims):]) == dims:
        return count
    size, rows = (sizes.m * sizes.k, sizes.m) if side == 0 else (sizes.k * sizes.n, sizes.n)
    return _pyarray("q", [starts[side] // size * rows for starts in batch_starts(sizes)]).tobytes()


def _kernel_operands(sizes: MatmulShape, A: Array, B: Array, radii: tuple = (None, None), flush_format=None, coefficients: bool = False, workers=None) -> tuple:
    """The product kernel's operands for every output batch: A's matrices by rows and B's
    transposed (row j of one of its matrices is that matrix's column j), each read once, and
    how the output batches read them (``_bases``); then the operands' radii read the same way
    (None where an operand is exact), widened to cover 0 where ``flush_format`` may flush a
    stored value. Returns ``(a, parts, a_bases, b_bases)``, with per output
    part ``(b, radii)``: for complex values A's parts interleaved make a row of 2k terms, and
    B's operand of each output part pairs them with (b_re, -b_im) or (b_im, b_re)
    (``_native.operand_parts``; ``coefficients``: B holds exact coefficients)."""
    m, k, n = sizes.m, sizes.k, sizes.n
    ta, tb = math.prod(sizes.a_shape[:-2]), math.prod(sizes.b_shape[:-2])
    complex_ = A.dtype.kind == "complex"
    terms = 2 * k if complex_ else k
    if complex_:  # A's parts as 2k columns; B's as rows 2j and 2j + 1 of its transpose
        shapes = ((ta * m, 2 * k, 2 * k, 1, max(ta * m, 1), 0), (tb * 2 * n, k, 1, 2 * n, max(2 * n, 1), 2 * k * n))
    else:
        shapes = ((ta * m, k, k, 1, max(ta * m, 1), 0), (tb * n, k, 1, n, max(n, 1), k * n))
    # The wide main range (two-word slots, 256-bit sums) for float64's 53-bit values, and with
    # radii, which are float64 values, so that every product of the operands and radii is wide.
    wide = radii != (None, None) or any(isinstance(_part_format(X.dtype), FloatFormat) and _part_format(X.dtype).precision > 32 for X in (A, B))
    found = [_native.operand(bytes(X.buffer), X.byteorder == "big", _part_format(X.dtype)._layout, 0, *shape, terms, wide, workers=workers)
             for X, shape in zip((A, B), shapes)]
    for X, radius, shape in zip((A, B), radii, shapes):
        if radius is None:
            found.append(None)
            continue
        component = _part_format(X.dtype)
        widened = _native.radius_parts(component._layout, bytes(X.buffer), X.byteorder == "big", float64._layout, bytes(radius.buffer),
                                       radius.byteorder == "big", 0, X.size * X.dtype.components, None if flush_format is None else flush_format.emin)
        found.append(_native.operand(widened, False, float64._layout, 0, *shape, terms, wide, workers=workers))
    a, b, ra, rb = found
    if complex_:
        bs = _native.operand_parts(b, coefficients)
        rbs = (None, None) if rb is None else _native.operand_parts(rb, False)
    else:
        bs, rbs = (b,), (rb,)
    parts = [(bp, None if ra is rbp is None else (ra, rbp)) for bp, rbp in zip(bs, rbs)]
    return a, parts, _bases(sizes.a_shape[:-2], sizes, 0), _bases(sizes.b_shape[:-2], sizes, 1)


def _part_bytes(data: bytes, size: int, part: int, parts: int) -> bytes:
    """Part ``part`` of every element of interleaved parts, each ``size`` bytes."""
    if parts == 1:
        return data
    return _pyarray(_WORDS[size], data)[part::parts].tobytes()


_WORDS = {1: "B", 2: "H", 4: "I", 8: "Q"}


def _tiles(sizes: MatmulShape) -> list:
    """The output elements in runs of about ``_TILE``, each one call into the kernel: whole rows
    of one batch where a batch has that many elements (the kernel then reads each column of B
    once for all of a tile's rows), else runs across batches."""
    m, n = sizes.m, sizes.n
    batches, mn = math.prod(sizes.batch), m * n
    if mn >= _TILE:
        rows = max(1, _TILE // n)
        return [(t * mn + i0 * n, t * mn + min(m, i0 + rows) * n) for t in range(batches) for i0 in range(0, m, rows)]
    total = batches * mn
    return [(o0, min(o0 + _TILE, total)) for o0 in range(0, total, _TILE)]


def _part_calls(parts: list, epilogue) -> list:
    """Per output part, B's operands, their radii and the epilogue of its calls: gemm's complex
    epilogue reads both of P's parts (its channels), so each part's calls take B's parts' operands
    as a pair, and the epilogue names the part."""
    if epilogue is None or len(parts) == 1:
        return [(b, radii, epilogue) for b, radii in parts]
    bs = tuple(b for b, _ in parts)
    radii = None
    if parts[0][1] is not None:
        ra, rb = parts[0][1][0], None if parts[0][1][1] is None else tuple(r[1] for _, r in parts)
        radii = (ra, rb)
    return [(bs, radii, epilogue[:-1] + (p,)) for p in range(2)]


def kernel_reference(sizes: MatmulShape, A: Array, B: Array, out, options: dict[str, Rounding | str], elementwise: bool, epilogue=None, coefficients: bool = False,
                  workers: int | None = None, shape: tuple | None = None) -> tuple[Array, ExactArray]:
    """The correctly rounded output and its exact entries, by the product kernel; ``epilogue``
    is gemm's (``kernel_epilogue``). An integer output outside its format with overflow
    ``"error"`` raises OverflowError naming the element, as an index in ``shape`` (default the
    product's output shape), and its value."""
    a, parts, a_bases, b_bases = _kernel_operands(sizes, A, B, coefficients=coefficients, workers=workers)
    component = _part_format(out)
    rnd, ovf = (0, _INT_OVERFLOW_CODES[options["overflow"]]) if out.kind == "int" else _float_codes(component, options["rounding"], options["overflow"])
    tiles = _tiles(sizes)
    found = _native.run([functools.partial(_native.product_reference, a, b, a_bases, b_bases, sizes.m, sizes.n, o0, o1, component._layout, rnd, ovf,
                                           elementwise, part_epilogue) for b, _, part_epilogue in _part_calls(parts, epilogue) for o0, o1 in tiles],
                        workers)
    for got in found:
        if got[0] is None:
            _, position, value = got
            index = unravel(position, sizes.out_shape if shape is None else shape)
            raise OverflowError(f"output element {index} is {value}, outside the {out.name} range [{out.min_value}, {out.max_value}]")
    per_part = _per_part(found, len(tiles), len(parts), 3)
    if len(parts) == 1:
        encodings, flags, midpoints = per_part[0]
    else:
        (er, fr, mr), (ei, fi, mi) = per_part
        encodings, flags, midpoints = (_native.interleave(er, ei, component.itemsize), _native.interleave(fr, fi, 1),
                                       _native.interleave(mr, mi, 24))
    return Array(out, sizes.out_shape, encodings), ExactArray._from_storage(sizes.out_shape, len(parts), flags, midpoints)


def _per_part(found: list, tiles: int, parts: int, streams: int) -> list[list[bytes]]:
    """Each part's ``streams`` byte strings, joined over its ``tiles`` tile calls' results in
    ``found`` (the calls part by part, each returning one byte string per stream)."""
    return [[b"".join(x) for x in zip(*found[p * tiles : (p + 1) * tiles])] if tiles else [b""] * streams for p in range(parts)]


def kernel_int_bounds(sizes: MatmulShape, A: Array, B: Array, reference: Array, overflow: str, radii: tuple, workers: int | None = None,
                       epilogue=None) -> Bounds:
    """An integer product's bounds by the product kernel (``integer_bounds``' rule): each
    element's exact values within the radii's deviation of its exact sum; ``epilogue`` is
    gemm's (``kernel_epilogue``), whose C may be an earlier result too."""
    a, parts, a_bases, b_bases = _kernel_operands(sizes, A, B, radii, workers=workers)
    ((b, read_radii),) = parts
    out = reference.dtype
    data, size = bytes(reference.buffer), out.itemsize
    found = _native.run([functools.partial(_native.product_int_bounds, a, b, a_bases, b_bases, sizes.m, sizes.n, o0, o1, out._layout, float64._layout,
                                           _INT_OVERFLOW_CODES[overflow], data[o0 * size : o1 * size], read_radii, epilogue) for o0, o1 in _tiles(sizes)],
                        workers)
    ((tolerance, low, high, flags),) = _per_part(found, len(found), 1, 4)
    shape = sizes.out_shape
    return Bounds(Array(float64, shape, tolerance), Array(out, shape, low), Array(out, shape, high), Array(uint8, shape, flags))


def kernel_run(sizes: MatmulShape, A: Array, B: Array, out, options: dict, accumulator, accumulator_rounding: Rounding, method: int, block: int,
                fma: bool, workers: int | None = None) -> Array:
    """A named algorithm's outputs by the product kernel (``MatmulAlgorithm.run``,
    ``MatmulSlots16.run``): method 1 sequential, 2 pairwise, 3 blocked, 4 exact accumulation, 5
    the 16-slot algorithm, each rounding emulated exactly."""
    a, ((b, _),), a_bases, b_bases = _kernel_operands(sizes, A, B, workers=workers)
    rnd, ovf = _float_codes(out, options["rounding"], options["overflow"])
    acc_rnd, _ = _float_codes(accumulator, accumulator_rounding, "ieee")
    found = _native.run([functools.partial(_native.product_run, a, b, a_bases, b_bases, sizes.m, sizes.n, o0, o1, accumulator._layout, acc_rnd, method,
                                           block, fma, out._layout, rnd, ovf) for o0, o1 in _tiles(sizes)], workers)
    return Array(out, sizes.out_shape, b"".join(found))


def kernel_normal(sizes: MatmulShape, A: Array, B: Array, out, compute, options: dict, flush: bool, coefficients: bool,
                   workers: int | None = None, epilogue=None, exact_terms: int = _TERMS_ROUNDED) -> Array:
    """The normal answer by the product kernel (``proofs/foundations.md``, Lemma E): every
    rounding emulated exactly, part by part for complex values; ``epilogue`` is gemm's
    (``kernel_epilogue``); ``exact_terms`` as ``_terms`` gives it, with ``_TERMS_EXACT`` no term
    rounded on its own."""
    a, parts, a_bases, b_bases = _kernel_operands(sizes, A, B, coefficients=coefficients, workers=workers)
    component = _part_format(out)
    rnd, ovf = _float_codes(component, options["rounding"], options["overflow"])
    tiles = _tiles(sizes)
    found = _native.run([functools.partial(_native.product_normal, a, b, a_bases, b_bases, sizes.m, sizes.n, o0, o1, compute._layout, flush,
                                           component._layout, rnd, ovf, part_epilogue, exact_terms)
                         for b, _, part_epilogue in _part_calls(parts, epilogue) for o0, o1 in tiles], workers)
    per_part = [b"".join(found[p * len(tiles) : (p + 1) * len(tiles)]) for p in range(len(parts))]
    encodings = per_part[0] if len(parts) == 1 else _native.interleave(per_part[0], per_part[1], component.itemsize)
    return Array(out, sizes.out_shape, encodings)


def kernel_bounds(sizes: MatmulShape, A: Array, B: Array, out, compute, rounding: Rounding, reference: Array, marks: tuple, *, flush: bool = False,
                  standard: str = "normal", exact_terms: int = _TERMS_ROUNDED, overflow: str = "ieee", epilogue=None, order=None,
                  w: int | None = None, mma=None, few: bool = False, radii: tuple = (None, None), coefficients: bool = False, workers: int | None = None,
                  reach=None) -> Bounds:
    """The tolerance's bounds by the product kernel (``proofs/matmul.md``, The tolerance);
    ``exact_terms`` as ``_terms`` gives it; ``marks`` (``Operand.marks``) reach every element
    of their row of A or column of B.
    ``epilogue`` is gemm's (``kernel_epilogue``); ``order`` a named algorithm, ``(kind,
    block)``: 1 sequential, 2 pairwise, 3 blocked (``block`` its block size), 4 exact
    accumulation, 5 the 16-slot algorithm (``block`` its accumulator's rounding); ``mma`` Lemma
    T's ``(G, H, reduction layout)``, followed by the epilogue stage's ``G, H`` with gemm's
    epilogue; ``w`` the bounds' working precision. With ``few``
    (one-row, one-column products), an element with exact inputs that the few-term rule covers
    gets the exact set of values the implementations compute (``proofs/foundations.md``, Lemma F).
    ``radii`` are the operands' radii when they are earlier results (Results as inputs),
    widened for ``flush``.
    ``reach`` is a coefficient product's A marks per part as bytes (``b""`` for none), its
    uncertain inputs reaching each part as the coefficient rule lets them (``product_reach``)."""
    a, parts, a_bases, b_bases = _kernel_operands(sizes, A, B, radii, compute if flush else None, coefficients, workers=workers)
    m, k, n = sizes.m, sizes.k, sizes.n
    count = len(parts)  # the parts of an element: its terms' marks reach both
    marks_a, marks_b = marks
    rows = columns = None
    if marks_a is not None:
        width = k * count
        rows = (bytes(functools.reduce(operator.or_, marks_a[r * width : (r + 1) * width], 0) for r in range(len(marks_a) // width)) if k
                else bytes(len(marks_a) // count))
    if marks_b is not None:
        columns = bytes(functools.reduce(operator.or_, (marks_b[(t * k * n + q * n + j) * count + p] for q in range(k) for p in range(count)), 0)
                        for t in range(len(marks_b) // max(k * n * count, 1)) for j in range(n))
    component = _part_format(out)
    rnd, ovf = _float_codes(component, rounding, overflow)
    w = bound_precision(compute, component) if w is None else w
    data, size = bytes(reference.buffer), component.itemsize
    tiles = _tiles(sizes)
    calls = []
    for p, (b, read_radii, part_epilogue) in enumerate(_part_calls(parts, epilogue)):
        refs = _part_bytes(data, size, p, count)
        calls += [functools.partial(_native.product_bounds, a, b, a_bases, b_bases, m, n, o0, o1, compute._layout, flush, standard == "normal",
                                    exact_terms, component._layout, float64._layout, rnd, ovf, refs[o0 * size : o1 * size], rows, columns, w,
                                    part_epilogue, order, mma, few, read_radii, reach) for o0, o1 in tiles]
    found = _native.run(calls, workers)
    per_part = _per_part(found, len(tiles), count, 4)
    shape = sizes.out_shape
    if count == 1:
        tolerance, low, high, flags = per_part[0]
        return Bounds(Array(float64, shape, tolerance), Array(out, shape, low), Array(out, shape, high), Array(uint8, shape, flags))
    (tr, lr, hr, fr), (ti, li, hi, fi) = per_part
    return Bounds(Array(complex128, shape, _native.interleave(tr, ti, 8)), Array(out, shape, _native.interleave(lr, li, size)),
                  Array(out, shape, _native.interleave(hr, hi, size)), Array(uint8, shape, _native.pair_flags(fr, fi)))


def kernel_epilogue(alpha: tuple, beta: tuple, C: Array | None, c_terms: int = _TERMS_HELD, c_radius: Array | None = None, c_marks: list | None = None,
                    flush_format=None) -> tuple[tuple, tuple, bytes | None, bool, tuple, int, bytes | None, bytes | None, tuple, int]:
    """gemm's epilogue for the product kernel: alpha and beta as ``(m, e)``, or for complex values
    ``(re, im)`` pairs of them (the kernel makes their channels, ``proofs/gemm.md``), C's elements
    (row-major over the output, parts interleaved; None when beta is 0), how beta c enters the sum
    with a unit beta (``_terms``: a value of the compute format, an exact value it need not hold, or
    rounded under a compute format narrower than the arrays'), and when C is an earlier result its
    radii (widened to cover 0 where
    ``flush_format`` may flush a stored value) and marks (``Operand.marks``, per part). The last
    field is the output part a call computes (``_part_calls``)."""
    component = _part_format(C.dtype) if C is not None else None
    data = (bytes(C.buffer), C.byteorder == "big", component._layout) if C is not None else (None, False, (8, True))
    radius = None
    if C is not None and c_radius is not None:
        radius = _native.radius_parts(component._layout, data[0], data[1], float64._layout, bytes(c_radius.buffer), c_radius.byteorder == "big", 0,
                                      C.size * C.dtype.components, None if flush_format is None else flush_format.emin)
    marks = None if C is None or c_marks is None else bytes(c_marks)
    return alpha, beta, data[0], data[1], data[2], c_terms, radius, marks, float64._layout, 0


def backward_operands(A: Operand, B: Operand, G: Operand, sizes: MatmulShape) -> tuple[tuple[Operand, Operand], tuple[Operand, Operand]]:
    """The backward's two products (``proofs/matmul.md``, Backward) as operand pairs:
    grad_A = G' @ B' and grad_B = A' @ G'', with A's and B's shapes' batches; the batch
    copies that broadcasting made of an operand are folded into the contraction."""
    m, k, n = sizes.m, sizes.k, sizes.n
    # Element positions, gathered below the way the operands' encodings are.
    a, b, g = range(A.array.size), range(B.array.size), range(G.array.size)
    a_batch, b_batch = sizes.a_shape[:-2], sizes.b_shape[:-2]

    # grad_A[alpha] = [G[beta_1] ... G[beta_F]] @ [conj(B[beta_1])^T; ...; conj(B[beta_F])^T]
    folds = _folds(sizes.batch, a_batch)
    width = n * len(folds[0]) if folds else 0
    left, right = [], []
    for betas in folds:
        g_starts = [index * m * n for index, _ in betas]
        b_starts = [MatmulShape._start(beta, sizes.b_shape) for _, beta in betas]
        for i in range(m):
            for start in g_starts:
                left.extend(g[start + i * n : start + i * n + n])
        for start in b_starts:
            for j in range(n):
                right.extend(b[start + j : start + k * n : n])
    grad_A = (G.take(left, a_batch + (m, width)), B.take(right, a_batch + (width, k), conjugate=True))

    # grad_B[beta_B] = [conj(A[beta_1])^T ... conj(A[beta_F])^T] @ [G[beta_1]; ...; G[beta_F]]
    folds = _folds(sizes.batch, b_batch)
    width = m * len(folds[0]) if folds else 0
    left, right = [], []
    for betas in folds:
        a_starts = [MatmulShape._start(beta, sizes.a_shape) for _, beta in betas]
        for r in range(k):
            for start in a_starts:
                left.extend(a[start + r : start + m * k : k])
        for index, _ in betas:
            right.extend(g[index * m * n : (index + 1) * m * n])
    grad_B = (A.take(left, b_batch + (k, width), conjugate=True), G.take(right, b_batch + (width, n)))
    return grad_A, grad_B


def _folds(batch: tuple, operand_batch: tuple) -> list[list]:
    """For each index of an operand's ``operand_batch`` (row-major), the indices of ``batch``
    broadcasting onto it, as ``(row-major position, index)`` pairs; all lists have one length."""
    groups: dict = {}
    for position, index in enumerate(iter_indices(batch)):
        groups.setdefault(broadcast_index(index, operand_batch), []).append((position, index))
    return [groups.get(index, []) for index in iter_indices(operand_batch)]


def batch_starts(sizes: MatmulShape) -> list[tuple[int, int]]:
    """For each output batch in row-major order, where its matrix of A and of B start."""
    return [(MatmulShape._start(index, sizes.a_shape), MatmulShape._start(index, sizes.b_shape)) for index in iter_indices(sizes.batch)]


def reshaped(inner: Result, shape: tuple, backward) -> Result:
    """``inner`` (one output) with its output in ``shape``, under its own operation and output names, and ``backward``."""
    return inner._view(None, shape, backward)


def gradient(op: str, G: Operand, column, target, shape: tuple, compute_base, dtype, rounding, overflow, workers: int | None = None,
              exact_terms: bool = False, coefficients: bool = False) -> Result:
    """For each element of ``target``, the sum over its broadcast copies of ``grad`` times
    the column ``column(gather, width)`` gives: one product whose rows are those copies
    (``coefficients``: the column holds exact coefficients, as ``product`` takes them).
    ``compute_base`` is the format whose default compute format applies: grad's where the
    column holds coefficients (add's gradients and gemm's grad_C), grad's promoted with
    the other factor's where it holds data (multiply's).
    An element is a reduction unless it has one copy and ``target`` has the output's number
    of dimensions: PyTorch sums the dimensions broadcasting adds in front, even of size 1."""
    copies = [[position for position, _ in group] for group in _folds(shape, target.shape)]
    width = len(copies[0]) if copies else 0
    gather = [p for group in copies for p in group]
    out = output_format(dtype, target.dtype)
    elementwise = width == 1 and len(target.shape) == len(shape)
    inner = product(op, G.take(gather, target.shape + (1, width)), column(gather, width), out, rounding, overflow, compute_base, workers=workers,
                    name="g", exact_terms=exact_terms, coefficients=coefficients, few_terms=True, elementwise=elementwise)
    return reshaped(inner, target.shape, None)


def combined(op: str, gradients: dict) -> Result:
    def tolerance(compute, ulps, flush, standard, workers=None):
        return {name: r._bounds_of(compute, ulps, flush, standard, workers) for name, r in gradients.items()}

    def normal(compute, flush, workers=None):
        return {name: r.normal(compute, flush_to_zero=flush, workers=workers) for name, r in gradients.items()}

    guarantee = next(iter(gradients.values())).guarantee
    return Result(op, guarantee, {n: r._single() for n, r in gradients.items()}, {n: r.exact for n, r in gradients.items()}, tolerance,
                  normal=normal, named=True)


def conjugate_scalar(a):
    if isinstance(a, ComplexValue):
        im = a.imag
        return ComplexValue(a.real, FloatValue(im.kind, not im.negative, im.significand, im.exponent))
    return a
