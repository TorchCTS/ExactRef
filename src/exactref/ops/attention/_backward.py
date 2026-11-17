"""Backward of scaled dot-product attention."""

from __future__ import annotations

import functools
import itertools
from array import array as _pyarray
from collections.abc import Callable
from typing import NamedTuple

from ... import _native
from ..._fenv import ieee
from ...array import Array
from ...boundaries import Boundary, OpSpec, Uses, register, required_settings
from ...errors import ReferenceUnavailableError
from ...exact import ExactArray
from ...formats import Format, Rounding, _float_codes, float64, uint8
from ...result import Bounds, Result
from .._operands import operand, unflagged_radius
from .._rounding import bound_precision, compute_format, encode_options, output_format, tolerance_ulps
from .._shapes import unravel
from ._problem import DEFAULT_ULPS, MAX_PRECISION, Problem, check_dropout, kernel_settings, radius_bytes, radius_operand, read_rows, row_of

register(
    OpSpec(
        "scaled_dot_product_attention.backward",
        boundaries={
            "scores": Uses("matmul", "dot products of each query row with each key row (Q K^T)"),
            "scale_factor": Boundary("function", "the scale (1/sqrt(E) unless given) held in the compute format"),
            "logits": Boundary("round", "each score times the scale, plus the float mask if any, rounded once"),
            "softmax": {
                "shift": Boundary("round", "each logit minus its row maximum"),
                "exp": Boundary("function", "exponential of each shifted logit"),
                "sum": Boundary("accumulate", "row sum of the exponentials", applies_to="sum"),
                "probabilities": Boundary("round", "each exponential divided by the row sum (P)"),
            },
            "grad_value": Uses("matmul", "P^T dO: probabilities times output-gradient rows, summed over query rows"),
            "grad_probabilities": Uses("matmul", "dP = dO V^T"),
            "grad_softmax": {
                "row_dot": Boundary("accumulate", "D: each row's sum of P times dP", applies_to="dot"),
                "difference": Boundary("round", "each dP minus its row's D"),
                "grad_scores": Boundary("round", "that difference times P (dS)"),
            },
            "grad_query": Uses("matmul", "dS K"),
            "grad_key": Uses("matmul", "dS^T Q, summed over query rows"),
            "scale_gradients": Boundary("round", "each grad_query and grad_key sum times the scale"),
        },
        guarantee="correctly_rounded",
        limitations=(
            "real floating-point inputs only",
            "dropout_p must be 0 (dropout is random)",
            "no gradient for a float attn_mask",
            "sums over several rows (grad_key, grad_value, and grad_query over broadcast copies of a query row) are "
            "found exact only when every term is rational (keys with weight 0, zero output-gradient rows, rows with "
            "equal weights); other such sums, even exact ones, are refined up to a precision cap that raises "
            "ReferenceUnavailableError",
            "an element within a heavily masked key's weight of a rounding boundary, where in some row the effect of the "
            "least masked such key cancels exactly (as with equal entries in a key column) and a more heavily masked "
            "one's does not, or where a heavily masked key's own effect cancels exactly (its dP equal to the kept keys' "
            "weighted mean, as when its value row equals the only kept key's) and leaves only its product with a less "
            "heavily masked key's weight, is refined up to that cap too; in a sum over rows, such a row's leftover also "
            "hides the other rows' effects",
            "PyTorch's math path multiplies the gradients of the pre-scaled Q and K by the square root of the scale "
            "at the end; with pre-scaled inputs and scale=1, the reference is the gradient before that rounded product",
        ),
    )
)

_TILE_ENTRIES = 1 << 16  # probabilities per tile: a tile has at most this many rows times keys
NAMES = ("grad_query", "grad_key", "grad_value")


@ieee
def scaled_dot_product_attention_backward(
    query,
    key,
    value,
    grad,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
    *,
    dtype: Format | str | None = None,
    rounding: Rounding = Rounding.NEAREST_EVEN,
    overflow: str | None = None,
    workers: int | None = None,
) -> Result:
    """Gradients of ``scaled_dot_product_attention`` for the output gradient ``grad``.

    The arguments are the forward's, with ``grad`` after ``value``; ``grad``
    has the output's shape and the inputs' dtype. Returns ``grad_query``,
    ``grad_key`` and ``grad_value`` with the inputs' shapes: with ``P`` the
    attention weights, ``dP = grad V^T``, ``D_i = sum_j P_ij dP_ij`` and
    ``dS = P * (dP - D)``,

        grad_value = P^T grad,   grad_query = scale * dS K,   grad_key = scale * dS^T Q,

    each summed over the copies that broadcasting or ``enable_gqa`` made of
    its input. Every output is the exact value correctly rounded to ``dtype``,
    which must be floating point, as the inputs are.

    A key a row does not keep (masked, or removed by a -inf logit, as in the
    forward) takes no part in that row's terms, whatever its ``dP``, score or
    value. Over the kept keys non-finite values propagate as IEEE arithmetic
    does on exact values through these formulas, as in PyTorch's math path:
    an infinite or NaN ``dP`` makes ``D`` and the row's ``dS`` non-finite, and
    a NaN row of the forward makes every gradient it touches through its
    other keys NaN. A row with no kept key contributes 0. An implementation
    that computes the terms of the keys a row does not keep (``P = 0`` there)
    meets NaN where one of their factors is not finite or ``dP - D``
    overflows, and so does a kept key whose computed ``P`` may be exactly 0
    times a non-finite output gradient; the tolerance's flags allow that NaN
    there. An exact zero is +0, or -0 when rounding toward
    -inf; an empty sum is +0. Dropout must be 0, and a float mask gets no
    gradient.

    Exactness: ``grad_query`` of a row is decided exactly (a
    Lindemann-Weierstrass test), unless heavily masked keys' effects cancel
    as the limitations describe. ``grad_key``, ``grad_value``, and
    ``grad_query`` summed over broadcast copies of a query row, are found
    exact when every term is rational. Otherwise, and in those cancelling
    cases, refinement stops at a precision cap and raises
    ``ReferenceUnavailableError`` rather than return an uncertain value (the
    accepted cases where an exact result can reach the cap).

    ``tolerance()`` budgets the required boundaries of the gradient
    computation: the forward's scores, scale factor, logits, shift, ``exp``
    within ``ulps`` (default 2) and row sum, then the probabilities ``P``
    (one rounding each), the accumulations of ``dP``, ``D``, ``grad_value``,
    ``dS K`` and ``dS^T Q``, the rounding of ``dP - D``, of ``dS`` and of the
    scale product, and the rounding to ``dtype``. The accumulations are normal
    accumulations by default (index order or 2 or 4 SIMD-style lanes, as
    ``matmul`` describes); ``standard="any_order"`` covers any order.
    ``normal()`` is that program with normal accumulation in index order and
    the scale and ``exp`` correctly rounded. Not
    budgeted: computing ``P`` with a reciprocal of the row sum, rounding to the
    input dtype in between, or computing ``D`` from the forward output; named
    algorithms (``AttentionAlgorithm``) budget these. Pre-scaling Q and K
    (PyTorch's math path scales each by the square root of the scale) changes
    the inputs: a consumer that wants it passes the pre-scaled values with
    ``scale=1``. PyTorch's math path multiplies the gradients of the
    pre-scaled Q and K by the square root of the scale at the end; with
    pre-scaled inputs and ``scale=1``, the reference is the gradient before
    that rounded product. Every input and ``grad`` may also be an earlier
    result, as ``Result`` describes.
    """
    check_dropout(dropout_p)
    problem = Problem(query, key, value, attn_mask, is_causal, scale, enable_gqa)
    return backward(GradProblem(problem, grad), dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)


def backward(grad_problem: GradProblem, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
             overflow: str | None = None, workers: int | None = None) -> Result:
    """The backward for validated inputs and output gradient (internal; ``Result.backward``
    and ``AttentionAlgorithm.backward_tolerance`` use it)."""
    problem = grad_problem.problem
    formats = gradient_formats(problem, dtype)
    out, shapes = formats["grad_query"][0], {name: shape for name, (_, shape) in formats.items()}
    options = encode_options(out, rounding, overflow)
    stored, pending, call = _kernel_reference(grad_problem, out, options, workers)
    _kernel_refine(grad_problem, call, pending, stored, workers)
    rounded = {name: Array(out, shapes[name], bytes(stored[name].enc)) for name in NAMES}
    exact = {name: ExactArray._from_storage(shapes[name], 1, bytes(stored[name].flags), bytes(stored[name].mids)) for name in NAMES}

    def tolerance(compute, ulps, flush, standard, workers=None):
        fmt = compute_format(compute, problem.dtype)
        settings = required_settings("scaled_dot_product_attention.backward", fmt, tolerance_ulps(ulps, DEFAULT_ULPS))
        radii = grad_problem.radii(compute, ulps, flush, standard, workers)
        return kernel_bounds(grad_problem, out, settings, fmt, rounding, rounded, flush, standard=standard, radii=radii, overflow=options["overflow"],
                             workers=workers)

    def normal(compute, flush, workers=None):
        given, fmt = grad_problem.normal_problem(compute, flush, workers), compute_format(compute, problem.dtype)
        found = _kernel_normal(given, out, options, fmt, flush, workers)
        return {name: Array(out, shapes[name], found[name]) for name in NAMES}

    return Result("scaled_dot_product_attention.backward", "correctly_rounded", rounded, exact, tolerance, normal=normal)


def gradient_formats(problem: Problem, dtype: Format | str | None) -> dict:
    """Per gradient, its format and shape: ``dtype`` (by default the inputs' format), and its
    input's shape."""
    out = output_format(dtype, problem.dtype)
    return {name: (out, x.shape) for name, x in zip(NAMES, (problem.q, problem.k, problem.v))}


class GradProblem:
    """The backward's inputs: the problem, the output gradient, and the output rows in
    tiles.

    A tile is ``(k_start, v_start, rows)``: rows that share one key block and
    one value block, in row-major order, each row as ``(row position, query
    position, q_start)`` (``Problem.tiles``).
    """

    def __init__(self, problem: Problem, grad) -> None:
        self.grad_source = operand(grad)
        G = self.grad_source.array
        if G.dtype != problem.dtype:
            raise TypeError(f"grad must have the inputs' dtype {problem.dtype.name}, got {G.dtype.name}")
        if G.shape != problem.out_shape:
            raise ValueError(f"grad must have the output's shape {problem.out_shape}, got {G.shape}")
        p = self.problem = problem
        self._grad = G
        self.tiles = p.tiles(_TILE_ENTRIES)
        self.query_rows: dict = {}  # query row start -> the tiles' rows that use it, as (k_start, v_start, row), in row order
        for k_start, v_start, rows in self.tiles:
            for row in rows:
                self.query_rows.setdefault(row[2], []).append((k_start, v_start, row))
        for found in self.query_rows.values():
            found.sort(key=lambda item: item[2][0])

    def radii(self, compute, ulps, flush: bool, standard: str, workers=None) -> dict | None:
        """The inputs' and the output gradient's radii under these tolerance arguments, or
        None when every one is exact (Results as inputs)."""
        found = dict(self.problem.radii(compute, ulps, flush, standard, workers) or {name: None for name in ("q", "k", "v", "mask")})
        found["grad"] = unflagged_radius(self.grad_source, compute, ulps, flush, standard, workers)
        return found if any(r is not None for r in found.values()) else None

    def normal_problem(self, compute, flush: bool, workers=None) -> GradProblem:
        """This backward on the inputs' normal values (an earlier result's ``normal()``)."""
        problem = self.problem.normal_problem(compute, flush, workers)
        if problem is self.problem and not self.grad_source.uncertain:
            return self
        return GradProblem(problem, self.grad_source.normal(compute, flush, workers))


# ---------------------------------------------------------------- the reference


def _blocks(grad_problem: GradProblem) -> dict:
    """For each gradient, the start of the block an element belongs to."""
    p = grad_problem.problem
    sizes = {"grad_query": p.E, "grad_key": p.S * p.E, "grad_value": p.S * p.Ev}
    return {name: (lambda index, n=n: index - index % n) for name, n in sizes.items()}


_FIRST_BITS = 64  # the refinement's first working precision, doubling each round
_REFINE_ELEMENTS = 1 << 20  # pending elements per refinement run: their running sums, about 3 KB each, about 3 GiB


class _Stored:
    """One gradient's encodings, entry flags and entry midpoints as bytes."""

    def __init__(self, enc: bytearray, flags: bytearray, mids: bytearray, size: int) -> None:
        self.enc, self.flags, self.mids, self.size = enc, flags, mids, size


def _kernel_reference(grad_problem: GradProblem, out, options: dict, workers=None):
    """The gradients' encodings and exact entries by the reference kernel (``csrc/attention_gref.c``:
    the exact analysis and the first enclosure, the weights from Lemma D's tables or Lemma X), as
    ``_Stored`` per gradient, the elements it leaves undecided by gradient and block, and the call
    (``attention_gref_call``) that ``_kernel_refine`` continues."""
    p = grad_problem.problem
    S, E, Ev = p.S, p.E, p.Ev
    q, k = p.operands(workers, ("q", "k"))  # V at its own width below
    scale = p.kernel_scale
    mask, mask_layout = p.kernel_mask_args()
    tables, score_unit = p.kernel_tables(q, k, None)  # V enters through dP's products; without tables, each weight from Lemma X
    nkv = _block_rows(grad_problem)
    nq = S * max((len(rows) for rows in grad_problem.query_rows.values()), default=1)
    G = grad_problem._grad
    slots = _broadcast_slots(grad_problem)
    call = _native.attention_gref_call(q, k, read_rows(G, Ev, Ev, workers), read_rows(p.v, Ev, Ev, workers), read_rows(G, Ev, 1, workers),
                                       read_rows(p.q, E, 1, workers), read_rows(p.k, E, 1, workers), tables, scale, E, score_unit, S, Ev,
                                       p.is_causal, mask, mask_layout, out._layout, *_float_codes(out, options["rounding"], options["overflow"]),
                                       256, nq, nkv, len(slots))
    size = out.itemsize
    stored = {name: _Stored(bytearray(n * size), bytearray(n), bytearray(n * 24), size) for name, n in zip(NAMES, (p.q.size, p.k.size, p.v.size))}
    status = {name: bytearray(len(stored[name].flags)) for name in NAMES}

    def put(name, start, found):
        enc, flags, mids, st = found
        n = len(flags)
        x = stored[name]
        x.enc[start * size : (start + n) * size] = enc
        x.flags[start : start + n] = flags
        x.mids[start * 24 : (start + n) * 24] = mids
        status[name][start : start + n] = st

    workers = _native.default_workers() if workers is None else workers
    _walk(grad_problem, _gref_kernel(), call, put, (size, 1, 24, 1), slots, _units(grad_problem), workers)
    block_of = _blocks(grad_problem)
    pending = {}
    for name in NAMES:
        waiting: dict = {}
        for index, st in enumerate(status[name]):
            if st:
                waiting.setdefault(block_of[name](index), []).append(index)
        pending[name] = waiting
    return stored, pending, call


def _kernel_refine(grad_problem: GradProblem, call, pending: dict, stored: dict, workers: int | None = None) -> None:
    """The elements the reference kernel left undecided, refined by the kernel's refinement
    (``csrc/attention_grad_refine.c``) from its first enclosure's bits up to the cap, its
    rows in waves of tiles on the threads; raises where one is still undecided there. Each element
    goes with the rows its sum runs over: a value or key block's rows, or a query row's copies. The
    blocks are refined in batches of at most ``_REFINE_ELEMENTS`` elements (a larger block's elements
    cut into several batches, each with the block's rows), each batch's sums held only while it is refined."""
    if not any(pending.values()):
        return
    p = grad_problem.problem
    E, Ev = p.E, p.Ev
    rows, at = [], {}
    by_k, by_v = {}, {}
    for k_start, v_start, tile in grad_problem.tiles:
        for position, i, q_start in tile:
            r = len(at)
            at[k_start, v_start, position] = r
            by_k.setdefault(k_start, []).append(r)
            by_v.setdefault(v_start, []).append(r)
            rows += [row_of(q_start, E), i, *p.mask_start(position), position, row_of(k_start, E), row_of(v_start, Ev)]
    rows = _pyarray("q", rows).tobytes()
    size = stored["grad_value"].size
    shapes = dict(zip(NAMES, (p.q.shape, p.k.shape, p.v.shape)))

    def refine(batch: list) -> None:
        elems, rowlist, order = [], [], []
        for kind, name, cols, start, indices, block in batch:
            first = len(rowlist)
            rowlist += block
            for index in indices:
                j, col = divmod(index - start, cols) if kind < 2 else (0, index - start)
                elems += [kind, j, col, first, len(block)]
                order.append((name, index))
        refinement = _native.attention_gref_refine(call, rows, _pyarray("q", elems).tobytes(), _pyarray("q", rowlist).tobytes(), _FIRST_BITS, MAX_PRECISION)
        while tiles := _native.attention_gref_refine_next(refinement):
            _native.run([functools.partial(_native.attention_gref_refine_tile, refinement, t) for t in range(tiles)], workers)
        enc, flags, mids, status = _native.attention_gref_refine_result(refinement)
        if any(status):
            name, index = order[next(e for e, s in enumerate(status) if s)]
            raise ReferenceUnavailableError(f"attention {name} element {unravel(index, shapes[name])} could not be rounded with certainty "
                                            f"within {MAX_PRECISION} bits")
        for e, (name, index) in enumerate(order):
            x = stored[name]
            x.enc[index * size : (index + 1) * size] = enc[e * size : (e + 1) * size]
            x.flags[index] = flags[e]
            x.mids[index * 24 : (index + 1) * 24] = mids[e * 24 : (e + 1) * 24]

    batch, held = [], 0
    for kind, name, cols, blocks in ((0, "grad_value", Ev, by_v), (1, "grad_key", E, by_k), (2, "grad_query", E, None)):
        for start, indices in pending[name].items():
            block = blocks[start] if blocks is not None else [at[k, v, row[0]] for k, v, row in grad_problem.query_rows[start]]
            for first in range(0, len(indices), _REFINE_ELEMENTS):
                chunk = indices[first : first + _REFINE_ELEMENTS]
                if batch and held + len(chunk) > _REFINE_ELEMENTS:
                    refine(batch)
                    batch, held = [], 0
                batch.append((kind, name, cols, start, chunk, block))
                held += len(chunk)
    if batch:
        refine(batch)


def _block_rows(grad_problem: GradProblem) -> int:
    """The most rows any key block or value block serves: the terms of a grad_key or grad_value sum."""
    rows: dict = {}
    for k_start, v_start, tile in grad_problem.tiles:
        rows["k", k_start] = rows.get(("k", k_start), 0) + len(tile)
        rows["v", v_start] = rows.get(("v", v_start), 0) + len(tile)
    return max(rows.values(), default=0)


def _units(grad_problem: GradProblem) -> tuple:
    """The key and value units in the model's order, as ``((k_start, v_start), rows)``, and per
    key block and per value block the index of the last unit that uses it (where its elements are
    finished: a block may serve several units, when K or V is broadcast where the other is not)."""
    units: dict = {}
    for k_start, v_start, rows in grad_problem.tiles:
        units.setdefault((k_start, v_start), []).extend(rows)
    units = list(units.items())
    return units, {k: u for u, ((k, _), _) in enumerate(units)}, {v: u for u, ((_, v), _) in enumerate(units)}


def _tile_info(p: Problem, tile: list, slots: dict, copies: dict) -> tuple:
    """A tile's rows for the backward kernels (``ATT_GR_INFO`` native int64 each: query row, query
    index, mask offset and step, output-gradient row, broadcast slot or -1, copy, copies), and
    which rows are their query row's last copy; ``copies`` advances."""
    info, last = [], []
    for position, i, q_start in tile:
        copy, count = copies[q_start]
        copies[q_start][0] += 1
        last.append(copy == count - 1)
        info += [row_of(q_start, p.E), i, *p.mask_start(position), position, slots.get(q_start, -1), copy, count]
    return _pyarray("q", info).tobytes(), last


def _kernel_normal(grad_problem: GradProblem, out, options: dict, compute, flush: bool, workers=None) -> dict:
    """The gradients' normal answer by the kernel (``csrc/attention_normal.c``), as encodings per
    gradient."""
    p = grad_problem.problem
    S, E, Ev = p.S, p.E, p.Ev
    scale = p.kernel_scale
    q, k = p.operands(workers, ("q", "k"))  # V at its own width below
    mask, mask_layout = p.kernel_mask_args()
    slots = _broadcast_slots(grad_problem)
    call = _native.attention_gnorm_call(q, k, read_rows(grad_problem._grad, Ev, Ev, workers), read_rows(p.v, Ev, Ev, workers), compute._layout, flush, scale, E, S, Ev, p.is_causal, mask, mask_layout,
                                        out._layout, *_float_codes(out, options["rounding"], options["overflow"]), len(slots))
    size = out.itemsize
    enc = {name: bytearray(n * size) for name, n in zip(NAMES, (p.q.size, p.k.size, p.v.size))}
    overflowed = {name: [] for name in NAMES}  # index order overflowed with every term finite: Lemma O's slots

    def put(name, start, found):
        e, st = found
        enc[name][start * size : start * size + len(e)] = e
        overflowed[name].extend(start + at for at, x in enumerate(st) if x)

    workers = _native.default_workers() if workers is None else workers
    units = _units(grad_problem)
    _walk(grad_problem, _gnorm_kernel(), call, put, (size, 1), slots, units, workers)
    if any(overflowed.values()):
        _normal_slots(grad_problem, call, units[0], overflowed, enc, size)
    return {name: bytes(enc[name]) for name in NAMES}


def _normal_slots(grad_problem: GradProblem, call, units: list, overflowed: dict, enc: dict, size: int) -> None:
    """The normal answer's elements whose index order overflowed with every term finite, from their
    rows again by Lemma O's slots (``att_gnorm_slots``): grad_value's and grad_key's over their
    block's units, grad_query's over its query row's copies."""
    p = grad_problem.problem
    E, Ev = p.E, p.Ev

    def rows_of(items):
        out = []
        for k_start, v_start, (position, i, q_start) in items:
            out += [row_of(q_start, E), i, *p.mask_start(position), position, -1, 0, 1, row_of(k_start, E), row_of(v_start, Ev)]
        return _pyarray("q", out).tobytes()

    block_of = _blocks(grad_problem)
    for which, name in ((0, "grad_value"), (1, "grad_key"), (2, "grad_query")):
        for index in overflowed[name]:
            start = block_of[name](index)
            if which == 2:
                items = grad_problem.query_rows[start]
            else:
                items = [(k, v, row) for (k, v), rows in units for row in rows if (v if which == 0 else k) == start]
            enc[name][index * size : (index + 1) * size] = _native.attention_gnorm_slots(call, which, index - start, rows_of(items))


def _tiles(rows: list) -> list:
    """A key and value block's rows in tiles of at most ``_KERNEL_TILE``, cut where a query row
    repeats (a tile holds a query row at most once)."""
    tiles, tile = [], []
    for row in rows:
        if len(tile) == _KERNEL_TILE or any(other[2] == row[2] for other in tile):
            tiles.append(tile)
            tile = []
        tile.append(row)
    if tile:
        tiles.append(tile)
    return tiles


def _broadcast_slots(grad_problem: GradProblem) -> dict:
    """Query rows broadcast over several key and value blocks: a slot each for their grad_query
    sums between copies."""
    return {q_start: slot for slot, q_start in enumerate(q for q, rows in grad_problem.query_rows.items() if len(rows) > 1)}


def _row_copies(grad_problem: GradProblem) -> dict:
    """Per query row its next copy and how many it has (``_tile_info`` advances it)."""
    return {q_start: [0, len(rows)] for q_start, rows in grad_problem.query_rows.items()}


def _key_chunks(S: int) -> list:
    """The keys in chunks of ``_KERNEL_KEYS``, one call of the kernels' keys' phase each."""
    return [(j, min(j + _KERNEL_KEYS, S)) for j in range(0, S, _KERNEL_KEYS)]


def _row_chunks(T: int, workers: int) -> list:
    """A tile's ``T`` rows in a chunk per thread (one chunk for ``workers`` 0 or 1)."""
    step = max(1, -(-T // max(1, workers)))
    return [(r, min(r + step, T)) for r in range(0, T, step)]


# ---------------------------------------------------------------- the tolerance


class _Wider(Exception):
    """A tile's P' or dS' ends past the bits the backward kernel's sums are sized for: ``need``, the bits they need."""

    def __init__(self, need: tuple):
        super().__init__(need)
        self.need = need


_KERNEL_TILE = 64  # rows per tile of the backward kernel
_KERNEL_KEYS = 16  # keys per call of its keys' phase


def kernel_bounds(grad_problem: GradProblem, out, settings: dict, compute, rounding: Rounding, references: dict, flush: bool, methods=None, *,
                  given=None, standard: str, radii: dict | None = None, overflow: str = "ieee", workers: int | None = None) -> dict[str, Bounds]:
    """Each gradient's ``Bounds`` by the backward kernel (``csrc/attention_grad_tol.c``):
    ``references`` the rounded gradients (``Array`` by name, in ``out``), ``methods`` the named algorithm's (``AttentionAlgorithm``), ``given`` the forward's given
    output and logsumexp (``_algorithms._Given``), ``radii`` the inputs' radii (``GradProblem.radii``,
    with the given output's and logsumexp's as ``out`` and ``lse``) when some are earlier results.
    The key and value units come in the model's order; each one's rows go through it in tiles:
    the tile's rows' P' and dS' (rows split among the threads), then its terms into its key block's
    grad_key and value block's grad_value sums (keys split among them) and the rows' grad_query; a
    block is finished after the last unit that uses it."""
    p = grad_problem.problem
    grouped = methods is not None and methods.grouped_heads
    blocks = None if methods is None else methods.backward_blocks
    S, E, Ev = p.S, p.E, p.Ev
    w = bound_precision(compute, out)
    units, last_k, last_v = _units(grad_problem)
    nkv = _block_rows(grad_problem)
    served: dict = {}  # per key block and value block the query heads (and batch copies) whose rows it serves
    for (k_start, v_start), rows in units:
        for key in (("k", k_start), ("v", v_start)):
            served.setdefault(key, set()).update(row[0] // p.L for row in rows)
    heads = max(map(len, served.values()), default=0)
    nq = S * max((len(rows) for rows in grad_problem.query_rows.values()), default=1)  # grad_query's terms: S per copy of its row
    native_settings = kernel_settings(p, settings, compute if flush else None, methods, w, backward=True)
    grad = _native.attention_gtol_build(native_settings, standard == "normal", True, p.scale_sign, float64._layout, Ev, nq, nkv, heads)
    q, k, v = p.operands(workers)
    scale = p.kernel_scale
    tables, score_unit = p.kernel_tables(q, k, v)
    G = grad_problem._grad
    refs = {name: bytes(references[name].buffer) for name in NAMES}
    mask, mask_layout = p.kernel_mask_args()
    od = ld = None
    if given is not None:
        od, ld = read_rows(given.out_array, Ev, Ev, workers), read_rows(given.lse_array, 1, 1, workers)
    kernel_radii = None
    if radii:  # inputs that are earlier results: their radii, widened where a flush may read a value as zero
        emin = compute.emin if flush else None
        r, g = functools.partial(radius_operand, workers=workers), radii.get
        kernel_radii = (r(p.q, g("q"), E, emin), r(p.k, g("k"), E, emin), r(p.v, g("v"), Ev, emin), r(G, g("grad"), Ev, emin),
                        None if given is None else r(given.out_array, g("out"), Ev, emin),
                        None if given is None else radius_bytes(given.lse_array, g("lse"), emin), radius_bytes(p.mask, g("mask"), emin))
    workers = _native.default_workers() if workers is None else workers

    def attempt(ends: tuple[int, int]) -> dict[str, Bounds]:
        call = _native.attention_gtol_call(grad, q, k, read_rows(G, Ev, Ev, workers), read_rows(p.v, Ev, Ev, workers), read_rows(G, Ev, 1, workers),
                                           read_rows(p.q, E, 1, workers), read_rows(p.k, E, 1, workers), od, ld, tables,
                                           scale, E, score_unit, S, Ev, p.is_causal, mask, mask_layout, out._layout, *_float_codes(out, rounding, overflow),
                                           float64._layout, refs["grad_value"], refs["grad_key"], refs["grad_query"], kernel_radii, *ends)
        size = out.itemsize
        sizes = {name: references[name].size for name in NAMES}
        parts = {name: [bytearray(n * 8), bytearray(n * size), bytearray(n * size), bytearray(n)] for name, n in sizes.items()}
        widths = (8, size, size, 1)

        def put(name, start, found):
            for part, data, width in zip(parts[name], found, widths):
                part[start * width : start * width + len(data)] = data

        slots = _broadcast_slots(grad_problem)
        if slots:
            _native.attention_gtol_query_slots(call, len(slots))
        if grouped or blocks is not None:
            # with grouped_heads, each query head's (and batch copy's) rows on their own, its sums folded at its end; with
            # backward_blocks each block of query rows of a head, its sums carried at its end (Lemma C)
            run_of = (lambda row: row[0] // p.L) if grouped else (lambda row: _row_block(row, p.L, blocks[0]))
            close, closed = ((_native.attention_gtol_group_fold, _native.attention_gtol_group_folded) if grouped
                             else (_native.attention_gtol_group_carry, _native.attention_gtol_group_carried))

            def segments(rows):
                return [list(x) for _, x in itertools.groupby(rows, key=run_of)]

            def after_segment(group, key_chunks):
                _native.run([functools.partial(close, group, call, j0, j1) for j0, j1 in key_chunks], workers)
                closed(group)
        else:
            segments = after_segment = None
        _walk(grad_problem, _gtol_kernel(), call, put, widths, slots, (units, last_k, last_v), workers, segments, after_segment)
        shapes = dict(zip(NAMES, (p.q.shape, p.k.shape, p.v.shape)))
        return {name: Bounds(Array(float64, shapes[name], bytes(parts[name][0])), Array(out, shapes[name], bytes(parts[name][1])),
                             Array(out, shapes[name], bytes(parts[name][2])), Array(uint8, shapes[name], bytes(parts[name][3]))) for name in NAMES}

    # a tile whose P' or dS' ends take more bits than its sums are sized for (kernels.md, Lemma J) has the call made again with ends that
    # hold them
    ends = (0, 0)
    while True:
        try:
            return attempt(ends)
        except _Wider as wider:
            ends = tuple(max(x, y) for x, y in zip(ends, wider.need))


class _Kernel(NamedTuple):
    """One backward kernel's native steps, as ``_walk`` drives them (``csrc/attention_gref.c``,
    ``attention_normal.c``, ``attention_grad_tol.c``)."""

    block: Callable  # (call, key, r0, cols): a key block (key True) or value block from row r0, its rows cols wide
    group: Callable  # (call, kr0, vr0, key block, value block, tile): a unit's group of rows
    tile: Callable  # (group, info): a tile's rows (``_tile_info``)
    tile_rows: Callable  # (group, call, r0, r1): the tile's rows r0 to r1
    after_rows: Callable | None  # (group, call): once all the tile's rows are done, before its keys
    tile_keys: Callable  # (group, call, j0, j1): the tile's terms into the blocks' sums of keys j0 to j1
    tile_queries: Callable  # (group, call, r0, r1): rows r0 to r1's grad_query streams
    tile_done: Callable | None  # (group, call): the tile done
    block_finish: Callable  # (block, call, key, j0, j1): a finished block's streams for keys j0 to j1


def _gtol_need(group, call) -> None:
    """A tile whose P' or dS' ends take more bits than its sums are sized for (``kernels.md``,
    Lemma J): the call is made again with ends that hold them."""
    need = _native.attention_gtol_tile_need(group)
    if need is not None:
        raise _Wider(need)


# Each kernel's steps, read from _native when a walk starts (the tests replace some there).


def _gref_kernel() -> _Kernel:
    return _Kernel(lambda call, key, r0, cols: _native.attention_gref_block(call, key), _native.attention_gref_group,
                   _native.attention_gref_tile, _native.attention_gref_tile_rows, _native.attention_gref_tile_prep,
                   _native.attention_gref_tile_keys, _native.attention_gref_tile_queries,
                   lambda group, call: _native.attention_gref_tile_done(group), _native.attention_gref_block_finish)


def _gnorm_kernel() -> _Kernel:
    # block_finish's key is grad_key's scaling
    return _Kernel(lambda call, key, r0, cols: _native.attention_gnorm_block(call, cols), _native.attention_gnorm_group,
                   _native.attention_gnorm_tile, _native.attention_gnorm_tile_rows, None, _native.attention_gnorm_tile_keys,
                   _native.attention_gnorm_tile_queries, None, _native.attention_gnorm_block_finish)


def _gtol_kernel() -> _Kernel:
    return _Kernel(lambda call, key, r0, cols: _native.attention_gtol_block(call, key, r0), _native.attention_gtol_group,
                   _native.attention_gtol_tile, _native.attention_gtol_tile_rows, _gtol_need, _native.attention_gtol_tile_keys,
                   _native.attention_gtol_tile_queries, _native.attention_gtol_tile_done, _native.attention_gtol_block_finish)


def _walk(grad_problem: GradProblem, kernel: _Kernel, call, put, widths: tuple, slots: dict, units: tuple, workers: int,
          segments=None, after_segment=None) -> None:
    """The backward's walk, the one the proofs' sums follow, for each kernel: the key and value
    units in the model's order (``units``, ``_units``' triple), each one's key and value blocks
    opened at their first unit, its rows through its group tile by tile (the tile's rows split among
    the threads, then its terms into the blocks' sums, keys split among them, and the rows'
    grad_query, which ``put`` stores at a query row's last copy, each stream ``widths`` bytes per
    element; ``slots``: the broadcast rows' slots), and a block's elements ``put`` after the last unit
    that uses it. A tile holds a query row at most once. ``segments(rows)`` splits a unit's rows into
    runs, each followed by ``after_segment(group, key_chunks)`` (``kernel_bounds``' grouped heads and
    backward blocks)."""
    p = grad_problem.problem
    E, Ev = p.E, p.Ev
    units, last_k, last_v = units
    key_chunks = _key_chunks(p.S)
    copies = _row_copies(grad_problem)
    open_blocks: dict = {}
    for unit_index, ((k_start, v_start), rows) in enumerate(units):
        kr0, vr0 = row_of(k_start, E), row_of(v_start, Ev)
        for key, r0, cols in ((True, kr0, E), (False, vr0, Ev)):
            if (key, r0) not in open_blocks:
                open_blocks[key, r0] = kernel.block(call, key, r0, cols)
        group = kernel.group(call, kr0, vr0, open_blocks[True, kr0], open_blocks[False, vr0], _KERNEL_TILE)
        for segment in [rows] if segments is None else segments(rows):
            for tile in _tiles(segment):
                info, last = _tile_info(p, tile, slots, copies)
                kernel.tile(group, info)
                row_chunks = _row_chunks(len(tile), workers)
                _native.run([functools.partial(kernel.tile_rows, group, call, r0, r1) for r0, r1 in row_chunks], workers)
                if kernel.after_rows is not None:
                    kernel.after_rows(group, call)
                calls = [functools.partial(kernel.tile_keys, group, call, j0, j1) for j0, j1 in key_chunks]
                calls += [functools.partial(kernel.tile_queries, group, call, r0, r1) for r0, r1 in row_chunks]
                for (r0, r1), found in zip(row_chunks, _native.run(calls, workers)[len(key_chunks):]):
                    for r in range(r0, r1):
                        if last[r]:
                            put("grad_query", tile[r][2], [x[(r - r0) * E * w : (r - r0 + 1) * E * w] for x, w in zip(found, widths)])
                if kernel.tile_done is not None:
                    kernel.tile_done(group, call)
            if after_segment is not None:
                after_segment(group, key_chunks)
        # a block's elements once its last unit is done
        for name, key, r0, start, cols, done in (("grad_key", True, kr0, k_start, E, last_k[k_start] == unit_index),
                                                 ("grad_value", False, vr0, v_start, Ev, last_v[v_start] == unit_index)):
            if done:
                found = _native.run([functools.partial(kernel.block_finish, open_blocks[key, r0], call, key, j0, j1) for j0, j1 in key_chunks], workers)
                for (j0, _), result in zip(key_chunks, found):
                    put(name, start + j0 * cols, result)
                del open_blocks[key, r0]


def _row_block(row: tuple, L: int, query_block: int) -> tuple:
    """A row's (query head, block of query rows): rows are ``(position, i, q_start)``,
    with ``position`` counting L rows per query head."""
    return row[0] // L, row[1] // query_block


