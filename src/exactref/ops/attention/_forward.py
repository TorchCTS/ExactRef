"""Scaled dot-product attention: the exact answer and its tolerance."""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

from ... import _native
from ..._fenv import ieee
from ...array import Array
from ...boundaries import Boundary, OpSpec, Uses, default_compute, register, required_settings
from ...exact import ExactArray, start_precision
from ...errors import ReferenceUnavailableError
from ...formats import Format, Rounding, _float_codes, float64, uint8
from ...result import Bounds, Result
from .._rounding import bound_precision, compute_format, encode_options, output_format, tolerance_ulps
from .._shapes import unravel
from . import _backward
from ._problem import DEFAULT_ULPS, MAX_PRECISION, Problem, check_dropout, kernel_settings, radius_bytes, radius_operand, row_of

register(
    OpSpec(
        "scaled_dot_product_attention",
        boundaries={
            "scores": Uses("matmul", "dot products of each query row with each key row (Q K^T)"),
            "scale_factor": Boundary("function", "the scale (1/sqrt(E) unless given) held in the compute format"),
            "logits": Boundary("round", "each score times the scale, plus the float mask if any, rounded once"),
            "softmax": {
                "shift": Boundary("round", "each logit minus its row maximum"),
                "exp": Boundary("function", "exponential of each shifted logit"),
                "sum": Boundary("accumulate", "row sum of the exponentials", applies_to="sum"),
            },
            "values": Uses("matmul", "unnormalized sums of exponentials times value rows"),
            "normalize": Boundary("round", "each unnormalized sum divided by the row sum"),
            "logsumexp": {
                "log": Boundary("function", "logarithm of the row sum (logsumexp)"),
                "add": Boundary("round", "the row maximum plus that logarithm (logsumexp)"),
            },
        },
        guarantee="correctly_rounded",
        limitations=(
            "real floating-point inputs only",
            "dropout_p must be 0 (dropout is random)",
            "window_size and alibi_slopes are consumer mappings, not parameters: a sliding window is a bool mask, "
            "and ALiBi biases (slope times query-key distance) are a float mask",
        ),
    )
)


_TILE_ENTRIES = 1 << 16  # rows times keys per tile; only speed and memory depend on it
_GROUPED_READ_BYTES = 1 << 31  # the operands' slots (8 bytes per element) above which a call reads them a group of heads at a time
_GROUP_BYTES = 1 << 32  # a group's operands at most this: its slots, radius operands and V packing (_head_groups)
_NO_RADII = (None, None, None, None, 0, None, None)  # _kernel_radii's value where every input is exact


@ieee
def scaled_dot_product_attention(
    query,
    key,
    value,
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
    """``softmax(Q K^T * scale + mask) V`` over the key axis.

    Shapes follow ``torch.nn.functional.scaled_dot_product_attention``:
    query ``(..., L, E)``, key ``(..., S, E)``, value ``(..., S, Ev)``; with
    ``enable_gqa`` the head axis (-3) of query is a multiple of key/value's.
    A bool mask keeps keys where True; a float mask is added to the scaled
    scores (-inf removes a key). ``is_causal`` keeps key j for query i when
    ``j <= i``. ``scale`` defaults to ``1/sqrt(E)``; a given one must be finite. With E = 0 the scores are
    empty sums and stay 0, as in PyTorch, so the logits are the mask alone.

    Every output is the exact value correctly rounded to ``dtype``, which
    must be floating point, as the inputs are.
    A masked key (False in a bool mask, -inf in a float mask, or above the
    diagonal with ``is_causal``) takes no part, whatever its score or value.
    At the other keys non-finite values propagate through the formula as
    IEEE arithmetic does on exact values, as in PyTorch's math path: a logit
    of +inf or NaN makes the row NaN, a -inf logit removes its key (which then
    takes no part either), and a row with no finite logit has every output 0.
    A non-finite value at a kept key gives its column that infinity (NaN for
    both signs or NaN). An implementation that computes the terms of the keys
    a row does not keep meets NaN where one of their factors is not finite (0
    times a non-finite value, or a masked key's NaN or overflowing score
    before its -inf bias), and so does a kept key whose computed weight may be
    exactly 0 (an exp that underflows, or a logit that may round to -inf)
    times a non-finite value; the tolerance's flags allow that NaN there
    (``may_be("nan")``). An exact zero is +0, or -0 when rounding toward
    -inf. Dropout is random and has no reference, so ``dropout_p`` must be 0.

    The second output, ``logsumexp`` (shape ``(..., Hq, L)``, as the fused
    kernels return it), is ``log(sum_j exp(l_j))`` over each row's kept keys,
    correctly rounded to the compute format of ``dtype`` (float32 for formats
    narrower than float32, else ``dtype`` itself). It follows IEEE arithmetic
    on the exact values at the keys the row does not mask: NaN if a logit is
    NaN, +inf if one is +inf, and -inf, log(0), for a row with no kept key.

    ``tolerance()`` budgets the required boundaries: the score and value
    accumulations, the scale factor, one rounding of each logit, the shift by
    the row maximum, ``exp`` within ``ulps`` (default 2), the row sum, the
    final division, and the rounding to ``dtype``; for ``logsumexp``, ``log``
    of the row sum within ``ulps`` and one rounding of the row maximum plus
    that logarithm. The three sums (scores, row sum, value sums) are normal
    accumulations by default (index order or 2 or 4 SIMD-style lanes, as
    ``matmul`` describes); ``standard="any_order"`` covers any order.
    ``normal()`` is the whole model with normal accumulation in index order
    and the scale, ``exp`` and ``log`` correctly rounded. Other roundings are
    not budgeted: a separate mask
    rounding, materialized probabilities, casting the weights to the input
    dtype before the value product (FlashAttention does this for bf16 and
    fp16), online-softmax rescaling, or a reciprocal of the row sum; named
    algorithms (``AttentionAlgorithm``) budget all of these except
    materialized probabilities (P normalized and rounded before the product
    with V, as PyTorch's math path does), which no tolerance budgets.
    Pre-scaling Q and K changes the inputs: a consumer that wants it passes
    the pre-scaled values with ``scale=1``. An implementation that adds such roundings can exceed
    the tolerance.

    Every input, the mask included, may also be an earlier result, as
    ``Result`` describes.
    """
    check_dropout(dropout_p)
    problem = Problem(query, key, value, attn_mask, is_causal, scale, enable_gqa)
    return forward(problem, dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)


def forward(problem: Problem, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
            workers: int | None = None) -> Result:
    """The forward for validated inputs (internal; ``AttentionAlgorithm.tolerance`` uses it)."""
    formats = output_formats(problem, dtype)
    (out, _), (lse_format, lse_shape) = formats["output"], formats["logsumexp"]
    options, lse_options = encode_options(out, rounding, overflow), encode_options(lse_format, rounding, overflow)
    rounded, exact = _reference(problem, out, options, lse_format, lse_options, workers)

    def tolerance(compute, ulps, flush, standard, workers=None):
        fmt = compute_format(compute, problem.dtype)
        settings = required_settings("scaled_dot_product_attention", fmt, tolerance_ulps(ulps, DEFAULT_ULPS))
        found, lse_found = kernel_bounds(problem, out, settings, fmt, rounding, rounded["output"], flush, lse=(lse_format, rounded["logsumexp"]),
                                  exact=(exact["output"], exact["logsumexp"]), standard=standard,
                                  radii=problem.radii(compute, ulps, flush, standard, workers), overflow=options["overflow"], workers=workers)
        return {"output": found, "logsumexp": lse_found}

    def backward(grad, **options):
        return _backward.backward(_backward.GradProblem(problem, grad), **options)

    def normal(compute, flush, workers=None):
        fmt = compute_format(compute, problem.dtype)
        given = problem.normal_problem(compute, flush, workers)
        found = _kernel_normal(given, out, options, fmt, flush, lse_format, lse_options, workers)
        return {"output": Array(out, problem.out_shape, found[0]), "logsumexp": Array(lse_format, lse_shape, found[1])}

    return Result("scaled_dot_product_attention", "correctly_rounded", rounded, exact, tolerance, backward, normal)


scaled_dot_product_attention.backward = _backward.scaled_dot_product_attention_backward


def kernel_bounds(problem: Problem, out, settings: dict, compute, rounding, reference: Array, flush: bool, methods=None, *, lse, exact,
                  standard: str, radii: dict | None = None, overflow: str = "ieee", workers: int | None = None) -> tuple[Bounds, Bounds]:
    """The ``Bounds`` of the output and of logsumexp (``lse`` is ``(format, reference)``), from
    ``final_bounds`` per element: ``(output's, logsumexp's)``; ``reference`` is the rounded output.

    Rows are worked in tiles sharing a key and a value block by the attention kernels
    (``csrc/attention_tol.c``): each row gets the closed form's bound where its validity conditions
    hold (``exact`` holds the reference's ``ExactArray`` for the output and logsumexp), and otherwise
    the step-by-step program's, whose intervals serve both outputs. ``radii`` are the inputs' radii
    (``Problem.radii``) when some are earlier results.
    """
    w = bound_precision(compute, out)
    flush_format = compute if flush else None
    native_settings = kernel_settings(problem, settings, flush_format, methods, w)
    tiles, done = problem.tiles(_TILE_ENTRIES), {}
    # the closed form's configuration, None where it does not cover the call: only it reads V transposed
    config = _native.attention_tolerance_build(native_settings, standard == "normal", float64._layout)
    groups = _head_groups(problem, tiles, workers, radii, transposed=config is not None)
    if groups is None:
        kernel_radii = None if radii is None else _kernel_radii(problem, radii, flush_format, workers)
        inputs = _kernel_inputs(problem, config is not None, workers)
        if config is not None:
            tiles, done = _closed_form_tiles(problem, inputs, config, tiles, out, lse[0], rounding, overflow, reference, lse[1], exact, workers,
                                             kernel_radii)
        done.update(_program_tiles(problem, inputs, _program_config(problem, native_settings), tiles, out, lse[0], rounding, overflow, reference,
                                   lse[1], workers, kernel_radii))
    else:
        # a group of heads at a time (_head_groups), the tables and constants the whole call's
        scans = problem.scans(workers)
        tables, u, scale = _scanned_tables(problem, scans)
        program_config = _program_config(problem, native_settings)
        e0 = None if radii is None else _v_exponent(problem, radii, flush_format, workers)
        for group in groups:
            q, k, v, *vt = _group_operands(problem, scans, group, config is not None, workers)  # V transposed only with config
            inputs = _KernelInputs(q, k, v, vt[0] if vt else None, tables, u, scale)
            group_radii = None if radii is None else _kernel_radii(problem, radii, flush_format, workers, group, e0)
            group_tiles = group[3]
            if config is not None:
                group_tiles, found = _closed_form_tiles(problem, inputs, config, group_tiles, out, lse[0], rounding, overflow, reference, lse[1],
                                                        exact, workers, group_radii)
                done.update(found)
            done.update(_program_tiles(problem, inputs, program_config, group_tiles, out, lse[0], rounding, overflow, reference, lse[1], workers,
                                       group_radii))
            del q, k, v, vt, inputs, group_radii
    Ev = problem.Ev
    rows_total = math.prod(problem.batch) * problem.L
    sizes, lse_sizes = (8, out.itemsize, out.itemsize, 1), (8, lse[0].itemsize, lse[0].itemsize, 1)
    parts = [bytearray(rows_total * Ev * size) for size in sizes]
    lse_parts = [bytearray(rows_total * size) for size in lse_sizes]
    for position, (row, lse_row) in done.items():  # every row: the kernels' bytes
        for part, data, size in zip(parts, row, sizes):
            part[position * Ev * size : (position + 1) * Ev * size] = data
        for part, data, size in zip(lse_parts, lse_row, lse_sizes):
            part[position * size : (position + 1) * size] = data
    shape, lse_shape = reference.shape, lse[1].shape

    def made(fmt, shape, data):
        return Bounds(Array(float64, shape, bytes(data[0])), Array(fmt, shape, bytes(data[1])), Array(fmt, shape, bytes(data[2])),
                      Array(uint8, shape, bytes(data[3])))

    return made(out, shape, parts), made(lse[0], lse_shape, lse_parts)

class _KernelInputs(NamedTuple):
    """What the tolerance kernels read for the whole call or a group of heads: the operands (``vt``, V transposed, only
    where the closed form runs), Lemma D's tables and their unit ``u`` (None and 0 without tables: the kernels then take
    each weight from Lemma X), and the exact scale (``Problem.kernel_scale``)."""

    q: object
    k: object
    v: object
    vt: object
    tables: object
    u: int
    scale: tuple | None


def _kernel_inputs(problem: Problem, transposed: bool, workers=None) -> _KernelInputs:
    """The whole call's kernel inputs, with V transposed where ``transposed`` (the closed form runs)."""
    q, k, v = problem.operands(workers)
    vt = None
    if transposed:
        S, Ev, x = problem.S, problem.Ev, problem.v
        vt = _native.operand(bytes(x.buffer), x.byteorder == "big", x.dtype._layout, 0, math.prod(x.shape[:-2]) * Ev, S, 1, Ev, Ev, S * Ev, S, False,
                             workers=workers)
    return _KernelInputs(q, k, v, vt, *problem.kernel_tables(q, k, v), problem.kernel_scale)


def _closed_form_tiles(problem: Problem, inputs: _KernelInputs, config, tiles: list, out, lse_format, rounding, overflow, reference: Array,
                       lse_reference: Array, exact, workers, radii=None) -> tuple:
    """The closed form's bounds for every tile's rows: ``(tiles of the rows it left, position -> (the
    row's four output parts, its four logsumexp parts) as bytes)``; ``config`` its configuration
    (``attention_tolerance_build``), ``radii`` as ``_kernel_radii`` gives them, or None."""
    radii = radii or _NO_RADII
    S, E, Ev = problem.S, problem.E, problem.Ev
    size, lse_size = out.itemsize, lse_format.itemsize
    codes, lse_codes = _float_codes(out, rounding, overflow), _float_codes(lse_format, rounding, overflow)
    ref, lse_ref = bytes(reference.buffer), bytes(lse_reference.buffer)
    (flags, mids), (lflags, lmids) = exact[0]._storage(), exact[1]._storage()
    mask, mask_layout = problem.kernel_mask_args()
    calls = []
    for k_start, v_start, rows in tiles:
        positions = [position for position, _, _ in rows]
        calls.append(functools.partial(
            _native.attention_tolerance, inputs.q, inputs.k, inputs.v, inputs.vt, inputs.tables, config, inputs.scale, E, inputs.u, problem.kernel_rows(rows), row_of(k_start, E), row_of(v_start, Ev), S, Ev,
            problem.is_causal, mask, mask_layout, out._layout, *codes, lse_format._layout, *lse_codes,
            _row_bytes(ref, positions, Ev * size), _row_bytes(lse_ref, positions, lse_size), _row_bytes(flags, positions, Ev),
            _row_bytes(mids, positions, Ev * 24), _row_bytes(lflags, positions, 1), _row_bytes(lmids, positions, 24), *radii))
    return _tile_bounds(tiles, _native.run(calls, workers), Ev, size, lse_size)


def _row_bytes(data: bytes, positions: list, width: int) -> bytes:
    """The rows at ``positions`` of ``data``, rows of ``width`` bytes, joined."""
    return b"".join(data[p * width : (p + 1) * width] for p in positions)


def _tile_bounds(tiles: list, results: list, Ev: int, size: int, lse_size: int) -> tuple:
    """A kernel's results per tile as ``(tiles of the rows it left, position -> (the row's four
    output parts, its four logsumexp parts) as bytes)``: a row is left where its status is set."""
    left, done = [], {}
    for (k_start, v_start, rows), (status, tol, low, high, fl, ltol, llow, lhigh, lfl) in zip(tiles, results):
        rest = []
        for r, row in enumerate(rows):
            if status[r]:
                rest.append(row)
                continue
            done[row[0]] = ((tol[r * Ev * 8 : (r + 1) * Ev * 8], low[r * Ev * size : (r + 1) * Ev * size], high[r * Ev * size : (r + 1) * Ev * size],
                             fl[r * Ev : (r + 1) * Ev]),
                            (ltol[r * 8 : (r + 1) * 8], llow[r * lse_size : (r + 1) * lse_size], lhigh[r * lse_size : (r + 1) * lse_size],
                             lfl[r : r + 1]))
        if rest:
            left.append((k_start, v_start, rest))
    return left, done


def _program_config(problem: Problem, native_settings):
    """The step-by-step program's configuration for this call (``csrc/attention_tol.c``)."""
    return _native.attention_program_build(native_settings, problem.scale_sign, float64._layout)

def _v_packing(problem: Problem, radii: dict, emin, lo: int, hi: int, e0, workers) -> tuple:
    """``_native.radius_packed`` of V's elements lo .. hi - 1 at the exponent of all of V (``e0``, or
    None for its first pass), its passes in tiles on the threads: (values, radii, e0, unbounded)."""
    x, r = problem.v, radii["v"]
    call = _native.radius_packed(x.dtype._layout, bytes(x.buffer), x.byteorder == "big", float64._layout, bytes(r.buffer), r.byteorder == "big", x.size,
                                 emin, e0, lo, hi)
    while tiles := _native.radius_packed_next(call):
        _native.run([functools.partial(_native.radius_packed_tile, call, t) for t in range(tiles)], workers)
    return _native.radius_packed_result(call)


def _v_exponent(problem: Problem, radii: dict, flush_format, workers=None) -> int:
    """The exponent at which ``_kernel_radii`` packs every head group's part of V: that of all of V
    (its first pass alone), 0 where V is exact."""
    if radii["v"] is None:
        return 0
    return _v_packing(problem, radii, None if flush_format is None else flush_format.emin, 0, 0, None, workers)[2]


def _kernel_radii(problem: Problem, radii: dict, flush_format, workers=None, group=None, e0=None) -> tuple:
    """The inputs' radii as the attention kernels take them (Results as inputs): Q's and K's as
    operands of their widened radii, V's values and radii as 128-bit integers at one exponent with
    each element's no-bound flag (``_native.radius_packed``), and the float mask's radii as float64,
    each None where exact. With a head group (``_head_groups``) only its rows of Q and K and its
    elements of V, these at the exponent ``e0`` of all of V (``_v_exponent``), as its operands are
    read; the mask's whole."""
    emin = None if flush_format is None else flush_format.emin
    (qa, qb), (ka, kb), (va, vb) = group[:3] if group is not None else ((0, problem.q.size), (0, problem.k.size), (0, problem.v.size))
    vv = vr = vunb = None
    if radii["v"] is not None:
        vv, vr, e0, vunb = _v_packing(problem, radii, emin, va, vb, e0, workers)
    return (radius_operand(problem.q, radii["q"], problem.E, emin, qa, qb, workers), radius_operand(problem.k, radii["k"], problem.E, emin, ka, kb, workers), vv, vr,
            e0 or 0, vunb, radius_bytes(problem.mask, radii["mask"], emin))


def _program_tiles(problem: Problem, inputs: _KernelInputs, config, tiles: list, out, lse_format, rounding, overflow, reference: Array,
                   lse_reference: Array, workers, radii=None) -> dict:
    """The step-by-step program's bounds for every tile's rows, by row, as ``_closed_form_tiles``
    gives them; ``config`` its configuration (``_program_config``), ``radii`` as ``_kernel_radii``
    gives them, or None."""
    radii = radii or _NO_RADII
    S, E, Ev = problem.S, problem.E, problem.Ev
    size, lse_size = out.itemsize, lse_format.itemsize
    codes, lse_codes = _float_codes(out, rounding, overflow), _float_codes(lse_format, rounding, overflow)
    ref, lse_ref = bytes(reference.buffer), bytes(lse_reference.buffer)
    mask, mask_layout = problem.kernel_mask_args()
    calls = []
    for k_start, v_start, rows in tiles:
        positions = [position for position, _, _ in rows]
        calls.append(functools.partial(
            _native.attention_program, inputs.q, inputs.k, inputs.v, config, E, problem.kernel_rows(rows), row_of(k_start, E), row_of(v_start, Ev), S, Ev, problem.is_causal, mask,
            mask_layout, out._layout, *codes, lse_format._layout, *lse_codes, _row_bytes(ref, positions, Ev * size), _row_bytes(lse_ref, positions, lse_size),
            *radii, inputs.tables, inputs.u, inputs.scale))
    left, done = _tile_bounds(tiles, _native.run(calls, workers), Ev, size, lse_size)
    if left:  # the program bounds every row or raises: a row left unbounded would get no tolerance
        raise RuntimeError("internal error: the attention program left a row without bounds")
    return done


def _kernel_normal(problem: Problem, out, options: dict, compute, flush: bool, lse_format, lse_options: dict, workers) -> tuple:
    """The normal answer by the attention kernel (``csrc/attention_normal.c``): the outputs' and
    logsumexp's encodings as bytes in row order."""
    S, E, Ev = problem.S, problem.E, problem.Ev
    scale = problem.kernel_scale
    q, k, v = problem.operands(workers)
    mask, mask_layout = problem.kernel_mask_args()
    codes, lse_codes = _float_codes(out, options["rounding"], options["overflow"]), _float_codes(lse_format, lse_options["rounding"], lse_options["overflow"])
    tiles = problem.tiles(_TILE_ENTRIES)
    calls = [functools.partial(_native.attention_normal, q, k, v, compute._layout, flush, scale, E, problem.kernel_rows(rows), row_of(k_start, E), row_of(v_start, Ev),
                               S, Ev, problem.is_causal, mask, mask_layout, out._layout, *codes, lse_format._layout, *lse_codes)
             for k_start, v_start, rows in tiles]
    size, lse_size = out.itemsize, lse_format.itemsize
    rows_total = math.prod(problem.batch) * problem.L
    enc, lse_enc = bytearray(rows_total * Ev * size), bytearray(rows_total * lse_size)
    for (_, _, rows), (t_enc, t_lse) in zip(tiles, _native.run(calls, workers)):
        for r, (position, _, _) in enumerate(rows):
            enc[position * Ev * size : (position + 1) * Ev * size] = t_enc[r * Ev * size : (r + 1) * Ev * size]
            lse_enc[position * lse_size : (position + 1) * lse_size] = t_lse[r * lse_size : (r + 1) * lse_size]
    return bytes(enc), bytes(lse_enc)


def output_formats(problem: Problem, dtype: Format | str | None) -> dict:
    """Per output, its format and shape: ``output`` in ``dtype`` (by default the inputs' format),
    ``logsumexp`` one per output row, in that format's default compute format."""
    out = output_format(dtype, problem.dtype)
    return {"output": (out, problem.out_shape), "logsumexp": (default_compute(out), problem.out_shape[:-1])}


def _reference(problem: Problem, out, options: dict, lse_format, lse_options: dict, workers: int | None = None) -> tuple:
    """The correctly rounded outputs and logsumexp, as ``Array``s, and their exact entries, as
    ``ExactArray``s (dicts by output name), by the attention kernel (``_kernel_reference``) in tiles
    sharing a key and a value block."""
    lse_shape = problem.out_shape[:-1]
    enc, flags, mids, lse_enc, lse_flags, lse_mids = _kernel_reference(problem, out, options, lse_format, lse_options, workers)
    return ({"output": Array(out, problem.out_shape, bytes(enc)), "logsumexp": Array(lse_format, lse_shape, bytes(lse_enc))},
            {"output": ExactArray._from_storage(problem.out_shape, 1, bytes(flags), bytes(mids)),
             "logsumexp": ExactArray._from_storage(lse_shape, 1, bytes(lse_flags), bytes(lse_mids))})


def _scanned_tables(problem: Problem, scans: dict) -> tuple:
    """Lemma D's tables from the operands' scans, as ``attention_tables`` gives them from the operands: ``(tables, u, sigma)``."""
    q, k, v = scans["q"], scans["k"], scans["v"]
    tables, u = problem.kernel_tables((q[1], q[0], k[1], k[0], q[2] and k[2], q[3] or k[3] or v[3]), None, None)
    return tables, u, problem.kernel_scale


def _head_groups(problem: Problem, tiles: list, workers, radii: dict | None = None, transposed: bool = True) -> list | None:
    """The tiles in groups of whole key heads where the operands' slots exceed ``_GROUPED_READ_BYTES``, each group's tiles
    relative to the first Q, K and V elements they read: ``[((qa, qb), (ka, kb), (va, vb), tiles)]``, as many heads per group
    as fill the threads' waves (tiles a multiple of the threads where few heads do that), and no more than keep what a
    group reads (``_group_operands``: the spans qa..qb, ka..kb and va..vb) within ``_GROUP_BYTES``: 8 bytes per
    element of each slot (V^T's with ``transposed``), 24 more per element of Q and K for the call's ``radii`` (a radius
    operand's two-word slot, and the float64 copy it is read from) and 33 for V's packing, and 8 per element of a K or V
    that is an earlier result, whose arrays the call holds besides. None to read the operands whole."""
    S, E, Ev = problem.S, problem.E, problem.Ev
    rq, rk, rv = (radii is not None and radii.get(name) is not None for name in ("q", "k", "v"))
    held = 8 * (problem.q.size + problem.k.size + 2 * problem.v.size) + (33 * problem.v.size if rv else 0)
    if held <= _GROUPED_READ_BYTES or not (S and E and Ev) or not tiles:
        return None
    by_head: dict = {}
    for tile in tiles:
        by_head.setdefault(tile[0] // (S * E), []).append(tile)
    heads = sorted(by_head)
    threads = max(1, _native.default_workers() if workers is None else workers)
    per = max(1, len(tiles) // len(heads))
    size = threads // math.gcd(per, threads)  # the fewest heads whose tiles (per each) fill whole waves of the threads
    per_q = 8 + 24 * rq
    per_k = 8 + 24 * rk + 8 * problem.sources["k"].uncertain
    per_v = 8 * (1 + transposed + problem.sources["v"].uncertain) + 33 * rv

    def spans(group):
        """the Q, K and V element spans a list of tiles reads"""
        qs = [row[2] for t in group for row in t[2]]
        return [min(qs), max(qs) + E, min(t[0] for t in group), max(t[0] for t in group) + S * E, min(t[1] for t in group),
                max(t[1] for t in group) + S * Ev]

    def joined(x, y):
        return [min(x[0], y[0]), max(x[1], y[1]), min(x[2], y[2]), max(x[3], y[3]), min(x[4], y[4]), max(x[5], y[5])]

    def read_bytes(x):
        return per_q * (x[1] - x[0]) + per_k * (x[3] - x[2]) + per_v * (x[5] - x[4])

    # whole heads in order, a group closed before the head that would take it past size heads or past the bytes
    closed, group, span, count = [], [], None, 0
    for h in heads:
        own = spans(by_head[h])
        if group and (count >= size or read_bytes(joined(span, own)) > _GROUP_BYTES):
            closed.append((group, span))
            group, span, count = [], None, 0
        group, span, count = group + by_head[h], own if span is None else joined(span, own), count + 1
    closed.append((group, span))
    groups = []
    for group, (qa, qb, ka, kb, va, vb) in closed:
        relative = [(k - ka, v - va, [(position, i, q_start - qa) for position, i, q_start in rows]) for k, v, rows in group]
        groups.append(((qa, qb), (ka, kb), (va, vb), relative))
    return groups if len(groups) > 1 else None


def _group_operands(problem: Problem, scans: dict, group, transposed: bool = True, workers=None) -> tuple:
    """A group's Q, K and V operands (``_head_groups``), and with ``transposed`` V transposed's, each slot as the whole
    operand's read gives it."""
    (qa, qb), (ka, kb), (va, vb), _ = group
    S, E, Ev = problem.S, problem.E, problem.Ev

    def given(name):
        top, _, any_, flagged = scans[name]
        return (top, flagged) if any_ else None

    found = []
    for name, x, a, b, cols, terms in (("q", problem.q, qa, qb, E, E), ("k", problem.k, ka, kb, E, E), ("v", problem.v, va, vb, Ev, 1)):
        rows = (b - a) // cols
        found.append(_native.operand(x.buffer, x.byteorder == "big", x.dtype._layout, a, rows, cols, cols, 1, max(rows, 1), 0, terms, False, given(name), workers=workers))
    x = problem.v
    if transposed:
        found.append(_native.operand(x.buffer, x.byteorder == "big", x.dtype._layout, va, (vb - va) // S, S, 1, Ev, Ev, S * Ev, S, False, given("vt"), workers=workers))
    return tuple(found)


def _kernel_reference(problem: Problem, out, options: dict, lse_format, lse_options: dict, workers: int | None = None) -> tuple:
    """The outputs' and logsumexp's encodings and entries as bytes, by the attention kernel
    (``csrc/attention.c``; without Lemma D's tables it takes each weight from Lemma X), which
    refines what its first enclosures leave undecided up to the precision cap."""
    E, Ev = problem.E, problem.Ev
    mask, mask_layout = problem.kernel_mask_args()
    tiles = problem.tiles(_TILE_ENTRIES)
    groups = _head_groups(problem, tiles, workers, transposed=False)
    if groups is None:
        operands = problem.operands(workers)
        scale = problem.kernel_scale
        tables, u = problem.kernel_tables(*operands)
        reads = [(lambda: operands, tiles)]
    else:
        scans = problem.scans(workers)
        tables, u, scale = _scanned_tables(problem, scans)
        reads = [(functools.partial(_group_operands, problem, scans, group, False, workers=workers), group[3]) for group in groups]
    rows_total = math.prod(problem.batch) * problem.L
    size, lse_size = out.itemsize, lse_format.itemsize
    enc, flags, mids = bytearray(rows_total * Ev * size), bytearray(rows_total * Ev), bytearray(rows_total * Ev * 24)
    lse_enc, lse_flags, lse_mids = bytearray(rows_total * lse_size), bytearray(rows_total), bytearray(rows_total * 24)
    codes = _float_codes(out, options["rounding"], options["overflow"])
    lse_codes = _float_codes(lse_format, lse_options["rounding"], lse_options["overflow"])
    p = start_precision(out)
    for read, group_tiles in reads:  # the whole call, or a group of heads at a time (_head_groups)
        q, k, v = read()
        calls = [functools.partial(_native.attention_reference, q, k, v, tables, scale, E, u, problem.kernel_rows(rows), row_of(k_start, E),
                                   row_of(v_start, Ev), problem.S, Ev, problem.is_causal, mask, mask_layout, out._layout, *codes, lse_format._layout, *lse_codes,
                                   p, MAX_PRECISION)
                 for k_start, v_start, rows in group_tiles]
        for (k_start, v_start, rows), (t_enc, t_flags, t_mids, t_status, t_lse, t_lflags, t_lmids, t_lstatus) in zip(group_tiles, _native.run(calls, workers)):
            if any(t_status):
                r, c = divmod(next(n for n, s in enumerate(t_status) if s), Ev)
                raise ReferenceUnavailableError(f"attention output element {unravel(rows[r][0] * Ev + c, problem.out_shape)} could not be "
                                                f"rounded with certainty within {MAX_PRECISION} bits")
            if any(t_lstatus):
                r = next(n for n, s in enumerate(t_lstatus) if s)
                raise ReferenceUnavailableError(f"attention logsumexp element {unravel(rows[r][0], problem.out_shape[:-1])} could not be "
                                                f"rounded with certainty within {MAX_PRECISION} bits")
            for r, (position, _, _) in enumerate(rows):
                a, b = position * Ev, r * Ev
                enc[a * size : (a + Ev) * size] = t_enc[b * size : (b + Ev) * size]
                flags[a : a + Ev] = t_flags[b : b + Ev]
                mids[a * 24 : (a + Ev) * 24] = t_mids[b * 24 : (b + Ev) * 24]
                lse_enc[position * lse_size : (position + 1) * lse_size] = t_lse[r * lse_size : (r + 1) * lse_size]
                lse_flags[position] = t_lflags[r]
                lse_mids[position * 24 : (position + 1) * 24] = t_lmids[r * 24 : (r + 1) * 24]
        del q, k, v, calls
    return enc, flags, mids, lse_enc, lse_flags, lse_mids
