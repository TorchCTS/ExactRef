"""The gateway to exactref's C (internal).

Only this module imports the extension ``exactref._c``. Arrays cross as ``bytes``: the
limited API of Python 3.10, which the extension is built for, has no buffer protocol. The last
section holds the primitives only the tests call.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import TypeVar

from . import _c


_T = TypeVar("_T")

@functools.lru_cache(maxsize=None)
def default_workers() -> int:
    """The default number of worker threads, as PyTorch picks its own: on Apple Silicon its
    performance cores; elsewhere the physical cores, at most the processors the process may
    run on."""
    return _c.physical_cores()


# Formats (csrc/formats.c): formats.py passes each format's layout tuple and codes in its own
# orders; exponents beyond 2**61 cross clamped, past every format's range either way.
LIMIT = _c.LIMIT
float_layout = _c.float_layout
float_decode = _c.float_decode
float_encode = _c.float_encode
float_encode_ratio = _c.float_encode_ratio
float_special = _c.float_special
float_narrow = _c.float_narrow
int_range = _c.int_range
int_decode = _c.int_decode
int_encode = _c.int_encode
round_integer = _c.round_integer

# Arrays' elements (array.py, csrc/reading.c): as Python scalars, and whether every or some element is nonzero, read
# from their encodings.
scalars = _c.scalars
nonzero = _c.nonzero

# DLPack imports (_dlpack.py, csrc/module.c): a producer's capsule taken and handed back, its
# tensor copied in row-major order.
dlpack_open = _c.dlpack_open
dlpack_close = _c.dlpack_close
gather = _c.gather

# The calling thread's floating-point environment (_fenv.py, csrc/system.c): saved, reset to the
# default and restored.
fenv_save = _c.fenv_save
fenv_reset = _c.fenv_reset
fenv_restore = _c.fenv_restore

# Error bounds' constants (csrc/errors.c): Lemma 1's units and Lemma 2's growth terms.
growth_terms = _c.growth_terms

# The final rounding of outputs (csrc/finish.c): Lemma 3's bounds, and Bounds' flags.
FLAG_PLUS_INF, FLAG_MINUS_INF, FLAG_NAN, FLAG_ANY = _c.FLAG_PLUS_INF, _c.FLAG_MINUS_INF, _c.FLAG_NAN, _c.FLAG_ANY
int_bounds = _c.int_bounds
widest = _c.widest
marks = _c.marks
covers = _c.covers

# Earlier results as operands, and formats holding others (csrc/operands.c).
radius_parts = _c.radius_parts
radius_packed = _c.radius_packed
radius_packed_next = _c.radius_packed_next
radius_packed_tile = _c.radius_packed_tile
radius_packed_result = _c.radius_packed_result
recode = _c.recode  # parts re-encoded in another format (csrc/reading.c's part_read and part_write)
holds = _c.holds
binary_holding = _c.binary_holding

# Exact values' entries (csrc/entries.c): ExactArray's layout, and deciding roundings.
ENTRY_WIDTH = _c.ENTRY_WIDTH
ENTRY_BITS = _c.ENTRY_BITS
exact_entry = _c.exact_entry
ratio_entry = _c.ratio_entry
interval_entry = _c.interval_entry
entry_read = _c.entry_read

# The product kernel (csrc/product.c, csrc/sums.c): operands read once (capsules), and the
# exact sums, encodings, entries and bounds of output elements.
operand_scan = _c.operand_scan
_READ_ON_WORKERS = 1 << 22  # an operand's elements from which operand reads it on the workers


def operand(data, big_endian, layout, start, rows, cols, row_step, col_step, inner, outer_step, terms, wide, given=None, workers=None) -> object:
    """_c.operand's read (rows in groups of inner each outer_step elements on; ``given`` a larger operand's (top,
    flagged)), on up to ``workers`` threads (default ``default_workers()``) for an operand of at least _READ_ON_WORKERS
    elements: its row ranges' largest top bits, unless given, then the ranges read into one operand, each range's first
    slot a multiple of 64 so that no two write one bitmap word. The same operand as one read."""
    args = (data, big_endian, layout, start, rows, cols, row_step, col_step, inner, outer_step, terms, wide)
    workers = default_workers() if workers is None else workers
    if workers <= 1 or rows * cols < _READ_ON_WORKERS or rows < 2:
        return _c.operand(*args) if given is None else _c.operand(*args, given)
    step = 64 // math.gcd(cols, 64) if cols else 1
    per = max(step, -(-rows // (4 * workers)) // step * step)
    ranges = [(r0, min(rows, r0 + per)) for r0 in range(0, rows, per)]
    strides = (row_step, col_step, inner, outer_step)
    if given is None:
        tops = run([functools.partial(_c.operand_rows_top, data, big_endian, layout, start, cols, *strides, r0, r1) for r0, r1 in ranges], workers)
        found = [top for any_, top in tops if any_]
        any_, top, flags = bool(found), max(found, default=0), False
    else:
        any_, (top, flags) = True, given
    x = _c.operand_new(layout, rows, cols, terms, wide, any_, top)
    filled = run([functools.partial(_c.operand_fill, x, data, big_endian, layout, start, *strides, r0, r1) for r0, r1 in ranges], workers)
    _c.operand_done(x, max(w for w, _ in filled), sum(f for _, f in filled), flags)
    return x
scan_flagged = _c.scan_flagged
operand_parts = _c.operand_parts
interleave = _c.interleave
pair_flags = _c.pair_flags
product_reference = _c.product_reference
product_bounds = _c.product_bounds
product_normal = _c.product_normal
product_int_bounds = _c.product_int_bounds
product_run = _c.product_run
order_depth = _c.order_depth

# The attention forward's reference (csrc/attention.c): Lemma D's weight tables, and per tile the
# rows' outputs and logsumexp, decided or left to refine.
attention_tables = _c.attention_tables
attention_reference = _c.attention_reference

# The attention forward's tolerance (csrc/attention_tol.c): the constants from the call's settings
# (the backward's too), the closed form and the step-by-step program.
attention_settings = _c.attention_settings
attention_tolerance_build = _c.attention_tolerance_build
attention_tolerance = _c.attention_tolerance
attention_program_build = _c.attention_program_build
attention_program = _c.attention_program

# The attention normal answers (csrc/attention_normal.c): the forward's, and the backward's by unit,
# tile and block.
attention_normal = _c.attention_normal
attention_gnorm_call = _c.attention_gnorm_call
attention_gnorm_group = _c.attention_gnorm_group
attention_gnorm_tile = _c.attention_gnorm_tile
attention_gnorm_tile_rows = _c.attention_gnorm_tile_rows
attention_gnorm_tile_keys = _c.attention_gnorm_tile_keys
attention_gnorm_tile_queries = _c.attention_gnorm_tile_queries
attention_gnorm_block = _c.attention_gnorm_block
attention_gnorm_block_finish = _c.attention_gnorm_block_finish
attention_gnorm_slots = _c.attention_gnorm_slots

# The attention backward's tolerance (csrc/attention_grad_tol.c), by unit, tile and block.
attention_gtol_build = _c.attention_gtol_build
attention_gtol_call = _c.attention_gtol_call
attention_gtol_group = _c.attention_gtol_group
attention_gtol_tile = _c.attention_gtol_tile
attention_gtol_tile_rows = _c.attention_gtol_tile_rows
attention_gtol_tile_keys = _c.attention_gtol_tile_keys
attention_gtol_tile_queries = _c.attention_gtol_tile_queries
attention_gtol_tile_done = _c.attention_gtol_tile_done
attention_gtol_tile_need = _c.attention_gtol_tile_need
attention_gtol_query_slots = _c.attention_gtol_query_slots
attention_gtol_group_carry = _c.attention_gtol_group_carry
attention_gtol_group_carried = _c.attention_gtol_group_carried
attention_gtol_group_fold = _c.attention_gtol_group_fold
attention_gtol_group_folded = _c.attention_gtol_group_folded
attention_gtol_block = _c.attention_gtol_block
attention_gtol_block_finish = _c.attention_gtol_block_finish

# The attention backward's reference (csrc/attention_gref.c) by unit, tile and block, and its
# refinement (csrc/attention_grad_refine.c).
attention_gref_call = _c.attention_gref_call
attention_gref_group = _c.attention_gref_group
attention_gref_tile = _c.attention_gref_tile
attention_gref_tile_rows = _c.attention_gref_tile_rows
attention_gref_tile_prep = _c.attention_gref_tile_prep
attention_gref_tile_keys = _c.attention_gref_tile_keys
attention_gref_tile_queries = _c.attention_gref_tile_queries
attention_gref_tile_done = _c.attention_gref_tile_done
attention_gref_block = _c.attention_gref_block
attention_gref_block_finish = _c.attention_gref_block_finish
attention_gref_refine = _c.attention_gref_refine
attention_gref_refine_next = _c.attention_gref_refine_next
attention_gref_refine_tile = _c.attention_gref_refine_tile
attention_gref_refine_result = _c.attention_gref_refine_result

# matrix_exp's reference and tolerance (csrc/matrix_exp.c): a batch built from an array's bytes, run
# phase by phase in tiles on the caller's threads (`run`).
matrix_exp_reference = _c.matrix_exp_reference
matrix_exp_reference_next = _c.matrix_exp_reference_next
matrix_exp_reference_tile = _c.matrix_exp_reference_tile
matrix_exp_reference_result = _c.matrix_exp_reference_result
matrix_exp_tolerance = _c.matrix_exp_tolerance
matrix_exp_tolerance_next = _c.matrix_exp_tolerance_next
matrix_exp_tolerance_tile = _c.matrix_exp_tolerance_tile
matrix_exp_tolerance_result = _c.matrix_exp_tolerance_result

# Sums by tensor-core steps (csrc/mma.c): Lemma T's constants.
mma_terms = _c.mma_terms
mma_extent = _c.mma_extent
mma_slack = _c.mma_slack


def run(calls: Sequence[Callable[[], _T]], workers: int | None = None) -> list[_T]:
    """The results of ``calls``, in order: functions of no arguments, each a call into the C,
    which releases the GIL while it computes, run on up to ``workers`` threads (default
    ``default_workers()``). Only the C calls run on the threads' behalf; their results come
    back to the calling thread."""
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(calls) <= 1:
        return [call() for call in calls]
    from concurrent.futures import ThreadPoolExecutor  # only when threads run: it adds about 8 ms to importing exactref

    with ThreadPoolExecutor(min(workers, len(calls))) as pool:
        return list(pool.map(lambda call: call(), calls))


# ---------------------------------------------------------------- for the tests
# C primitives that nothing in exactref calls on its own, which the tests check directly against MPFR, FLINT and
# exact arithmetic: the dyadic operations and function enclosures (csrc/bigfloat.c, csrc/functions.c) and single
# steps of the kernels.

# Dyadics cross as pairs (m, e). Their exponents are unbounded Python ints and the C's are
# 64-bit, so a call whose exponents are beyond 2**61 returns NotImplemented, and it is made
# again with them moved to a base: each operation's result moves with its operands' exponents,
# and an exponent more than _FAR below the base belongs to an operand too small to change the
# result, so it crosses as -_FAR.
_FAR = 1 << 61


def _near(e: int) -> int:
    return max(-_FAR, min(_FAR, e))


def dyadic_rnd(x: tuple[int, int], p: int, up: bool) -> tuple[int, int]:
    """``x`` rounded to at most ``p`` significant bits toward -inf or +inf."""
    r = _c.rnd(x, p, up)
    if r is NotImplemented:
        m, e = _c.rnd((x[0], 0), p, up)
        r = (m, x[1] + e) if m else (0, 0)
    return r


def dyadic_add(x: tuple[int, int], y: tuple[int, int], p: int, up: bool) -> tuple[int, int]:
    """``x + y`` rounded to ``p`` bits; operands far apart never get aligned exactly."""
    r = _c.add(x, y, p, up)
    if r is NotImplemented:
        mx, ex = x
        my, ey = y
        # the base: the exponent of the operand whose top bit is higher (of the nonzero one)
        base = ex if not my or (mx and ex + abs(mx).bit_length() >= ey + abs(my).bit_length()) else ey
        m, e = _c.add((mx, _near(ex - base)), (my, _near(ey - base)), p, up)
        r = (m, base + e) if m else (0, 0)
    return r


def dyadic_mul(x: tuple[int, int], y: tuple[int, int], p: int, up: bool) -> tuple[int, int]:
    r = _c.mul(x, y, p, up)
    if r is NotImplemented:
        m, e = _c.mul((x[0], 0), (y[0], 0), p, up)
        r = (m, x[1] + y[1] + e) if m else (0, 0)
    return r


def dyadic_div(x: tuple[int, int], y: tuple[int, int], p: int, up: bool) -> tuple[int, int]:
    r = _c.div(x, y, p, up)
    if r is NotImplemented:
        m, e = _c.div((x[0], 0), (y[0], 0), p, up)
        r = (m, x[1] - y[1] + e) if m else (0, 0)
    return r


def dyadic_sqrt(x: tuple[int, int], p: int, up: bool) -> tuple[int, int]:
    r = _c.sqrt(x, p, up)
    if r is NotImplemented:
        low = x[1] & 1
        m, e = _c.sqrt((x[0], low), p, up)
        r = (m, (x[1] - low) // 2 + e) if m else (0, 0)
    return r


def ln2_bounds(p: int):
    """``(lo, hi)``: dyadics enclosing log(2) (``proofs/foundations.md``, Lemma Q)."""
    return _c.ln2(p)


def exp_bounds(x, p: int):
    """``(lo, hi)``: dyadics enclosing exp(x) for a dyadic ``x`` (Lemma X). Below 2**-(p + 4) in
    magnitude the result depends only on the sign of ``x``, so a tinier exponent crosses as
    -_FAR."""
    m, e = x
    if m and e > _FAR:
        raise OverflowError("exp of a dyadic with an exponent beyond 2**61")
    lo, hi, n = _c.exp((m, max(e, -_FAR)), p)
    return (lo[0], lo[1] + n), (hi[0], hi[1] + n)


def expm1_end(x, p: int, up: bool):
    """e**x - 1 for a dyadic ``x >= 0`` (below 2**4), rounded down or up to ``p`` bits: the end the
    attention tolerance takes (``csrc/attention_tol.c``). OverflowError for an exponent beyond 2**61."""
    return _c.expm1_end(x, p, up)


def exp_near_one(x, p: int):
    """``(lo, hi)``: exp(x) rounded down and up to ``p`` bits for a dyadic ``x`` below 2**-4 in
    magnitude (Lemma W at three words), or None where ``x`` is not."""
    return _c.exp_near_one(x, p)


def log_bounds(x, p: int):
    """``(lo, hi)``: dyadics enclosing log(x) for a dyadic ``x > 0`` (Lemma Y)."""
    return _c.log(x[0], x[1], p)


def pi_bounds(p: int):
    """``(lo, hi)``: dyadics enclosing pi (Lemma M)."""
    return _c.pi(p)


def cos_sin_bounds(x, p: int):
    """``((lo, hi), (lo, hi))``: dyadics enclosing cos(x) and sin(x) for a dyadic ``x`` (Lemma Z)."""
    return _c.cos_sin(x, p)


def cos_sin_range(lo, hi, p: int):
    """``((lo, hi), (lo, hi))``: dyadics enclosing cos and sin over ``[lo, hi]`` (Lemma Z over an interval)."""
    return _c.cos_sin_range(lo, hi, p)


final_bounds = _c.final_bounds  # Lemma 3's bounds of one element (csrc/finish.c)
read_operand = _c.read_operand  # the read kernel's slots (csrc/reading.c)
entry_bounds = _c.entry_bounds  # an entry's bounds from its bytes (csrc/entries.c)
lemma1_ends = _c.lemma1_ends  # Lemma 1 at a boundary on an interval (csrc/errors.c)
accumulation_error = _c.accumulation_error  # an accumulation's error (csrc/attention_tol.c)
ivl_op = _c.ivl_op  # interval operations (csrc/bigfloat.c)
attention_carry = _c.attention_carry  # Lemma C's enclosure (csrc/attention_grad_tol.c)
attention_gref_refine_waves = _c.attention_gref_refine_waves  # (waves, split waves) of a refinement run
# interval matrix exponentials (csrc/matrices.c), phase by phase
matrix_exp_new = _c.matrix_exp_new
matrix_exp_next = _c.matrix_exp_next
matrix_exp_tile = _c.matrix_exp_tile
matrix_exp_ends = _c.matrix_exp_ends
matrix_exp_rational = _c.matrix_exp_rational
