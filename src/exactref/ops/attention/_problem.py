"""Attention's validated inputs and what its forward and backward share: the logits' scale
and mask, the algorithm methods, the inputs' radii, and the kernels' operands, rows and
settings (internal)."""

from __future__ import annotations

import functools
import math
from array import array as _pyarray
from dataclasses import dataclass
from fractions import Fraction

from ... import _native
from ...accumulation import ANY_ORDER, MMAStep
from ...boundaries import Accumulation, RoundTo
from ...formats import _ROUNDING_CODES, FloatFormat, float64
from ...values import FloatValue
from .._operands import operand, unflagged_radius
from .._shapes import Grid, broadcast_index, broadcast_shapes, iter_indices, unravel


MAX_PRECISION = 1 << 14
DEFAULT_ULPS = 2


class Problem:
    """Validated inputs, shapes and per-row access."""

    def __init__(self, query, key, value, attn_mask, is_causal, scale, enable_gqa):
        sources = {"q": operand(query), "k": operand(key), "v": operand(value), "mask": None if attn_mask is None else operand(attn_mask)}
        self.sources = sources  # the inputs, which may be earlier results (Results as inputs)
        self._options = (is_causal, scale, enable_gqa)
        q, k, v = sources["q"].array, sources["k"].array, sources["v"].array
        if not (q.dtype == k.dtype == v.dtype):
            raise TypeError(f"query, key and value must share a dtype, got {q.dtype.name}, {k.dtype.name}, {v.dtype.name}")
        if q.dtype.kind != "float":
            raise TypeError(f"scaled_dot_product_attention needs real floating-point inputs, got {q.dtype.name}")
        if min(q.ndim, k.ndim, v.ndim) < 2:
            raise ValueError("query, key and value need at least 2 dimensions")
        self.L, self.E = q.shape[-2:]
        self.S, e_key = k.shape[-2:]
        s_value, self.Ev = v.shape[-2:]
        if e_key != self.E or s_value != self.S:
            raise ValueError(f"incompatible shapes: query {q.shape}, key {k.shape}, value {v.shape}")
        if enable_gqa:
            if min(q.ndim, k.ndim, v.ndim) < 3:
                raise ValueError("enable_gqa needs a head dimension (at least 3 dimensions)")
            hq, hk, hv = q.shape[-3], k.shape[-3], v.shape[-3]
            if hk != hv or hk == 0 or hq % hk:
                raise ValueError(f"grouped-query attention needs Hq divisible by Hkv; got Hq={hq}, Hk={hk}, Hv={hv}")
            self.group = hq // hk
            self.batch = broadcast_shapes(q.shape[:-3], k.shape[:-3], v.shape[:-3]) + (hq,)
        else:
            self.group = 1
            self.batch = broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
        self.out_shape = self.batch + (self.L, self.Ev)
        if attn_mask is not None and is_causal:
            raise ValueError("pass either attn_mask or is_causal=True, not both")
        self.is_causal = bool(is_causal)
        self.mask = None if attn_mask is None else sources["mask"].array
        if self.mask is not None:
            if self.mask.dtype.kind not in ("bool", "float"):
                raise TypeError(f"attn_mask must be bool or floating point, got {self.mask.dtype.name}")
            scores = self.batch + (self.L, self.S)
            if broadcast_shapes(self.mask.shape, scores) != scores:
                raise ValueError(f"attn_mask of shape {self.mask.shape} does not broadcast to the scores' shape {scores}")
        given = scale
        if isinstance(scale, float):
            scale = FloatValue.from_float(scale)  # by its bits, whatever the thread's floating-point environment
        if isinstance(scale, FloatValue) and not scale.is_finite:
            raise ValueError(f"scale must be finite, got {given}")
        if scale is not None:
            scale = scale.to_fraction() if isinstance(scale, FloatValue) else Fraction(scale)
        self.scale = scale
        self.dtype = q.dtype
        self.q, self.k, self.v = q, k, v
        self.qg, self.kg, self.vg = Grid(q.shape), Grid(k.shape), Grid(v.shape)
        self.mg = Grid(self.mask.shape) if self.mask is not None else None
        self._scans = None  # scans' result

    def radii(self, compute, ulps, flush: bool, standard: str, workers=None) -> dict | None:
        """The inputs' radii under these tolerance arguments, by input, or None when every
        input is exact (Results as inputs); inf where a stored value may be non-finite
        (``unflagged_radius``)."""
        found = {name: None if x is None else unflagged_radius(x, compute, ulps, flush, standard, workers) for name, x in self.sources.items()}
        return found if any(r is not None for r in found.values()) else None

    def normal_problem(self, compute, flush: bool, workers=None) -> Problem:
        """This problem on the inputs' normal values (an earlier result's ``normal()``)."""
        if not any(x is not None and x.uncertain for x in self.sources.values()):
            return self
        values = [None if x is None else x.normal(compute, flush, workers) for x in (self.sources[name] for name in ("q", "k", "v", "mask"))]
        return Problem(*values, *self._options)

    def rows(self):
        """Yield ``(batch index, query row)`` for every output row, row-major."""
        for batch_index in iter_indices(self.batch):
            for i in range(self.L):
                yield batch_index, i

    def row_inputs(self, batch_index, i) -> tuple[int, int, int]:
        """Where one output row's inputs start among the inputs' elements (row-major offsets).

        Returns ``(q_start, k_start, v_start)``: the query row is ``E`` elements
        from ``q_start``, key row j is ``E`` elements from ``k_start + j * E``,
        and value row j is ``Ev`` elements from ``v_start + j * Ev``.
        """
        q_batch = broadcast_index(batch_index, self.q.shape[:-2])
        if self.group > 1:
            kv_index = batch_index[:-1] + (batch_index[-1] // self.group,)
        else:
            kv_index = batch_index
        k_batch = broadcast_index(kv_index, self.k.shape[:-2])
        v_batch = broadcast_index(kv_index, self.v.shape[:-2])
        q_start = self.qg.offset(q_batch + (i, 0))
        k_start = self.kg.offset(k_batch + (0, 0))
        v_start = self.vg.offset(v_batch + (0, 0))
        return q_start, k_start, v_start

    def mask_start(self, position: int) -> tuple[int, int]:
        """Where output row ``position``'s mask row starts among the mask's elements, and its step
        along the keys (0 where the mask broadcasts over them); (0, 0) without a mask."""
        if self.mask is None:
            return 0, 0
        index = unravel(position // self.L, self.batch) + (position % self.L, 0) if self.L else (0, 0)
        start = self.mg.offset(broadcast_index(index, self.mask.shape))
        return start, 1 if self.mask.shape and self.mask.shape[-1] == self.S else 0

    @property
    def exact_scale(self) -> Fraction | None:
        """The scale as an exact rational, or None when it is the irrational 1/sqrt(E)."""
        if self.scale is not None:
            return self.scale
        if self.E == 0:
            return Fraction(0)  # the scores are empty sums, exactly 0, so the scale has no effect
        root = math.isqrt(self.E)
        return Fraction(1, root) if root * root == self.E else None

    def tiles(self, entries: int) -> list[tuple[int, int, list[tuple[int, int, int]]]]:
        """The output rows in tiles that share a key block and a value block: ``(k_start,
        v_start, rows)`` with rows ``(position, query row, q_start)`` in row-major order, at
        most ``entries`` rows times keys per tile."""
        units: dict = {}
        for position, (batch_index, i) in enumerate(self.rows()):
            q_start, k_start, v_start = self.row_inputs(batch_index, i)
            units.setdefault((k_start, v_start), []).append((position, i, q_start))
        size = max(1, entries // max(self.S, 1))
        return [(k, v, rows[s : s + size]) for (k, v), rows in units.items() for s in range(0, len(rows), size)]

    @property
    def kernel_scale(self) -> tuple[int, int] | None:
        """The exact scale as the kernels take it, ``(numerator, denominator)``, or None where it
        is the irrational 1/sqrt(E)."""
        sigma = self.exact_scale
        return None if sigma is None else (sigma.numerator, sigma.denominator)

    @property
    def scale_sign(self) -> int:
        """The exact scale's sign (1/sqrt(E) without one), for IEEE's rules on the logits."""
        sigma = self.exact_scale
        return 1 if sigma is None else (sigma > 0) - (sigma < 0)

    def kernel_mask_args(self) -> tuple[bytes | None, tuple[tuple, bool] | None]:
        """The mask as the kernels take it: ``(bytes, (float layout, big-endian))``, each None where
        it does not apply."""
        mask = self.mask
        if mask is None:
            return None, None
        return bytes(mask.buffer), ((mask.dtype._layout, mask.byteorder == "big") if mask.dtype.kind == "float" else None)

    def kernel_rows(self, rows: list) -> bytes:
        """A tile's rows for the attention kernels: per row (query row, query index, mask offset,
        mask step) as native int64."""
        return _pyarray("q", [x for position, i, q_start in rows for x in (row_of(q_start, self.E), i, *self.mask_start(position))]).tobytes()

    def operands(self, workers=None, names=("q", "k", "v")) -> tuple[object, ...]:
        """The rows of the operands ``names`` (of Q, K and V, in that order) read once by the read
        kernel (``read_rows``): Q and K at the product kernel's main range for E terms, V at one word."""
        found = []
        for name, x, cols, terms in (("q", self.q, self.E, self.E), ("k", self.k, self.E, self.E), ("v", self.v, self.Ev, 1)):
            if name in names:
                found.append(read_rows(x, cols, terms, workers))
        return tuple(found)

    def scans(self, workers=None) -> dict[str, tuple[int, int, bool, bool]]:
        """Per operand (q, k, v and V transposed, as ``operands`` and the closed form's configuration read them), the read kernel's scan of
        its values (``csrc/reading.c``, ``scan_operand``) as ``(top, low, any, flagged)``: chunks of rows on the threads, merged;
        V transposed holds V's values, flagged at its own main bits. Kept for the problem's later calls."""
        found = self._scans
        if found is not None:
            return found
        S, E, Ev = self.S, self.E, self.Ev
        chunks = max(1, _native.default_workers() if workers is None else workers)
        raw = {}
        for name, x, cols, terms in (("q", self.q, E, E), ("k", self.k, E, E), ("v", self.v, Ev, 1)):
            rows = x.size // cols if cols else math.prod(x.shape[:-1])
            step = max(1, -(-rows // chunks))
            parts = _native.run([functools.partial(_native.operand_scan, x.buffer, x.byteorder == "big", x.dtype._layout, r0 * cols, min(step, rows - r0), cols,
                                                   cols, 1, max(min(step, rows - r0), 1), 0, terms, False) for r0 in range(0, max(rows, 1), step)], workers)
            seen = [part for part in parts if part[2]]
            top, low = (max(part[0] for part in seen), min(part[1] for part in seen)) if seen else (0, 0)
            raw[name] = (top, low, bool(seen), any(part[3] for part in parts), parts[0][4])
        x = self.v
        raw["vt"] = raw["v"][:4] + (_native.operand_scan(x.buffer, x.byteorder == "big", x.dtype._layout, 0, 0, S, 1, Ev, Ev, S * Ev, S, False)[4],)
        found = {name: (top, low, seen, _native.scan_flagged(top, low, seen, nonfinite, bits)) for name, (top, low, seen, nonfinite, bits) in raw.items()}
        self._scans = found
        return found

    def kernel_tables(self, q, k, v) -> tuple[object | None, int]:
        """Lemma D's tables and their unit for the kernels, ``(tables, u)``, from the operands
        (``_native.attention_tables``); ``(None, 0)`` where the call has none, and the kernels then
        take each weight from Lemma X."""
        found = _native.attention_tables(q, k, v, self.kernel_scale, self.E, *self.kernel_mask_args())
        return (None, 0) if found is None else found[:2]



def check_dropout(dropout_p) -> None:
    """Dropout is random and has no reference, so ``dropout_p`` must be 0."""
    if dropout_p != 0:
        raise ValueError("dropout is random and has no reference; dropout_p must be 0")


def row_of(start: int, width: int) -> int:
    """The row an input's flat element offset starts, for rows of ``width`` elements (row 0 where
    the rows are empty: nothing in them is read)."""
    return start // width if width else 0


def read_rows(x, cols: int, terms: int, workers=None):
    """Array ``x`` as rows of ``cols`` elements read once by the read kernel (``_native.operand``)
    for sums of ``terms`` products, on up to ``workers`` threads."""
    rows = x.size // cols if cols else math.prod(x.shape[:-1])
    return _native.operand(bytes(x.buffer), x.byteorder == "big", x.dtype._layout, 0, rows, cols, cols, 1, max(rows, 1), 0, terms, False, workers=workers)


def radius_bytes(x, radius, emin, a: int = 0, b: int | None = None):
    """An earlier result's radii of elements a .. b - 1 (all by default), widened where a flush below
    ``emin`` may read a value as zero, as float64 bytes, rounded up, or None where it is exact."""
    if radius is None:
        return None
    b = x.size if b is None else b
    return _native.radius_parts(x.dtype._layout, bytes(x.buffer), x.byteorder == "big", float64._layout, bytes(radius.buffer), radius.byteorder == "big",
                                a, b - a, emin)


def radius_operand(x, radius, cols: int, emin, a: int = 0, b: int | None = None, workers=None):
    """``radius_bytes`` of elements a .. b - 1 (all by default) as an operand of rows of ``cols``, or None."""
    b = x.size if b is None else b
    widened = radius_bytes(x, radius, emin, a, b)
    if widened is None:
        return None
    rows = (b - a) // cols if cols else math.prod(x.shape[:-1])
    return _native.operand(widened, False, float64._layout, 0, rows, cols, cols, 1, max(rows, 1), 0, cols, True, workers=workers)


@dataclass(frozen=True)
class Methods:
    """The extra rounding points of a named attention algorithm (internal; see
    ``_algorithms.AttentionAlgorithm``). None stands for the standard
    algorithm, which rounds only at the required boundaries."""

    online_softmax: bool = False
    block_size: int | None = None
    exp2: bool = False
    weights: FloatFormat | None = None
    bias: FloatFormat | None = None
    reciprocal: bool = False
    backward_blocks: tuple[int, int] | None = None  # (query block, key block) of a fused backward
    mma: MMAStep | None = None  # the products' sums by tensor-core steps
    scale_ds: bool = False  # the backward scales dS before its rounding, not the gradients after their sums
    rounded_p: bool = False  # the backward forms dS from P rounded to the weights format
    grouped_heads: bool = False  # each query head's grad_key and grad_value rounded to the output format, then summed
    split_kv: int | None = None  # keys per split: each split's online softmax normalized on its own, merged through the logsumexps
    fast_log: bool = False  # the forward's logsumexp log within c ulps of the larger of its result and 1


# Each boundary's and accumulation's path in the forward's settings and in the backward's (None where the op has no such
# boundary), in the order the kernels' settings take them (attention_tol.c's ATT_SR_ and ATT_SC_).
_ROUND_SLOTS = (("scale_factor", "scale_factor"), ("exp2_scale", "exp2_scale"), ("scores.output", "scores.output"), ("logits", "logits"),
                ("bias", "bias"), ("exp2_bias", "exp2_bias"), ("softmax.shift", "softmax.shift"), ("softmax.exp", "softmax.exp"),
                ("softmax.rescale", None), ("weights", "weights"), ("values.output", None), ("reciprocal", "softmax.reciprocal"), ("normalize", None),
                ("logsumexp.log", None), ("logsumexp.add", None), ("exp2_logsumexp", "exp2_logsumexp"), ("split_kv.shift", None),
                ("split_kv.exp", None), ("split_kv.log", None), ("split_kv.add", None), ("split_kv.product", None),
                (None, "softmax.probabilities"), (None, "grad_probabilities.output"), (None, "grad_softmax.difference"),
                (None, "grad_softmax.grad_scores"), (None, "scale_gradients"), (None, "scale_ds"), (None, "grad_query.output"),
                (None, "grad_key.output"), (None, "grad_value.output"), (None, "grad_query.carry"), (None, "grad_key.carry"),
                (None, "grad_value.carry"), (None, "grad_key.group"), (None, "grad_value.group"))
_ACCUMULATION_SLOTS = (("scores.accumulate", "scores.accumulate"), ("softmax.sum", "softmax.sum"), ("values.accumulate", None),
                       ("split_kv.sum", None), ("split_kv.accumulate", None), (None, "grad_probabilities.accumulate"),
                       (None, "grad_softmax.row_dot"), (None, "grad_query.accumulate"), (None, "grad_key.accumulate"),
                       (None, "grad_value.accumulate"), (None, "grad_key.group_sum"), (None, "grad_value.group_sum"))
# per side (the forward's, the backward's), every boundary path a slot reads
_SLOTTED = tuple(frozenset(slot[side] for slot in _ROUND_SLOTS + _ACCUMULATION_SLOTS) - {None} for side in (0, 1))
_SA_ANY_ORDER, _SA_MMA = 1, 2


def _round_tuple(x):
    """A boundary's setting as the kernels read it: ``(layout, rounding, ulps numerator,
    ulps denominator, flush)``, None for anything but a rounding."""
    return (x.format._layout, _ROUNDING_CODES[x.rounding], x.ulps.numerator, x.ulps.denominator, x.flush) if isinstance(x, RoundTo) else None


def _accumulation_tuple(x):
    """An accumulation's setting as the kernels read it: ``(kind, layout, rounding, x, y)``, with
    mma's step and slack, or any order (whose bound covers every order; the C has its counts),
    None for anything but an accumulation."""
    if not isinstance(x, Accumulation):
        return None
    if x.mma is not None:
        step, slack = x.mma.step, Fraction(x.mma.slack)
        return (_SA_MMA, x.format._layout, _ROUNDING_CODES[x.rounding],
                (step.products, step.fraction_bits, step.exponent_floor, step.split, -1 if x.mma.count is None else x.mma.count),
                (None if x.mma.reduction is None else x.mma.reduction._layout, (slack.numerator, 1 - slack.denominator.bit_length())))
    return (_SA_ANY_ORDER, x.format._layout, _ROUNDING_CODES[x.rounding], None, None)


def _carry_accumulation(x):
    """backward_blocks: how a gradient's blocks are summed for Lemma C, in any order on the
    accumulator of its setting ``x``; None for anything but an accumulation."""
    return Accumulation(ANY_ORDER.key, x.format, rounding=x.rounding) if isinstance(x, Accumulation) else None


def kernel_settings(problem: Problem, settings: dict, flush_format: FloatFormat | None, methods: Methods | None, w: int,
                    backward: bool = False) -> object:
    """The call's settings for the kernels' constants (``_native.attention_settings``): each
    boundary's format, rounding, ulps and flush, each accumulation's (``_accumulation_tuple``),
    the scale and the methods. ``settings`` are the boundary settings by path, ``flush_format``
    the compute format when subnormal values in it may be flushed to zero, else None. Raises
    what ``_native.attention_settings`` raises for an unusable setting. ``backward`` says these
    are the backward's settings."""
    S, E = problem.S, problem.E
    flush = flush_format is not None
    side = 1 if backward else 0
    unslotted = settings.keys() - _SLOTTED[side]
    if unslotted:  # a setting no slot reads would leave its rounding out of the bound
        raise RuntimeError(f"internal error: no attention kernel slot for {', '.join(sorted(unslotted))}")
    rounds = [_round_tuple(None if slot[side] is None else settings.get(slot[side])) for slot in _ROUND_SLOTS]
    accumulation = [None if slot[side] is None else settings.get(slot[side]) for slot in _ACCUMULATION_SLOTS]
    # backward_blocks: each gradient's blocks carried in any order on its accumulator (Lemma C)
    if backward and methods is not None and methods.backward_blocks is not None:
        accumulation += [_carry_accumulation(settings.get(f"{name}.accumulate")) for name in ("grad_query", "grad_key", "grad_value")]
    else:
        accumulation += [None] * 3
    accumulations = [_accumulation_tuple(x) for x in accumulation]
    named = None
    if methods is not None:
        named = (methods.online_softmax and not backward, methods.exp2, methods.reciprocal, methods.fast_log, methods.block_size or 0,
                 methods.split_kv or 0, None if methods.weights is None else methods.weights._layout, methods.bias is not None, methods.mma is not None,
                 methods.rounded_p, methods.scale_ds, methods.grouped_heads, *(methods.backward_blocks or (0, 0)))
    return _native.attention_settings(tuple(rounds), tuple(accumulations), flush, flush_format.emin if flush else 0, w, S, E,
                                      problem.kernel_scale, named, backward)
