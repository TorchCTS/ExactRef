"""Named attention algorithms: tolerance-only descriptions of how fused kernels round."""

from __future__ import annotations

import math
from fractions import Fraction

from ..._fenv import ieee
from ...algorithm import Algorithm
from ...accumulation import MMA, MMAStep
from ...array import Array
from ...boundaries import Accumulation, RoundTo, required_settings
from ...formats import FloatFormat, Format, Rounding, bfloat16, float16, float32, get_format
from ...result import Outputs, check_kind, check_standard, checked_output, named_outputs
from .._mma import check_step, check_unflushed, extent, slack
from .._operands import operand, unflagged_radius
from .._rounding import compute_format, encode_options, output_format, require_real_float, tolerance_ulps
from . import _backward, _forward
from ._problem import DEFAULT_ULPS, Methods, Problem, check_dropout


class _Given:
    """The forward's rounded output and logsumexp, as a fused backward kernel receives them
    (arrays, or earlier results such as ``forward["output"]`` and ``forward["logsumexp"]``)."""

    def __init__(self, problem: Problem, out, logsumexp) -> None:
        self.sources = {"out": operand(out), "lse": operand(logsumexp)}
        out, logsumexp = self.sources["out"].array, self.sources["lse"].array
        self.out_array, self.lse_array = out, logsumexp
        if out.shape != problem.out_shape:
            raise ValueError(f"out must have the output's shape {problem.out_shape}, got {out.shape}")
        if logsumexp.shape != problem.out_shape[:-1]:
            raise ValueError(f"logsumexp must have shape {problem.out_shape[:-1]}, got {logsumexp.shape}")
        for name, x in (("out", out), ("logsumexp", logsumexp)):
            if x.dtype.kind != "float":
                raise TypeError(f"{name} must be floating point, got {x.dtype.name}")

    def radii(self, compute, ulps, flush: bool, standard: str, workers=None) -> dict[str, Array | None]:
        return {name: unflagged_radius(x, compute, ulps, flush, standard, workers) for name, x in self.sources.items()}


class AttentionAlgorithm(Algorithm):
    """A fused-kernel way of computing attention, described by its extra rounding points.

    The reference is unchanged (``scaled_dot_product_attention``); an algorithm
    has only bounds (``tolerance()`` and the other views ``finite_range()``,
    ``may_be()`` and ``covers()``, each with a ``backward_`` form): how far an
    implementation that rounds at the required boundaries and at these points
    can be from the correct answer.
    It has no ``run()``, because kernels differ in details these methods leave
    open. Methods, any combination:

    - ``online_softmax``: keys processed in blocks with a running maximum and
      sum, rescaling earlier partial sums by ``exp(old max - new max)``. With
      ``block_size`` None, any blocking and order; otherwise at most
      ceil(S / ``block_size``) blocks of at most ``block_size`` keys (as kernels
      split the keys: every block full but one), in any order.
    - ``exp2``: weights computed as ``exp2`` of logits in base-2 units (the
      scale times log2(e), and mask values times log2(e), held in the compute
      format within ``ulps``).
    - ``weights``: the weights rounded to this format before the value
      product (and in the backward, P and dS before their products).
    - ``accumulate``: every accumulation in this format instead of the compute
      format.
    - ``bias``: float-mask values rounded to this format before they are added
      (a kernel computing ALiBi biases in lower precision).
    - ``reciprocal``: normalization by multiplying with the row sum's
      reciprocal, rounded once, instead of dividing: for the output, and for
      the backward's probabilities (PyTorch's CPU softmax and its CPU flash
      kernel do this). That kernel's forward for bfloat16 and float16 inputs
      also needs ``exp_ulps``.
    - ``exp_ulps``: for a kernel with an approximate exp, each key's softmax
      exponential (exp, or exp2 with ``exp2``) within this many ulps instead
      of the call's ``ulps``, and where the exact result is below the
      compute format's smallest normal value, within that value of it (an
      exp that returns 0 or a wrongly encoded subnormal there). The
      logsumexp's log, online softmax's rescaling factors and the backward
      keep ``ulps``.
    - ``mma``: an ``MMAStep``: the products' sums (the scores and the value
      sums, and in the backward the scores, dP and the three gradients' sums)
      by tensor-core steps with its parameters (``proofs/foundations.md``,
      Lemma T), in float32. The row sums and the backward's D keep the
      compute format. Needs ``weights`` equal to the inputs' format (float16
      or bfloat16), which P and dS are rounded to before their products; not
      with ``accumulate`` or ``backward_blocks``. ``standard="normal"`` gives
      the any-order bound, and every row's bound is the step-by-step program's.
    - ``split_kv``: keys per split, a positive int, for a forward that cuts
      each row's keys into splits of this many consecutive keys (the last may
      be shorter), as FlashAttention-2's split-KV kernels do. Each split is an
      online softmax of its own (``online_softmax`` is required), whose value
      sums are normalized by its own row sum in the compute format and whose
      logsumexp is computed as the forward's. The splits are merged in the
      compute format: with M the largest
      split logsumexp, the logsumexp is ``log(sum exp(lse_i - M)) + M`` and
      the output ``sum exp(lse_i - logsumexp) O_i``, rounded at each step,
      exp and log within ``ulps``, sums in any order (as FlashAttention-2
      merges). Every row's bound is the step-by-step program's.
    - ``fast_log``: the forward's logsumexp takes the log of the row sum
      within ``ulps`` ulps of the larger of its result and 1, as a fast
      logarithm does that is accurate in absolute terms near 1 (CUDA's
      ``__logf``, which FlashAttention-2 calls); with ``split_kv``, each
      split's log too.
    - ``scale_ds``: the backward multiplies dS by the scale, rounded once in
      the compute format, before its rounding to ``weights`` (cuDNN's and
      PyTorch's memory-efficient kernels), so grad_query and grad_key are not
      scaled after their sums (no ``scale_gradients`` rounding). Not with
      ``backward_blocks``.
    - ``rounded_p``: the backward forms dS from P already rounded to
      ``weights`` (PyTorch's memory-efficient kernel). Needs ``weights``.
    - ``grouped_heads``: the backward rounds each query head's grad_key and
      grad_value (each batch copy's too) to the output format, then adds
      them in the compute format in any order and rounds once more (PyTorch's
      fused kernels with grouped or broadcast key and value heads).
      ``standard="normal"`` gives the any-order bound; not with
      ``backward_blocks``.
    - ``backward_blocks``: ``(query_block, key_block)``, two positive ints, for
      a fused backward kernel that adds the gradients' products block by
      block into gradients held in the output format: grad_key and
      grad_value are rounded to it after each block of ``query_block``
      queries (each query head of a group in turn), grad_query after each
      block of ``key_block`` keys, and the scale is applied to each block.
      This replaces those gradients' nested output roundings and, for
      grad_query and grad_key, the ``scale_gradients`` rounding. Fused
      backward only (``backward_tolerance`` with ``out`` and
      ``logsumexp``); a broadcast query, or a key or value broadcast across
      batch dimensions, raises ValueError (pass full-size inputs).

    ``backward_tolerance`` given the forward's rounded ``out`` and
    ``logsumexp`` describes a fused backward kernel: it recomputes
    ``P = exp(logit - logsumexp)`` and takes ``D = sum grad * out``. Without
    them it describes a backward that recomputes the softmax, as PyTorch's
    math path does. ``online_softmax`` applies to the forward only.

    Every input of ``tolerance`` and ``backward_tolerance``, ``out`` and
    ``logsumexp`` included, may also be an earlier result, as ``Result``
    describes. The bounds are proven in ``proofs/attention_algorithms.md``.
    """

    op = "scaled_dot_product_attention"

    def __init__(
        self,
        *,
        online_softmax: bool = False,
        block_size: int | None = None,
        exp2: bool = False,
        weights: Format | str | None = None,
        accumulate: Format | str | None = None,
        bias: Format | str | None = None,
        reciprocal: bool = False,
        backward_blocks: tuple[int, int] | None = None,
        exp_ulps: int | Fraction | None = None,
        mma: MMAStep | None = None,
        scale_ds: bool = False,
        rounded_p: bool = False,
        grouped_heads: bool = False,
        split_kv: int | None = None,
        fast_log: bool = False,
    ) -> None:
        if block_size is not None and type(block_size) is not int:
            raise TypeError("block_size must be an int")
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be positive")
        if block_size is not None and not online_softmax:
            raise ValueError("block_size applies to online_softmax")
        if split_kv is not None and type(split_kv) is not int:
            raise TypeError("split_kv must be an int")
        if split_kv is not None and split_kv < 1:
            raise ValueError("split_kv must be positive")
        if split_kv is not None and not online_softmax:
            raise ValueError("split_kv needs online_softmax: each split is an online softmax")
        if backward_blocks is not None:
            if not isinstance(backward_blocks, (tuple, list)) or any(type(b) is not int for b in backward_blocks):
                raise TypeError("backward_blocks must be (query_block, key_block), two ints")
            if len(backward_blocks) != 2 or any(b < 1 for b in backward_blocks):
                raise ValueError("backward_blocks must be (query_block, key_block), two positive ints")
            backward_blocks = tuple(backward_blocks)
        self.exp_ulps = None if exp_ulps is None else tolerance_ulps(exp_ulps, None)
        formats = {}
        for name, value in (("weights", weights), ("accumulate", accumulate), ("bias", bias)):
            formats[name] = None if value is None else require_real_float(get_format(value), f"an attention algorithm's {name}")
        if mma is not None:
            check_step(mma)
            if formats["weights"] is None:
                raise ValueError("mma needs weights: the kernel rounds P (and dS) to the inputs' format before their products")
            if formats["accumulate"] is not None or backward_blocks is not None:
                raise ValueError("mma does not combine with accumulate or backward_blocks")
        if (scale_ds or grouped_heads) and backward_blocks is not None:
            raise ValueError("scale_ds and grouped_heads do not combine with backward_blocks")
        if rounded_p and formats["weights"] is None:
            raise ValueError("rounded_p needs weights: dS is formed from P rounded to that format")
        self.methods = Methods(online_softmax, block_size, exp2, formats["weights"], formats["bias"], bool(reciprocal), backward_blocks, mma,
                               bool(scale_ds), bool(rounded_p), bool(grouped_heads), split_kv, bool(fast_log))
        self.accumulate: FloatFormat | None = formats["accumulate"]
        weights, accumulation, bias = formats["weights"], formats["accumulate"], formats["bias"]
        # (name part, description) of each method in use, in order; a method not in use is falsy and dropped
        methods = [m for m in (
            online_softmax and ("online_softmax" if block_size is None else f"online_softmax{block_size}", "online softmax over "
                                + ("any blocking" if block_size is None else f"at most ceil(S / {block_size}) blocks of at most {block_size} keys")),
            exp2 and ("exp2", "exp2 of base-2 logits"),
            weights is not None and (f"weights_{weights.name}", f"weights rounded to {weights.name}"),
            accumulation is not None and (f"accumulate_{accumulation.name}", f"accumulation in {accumulation.name}"),
            bias is not None and (f"bias_{bias.name}", f"mask values rounded to {bias.name}"),
            reciprocal and ("reciprocal", "normalization by a rounded reciprocal"),
            backward_blocks is not None and ("backward_blocks{}x{}".format(*backward_blocks),
                                             "backward sums carried in the output format over blocks of {} queries and {} keys".format(*backward_blocks)),
            self.exp_ulps is not None and (f"exp_ulps{self.exp_ulps}", f"each key's exp within {self.exp_ulps} ulps"),
            mma is not None and (f"mma{mma.products}_f{mma.fraction_bits}", f"products summed by tensor-core steps of {mma.products} products, "
                                 f"{mma.fraction_bits} fraction bits, exponent floor {mma.exponent_floor}"),
            split_kv is not None and (f"split_kv{split_kv}", f"keys in splits of {split_kv}, each normalized on its own, merged through their logsumexps"),
            fast_log and ("fast_log", "the logsumexp's log within ulps of 1 near 1"),
            scale_ds and ("scale_ds", "dS scaled before its rounding"),
            rounded_p and ("rounded_p", "dS from the rounded weights"),
            grouped_heads and ("grouped_heads", "gradients of grouped heads rounded per head, then summed"),
        ) if m]
        self.name = "attention_" + "_".join(part for part, _ in methods) if methods else "attention_standard"
        self.description = ", ".join(described for _, described in methods) if methods else "the required boundaries only"

    def _check_mma_inputs(self, problem: Problem) -> None:
        """With ``mma``, the inputs' format is the weights' one, float16 or bfloat16 (checked before any radius)."""
        weights, fmt = self.methods.weights, problem.dtype
        if self.methods.mma is not None and (fmt not in (float16, bfloat16) or weights != fmt):
            raise TypeError(f"mma takes float16 or bfloat16 inputs in its weights format {weights.name}, got {fmt.name}")

    def _settings(self, op: str, fmt: FloatFormat, ulps, sums: dict | None = None) -> dict:
        """The boundary settings; ``sums`` maps each product sum's path to its ``(count, slack)``
        for ``mma``."""
        c = tolerance_ulps(ulps, DEFAULT_ULPS)
        settings = required_settings(op, fmt, c)
        for path, (count, sigma) in (sums or {}).items():
            settings[path] = Accumulation("mma", float32, mma=MMA(self.methods.mma, None, sigma, count))
        if self.methods.split_kv is not None and op == "scaled_dot_product_attention":  # the merge of the splits
            settings.update({"split_kv.shift": RoundTo(fmt), "split_kv.exp": RoundTo(fmt, ulps=c), "split_kv.sum": Accumulation("any_order", fmt),
                             "split_kv.log": RoundTo(fmt, ulps=c), "split_kv.add": RoundTo(fmt), "split_kv.product": RoundTo(fmt),
                             "split_kv.accumulate": Accumulation("any_order", fmt)})
        if self.accumulate is not None:
            for path, setting in settings.items():
                if isinstance(setting, Accumulation):
                    settings[path] = Accumulation("any_order", self.accumulate)
        m = self.methods
        if m.weights is not None:
            settings["weights"] = RoundTo(m.weights)
        if m.bias is not None:
            settings["bias"] = RoundTo(m.bias)
        if m.exp2:
            for path in ("exp2_scale", "exp2_bias", "exp2_logsumexp"):
                settings[path] = RoundTo(fmt, ulps=c)
        if m.reciprocal:
            settings["reciprocal" if op == "scaled_dot_product_attention" else "softmax.reciprocal"] = RoundTo(fmt)
        if m.scale_ds and op != "scaled_dot_product_attention":
            settings["scale_ds"] = RoundTo(fmt)
        if self.exp_ulps is not None and op == "scaled_dot_product_attention":  # the forward's own exponentials only
            settings["softmax.exp"] = RoundTo(fmt, ulps=self.exp_ulps, flush=True)
            if m.online_softmax:
                settings["softmax.rescale"] = RoundTo(fmt, ulps=c)  # the rescaling factors keep the call's ulps
        return settings

    def _bounds(
        self,
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
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> dict:
        """Per output (``output``, ``logsumexp``), its ``Bounds``; the arguments are ``tolerance``'s."""
        check_dropout(dropout_p)
        if self.methods.mma is not None:
            check_unflushed(flush_to_zero)
        problem = Problem(query, key, value, attn_mask, is_causal, scale, enable_gqa)
        self._check_mma_inputs(problem)
        options = encode_options(output_format(dtype, problem.dtype), rounding, overflow)
        fmt = compute_format(compute, problem.dtype)
        standard = check_standard(standard)
        # the inputs' radii under the caller's standard; mma's own sums are any-order under both
        radii = problem.radii(compute, ulps, flush_to_zero, standard, workers)
        sums = None if self.methods.mma is None else _mma_sums(problem, radii, None, None, "forward")
        settings = self._settings("scaled_dot_product_attention", fmt, ulps, sums)
        reference = _forward.forward(problem, dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)
        out, lse = reference.rounded.output, reference.rounded.logsumexp
        exact = (reference.exact.output, reference.exact.logsumexp)
        found, lse_found = _forward.kernel_bounds(problem, out.dtype, settings, fmt, rounding, out, flush_to_zero, self.methods, lse=(lse.dtype, lse),
                                                  exact=exact, standard="any_order" if self.methods.mma is not None else standard, radii=radii,
                                                  overflow=options["overflow"], workers=workers)
        return {"output": found, "logsumexp": lse_found}

    @ieee
    def tolerance(
        self,
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
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """Worst-case ``|output - reference|`` per element of ``output`` and ``logsumexp``, over
        this algorithm's finite outputs (``Result.tolerance``).

        The arguments are ``scaled_dot_product_attention``'s (``overflow`` the implementation's
        overflow mode); ``compute``, ``ulps``, ``flush_to_zero``, ``standard`` and ``workers`` are as in
        ``Result.tolerance``.
        With ``online_softmax``, ``standard="normal"`` gives the any-order bound."""
        found = self._bounds(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, dtype=dtype, rounding=rounding,
                             overflow=overflow, compute=compute, ulps=ulps, flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: b.tolerance for name, b in found.items()})

    @ieee
    def finite_range(
        self,
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
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """Per output, the lowest and highest finite output of this algorithm, as
        ``Result.finite_range`` describes; the arguments are ``tolerance``'s."""
        found = self._bounds(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, dtype=dtype, rounding=rounding,
                             overflow=overflow, compute=compute, ulps=ulps, flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: (b.low, b.high) for name, b in found.items()})

    @ieee
    def may_be(
        self,
        kind: str,
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
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """Per output, where this algorithm may output ``kind`` (``"+inf"``, ``"-inf"`` or
        ``"nan"``), as ``Result.may_be`` describes; the other arguments are ``tolerance``'s."""
        check_kind(kind)
        found = self._bounds(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, dtype=dtype, rounding=rounding,
                             overflow=overflow, compute=compute, ulps=ulps, flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: b.may_be(kind) for name, b in found.items()})

    @ieee
    def covers(
        self,
        outputs: dict,
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
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """For each output named in ``outputs`` (``{"output": ..., "logsumexp": ...}``, either
        or both), where this algorithm can produce that value, as ``Result.covers``
        describes; the other arguments are ``tolerance``'s."""
        problem = Problem(query, key, value, attn_mask, is_causal, scale, enable_gqa)
        formats = _forward.output_formats(problem, dtype)
        outputs = {name: checked_output(array, *formats[name]) for name, array in named_outputs(outputs, formats).items()}
        s = problem.sources
        found = self._bounds(s["q"], s["k"], s["v"], s["mask"], dropout_p, is_causal, scale, enable_gqa, dtype=dtype, rounding=rounding,
                             overflow=overflow, compute=compute, ulps=ulps, flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: found[name].covers(array) for name, array in outputs.items()})

    def _backward_bounds(
        self,
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
        out=None,
        logsumexp=None,
        dtype: Format | str | None = None,
        rounding: Rounding = Rounding.NEAREST_EVEN,
        overflow: str | None = None,
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> dict:
        """Per gradient, its ``Bounds``; the arguments are ``backward_tolerance``'s."""
        check_dropout(dropout_p)
        if self.methods.mma is not None:
            check_unflushed(flush_to_zero)
        if (out is None) != (logsumexp is None):
            raise ValueError("pass both out and logsumexp, or neither")
        problem = Problem(query, key, value, attn_mask, is_causal, scale, enable_gqa)
        self._check_mma_inputs(problem)
        options = encode_options(output_format(dtype, problem.dtype), rounding, overflow)
        fmt = compute_format(compute, problem.dtype)
        standard = check_standard(standard)
        blocked = self.methods.backward_blocks is not None
        if blocked:
            _check_blocked(problem, out)
        grad_problem = _backward.GradProblem(problem, grad)
        given = _Given(problem, out, logsumexp) if out is not None else None
        # the inputs' radii under the caller's standard; mma's and grouped_heads' own sums are any-order under both
        radii = grad_problem.radii(compute, ulps, flush_to_zero, standard, workers) or {}
        if given is not None:
            radii.update((name, r) for name, r in given.radii(compute, ulps, flush_to_zero, standard, workers).items() if r is not None)
        sums = None
        if self.methods.mma is not None:
            sums = _mma_sums(problem, radii, grad_problem.grad_source, (compute, ulps, flush_to_zero, standard, workers), "backward")
        settings = self._settings("scaled_dot_product_attention.backward", fmt, ulps, sums)
        reference = _backward.backward(grad_problem, dtype=dtype, rounding=rounding, overflow=overflow, workers=workers)
        names = _backward.NAMES
        references = {name: reference.rounded[name] for name in names}
        out_format = references["grad_value"].dtype
        if blocked:
            for name in names:
                settings[f"{name}.carry"] = RoundTo(out_format)  # the gradients held in the output format
        if self.methods.grouped_heads:
            for name in ("grad_key", "grad_value"):
                settings[f"{name}.group"] = RoundTo(out_format)  # each head's gradient in the output format
                settings[f"{name}.group_sum"] = Accumulation("any_order", fmt)  # their sum
        kernel_standard = "any_order" if self.methods.mma is not None or self.methods.grouped_heads else standard
        return _backward.kernel_bounds(grad_problem, out_format, settings, fmt, rounding, references, flush_to_zero, self.methods, given=given,
                                       standard=kernel_standard, radii=radii or None, overflow=options["overflow"],
                                       workers=workers)

    @ieee
    def backward_tolerance(
        self,
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
        out=None,
        logsumexp=None,
        dtype: Format | str | None = None,
        rounding: Rounding = Rounding.NEAREST_EVEN,
        overflow: str | None = None,
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """Worst-case ``|gradient - reference|`` per element of ``grad_query``,
        ``grad_key`` and ``grad_value``, over this algorithm's finite outputs. Given the forward's rounded ``out`` and
        ``logsumexp`` (any values of the right shapes; a non-finite logsumexp
        or output gives an infinite bound where it is used), for a fused
        backward kernel; without them, for a backward that recomputes the
        softmax. ``compute``, ``ulps``, ``flush_to_zero``, ``standard`` and
        ``workers`` are as in ``Result.tolerance``."""
        found = self._backward_bounds(query, key, value, grad, attn_mask, dropout_p, is_causal, scale, enable_gqa, out=out, logsumexp=logsumexp,
                                      dtype=dtype, rounding=rounding, overflow=overflow, compute=compute, ulps=ulps,
                                      flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: b.tolerance for name, b in found.items()})

    @ieee
    def backward_finite_range(
        self,
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
        out=None,
        logsumexp=None,
        dtype: Format | str | None = None,
        rounding: Rounding = Rounding.NEAREST_EVEN,
        overflow: str | None = None,
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """Per gradient, the lowest and highest finite output of this algorithm, as
        ``Result.finite_range`` describes; the arguments are ``backward_tolerance``'s."""
        found = self._backward_bounds(query, key, value, grad, attn_mask, dropout_p, is_causal, scale, enable_gqa, out=out, logsumexp=logsumexp,
                                      dtype=dtype, rounding=rounding, overflow=overflow, compute=compute, ulps=ulps,
                                      flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: (b.low, b.high) for name, b in found.items()})

    @ieee
    def backward_may_be(
        self,
        kind: str,
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
        out=None,
        logsumexp=None,
        dtype: Format | str | None = None,
        rounding: Rounding = Rounding.NEAREST_EVEN,
        overflow: str | None = None,
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """Per gradient, where this algorithm may output ``kind`` (``"+inf"``, ``"-inf"`` or
        ``"nan"``), as ``Result.may_be`` describes; the other arguments are
        ``backward_tolerance``'s."""
        check_kind(kind)
        found = self._backward_bounds(query, key, value, grad, attn_mask, dropout_p, is_causal, scale, enable_gqa, out=out, logsumexp=logsumexp,
                                      dtype=dtype, rounding=rounding, overflow=overflow, compute=compute, ulps=ulps,
                                      flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: b.may_be(kind) for name, b in found.items()})

    @ieee
    def backward_covers(
        self,
        gradients: dict,
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
        out=None,
        logsumexp=None,
        dtype: Format | str | None = None,
        rounding: Rounding = Rounding.NEAREST_EVEN,
        overflow: str | None = None,
        compute: Format | str | None = None,
        ulps: int | Fraction | None = None,
        flush_to_zero: bool = False,
        standard: str = "normal",
        workers: int | None = None,
    ) -> Outputs:
        """For each gradient named in ``gradients`` (``grad_query``, ``grad_key``,
        ``grad_value``), where this algorithm can produce that value, as ``Result.covers``
        describes; the other arguments are ``backward_tolerance``'s."""
        problem = Problem(query, key, value, attn_mask, is_causal, scale, enable_gqa)
        formats = _backward.gradient_formats(problem, dtype)
        gradients = {name: checked_output(array, *formats[name]) for name, array in named_outputs(gradients, formats).items()}
        s = problem.sources
        found = self._backward_bounds(s["q"], s["k"], s["v"], grad, s["mask"], dropout_p, is_causal, scale, enable_gqa, out=out,
                                      logsumexp=logsumexp, dtype=dtype, rounding=rounding, overflow=overflow, compute=compute, ulps=ulps,
                                      flush_to_zero=flush_to_zero, standard=standard, workers=workers)
        return Outputs({name: found[name].covers(array) for name, array in gradients.items()})


def _mma_sums(problem: Problem, radii, grad, grad_options, direction: str) -> dict:
    """For ``mma``, each product sum's path and its ``(count, slack)``: the products its kernel
    processes per element, zeros and broadcast copies included, and Lemma T's sigma. P and dS
    are rounded to the inputs' format W before their products, so they are at most W's largest
    value and may be subnormal; the inputs' extents come from their values and radii."""
    fmt = problem.dtype
    radii = radii or {}
    found = {name: extent(x, radii.get(name)) for name, x in (("q", problem.q), ("k", problem.k), ("v", problem.v))}
    if grad is not None:
        found["g"] = extent(grad.array, grad.radius(*grad_options))
    computed = (True, fmt.max_finite.to_fraction())  # P or dS

    rows = math.prod(problem.batch)

    def copies(x) -> int:  # how many output rows of the batch each row of x serves (none in an empty batch)
        return rows // math.prod(x.shape[:-2]) if rows else 0

    q, k, v = found["q"], found["k"], found["v"]
    if direction == "forward":
        return {"scores.accumulate": (problem.E, slack(fmt, q, k)), "values.accumulate": (problem.S, slack(fmt, computed, v))}
    g = found["g"]
    return {
        "scores.accumulate": (problem.E, slack(fmt, q, k)),
        "grad_value.accumulate": (problem.L * copies(problem.v), slack(fmt, computed, g)),
        "grad_probabilities.accumulate": (problem.Ev, slack(fmt, g, v)),
        "grad_query.accumulate": (problem.S * copies(problem.q), slack(fmt, computed, k)),
        "grad_key.accumulate": (problem.L * copies(problem.k), slack(fmt, computed, q)),
    }


def _check_blocked(problem: Problem, out) -> None:
    """``backward_blocks`` describes a fused backward on full-size inputs: it needs the forward's
    out and logsumexp, a query that is not broadcast, and a key and value not broadcast
    across batch dimensions (grouped or broadcast heads are allowed)."""
    if out is None:
        raise ValueError("backward_blocks describes a fused backward: pass out and logsumexp")
    batch = problem.batch
    if _broadcast(problem.q.shape[:-2], batch):
        raise ValueError("backward_blocks needs a query that is not broadcast; pass full-size inputs")
    for name, x in (("key", problem.k), ("value", problem.v)):
        if _broadcast(x.shape[:-3], batch[:-1]):
            raise ValueError(f"backward_blocks needs a {name} that is not broadcast across batch dimensions; pass full-size inputs")


def _broadcast(shape: tuple, target: tuple) -> bool:
    """Whether ``shape`` broadcasts to ``target`` other than as itself."""
    return (1,) * (len(target) - len(shape)) + tuple(shape) != tuple(target)
