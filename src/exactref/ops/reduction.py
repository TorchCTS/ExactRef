"""``sum`` over chosen axes (category ``reduction.arithmetic``).

One call of matmul's product (a proxy): the reduced elements of each output are a row
and the column is ones, so every output is the exact sum rounded once, and the
tolerance, normal answer and results-as-inputs rules are matmul's with exact terms
(``proofs/sum.md``). A complex sum adds its real and imaginary parts separately.
"""

from __future__ import annotations

import builtins
import math
import operator

from .. import _native
from .._fenv import ieee
from ..array import Array, as_parts_layout, row_major_strides
from ..boundaries import NO_INTEGER_GRADIENT, OpSpec, Uses, register
from ..formats import Format, Rounding, int64, uint8
from ..result import Result, integer_bounds
from ._operands import Operand, grad_operand, holds, operand, repeated_operand, scalar_operand
from ._rounding import encode_options, output_format
from ._shapes import iter_indices
from ._product_core import combined, product, reshaped

register(
    OpSpec(
        "sum",
        boundaries={"accumulate": Uses("matmul", "the reduced elements times ones: a sum of exact terms")},
        guarantee="correctly_rounded",
        limitations=("integer and bool inputs are summed in int64, as PyTorch sums them",),
    )
)
register(
    OpSpec(
        "sum.backward",
        boundaries={"grad_x": Uses("matmul", "grad copied to every reduced element (grad * 1)")},
        guarantee="correctly_rounded",
        limitations=(NO_INTEGER_GRADIENT,),
    )
)

_ONE = 1


def _axes(axis, ndim: int) -> tuple:
    """The reduced dimensions, sorted: all for None, else the given int or ints."""
    if axis is None:
        return tuple(range(ndim))
    try:
        found = (operator.index(axis),)
    except TypeError:
        found = tuple(axis)
    dims = max(ndim, 1)  # a scalar takes axis 0 or -1, as in PyTorch, and reduces nothing
    axes = []
    for a in found:
        a = operator.index(a)
        if not -dims <= a < dims:
            raise ValueError(f"axis {a} is out of range for {ndim} dimensions")
        axes.append(a % dims)
    if len(set(axes)) != len(axes):
        raise ValueError(f"axis {axis} repeats a dimension")
    return tuple(sorted(axes)) if ndim else ()


def _reduction_groups(shape: tuple, axes: tuple, keepdims: bool) -> tuple:
    """For a reduction of ``shape`` over ``axes``: the positions of the elements grouped by
    output element (each group in row-major order; a range when the reduced dimensions are the
    last ones, so the groups are the rows), the number of outputs, the group size and the
    output shape."""
    kept = [d for d in range(len(shape)) if d not in axes]
    outer = [shape[d] for d in kept]
    reduced = [shape[d] for d in axes]
    out_shape = tuple(1 if d in axes else shape[d] for d in range(len(shape))) if keepdims else tuple(outer)
    if tuple(axes) == tuple(range(len(shape) - len(axes), len(shape))):
        return range(math.prod(shape)), math.prod(outer), math.prod(reduced), out_shape
    strides = row_major_strides(shape)
    bases = [builtins.sum(i * strides[d] for i, d in zip(o, kept)) for o in iter_indices(tuple(outer))]
    offsets = [builtins.sum(i * strides[d] for i, d in zip(r, axes)) for r in iter_indices(tuple(reduced))]
    return [b + o for b in bases for o in offsets], math.prod(outer), math.prod(reduced), out_shape


@ieee
def sum(x, axis=None, *, keepdims: bool = False, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
        workers: int | None = None) -> Result:
    """Sum over ``axis`` (None: every dimension; an int or a tuple of ints), keeping the
    reduced dimensions as 1 with ``keepdims``.

    Each output element is the exact sum rounded once to ``dtype`` (default: the input
    dtype; int64 for integer and bool inputs, as PyTorch sums them), which must be of
    that kind. ``tolerance()``
    budgets accumulation in the compute format and the final rounding; by default normal
    accumulation (index order of the reduced elements, row-major, or 2 or 4 SIMD-style
    lanes), with ``standard="any_order"`` any order. Each term is an input value, so it
    is not rounded on its own. A complex sum adds its real and imaginary parts as two
    separate sums. An exact zero sum is +0 (-0 when rounding toward -inf); an empty sum
    is +0. Integer results wrap on overflow unless ``overflow`` says otherwise. ``x`` may
    also be an earlier result. PyTorch's ``sum(x, dtype=d)`` converts ``x`` to ``d``
    before summing: for ``d`` of ``x``'s kind that is ``sum(multiply(x, 1, dtype=d))``,
    whose sum accumulates in ``d``'s compute format, and for another kind (an integer
    ``x`` summed in a float format) ``sum(asarray(x, d))``.
    """
    source = X = operand(x)
    fmt = int64 if X.array.dtype.kind in ("int", "bool") else X.array.dtype
    out = output_format(dtype, fmt)
    encode_options(out, rounding, overflow)  # the arguments, checked before any element is read
    axes = _axes(axis, X.array.ndim)
    if fmt is int64:
        X = X.converted(int64) if X.array.dtype.kind == "int" and holds(int64, X.array.dtype) else _as_int64(X)
    positions, outer, k, out_shape = _reduction_groups(X.array.shape, axes, keepdims)

    def backward(grad, **options):
        return _sum_backward(source, grad, axis=axis, keepdims=keepdims, **options)

    if fmt.kind != "complex":
        inner = product("sum", X.take(positions, (outer, k)), repeated_operand(_ONE, fmt.kind, k, fmt), out, rounding, overflow, fmt,
                        name="s", exact_terms=True, workers=workers, shape=out_shape)
        return reshaped(inner, out_shape, backward)
    # A complex sum: its parts are two real sums, side by side.
    parts = X.as_real()
    pair_positions = [] if k == 0 else [2 * p + part for start in range(0, len(positions), k) for part in (0, 1) for p in positions[start : start + k]]
    component = fmt.component
    inner = product("sum", parts.take(pair_positions, (2 * outer, k)), repeated_operand(_ONE, "float", k, component), out.component,
                    rounding, overflow, component, name="s", exact_terms=True, workers=workers)

    def tolerance(compute, ulps, flush, standard, workers=None):
        return {"s": inner._bounds_of(compute, ulps, flush, standard, workers).paired(out, out_shape)}

    def normal(compute, flush, workers=None):
        return {"s": as_parts_layout(inner.normal(compute, flush_to_zero=flush, workers=workers), out, out_shape)}

    exact = inner.exact.paired(out_shape)
    return Result("sum", inner.guarantee, {"s": as_parts_layout(inner._single(), out, out_shape)}, {"s": exact}, tolerance, backward, normal)


def _as_int64(X: Operand) -> Operand:
    """A bool or uint64 operand's values in int64, uint64 wrapping as in PyTorch's sum (the
    other integer formats widen exactly, ``Operand.converted``). An earlier result's stored values
    convert the same way: its range's ends do, and a range that wrapping splits covers every
    int64 value (``integer_bounds``)."""
    shape = X.array.shape

    def converted(array: Array) -> Array:
        # the C re-encodes every element at once, wrapping (a bool's byte read as uint8)
        source = uint8 if array.dtype.kind == "bool" else array.dtype
        return Array(int64, shape, _native.recode(bytes(array.buffer), array.byteorder == "big", source._layout, int64._layout))

    values = converted(X.array)
    if not X.uncertain:
        return Operand(values)

    def bounds(compute, ulps, flush, standard, workers=None):
        found = X.bounds(compute, ulps, flush, standard, workers)
        return integer_bounds(values, "wrap", [(int(lo), int(hi)) for lo, hi in zip(found.low.values(), found.high.values())])

    def normal(compute, flush, workers=None):
        return converted(X.normal(compute, flush, workers))

    return Operand(values, bounds, normal)


@ieee
def _sum_backward(x, grad, axis=None, *, keepdims: bool = False, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
                  overflow: str | None = None, workers: int | None = None) -> Result:
    """Gradient of ``sum(x, axis, keepdims=keepdims)`` for the output gradient ``grad``.

    ``grad_x`` holds, at every element of ``x``, the gradient of the output it was summed
    into, in ``x``'s dtype (rounded once if ``grad``'s dtype is wider); an exact zero keeps
    ``grad``'s sign, as in PyTorch. ``grad`` must have the output's shape. ``x`` and
    ``grad`` may also be earlier results.
    """
    X = operand(x)
    out = output_format(dtype, X.array.dtype)
    encode_options(out, rounding, overflow)  # the arguments, checked before any element is read
    axes = _axes(axis, X.array.ndim)
    positions, outer, k, out_shape = _reduction_groups(X.array.shape, axes, keepdims)
    G = grad_operand("sum", grad, out_shape, X.array.dtype)
    # Element p of x was summed into output source[p].
    source = [0] * X.array.size
    for at, p in enumerate(positions):
        source[p] = at // k if k else 0
    shape = X.array.shape
    fmt = G.array.dtype
    one = scalar_operand([_ONE], fmt.kind, (1, 1), fmt)
    inner = product("sum.backward", G.take(source, shape + (1, 1)), one, out, rounding, overflow, fmt,
                    workers=workers,
                    name="grad_x", exact_terms=True, coefficients=True, few_terms=True, elementwise=True)
    return combined("sum.backward", {"grad_x": reshaped(inner, shape, None)})


sum.backward = _sum_backward
