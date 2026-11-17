"""Elementwise ``multiply`` and ``add`` (category ``arithmetic.basic``).

Each is one call of matmul's product (a proxy): ``multiply`` makes every element a
one-term product, ``add`` makes every element the two-term dot ``[x, y] . [1, alpha]``,
and each backward gradient is one product summed over the broadcast copies. So every
output is the exact value rounded once, the normal answer and the results-as-inputs
rules are matmul's, and so is the tolerance, tightened for elements of few terms (Lemma
F) and for ``add`` with ``alpha`` 1 or -1 (Lemma U) (``proofs/multiply.md``,
``proofs/add.md``, ``proofs/foundations.md``). The named algorithms
``multiply_held_scalar`` and ``add_held_scalar`` describe implementations that round each
scalar operand (and ``alpha``) to a format first: the operation's model at the held
values, its bounds measured from the operation's reference (Held scalars).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from .._fenv import ieee
from ..algorithm import Algorithm
from ..array import Array
from ..boundaries import NO_INTEGER_GRADIENT, OpSpec, Uses, register
from ..formats import FORMATS, ComplexFormat, FloatFormat, Format, Rounding, _part_format, get_format
from ..result import Result, check_kind, check_standard, check_ulps
from ..values import ComplexValue, FloatValue
from ._operands import (Operand, broadcast_to, exact_scalar, grad_operand, holding_format, holds, is_unit, operand, promoted_format, repeated_operand,
                        scalar_operand, stacked)
from ._rounding import compute_format, encode_options, output_format
from ._shapes import broadcast_shapes
from ._product_core import combined, conjugate_scalar, gradient, product, reshaped

_LIMITATIONS = (
    "array operands must be of one kind (integer, floating-point or complex); other mixes are converted by the consumer",
    "bool operands are not supported",
)

register(
    OpSpec(
        "multiply",
        boundaries={"product": Uses("matmul", "x * y: one product per element")},
        guarantee="correctly_rounded",
        limitations=_LIMITATIONS,
    )
)
register(
    OpSpec(
        "multiply.backward",
        boundaries={
            "grad_x": Uses("matmul", "grad * conj(y), summed over x's broadcast copies"),
            "grad_y": Uses("matmul", "grad * conj(x), summed over y's broadcast copies"),
        },
        guarantee="correctly_rounded",
        limitations=(
            NO_INTEGER_GRADIENT,
            "where an operand is broadcast, PyTorch rounds each product to the dtype before summing the copies, "
            "an extra rounding the default tolerance does not budget",
        ),
    )
)
register(
    OpSpec(
        "add",
        boundaries={"sum": Uses("matmul", "x * 1 + y * alpha: two products per element")},
        guarantee="correctly_rounded",
        limitations=_LIMITATIONS,
    )
)
register(
    OpSpec(
        "add.backward",
        boundaries={
            "grad_x": Uses("matmul", "grad, summed over x's broadcast copies"),
            "grad_y": Uses("matmul", "grad * conj(alpha), summed over y's broadcast copies"),
        },
        guarantee="correctly_rounded",
        limitations=(
            NO_INTEGER_GRADIENT,
            "where an operand is broadcast, PyTorch may round each product to the dtype before summing the copies, "
            "an extra rounding the default tolerance does not budget",
        ),
    )
)

_SCALARS = (int, float, complex, Fraction, FloatValue, ComplexValue)


def promoted(op: str, xs: list, dtype=None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None) -> tuple[list[Operand], Format, Format]:
    """Array operands (arrays or earlier results) as operands in their promoted format, that
    format, and the output format ``dtype`` gives, checked with ``rounding`` and ``overflow``
    before any operand is converted. As in PyTorch, operands of shape () take part in the
    promotion only when every operand has shape (); one the promoted format does not hold
    (wider, beside an operand with dimensions) stays in its own format."""
    found = [operand(x) for x in xs]
    dtypes = [X.array.dtype for X in found]
    fmt = promoted_format([X.array.dtype for X in found if X.array.ndim] or dtypes)
    if any(f.kind != fmt.kind for f in dtypes):
        raise TypeError(f"{op} operands must be of one kind, got {', '.join(f.name for f in dtypes)}")
    if fmt.kind == "bool":
        raise TypeError(f"{op} does not support bool operands")
    out_format = output_format(dtype, fmt)
    encode_options(out_format, rounding, overflow)
    return [X.converted(fmt) if holds(fmt, X.array.dtype) else X for X in found], fmt, out_format


def _sources(*xs) -> tuple:
    """The operands as the forward reads them: arrays and earlier results as ``Operand``s, scalars
    as given; the backward is computed from these, so a caller's later change to an input does
    not reach it."""
    return tuple(x if isinstance(x, _SCALARS) else operand(x) for x in xs)


def _inputs(op: str, *xs, dtype=None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None) -> tuple:
    """Array operands (arrays or earlier results, ``promoted``) and scalars: each as an
    operand broadcast to the common shape; then the promoted format, the output format
    (``promoted`` checks it and the modes first) and the shape. Scalars do not take part in
    the promotion (as in PyTorch), and neither does an operand of shape () beside one with
    dimensions. A scalar, and an array of shape () that the promoted format does not hold,
    is taken by its value, held exactly (``scalar_operand``); an earlier result of shape ()
    stays an operand in its own format, with its bounds (``proofs/multiply.md``)."""
    at = [i for i, x in enumerate(xs) if not isinstance(x, _SCALARS)]
    if not at:
        raise TypeError(f"{op} needs at least one array operand")
    arrays, fmt, out = promoted(op, [xs[i] for i in at], dtype, rounding, overflow)
    arrays = dict(zip(at, arrays))
    shape = broadcast_shapes(*(X.array.shape for X in arrays.values()))
    found = []
    for i, x in enumerate(xs):
        X = arrays.get(i)
        if X is None or (X.array.dtype != fmt and not X.uncertain):
            X = scalar_operand([exact_scalar(x if X is None else X.array)], fmt.kind, (), fmt)
        found.append(broadcast_to(X, shape))
    return found, fmt, out, shape


@ieee
def multiply(x, y, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
             overflow: str | None = None, workers: int | None = None) -> Result:
    """Elementwise product with broadcasting (NumPy rules).

    Each output element is the exact product rounded once to ``dtype`` (default: the
    operands' promoted format: the one that holds both, else the narrowest standard format
    that does, as PyTorch promotes); ``dtype`` must be of the inputs' kind; convert across
    kinds with ``asarray(x, d)`` first. As in PyTorch, an operand of shape () takes part
    in the promotion only when both are of shape (). A scalar operand (int, float,
    complex, a ``Fraction`` with a power-of-two denominator, a ``FloatValue`` or
    ``ComplexValue``, or an array of shape () of any format beside one with dimensions)
    is taken by its value and does not change the format. ``tolerance()`` budgets the
    product's rounding in the compute format and the final rounding; a complex product's
    parts are two-term sums, budgeted as in ``matmul``. An implementation that rounds a
    scalar operand to a format first is ``multiply_held_scalar``. Integer results wrap on
    overflow unless ``overflow`` says otherwise. A complex product with a non-finite part
    follows IEEE rules in each part, as PyTorch's does: ``(inf + 0j) * (1 + 0j)`` is
    ``inf + nanj``. An exact zero takes the sign IEEE arithmetic gives the product
    (``-0.0 * 1`` is -0), as in PyTorch. ``x`` and ``y`` may also be earlier results, as
    ``Result`` describes.
    """
    x, y = _sources(x, y)

    def backward(grad, **options):
        return _multiply_backward(x, y, grad, **options)

    return _multiply_result(_inputs("multiply", x, y, dtype=dtype, rounding=rounding, overflow=overflow), rounding, overflow, workers, backward)


def _multiply_result(inputs: tuple, rounding: Rounding, overflow, workers, backward=None, reference: tuple | None = None) -> Result:
    """multiply's result for ``_inputs``' operands; ``reference`` as ``product`` takes it."""
    (X, Y), fmt, out, shape = inputs
    size = X.array.size
    inner = product("multiply", X.take(range(size), shape + (1, 1)), Y.take(range(size), shape + (1, 1)), out, rounding, overflow, fmt,
                    name="output", few_terms=True, elementwise=True, workers=workers, shape=shape, reference=reference)
    return reshaped(inner, shape, backward)


@ieee
def add(x, y, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
        workers: int | None = None) -> Result:
    """Elementwise ``x + alpha * y`` with broadcasting (NumPy rules).

    Each output element is the exact value rounded once to ``dtype`` (default: the operands' promoted format, with
    operands of shape () as in ``multiply``); ``dtype`` must be of the inputs' kind; convert across kinds with
    ``asarray(x, d)`` first. ``alpha`` is taken exactly as given, like a scalar operand. ``tolerance()`` budgets
    ``alpha * y`` rounded in the compute format, the sum rounded in it, and the final rounding; with ``alpha`` 1
    or -1 only the sum is rounded, its terms taken exactly, a scalar operand the compute format does not hold
    included (a compute format narrower than the array operands' rounds every term first). An implementation that stores ``alpha * y`` in a narrower format first
    is the chain ``add(x, multiply(y, alpha, dtype=...))``; ``any_of`` covers both when either may run. An
    implementation that rounds a scalar operand and ``alpha`` to a format first is ``add_held_scalar``.
    Integer results wrap on overflow unless ``overflow`` says otherwise. An exact zero takes the sign IEEE
    arithmetic gives ``x + alpha * y``, as in PyTorch: ``-0.0 + -0.0`` is -0, and a cancellation is +0 (-0 when
    rounding toward -inf). In a complex sum, a zero part of ``alpha`` (the imaginary part of 1, say) adds no term,
    by the rule below, where PyTorch multiplies ``alpha * y`` out: ``(-0+1j) + (-0-1j)`` has the real part -0,
    PyTorch's +0. A complex non-finite part reaches only the parts it is added to, as in PyTorch: ``(inf + 1j) +
    (1 + 1j)`` is ``inf + 2j``. A complex ``alpha`` follows the same rule, which is the mathematical answer:
    ``add(1+1j, inf+1j, alpha=1j)`` is ``infj``, where PyTorch, multiplying ``alpha * y`` in complex arithmetic,
    gives ``nan+infj``. ``x`` and ``y`` may also be earlier results.
    """
    x, y = _sources(x, y)

    def backward(grad, **options):
        return _add_backward(x, y, grad, alpha=alpha, **options)

    inputs = _inputs("add", x, y, dtype=dtype, rounding=rounding, overflow=overflow)
    return _add_result((x, y), inputs, exact_scalar(alpha), rounding, overflow, workers, backward)


def _add_result(sources: tuple, inputs: tuple, a, rounding: Rounding, overflow, workers, backward=None, reference: tuple | None = None) -> Result:
    """add's result for ``_inputs``' operands of ``sources`` (``_sources``) and the exact ``alpha`` ``a``;
    ``reference`` as ``product`` takes it."""
    (X, Y), fmt, out, shape = inputs
    coefficients = [1 if fmt.kind == "int" else FloatValue("finite", False, 1, 0), a]
    terms = [X, Y]
    # the terms are read as one array: a scalar in a format the array's does not hold widens
    # both to a format holding the two (exactly); an integer scalar that no standard format
    # holds with the array (a negative one with uint64) joins its coefficient instead
    dtypes = [X.array.dtype, Y.array.dtype]
    if fmt.kind == "int" and holding_format(dtypes) is None:
        k = 0 if not holds(fmt, dtypes[0]) else 1
        source = sources[k]
        coefficients[k] *= exact_scalar(source if isinstance(source, _SCALARS) else source.array)
        terms[k] = broadcast_to(scalar_operand([1], "int", (), fmt), shape)
    else:
        common = promoted_format(dtypes)
        terms = [T.converted(common) for T in terms]
    inner = product("add", stacked(terms, shape + (1, 2)), scalar_operand(coefficients, fmt.kind, (2, 1), fmt), out, rounding, overflow, fmt,
                    name="output", unrounded_terms=fmt if is_unit(a) else None, coefficients=True, few_terms=True, elementwise=True, workers=workers, shape=shape,
                    reference=reference)
    return reshaped(inner, shape, backward)


@ieee
def _multiply_backward(x, y, grad, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                       workers: int | None = None) -> Result:
    """Gradients of ``multiply(x, y)`` for the output gradient ``grad``.

    ``grad_x = grad * conj(y)`` and ``grad_y = grad * conj(x)``, each summed over the
    copies that broadcasting made of its operand, in that operand's shape and dtype. Each
    is one product (the copies are folded into its sum), so every element is one exact
    sum rounded once. A scalar operand gets no gradient. Where an operand is broadcast,
    PyTorch rounds each product to the dtype before summing the copies, an extra rounding
    the default tolerance does not budget. An element with one copy, of an operand with
    the output's number of dimensions, is a single product, so an exact zero takes that
    product's IEEE sign, as in PyTorch. Any other element is a sum, as in PyTorch, which
    also sums the dimensions broadcasting adds in front: its exact zero is +0 (-0 when
    rounding toward -inf). ``grad`` must have the output's shape.
    """
    (X, Y), fmt, _, shape = _inputs("multiply", x, y, dtype=dtype, rounding=rounding, overflow=overflow)
    G = grad_operand("multiply", grad, shape, fmt)
    base = promoted_format([G.array.dtype, fmt])
    gradients = {}
    for name, original, other in (("grad_x", x, Y), ("grad_y", y, X)):
        if isinstance(original, _SCALARS):
            continue
        target = operand(original).array

        def column(gather, width, other=other, target=target):
            return other.take(gather, target.shape + (width, 1), conjugate=True)

        gradients[name] = gradient("multiply.backward", G, column, target, shape, base, dtype, rounding, overflow, workers)
    return combined("multiply.backward", gradients)


@ieee
def _add_backward(x, y, grad, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                  workers: int | None = None) -> Result:
    """Gradients of ``add(x, y, alpha=alpha)`` for the output gradient ``grad``.

    ``grad_x = grad`` and ``grad_y = conj(alpha) * grad``, each summed over the copies that
    broadcasting made of its operand, in that operand's shape and dtype; each is one
    product, so every element is one exact sum rounded once. An element with one copy, of
    an operand with the output's number of dimensions, is a single product, so an exact zero
    takes that product's IEEE sign. Any other element is a sum, as in PyTorch, which also
    sums the dimensions broadcasting adds in front: its exact zero is +0 (-0 when rounding
    toward -inf). A zero part of a complex ``conj(alpha)`` adds no term, as in ``add``,
    where PyTorch multiplies ``grad`` by ``conj(alpha)`` in complex arithmetic unless
    ``alpha`` is 1: with ``alpha=2``, a gradient ``1-0j`` gives ``grad_y`` ``2-0j``, PyTorch's
    ``2+0j``. A scalar operand gets no gradient. ``grad`` must have the output's shape.
    """
    (X, Y), fmt, _, shape = _inputs("add", x, y, dtype=dtype, rounding=rounding, overflow=overflow)
    G = grad_operand("add", grad, shape, fmt)
    a = exact_scalar(alpha)
    gradients = {}
    for name, original, c in (("grad_x", x, FloatValue("finite", False, 1, 0)), ("grad_y", y, conjugate_scalar(a))):
        if isinstance(original, _SCALARS):
            continue

        def column(gather, width, c=c):
            return repeated_operand(c, G.array.dtype.kind, width, G.array.dtype)

        gradients[name] = gradient("add.backward", G, column, operand(original).array, shape, G.array.dtype, dtype, rounding, overflow, workers,
                                    exact_terms=is_unit(c), coefficients=True)
    return combined("add.backward", gradients)


multiply.backward = _multiply_backward
add.backward = _add_backward


class _HeldScalar(Algorithm):
    """An implementation of an elementwise operation that first rounds each scalar operand (and ``add``'s
    ``alpha``) to nearest even to a format, ``held``, or to each of a sequence of formats in turn
    (``MultiplyHeldScalar``, ``AddHeldScalar``)."""

    _what: str  # the values it rounds, for the description

    def __init__(self, held: Format | str | Sequence[Format | str]) -> None:
        sequence = isinstance(held, (list, tuple))
        steps = [self._step(h) for h in (held if sequence else [held])]
        if not steps:
            raise ValueError(f"{self.name} holds scalars in at least one format, got an empty sequence")
        self.held = tuple(steps) if sequence else steps[0]
        self._steps = steps
        targets = ["the operands' format" if h == "operand" else "the compute format" if h == "compute" else h.name for h in steps]
        self.description = f"{self._what} rounded to nearest even to {', then to '.join(targets)}, then {self.op}"

    def _step(self, held):
        """One format ``held`` names: "operand", "compute" or a floating-point or complex format."""
        if isinstance(held, str) and held in ("operand", "compute"):
            return held
        found = get_format(held)
        if found.kind not in ("float", "complex"):
            raise TypeError(f"{self.name} holds scalars in a floating-point or complex format, got {found.name}")
        return found

    def _format(self, step, fmt: Format, compute) -> Format:
        """The format ``step`` holds a scalar in, of the inputs' kind, for array operands of the promoted format ``fmt``:
        a complex one for complex inputs, whose parts are held in its component."""
        found = fmt if step == "operand" else compute_format(compute, fmt) if step == "compute" else step
        if fmt.kind == "float":
            if found.kind != "float":
                raise TypeError(f"{self.name} holds a real scalar in a floating-point format, got {found.name}")
            return found
        if found.kind == "complex":
            return found
        return next((f for f in FORMATS.values() if f.kind == "complex" and f.component == found), None) or ComplexFormat(f"complex_{found.name}", found)

    def _held(self, x, y, alpha, dtype, rounding: Rounding, overflow, compute, workers: int | None = None) -> Result:
        """The operation's result on these inputs (``alpha`` None for ``multiply``): its ``rounded`` and ``exact``,
        with the bounds and the normal answer of its model at the held scalars, the bounds measured from that
        ``rounded`` (``proofs/multiply.md``, Held scalars). A scalar operand is one the operation takes by its value:
        a scalar, or an operand of shape () beside one with dimensions."""
        x, y = _sources(x, y)
        inputs = _inputs(self.op, x, y, dtype=dtype, rounding=rounding, overflow=overflow)
        fmt = inputs[1]
        if fmt.kind not in ("float", "complex"):
            raise TypeError(f"{self.name} supports floating-point and complex formats, not {fmt.name}")
        held = [self._format(step, fmt, compute) for step in self._steps]
        dims = any(not isinstance(s, _SCALARS) and s.array.ndim for s in (x, y))
        scalars = [isinstance(s, _SCALARS) or (dims and not s.array.ndim) for s in (x, y)]
        if any(scalar and not isinstance(s, _SCALARS) and s.uncertain for s, scalar in zip((x, y), scalars)):
            chain = "s"
            for f in held:
                chain = f"multiply({chain}, 1, dtype={f.name})"
            raise ValueError(f"{self.name} rounds a scalar operand to {', then '.join(f.name for f in held)}, but an earlier result of shape () may "
                             f"be stored anywhere its bounds allow; hold it with the chain {chain} and pass that to {self.op}")
        a = None if alpha is None else exact_scalar(alpha)
        given = self._result((x, y), inputs, a, rounding, overflow, workers)
        # each scalar's held value, once, by the C codec, in place of the scalar's copies; the per-element work is the
        # product's kernels'
        parts, (operands, _, _, shape) = [_part_format(f) for f in held], inputs
        sources, found = [], []
        for s, scalar, X in zip((x, y), scalars, operands):
            if scalar:
                s = _held_value(exact_scalar(s if isinstance(s, _SCALARS) else s.array), parts)
                X = Operand(repeated_operand(s, fmt.kind, X.array.size, fmt).array.reshape(shape))
            sources.append(s)
            found.append(X)
        a = None if a is None else _held_value(a, parts)
        return self._result(tuple(sources), (found,) + inputs[1:], a, rounding, overflow, workers, reference=(given.rounded, given.exact))

    def _result(self, sources: tuple, inputs: tuple, a, rounding: Rounding, overflow, workers, reference: tuple | None = None) -> Result:
        raise NotImplementedError


def _check(standard: str, ulps) -> None:
    """A query's ``standard`` and ``ulps``, checked before any work."""
    check_standard(standard)
    check_ulps(ulps)


def _held_value(v, parts: list[FloatFormat]):
    """The exact scalar ``v`` (``exact_scalar``'s) rounded to nearest even to each format of ``parts`` in turn, each part
    of a complex value; a value past a format's range gives an infinity (NaN in a format without one), as a cast
    does."""

    def held(p):
        for part in parts:
            p = part.decode(part.encode(p, overflow="ieee" if part.has_infinity else "nan"))
        return p

    return ComplexValue(held(v.real), held(v.imag)) if isinstance(v, ComplexValue) else held(v)


class MultiplyHeldScalar(_HeldScalar):
    """``multiply`` with each scalar operand first rounded to nearest even to ``held`` (``proofs/multiply.md``, Held
    scalars). A scalar operand is one ``multiply`` takes by its value: a Python scalar, ``FloatValue`` or
    ``ComplexValue``, or an array of shape () beside an operand with dimensions; a complex one is rounded part by part,
    and a value past ``held``'s range gives an infinity (NaN in a format without one), as a cast does. Then
    ``multiply``'s model runs on the held value. ``held`` is ``"operand"`` (the array operands' promoted format),
    ``"compute"`` (the compute format, the queries' ``compute``) or a format: for complex inputs a complex format, or
    the format of the parts. It may also be a sequence (list or tuple) of these, the scalar then rounded to each in
    turn, as an implementation that converts it through another format first does: ``("float32", "operand")``. The
    bounds are measured from ``multiply``'s reference, with the scalar as given, so the tolerance holds the scalar's
    rounding; ``normal()`` is the normal answer at the held value. Real and complex floating-point inputs; an earlier
    result of shape () as the scalar raises ValueError (the chain ``multiply(s, 1, dtype=...)`` holds it, and
    ``multiply`` takes that). No backward and no ``run()``.

    PyTorch's CPU ``mul`` rounds a scalar operand first, in a format that depends on the scalar's position:

    - float16 and bfloat16 with the scalar as mul's second argument (``x * s``, ``s * x`` with a Python scalar ``s``,
      ``torch.mul(x, s)``): ``"compute"`` (float32, which it computes in).
    - float16 and bfloat16 with the scalar first (``torch.mul(s, x)``, ``s0d * x`` with a tensor ``s0d`` of shape ()):
      the dtype, ``"operand"`` for float16 and ``("float32", "operand")`` for bfloat16, which PyTorch converts a
      float64 value to through float32.
    - float32 and complex64 in either position: ``"operand"``.
    - complex32 in either position: ``("complex64", "operand")``, as PyTorch converts a complex128 or float64 value to
      complex32 through complex64.

    ``"operand"`` for float16 holds where PyTorch converts a float64 value to float16 in one rounding, as on aarch64
    (Linux and macOS), where c10::Half converts from the platform's own half type; elsewhere, x86_64 included, c10::Half
    converts from float32, so the float64 value reaches float16 through float32: ``("float32", "operand")``."""

    op = "multiply"
    name = "multiply_held_scalar"
    _what = "each scalar operand"

    def _result(self, sources: tuple, inputs: tuple, a, rounding: Rounding, overflow, workers, reference: tuple | None = None) -> Result:
        return _multiply_result(inputs, rounding, overflow, workers, reference=reference)

    @ieee
    def tolerance(self, x, y, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                  compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False, standard: str = "normal",
                  workers: int | None = None) -> Array:
        """Worst-case ``|output - multiply(x, y).rounded|`` per element over this algorithm's finite outputs, as
        ``Result.tolerance`` describes; the arguments are ``multiply``'s (``overflow`` the implementation's overflow
        mode) and ``Result.tolerance``'s."""
        _check(standard, ulps)
        return self._held(x, y, None, dtype, rounding, overflow, compute, workers).tolerance(compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                            standard=standard, workers=workers)

    @ieee
    def finite_range(self, x, y, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                     compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False, standard: str = "normal",
                     workers: int | None = None) -> tuple[Array, Array]:
        """The lowest and highest finite output per element, as ``Result.finite_range`` describes; the arguments are
        ``tolerance``'s."""
        _check(standard, ulps)
        return self._held(x, y, None, dtype, rounding, overflow, compute, workers).finite_range(compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                               standard=standard, workers=workers)

    @ieee
    def may_be(self, kind: str, x, y, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
               compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False, standard: str = "normal",
               workers: int | None = None) -> Array:
        """Where this algorithm may output ``kind`` (``"+inf"``, ``"-inf"`` or ``"nan"``), as ``Result.may_be``
        describes; the other arguments are ``tolerance``'s."""
        check_kind(kind)
        _check(standard, ulps)
        return self._held(x, y, None, dtype, rounding, overflow, compute, workers).may_be(kind, compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                         standard=standard, workers=workers)

    @ieee
    def covers(self, output, x, y, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
               compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False, standard: str = "normal",
               workers: int | None = None) -> Array:
        """Where this algorithm can produce ``output``, as ``Result.covers`` describes; the other arguments are
        ``tolerance``'s."""
        _check(standard, ulps)
        return self._held(x, y, None, dtype, rounding, overflow, compute, workers).covers(output, compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                         standard=standard, workers=workers)

    @ieee
    def normal(self, x, y, *, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
               compute: Format | str | None = None, flush_to_zero: bool = False, workers: int | None = None) -> Array:
        """``multiply``'s normal answer (``Result.normal``) with each scalar operand held."""
        return self._held(x, y, None, dtype, rounding, overflow, compute, workers).normal(compute, flush_to_zero=flush_to_zero, workers=workers)


class AddHeldScalar(_HeldScalar):
    """``add`` with each scalar operand and ``alpha`` first rounded to nearest even to ``held`` (``proofs/add.md``,
    Held scalars), as ``MultiplyHeldScalar`` rounds a scalar operand; then ``add``'s model runs on the held values.
    ``held`` is ``"operand"`` (the array operands' promoted format), ``"compute"`` (the compute format, the queries'
    ``compute``), a format, or a sequence of these rounded to in turn, as in ``MultiplyHeldScalar``. The bounds are
    measured from ``add``'s reference, with the scalars as given; ``normal()`` is the normal answer at the held values.
    Real and complex floating-point inputs; an earlier result of shape () as the scalar raises ValueError. No backward
    and no ``run()``.

    PyTorch's CPU ``add`` rounds a scalar operand and ``alpha`` to the dtype first: ``"operand"`` for float32,
    complex64 and float16 (where PyTorch converts a float64 value to float16 in one rounding, as ``MultiplyHeldScalar``
    says), ``("float32", "operand")`` for bfloat16 and ``("complex64", "operand")`` for complex32, which PyTorch
    converts a float64 or complex128 value to through float32 or complex64. With ``alpha`` other than 1, its bfloat16
    and complex32 kernels, and its float16 kernel for some elements (of a short broadcast row), also store ``alpha * y``
    in the dtype before the addition, a rounding this model does not have: ``add``'s chain ``add(x, multiply(y, alpha,
    dtype=...))`` with ``alpha`` so rounded, which ``any_of`` combines with ``add``."""

    op = "add"
    name = "add_held_scalar"
    _what = "each scalar operand and alpha"

    def _result(self, sources: tuple, inputs: tuple, a, rounding: Rounding, overflow, workers, reference: tuple | None = None) -> Result:
        return _add_result(sources, inputs, a, rounding, overflow, workers, reference=reference)

    @ieee
    def tolerance(self, x, y, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
                  compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False, standard: str = "normal",
                  workers: int | None = None) -> Array:
        """Worst-case ``|output - add(x, y, alpha=alpha).rounded|`` per element over this algorithm's finite outputs, as
        ``Result.tolerance`` describes; the arguments are ``add``'s (``overflow`` the implementation's overflow mode)
        and ``Result.tolerance``'s."""
        _check(standard, ulps)
        return self._held(x, y, alpha, dtype, rounding, overflow, compute, workers).tolerance(compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                             standard=standard, workers=workers)

    @ieee
    def finite_range(self, x, y, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
                     overflow: str | None = None, compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False,
                     standard: str = "normal", workers: int | None = None) -> tuple[Array, Array]:
        """The lowest and highest finite output per element, as ``Result.finite_range`` describes; the arguments are
        ``tolerance``'s."""
        _check(standard, ulps)
        return self._held(x, y, alpha, dtype, rounding, overflow, compute, workers).finite_range(compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                                standard=standard, workers=workers)

    @ieee
    def may_be(self, kind: str, x, y, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
               overflow: str | None = None, compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False,
               standard: str = "normal", workers: int | None = None) -> Array:
        """Where this algorithm may output ``kind`` (``"+inf"``, ``"-inf"`` or ``"nan"``), as ``Result.may_be``
        describes; the other arguments are ``tolerance``'s."""
        check_kind(kind)
        _check(standard, ulps)
        return self._held(x, y, alpha, dtype, rounding, overflow, compute, workers).may_be(kind, compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                          standard=standard, workers=workers)

    @ieee
    def covers(self, output, x, y, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN,
               overflow: str | None = None, compute: Format | str | None = None, ulps=None, flush_to_zero: bool = False,
               standard: str = "normal", workers: int | None = None) -> Array:
        """Where this algorithm can produce ``output``, as ``Result.covers`` describes; the other arguments are
        ``tolerance``'s."""
        _check(standard, ulps)
        return self._held(x, y, alpha, dtype, rounding, overflow, compute, workers).covers(output, compute, ulps=ulps, flush_to_zero=flush_to_zero,
                                                                                          standard=standard, workers=workers)

    @ieee
    def normal(self, x, y, *, alpha=1, dtype: Format | str | None = None, rounding: Rounding = Rounding.NEAREST_EVEN, overflow: str | None = None,
               compute: Format | str | None = None, flush_to_zero: bool = False, workers: int | None = None) -> Array:
        """``add``'s normal answer (``Result.normal``) with each scalar operand and ``alpha`` held."""
        return self._held(x, y, alpha, dtype, rounding, overflow, compute, workers).normal(compute, flush_to_zero=flush_to_zero, workers=workers)


def multiply_held_scalar(held: Format | str | Sequence[Format | str]) -> MultiplyHeldScalar:
    """``multiply`` with each scalar operand first rounded to ``held``, or to each format of a sequence in turn
    (``MultiplyHeldScalar``)."""
    return MultiplyHeldScalar(held)


def add_held_scalar(held: Format | str | Sequence[Format | str]) -> AddHeldScalar:
    """``add`` with each scalar operand and ``alpha`` first rounded to ``held``, or to each format of a sequence in turn
    (``AddHeldScalar``)."""
    return AddHeldScalar(held)
