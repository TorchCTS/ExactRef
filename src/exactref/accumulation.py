"""Accumulation methods: how an implementation may add up a list of terms.

Each method in ``METHODS`` is a fully specified summation order, which the matmul
algorithms' ``run()`` emulates exactly; ``ANY_ORDER`` and ``MMA_METHOD`` describe
families of orders, for tolerances only. For tolerances,
a method of ``METHODS`` reports ``depth`` (the most roundings any one term passes through) and
``roundings`` (the total), which give the standard worst-case bound

    ``|computed - exact| <= gamma(depth) * sum|t_i| + roundings * eta * (1 + gamma(depth))``

with ``gamma(n) = n*u / (1 - n*u)`` (Higham, Accuracy and Stability of
Numerical Algorithms, 2nd ed., chapters 3-4). The ``exact`` method sums
exactly and rounds once, which no implementation can beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from . import _native
from .formats import FloatFormat


@dataclass(frozen=True)
class AccumulationMethod:
    """An accumulation method: its ``key``, a short ``label`` and a ``description``, whether
    its result depends on the order of the terms, the ``parameters`` it takes (such as
    ``block_size``) and its ``references``. For a method of ``METHODS``, ``depth(n)`` and
    ``roundings(n)`` give the worst-case bound's counts for ``n`` terms (the module
    docstring); ``ANY_ORDER`` and ``MMA_METHOD`` have none."""

    key: str
    label: str
    description: str
    order_dependent: bool
    parameters: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    code: int = 0  # the product kernel's order code (csrc/product.c), 0 for a method it does not order

    def depth(self, n: int, *, block_size: int | None = None) -> int:
        raise NotImplementedError

    def roundings(self, n: int, *, block_size: int | None = None, flush: bool = False) -> int:
        raise NotImplementedError

    def describe(self) -> dict[str, str | bool | list[str]]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "order_dependent": self.order_dependent,
            "parameters": list(self.parameters),
            "references": list(self.references),
        }


class _Exact(AccumulationMethod):
    def depth(self, n, *, block_size=None):
        return 1 if n else 0

    def roundings(self, n, *, block_size=None, flush=False):
        return 1 if n else 0


class _Sequential(AccumulationMethod):
    def depth(self, n, *, block_size=None):
        return _native.order_depth(self.code, n, 0) + 1 if n else 0  # the term's own rounding, then its additions (csrc/product.c)

    def roundings(self, n, *, block_size=None, flush=False):
        return n


class _Pairwise(AccumulationMethod):
    def depth(self, n, *, block_size=None):
        return _native.order_depth(self.code, n, 0) + 1 if n else 0

    def roundings(self, n, *, block_size=None, flush=False):
        return 2 * n - 1 if n else 0


class _Blocked(AccumulationMethod):
    def depth(self, n, *, block_size: int):
        return _native.order_depth(self.code, n, block_size) + 1 if n else 0

    def roundings(self, n, *, block_size: int, flush=False):
        return n + (n + block_size - 1) // block_size if n else 0


_HIGHAM = "Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed. (SIAM 2002), ch. 4"

_EXACT = _Exact(
    "exact",
    "Exact accumulation",
    "Sum the terms exactly (as a long fixed-point or Kulisch accumulator would), then round once to the "
    "accumulator format. Order-independent and the most accurate method possible.",
    order_dependent=False,
    references=("Kulisch, Computer Arithmetic and Validity (de Gruyter 2008)", "Neal, arXiv:1505.05571"),
    code=4,
)
_SEQUENTIAL = _Sequential(
    "sequential",
    "Sequential (recursive) summation",
    "Start from an exact zero and add the terms one at a time in index order, rounding after every "
    "addition. Its worst-case bound also covers every other order of pairwise additions.",
    order_dependent=True,
    references=(_HIGHAM,),
    code=1,
)
_PAIRWISE = _Pairwise(
    "pairwise",
    "Pairwise (tree) summation",
    "Round each term, then recursively add the sum of the first ceil(n/2) terms to the sum of the rest, "
    "rounding after every addition.",
    order_dependent=True,
    references=(_HIGHAM,),
    code=2,
)
_BLOCKED = _Blocked(
    "blocked",
    "Blocked summation",
    "Sum consecutive blocks of block_size terms sequentially, then sum the block results sequentially.",
    order_dependent=True,
    parameters=("block_size",),
    references=(_HIGHAM,),
    code=3,
)

METHODS: dict[str, AccumulationMethod] = {m.key: m for m in (_EXACT, _SEQUENTIAL, _PAIRWISE, _BLOCKED)}


# Every way to sum the terms: any tree of additions or fused multiply-adds, a term perhaps rounded on its own first. For
# required-boundary tolerances only: not one order, so it cannot be emulated and no algorithm can use it; its depth and
# rounding counts are csrc/errors.c's any_order_counts (proofs/foundations.md, Lemma 2).
ANY_ORDER = AccumulationMethod("any_order", "Any order", "Any summation order, with or without fused multiply-add.", order_dependent=True)


@dataclass(frozen=True)
class MMAStep:
    """One tensor-core step (``proofs/foundations.md``, Lemma T): up to ``products`` exact
    products and the float32 accumulator, aligned to the largest exponent (not below
    ``exponent_floor``), each truncated to ``fraction_bits`` bits below it, added exactly, and
    the sum truncated to float32. The values are measured per GPU and input format (Khattak and
    Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004).

    ``split`` states how the kernels being checked split a sum into pieces: every piece but one
    holds a multiple of ``split`` products (1: any split). CUTLASS splits K on multiples of
    128 / element bits (8 for 16-bit inputs); a multiple of ``products`` costs no extra steps.
    Each piece is summed in full steps: every step but the piece's last takes ``products``
    products, as a tensor-core instruction takes a fixed number (zero padding counts among them).
    A kernel whose steps take fewer is outside this model."""

    products: int
    fraction_bits: int
    exponent_floor: int
    split: int

    def __post_init__(self) -> None:
        for name in ("products", "fraction_bits", "exponent_floor", "split"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"MMAStep's {name} must be an int")
        if self.products < 1 or self.split < 1:
            raise ValueError("MMAStep's products and split must be positive")
        # So a step that adds no product leaves the accumulator unchanged (Lemma T).
        if self.fraction_bits < 23 or self.exponent_floor - self.fraction_bits > -149:
            raise ValueError("MMAStep needs fraction_bits >= 23 and exponent_floor - fraction_bits <= -149")


@dataclass(frozen=True)
class MMA:
    """An accumulation by tensor-core steps (``proofs/foundations.md``, Lemma T): the step, the
    optional reduction format that split-K partial sums are rounded to, ``slack`` (the model's
    allowance sigma for subnormal factors, a ``Fraction``), and
    ``count``, the products each sum's kernel processes (zeros included) where a caller's term
    count leaves zeros out (None: the caller's count)."""

    step: MMAStep
    reduction: FloatFormat | None = None
    slack: Fraction = Fraction(0)
    count: int | None = None


# Tensor-core kernels (Lemma T): chains of steps and float32 additions in any order. Not one order, so it cannot be
# emulated, and no depth or rounding count describes it; its bound is ops._mma.mma_terms.
MMA_METHOD = AccumulationMethod("mma", "Tensor-core steps", "Chains of tensor-core steps added in any order.", order_dependent=True)


def get_method(key: str) -> AccumulationMethod:
    """The accumulation method named ``key``: one of ``METHODS``, ``"any_order"`` or ``"mma"``
    (ValueError for any other)."""
    if key == ANY_ORDER.key:
        return ANY_ORDER
    if key == MMA_METHOD.key:
        return MMA_METHOD
    try:
        return METHODS[key]
    except KeyError:
        raise ValueError(f"unknown accumulation method {key!r}; known: {', '.join((*METHODS, ANY_ORDER.key, MMA_METHOD.key))}") from None
