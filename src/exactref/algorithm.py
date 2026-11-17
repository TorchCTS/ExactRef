"""Named algorithms: concrete implementations of an operation.

An operation's definition gives the exact answer and the tolerance of every
correct implementation. An algorithm describes one implementation:

- metadata: ``op`` (the key of the operation it implements), ``name``, ``description``;
- ``error_model(...)``: its error formula, independent of the input values,
  with a coefficient that depends only on shapes and formats; metadata for
  comparing algorithms, not a pass/fail tolerance. Only matmul's
  ``MatmulAlgorithm`` and ``MatmulSlots16`` have one, which raises ValueError
  where the sum's depth times the unit roundoff reaches 1 (no finite bound);
  attention's error grows with its logits, so its algorithms have none;
- ``tolerance(...)``: that bound evaluated on actual inputs, measured from the
  definition's rounded output;
- ``finite_range(...)``, ``may_be(...)`` and ``covers(...)``: the other views of
  the same bounds, as ``Result``'s; attention's algorithms also have
  ``backward_tolerance``, ``backward_finite_range``, ``backward_may_be`` and
  ``backward_covers``, for the backward, and the held-scalar algorithms
  (``multiply_held_scalar``, ``add_held_scalar``) ``normal(...)``, the normal
  answer at the held values;
- ``run(...)``: its emulated output, only when its rounding sequence is fully
  specified.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._fenv import ieee


@dataclass(frozen=True)
class ErrorModel:
    """``|error| <= relative * scale + absolute``, with ``scale`` naming the input quantity."""

    relative: Fraction
    absolute: Fraction
    scale: str

    @ieee
    def __str__(self) -> str:
        return f"{float(self.relative):.3g} * {self.scale} + {float(self.absolute):.3g}"


class Algorithm:
    """Base for named algorithms: one implementation of an operation, with its ``op``,
    ``name`` and ``description``, the bounds of its outputs on actual inputs
    (``tolerance``, ``finite_range``, ``may_be`` and ``covers``), and, where its rounding
    sequence is fully specified, ``run``."""

    op: str
    name: str
    description: str

    def __repr__(self) -> str:
        return f"<{self.name}: {self.description}>"
