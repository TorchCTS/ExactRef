"""Shapes and indices shared by operation implementations (internal)."""

from __future__ import annotations

from collections.abc import Iterator

from ..array import row_major_strides


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    ndim = max((len(s) for s in shapes), default=0)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    out = []
    for axis in range(ndim):
        sizes = {s[axis] for s in padded} - {1}
        if len(sizes) > 1:
            raise ValueError(f"shapes {', '.join(map(str, shapes))} cannot be broadcast together")
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


def broadcast_index(index: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """Map an index of a broadcast shape back to an operand of ``shape``."""
    trimmed = index[len(index) - len(shape) :] if shape else ()
    return tuple(0 if n == 1 else i for i, n in zip(trimmed, shape))


def unravel(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major index of position ``flat`` in ``shape``."""
    index = []
    for n in reversed(shape):
        flat, i = divmod(flat, n)
        index.append(i)
    return tuple(reversed(index))


def iter_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if not shape:
        yield ()
        return
    if 0 in shape:
        return
    index = [0] * len(shape)
    while True:
        yield tuple(index)
        for axis in range(len(shape) - 1, -1, -1):
            index[axis] += 1
            if index[axis] < shape[axis]:
                break
            index[axis] = 0
        else:
            return


class Grid:
    """Row-major offsets of an array's elements."""

    __slots__ = ("shape", "strides")

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.strides = row_major_strides(shape)

    def offset(self, index: tuple[int, ...]) -> int:
        return sum(i * s for i, s in zip(index, self.strides))
