"""Exceptions raised by reference operations."""


class ReferenceUnavailableError(ArithmeticError):
    """A result could not be established with the promised guarantee.

    This is raised instead of returning an unverified answer: for example when
    an exact value may lie exactly where the rounding to the output format
    changes (halfway between two of its values, for rounding to nearest) and
    the refinement's precision cap is reached before that is decided.
    """
