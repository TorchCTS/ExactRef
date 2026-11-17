"""ExactRef: exact reference answers, with proven error bounds, for testing numerical code.

Data lives in ``Array`` (immutable, for reference inputs) and
``MutableArray`` (its elements can be set): one flat byte buffer plus the
dtype, shape and byte order describing it. ``asarray`` imports data from
Python values, the buffer protocol (NumPy, ``array``) and DLPack (PyTorch,
JAX), without needing NumPy.

Operations live under their category, such as
``exactref.linalg.products.matmul``, with top-level shortcuts such as
``exactref.matmul``. Each returns a ``Result``: ``rounded`` (the exact answer
rounded once to the output format), ``exact`` (certified enclosures of the
exact answer) and ``tolerance()`` (a proven bound on how far a correct
implementation's output can be from ``rounded`` on those inputs). Named
algorithms, such as ``linalg.products.matmul_pairwise``, describe concrete
implementations. A result can be passed to a later operation in place of an
array, and ``any_of`` covers an implementation that computes one of several
expressions.
"""

from . import arithmetic, linalg, nn, reduction
from .accumulation import MMAStep
from .algorithm import Algorithm, ErrorModel
from .array import Array, MutableArray
from .boundaries import Boundary
from .convert import asarray, round_to_tf32
from .errors import ReferenceUnavailableError
from .exact import Enclosure, ExactArray
from .formats import (
    FORMATS,
    BoolFormat,
    ComplexFormat,
    FloatFormat,
    Format,
    IntFormat,
    Rounding,
    UnsupportedFormatError,
    bfloat16,
    bool_,
    complex32,
    complex64,
    complex128,
    float8_e3m4,
    float8_e4m3,
    float8_e4m3b11fnuz,
    float8_e4m3fn,
    float8_e4m3fnuz,
    float8_e5m2,
    float8_e5m2fnuz,
    float16,
    float32,
    float64,
    get_format,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from .arithmetic.basic import add, multiply
from .linalg.matrix_function import matrix_exp, matrix_exp_frechet
from .linalg.products import gemm, matmul
from .nn.attention import scaled_dot_product_attention
from .ops.any_of import any_of
from .reduction.arithmetic import sum
from .result import Outputs, Result
from .values import ComplexValue, FloatValue

__all__ = [
    "Array",
    "MutableArray",
    "asarray",
    "round_to_tf32",
    "FloatValue",
    "ComplexValue",
    "Format",
    "FloatFormat",
    "IntFormat",
    "BoolFormat",
    "ComplexFormat",
    "Rounding",
    "UnsupportedFormatError",
    "ReferenceUnavailableError",
    "FORMATS",
    "get_format",
    "Result",
    "Outputs",
    "ExactArray",
    "Enclosure",
    "Boundary",
    "Algorithm",
    "ErrorModel",
    "MMAStep",
    "arithmetic",
    "linalg",
    "nn",
    "reduction",
    "add",
    "multiply",
    "sum",
    "matmul",
    "gemm",
    "scaled_dot_product_attention",
    "matrix_exp",
    "matrix_exp_frechet",
    "any_of",
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "float8_e5m2",
    "float8_e4m3",
    "float8_e3m4",
    "float8_e4m3fn",
    "float8_e5m2fnuz",
    "float8_e4m3fnuz",
    "float8_e4m3b11fnuz",
    "complex32",
    "complex64",
    "complex128",
]
