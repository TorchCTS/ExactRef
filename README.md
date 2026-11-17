# ExactRef

[![PyPI](https://img.shields.io/pypi/v/exactref)](https://pypi.org/project/exactref/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/exactref/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/TorchCTS/ExactRef/blob/main/LICENSE)

Exact reference answers, with proven error bounds, for testing numerical code.

When a kernel's output differs from a reference computed in float64, is that a bug, or ordinary rounding? ExactRef
answers per element. For an operation and its inputs, it computes the exact answer, rounds it once to the output format,
and gives a tolerance: how far the output of a correct finite-precision implementation can be from that answer, on
those inputs. An output beyond the tolerance cannot come from a correct implementation.

- Documentation: https://exactref.ai
- Source and bug reports: https://github.com/TorchCTS/ExactRef
- Proofs of every guarantee: https://github.com/TorchCTS/ExactRef/tree/main/proofs

## Install

```bash
pip install exactref
```

ExactRef needs CPython 3.10 or newer with the GIL (free-threaded builds are not supported) and has no dependencies:
its arithmetic is its own C extension. NumPy and PyTorch are not needed, but ExactRef reads their arrays and tensors
directly when you have them, as well as Python lists and any object with the buffer protocol.

## Quick start

A sum where the order of additions matters:

```python
>>> import exactref as er
>>> r = er.sum(er.asarray([1e16, 1.0, -1e16], "float64"))
>>> r.rounded.tolist()      # the exact sum, rounded once to float64
1.0
>>> r.normal().tolist()     # what adding in index order in float64 gives
0.0
>>> r.tolerance().tolist()  # how far a correct float64 implementation can be from 1.0
1.1102230246251574
```

The exact sum is 1.0. Adding left to right in float64 loses the 1.0 and gives 0.0, which is still correct: a correct
implementation can be up to 1.11 away. An implementation that returned 8.0 would not be.

Checking PyTorch's matrix product:

```python
import torch
import exactref as er

A, B = torch.randn(64, 256), torch.randn(256, 32)
r = er.matmul(A, B)
assert r.covers(A @ B).all()  # every element is one a correct implementation can produce
```

## What a check tells you

The tolerance describes an implementation that rounds only where the operation requires, such as each partial sum of
a matrix product and the final rounding to the output format, and that adds in an ordinary order: index order, or the
lanes of a SIMD loop. ExactRef calls this normal accumulation. Each output then falls into one of three cases:

- **Within `tolerance()`**: no worse than normal accumulation can be.
- **Beyond it, but within `tolerance(standard="any_order")`**: still correct, but worse than normal accumulation can
  be, for example from a poor order of additions.
- **Beyond the any-order tolerance**: no implementation that rounds only where the operation requires, without
  overflowing, could produce it.

`r.covers(output)` makes that judgment for every element at once, including infinities and NaN. A kernel that rounds
more than the operation requires, such as FlashAttention when it casts the attention weights to bfloat16, can fall
outside `tolerance()` without a bug. Describe such a kernel with a named algorithm (below), whose tolerance includes
its extra roundings.

## Reading a result

Every operation returns a `Result`:

```python
r = er.matmul(A, B)
r.rounded                 # the exact answer, correctly rounded once to the output format
r.exact                   # certified enclosures of the exact answer (an ExactArray)
r.exact.value((0, 1))     # Enclosure(midpoint, radius): the true value lies within radius of midpoint
r.guarantee               # "correctly_rounded"
r.tolerance()             # per element: max |implementation - r.rounded| over finite outputs, for normal accumulation
r.tolerance(standard="any_order")  # the same for any order of additions that does not overflow
r.tolerance("bfloat16")   # if implementations compute in bfloat16
r.tolerance(flush_to_zero=True)  # if they flush subnormals to zero
r.finite_range()          # (low, high): the lowest and highest finite output of a correct implementation
r.may_be("+inf")          # where a correct implementation may output +inf (or "-inf", "nan")
r.covers(output)          # where a correct implementation can produce the values in output
r.normal()                # the output of normal accumulation in index order (computed when asked for)
r.boundaries              # the operation's rounding boundaries
```

Where an overflow is possible, `may_be()` tells you which infinities or NaN a correct implementation may produce, and
it errs on the side of yes. An infinite tolerance marks an element with no bound at all, and a NaN tolerance an element
whose exact answer is not finite. For complex outputs, `tolerance()` is complex128 with one bound per part, and
`may_be()` and `covers()` have a last axis of 2: the real part and the imaginary part.

## Operations

| Operation | Shortcut | Inputs | Backward | `normal()` |
|---|---|---|---|---|
| `linalg.products.matmul` | `er.matmul` | real, complex, integer | yes | yes |
| `linalg.products.gemm` (`beta C + alpha A @ B`) | `er.gemm` | real, complex, integer | yes | yes |
| `arithmetic.basic.add`, `arithmetic.basic.multiply` | `er.add`, `er.multiply` | real, complex, integer | yes | yes |
| `reduction.arithmetic.sum` | `er.sum` | real, complex, integer, bool | yes | yes |
| `linalg.matrix_function.matrix_exp` | `er.matrix_exp` | real, complex | yes | no |
| `linalg.matrix_function.matrix_exp_frechet` | `er.matrix_exp_frechet` | real, complex | no | no |
| `nn.attention.scaled_dot_product_attention` | `er.scaled_dot_product_attention` | real | yes | yes |

Backward operations take floating-point and complex inputs. Formats run from the float8 variants through float16,
bfloat16, float32, and float64 to complex and integer formats, and the reference can round in any of the five IEEE
rounding modes.

## Beyond one operation

- **Chains.** Pass a result where an array goes. The next operation's tolerance then carries the uncertainty of the
  stored intermediate, so a computation with several steps, such as a transformer layer, is checked step by step.
- **Named algorithms.** These give tolerances for specific implementations: fixed orders of additions, NVIDIA tensor
  cores (with your GPU's measured parameters), fused attention kernels, and an `add` or a `multiply` that rounds a
  scalar operand to a format first, as PyTorch's CPU kernels do when you write `x * 0.1`.

## Guarantees and limits

- **The reference is exact or the call raises.** Every element of `rounded` is the exact answer rounded once, or the
  call raises `ReferenceUnavailableError`. ExactRef raises its working precision until it knows which way each exact
  value rounds, up to a cap. Two kinds of call can reach the cap:
  - attention's backward, where `grad_key`, `grad_value`, and `grad_query` summed over broadcast copies of a query row
    add terms from rows with different softmax denominators. Unless every term is rational, such a sum that lands
    exactly on a rounding boundary, or cancels to exactly zero, reaches the cap, and so can an element very close to a
    rounding boundary where the effect of a heavily masked key cancels exactly;
  - the matrix exponential, its Frechet derivative, and its backward, for an entry that needs more than 8192 bits, a
    very large matrix (from about 6600 x 6600 real or 3300 x 3300 complex), or a matrix that is not diagonal with
    `||A||_inf` of 2**60 or more. Their docstrings give the limits exactly.
- **What the tolerance covers.** Every implementation that loses precision only at the operation's rounding boundaries
  and rounds to nearest there. The `rounding` argument sets the rounding of the reference and of the final rounding to
  the output format.
- **The matrix exponential is different.** No rounding step is shared by every algorithm (Taylor or Pade, any number of
  squarings), so its tolerance bounds a backward error: the output is `exp(A + dA)` with
  `||dA||_inf <= ulps * (n + 16) * u * max(||A||_inf, 1)` (`2n + 16` for `matrix_exp_frechet` and `matrix_exp`'s
  backward, on their 2n x 2n block matrix), where `u` is the compute format's unit roundoff. That allowance is an
  assumption, measured on random matrices up to 64 x 64, where Higham's scaling and squaring and PyTorch's
  `matrix_exp` mostly stay within half of it. The guide lists the cases measured outside it (matrices near a multiple
  of the identity, and PyTorch's results at small norms), and `ulps` raises the allowance.
- **Formats without infinity** (`float8_e4m3fn` and the `fnuz` formats). A finite result too large for the format
  rounds to its largest value, or to NaN where a directed rounding points away from zero. PyTorch's conversions turn
  infinities into the largest value for `float8_e4m3fn` (`overflow="saturate"` reproduces this) and give NaN for the
  `fnuz` formats (`overflow="nan"`).
- **Flushing subnormals.** Some code sets the CPU to flush subnormal numbers, the tiny values below the normal range,
  to zero, for example `torch.set_flush_denormal(True)`. ExactRef turns that off during each call and restores it
  afterwards. Where the platform offers no way to do that, the call raises `ReferenceUnavailableError`.
- **Threads.** Operations, and the methods that compute bounds, take `workers`, the number of threads to use, by
  default as many as PyTorch uses: the physical cores, or the performance cores on Apple Silicon. No result depends on
  it.

## Documentation

The guide at https://exactref.ai/guide/ covers how tolerances work and how to read complicated ones, results, data in
and out, backward operations, chains and `any_of`, named algorithms, and the guarantees and limits. Each operation's
reference page gives its rules (shapes, promotion, special values, and the roundings its tolerance allows for), and the
reference maps PyTorch's calls onto ExactRef's.

## How ExactRef is tested

The tests are developed separately. They check ExactRef's answers against independent references: MPFR and MPC
(through gmpy2), FLINT (through python-flint), SymPy, NumPy (for integer wraparound), and the Python standard library.
PyTorch is never a reference there: it is an implementation whose outputs the tests check against ExactRef's bounds.
Each operation's guarantees and tolerances are proven in [`proofs/`](https://github.com/TorchCTS/ExactRef/tree/main/proofs).

## License

ExactRef is released under the [MIT License](https://github.com/TorchCTS/ExactRef/blob/main/LICENSE).
