# matrix_exp_frechet and matrix_exp backward: proofs

This file proves what `exactref.matrix_exp_frechet` and `matrix_exp.backward`
claim. Both rest on `proofs/matrix_exp.md`.

## The block identity

For square matrices A and E of size n, let M = [[A, E], [0, A]] (size 2n).
The Frechet derivative of exp at A in the direction E is

    L(A, E) = d/dt exp(A + t E) at t = 0.

**Claim.** The top-right n x n block of exp(M) is L(A, E).

*Proof.* By induction on k, M**k = [[A**k, S_k], [0, A**k]] with
S_k = sum_{j=0}^{k-1} A**j E A**(k-1-j): this holds for k = 1, and multiplying
by M gives the top-right block A**k E + S_k A, which is S_{k+1}. So the
top-right block of exp(M) = sum_k M**k / k! is sum_k S_k / k!.

On the other side, (A + t E)**k is a polynomial in t whose coefficient of t is
exactly S_k (each term of the expanded product takes E from one factor and A
from the others). The series sum_k (A + t E)**k / k! converges absolutely and
uniformly for t in a bounded interval, and so does the series of its
termwise derivatives, since every norm is bounded by a convergent series of
the form sum_k k c**(k-1) / k!. So it may be differentiated term by term, and
its derivative at t = 0 is sum_k S_k / k!. []

This is a known identity (R. Mathias, "A Chain Rule for Matrix Functions and
Applications", SIAM J. Matrix Anal. Appl. 17(3), 610-620, 1996; N. J. Higham,
"Functions of Matrices", SIAM, 2008); the argument above makes this file
self-contained.

## The reference output

`matrix_exp_frechet(A, E)` has matrix_exp's reference kernel
(`csrc/matrix_exp.c`) build M exactly (copying values, with exact zeros in the
bottom-left block) and compute its top-right entries only. By the block
identity these entries are exactly L(A, E).

The kernel first scales E by an exact power of two. L(A, E) is linear in E,
so L(A, 2**-k E) = 2**-k L(A, E), and M' = [[A, 2**-k E], [0, A]] is again a
matrix of exact dyadics. With t_A and t_E the least integers bounding A's and
E's row sums of |part| (|re| + |im| for complex parts) by 2**t, it takes
k = t_E - max(t_A, 1), so that E's row sums lie in the binade of
max(||A||_inf, 1), except where t_A > 57: there it takes k = max(t_E - 57, 0),
scaling E only down, below 2**57, so that M''s row sums stay below A's plus
2**57 and never pass those of the unscaled block. M' is then within the
kernel's range whenever A's row sums are below 2**58, however large E is, and
a small E costs the top-right block no precision while t_A <= 57. The kernel decides each wanted entry of exp(M') as below, with
each enclosure's ends, and each exact rational value, multiplied by 2**k
exactly before the only rounding: the decision, the zero rule, the exact entry
and the statuses all act on L(A, E)'s own values, and no rounded output is
rescaled. `proofs/matrix_exp.md`, The reference output, applies to M' (its
norms and sizes are M''s); k sets only the cost.

Everything in `proofs/matrix_exp.md`, The reference output, applies entry by
entry: M is a rational matrix, so each part of an entry of exp(M) is rational
or transcendental (Lindemann-Weierstrass); structural zeros come from M's
graph; refinement ends for transcendental parts; parts still unresolved are
tested once for being rational, when the test is due, and rounded from their
exact values. Restricting the work to the top-right entries changes no
entry's value or rounding; it only stops the other entries from being
computed (and from raising at the precision cap). Every returned entry is
therefore correctly rounded once, or the call raises
`ReferenceUnavailableError`.

`matrix_exp.backward(A, grad)` is `matrix_exp_frechet(A^H, grad)`, where A^H is
built by transposing and flipping imaginary sign bits, both exact. This is the
gradient PyTorch computes: the adjoint of v -> L(A, v) under the real inner
product <X, Y> = Re trace(X^H Y) is G -> L(A^H, G).

## The tolerance

**Model.** A correct implementation computes exp(M) for the block matrix, as
PyTorch's `matrix_exp` backward does, and is correct in the sense of
`proofs/matrix_exp.md`, The tolerance, for that 2n x 2n matrix: its output
entries are fl_F(exp(M + dM)_ij) with
||dM||_inf <= c * (2n + 16) * u * max(||M||_inf, 1), then rounded to the
output format.

**Claim.** Under the model, `tolerance()` is at least |output - ref| for every
returned entry.

**Proof.** The tolerance kernel (`csrc/matrix_exp.c`, `mtol`) runs on M with
size 2n and keeps the top-right positions. That is the bound proven in
`proofs/matrix_exp.md`, The tolerance, for the matrix M, evaluated at those
entries. []

The model lets dM perturb the zero block as well, which a block-structured
implementation never does. That only makes the bound larger, so it stays
valid. The tolerance kernel takes M unscaled, as the model states it: where
M's norm is past the kernel's range there is no box and no ball, and the
tolerance is infinite. The model allows that: its dM may then reach
c (2n + 16) u 2**60 in norm, which in float64 already lets exp(M + dM) reach
e**2304.

When A or E (or grad) is an earlier result, each entry of M gets its
radius: A's in both diagonal blocks, E's in the top-right block, 0 in the
zero block. That is `proofs/matrix_exp.md`, Results as inputs, for M; the two
copies of A are bounded independently, which only widens the sets enclosed. For the
backward, A^H's radii are A's, transposed (conjugation keeps each part's
magnitude).
