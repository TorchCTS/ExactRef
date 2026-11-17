# matmul: proofs

This file proves what `exactref.matmul` claims: the reference output is the
exact product correctly rounded; `tolerance()` bounds normal accumulation and
`tolerance(standard="any_order")` every implementation that rounds only at
the required boundaries without overflowing; and `normal()` is the output of
normal accumulation in index order. The notation, units and lemmas it cites
(1 to 3, S, B, L, O, A, E and T) are in `foundations.md`; Lemmas R and P are
in `kernels.md`.

## The reference output

Every finite input is a dyadic rational m * 2**e. For real floating-point
operands of at most 64 bits, the product kernel (`csrc/product.c`;
`kernels.md`, Lemmas R and P) reads each operand's values as integers at one
main exponent, values below that range kept as flagged slots, and computes
each output element's exact sum: in 128-bit words where every value it reads
is a slot's integer, and otherwise in multi-word integers at the exponent of
its finest term. The same read covers every operand: a complex matrix's parts
make rows of 2k terms (A's real and imaginary parts interleaved, and B's
operand for each output part pairing them with (b_re, -b_im) or (b_im, b_re),
`operand_parts`), integer values are read as integers at exponent 0, and
float64 values, like every operand with radii, take the wide main range
(two-word slots and 256-bit sums). The exact
sum is rounded once, in the requested rounding direction (`formats.c`), so
the output is the exact product correctly rounded.
Non-finite products and sums follow the IEEE rules on exact values
(`product.c`'s `special_sum`). An exact zero sum is +0 (-0 when
rounding toward -inf), as PyTorch's sums start from +0; an empty sum is +0.
An element that is not a reduction (`multiply`'s and `add`'s elements, a
gradient element with one copy of an operand with the output's number of
dimensions) instead takes the sign IEEE 754 6.3 gives its expression, as
PyTorch's elementwise operations do. Each term's sign is the exclusive or of
its factors' signs, and a term the coefficient rule leaves out takes no part.
A part whose terms are all zeros of one sign keeps that sign; otherwise
(opposite signs, or nonzero terms that cancel) the sum rule applies. Integer
outputs are exact and wrap (or raise or saturate) on overflow. Complex outputs use the four real
products per term, exactly, and each part follows the IEEE rules on its own
real terms (a_re b_re and -a_im b_im for the real part, a_re b_im and
a_im b_re for the imaginary part), as PyTorch's elementwise and batched
complex products do. When B holds exact coefficients (`add`'s column [1, a],
the gradients' constant columns), a zero part of a nonzero coefficient adds
no term: a real coefficient scales each part and an imaginary one rotates
them, which is the mathematical answer. A coefficient that is exactly 0 has
no such answer against an infinity, so it multiplies by IEEE rules, as
PyTorch does (`special_sum`; the normal answer follows the same rule). A
finite part next to a non-finite one is exact: its terms are all finite.
`exact` holds the same exact sums (before any wrap), with radius 0 unless a
sum is longer than 128 bits.

## The tolerance

**Model.** A correct implementation multiplies the given input values exactly
(possibly as part of a fused multiply-add) or rounds each product to the
compute format F, accumulates the k products of each output element in F,
and rounds the result to the output format. Not covered: rounding the inputs
themselves to another format first (TF32 and similar); consumers pass such
inputs already rounded. A complex product uses the four real products. Each
part is a sum of 2k real products, in index order a_re b_re, -a_im b_im for
each k (real part) or a_re b_im, a_im b_re (imaginary part). A Karatsuba
three-product formula has different terms and is not covered.

**Claim.** For each output element, `tolerance()` is at least |output - ref|
for normal accumulation (index order or lanes 2 and 4, with the slots), and
`tolerance(standard="any_order")` for every order that does not overflow.

**Proof.** For an element, s, the mass and N (`foundations.md`, The normal
bound) are exact integers, for each part, from the product kernel
(`csrc/product.c`), every node exact. With flush, the kernel adds into moved
the magnitude of each term with a factor subnormal in F, so moved is exact
too.

- Normal: when mass + B < Theta_F (Lemma O (e)), the interval is
  [s - B', s + B'], with B' the smaller of B and Lemma A's bound (Lemma O (e):
  no normal order overflows, so Lemma A's bound holds for them too); else it
  is Lemma A's interval, which holds for every finite output, with the flags
  of Lemma O (f) (`csrc/sums.c`).
- Any order: Lemma A's interval, with the flags of the sides on which the
  final value may overflow and of a term's own overflow (Lemma A).

Lemma 3 then turns the interval into the tolerance, the finite range and the
flags (`foundations.md`, Finite outputs and overflow). []

## Algorithms

A matmul algorithm (`matmul_sequential`, `matmul_pairwise`, `matmul_blocked`,
`matmul_exact_accumulation`) fixes one accumulation method in an accumulator
format F rounding to nearest (ties to even or away; a directed rounding is
not modeled, `foundations.md`, The normal standard, Units), with products
exact (fused multiply-add) or rounded to F first. `run` emulates it exactly
with integer arithmetic (`emulate.c`). Its `tolerance(A, B)` is Lemma S
over the method's own tree, in F's units (`MatmulAlgorithm.tolerance`), with
every node of the tree taken exactly by the product kernel (Lemma B's blocks
for operands with radii):

- sequential: index order, D = k - 1 (the first addition, 0 + t_1, is the
  leaf's own rounding);
- blocked: each block of b terms is a chain from its first term, then the
  block sums in order; D = min(b, k) - 1 + ceil(k / b) - 1;
- pairwise: the balanced tree (the first ceil(n/2) terms, then the rest),
  each node an exact difference of running sums at its ends;
  D = ceil(log2 k);
- exact accumulation: the exact sum rounded once (Lemma 1).

A fixed order has no slots, and it is itself the implementation, so its
partial sums can overflow. Where mass + B could reach Theta_F, Lemma S's
interval over its tree still holds for every finite output (`foundations.md`,
Finite outputs and overflow), and the flags are +inf where the positive terms'
total plus B could reach Theta_F, -inf for the negative terms' total, and NaN
where both could (`csrc/sums.c`, `sum_bounds_tree`); in an F without infinity each of
these is NaN (`foundations.md`, Lemma 3: the compute format keeps the IEEE
rule with NaN for the missing infinity). Exact accumulation that may round
past F's largest finite value holds the sum no further out than that value for
its finite outputs, with the flags of the sides that may overflow, NaN in an F
without infinity. Lemma 3 then gives the distance from the definition's
rounded output. `error_model(k)` reports Lemma 2's coefficient gamma(D) and
its absolute term R * eta * (1 + gamma(D)) for k products, as exact rationals:
D is the method's depth, which counts each term's own rounding (the product
rounded first, or the fused 0 + t_1), so it is the same with or without fused
products, and R the method's count of roundings. It is metadata for comparing
algorithms, not a tolerance.

**The 16-slot algorithm** (`matmul_slots16`). Per output element, each product
is rounded to F, to nearest as above. The finite products go into 16 slots by
magnitude: slot s holds the products t with 2**(s p) <= (T / |t|)**16 < 2**((s
+ 1) p), where T is the largest finite magnitude, so each slot spans p / 16
binades below T (1.5 for float32), and anything smaller goes to the last slot;
exact zeros add nothing. Each slot adds its values in increasing magnitude,
alternating positive and negative values (positive first, then whichever
remain), and the slot sums are added from the smallest slot up. A product that
is not finite (a factor not finite, or a rounding past the largest value)
takes no slot: those products join the slots' sum last, by IEEE rules, so an
infinity meets a slot sum that overflowed to the other one as NaN. Where one
occurs and the reference is finite, the element has no finite bound; where
the reference is not finite, its flags are the infinities rule's
(`foundations.md`, Results as inputs; the sides of any tree). This is a
summation tree, so Lemma S applies with each slot's running sums and the
combination's partial sums as its nodes, computed exactly per element from the
exact products, with D the longest path. Its nodes are not adjacent ranges, so
the tolerance sorts each element's terms. The overflow check uses the largest
exact node or term magnitude in place of the mass, which is still an upper
bound on every value it rounds. `error_model` is sequential summation's with
rounded products, the worst case (one slot holding every product).

**Tensor cores** (`matmul_mma`). Each output element is computed by a
tensor-core kernel (`foundations.md`, Lemma T) with the caller's step
parameters and split granularity
`MMAStep(products, fraction_bits, exponent_floor, split)` and an optional
reduction format. The inputs are float16, bfloat16, or float32
values that are tf32 values (`round_to_tf32`), so every product is exact.
Per element, s and the mass M are the exact integer sums above, and
Pos = (M + s) / 2 and Neg = (M - s) / 2. sigma is
2**emin_I max|B| where A holds a subnormal value, plus 2**emin_I max|A|
where B does, plus 2**(2 emin_I) where both do (zero without subnormal
inputs). Lemma T gives the interval [s - G M - H, s + G M + H] for every
finite output, with the overflow flags of Lemma T's Overflow paragraph, and
Lemma 3 turns it into the tolerance, the finite range and the flags, also
where the reference is infinite because the exact sum rounds past the output
format's largest value (a finite output may still occur there). Both
standards give this bound. Earlier results as inputs widen s by the
deviation and M by it, as for the exact accumulation. A float32 operand is
an array of tf32 values, never an earlier result: that result's stored
values need not be tf32 values, and a kernel's rounding of them to tf32 is
not in this bound. An operand that is an
earlier result counts as one that may hold subnormal values, with its
maximum taken over |x| + r (r its radius).

## Results as inputs

The general rule is in `foundations.md` (Results as inputs).

**Claim.** With y, q the other operand's values and radii (per part for
complex values), `tolerance()` is at least |output - ref| for every finite
output of every implementation of the standard (or of the algorithm) on every
x', y' whose parts lie within r and q of x's and y's, where ref is the
reference output on x and y. A part whose radius is inf has no such bound, and
every output part it reaches has an infinite tolerance where its reference is
finite; stored infinities and NaN pass on as flags (`foundations.md`, Results
as inputs).

**Proof.** A term t' = x'_k y'_k of the implementation satisfies

    |t' - x_k y_k| = |x'_k (y'_k - y_k) + (x'_k - x_k) y_k| <= |x_k| q_k + r_k |y_k| + r_k q_k = delta_k.

For a complex part each of its two real terms is a product of one part of x
and one part of y, each with its own radius: the real part's terms x_re y_re
and x_im y_im move by at most
|x_re| q_re + r_re |y_re| + r_re q_re + |x_im| q_im + r_im |y_im| + r_im q_im,
and the imaginary part's terms x_re y_im and x_im y_re by the same sum with
y's parts swapped. Each is at most the bound with r and q the moduli. Delta,
per part, is computed exactly (`product_deviation`: the integer products
|A_p| Q_q + R_p |B_q| + R_p Q_q over the part's two pairings). So the
implementation's exact sum lies within Delta of s, the exact sum of the
midpoints' terms, and its mass is at most mass + Delta.

- Normal: Lemma S's remark on interval terms gives the bound with the
  nodes' midpoint sums, the mass mass + Delta and Delta added to moved; the
  nodes come from the midpoints as before, so N + 2 Delta
  bounds 2 (mass + Delta + sum of |v_node|). The overflow check of Lemma O
  uses mass + Delta, which bounds every term and partial sum of the
  implementation.
- Any order: Lemma A bounds the distance from the implementation's exact
  sum with mass + Delta; adding the Delta between that sum and s gives
  Lemma A with Delta added to moved, which counts moved at least once in
  both forms. Its overflow check with |s| covers |s| + Delta the same way.
- Flush: a stored value whose box reaches below 2**emin of the compute
  format may be read as zero. Such a part's radius is first widened to cover
  0 (to max(r, |x|)), so the term it makes is still within its box. Flushes
  of the midpoints count in moved as before.
- No bound: a part with radius inf gives no interval for the output parts
  that read it: in a plain product, both parts of every output element in
  its row of A or column of B; in a coefficient product (B exact, as in
  `add`'s column [1, a] and the gradients), the output parts that read it
  through a coefficient part the coefficient rule keeps. Lemma 3 makes their
  tolerance infinite, or NaN where the reference is not finite.
- Flags: a stored infinity or NaN passes on as `foundations.md` (Results as
  inputs) describes; in a plain product the other factor is data, so any
  flag in the row or column sets all three (`kernel_bounds`), and in a
  coefficient product the coefficient part's sign decides (`product_reach`).
  Under either standard NaN is flagged where two infinities of opposite
  signs can meet, as `foundations.md` (Results as inputs) lists them: a
  passed one, the exact value's own, a term's overflow (Lemma O's term flags,
  with the radii) and, where one of those occurs, the finite terms of the
  other sign reaching the threshold together (Lemma O (g); for a named order,
  the subtrees that join an infinite term's path) (`infinities_meet`). A term
  is at most X Y and a side at most k X Y, X and Y the row's and the column's
  largest |x| + r, so the terms are examined one by one only where X Y can
  reach the threshold, or where one of those events occurs and k X Y (by
  Lemma O (g)) can.

The sums are expressed exactly at the finer of their exponent and Delta's.
Lemma 3 then gives the distance from ref. []

The algorithms take the same Delta: `MatmulAlgorithm.tolerance` is Lemma S
over its fixed tree with mass + Delta, N + 2 Delta and moved Delta, and exact
accumulation rounds a sum in [s - Delta, s + Delta] (Lemma 1). The 16-slot
algorithm sorts the stored products, which are not known, so an element with
uncertain terms gets Lemma S for any tree: D = n - 1 and each |v_node| at
most the midpoints' mass, so N = 2 (mass + Delta + (n - 1) mass), with
mass + Delta as the overflow check's largest magnitude. `run` uses the
values x.

`normal()` is normal accumulation on the stored values of normal
implementations at every step: on the earlier results' `normal()`, by
Lemma E. The backward's gradients are matmuls of rearranged operands: each
element keeps its radii (conjugation keeps each part's magnitude, and swaps
the imaginary part's infinities and its range's ends with their negatives)
and its normal value.

## Backward

`matmul.backward(A, B, grad)` returns grad_A = grad @ B^H and
grad_B = A^H @ grad (^H: conjugate transpose), each summed over the batch
copies that broadcasting made of its operand, with that operand's shape.

**Claim.** Every gradient element is the exact value correctly rounded once,
its `exact` entry encloses it, and its tolerances bound, as the forward's do,
normal accumulation of its products in the compute format (`tolerance()`) and
every order that does not overflow (`tolerance(standard="any_order")`), each
followed by one rounding to the output format.

**Proof.** Take an element of grad_A: batch index alpha of A, row i, column r.
Its exact value is the sum, over the output batch indices beta that broadcast
onto alpha and over j, of grad[beta]_ij * conj(B[beta'])_rj, where beta' is
the batch index of B that beta reads. With F such indices beta, these are F * n
exact products. `_matmul_backward` lays them out as row i of the matrix
[grad[beta_1] ... grad[beta_F]] and column r of the matrix whose rows are the
rows of conj(B[beta_1])^T, ..., conj(B[beta_F])^T. Building these matrices only
copies encodings and flips imaginary sign bits, which is exact. So the element
is one entry of one matmul with contraction length F * n, and the forward
proof (The exact output, The reference output, The tolerance, Lemma E)
applies to it unchanged. grad_B is the same with the F * m products
conj(A[beta''])_ir * grad[beta]_ij. A 1-D operand is promoted as in the
forward, so a dot product's gradient is a matmul with contraction length 1. []

An implementation that rounds each batch's product before summing the copies
adds a rounding this tolerance does not budget. PyTorch does so when both
operands keep batch dimensions that broadcast against each other.
