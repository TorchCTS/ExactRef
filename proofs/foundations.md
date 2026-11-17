# Shared definitions and lemmas

The definitions and lemmas every operation's proof builds on: the notation,
one rounding of an interval (Lemma 1), a sum in any order (Lemma 2), block
sums carried in a second format (Lemma C), sums by tensor-core steps (Lemma
T), the final rounding and the tolerance (Lemma 3), how `exact` stores
enclosures and decides a rounding by a sign, directed dyadic arithmetic and
the enclosures of ln 2, exp, log, pi, cos and sin (Lemmas Q, X, W, W3, Y, M
and Z), the normal standard (Lemmas S, B, L, O and A, exact terms in Lemma U
and few terms in Lemma F), the normal answer (Lemma E), and results as
inputs.

## Notation

A floating-point format F has precision p (significand bits, including the
implicit bit) and smallest normal exponent emin. For a real y != 0 let
e(y) = floor(log2 |y|), and let

    ulp(y) = 2**(max(e(y), emin) - p + 1),

the spacing of F's values near y (subnormal spacing below 2**emin). Rounding y
to F with an error of at most c units in the last place means

    |fl(y) - y| <= c * ulp(y).

Correct rounding to nearest has c = 1/2 and directed rounding has c = 1. A
library function that is accurate to c ulps, such as `exp` within 2 ulps,
satisfies the same inequality with that c. In an operation's tolerance model,
every rounding in the compute format is to nearest, whatever `rounding` the
operation is given; `rounding` sets the reference's rounding and the model's
final rounding to the output format (Lemma 3). A named algorithm sets its own
roundings. The flush-to-zero option changes
only one thing: a result whose rounded value would be subnormal may be replaced
by zero.

## Lemma 1: one rounding of an interval

Let X = [lo, hi], M = max(|lo|, |hi|), and L be F's largest finite value. The
overflow threshold T is L plus half an ulp of L for correct rounding to
nearest (c = 1/2), and L itself in the directed modes and for library
functions within c > 1/2 ulps. If M < T (M <= L in the directed modes and for
library functions), then every fl(y) with y in X is finite and lies in
(X + [-r, r]) intersected with [-L, L], where

    r = c * ulp(M)                       (no flushing)
    r = c * ulp(M) + 2**emin             (flushing, if X reaches below 2**emin in magnitude)

*Proof.* ulp is nondecreasing in |y|, so for y in X,
|fl(y) - y| <= c * ulp(y) <= c * ulp(M). With flushing, a flushed result is 0
and comes from a y whose rounded value is subnormal, so |y| < 2**emin and the
error is |y| < 2**emin; a result that is not flushed obeys the first bound.
Rounding to nearest gives a value of magnitude at most L for every |y| < T: a
y between L and T rounds to L. A directed rounding of |y| <= L gives a value
between y and the next value of F toward its direction, so at most L; a
library function within c ulps returns a value of F, and F has no finite
value beyond L. So every fl(y) is finite, of magnitude at most L, and within
r of y. []

A boundary may take the flushing case on its own, without the flush-to-zero
option (its setting's `flush`): a library function that, for a result below
2**emin, returns 0 or another value of F within 2**emin of it. The proof's
flushing case covers it unchanged, since that error is below 2**emin.
Every reader of a setting takes the flushing case under the flush-to-zero
option or where its setting's `flush` allows it.

Every fl(y) is also a value of F (0 included, with flushing), so the
enclosure is snapped inward to F's values: its lower end rounded up and its
upper end rounded down to F's grid (the multiples of ulp at each end's
magnitude, the subnormal spacing below 2**emin; a format without subnormals
holds a subset of these, so the snapped ends still enclose its values). This
matters where r spans less than one spacing of F, as near a huge mask value:
a logit within half an ulp of -L has the single value -L, where r alone
would leave an enclosure about one ulp wide. Should the snapped ends cross
(no value of F in the enclosure), the unsnapped enclosure is kept.

The snapped enclosure bounds the rounded values, not the rounding's error:
its ends can lie closer to X than r (an end of X on F's grid stays where it
is). A bound that needs the error itself, |fl(y) - y| <= r, takes r (the
`lean_error`, `errors.c`).

This is `lean_ends` (`errors.c`). Where M
reaches the threshold it reports no bound, and the tolerance is infinite. Below it, clipping to [-L, L] changes nothing when M + r <= L;
when M + r > L it keeps a bound that is still finite.

## Lemma 2: a sum in any order

Let t_1 .. t_n be exact real terms and s = sum t_i. An implementation computes
s by any sequence of rounded operations to F: additions of two values, fused
multiply-adds that merge a term into a value, and roundings of a single term
(for example `fma(a, b, 0)`). Each rounding obeys

    fl(x) = x * (1 + d) + h,   |d| <= u,   |h| <= eta,

with u = 2c * 2**-p, and eta = 2c * 2**(emin - p) (the absolute error of a
subnormal result), or eta = 2**emin with flushing (a flushed result has
d = 0 and h = -x). Let D be the most roundings on any term's path to the result
and R the number of roundings whose h can be nonzero. If D * u < 1, the computed
sum s' obeys

    |s' - s| <= gamma(D) * sum |t_i| + R * eta * (1 + gamma(D)),
    gamma(D) = D * u / (1 - D * u).

For any order, D <= n, and R <= n without flushing (R <= 2n - 1 with flushing).

*Proof.* Unrolling the operations,

    s' = sum_i t_i * prod_{j in path(i)} (1 + d_j) + sum_k h_k * prod_{j in after(k)} (1 + d_j),

where path(i) are the roundings on term i's way to the result and after(k) are
the roundings after rounding k on its way to the result; |path(i)| <= D and
|after(k)| <= D - 1. By Higham (Accuracy and Stability of Numerical Algorithms,
2nd ed., Lemma 3.1), |prod_{j=1..m} (1 + d_j) - 1| <= gamma(m) and
prod_{j=1..m} (1 + |d_j|) <= 1 + gamma(m) for m * u < 1, and gamma is increasing.
Subtracting s gives the bound.

For the counts: the operations form a tree whose leaves are the n terms. A tree
with n leaves has height at most n - 1, and a term may also be rounded once on
its own before it enters the tree, so D <= n.

Without flushing, an addition of two numbers of F has h = 0. Every number of F
is an integer multiple of 2**(emin - p + 1), so their sum is one too. If the
sum's magnitude is below 2**emin, it is a subnormal number of F and the
addition is exact; otherwise the result is normal and only d applies. So h can
be nonzero only at a rounding that involves an exact term: a fused multiply-add
that merges it, or the rounding of a term on its own. Each term passes exactly
one such rounding, so R <= n. With flushing, an addition whose result would be
subnormal is flushed, so every rounding counts. Every addition or fused
multiply-add merges two values into one, so exactly n - 1 of them reduce n
values to one, and at most n roundings act on single terms, so R <= 2n - 1. []

Any order uses D = n with R = n, or R = 2n - 1 with flushing (`errors.c`'s
`any_order_counts`, which `sums.c` and the attention programs share). So the bound holds for every order, with or
without fused multiply-add, and also when each product is rounded first. The
code evaluates it with outward-rounded dyadic arithmetic
(`attention_tol.c`'s `acc_of` and `acc_apply`, which the step-by-step
programs use; the tolerances' any-order level uses it through Lemma A below).
If D * u >= 1, or if mass + error could exceed F's largest finite value (so a
partial sum could overflow), they report no bound. The partial sums are
bounded by mass + error by the same unrolling.

**Exact zero terms.** An implementation may also add terms that are exactly
0 in every implementation (the keys an attention row does not keep). Such a
term's own rounding is exact, and so is an operation that adds one, or a sum
of them, to a value x (x + 0 = x, fma(0, b, x) = x), except that with
flushing it may give 0 for a subnormal x. That x is the result of a rounding
of the sum without the zeros: a merge of two of its values, or the operation
where a term enters it (its own rounding, or a fused multiply-add taking it
into a sum of zeros, which rounds that term alone). Its result x being
subnormal, that rounding may flush it itself, and from there to the zero's
addition x passes only additions of zeros, which leave it. So every
computation with the zeros is one of the sum without them, and the bound for
n, the terms that can be nonzero, holds: R counts a rounding of its own for
every term, where each such flush is charged. Without that count (exact
terms, Lemma U), a subnormal term could reach a zero's addition with no
rounding before it, and each term's flush there would be one more.

## Lemma C: block sums carried in a second format

Some kernels add a sum block by block into a value held in a second format C,
rounding it to C after every block: a matrix product with beta = 1 into an
output of a narrower format, called once per block of the sum.

**Model.** The terms t_1 .. t_n are split into blocks 1 .. m in order; block j
has exact sum B_j, mass M_j (the sum of its |t_i|) and n_j terms that can be
nonzero. Each term is a product t_i = x_i y_i of two exact factors, and alpha
is a real scale in a known interval. Starting from c_0 = 0, the implementation
computes, for each block,

    c_j = fl_C(v_j),

where v_j is computed in the accumulator format A (units u, eta and gamma as
in Lemma 2) from c_(j-1), taken as one more exact term, and the block's terms
scaled by alpha, using Lemma 2's operations, with the scale applied in one of
these ways:

1. to the block's sum: the block's terms are summed in A, the result is
   multiplied by alpha and rounded, and c_(j-1) is added (or both in one fused
   multiply-add);
2. to a factor of each term: x_i (or y_i) is multiplied by alpha and rounded
   to A before the product;
3. not at all, when alpha = 1 exactly.

The value wanted is s = alpha (B_1 + ... + B_m). Every rounding to C before
the last is to nearest.

**Statement.** Let D = n_j + 2, R be Lemma 2's count for n_j + 1 terms plus 1,
and F_j = sum of |x_i| + |y_i| over the block's terms in way 2 (F_j = 0 in
ways 1 and 3). If D u < 1, then

    |v_j - c_(j-1) - alpha B_j| <= gamma(D) (|alpha| M_j + |c_(j-1)|) + (R max(1, |alpha|) + F_j) eta (1 + gamma(D)),

provided this error plus |alpha| M_j + |c_(j-1)| is at most A's largest finite
value, and in way 2 |alpha| F_j is too (so no partial sum and no scaled
factor overflows). So if c_(j-1) lies in an interval X, then v_j lies in
X + alpha B_j widened by this error with |c_(j-1)| <= max |X|, and c_j lies in
Lemma 1's enclosure (format C) of that interval. By induction on j, every c_j
has such an enclosure; the interval of the last block not skipped (below),
before its rounding, goes to Lemma 3 as the value before the final rounding.

*Proof.* Way 2: a scaled factor is fl(alpha x_i) = alpha x_i (1 + d) + h with
|d| <= u and |h| <= eta, so the term the implementation forms is
alpha t_i (1 + d) + h y_i (or h x_i). Lemma 2 for the n_j + 1 terms (these and
c_(j-1)) unrolls v_j as in its proof: each term's path has at most n_j + 1
roundings, so alpha t_i has at most n_j + 2 factors (1 + d) and
|alpha t_i (prod (1 + d) - 1)| <= gamma(D) |alpha t_i|; c_(j-1) has at most
n_j + 1 <= D; the h y_i are multiplied by at most n_j + 1 factors, so they add
at most eta (|x_i| + |y_i|) (1 + gamma(D)) each, F_j eta (1 + gamma(D)) in all;
and Lemma 2's own absolute parts add at most its count times
eta (1 + gamma(D)).

Way 1: the block's sum w is Lemma 2's computation of its n_j terms, so
w = sum_i t_i prod_path(i) (1 + d) + sum_k h_k prod_after(k) (1 + d), with at
most n_j roundings on each path and at most Lemma 2's count for n_j terms of
absolute parts. Then fl(alpha w) multiplies everything by alpha (1 + d) and
adds an h; c_(j-1) may be rounded to A on its own (one more h, where it is not
a value of A); and the addition multiplies by one more (1 + d) and adds an h
that is zero without flushing (an addition of two numbers of A). So
alpha t_i's path has at most n_j + 2 roundings and c_(j-1)'s at most 2, and
the absolute parts are those of w, multiplied by |alpha|, plus at most three
more (two without flushing). Lemma 2's count is n for n terms (2n - 1 with
flushing), so that is at most n_j + 2 (2 n_j + 2 with flushing) = R, each part
at most max(1, |alpha|) eta (1 + gamma(D)). (With n_j = 0, w = 0 and its
scaling is exact, so at most two, and R = 2.) A fused multiply-add rounds once
instead of twice, which these counts cover.

Way 3 is way 2 with exact scaled factors (h = 0, d = 0), so F_j = 0.

Each case gives the bound by the triangle inequality, as in Lemma 2. The
partial sums are bounded by |alpha| M_j + |c_(j-1)| plus the error, by the same
unrolling, and in way 2 each scaled factor by |alpha| F_j, so the proviso
rules out overflow. Then c_j = fl_C(v_j) lies in Lemma 1's enclosure of the
interval containing v_j, which gives no bound where C could overflow. []

**Empty blocks.** A block with no term that can be nonzero (n_j = 0) leaves
c unchanged when every value of C is a value of A: every product is an exact
zero, every addition to c_(j-1) is then exact, and c_(j-1) is already a value
of C. Such a block is skipped, as is a block the kernel does not compute at
all; after the last block not skipped, the output is that block's rounding. Where C is not contained in A, the block is taken like any other: its
enclosure contains c_(j-1) as well.

The backward kernel applies it (`csrc/attention_grad_tol.c`, `carry_add`). It
evaluates the error with Lemma 2's terms of the any-order method at n_j + 2
terms, whose depth n_j + 2 and count (n_j + 2, or 2 n_j + 3 with flushing) are
at least D and R, and the rounding to C with Lemma 1.

## Lemma T: sums by tensor-core steps

A tensor core (a matrix unit that adds a block of products in one operation)
does not round each addition to nearest. It adds the block and the
accumulator at once, after aligning them to the largest exponent and
truncating the bits that fall off. This lemma bounds sums computed that way.

**The step.** A step has parameters N (products per step), f (fraction bits
kept in the alignment) and e_f (the floor of the alignment exponent). Its
inputs are values of a format I (float16, bfloat16, or float32 values that
are tf32 values), with smallest normal exponent emin_I, and its accumulator
is float32 (p = 24, emin = -126). For a value x of a format, e(x) is the
exponent of its encoding: floor(log2 |x|) for a normal value, the format's
emin for a subnormal one. So 2**e(x) <= max(|x|, 2**emin).

A step takes the accumulator c, a float32 value, and m <= N products
t_i = x_i y_i of inputs:

1. E is the largest of e_f, e(c) (if c != 0) and e(x_i) + e(y_i) (for each
   t_i != 0).
2. c and each t_i are truncated to a multiple of 2**(E - f).
3. The truncated values are added exactly.
4. The sum is truncated to float32, to infinity from 2**128 on.

This is the model of Khattak and Mikaitis ("Accurate Models of NVIDIA Tensor
Cores", arXiv 2512.07004), whose parameters are measured per GPU and input
format. The bound uses only that the truncations in steps 2 and 4 err by less
than one unit of their grids, in either direction, and that step 3 is exact.

**One step.** Let v = c + sum t_i, mu = |c| + sum |t_i|, and sigma >= 0 with

    2**(e(x_i) + e(y_i)) <= |t_i| + sigma

for every t_i of the step. sigma = 0 suffices when no input is subnormal.
Otherwise 2**e(x) <= |x| + 2**emin_I gives sigma = 2**emin_I Y + 2**emin_I X
+ 2**(2 emin_I), where:
- the term 2**emin_I Y counts only where some x may be subnormal, with Y at
  least every |y|;
- the term 2**emin_I X likewise for y, with X at least every |x|;
- the term 2**(2 emin_I) counts only where both may be.

With rho = 2**-f (1 + 2**-23) and tau = 2**-23, if the step's output s is
finite, then

    |s - v| <= ((m + 1) rho + tau) mu + h,
    h = (N + 1) rho (sigma + 2**e_f + 2**-126) + 2**-149.

If f >= 23 and e_f - f <= -149, a step whose products are all zero leaves c
unchanged.

*Proof.* 2**e(c) <= |c| + 2**-126 for c != 0, so the largest of the step's
2**e values is at most mu + sigma + 2**-126, and 2**E <= mu + sigma + 2**-126
+ 2**e_f. At most m + 1 values are nonzero, and each truncation in step 2
moves its value by less than 2**(E - f). So the exact sum S of step 3 has

    |S - v| < A = (m + 1) 2**(E - f) <= (m + 1) 2**-f (mu + sigma + 2**e_f + 2**-126).

Step 4 moves S by less than one float32 ulp of S, which is at most
2**-23 |S| + 2**-149, and |S| <= mu + A. Adding the two:

    |s - v| <= A (1 + 2**-23) + 2**-23 mu + 2**-149,

which is the bound, since m + 1 <= N + 1 in its absolute part.

With all products zero, E = max(e_f, e(c)). For E = e(c), c is a multiple of
2**(e(c) - 23), so of 2**(E - f) when f >= 23. For E = e_f > e(c), c is a
multiple of 2**-149, so of 2**(E - f) when e_f - f <= -149. Either way steps
2 and 4 leave c unchanged. []

**Tensor-core kernels.** An implementation computes each output's sum of K
products as follows, with g the caller's `split`:
- The products are split into s >= 1 pieces of at least one product each,
  and every piece but one holds a multiple of g products. CUTLASS, for
  example, splits K on multiples of 128 / (element bits): 8 for 16-bit
  inputs (`params_universal_base.h`).
- Each piece of K_p products is summed by ceil(K_p / N) steps, each step's
  output the next one's c: every step but the piece's last takes N
  products, as a tensor-core instruction takes a fixed number of them (a
  kernel's zero padding counts among them; a zero product changes no step).
  The first c is 0, or the float32 result of another piece (a continuation,
  as a kernel that reloads a partial sum into its accumulator does). A
  kernel whose steps take fewer products is outside this model: 64 products
  in 64 steps of one can exceed the bound below for N = 16.
- A piece result that is not continued is optionally rounded to nearest in a
  format R, the reduction format, as split-K reductions in a lower precision
  do.
- These results are added in float32, each addition rounded to nearest, in
  any order. These are split-K and stream-K reductions, and atomic
  additions.

Let r = ceil(K / N) where g is a multiple of N, else ceil(K / N) +
ceil(K / g) - 1. Let r_all = ceil(K / N) + ceil(K / g) - 1, and M = sum |t_i|,
with sigma for all the products. If the computed sum s' is finite,

    |s' - s| <= G M + H,
    G = (1 + rho)**(K + r) (1 + tau)**r (1 + u_R) - 1,
    H = (G + 1) (r_all h + s eta_R),

where u_R = 2**-p_R and eta_R = 2**(emin_R - p_R) for R (both 0 without a
reduction format).

*Proof.* The operations form a tree. Its leaves are the products; its nodes
are the steps (a step's children are its products and its c), the
roundings to R, and the float32 additions. Write E_node for a node's
computed value minus the exact sum of its leaves, and mass_node for the sum
of |t| over its leaves. Each node's error is at most u_node times its
computed inputs' magnitudes plus h_node, where:
- a step of m products has u_node = (m + 1) rho + tau and h_node = h;
- a rounding to R has u_R and eta_R (Lemma 1);
- a float32 addition has 2**-24 <= tau and no absolute part, since a sum
  of two float32 values below 2**-126 is exact.

The computed inputs' magnitudes are at most mass_node + sum E_child, so

    E_node <= (1 + u_node) sum E_child + u_node mass_node + h_node.

Unrolled to the leaves, which are exact, each leaf's |t| is multiplied by
prod (1 + u_node) - 1 over the nodes on its path, and each h_node by
prod (1 + u) over the nodes above it. For a step,
1 + (m + 1) rho + tau <= (1 + rho)**(m + 1) (1 + tau).

A leaf's path holds at most one rounding to R, some a additions, and the
steps of the pieces continued into one another that hold the leaf, from
the leaf's own step on.
- Their products are distinct, so the steps' m add up to at most K, and
  their (m + 1) to at most K plus the number of steps.
- Each addition's 1 + 2**-24 is at most 1 + tau.

So the factor is at most (1 + rho)**(K + q) (1 + tau)**q (1 + u_R), where
q is the path's steps plus its additions. It remains to show q <= r.
- The path's steps are at most the sum of ceil(K_p / N) over the c pieces
  it runs through (each piece's steps are full but its last). Its additions
  are at most s - c, one per other piece.
- Where g is a multiple of N, every piece but one holds a multiple of N
  products. So the ceil(K_p / N) of all s pieces add up to ceil(K / N).
  Each other piece holds at least one step, so q <= ceil(K / N) - (s - c)
  + (s - c) = ceil(K / N) = r.
- Otherwise each piece adds at most one partial step: the steps over the
  path's pieces are at most ceil(K / N) + c - 1. Since s <= ceil(K / g),
  q <= ceil(K / N) + s - 1 <= r.

There are at most r_all steps (ceil(K / N) + s - 1 by the same count) and s
roundings to R in all, each multiplied by at most G + 1. That gives H.

In addition, multiplication and rounding, inf and NaN never become finite
again, and a step with an infinite c or product gives inf or NaN. So when s'
is finite, no operation overflowed, and every operation obeyed its bound.
[]

**Overflow.** Each computed intermediate value lies within G M + H of the
exact sum of its leaves. That sum lies in [-Neg, Pos], where Pos and Neg are
the totals of the positive and the negative terms. A rounding can overflow
to +inf only where Pos + G M + H reaches its threshold: 2**128 for a step,
and Lemma 1's threshold to nearest for a float32 addition (below 2**128) or
a rounding to R. The same holds for -inf with Neg. A NaN can come from
adding infinities of both signs, so only where both can occur. These are
the flags. In a format without infinity, each is NaN.

**gemm.** For D = alpha (A @ B) + beta C, add these to the model:
- alpha, a float32 value, is applied once on every product's path, by a
  multiplication rounded to nearest in float32: to a piece's result, a
  partial sum, or the sum.
- beta c, rounded to nearest in float32, is added by a float32 addition
  rounded to nearest, after alpha's multiplication on that path (or fused
  with it, one rounding fewer).

Measure every value in output units: an error below alpha's multiplication
is multiplied by |alpha|. The tree gains a leaf beta c, whose path has two
nodes (its rounding and its join). Each product's path gains at most two
float32 roundings: alpha's multiplication and beta c's join. With
M' = |alpha| M + |beta c|,

    |d' - d| <= G' M' + H',
    G' = (1 + rho)**(K + r) (1 + tau)**r (1 + 2**-24)**2 (1 + u_R) - 1,
    H' = (G' + 1) (max(1, |alpha|) (r_all h + s eta_R) + 2 * 2**-150),

where 2 * 2**-150 bounds the absolute parts of alpha's multiplication and
beta c's rounding. Each multiplication by alpha and each rounding of beta c
that is exact (alpha = 1 or -1; beta c a float32 value) only loosens this
bound.

The overflow flags come from two stages, each in its own units. The
products' stage (steps, float32 additions, roundings to R) is the sum above
before alpha: its flags are as above, with G M + H of that sum (without
max(1, |alpha|)) and Pos and Neg of the terms t_i. alpha's multiplication
keeps an infinity, with its sign flipped when alpha < 0, and adding the
finite beta c keeps it too, so those flags pass to the output. The
epilogue's float32 roundings (alpha's multiplication, beta c's rounding and
its join) can overflow only where the totals Pos and Neg of the terms
alpha t_i and beta c, plus G' M' + H', reach float32's threshold to nearest.
NaN where both infinities can occur; in a format without infinity, each is
NaN. (Checking the products' stage in output units would miss an overflow
that a small |alpha| scales back into range.)

This is `ops/_mma.py`'s `mma_terms` and `MMAStep` in `accumulation.py`, used by
`matmul_mma`, `gemm_mma` and the attention method `mma`.

## Lemma 3: the final rounding and the bounds

Let X be an interval that contains the implementation's value just before its
final rounding to the output format O, and let ref be the reference output.
Every output the implementation can produce is the rounding of a value y of
X. Below O's overflow threshold (Lemma 1), fl(y) is finite and lies in X' = X
widened by Lemma 1 for O; past it, fl(y) is what O encodes beyond its largest
finite value M (below), which is M itself when finite. Either way a finite
output is a value of O in [lo', hi'], where lo' is the smallest value of O at
or above X'.lo and hi' the largest at or below X'.hi, each clamped to
[-M, M]. [lo', hi'] is the finite range (`finite_range()`), and for a finite
output

    |output - ref| <= max(hi' - ref, ref - lo'),

which `finish.c`'s `final_bounds` rounds up to float64, from the exact
differences: the tolerance. This is why it is snapped to representable
outputs. Where X reaches past the value that rounds to M, an output may be
what O encodes beyond M (`FloatFormat.encode` with the operation's
`overflow`):

- in a format with infinity, an infinity where IEEE 754 overflows (to
  nearest, or rounding outward), and M toward zero or inward;
- in a format without infinity, by default the value of O the rounding
  reaches: M to nearest, toward zero or inward, and NaN rounding outward,
  where O has no value. `overflow="nan"` gives NaN to nearest too (the IEEE
  rule with NaN for the missing infinity), and `"saturate"` gives M always.

An infinity or NaN sets its flag (`may_be()`), and M joins the range.
Infinities and NaN that reach the final rounding from earlier steps (Lemmas O
and A, Results as inputs, attention's masked keys) map through O the same
way: an infinity gives NaN in a format without one (M with `"saturate"`). The compute format's roundings
(Lemmas O and A, Lemma F's cases, the normal answer, algorithms) keep the IEEE rule with NaN
for the missing infinity, so a finite output still means nothing overflowed
on the way.

Complex outputs get one bound per part: the real and the imaginary part each
have their own X, range and flags, and the tolerance is complex128, its parts
the two bounds.

If the reference is NaN or infinite, there is no distance to bound: the
tolerance is NaN and the reference's own value is a possible output. Where
its exact value is finite (it rounded past M), [lo', hi'] still gives the
finite outputs that can occur, and is empty when every value of X rounds past
M. If no finite enclosure exists, the tolerance is infinite, the range holds
every finite value and every flag is set.

### A finite part next to a non-finite input

Any term with a non-finite factor is non-finite (inf * 0 = NaN, inf * x = inf
for x != 0), and a sum with a non-finite term is non-finite. So every term of
a part whose exact value is finite is finite, apart from the terms the
coefficient rule leaves out (`matmul.md`, The reference output). The bounds'
integer quantities read non-finite entries as 0, so a left-out term counts as
an exact 0, and an extra exact 0 only loosens Lemma S's bound. Such a part
gets the interval its quantities give, as an element with finite inputs does.

### Finite outputs and overflow

In addition and multiplication, inf and NaN never become finite again. So an
implementation whose output is finite had no rounding that overflowed, and a
bound that assumes no overflow holds for that output even where another
implementation, or another order, would overflow. The tolerance therefore
bounds the finite outputs, and the flags hold the overflows the standard's own
implementations can produce (Lemmas O and A).

The flags can be wider than needed, never narrower. Where an overflow check
fails, the flags of the sides it fails on come from the error bound, which does
not know which orders the standard allows. For example, the float32 sum of
[M, M, -M] (M the largest finite value) gets the +inf flag, although every
allowed order gives M: the only order that overflows adds M + M first, which
both standards exclude.

## The exact output

`exact` stores each element as a midpoint m with a 128-bit significand and a
radius r, and the true value lies in [m - r, m + r] (`exact.py`). Write unit(m)
for one unit in the 128th bit of m.

- An exactly known value that fits in 128 bits is stored with r = 0. A longer one
  is rounded to nearest, so |m - value| <= unit(m) / 2, and r = unit(m) / 2.
- A value known only to lie in a dyadic interval [lo, hi] is stored with m the
  center rounded to 128 bits and r the smallest power-of-two multiple of
  unit(m), from 2**-62 to 1 unit, that is at least max(hi - m, m - lo). So
  [m - r, m + r] contains [lo, hi]. If no allowed radius is large enough, the
  operation refines further.
- An interval that contains zero is stored as "near zero" with
  r = max(|lo|, |hi|) rounded up: the value lies in [-r, r]. A radius below
  2**-LIMIT is raised to 2**-LIMIT, which still contains [lo, hi].
- Stored magnitudes lie between 2**-LIMIT and 2**LIMIT (LIMIT = 2**20). A
  nonzero value, or an interval of one sign, wholly at or above 2**LIMIT in
  magnitude is stored as an infinity with its sign, marked out of range; one
  wholly below 2**-LIMIT as a zero with its sign, marked out of range. Either
  mark states that the value is finite, nonzero, of that sign and beyond that
  bound. An interval that crosses one of these bounds is refined further.
  `FloatFormat` accepts only formats with emax + 1 <= LIMIT and smallest
  subnormal at least 2**(1 - LIMIT), so a value beyond either bound rounds to
  every accepted format by its sign and the rounding direction alone.

The operations' proofs show that the values or intervals they store this way
contain the true value.

### Deciding a rounding by a sign

Some values lie closer to a point where rounding to F changes than an
enclosure at the precision cap can resolve, for example 1 + e**-30000. Their
rounding follows from the sign of their distance to that point
(`entries.c`'s `offset_rounding`).

**Grid points.** Call b a grid point of F if b = 0 or b is a multiple of
ulp(b) / 2. Every point where rounding to F changes, in any rounding mode, is
a grid point: zero (where the sign of a zero result changes), F's values, the
midpoints of neighbouring values, and the overflow threshold (the midpoint of
L and the next multiple of ulp(L)). Near a grid point b, the other grid points
are multiples of ulp(b) / 2, or of ulp(b) / 4 just below a power of two, so
none lies within eps = ulp(b) / 8 of b (eps = 2**(emin - p - 2) for b = 0).

**Claim.** Let x = c + f, with c exact and f in a dyadic interval
[f_lo, f_hi].

1. If lo <= c + f_lo and hi >= c + f_hi (the ends rounded outward) round to
   the same encoding, x rounds to it.
2. If c is a grid point, f_lo > 0, hi >= c + f_hi, and c + eps rounds to the
   same encoding as hi, x rounds to it. Likewise if f_hi < 0, lo <= c + f_lo,
   and c - eps rounds as lo does.

*Proof.* Rounding is monotone and keeps the sign of a nonzero value. (1) Every
value in [lo, hi] rounds between the equal roundings of its ends. (2) On
[c + eps, hi] the same holds. On (c, c + eps) no grid point lies, so rounding
is constant there and equal to that of c + eps. So every value in (c, hi],
x included, rounds to that encoding. []

The entry stores [lo, hi] in case 1. In case 2 it stores [f_lo, f_hi] when
c = 0, and [c, hi] or [lo, c] otherwise. Each contains x.

**Tiny offsets.** Every grid point of a format `FloatFormat` accepts is a
multiple of 2**-LIMIT (The exact output): its values are multiples of its
smallest subnormal, at least 2**(1 - LIMIT), so their midpoints and the
overflow threshold are multiples of half that. So for a grid point c and
0 < f < d = 2**-(LIMIT + 1), no grid point lies in (c, c + d], and c + f
rounds as c + d in every accepted format and rounding mode; likewise below c.
Where an operation knows the sign of f and |f| < d, but not f itself (its
exponent past what it holds), case 2 decides x with f taken as the point d of
f's sign: hi >= c + d >= c + f, and the entry stores [c, hi] or [lo, c], or,
for c = 0, d of f's sign, wholly below 2**-LIMIT and so stored as a zero of
that sign marked out of range, which holds for x as well.

## Directed dyadic arithmetic

A dyadic is m * 2**e with integers m and e; top(x) is the integer with
2**(top - 1) <= |x| < 2**top. The rounding operations of `csrc/bigfloat.c`
(`dyad_rnd`, `dyad_add`, `dyad_mul`, `dyad_div`, `dyad_sqrt`, `dyad_power`,
`dyad_to_fixed`) each round an exact result once, to at most p significant
bits, toward -inf for a lower end and toward +inf for an upper end:
`dyad_rnd` shifts the exact integer; `dyad_add` aligns its operands exactly
unless the smaller lies below 2**unit, unit = min(e_x, top(x) - p - 2), where
it rounds x + sign(y) 2**(unit - 1) instead, which rounds the same way (every
p-bit value near x is a multiple of 2**unit); `dyad_div` and `dyad_sqrt`
compute p + 2 quotient or root bits and whether a remainder is left. A
directed rounding of y to p bits moves it by less than one unit of its p-th
bit, so by less than 2**(1 - p) |y|.

Taken as given, and checked by the tests against MPFR's correctly rounded
results in both directions: these operations round their exact results so,
on the exact integer arithmetic of `csrc/natural.c` (adapted from LibBF:
sums, products, quotients with remainder, and square roots with remainder,
of natural numbers). The interval operations built on them
(`bigfloat.c`'s `ivl_add`, `ivl_sub`, `ivl_mul`, `ivl_div`, `ivl_max`)
round each end outward, so an interval contains every value of its formula
over its operands' intervals (the inclusion property).

### Lemma Q: ln 2

For p >= 2, `ln2(p)` (`ln2_bounds`) returns dyadics lo <= ln 2 <= hi of at
most p bits with hi - lo <= 2**(2 - p) ln 2.

*Proof.* ln 2 = 2 atanh(1/3) = sum over j >= 0 of 2 / ((2j + 1) 3**(2j + 1)).
For a precision W >= 17 let c = ceil(W / 1.5849) and N = floor(c / 2) + 1, so
2N + 1 > c and (2N + 1) log2 3 > W, that is 3**(2N + 1) > 2**W; also
2N + 1 <= c + 3 <= W. With a_j = floor(2**(W + 1) / 3**(2j + 1)) and
b_j = ceil(2**(W + 1) / 3**(2j + 1)) (obtained from a_0 and b_0 by floor and
ceiling division by 9, since floor(floor(u) / 9) = floor(u / 9) and likewise
for the ceiling), the sums L = sum over j < N of floor(a_j / (2j + 1)) and
U = sum over j < N of ceil(b_j / (2j + 1)) satisfy L <= 2**W S_N <= U, S_N the
sum of the first N terms. The terms are positive, so ln 2 >= S_N >= L 2**-W.
The rest is at most (2 / (2N + 1)) 3**-(2N + 1) times the sum of 9**-i, that
is 9 / (4 (2N + 1)) 3**-(2N + 1) <= 3**-(2N + 1) < 2**-W, so
ln 2 <= (U + 1) 2**-W. Since b_j <= a_j + 1, each ceiling exceeds its floor by
at most 2, so U + 1 - L <= 2N + 1 <= W, and [L, U + 1] 2**-W is less than
W 2**-W wide.

For p <= 4096, the enclosure is this one at W = 4160, less than 2**-4147
wide, computed once with exact integers and stored
(`LN2_LO`, `LN2_HI` in `functions.c`); the tests check it against MPFR's ln 2
rounded down and up. For larger p, the code sums the series at
W = p + 8 + b, b the bit length of p + 72; then W < 2**b and the enclosure is
less than 2**-(p + 8) wide. Rounding lo down and hi up to p bits moves each by
less than 2**-p (ln 2 lies in [1/2, 1)), so hi - lo < 2**-(p + 8) + 2**(1 - p)
<= 2**(2 - p) ln 2. []

### Lemma X: exp

For a dyadic x and p >= 2, `exp_bounds(x, p)` returns dyadics
lo <= exp(x) <= hi; for p >= 3, hi - lo <= 2**(3 - p) lo. The C
(`exp_bounds`) returns (lo', hi', n) with lo = lo' 2**n, hi = hi' 2**n and n
an integer of any size, lo' and hi' of at most p bits.

1. If x = 0, lo = hi = 1. If top(x) <= -(p + 4), so |x| < 2**-(p + 4): for
   x > 0, 1 < exp(x) <= 1 + x + x**2 < 1 + 2**-(p + 2), and lo = 1, hi the
   last bound rounded up to p bits (1 + 2**(1 - p)); for x < 0,
   1 > exp(x) >= 1 + x > 1 - 2**-(p + 3), and hi = 1, lo the last bound
   rounded down to p bits (1 - 2**-p). Both widths are within 2**(1 - p).
2. Otherwise let T = max(0, top(x)), K = floor(sqrt(p)) + 1 (at least 2) and
   F = p + K + 26. Lemma Q at W = F + T + 8 bits gives l_lo <= ln 2 <= l_hi
   with l_hi - l_lo < 2**(2 - W). Let q be x / l_lo rounded down to T + 16
   bits and n = floor(q + 1/2); |q - x / l_lo| < 2**-14, since |q| < 2**(T + 1),
   so n is within 1/2 + 2**-14 of x / l_lo.
3. t = x - n ln 2 lies in [t_lo, t_hi], the exact values x - n l_hi and
   x - n l_lo, ordered by the sign of n. Then |t| <= l_lo (1/2 + 2**-14) +
   |n| (l_hi - l_lo) < 0.35 + 2**-(F + 2), and t_hi - t_lo =
   |n| (l_hi - l_lo) < 2**(T + 2) 2**(2 - W) = 2**-(F + 2). (The C checks
   |t_lo|, |t_hi| < 1 and reports an internal failure otherwise.)
4. For each end, s = t_end / 2**K, so a = |s| < 2**-K <= 1/4. By Taylor's
   theorem exp(a) = sum over j < N of a**j / j! + R with
   0 <= R <= a**N e**a / N! <= 2**(1 - K N) / N!. N is the least integer with
   K N + sum over 2 <= j <= N of floor(log2 j) >= F + 2, and
   N! >= 2**(that sum), so R <= 2**-(F + 1). The sum is taken in fixed point
   with F fractional bits, on one side: for a lower bound, A = floor(a 2**F),
   u_0 = 2**F and u_j = floor(u_(j - 1) A / (j 2**F)); for an upper bound the
   same with ceilings, plus one unit for R. All terms are positive, so by
   induction u_j <= 2**F a**j / j! on the lower side (and >= on the upper),
   and the sum bounds 2**F exp(a) from that side. For s >= 0 that is the
   side wanted; for s < 0, exp(s) = 1 / exp(a), and the bound wanted is
   floor(2**(2F) / U) from the upper sum U, or ceil(2**(2F) / L) from the
   lower sum L.
5. Squaring K times, E to floor(E**2 / 2**F) for a lower bound and to
   ceil(E**2 / 2**F) for an upper one (E > 0), gives a bound of
   2**F exp(s)**(2**K) = 2**F exp(t_end) from the same side.
6. exp is increasing and exp(x) = 2**n exp(t) with t in [t_lo, t_hi], so
   exp(x) lies between 2**n times the lower bound computed at t_lo and 2**n
   times the upper bound computed at t_hi (each times 2**-F); these are
   rounded outward to p bits.

*Proof.* Each step bounds its value from its side: step 1 by the stated
inequalities, and steps 2 to 6 by Lemma Q, the exactness of t's ends,
Taylor's bound, induction on the positive terms (a floor of a product of
lower bounds of positive numbers is a lower bound, and likewise for
ceilings), and the monotonicity of 1/y, of squaring on positive numbers and
of exp.

For the width, values are counted in units of 2**-F. Each term of step 4 lies
within at most 4 units of 2**F a**j / j! (its distance shrinks by a / j <= 1/4
per term, and each term adds at most one unit for its floor or ceiling and
one for A's), so each sum lies within 4N + 1 units of 2**F exp(a), and the
reciprocal adds one unit; with exp(s) > 0.7, the logarithm of each end's
ratio to its exact value is at most rho_0 <= (4N + 3) 2**(1 - F). A squaring
doubles rho and its floor or ceiling adds at most 2**(1 - F) (the squares stay
above 0.7 2**F), so after K squarings rho_K <= 2**K (rho_0 + 2**(1 - F)) <=
2**(K + bits(N) + 4 - F) <= 2**-(p + 6), since bits(N) <= 16 for the
precisions used (N < 2**16). With t_hi - t_lo < 2**-(F + 2), hi / lo before the
last rounding is at most exp(2**-(F + 2) + 2**-(p + 5)) <= 1 + 2**-(p + 4). The
last rounding moves each end by less than 2**(1 - p) of itself, so for p >= 3,
hi / lo <= (1 + 2**-(p + 4)) (1 + 2**(1 - p)) / (1 - 2**(1 - p))
<= 1 + 2**(3 - p). []

### Lemma W: exp of small arguments on words

For R = r 2**-s with 2**63 <= r < 2**64 and s >= 68 (so 0 < R < 2**-4),
`exp_small` (`functions.c`) returns m < 2**64 and e with
exp(R) - 1 <= m 2**-e <= (exp(R) - 1)(1 + 2**-56), on 64-bit words:

1. rho = ceil(r 2**(64 - s)), so rho >= R 2**64 and rho < 2**60 + 1.
2. t_0 = 2**62 and, for k >= 1, t_k = ceil(ceil(t_(k - 1) rho / 2**64) / (k + 1)),
   up to the first K with t_K <= 1; S = t_0 + ... + t_K + 1.
3. r S, rounded up to 64 bits, is m 2**(e - s - 62).

*Proof.* exp(R) - 1 = R sum over k >= 0 of x_k 2**-62, with
x_k = 2**62 R**k / (k + 1)!. By induction t_k >= x_k: x_k = x_(k - 1) R / (k + 1),
t_(k - 1) rho / 2**64 >= x_(k - 1) R, and a ceiling of a product of upper
bounds of nonnegative numbers is an upper bound. The terms
after K satisfy x_(K + i) <= x_K R**i <= t_K R**i, so they sum to at most
t_K R / (1 - R) <= 1/15 < 1, the unit added to S. So S >= sum over all k of
x_k, and exp(R) - 1 <= R S 2**-62 <= r S 2**-(s + 62).

The loop ends: while t_(k - 1) >= 2, t_(k - 1) rho / 2**64 < t_(k - 1) / 16 + 1,
so t_k < t_(k - 1); and t_(k - 1) <= 3 gives t_k = 1.

Width: t_(k - 1) < 2**63 and rho <= R 2**64 + 1 give
t_(k - 1) rho / 2**64 <= t_(k - 1) R + 1, so
t_k <= (t_(k - 1) R + 2) / (k + 1) + 1, and d_k = t_k - x_k satisfies
d_k <= d_(k - 1) / 32 + 2, d_0 = 0, so d_k < 2.07. x_k < 1 for k >= 11 (since
4k + log2((k + 1)!) > 62), and then t_(k + 1) = 1, so K <= 13 and
S - sum x_k < 2.07 (K + 1) + 1 < 31 <= 2**-57 2**62. S < 2**63, so r S < 2**127
fits two words, and its rounding up to 64 bits adds less than 2**-63 of it.
With sum x_k >= 2**62, m 2**-e <= (exp(R) - 1)(1 + 2**-57)(1 + 2**-63).

A lower bound takes floors throughout: rho = floor(r 2**(64 - s)), each t_k
the floor of the floor, the loop stopped at the first t_K = 0, no unit for
the rest, and r S rounded down. By the same induction with floors of products
of lower bounds of nonnegative numbers, t_k <= x_k, and every term left out
is nonnegative, so r S 2**-(s + 62) <= exp(R) - 1. []

### Lemma W3: exp of small arguments on three words

For R = r 2**-s with 2**191 <= r < 2**192 and s >= 196 (so 0 < R < 2**-4),
`exp_small3` (`functions.c`) returns m < 2**192 and e with
exp(R) - 1 <= m 2**-e <= (exp(R) - 1)(1 + 2**-183), and with floors
throughout (as in Lemma W's lower bound) (exp(R) - 1)(1 - 2**-183) <= m 2**-e
<= exp(R) - 1. Its steps are Lemma W's with 64 replaced by 192: rho =
ceil(r 2**(192 - s)) < 2**188 + 1, t_0 = 2**190, t_k = ceil(ceil(t_(k - 1) rho /
2**192) / (k + 1)) until t_K <= 1, S = t_0 + ... + t_K + 1, and r S rounded
up to 192 bits.

*Proof.* As for Lemma W, with x_k = 2**190 R**k / (k + 1)!: t_k >= x_k by
induction, the terms after K sum to at most t_K R / (1 - R) < 1, and the loop
ends. Width: d_k = t_k - x_k <= d_(k - 1) / 16 + 2, so d_k < 2.14; x_k < 1 for
k >= 26 (4k + log2((k + 1)!) > 190), so K <= 28, and S - sum x_k <
2.14 (K + 1) + 1 < 64 = 2**-184 2**190 <= 2**-184 sum x_k. S < 2**191, so r S
< 2**383 fits six words, and its rounding to 192 bits adds less than 2**-191
of it: (1 + 2**-184)(1 + 2**-191) < 1 + 2**-183. The floors give t_k <= x_k
and the same distance, as in Lemma W. []

`exp_near_one` gives exp(x) for |x| < 2**-4 rounded down or up to p bits: |x|
rounded to 192 bits toward the side wanted (exp(|x|) - 1 is increasing), Lemma
W3 on it, then 1 plus that, exactly, rounded to p bits for x > 0, or 1 over it,
rounded (`dyad_div`) to the other side's end for x < 0, since exp(x) =
1 / exp(|x|). An |x| that rounds to 2**-4 is left to Lemma X.

### Lemma M: pi

For p >= 2, `pi_bounds(p)` (`functions.c`) returns dyadics lo <= pi <= hi of
at most p bits with hi - lo <= 2**(4 - p).

*Proof.* By Machin's formula pi = 16 atan(1/5) - 4 atan(1/239), with
atan(1/k) = sum over j >= 0 of (-1)**j / ((2j + 1) k**(2j + 1)), a series
whose terms alternate in sign and decrease, so the rest after any number of
terms is at most the first term left out.

Let b be the bit length of p and W = p + 16 + b. For k = 5 and k = 239, with
d_5 = 23219 and d_239 = 79009 (each below 10000 log2 k),
c_k = ceil((W + 2) 10000 / d_k) and N_k = floor(c_k / 2) + 1: then
2 N_k + 1 > c_k, so k**(2 N_k + 1) > 2**(W + 2), and the rest after N_k terms
is below 2**-(W + 2). With a_j = floor(2**W / k**(2j + 1)) and
b_j = ceil(2**W / k**(2j + 1)) (from a_0 and b_0 by floor and ceiling division
by k**2, as in Lemma Q), let L_k be the sum over even j < N_k of
floor(a_j / (2j + 1)) less the sum over odd j < N_k of ceil(b_j / (2j + 1)),
and U_k the same with the ceilings and floors exchanged. Each term is bounded
from its side, so L_k <= 2**W S_k <= U_k for the sum S_k of the first N_k
terms, and 2**W atan(1/k) lies in [L_k - 1, U_k + 1]. So pi lies in

    [16 (L_5 - 1) - 4 (U_239 + 1), 16 (U_5 + 1) - 4 (L_239 - 1)] 2**-W,

and lo and hi are its ends rounded down and up to p bits.

For the width: b_j <= a_j + 1, so each ceiling exceeds its floor by at most
2, U_k - L_k <= 2 N_k, and the enclosure is at most 32 N_5 + 8 N_239 + 40
units of 2**-W wide. N_k <= c_k / 2 + 1 <= (W + 2) 5000 / d_k + 3/2, so
N_5 <= W / 4.64 + 2 and N_239 <= W / 15.8 + 2, and the enclosure is less than
(7.5 W + 120) 2**-W < 2**-(p + 2) wide (2**b > p). pi < 4, so each rounding to
p bits moves an end by less than 2**(2 - p), and
hi - lo < 2**-(p + 2) + 2**(3 - p) <= 2**(4 - p). []

### Lemma Z: cos and sin

For a dyadic x and p >= 2, `cos_sin_bounds(x, p)` (`functions.c`) returns
dyadics c_lo <= cos x <= c_hi and s_lo <= sin x <= s_hi of at most p bits,
with c_hi - c_lo <= 2**(3 - p) and s_hi - s_lo <= 2**(4 - p) min(1, |x|).

1. If x = 0, cos x = 1 and sin x = 0 exactly. If top(x) <= -(p + 4), so
   |x| < 2**-(p + 4): 1 >= cos x >= 1 - x**2 / 2 > 1 - 2**-(2p + 9), and
   sin x lies between x and x - x**3 / 6, so between x and
   x (1 - 2**-(2p + 8)); these bounds are rounded outward to p bits.
2. Otherwise let b be the bit length of p, T = max(0, top(x)) and
   G = p + 16 + b + max(0, -top(x)). If top(x) <= -1, so |x| < 1/2, then n = 0
   and t_lo = t_hi = x. Otherwise Lemma M at W = G + T + 8 bits gives
   pi_lo <= pi <= pi_hi with pi_hi - pi_lo <= 2**(4 - W). Let q be 2x / pi_lo
   rounded down to T + 16 bits and n = floor(q + 1/2); |q - 2x / pi_lo| < 2**-14,
   since |q| < 2**(T + 1), so n is within 1/2 + 2**-14 of 2x / pi_lo, and
   |n| <= 2**(T + 1).
3. t = x - n pi / 2 lies in [t_lo, t_hi], the exact values x - n pi_hi / 2 and
   x - n pi_lo / 2, ordered by the sign of n. Then
   |t| <= (pi_lo / 2)(1/2 + 2**-14) + |n| (pi_hi - pi_lo) / 2 < 0.8, and
   t_hi - t_lo = |n| (pi_hi - pi_lo) / 2 <= 2**(T + 4 - W) = 2**-(G + 4). (The C
   checks |t_lo|, |t_hi| < 1 and reports an internal failure otherwise.)
4. cos and sin of t0 = t_lo by their Taylor series, in fixed point with G
   fractional bits: with A_lo = floor(|t0| 2**G) and A_hi = ceil(|t0| 2**G),
   l_0 = h_0 = 2**G, l_k = floor(l_(k - 1) A_lo / (k 2**G)) and
   h_k = ceil(h_(k - 1) A_hi / (k 2**G)) bound u_k = 2**G |t0|**k / k! from
   below and above. |t0| < 1, so the u_k decrease, and
   cos t0 = (u_0 - u_2 + u_4 - ...) 2**-G and
   sin t0 = sign(t0) (u_1 - u_3 + u_5 - ...) 2**-G are series whose terms
   alternate in sign and decrease: each partial sum is within its first term
   left out of the value. The terms run through k = K, the first with
   h_K <= 1, so every term left out is below one unit. Each series' lower
   bound adds l_k for its terms of sign + and subtracts h_k for those of
   sign -, then subtracts one unit; its upper bound the reverse, then adds one
   unit. For t0 < 0, sin's bounds are those of the series negated and
   exchanged.
5. By the quadrant n mod 4, (cos x, sin x) is (cos t, sin t), (-sin t, cos t),
   (-cos t, -sin t) or (sin t, -cos t) for n = 0, 1, 2 or 3 modulo 4.
6. Every t in [t_lo, t_hi] is within t_hi - t_lo of t0, and cos and sin have
   derivatives of magnitude at most 1, so each enclosure moves t_hi - t_lo
   outward; then its ends are rounded outward to p bits.

*Proof.* Step 1 is Taylor's theorem with |x| < 2**-(p + 4). Steps 2 and 3
enclose t exactly at pi's ends; step 4 bounds each term from its side by
induction (a floor of a product of lower bounds of nonnegative numbers is a
lower bound, and a ceiling of upper bounds an upper bound) and the rest by
the alternating series' bound; steps 5 and 6 are the identities for
cos(t + n pi / 2) and sin(t + n pi / 2) and the mean value theorem.

For the width, in units of 2**-G, let e_k = u_k - l_k and d_k = h_k - u_k.
e_0 = d_0 = 0 and e_1, d_1 < 1 (l_1 = A_lo, h_1 = A_hi). For k >= 2,
|t0| / k < 0.4 and u_(k - 1) <= u_1 < 0.8 2**G, so A_lo >= |t0| 2**G - 1 gives
e_k < 0.4 e_(k - 1) + 0.4 + 1, and A_hi <= |t0| 2**G + 1 gives
d_k < 0.4 d_(k - 1) + (0.8 2**G + d_(k - 1)) / (2 2**G) + 1; by induction both
stay below 3. The terms number at most G: u_k < 2**G 0.4**k for k >= 4
(k! >= 2**k), so u_k < 1 for some k0 <= G / 1.32 + 1; then h_k0 <= 3 and
h_(k0 + 1) <= 1, so K + 1 <= G / 1.32 + 3 <= G. Each bound is then within
3 G + 2 units of 2**G times its value, and with step 6's 2**-(G + 4) each
enclosure is less than (6 G + 5) 2**-G wide before the last rounding. For
top(x) >= 0, G = p + 16 + b and 6 G + 5 < 2**(16 + b), so that is below
2**-p. For -(p + 3) <= top(x) <= -1, G <= 2 p + 19 + b and
2**-G <= 2 |x| 2**-(p + 16 + b), and 2 (6 G + 5) < 2**(16 + b), so it is below
|x| 2**-p.

Each value lies in [-1, 1] and each end within less than 2 in magnitude, so
a rounding to p bits moves an end by less than 2**(1 - p): each enclosure is
less than 2**-p + 2**(2 - p) <= 2**(3 - p) wide. For sin with |x| < 1/2, its
ends are within |x| (1 + 2**-p) in magnitude, so the roundings move them by
less than 2**(1 - p) |x| (1 + 2**-p) each, and the enclosure is less than
2**(3 - p) |x| wide; in step 1 likewise, x 2**-(2p + 8) plus the roundings
is less than 2**(3 - p) |x|, and cos's enclosure is [1 - 2**-p, 1]. For
|x| >= 1/2, 2**(3 - p) <= 2**(4 - p) min(1, |x|). []

Over an interval: for dyadics lo <= hi, `cos_sin_range(lo, hi, p)`
(`functions.c`) returns enclosures of cos and sin over [lo, hi]. If
hi - lo >= 7 (more than 2 pi, a whole period), both are [-1, 1]. Otherwise
each is the hull of Lemma Z's enclosures at lo and at hi, with 1 or -1 added
for each integer j with u_lo <= j <= u_hi, where u_lo and u_hi are 2 lo / pi
and 2 hi / pi rounded down and up to p + 8 + max(0, top(lo), top(hi)) bits
(with Lemma M's enclosure of pi at as many bits, its end chosen by the sign
of lo and of hi): cos's gets 1 for j = 0 and -1
for j = 2 modulo 4, sin's 1 for j = 1 and -1 for j = 3. *Proof.* sin' = cos
vanishes exactly at the odd multiples j pi / 2, where sin is 1 or -1 by
j modulo 4, and cos' = -sin at the even ones, where cos is 1 or -1. Between
consecutive critical points each is monotone, so on [lo, hi] its extremes
lie at lo, at hi or at a critical point inside, and every critical point
x = j pi / 2 inside has j = 2x / pi in [u_lo, u_hi]. []

### Lemma Y: log

For a dyadic x > 0 and p >= 2, `log_bounds(x, p)` returns dyadics
lo <= log x <= hi of at most p bits; for p >= 3 and x != 1, hi - lo <=
2**(3 - p) |log x| (lo = hi = 0 for x = 1).

1. Write x = m 2**e with integers m > 0 and e, and b for m's bit length. If
   m's two top bits are set, y = m 2**-b, in [3/4, 1), and k = e + b;
   otherwise y = m 2**-(b - 1), in [1, 3/2), and k = e + b - 1. So
   x = y 2**k exactly, and log x = log y + k ln 2.
2. d = y - 1 and y + 1 are exact. If d = 0, log y = 0. Otherwise
   z = d / (y + 1) lies in [-1/7, 1/5), log y = 2 atanh z, and z_lo <= z <= z_hi
   are the quotient rounded down and up to w = p + 16 bits.
3. atanh is increasing and odd, so log y lies in [2 atanh(z_lo),
   2 atanh(z_hi)], and atanh(zeta) = -atanh(|zeta|) for zeta < 0. For a
   dyadic a = |zeta| < 1/4 (both ends are below it), atanh a = sum over
   j >= 0 of a**(2j + 1) / (2j + 1), positive terms; with N = floor((G + 3) / 4)
   + 1 terms, G = w - top(a) + 8, the rest is at most
   a**(2N + 1) / ((2N + 1)(1 - a**2)) <= a 16**-N <= 2**-(G + 6). The sum is
   taken in fixed point with G fractional bits on one side, as in Lemma X:
   A = floor(a 2**G), A2 = floor(A**2 / 2**G), P_0 = A,
   P_j = floor(P_(j - 1) A2 / 2**G) and the terms floor(P_j / (2j + 1)) for a
   lower bound; ceilings and one unit for the rest for an upper bound. A
   lower bound of atanh(zeta) for zeta < 0 is minus an upper bound of
   atanh a, and conversely; doubling is exact.
4. If k != 0, Lemma Q at W = w + bits(k) + 8 gives l_lo <= ln 2 <= l_hi;
   k ln 2 lies in [k l_lo, k l_hi] for k > 0 and [k l_hi, k l_lo] for k < 0,
   the products exact, and the sums with log y's ends are rounded outward to
   w + bits(k) bits.
5. Both ends are rounded outward to p bits.

*Proof.* Each step bounds its value from its side by exactness, the
monotonicity and oddness of atanh, induction on the positive terms, the bound
on the rest of the series, and Lemma Q.

For the width with x != 1: the ends of z lie within 2**(1 - w) |z| of z. In
step 3, a 2**G >= 2**(w + 7), each term lies within 4 units of its exact
value (the distance shrinks by a**2 < 1/16 per term), so each sum lies within
4N + 2 units, less than 2**(bits(N) + 3 - w - 7) |atanh a| relatively. So
each end of log y lies within 2**(3 - w) |log y| of it (atanh(z_end) differs
from atanh(z) by less than 2**(2 - w) of it). For k = 0, log x = log y. For
k != 0, |log x| >= ln 2 - ln(3/2) > 0.28 and |log y| < 0.41, k ln 2's
enclosure is narrower than 2**bits(k) 2**(2 - W) = 2**-(w + 6), and the
sum's rounding adds less than 2**(1 - w) |log x|; so the ends lie within
2**(6 - w) |log x| of log x. The last rounding moves each end by less than
2**(1 - p) of itself, so for p >= 3 the width is at most 2**(3 - p) |log x|.
[]

## The normal standard

Normal accumulation is how ordinary kernels add up a sum. It adds the terms
in index order, or in 2 or 4 SIMD-style lanes (lane i adds the terms
i, i + L, i + 2L, ... in order, then the lanes are combined), in the
accumulator format A, rounding each product and then adding it. Where adding
the next value would overflow, it starts a new partial sum (Lemma O). Its
index-order output is the normal answer.

**Units.** A has precision p, smallest normal value 2**emin and largest
finite value max_A. Every accumulator here rounds to nearest (ties to even
or away): the tolerance model's compute roundings, and every named
algorithm's. A directed rounding is not modeled: past max_A it saturates on
its inward side and toward zero, with an error up to |y| - max_A that no
bound r |y| + a covers, and a saturated partial sum can move the result by
up to the size of the terms. A rounding to nearest moves y by at most
r |y| + a, by Lemma 1: r = 2**-p, and a = 2**(emin - p) (a subnormal result),
or a = 2**emin with flush to zero (a subnormal result may become 0). An
addition of two numbers of A has no absolute part without flush (a subnormal
sum is exact); write a' for its absolute part: 0 without flush, 2**emin with
flush. The overflow threshold Theta_A is the smallest magnitude that can
round to infinity (or NaN, in a format without infinity): max_A plus half an
ulp of max_A (`formats.c`'s `floatfmt_threshold`). The growth factor is
Growth(D) = (1 + r)**D, evaluated rounded up (`errors.c`'s `growth_up`): it
is valid for every D, unlike gamma(D), which needs D r < 1. Units per setting
are computed in `sums.c`'s `sum_bounds_init` and by `attention_tol.c`'s
`acc_units`.

**Notation.** One sum has terms t_1 .. t_n in index order, running sums
s_k = t_1 + ... + t_k, the exact sum s = s_n and mass = sum_k |t_k|. With flush,
moved is the sum of |t_k| over the terms with a factor subnormal in the
compute format (it may be read as zero).

### Lemma S: summation trees

**Statement.** An implementation computes s by a binary tree. Each leaf is a
term: rounded to A, or exact where a fused multiply-add takes it, or 0 where a
flush reads one of its factors as zero. Each internal node adds the computed
values of its two children and rounds once. Each node covers a set of terms;
v_node is the exact sum of those terms. D is the most additions on any path
from a leaf to the root (at most n - 1). If no rounding overflows, the
computed sum s' satisfies

    |s' - s| <= Growth(D) * ( r * (mass + sum over internal nodes |v_node|) + n a + moved + (n - 1) a' ).

**Proof.** A rounded leaf's error is at most r |t_k| + a; a fused leaf has
none; a flushed leaf's error is |t_k|, which moved counts. For an internal
node whose children have errors E_L and E_R (computed value minus exact
value), the node computes fl(L' + R') = L' + R' + d with

    |d| <= r |L' + R'| + a' <= r (|v_node| + E_L + E_R) + a',

so E_node <= (1 + r)(E_L + E_R) + r |v_node| + a'. (A node that merges an
exact term by a fused multiply-add has absolute part a instead of a'; the
term's a is then counted at that node instead of at its leaf, so n a still
covers it.) Unrolled down to the leaves, every leaf's error and every node's
r |v_node| + a' is multiplied by (1 + r) once per addition above it, at most D
times. []

If every term is exactly 0 (mass = 0), every leaf and every addition is exact
and s' = s = 0. A term that differs from a reference term by at most rad
(interval terms, as in the attention backward) has leaf error at most
r |t_k| + a + rad: the same proof holds with v_node the sums of the reference
terms, mass the sum of the actual terms' magnitudes, and the radii added to
moved.

**Never worse than Lemma 2.** Each |v_node| is at most the mass, so the sum
over the n - 1 internal nodes is at most (n - 1) mass, and the first-order part
is at most Growth(n - 1) n r mass <= gamma(n) mass when n r < 1.

### Lemma B: running sums from block products

Index order's internal nodes are the running sums s_2 .. s_n. Split the terms
after the first into blocks of consecutive terms. For a block j0 < k <= j1
with mass M (the sum of |t_i| for j0 < i <= j1), each running sum inside it
satisfies |s_k| <= |s_j0| + (sum of |t_i| for j0 < i <= k) and
|s_k| <= |s_j1| + (sum of |t_i| for k < i <= j1); the smaller of two numbers is
at most their average, so |s_k| <= (|s_j0| + |s_j1| + M) / 2 and

    sum over j0 < k <= j1 of |s_k| <= (j1 - j0 - 1)(|s_j0| + |s_j1| + M) / 2 + |s_j1|.

For a matrix product, the running sums at block ends are exact integer matrix
products of column blocks of A with row blocks of B, added up block by block.

### Lemma L: lanes

With L lanes, lane i adds the terms i, i + L, i + 2L, ... in order, then the L
lane sums are combined: in order, pairwise (the first ceil(L/2), then the
rest), or by folding the upper half onto the lower ((T_0 + T_2), (T_1 + T_3),
then their sum). This is a summation tree, so Lemma S applies: its internal
nodes are each lane's running sums from its second term on and the
combination's nodes. Each lane is a chain, so Lemma B bounds its running sums
from its sums at block ends, or they are taken exactly (blocks of one term).
The lane totals are exact, so the combination's nodes are exact. A tree with n leaves has at most n - 1 additions on any path,
so Growth(n - 1) serves every order.

A term that is exactly 0 in every implementation (a masked key's, in
attention) adds exactly, x + 0 = x, so the tree with it left out, its parent
replaced by its other child, computes the same value: the bound may take that
tree over the other terms, each keeping its position's lane. With flushing,
x + 0 may give 0 for a subnormal x; x is then the value of a leaf or node of
that tree, whose rounding may flush it itself (Lemma 2, Exact zero terms), so
the tree computes the same value with that flush, which its n a and
(n - 1) a' count. Its internal
nodes are each lane's running sums from its first remaining term's successor
on, and the combination's nodes where both sides hold a remaining term (an
empty lane joins none).

### The normal bound

The normal tolerance bounds index order and lanes 2 and 4 (with every
combination above) at once: it is Lemma S with Growth(n - 1) and the largest of
their node sums. Per output element it takes, for each order, an integer
N >= 2 (mass + sum of |node|):

- The product kernel (`csrc/product.c`: matmul, gemm and the operations on
  matmul's product) and the attention kernels take every node exactly, so
  N = 2 (mass + sum of |node|). Index order's nodes are its running sums
  s_2 .. s_n; lanes L's are each lane's running sums from its second term on,
  and the combination's nodes, from the exact lane totals (the largest of the
  three combinations). This is Lemma B with blocks of one term.

The bound is then

    B = Growth(n - 1) ( r N / 2 + n a + moved + (n - 1) a' ),

computed exactly as one integer at a unit fine enough to hold the constant
part exactly (`csrc/sums.c`; the growth factor's multiple of moved and the
threshold are rounded up and down respectively).

### Lemma O: overflow-safe slots

**The algorithm.** Pass 1 goes through the rounded terms in index order with
one current partial sum: if adding the next value gives a finite rounded
result, keep it; otherwise close the current partial sum and start a new one
with the next value. Each later pass does the same with the previous pass's
partial sums, in order, until one value remains. A pass that merges nothing
ends with +inf or -inf (the sign of its values).

**(a) No inf or NaN from finite terms,** except that stopping result: a sum is
kept only when its rounded value is finite. If a rounded term itself
overflows, every implementation that rounds it overflows.

**(b) It ends.** Two values of opposite sign, or a zero, have a sum no larger
in magnitude than the larger of the two, so its rounding cannot overflow and
they merge. So a pass merges at least once, or all its values have one sign
and every neighbouring sum reaches Theta_A; then it stops. Each merge lowers
the count by one.

**(c) Finite whenever the exact sum leaves room.** Let B_any be Lemma S's
bound with every |v_node| replaced by the mass and D = n - 1 (it holds for any
tree, so for whatever the passes have merged so far). A pass that stops has
values of one sign whose total reaches Theta_A in magnitude, and that total is
within B_any of s, so |s| >= Theta_A - B_any. So if |s| + B_any < Theta_A, the
result is finite.

**(d) Its bound.** Every addition merges two values covering adjacent ranges
of terms, so the computation is a summation tree with depth at most n - 1, and
Lemma S applies.

**(e) When no slot can open.** If mass + B < Theta_A, with B the normal bound
above, no rounding of index order or of the lanes can overflow: each value
such an order rounds has magnitude at most |v_node| + E_L + E_R (or a term's
|t_k|), and |v_node| <= mass while E_L + E_R <= B (they are disjoint parts of
the unrolled sum). So index order never opens a slot, and the normal bound
holds. No normal order overflows then, so Lemma A's bound, which holds for
every order that does not overflow, holds for them too: the normal tolerance
may take the smaller of B and Lemma A's bound, the intersection of two
intervals centered on s (matmul and gemm do).

**(f) Otherwise**, the normal tolerance is Lemma A's bound, which covers
every tree that does not overflow (the slots' tree and every lane order among
them), so every finite output. By (a), normal accumulation gives inf or NaN
only where a rounded term overflows, with that term's sign (NaN where terms
of both signs can), or where its last pass stops because the final value
reaches Theta_A, on the side where s + B_any or -s + B_any does. These are
its flags. A term's values are bounded by its factors' products at their
ends, each factor anywhere within its radius where the input is an earlier
result (Results as inputs; `product_terms`).

**(g) A side's reach.** Rounding and addition are monotone, so in any tree of
roundings over the terms, replacing each negative term by 0 can only raise a
node: a positive node value is at most that node's value over the positive
terms alone. There every value is at least 0, a rounded term at most
(1 + r) t + a and an addition at most (1 + r)(x + y) + a', so every node is
at most Growth(n - 1)((1 + r) side + n a + (n - 1) a'), side the sum of the
positive terms' largest values (each factor anywhere within its radius); and
at most side + B_any, every node being within B_any of its exact partial
sum. The same holds for the negative terms. A node over one nonzero term is
that term's rounded value (adding 0 is exact), below Theta_A unless the term
itself reaches Theta_A. So one sign's finite terms reach Theta_A together only
where at least two are nonzero and both bounds reach it (`side_reaches`). This
holds for every tree, so for index order, the lanes and the slots' passes as
for any order. For a fixed order, a subtree is a range of terms, and the bound
over that range alone holds for its nodes (`order_sides`).

### Lemma A: any order without overflow

Lemma 2 requires that no partial sum can overflow. For an implementation
that does not overflow, its proof holds without that requirement, and more:
each value an addition rounds has magnitude below Theta_A, so each addition
moves it by at most r Theta_A + a'. So

    |s' - s| <= min( gamma(n) mass + n a (1 + gamma(n)) + moved,
                     Growth(n) (r mass + n a + moved + (n - 1)(r Theta_A + a')) ),

the first form Lemma 2's (with R = 2n - 1 and eta = 2**emin with flush, as
there; it needs n r < 1), the second finite even when the mass exceeds max_A
(it needs Theta_A). Such an implementation exists when |s| + B_any < Theta_A
(Lemma O (c)). The bound holds for every order that does not overflow, so it
is the any-order tolerance in every case. Where s + B_any (or -s + B_any)
reaches Theta_A, the final value may overflow on that side, which sets that
side's flag (example: float32 terms [max_f32, 2**103, -2**78], exact sum
max_f32: every order gives inf, so no finite output occurs, and the +inf flag
is set). A term that rounds past Theta_A on its own gives every
implementation that rounds it an infinity (Lemma O (a)), so its side is a flag
too. An order whose partial sums overflow while the exact value is finite and
no term overflows on its own is outside the standard and sets no flag; where
an infinity occurs anyway, the infinities rule (Results as inputs) covers what
the partial sums meet. `csrc/sums.c` computes it (`sum_bounds_any_order`).
If every term is exactly 0 (mass = 0) and a flush moves nothing, every
addition in every order is exact and s' = s = 0, as in Lemma S.

### Lemma U: exact terms

**Statement.** Let every term t_k enter the sum exactly, never rounded on its
own: a value of the accumulator's format A, as x_k * 1 is in a sum when A
holds the input format, or an exact value that the model does not round,
which A need not hold (`add`'s terms with a = 1 or -1, `add.md`). Then Lemma S
holds without its leaf errors:

    |s' - s| <= Growth(D) * ( r * sum over internal nodes |v_node| + moved + (n - 1) a'' ),

with a'' = a' where every term is a value of A, and a'' = a otherwise; and
Lemma 2 holds with D <= n - 1 and R = 0 when every term is a value of A
without flushing, R <= n - 1 otherwise.

**Proof.** Lemma S's proof bounds a rounded leaf's error by r |t_k| + a. No
term is rounded on its own here (a value of A rounds to itself in any case),
so that error is 0; a fused multiply-add that takes x_k * 1 adds x_k exactly,
which is an addition node. The rest of the proof is unchanged, except for the
absolute part of an addition: of two values of A it is a' (a subnormal sum is
exact without flushing), but an addition that takes a term that is not a
value of A rounds an exact value that need not be a multiple of A's subnormal
spacing, so its absolute part is a (Lemma 1), at most one per addition. In
Lemma 2, no term is rounded on its own, so D <= n - 1, the height of a tree
with n leaves. Every rounding is one of the n - 1 additions; its h is 0
when it adds two values of A without flushing (Lemma 2's argument), and
otherwise it counts. []

The code takes N >= 2 (mass + sum |v_node|) from the running sums and uses
N - 2 mass, which is at least 2 sum |v_node| (`csrc/sums.c`, `sum_bounds`'s
exact terms; `TERMS_EXACT` where A need not hold the terms, with a'' = a and
R = n - 1). Sums whose terms are input values use it: `sum`, and `add`
with a = 1 or -1 (`sum.md`, `add.md`).

### Lemma F: few terms

**Statement.** Let an element have exact inputs, no flushing, and at most two
nonzero exact terms t_1, t_2 in each part (a term with an exact zero factor is
exactly 0). An implementation that performs only some of the model's
roundings to the compute format C, computing the others exactly, is correct:
the reference itself performs none of them. Write r_k for fl(t_k). Every
such implementation, the model's own included, computes one of

    fl(t_1), t_1                                                (one term),
    fl(r_1 + r_2), fl(t_1 + r_2), fl(r_1 + t_2), fl(t_1 + t_2),
    r_1 + r_2, t_1 + r_2, r_1 + t_2, t_1 + t_2                  (two terms),

each fl rounding to nearest in C with ties to even or away (Units; each
rounding may take either), and then rounds it to the output format O. So the
largest distance from the reference output among these values rounded to O
is the exact worst case, and the reference, t_1 or t_1 + t_2 rounded to O, is
among them.

**Proof.** In the model, a product is rounded to C on its own or merged by a
fused multiply-add, and each addition of two values is rounded to C; an
implementation may leave out any of these roundings. A term that is exactly 0
is 0 in every implementation, and adding 0 to a value is exact. With one
nonzero term there is no addition: the only rounding is the product's,
fl(t_1), since a fused multiply-add with a zero addend rounds the same
product, or none, which leaves t_1. With two, each product enters the one
addition rounded (r_k) or exact (merged by a fused multiply-add, or its
rounding left out), and the addition is rounded or exact; addition is
commutative, so these are all the cases. Rounding to nearest with a given tie
rule is a function, so each case and choice of tie rules gives one value, and
the output is that value rounded to O. []

Where the model rounds no term on its own (Lemma U's exact terms), the
implementations' cases are those with r_k = t_k: fl(t_1 + t_2) and t_1 + t_2
for two terms, and for one, fl(t_1), the addition of t_1 and an exact 0
rounded in C, and t_1. `multiply.md` says how `multiply` and `add` apply it.

### Lemma E: the normal answer

`normal(compute=None, flush_to_zero=False)` emulates the normal standard in
index order, with the overflow-safe slots, in exactref's emulation of the
accumulator format A (`emulate.c`), for every accepted float format (at most
64 bits wide, so at most 62 bits of precision): a finite value is a sign and
sig 2**e with sig below 2**p, and each operation rounds its exact result once
to A as IEEE 754 defines it, to nearest (ties to even or away), an overflow
giving an infinity (NaN in a format without one):

- A product of two values: the significands' product is exact in two words
  (below 2**124), then rounded.
- A sum of two values, each significand moved up to p bits: when their units
  are at most p + 3 apart, the larger's significand moved up by the gap
  (below 2**(2p + 3) <= 2**127) and the smaller's added or subtracted are
  exact in two words, then rounded. Otherwise the smaller lies below an
  eighth of the larger's unit (it is below 2**p times its own unit, more than
  p + 3 bits lower): the larger with three guard bits (p + 3 <= 65 bits, in
  two words), less one guard unit when the signs differ, and a sticky bit
  standing for a fraction of a guard unit, round as the exact sum does, since
  no rounding boundary of A lies strictly between them.
- A sum of a value and an exact product of two values, rounded once (a fused
  multiply-add, as the named algorithms' `run` with exact products adds each
  term; `fused_add`): the product's significand has at most 128 bits, and the
  value's fits one word (a value of A, or an exact term below). Let big
  be the operand with the higher top and G = min(big's unit, big's top - p -
  3). If the other operand lies wholly below 2**G, it is replaced by its sign
  times 2**(G - 1): big is a multiple of 2**G; the sum's top is at least big's
  top - 1, so its unit, the subnormal spacing included, is at least
  2**(big's top - p - 1), and every value of A it may round to and every
  midpoint between two of them is a multiple of 2**(G + 1); so no rounding
  boundary lies strictly between big and big +- 2**G, and the rounding is the
  same for every operand of that sign below 2**G. The exact sum then spans
  fewer than max(w, p + 3) + w' <= 65 + 128 bits, w the width of big's
  significand and w' the other's (big's top lies max(w, p + 3) bits above
  2**G, and the other reaches below 2**G by less than w'), exact in four words
  (in two when it spans at most 127), and its two words from the highest
  nonzero one, with a sticky bit for the words below, round as the exact sum
  does.
- With exact terms the model does not round on their own and A need not hold
  (Lemma U: `add`'s terms with a = 1 or -1, `add.md`; gemm's beta c with beta
  1 or -1, `gemm.md`), no term is rounded on its own: the first two terms'
  exact sum is rounded once and each later term is added to the sum and
  rounded once, both by the fused sum above (`femu_add_exact`, every such
  term a one-word value: a value times a factor of 1); a term alone stays
  exact until the rounding to the output.
- A rounding takes the bits below the result's unit (p bits below its top, no
  finer than the subnormal spacing) and the sticky bit against half a unit.
- With flush to zero, inputs subnormal in A are read as zero, and results
  below 2**emin become zeros of their sign.
- Elements are summed in index order; a sum that overflows while every term
  is finite starts a new partial sum (Lemma O's slots, `femu_slot_sum`).

ExactRef's test suite, developed separately, compares it with MPFR's
term-by-term rounding, for formats up to 62 bits of precision, and with
PyTorch's scalar arithmetic for flush to zero.

## Results as inputs

An operand may be an earlier result. Its values x are that result's
`rounded`, and its radius r per element and part is that result's tolerance,
taken with the same arguments (for an algorithm: its `standard` and
`flush_to_zero`, with the default compute format and ulps; for an attention
algorithm, its compute format and ulps too, as the fused backward's given
output and logsumexp come from the same kernels' forward). The
implementation under test stored the earlier output, so by the earlier
operation's Lemma 3 each stored value x' is a value of the earlier output
format with each part within r of its midpoint part, or an infinity or NaN
that the earlier result's flags allow, or, where the earlier reference is an
infinity whose exact value was finite, a value of its finite range.

Each part of the later result reads the stored parts its terms use. A radius
that is not finite on a part whose reference is finite leaves no bound in the
parts that read it; a part whose reference is not finite has no finite
midpoint, so its stored values are those its flags give (any finite value
among them where it has a finite range). A flagged stored part
passes on as flags: through an exact nonzero coefficient part (the coefficient
rule's terms), an infinity keeps its sign or flips it with a negative
coefficient part, and NaN stays NaN; through a coefficient that is exactly 0,
an infinity gives NaN; through a data factor, which may be anything, any flag
sets all three; infinities of both signs give NaN. Under either standard,
so do two infinities of opposite signs that an implementation's sum can meet
(the infinities rule): a passed one, the exact value's own (an infinite input,
or C's in gemm), a term rounded on its own past the threshold (Lemma O (a),
each factor anywhere within its radius; both standards' models round each
product), and, where one of those occurs, the finite terms of the other sign
reaching the threshold together (Lemma O (g)). Each side, and each finite term,
is at most the finite terms' mass with the radii's deviation, so where that mass
through Lemma O (g)'s reach stays below the threshold none reaches it, and the
terms need no walk. A term's own infinity is also a
flag, unless the exact value is infinite: every implementation's value is
then that infinity or NaN. An infinite factor times a value an
implementation may read as 0 (a stored value whose radius reaches 0, or with
flush a value below the compute format's smallest normal) gives NaN, in every
implementation that read 0, on every path. With the products exact (tensor-core steps, exact
accumulation) no term rounds past the threshold on its own, so the events are
the passed infinities and the exact value's own. Exact accumulation sums the
finite terms exactly, so they reach no threshold. Tensor-core steps can (Lemma
T, Overflow): with an event of one sign, NaN where the other sign's total of
the finite terms (Neg for +inf, Pos for -inf, each factor anywhere within its
radius) plus G M + H reaches a step's threshold or a rounding to R's, in gemm
through alpha's sign, or where the epilogue's total of that sign with beta c,
plus G' M' + H', reaches float32's; where R has no infinity, an event is NaN
itself, since a piece that holds it may be rounded to R. Each total is at most
the finite terms' mass with the radii's deviation, M_f, so where M_f (1 + G) + H
stays below the least of those thresholds, and with the epilogue
|alpha| (M_f (1 + G) + H) + |beta c| + its radius, plus
G' (|alpha| M_f + |beta c| + its radius) + H', below float32's, no side reaches
one and the terms need no walk. In a
compute format without infinities each of these events is NaN itself. The finite tolerance does
not change: a stored infinity or NaN makes the part it reaches non-finite, or
is left out by the coefficient rule. A stored finite value where the reference
used an infinity, and a stored part with no bound (an infinite radius: it may
be any finite value or any of the flags), give the parts they reach every
finite value and every flag, whatever those parts' references are: an
implementation's finite stored value can give a finite output even where the
later reference is not finite. Operations whose exp or division can turn a
non-finite value back into a finite one (attention, matrix_exp) give no bound
where a stored value may be non-finite, and so also every finite value and
every flag there.
