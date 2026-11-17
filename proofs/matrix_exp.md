# matrix_exp: proofs

This file proves what `exactref.matrix_exp` claims: each entry of the reference
output is the exact value correctly rounded, and `tolerance()` bounds every
implementation that satisfies the backward-error model. Lemmas 1 and 3 are in
`foundations.md`.

## The enclosure

The interval matrix exponential (`csrc/matrices.c`) returns, for an interval
matrix B, an interval matrix that contains exp(X) for every matrix X in B,
or reports that it could not bound it.

**Interval matrices.** An interval matrix [M, R] has dyadic midpoints M_ij and
dyadic radii R_ij >= 0, and contains a real matrix X when
|X_ij - M_ij| <= R_ij for every i and j. A complex interval matrix is a pair
of real ones, for the real and the imaginary parts. A midpoint is held as L
signed limbs of 53 bits times a power of 2 (an integer of at most 53 L
bits), a radius as an upper bound r * 2**f with r < 2**32. Every step below
keeps the part of a midpoint it drops in the radius and rounds radii up, so
each result contains what its formula gives for every member of its
operands.

**Products.** If [M, R] contains X and [N, S] contains Y, then for any C,
entrywise,

    |XY - C| <= |MN - C| + |M| S + R |N| + R S,

since XY - MN = M (Y - N) + (X - M) N + (X - M)(Y - N). The kernel first
puts each row of M and each column of N in fixed point. A line (a row or a
column) whose entries' top bits lie within 53 of each other goes on one
exponent, 53 L bits below its largest entry's top. A wider line is banded:
an entry whose top lies in band b, 53 b to 53 (b + 1) bits below the
line's largest (b < 64, the last band taking every lower entry), keeps
L + 1 limbs on the exponent 53 (L + 1 + b) bits below the line's top, so at
least 53 L bits of its own. Each entry is truncated toward zero and the part
dropped, less than one unit of its exponent, is added to its radius, so the
new pair [M', R'] still contains X (and [N', S'] contains Y). C is M'N'
truncated toward zero to L limbs: the limbs' products are summed exactly in
accumulators of 128 bits, each product of two banded entries placed as many
limbs up as their bands lie above their lines' lowest, so that every pair of
bands meets on one array of digits. Each limb product is below 2**106 and at
most P (L + 1) n <= 2**20 of them meet in one accumulator (P parts; the
kernel bounds nothing past that), so the sums are exact, and |M'N' - C| is
less than one unit of C's last limb. |M'|, R', |N'| and S' are rounded up to
Q = 52 bits, each on the exponent of its entry's band (for a radius, its
band among the line's radii, set the same way), so each product of two is
below 2**104 and the sum of each pair of bands' products is exact in 128
bits; those sums, rounded up, plus one unit of C's last limb where C was
truncated, bound the right side, so [C, T] contains XY for every X in
[M, R] and Y in [N, S]. A complex product is
(X + iY)(Z + iW) = (XZ - YW) + i(XW + YZ): each part's midpoint sum is exact
as above, and its radius is the sum of the right sides over its two
pairings; a sum or difference of two enclosures, with the radii added,
encloses the sum or difference of their members.

**Complex disks.** A complex entry also carries a disk radius d bounding
|z - m|, the modulus of its error, for every member z (m its midpoint),
besides its parts' radii r_re and r_im; an input entry's is
hypot(r_re, r_im) rounded up. For complex z and w with |z - m| <= d and
|w - k| <= e, |zw - mk| <= |m| e + d |k| + d e, so a product's disk radius is
the sum over k of |M_ik| e_kj + d_ik |N_kj| + d_ik e_kj, with the moduli
|M_ik| and |N_kj| rounded up from their parts' Q-bit magnitudes, plus both
parts' truncations; sums and scalars likewise. Each part's error is at most
the modulus of the error, and the modulus at most hypot of the parts'
errors, so after every product and sum the kernel takes each part's radius
as min(r, d) and d as min(d, hypot(r_re, r_im)) rounded up; an entry's ends
use min(r, d) too. The remainder below bounds each entry's modulus, so it is
added to the disk radius as well as to the parts'.

**Sums and scalars.** For an interval scalar [c, r] and enclosures [M, R] of
X and [P, U] of Z, |(c' X + Z) - (c M + P)| <= |c| R + r |M| + r R + U for
every c' in [c, r]. The kernel computes c M + P on one exponent with a
guard limb and truncates it toward zero to L limbs, the parts dropped in the
radius.

**The Taylor remainder.** For a real or complex matrix Y with
||Y||_inf <= b <= 1 and N >= 0,

    |exp(Y) - sum over k <= N of Y**k / k!|_ij <= 2 b**(N + 1) / (N + 1)!

for every i and j, and the left side is 0 where no walk of length 1 or more
in Y's nonzero pattern leads from i to j.

*Proof.* |(Y**k)_ij| <= ||Y**k||_inf <= b**k: an entry is at most its row's
absolute sum, and the norm is submultiplicative. So the left side is at most
the sum over k > N of b**k / k!, which is b**(N + 1) / (N + 1)! times the sum
over j >= 0 of b**j (N + 1)! / (N + 1 + j)!; each of those terms is at most
(b / (N + 2))**j <= 2**-j, since b <= 1 <= (N + 2) / 2. (Y**k)_ij is the sum,
over the walks of length k from i to j, of the products of their entries,
so it is 0 for every k >= 1 where no such walk exists. []

**The algorithm.** For an interval matrix A (a point matrix for the
reference):

1. a bounds ||X||_inf for every X in A: the largest row sum of
   |M_ij| + R_ij, rounded up (for a complex entry, the smaller of
   |M_ij| + d_ij and the sum of its parts' |midpoint| + radius).
2. For an integer s >= 0 with b = a 2**-s <= 1, B = A 2**-s (exponents
   moved, exactly) contains X 2**-s for every X in A, and
   ||X 2**-s||_inf <= b.
3. The Taylor polynomial of degree N, T(Y) = sum over k <= N of Y**k / k!,
   is evaluated on B by Paterson and Stockmeyer's scheme: with
   q = ceil(sqrt(N + 1)), the powers B**2 to B**q, then Horner's rule in
   B**q over blocks of q terms. It is the same polynomial, and every
   operation encloses its result for every member of its operands, so the
   result contains T(Y) for every Y in B. The coefficients 1/k! are
   interval scalars: 1/(k - 1)! divided by k toward zero, the remainder in
   the radius.
4. Adding 2 b**(N + 1) / (N + 1)! to the radius of each entry (i, j) that
   a walk of length 1 or more connects in A's pattern (the entries whose
   interval is not exactly 0; the walks by Warshall's transitive closure on
   bit sets) gives an enclosure of exp(Y) for every Y in B, by the
   remainder: Y's pattern lies within A's. Every other entry is exactly 0
   in each power and stays exactly 0 in every product.
5. s squarings: exp(X) = exp(X 2**-s)**(2**s), and the product of an
   enclosure with itself contains the square of each of its members.

s and N are chosen for the fewest products, s + (q - 1) + (ceil((N + 1) / q) - 1),
with the remainder at most 2**-(53 L). Where some radius exceeds
2**(c - 53 L / 2), with every ||X||_inf < 2**c (a box rather than rounding
errors), s is at least c + 10, so ||B||_inf <= 2**-10: a Taylor sum's
interval radius on a box can exceed the spread of its members by a factor
up to e**(2 ||B||), which with few squarings would approach e**(2 ||A||).
These choices set the cost and the tightness, not the containment. The
kernel reports that it could not bound the result where an exponent would
leave the range it holds, a ||X||_inf bound of 2**60 or more. []

**Diagonal matrices.** If every entry of A off the diagonal is exactly 0,
midpoint and radius, then so is every member's, and exp(X) is diagonal with
entries exp(X_ii). The kernel then takes each from the scalar functions
(`foundations.md`) at p = 53 L + 8 bits, with no products. A real entry in
[a_lo, a_hi] gives [exp(a_lo), exp(a_hi)] by Lemma X, its ends rounded
outward (exp is increasing). A complex entry with parts in [a] and [b] (each
part's ends within its disk radius too) has exp(a + ib) = e**a cos b + i
e**a sin b; Lemma Z over the interval [b] encloses cos b and sin b for every
b in [b]. With e**a in [E_lo, E_hi], E_lo > 0, each part is enclosed by the
product of the intervals, [E_lo f_lo, or E_hi f_lo for f_lo < 0; E_hi f_hi,
or E_lo f_hi for f_hi < 0] for f the cos or sin enclosure, rounded outward;
the rectangle of the two contains exp(a + ib) for every a in [a] and b in
[b]. Each interval becomes a midpoint and radius exactly, the midpoint
truncated to L limbs and the part dropped in the radius, and a complex
entry's disk radius is hypot of its parts' radii. Off the diagonal the
result is exactly 0.

The tests check the kernel's enclosures against python-flint's `arb_mat.exp`
and `acb_mat.exp` at a higher precision, and against MPFR.

## The reference output

**Structural zeros.** If no walk in A's graph leads from i to j (i != j), then
(A**k)_ij = 0 for every k >= 1, so exp(A)_ij = 0 exactly.

**Ziv refinement.** The reference kernel (`csrc/matrix_exp.c`) encloses every
entry with the enclosure above (A as a point matrix) at precision p plus
4 log2(n) + 32 bits plus the integer bits of A's norm bound, at most 61 (the
kernel runs no matrix whose bound reaches 2**60), rounded up to whole limbs,
starting at p = 160. Rounding is monotone, so if both
ends of a part's interval round to the same value, that value is the correct
rounding of the exact part. A part is done once that holds, its interval
excludes zero, and the interval fits an exact entry (`foundations.md`, The exact
output). Otherwise, or when an entry could not be bounded, p doubles. At the
precision cap an interval that still contains zero is accepted when both ends
round to the same value: the rounding is decided, and the exact entry is
stored as near zero.

**Rational parts.** Each part of an entry is rational or transcendental.
exp(A)_ij = sum over the distinct eigenvalues lam of e**lam * c_lam, with
algebraic c_lam. By the Lindemann-Weierstrass theorem, the numbers e**lam are
linearly independent over the algebraic numbers. So the entry is algebraic only
if c_lam = 0 for every lam != 0, and it then equals c_0, a rational. A
transcendental part is never a rounding boundary, so its refinement ends. A
rational part can be one (it may be exactly representable, or a tie), and
it may be zero. Parts still unfinished are therefore tested exactly, once,
and a rational part is rounded once from its exact value and stored as an
exact rational. The test runs at the first precision where its estimated cost
is at most the next refinement's, and at the latest before the last attempt
(the one at the precision cap, which decides what is left), so a rational
part that no refinement decides is tested before that decision; when it runs
changes only the cost, never an output. The kernel's estimate counts a
refinement as (n**3 per matrix product, plus n**2 for the decisions) L**2 for
L-limb entries, the next refinement four times the last, and the test as
64 M**4 ceil(B / 64), for the largest vertex set S of an unfinished part (M its
vertices times the parts, B the bits A's entries span): a matrix without
products is tested within a precision or two where B is at most 64, and one
precision later for each fourfold growth of ceil(B / 64). This estimate only
orders the work: the test's real cost
can differ from it by a large factor (it grows about as M**5 through the
powers of P, falls where a power of P is 0, and grows with the other vertex
sets and with the denominators of the eigenvalue-0 part). So a part that only
needs more precision, such as an entry far below its row's largest, is
decided without the test, and a rational part pays, in the estimate's units,
at most a few refinements cheaper than its test.

The test (`csrc/matrix_exp.c`) uses exact integers and fractions only
(`csrc/rational.c`). Every entry of A is a dyadic, so P = 2**k A is an
integer matrix for some k >= 0; its characteristic polynomial chi is monic
with integer coefficients, and A's eigenvalues are y = mu 2**-k for chi's
roots mu.

*The construction.* chi is split into square-free factors g, each with its
multiplicity m (Yun's algorithm); each g is monic with integer coefficients
(Gauss's lemma), and the factor z is taken out of its g. For a root mu of g,
write chi = (z - mu)**m q. The Hermite interpolation of e**x at A's
eigenvalues gives exp(A) = sum over the eigenvalues y of e**y f_y(A); in
terms of P, f_y(A) = q(P) T(P), with T(z) = sum over j < m of
tau_j (z - mu)**j and tau_j the Taylor coefficients at v = 0 of
e**(v 2**-k) / q(mu + v) (substitute x = z 2**-k in the interpolation for A:
the powers of 2 from q, from (x - y)**j and from the derivatives cancel). So
c_y = (q(P) T(P))_ij. The coefficients of q T are polynomials in mu,
computed in Z[mu]/(g) (g monic, so reducing modulo g keeps integers): one
computation covers every root of g, and c_y vanishes at every root of g
exactly when the entry's polynomial is 0 modulo g, since g's roots are
simple.

*Scaling by a unit.* For a factor other than z, q T is multiplied by
u = w**m 2**(k (m - 1)) (m - 1)!, w = q(mu): q does not vanish at a root of
g, whose multiplicity in chi is exactly m, so u is a unit modulo g and the
product is 0 modulo g exactly when the entry's polynomial is. Every
quantity is then an integer: with t_j the Taylor coefficients of q at mu,
J_0 = 1 and J_j = -(sum over 1 <= i <= j of t_i w**(i - 1) J_(j - i)) are
w**(j + 1) times the series of 1 / q(mu + v), and u tau_j is the sum over
a <= j of w**(m - a - 1) J_a e_(j - a), with the integers
e_b = 2**(k (m - 1 - b)) (m - 1)! / b!. For the factor z (mu = 0) the
construction runs without the scaling, in rational constants, and gives
c_0.

*chi.* chi is computed from its residues modulo word primes (Hessenberg's
method modulo each prime) by the Chinese remainder theorem. The coefficient
of z**(M - j) (P is M x M) is a signed sum of the C(M, j) principal minors of
order j, each at most R**j in magnitude by Hadamard's inequality, R < 2**r
the largest row's Euclidean length; so every coefficient is below
2**(M + M r) in magnitude, and primes whose product passes 2**(M + M r + 2)
determine it, read in the symmetric range. The primes are tested by
Miller-Rabin with the first 12 primes as bases, exact below 3.3 * 10**24.

*Square-free.* Whether chi is square-free is decided modulo the prime
p = 2**61 - 1 first: a monic common factor of chi and chi' over the
rationals has integer coefficients and divides both modulo p with its
degree kept, so if chi and chi' have no common factor modulo p, chi is
square-free. Otherwise Yun's algorithm runs over the integers, its gcds by
Euclid's algorithm on primitive pseudo-remainders (a gcd of factors of
monic integer polynomials is monic, up to sign, once primitive).

The test applies two reductions, neither of which changes its answer:

- Every walk from i to j stays among the vertices S on such walks, so
  exp(A)_ij = exp(A[S][S])_ij.
- A complex matrix X + iY is handled through the real matrix [[X, -Y], [Y, X]].
  X + iY maps to it as a ring homomorphism, so its exponential is the image of
  exp(X + iY), whose blocks hold the real and imaginary parts.

An exact zero is +0, or -0 when rounding toward -inf. A part so close to a
boundary that rounding it would need more than 8192 bits raises
`ReferenceUnavailableError` instead of returning a guess. So does a part far
below the largest entries of its row or column: the products truncate an
entry more than 53 (L + 64) bits below its line's largest into its radius
(Products), so such a part's interval keeps 0 at every precision. In
exp([[-x, 1], [0, 0]]) the squarings put e**(-x / 2**j) next to about 1/x, and
from about x = 13800 entry (0, 0), e**-x, is never decided, though it rounds
to 0. A matrix with a non-finite entry is NaN throughout.

**Norms of 2**60 or more.** The kernel's exponents are 64-bit integers, and
exp(A)'s entries reach e**||A||_inf in magnitude, so the interval exponential
bounds nothing for ||A||_inf >= 2**60. Its one range test is on its
rounded-up bound of ||A||_inf (`mexp_init`, a complex entry by its modulus as
in step 1), which also reaches 2**60 for some norms just below it; the
reference runs no test of its own, and such a matrix is treated as here. A
diagonal matrix's exponential is the
diagonal of its entries' exponentials: e**a is enclosed by Lemma X
(`foundations.md`), whose scale 2**n has an exponent of any size (one past
2**61 in magnitude lies beyond every format and the exact entries' range,
so it is decided as such), and for a complex entry e**a (cos b + i sin b)
by Lemmas X and Z, each part the product of e**a's enclosure and cos b's or
sin b's, rounded outward. A format's value b = u / v has |b| < 2**(2**20) and
v <= 2**(2**20), so by Mahler's bound |pi - p/q| > q**-42 for integers p and
q >= 2, b lies more than (|k| / 2)(|k| v)**-42 > 2**-(2**27) from each k pi / 2
with k != 0 (|k| < 2**(2**20 + 1); for |k| v = 1, b is an integer, 0.4 or
more from pi / 2), and a nonzero b lies 2**-(2**20) or more from 0. So
|cos b| exceeds 2**-(2**27), and so does |sin b| unless b = 0 (where the part
is exactly 0, and is rounded as the rational parts' zeros are: +0, or -0
rounding toward -inf), and a part whose scale is clamped still lies past every
format's range. Each is refined as above. Any other
such matrix raises `ReferenceUnavailableError`, and so does a part still
undecided at a precision whose limbs L make P (L + 1) n exceed 2**20, past
what the kernel's products hold (`imat_product_fits`), with its own message:
at the 8192-bit cap from about n = 6600 real and 3300 complex, half those for
the Frechet derivative's and the backward's block matrices.

## The tolerance

**Model (backward error).** No internal value is shared by all correct
algorithms: Taylor or Pade, any degree, any number of squarings. So
`matrix_exp` defines a correct implementation by its backward error. Its output
entries are

    fl_F(exp(A + dA)_ij),   ||dA||_inf <= c * (n + 16) * u * max(||A||_inf, 1),

with u = 2**-p of the compute format F and c = `ulps` (default 1).

The allowance follows how the algorithms err. Each evaluates an approximant
r(B) of exp(B) at a scaled B = A / 2**s and squares the result s times. An
error d in r(B) becomes about 2**s d after the squarings: a backward error of
about 2**s d, where 2**s is about ||A|| over the scaled norm the algorithm
aims for, so it grows with ||A||. The squarings' own roundings are much
smaller than this carried error (measured below). The evaluation's error
comes from inner products of length n and from a fixed number of other
roundings, the approximant's short sums and scalings and the linear solve:
hence n + 16. The floor of 1 is there because every algorithm evaluates and
rounds quantities of size about 1 (the identity part of exp(A)) even when A
is tiny, so its error is at least about u per entry: Higham's degree-3 Pade,
which scipy and MATLAB use for ||A||_1 <= 0.015, leaves such an error in its
linear solve. A backward error proportional to ||A|| alone could not produce
it; with the floor, exp(A + dA) moves by about dA for a small A.

Each entry is held in F with one rounding, then rounded to the output format.
With `flush_to_zero=True`, an entry part of A that is subnormal in F may be
read as zero, and a held result may be flushed (Lemma 1 with the 2**emin
term). The model covers any flushing inside the algorithm only through dA.

No published bound gives this allowance. Higham's bound on the squarings
needs their number, his bound on the evaluation needs the degree, and he calls
the effect of rounding errors on the squaring phase an open question (Higham,
"The scaling and squaring method for the matrix exponential revisited", SIAM
J. Matrix Anal. Appl. 26(4), 2005, pp. 1184, 1187, 1192). So the allowance
is an assumption about what correct implementations do, measured with a
margin of about 2 against the tolerance below. On random Gaussian, symmetric,
non-normal and skew-symmetric matrices, n = 2 to 64, ||A||_inf from 0.03 to
512, in float32 and float64, degree-13 Pade with scaling and squaring,
scaled as in Higham's Algorithm 2.3, reaches at most 0.47 of it for n <= 8
and 0.18 for n = 16 to 64; PyTorch's `matrix_exp` at most 0.35 and 0.23,
apart from its own errors at small norms. Outside it lie, with the
reference matching mpmath at 40 digits:
- near a multiple of the identity with eigenvalues near +-5.37 * 2**s
  (Pade-13's scaling points), the degree-13 Pade formula's own rounding: its
  numerator cancels for negative eigenvalues and its denominator for
  positive ones, an error of up to about e**5.37 u (211 u for the 1 x 1
  matrix [[-5.35]] in float64), up to about 3 times the allowance on 2 x 2
  to 16 x 16 matrices; there PyTorch's float32 results reach about 6 times
  for negative eigenvalues;
- PyTorch's own errors at small norms: in float64 at ||A||_inf near 0.03
  for every matrix measured, and in float32 for some matrices at ||A||_inf
  up to 1;
- `scipy.linalg.expm` 1.18.1 in float64 for some matrices from ||A||_inf 4,
  with normwise relative errors up to about 1950 u where Pade-13 stays below
  22 u.

**Claim.** Under the model, `tolerance()` is at least |output - ref| for every
entry.

**Proof.** The tolerance kernel (`csrc/matrix_exp.c`, `mtol`) computes,
rounding upward, a bound r >= c * (n + 16) * u * max(||A||_inf, 1). With flushing, each subnormal entry
part is first widened to the hull of {0, a}. That contains the value the
implementation reads, and leaves ||A||_inf, and so r, unchanged. It encloses
exp(A + dA), for every allowed dA, in two ways below, and keeps for each entry
part the intersection of the two enclosures; where one is missing, the other
alone, and where both are, no bound (an infinite tolerance). Lemma 1 widens
the result for the rounding to F, and Lemma 3 for the rounding to the output
format and the tolerance.

*The box.* Every allowed dA satisfies |dA_ij| <= ||dA||_inf <= r, so the box
of entries [A_ij - r, A_ij + r] contains every A + dA, and the enclosure above
contains exp(A + dA). It is missing where the kernel cannot bound it.

*The ball.* Write A0 for the matrix of the entries'
midpoints and A + dA = A0 + F with ||F||_inf <= rho (here A0 = A and rho = r;
Results as inputs gives a larger rho). Absolute values of complex numbers are
moduli.

1. For every F, differentiating exp((1-s)(A0+F)) exp(s A0) in s gives
   exp(A0 + F) - exp(A0) = integral_0^1 exp((1-s)(A0+F)) F exp(s A0) ds.
2. So, with X = exp((1-s)(A0+F)) and Y = exp(s A0),
   |exp(A0 + F) - exp(A0)|_ij <= integral_0^1 sum_k |X_ik| sum_l |F_kl| |Y_lj| ds
   <= rho integral_0^1 xF(1-s)_i y(s)_j ds,
   where xF(t)_i is row i's absolute sum of exp(t(A0+F)) and y(s)_j is the
   largest |exp(s A0)_lj| over l, since sum_l |F_kl| <= rho for every k.
3. Pieces. [0, 1] is cut into m pieces of width h = 1/m. On piece k we
   need bounds x_k on every row's absolute sum of exp(t A0), y_k on every
   column's largest absolute entry, and g_k = max_i x_(k,i) >=
   ||exp(t A0)||_inf. Two constructions give such bounds, and each of x_k's
   and y_k's entries is the smaller of the two, itself such a bound. The
   enclosure above gives W containing exp(h A0); a product of enclosures
   contains the products of their members.
   - The box. B contains exp(tau A0) for every tau in [0, h] (the interval
     matrix A0 [0, h], midpoints A0 h/2 and radii |A0| h/2, contains every
     tau A0), and with G_0 = I, G_(k+1) = G_k W, G_k contains exp(k h A0)
     and G_k B contains exp(t A0) for every t in piece k. The upper bounds
     of G_k B's absolute values give the row sums and column maxima.
   - The centered expansion. H_0 contains exp(h A0 / 2) and H_(k+1) = H_k W
     contains exp((k + 1/2) h A0). Every t in piece k is (k + 1/2) h + delta
     with |delta| <= h/2, and exp(delta A0) = I + delta A0 + R with
     ||R||_inf <= e**s - 1 - s <= rest = (s**2 / 2)(1 + 2 s),
     s = ||A0||_inf h / 2 <= 1/2 (the series' terms from s**3 on add at most
     s**2 / 2 times 2 s for s <= 1/2). So with X = exp((k + 1/2) h A0),
     exp(t A0) = X + delta X A0 + X R and, entrywise,
     |exp(t A0)| <= |X| + (h/2) |X A0| + |X| |R|. The upper bounds of H_k's
     absolute values plus h/2 times those of the interval product H_k A0
     bound the first two terms; the third adds at most rest times row i's
     sum of |X| to row i's sum, and at most rest max_i (row i's sum of |X|)
     to any entry, so to every column's maximum. The first-order term is one
     interval product with the point A0, so it keeps the correlation in
     delta that B, an enclosure over a whole piece, loses; the box is the
     tighter where exp(t A0)'s entries span many binades, which the column
     term's largest row sum overstates.
4. The left factor. Step 1 for the time t and the matrix tA0 gives
   exp(t(A0+F)) - exp(t A0) = integral_0^t exp((t-sigma)(A0+F)) F exp(sigma A0) dsigma,
   and, as in step 2, xF(t) <= x(t) + rho integral_0^t g(sigma) xF(t - sigma) dsigma
   (componentwise, x(t) the row sums of exp(t A0), g(sigma) = ||exp(sigma A0)||_inf).
   For t in piece q and sigma in piece j, t - sigma lies in pieces q - j - 1
   and q - j. So with Q_k bounding xF on piece k (Q_(-1) = 0),
   a = x_q + rho h sum_(j=1..q) g_j (Q_(q-j) + Q_(q-j-1)) (a sum bounds the
   larger of the two) and c0 = rho h g_0 < 1, the supremum S of xF on piece q
   (finite, by continuity) satisfies S <= a + c0 max(S, Q_(q-1)). Both
   S <= a / (1 - c0) (when S is the larger) and S <= a + c0 Q_(q-1) are at
   most Q_q = (a + c0 Q_(q-1)) / (1 - c0). If c0 >= 1 the ball is missing.
   Collected by the distance k between pieces, this is
   Q_q = b_q + sum_(k=1..q) kappa_k Q_(q-k), with b_q = x_q / (1 - c0) and
   kappa_k = rho h (g_k + g_(k-1)) / (1 - c0), whose
   solution is Q_q = sum_(k=0..q) l_k b_(q-k) with l_0 = 1 and
   l_q = sum_(k=1..q) kappa_k l_(q-k): one product of the lower triangular
   matrix (l_(q-q')) and the matrix of the b_q. Every coefficient is
   nonnegative, so upper bounds for x, g, kappa and l give upper bounds for Q.
5. Every s in piece k has 1 - s in piece m - 1 - k, so
   |exp(A0 + F) - exp(A0)|_ij <= D_ij = rho h sum_k Q_(m-1-k, i) y_(k, j),
   and each part of exp(A0 + F)_ij lies within D_ij of the enclosure of
   exp(A0)_ij (above, for the point A0).

The bound holds for any m and any working precision; they only set how tight
and how costly it is. The kernel takes m = max(1, min(ceil(4 ||A||_inf), 256))
and 64 + ceil(1.45 ||A||_inf) bits, and computes the ball only for
||A||_inf <= 256 (beyond, the box alone). Then h ||A||_inf <= 1/4 up to
||A||_inf = 64 and h ||A||_inf <= 1 beyond, so s <= 1/8 or s <= 1/2, taking
the bound on ||A||_inf for ||A0||_inf. Over a piece exp(t A0) = exp(k h A0)
exp(tau A0) changes by a factor ||exp(tau A0)||_inf <= e at most, which bounds
how much the box's suprema overstate; the centered expansion's overstate by
the second-order rest and the column term. The bounds read from G_k B, H_k
and H_k A0 are each entry's largest |part|, for a complex entry the smaller of
sqrt(re**2 + im**2) of its parts' largest magnitudes and |midpoint| plus its
disk radius; every square root, sum, product and quotient is rounded up to 64
bits, and 1 - c0 is held as a lower bound, rounded down. The two matrix
products of steps 4 and 5, of nonnegative upper bounds, are interval products
whose upper ends are read. []

## Results as inputs

When A is an earlier result, the implementation computes the model on its
stored A', each part of A'_ij within rho_ij of A_ij's, rho the earlier
tolerance taken with the same arguments (for complex entries one radius per
part). Where a stored value may be non-finite (the earlier result's flags),
exp can turn it back into a finite value, so there is no bound: the
tolerance is infinite, the finite range holds every finite value and every
flag is set. Its output is fl_F(exp(A' + dA')_ij) with
||dA'||_inf <= c * (n + 16) * u * max(||A'||_inf, 1).

**Claim.** `tolerance()` is at least |output - ref| for every such A' and dA',
where ref is the reference output on A.

**Proof.** The kernel first widens each entry to [A_ij - rho_ij,
A_ij + rho_ij], which contains A'_ij, and with flushing then widens each part
that may be subnormal in F (an interval whose end nearest zero is below
2**emin) to include 0, which contains the value read. The norm of these
intervals bounds ||A'||_inf, so r bounds every entry of dA', and the box
contains every A' + dA'. For the ball, A0 is the matrix of these intervals'
midpoints: each |A'_ij - A0_ij| is at most the sum of the entry's parts'
half-widths, so A' + dA' = A0 + F with ||F||_inf <= r plus the largest row
sum of those half-widths, which is rho. The rest is the proof above. An entry
whose radius is inf or NaN, or whose stored value may be non-finite, has no
bound: every entry of its matrix gets an infinite tolerance, or NaN where the
reference is not finite, with every finite value in its range and every flag
set. []
