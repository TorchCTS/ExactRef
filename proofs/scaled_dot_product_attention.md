# scaled_dot_product_attention: proofs

This file proves what `exactref.scaled_dot_product_attention` claims: the
reference output is the exact value correctly rounded, `tolerance()` bounds
every implementation that rounds only at the required boundaries with normal
accumulation, `tolerance(standard="any_order")` every such implementation that
does not overflow, and `normal()` is the model computed with normal
accumulation. Lemmas 1 to 3 and the normal standard's lemmas (S, B, L, O, A,
E) are in `foundations.md`.

## The exact value

For one output row, a masked key (False in a bool mask, -inf in a float mask,
or above the diagonal with `is_causal`) takes no part, whatever its score or
value: its logit is -inf. At every other key the logit is the score times
sigma plus the mask value, by IEEE's rules on exact values where the query or
key row holds NaN or an infinity (the product's rules, times sigma's sign): a
NaN or +inf logit makes the row NaN, and a -inf one (a -inf score times sigma
beside a finite mask value) removes the key, which then takes no part either.
The kept keys are those whose logit is finite. A row with no kept key has
every output 0. A value that is not finite enters its column where its key is
kept (an infinity, or NaN for both signs or a NaN) and nowhere else. These
rules follow PyTorch's math path with IEEE arithmetic on exact values, except
that a key that takes no part adds nothing where the math path multiplies its
weight 0 by a non-finite value or adds a NaN or +inf score to its -inf bias;
the docstring and the Conventions in project-scope.html list them. For the
kept keys j,

    l_j = (q . k_j) * sigma + b_j,     o_c = sum_j exp(l_j) v_jc / sum_j exp(l_j),

with sigma the given scale or 1/sqrt(E), and b_j the float mask value or 0.
When E = 0, every q . k_j is an empty sum, 0, so l_j = b_j, as in PyTorch.

## The reference output

**Exact cases.** Group the kept keys by equal logits. Write o = N / Z with
N = sum_g S_g exp(s_g) and Z = sum_g |g| exp(s_g), where s_g are the distinct
logits and S_g the sums of v over each group. The logits are algebraic: dot
products of dyadic rationals, times sigma (rational or 1/sqrt(E)), plus dyadic
mask values. If o were algebraic, sum_g (S_g - o |g|) exp(s_g) = 0 would be a
vanishing combination of exponentials of distinct algebraic numbers with
algebraic coefficients. By the Lindemann-Weierstrass theorem every coefficient
is then 0, so every group has the same mean S_g / |g| = o. So o is algebraic
exactly when all group means are equal, and then it is that mean, a rational.
The attention kernel (`csrc/attention.c`) checks this with exact integers and rounds the mean once.
Groups are formed with exact rationals when sigma is rational. When
sigma = 1/sqrt(E) is irrational, two logits l_i = l_j with different (dot, mask)
pairs would make sigma rational, so grouping by the exact (dot, mask) pair is
exact.

**Other outputs** are transcendental, so they are never a rounding boundary.
The attention kernel encloses every such output first with the weight tables
of Lemma D at F = 191 bits, and decides it when both ends round alike and the
enclosure fits an exact entry. For a call without tables (the scores' span
past 127 bits, a mask's past its tables, or an operand with flagged or
two-word slots), and for a row whose tables' row sum is wider than 2**-20 of
its largest weight's lower end (the largest score and the largest mask value
on keys far apart, every weight a few units of 2**-F), it takes the exact
scores from the product kernel and each
w'_j = exp(-x_j) from Lemma X at both ends of an enclosure of its argument
x_j = l_r - l_j at the first precision, r the reference key: the kept key of
the largest logit, compared exactly (Logit differences below), the first of
equal ones (without mask values, the key of s_top, the kept keys' largest
score, the least with sigma < 0). Then
w'_j = exp(l_j - l_r), and x_r = 0 exactly, so w'_r = 1 and the row sum is at
least 1 however far apart the largest score and the largest mask value are; the
quotient's ends are the sums of the weights' ends times the values (each
value with the end that makes its product least, or largest) over the row
sum's end that makes the quotient least (largest). The kernel then refines the
outputs left undecided, row by row, from twice the first precision, doubling p
each round: it repeats that enclosure at precision p without tables. Each
argument x_j comes from Logit differences at p bits (X = s_r - s_j and
Y = b_r - b_j, exact): its lower end is at least 0, so every weight is at most
1 and the row sum lies in [1, n], and its width is within 2**(4 - p) of its
size. Lemma X gives the weight's ends, an argument past
2**40 the ends 0 and exp(-2**40) (`exp_clamped`). The sums N and Z are
rounded outward at p + 64 bits and the two quotients at p bits. A round that
leaves an output undecided runs the sign test below at the same p.

**Logit differences.** Every comparison of two logits, and every argument the
kernels pass to exp, is v = sigma X + Y with exact dyadics X (a score or a
difference of scores) and Y (a mask value, or a difference of mask values and
grid points; 0 without a mask). `attention.c`'s `att_affine` gives v's exact
sign and an enclosure within 2**(4 - p) |v| at p bits, |sigma|'s ends taken at
p bits. Rounding sigma X to p bits before adding Y would not do: where the two
terms cancel, that enclosure is 2**-p of the terms wide, not of v, and an end
can pass exp's range.

For sigma = num / den (den > 0; num = 0 for sigma = 0): v = N / den with
N = num X + den Y exact, so sign(v) = sign(N), and N / den rounded down and up
to p bits encloses v.

For sigma = 1/sqrt(E) with enclosure [s_lo, s_hi], when X and Y do not have
opposite signs, v adds two terms of one sign: X times the end of sigma chosen
by X's sign, at p + 2 bits, plus Y, each side rounded outward. sign(v) is X's
sign, or Y's when X = 0.

Otherwise (X + Y sqrt(E)) (X - Y sqrt(E)) = X**2 - E Y**2, and
X - Y sqrt(E) = sign(X) sqrt(E) Q with Q = sigma |X| + |Y| > 0, so
v = sign(X) N / (E Q) with N = X**2 - E Y**2 exact. Then
sign(v) = sign(X) sign(N), and |N| over the ends of E Q (from sigma's ends at
p + 4 bits), rounded outward, gives |v|'s ends.

Each end is a directed rounding of a value of v's sign, so it never crosses 0.

**Lemma D: weights from digit tables.** Every product q_e k_je of a call is
a multiple of U = 2**u, u the sum of the lowest set bits of Q's and K's
nonzero values, so every score is an integer S_j times U. In a row without
mask values, with sigma > 0 the largest logit is at the largest kept score
S_top, and with sigma < 0 at the smallest; so w_j = exp(l_j - m) =
exp(-c D_j) with c = |sigma| U and D_j = |S_top - S_j|, a nonnegative
integer (sigma = 0 gives D_j = 0). Write D_j in base 2**b with k digits
d_ji (k covers the call's largest distance), so
w_j = prod over i of e_i(d_ji), e_i(d) = exp(-c d 2**(b i)).

*Tables.* For each i, the entries T_i[d], d < 2**b, are integers with
T_i[d] <= 2**F e_i(d) < T_i[d] + 2 and T_i[0] = 2**F. They are built from one
enclosure: with c in [c_lo, c_hi], c_hi - c_lo <= 2**-(G + b i + 2), and
G = F + b + 8, Lemma X at G + 2 bits gives a_lo <= exp(-c_hi 2**(b i)) and
a_hi >= exp(-c_lo 2**(b i)), and a_hi - a_lo <= 2**(4 - G) a_lo (each end is
within 2**(1 - G) of its value, and c's width moves the value by a factor
of at most exp(2**-(G + 2))); then
A_lo = floor(2**G a_lo), A_hi = ceil(2**G a_hi), L_0 = H_0 = 2**G,
L_d = floor(L_(d - 1) A_lo / 2**G), H_d = ceil(H_(d - 1) A_hi / 2**G), and
T_i[d] = floor(L_d 2**(F - G)). By induction L_d <= 2**G e_i(d) <= H_d (floors
and ceilings of products of bounds of positive numbers). Each step widens
H_d - L_d by at most (A_hi - A_lo) + 2 <= 20 (all factors are at most 2**G),
so H_d - L_d <= 20 2**b < 2**(G - F - 3), and then
T_i[d] <= 2**F e_i(d) <= H_d 2**(F - G) < L_d 2**(F - G) + 1 < T_i[d] + 2.

*Weights.* For key j, take the entries of its nonzero digits, z_j of them:
p = the first entry, then p = floor(p T / 2**F) for each further entry T. A
digit 0 leaves the weight unchanged (its entry is exactly 2**F), and with no
nonzero digit p_j = 2**F = 2**F w_j. Otherwise

    p_j <= 2**F w_j < p_j + 3 z_j - 1.

By induction on the entries taken, with x the exact product so far times
2**F (at most 2**F): after the first, 0 <= x - p < 2; after a further entry
T with T <= 2**F e < T + 2, the new difference is
x e - floor(p T / 2**F) < (x - p) e + p (2**F e - T) / 2**F + 1 < (x - p) + 3,
and it stays at least 0 since p T / 2**F <= x e.

*Masks.* A float mask's values are dyadics; over a row's kept keys they are
integers M'_j times B = 2**v, v the lowest set bit of the row's nonzero mask
values. With b_top the kept keys' largest mask value, every logit is
m' - c D_j - B M_j with m' = sigma U S_top + b_top and M_j = (b_top - b_j) / B
a nonnegative integer, so exp(l_j - m') = exp(-c D_j) exp(-B M_j) <= 1, and a
second set of tables for exp(-B d 2**(b i)) gives its digits. The weights are
then relative to m' instead of m (the ratio o = N / Z is the same), each a
product of both sets' entries, with z_j counting the nonzero digits of both.

*Sums.* With h = 3 max_j z_j - 1, Z 2**F lies in [Zp, Zp + n h], Zp = sum_j p_j,
and N_c 2**F in [Np_c + h Vneg_c, Np_c + h Vpos_c], Np_c = sum_j p_j v_jc,
Vneg_c and Vpos_c the sums of the kept keys' negative and positive v_jc. The
ends of o_c = N_c / Z divide each end of N_c by the end of Z that makes the
quotient smallest (or largest), rounded down (or up). These are exact integer
sums, so the enclosure holds for every row; it is narrow when the weights
are not all tiny (Z of at least about 1, so the n h units of 2**-F are small
beside it). A row whose weights are all tiny, as with large negative mask
values on every kept key, leaves its outputs undecided, and they are refined.

Rounding is monotone, so if lo and hi round to the same value, that value is
the correct rounding of o. Each row's refinement starts at 160 bits and
continues, doubling p, until that holds, the interval excludes zero (o is not zero), and it fits an
exact entry (`foundations.md`, The exact output). It ends because o is not a
boundary and not zero. A precision cap (2**14 bits) raises
`ReferenceUnavailableError` instead of returning a guess; at the cap, an
interval that still contains zero is accepted when both ends round to the
same value (the rounding is decided, and the exact entry is stored as near
zero). The exact means of the
first case are stored as exact rationals.

**The sign test.** Each round's enclosure of o is about 2**-p wide relative
to o, so when o lies within a far key's weight of a point where rounding
changes, refinement up to the cap cannot decide it; a mask of -30000 or -1e9
does this. Example: v = (1, 2) with mask (0, -30000)
gives o = 1 + e**-30000 / (1 + e**-30000), which rounds up to the next value
after 1. So for an output that a round leaves undecided, the code looks for
a grid point b in [lo, hi] (`foundations.md`, Deciding a rounding by a sign):
zero, the value lo or hi rounds to, or their midpoint. When lo or hi rounds
past the largest value L (to an infinity, or to NaN in a format without one),
the midpoint is not available, and the candidates are instead the overflow
thresholds +-(L + ulp(L)/2), where rounding to nearest switches (Lemma 1,
`formats.c`'s `floatfmt_threshold`). Each candidate is kept
only when rounding switches there from lo's rounding to hi's, which the code
checks at b - eps and b + eps, on each side of b that [lo, hi] reaches
(`entries.c`'s `switch_point`). Then

    o - b = sum_g C_g e**s_g / Z,   C_g = S_g - b |g|,

over the groups of equal logits, with exact rational C_g (a group's members
share one logit, so any member's score and mask value stand for the group's). Groups with C_g = 0
drop out. The others have distinct algebraic s_g and nonzero C_g, so by
Lindemann-Weierstrass the sum is nonzero (o is transcendental here). The
kernel takes a, among the groups with C_g != 0, the one of the largest logit,
compared exactly (Logit differences), and writes

    o - b = e**-(x_a) t' / Z',   t' = sum_g C_g e**(x_a - x_g),   Z' = sum_g |g| e**(-x_g),

with x_g each group's argument from the reference key (as above),
so t' has the term C_a exactly and every other term's exponent
x_a - x_g = l_g - l_a, at most 0, from Logit differences, so e**(x_a - x_g)
is at most 1 by Lemma X (past -2**40, [0, exp(-2**40)]); Z' is
the first part's row sum. When t' excludes zero: if x_a's lower end is below
2**40, e**-(x_a) by Lemma X (its exponent within the dyadics' range) gives
o - b's enclosure; from 2**40 (where the exponential's lower end is clamped to
0), |o - b| <= |t'| e**-(2**40) / Z' is far below
2**-(LIMIT + 1) (|t'| / Z' <= sum_g |C_g| <= 2**(LIMIT + 2) n, since Z' >= 1,
the reference key's weight being exactly 1),
and the sign of t' decides by the tiny-offset rule (`foundations.md`,
Deciding a rounding by a sign). Case 2 of that lemma then decides the rounding
and gives the entry; case 1 with c = b is tried first. A sum whose largest
terms nearly cancel needs more precision, as before; the cap still applies.

## The tolerance

**Model.** An implementation rounds only at the required boundaries, in the
compute format F, with library `exp` within c ulps (`ulps`, default 2):

1. scores `q . k_j`: a sum of E products (the accumulation, below), then rounded to F (`scores.output`);
2. the scale: sigma rounded to F within c ulps (`scale_factor`);
3. logits: score * scale + mask value, rounded once (`logits`);
4. the row maximum m of the computed logits (an exact selection);
5. `logit - m`, rounded (`softmax.shift`);
6. exp of that, within c ulps (`softmax.exp`);
7. the row sum of the weights (`softmax.sum`);
8. per output column, the sum of weight * value, rounded to F (`values.accumulate`, `values.output`);
9. that sum divided by the row sum, rounded (`normalize`);
10. the final rounding to the output format (Lemma 3).

The three sums (steps 1, 7 and 8) are normal accumulations by default: index
order or lanes 2 and 4 (over the head dimension for the scores, over the key
indices for the others), each product rounded and then added, with the
overflow-safe slots (`foundations.md`, The normal standard). With
`standard="any_order"` they are sums in any order that does not overflow
(Lemma A).

**Masked keys.** The reference leaves out the keys a row does not keep (The
exact value). An implementation may skip them or compute their terms: a
masked key's score and its logit before the -inf bias, and for every key it
does not keep (masked, or removed by a -inf score) the weight
exp(-inf - m') = 0 and the products 0 * v. Those terms are exactly 0 and add
nothing, whether included or skipped (the lanes are by key index, lane l
holding the kept keys whose index is l mod L; Lemma L leaves the exact zeros
out). With flush to zero, adding one may flush a subnormal partial sum, which
a rounding the bounds count could have flushed already (`foundations.md`,
Lemma 2, Exact zero terms), so the bounds over the kept keys hold. A kept key
that surely vanishes (`attention_algorithms.md`, Keys that may vanish) adds
such exact zeros too. Every term is 0 except where a factor is not finite:

- a masked key's score times the computed scale, where it is NaN or +inf,
  makes its logit NaN (+inf - inf), so the row's maximum, outputs and
  logsumexp NaN. This is possible where the key's or the query's row holds
  NaN or an infinity, or where the score's accumulation, its rounding or the
  product with the scale may overflow (condition 4 below, and the
  step-by-step program's masked keys);
- 0 times a value that is not finite is NaN: the row's output in that
  column.

Elsewhere every such term is 0, and the implementation computes what the
model computes over the kept keys alone. A kept key's computed weight may be
exactly 0 too, where its interval reaches 0 (a key that may vanish, or an exp
that may underflow): times a value that is not finite it is NaN, while the
reference's weight is positive, so that column's output may be NaN as well.
So each bound below holds for every implementation's output that is not NaN,
and Lemma 3's flags (`may_be`) allow NaN in the outputs (and logsumexp) where
one of these cases may occur, and add no flag elsewhere. The closed form
flags a kept key's columns where the key may vanish, or where its weight's
least value, from exp((l_j - m) - R_j) / exp(Lam_up) (step 4's K and steps 5
and 6's R_j), is at most 0 after the exp's rounding: that rounding moves a y
by at most rel |y| + abs (Lemma 1's units: rel = c 2**-p' and
abs = c 2**(emin - p') for p' the fraction bits, plus 2**emin with flush),
so it rounds no y with y (1 - rel) > abs to 0. The step-by-step program flags
them where the lower end of the key's weight interval is at most 0
(`attention_algorithms.md` adds its methods' cases). An implementation that deals the kept keys to the lanes in turn,
skipping the masked ones, is covered by `standard="any_order"`. Not covered: rounding
done elsewhere, such as rounding the mask separately, materializing the
probabilities, casting the weights to the input format before the value product
(FlashAttention does this for bf16 and fp16), rescaling in an online softmax, or
a reciprocal of the row sum (named algorithms cover all but materializing the
probabilities, `attention_algorithms.md`). Pre-scaling Q and K changes the inputs; a consumer
that wants it passes the pre-scaled values with scale 1.

**Claim.** For each output element, `tolerance()` is at least |output - ref| for
every such implementation, except an output NaN that the element's flags
allow (Masked keys).

The attention kernel (`csrc/attention_tol.c`) proves the claim for each row in
one of two ways: the closed form (`att_tolerance_row`) when the row's
validity conditions hold, and otherwise the step-by-step program
(`att_program_row`), intersected with the bound for fully padded
rows when that one holds. All three are proved below. The step-by-step
program bounds every sum in any order (Lemma 2, which also requires that no
partial sum overflow), so it covers both standards.

### The closed form

**Notation.** One output row with kept keys j = 1..n (n >= 1). For each: the
exact score s_j, its magnitude sum M_j = sum_e |q_e k_je|, the float-mask value
b_j (0 without one) and the exact logit l_j = sigma s_j + b_j. Let
m = max_j l_j, w_j = exp(l_j - m) (the largest is 1, so W = sum_j w_j lies in
[1, n]), P_j = w_j / W, and o_c = sum_j P_j v_jc. The reference's enclosure of
o_c is [o_lo, o_hi]; O_c bounds |o_c| from above, and O'_c from below (0 if
the enclosure contains 0).

For a rounding or function boundary, rnd(Y, Y') is c ulp(Y) in its format,
plus 2**emin with flush unless Y' > 2**emin (rnd(Y) has no lower bound, so
always includes it with flush): by Lemma 1, the most one rounding there moves
a value whose magnitude is at most Y and at least Y'; Theta is its format's
overflow threshold (`foundations.md`, Units), and a value of magnitude below Theta
rounds to a finite value. For an accumulation, r, a, a' and Theta are the
accumulator's units, Growth its growth factor, and gamma(n) and
kappa(n) = R(n) eta (1 + gamma(n)) Lemma 2's terms (R(0) = 0).

Every quantity below is computed from the exact inputs as an enclosure that
contains its formula's exact value, in the attention kernel
(`csrc/attention_tol.c`) as intervals of dyadics rounded outward at 128 bits.
A condition counts as true only when the enclosures prove it, and the bound
uses their upper ends. The kernel takes the weights from Lemma D's tables: they give
w'_j = exp(l_j - m') with m' >= m (m' = m without mask values), and
w_j = w'_j / exp(m - m') with exp(m - m') = max_j w'_j enclosed by the largest
lower and the largest upper end of the w'_j (for any reference m'). Where the
call has no tables (the scores' span past 127 bits, or a mask's past its
tables), or a float mask leaves the tables' row sum wider than 2**-20 of the
largest table weight's lower end (the largest score and the largest mask
value on keys far apart: the largest weight is then known only to a few units
of 2**-F, and every weight would carry the row sum's width over it), each
w'_j comes from Lemma X at both ends of an enclosure of l_j - m' = -x_j, the
argument from the reference key r as in the reference output above, so
m' = l_r and w'_r = 1 exactly, and every argument is at least 0 (Logit
differences), so the largest weight lies in [1, 1 + 2**(3 - p)]; below -2**40
the lower end is 0 and the upper exp(-2**40). exp(R_j) - 1 comes from Lemma W
for R_j < 2**-4 and Lemma X above. For a key with R_j > 16 (the shift of a
logit far below the row's maximum) the kernel bounds P_j T_j and exp's
argument together, P_j T_j <= w_j exp(R_j) / W = exp(l_j - m + R_j) / W, with
the argument's upper end raised to -2**20 where it is lower (exp is
increasing), and eps_j with that value in place of w_j exp(R_j). The kernel
takes the methods of `attention_algorithms.md` the same way: with exp2 the
argument is (l_j - m) log 2 + R_j, and with online softmax
P_j T_j <= P_j exp(R_j) up, so the bound is that value times up.

**Steps 1 to 3, the logits.** The implementation's score s'_j sums E
products and rounds to F. Its accumulation is within acc_j of s_j: under
normal accumulation, the normal bound of `foundations.md` over the head dimension
(index order and lanes 2 and 4, Lemma S with N from the product kernel's
node sums), with Lemma O's rule when a
partial sum could overflow; in any order, Lemma A. Both include the flush
term moved_j. (No finite bound sends the row to the step-by-step program.)
Then

    |s'_j - s_j| <= A_j = acc_j + rnd(|s_j| + acc_j, |s_j| - acc_j).

The computed scale lies in the interval I of the step-by-step program's step
2, and so does sigma, so it is within B = max(I.hi - sigma, sigma - I.lo) of
sigma. The logit rounds s'_j sigma' + b_j once; with flush, a subnormal b_j
may be read as zero (beta_j = |b_j| then, else 0). Before the rounding, its
distance from l_j is at most C_j = (|s_j| + A_j) B + |sigma| A_j + beta_j, so
when the computed logit is finite

    |l'_j - l_j| <= L_j = C_j + rnd(|l_j| + C_j, |l_j| - C_j).

It is finite when |l_j| + C_j < Theta. Otherwise, if l_j + C_j < 0, the value
before the rounding is negative and may overflow toward -inf only: the key
*may vanish* (its logit is -inf or within L_j, and a -inf logit gives weight
exactly 0). If l_j + C_j >= 0, an overflow toward +inf would make the row NaN:
the row goes to the step-by-step program. At least one key must not vanish,
so the row's maximum is finite. (A key at the format's lowest value, as
Hugging Face pads, vanishes this way; its weight is negligible anyway.)

**Step 4, the maximum.** m' is the largest computed logit, and each l'_j lies
in [l_j - L_j, l_j + L_j] or is -inf, so m' lies in [m_lo, m_up] with
m_up = max_j (l_j + L_j) and m_lo = max (l_j - L_j) over the keys that cannot
vanish. Let Lam_up = m_up - m and Lam_lo = m - m_lo, both >= 0 (m_up >= l_k + L_k >= m for
the key k of m, and l_j - L_j <= m). The common factor K = exp(m - m') lies in
[exp(-Lam_up), exp(Lam_lo)].

**Steps 5 and 6, the weights.** The shift l'_j - m' lies in
[-(m_up - l_j) - L_j, 0], and its rounding x'_j keeps it at most 0 (rounding
is monotone and keeps 0). So exp(x'_j) = K w_j exp(rho_j), where rho_j is the
logit's error plus the shift's rounding:

    |rho_j| <= R_j = L_j + rnd((m_up - l_j) + L_j, (m_lo - l_j) - L_j).

The shift is at most 0, so it can overflow only toward -inf, and only when
(m_up - l_j) + L_j reaches Theta: then the key may vanish too. A key that may
vanish has weight exactly 0 (t_j = -1 below) or the bound above, so R_j is
raised to at least log 2 (1 in base-2 units with exp2), which makes
T_j = exp(R_j) - 1 >= 1 >= |t_j|.

The library exp returns w'_j = exp(x'_j) + h_j with
|h_j| <= eps_j = rnd(exp(Lam_lo) w_j exp(R_j), exp(-Lam_up) w_j exp(-R_j)),
since exp(x'_j) lies between those two values. With T_j = exp(R_j) - 1 and
t_j = exp(rho_j) - 1, |t_j| <= T_j (because 1 - exp(-R) <= exp(R) - 1), and

    w'_j = K w_j (1 + t_j) + h_j.

Since x'_j <= 0, w'_j is at most 1 + r_e + a_e (r_e and a_e: Lemma 1's units
for exp). Per key and per row:

    E_j = P_j T_j + H_j,   H_j = eps_j exp(Lam_up) / W   (at least |h_j| / (K W)),
    tau = sum_j E_j,   D = 1 - tau,   D' = 1 + tau,   g = exp(Lam_up) / (W D).

**Step 7, the ratio of exact sums.** Let G = sum_j w'_j and
Q' = (sum_j w'_j v_jc) / G. Since sum_j w_j (v_jc - o_c) = 0 by the definition
of o_c,

    G (Q' - o_c) = sum_j (K w_j t_j + h_j) (v_jc - o_c).

Divided by K W, with |v_jc - o_c| <= |v_jc| + O_c, the right side is at most
U_c + O_c tau, where U_c = sum_j E_j |v_jc|. And
G / (K W) = sum_j (P_j (1 + t_j) + h_j / (K W)) lies in [D, D']. So for D > 0,

    |Q' - o_c| <= E_Q = (U_c + O_c tau) / D,

and in the same way (sum_j w'_j |v_jc|) / G <= H = (S_c + U_c) / D with
S_c = sum_j P_j |v_jc|. G lies in [G_lo, G_hi] = [W D exp(-Lam_up), W D' exp(Lam_lo)],
so 1 / G <= g = 1 / G_lo.

**Step 8, the implementation's sums.** The row sum Z' adds the w'_j, so
|Z' - G| <= G E_Z. The value sum adds the products w'_j v_jc, each rounded to
the accumulator format (or fused). With flush, a value that is subnormal in F
may be read as zero, which moves the exact sum by at most G phi_c, where
phi_c = (F_c + UF_c) / D with F_c = sum_j P_j |vf_jc| and
UF_c = sum_j E_j |vf_jc| over those values vf (by step 7's bound on each w'_j);
phi_c = 0 without flush. The value sum is within G e_N of G Q'.

*Normal accumulation* (Lemma S over the kept keys in key order, the lanes by
key index, divided by G >= K W D). The row sum's terms w'_j are nonnegative, so a node's sum is at
most G and at most K W C_k with C_k = sum over i <= k of (P_i + E_i) (step 7's
bound on each w'_i). In index order the nodes are the running sums, so with
X = sum over k >= 2 of min(C_k, D),

    E_Z = Growth(n - 1) ( r (X / D + min(n - 1, 3)) + (n - 1) a' g ),

plus Growth(n - 1)(r + n a g) when the accumulator is narrower than exp's
format (each weight is rounded as it enters). The lanes are covered by the
min(n - 1, 3) term: a lane node sums a subset of the terms up to some key i,
at most C_i, and distinct lane nodes end at distinct keys i >= 2, so the lane
nodes add up to at most X / D; the combination adds at most three nodes of
size up to G. For the value sums, a node's sum of w'_i v_ic is at most
K W (|sum over the node of P_i v_ic| + sum over the node of E_i |v_ic|), so

    e_N = Growth(n - 1) ( r (H + Pi_c) + phi_c + (n a + (n - 1) a') g ),
    Pi_c = (Nd_c + (n - 1) U_c) / D,

where Nd_c bounds the sum over the nodes (index order and lanes 2 and 4, the
largest) of |sum over the node of P_i v_ic|, and each term is in at most
n - 1 nodes. Nd_c comes from the exact node sums (`product_fast_nodes`, every
node exact) of the row's P rounded up to integers times the column, their
distance from P (the ball's width and the rounding) as radii:
Nd_c = (N - 2 mass) / 2 + (n - 1) rad. Lemma O
(e) needs mass + B < Theta: for the row sum G_hi (1 + E_Z) < Theta (its terms
are nonnegative, so their mass is their sum, at most G_hi, and B is at most
G_hi E_Z), and wmax_v V_c + G_hi e_N < Theta for each value sum (every term
and node is at most those, as in `foundations.md`, Lemma O); otherwise the row
goes to the step-by-step program.

*Any order* (Lemma A, divided by G):

    E_Z = min( gamma(n) + kappa(n) g,
               Growth(n) ( r + (n a + (n - 1)(r Theta + a')) g ) ),
    e_N = min( gamma(n) H + kappa(n) g,
               Growth(n) ( r H + (n a + (n - 1)(r Theta + a')) g ) ) + phi_c,

each form where it exists, when G_hi + Growth(n - 1)(r n G_hi + n a + (n - 1) a')
and wmax_v V_c + Growth(n - 1)(r n wmax_v V_c + n a + (n - 1) a') are below
Theta (an implementation without overflow exists); otherwise the row goes to
the step-by-step program.

Its rounding to F (`values.output`) acts on a value of magnitude at most
Y = G_hi (Q + e_N), with Q = O_c + E_Q, and at least G_lo (O'_c - E_Q - e_N).
It adds at most d = rnd(Y, G_lo (O'_c - E_Q - e_N)), which is at most
G d / G_lo. So the rounded value sum N' satisfies |N' - G Q'| <= G E_N with
E_N = e_N + d / G_lo.

**Step 9, the division.** N' / Z' - Q' = ((N' - G Q') - Q' (Z' - G)) / Z', and
|Q'| <= Q. So for E_Z < 1, the quotient before its rounding is within

    A = E_Q + E_R,   E_R = (E_N + Q E_Z) / (1 - E_Z),

of o_c, that is in [o_lo - A, o_hi + A]. Lemma 1 widens that interval for
`normalize`, and Lemma 3 gives the tolerance. []

**The integer products.** P_j and E_j (E_j + HW_j when the weights are
rounded, `attention_algorithms.md`) are rounded up to integers at one unit per
row and multiplied by the value block's magnitudes in one integer matrix
product per tile (with flush, also by the magnitudes of the flushable values;
with uncertain values, also by their radii). So U_c, S_c, F_c, UF_c and D_c
are upper bounds, and V_c = sum_j |v_jc| is exact (with uncertain values,
sum_j (|v_jc| + rv_jc)). Rounding up keeps every bound valid at any unit,
since all entries are nonnegative. For Nd_c, P_j's lower end is also rounded
down at that unit.

The units are first 2**uP, 62 bits below the largest P_j (under the normal
standard min(62, 126 - b - 2 ceil(log2(n + 1))) bits, b the bits of V transposed's
integers, so that the node sums of `product_fast_nodes` fit its 128-bit words;
below 24 bits the row goes to the step-by-step program), and 2**uE, 62 bits
below the largest E_j, which keeps the integers short. An entry below its unit
counts as a whole unit, so the integers exceed the exact sums by at most
2**uE V_c in U_c, 2**uP V_c in S_c, 2**uP VF_c in F_c and 2**uE VF_c in UF_c
(VF_c = sum_j |vf_jc| <= V_c), 2**uP RV_c in D_c (RV_c = sum_j rv_jc <= V_c),
and 3 (n - 1) 2**uP V_c in Nd_c. For Nd_c = (N - 2 mass) / 2 + (n - 1) rad,
each term weighs at most n - 1 in N (Lemma B with blocks of one term), the
number of nodes that can hold it. A term at index s >= 2 is in the nodes
s_s .. s_n, n - s + 1 of them, and the first term in n - 1; per lane, the
combination's nodes add at most the number of other lanes, so the total stays
at most n - 1. So (N - 2 mass) / 2 grows by at most (n - 1)
units per |v_jc|. Each term of rad, rad_j |v_jc| + (P_j + rad_j) rv_jc, grows
by at most 2 units per |v_jc| and 3 per rv_jc, since P_j grows by less than
one unit and rad_j by less than two.

Let h, pi and f be the factors of H, Pi and phi_c in e_N, and w that of H in
E_Q:

- normal: r Growth(n - 1), r Growth(n - 1), Growth(n - 1), 0;
- any order: the larger of gamma(n) and Growth(n) r (of the forms that
  exist), 0, 1, 0;
- online softmax: gamma, 0, 1, and r_W when the weights are rounded (else 0),
  with H and phi_c grown by 1 + w.

By steps 7 and 8 the integers then add at most X_c / D to E_Q + e_N and to the
enclosure of o_c, with

    X_c = ((1 + c) 2**uE + c 2**uP) V_c + f' (2**uE + 2**uP) VF_c + 2**uP RV_c,
    c = w + h (1 + w) + 3 (n - 1) pi,   f' = (1 + w) f.

Since E_Q >= U_c / D, that is at most 2**-16 E_Q when 2**16 X_c <= U_c in
every column. For any other row the code computes the row's bound again with
one unit 2**u for P and E, the largest power of 2 at most
2**-16 E_min / (2 + 2c + 2f'), where E_min is the smallest lower end of the
row's E_j, and intersects each output's two intervals (both contain the
quotient). At 2**u the row passes the check, since U_c >= E_min V_c. And
E_min >= min_j H_j > 0, which is at least Lemma 1's absolute part for exp
divided by W <= n, so 2**u is at most about p - emin + log2(n) + 20 bits
below 1 (p and emin of the compute format). The attention kernel's second
pass takes no unit: it sums P_j's and E_j's upper ends times the values
exactly, with P_j's lower end at least 0 in rad, and the node sums of Nd_c over
P_j's upper ends exactly (every unit-rounded integer is at least the end it
rounds, so these sums are at most the finer unit's). With uncertain values
the exact sums take |v_jc| + rv_jc in S_c, U_c and V_c, add
(2 P_j - max(P_j's lower end, 0)) rv_jc to rad and give D_c = sum_j P_j rv_jc,
rv_jc the packed radius (at least the radius). For V with flagged or two-word
slots (values outside one word's range, or not finite) these exact sums are
the kernel's only pass; a value that is not finite counts as 0 there, its
column's reference being not finite. Before these exact sums every end of P_j
and E_j more than 4096 bits below the largest is moved outward to that power
of 2 (an upper end up to it, a lower end down to 0), so the integers stay a
few thousand bits wide where an end is as small as exp(-2**40); each bound
still holds (a wider enclosure of P_j and E_j), and what the move adds is
below every format's least value.

**Validity conditions.** The closed form covers a row only when all of these
hold but 4, each checked with the bounds above; otherwise the row gets the
step-by-step program.

1. Every kept key's score has a finite accumulation bound (normal: Lemma O;
   any order: Lemma A).
2. The score rounding: |s_j| + A_j is at most F's largest value.
3. The scale: the step-by-step program's step 2 is finite.
4. Masked keys whose score is finite (their logit is computed before the -inf
   bias): each has a finite accumulation bound, and
   (|s_j| + A_j) |I| (1 + r) + a is at most F's largest value. It is checked at
   the largest |s_j|, M_j and moved amount among them, since the bound grows
   with each. Where it fails, or a masked key's score is not finite, the row
   keeps the closed form's bound and its outputs and logsumexp may be NaN
   (Masked keys).
5. The logits: none may overflow toward +inf, and at least one key cannot
   vanish (steps 1 to 3).
6. The shifts may overflow toward -inf only (they are at most 0); such keys
   may vanish (steps 5 and 6).
7. D > 0 and E_Z < 1.
8. The row sum: Lemma O (e)'s or Lemma A's condition (step 8).
9. Each value sum: Lemma O (e)'s or Lemma A's condition (step 8), and Y + d is
   at most F's largest value.
10. `normalize`'s rounding satisfies Lemma 1's hypothesis.
11. logsumexp's conditions (below).

Rows with no kept key and NaN rows use the step-by-step program. Where a
query or key row holds NaN or an infinity, IEEE's rules on the logits decide
the kept keys first, as in every implementation: such a score is NaN or an
infinity by the product's rules (times the scale's sign); at a key the row
does not mask, a NaN or +inf logit makes the row NaN and a -inf one removes
the key; a masked key takes no part whatever its score (Masked keys: its NaN
is the flag's). A column with a non-finite value among the kept keys has a
non-finite reference and a NaN tolerance (with uncertain inputs, any value or
flag); the row's other columns keep theirs, and a non-finite value at a key
the row does not keep only adds the NaN flag to its column. With
online softmax (`attention_algorithms.md`) the sums keep the any-order bounds
of that file under both standards.

### The step-by-step program

The attention kernel's `att_program_row` runs the model on intervals of
dyadics at the bounds' precision. The invariant is that each interval contains
every value that any such implementation can hold at that step. It holds step
by step:

1. Scores: the exact dot product and the magnitude sum are integers over one
   shared exponent (the product kernel's), so Lemma 2 with n = E gives an
   interval for the accumulated score, and Lemma 1 widens it for the rounding
   to F.
2. The scale: the given scale, or an enclosure of 1/sqrt(E) (1/root exactly
   when E is a perfect square), widened by Lemma 1 with c ulps.
3. Logits: the interval product and sum (outward rounded) contain every
   score * scale + mask; one widening by Lemma 1 follows.
4. The maximum: an implementation's m is one of its logits, so it lies in the
   interval [max of lower ends, max of upper ends].
5. and 6. Interval subtraction, then exp of the interval's ends, which
   encloses exp of the interval because exp is increasing (its ends bounded by
   Lemma X; an end whose argument is below -2**40 is taken as 0 for the lower
   end and exp(-2**40) for the upper one, since exp is positive and
   increasing), then Lemma 1 widenings.
7. The sum: interval sums of the weights, with magnitude sum
   sum max(|lo_j|, |hi_j|), then Lemma 2 with n the number of kept keys
   whose weight can be nonzero (a key that surely vanishes adds an exact 0,
   Masked keys). With online softmax the weights' ends are added rounded
   outward at 64 bits beyond the precision (lower ends down, upper ends and
   magnitudes up), before the rounding to the precision.
8. Weighted sums: for weights w_j in [lo_j, hi_j] and exact values v_j, the sum
   sum_j w_j v_j lies between sum_j (v_j >= 0 ? lo_j : hi_j) * v_j and
   sum_j (v_j >= 0 ? hi_j : lo_j) * v_j, and the magnitude sum is
   sum_j max(|lo_j|, |hi_j|) |v_j|. The kernel takes the weight ends as
   two-word integers at a unit 2**u, u 126 bits below the largest end's
   magnitude, the lower ends rounded down and the upper ones up, so each
   weight's interval contains the one before and its magnitude bound is at
   least the one before; the values are V's integers, and the sums of products
   are exact in six-word integers. Where the units could matter,
   2**u (2 sum_j |v_jc| + sum_j rv_jc) above 2**-(p + 40) of the larger
   magnitude of the column's two sums (p the output format's precision), and
   for V with flagged or two-word slots, the row's sums are the exact ones of
   the ends times the values instead (with uncertain values, the deviation sum
   too), with no unit, after the ends more than 4096 bits below the largest
   are moved outward to that power of 2 as in the closed form. Lemma 2 at
   step 7's n then applies to whichever terms the implementation's weights
   give, followed by Lemma 1 for `values.output`.
9. Division: the interval quotient (the row-sum interval excludes 0, or the
   bound is infinite), widened by Lemma 1.
10. Lemma 3.

Where a kept key's logit or shift rounding may overflow toward -inf only (its
interval below 0 reaches the threshold), the key may vanish, and the program
takes the rounding's finite outcomes, the maximum's lower end over the keys
whose logit cannot be -inf, and the key's weight interval's hull with 0, or
[0, 0] where the rounding has no finite result; a row whose kept keys may all
vanish has no bound (`attention_algorithms.md`, Keys that may vanish).

For masked keys whose inputs are finite, the kernel bounds the score and the
logit (with bias 0) the same way, only to detect overflow: if one could
overflow, or a masked key's score is not finite, the row keeps the program's
bound and its outputs and logsumexp may be NaN (Masked keys). Keys the row
does not mask with a non-finite score are removed or make the row NaN by
IEEE rules in every implementation.

Every step only enlarges what it encloses, so the final interval contains every
possible output, and Lemma 3 gives the tolerance. []

**Flush to zero.** With `flush_to_zero=True`:

- Every rounding uses Lemma 1 with the 2**emin term, and every accumulation uses
  Lemma 2 with eta = 2**emin.
- An input that is subnormal in F may be read as zero. For the scores, as in
  `matmul.md`, the exact sum moves by at most the products with such a factor.
- A value v that is subnormal in F lies in the hull of {0, v}. A weight in
  [lo, hi] times it lies between min(lo, 0) * v and max(hi, 0) * v when v > 0,
  and between max(hi, 0) * v and min(lo, 0) * v when v < 0. The code adds those
  terms with exact integer dot products.
- A float mask value that is subnormal in F becomes the interval hull {0, b}.
- Intermediate results are produced already flushed, so reading them changes
  nothing further.

**Special rows.** A row with no kept key has every weight exactly 0, so every
implementation returns exactly 0, the reference, or NaN where a key's term
meets a non-finite factor (Masked keys). Its outputs have tolerance 0, with
the NaN flag where Masked keys gives it. Outputs whose reference is NaN or
infinite have a NaN tolerance (Lemma 3).

### Fully padded rows

When every kept key carries the same large mask value (padding at -1e9, say),
the logits' rounding wipes out the scores, so the closed form fails (D <= 0),
and the step-by-step program's intervals are very wide. For the standard
algorithm, the kernel's `padded_row` bounds such a row (any row the closed form does not
cover) from the weights alone, for both standards.

**Conditions.** Every kept key's computed logit is below Theta (no +inf), and
at least one is finite (its value before rounding is below Theta in
magnitude), so the row's maximum m' is finite; the masked keys' check as
above. The scores use Lemma A's bound with Lemma 1 for their rounding. n
counts the kept keys whose weight can be nonzero, and a key that may vanish
is never the finite one (`attention_algorithms.md`, Keys that may vanish).

**Bound.** The key of m' has shift exactly 0 and weight exp(0) = 1 within c
ulps, so G = sum_j w'_j >= g_min = 1 - r_e - a_e > 0, and every weight is
between 0 and wmax = 1 + r_e + a_e. Before the sums' roundings, the ratio
Q' = sum_j w'_j v_jc / G is a convex combination of the values, so it lies in
[lowest_c, highest_c], their least and greatest over the kept keys (and 0,
with flush, since a value may be read as zero), and |Q'| <= vmax_c, the larger
magnitude. Every node of either sum is at most G vmax_c (value sum) or G (row
sum), in every order, so Lemma S gives, relative to G,

    e_N = Growth(n - 1) ( r n vmax_c + (n a + (n - 1) a') / g_min ),
    E_Z = Growth(n - 1) ( r (n - 1) + (n - 1) a' / g_min ),

plus Growth(n - 1)(r + n a / g_min) in E_Z with a narrower row-sum
accumulator. Lemma O (e) needs n wmax (vmax_c + e_N) and n wmax (1 + E_Z) below
Theta. The value sum's rounding to F moves it by at most d, Lemma 1's error r
at n wmax (vmax_c + e_N) (`lean_error`), which is at most d / g_min
relative to G. As in step 9, the quotient before
its rounding lies in

    [lowest_c - E_R, highest_c + E_R],   E_R = (e_N + d / g_min + vmax_c E_Z) / (1 - E_Z),

then Lemma 1 for `normalize`. The row's tolerance is the intersection of this
interval with the step-by-step program's, both of which hold. []


## logsumexp

**The exact value.** For each output row, `logsumexp` is log(Z) with
Z = sum_j exp(l_j) over the kept keys, in the format `default_compute(dtype)`
(float32 for formats narrower than float32, as the fused kernels return it).
Masked keys take no part, as in the output. The other non-finite logits
follow IEEE arithmetic on exact values: a NaN logit gives NaN, a +inf logit
gives +inf, and a row with no kept key has Z = 0 and gives log(0) = -inf, as
`torch.logsumexp` does. (PyTorch's CPU flash kernel returns 0 for such a row;
that is a kernel deviation, not the reference.)

**The reference output.** With one kept key, log(Z) is that key's logit:
exact when sigma is rational (or the score is 0), and otherwise
score / sqrt(E) plus a dyadic mask value, irrational. With more kept keys,
group them by equal logits s_g (counts n_g), so Z = sum_g n_g exp(s_g). If
log(Z) = a were algebraic, exp(a) - sum_g n_g exp(s_g) = 0 would be a
vanishing combination of exponentials of algebraic numbers; by
Lindemann-Weierstrass, if a differs from every s_g all coefficients are 0,
which is false, and if a = s_h then 1 - n_h = 0 and every other n_g = 0, which
leaves one key. So log(Z) is transcendental: never a rounding boundary and
never 0. The attention kernel encloses it first from Lemma D's sums:
log(Z) = m + log(Z e**-m), with Z e**-m in [Zp, Zp + n h] 2**-F (with mask
values, m' and its weights instead), m enclosed from S_top and sigma's
enclosure, and the log of each end by Lemma Y (log is increasing). Values it
leaves undecided are refined with the outputs: the same sum at precision p
without tables (the weights as above, m' = l_r by Logit differences) and Lemma
Y on each end of Z', from
twice the first precision, doubling, cap 2**14 bits, until both ends round to
the same value, the interval excludes 0 (at the cap, an interval containing 0
whose ends round alike is accepted, as for the output) and it fits an exact
entry. A row with one kept key has the logit itself: when sigma is rational
or the score is 0, the exact rational sigma s + b rounded once
(`floatfmt_encode_ratio`); otherwise s / sqrt(E) + b, irrational and nonzero,
enclosed by Logit differences at p bits and refined the same way.

A round that leaves log(Z) undecided runs the output's sign test, with the
same candidates b. log(Z) - b has the sign of

    T = Z e**-b - 1 = sum_g n_g e**(s_g - b) - 1.

A group with s_g = b exactly contributes n_g - 1 instead of n_g e**0, and so
does nothing when n_g = 1. The equality is checked exactly: s_g - b =
sigma s + (b_g - b) has sign 0 by Logit differences. T is nonzero, since
log(Z) is not b. Each s_g - b is enclosed by Logit differences at p bits.
Since Z >= e**(s_g) for every group and Z <= n e**(max s_g), log(Z) - b lies
between the largest lower end and the largest upper end plus log n, rounded
up. The kernel forms T where every upper end is at most 1 and the largest
upper end plus log n is at least 0: each term is then at most n_g e, and
1 + T = Z e**-b is at least e**(max s_g - b), about 1/n or more, so the exact
1 + T below stays within a few binades of 1. Elsewhere, where the largest
lower end is positive or the largest upper end plus log n negative, that
interval is the offset in `foundations.md`'s lemma at once, however wide
log(Z)'s enclosure (a fully padded row whose logits lie far above its mask
value, b being that value). Otherwise the round leaves it to the next: the
enclosures narrow, so some lower end turns positive, or every upper end falls
to 1 with the largest at least -log n, or the largest upper end plus log n
falls below 0. The kernel encloses
each term n_g e**(s_g - b) by Lemma X at both ends of s_g - b's enclosure
(past -2**40, [0, n_g exp(-2**40)]), and the group
that joins the -1 (the one with s_g = b, else the one whose term's upper end
is largest) as n_g expm1(x) + (n_g - 1), with expm1(x) = e**x - 1 enclosed by
Lemma X's ends at p + max(0, -top(x)) + 8 bits less 1, exactly, which keeps
its relative precision for small x. Once T's enclosure excludes zero,
log(Z) - b = log1p(T) is the offset in `foundations.md`'s lemma. Where
|T| < 2**-p it is enclosed by T - 2 T**2 <= log1p(T) <= T, both ends rounded
outward at p bits: for 0 < T, log1p(T) >= T - T**2 / 2; for -1/2 <= T < 0,
log1p(T) >= T / (1 + T) = T - T**2 / (1 + T) >= T - 2 T**2; and T - 2 T**2
increases below 1/4, so the ends are those at T's ends. Otherwise each end
of T is taken by its own size: an end of magnitude below 2**-p by the same
bounds (T - 2 T**2 rounded down at a lower end, T rounded up at an upper
one), and an end of magnitude at least 2**-p by Lemma Y at that end of
1 + T (an exact sum, at most about 2p + 64 bits long since the end is at
least 2**-p; Lemma Y keeps y - 1 exact, so a small T keeps its relative
precision). log1p is increasing, so the two ends enclose it. When the only
terms left are those past -2**40 (the group at b has n_g = 1), T is positive
and below 2**-(LIMIT + 1), and its sign decides by the tiny-offset rule.
Example: mask
(0, -30000) gives log(Z) = log(1 + e**-30000), which rounds toward +inf to the
smallest positive value.

**The tolerance.** Model: the output's steps 1 to 7 (scores, scale, logits,
row maximum m, shift, exp, row sum), then log of the row sum within c ulps
(`logsumexp.log`) and m plus that logarithm rounded once (`logsumexp.add`),
then the rounding to the logsumexp format (Lemma 3).

With the closed form (same notation): log Z' = log K + log W + log(Z' / (K W))
and m' + log K = m, so m' + log Z' = logsumexp + log(Z' / (K W)). By steps 7
and 8, Z' lies in [G (1 - E_Z), G (1 + E_Z)] and G / (K W) in [D, D'], so
Z' / (K W) lies in [Z_lo, Z_hi] with

    Z_lo = D (1 - E_Z),   Z_hi = D' (1 + E_Z)

(with online softmax, `attention_algorithms.md`'s
Z_lo = D (1 - gamma) - alpha, Z_hi = D' (1 + gamma) + alpha, alpha its
absolute terms over K W),

so for Z_lo > 0, |log(Z' / (K W))| <= lambda = max(-log Z_lo, log Z_hi). The
library log's argument has |log Z'| <= log n + max(Lam_up, Lam_lo) + lambda
(W lies in [1, n] and log K in [-Lam_up, Lam_lo]), so before the last rounding
the result is within E_log = lambda + rnd(log n + max(Lam_up, Lam_lo) + lambda)
of logsumexp. Lemma 1 widens [lse_lo - E_log, lse_hi + E_log] for `add`, and
Lemma 3 gives the tolerance.

The step-by-step program runs this on intervals with the output's, sharing
each row's steps 1 to 7: m lies in [max of lower ends, max of upper ends];
log of the row-sum interval is enclosed by its ends (Lemma Y at each end: log
is increasing; the interval is positive, since the largest weight is about
1), then widened by Lemma 1;
interval addition and
Lemma 1 for `add`. Masked keys are checked as above (the NaN flag). Rows whose
reference is not finite have a NaN tolerance.

## Results as inputs

When query, key, value or a float mask is an earlier result, the
implementation computes the model on its stored values q', k', v', b', each
within its radius (rq, rk, rv, rb: the earlier tolerance with the same
arguments) of the values q, k, v, b that the reference uses.

**Claim.** `tolerance()` bounds |output - ref| (and logsumexp's distance) for
every such stored input.

**Proof.** Each step of the model takes the stored inputs into account as
follows.

- No bound: an element whose radius is inf or NaN bounds nothing, and nor
  does one whose stored value may be non-finite (the earlier result's flags):
  exp and division can turn a non-finite value back into a finite one
  (exp(-inf) = 0), so such a value may move a finite output anywhere
  (`unflagged_radius` reads its radius as inf). A mask element (whose stored
  -inf may be finite, so that its key may be kept), and a query or key element
  that reaches a key the row does not mask, reach a logit the row may keep, so
  those rows' outputs and logsumexp get infinite tolerances (`row_unbounded`
  in `csrc/attention_tol.c`). A query or key element that reaches only masked
  keys (a key row the row masks, or the query row of a row that masks every
  key) moves only logits that stay -inf or become NaN: the row keeps its
  bound, and its outputs and logsumexp may be NaN (Masked keys). A value
  element at a key the row keeps reaches its column, which gets an infinite
  tolerance; at a key the row does not keep, it only adds the NaN flag to its
  column (0 times the stored value).
  Where the reference is not finite the tolerance stays NaN. Where there is
  no bound, the finite range holds every finite value and every flag is set
  (Lemma 3).
- Flush: a radius is first widened to cover 0 where its box reaches below
  2**emin of F, so a stored value read as zero stays within its box.
- Scores (step 1): the stored inputs' exact score lies within
  Delta_j = sum_e (|q_e| rk_je + rq_e |k_je| + rq_e rk_je) of s_j, and its
  magnitude sum is at most M_j + Delta_j (`matmul.md`, Results as inputs).
  The kernel adds Delta_j to the magnitude sum and to the moved amount and
  2 Delta_j to N, so every bound built on a score (the closed form's
  accumulation error, the step-by-step program's interval, the masked keys'
  check and the fully padded rows) covers the stored inputs, still around s_j.
- Mask values (step 3): b'_j lies within rb_j of b_j. The closed form adds
  rb_j to the mask value's error beta_j, uses |b_j| + rb_j in its rounding and
  overflow conditions, and takes a key's value as surely negative only when
  rb_j < |b_j|; the step-by-step program widens the mask value's interval by
  rb_j.
- Values (step 8): with the reference's probabilities P_j, the output on the
  stored values o'_c = sum_j P_j v'_jc lies within
  D_c = sum_j P_j rv_jc <= sum_j P^up_j rv_jc of o_c, so widening o_c's
  enclosure by that (P^up: the integers rounded up that bound P) encloses
  o'_c. The closed form then bounds the distance from o'_c with every |v|
  replaced by |v| + rv (each term only grows with them), and the normal
  standard's node bound N_d takes the values' radii as interval terms
  (Lemma S's remark). The step-by-step program widens each weighted sum by
  sum_j max(|lo_j|, |hi_j|) rv_jc and uses |v| + rv in its magnitude sum; the
  fully padded rows' convex combination uses the values' boxes, v - rv to
  v + rv.
- logsumexp depends only on the logits, which the score and mask terms cover.

The rest of each bound is unchanged. The attention kernels take the same
terms: Delta_j from the radii's own operands (`product_deviation`), rb_j as a
float64 radius per key, and the values and their radii as integers at one
exponent, with N_d's interval terms as sum_j ((P^up_j - P^lo_j) |v_jc| +
(2 P^up_j - P^lo_j) rv_jc), the excess of (|a| + rad)(|b| + brad) over |a| |b|
for a = P^up with rad = P^up - P^lo. []

`normal()` runs the normal answer on the earlier results' `normal()`.

## The normal answer

`normal(compute=None, flush_to_zero=False)` computes the model in the
compute format F with every sum a normal accumulation in index order, by
the attention kernel (`csrc/attention_normal.c`), for every key in key order,
as an implementation that computes the terms of the keys a row does not keep
(Masked keys: their weight is exactly 0, and their NaN the flags'):

1. each score: the E products q_e k_je rounded to F, added in index order,
   each addition rounded, with Lemma O's slots where index order overflows;
2. the scale: 1/sqrt(E) or the given scale, correctly rounded to F;
3. each logit: score times scale plus the mask value, rounded once (exact
   rational arithmetic, then one rounding); a masked key's logit is its score
   times the scale plus -inf: -inf, or NaN for a NaN or +inf product;
4. the maximum m'; a NaN or +inf logit makes the row NaN (logsumexp NaN or
   +inf), and a row with no finite logit has every weight 0 (logsumexp -inf);
5. each shift l' - m', rounded;
6. each weight, exp of the shift correctly rounded to F (Lemma X's
   enclosures, refined until both ends round alike);
7. the row sum and 8. each column's sum of the products w' v, rounded, as in
   step 1;
9. each quotient, rounded;
10. the rounding to the output format; logsumexp is m' plus log of the row
    sum (correctly rounded), rounded once in F, then rounded to its format.

With flush to zero, inputs (and mask values) subnormal in F are read as zero
and subnormal results become zeros of the same sign. The arithmetic is Lemma
E's emulation in integers (`emulate.c`): each product, sum and difference
rounded once as IEEE 754 defines it, the logits and quotients as exact
values rounded once, exp and log refined as above, the scale held in F by
`normal_scale` (a rational scale rounded once, 1/sqrt(E) refined until both
ends round alike; a scale past F's range is an infinity, or NaN in a format
without one, and each score times it then follows IEEE 754's special values).
The test suite compares it with the same steps rounded one by one by MPFR.
