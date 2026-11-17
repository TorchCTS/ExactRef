# scaled_dot_product_attention backward: proofs

This file proves what `exactref.scaled_dot_product_attention.backward` claims:
every returned gradient is the exact value correctly rounded (or the call
raises `ReferenceUnavailableError`); `tolerance()` bounds every
implementation that rounds only at the required boundaries with normal
accumulation, and `tolerance(standard="any_order")` every such implementation
in any order without overflow; `normal()` is the model computed with normal
accumulation. Lemmas 1 to 3 and the normal standard's lemmas are in
`foundations.md`; the forward's exact value and its notation (kept keys, logits
l_j, weights w_j = exp(l_j), Z = sum w_j, sigma) are in
`scaled_dot_product_attention.md`.

## The exact values

For one output row with kept keys K, weights P_j = w_j / Z for j in K. With
the output gradient row g (`grad`):

    dP_j = g . v_j,   D = sum_(j in K) P_j dP_j,   dS_j = P_j (dP_j - D)   (j in K),

and the gradients are

    grad_value[j]  = sum over rows keeping j of P_j g,
    grad_query     = sigma * sum_(j in K) dS_j k_j,
    grad_key[j]    = sigma * sum over rows keeping j of dS_j q,

each summed over every output row that uses that value row, key row or query
row (broadcast copies and grouped-query heads). A key the row does not keep
(masked, or removed by a -inf logit, as in the forward) takes no part: it adds
no term to D or to any gradient, whatever its dP, score or value. A row with
no kept key adds nothing. These are the derivatives of the forward's exact
value: softmax's Jacobian is diag(P) - P P^T over the kept keys, so the
gradient of their logits is dS, and the logits are sigma * q . k_j + b_j.

Non-finite values follow IEEE arithmetic on exact values through these
formulas, as in PyTorch's math path over the kept keys: a NaN row of the
forward has P = NaN at every key whose logit is not -inf (its other keys take
no part); an infinite or NaN product enters the sums by IEEE multiplication
and addition of the special values. The reference kernel applies the rules
case by case. Two facts keep the case list short:

- **Any non-finite dP at a kept key makes D non-finite.** P_j is positive
  there, and a positive P times inf is inf and x * NaN is NaN. So a row's dS
  is either finite for every kept key or non-finite for every kept key.
- **A non-finite query or key entry meets no finite term.** If q_e is
  infinite or NaN, every score q . k_j is infinite or NaN (q_e k_je is NaN
  when k_je = 0). So every logit is -inf (the key takes no part), +inf or
  NaN (the row is NaN): the row keeps no key and adds no term unless it is a
  NaN row. A key with a non-finite entry likewise never has a finite logit,
  so no row keeps it.

An exact zero is +0, or -0 when rounding toward -inf; an empty sum is +0.

## The reference output

Every finite input is a dyadic rational, so dP, the scores and the products
with q, k and g are exact integer sums. Only P and dS can be
irrational.

**Which terms are rational.** Group a row's kept keys by equal logits s_g
(compared exactly, as in the forward: as rationals when sigma is rational, and
as (score, mask) pairs when sigma = 1/sqrt(E) is irrational). Let n_g be the
group sizes and S_g the sums of dP over a group.

- *Uniform rows* (one group): P_j = 1/n and D is the mean of dP, so
  dS_j = (n dP_j - sum dP) / n**2, rational.
- *Other rows:* P_j is transcendental. If P_j = x were algebraic, then
  e**s_g(j) - x * sum_h n_h e**s_h = 0, an algebraic combination of exponentials
  of distinct algebraic numbers. By the Lindemann-Weierstrass theorem every
  coefficient is 0, but the coefficient of e**s_g is 1 - x n_g and that of any
  other group is -x n_h, which cannot all vanish.
- In such a row, dS_j is 0 or transcendental. Write
  dS_j = N_j / Z**2 with N_j = sum_h e**(s_g + s_h) (n_h dP_j - S_h), g = g(j),
  and Z**2 = sum_(h,h') n_h n_h' e**(s_h + s_h'). If dS_j = x is algebraic,
  N_j - x Z**2 vanishes coefficient by coefficient. The exponent 2 s_max arises
  in Z**2 only from (max, max), with coefficient n_max**2 > 0, and in N_j only
  if j is in the largest group; likewise 2 s_min. A row with two or more groups
  has j outside one of them, so x n**2 = 0 for that group: x = 0.
- dS_j = 0 exactly when every group has mean dP equal to one value m and
  dP_j = m: dS_j = 0 means dP_j = D, that is
  sum_h (n_h dP_j - S_h) e**s_h = 0, and by Lindemann-Weierstrass every
  coefficient is 0. The reference kernel (`csrc/attention_gref.c`) checks
  this with integers.

A product with an exactly zero factor is exactly 0. So for each output
element the kernel knows whether every term is rational. Its rational part
is summed exactly as a fraction (a uniform row's terms over n or n**2, added
and reduced by `rational.c`), and the number of irrational terms is counted.
An element with no irrational term is that exact rational, times sigma for
grad_query and grad_key, rounded once. With sigma = 0 every finite element is
exactly 0.

**The first enclosure.** The kernel encloses every other element once,
from integer bounds on the weights. For a row with kept keys K, let
w_j = exp(l_j - m') for a reference logit m' of the row (the forward's: its
extreme score with its largest mask value, `scaled_dot_product_attention.md`,
Lemma D), W = sum_K w_j, and let integers p_j and h_j >= 0 satisfy
p_j <= 2**F w_j <= p_j + h_j, with F = 191. Lemma D's digit tables give them:
h_j = 3 z_j - 1 for z_j >= 1 nonzero digits, 0 for none, and p_j = 0, h_j = 1
for a key whose mask value lies 133 or more below the row's largest (a capped
key, weight below 2**-F). A call without tables takes them from Lemma X at
both ends of each logit's enclosure, as the forward does. With
Wp = sum_K p_j and H = sum_K h_j, Wp <= 2**F W <= Wp + H, and for a row with
Wp > 0:

- P_j = w_j / W lies in [p_j / (Wp + H), (p_j + h_j) / Wp], and in [0, 1].
- dS_j = P_j (dP_j - D) = w_j X_j / W**2 with
  X_j = sum_{k != j} w_k (dP_j - dP_k), since sum_k P_k = 1. Write
  X_j = dP_j A_j - B_j with A_j = W - w_j and B_j = sum_{k != j} w_k dP_k.
  Then 2**F A_j lies in [Wp - p_j, Wp + H - p_j - h_j], and 2**F B_j in
  [Bl - b_j, Bh - c_j], where b_k and c_k are the least and the largest of
  p_k dP_k and (p_k + h_k) dP_k, and Bl and Bh their sums over K. These are
  exact integers (the dP_k are exact), the row's sums are formed once, and
  the row costs O(n). So 2**F X_j lies in [x_l, x_h], x_l being the least of
  dP_j times either end of 2**F A_j, minus Bh - c_j, and x_h the largest,
  minus Bl - b_j. With u = 2**F w_j in [p_j, p_j + h_j] and Omega = 2**F W
  in [Wp, Wp + H], dS_j = u (2**F X_j) / Omega**2 lies between
  p_j x_l / (Wp + H)**2 (when x_l >= 0, else (p_j + h_j) x_l / Wp**2) and
  (p_j + h_j) x_h / Wp**2 (when x_h >= 0, else p_j x_h / (Wp + H)**2).

The kernel holds these ends as integers at fixed units, lower ends rounded
down and upper ends up. W's bounds enter only as a relative factor, and the
subtraction in X_j is between exact integer ends that leave out key j's own
width: the cancellation-free form of dS (below), on the weights. Then
grad_value, grad_key and grad_query are sums over rows (and keys) of P_j g,
dS_j q and dS_j k with exact factors, so their ends are exact integer sums,
each term taking the end of its interval factor that makes it least (or
largest) by the sign of its exact factor. grad_query and grad_key are then
multiplied by sigma: exactly when it is rational, and by an outward-rounded
enclosure of 1/sqrt(E) otherwise. Each enclosure holds its element (the ends
of a sum are the sums of the ends, and a product of an interval with an
exact factor, or with a nonnegative interval, has the ends named), so when
its ends round alike and it fits an exact entry (`foundations.md`, The exact
output), it decides the element. A row with Wp = 0 (every kept key capped)
leaves the elements it reaches undecided. The weights' bounds are absolute,
within 2**-F of the reference weight, so this decides the elements whose
rows have no weight that matters far below that; the far keys and
refinement (below) take the others.

**Refinement.** The elements the first enclosure leaves undecided are
enclosed again by the reference kernel at a working precision p, from
64 bits, doubling each round. Each round streams the rows that reach a
pending element, in tiles on the threads; for each such row the kernel
encloses at precision p, once for every form below:
each kept key's argument
x_j = l_r - l_j from the reference key r, the largest logit's, by Logit
differences (as in the forward's enclosure without tables,
`scaled_dot_product_attention.md`; x_r = 0 and every x_j at least 0, so every
weight is at most 1, Z lies in [1, n] and the reference key is never far);
w_j = e**-(x_j) by Lemma X at both ends (an argument past
2**40 gets [0, e**-(2**40)]), except in the first round for a key with the
reference key's mask value and a score on the reference's side of it, whose
argument is then |sigma| D_j 2**u for its exact score distance D_j >= 0:
Lemma D's tables give its weight in [p_j, p_j + h_j] 2**-F, as in the first
enclosure, where that is at least 2**-48 (h_j <= 23, so within 2**-138 of the
weight, finer than the round's 2**-64);
Z and D = (sum_j w_j dP_j) / Z; and P, dS and the forms below only at the keys
the row's pending elements read (an element's key for grad_value and grad_key,
every key for grad_query), all by interval arithmetic
on dyadics, every operation's ends rounded outward at p + 64 bits, so each
interval contains its exact value (the inclusion property, `foundations.md`,
Directed dyadic arithmetic). An element is a sum over rows (grad_value:
P_rj g_rc; grad_key: dS_rj q_re) or over keys (grad_query: dS_rj k_je) of
an interval times an exact factor, times sigma for grad_key and grad_query
(exactly when it is rational, else by 1/sqrt(E)'s ends): each term's ends are
its interval's ends times the factor, taken by the factor's sign, and the
sum's ends the sums of the ends, rounded outward. Every dyadic keeps its own
exponent, so a term of size e**-30000 keeps its relative precision. Non-finite
inputs enter these sums as 0: an element whose terms include a non-finite
factor is not refined (its value is decided by the IEEE rules above), so this
changes no refined element. If both ends of an element's interval round to
the same output value, that is the correct rounding (rounding is monotone),
and once the interval also excludes zero and fits an exact entry
(`foundations.md`, The exact output) it is stored. Otherwise p doubles, up to
a cap of 2**14 bits, where the call raises `ReferenceUnavailableError`; at the
cap, an interval that still contains zero is accepted when both ends round
to the same value (the rounding is decided, and the exact entry is stored as
near zero).

Refinement ends for every element whose value is not a rounding boundary and
not zero: the intervals shrink to the value as p grows. Values with an
irrational part are never rounding boundaries, which are dyadic rationals.

**The cancellation-free dS.** In a row dominated by key j (the other weights
tiny), dP_j - D cancels: D's enclosure is about 2**-p |dP_j| wide, which can
exceed the value. So a round that leaves an element undecided evaluates
it again with

    dS_j = P_j (dP_j A_j - B_j),   A_j = sum_{k != j} P_k,   B_j = sum_{k != j} P_k dP_k,

the same value, since sum_k P_k = 1. A_j and B_j are the other keys' sums of
w_k and of w_k dP_k over Z, from prefix and suffix sums over the row's keys
(O(S) per row) where the row reads more than two keys, else from a pass over
the others for each; they contain no subtraction, so a tiny dS_j keeps its
relative precision. An element the first form decides keeps
its entry.

**Far keys.** If a gradient's value lies within the masked keys' weights of a
grid point (`foundations.md`, Deciding a rounding by a sign), no enclosure at
the cap decides it. For example, one row with equal scores and mask (0, -30000, -30000) gives
grad_value of key 0 equal to g / (1 + 2 e**-30000), just below g. For an
element still undecided, each row's keys are split at the round's precision
p:

- a key is *far* when its argument's lower end exceeds p, so its weight is
  below e**-p, less than 2**-p of the row's largest (whose weight is 1);
- the others are *kept*.

Any split is valid; this one makes the kept part exact where it can be. Let
G^K be the gradient of the same inputs with the far keys removed (weight 0),
and Delta = G - G^K the far keys' effect. With w_j the weights, Z_K and Z_F
the kept and far sums, Z = Z_K + Z_F, phi = Z_F / Z, and P^K_j = w_j / Z_K on
the kept keys:

    Delta P_j  = P_j - P^K_j = -P^K_j phi (kept j),   P_j (far j),
    D^K        = sum_kept P^K_j dP_j = sum_kept w_j dP_j / Z_K,
    Delta D    = D - D^K = -phi D^K + sum_far P_j dP_j,
    Delta dS_j = dS_j - dS^K_j = Delta P_j (dP_j - D^K) - P_j Delta D   (every j; dS^K_j = 0 for far j).

These are exact identities: P_j - P^K_j = w_j (Z_K - Z) / (Z Z_K), and
D = sum_kept (P^K_j + Delta P_j) dP_j + sum_far P_j dP_j. Every term carries a
factor phi or a far weight. Each row takes F, its far key of the largest
logit (compared exactly, `scaled_dot_product_attention.md`, Logit
differences), and for the far keys omega_j = e**(x_F - x_j), at most 1 since
x_F - x_j = l_j - l_F <= 0 by Logit differences, and Omega = sum_far omega_j,
so phi = e**-(x_F) Omega / Z. Where a row's pending elements read only far
keys, each omega_j but F's (exactly 1) is enclosed by [0, 1] instead of Lemma
X: it then enters only the far keys' e**-(x_F) Delta D', their second-order
term. A kept key's Delta P_j and Delta dS_j are
e**-(x_F) times intervals computed without e**-(x_F). A far key's are its own
weight times such intervals, since P_j = e**-(x_j) / Z (the reference weight
being exactly 1):

    Delta P_j = e**-(x_j) / Z,   Delta dS_j = e**-(x_j) ((dP_j - D^K) - e**-(x_F) Delta D') / Z   (far j),

with Delta D = e**-(x_F) Delta D'. An element's Delta is so a sum of terms
c_t e**-(y_t) T_t, each an interval T_t times its exact factor c_t (as above)
at the scale y_t: x_F for a kept key's term, x_j for a far key's. A term with
c_t = 0 is exactly 0 and is left out, and so is a row's term that is exactly
0: its interval the point 0, or a dS term the exact analysis proves 0 (every
group of equal logits over the row's keys has one mean dP m, so D = m, and the
key's dP is m; for grad_query every key's dP is one value), checked where the
term's enclosure holds 0. Such a row sets no scale, so it cannot hide the other
rows' terms. Each row takes as its scale the least
y_t among its terms with a nonzero factor (x_F where a kept key or F has one;
else, for grad_query, the far key s of the largest logit with one, its logits
compared exactly by Logit differences as F's), its other terms times
e**(y_row - y_t) (omega_j, or e**(x_s - x_j), with x_s - x_j exactly at most
0). Delta = e**-(x_m) Delta', the rows' terms added in row order at the least
scale x_m so far: a row whose scale is not below it adds its terms times
e**(x_m - y_row), and one whose scale is below it first multiplies the sum so
far by e**(y_row - x_m), its scale becoming x_m (the columns of one key over
one row list share the scale), and then moves any end of the sum below
2**-(2**61) in magnitude outward, to 0 or 2**-(2**61), so that repeated
rescaling keeps the exponents in range. Each scale is sigma X + Y with X and Y
the exact score and mask differences from its row's reference key, so a row's
scale and x_m are compared exactly, and y_row - x_m is enclosed by Logit
differences from the differences of their parts: the enclosure has the
difference's sign and is within 2**-p of it however large the scales are (a
mask at a format's lowest value puts them near 2**128).

A key that is far in every row keeping it (a *far group*: its grad_value or
grad_key columns over one row list, as with padded keys) takes its least
scale first. A pass over the rows finds, from the exact score and mask
differences alone, whether some row keeps the key not far (the group then
sums as above), and x_m, the least scale over the rows where the key is far
and a term is nonzero (a pending element's factor nonzero, and for grad_key
the row's dS not exactly 0 by the exact test: every dP equal makes it 0;
n dP_j unequal to the sum of the row's dP rules it out; otherwise it is 0
exactly where every group of equal logits over the row's keys has one mean
dP, m, and dP_j = m). Each row then gives
one interval per far group, F = Delta P'_j or Delta dS'_j times
e**(x_m - y_row), shared by the group's columns and at most the interval
itself since y_row >= x_m. Such an element has G^K = 0 (its key is in no
row's kept part), so G = Delta, and its plain and cancellation-free sums are
not formed: they enclose the same value. Its Delta' is the sum over the rows
of F times its exact factor c_t, as integers on a grid 2**f: F's ends rounded
outward to multiples of 2**f, and each end times c_t added exactly (the ends
swapped for c_t < 0). The grid lies p + 72 bits below the power of two above
the factor that set it, and moves up, the sums shifted with outward rounding,
when a factor's top passes that one's by more than 48 bits, so every factor
is below 2**(p + 120) units and every move coarsens the grid at least 2**48
times. Each row's rounding adds at most 2**f |c_t| to an end, and the moves at most two units of
the last sum's grid together, 2**(f + u_x) each with 2**u_x the factors' unit
(c_t an integer times it). The factor that set the last grid is at most the
largest factor F_max, so 2**f <= 2**-(p + 71) |F_max|, and the grid adds at
most (sum |c_t| + 2 2**u_x) 2**-(p + 71) |F_max| to each end, at most
(sum |c_t| + 2 max |c_t|) 2**-(p + 71) |F_max|. No sum is rescaled, so no
exponent drifts. These splittings are
exact, and every factor e**y here has y at most 0, its enclosure's upper end
too. It comes from Lemma X at the ends of y's enclosure while y > -2**56,
and is [0, 2**-(2**56)] below (e**y < 2**-(2**56) there), so it never needs an
exponent past the dyadics' range: each factor is 0 or above
2**-(1.45 2**56 + p + 2), a term has at most four of them (e**-(x_F) and
Delta D' in a far key's dS, the row's factor and the rows' factor), and its
exponent stays above -2**58.6. Once Delta' excludes zero, Lemma X gives e**-(x_m) as
[lo, hi] 2**n with an exponent n of any size: for n at least -2**61 at both
ends, Delta = e**-(x_m) Delta' is enclosed directly (its exponent above
-2.72e18, within the dyadics' range); with n below -2**61 at both ends, and
Delta''s ends below 2**(2**60) in magnitude (which the kernel checks),
|Delta| < 2**(1 - 2**61 + 2**60) < 2**-(LIMIT + 1), and its sign decides by the
tiny-offset rule (`foundations.md`, Deciding a rounding by a sign). G^K is the gradient of an ordinary
problem whose masked keys include the far ones. So the exact analysis above
(Which terms are rational) applies to it, and an element with no irrational
kept term has an exact rational G^K. The grad_query pair test (grad_query of
one row, decided exactly) also applies to the kept rows, summed over the rows
that use the query row. It finds G^K where counting irrational terms cannot:
for example, 0 when the kept keys have equal entries in a key column, since
their dS add to zero. For grad_query and grad_key, G^K includes the factor
sigma. This is rational when sigma is, or when the sum is 0.

For such an element, G = G^K + Delta, and the lemma's case 1 or 2 decides it
with c = G^K; where every row's far terms are exactly 0, Delta = 0 and G = G^K
is its exact value. Case 2 applies when G^K is a grid point and Delta's enclosure
excludes zero. That happens unless the first-order part of the terms at the
least scale vanishes exactly. Elements whose G^K is not exact are refined as
before.

**What can still raise** is a Delta whose enclosure still contains 0 at the cap:
an element whose first-order far effect cancels exactly, such as an exact
sum over rows (below), or, in some row, a nearest far key whose first-order
term is exactly 0 while a farther key's is not (the cancelling terms'
enclosure, about 2**-p e**-(x_F) wide, hides the farther key's), or a far key
J other than F whose own first-order term is exactly 0 (dP_J = D^K, as when
J's value row equals the only kept key's), which leaves its second-order term
e**-(x_J) e**-(x_F) through F, hidden by [0, 2**-(2**56)] once x_F >= 2**56 or
by the 2**-p width at the cap. In a sum over rows such a row's enclosure, at its
lower scale, also hides the other rows' first-order terms at higher scales.

**grad_query of one row, decided exactly.** After the first refinement round,
the kernel tests the grad_query elements that are still unresolved, with exact
integers and fractions (`rational.c`).
For one row and column e, write d_j = dP_j and c_j = k_je. Then

    sum_j dS_j c_j = sum_j P_j d_j c_j - D sum_j P_j c_j
                   = sum_(j,k) w_j w_k c_j (d_j - d_k) / Z**2
                   = (1/2) sum_(j,k) w_j w_k (d_j - d_k)(c_j - c_k) / Z**2,

the last step by averaging with the same sum with j and k swapped. Group the
ordered pairs by t = s_g + s_h (compared exactly as above; for groups g, h the
pairs contribute A_gh = n_h a_g - d_g c_h - d_h c_g + n_g a_h, with d_g, c_g,
a_g the group sums of d, c and d c). This gives

    (1/2) sum_t C_t e**t / sum_t N_t e**t,

with rational C_t (the sum of the A_gh with s_g + s_h = t) and positive
integers N_t (the pair counts). If the value x is algebraic,
sum_t (C_t / 2 - x N_t) e**t = 0, and by Lindemann-Weierstrass every
coefficient is 0: x = C_t / (2 N_t) for every t. Conversely, if all ratios
C_t / N_t are equal, x is that ratio over 2. So the test gives the
exact rational x, or reports that x is transcendental. Where every kept key's
c_j is one value, each A_gh is 0 and x = 0 at once. Otherwise two tests of
O(G) steps come first (G groups, group 1 the first in the kernel's order).
With V_g = (n_g a_g - d_g c_g) / n_g**2 and the means m_g = d_g / n_g,
u_g = c_g / n_g, substituting a_g = n_g (V_g + m_g u_g) gives
A_gh = n_g n_h (V_g + V_h + (m_g - m_h)(u_g - u_h)), h = g included. If every
V_g is V_1, and every m_g is m_1 or every u_g is u_1, then every
A_gh = 2 V_1 n_g n_h, every ratio is 2 V_1, and x = V_1. Else the kernel tests
whether a ratio differs. The least t, 2 s_1, holds the pair (1, 1) alone, with
ratio 2 V_1, so every ratio equals it exactly when
F = n_1**2 (N A - D C) - k_1 N**2 is 0, with k_1 = n_1 a_1 - d_1 c_1 and N, D,
C, A the sums over the groups of n_g, d_g, c_g, a_g times formal monomials
X**s_g, X**s X**s' = X**(s + s') over the logits' parts
(sum_t C_t X**t = 2 (N A - D C) and sum_t N_t X**t = N**2). The kernel maps F
to the integers modulo the prime p = 2**61 - 1 by a ring homomorphism h: a
dyadic coefficient m 2**z to itself modulo p (2 is invertible), and X**s, for
parts (s_1, s_2), to y_1**psi(s_1) y_2**psi(s_2), with psi the map of the
dyadics to the integers modulo q = 2**60 - 1 (odd, so 2 is invertible and psi
is additive) and y_i = z_i**2 for fixed z_i not divisible by p, so that
y_i**q = z_i**(p - 1) = 1. F = 0 gives h(F) = 0; so h(F) != 0 proves a ratio
unequal and x transcendental.

Where h(F) = 0 (a false zero is possible, for instance from values v and
v 2**61 in one column), a witness check comes next. A run whose pairs all
keep the ratio 2 V_1 keeps it, so the kernel looks for pairs whose own ratio
A_ab / (n_a n_b) is not 2 V_1, among (1, h), (h, h) and (h, G) for
h = 2, 3, ..., up to four of them; if none of those differs, every V_h is V_1
and every (m_1 - m_h)(u_1 - u_h) is 0, and since the first test failed some
a has m_a != m_1 (so u_a = u_1) and some b has u_b != u_1 (so m_b = m_1),
whose pair's ratio differs by (m_a - m_1)(u_1 - u_b) != 0. For each such pair
it gathers the pair's whole run of t: the groups' logits are distinct and
their order translation-invariant, so for each g at most one h has
s_g + s_h = t, found by walking in from both ends of the sorted groups. A run
whose ratio C_t / N_t is not 2 V_1 proves x transcendental. Only where none
does, the kernel produces the pairs in the order of t without holding them
all: for each g its pairs with h = g, g + 1, ... (the groups sorted by logit)
come in that order, so a heap of one pair per g merges them, and the first
unequal ratio ends the test. The pairs' logit sums are exact: integers at one
exponent per part where the row's parts span at most 254 bits, else dyadic
sums. A row's pending columns are tested together: their groups and the order
of their pairs are the row's, so one merge serves every column the tests
leave, each with its own c, a and runs' sums. An element whose rows
(one row, or several broadcast copies) all give rational x is that exact sum
times sigma: rational when sigma is, and otherwise a nonzero rational times
1/sqrt(E), irrational, so refinement ends. A transcendental x is never a
rounding boundary or zero, so refinement ends for a single row. The test runs
for the elements the first refinement pass left unresolved.

**Exact sums over rows.** grad_key and grad_value sum terms from different rows,
and grad_query sums them over broadcast copies of a query row. Such a sum of
transcendental terms with different denominators Z can be algebraic, for
example exactly 0 when two equal rows have opposite gradients. These sums are
not tested; if one is a rounding boundary or zero, refinement reaches the cap
and the call raises instead of returning a guess. This is the one accepted
case where an exact result can reach the cap, stated in the `OpSpec`
limitations.

## The tolerance

**Model.** An implementation computes the gradients from the inputs, rounding
only at these boundaries in the compute format F (library `exp` within c ulps,
`ulps`, default 2):

1. to 6. the forward's steps 1 to 6 (`scaled_dot_product_attention.md`): the
   scores, the scale factor, the logits, the row maximum, the shift and exp,
   for the kept keys. It also computes the masked keys' scores and logits
   before adding their -inf bias.
7. the row sum Z' of the weights (`softmax.sum`);
8. P'_j = w'_j / Z', rounded (`softmax.probabilities`); P' = 0 off the kept keys;
9. dP'_j = g . v_j, rounded (`grad_probabilities`);
10. D' = sum_j P'_j dP'_j (`grad_softmax.row_dot`);
11. dP'_j - D', rounded (`grad_softmax.difference`), times P'_j, rounded
    (`grad_softmax.grad_scores`): dS'_j;
12. grad_value = sum over rows of P'_j g, rounded (`grad_value`), and
    likewise dS' K (`grad_query`) and dS'^T Q (`grad_key`), each sum over all
    the rows or keys and copies it covers;
13. those two sums times the scale factor, rounded (`scale_gradients`);
14. the final rounding to the output format (Lemma 3).

The sums (steps 7, 9, 10 and 12) are normal accumulations by default: index
order or lanes 2 and 4, each product rounded and then added, with the
overflow-safe slots. The order of a gradient sum's terms is the order of its
rows: the output rows grouped by the key and value block they use, the groups
in the order of their first rows and each group's rows in row-major order
(`Problem.tiles`). grad_value and grad_key add one term per row in that order;
grad_query adds each row's terms in key order, its rows (a query row's
broadcast copies) in that order. The sums over a row's keys (steps 7, 10 and
grad_query's) run their lanes over the key indices, a masked key's exactly 0
term keeping its place (as in the forward). With `standard="any_order"` they
are sums in any order without overflow.

Not covered: rounding elsewhere, such as a reciprocal of the row sum instead
of the division, rounding intermediates to the input format, or computing D
from the forward output instead of P' and dP' (named algorithms cover these,
`attention_algorithms.md`). Pre-scaling Q and K (PyTorch's math path
multiplies each by the square root of the scale) changes the inputs; a
consumer that wants it passes the pre-scaled values with scale 1.

**Masked keys.** As in the forward (`scaled_dot_product_attention.md`, Masked
keys), an implementation may skip the keys a row does not keep or compute
their terms with P' = 0: dP'_j, P'_j dP'_j in D', dS'_j, and the products
P'_j g, dS'_j q and dS'_j k_j. Those are exactly 0 where every factor is
finite and dP'_j - D' does not overflow, so the implementation then computes
the model over the kept keys (with flush to zero too, `foundations.md`,
Lemma 2, Exact zero terms; a kept key that surely vanishes, with P' = 0,
adds such zeros as well). Otherwise they are NaN, and the NaN reaches:

1. a row whose P' may be NaN (a NaN row of the forward, or a masked key whose
   logit may be NaN, as in the forward): every gradient the row touches,
   grad_query of its query row and grad_key and grad_value of every key of
   its block, since P' is then NaN at every key. With the forward's given
   logsumexp (`attention_algorithms.md`, The backward of fused kernels),
   P' = exp(l - logsumexp) is NaN only at such a masked key and D' comes from
   the given output: grad_query of the row and grad_key and grad_value of
   that key (every key where a query entry with no bound reaches them all);
2. a row whose D' may be NaN (dP' at a key it does not keep may be
   non-finite: a non-finite entry of g or of that key's value row, or dP''s
   sum may overflow; or D is not finite): its dS' at every key, so grad_query
   of its query row and grad_key of every key of its block. And at a key the
   row does not keep, dP'_j - D' may overflow with both finite: that pair's
   dS' is P' = 0 times an infinity, NaN, so grad_query of the row and
   grad_key of that key;
3. 0 times a non-finite entry of a key's row or the query row at a pair the
   row does not keep: grad_query's column of a key row entry that is not
   finite, and grad_key's column of a query entry that is not finite;
4. P' = 0 times a non-finite entry of g at a pair whose P' may be 0, kept or
   not (a key the row does not keep, or P''s lower end at most 0), where the
   reference's P is positive at a kept key: grad_value's column of that
   entry, at that key.

(A masked key's non-finite row entry makes its logit possibly NaN, case 1.)
So each bound below holds for every output that is not NaN, and the flags
allow NaN where one of these cases reaches; where none does, the keys a row
does not keep add no flag.

**Claim.** For each output element, `tolerance()` is at least |output - ref|
for every such implementation, except an output NaN that the element's flags
allow (Masked keys).

**Proof.** The backward kernel (`csrc/attention_grad_tol.c`) runs the model
on intervals, with the invariant that each interval contains every value any
such implementation can hold at that step.

*Values of a format.* A quantity the model rounds to a format, or accumulates
in one, is a value of that format in every implementation (or is not finite,
which the bounds below exclude or report as unbounded). So an interval
[lo, hi] for it may be narrowed to [up(lo), down(hi)], up(lo) the least value
of the format at or above lo and down(hi) the greatest at or below hi; the
narrowed interval still holds every value the implementation can have. Every
finite value of a format is an integer multiple of its least positive value
2**(emin - p + 1), subnormals included (with flush to zero, its values are a
subset of these). The kernel narrows P', dP', D' and dS', each to the format
of the rounding that produced it, so their ends are integers at that unit, and
every product and sum of them below is an exact integer operation. Narrowing
only shrinks the intervals, so each bound below is at most the one without it.

- Steps 1 to 8, the probabilities P'_j: from the closed form when the row's
  validity conditions hold (below), otherwise as in the forward's step-by-step
  program (`scaled_dot_product_attention.md`) through the row sum, then the
  interval quotient widened by Lemma 1; or, for a named algorithm that takes
  the forward's output and logsumexp, as in `attention_algorithms.md`.
  For masked keys with finite inputs the score and logit are bounded too,
  only to detect overflow: an overflowed score gives inf - inf = NaN and a NaN
  row, so a possible overflow there, or a masked key's score that is not
  finite, makes the row's P' possibly NaN (Masked keys, case 1), and the
  row keeps its bounds. Keys the row does not mask with a non-finite score
  are removed or make the row NaN by IEEE rules in every implementation.
- Step 9: the exact sums, magnitude sums and (with flush) the moved amounts
  come from the product kernel, then, any order, Lemma 2 with n = Ev, or,
  normal, the normal bound of `foundations.md` over Ev with every node exact,
  and Lemma O's rule; then Lemma 1, at the kept keys. At the other keys dP'
  is checked the same way only to detect a non-finite value: P' = 0 times
  one would be NaN (Masked keys, case 2); and where it is finite, once D' is
  bounded (step 10, or the given output's), dP'_j - D' is checked against
  `grad_softmax.difference`'s threshold (Lemma 1), the pair's case 2.
- Step 10: P'_j dP'_j over the kept keys (the other terms are exactly 0 in
  every implementation: they add nothing, but keep their places, so the lanes
  are by key index, Lemma L). Each product of [p_lo, p_hi] and
  [d_lo, d_hi] lies between the least and the largest product of their ends,
  exact integers. Any order: Lemma 2 over the products that can be nonzero,
  with the magnitude sum of the largest end products. Normal: each factor as
  twice its midpoint and twice its radius (a = p_lo + p_hi, ra = p_hi - p_lo,
  and likewise b, rb for dP'), the products a b as the reference terms, whose
  every node is taken exactly (index order, lanes 2 and 4), and R, the sum of
  (|a| + ra)(|b| + rb) - |a| |b| over the terms, as the distance of each term
  from its reference term: Lemma S with those distances as leaf errors
  (`foundations.md`, Lemma S: mass + R, N + 2R, moved R), every doubled
  product being four times its value. Lemma O's rule, then the narrowing.
- Step 11: interval subtraction, widened by Lemma 1, narrowed; the interval
  product with P', widened by Lemma 1, narrowed. Using the same P' interval
  in D' and in dS' is valid: each implementation uses one value of P' in both,
  and that value lies in the interval.
- Step 12, any order: for w in [lo, hi] and exact x, w x lies between lo x and
  hi x (in the order given by the sign of x). Per element, the kernel adds
  these ends exactly over the element's terms, and the magnitude sum
  sum max(|lo|, |hi|) |x|; Lemma 2 applies with n the number of products that
  can be nonzero: a product that is exactly 0 in every implementation adds
  nothing and rounds nothing (x + 0 = x and fma(0, b, x) = x; with flush to
  zero, Lemma 2's Exact zero terms). Then Lemma 1 for the rounding.
- Step 12, normal: per element, over its terms in the model's order, the
  interval factor (P' or dS') as twice its midpoint and twice its radius, and
  the exact factor x as 2x with radius 0, or, where a flush may read x as zero,
  as the hull of 0 and x (twice the midpoint x, twice the radius |x|). The
  midpoints' products are the reference terms, every node taken exactly
  (index order, lanes 2 and 4, and the lanes' combinations from the exact lane
  totals), and R as in step 10 bounds each term's distance from its reference
  term: Lemma S with those distances as leaf errors, Lemma O's rule, then
  Lemma 1. An element whose terms are all exactly 0 is exactly 0.
- Step 13: interval product with the scale-factor interval, Lemma 1.
- Step 14: Lemma 3.

Every step only enlarges what it encloses, or narrows it to values the
implementation can hold, so the final interval contains every possible
output. []

**The kernel's order.** The kernel's threads share one group of rows (one key
and value block) at a time, tile by tile: the tile's rows are split among the
threads for steps 1 to 11, and then its keys (for grad_value and grad_key)
and rows (for grad_query). Each element's sums take a tile's terms in row
order, and the tiles in order, so its exact sums and node sums are those of
the model's order. A query row that appears in several groups (broadcast
copies) keeps its grad_query sums from one group to the next, in the groups'
order. The kernel's words hold every integer above exactly (`kernels.md`,
Lemma J).

**The probability step in closed form.** In the notation of the forward's
closed form (`scaled_dot_product_attention.md`), steps 1 to 8 give
w'_j = K w_j (1 + t_j) + h_j and Z' in [G (1 - E_Z), G (1 + E_Z)] with
G / (K W) in [D, D'], E_Z under the chosen standard (the forward's step 8).
So w'_j / Z' lies in [Y_lo, Y_hi]:

    Y_hi = (P_j exp(R_j) + H_j) / (D (1 - E_Z)),
    Y_lo = P_j exp(-R_j) / (D' (1 + E_Z)) - H_j / (D (1 - E_Z)),

since K w_j exp(R_j) + |h_j| over Z' >= K W D (1 - E_Z) is at most Y_hi
(H_j >= |h_j| / (K W)), and K w_j exp(-R_j) over Z' <= K W D' (1 + E_Z),
minus |h_j| / Z', is at least Y_lo. For a key that may vanish (its logit or
shift may overflow toward -inf, the forward's steps 1 to 6), the lower end
is -H_j / (D (1 - E_Z)) instead, since its weight may be exactly 0. The rounding to F
(`softmax.probabilities`) adds at most rnd(Y_hi, Y_lo), and P'_j >= 0. The
row's validity conditions are the forward's 1 to 8. With `reciprocal`,
P'_j = fl(w'_j fl(1 / Z')) with fl(1 / Z') = 1 / Z' + f and |f| <= d_r (as
in `attention_algorithms.md`, reciprocal, which also requires that it not
overflow), so w'_j fl(1 / Z') lies in [Y_lo - wh_j d_r, Y_hi + wh_j d_r],
where wh_j = exp(Lam_lo) w_j exp(R_j) + eps_j is w'_j's largest value. A fused backward's P is in
`attention_algorithms.md` (The backward of fused kernels). The kernel
evaluates these bounds as the forward's closed form does: intervals of
dyadics rounded outward at 128 bits, the weights from Lemma D's tables or
Lemma X, exp(R_j) - 1 by Lemma W or X, and exp(-R_j) as the reciprocal of
exp(R_j)'s upper end, rounded down.

**Flush to zero.** As in the forward's proof: every rounding and accumulation
uses the 2**emin terms; an input subnormal in F may be read as zero, which
moves exact sums of exact products by at most the product kernel's moved
amount, and turns a product w x with an interval factor into one between
min(lo, 0) x and max(hi, 0) x (x > 0) or the reverse (x < 0), whose ends the
kernel adds (step 12, any order) or whose hull it takes (normal).
Intermediate results are produced already flushed, and narrowing keeps them
values of their formats.

**Special elements.** An element whose reference is NaN or infinite has a NaN
tolerance (Lemma 3). The program leaves out the terms that only reach such
elements (NaN rows, and the dS of rows whose D is not finite); the elements
of the keys such a row does not keep get the NaN flag instead (Masked keys,
cases 1 and 2). An element with no possibly nonzero term, such as grad_value
for keys masked in every row, has tolerance 0.

## Results as inputs

When query, key, value, a float mask or grad is an earlier result, the
implementation computes the model on its stored values, each within its
radius (the earlier tolerance with the same arguments) of the values the
reference uses. A stored value that may be non-finite (the earlier result's
flags) counts as having radius inf, as in the forward: exp can turn it back
into a finite value.

**Claim.** `tolerance()` bounds |output - ref| for every such stored input.

**Proof.** As in the forward (`scaled_dot_product_attention.md`, Results as
inputs), radii are first widened to cover 0 where a flush may read a stored
value as zero, and each exact operand of an integer product carries its
radii:

- No bound: a mask element with radius inf or NaN, or a query or key element
  with one that reaches a key the row does not mask, makes its row's P
  unbounded, and with it the row's grad_query and the grad_key and
  grad_value blocks it reaches; one that reaches only masked keys makes the
  row's P' possibly NaN instead (as in the forward; Masked keys, case 1). A
  value element with none at a key the row keeps, or a grad element with
  none, makes dP, so D and dS, unbounded for its rows (grad_query and
  grad_key), and a grad element also its grad_value block; a value element
  at a key the row does not keep makes its D' possibly NaN (case 2).
- Steps 1 to 8: the scores' exact sums take the deviation of the stored
  inputs (the product kernel adds it to the magnitude sum and the moved
  amount, and twice it to N), and the mask values their radii, exactly as in
  the forward.
- Step 9: dP's exact sums take the deviation of the stored grad and values
  the same way.
- Step 12: for a stored factor within r of x, w x' lies within |w| r of w x,
  so each end widens by sum max(|lo|, |hi|) r and the magnitude sum uses
  |x| + r; a product counts as possibly nonzero when |x| + r > 0. For normal
  accumulation the exact factor's doubled radius takes 2r more, so R
  includes it.

Every other step is unchanged, and every interval still contains every
value an implementation can hold. []

`normal()` runs the normal answer on the earlier results' `normal()`.

## The normal answer

`normal(compute=None, flush_to_zero=False)` computes the model in the
compute format F with every sum a normal accumulation in index order
(`csrc/attention_normal.c`): the forward's steps 1 to 7
as in its normal answer (`scaled_dot_product_attention.md`), then
P'_j = w'_j / Z' rounded (0 for every key of a row with no kept key, NaN for a
NaN row), dP' over the Ev products rounded and added in index order, D' over
every key's product P'_j dP'_j likewise (the keys a row does not keep
included, as Masked keys describes, so its NaN is the flags'), dS'_j = fl(fl(dP'_j - D') P'_j), and
each gradient sum over its terms in the order above (each product rounded,
then added, with Lemma O's slots where index order overflows with every term
finite: the kernel keeps each sum's running value and, for such an element,
forms its terms again from its rows), with grad_query and grad_key times the
correctly rounded scale, rounded once. The arithmetic is Lemma E's, as in
the forward. ExactRef's test suite, developed
separately, compares it with the same steps rounded one by one by MPFR.
