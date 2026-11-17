# Attention algorithms: proofs

This file proves the tolerances of `exactref.nn.attention.AttentionAlgorithm`.
An algorithm is the standard computation of `scaled_dot_product_attention.md`
(and of `scaled_dot_product_attention_backward.md`) with extra rounding
points, its methods. The reference is unchanged; each tolerance bounds
|output - reference| for every implementation that rounds at the required
boundaries and at the method's points, with normal accumulation
(`standard="normal"`, the default) or in any order without overflow
(`standard="any_order"`), as the definition's tolerance does. With
`online_softmax`, the forward's sums keep the any-order bounds below under
both standards: online softmax also multiplies its running sums by
rescaling factors, which Lemma S does not cover. Lemmas 1 to 3 and the normal
standard's lemmas are in `foundations.md`.

The bounds are the standard ones with the method steps switched on
(`Methods` in `ops/attention/_problem.py`, `methods` and `given` in
`ops/attention/_backward.py`): each method's section says how it changes the
closed form (`scaled_dot_product_attention.md`, The closed form, whose
notation it uses) and the step-by-step program, which covers the rows the
closed form does not. With no method, an algorithm's tolerance is the
definition's, bit for bit, because it runs the same code with the same
settings. Below, F is the compute format, c the library-function accuracy in
ulps (`ulps`), and for a format X the dyadic pair (rel_X, abs_X) is
`error_units`: one rounding to X moves y by at most rel_X |y| + abs_X
(abs_X = 2**emin with flushing). This follows from Lemma 1:
c ulp(y) <= 2c 2**-p |y| for normal y and 2c 2**(emin - p) below.

## accumulate

Every accumulation boundary uses the format A instead of F. Lemmas 2, S and A
hold for any accumulator format, so the sums are bounded as before, in A: in
the closed form, the accumulations' units (r, a, a', Theta, Growth, gamma,
kappa) and the largest values of conditions 1, 8 and 9 are A's, and the row
sum counts its narrower accumulator (step 8).

## bias

Each float-mask value b is rounded to the format B before it is added
(`bias`): Lemma 1 on the point b. Masked keys (-inf) are unaffected. In the
closed form, beta_j gains rnd_B(|b_j|, |b_j|) (rnd_B(|b_j|, 0) when a flush
may read b_j as zero). When |b_j| reaches B's overflow threshold, a negative
b_j rounds to -inf or within that bound, so the key may vanish
(`scaled_dot_product_attention.md`, steps 1 to 3: rounding float32's lowest
value to bfloat16 gives -inf); a positive one sends the row to the
step-by-step program.

## weights

**Forward.** Each weight w_j (the exponential after the shift, in F) is
rounded to the format W before the value product; the row sum uses the
unrounded weights, as kernels compute it in F. Lemma 1 widens each weight
interval for the value sums only. (With online softmax, see below.)

In the closed form, values are multiplied by w~_j = round_W(w'_j) while the
row sum uses w'_j. In step 7 the numerator gains sum_j (w~_j - w'_j) v_jc.
w'_j lies between exp(-Lam_up) w_j exp(-R_j) - eps_j and
exp(Lam_lo) w_j exp(R_j) + eps_j, so Lemma 1 at those bounds gives
|w~_j - w'_j| <= ew_j, and ew_j / G <= HW_j / D with
HW_j = ew_j exp(Lam_up) / W. Adding HW_j to the key's error weight in the
value sums (E_j + HW_j, not in tau, since the row sum uses w'_j) covers the
change in E_Q, in H (the rounded weights' mass) and, with flush, in phi_c.
With online softmax the weight is rounded before its rescaling factors, so
there E_Q gains r_W H + a_W g V_c and H becomes (1 + r_W) H + a_W g V_c (with
flush, phi_c becomes (1 + r_W) phi_c + a_W g VF_c, VF_c = sum_j |vf_jc|),
with r_W, a_W Lemma 1's units for W and a_W multiplied by `growth`. Every
rounded weight is at most w = (1 + r_e + a_e)(1 + r_W) + a_W (times `growth`
with online softmax), which must be at most W's largest value and replaces
1 + r_e + a_e in condition 9.

**Backward.** P and dS are rounded to W before the products P^T grad, dS K
and dS^T Q: Lemma 1 on each interval, then the narrowing to W's values
(`scaled_dot_product_attention_backward.md`, Values of a format); an overflow
makes the affected gradients unbounded. This comes after the probability step,
so the closed form there is unchanged.

## exp2

The weights are computed as exp2 of logits in base-2 units:

1. the scale times log2(e), held in F within c ulps (`exp2_scale`);
2. each mask value (after `bias`, if set) times log2(e), within c ulps
   (`exp2_bias`);
3. base-2 logits fma(score, step 1, step 2), rounded once (`logits`);
4. the shift by the row maximum, rounded (`softmax.shift`);
5. exp2 within c ulps (`softmax.exp`);
6. for `logsumexp`, the base-2 maximum times log(2) within c ulps
   (`exp2_logsumexp`), plus log of the row sum within c ulps, rounded once
   (`logsumexp.add`).

The programs enclose log(2) by Lemma Q (foundations.md; `functions.c`'s
`ln2_bounds`) and log2(e) as its interval reciprocal, so each exact constant
product lies in the interval product; exp2(x) = exp(x log 2) is enclosed by
exp of the interval product (Lemma X at both ends). Every rounding is then
Lemma 1 with c ulps or one half. Since the base-2 logits times log(2) are the natural ones,
the rest of the program (sums, values, division) is unchanged. In the
backward, the given logsumexp times log2(e) within c ulps replaces the row
maximum (step 6 reversed).

In the closed form, steps 1 to 5 run in base-2 units: the exact factor is
sigma log2(e) and I is step 1's interval (`exp2_scale`). A mask value's error
in natural units, delta_j (from a flush and from `bias`), becomes
delta_j log2(e) + rnd_X((|b_j| + delta_j) log2(e), (|b_j| - delta_j) log2(e))
with X = `exp2_bias`. When (|b_j| + delta_j) log2(e) reaches X's overflow
threshold, the key may vanish for a negative b_j (a mask of -3e38 in float32
times log2(e) overflows toward -inf), and the row falls back for a positive
one. This gives base-2 L_j, m_up, m_lo and R_j,
and conditions 5 and 6 are checked in base-2 units. Since
exp2(x) = exp(x log 2), R_j, Lam_up and Lam_lo are then multiplied by log(2),
and P_j comes from the natural logits; steps 6 to 9 are unchanged, with the
common factor 2**(mu - mu') for the base-2 maximum mu. logsumexp computes the
base-2 maximum times log(2) within c ulps, which adds
rnd(|m| + max(Lam_up, Lam_lo), |m| - max(Lam_up, Lam_lo)) to E_log.

## exp_ulps

Each key's own softmax exponential (exp, or exp2 with `exp2`) is within
c_e = `exp_ulps` ulps, in place of the call's `ulps`. Below 2**emin it may
also be off by up to 2**emin: Lemma 1's flushing case for that boundary
alone (`foundations.md`, Lemma 1). The `softmax.exp` setting carries both, so
every bound that uses each key's exp uses them:
- the step-by-step program (Lemma 1 at each exp);
- the closed form (its units r_e and a_e, and its Lemma 1 for each weight);
- online softmax's weights (below).

Everything else keeps the call's `ulps`: the logsumexp's log, the scale,
exp2's products, and online softmax's rescaling factors (`softmax.rescale`).
`exp_ulps` applies to the forward only, because fused backward kernels
recompute P with their accurate exp.

PyTorch's CPU flash forward computes these exponentials with `fexp_u20` for
bfloat16 and float16 inputs. PyTorch documents it as accurate to one
bfloat16 or float16 step. Measured by emulating each of its float32
operations with MPFR, exhaustively where its error peaks:
- it is within 3647 float32 ulps on aarch64 without SVE, so `exp_ulps=4096`
  covers that build;
- it is within 986 ulps on x86 with AVX2 or AVX-512, so `exp_ulps=1024`
  covers those;
- below exp(x) = 2**-126 it returns 0, or a value within 2**-126 of the true
  one;
- SVE builds use another method, not measured; PyTorch's documented float16
  step is 8192 float32 ulps.

## fast_log

**Model.** The forward's log of the row sum (`logsumexp.log`) is within c
ulps of the larger of |log Z| and 1: at most c 2**(1 - p) where
|log Z| < 1, with p F's precision. A fast logarithm is accurate in absolute
terms near 1, not in ulps of its result. FlashAttention-2 computes each
row's logsumexp with CUDA's `__logf` (`normalize_softmax_lse`), which CUDA
documents as within 2**-21.41 on [0.5, 2] and 3 ulps elsewhere. A row sum is
at least about 1 (the maximum's own weight), so log Z is often near 0.
2**-21.41 is 3.01 2**-23, so the documented bound needs c >= 4. The GB10's
measured error needs only c = 3 (Checks).

**Bounds.** The step-by-step program widens the log's interval by
c 2**(1 - p) beyond Lemma 1's c ulps, which covers both cases. In the closed
form, E_log gains c 2**(1 - p). With `split_kv`, each split's log does the
same. The backward takes the logsumexp as given, so the option is the
forward's.

## online_softmax

**Model.** Each row's keys are processed in blocks b = 1 .. nb, in any
order of keys and blocks. With m_0 = -inf and l_j the computed logits:

- m_b = max(m_(b-1), max of the block's logits), an exact selection;
- each weight w_j = exp_c(fl(l_j - m_b)) (shift and exp as usual);
- for b >= 2 the rescaling factor a_b = exp_c(fl(m_(b-1) - m_b));
- Z_b = Z_(b-1) a_b + s_b and acc_b = acc_(b-1) a_b + t_b, where s_b and t_b
  are the block's sums of w_j and of w_j v_j (or of the rounded weights),
  each in any order, and the update rounds once or twice (fused or not);
- finally out = acc_nb / Z_nb rounded, and logsumexp from m_nb and Z_nb.

A block whose running maximum m_b is -inf (every logit so far -inf: masked
keys, a left-padded first block, or keys that vanished, Keys that may vanish)
adds 0: kernels guard the shift there (FlashAttention shifts by 0), so each of
its weights is exp(-inf) = 0, its sums stay 0, and the next block's rescaling
factor is exp(-inf) = 0, whose argument is <= 0 like every other factor's.
Such blocks hold no term, and the bounds below hold with them.

nb is at most S for any blocking (`block_size` None). With `block_size` B,
the keys are split into at most ceil(S / B) blocks of at most B keys (as
kernels split them: every block full but one), so nb is at most ceil(S / B).

**The exponent.** Unrolled, key j in block b reaches Z_nb as
w_j a_(b+1) ... a_nb times rounding factors. The exact exponent of that
product is fl(l_j - m_b) + sum over later blocks of fl(m_(b'-1) - m_b'). Each
difference is nonpositive (m is nondecreasing and m_b >= l_j), and their
exact values add up to l_j - m with m = m_nb, the final maximum: the sum
telescopes. Each rounding moves its difference d by at most
rel_F |d| + abs_F, and the |d| add up to m - l_j, so the exponent lies within

    E_j = rel_F (m - l_j) + nb * abs_F

of l_j - m. The standard model's single shift has the same relative part.
The step-by-step program widens the interval of l_j - m (the logit interval minus the
maximum's interval) by E_j, using the interval's largest magnitude for
m - l_j.

**The exp calls.** On that path there are at most nb calls to exp: the
weight's own and nb - 1 rescaling factors. Each returns exp(x)(1 + e) + h.
- The weight's own call has |e| <= rel_e and |h| <= h_e, the `softmax.exp`
  boundary's units (c ulps, or `exp_ulps` with its allowance).
- Each rescaling factor has |e| <= rel_r and |h| <= h_r, the
  `softmax.rescale` units (c ulps). Without `exp_ulps` the two kinds are
  equal.

Split each rescaling factor as exp(y)(1 + e) plus h. The relative parts then
multiply the term by a factor in [down, up], where

    down = (1 - rel_e)(1 - rel_r)**(nb - 1),   up = (1 + rel_e)(1 + rel_r)**(nb - 1).

The weight's own h is multiplied only by later factors' relative parts and
by exp(sum of later y) <= 1. Each rounded y is <= 0, since rounding is
monotone and fl(0) = 0. So h adds at most own = h_e (1 + rel_r)**(nb - 1).
This gives each term's interval. With exp2, the exponent interval is multiplied by the
log(2) interval before exp, as above. If exp of the interval reaches F's
overflow threshold (Lemma 1, the test `lean_ends` applies to exp in the
standard model), an exp call may overflow, and the program gives the row no
bound; the fully padded rows' bound below may still give one.

**The sums.** The unrolled Z_nb is a sum of these terms, each multiplied by
the (1 + d) factors of the roundings on its path, plus absolute parts. A
term passes at most B roundings inside its block (any order; B = S for any
blocking), one when the block sum joins the carried sum, and two per later
block (the multiplication by a_b' and the addition), so its depth is at most
B + 2 nb - 1; with the value products rounded, one more. Without flushing,
the roundings with an absolute part are those that involve an exact product:
the rescaling multiplications (nb) and, for acc, the value products (S); with
flushing every rounding counts, at most 2S + 2nb. The any-order bound of
Lemma 2 for n terms has depth n and n roundings (2n - 1 with flushing), so
n = max(B + 2 nb, S + nb + 1) + 1 covers both sums (the online count), and
Lemma 2 with that n applies to the terms' exact interval sum.

**The rescalings' absolute parts.** Rescaling by a_b = exp(y)(1 + e) + h adds
Z_(b-1) h, an absolute error. Arguments are <= 0, so every weight is at most
1 + rel_e + h_e and every factor at most 1 + rel_r + h_r. A partial sum is
therefore at most S such terms grown by K = growth (1 + gamma(n)), with

    growth = (1 + rel_e + h_e)(1 + rel_r + h_r)**(nb - 1),

plus the accumulator's absolute errors n abs_A (1 + gamma(n)). Each of the
nb - 1 additions is then grown by at most K again. So the sum moves by at
most (nb - 1) h_r K * (S largest + n abs_A) K. Here largest bounds a term
before rescaling:
- for Z, 1 + rel_e + h_e (`growth` is used, which is larger);
- for acc, that times max |v| (times 1 + rel_W + abs_W with rounded weights)
  (`online_bound` in `csrc/attention_tol.c`).

**Fully padded rows.** Where the program's intervals give no bound (every
key of a row carrying the lowest finite mask, say), the fully padded rows'
bound of the standard model (`scaled_dot_product_attention.md`, Fully padded
rows) carries over, without rounded weights or a rounded reciprocal
(the kernel's `padded_row` with the online constants). A key that may vanish
(Keys that may vanish) never counts as its key with a finite logit, so a row
whose kept keys may all vanish gets no such bound. Let w'_j be key j's computed
weight times its block's later computed rescaling factors, an exact product
of computed values. Every computed weight and factor is
an exp of an argument <= 0 (The rescalings' absolute parts, above), so
0 <= w'_j <= wmax = growth. The key of the final maximum m' has shift
fl(0) = 0 in its block, and every later block keeps m', so each of its later
factors is an exp of fl(0) too. Its w' is therefore at least
g_min = (1 - rel_e - h_e)(1 - rel_r - h_r)**(nb - 1), and
G = sum_j w'_j >= g_min. Unrolled, the
computed row sum is sum_j w'_j (1 + d_j) plus the roundings' absolute parts,
and the value sum is sum_j w'_j v_jc (1 + d'_j) plus theirs, with each path's
depth and rounding count covered by Lemma 2 for n + 1 terms, n the online count
(The sums, above): |d_j| <= gamma_Z, and likewise gamma_V for the value
sum's format. An absolute part is added in some block and then multiplied by
that block's later computed rescaling factors: fewer than nb of them, each
at most 1 + rel_r + h_r, so their product is at most
(1 + rel_r + h_r)**(nb - 1) <= wmax. So the absolute parts are at most
const_Z wmax (and const_V wmax). So, relative to G,

    E_Z = gamma_Z + const_Z wmax / g_min,   e_N = gamma_V vmax_c + const_V wmax / g_min,

and the rest is the standard bound: Lemma O (e) with n wmax (1 + E_Z) and
n wmax (vmax_c + e_N) below each format's threshold, the value sum's rounding
to F, and the quotient in [lowest_c - E_R, highest_c + E_R], then Lemma 1
for `normalize`. The row's tolerance is this interval intersected with the
program's, where that one exists. []

**Rounded weights.** A weight is rounded to W before its rescaling factors:
w (1 + d) + h with |d| <= rel_W, |h| <= abs_W; after the factors, the term
moves by at most rel_W |term| + abs_W growth.

**The rest** is the standard program: the values' sum is rounded to F
(`values.output`, which only widens), divided by the row sum, and rounded;
logsumexp adds log of the online row sum. Every step encloses every value an
implementation can hold, so the claim follows as in the standard proof. []

**Closed form.** The exponent of a weight times its later rescaling factors
is l'_j - m' plus at most nb shift roundings of total size
rel_F (m' - l'_j) + nb abs_F (The exponent, above), so
R_j = L_j + rel_F ((m_up - l_j) + L_j) + nb abs_F; m' is the largest computed
logit, so step 4 is unchanged. The nb exp calls' relative parts give
T_j = (exp(R_j) - 1) up + (up - 1), with up and down as in The exp calls.

The term's relative error lies between exp(-R_j) down - 1 and
exp(R_j) up - 1. T_j covers both ends because
exp(R) up + exp(-R) down >= up + down >= 2:
- the first inequality holds since up >= down and R >= 0;
- for the second, with P = (1 + rel_r)**(nb - 1) and
  Q = (1 - rel_r)**(nb - 1), P + Q >= 2 and P >= Q, so
  up + down = P + Q + rel_e (P - Q) >= 2.

The weight's own absolute part is multiplied only by the later factors'
relative parts and by exponentials of nonpositive numbers, so
H_j = own exp(Lam_up) / W. The sums use the step-by-step program's count
(the online count + 1) in gamma and kappa, under both standards. The rescalings' absolute parts X_Z and
X_N, as `online_bound` computes them (X_N with the column's largest
|v_jc| over the kept keys), are absolute errors of Z' and of the value sum, so
E_Z gains X_Z g, e_N gains X_N g, and alpha gains X_Z exp(Lam_up) / W. Every
term is at most `growth`, which replaces 1 + r_e + a_e in conditions 8 and 9.
Without `exp_ulps`, the own and rescaling units are equal and every formula
here is the one for nb exp calls alike.

The any-blocking bound grows with S (each key may be its own block), so it
is much looser than the bound for a known block size; both cover every key
order.

## reciprocal

The row sum Z is normalized by its reciprocal rounded once, r = fl(1 / Z)
(`reciprocal`, or `softmax.reciprocal` in the backward), and a product with r
rounded once (`normalize` for the output, `softmax.probabilities` for the
backward's P) instead of one rounded division. The step-by-step programs
compute the interval quotient of 1 by Z's interval (which excludes 0, or the
bound is infinite), widen it by Lemma 1, then multiply and widen again by
Lemma 1; each step contains every value an implementation can hold.

In the closed form, Z' lies in [G_lo (1 - E_Z), G_hi (1 + E_Z)] (steps 7 and
8), so by Lemma 1 fl(1 / Z') = 1 / Z' + f with
|f| <= d_r = rnd(1 / (G_lo (1 - E_Z)), 1 / (G_hi (1 + E_Z))), and
1 / (G_lo (1 - E_Z)) + d_r must be at most F's largest value. Then
N' fl(1 / Z') = N' / Z' + N' f with |N'| <= G (Q + E_N) <= G_hi (Q + E_N), so
E_R becomes E_R + G_hi (Q + E_N) d_r, and the product's rounding is
`normalize`. For the backward, see its probability step.

## mma

**Model.** The products' sums are computed by tensor-core kernels
(`foundations.md`, Lemma T) with the step `mma`, whose `split` states how
the kernels split each sum:
- the scores' sums over E, and the value sums over the keys;
- in the backward: the scores; dP over Ev; grad_value and grad_key over the
  queries and their broadcast copies; grad_query over the keys and their
  copies.

Their factors are values of the inputs' format W: the inputs themselves, and
P and dS, which `weights` rounds to W before their products. So `mma` needs
`weights` = W, float16 or bfloat16. The row sums and D keep the compute
format. A kernel processes every product of a sum, masked ones (zero)
included. So each sum's count is its full length, not the number of terms
that can be nonzero.

**sigma.** For products x y of two inputs, sigma is as in Lemma T, with X
and Y the inputs' largest magnitudes (radii added for earlier results). P
and dS may be subnormal, and they are at most W's largest finite value L_W
(a larger value rounds to infinity, and the output is then not finite). So
a sum whose first factor is P or dS has

    sigma = 2**emin_W max |y| + [y may be subnormal] (2**emin_W L_W + 2**(2 emin_W)).

**Bounds.** The step-by-step program bounds every accumulation with Lemma 2.
Here it takes Lemma T's G and H instead, with T = ceil(count / N). The program's argument is unchanged: it
uses only that a finite sum's computed value lies within G mass + H of its
exact sum, and that its partial sums stay below the overflow threshold
(mass + G mass + H at most float32's largest value).

The closed form and the fully padded rows' bound rely on round-to-nearest
units (Lemma S, Lemma O, theta). So with `mma` every row takes the
program's bound, and both standards give the any-order bound.

**Online softmax.** With `online_softmax`, a value product passes:
- the steps of its own block and of each later block, at most
  ceil(S / N) + nb - 1 in all (each block adds at most one partial step);
- per later block, a rescaling multiplication and possibly a join, both
  float32 roundings to nearest.

The blocks are continued pieces. So Lemma T's tree argument applies with
r = r_all = ceil(S / N) + nb - 1 steps and 2 nb float32 roundings more on a
path:

    G = (1 + rho)**(S + r) (1 + tau)**r (1 + 2**-24)**(2 nb) - 1.

These G and H replace gamma(n) and n abs_A for the value sums in The sums
and The rescalings' absolute parts (above), with K = growth (1 + G). The row
sum keeps its Lemma 2 constants.

## split_kv

**Model.** `split_kv` = K cuts each row's keys into splits of K consecutive
keys: split i holds keys iK to min(S, (i + 1) K) - 1. Each split is an online
softmax of its own (`online_softmax` is required): its maximum, blocks and
sums cover its keys only. It ends in the compute format F with:
- O_i, its value sums (rounded to F, `values.output`) divided by its row sum
  Z_i, or times its reciprocal with `reciprocal`, rounded to F (`normalize`),
  not to the output format;
- lse_i, its logsumexp as the forward computes logsumexp (`logsumexp.log`,
  with `fast_log` if set, `logsumexp.add`, and `exp2_logsumexp` with exp2).

A split with no kept key has lse_i = -inf and adds exactly 0 below
(exp(-inf) = 0), so it is left out. Every computed exp is nonnegative, as
online softmax's fully padded rows also use. The splits are then merged in
F:
1. M = max_i lse_i, an exact selection;
2. Sigma = sum_i exp_c(fl(lse_i - M)), in any order (`split_kv.shift`,
   `split_kv.exp`, `split_kv.sum`);
3. the logsumexp output, lse = fl(log_c(Sigma) + M) (`split_kv.log`,
   `split_kv.add`);
4. s_i = exp_c(fl(lse_i - lse));
5. the output, sum_i s_i O_i, each product rounded or fused into its
   addition, in any order (`split_kv.product`, `split_kv.accumulate`), then
   rounded to the output format.

FlashAttention-2's split-KV kernels merge this way
(`combine_attn_seqk_parallel` in `flash_fwd_kernel.h`). In PyTorch's build,
the combine kernel's `expf` and `logf` are CUDA's accurate ones: its machine
code has their standard instruction sequences, not fast math. Each of their
splits is an online softmax (`compute_attn_1rowblock_splitkv`) whose lse_i
uses `__logf` (`fast_log`). It forms its rescaling factors from the raw
score maxima, not from the rounded base-2 maxima of online_softmax's model,
and lse_i from the raw maximum times the scale. Whether a call's `ulps`
covers those differences is for the caller to establish. Each split holds a
whole number of key blocks of N_k keys, so K = ceil(ceil(S / N_k) / splits)
N_k. PyTorch picks the number of splits from the shape and the GPU's
multiprocessor count (`num_splits_heuristic` in `flash_api.cpp`), and with
one split it runs no merge.

**Bounds.** Let l_j be the computed logits, T their maximum and T_i split
i's, and m and mu_i those maxima in natural units (times log 2 with exp2).
Z_i and N_i are split i's computed row sum and value sum (one column), its
weights taken relative to T_i. P is the number of splits with a kept key.

*The merged sums.* The bound goes through Z^ = sum_i exp(mu_i - m) Z_i and
N^ = sum_i exp(mu_i - m) N_i, which no kernel computes. Unrolled
(online_softmax, The sums), Z_i sums each key's weight times its later
rescaling factors, with rounding factors and absolute parts. Times
exp(mu_i - m) <= 1, key j's term has exponent l_j - T plus split i's shift
roundings, at most rel_F (T_i - l_j) + nb abs_F <= rel_F (T - l_j) +
nb abs_F (The exponent), and at most nb exp calls, with nb the blocks of one
split (ceil(min(S, K) / B) with `block_size` B, else min(S, K)). So the term
lies in the program's interval of it for the whole row's logits and maximum T,
with one split's constants. With `weights`, its rounded weight lies in
the rounded weights' interval, since the scaled absolute part only shrinks.

A term's rounding path stays within its split, so the relative constants are
one split's: Lemma 2 for min(S, K) keys, or Lemma T for that count with
`mma`. The absolute parts are Lemma 2's constant, Lemma T's H and the
rescalings' absolute parts. Each split has its own, multiplied by
exp(mu_i - m) <= 1, so together they are at most P times one split's. Hence
Z^ and N^ lie in the single-pass program's intervals with one split's
constants and P times its absolute parts (`online_bound` with P passes). So does
A^ = sum_i exp(mu_i - m) |N_i|, bounded from the products' magnitudes. One
product per row covers every split. A kernel that skips some of a split's
keys (causal rows), or adds masked keys' exact zeros, only lowers the
counts.

*Each split's logsumexp.* lse_i = mu_i + log Z_i + gamma_i, where gamma_i
collects three roundings:
- `exp2_logsumexp`'s, at |mu_i|;
- the log's, at |log Z_i|, plus c 2**(1 - p) with `fast_log`;
- `add`'s, at |lse_i|.

The program bounds these magnitudes with each split's maximum, its lower end
taken over the split's keys whose logit cannot be -inf (Keys that may
vanish; a split whose kept keys may all vanish leaves the row no bound, since
its Z_i may be 0), and with Z_i in [z_i, R_i growth + d_i]. The split
maximum's own term is at least g_min, every term is at most `growth`, and d_i
is one split's sum error for R_i terms that size. z_i is the larger of
g_min - d_i and g_min (1 - E_Z), E_Z = gamma_Z + const_Z wmax / g_min with
Lemma 2's terms at one split's online count n + 1: as in the fully padded
rows' bound (online_softmax), Z_i sums the split's terms w'_j (1 + d_j) plus
absolute parts at most const_Z wmax, each w'_j >= 0 and their sum G_i at
least g_min, so Z_i >= G_i (1 - gamma_Z) - const_Z wmax >= g_min (1 - E_Z).
R_i is the smaller of S_i and the largest, over the lower ends M of the
split's logits, of

    R(M) = min(n_M, exp(T - M) sum_{j: lo(l_j) <= M} W_j + n_M H),

with W_j key j's relative part, the upper end of the program's interval of
its weight relative to the row's maximum T with its later rescaling factors,
before the absolute parts (exp(hi(l_j) - lo(T) + E_j) up, with E_j the shift
roundings of The exponent), H a key's absolute part (own), and n_M the
number of the split's keys with lo(l_j) <= M. A key that may vanish takes
lo(l_j) = -inf here (its logit is -inf or in its interval), so it counts in
every n_M, and its relative part's lower end is 0 (its hull with 0). The
program bounds exp(T - M) by the reciprocal of the lower end of a key at M's
relative part (exp(lo(l_j) - hi(T) - E_j) down, at most exp(M - T) since
lo(l_j) = M). That end is 0 only where exp's clamp leaves the key 2**40 or
more below T, or for keys that may vanish (M = -inf), and there R(M) is n_M.
With
`weights` each held weight is at most 1 + rel_W times its weight plus
abs_W growth (Rounded weights), so R_i is 1 + rel_W times that largest
R(M) plus the split's key count times abs_W growth. In whatever order the
kernel takes the split's keys, and with whatever block boundaries, after any
key t the pass holds its sums relative to its running maximum, which is at
least M_t, the largest lower end among the keys so far, and each of those
keys has lo(l_j) <= M_t. So each one's relative part is at most
exp(T - M_t) W_j, and its absolute part, added in the pass's own units
whatever its maximum, at most H; and each held term is at most `growth`
(above), so the at most n_(M_t) of them are at most n_(M_t) growth: each
partial sum is at most R(M_t) growth. The cap n_M keeps the bound where the
relative parts' own widths pass the key count (a key so far below T that
its shift's roundings E_j are wide).
Neither the split's final maximum nor its keys' index order would bound the
earlier sums (FlashAttention-2's split kernel takes its blocks last to
first). Lemma 1 at those magnitudes gives Gamma_i >= |gamma_i|.

*Each split's share.* A split far below the maximum may have a large
|lse_i|, and so large rounding errors, but it adds little to the row. Its
share w_i = exp(mu_i - m) Z_i / Z^ weighs its errors; the shares add up to
1. Since log Z^ + m >= lse_K - gamma_K for the split K with the largest
lse_i (M), w_i <= exp(lse_i - M + gamma_K - gamma_i), so
W_i = min(1, exp(lse_i^hi - M^lo + Gamma_i + Gamma)) bounds it from the
intervals, for every split i, where Gamma >= |gamma_K|. K is one of the
splits with lse_i^hi >= M^lo (lse_K = M >= M^lo, M's interval being the
largest of the lse_i's intervals), so Gamma is the largest Gamma_i over those
splits only: a split far below the maximum, whose Gamma_i is large, cannot be
K. For numbers g_i with |g_i| <= G_i, and G_0 the smallest G_i,

    sum_i w_i exp(g_i) <= exp(G_0) + sum_i W_i (exp(G_i) - exp(G_0)),
    sum_i w_i exp(g_i) >= exp(-G_0) - sum_i W_i (exp(-G_0) - exp(-G_i)),

since sum_i w_i = 1 and every excess term is nonnegative. In the same way,
for numbers x_i <= X_i and any split k,

    sum_i w_i x_i <= sum_i w_i X_i <= X_k + sum_i W_i (X_i - X_k)+,

and the middle sum is also at most the largest X_i.

*The logsumexp.* The program also runs the merge on these intervals step by
step, but only to bound the magnitudes of its arguments: lse_i - M,
log Sigma, lse and lse_i - lse. From them:
- Lemma 1 bounds each shift's rounding,
  fl(lse_i - M) = lse_i - M + eta_i with |eta_i| <= H_i;
- Lemma 1 bounds the log's error |lambda| <= Lambda and the addition's
  |r| <= R;
- exp is within rel_e |y| + abs_e (`split_kv.exp`);
- Lemma 2 for nt terms gives gamma_P and C_P (`split_kv.sum`), nt = P, or
  ceil(S / K) with flush_to_zero (below).

The step-by-step Sigma's lower end is the larger of its interval's and
(1 - rel_e - abs_e)(1 - gamma_P) - C_P: split K's term is
exp_c(fl(lse_K - M)) = exp_c(0) >= 1 - rel_e - abs_e, every term is
nonnegative, and Lemma 2 bounds the computed sum below by its exact sum times
1 - gamma_P less C_P. Likewise Z^ >= g_min (1 - E_Z): the split K' holding
the row's maximum has exp(mu_K' - m) = 1 and Z_K' >= g_min (1 - E_Z) (Each
split's logsumexp), and every other term is nonnegative, so Z^'s lower end
is the larger of that and the single-pass interval's.

Each term is

    e_i = exp(lse_i - M + eta_i)(1 + eps_i) + h_i
        = exp(mu_i - M) Z_i exp(gamma_i + eta_i)(1 + eps_i) + h_i,

and sum_i exp(mu_i - M) Z_i = exp(m - M) Z^ = X. X is at least exp(-Gamma),
since the split K has exp(mu_K - M) Z_K = exp(-gamma_K). So
sum_i e_i = X sum_i w_i exp(gamma_i + eta_i)(1 + eps_i) + sum_i h_i, and the
shares' bounds with G_i = Gamma_i + H_i give E_lo and E_hi. The computed
Sigma is X F, with F in [F_lo, F_hi]:

    F_hi = (E_hi (1 + rel_e) + P abs_e exp(Gamma))(1 + gamma_P) + C_P exp(Gamma),
    F_lo = (E_lo (1 - rel_e) - P abs_e exp(Gamma))(1 - gamma_P) - C_P exp(Gamma),

and F_lo must be positive. The logsumexp output is
log Sigma + lambda + M + r = m + log Z^ + Delta, with Delta in
[log F_lo - Lambda - R, log F_hi + Lambda + R]. m and Z^ lie in the
single-pass intervals, so lse lies in m + log Z^'s interval widened by
Delta's range.

*The output.* With fl(lse_i - lse) = lse_i - lse + eta'_i, where
|eta'_i| <= H'_i,

    s_i = exp(lse_i - lse + eta'_i)(1 + eps'_i) + h'_i = w_i phi'_i + h'_i,
    phi'_i = exp(gamma_i + eta'_i - Delta)(1 + eps'_i),

and O_i = (N_i / Z_i) rho_i + o_i. Here rho_i collects the relative parts of
`values.output`, `reciprocal` and `normalize` (Lemma 1), and o_i their
absolute parts. So

    sum_i s_i O_i = (1 / Z^) sum_i exp(mu_i - m) N_i phi_i + sum_i (w_i phi'_i o_i + h'_i O_i),

with phi_i = phi'_i rho_i in [phi_lo_i, phi_hi_i], from
Gamma_i + H'_i. Take phi_c, the midpoint of the split with the smallest
Gamma_i + H'_i, and r_0, half its width; every |phi_i - phi_c| is at most
some r_i. Since exp(mu_i - m) |N_i| / Z^ = w_i |N_i| / Z_i <= W_i V, with V
bounding |N_i| / Z_i, the first sum lies in

    (phi_c N^ +- r_0 A^) / Z^ +- V sum_i W_i (r_i - r_0)+.

In the second, sum_i w_i phi'_i is at most
Phi = phi'_hi_k + sum_i W_i (phi'_hi_i - phi'_hi_k)+, with phi'_hi_i
phi'_i's upper end and k the split with the smallest Gamma_i + H'_i, and at
most max phi'_hi (Each split's share), so the second sum is at most the
smaller of the two times max |o_i|, plus P abs_e max |O_i|. A split far below
the maximum then weighs in by its share: its phi'_hi may be huge, its W_i
is tiny. V, |O_i| and |o_i| follow from Z_i's lower end and |N_i|, which is
at most R_i (the largest over the splits) of one split's terms at their
largest plus its error. V is also at most

    (1 + rel_a)(c_W vmax (1 + gamma_V) + const_V wmax / g_min) / (1 - E_Z) + abs_a / (g_min (1 - E_Z)),

and the program takes the smaller. As for Z_i above, N_i sums the split's
terms w'_j v_j (1 + d_j) plus absolute parts at most const_V wmax, so with
G_i >= g_min, |N_i| <= c_W G_i vmax (1 + gamma_V) + const_V wmax and
Z_i >= G_i (1 - E_Z); `values.output` (rel_a, abs_a) moves N_i by at most
rel_a |N_i| + abs_a, and Z_i >= g_min (1 - E_Z). Here vmax is the column's
largest |v| over the kept keys, gamma_V and const_V are Lemma 2's terms at
one split's online count n + 1, or Lemma T's G and H with `mma`, and
c_W = 1 + rel_W + n_i abs_W growth / g_min with `weights` (Rounded weights,
n_i the split's kept keys: the held weights add up to at most
(1 + rel_W) G_i + n_i abs_W growth), 1 without. With `reciprocal`, its
absolute part abs_b multiplies |fl(N_i)|, whose bound is unchanged.

Last come the products' roundings (`split_kv.product`) and their sum
(`split_kv.accumulate`, Lemma 2 for P terms). They move the result by at
most rel_pi Q + P abs_pi + gamma'_P (Q (1 + rel_pi) + P abs_pi) + C'_P,
where Q is at least sum_i |s_i O_i|: phi_hi_0 A^ / Z^_lo +
V sum_i W_i (phi_hi_i - phi_hi_0)+ plus the second sum's bound, with
phi_hi_0 that of the split with the smallest Gamma_i + H'_i. A product fused
into its addition is not rounded on its own, which only removes rel_pi and
abs_pi.

In exact arithmetic every gamma, eta, eps, rho, h and o is 0, so Delta = 0
and phi = 1. The bounds are then the single-pass sums' quotient and
logsumexp: exp(lse_i - lse) O_i = exp(mu_i - m) N_i / Z^ cancels each Z_i.
The merge widens them only by its own roundings and the lse_i's errors, each
split's weighed by its share.

The row gets no bound when a split's bounds fail, F_lo is not
positive, or a value may overflow. The merged sums' checks do not cover a
split's own sums, whose terms they scale by exp(mu_i - m) <= 1. So each
split's row sum and value sums must stay below their accumulators' largest
values: one split's terms at their largest plus its error, Lemma 2's or
Lemma T's condition. Lemma 1's overflow test also applies at
`values.output`, `reciprocal`, `normalize` and the products, at the
magnitudes above. With flush_to_zero, adding an empty split's exact zero may
flush a subnormal partial sum, so the merge's two sums count every split,
ceil(S / K). A column with a non-finite value among the kept keys is not
finite, as in the standard program. The closed form and
the fully padded rows' bound do not model the merge, so with `split_kv`
every row takes the program. Both standards give the any-order bound, as
with online softmax. The backward is unchanged: fused backward kernels take
the merged logsumexp as given.

## scale_ds

**Model.** In the backward, dS is multiplied by the scale and rounded once in
F, then rounded to W (`weights`) before its products:

    dS' = fl(fl(fl(dP - D) p) sigma),

with p the softmax probability, or its rounding to W with `rounded_p`
(below). grad_query and grad_key are then the products' sums, rounded to the
output format with no scaling after them (no `scale_gradients` rounding).
cuDNN's and PyTorch's memory-efficient backward kernels do this. The latter's
source computes `(dP - Di) * attn`, then `* scale`, then casts to the inputs'
format.

**Bounds.** In the program, dS's interval is multiplied by sigma's interval
and rounded once in F (Lemma 1, the `scale_ds` setting). It is then rounded
to W as `weights` rounds it. The exact reference multiplies the gradients by
sigma, and the dS intervals already hold that factor, so the scaling after
the sums is skipped. Every step encloses every value such an implementation
holds, so the claim follows as in the standard proof.

## rounded_p

**Model.** The backward forms dS from P already rounded to W (`weights`),
the P its value product takes, instead of the unrounded P:
dS = fl(fl(dP - D) fl_W(p)). PyTorch's memory-efficient kernel reads `attn`
in the inputs' format, while FlashAttention-2 and cuDNN take the float32 P.

**Bounds.** The factor of dS's product is P's interval rounded to W (Lemma 1,
the rounding the value product applies). The rest is unchanged.

## grouped_heads

**Model.** Where several query heads share a key and value head (grouped or
broadcast heads), the kernel computes grad_key and grad_value per query head
and batch copy, rounds each to the output format O, then adds them:

    dK = fl_O(sum over the group, in F in any order, of fl_O(dK_h)).

PyTorch's fused backward kernels do this. FlashAttention-2's and the
memory-efficient kernel's wrappers write each query head's gradients into an
expanded tensor in the inputs' format, then `at::sum_out` over the group.
cuDNN's outputs match the same arithmetic.

**Bounds.** Each query head's and batch copy's gradient sum is bounded as
before: its accumulation, its rounding to F, and for grad_key its scaling.
Each is then rounded to O (Lemma 1, the `group` setting). Their sum is an
any-order sum of these intervals in F (Lemma 2, the `group_sum` setting).
Lemma 3 then gives the final rounding. The kernel keeps each head's sums
apart by cutting its tiles at head boundaries. Both standards give this
any-order bound.

## Keys that may vanish

The step-by-step programs round each kept key's float-mask value (`bias`,
and `exp2_bias` with exp2), its logit (`logits`) and, without online
softmax, its shift (`softmax.shift`); the fused backward's program rounds its
shift by the given logsumexp. Where the interval X = [lo, hi] such a rounding
takes reaches its overflow threshold (Lemma 1), a point of X may round to an
infinity:

- if hi < 0 (for a mask value b of an earlier result with radius r, b + r < 0,
  as in the closed form's steps 1 to 3), the result is -inf or finite: the key
  *may vanish*. With -inf its logit is -inf and its weight exactly 0
  (exp(-inf) = 0; in online softmax after its model's guard);
- otherwise a point may round to +inf, which makes the row NaN, and the row
  has no bound.

**The finite outcomes.** Let L be the format's largest value. A finite result
of rounding a point y of X is a value of the format, at least -L, and either
within c ulp(y) of y (Lemma 1's argument, which holds for every finite
result whatever |y|: a rounding to nearest or a directed one to a neighbouring
value, a library function within c ulps) or -L, where a directed rounding of
a y below -L stops at the format's largest magnitude. The values within
c ulp(y) of a point of X are at most the larger of

- hi + c ulp(hi), for the points in hi's binade, and
- where X reaches 2**k, the power of 2 just above |hi|: -2**k + c ulp(M),
  M = |lo|, for the points at or beyond 2**k in magnitude (each is at most
  -2**k, and its ulp at most M's),

with 2**emin added to the first where flushing may read a y below 2**emin
as 0. So every finite result lies in [-L, U], U the larger of -L and that
bound rounded down to the format's grid.

**No finite result.** The rounding has no finite result, and the key *surely
vanishes*, in two cases:

- a correct rounding to nearest (c = 1/2) of an X with hi <= -T: a point at
  or past Lemma 1's threshold rounds to -inf (-T is a tie between -L and
  minus the power of 2 above L; ties to even and ties away from zero both
  take the latter, past the format's range);
- c < 1 and the bound above below -L: no value of the format lies within
  c ulp(y) of any point y of X, and the outcome -L needs a directed rounding,
  which has c = 1 (`foundations.md`).

Every later step keeps -inf: a finite score times the scale plus -inf, -inf
times log2(e), and -inf minus a finite maximum or logsumexp are -inf. So the
key's logit and shift are -inf and its weight is exactly 0 in every
implementation of the model.

**The program.** The rounded interval is [-L, U] ([-L, -L] for a key that
surely vanishes, which has no finite outcome to enclose), the key is marked as
one that may vanish, and the program runs on as for any finite value; every
step encloses the key's finite outcomes, and the outcome -inf is taken as
follows.

- The row's maximum m' is the largest computed logit that is finite, so it
  lies in [the largest lower end over the keys whose logit cannot be -inf,
  the largest upper end over every kept key]: U can make a key that may
  vanish the maximum, and the upper end takes it. A row whose kept keys may
  all vanish has no bound (every logit -inf gives 0 / 0).
- A key that may vanish has its weight interval's hull with 0 (its lower end
  at most 0), which holds the weight exactly 0 of the outcome -inf whatever
  the row's maximum, as the closed form raises R_j for such a key. With online
  softmax this is the interval of the weight times its later rescaling
  factors, and with split_kv also of its relative part (split_kv); in the
  backward it is P'_j's interval.
- A key that surely vanishes has weight interval [0, 0] in place of the hull,
  since 0 is its only weight: with online softmax its weight times its later
  rescaling factors and its relative part are 0, with no absolute part (the
  exp of -inf is exactly 0), a held weight (`weights`) is fl(0) = 0, and in
  the backward P'_j is 0. It is still a key that may vanish for every other
  rule here and below.
- The fully padded rows' bound (online_softmax) and the split_kv merge take
  such keys as their sections state.

Every interval still holds every value an implementation can hold at its
step, so each program's claim follows as before. []

## Weights that may be 0

A kept key whose held weight may be exactly 0, times a value that is not
finite, gives NaN in that column (`scaled_dot_product_attention.md`, Masked
keys). With the methods, the flags take these cases:

- Online softmax: the term's factor is the weight times its later rescaling
  factors, whose interval [y_lo down - own, y_hi up + own] (online_softmax)
  holds every value it can take; the program flags where its lower end is at
  most 0. A rescaling factor of 0 times an accumulator holding an infinity is
  NaN too. Such a factor's computed argument bounds the term's exponent
  (every other part is at most 0), so the term's relative part is at most
  that factor's absolute unit h_r, which own is at least (the same exp, or
  `exp_ulps`'s, which flushes and so has the larger unit 2**emin): the
  interval reaches 0 then as well. The closed form takes its least value
  e down - own >= e (2 - up) - own, e = exp((l_j - m) - R_j) / exp(Lam_up)
  with R_j its shift roundings (down >= 2 - up, as
  (1 - a)(1 - b)**k >= 1 - a - k b and up - 1 >= a + k b).
- `weights`: the held weight is the weight rounded to W, which moves it by at
  most rel_W |y| + abs_W, so it may be 0 only where the weight is at most
  abs_W / (1 - rel_W) <= 2 abs_W. The program's interval of the held weight
  includes that rounding; the closed form takes e (1 - rel) - abs after the
  exp's rounding and again after W's, and online e (2 - up) - own minus
  2 abs_W growth (the factors after the rounding are at most growth).
- `split_kv`: the merge multiplies a split's output O_i by
  s_i = exp_c(fl(lse_i - lse)), which may be 0 where O_i is infinite. The
  merge takes s_i's least value from the least lse_i - lse, rounded
  (`split_kv.shift`), its exp, rounded (`split_kv.exp`); where it is at most
  0 (or the merge has no bound) the split's kept keys flag their values'
  columns that may not be finite.

## The backward of fused kernels

`backward_tolerance` takes the forward's rounded `out` and `logsumexp` as
inputs, as fused backward kernels do, and models:

- P_j = exp_c(fl(l_j - logsumexp)) for kept keys (with exp2, in base-2 units
  as above), and exactly 0 for the keys a row does not keep (exp(-inf) = 0;
  NaN where a masked key's logit may be NaN, which the flags allow,
  `scaled_dot_product_attention_backward.md`, Masked keys). The given
  logsumexp is an exact input, so the interval of l_j minus it, rounded and
  exponentiated, contains P_j. Where l_j or that shift may overflow toward
  -inf, the key may vanish, and P_j's interval is its hull with 0, or [0, 0]
  where it surely vanishes (Keys that may vanish). A non-finite given
  logsumexp makes the row's
  gradients unbounded (-inf - logsumexp or l_j - logsumexp is then not a
  finite difference).
- D = sum_c grad_c out_c in any order (`grad_softmax.row_dot`): exact products
  of exact inputs, bounded with integers and Lemma 2 as in `foundations.md`,
  including flushed inputs, under both standards (any order covers normal
  accumulation). A non-finite given output makes D unbounded.
- Everything else as in the standard backward, with `weights` rounding P and
  dS before the products.

The reference stays the exact gradients, so the tolerance also covers how
far the given out and logsumexp are from their exact values.

**Closed form for P.** With the given logsumexp lse_g (an exact input) and
the closed form's L_j, the computed shift l'_j - lse_g is within L_j + k of
l_j - lse_g, where k = |lse_g| when a flush may read lse_g as zero, else 0.
Its rounding adds rnd(|l_j - lse_g| + L_j + k, |l_j - lse_g| - L_j - k), so the
exponent is within R'_j = L_j + k + that rounding of l_j - lse_g, and

    P'_j in [exp(l_j - lse_g - R'_j) - eps_j, exp(l_j - lse_g + R'_j) + eps_j],

with eps_j = rnd(exp(l_j - lse_g + R'_j), exp(l_j - lse_g - R'_j)) for exp,
and the lower end at least 0. With exp2 this runs in base-2 units: the given
logsumexp times log2(e) is held within c ulps, which adds
rnd(|lse_g log2(e)|, |lse_g log2(e)|) to k (and k is |lse_g| log2(e) for a
flush), and R'_j is then multiplied by log(2). The conditions are those of the
closed form's steps 1 to 3, the shift's rounding and exp's result at most F's
largest value (a given logsumexp far below the true one can make P overflow),
and with exp2 the logsumexp product's rounding; a row that fails one gets the
step-by-step program. A non-finite given logsumexp is unbounded either way.

Without the forward's out and logsumexp, `backward_tolerance` runs the
standard backward program (`scaled_dot_product_attention_backward.md`) with
the methods' points: `reciprocal` for P, `weights` before the products, `exp2`
for the weights, `bias` and `accumulate`. Online softmax applies to the forward
only; with no methods this is the definition's tolerance, bit for bit.

### backward_blocks

`backward_blocks=(Q, K)` describes a fused backward kernel that adds the
gradients' products block by block straight into gradients held in the output
format O, as PyTorch's CPU flash kernel does. For each batch and key-value
head, it takes each query head of the group in order, then each block of Q
consecutive queries, then each block of K consecutive keys. Each product is
one matrix product with beta = 1 into its gradient:

- grad_value += P^T grad over the query block (scale 1);
- grad_key += scale dS^T q over the query block;
- grad_query += scale dS k over the key block.

So each element of grad_value and grad_key is a sum carried in O over the
blocks (query head, query block), in that order, and each element of
grad_query is carried over its row's key blocks. Lemma C (`foundations.md`)
bounds each, with:

- A the gradient's accumulation format (the compute format, or
  `accumulate`'s);
- C = O;
- alpha the scale factor's enclosure (`scale_factor`), or 1 for grad_value;
- the block's exact sums, masses and counts of terms that can be nonzero from
  the exact products as in the fused backward, with P and dS as above
  (rounded by `weights`);
- F_j the block's sum of |dS| and |q| (grad_key) or of |dS| and |k|
  (grad_query), taken over its terms' factors.

The scale is covered wherever a kernel applies it, because Lemma C covers
every way:

- PyTorch's `gemm_stub` applies it to dS, and to the block's sum in one
  product;
- the scale on q or k is covered too.

Keys that causal masking or a mask removes give exact zero terms, so a key
block the kernel skips and a block whose terms are all zero are the same: an
empty block, skipped when O's values are values of A (Lemma C).

These sums replace roundings that the kernel never makes. For all three
gradients, the carry replaces the nested products' output rounding. For
grad_query and grad_key, it also replaces the `scale_gradients` rounding,
because the kernel scales each block. The last block's value before its
rounding goes to Lemma 3, as the definition's gradients do.

The block sums are bounded in any order (Lemma 2 inside Lemma C) under both
standards, as online softmax's are. P, dS, dP and D are bounded as in the
fused backward. Earlier results' radii enter each block's exact sums and
masses through the products as above, and F_j through the factors'
magnitudes plus their radii.

The option applies to the fused backward only. `backward_tolerance` raises
`ValueError` with `backward_blocks` in these cases:

- without the forward's out and logsumexp;
- with a query broadcast in any dimension;
- with a key or value broadcast across batch dimensions (any dimension before
  the last three).

The kernel keeps each gradient in its own input's shape. PyTorch handles a
broadcast by expanding the inputs to full size first and summing the
per-batch gradients afterwards, and that sum is not a carried chain. Grouped
or broadcast key and value heads are allowed: the kernel treats them as
groups, in the order above.

## The exponentials from Lemma D's tables

The step-by-step program's exponentials (each key's weight exp(d), d the
interval of its computed shift l'_j - m') and the fused backward's (P'_j's
ends, exp(l_j - lse_g -+ R'_j) in the closed form above and exp of the
program's interval of l'_j - lse_g) take an end y of such an interval to
exp(y) rounded outward. Where the call has Lemma D's tables (`table_exp`,
`csrc/attention_tol.c`), the logits are sigma times exact scores (no float
mask, no input an earlier result, no flush) and sigma is not 0, each end is

    exp(y) = exp(delta_j) exp(c) exp(y - delta_j - c),

an identity for any exact c: delta_j = -|sigma| D_j 2**u is key j's exact logit
shift from the top key's (D_j its score's distance from the top key's at the
tables' unit), whose exponential the tables enclose in
[p_j, p_j + h_j] 2**-F as for the closed form's weights (Lemma D); c is 0 for the
program's weights and, for the given logsumexp, sigma's lower end times the
top key's score less lse_g, an exact dyadic, whose exponential is taken once
per row by Lemma X; y - delta_j - c is bounded outward with |sigma|'s ends at
the working precision plus 128 bits, and its exponential is Lemma W3's
(`exp_near_one`, `foundations.md`). Each factor is enclosed to its side, so
the product of the ends to that side encloses exp(y), and it is rounded to the
working precision outward. The tables' factor is used only where exp(delta_j)
is at least 2**-48 (its absolute 2**-F then keeps the working precision
relative to it), and Lemma W3 only where the rest is below 2**-4; otherwise the
end is Lemma X's as before. Every bound downstream uses the enclosure, so it
holds either way.

## Results as inputs

Every input of `tolerance` and `backward_tolerance` may be an earlier
result, including the given `out` and `logsumexp` (for example
`forward["output"]` and `forward["logsumexp"]`). The bounds take the stored
values' radii as `scaled_dot_product_attention.md` and
`scaled_dot_product_attention_backward.md` (Results as inputs) describe, with
the methods' points unchanged. For the fused backward:

- the given logsumexp lies within its radius rho of lse_g, so the closed form
  adds rho to k and uses |lse_g| + rho for the product with log2(e), and the
  step-by-step program widens the given logsumexp's interval by rho;
- the given output's radii enter D's exact sums like any exact factor's;
- a logsumexp element with radius inf or NaN makes its row's P unbounded,
  and an output element its row's D.

## Checks

ExactRef's test suite, developed separately, checks these bounds against
PyTorch 2.14.1 implementations on the development machine (macOS): the CPU
flash kernel forward and backward (online softmax,
blocks of 512 keys, weights rounded for bfloat16 and float16; with
`backward_blocks`, backwards past one block of queries or keys, up to 800
queries and 1100 keys, with grouped heads and masks; the forward in
bfloat16 and float16 with `exp_ulps=4096`, including two key blocks and a
weight below 2**-126), MPS's fused
math kernel (within the standard algorithm), and PyTorch compositions that
round as exp2, online softmax in blocks of 2 keys, and a bias computed in
bfloat16 do. With `mma`, it checks recorded outputs of PyTorch's CUDA
kernels on an NVIDIA GB10, in float16 and bfloat16:
- FlashAttention-2, forward and backward;
- the memory-efficient kernel, forward and backward (with `scale_ds` and
  `rounded_p`);
- cuDNN, forward and backward (with `scale_ds`). Its source is closed, so
  its methods are inferred from the recordings and from emulations of its
  arithmetic, not read from its code.

The recordings include non-causal and causal attention, head dimensions 64
and 128 (FlashAttention-2's split-KV kernel, with `split_kv`), a float mask,
grouped key and value heads (with `grouped_heads`), and a scale that is not
a power of two. They use c = 3, and c = 4 with `fast_log` for
FlashAttention-2. The
functions were measured on the GB10 over every float32 argument in the
ranges the kernels meet:
- `__logf` is within 1.47 2**-23 on [0.5, 2), and 2.28 ulps from 2 up;
- the split-KV merge's `logf` is within 0.87 ulps from 0.5 up;
- its `expf` is within 1.94 ulps on [-104, 1], and 1.41 2**-149 below
  2**-126.
