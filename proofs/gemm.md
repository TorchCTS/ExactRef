# gemm: proof

`exactref.gemm(A, B, C, alpha=1, beta=1, reduce_batch=False)` returns
D = beta C + alpha (A @ B). This file proves that its reference output is the
exact value rounded once, that `tolerance()` bounds the normal standard below
and `tolerance(standard="any_order")` every order that does not overflow, and
that `normal()` is one implementation of the normal standard. Notation, units
and Lemmas 1 to 3, S, L, O and E are in `foundations.md`.

## What gemm returns

A @ B is matmul's product: NumPy batching and 1-D promotion. C is broadcast
to the output's shape. With `reduce_batch=True` the output's batch dimensions
are summed: the batches fold into the contraction, as in matmul's backward
(row i of A's matrices side by side, column j of B's stacked), so each output
element is one sum of products over all batches (PyTorch's `addbmm` rounds
after each batch instead: that is a loop of gemm calls, results as inputs).
Operands of different formats are promoted as `add` promotes them; a C of
shape () in a wider format takes no part in the promotion and is taken by
value, in its own format (with beta 1 or -1, step 3 below takes beta c
exactly though F does not hold it, wherever F holds A's and B's formats). alpha and beta are exact dyadics, held in
formats that represent them.

For one output element, with k the contraction length (all batches with
`reduce_batch`),

    s = beta c + alpha P,   P = sum_{q=1..k} a_q b_q.

**Unused inputs.** With beta = 0 the C term is absent: c is not read, so a
NaN or inf in C, or an uncertain C's radius, plays no part. With alpha = 0
the product term is absent likewise. This is PyTorch's and BLAS's
definition; the expression then has no such term, so no mathematics is
contradicted.

## The exact and the reference output

P is matmul's exact sum: an integer times 2**e per batch (`matmul.md`, The
reference output). alpha, beta and c are dyadics, so s is an exact dyadic,
computed as one integer at the finest of the exponents. The reference output
is s rounded once to the output format O, in the requested direction;
`exact` holds s. Integer outputs are exact and wrap (or raise or saturate)
once. For complex values each part of s is computed exactly from the parts:
Re(alpha P) = alpha_re P_re - alpha_im P_im and so on.

**Special values.** A non-finite term follows IEEE rules on the exact
expression: matmul's rule gives P, then alpha P and beta c are IEEE products
and s their IEEE sum. For complex values each part follows these rules
separately: P's parts by matmul's rule, and alpha and beta as exact
coefficients, through the channels below, so a zero part of a nonzero alpha
or beta adds no term (beta = 1 and c = inf + 1i give an imaginary part of 1
plus P's); alpha = 0 and beta = 0 remove their term (unused inputs).
The unused-inputs rule comes first.

## Channels

Fix an output element and a part p (the element itself when it is real). The
terms of P's part q are matmul's: the k products for real values, the 2k real
products a_re b_re, -a_im b_im (q = re) or a_re b_im, a_im b_re (q = im) for
complex ones. Write K for their number, mass_q for their magnitude sum, and
N_q >= 2 (mass_q + sum of |node|) for the normal orders (`foundations.md`,
The normal bound: equal in the product kernel, every node exact).

Part p of alpha P is a sum of channels sigma_j P_(q_j): for real values the
one channel (alpha, P); for complex values

    Re(alpha P) = alpha_re P_re + (-alpha_im) P_im,
    Im(alpha P) = alpha_re P_im + alpha_im P_re,

leaving out channels with sigma = 0. Part p of beta c is likewise a sum of
c channels tau_l c_(y_l). Write J for the number of channels (0 when
alpha = 0), L_c for the number of c channels (0 when beta = 0),
M_alpha = sum_j |sigma_j| mass_(q_j) and C = sum_l |tau_l| |c_(y_l)|. With
flush, moved_alpha = sum_j |sigma_j| moved_(q_j) (matmul's moved: the terms a
flush of a subnormal factor may make 0) and Cmoved is the part of C whose c
part is subnormal in F (it may be read as 0).

## The model

An implementation computes part p in the compute format F as follows.

1. Each term is rounded to F, or merged into an addition by a fused
   multiply-add (matmul's model).
2. alpha is applied once on every term's path, in one of these places: to
   an operand element (a or b replaced by alpha a or alpha b rounded to F
   before the product), to a term, to a node of the terms' summation tree
   (a partial sum, such as a block), or to the whole sum. Applying it
   computes each part of the scaled value from its channels: each product
   sigma v rounded to F or fused, and with two channels their sum rounded
   once. With alpha 1 or -1 (or i or -i for complex values) it is exact.
3. beta c is computed per part from its c channels the same way: exact when
   beta is 1 or -1 (or i or -i) and F holds A's and B's formats (the default
   compute format always does); F need not hold C's format then (a wider C of
   shape (), taken by value), and the addition that merges such an exact
   beta c rounds a value that is not one of F's. An F narrower than A's and
   B's formats rounds beta c as it rounds the products.
4. The values are added in F by one tree: the terms' tree with alpha's
   applications, and beta c joined at any node (as the accumulator's start,
   as PyTorch's CPU kernels and Netlib's reference BLAS without a transposed A
   do, or at the root, as the reference BLAS does with A transposed).
   For the normal standard the terms' tree is one of matmul's normal orders
   (index order, or lanes 2 and 4) and alpha's places do not change it; for
   any order, any tree.
5. The result is rounded to O.

This covers the reproduced PyTorch CPU kernels: y = fl(beta C), then the
products in the kernel's tree with alpha applied per block,
y = fma(alpha, block, y); or alpha folded into an operand; or the whole sum
followed by fl(fl(alpha S) + fl(beta C)) or a fused form of it. PyTorch's
`addbmm` rounds each batch's product to the dtype, which is a chain.

## Lemma G: the error of the model

Let kappa = 0 when alpha's application is exact, else J (the roundings in
one application: 1, or 2 with two channels); kappa_c = 0 when beta c is
exact, else L_c; W at least every part of every stored entry of A and B the
element reads. With r, a, a' as in `foundations.md` (Units):

**(normal, alpha real)** |s' - s| <= Growth(K + 2) (r X + Abs + |alpha| moved + Cmoved),

    X   = |alpha| (N/2 + (1 + kappa) mass) + (K_c + kappa_c) C,
    Abs = max(1, |alpha|) (K a + (K - 1) a') + kappa K a (1 + W) + (2 kappa_c - 1)+ a + a,

with N, mass and moved those of P's part p, and K_c = K when alpha != 0
(else 0, and with alpha = 0 the terms and the join are absent).

**(any order)** with n = J K + L_c leaves,

    |s' - s| <= Growth(n - 1 + kappa) (r X' + Abs' + moved_alpha + Cmoved),
    X'   = (1 + kappa) M_alpha + (n - 1) (M_alpha + C) + [kappa_c > 0] C,
    Abs' = max(1, max_j |sigma_j|) (J K a + (n - 1) a') + kappa K a + (2 kappa - 1)+ K a W + c_abs L_c a,

with c_abs = 1 where kappa_c > 0, or where beta c is exact, F does not hold
C's format and K > 0 (beta c is merged into the terms), else 0.

**Proof.** Unroll the computation as in Lemma S. Every rounding moves its
exact input y by at most r |y| + h, with h at most a (a' for an addition of
two values of F, which has no absolute part without flush; a fused merge of
an exact value has a, and that value's own rounding is then absent, as has
the addition that merges an exact beta c that F does not hold). An
error made below an application of alpha is multiplied by |sigma| when the
application scales it, and by (1 + r) at every rounding above it. Measure
every value in output units: a value below an application, scaled by its
|sigma|. Then the error of s' is at most the sum, over the roundings, of
r times the output-unit magnitude of the rounded exact value, plus its h
(scaled by |sigma| below an application, so by at most max(1, |sigma|)),
each multiplied by (1 + r) once per rounding above it.

Normal, the roundings and their magnitudes:

- The K terms: |alpha| mass. Their h: K a, below alpha.
- The tree's K - 1 additions: |alpha| sum |v_node|, plus C for each node
  that contains beta c. beta c joins one node, which adds one addition of
  magnitude at most |alpha| mass + C, and every node on its path to the root
  contains it: at most K nodes. Their h: K - 1 times a' below alpha, and a
  for the join (fused or not).
- alpha's applications: their places cut every path once, so the values
  they scale cover disjoint sets of terms and their magnitudes add up to at
  most |alpha| mass, once per rounding in an application (kappa). Their h:
  at most K applications, each a; applied to an operand element, the h of
  fl(alpha a) is multiplied by the other factor b in the product, so the
  sum over the K terms is at most K a W.
- beta c: kappa_c roundings of magnitude at most C (two products and their
  sum with two channels), h at most (2 kappa_c - 1) a; an exact beta c that F
  does not hold adds no rounding of its own, and its join's h is the a
  above.

Adding up, with sum |v_node| <= N/2 - mass, gives r X and Abs. A flushed
factor makes its term 0: an error of |t| in its own units, |alpha| moved in
all; a flushed c part, Cmoved. The most roundings above any rounding: K - 1
additions in the terms' tree, the join, and kappa <= 1 for a real alpha, so
K + 1; Growth(K + 2) covers it and the complex beta's two roundings below
its join.

Any order: the same accounting with every node's magnitude at most
M_alpha + C (each node is a partial sum of the n leaves, J K scaled terms
and L_c c-channel values), n - 1 additions, and kappa roundings per
application, two per operand element with two channels (its parts are
two-term sums), whose h is at most 3a, hence (2 kappa - 1) K a W; a
c channel's product is a leaf rounding, and their sum is one of the tree's
additions. An exact beta c that F does not hold is not rounded on its own,
but the one addition that merges it into the terms (when K > 0; one per c
channel) has h = a instead of a', which c_abs L_c a counts. A path has at
most n - 1 additions and kappa roundings of alpha's application above a
term's rounding. []

With alpha 1 or -1, kappa = 0 and the normal form is matmul's Lemma S over
the terms with beta c joined: its N/2 + mass and K a + (K - 1) a' with the
join's additions.

**Normal with a complex alpha.** Folding a complex alpha into an operand
gives terms a'_re b_re, -a'_im b_im whose channel P_im is P_im's terms with
each pair's two terms swapped; matmul's normal orders do not track that
order. So for alpha with a nonzero imaginary part the normal tolerance is
the any-order one, with the normal standard's flags (Overflow), as for any
alpha; the infinities rule is the same too.

## Overflow

The bounds hold when no rounding overflows. Every exact value a rounding
sees is at most, in its own units, L = max(mass_q (each channel's unscaled
terms and nodes), M_alpha + C (output units), C), and part p of an operand
element's scaled value is at most S_p W, S_p the sum of part p's |sigma_j|.
A computed value is within its error of the exact one: at most B in output
units, at most B / |sigma_j| in channel j's units. Part p of alpha x rounded
to F, every part of x at most X, is at most F_p(X) = (1 + r) S_p X + a with
one channel; with two, each channel product is rounded and their sum rounded
once, so it is at most (1 + r)((1 + r) S_p X + 2 a) + a <= F_p(X) =
(1 + r)**2 (S_p X + 3 a) (`gemm.c`'s `gemm_fold_set`). So if

    L + B max(1, max_j 1 / |sigma_j|) < Theta_F   and   F_p(W) < Theta_F for every part p,

no rounding overflows and the bound B covers every implementation; this is
the check (1 / |sigma| rounded up to a power of two). The normal tolerance is
the normal bound when its check passes, else the any-order bound; the
any-order tolerance is its bound. Each holds for every finite output even
where its check fails (`foundations.md`, Finite outputs and overflow). Where
the check fails, the flags are the sides on which the final value S +- B may
reach Theta_F, and under the normal standard every flag where a product or P
itself may overflow (the first check's terms other than the final sum, and
the second check), since its sign is not known (`gemm.c`'s `gemm_box`).

**Infinities.** `foundations.md`'s infinities rule (Results as inputs) holds
under both standards, with these events and sides (`infinities_meet`). Its
events: an infinity C's stored values may pass through beta's parts, the
exact value's own (A's, B's or C's), and a value rounded past Theta_F on its
own. For channel j's term t = x y (each factor anywhere within its radius)
that is, by where alpha is applied: t itself (alpha applied above it),
|sigma_j| (|t| (1 + r) + a) (to the term), |sigma_j| (|x| + r_x) (to an
operand element where part p has one channel: an infinite factor, its
product with y an infinity, NaN where y may be 0), and (|sigma_j| (|x| + r_x)
(1 + r) + a) (|y| + r_y) (that factor's product), each on t's side, flipped
where sigma_j < 0; where part p has two channels (a complex alpha), the
folded element's part, at most F_p(X), and its product, at most F_p(X) Y, X
and Y the two elements' largest part plus radius, on both sides and NaN (its
sign mixes the element's two parts); and beta c's
channel products |tau_l| (|c| + r_C) on c tau_l's side (both sides where C's
radius spans 0). Each of these is also a flag, unless the exact value is
infinite. Its sides: a node below alpha's application is a node of channel
j's tree over its K unscaled terms (Lemma O (g)), its side flipped where
sigma_j < 0. Every other value is a node of the tree with leaves of an
effective sign: replacing the leaves of the other one by 0 can only raise it,
since multiplying by sigma_j keeps the order of its channel's values in
effective units. With M = 2 max(1, |sigma|): each leaf is at most (1 + r)**2
times its effective magnitude (|sigma_j| |t|, or |tau_l| (|c| + r_C)) plus
M a (2 + W) (the operand form's a |y| at most a W); an application of alpha
multiplies by |sigma_j| and adds one rounding (a); an addition adds one
rounding (a'), scaled by |sigma_j| below an application (at most M a'). A
path has at most n - 1 additions, one application and two roundings of its
leaf, n the part's leaves (n_p K + n_c), so every such node of one sign is at
most

    (1 + r)**(n + 2) (S + n M (a (3 + W) + a')),

S the effective magnitudes of that sign's leaves; beta c's own sum (a
complex beta's two channel products) is one of these nodes. As in Lemma O (g),
a side with one leaf reaches Theta_F only by that leaf's own overflow. The
three single-channel scaled forms of a term are at most |sigma| (1 + r)**2 X Y +
(|sigma| + 1) a (1 + X + Y), X and Y the row's and the column's largest
|x| + r, so where that, X Y, |sigma| X and |sigma| Y stay below Theta_F, and
with two channels F_p(X), F_p(Y), F_p(X) Y and F_p(Y) X (S_p X below Theta_F
is not enough: the fold rounds twice), no term rounds past Theta_F on its own.
The terms are then examined one by one only where an event occurs and k X Y
(each of one sign at most X Y) may reach Theta_F by Lemma O (g).

## Results as inputs

A's and B's radii move each part q of P's terms by Delta_q
(`product_deviation`, per part); C's radius r_C,y bounds the move of
c's part y, so the c channels move by Dc = sum_l |tau_l| r_C,y(l). As in
`matmul.md`, mass becomes mass + Delta, N becomes N + 2 Delta, C becomes
C + Dc, and the moved quantities gain |sigma| Delta and Dc; W bounds the
stored values (entry plus radius). An output part reading a radius that is
inf has no bound: through P (alpha != 0), any such part in the A row or B
column it reads, and through its c channels, such a part of c they read.
Flagged stored values pass on as in `matmul.md`: A's and B's, through P, as
every flag; C's, through the exact beta parts of the channels, with their
signs. With flush, radii are widened to cover 0 where a stored value may
flush, as there.

## The tolerance

**Claim.** For each output element, `tolerance()` is at least
|output - ref| for every implementation of the normal standard, and
`tolerance(standard="any_order")` for every implementation of the model that
does not overflow.

**Proof.** s, the masses, N, moved and Delta are exact integers per batch
(the same quantities matmul's tolerance uses), and C,
Cmoved and Dc are exact per element. Lemma G gives the interval
[s - B, s + B] and the overflow check its flags. Under the normal standard,
where the check passes, no rounding of a normal implementation overflows; the
any-order form holds for every implementation of the model in which no
rounding overflows (Lemma G), so it covers the normal implementations too, and
B is the smaller of the two forms (the intersection of two intervals centered
on s). Lemma 3 turns the interval and the flags into
the bounds, one per part for complex values. []

## Tensor cores

`gemm_mma(step, reduction=None)` is a tensor-core kernel with a float32
epilogue (`foundations.md`, Lemma T, its gemm paragraph). A and B are real,
in one format the tensor cores multiply: float16, bfloat16, or float32 values
that are tf32 values (A's and B's promoted format, whatever C's is). C is in
a format float32 holds, as the epilogue holds it. alpha and beta are
real values of float32. Per element:
- s and M are matmul's exact sums for the element;
- c is C's element;
- d = alpha s + beta c exactly;
- M' = |alpha| M + |beta c|;
- Pos and Neg are the totals of the positive and the negative terms among
  alpha t_i and beta c: (|alpha| M + alpha s) / 2 and (|alpha| M - alpha s) / 2,
  plus beta c on its side;
- sigma is as for `matmul_mma` (`matmul.md`, Algorithms), from A's and B's format.

Lemma T gives [d - G' M' - H', d + G' M' + H'] for every finite output, with
its flags, and Lemma 3 gives the bounds, also where the reference rounds past
the output format's largest value. With alpha = 0, A and B are not
read and the interval is beta c rounded once, so G' and H' still bound it.
Earlier results as inputs widen d by alpha times the deviation and by beta
times C's radius, and M' by the same, as in Lemma G. Float32 A and B are
arrays of tf32 values, never earlier results, as for `matmul_mma`; C may be
any earlier result.

## The normal answer

`normal()` computes P's parts as matmul's normal answer before its final
rounding (index order with the slots, Lemma E), then per part
fl(alpha P') and fl(beta c') from their channels as in the model (each
product rounded, then their sum; exact when alpha or beta is 1 or -1, or i
or -i, beta c' then taken as it is, which F need not hold), then fl of their
sum in F (Lemma E's fused sum where beta c' is exact), then rounds to O. It
is the model with the terms in index order, alpha applied to the whole sum
and beta c joined at the root, so the normal tolerance covers it.

## Backward

`gemm.backward(A, B, C, grad)` returns grad_A = conj(alpha) grad @ B^H,
grad_B = conj(alpha) A^H @ grad and grad_C = conj(beta) grad, each summed
over the copies that broadcasting (and `reduce_batch`) made of its operand.
grad_A and grad_B are gemm with beta = 0 on matmul's backward operands
(`matmul.md`, Backward): each element is one exact value rounded once, and
its tolerance is Lemma G's with alpha replaced by conj(alpha). PyTorch
computes the product and then scales it by conj(alpha), which is alpha
applied to the whole sum. grad_C is one product whose rows are grad's
copies and whose column holds conj(beta) (`add.md`, The backward).
