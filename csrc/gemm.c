/* gemm's epilogue (internal; proofs/gemm.md): the exact output alpha P + beta c per part from
   matmul's exact sums P (the product kernel) and C's parts through the channels, its IEEE value
   where a term is not finite, and Lemma G's bound: the normal form, intersected with the
   any-order form where the overflow check passes, and the any-order form. The constants are
   rounded up once; each element's error is then exact. */
#include "exactref.h"

#define GEMM_DYADS(g) {&(g)->half_r, &(g)->constant, &(g)->g, &(g)->theta, &(g)->tiny}

void gemm_channels_free(gemm_channels *x) {
    for (int p = 0; p < 2; p++) {
        for (int k = 0; k < 2; k++) {
            dyad_free(&x->p[p][k].s);
            dyad_free(&x->c[p][k].s);
        }
        dyad_free(&x->S[p]);
    }
}

/* One coefficient's channels (proofs/gemm.md, Channels): for a real v the one channel (v, 0); for
   a complex v = re + i im, part 0 reads (re, 0) and (-im, 1), part 1 (re, 1) and (im, 0). A zero
   coefficient leaves its channel out (a zero part of a nonzero v by the coefficient rule, and a
   zero v reads nothing). */
static int channels_of(gemm_channel (*to)[2], int count[2], const dyad *v, int parts) {
    static const int source[2][2] = {{0, 1}, {1, 0}}; /* per part, the source parts of v's re and im */
    int rc = 0;
    for (int p = 0; p < parts; p++) {
        count[p] = 0;
        for (int k = 0; !rc && k < parts; k++) {
            if (!v[k].m.n) continue;
            gemm_channel *x = &to[p][count[p]++];
            x->q = parts == 1 ? 0 : source[p][k];
            rc = dyad_copy(&x->s, &v[k]);
            if (!rc && parts == 2 && p == 0 && k == 1) x->s.m.neg = !x->s.m.neg; /* -im */
        }
    }
    return rc;
}

/* The channels of alpha and beta, each given by its parts (one for real values, re and im for
   complex ones). */
int gemm_channels_set(gemm_channels *x, const dyad *alpha, const dyad *beta, int parts) {
    for (int p = 0; p < 2; p++) {
        for (int k = 0; k < 2; k++) {
            dyad_init(&x->p[p][k].s);
            dyad_init(&x->c[p][k].s);
        }
        dyad_init(&x->S[p]);
    }
    x->parts = parts;
    x->alpha_real = parts == 1 || !alpha[1].m.n;
    int rc = channels_of(x->p, x->n_p, alpha, parts);
    if (!rc) rc = channels_of(x->c, x->n_c, beta, parts);
    /* S_p: the sum of part p's |sigma| */
    for (int p = 0; !rc && p < parts; p++)
        for (int k = 0; !rc && k < x->n_p[p]; k++) {
            dyad s = x->p[p][k].s; /* a read-only view of |sigma| */
            s.m.neg = 0;
            rc = dyad_add_exact(&x->S[p], &x->S[p], &s);
        }
    if (rc) gemm_channels_free(x);
    return rc;
}

void gemm_fold_init(gemm_fold *x) {
    for (int p = 0; p < 2; p++) {
        dyad_init(&x->mul[p]);
        dyad_init(&x->add[p]);
    }
}

void gemm_fold_free(gemm_fold *x) {
    for (int p = 0; p < 2; p++) {
        dyad_free(&x->mul[p]);
        dyad_free(&x->add[p]);
    }
}

/* alpha folded into an operand element in the compute format f (proofs/gemm.md, Overflow), from the parts' S_p: with one
   channel, fl(sigma x) <= (1 + r) S_p X + a; with two, each channel product rounded and their sum rounded once,
   (1 + r) ((1 + r) S_p X + 2 a) + a <= (1 + r)**2 (S_p X + 3 a). r = 2**-p; a = 2**(emin - p), or 2**emin with flush. Both
   parts of a fold have the same channels' magnitudes (channels_of), so either part's bound holds for the other. */
int gemm_fold_set(gemm_fold *x, const gemm_channels *ch, const floatfmt *f, int flush) {
    dyad one_r, a, t;
    dyad_init(&one_r);
    dyad_init(&a);
    dyad_init(&t);
    int rc = dyad_set_si(&one_r, 1, 0);
    if (!rc && !(rc = dyad_set_si(&t, 1, -(int64_t)(f->man_bits + 1))) && !(rc = dyad_add_exact(&one_r, &one_r, &t)))
        rc = dyad_set_si(&a, 1, flush ? f->emin : f->emin - (int64_t)(f->man_bits + 1));
    for (int p = 0; !rc && p < 2; p++) {
        int n = p < ch->parts ? ch->n_p[p] : 0;
        if ((rc = dyad_set_si(&x->mul[p], 0, 0)) || (rc = dyad_set_si(&x->add[p], 0, 0)) || !n) continue;
        if (n == 1) {
            if (!(rc = dyad_mul_exact(&x->mul[p], &ch->S[p], &one_r))) rc = dyad_copy(&x->add[p], &a);
        } else if (!(rc = dyad_mul_exact(&t, &one_r, &one_r)) && !(rc = dyad_mul_exact(&x->mul[p], &ch->S[p], &t)) &&
                   !(rc = dyad_mul_exact(&t, &t, &a))) {
            rc = dyad_set_si(&x->add[p], 3, 0);
            if (!rc) rc = dyad_mul_exact(&x->add[p], &x->add[p], &t);
        }
    }
    dyad_free(&one_r);
    dyad_free(&a);
    dyad_free(&t);
    return rc;
}

/* r = part p's fold bound at X, mul X + add, rounded up to w bits */
int gemm_fold_bound(dyad *r, const gemm_fold *x, int part, const dyad *X, int64_t w) {
    int rc = dyad_mul(r, &x->mul[part], X, w, 1);
    return rc ? rc : dyad_add(r, r, &x->add[part], w, 1);
}

/* Whether applying these channels is exact: per part at most one, of magnitude 1. */
static int unit_channels(const gemm_channel x[2][2], const int count[2], int parts) {
    for (int p = 0; p < parts; p++) {
        if (count[p] > 1) return 0;
        if (count[p] && !(x[p][0].s.m.n == 1 && x[p][0].s.m.w[0] == 1 && x[p][0].s.e == 0)) return 0;
    }
    return 1;
}

/* How beta c enters the sum (TERMS_*), given how a unit beta's c enters it (unit: TERMS_HELD where the compute format
   holds C's parts, TERMS_EXACT where it holds the arrays' but not a wider C of shape (), TERMS_ROUNDED where it is
   narrower than the arrays'): rounded where its channels are not exact, held where there is no c (gemm.md, step 3). */
int gemm_c_terms(const gemm_channels *ch, int unit) {
    int has_c = ch->n_c[0] || (ch->parts == 2 && ch->n_c[1]);
    if (!has_c) return TERMS_HELD;
    return unit_channels(ch->c, ch->n_c, ch->parts) ? unit : TERMS_ROUNDED;
}

void gemm_bounds_free(gemm_bounds *g) {
    dyad *all[] = GEMM_DYADS(g);
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
}

/* Lemma G's constants for one output batch in the compute format f: the channels ch, K terms per
   part of P (0 without alpha's terms), how beta c enters the sum (gemm_c_terms), W at least every entry the
   batch's matrices hold (used when alpha's channels are not exact), each rounded up to w bits.
   The normal form is for a real alpha only. */
int gemm_bounds_init(gemm_bounds *g, const floatfmt *f, int flush, int normal, int64_t K, const gemm_channels *ch, const gemm_fold *fold, int c_terms,
                     const dyad *W, int64_t w) {
    dyad *all[] = GEMM_DYADS(g);
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    dyad r, a, a2, top, absolute, t, u, G;
    dyad *tmp[] = {&r, &a, &a2, &top, &absolute, &t, &u, &G};
    for (size_t i = 0; i < sizeof tmp / sizeof *tmp; i++) dyad_init(tmp[i]);
    int64_t J = 0, L_c = 0;
    for (int p = 0; p < ch->parts; p++) {
        if (ch->n_p[p] > J) J = ch->n_p[p];
        if (ch->n_c[p] > L_c) L_c = ch->n_c[p];
    }
    g->ch = ch;
    g->normal = normal;
    g->flush = flush;
    g->has_p = J > 0 && K > 0;
    g->has_c = L_c > 0;
    g->has_infinity = f->special == SPECIAL_IEEE;
    g->K = g->has_p ? K : 0;
    J = g->has_p ? J : 0;
    g->kappa = g->has_p && !unit_channels(ch->p, ch->n_p, ch->parts) ? J : 0;
    g->kappa_c = g->has_c && c_terms == TERMS_ROUNDED ? L_c : 0;
    int64_t p = f->man_bits + 1;
    int rc = dyad_set_si(&r, 1, -p);
    if (!rc) rc = dyad_set_si(&a, 1, flush ? f->emin : f->emin - p);
    if (!rc) rc = dyad_set_si(&a2, flush ? 1 : 0, f->emin);
    if (!rc) rc = dyad_set_si(&g->tiny, 1, f->emin);
    /* top = max(1, |sigma|) over the channels; 1 / |sigma| <= 2**inverse for each */
    g->inverse = 0;
    if (!rc) rc = dyad_set_si(&top, 1, 0);
    for (int q = 0; !rc && g->has_p && q < ch->parts; q++)
        for (int k = 0; !rc && k < ch->n_p[q]; k++) {
            dyad s = ch->p[q][k].s; /* a read-only view of |sigma| */
            s.m.neg = 0;
            if (dyad_cmp(&s, &top) > 0) rc = dyad_copy(&top, &s);
            int64_t x = -(s.e + big_bits(&s.m) - 1);
            if (x > g->inverse) g->inverse = x;
        }
    int64_t depth = 0;
    if (!rc && normal) {
        /* top (K a + (K - 1) a2) + kappa K a (1 + W) + ((2 kappa_c - 1)+ + join) a, over a tree of depth K + 2 */
        int64_t join = g->K && g->has_c, edges = g->K ? g->K - 1 : 0;
        g->c_factor = g->K + g->kappa_c;
        if (!(rc = dyad_times(&t, &a, g->K, w)) && !(rc = dyad_times(&u, &a2, edges, w)) && !(rc = dyad_add(&t, &t, &u, w, 1)))
            rc = dyad_mul(&absolute, &top, &t, w, 1);
        if (!rc && !(rc = dyad_set_si(&t, 1, 0)) && !(rc = dyad_add(&t, &t, W, w, 1)) && !(rc = dyad_mul(&t, &a, &t, w, 1)) &&
            !(rc = dyad_times(&t, &t, g->kappa * g->K, w)))
            rc = dyad_add(&absolute, &absolute, &t, w, 1);
        if (!rc && !(rc = dyad_times(&t, &a, (2 * g->kappa_c - 1 > 0 ? 2 * g->kappa_c - 1 : 0) + join, w))) rc = dyad_add(&absolute, &absolute, &t, w, 1);
        depth = g->K + 2;
    } else if (!rc) {
        /* top (J K a + edges a2) + kappa K a + (2 kappa - 1)+ K a W + c_abs L_c a, over edges + kappa: c_abs for the c
           channels' roundings (c_extra), or for the merge of an exact c that f does not hold */
        int64_t leaves = J * g->K + L_c;
        g->edges = leaves > 0 ? leaves - 1 : 0;
        g->c_extra = g->kappa_c ? 1 : 0;
        int64_t c_abs = g->c_extra || (c_terms == TERMS_EXACT && g->K) ? 1 : 0;
        if (!(rc = dyad_times(&t, &a, J * g->K, w)) && !(rc = dyad_times(&u, &a2, g->edges, w)) && !(rc = dyad_add(&t, &t, &u, w, 1)))
            rc = dyad_mul(&absolute, &top, &t, w, 1);
        if (!rc && !(rc = dyad_times(&t, &a, g->kappa * g->K, w))) rc = dyad_add(&absolute, &absolute, &t, w, 1);
        if (!rc && !(rc = dyad_mul(&t, &a, W, w, 1)) && !(rc = dyad_times(&t, &t, (2 * g->kappa - 1 > 0 ? 2 * g->kappa - 1 : 0) * g->K, w)))
            rc = dyad_add(&absolute, &absolute, &t, w, 1);
        if (!rc && !(rc = dyad_times(&t, &a, c_abs * L_c, w))) rc = dyad_add(&absolute, &absolute, &t, w, 1);
        depth = g->edges + g->kappa;
    }
    /* G = (1 + r)**depth; half_r = G r / 2; constant = G absolute */
    if (!rc && !(rc = dyad_set_si(&t, 1, 0)) && !(rc = dyad_add(&t, &t, &r, w, 1)) && !(rc = dyad_power(&G, &t, depth, w, 1)) &&
        !(rc = dyad_copy(&g->g, &G))) {
        dyad half = r; /* a read-only view of r / 2 */
        if (!(rc = dyad_shift(&half, -1)) && !(rc = dyad_mul(&g->half_r, &G, &half, w, 1))) rc = dyad_mul(&g->constant, &G, &absolute, w, 1);
    }
    /* theta: the largest finite value plus half its ulp */
    if (!rc) rc = floatfmt_threshold(&g->theta, NULL, f);
    /* an operand element scaled by alpha must not overflow: each part's fold bound at W below theta */
    g->operand_ok = 1;
    for (int q = 0; !rc && g->kappa && g->operand_ok && q < ch->parts; q++)
        if (ch->n_p[q] && !(rc = gemm_fold_bound(&t, fold, q, W, w))) g->operand_ok = dyad_cmp(&t, &g->theta) < 0;
    for (size_t i = 0; i < sizeof tmp / sizeof *tmp; i++) dyad_free(tmp[i]);
    if (rc) gemm_bounds_free(g);
    return rc;
}

/* r += x * c exactly, for a dyadic x. */
static int add_product(dyad *r, const dyad *x, const dyad *c, dyad *t) {
    int rc = dyad_mul_exact(t, x, c);
    return rc ? rc : dyad_add_exact(r, r, t);
}

/* Lemma G's interval for part `part` of one element: P's exact quantities per source part (P[q],
   NULL without alpha's terms; its mass, N and moved hold the deviation of A's and B's radii), C's
   finite parts c[y] and their radii rc[y] (NULL without beta, rc[y] NULL where C is exact):
   [lo, hi] = s -+ B, s the part's alpha P + beta c, *checked where the overflow check passes, else
   *flags: the sides the final value may reach, and with normal_flags (the call's standard, apart from
   g's) every flag where another value may overflow (proofs/gemm.md, Overflow and Results as inputs). */
int gemm_box(dyad *lo, dyad *hi, int *checked, int *flags, const gemm_bounds *g, int normal_flags, int part, const product_sums *const P[2],
             const dyad *const c[2], const dyad *const rc_c[2]) {
    dyad s, M, X2, moved, unscaled, Cv, err, t, v, q;
    dyad *all[] = {&s, &M, &X2, &moved, &unscaled, &Cv, &err, &t, &v, &q};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    const gemm_channels *ch = g->ch;
    int rc = 0;
    *checked = 1;
    *flags = 0;
    for (int k = 0; !rc && g->has_p && k < ch->n_p[part]; k++) {
        const gemm_channel *x = &ch->p[part][k];
        const product_sums *r = P[x->q];
        if (!r) continue;
        dyad sigma = x->s, view; /* read-only views: |sigma|, and P's quantities at its exponent */
        sigma.m.neg = 0;
        dyad_view(&view, &r->s, r->e);
        if ((rc = add_product(&s, &view, &x->s, &t))) break;
        dyad_view(&view, &r->mass, r->e);
        if ((rc = add_product(&M, &view, &sigma, &t))) break;
        if (dyad_cmp(&view, &unscaled) > 0 && (rc = dyad_copy(&unscaled, &view))) break;
        dyad_view(&view, &r->moved, r->e);
        if ((rc = add_product(&moved, &view, &sigma, &t))) break;
        if (g->normal) {
            dyad_view(&view, &r->N, r->e);
            rc = add_product(&X2, &view, &sigma, &t); /* N, which holds 2 mass already */
        }
    }
    for (int k = 0; !rc && g->has_c && k < ch->n_c[part]; k++) {
        const gemm_channel *x = &ch->c[part][k];
        const dyad *cv = c[x->q], *rv = rc_c[x->q];
        if (!cv) continue;
        dyad tau = x->s, mag = *cv; /* read-only views of |tau| and |c| */
        tau.m.neg = 0;
        mag.m.neg = 0;
        if ((rc = add_product(&s, cv, &x->s, &t)) || (rc = dyad_mul_exact(&v, &mag, &tau)) || (rc = dyad_add_exact(&Cv, &Cv, &v))) break;
        if (g->flush && mag.m.n && dyad_cmp(&mag, &g->tiny) < 0 && (rc = dyad_add_exact(&moved, &moved, &v))) break; /* may be read as 0 */
        /* C's radius: Dc = |tau| rc counts with C, and once more in the moved part */
        if (rv && rv->m.n && !(rc = dyad_mul_exact(&v, rv, &tau)) && !(rc = dyad_add_exact(&Cv, &Cv, &v))) rc = dyad_add_exact(&moved, &moved, &v);
    }
    /* X2: normal N + 2 (1 + kappa) M + 2 c_factor C; any order 2 ((1 + kappa) M + edges (M + C) + c_extra C) */
    if (!rc) {
        if (g->normal) {
            if (!(rc = dyad_set_si(&q, 2 * (1 + g->kappa), 0)) && !(rc = add_product(&X2, &M, &q, &t)) && !(rc = dyad_set_si(&q, 2 * g->c_factor, 0)))
                rc = add_product(&X2, &Cv, &q, &t);
        } else {
            if (!(rc = dyad_set_si(&q, 2 * (1 + g->kappa + g->edges), 0)) && !(rc = add_product(&X2, &M, &q, &t)) &&
                !(rc = dyad_set_si(&q, 2 * (g->edges + g->c_extra), 0)))
                rc = add_product(&X2, &Cv, &q, &t);
        }
    }
    /* B = X2 G r / 2 + G absolute + moved G */
    if (!rc && !(rc = dyad_mul_exact(&err, &X2, &g->half_r)) && !(rc = dyad_add_exact(&err, &err, &g->constant)))
        rc = add_product(&err, &moved, &g->g, &t);
    if (!rc) {
        dyad n = err; /* a read-only view of -B */
        n.m.neg = n.m.n != 0;
        if (!(rc = dyad_add_exact(lo, &s, &n))) rc = dyad_add_exact(hi, &s, &err);
    }
    if (rc) goto done;
    /* the check: the unscaled values within B / |sigma| of theta, and M + C + B below it */
    int inner = !g->operand_ok;
    if (!inner && g->has_p) {
        dyad scaled = err; /* a read-only view of B 2**inverse */
        if (!(rc = dyad_shift(&scaled, g->inverse)) && !(rc = dyad_add_exact(&t, &unscaled, &scaled))) inner = dyad_cmp(&t, &g->theta) >= 0;
    }
    if (!rc && !(rc = dyad_add_exact(&t, &M, &Cv)) && !(rc = dyad_add_exact(&t, &t, &err))) *checked = !inner && dyad_cmp(&t, &g->theta) < 0;
    if (rc || *checked) goto done;
    if (normal_flags) {
        int any = inner;
        if (!any && !(rc = dyad_add_exact(&t, &M, &err))) any = dyad_cmp(&t, &g->theta) >= 0;
        if (!rc && !any && !(rc = dyad_add_exact(&t, &Cv, &err))) any = dyad_cmp(&t, &g->theta) >= 0;
        if (any) *flags = FLAG_ALL;
    }
    if (!rc && !(rc = dyad_add_exact(&t, &s, &err)) && dyad_cmp(&t, &g->theta) >= 0) *flags |= FLAG_PLUS_INF;
    if (!rc) {
        dyad n = s; /* B - s */
        n.m.neg = !n.m.neg && n.m.n;
        if (!(rc = dyad_add_exact(&t, &err, &n)) && dyad_cmp(&t, &g->theta) >= 0) *flags |= FLAG_MINUS_INF;
    }
    if (*flags && !g->has_infinity) *flags = FLAG_NAN;
done:
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
    return rc;
}

/* The IEEE value of part `part` of alpha P + beta c where a term is not finite: each channel's
   term is an IEEE product of an exact nonzero coefficient, so a non-finite source (P's part by
   matmul's rule, NULL without alpha's terms; C's part, absent without beta) gives NaN for NaN and
   else the infinity with the signs' product; then the sum's rule: NaN where a term is NaN or
   infinities of both signs meet. *kind is VALUE_FINITE when every term is finite. */
void gemm_special(int *kind, int *neg, const gemm_channels *ch, int part, const product_sums *const P[2], const int c_kind[2], const int c_neg[2]) {
    int nan = 0, plus = 0, minus = 0;
    for (int side = 0; side < 2; side++) {
        int count = side ? ch->n_c[part] : ch->n_p[part];
        for (int k = 0; k < count; k++) {
            const gemm_channel *x = side ? &ch->c[part][k] : &ch->p[part][k];
            int source_kind, source_neg;
            if (side) {
                source_kind = c_kind[x->q];
                source_neg = c_neg[x->q];
            } else {
                if (!P[x->q]) continue;
                source_kind = P[x->q]->kind;
                source_neg = P[x->q]->neg;
            }
            if (source_kind == VALUE_FINITE) continue;
            if (source_kind == VALUE_NAN)
                nan = 1;
            else if (source_neg != (x->s.m.neg != 0))
                minus = 1;
            else
                plus = 1;
        }
    }
    *kind = nan || (plus && minus) ? VALUE_NAN : plus || minus ? VALUE_INFINITE : VALUE_FINITE;
    *neg = *kind == VALUE_INFINITE && minus;
}
