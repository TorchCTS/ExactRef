/* The attention forward's tolerance (proofs/scaled_dot_product_attention.md, The tolerance), per row of a tile:
   the closed form, every quantity an interval of dyadics at its precision, the weights from Lemma D's tables (or
   Lemma X without them) and exp(R) - 1 from Lemma W, a row whose validity conditions fail left to the caller
   (status 1); the step-by-step program, which bounds every row; and their constants from the call's settings. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

/* exp(x) - 1 for x >= 0 rounded to the side: Lemma W below 2**-4, else Lemma X */
int expm1_end(dyad *r, const dyad *x, int64_t p, int up) {
    if (!dyad_sign(x)) return dyad_set_si(r, 0, 0);
    dyad t;
    dyad_init(&t);
    int rc = dyad_rnd(&t, x, 64, up);
    if (!rc) {
        int64_t bits = big_bits(&t.m);
        if (bits && bits < 64 && (rc = big_shl(&t.m, &t.m, 64 - bits)) == 0) t.e -= 64 - bits; /* t nonzero */
    }
    if (!rc && t.e + 64 <= -4) {
        uint64_t m;
        int64_t e;
        exp_small(&m, &e, t.m.w[0], -t.e, up);
        rc = big_set_word(&r->m, m, 0);
        r->e = r->m.n ? -e : 0;
    } else if (!rc) {
        dyad one;
        dyad_init(&one);
        rc = exp_end(&t, x, p + 8, up);
        if (!rc && !(rc = dyad_set_si(&one, -1, 0))) rc = dyad_add(r, &t, &one, p, up);
        dyad_free(&one);
    }
    dyad_free(&t);
    return rc;
}

/* Whether exp(x) overflows every format: x at least 2**20 (the formats' largest values are below
   2**(2**16)), so a rounding of it may overflow (Lemma 1) and the step has no bound. */
int exp_overflows(const dyad *x) { return dyad_sign(x) > 0 && top_bit(x) >= 20; }


void att_tol_init(att_tol *t) {
    memset(t, 0, sizeof *t);
    for (int i = 0; i < ATT_ROUNDINGS; i++) dyad_init(&t->rnd[i].ulps);
    round_bound *l[2] = {&t->normalize, &t->add};
    for (int i = 0; i < 2; i++) {
        dyad_init(&l[i]->ulps);
        dyad_init(&l[i]->largest);
        dyad_init(&l[i]->threshold);
    }
    for (int i = 0; i < ATT_CONSTS; i++) dyad_init(&t->c[i]);
    dyad_init(&t->sig_lo);
    dyad_init(&t->sig_hi);
    dyad_init(&t->B_lo);
    dyad_init(&t->B_hi);
    dyad_init(&t->ln2_lo);
    dyad_init(&t->ln2_hi);
    dyad_init(&t->log2e_lo);
    dyad_init(&t->log2e_hi);
    dyad_init(&t->abs_lo);
    dyad_init(&t->abs_hi);
}

void att_tol_free(att_tol *t) {
    for (int i = 0; i < ATT_ROUNDINGS; i++) dyad_free(&t->rnd[i].ulps);
    round_bound *l[2] = {&t->normalize, &t->add};
    for (int i = 0; i < 2; i++) {
        dyad_free(&l[i]->ulps);
        dyad_free(&l[i]->largest);
        dyad_free(&l[i]->threshold);
    }
    for (int i = 0; i < ATT_CONSTS; i++) dyad_free(&t->c[i]);
    if (t->per_n) {
        for (size_t i = 0; i < (t->nmax + 1) * ATT_PER_N; i++) dyad_free(&t->per_n[i]);
        free(t->per_n);
    }
    free(t->per_n_has);
    dyad_free(&t->sig_lo);
    dyad_free(&t->sig_hi);
    dyad_free(&t->B_lo);
    dyad_free(&t->B_hi);
    dyad_free(&t->ln2_lo);
    dyad_free(&t->ln2_hi);
    dyad_free(&t->log2e_lo);
    dyad_free(&t->log2e_hi);
    dyad_free(&t->abs_lo);
    dyad_free(&t->abs_hi);
    sum_bounds_free(&t->scores);
}

/* sigma's enclosure, B (how far the computed factor may be from sigma), ln 2, and the scores'
   accumulation bounds, from the loaded constants. */
int att_tol_prepare(att_tol *t, const att_call *a, const floatfmt *scores, int64_t w) {
    int64_t p = t->p;
    int rc;
    if (a->den.n) {
        dyad x, y;
        dyad_init(&x);
        dyad_init(&y);
        rc = big_copy(&x.m, &a->num);
        if (!rc && !(rc = big_copy(&y.m, &a->den)) && !(rc = dyad_div(&t->sig_lo, &x, &y, p, 0))) rc = dyad_div(&t->sig_hi, &x, &y, p, 1);
        dyad_free(&x);
        dyad_free(&y);
    } else {
        dyad e, r, one;
        dyad_init(&e);
        dyad_init(&r);
        dyad_init(&one);
        rc = dyad_set_si(&e, a->E, 0);
        if (!rc) rc = dyad_set_si(&one, 1, 0);
        if (!rc && !(rc = dyad_sqrt(&r, &e, p + 4, 1))) rc = dyad_div(&t->sig_lo, &one, &r, p, 0);
        if (!rc && !(rc = dyad_sqrt(&r, &e, p + 4, 0))) rc = dyad_div(&t->sig_hi, &one, &r, p, 1);
        dyad_free(&e);
        dyad_free(&r);
        dyad_free(&one);
    }
    /* |sigma|'s ends, for the weights without tables */
    if (!rc) {
        ivl sig, mag;
        sig.lo = t->sig_lo; /* read-only views */
        sig.hi = t->sig_hi;
        ivl_init(&mag);
        if (!(rc = ivl_abs(&mag, &sig)) && !(rc = dyad_copy(&t->abs_lo, &mag.lo))) rc = dyad_copy(&t->abs_hi, &mag.hi);
        ivl_free(&mag);
    }
    /* ln 2, log2(e) = 1 / ln 2, and with exp2 the exact factor sigma log2(e) */
    if (!rc) rc = ln2_bounds(&t->ln2_lo, &t->ln2_hi, p);
    dyad x, y;
    dyad_init(&x);
    dyad_init(&y);
    if (!rc && !(rc = dyad_set_si(&x, 1, 0)) && !(rc = dyad_div(&t->log2e_lo, &x, &t->ln2_hi, p, 0))) rc = dyad_div(&t->log2e_hi, &x, &t->ln2_lo, p, 1);
    if (!rc && (t->methods & ATT_T_EXP2)) {
        ivl sig, l2e;
        ivl_init(&sig);
        l2e.lo = t->log2e_lo; /* read-only views */
        l2e.hi = t->log2e_hi;
        if (!(rc = dyad_copy(&sig.lo, &t->sig_lo)) && !(rc = dyad_copy(&sig.hi, &t->sig_hi)) && !(rc = ivl_mul(&sig, &sig, &l2e, p)) &&
            !(rc = dyad_copy(&t->sig_lo, &sig.lo)))
            rc = dyad_copy(&t->sig_hi, &sig.hi);
        ivl_free(&sig);
    }
    /* B = max(factor_hi - sigma, sigma - factor_lo) */
    if (!rc && !(rc = sub_dir(&x, &t->c[ATT_C_FACTOR_HI], &t->sig_hi, p, 0))) rc = sub_dir(&y, &t->sig_lo, &t->c[ATT_C_FACTOR_LO], p, 0);
    if (!rc) rc = dyad_copy(&t->B_lo, dmax(&x, &y));
    if (!rc && !(rc = sub_dir(&x, &t->c[ATT_C_FACTOR_HI], &t->sig_lo, p, 1))) rc = sub_dir(&y, &t->sig_hi, &t->c[ATT_C_FACTOR_LO], p, 1);
    if (!rc) rc = dyad_copy(&t->B_hi, dmax(&x, &y));
    dyad_free(&x);
    dyad_free(&y);
    if (!rc) rc = sum_bounds_init(&t->scores, scores, t->flush, a->E, a->E > 0 ? a->E - 1 : 0, 0, w);
    return rc;
}

/* One score's accumulation error under the standard: *none when it has no finite bound. */
static int score_error(dyad *err, int *none, const att_tol *t, product_sums *ps) {
    dyad lo, hi, s;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&s);
    int flags = 0, checked = 1, rc;
    if (t->normal)
        rc = sum_bounds_normal(&lo, &hi, &checked, &flags, &t->scores, &ps->s, &ps->mass, &ps->N, &ps->moved, ps->e);
    else
        rc = sum_bounds_any_order(&lo, &hi, &flags, &t->scores, &ps->s, &ps->mass, &ps->moved, ps->e);
    *none = !checked ? flags != 0 : 0;
    if (!t->normal) *none = flags != 0;
    if (!rc && !*none && !(rc = dyad_from_big(&s, &ps->s, ps->e))) {
        rc = sub_dir(err, &hi, &s, 1 << 20, 1); /* exact: hi - s */
    }
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&s);
    return rc;
}

#define C(name) (&t->c[ATT_C_##name])
#define HAS(name) (t->has[ATT_C_##name])

/* a dyad from n words of magnitude at exponent e */
int dyad_words(dyad *r, const uint64_t *w, size_t n, int64_t e) {
    while (n && !w[n - 1]) n--;
    int rc = big_reserve(&r->m, n ? n : 1);
    if (rc) return rc;
    memcpy(r->m.w, w, n * sizeof *w);
    r->m.n = n;
    r->m.neg = 0;
    r->e = n ? e : 0;
    return 0;
}

static void acc_word(uint64_t *acc, size_t n, uint64_t x, uint64_t y) {
    uint64_t hi, lo = word_mul(&hi, x, y), carry;
    acc[0] += lo;
    carry = acc[0] < lo;
    uint64_t s = acc[1] + hi;
    uint64_t c2 = s < hi;
    s += carry;
    c2 += s < carry;
    acc[1] = s;
    for (size_t i = 2; c2 && i < n; i++) {
        acc[i] += c2;
        c2 = acc[i] < c2;
    }
}

#define ATT_PW 4 /* words of a column's integer products */

/* the 128-bit little-endian integer at x (two's complement with is_signed) */
static w128 read128(const unsigned char *x) {
    uint64_t lo = 0, hi = 0;
    for (int b = 7; b >= 0; b--) {
        lo = lo << 8 | x[b];
        hi = hi << 8 | x[8 + b];
    }
    return w128_make(lo, hi);
}

/* a < b for two's-complement 128-bit integers */
static int s128_lt(w128 a, w128 b) {
    w128 bias = w128_make(0, (uint64_t)1 << 63);
    return w128_less(w128_add(a, bias), w128_add(b, bias));
}

/* acc (nw words) += a b for a word a and a nonnegative 128-bit b */
static void acc_wide(uint64_t *acc, size_t nw, uint64_t a, w128 b) {
    uint64_t h0, l0 = word_mul(&h0, a, w128_lo(b)), h1, l1 = word_mul(&h1, a, w128_hi(b));
    uint64_t x[3] = {l0, h0 + l1, h1 + (h0 + l1 < h0)}, carry = 0;
    for (size_t i = 0; i < nw; i++) {
        uint64_t y = i < 3 ? x[i] : 0, s = acc[i] + y, c = s < y;
        s += carry;
        c += s < carry;
        acc[i] = s;
        carry = c;
        if (i >= 3 && !carry) break;
    }
}

/* a dyad from a signed 128-bit integer at exponent e */
static int dyad_s128(dyad *r, w128 x, int64_t e) {
    int neg = (int)(w128_hi(x) >> 63);
    if (neg) x = w128_sub(w128_zero(), x);
    uint64_t m[2] = {w128_lo(x), w128_hi(x)};
    int rc = dyad_words(r, m, 2, e);
    if (!rc && r->m.n) r->m.neg = neg;
    return rc;
}

/* an earlier result's float64 radius at x as a dyad; *finite 0 for inf or NaN */
int radius_f64(dyad *r, int *finite, const floatfmt *f64, const unsigned char *x) {
    bigint bits, sig;
    big_init(&bits);
    big_init(&sig);
    int kind = VALUE_NAN, neg;
    int64_t e;
    int rc = bytes_load(&bits, x, 8, 0);
    if (!rc) rc = floatfmt_decode(&kind, &neg, &sig, &e, f64, &bits);
    *finite = kind == VALUE_FINITE;
    if (!rc && *finite && !(rc = big_copy(&r->m, &sig))) r->e = e;
    big_free(&bits);
    big_free(&sig);
    return rc;
}

/* The exponent 4096 bits below the largest magnitude of the n intervals' ends (INT64_MIN where every end is 0): the
   exact sums take ends below it as that power of 2 (tiny_widened), so their integers stay a few thousand bits wide
   (an end may be exp(-2**40)), while what that adds is below every format's least value. */
#define ATT_TINY 4096
static int64_t tiny_floor(const ivl *x, size_t n) {
    int64_t top = INT64_MIN;
    for (size_t j = 0; j < n; j++) {
        if (dyad_sign(&x[j].lo) && top_bit(&x[j].lo) > top) top = top_bit(&x[j].lo);
        if (dyad_sign(&x[j].hi) && top_bit(&x[j].hi) > top) top = top_bit(&x[j].hi);
    }
    return top == INT64_MIN ? top : top - ATT_TINY;
}

/* x with each nonzero end below 2**f moved outward: an upper end to 2**f (0 where negative), a lower end to -2**f
   (0 where positive) */
static int tiny_widened(ivl *r, const ivl *x, int64_t f) {
    int rc = ivl_copy(r, x);
    if (!rc && dyad_sign(&r->lo) && top_bit(&r->lo) < f) rc = dyad_sign(&r->lo) > 0 ? dyad_set_si(&r->lo, 0, 0) : dyad_set_si(&r->lo, -1, f);
    if (!rc && dyad_sign(&r->hi) && top_bit(&r->hi) < f) rc = dyad_sign(&r->hi) < 0 ? dyad_set_si(&r->hi, 0, 0) : dyad_set_si(&r->hi, 1, f);
    return rc;
}

/* The value sums for V with flagged or two-word slots (values past one word's range, or not finite),
   exactly (the weights' ends as tiny_widened leaves them): per column lo, hi and mass of the weights' ends wt times the values (with flush, the hull 0 to
   w v for a value below 2**emin), and the kept keys' least and largest value; special per column where a
   kept key's value is not finite. */
static int general_values(dyad *clo, dyad *chi, dyad *cmass, dyad *vlo, dyad *vhi, unsigned char *special, const att_call *a, const size_t *keys,
                          size_t n, size_t vr0, const ivl *wt, int products, int flush, int64_t emin) {
    size_t Ev = a->Ev;
    dyad x, t, zero_d, size;
    dyad_init(&x);
    dyad_init(&t);
    dyad_init(&zero_d);
    dyad_init(&size);
    ivl wj;
    ivl_init(&wj);
    int64_t f = tiny_floor(wt, n);
    int rc = 0;
    for (size_t c = 0; !rc && c < Ev; c++) {
        dyad *zero[] = {&clo[c], &chi[c], &cmass[c], &vlo[c], &vhi[c], &zero_d};
        for (size_t z = 0; !rc && z < 6; z++) rc = dyad_set_si(zero[z], 0, 0);
        int first = 1;
        for (size_t j = 0; !rc && j < n; j++) {
            int kind, neg;
            uint64_t sig;
            int64_t e;
            element_decode_slot(&kind, &neg, &sig, &e, a->v, (vr0 + keys[j]) * Ev + c);
            if (kind != VALUE_FINITE) {
                special[c] = 1;
                continue;
            }
            if ((rc = big_set_word(&x.m, sig, neg && sig))) break;
            x.e = sig ? e : 0;
            if (first || dyad_cmp(&x, &vlo[c]) < 0) rc = dyad_copy(&vlo[c], &x);
            if (!rc && (first || dyad_cmp(&x, &vhi[c]) > 0)) rc = dyad_copy(&vhi[c], &x);
            first = 0;
            if (rc || !products || !sig) continue;
            /* the weight's ends times v, exactly; a value below 2**emin may be read as zero (its product between 0 and w v) */
            if ((rc = tiny_widened(&wj, &wt[j], f))) break;
            int flushed = flush && e + (int64_t)word_bits(sig) <= emin;
            const dyad *below = dyad_sign(&wj.lo) < 0 ? &wj.lo : &zero_d, *above = dyad_sign(&wj.hi) > 0 ? &wj.hi : &zero_d;
            const dyad *fl = flushed ? (neg ? above : below) : (neg ? &wj.hi : &wj.lo), *fh = flushed ? (neg ? below : above) : (neg ? &wj.lo : &wj.hi);
            if ((rc = ends_magnitude(&size, &wj.lo, &wj.hi))) break;
            const dyad *f3[3] = {fl, fh, &size};
            dyad *sums[3] = {&clo[c], &chi[c], &cmass[c]};
            for (int z = 0; !rc && z < 3; z++) {
                if (z == 2) x.m.neg = 0; /* the mass: sizes times |v| */
                if (!(rc = dyad_mul(&t, f3[z], &x, INT64_C(1) << 20, 1))) rc = dyad_add_exact(sums[z], sums[z], &t);
            }
        }
    }
    dyad_free(&x);
    dyad_free(&t);
    dyad_free(&zero_d);
    dyad_free(&size);
    ivl_free(&wj);
    return rc;
}

/* Whether a mask element of the row, or a query or key element that reaches a key the row does not mask, has no bound
   as an earlier result: then neither have the row's outputs nor its logsumexp (proofs/scaled_dot_product_attention.md,
   Results as inputs). *nan is set where such a query or key element reaches only masked keys, whose logits it may make
   NaN (Masked keys). att_row_logits has given w's kinds. */
int row_unbounded(const att_call *a, const floatfmt *f64, const att_row *w, size_t qr, size_t kr0, size_t moff, size_t mstep, int *nan) {
    size_t S = a->S;
    for (size_t j = 0; a->rk && j < S; j++)
        if (a->rk->special_rows[kr0 + j]) {
            if (w->kinds[j] != ATT_KEY_MASKED) return 1;
            *nan = 1;
        }
    if (a->rq && a->rq->special_rows[qr]) {
        for (size_t j = 0; j < S; j++)
            if (w->kinds[j] != ATT_KEY_MASKED) return 1;
        *nan = 1;
    }
    int unbounded = 0;
    dyad r;
    dyad_init(&r);
    for (size_t j = 0; a->mrad && !unbounded && j < a->S; j++) {
        int finite;
        if (radius_f64(&r, &finite, f64, a->mrad + 8 * (moff + j * mstep)) || !finite) unbounded = 1;
    }
    dyad_free(&r);
    return unbounded;
}

/* Whether key j's value row may hold an entry that is not finite: one that is not, or with V's radii one with no bound */
static int value_special(const att_call *a, size_t vr0, size_t j) {
    if (a->v->special_rows[vr0 + j]) return 1;
    for (size_t c = 0; a->vunb && c < a->Ev; c++)
        if (a->vunb[(vr0 + j) * a->Ev + c]) return 1;
    return 0;
}

/* col[c] where key j's value in column c may not be finite (value_special): a weight of exactly 0 times it is NaN */
static void value_nan(unsigned char *col, const att_call *a, size_t vr0, size_t j) {
    size_t Ev = a->Ev;
    const operand *v = a->v;
    for (size_t c = 0; a->vunb && c < Ev; c++) col[c] |= a->vunb[(vr0 + j) * Ev + c];
    for (size_t c = 0; v->special_rows[vr0 + j] && c < Ev; c++) {
        int kind, neg;
        uint64_t sig;
        int64_t e;
        element_decode_slot(&kind, &neg, &sig, &e, v, (vr0 + j) * Ev + c);
        if (kind != VALUE_FINITE) col[c] = 1;
    }
}

/* The NaN that an implementation computing the terms of the keys a row does not keep may reach
   (proofs/scaled_dot_product_attention.md, Masked keys), beyond the overflow of masked keys' finite scores, which the
   bounds check: *row where a masked key's score is not finite (its logit, and so the row, may be NaN), col[c] where a
   key the row does not keep has a value in column c that may not be finite (0 times it is NaN, value_nan; col
   NULL: the row's only). The kept keys whose weight may be exactly 0 add their columns once their weights are bounded.
   att_row_logits has given w's kinds. */
void att_masked_nan(int *row, unsigned char *col, const att_row *w, const att_call *a, size_t vr0) {
    for (size_t j = 0; j < a->S; j++) {
        if (w->kinds[j] == ATT_KEY_KEPT) continue;
        if (w->kinds[j] == ATT_KEY_MASKED && !w->scored[j]) *row = 1;
        if (col) value_nan(col, a, vr0, j);
    }
}

/* A float mask's value b in the logits' units (times log2(e) with exp2) into out, and beta, the bound on
   its computed value's error from the bias and exp2_bias roundings; *vanish where a rounding may overflow toward
   -inf, *fail toward +inf. */
static int mask_term(ivl *out, dyad *beta, int *vanish, int *fail, const att_tol *t, const dyad *b, const dyad *rho, int64_t p) {
    dyad size, err, held, low, least;
    dyad_init(&size);
    dyad_init(&err);
    dyad_init(&held);
    dyad_init(&low);
    dyad_init(&least);
    int rc = dyad_abs(&size, b);
    /* an earlier result's value lies within rho of b: the magnitudes |b| + rho at most and |b| - rho at least */
    if (!rc) rc = rho ? sub_dir(&least, &size, rho, p, 0) : dyad_copy(&least, &size);
    if (!rc && rho) rc = dyad_add(&size, &size, rho, p, 1);
    int negative = dyad_sign(b) < 0 && (!rho || dyad_sign(&least) > 0);
    /* with flush, a value below 2**emin may be read as zero: its error is at most its magnitude */
    int flushable = t->flush && dyad_sign(b) && top_bit(b) < t->emin;
    if (!rc) rc = flushable ? dyad_copy(beta, &size) : rho ? dyad_copy(beta, rho) : dyad_set_si(beta, 0, 0);
    if (!rc && !(rc = dyad_set_si(&low, 0, 0)) && t->rnd[ATT_R_BIAS].active) {
        rc = boundary_error(&err, &t->rnd[ATT_R_BIAS], &size, flushable ? &low : &least);
        if (!rc && HAS(THETA_BIAS) && dyad_cmp(&size, C(THETA_BIAS)) >= 0) {
            if (!negative) *fail = 1;
            *vanish = 1;
        }
        if (!rc) rc = dyad_add(beta, beta, &err, p, 1);
    }
    if (!rc && !*fail && (t->methods & ATT_T_EXP2)) {
        ivl l2e, pt;
        l2e.lo = t->log2e_lo; /* read-only views */
        l2e.hi = t->log2e_hi;
        ivl_init(&pt);
        if (!(rc = dyad_add(&held, &size, beta, p, 1)) && !(rc = dyad_mul(&held, &held, &t->log2e_hi, p, 1)) && !(rc = sub_dir(&low, &size, beta, p, 0)) &&
            !(rc = dyad_mul(&low, &low, &t->log2e_lo, p, 0)))
            rc = boundary_error(&err, &t->rnd[ATT_R_EXP2_BIAS], &held, &low);
        if (!rc && HAS(THETA_EXP2_BIAS) && dyad_cmp(&held, C(THETA_EXP2_BIAS)) >= 0) {
            if (!negative) *fail = 1;
            *vanish = 1;
        }
        if (!rc && !(rc = dyad_mul(beta, beta, &t->log2e_hi, p, 1))) rc = dyad_add(beta, beta, &err, p, 1);
        if (!rc && !(rc = ivl_point(&pt, b))) rc = ivl_mul(out, &pt, &l2e, p);
        ivl_free(&pt);
    } else if (!rc) {
        rc = ivl_point(out, b);
    }
    dyad_free(&size);
    dyad_free(&err);
    dyad_free(&held);
    dyad_free(&low);
    dyad_free(&least);
    return rc;
}

/* The closed form's column sums from exact products (its second pass, and its only one for V with flagged or
   two-word slots), at XC_* times nx: per column S = sum P.hi |v|, U = sum E.hi |v|, V = sum |v| and M = the
   largest |v|; with flush F, UF and VF the same over the values below 2**emin; with the normal standard N = the
   largest node sum of index order and lanes 2 and 4 over P.hi v (every node exact, each lane's first term a leaf; the
   lanes by key index, a masked key's exact zero left out, Lemma L) plus (n - 1) rad, rad = sum (P.hi - max(P.lo, 0)) |v|. A value that is not finite counts as 0 (its column's
   reference is not finite). With V's radii r, S, U, V and M take |v| + r, rad gains (2 P.hi - max(P.lo, 0)) r,
   and D = sum P.hi r (the reference on the stored values is within D of the reference). P and E as tiny_widened
   leaves them. */
enum { XC_U, XC_S, XC_F, XC_UF, XC_N, XC_V, XC_VF, XC_M, XC_D, XC_SUMS };

static int exact_columns(dyad *x, size_t nx, const att_call *a, const ivl *P, const ivl *E, const size_t *keys, size_t n, size_t vr0, int flush,
                         int64_t emin, int normal) {
    size_t Ev = a->Ev;
    dyad vd, mag, t, m, rad, s, n1, n2, n4, l2[2], l4[4], c2, c4, plo, r;
    dyad *ds[] = {&vd, &mag, &t, &m, &rad, &s, &n1, &n2, &n4, &l2[0], &l2[1], &l4[0], &l4[1], &l4[2], &l4[3], &c2, &c4, &plo, &r};
    const size_t nds = sizeof ds / sizeof *ds;
    for (size_t z = 0; z < nds; z++) dyad_init(ds[z]);
    ivl Pw, Ew;
    ivl_init(&Pw);
    ivl_init(&Ew);
    int64_t fP = tiny_floor(P, n), fE = tiny_floor(E, n);
    int rc = 0;
#define XC(name) (&x[XC_##name * nx + c])
    for (size_t c = 0; !rc && c < Ev; c++) {
        for (size_t z = 0; !rc && z < nds; z++) rc = dyad_set_si(ds[z], 0, 0);
        for (size_t z = 0; !rc && z < XC_SUMS; z++) rc = dyad_set_si(&x[z * nx + c], 0, 0);
        unsigned has2 = 0, has4 = 0; /* the lanes with terms so far */
        for (size_t j = 0; !rc && j < n; j++) {
            size_t at = (vr0 + keys[j]) * Ev + c;
            int kind, neg;
            uint64_t sig;
            int64_t e;
            element_decode_slot(&kind, &neg, &sig, &e, a->v, at);
            if (kind != VALUE_FINITE) sig = 0;
            if ((rc = tiny_widened(&Pw, &P[j], fP)) || (rc = tiny_widened(&Ew, &E[j], fE)) || (rc = big_set_word(&vd.m, sig, neg && sig))) break;
            vd.e = sig ? e : 0;
            if ((rc = dyad_abs(&mag, &vd))) break;
            if (sig) { /* a zero term: the lanes' running sums still count their nodes */
                const dyad *f2[2] = {&Pw.hi, &Ew.hi};
                dyad *to[2] = {XC(S), XC(U)}, *fl[2] = {XC(F), XC(UF)};
                int flushed = flush && e + (int64_t)word_bits(sig) <= emin;
                for (int z = 0; !rc && z < 2; z++)
                    if (!(rc = dyad_mul(&m, f2[z], &mag, 1 << 20, 1)) && !(rc = dyad_add_exact(to[z], to[z], &m)) && flushed) rc = dyad_add_exact(fl[z], fl[z], &m);
                if (!rc) rc = dyad_add_exact(XC(V), XC(V), &mag);
                if (!rc && flushed) rc = dyad_add_exact(XC(VF), XC(VF), &mag);
            }
            if (!rc && a->vv) { /* the value's radius (at 2**ve) */
                if (!(rc = dyad_s128(&r, read128(a->vr + 16 * at), a->ve)) && dyad_sign(&r) &&
                    !(rc = dyad_mul(&m, &Pw.hi, &r, 1 << 20, 1)) && !(rc = dyad_add_exact(XC(S), XC(S), &m)) && !(rc = dyad_add_exact(XC(D), XC(D), &m)) &&
                    !(rc = dyad_mul(&m, &Ew.hi, &r, 1 << 20, 1)) && !(rc = dyad_add_exact(XC(U), XC(U), &m)))
                    rc = dyad_add_exact(XC(V), XC(V), &r);
                if (!rc) rc = dyad_add_exact(&t, &mag, &r);
            } else if (!rc) {
                rc = dyad_copy(&t, &mag);
            }
            if (!rc && dyad_cmp(&t, XC(M)) > 0) rc = dyad_copy(XC(M), &t);
            if (rc) break;
            if (!normal) continue;
            /* rad (with radii plus (2 P.hi - max(P.lo, 0)) r), and the term P.hi v into index order and its lanes of two and four */
            if ((rc = dyad_sign(&Pw.lo) > 0 ? dyad_copy(&plo, &Pw.lo) : dyad_set_si(&plo, 0, 0)) || (rc = dyad_neg(&plo, &plo)) ||
                (rc = dyad_add_exact(&plo, &Pw.hi, &plo)) || (sig && ((rc = dyad_mul(&m, &plo, &mag, 1 << 20, 1)) || (rc = dyad_add_exact(&rad, &rad, &m)))) ||
                (a->vv && dyad_sign(&r) &&
                 ((rc = dyad_add_exact(&m, &plo, &Pw.hi)) || (rc = dyad_mul(&m, &m, &r, 1 << 20, 1)) || (rc = dyad_add_exact(&rad, &rad, &m)))) ||
                (rc = dyad_mul(&t, &Pw.hi, &vd, 1 << 20, 1)))
                break;
            /* the lanes by the key's index; each order's or lane's first term is a leaf */
            size_t k = keys[j];
            dyad *runs[3] = {&s, &l2[k & 1], &l4[k & 3]}, *sums[3] = {&n1, &n2, &n4};
            int leaf[3] = {!j, !(has2 >> (k & 1) & 1), !(has4 >> (k & 3) & 1)};
            for (int z = 0; !rc && z < 3; z++)
                if (!(rc = dyad_add_exact(runs[z], runs[z], &t)) && !leaf[z] && !(rc = dyad_abs(&m, runs[z]))) rc = dyad_add_exact(sums[z], sums[z], &m);
            has2 |= 1u << (k & 1);
            has4 |= 1u << (k & 3);
        }
        if (!rc && normal) {
            if (!(rc = lane_nodes_dyad(&c2, l2, has2, 2)) && !(rc = lane_nodes_dyad(&c4, l4, has4, 4)) &&
                !(rc = dyad_add_exact(&n2, &n2, &c2)) && !(rc = dyad_add_exact(&n4, &n4, &c4)) && !(rc = dyad_set_si(&m, (int64_t)n - 1, 0)) &&
                !(rc = dyad_mul(&m, &m, &rad, 1 << 20, 1)))
                rc = dyad_add_exact(XC(N), dmax(dmax(&n1, &n2), &n4), &m);
        }
    }
#undef XC
    for (size_t z = 0; z < nds; z++) dyad_free(ds[z]);
    ivl_free(&Pw);
    ivl_free(&Ew);
    return rc;
}

void att_soft_init(att_soft *s) {
    memset(s, 0, sizeof *s);
    ivl *iv[] = {&s->m, &s->lam_up, &s->lam_lo, &s->kup, &s->klo, &s->W, &s->D, &s->tau};
    for (size_t z = 0; z < sizeof iv / sizeof *iv; z++) ivl_init(iv[z]);
    dyad *dv[] = {&s->g, &s->Glo, &s->Ghi, &s->EZ, &s->dr};
    for (size_t z = 0; z < sizeof dv / sizeof *dv; z++) dyad_init(dv[z]);
}

void att_soft_free(att_soft *s) {
    for (size_t j = 0; j < s->n; j++) {
        ivl_free(&s->xs[j]);
        ivl_free(&s->Ls[j]);
        ivl_free(&s->Rs[j]);
        ivl_free(&s->Ps[j]);
        ivl_free(&s->PTs[j]);
        if (s->PVs) ivl_free(&s->PVs[j]);
        if (s->Hs) {
            ivl_free(&s->Hs[j]);
            dyad_free(&s->pe_hi[j]);
            dyad_free(&s->pe_lo[j]);
            dyad_free(&s->wh[j]);
        }
    }
    free(s->xs);
    free(s->Ls);
    free(s->Rs);
    free(s->Ps);
    free(s->PTs);
    free(s->PVs);
    free(s->Hs);
    free(s->pe_hi);
    free(s->pe_lo);
    free(s->wh);
    free(s->vanish);
    ivl *iv[] = {&s->m, &s->lam_up, &s->lam_lo, &s->kup, &s->klo, &s->W, &s->D, &s->tau};
    for (size_t z = 0; z < sizeof iv / sizeof *iv; z++) ivl_free(iv[z]);
    dyad *dv[] = {&s->g, &s->Glo, &s->Ghi, &s->EZ, &s->dr};
    for (size_t z = 0; z < sizeof dv / sizeof *dv; z++) dyad_free(dv[z]);
    memset(s, 0, sizeof *s);
}

/* Condition 4 (The closed form) for masked keys whose scores have magnitude at most s_abs and accumulation errors at most
   acc: *nan where a score's rounding or its product with the scale may overflow (its logit, before the -inf bias, may then
   be NaN: Masked keys) */
static int masked_overflow(int *nan, const att_tol *t, const dyad *s_abs, const dyad *acc, int64_t p) {
    dyad d1, d2, d3, err, A;
    dyad *ds[] = {&d1, &d2, &d3, &err, &A};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int rc = 0;
    if (!(rc = dyad_add(&d1, s_abs, acc, p, 1)) && !(rc = sub_dir(&d3, s_abs, acc, p, 0)) && !(rc = boundary_error(&err, &t->rnd[ATT_R_SCORES], &d1, &d3)) &&
        !(rc = dyad_add(&A, acc, &err, p, 1)) && !(rc = dyad_add(&d1, s_abs, &A, p, 1))) {
        *nan = dyad_cmp(&d1, C(MAX_SCORES)) > 0;
        if (!*nan && !(rc = dyad_mul(&d1, &d1, C(FAC), p, 1)) && !(rc = dyad_set_si(&d2, 1, 0)) && !(rc = dyad_add(&d2, &d2, C(R_L), p, 1)) &&
            !(rc = dyad_mul(&d1, &d1, &d2, p, 1)) && !(rc = dyad_add(&d1, &d1, C(A_L), p, 1)))
            *nan = dyad_cmp(&d1, C(MAX_LOGITS)) > 0;
    }
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* The closed form's steps 1 to 8 for one row (The closed form): its quantities into sf (att_soft), *ok 1 where every
   validity condition of those steps holds and some key cannot vanish, sf->masked_nan where condition 4 fails (the row
   may be NaN: Masked keys). att_row_logits has given the row's n >= 1 kept
   keys. probs: the backward's probability step too (scaled_dot_product_attention_backward.md, The probability step in
   closed form): per key H, the upper end of P exp(R), the lower end of P exp(-R) and w'_j's largest value. logits_only:
   steps 1 to 3 only (the logits x, their bounds L and the keys that may vanish), for a given logsumexp. mnan (NULL: none):
   per key, 1 at a masked key with a finite score that fails condition 4 on its own (its logit may be NaN). */
int att_soft_row(att_soft *sf, int *ok, att_row *w, const att_call *a, const att_tol *t, size_t n, size_t qr, size_t kr0, size_t moff,
                 size_t mstep, int probs, int logits_only, unsigned char *mnan) {
    const operand *q = a->q, *k = a->k;
    size_t S = a->S;
    int64_t p = t->p;
    size_t *keys = w->keys;
    w128 *scores = w->scores;
    *ok = 0;
    ivl x, L, R, P, T, H, PT, tau, top, low, m, iv1, iv2, sig_abs, lam_up, lam_lo, kup, klo, W, D, one_iv;
    ivl *keyed[] = {&x, &L, &R, &P, &T, &H, &PT, &tau, &top, &low, &m, &iv1, &iv2, &sig_abs, &lam_up, &lam_lo, &kup, &klo, &W, &D, &one_iv};
    for (size_t z = 0; z < sizeof keyed / sizeof *keyed; z++) ivl_init(keyed[z]);
    dyad s, sabs, acc, A, Cc, err, d1, d2, d3, best_s, best_acc, beta, g, Glo, Ghi, EZ, dr, nd, n1d, big_r, floor_arg, upm1_lo, upm1_hi;
    dyad *ds[] = {&s,  &sabs, &acc, &A,  &Cc, &err, &d1, &d2,  &d3,  &best_s,    &best_acc, &beta,
                  &g,  &Glo,  &Ghi, &EZ, &dr, &nd,  &n1d, &big_r, &floor_arg, &upm1_lo, &upm1_hi};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    product_sums ps, dev_ps;
    product_sums_init(&ps);
    product_sums_init(&dev_ps);
    dyad *sd = NULL; /* without tables, the kept keys' scores */
    int fail = 0, rc = 0;
    if (!(sf->xs = malloc(n * sizeof *sf->xs)) || !(sf->Ls = malloc(n * sizeof *sf->Ls)) || !(sf->Rs = malloc(n * sizeof *sf->Rs)) ||
        !(sf->Ps = malloc(n * sizeof *sf->Ps)) || !(sf->PTs = malloc(n * sizeof *sf->PTs)) || !(sf->vanish = calloc(n, 1)) ||
        ((t->methods & ATT_T_WEIGHTS) && !(sf->PVs = malloc(n * sizeof *sf->PVs))) ||
        (probs && (!(sf->Hs = malloc(n * sizeof *sf->Hs)) || !(sf->pe_hi = malloc(n * sizeof *sf->pe_hi)) ||
                   !(sf->pe_lo = malloc(n * sizeof *sf->pe_lo)) || !(sf->wh = malloc(n * sizeof *sf->wh)))) ||
        ((!a->tables || a->mask_float) && !(sd = malloc(n * sizeof *sd)))) {
        rc = EXACTREF_NOMEM;
        goto out;
    }
    for (size_t j = 0; sd && j < n; j++) dyad_init(&sd[j]);
    for (size_t j = 0; j < n; j++) {
        ivl_init(&sf->xs[j]);
        ivl_init(&sf->Ls[j]);
        ivl_init(&sf->Rs[j]);
        ivl_init(&sf->Ps[j]);
        ivl_init(&sf->PTs[j]);
        if (sf->PVs) ivl_init(&sf->PVs[j]);
        if (probs) {
            ivl_init(&sf->Hs[j]);
            dyad_init(&sf->pe_hi[j]);
            dyad_init(&sf->pe_lo[j]);
            dyad_init(&sf->wh[j]);
        }
    }
    sf->n = n;
    ivl *xs = sf->xs, *Ls = sf->Ls, *Rs = sf->Rs, *Ps = sf->Ps, *PTs = sf->PTs, *PVs = sf->PVs;
    unsigned char *vanish = sf->vanish;
    {
        ivl sig;
        sig.lo = t->sig_lo; /* read-only views */
        sig.hi = t->sig_hi;
        if ((rc = ivl_abs(&sig_abs, &sig))) goto out; /* |sigma|'s ends, for either sign */
    }
    int what = t->normal ? PRODUCT_NODES : PRODUCT_MASS, any_masked = 0, masked_none = 0;
    size_t kept = 0;
    for (size_t j = 0; !rc && !fail && j < S; j++) {
        /* a key the row does not keep: only a masked one with a finite score, for condition 4, until one fails it (a
           masked key's score that is not finite is att_masked_nan's) */
        if (w->kinds[j] != ATT_KEY_KEPT && (w->kinds[j] != ATT_KEY_MASKED || !w->scored[j] || (masked_none && !mnan))) continue;
        if ((rc = product_element(&ps, q, qr, k, kr0 + j, what, t->flush, t->emin, NULL))) break;
        if (a->rq || a->rk) { /* the radii's deviation, in the mass, the moved part and N (proofs/matmul.md, Results as inputs) */
            int unb = 0;
            if ((rc = product_deviation(&ps, &unb, &dev_ps, q, a->rq, qr, k, a->rk, kr0 + j, 0))) break;
        }
        int none;
        if ((rc = score_error(&acc, &none, t, &ps))) break;
        /* the score at the unit 2**u, exactly (for the tables) */
        bigint sh;
        big_init(&sh);
        if (a->tables && !(rc = big_shr(&sh, &ps.s, a->u - ps.e, 0))) {
            uint64_t lo = sh.n > 0 ? sh.w[0] : 0, hi = sh.n > 1 ? sh.w[1] : 0;
            w128 v128 = w128_make(lo, hi);
            if (kept < n && keys[kept] == j) scores[kept] = sh.neg ? w128_sub(w128_zero(), v128) : v128;
        }
        big_free(&sh);
        if (rc) break;
        if ((rc = dyad_from_big(&s, &ps.s, ps.e))) break;
        if (kept >= n || keys[kept] != j) {
            /* a masked key: its largest |s| and error, for condition 4 */
            if (none) masked_none = 1;
            if ((rc = dyad_copy(&sabs, &s))) break;
            sabs.m.neg = 0;
            if (!any_masked || dyad_cmp(&sabs, &best_s) > 0) rc = dyad_copy(&best_s, &sabs);
            if (!rc && !none && (!any_masked || dyad_cmp(&acc, &best_acc) > 0)) rc = dyad_copy(&best_acc, &acc);
            any_masked = 1;
            int own = none;
            if (!rc && mnan && !none) rc = masked_overflow(&own, t, &sabs, &acc, p);
            if (mnan) mnan[j] = (unsigned char)own;
            continue;
        }
        if (none) {
            fail = 1;
            break;
        }
        size_t kj = kept++;
        if (sd && (rc = dyad_copy(&sd[kj], &s))) break;
        /* steps 1 to 3: A, the logit x = sigma s, C and L */
        if ((rc = dyad_copy(&sabs, &s))) break;
        sabs.m.neg = 0;
        if ((rc = dyad_add(&d1, &sabs, &acc, p, 1)) || (rc = sub_dir(&d3, &sabs, &acc, p, 0)) || (rc = boundary_error(&err, &t->rnd[ATT_R_SCORES], &d1, &d3)) ||
            (rc = dyad_add(&A, &acc, &err, p, 1)) ||
            (rc = dyad_add(&d1, &sabs, &A, p, 1)))
            break;
        if (dyad_cmp(&d1, C(MAX_SCORES)) > 0) {
            fail = 1;
            break;
        }
        if ((rc = ivl_point(&iv1, &s))) break;
        ivl sig;
        sig.lo = t->sig_lo; /* read-only views */
        sig.hi = t->sig_hi;
        if ((rc = ivl_mul(&xs[kj], &iv1, &sig, p))) break;
        /* a float mask's value, rounded as the methods say, and beta, its error */
        int mask_vanish = 0;
        if ((rc = dyad_set_si(&beta, 0, 0))) break;
        int finite = 1, has_rho = 0;
        if (a->mrad && !(rc = radius_f64(&d3, &finite, &t->f64, a->mrad + 8 * (moff + j * mstep)))) has_rho = dyad_sign(&d3) > 0;
        if (rc) break;
        if (a->mask_float && (dyad_sign(&w->bias[kj]) || has_rho)) {
            if ((rc = mask_term(&iv2, &beta, &mask_vanish, &fail, t, &w->bias[kj], has_rho ? &d3 : NULL, p)) || fail) break;
            if ((rc = ivl_add(&xs[kj], &xs[kj], &iv2, p))) break;
        }
        /* C = (|s| + A) B + |sigma| A + beta, both ends */
        for (int up = 0; !rc && up < 2; up++) {
            dyad *end = up ? &Cc : &d2;
            if (!(rc = dyad_add(&d1, &sabs, &A, p, up)) && !(rc = dyad_mul(&d1, &d1, up ? &t->B_hi : &t->B_lo, p, up)) &&
                !(rc = dyad_mul(&d3, up ? &sig_abs.hi : &sig_abs.lo, &A, p, up)) && !(rc = dyad_add(end, &d1, &d3, p, up)))
                rc = dyad_add(end, end, &beta, p, up);
        }
        if (rc) break;
        if (mask_vanish) vanish[kj] = 1;
        /* L = C + rnd(logits, |x| + C) */
        if ((rc = ivl_abs(&iv1, &xs[kj])) || (rc = dyad_add(&d1, &iv1.hi, &Cc, p, 1)) || (rc = sub_dir(&d3, &iv1.lo, &Cc, p, 0)) ||
            (rc = boundary_error(&err, &t->rnd[ATT_R_LOGITS], &d1, &d3)) ||
            (rc = dyad_add(&Ls[kj].hi, &Cc, &err, p, 1)) || (rc = dyad_add(&Ls[kj].lo, &d2, &err, p, 0)))
            break;
        if (HAS(THETA_LOGITS) && dyad_cmp(&d1, C(THETA_LOGITS)) >= 0) {
            /* the value before the rounding may overflow: toward -inf only (x + C < 0) */
            if ((rc = dyad_add(&d3, &xs[kj].hi, &Cc, p, 1))) break;
            if (dyad_sign(&d3) >= 0) {
                fail = 1;
                break;
            }
            vanish[kj] = 1;
        }
    }
    if (rc || fail) goto out;
    {
        int all = 1;
        for (size_t j = 0; j < n; j++) all &= vanish[j];
        if (all) goto out;
    }
    /* condition 4: masked keys whose logit is computed before the -inf bias; where it fails the row keeps its bound and
       may be NaN (Masked keys) */
    sf->masked_nan = masked_none;
    if (any_masked && !masked_none && (rc = masked_overflow(&sf->masked_nan, t, &best_s, &best_acc, p))) goto out;
    if (logits_only) {
        *ok = 1;
        goto out;
    }
    /* step 4: m, the upper end top and the lower end low of the computed maximum */
    for (size_t j = 0; !rc && j < n; j++) {
        ivl xl;
        ivl_init(&xl);
        if (!(rc = dyad_add(&xl.lo, &xs[j].lo, &Ls[j].lo, p, 0))) rc = dyad_add(&xl.hi, &xs[j].hi, &Ls[j].hi, p, 1);
        if (!rc) rc = j ? ivl_max(&m, &m, &xs[j]) : ivl_copy(&m, &xs[j]);
        if (!rc) rc = j ? ivl_max(&top, &top, &xl) : ivl_copy(&top, &xl);
        if (!rc && !vanish[j]) {
            if (!(rc = sub_dir(&xl.lo, &xs[j].lo, &Ls[j].hi, p, 0))) rc = sub_dir(&xl.hi, &xs[j].hi, &Ls[j].lo, p, 1);
            if (!rc) {
                int first = 1;
                for (size_t z = 0; z < j; z++) first &= vanish[z];
                rc = first ? ivl_copy(&low, &xl) : ivl_max(&low, &low, &xl);
            }
        }
        ivl_free(&xl);
    }
    if (rc) goto out;
    /* lam_up = top - m, lam_lo = m - low; k_up = exp(lam_up), k_lo = exp(lam_lo) */
    if ((rc = dyad_set_si(&one_iv.lo, 1, 0)) || (rc = dyad_set_si(&one_iv.hi, 1, 0))) goto out;
    int online = t->methods & ATT_T_ONLINE, exp2 = t->methods & ATT_T_EXP2;
    ivl ln2;
    ln2.lo = t->ln2_lo; /* read-only views */
    ln2.hi = t->ln2_hi;
    if ((rc = ivl_sub(&lam_up, &top, &m, p)) || (rc = ivl_sub(&lam_lo, &m, &low, p))) goto out;
    if (exp2 && ((rc = ivl_mul(&lam_up, &lam_up, &ln2, p)) || (rc = ivl_mul(&lam_lo, &lam_lo, &ln2, p)))) goto out; /* natural units */
    if ((rc = exp_end(&kup.lo, &lam_up.lo, p, 0)) || (rc = exp_end(&kup.hi, &lam_up.hi, p, 1)) || (rc = exp_end(&klo.lo, &lam_lo.lo, p, 0)) ||
        (rc = exp_end(&klo.hi, &lam_lo.hi, p, 1)))
        goto out;
    /* steps 5 and 6: per key R = L + the shift's rounding (at least ln 2 for a key that may vanish, 1 in
       base 2); with online softmax the shifts' r_h |d| + a_h */
    for (size_t j = 0; !rc && j < n; j++) {
        if ((rc = sub_dir(&d1, &top.hi, &xs[j].lo, p, 1)) || (rc = dyad_add(&d1, &d1, &Ls[j].hi, p, 1))) break;
        if (online) {
            if ((rc = dyad_mul(&err, C(R_H), &d1, p, 1)) || (rc = dyad_add(&err, &err, C(A_H), p, 1)) || (rc = dyad_add(&d2, &d1, &err, p, 1))) break;
            if (dyad_cmp(&d2, C(MAX_SHIFT)) > 0) {
                fail = 1;
                break;
            }
        } else {
            if ((rc = sub_dir(&d3, &low.lo, &xs[j].hi, p, 0)) || (rc = sub_dir(&d3, &d3, &Ls[j].hi, p, 0)) ||
                (rc = boundary_error(&err, &t->rnd[ATT_R_SHIFT], &d1, &d3)))
                break;
            if (HAS(THETA_SHIFT) && dyad_cmp(&d1, C(THETA_SHIFT)) >= 0) vanish[j] = 1;
        }
        if ((rc = dyad_add(&Rs[j].lo, &Ls[j].lo, &err, p, 0)) || (rc = dyad_add(&Rs[j].hi, &Ls[j].hi, &err, p, 1))) break;
        if (vanish[j]) {
            const dyad *flo = exp2 ? &one_iv.lo : &t->ln2_lo, *fhi = exp2 ? &one_iv.hi : &t->ln2_hi;
            if (dyad_cmp(&Rs[j].lo, flo) < 0) rc = dyad_copy(&Rs[j].lo, flo);
            if (!rc && dyad_cmp(&Rs[j].hi, fhi) < 0) rc = dyad_copy(&Rs[j].hi, fhi);
        }
        if (!rc && exp2) rc = ivl_mul(&Rs[j], &Rs[j], &ln2, p);
    }
    if (rc || fail) goto out;
    /* the weights w_j = exp(x_j - m) from Lemma D's tables: [p_j, p_j + h_j] 2**-F, and W */
    int tabled = a->tables != NULL;
    if (tabled) {
        w128 topv = scores[0];
        for (size_t j = 1; j < n; j++)
            if (a->sign > 0 ? w128_less(w128_add(topv, w128_make(0, (uint64_t)1 << 63)), w128_add(scores[j], w128_make(0, (uint64_t)1 << 63)))
                            : w128_less(w128_add(scores[j], w128_make(0, (uint64_t)1 << 63)), w128_add(topv, w128_make(0, (uint64_t)1 << 63))))
                topv = scores[j];
        uint64_t Wlo[ATT_ZW] = {0}, Whi[ATT_ZW] = {0};
        for (size_t j = 0; j < n; j++) {
            uint64_t pw[3] = {0, 0, (uint64_t)1 << (ATT_F - 128)};
            int z = 0, capped = a->mask_float && w->capped[j];
            if (capped) {
                pw[2] = 0; /* 0 <= 2**F w' < 1 */
            } else {
                if (a->sign && (rc = att_digits_mul(pw, &z, a->sign > 0 ? w128_sub(topv, scores[j]) : w128_sub(scores[j], topv), a->tables))) break;
                if (a->mask_float && (rc = att_digits_mul(pw, &z, ((w128 *)w->mdist)[j], a->mtables))) break;
            }
            uint64_t h = z ? 3 * (uint64_t)z - 1 : (uint64_t)capped, ph[3] = {pw[0], pw[1], pw[2]};
            /* ph = pw + h */
            uint64_t c = h;
            for (int ii = 0; ii < 3 && c; ii++) {
                ph[ii] += c;
                c = ph[ii] < c;
            }
            if ((rc = dyad_words(&Ps[j].lo, pw, 3, -ATT_F)) || (rc = dyad_words(&Ps[j].hi, ph, 3, -ATT_F))) break;
            /* the sums of the ends into W */
            uint64_t cl = 0, ch = 0;
            for (int ii = 0; ii < ATT_ZW; ii++) {
                uint64_t xl = ii < 3 ? pw[ii] : 0, xh = ii < 3 ? ph[ii] : 0;
                uint64_t sl = Wlo[ii] + xl, sh2 = Whi[ii] + xh;
                uint64_t c1 = sl < xl, c3 = sh2 < xh;
                sl += cl;
                c1 += sl < cl;
                sh2 += ch;
                c3 += sh2 < ch;
                Wlo[ii] = sl;
                Whi[ii] = sh2;
                cl = c1;
                ch = c3;
            }
        }
        if (!rc && !(rc = dyad_words(&W.lo, Wlo, ATT_ZW, -ATT_F))) rc = dyad_words(&W.hi, Whi, ATT_ZW, -ATT_F);
        /* with a float mask the weights are relative to m' >= m, and w = w' / M with M = max w', which may be only a few
           units of 2**-F (the largest score and the largest mask value on keys far apart); each w then carries W's absolute
           width over M. The tables are kept where W's width (the sum of the h_j units) is at most 2**-20 of M's lower end,
           and else Lemma X, below */
        if (!rc && a->mask_float) {
            dyad mlo, wd;
            dyad_init(&mlo);
            dyad_init(&wd);
            rc = dyad_set_si(&mlo, 0, 0);
            for (size_t j = 0; !rc && j < n; j++)
                if (dyad_cmp(&Ps[j].lo, &mlo) > 0) rc = dyad_copy(&mlo, &Ps[j].lo);
            if (!rc && !(rc = dyad_neg(&wd, &W.lo)) && !(rc = dyad_add_exact(&wd, &W.hi, &wd)) && !(rc = dyad_shift(&wd, 20)))
                tabled = dyad_sign(&mlo) > 0 && dyad_cmp(&wd, &mlo) <= 0;
            dyad_free(&mlo);
            dyad_free(&wd);
        }
    }
    if (!rc && !tabled) {
        /* without tables (the scores' span past 127 bits, or a mask's past its tables): Lemma X per weight,
           w'_j = exp(l_j - m'), m' the logit of the largest logit's key (att_argument), whose weight is exactly 1 */
        size_t ktop = 0, ref = 0;
        for (size_t j = 1; j < n; j++)
            if (a->sign > 0 ? dyad_cmp(&sd[j], &sd[ktop]) > 0 : dyad_cmp(&sd[j], &sd[ktop]) < 0) ktop = j;
        const dyad *bias = a->mask_float ? w->bias : NULL;
        rc = att_reference_key(&ref, a, sd, bias, n, ktop);
        if (!rc && !(rc = dyad_set_si(&W.lo, 0, 0))) rc = dyad_set_si(&W.hi, 0, 0);
        for (size_t j = 0; !rc && j < n; j++) {
            if ((rc = att_argument(&d2, &d3, a, sd, bias, ref, j, &t->abs_lo, &t->abs_hi, p))) break;
            if ((rc = dyad_neg(&d2, &d2)) || (rc = dyad_neg(&d3, &d3)) || (rc = exp_clamped(&Ps[j].lo, &d3, p, 0)) ||
                (rc = exp_clamped(&Ps[j].hi, &d2, p, 1)))
                break;
            rc = ivl_add(&W, &W, &Ps[j], p);
        }
    }
    if (!rc && a->mask_float) {
        /* the w' = exp(l - m') for a reference m': w = w' / exp(m - m'), and exp(m - m') = max w' lies in [max of the
           lower ends, max of the upper ends] */
        ivl M;
        ivl_init(&M);
        rc = ivl_copy(&M, &Ps[0]);
        for (size_t j = 1; !rc && j < n; j++) rc = ivl_max(&M, &M, &Ps[j]);
        if (!rc && !(rc = dyad_set_si(&W.lo, 0, 0))) rc = dyad_set_si(&W.hi, 0, 0);
        for (size_t j = 0; !rc && j < n; j++)
            if (!(rc = ivl_div(&Ps[j], &Ps[j], &M, p))) rc = ivl_add(&W, &W, &Ps[j], p);
        ivl_free(&M);
    }
    if (rc) goto out;
    /* P = w / W; T = exp(R) - 1; eps, H = eps k_up / W; PT = P T + H; tau = sum PT */
    if ((rc = dyad_set_si(&tau.lo, 0, 0)) || (rc = dyad_set_si(&tau.hi, 0, 0))) goto out;
    if ((rc = dyad_set_si(&big_r, 16, 0)) || (rc = dyad_set_si(&floor_arg, -1, 20))) goto out;
    if (online && (!(rc = sub_dir(&upm1_lo, C(UP), &one_iv.lo, p, 0)) && !(rc = sub_dir(&upm1_hi, C(UP), &one_iv.lo, p, 1)))) {
        /* online: H = own k_up / W for every key */
        if (!(rc = dyad_mul(&H.lo, C(OWN), &kup.lo, p, 0)) && !(rc = dyad_div(&H.lo, &H.lo, &W.hi, p, 0)) && !(rc = dyad_mul(&H.hi, C(OWN), &kup.hi, p, 1)))
            rc = dyad_div(&H.hi, &H.hi, &W.lo, p, 1);
    }
    for (size_t j = 0; !rc && j < n; j++) {
        ivl wj;
        ivl_init(&wj);
        int big = 0;
        if (!(rc = ivl_copy(&wj, &Ps[j])) && !(rc = ivl_div(&Ps[j], &wj, &W, p)) && dyad_cmp(&Rs[j].hi, &big_r) > 0) {
            /* a large R (the shift of a logit far below the row's maximum): P T <= w exp(R) / W (online: times up)
               and w exp(R) = exp(x - m + R) (x - m times ln 2 with exp2), its argument's upper end at least -2**20 */
            big = 1;
            if (!(rc = sub_dir(&d1, &xs[j].hi, &m.lo, p, 1)) && (!exp2 || !(rc = dyad_mul(&d1, &d1, dyad_sign(&d1) >= 0 ? &t->ln2_hi : &t->ln2_lo, p, 1))) &&
                !(rc = dyad_add(&d1, &d1, &Rs[j].hi, p, 1))) {
                if (dyad_cmp(&d1, &floor_arg) < 0) rc = dyad_copy(&d1, &floor_arg);
                if (!rc) rc = exp_end(&d2, &d1, p, 1);
            }
            /* the exp's upper end, k_lo w exp(R); its lower end w exp(-R) / k_up = exp(x - m - R) / k_up for exp_ulps */
            if (!rc && !online && !(rc = dyad_mul(&d1, &d2, &klo.hi, p, 1)) && t->rnd[ATT_R_EXP].flush) {
                if (!(rc = sub_dir(&d3, &xs[j].lo, &m.hi, p, 0)) &&
                    (!exp2 || !(rc = dyad_mul(&d3, &d3, dyad_sign(&d3) >= 0 ? &t->ln2_lo : &t->ln2_hi, p, 0))) && !(rc = sub_dir(&d3, &d3, &Rs[j].hi, p, 0))) {
                    if (dyad_cmp(&d3, &floor_arg) < 0) rc = dyad_set_si(&d3, 0, 0);
                    else if (!(rc = exp_end(&d3, &d3, p, 0))) rc = dyad_div(&d3, &d3, &kup.hi, p, 0);
                }
            }
        } else if (!rc && !(rc = expm1_end(&T.lo, &Rs[j].lo, p, 0)) && !(rc = expm1_end(&T.hi, &Rs[j].hi, p, 1)) && !online) {
            /* the exp's upper end k_lo w exp(R) <= k_lo w (1 + T.hi), its lower end w exp(-R) / k_up >= w / ((1 + T.hi) k_up) */
            if (!(rc = dyad_add(&d2, &one_iv.hi, &T.hi, p, 1)) && !(rc = dyad_mul(&d1, &d2, &wj.hi, p, 1)) && !(rc = dyad_mul(&d1, &d1, &klo.hi, p, 1)) &&
                t->rnd[ATT_R_EXP].flush && !(rc = dyad_mul(&d3, &d2, &kup.hi, p, 1)))
                rc = dyad_div(&d3, &wj.lo, &d3, p, 0);
        }
        if (!rc && !online) {
            /* eps = rnd(exp, high, low); H = eps k_up / W */
            if (!(rc = boundary_error(&err, &t->rnd[ATT_R_EXP], &d1, t->rnd[ATT_R_EXP].flush ? &d3 : NULL)) && !(rc = dyad_mul(&H.lo, &err, &kup.lo, p, 0)) &&
                !(rc = dyad_div(&H.lo, &H.lo, &W.hi, p, 0)) && !(rc = dyad_mul(&H.hi, &err, &kup.hi, p, 1)))
                rc = dyad_div(&H.hi, &H.hi, &W.lo, p, 1);
            if (!rc && probs) rc = dyad_add(&sf->wh[j], &d1, &err, p, 1); /* w'_j's largest value, k_lo w exp(R) + eps */
        }
        if (!rc && big) {
            /* PT = [H.lo, w exp(R) / W (up) + H.hi] */
            if (!(rc = dyad_div(&d2, &d2, &W.lo, p, 1)) && probs) rc = dyad_copy(&sf->pe_hi[j], &d2); /* exp(x - m + R) / W */
            if (!rc && (!online || !(rc = dyad_mul(&d2, &d2, C(UP), p, 1))) && !(rc = dyad_copy(&PTs[j].lo, &H.lo)))
                rc = dyad_add(&PTs[j].hi, &d2, &H.hi, p, 1);
        } else if (!rc) {
            /* online: T = (exp(R) - 1) up + (up - 1) */
            if (online && !(rc = dyad_mul(&T.lo, &T.lo, C(UP), p, 0)) && !(rc = dyad_add(&T.lo, &T.lo, &upm1_lo, p, 0)) &&
                !(rc = dyad_mul(&T.hi, &T.hi, C(UP), p, 1)))
                rc = dyad_add(&T.hi, &T.hi, &upm1_hi, p, 1);
            if (!rc && !(rc = ivl_mul(&PTs[j], &Ps[j], &T, p))) rc = ivl_add(&PTs[j], &PTs[j], &H, p);
        }
        if (!rc) rc = ivl_add(&tau, &tau, &PTs[j], p);
        if (!rc && PVs) {
            /* weights rounded before the value product (not online): the error weight gains rnd(weights, high + eps) k_up / W */
            int lw = t->rnd[ATT_R_EXP].flush;
            if (lw) rc = sub_dir(&d3, &d3, &err, p, 0);
            if (!rc && !(rc = dyad_add(&d1, &d1, &err, p, 1)) && !(rc = boundary_error(&err, &t->rnd[ATT_R_WEIGHTS], &d1, lw ? &d3 : NULL)) &&
                !(rc = dyad_mul(&d2, &err, &kup.lo, p, 0)) && !(rc = dyad_div(&d2, &d2, &W.hi, p, 0)) && !(rc = dyad_add(&PVs[j].lo, &PTs[j].lo, &d2, p, 0)) &&
                !(rc = dyad_mul(&d2, &err, &kup.hi, p, 1)) && !(rc = dyad_div(&d2, &d2, &W.lo, p, 1)))
                rc = dyad_add(&PVs[j].hi, &PTs[j].hi, &d2, p, 1);
        }
        if (!rc && probs) {
            /* the lower end of P exp(-R): exp(x - m - R) / W for a large R (0 below exp(-2**20)), else P / (1 + T); the upper
               end of P exp(R) for a small R, P (1 + T); and H */
            if (big) {
                if (!(rc = sub_dir(&d3, &xs[j].lo, &m.hi, p, 0)) &&
                    (!exp2 || !(rc = dyad_mul(&d3, &d3, dyad_sign(&d3) >= 0 ? &t->ln2_lo : &t->ln2_hi, p, 0))) && !(rc = sub_dir(&d3, &d3, &Rs[j].hi, p, 0))) {
                    if (dyad_cmp(&d3, &floor_arg) < 0) rc = dyad_set_si(&sf->pe_lo[j], 0, 0);
                    else if (!(rc = exp_end(&sf->pe_lo[j], &d3, p, 0))) rc = dyad_div(&sf->pe_lo[j], &sf->pe_lo[j], &W.hi, p, 0);
                }
            } else if (!(rc = dyad_add(&d3, &one_iv.hi, &T.hi, p, 1)) && !(rc = dyad_mul(&sf->pe_hi[j], &Ps[j].hi, &d3, p, 1))) {
                rc = dyad_div(&sf->pe_lo[j], &Ps[j].lo, &d3, p, 0);
            }
            if (!rc) rc = ivl_copy(&sf->Hs[j], &H);
        }
        ivl_free(&wj);
    }
    if (rc) goto out;
    /* D = 1 - tau > 0; g = k_up / (W D); G_lo = W D / k_up; G_hi = W (1 + tau) k_lo */
    if ((rc = dyad_set_si(&one_iv.lo, 1, 0)) || (rc = dyad_set_si(&one_iv.hi, 1, 0)) || (rc = ivl_sub(&D, &one_iv, &tau, p))) goto out;
    if (dyad_sign(&D.lo) <= 0) goto out;
    if ((rc = dyad_mul(&d1, &W.lo, &D.lo, p, 0)) || (rc = dyad_div(&g, &kup.hi, &d1, p, 1)) || (rc = dyad_div(&Glo, &d1, &kup.hi, p, 0)) ||
        (rc = dyad_add(&d2, &one_iv.hi, &tau.hi, p, 1)) || (rc = dyad_mul(&Ghi, &W.hi, &d2, p, 1)) || (rc = dyad_mul(&Ghi, &Ghi, &klo.hi, p, 1)))
        goto out;
    const dyad *pn = t->per_n + n * ATT_PER_N;
    const unsigned char *pnh = t->per_n_has + n * ATT_PER_N;
    if ((rc = dyad_set_si(&nd, (int64_t)n, 0)) || (rc = dyad_set_si(&n1d, (int64_t)n - 1, 0))) goto out;
    if (online) {
        /* the online sums' Lemma 2 at the online count: n wmax (1 + g_z) + k_z <= lim_z; E_Z = g_z + (k_z + x_z) g */
        if (!HAS(ON_G_Z) || !HAS(ON_G_V)) {
            fail = 1;
        } else if (!(rc = dyad_mul(&d1, &nd, C(WMAX), p, 1)) && !(rc = dyad_add(&d2, &one_iv.hi, C(ON_G_Z), p, 1)) && !(rc = dyad_mul(&d1, &d1, &d2, p, 1)) &&
                   !(rc = dyad_add(&d1, &d1, C(ON_K_Z), p, 1))) {
            if (dyad_cmp(&d1, C(ON_LIM_Z)) > 0)
                fail = 1;
            else if (!(rc = dyad_add(&d1, C(ON_K_Z), C(X_Z), p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1)))
                rc = dyad_add(&EZ, C(ON_G_Z), &d1, p, 1);
        }
    } else if (t->normal) {
        /* step 8, the row sum in index order and lanes: X = sum over k >= 2 of min(C_k, D) */
        ivl run;
        ivl_init(&run);
        dyad X;
        dyad_init(&X);
        rc = dyad_set_si(&X, 0, 0);
        if (!rc && !(rc = dyad_set_si(&run.lo, 0, 0))) rc = dyad_set_si(&run.hi, 0, 0);
        for (size_t j = 0; !rc && j < n; j++) {
            if ((rc = ivl_add(&run, &run, &Ps[j], p)) || (rc = ivl_add(&run, &run, &PTs[j], p))) break;
            if (j) rc = dyad_add(&X, &X, dmin(&run.hi, &D.hi), p, 1);
        }
        /* E_Z = Gz (r_z (X / D + min(n - 1, 3)) + (n - 1) a2_z g) */
        if (!rc && !(rc = dyad_div(&d1, &X, &D.lo, p, 1)) && !(rc = dyad_set_si(&d2, n - 1 < 3 ? (int64_t)n - 1 : 3, 0)) &&
            !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, C(R_Z), p, 1)) && !(rc = dyad_mul(&d2, &n1d, C(A2_Z), p, 1)) &&
            !(rc = dyad_mul(&d2, &d2, &g, p, 1)) && !(rc = dyad_add(&d1, &d1, &d2, p, 1)))
            rc = dyad_mul(&EZ, &pn[ATT_N_GZ], &d1, p, 1);
        if (!rc && HAS(NARROW_Z)) {
            if (!(rc = dyad_mul(&d1, &nd, C(A_Z), p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1)) && !(rc = dyad_add(&d1, &d1, C(R_Z), p, 1)) &&
                !(rc = dyad_mul(&d1, &d1, &pn[ATT_N_GZ], p, 1)))
                rc = dyad_add(&EZ, &EZ, &d1, p, 1);
        }
        /* below theta_z: G_hi (1 + E_Z) */
        if (!rc && HAS(THETA_Z) && !(rc = dyad_add(&d1, &one_iv.hi, &EZ, p, 1)) && !(rc = dyad_mul(&d1, &d1, &Ghi, p, 1)) &&
            dyad_cmp(&d1, C(THETA_Z)) >= 0)
            fail = 1;
        ivl_free(&run);
        dyad_free(&X);
    } else {
        if (HAS(THETA_Z)) {
            /* G_hi + Gz1 (r_z n G_hi + n a_z + (n - 1) a2_z) < theta_z */
            if (!(rc = dyad_mul(&d1, C(R_Z), &nd, p, 1)) && !(rc = dyad_mul(&d1, &d1, &Ghi, p, 1)) && !(rc = dyad_mul(&d2, &nd, C(A_Z), p, 1)) &&
                !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d2, &n1d, C(A2_Z), p, 1)) && !(rc = dyad_add(&d1, &d1, &d2, p, 1)) &&
                !(rc = dyad_mul(&d1, &d1, &pn[ATT_N_GZ1], p, 1)) && !(rc = dyad_add(&d1, &d1, &Ghi, p, 1)) && dyad_cmp(&d1, C(THETA_Z)) >= 0)
                fail = 1;
        }
        int forms = 0;
        if (!rc && !fail && pnh[ATT_N_G_Z]) {
            if (!(rc = dyad_mul(&d1, &pn[ATT_N_K_Z], &g, p, 1))) rc = dyad_add(&EZ, &pn[ATT_N_G_Z], &d1, p, 1);
            forms = 1;
        }
        if (!rc && !fail && HAS(THETA_Z)) {
            /* Gzn (r_z + (n a_z + (n - 1)(r_z theta_z + a2_z)) g) */
            if (!(rc = dyad_mul(&d1, C(R_Z), C(THETA_Z), p, 1)) && !(rc = dyad_add(&d1, &d1, C(A2_Z), p, 1)) && !(rc = dyad_mul(&d1, &d1, &n1d, p, 1)) &&
                !(rc = dyad_mul(&d2, &nd, C(A_Z), p, 1)) && !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1)) &&
                !(rc = dyad_add(&d1, &d1, C(R_Z), p, 1)) && !(rc = dyad_mul(&d1, &d1, &pn[ATT_N_GZN], p, 1)) && (!forms || dyad_cmp(&d1, &EZ) < 0))
                rc = dyad_copy(&EZ, &d1);
            forms = 1;
        }
        if (!forms) fail = 1;
    }
    if (rc || fail) goto out;
    if (dyad_cmp(&EZ, &one_iv.hi) >= 0) goto out;
    /* reciprocal: d_r = rnd(reciprocal, 1 / (G_lo (1 - E_Z)), 1 / (G_hi (1 + E_Z))), below its largest value */
    if ((t->methods & ATT_T_RECIPROCAL) &&
        (!(rc = sub_dir(&d1, &one_iv.lo, &EZ, p, 0)) && !(rc = dyad_mul(&d1, &Glo, &d1, p, 0)) && !(rc = dyad_div(&d1, &one_iv.hi, &d1, p, 1)) &&
         !(rc = dyad_add(&d2, &one_iv.hi, &EZ, p, 1)) && !(rc = dyad_mul(&d2, &Ghi, &d2, p, 1)) && !(rc = dyad_div(&d2, &one_iv.lo, &d2, p, 0)) &&
         !(rc = boundary_error(&dr, &t->rnd[ATT_R_RECIPROCAL], &d1, &d2)) && !(rc = dyad_add(&d1, &d1, &dr, p, 1)))) {
        if (dyad_cmp(&d1, C(MAX_RECIPROCAL)) > 0) goto out;
    }
    if (rc) goto out;
    *ok = 1;
    /* the row's quantities into sf */
    ivl *from[] = {&m, &lam_up, &lam_lo, &kup, &klo, &W, &D, &tau}, *to[] = {&sf->m, &sf->lam_up, &sf->lam_lo, &sf->kup, &sf->klo, &sf->W, &sf->D, &sf->tau};
    for (size_t z = 0; !rc && z < sizeof from / sizeof *from; z++) rc = ivl_copy(to[z], from[z]);
    dyad *dfrom[] = {&g, &Glo, &Ghi, &EZ, &dr}, *dto[] = {&sf->g, &sf->Glo, &sf->Ghi, &sf->EZ, &sf->dr};
    for (size_t z = 0; !rc && z < sizeof dfrom / sizeof *dfrom; z++) rc = dyad_copy(dto[z], dfrom[z]);
    if (rc) *ok = 0;
out:
    if (sd) {
        for (size_t j = 0; j < n; j++) dyad_free(&sd[j]);
        free(sd);
    }
    for (size_t z = 0; z < sizeof keyed / sizeof *keyed; z++) ivl_free(keyed[z]);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    product_sums_free(&ps);
    product_sums_free(&dev_ps);
    return rc;
}

/* x's least value after a rounding at b that moves y by at most rel |y| + abs (rel = c 2**-mantissa, abs = c 2**(emin -
   mantissa), plus 2**emin with flush), rounded down; x at most 0 stays */
static int unit_low(dyad *x, const round_units *b, int64_t p) {
    if (!b->active || dyad_sign(x) <= 0) return 0;
    dyad r;
    dyad_init(&r);
    int rc = dyad_copy(&r, &b->ulps);
    if (!rc && !(rc = dyad_shift(&r, -b->mantissa)) && !(rc = dyad_mul(&r, &r, x, p, 1)) && !(rc = sub_dir(x, x, &r, p, 0)) && !(rc = dyad_copy(&r, &b->ulps)) &&
        !(rc = dyad_shift(&r, b->emin - b->mantissa)) && !(rc = sub_dir(x, x, &r, p, 0)) && b->flush && !(rc = dyad_set_si(&r, 1, b->emin)))
        rc = sub_dir(x, x, &r, p, 0);
    dyad_free(&r);
    return rc;
}

/* The closed form's kept key j: *zero where its held weight (online: times its later rescaling factors) may be exactly 0
   (scaled_dot_product_attention.md, Masked keys): a key that may vanish, or one whose weight's least value is at most 0.
   Its exponential is at least e = exp((x_j - m) - R_j) / k_up (x_j - m times ln 2 with exp2; 0 below -2**20). Without
   online softmax the exp's rounding, and with weights theirs, each leave at least y (1 - rel) - abs of a y (unit_low);
   online, the weight times its factors is at least e down - own >= e (2 - up) - own, and with weights a held weight of 0
   needs that at most 2 abs_W growth. */
static int closed_zero(int *zero, const att_soft *sf, const att_tol *t, size_t j, int64_t p) {
    *zero = 1;
    if (sf->vanish[j]) return 0;
    dyad d, e, r;
    dyad_init(&d);
    dyad_init(&e);
    dyad_init(&r);
    int rc = sub_dir(&d, &sf->xs[j].lo, &sf->m.hi, p, 0);
    if (!rc && (t->methods & ATT_T_EXP2)) rc = dyad_mul(&d, &d, dyad_sign(&d) >= 0 ? &t->ln2_lo : &t->ln2_hi, p, 0);
    if (!rc) rc = sub_dir(&d, &d, &sf->Rs[j].hi, p, 0);
    if (!rc && !(rc = dyad_set_si(&r, -1, 20)) && dyad_cmp(&d, &r) >= 0 && !(rc = exp_end(&e, &d, p, 0)) && !(rc = dyad_div(&e, &e, &sf->kup.hi, p, 0))) {
        if (t->methods & ATT_T_ONLINE) {
            if (!(rc = dyad_set_si(&r, 2, 0)) && !(rc = sub_dir(&r, &r, C(UP), p, 0)))
                rc = dyad_sign(&r) > 0 ? dyad_mul(&e, &e, &r, p, 0) : dyad_set_si(&e, 0, 0);
            if (!rc) rc = sub_dir(&e, &e, C(OWN), p, 0);
            if (!rc && HAS(R_W) && !(rc = dyad_mul(&r, C(A_W), C(WMAX), p, 1)) && !(rc = dyad_shift(&r, 1))) rc = sub_dir(&e, &e, &r, p, 0);
        } else if (!(rc = unit_low(&e, &t->rnd[ATT_R_EXP], p)) && (t->methods & ATT_T_WEIGHTS)) {
            rc = unit_low(&e, &t->rnd[ATT_R_WEIGHTS], p);
        }
        if (!rc) *zero = dyad_sign(&e) <= 0;
    }
    dyad_free(&d);
    dyad_free(&e);
    dyad_free(&r);
    return rc;
}

int att_tolerance_row(att_row *w, const att_call *a, const att_tol *t, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0,
                      const entry *out_entries,
                      const unsigned char *out_refs, const entry *lse_entry, const unsigned char *lse_ref, unsigned char *tol, unsigned char *lows,
                      unsigned char *highs, unsigned char *flags, unsigned char *lse_tol, unsigned char *lse_low, unsigned char *lse_high,
                      unsigned char *lse_flag, unsigned char *status) {
    const operand *v = a->v;
    size_t S = a->S, Ev = a->Ev;
    int64_t p = t->p;
    *status = 1;
    int rc = att_row_reserve(w, S, Ev);
    if (rc) return rc;
    int kept_count;
    if ((rc = att_row_logits(w, a, qr, kr0, i, moff, mstep, NULL, &kept_count))) return rc;
    if (kept_count <= 0) return 0; /* a NaN row, or no kept key: the step-by-step program and the padded rows' bound */
    int nan_row = 0;                /* the row's outputs and logsumexp may be NaN (Masked keys) */
    if (a->uncertain && row_unbounded(a, &t->f64, w, qr, kr0, moff, mstep, &nan_row)) return 0; /* an input with no bound: the row has none */
    size_t n = (size_t)kept_count, *keys = w->keys;
    int general = v->flagged_count || v->slots_hi; /* V with flagged or two-word slots: the column sums from exact products */
    if (t->normal && !general && a->vt->slots_hi) general = 1;
    for (size_t c = 0; t->normal && !general && a->vt->flagged_count && c < Ev; c++)
        general = a->vt->flagged_rows[(vr0 / S) * Ev + c]; /* the value nodes read V transposed, flagged in this head too */
    att_soft sf;
    att_soft_init(&sf);
    ivl one_iv;
    ivl_init(&one_iv);
    dyad err, d1, d2, d3, nd, n1d;
    dyad *ds[] = {&err, &d1, &d2, &d3, &nd, &n1d};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    uint64_t *pint = NULL, *plo = NULL, *eint = NULL, *prod = NULL, *mcol = NULL;
    w128 *rmaxc = NULL;          /* with V's radii, each column's largest |v| + r over the kept keys (at 2**ve) */
    unsigned char *cunb = NULL;  /* ... and whether an element of the column has no bound */
    unsigned char *cnone = NULL; /* per column: its reference is not finite (no bound needed) */
    unsigned char *cnan = NULL;  /* ... and whether it may be NaN (Masked keys) */
    int fail = 0, ok = 0;
    int prefix = keys[n - 1] == n - 1;
    if (!(pint = malloc(n * sizeof *pint)) || !(plo = malloc(n * sizeof *plo)) || !(eint = malloc(n * sizeof *eint)) ||
        !(prod = calloc(Ev * 11 * ATT_PW, sizeof *prod)) || !(mcol = calloc(Ev ? Ev : 1, sizeof *mcol)) ||
        !(rmaxc = calloc(Ev ? Ev : 1, sizeof *rmaxc)) || !(cunb = calloc(Ev ? Ev : 1, 1)) || !(cnone = calloc(Ev ? Ev : 1, 1)) ||
        !(cnan = calloc(Ev ? Ev : 1, 1))) {
        rc = EXACTREF_NOMEM;
        goto done;
    }
    if ((rc = att_soft_row(&sf, &ok, w, a, t, n, qr, kr0, moff, mstep, 0, 0, NULL)) || !ok) goto done;
    nan_row |= sf.masked_nan;
    att_masked_nan(&nan_row, cnan, w, a, vr0);
    /* a kept key whose held weight may be exactly 0 times a value entry that may not be finite: NaN (Masked keys) */
    for (size_t j = 0; !rc && j < n; j++) {
        int zero = 0;
        if (value_special(a, vr0, keys[j]) && !(rc = closed_zero(&zero, &sf, t, j, t->p)) && zero) value_nan(cnan, a, vr0, keys[j]);
    }
    if (rc) goto done;
    if ((rc = dyad_set_si(&one_iv.lo, 1, 0)) || (rc = dyad_set_si(&one_iv.hi, 1, 0))) goto done;
    ivl *Ps = sf.Ps, *PTs = sf.PTs, *PVs = sf.PVs;
    /* read-only views of the row's quantities */
    ivl m = sf.m, lam_up = sf.lam_up, lam_lo = sf.lam_lo, D = sf.D, tau = sf.tau;
    dyad g = sf.g, Glo = sf.Glo, Ghi = sf.Ghi, EZ = sf.EZ, dr = sf.dr;
    int online = t->methods & ATT_T_ONLINE, exp2 = t->methods & ATT_T_EXP2;
    const dyad *pn = t->per_n + n * ATT_PER_N;
    const unsigned char *pnh = t->per_n_has + n * ATT_PER_N;
    if ((rc = dyad_set_si(&nd, (int64_t)n, 0)) || (rc = dyad_set_si(&n1d, (int64_t)n - 1, 0))) goto done;
    /* the integers: P up and down and the error weights up, at units 62 bits (fewer where the value
       nodes need it) below their largest */
    {
        const ivl *Es = PVs ? PVs : PTs; /* the error weights of the value sums (with rounded weights, PV) */
        const dyad *pmax = &Ps[0].hi, *emax = &Es[0].hi;
        for (size_t j = 1; j < n; j++) {
            pmax = dmax(pmax, &Ps[j].hi);
            emax = dmax(emax, &Es[j].hi);
        }
        int64_t pbits = 62;
        if (t->normal && !general) {
            int64_t room = 126 - a->vt->width - 2 * (int64_t)word_bits((uint64_t)n);
            if (room < pbits) pbits = room;
            if (pbits < 24) goto done;
        }
        int64_t uP = top_bit(pmax) + 1 - pbits, uE = dyad_sign(emax) > 0 ? top_bit(emax) + 1 - 62 : 0;
        for (size_t j = 0; !rc && j < n; j++) {
            bigint bi;
            big_init(&bi);
            dyad tmp;
            dyad_init(&tmp);
            /* ceil(P.hi / 2**uP), floor(P.lo / 2**uP), ceil(E.hi / 2**uE) */
            if (!(rc = dyad_copy(&tmp, &Ps[j].hi)) && !(rc = dyad_shift(&tmp, -uP))) {
                rc = dyad_to_fixed(&bi, &tmp, 0, 1);
            }
            pint[j] = bi.n ? bi.w[0] : 0;
            if (!rc && !(rc = dyad_copy(&tmp, &Ps[j].lo)) && !(rc = dyad_shift(&tmp, -uP))) {
                rc = dyad_sign(&tmp) > 0 ? dyad_to_fixed(&bi, &tmp, 0, 0) : big_set_word(&bi, 0, 0);
            }
            plo[j] = bi.n ? bi.w[0] : 0;
            if (!rc && !(rc = dyad_copy(&tmp, &Es[j].hi)) && !(rc = dyad_shift(&tmp, -uE))) {
                rc = dyad_to_fixed(&bi, &tmp, 0, 1);
            }
            eint[j] = bi.n ? bi.w[0] : 0;
            big_free(&bi);
            dyad_free(&tmp);
        }
        if (rc) goto done;
        for (size_t m = 0; a->vunb && m < n; m++) /* a kept key's value with no bound leaves its column none */
            for (size_t c = 0; c < Ev; c++) cunb[c] |= a->vunb[(vr0 + keys[m]) * Ev + c];
        /* per column: S = sum P |v|, U = sum E |v|, V = sum |v|, rad = sum (P - p_lo) |v|; with flush, F, UF and VF the
           same sums of the values below 2**emin (V's integers below limit) */
        uint64_t limit = t->flush && t->emin > v->exponent ? (t->emin - v->exponent < 64 ? (uint64_t)1 << (t->emin - v->exponent) : UINT64_MAX) : 0;
        for (size_t j = 0; !general && j < n; j++) {
            const uint64_t *vs = v->slots + (vr0 + keys[j]) * Ev;
            for (size_t c = 0; c < Ev; c++) {
                uint64_t x0 = vs[c], mag = x0 >> 63 ? ~x0 + 1 : x0;
                if (!mag) continue;
                if (mag > mcol[c]) mcol[c] = mag; /* the column's largest |v| over the kept keys */
                uint64_t *col = prod + c * 11 * ATT_PW;
                acc_word(col, ATT_PW, pint[j], mag);
                acc_word(col + ATT_PW, ATT_PW, eint[j], mag);
                acc_word(col + 2 * ATT_PW, ATT_PW, 1, mag);
                if (pint[j] != plo[j]) acc_word(col + 3 * ATT_PW, ATT_PW, pint[j] - plo[j], mag);
                if (mag < limit) {
                    acc_word(col + 4 * ATT_PW, ATT_PW, pint[j], mag);
                    acc_word(col + 5 * ATT_PW, ATT_PW, eint[j], mag);
                    acc_word(col + 6 * ATT_PW, ATT_PW, 1, mag);
                }
            }
            for (size_t c = 0; a->vv && c < Ev; c++) {
                /* V's radii (at 2**ve): P r, E r, r and (2 P - p_lo) r, and the largest |v| + r */
                size_t at = (vr0 + keys[j]) * Ev + c;
                w128 vv = read128(a->vv + 16 * at), r = read128(a->vr + 16 * at), mag = w128_add(w128_hi(vv) >> 63 ? w128_sub(w128_zero(), vv) : vv, r);
                uint64_t *col = prod + c * 11 * ATT_PW;
                if (w128_less(rmaxc[c], mag)) rmaxc[c] = mag;
                acc_wide(col + 7 * ATT_PW, ATT_PW, pint[j], r);
                acc_wide(col + 8 * ATT_PW, ATT_PW, eint[j], r);
                acc_wide(col + 9 * ATT_PW, ATT_PW, 1, r);
                acc_wide(col + 10 * ATT_PW, ATT_PW, 2 * pint[j] - plo[j], r);
            }
        }
        /* the check on the integers (The integer products): 2**16 X_c <= U_c with X_c's factors rounded up to 32 bits */
        dyad cfac, fa, fb, hmax;
        dyad_init(&cfac);
        dyad_init(&fa);
        dyad_init(&fb);
        dyad_init(&hmax);
        /* the value sums' forms: normal h = Gv r_v, k = Gv (n a_v + (n - 1) a2_v) g, pi = h; any order: Lemma A's */
        dyad hf[2], kf[2], pi;
        int nf = 0;
        for (int z = 0; z < 2; z++) {
            dyad_init(&hf[z]);
            dyad_init(&kf[z]);
        }
        dyad_init(&pi);
        rc = dyad_set_si(&pi, 0, 0);
        /* online softmax rounding the weights: E_Q and H take w = r_w (weights_w); f, phi's factor in e_N */
        int weights_w = online && HAS(R_W);
        const dyad *fphi = !online && t->normal ? &pn[ATT_N_GV] : &one_iv.hi;
        if (!rc && online) {
            /* h = g_v, k = (k_v + x_n0) g, pi = 0 */
            if (!(rc = dyad_copy(&hf[0], C(ON_G_V))) && !(rc = dyad_add(&d1, C(ON_K_V), C(XN0), p, 1))) rc = dyad_mul(&kf[0], &d1, &g, p, 1);
            nf = 1;
        } else if (!rc && t->normal) {
            if (!(rc = dyad_mul(&hf[0], &pn[ATT_N_GV], C(R_V), p, 1)) && !(rc = dyad_mul(&d1, &nd, C(A_V), p, 1)) &&
                !(rc = dyad_mul(&d2, &n1d, C(A2_V), p, 1)) && !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1)) &&
                !(rc = dyad_mul(&kf[0], &pn[ATT_N_GV], &d1, p, 1)))
                rc = dyad_copy(&pi, &hf[0]);
            nf = 1;
        } else if (!rc) {
            if (pnh[ATT_N_G_V]) {
                if (!(rc = dyad_copy(&hf[nf], &pn[ATT_N_G_V]))) rc = dyad_mul(&kf[nf], &pn[ATT_N_K_V], &g, p, 1);
                nf++;
            }
            if (!rc && HAS(THETA_V)) {
                if (!(rc = dyad_mul(&hf[nf], &pn[ATT_N_GVN], C(R_V), p, 1)) && !(rc = dyad_mul(&d1, C(R_V), C(THETA_V), p, 1)) &&
                    !(rc = dyad_add(&d1, &d1, C(A2_V), p, 1)) && !(rc = dyad_mul(&d1, &d1, &n1d, p, 1)) && !(rc = dyad_mul(&d2, &nd, C(A_V), p, 1)) &&
                    !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1)))
                    rc = dyad_mul(&kf[nf], &pn[ATT_N_GVN], &d1, p, 1);
                nf++;
            }
        }
        /* c = h (h the largest form's factor) + 3 (n - 1) pi; factors (1 + c) 2**uE + c 2**uP */
        if (!rc && nf) {
            rc = dyad_copy(&hmax, &hf[0]);
            if (!rc && nf > 1 && dyad_cmp(&hf[1], &hmax) > 0) rc = dyad_copy(&hmax, &hf[1]);
            if (!rc && !(rc = dyad_set_si(&d1, 3 * ((int64_t)n - 1), 0)) && !(rc = dyad_mul(&d1, &d1, &pi, p, 1)))
                rc = dyad_add(&cfac, &hmax, &d1, p, 1);
            /* with w: c = w + h (1 + w) + 3 (n - 1) pi */
            if (!rc && weights_w && !(rc = dyad_add(&d2, &one_iv.hi, C(R_W), p, 1)) && !(rc = dyad_mul(&d2, &hmax, &d2, p, 1)) &&
                !(rc = dyad_add(&d2, &d2, C(R_W), p, 1)))
                rc = dyad_add(&cfac, &d2, &d1, p, 1);
            if (!rc && !(rc = dyad_set_si(&d1, 1, uE)) && !(rc = dyad_set_si(&d2, 1, 0)) && !(rc = dyad_add(&d2, &d2, &cfac, p, 1)) &&
                !(rc = dyad_mul(&fa, &d1, &d2, p, 1)) && !(rc = dyad_set_si(&d1, 1, uP)) && !(rc = dyad_mul(&d1, &d1, &cfac, p, 1)) &&
                !(rc = dyad_add(&fa, &fa, &d1, p, 1)))
                rc = dyad_rnd(&fa, &fa, 32, 1);
            /* with flush, VF's factor f' (2**uE + 2**uP), f' = (1 + w) f */
            if (!rc && t->flush && !(rc = dyad_set_si(&d1, 1, uE)) && !(rc = dyad_set_si(&d2, 1, uP)) && !(rc = dyad_add(&d1, &d1, &d2, p, 1)) &&
                !(rc = dyad_mul(&fb, fphi, &d1, p, 1)) && (!weights_w || (!(rc = dyad_add(&d2, &one_iv.hi, C(R_W), p, 1)) && !(rc = dyad_mul(&fb, &fb, &d2, p, 1)))))
                rc = dyad_rnd(&fb, &fb, 32, 1);
        }
        int coarse = 0;
        for (size_t c = 0; !rc && nf && !general && c < Ev && !coarse; c++) {
            uint64_t *col = prod + c * 11 * ATT_PW;
            dyad Vd, Ud;
            dyad_init(&Vd);
            dyad_init(&Ud);
            /* 2**16 (fa V_c + fb VF_c + 2**uP RV_c) > U_c, with V's radii in V_c and U_c (|v| + r) and RV_c = sum r */
            int64_t ev = v->exponent;
            if (!(rc = dyad_words(&Vd, col + 2 * ATT_PW, ATT_PW, ev + 16)) && !(rc = dyad_words(&Ud, col + ATT_PW, ATT_PW, uE + ev)) && a->vv &&
                !(rc = dyad_words(&d1, col + 9 * ATT_PW, ATT_PW, a->ve + 16)) && !(rc = dyad_add_exact(&Vd, &Vd, &d1)) &&
                !(rc = dyad_words(&d1, col + 8 * ATT_PW, ATT_PW, uE + a->ve)))
                rc = dyad_add_exact(&Ud, &Ud, &d1);
            if (!rc) rc = dyad_mul(&Vd, &Vd, &fa, 1 << 20, 1);
            if (!rc && t->flush && !(rc = dyad_words(&d1, col + 6 * ATT_PW, ATT_PW, ev + 16)) && !(rc = dyad_mul(&d1, &d1, &fb, 1 << 20, 1)))
                rc = dyad_add(&Vd, &Vd, &d1, 1 << 20, 1);
            if (!rc && a->vv && !(rc = dyad_words(&d1, col + 9 * ATT_PW, ATT_PW, a->ve + 16 + uP))) rc = dyad_add(&Vd, &Vd, &d1, 1 << 20, 1);
            if (!rc) coarse = dyad_cmp(&Vd, &Ud) > 0;
            dyad_free(&Vd);
            dyad_free(&Ud);
        }
        if (!nf) fail = 1;
        /* step 7 to 9 per column (elements) */
        dyad inverse, divisor;
        dyad_init(&inverse);
        dyad_init(&divisor);
        if (!rc && !fail && !(rc = dyad_set_si(&d1, 1, 0)) && !(rc = dyad_div(&inverse, &d1, &D.lo, p, 1)) &&
            !(rc = sub_dir(&d2, &d1, &EZ, p, 0)))
            rc = dyad_div(&divisor, &d1, &d2, p, 1);
        dyad *los = NULL, *his = NULL;
        if (!rc && !fail) {
            los = malloc(2 * (Ev ? Ev : 1) * sizeof *los); /* the first pass's, then the second's */
            his = malloc(2 * (Ev ? Ev : 1) * sizeof *his);
            if (!los || !his) rc = EXACTREF_NOMEM;
            for (size_t c = 0; !rc && c < 2 * Ev; c++) {
                dyad_init(&los[c]);
                dyad_init(&his[c]);
            }
        }
        /* the column sums from exact products of P's and the error weights' upper ends (exact_columns, P's lower end at least 0,
           no unit rounding them): the only pass for V with flagged or two-word slots, else the second pass, for a row whose check
           on the integers fails (the proof's finer unit); each output keeps the intersection with the first
           pass, and the row keeps the first pass where the second fails */
        dyad *xc = NULL;
        size_t nx = Ev ? Ev : 1;
        for (int pass = 0; pass < 2 && !rc && !fail; pass++) {
            int pf = 0, ex = pass || general;
            dyad *dlo = los + (pass ? Ev : 0), *dhi = his + (pass ? Ev : 0);
            if (pass && (general || !coarse)) break;
            if (ex) {
                if (!(xc = malloc(XC_SUMS * nx * sizeof *xc))) {
                    rc = EXACTREF_NOMEM;
                    break;
                }
                for (size_t z = 0; z < XC_SUMS * nx; z++) dyad_init(&xc[z]);
                if ((rc = exact_columns(xc, nx, a, Ps, Es, keys, n, vr0, t->flush, t->emin, t->normal))) break;
            }
#define XS(name) (&xc[XC_##name * nx + c])
            for (size_t c = 0; !rc && !pf && c < Ev; c++) {
                if (cnone[c]) continue;
                uint64_t *col = prod + c * 11 * ATT_PW;
                int none = 0;
                dyad olo, ohi, O, Olo, U, Sd, V, EQ, Hd, Q, eN, Nd, hi2, EN, ER, Ad;
                dyad *cd[] = {&olo, &ohi, &O, &Olo, &U, &Sd, &V, &EQ, &Hd, &Q, &eN, &Nd, &hi2, &EN, &ER, &Ad};
                for (size_t z = 0; z < sizeof cd / sizeof *cd; z++) dyad_init(cd[z]);
                rc = entry_bounds(&olo, &ohi, &none, &out_entries[c]);
                if (!rc && none) cnone[c] = 1; /* a reference that is not finite: no bound needed */
                if (!rc && !none) {
                    int64_t ev = v->exponent;
                    rc = ex ? dyad_copy(&U, XS(U)) : dyad_words(&U, col + ATT_PW, ATT_PW, uE + ev);
                    if (!rc) rc = ex ? dyad_copy(&Sd, XS(S)) : dyad_words(&Sd, col, ATT_PW, uP + ev);
                    if (!rc) rc = ex ? dyad_copy(&V, XS(V)) : dyad_words(&V, col + 2 * ATT_PW, ATT_PW, ev);
                    if (!rc && a->vv && ex) {
                        /* V's radii are in the exact sums; the reference on the stored values is within D = sum P r of it */
                        if (!(rc = sub_dir(&olo, &olo, XS(D), t->w, 0))) rc = dyad_add(&ohi, &ohi, XS(D), t->w, 1);
                    } else if (!rc && a->vv) {
                        /* V's radii: S, U and V over |v| + r, and the reference on the stored values within D = sum P r of it */
                        if (!(rc = dyad_words(&d1, col + 8 * ATT_PW, ATT_PW, uE + a->ve)) && !(rc = dyad_add_exact(&U, &U, &d1)) &&
                            !(rc = dyad_words(&d1, col + 7 * ATT_PW, ATT_PW, uP + a->ve)) && !(rc = dyad_add_exact(&Sd, &Sd, &d1)) &&
                            !(rc = sub_dir(&olo, &olo, &d1, t->w, 0)) && !(rc = dyad_add(&ohi, &ohi, &d1, t->w, 1)) &&
                            !(rc = dyad_words(&d1, col + 9 * ATT_PW, ATT_PW, a->ve)))
                            rc = dyad_add_exact(&V, &V, &d1);
                    }
                    /* O and O' */
                    dyad al, ah;
                    dyad_init(&al);
                    dyad_init(&ah);
                    if (!rc && !(rc = dyad_copy(&al, &olo)) && !(rc = dyad_copy(&ah, &ohi))) {
                        al.m.neg = 0;
                        ah.m.neg = 0;
                        rc = dyad_copy(&O, dmax(&al, &ah));
                        if (!rc) rc = (dyad_sign(&olo) <= 0 && dyad_sign(&ohi) >= 0) ? dyad_set_si(&Olo, 0, 0) : dyad_copy(&Olo, dmin(&al, &ah));
                    }
                    dyad_free(&al);
                    dyad_free(&ah);
                    /* online: wmax_v V (1 + g_v) + k_v <= lim_v */
                    if (!rc && online && !(rc = dyad_mul(&d1, C(WMAX_V), &V, p, 1)) && !(rc = dyad_add(&d2, &one_iv.hi, C(ON_G_V), p, 1)) &&
                        !(rc = dyad_mul(&d1, &d1, &d2, p, 1)) && !(rc = dyad_add(&d1, &d1, C(ON_K_V), p, 1)) && dyad_cmp(&d1, C(ON_LIM_V)) > 0)
                        pf = 1;
                    /* E_Q = (U + O tau) / D; H = (S + U) / D; with w: E_Q += w H + a_w g V, H = (1 + w) H + a_w g V; Q = O + E_Q */
                    if (!rc && !(rc = dyad_mul(&d1, &O, &tau.hi, p, 1)) && !(rc = dyad_add(&d1, &d1, &U, p, 1)) && !(rc = dyad_mul(&EQ, &d1, &inverse, p, 1)) &&
                        !(rc = dyad_add(&d1, &Sd, &U, p, 1)))
                        rc = dyad_mul(&Hd, &d1, &inverse, p, 1);
                    if (!rc && weights_w && !(rc = dyad_mul(&d3, C(A_W), &g, p, 1)) && !(rc = dyad_mul(&d3, &d3, &V, p, 1)) &&
                        !(rc = dyad_mul(&d1, C(R_W), &Hd, p, 1)) && !(rc = dyad_add(&EQ, &EQ, &d1, p, 1)) && !(rc = dyad_add(&EQ, &EQ, &d3, p, 1)) &&
                        !(rc = dyad_add(&d1, &one_iv.hi, C(R_W), p, 1)) && !(rc = dyad_mul(&Hd, &d1, &Hd, p, 1)))
                        rc = dyad_add(&Hd, &Hd, &d3, p, 1);
                    if (!rc) rc = dyad_add(&Q, &O, &EQ, p, 1);
                    /* with flush, phi = (F + UF) / D (with w: (1 + w) phi + a_w g VF), and e_N gains f phi below */
                    dyad phi;
                    dyad_init(&phi);
                    if (!rc) rc = dyad_set_si(&phi, 0, 0);
                    if (!rc && t->flush && !(rc = ex ? dyad_copy(&d1, XS(F)) : dyad_words(&d1, col + 4 * ATT_PW, ATT_PW, uP + ev)) &&
                        !(rc = ex ? dyad_copy(&d2, XS(UF)) : dyad_words(&d2, col + 5 * ATT_PW, ATT_PW, uE + ev)) &&
                        !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&phi, &d1, &inverse, p, 1)) && weights_w &&
                        !(rc = dyad_add(&d1, &one_iv.hi, C(R_W), p, 1)) && !(rc = dyad_mul(&phi, &d1, &phi, p, 1)) && !(rc = dyad_mul(&d1, C(A_W), &g, p, 1)) &&
                        !(rc = ex ? dyad_copy(&d2, XS(VF)) : dyad_words(&d2, col + 6 * ATT_PW, ATT_PW, ev)) && !(rc = dyad_mul(&d1, &d1, &d2, p, 1)))
                        rc = dyad_add(&phi, &phi, &d1, p, 1);
                    if (!rc && !t->normal && !online && HAS(THETA_V)) {
                        /* an implementation without overflow exists: size + Gv1 (r_v n size + n a_v + (n - 1) a2_v) < theta_v */
                        if (!(rc = dyad_mul(&d3, C(WMAX_V), &V, p, 1)) && !(rc = dyad_mul(&d1, C(R_V), &nd, p, 1)) && !(rc = dyad_mul(&d1, &d1, &d3, p, 1)) &&
                            !(rc = dyad_mul(&d2, &nd, C(A_V), p, 1)) && !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d2, &n1d, C(A2_V), p, 1)) &&
                            !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, &pn[ATT_N_GV1], p, 1)) &&
                            !(rc = dyad_add(&d1, &d1, &d3, p, 1)) && dyad_cmp(&d1, C(THETA_V)) >= 0)
                            pf = 1;
                    }
                    /* e_N = the least form h H + k, plus pi (Nd + (n - 1) U) / D (normal) */
                    for (int z = 0; !rc && !pf && z < nf; z++) {
                        if (!(rc = dyad_mul(&d1, &hf[z], &Hd, p, 1)) && !(rc = dyad_add(&d1, &d1, &kf[z], p, 1)) && (!z || dyad_cmp(&d1, &eN) < 0))
                            rc = dyad_copy(&eN, &d1);
                    }
                    if (!rc && !pf && t->flush && !(rc = dyad_mul(&d1, fphi, &phi, p, 1))) rc = dyad_add(&eN, &eN, &d1, p, 1);
                    dyad_free(&phi);
                    /* online: the rescalings' part x_n1 M g, M the column's largest |v| (with V's radii, |v| + r) */
                    if (!rc && !pf && online && ex) {
                        if (!(rc = dyad_mul(&d1, C(XN1), XS(M), p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1))) rc = dyad_add(&eN, &eN, &d1, p, 1);
                    } else if (!rc && !pf && online && a->vv) {
                        uint64_t mw[2] = {w128_lo(rmaxc[c]), w128_hi(rmaxc[c])}; /* the largest |v| + r */
                        if (!(rc = dyad_words(&d2, mw, 2, a->ve)) && !(rc = dyad_mul(&d1, C(XN1), &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1)))
                            rc = dyad_add(&eN, &eN, &d1, p, 1);
                    } else if (!rc && !pf && online && !(rc = big_set_word(&d2.m, mcol[c], 0))) {
                        d2.e = d2.m.n ? ev : 0;
                        if (!(rc = dyad_mul(&d1, C(XN1), &d2, p, 1)) && !(rc = dyad_mul(&d1, &d1, &g, p, 1))) rc = dyad_add(&eN, &eN, &d1, p, 1);
                    }
                    if (!rc && !pf && t->normal) {
                        /* Nd: the node sums of P's integers times the column (every node exact) plus (n - 1) rad */
                        w128 sum, mass, Nn;
                        if (ex) { /* the exact node bound: Nd and the (n - 1) rad part together */
                            if (!(rc = dyad_mul(&d1, &n1d, &U, p, 1)) && !(rc = dyad_add(&d1, &d1, XS(N), p, 1)) && !(rc = dyad_mul(&d1, &d1, &inverse, p, 1)) &&
                                !(rc = dyad_mul(&d1, &d1, &pi, p, 1)))
                                rc = dyad_add(&eN, &eN, &d1, p, 1);
                        } else {
                            const uint64_t *col_t = a->vt->slots + ((vr0 / S) * Ev + c) * a->vt->cols;
                            if (prefix) /* the lanes by key index: by rank when the kept keys are the first n */
                                product_fast_nodes(&sum, &mass, &Nn, pint, col_t, n);
                            else
                                product_fast_nodes_at(&sum, &mass, &Nn, pint, col_t, keys, n);
                            w128 best = w128_sub(w128_make(w128_lo(Nn) >> 1 | w128_hi(Nn) << 63, w128_hi(Nn) >> 1), mass);
                            uint64_t bw[2] = {w128_lo(best), w128_hi(best)};
                            if (!(rc = dyad_words(&Nd, bw, 2, uP + a->vt->exponent)) && !(rc = dyad_words(&d1, col + 3 * ATT_PW, ATT_PW, uP + ev)) &&
                                (!a->vv || (!(rc = dyad_words(&d2, col + 10 * ATT_PW, ATT_PW, uP + a->ve)) && !(rc = dyad_add_exact(&d1, &d1, &d2)))) &&
                                !(rc = dyad_mul(&d1, &d1, &n1d, p, 1)) && !(rc = dyad_add(&Nd, &Nd, &d1, p, 1)) && !(rc = dyad_mul(&d1, &n1d, &U, p, 1)) &&
                                !(rc = dyad_add(&d1, &d1, &Nd, p, 1)) && !(rc = dyad_mul(&d1, &d1, &inverse, p, 1)) && !(rc = dyad_mul(&d1, &d1, &pi, p, 1)))
                                rc = dyad_add(&eN, &eN, &d1, p, 1);
                        }
                        /* below theta_v: wmax_v V + G_hi e_N */
                        if (!rc && HAS(THETA_V) && !(rc = dyad_mul(&d1, C(WMAX_V), &V, p, 1)) && !(rc = dyad_mul(&d2, &Ghi, &eN, p, 1)) &&
                            !(rc = dyad_add(&d1, &d1, &d2, p, 1)) && dyad_cmp(&d1, C(THETA_V)) >= 0)
                            pf = 1;
                    }
                    /* the value sum's rounding at G_hi (Q + e_N); E_N = e_N + error / G_lo; E_R = (E_N + Q E_Z) / (1 - E_Z) */
                    if (!rc && !pf && !(rc = dyad_add(&d1, &Q, &eN, p, 1)) && !(rc = dyad_mul(&hi2, &Ghi, &d1, p, 1)) &&
                        !(rc = sub_dir(&d3, &Olo, &EQ, p, 0)) && !(rc = sub_dir(&d3, &d3, &eN, p, 0)) && !(rc = dyad_mul(&d3, &Glo, &d3, p, 0)) &&
                        !(rc = boundary_error(&err, &t->rnd[ATT_R_VALUES], &hi2, &d3)) && !(rc = dyad_add(&d1, &hi2, &err, p, 1))) {
                        if (dyad_cmp(&d1, C(MAX_VALUES)) > 0) pf = 1;
                    }
                    if (!rc && !pf && !(rc = dyad_div(&d1, &err, &Glo, p, 1)) && !(rc = dyad_add(&EN, &eN, &d1, p, 1)) &&
                        !(rc = dyad_mul(&d1, &Q, &EZ, p, 1)) && !(rc = dyad_add(&d1, &d1, &EN, p, 1)) && !(rc = dyad_mul(&ER, &d1, &divisor, p, 1))) {
                        /* reciprocal: E_R += G_hi (Q + E_N) d_r */
                        if ((t->methods & ATT_T_RECIPROCAL) && !(rc = dyad_add(&d1, &Q, &EN, p, 1)) && !(rc = dyad_mul(&d1, &Ghi, &d1, p, 1)) &&
                            !(rc = dyad_mul(&d1, &d1, &dr, p, 1)))
                            rc = dyad_add(&ER, &ER, &d1, p, 1);
                        if (!rc) rc = dyad_add(&Ad, &EQ, &ER, p, 1);
                    }
                    /* the interval before the final rounding: [o.lo - A, o.hi + A], then normalize's rounding */
                    if (!rc && !pf && !(rc = sub_dir(&dlo[c], &olo, &Ad, t->w, 0)) && !(rc = dyad_add(&dhi[c], &ohi, &Ad, t->w, 1))) {
                        int unbounded;
                        rc = lean_ends(&dlo[c], &dhi[c], &unbounded, &t->normalize, t->w);
                        if (unbounded) pf = 1;
                    }
                }
                for (size_t z = 0; z < sizeof cd / sizeof *cd; z++) dyad_free(cd[z]);
            }
            if (!pass) {
                fail |= pf;
            } else if (!pf && !rc) {
                for (size_t c = 0; !rc && c < Ev; c++) { /* the intersection of the two passes' intervals */
                    if (cnone[c]) continue;
                    if (dyad_cmp(&dlo[c], &los[c]) > 0) rc = dyad_copy(&los[c], &dlo[c]);
                    if (!rc && dyad_cmp(&dhi[c], &his[c]) < 0) rc = dyad_copy(&his[c], &dhi[c]);
                }
            }
        }
#undef XS
        if (xc) {
            for (size_t z = 0; z < XC_SUMS * nx; z++) dyad_free(&xc[z]);
            free(xc);
        }
        /* logsumexp: Z' / (K W) in [D (1 - E_Z), (1 + tau)(1 + E_Z)], E_log = lam + rnd(log, log n + spread + lam) */
        dyad llo, lhi;
        dyad_init(&llo);
        dyad_init(&lhi);
        int lse_none = 1;
        if (!rc && !fail) rc = entry_bounds(&llo, &lhi, &lse_none, lse_entry);
        if (!rc && !fail && !lse_none) {
            dyad Zlo, Zhi, lam, spread, El;
            dyad_init(&Zlo);
            dyad_init(&Zhi);
            dyad_init(&lam);
            dyad_init(&spread);
            dyad_init(&El);
            if (online) {
                /* alpha = (k_z + x_z) g D; Z' / (K W) in [D (1 - g_z) - alpha, (1 + tau)(1 + g_z) + alpha] */
                if (!(rc = dyad_add(&d3, C(ON_K_Z), C(X_Z), p, 1)) && !(rc = dyad_mul(&d3, &d3, &g, p, 1)) && !(rc = dyad_mul(&d3, &d3, &D.hi, p, 1)) &&
                    !(rc = sub_dir(&d1, &one_iv.lo, C(ON_G_Z), p, 0)) && !(rc = dyad_mul(&Zlo, &D.lo, &d1, p, 0)) && !(rc = sub_dir(&Zlo, &Zlo, &d3, p, 0)) &&
                    !(rc = dyad_add(&d1, &one_iv.hi, C(ON_G_Z), p, 1)) && !(rc = dyad_add(&d2, &one_iv.hi, &tau.hi, p, 1)) &&
                    !(rc = dyad_mul(&Zhi, &d1, &d2, p, 1)))
                    rc = dyad_add(&Zhi, &Zhi, &d3, p, 1);
            } else if (!(rc = sub_dir(&d1, &one_iv.lo, &EZ, p, 0)) && !(rc = dyad_mul(&Zlo, &D.lo, &d1, p, 0)) &&
                       !(rc = dyad_add(&d1, &one_iv.hi, &EZ, p, 1)) && !(rc = dyad_add(&d2, &one_iv.hi, &tau.hi, p, 1))) {
                rc = dyad_mul(&Zhi, &d1, &d2, p, 1);
            }
            if (!rc && dyad_sign(&Zlo) <= 0) fail = 1;
            if (!rc && !fail && !(rc = log_end(&d1, &Zlo, p, 0)) && !(rc = dyad_neg(&d1, &d1)) && !(rc = log_end(&d2, &Zhi, p, 1)))
                rc = dyad_copy(&lam, dmax(&d1, &d2));
            if (!rc && !fail && !(rc = dyad_copy(&spread, dmax(&lam_up.hi, &lam_lo.hi))) && !(rc = log_end(&d1, &nd, p, 1)) &&
                !(rc = dyad_add(&d1, &d1, &spread, p, 1)) && !(rc = dyad_add(&d1, &d1, &lam, p, 1)) &&
                !(rc = boundary_error(&err, &t->rnd[ATT_R_LOG], &d1, NULL)) && !(rc = dyad_add(&El, &lam, &err, p, 1))) {
                /* fast_log: c 2**(1 - p) more */
                if (HAS(LOG_FLOOR)) rc = dyad_add(&El, &El, C(LOG_FLOOR), p, 1);
                /* exp2: the base-2 maximum times log(2), within ulps: rnd(|m ln 2| + spread) */
                if (!rc && exp2 && !(rc = ends_magnitude(&d1, &m.lo, &m.hi)) && !(rc = dyad_mul(&d1, &d1, &t->ln2_hi, p, 1)) &&
                    !(rc = sub_dir(&d3, &d1, &spread, p, 0)) && !(rc = dyad_add(&d1, &d1, &spread, p, 1)) &&
                    !(rc = boundary_error(&err, &t->rnd[ATT_R_EXP2_LSE], &d1, &d3)) &&
                    !(rc = dyad_add(&d2, &d1, &err, p, 1))) {
                    if (dyad_cmp(&d2, C(MAX_EXP2_LOGSUMEXP)) > 0) fail = 1;
                    else rc = dyad_add(&El, &El, &err, p, 1);
                }
                if (!rc && !fail && !(rc = sub_dir(&llo, &llo, &El, t->w, 0)) && !(rc = dyad_add(&lhi, &lhi, &El, t->w, 1))) {
                    int unbounded;
                    rc = lean_ends(&llo, &lhi, &unbounded, &t->add, t->w);
                    if (unbounded) fail = 1;
                }
            }
            dyad_free(&Zlo);
            dyad_free(&Zhi);
            dyad_free(&lam);
            dyad_free(&spread);
            dyad_free(&El);
        }
        /* Lemma 3 per element, and for logsumexp */
        if (!rc && !fail) {
            bigint tb, lb, hb, ref;
            big_init(&tb);
            big_init(&lb);
            big_init(&hb);
            big_init(&ref);
            for (size_t c = 0; !rc && c < Ev; c++) {
                /* a column with a value with no bound, or whose reference is not finite, has none (with uncertain inputs any value
                   or flag); NaN where Masked keys allows it */
                int fl = 0, ok = !cunb[c] && !cnone[c], nan = nan_row || cnan[c] ? FLAG_NAN : 0;
                if (!(rc = bytes_load(&ref, out_refs + c * a->out_size, a->out_size, 0)) &&
                    !(rc = final_bounds(&tb, &lb, &hb, &fl, &a->out, &t->f64, a->rnd, a->ovf, ok ? &los[c] : NULL, ok ? &his[c] : NULL, &ref, t->w,
                                        (ok || !a->uncertain ? 0 : FLAG_ANY) | nan))) {
                    bytes_store(tol + 8 * c, 8, &tb);
                    bytes_store(lows + c * a->out_size, a->out_size, &lb);
                    bytes_store(highs + c * a->out_size, a->out_size, &hb);
                    flags[c] = (unsigned char)fl;
                }
            }
            if (!rc) {
                int fl = 0;
                if (!(rc = bytes_load(&ref, lse_ref, a->lse_size, 0)) &&
                    !(rc = final_bounds(&tb, &lb, &hb, &fl, &a->lse, &t->f64, a->lrnd, a->lovf, lse_none ? NULL : &llo, lse_none ? NULL : &lhi, &ref,
                                        t->w, nan_row ? FLAG_NAN : 0))) {
                    bytes_store(lse_tol, 8, &tb);
                    bytes_store(lse_low, a->lse_size, &lb);
                    bytes_store(lse_high, a->lse_size, &hb);
                    *lse_flag = (unsigned char)fl;
                    *status = 0;
                }
            }
            big_free(&tb);
            big_free(&lb);
            big_free(&hb);
            big_free(&ref);
        }
        dyad_free(&llo);
        dyad_free(&lhi);
        if (los) {
            for (size_t c = 0; c < 2 * Ev; c++) {
                dyad_free(&los[c]);
                dyad_free(&his[c]);
            }
        }
        free(los);
        free(his);
        dyad_free(&inverse);
        dyad_free(&divisor);
        for (int z = 0; z < 2; z++) {
            dyad_free(&hf[z]);
            dyad_free(&kf[z]);
        }
        dyad_free(&pi);
        dyad_free(&cfac);
        dyad_free(&fa);
        dyad_free(&fb);
        dyad_free(&hmax);
    }
done:
    att_soft_free(&sf);
    ivl_free(&one_iv);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    free(mcol);
    free(rmaxc);
    free(cunb);
    free(cnone);
    free(cnan);
    free(pint);
    free(plo);
    free(eint);
    free(prod);
    if (rc == EXACTREF_OVERFLOW) { /* an exponent beyond the dyadics' range: the row is left to the caller */
        *status = 1;
        rc = 0;
    }
    return rc;
}

/* ---------------------------------------------------------------- the step-by-step program
   (proofs/scaled_dot_product_attention.md, The step-by-step program and Fully padded rows;
   attention_algorithms.md): each boundary rounds by Lemma 1 on intervals of dyadics at the bounds'
   precision w. The weights' value sums are integer products at a unit 126 bits below the largest
   weight end, the ends rounded outward, or exact sums where that unit may matter. */

#define OC(name) (&g->o[ATT_O_##name])

void att_prog_init(att_prog *g) {
    memset(g, 0, sizeof *g);
    for (int i = 0; i < ATT_P_LEANS; i++) lean_init(&g->lean[i]);
    for (int i = 0; i < ATT_O_CONSTS; i++) dyad_init(&g->o[i]);
    dyad_init(&g->ln2_lo);
    dyad_init(&g->ln2_hi);
    dyad_init(&g->log2e_lo);
    dyad_init(&g->log2e_hi);
}

void att_prog_free(att_prog *g) {
    for (int i = 0; i < ATT_P_LEANS; i++) lean_free(&g->lean[i]);
    for (int i = 0; i < ATT_A_KINDS; i++) {
        att_acc *c = &g->acc[i];
        if (c->terms) {
            for (size_t z = 0; z < 3 * (c->nmax + 1); z++) dyad_free(&c->terms[z]);
            free(c->terms);
        }
        free(c->has);
    }
    for (int i = 0; i < ATT_O_CONSTS; i++) dyad_free(&g->o[i]);
    dyad_free(&g->ln2_lo);
    dyad_free(&g->ln2_hi);
    dyad_free(&g->log2e_lo);
    dyad_free(&g->log2e_hi);
    sum_bounds_free(&g->scores);
}

/* ln 2 and log2(e) at w (_problem.log2e), and the scores' any-order bound for the padded rows */
int att_prog_prepare(att_prog *g, const floatfmt *scores, int64_t E) {
    int64_t w = g->w;
    dyad one;
    dyad_init(&one);
    int rc = ln2_bounds(&g->ln2_lo, &g->ln2_hi, w);
    if (!rc) rc = dyad_set_si(&one, 1, 0);
    if (!rc && !(rc = dyad_div(&g->log2e_lo, &one, &g->ln2_hi, w, 0))) rc = dyad_div(&g->log2e_hi, &one, &g->ln2_lo, w, 1);
    if (!rc && (g->methods & ATT_M_PADDED)) rc = sum_bounds_init(&g->scores, scores, g->flush, E, E > 0 ? E - 1 : 0, 0, w);
    dyad_free(&one);
    return rc;
}

/* the accumulation of n terms of magnitudes adding up to at most mass */
int acc_apply(ivl *x, const att_acc *c, size_t n, const dyad *mass, int64_t w, int *unb) {
    if (c->kind == ATT_ACC_NONE) return 0;
    if (n > c->nmax || !c->has[n]) {
        *unb = 1;
        return 0;
    }
    const dyad *t = c->terms + 3 * n;
    dyad err, s;
    dyad_init(&err);
    dyad_init(&s);
    int rc = dyad_mul(&err, &t[0], mass, w, 1);
    if (!rc) rc = dyad_add(&err, &err, &t[1], w, 1);
    if (!rc) rc = dyad_add(&s, mass, &err, w, 1);
    if (!rc && dyad_cmp(&s, &t[2]) > 0) *unb = 1;
    if (!rc && !*unb) rc = widen_by(x, &err, w);
    dyad_free(&err);
    dyad_free(&s);
    return rc;
}

/* a key's score: the exact sum's accumulation, then its rounding */
int prog_score(ivl *s, int *unb, const att_prog *g, const product_sums *ps, int64_t E) {
    dyad x, mass;
    dyad_init(&x);
    dyad_init(&mass);
    int64_t w = g->w;
    /* the exact sum, within the magnitudes a flush may remove (moved) */
    bigint t;
    big_init(&t);
    int rc = big_copy(&t, &ps->moved);
    if (t.n) t.neg = !t.neg;
    if (!rc) rc = big_add(&t, &ps->s, &t);
    if (!rc && !(rc = dyad_from_big(&x, &t, ps->e))) rc = dyad_rnd(&s->lo, &x, w, 0);
    if (!rc && !(rc = big_add(&t, &ps->s, &ps->moved)) && !(rc = dyad_from_big(&x, &t, ps->e))) rc = dyad_rnd(&s->hi, &x, w, 1);
    big_free(&t);
    if (!rc && !(rc = dyad_from_big(&x, &ps->mass, ps->e))) rc = dyad_rnd(&mass, &x, w, 1);
    if (!rc) rc = acc_apply(s, &g->acc[ATT_A_SCORES], (size_t)E, &mass, w, unb);
    if (!rc && !*unb) rc = lean_ivl(s, &g->lean[ATT_P_SCORES_OUT], w, unb);
    dyad_free(&x);
    dyad_free(&mass);
    return rc;
}

/* Lemma 1 at a boundary on x (lean_ivl), for a kept key's value (attention_algorithms.md, Keys that may vanish): where a
   point may overflow and x lies below 0, with vanish given, the result is -inf or finite, so *vanish is set and x becomes
   the finite outcomes [-largest, U], U the larger of -largest and the largest value within c ulps of a point of x
   (x.hi + c ulp(x.hi), with 2**emin where a flush may give 0, and -2**k + c ulp(x.lo) where x reaches 2**k, the power of 2
   just above |x.hi|), rounded down to the grid; otherwise *unb is set where a point may overflow. Where the rounding has
   no finite result (No finite result: to nearest with x.hi at most -threshold, or c < 1 and that bound below
   -largest), *vanish is 2, the key surely vanishes, and x is [-largest, -largest]; a 2 from an earlier rounding stays. */
int lean_vanish(ivl *x, int *unb, unsigned char *vanish, const round_bound *b, int64_t w) {
    int u = 0, rc = lean_ends(&x->lo, &x->hi, &u, b, w);
    if (rc || !u) return rc;
    if (!vanish || dyad_sign(&x->hi) >= 0) {
        *unb = 1;
        return 0;
    }
    if (!*vanish) *vanish = 1;
    dyad up, r, t;
    dyad_init(&up);
    dyad_init(&r);
    dyad_init(&t);
    /* to nearest, every point at or past the threshold rounds to -inf */
    int sure = 0;
    if (b->has_threshold && !(rc = dyad_neg(&t, &b->threshold))) sure = dyad_cmp(&x->hi, &t) <= 0;
    int64_t top = top_bit(&x->hi), at = top > b->emin ? top : b->emin;
    if (!rc && !(rc = dyad_copy(&r, &b->ulps)) && !(rc = dyad_shift(&r, at - b->mantissa)) && !(rc = dyad_add(&up, &x->hi, &r, w, 1)) && b->flush &&
        top < b->emin && !(rc = dyad_set_si(&t, 1, b->emin)))
        rc = dyad_add(&up, &up, &t, w, 1);
    if (!rc && !(rc = dyad_set_si(&t, -1, top + 1)) && dyad_cmp(&x->lo, &t) <= 0) {
        int64_t top2 = top_bit(&x->lo), at2 = top2 > b->emin ? top2 : b->emin;
        if (!(rc = dyad_copy(&r, &b->ulps)) && !(rc = dyad_shift(&r, at2 - b->mantissa)) && !(rc = dyad_add(&t, &t, &r, w, 1)) && dyad_cmp(&t, &up) > 0)
            rc = dyad_copy(&up, &t);
    }
    if (!rc && !(rc = grid_end(&up, b->emin, b->mantissa, 0)) && !(rc = dyad_neg(&x->lo, &b->largest))) {
        /* no value within c ulps of a point of x, and with c < 1 no directed rounding to -largest */
        if (dyad_cmp(&up, &x->lo) < 0 && top_bit(&b->ulps) < 0) sure = 1;
        if (sure || dyad_cmp(&up, &x->lo) < 0) {
            rc = dyad_copy(&x->hi, &x->lo);
        } else {
            rc = dyad_copy(&x->hi, dyad_cmp(&up, &b->largest) > 0 ? &b->largest : &up);
        }
    }
    if (!rc && sure) *vanish = 2;
    dyad_free(&up);
    dyad_free(&r);
    dyad_free(&t);
    return rc;
}

/* the kept keys whose weight can be nonzero: those that do not surely vanish (lean_vanish's 2); a key that does adds an
   exact 0 to each sum, which leaves Lemma 2 at the other keys' count (scaled_dot_product_attention.md, Masked keys) */
static size_t live_keys(const unsigned char *vanish, size_t n) {
    size_t live = 0;
    for (size_t j = 0; j < n; j++) live += vanish[j] != 2;
    return live;
}

/* fma at logits: score times the factor plus the mask's value (bias NULL for none), rounded once; vanish as lean_vanish
   (NULL: an overflow sets *unb) */
int prog_logit(ivl *x, int *unb, unsigned char *vanish, const att_prog *g, const ivl *score, const ivl *bias) {
    ivl f;
    f.lo = *OC(FACTOR_LO); /* read-only views */
    f.hi = *OC(FACTOR_HI);
    int rc = ivl_mul(x, score, &f, g->w);
    if (!rc && bias) rc = ivl_add(x, x, bias, g->w);
    if (!rc) rc = lean_vanish(x, unb, vanish, &g->lean[ATT_P_LOGITS], g->w);
    return rc;
}

/* a kept key's float-mask value as the implementation adds it; *vanish where a rounding may overflow toward -inf
   (lean_vanish) */
int prog_bias(ivl *b, int *unb, unsigned char *vanish, const att_prog *g, const dyad *value, const dyad *radius) {
    int rc = ivl_point(b, value);
    /* with flush, a value below 2**emin may be read as zero: the hull of it and 0 */
    if (!rc && g->flush && dyad_sign(value) && top_bit(value) < g->emin) rc = dyad_set_si(dyad_sign(value) < 0 ? &b->hi : &b->lo, 0, 0);
    /* an earlier result's mask value lies within its radius of the stored one */
    if (!rc && radius && dyad_sign(radius)) rc = widen_by(b, radius, g->w);
    if (!rc && (g->methods & ATT_M_METHODS)) {
        if (g->lean[ATT_P_BIAS].active) rc = lean_vanish(b, unb, vanish, &g->lean[ATT_P_BIAS], g->w);
        if (!rc && (g->methods & ATT_M_EXP2)) {
            ivl l;
            l.lo = g->log2e_lo;
            l.hi = g->log2e_hi;
            if (!(rc = ivl_mul(b, b, &l, g->w))) rc = lean_vanish(b, unb, vanish, &g->lean[ATT_P_EXP2_BIAS], g->w);
        }
    }
    return rc;
}

/* acc (nw words, two's complement) += a b for a two's-complement 128-bit a */
static void acc_s128(uint64_t *acc, size_t nw, w128 a, int64_t b) {
    int na = (int)(w128_hi(a) >> 63);
    if (na) a = w128_sub(w128_zero(), a);
    if (!b || (!w128_lo(a) && !w128_hi(a))) return;
    uint64_t mb = b < 0 ? (uint64_t)0 - (uint64_t)b : (uint64_t)b, h0, l0 = word_mul(&h0, w128_lo(a), mb), h1, l1 = word_mul(&h1, w128_hi(a), mb);
    uint64_t x[3] = {l0, h0 + l1, h1 + (h0 + l1 < h0)}, ext = 0;
    if (na != (b < 0)) { /* subtract: add the two's complement */
        uint64_t c = 1;
        for (int i = 0; i < 3; i++) {
            x[i] = ~x[i] + c;
            c = c && !x[i];
        }
        ext = ~(uint64_t)0;
    }
    uint64_t carry = 0;
    for (size_t i = 0; i < nw; i++) {
        uint64_t y = i < 3 ? x[i] : ext, t = acc[i] + y, c = t < y;
        t += carry;
        c += t < carry;
        acc[i] = t;
        carry = c;
    }
}

/* acc (nw words) += a b for nonnegative 128-bit a and b */
static void acc_u128(uint64_t *acc, size_t nw, w128 a, w128 b) {
    acc_wide(acc, nw, w128_lo(a), b);
    if (w128_hi(a)) acc_wide(acc + 1, nw - 1, w128_hi(a), b);
}

/* floor (up: ceil) of x / 2**u as a two's-complement 128-bit integer; |x| < 2**(u + 127) */
static int to_w128(w128 *r, const dyad *x, int64_t u, int up) {
    dyad t;
    bigint b;
    dyad_init(&t);
    big_init(&b);
    int rc = dyad_copy(&t, x);
    if (!rc && !(rc = dyad_shift(&t, -u))) {
        rc = dyad_to_fixed(&b, &t, 0, up);
    }
    if (!rc) {
        w128 m = w128_make(b.n > 0 ? b.w[0] : 0, b.n > 1 ? b.w[1] : 0);
        *r = b.neg ? w128_sub(w128_zero(), m) : m;
    }
    dyad_free(&t);
    big_free(&b);
    return rc;
}

#define ATT_QW 6 /* words of a column's sums of the program's two-word weights times values */

/* a dyad from nw <= ATT_QW words of two's complement at exponent e */
static int dyad_signed_words(dyad *r, const uint64_t *acc, size_t nw, int64_t e) {
    uint64_t m[ATT_QW];
    int neg = (int)(acc[nw - 1] >> 63);
    uint64_t carry = 1;
    for (size_t i = 0; i < nw; i++) {
        m[i] = neg ? ~acc[i] + carry : acc[i];
        if (neg) carry = carry && !m[i];
    }
    int rc = dyad_words(r, m, nw, e);
    if (!rc && r->m.n) r->m.neg = neg;
    return rc;
}

/* Online softmax's sum bound over `parts` passes (one, or split_kv's splits): the accumulation of n + 1 terms
   (the other passes' constants added), or Lemma T's for mma value sums, then the rescalings' absolute
   part parts (count largest + eta) K rescale */
static int online_bound(ivl *x, int *unb, const att_prog *g, int acc, const dyad *mass, const dyad *largest, int64_t parts) {
    int64_t w = g->w;
    size_t on = (size_t)g->online_n + 1;
    dyad err, d1, cnt;
    dyad_init(&err);
    dyad_init(&d1);
    dyad_init(&cnt);
    int rc = dyad_set_si(&cnt, g->online_S, 0), mma = acc == ATT_A_VALUES && (g->methods & ATT_M_MMA_VALUES);
    if (!rc && mma) {
        if (!(rc = dyad_mul(&err, OC(MMA_G), mass, w, 1)) && !(rc = dyad_times(&d1, OC(MMA_H), parts, w)) && !(rc = dyad_add(&err, &err, &d1, w, 1)) &&
            !(rc = dyad_add(&d1, mass, &err, w, 1))) {
            if (dyad_cmp(&d1, OC(MMA_TOP)) > 0)
                *unb = 1;
            else
                rc = widen_by(x, &err, w);
        }
        /* partial = (count largest + H) k_mma; widened by parts rescale_mma partial */
        if (!rc && !*unb && !(rc = dyad_mul(&d1, &cnt, largest, w, 1)) && !(rc = dyad_add(&d1, &d1, OC(MMA_H), w, 1)) &&
            !(rc = dyad_mul(&d1, &d1, OC(MMA_K), w, 1)) && !(rc = dyad_mul(&d1, OC(MMA_RESCALE), &d1, w, 1)) && !(rc = dyad_times(&d1, &d1, parts, w)))
            rc = widen_by(x, &d1, w);
    } else if (!rc) {
        rc = acc_apply(x, &g->acc[acc], on, mass, w, unb);
        if (!rc && !*unb && parts > 1 && !(rc = dyad_times(&d1, &g->acc[acc].terms[3 * on + 1], parts - 1, w))) rc = widen_by(x, &d1, w);
        if (!rc && !*unb && !(rc = dyad_mul(&d1, &cnt, largest, w, 1)) && !(rc = dyad_add(&d1, &d1, OC(ETA), w, 1)) &&
            !(rc = dyad_mul(&d1, &d1, OC(K), w, 1)) && !(rc = dyad_mul(&d1, OC(RESCALE), &d1, w, 1)) && !(rc = dyad_times(&d1, &d1, parts, w)))
            rc = widen_by(x, &d1, w);
    }
    dyad_free(&err);
    dyad_free(&d1);
    dyad_free(&cnt);
    return rc;
}

/* Online softmax's one pass: (m, error) for count terms of magnitude at most largest before their
   rescaling factors, m at most own where given (the pass's own bound: split_merge); *unb where the pass's sum may
   overflow its accumulator */
static int pass_error(dyad *m, dyad *error, int *unb, const att_prog *g, int acc, int64_t count, const dyad *largest, const dyad *own) {
    int64_t w = g->w;
    size_t on = (size_t)g->online_n + 1;
    const att_acc *c = &g->acc[acc];
    dyad sums, partial, d1;
    dyad_init(&sums);
    dyad_init(&partial);
    dyad_init(&d1);
    int rc = dyad_times(m, largest, count, w);
    if (!rc && own && dyad_cmp(own, m) < 0) rc = dyad_copy(m, own);
    const dyad *limit = NULL, *rescale = NULL;
    if (!rc && acc == ATT_A_VALUES && (g->methods & ATT_M_MMA_VALUES)) {
        /* sums = G m + H, partial = (m + H) k_mma */
        if (!(rc = dyad_mul(&sums, OC(MMA_G), m, w, 1)) && !(rc = dyad_add(&sums, &sums, OC(MMA_H), w, 1)) && !(rc = dyad_add(&partial, m, OC(MMA_H), w, 1)))
            rc = dyad_mul(&partial, &partial, OC(MMA_K), w, 1);
        limit = OC(MMA_TOP);
        rescale = OC(MMA_RESCALE);
    } else if (!rc) {
        if (c->kind != ATT_ACC_TERMS || on > c->nmax || !c->has[on]) {
            *unb = 1;
        } else if (!(rc = dyad_mul(&sums, &c->terms[3 * on], m, w, 1)) && !(rc = dyad_add(&sums, &sums, &c->terms[3 * on + 1], w, 1)) &&
                   !(rc = dyad_add(&partial, m, OC(ETA), w, 1))) {
            rc = dyad_mul(&partial, &partial, OC(K), w, 1);
        }
        limit = c->kind == ATT_ACC_TERMS && on <= c->nmax ? &c->terms[3 * on + 2] : NULL;
        rescale = OC(RESCALE);
    }
    if (!rc && !*unb && !(rc = dyad_mul(&d1, rescale, &partial, w, 1)) && !(rc = dyad_add(error, &sums, &d1, w, 1)) && !(rc = dyad_add(&d1, m, error, w, 1)) &&
        dyad_cmp(&d1, limit) > 0)
        *unb = 1;
    dyad_free(&sums);
    dyad_free(&partial);
    dyad_free(&d1);
    return rc;
}

/* the fully padded rows' bound (Fully padded rows) from the kept keys' logits before their rounding
   (ys), each column's least and largest value (lowest, highest at V's exponent): into pad[c], or *none. A key that may
   vanish (vanish, att_prog_softmax) is never the key whose logit is finite (attention_algorithms.md, Keys that may vanish). */
static int padded_row(ivl *pad, int *none, const att_prog *g, const att_call *a, const ivl *ys, const unsigned char *vanish, size_t n, const dyad *vlo,
                      const dyad *vhi) {
    int64_t w = g->w;
    size_t Ev = a->Ev;
    int finite = 0, rc = 0;
    *none = 1;
    for (size_t j = 0; j < n; j++) {
        if (dyad_cmp(&ys[j].hi, OC(THETA_LOGITS)) >= 0) return 0; /* the logit may be +inf */
        dyad m;
        dyad_init(&m);
        if (!vanish[j] && !(rc = ends_magnitude(&m, &ys[j].lo, &ys[j].hi)) && dyad_cmp(&m, OC(THETA_LOGITS)) < 0) finite = 1;
        dyad_free(&m);
        if (rc) return rc;
    }
    if (!finite) return 0;
    /* the sums' terms that can be nonzero: a key that surely vanishes adds an exact 0 */
    size_t live = live_keys(vanish, n);
    int online = g->methods & ATT_M_ONLINE, unb = 0;
    dyad gmin, wmax, big, EZ, d1, d2, d3, one, divisor;
    dyad *ds[] = {&gmin, &wmax, &big, &EZ, &d1, &d2, &d3, &one, &divisor};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    rc = dyad_set_si(&one, 1, 0);
    if (!rc && !online) {
        /* the maximum's weight is exp(0) within its ulps */
        if (!(rc = dyad_add(&d1, OC(EXP_R), OC(EXP_A), w, 1)) && !(rc = sub_dir(&gmin, &one, &d1, w, 0))) rc = dyad_add(&wmax, &one, &d1, w, 1);
    } else if (!rc) {
        if (!(rc = dyad_copy(&gmin, OC(G_MIN)))) rc = dyad_copy(&wmax, OC(GROWTH));
    }
    if (rc || dyad_sign(&gmin) <= 0) goto out;
    const att_acc *cz = &g->acc[ATT_A_SUM], *cv = &g->acc[ATT_A_VALUES];
    size_t on = (size_t)g->online_n + 1;
    if ((rc = dyad_times(&big, &wmax, (int64_t)live, w))) goto out;
    if (!online) {
        /* E_Z = growth(r_z, n - 1) ((n - 1) r_z + over((n - 1) a2_z) + (narrow: r_z + over(n a_z))) */
        if ((rc = dyad_times(&d1, OC(Z_R), (int64_t)live - 1, w)) || (rc = dyad_times(&d2, OC(Z_A2), (int64_t)live - 1, w)) || (rc = dyad_div(&d2, &d2, &gmin, w, 1)) ||
            (rc = dyad_add(&d1, &d1, &d2, w, 1)))
            goto out;
        if (g->methods & ATT_M_NARROW) {
            if ((rc = dyad_times(&d2, OC(Z_A), (int64_t)live, w)) || (rc = dyad_div(&d2, &d2, &gmin, w, 1)) || (rc = dyad_add(&d2, OC(Z_R), &d2, w, 1))) goto out;
        } else if ((rc = dyad_set_si(&d2, 0, 0))) {
            goto out;
        }
        if ((rc = dyad_add(&d1, &d1, &d2, w, 1)) || (rc = growth_up(&d2, OC(Z_R), (int64_t)live - 1, w)) || (rc = dyad_mul(&EZ, &d2, &d1, w, 1))) goto out;
    } else {
        /* Lemma 2 over the online sums' depth: E_Z = gamma_z + over(constant_z wmax) */
        if (cz->kind != ATT_ACC_TERMS || cv->kind != ATT_ACC_TERMS || on > cz->nmax || on > cv->nmax || !cz->has[on] || !cv->has[on]) goto out;
        if ((rc = dyad_mul(&d1, &cz->terms[3 * on + 1], &wmax, w, 1)) || (rc = dyad_div(&d1, &d1, &gmin, w, 1)) ||
            (rc = dyad_add(&EZ, &cz->terms[3 * on], &d1, w, 1)))
            goto out;
    }
    if (dyad_cmp(&EZ, &one) >= 0) goto out;
    if ((rc = dyad_add(&d1, &one, &EZ, w, 1)) || (rc = dyad_mul(&d1, &big, &d1, w, 1))) goto out;
    if (dyad_cmp(&d1, OC(Z_THETA)) >= 0) goto out;
    if ((rc = sub_dir(&divisor, &one, &EZ, w, 0))) goto out;
    dyad gv, nvr, avs, vmax, eN, size, err, ER, neg, lo_d, hi_d;
    dyad *vs[] = {&gv, &nvr, &avs, &vmax, &eN, &size, &err, &ER, &neg, &lo_d, &hi_d};
    for (size_t z = 0; z < sizeof vs / sizeof *vs; z++) dyad_init(vs[z]);
    if (!online) {
        /* growth(r_v, n - 1), n r_v and over(n a_v + (n - 1) a2_v), shared by the columns */
        if (!(rc = growth_up(&gv, OC(V_R), (int64_t)live - 1, w)) && !(rc = dyad_times(&nvr, OC(V_R), (int64_t)live, w)) &&
            !(rc = dyad_times(&d1, OC(V_A), (int64_t)live, w)) && !(rc = dyad_times(&d2, OC(V_A2), (int64_t)live - 1, w)) && !(rc = dyad_add(&d1, &d1, &d2, w, 1)))
            rc = dyad_div(&avs, &d1, &gmin, w, 1);
    } else if (!(rc = dyad_mul(&d1, &cv->terms[3 * on + 1], &wmax, w, 1))) {
        rc = dyad_div(&avs, &d1, &gmin, w, 1);
    }
    for (size_t c = 0; !rc && !unb && c < Ev; c++) {
        /* the column's least and largest value (with V's radii, v - r and v + r); with flush, a value may be read as zero */
        if ((rc = dyad_copy(&lo_d, &vlo[c])) || (rc = dyad_copy(&hi_d, &vhi[c]))) break;
        if (g->flush && dyad_sign(&lo_d) > 0 && (rc = dyad_set_si(&lo_d, 0, 0))) break;
        if (g->flush && dyad_sign(&hi_d) < 0 && (rc = dyad_set_si(&hi_d, 0, 0))) break;
        if ((rc = ends_magnitude(&vmax, &lo_d, &hi_d)) || (rc = dyad_abs(&d1, &hi_d))) break;
        if (dyad_cmp(&d1, &vmax) > 0 && (rc = dyad_copy(&vmax, &d1))) break;
        if (!online) {
            /* e_N = growth(r_v, n - 1) (n r_v vmax + over(n a_v + (n - 1) a2_v)) */
            if ((rc = dyad_mul(&d1, &nvr, &vmax, w, 1)) || (rc = dyad_add(&d1, &d1, &avs, w, 1)) || (rc = dyad_mul(&eN, &gv, &d1, w, 1))) break;
        } else if ((rc = dyad_mul(&d1, &cv->terms[3 * on], &vmax, w, 1)) || (rc = dyad_add(&eN, &d1, &avs, w, 1))) {
            break;
        }
        /* the value sum's magnitude, below theta_v; its rounding to F */
        if ((rc = dyad_add(&d1, &vmax, &eN, w, 1)) || (rc = dyad_mul(&size, &big, &d1, w, 1))) break;
        if (dyad_cmp(&size, OC(V_THETA)) >= 0) {
            unb = 1;
            break;
        }
        if ((rc = dyad_neg(&neg, &size)) || (rc = lean_error(&err, &neg, &size, &g->lean[ATT_P_VALUES_OUT], &unb)) || unb) break;
        /* E_N = e_N + over(err); E_R = (E_N + vmax E_Z) / (1 - E_Z) */
        if ((rc = dyad_div(&d1, &err, &gmin, w, 1)) || (rc = dyad_add(&d1, &eN, &d1, w, 1)) || (rc = dyad_mul(&d2, &vmax, &EZ, w, 1)) ||
            (rc = dyad_add(&d1, &d1, &d2, w, 1)) || (rc = dyad_div(&ER, &d1, &divisor, w, 1)))
            break;
        if ((rc = sub_dir(&pad[c].lo, &lo_d, &ER, w, 0)) || (rc = dyad_add(&pad[c].hi, &hi_d, &ER, w, 1)))
            break;
        rc = lean_ivl(&pad[c], &g->lean[ATT_P_NORMALIZE], w, &unb);
    }
    if (!rc && !unb) *none = 0;
    for (size_t z = 0; z < sizeof vs / sizeof *vs; z++) dyad_free(vs[z]);
out:
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

static ivl *ivls_new(size_t n) {
    ivl *x = malloc((n ? n : 1) * sizeof *x);
    if (x)
        for (size_t j = 0; j < n; j++) ivl_init(&x[j]);
    return x;
}

static void ivls_free(ivl *x, size_t n) {
    if (x)
        for (size_t j = 0; j < n; j++) ivl_free(&x[j]);
    free(x);
}

/* a kept key's logit lower end and its relative part's ends, for a split's own bound in any order of its keys (split_own);
   vanish: the key may vanish, its lower end then -inf */
typedef struct {
    const dyad *lo, *wlo, *hi;
    int vanish;
} own_key;

static int own_key_cmp(const void *x, const void *y) {
    const own_key *a = x, *b = y;
    if (a->vanish || b->vanish) return b->vanish - a->vanish;
    return dyad_cmp(a->lo, b->lo);
}

/* split_kv: each of the np splits' own bound (attention_algorithms.md, split_kv) into own[i], with own_ok[i] where it is below
   the split's key count, from the row's logits L, maximum top and the weights' relative parts R (att_prog_softmax: the weight
   relative to the row's maximum with its later rescaling factors, before own and before methods' weights rounds it). In
   any order of a split's keys and blocks, after a key its pass holds its sums relative to its running maximum, at least M,
   the largest lower end of the keys so far, and each of those keys has L.lo <= M. So they total at most e**(top.lo - M)
   times the sum of their relative parts' upper ends plus their count times own, which the pass adds in its own units,
   and at most their count, each held term being at most 1 relative to the running maximum (growth apart); own[i] is the
   largest over the split's distinct lower ends M. e**(top.lo - M) is at most 1 / R.lo of a key at M, R.lo being below
   e**(L.lo - top.hi). R.lo is 0 only where exp's clamp leaves the key 2**40 or more below the maximum (exp_clamped;
   Lemma D's tables give positive ends) or for a key that may vanish (vanish), whose lower end is -inf (Keys that may
   vanish: it comes first, in every count), and there the count bounds. With weights, a held weight is at most 1 + rel_W
   times its weight plus abs_W growth (Rounded weights), so own[i] takes 1 + rel_W times that bound plus its key count
   times abs_W growth. */
static int split_own(dyad *own, unsigned char *own_ok, const att_prog *g, const att_call *a, const size_t *keys, size_t n, const ivl *L,
                     const ivl *R, const unsigned char *vanish, size_t np) {
    int64_t w = g->w, size = g->split;
    size_t S = a->S;
    int weights = g->methods & ATT_M_WEIGHTS, rc = 0;
    own_key *ord = malloc((n ? n : 1) * sizeof *ord);
    dyad mass, d1, d2, e1, e2, rw;
    dyad *ds[] = {&mass, &d1, &d2, &e1, &e2, &rw};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    if (!ord) rc = EXACTREF_NOMEM;
    if (!rc && weights && !(rc = dyad_mul(&e1, OC(ABS_W), OC(GROWTH), w, 1)) && !(rc = dyad_set_si(&rw, 1, 0))) rc = dyad_add(&rw, &rw, OC(REL_W), w, 1);
    size_t z0 = 0;
    for (size_t i = 0; !rc && i < np; i++) {
        size_t first = z0, idx = keys[z0] / (size_t)size;
        while (z0 < n && keys[z0] / (size_t)size == idx) z0++;
        size_t end = (idx + 1) * (size_t)size < S ? (idx + 1) * (size_t)size : S, cnt = z0 - first;
        own_ok[i] = 0;
        for (size_t z = 0; z < cnt; z++) {
            ord[z].lo = &L[first + z].lo;
            ord[z].wlo = &R[first + z].lo;
            ord[z].hi = &R[first + z].hi;
            ord[z].vanish = vanish[first + z] != 0;
        }
        qsort(ord, cnt, sizeof *ord, own_key_cmp);
        if ((rc = dyad_set_si(&mass, 0, 0)) || (rc = dyad_set_si(&own[i], 0, 0))) break;
        for (size_t z = 1; !rc && z <= cnt; z++) {
            rc = dyad_add(&mass, &mass, ord[z - 1].hi, w, 1);
            if (rc || (z < cnt && own_key_cmp(&ord[z], &ord[z - 1]) <= 0)) continue;
            /* every key with L.lo <= ord[z - 1].lo counted: R(M) at that lower end */
            int bounded = dyad_sign(ord[z - 1].wlo) > 0;
            if ((rc = dyad_set_si(&e2, (int64_t)z, 0)) ||
                (bounded && ((rc = dyad_div(&d1, &mass, ord[z - 1].wlo, w, 1)) || (rc = dyad_times(&d2, OC(OWN), (int64_t)z, w)) ||
                             (rc = dyad_add(&d1, &d1, &d2, w, 1)))))
                break;
            if ((!bounded || dyad_cmp(&d1, &e2) > 0) && (rc = dyad_copy(&d1, &e2))) break; /* at most the count */
            if (dyad_cmp(&d1, &own[i]) > 0) rc = dyad_copy(&own[i], &d1);
        }
        if (rc) break;
        if (weights && ((rc = dyad_mul(&own[i], &own[i], &rw, w, 1)) || (rc = dyad_times(&d2, &e1, (int64_t)cnt, w)) || (rc = dyad_add(&own[i], &own[i], &d2, w, 1))))
            break;
        if (!(rc = dyad_set_si(&d1, (int64_t)(end - idx * (size_t)size), 0))) own_ok[i] = dyad_cmp(&own[i], &d1) < 0;
    }
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    free(ord);
    return rc;
}

/* split_kv's merge (attention_algorithms.md, split_kv): the row's logsumexp into *lse and each column's output into
   out[c] (on entry each split pass's bound on the value sum, x, with sizes[c] bounding its terms' magnitudes, largest[c] a
   term before its rescaling factors and [vlo[c], vhi[c]] the kept keys' values; values_ok 0 where they have no bound),
   from the row's logits L (vanish: the keys that may vanish), maximum top and row sum total over P splits. *lse_ok and
   *row_ok say which exist. */
static int split_merge(ivl *out, int *row_ok, ivl *lse, int *lse_ok, unsigned char *szero, const att_prog *g, const att_call *a, const size_t *keys,
                       size_t n, const ivl *L, const unsigned char *vanish, const ivl *top, const ivl *total, const dyad *sizes, const dyad *largest,
                       const dyad *vlo, const dyad *vhi, const dyad *own, const unsigned char *own_ok, int64_t P, int values_ok) {
    int64_t w = g->w, size = g->split;
    size_t np = (size_t)P, S = a->S, Ev = a->Ev;
    int exp2 = g->methods & ATT_M_EXP2, unb = 0, rc = 0;
    *row_ok = *lse_ok = 0;
    memset(szero, 1, np);
    ivl *lses = ivls_new(np), mu, lg, held, d, sigma, topl, crude, zhat, ln2;
    ln2.lo = g->ln2_lo; /* read-only views */
    ln2.hi = g->ln2_hi;
    dyad *gam = malloc(np * sizeof *gam), *eta = malloc(np * sizeof *eta), *share = malloc(np * sizeof *share), *spread = malloc(np * sizeof *spread),
         *phi_out = malloc(np * sizeof *phi_out), *phi_hi = malloc(np * sizeof *phi_hi), *phi_lo = malloc(np * sizeof *phi_lo),
         *radius = malloc(np * sizeof *radius);
    dyad *arrays[] = {gam, eta, share, spread, phi_out, phi_hi, phi_lo, radius};
    ivl *ivs[] = {&mu, &lg, &held, &d, &sigma, &topl, &crude, &zhat};
    for (size_t z = 0; z < sizeof ivs / sizeof *ivs; z++) ivl_init(ivs[z]);
    dyad m, err, zlo, gamma, mass, d1, d2, d3, slack, elo, ehi, flo, fhi, one, rho_hi, rho_lo, phi_c, rmore, pmore, inv_low, recip, big_phi, ez, floor_z,
        weighed;
    dyad *ds[] = {&m, &err, &zlo, &gamma, &mass, &d1, &d2, &d3, &slack, &elo, &ehi, &flo, &fhi, &one, &rho_hi, &rho_lo, &phi_c, &rmore, &pmore,
                  &inv_low, &recip, &big_phi, &ez, &floor_z, &weighed};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int ready = lses && gam && eta && share && spread && phi_out && phi_hi && phi_lo && radius;
    if (ready)
        for (size_t z = 0; z < sizeof arrays / sizeof *arrays; z++)
            for (size_t i = 0; i < np; i++) dyad_init(&arrays[z][i]);
    if (!ready) {
        rc = EXACTREF_NOMEM;
        goto out;
    }
    if ((rc = dyad_set_si(&one, 1, 0))) goto out;
    /* E_Z = gamma_Z + const_Z growth / g_min at one split's online count, and floor_z = g_min (1 - E_Z) where that is
       positive: each split's row sum, and Z^, are at least floor_z (Each split's logsumexp; The logsumexp) */
    const att_acc *cz = &g->acc[ATT_A_SUM];
    size_t on = (size_t)g->online_n + 1, most = 0;
    int floored = 0;
    if (cz->kind == ATT_ACC_TERMS && on <= cz->nmax && cz->has[on] && !(rc = dyad_mul(&d1, &cz->terms[3 * on + 1], OC(GROWTH), w, 1)) &&
        !(rc = dyad_div(&d1, &d1, OC(G_MIN), w, 1)) && !(rc = dyad_add(&ez, &cz->terms[3 * on], &d1, w, 1)) && !(rc = sub_dir(&d1, &one, &ez, w, 0)) &&
        dyad_sign(&d1) > 0 && !(rc = dyad_mul(&floor_z, OC(G_MIN), &d1, w, 0)))
        floored = dyad_sign(&floor_z) > 0;
    /* Z^ in the single pass's interval, its lower end at least floor_z (the split of the row's maximum) */
    if (rc || (rc = ivl_copy(&zhat, total)) || (floored && dyad_cmp(&floor_z, &zhat.lo) > 0 && (rc = dyad_copy(&zhat.lo, &floor_z))) ||
        dyad_sign(&zhat.lo) <= 0)
        goto out;
    /* each split's logsumexp as an interval, and Gamma_i >= |lse_i - (mu_i + log Z_i)| */
    size_t j = 0;
    for (size_t i = 0; !rc && !unb && i < np; i++) {
        size_t first = j, idx = keys[j] / (size_t)size, k0 = first;
        while (j < n && keys[j] / (size_t)size == idx) j++;
        size_t end = (idx + 1) * (size_t)size < S ? (idx + 1) * (size_t)size : S;
        if (j - first > most) most = j - first;
        /* a split whose keys may all vanish may have Z_i = 0: no bound */
        while (k0 < j && vanish[k0]) k0++;
        if (k0 == j) {
            unb = 1;
            break;
        }
        if (own_ok[i] && (rc = dyad_mul(&d3, &own[i], OC(GROWTH), w, 1))) break;
        if ((rc = pass_error(&m, &err, &unb, g, ATT_A_SUM, (int64_t)(end - idx * (size_t)size), OC(GROWTH), own_ok[i] ? &d3 : NULL)) || unb) break;
        /* Z_i in [the larger of g_min - error and floor_z, m + error] */
        if ((rc = sub_dir(&d1, OC(G_MIN), &err, w, 0)) || (rc = dyad_add(&d2, &m, &err, w, 1))) break;
        if (floored && dyad_cmp(&floor_z, &d1) > 0 && (rc = dyad_copy(&d1, &floor_z))) break;
        if (dyad_sign(&d1) <= 0) {
            unb = 1;
            break;
        }
        if ((!i || dyad_cmp(&d1, &zlo) < 0) && (rc = dyad_copy(&zlo, &d1))) break;
        if ((rc = log_end(&lg.lo, &d1, w, 0)) || (rc = log_end(&lg.hi, &d2, w, 1))) break;
        /* the split's maximum: its lower end over the keys that cannot vanish */
        if ((rc = ivl_copy(&mu, &L[k0]))) break;
        for (size_t z = first; !rc && z < j; z++) {
            if (dyad_cmp(&L[z].hi, &mu.hi) > 0) rc = dyad_copy(&mu.hi, &L[z].hi);
            if (!rc && !vanish[z] && dyad_cmp(&L[z].lo, &mu.lo) > 0) rc = dyad_copy(&mu.lo, &L[z].lo);
        }
        if (!rc && exp2) rc = ivl_mul(&mu, &mu, &ln2, w);
        /* Gamma_i: the roundings of mu (exp2), of the log (and fast_log's floor) and of their sum */
        if (rc || (rc = lean_error(&gam[i], &mu.lo, &mu.hi, &g->lean[ATT_P_EXP2_LSE], &unb)) ||
            (rc = lean_error(&d1, &lg.lo, &lg.hi, &g->lean[ATT_P_LOG], &unb)) || (rc = dyad_add(&gam[i], &gam[i], &d1, w, 1)) || unb)
            break;
        if ((g->methods & ATT_M_FAST_LOG) && (rc = dyad_add(&gam[i], &gam[i], OC(FLOOR), w, 1))) break;
        if ((rc = lean_ivl(&mu, &g->lean[ATT_P_EXP2_LSE], w, &unb)) || (rc = lean_ivl(&lg, &g->lean[ATT_P_LOG], w, &unb)) || unb) break;
        if ((g->methods & ATT_M_FAST_LOG) && (rc = widen_by(&lg, OC(FLOOR), w))) break;
        if ((rc = ivl_add(&held, &mu, &lg, w)) || (rc = lean_error(&d1, &held.lo, &held.hi, &g->lean[ATT_P_ADD], &unb)) || unb ||
            (rc = dyad_add(&gam[i], &gam[i], &d1, w, 1)) || (rc = ivl_copy(&lses[i], &held)))
            break;
        rc = lean_ivl(&lses[i], &g->lean[ATT_P_ADD], w, &unb);
    }
    if (rc || unb) goto out;
    /* the merge on those intervals: M = max lse_i, sigma = sum exp(lse_i - M), log(sigma) + M */
    rc = ivl_copy(&topl, &lses[0]);
    for (size_t i = 1; !rc && i < np; i++) rc = ivl_max(&topl, &topl, &lses[i]);
    /* Gamma >= |gamma_K| for the split K of the largest lse: the largest Gamma_i of the splits that may be K, lse_i.hi >= M.lo */
    if (!rc) rc = dyad_set_si(&gamma, 0, 0);
    for (size_t i = 0; !rc && i < np; i++)
        if (dyad_cmp(&lses[i].hi, &topl.lo) >= 0 && dyad_cmp(&gam[i], &gamma) > 0) rc = dyad_copy(&gamma, &gam[i]);
    if (rc) goto out;
    /* the merge's sums over nt terms: the P splits, or with flush every split (an empty split's zero may flush a partial sum) */
    const att_acc *cs = &g->acc[ATT_A_SPLIT_SUM], *ca = &g->acc[ATT_A_SPLIT_ACC];
    size_t nt = g->flush ? (S + (size_t)size - 1) / (size_t)size : np;
    if (cs->kind != ATT_ACC_TERMS || ca->kind != ATT_ACC_TERMS || nt > cs->nmax || nt > ca->nmax || !cs->has[nt] || !ca->has[nt]) {
        unb = 1;
        goto out;
    }
    const dyad *gp = &cs->terms[3 * nt], *kp = &cs->terms[3 * nt + 1];
    if (!(rc = dyad_set_si(&sigma.lo, 0, 0)) && !(rc = dyad_set_si(&sigma.hi, 0, 0))) rc = dyad_set_si(&mass, 0, 0);
    for (size_t i = 0; !rc && !unb && i < np; i++) {
        if ((rc = ivl_sub(&d, &lses[i], &topl, w)) || (rc = lean_error(&eta[i], &d.lo, &d.hi, &g->lean[ATT_P_SPLIT_SHIFT], &unb)) || unb ||
            (rc = lean_ivl(&d, &g->lean[ATT_P_SPLIT_SHIFT], w, &unb)) || unb || (rc = exp_end(&d1, &d.lo, w, 0)) || (rc = exp_end(&d.hi, &d.hi, w, 1)) ||
            (rc = dyad_copy(&d.lo, &d1)) || (rc = lean_ivl(&d, &g->lean[ATT_P_SPLIT_EXP], w, &unb)) || unb || (rc = ivl_add(&sigma, &sigma, &d, w)) ||
            (rc = ends_magnitude(&d1, &d.lo, &d.hi)))
            break;
        rc = dyad_add(&mass, &mass, &d1, w, 1);
    }
    if (!rc && !unb) rc = acc_apply(&sigma, cs, nt, &mass, w, &unb);
    /* Sigma >= (1 - rel_e - abs_e)(1 - gamma_P) - C_P: the split K's term is exp(fl(0)), every term nonnegative (The logsumexp) */
    if (!rc && !unb && !(rc = dyad_add(&d1, OC(SE_REL), OC(SE_ABS), w, 1)) && !(rc = sub_dir(&d1, &one, &d1, w, 0)) && !(rc = sub_dir(&d2, &one, gp, w, 0)) &&
        dyad_sign(&d1) > 0 && dyad_sign(&d2) > 0 && !(rc = dyad_mul(&d1, &d1, &d2, w, 0)) && !(rc = sub_dir(&d1, &d1, kp, w, 0)) && dyad_cmp(&d1, &sigma.lo) > 0)
        rc = dyad_copy(&sigma.lo, &d1);
    if (rc || unb || dyad_sign(&sigma.lo) <= 0) {
        unb = 1;
        goto out;
    }
    if ((rc = log_end(&lg.lo, &sigma.lo, w, 0)) || (rc = log_end(&lg.hi, &sigma.hi, w, 1)) ||
        (rc = lean_error(&slack, &lg.lo, &lg.hi, &g->lean[ATT_P_SPLIT_LOG], &unb)) || unb || (rc = ivl_copy(&held, &lg)) ||
        (rc = lean_ivl(&held, &g->lean[ATT_P_SPLIT_LOG], w, &unb)) || unb || (rc = ivl_add(&held, &held, &topl, w)) ||
        (rc = lean_error(&d1, &held.lo, &held.hi, &g->lean[ATT_P_SPLIT_ADD], &unb)) || unb || (rc = dyad_add(&slack, &slack, &d1, w, 1)) ||
        (rc = ivl_copy(&crude, &held)) || (rc = lean_ivl(&crude, &g->lean[ATT_P_SPLIT_ADD], w, &unb)) || unb)
        goto out;
    /* W_i >= each split's share of the row sum: min(1, exp(lse_i.hi - M.lo + Gamma_i + Gamma)) */
    for (size_t i = 0; !rc && i < np; i++) {
        if ((rc = sub_dir(&d1, &lses[i].hi, &topl.lo, w, 1)) || (rc = dyad_add(&d1, &d1, &gam[i], w, 1)) || (rc = dyad_add(&d1, &d1, &gamma, w, 1)) ||
            (rc = exp_overflows(&d1) ? dyad_copy(&share[i], &one) : exp_end(&share[i], &d1, w, 1)))
            break;
        if (dyad_cmp(&share[i], &one) >= 0) rc = dyad_copy(&share[i], &one);
    }
    if (rc) goto out;
    /* sum_i W_i exp(g_i) for |g_i| <= spread_i (weighed): with spread_i = Gamma_i + eta_i */
    size_t k = 0;
    for (size_t i = 0; !rc && i < np; i++) {
        if ((rc = dyad_add(&spread[i], &gam[i], &eta[i], w, 1))) break;
        if (dyad_cmp(&spread[i], &spread[k]) < 0) k = i;
    }
    if (rc || (rc = exp_end(&d1, &spread[k], w, 0)) || (rc = dyad_neg(&d2, &spread[k])) || (rc = exp_end(&d2, &d2, w, 1)) ||
        (rc = dyad_set_si(&ehi, 0, 0)) || (rc = dyad_set_si(&elo, 0, 0)))
        goto out;
    /* more = sum W_i (exp(g_i) - exp(G_k)) into ehi, less = sum W_i (exp(-G_k) - exp(-g_i)) into elo */
    for (size_t i = 0; !rc && i < np; i++) {
        if ((rc = exp_end(&d3, &spread[i], w, 1)) || (rc = sub_dir(&d3, &d3, &d1, w, 1)) || (rc = dyad_mul(&d3, &share[i], &d3, w, 1)) ||
            (rc = dyad_add(&ehi, &ehi, &d3, w, 1)) || (rc = dyad_neg(&d3, &spread[i])) || (rc = exp_end(&d3, &d3, w, 0)) || (rc = sub_dir(&d3, &d2, &d3, w, 1)) ||
            (rc = dyad_mul(&d3, &share[i], &d3, w, 1)))
            break;
        rc = dyad_add(&elo, &elo, &d3, w, 1);
    }
    if (rc || (rc = dyad_neg(&d3, &spread[k])) || (rc = exp_end(&d3, &d3, w, 0)) || (rc = sub_dir(&elo, &d3, &elo, w, 0)) ||
        (rc = exp_end(&d3, &spread[k], w, 1)) || (rc = dyad_add(&ehi, &d3, &ehi, w, 1)))
        goto out;
    /* F's ends: Sigma = exp(m - M) Z^ F; then Delta = log F within slack */
    dyad egam, absolute;
    dyad_init(&egam);
    dyad_init(&absolute);
    if (!(rc = exp_end(&egam, &gamma, w, 1)) && !(rc = dyad_times(&absolute, OC(SE_ABS), P, w)) && !(rc = dyad_mul(&absolute, &absolute, &egam, w, 1)) &&
        !(rc = dyad_add(&d1, &one, OC(SE_REL), w, 1)) && !(rc = dyad_mul(&fhi, &ehi, &d1, w, 1)) && !(rc = dyad_add(&fhi, &fhi, &absolute, w, 1)) &&
        !(rc = dyad_add(&d1, &one, gp, w, 1)) && !(rc = dyad_mul(&fhi, &fhi, &d1, w, 1)) && !(rc = dyad_mul(&d3, kp, &egam, w, 1)) &&
        !(rc = dyad_add(&fhi, &fhi, &d3, w, 1)) && !(rc = sub_dir(&d1, &one, OC(SE_REL), w, 0)) && !(rc = dyad_mul(&flo, &elo, &d1, w, 0)) &&
        !(rc = sub_dir(&flo, &flo, &absolute, w, 0)) && !(rc = sub_dir(&d1, &one, gp, w, 0)) && !(rc = dyad_mul(&flo, &flo, &d1, w, 0)))
        rc = sub_dir(&flo, &flo, &d3, w, 0);
    dyad_free(&egam);
    dyad_free(&absolute);
    if (rc) goto out;
    if (dyad_sign(&flo) <= 0) {
        unb = 1;
        goto out;
    }
    ivl delta;
    ivl_init(&delta);
    if (!(rc = log_end(&delta.lo, &flo, w, 0)) && !(rc = sub_dir(&delta.lo, &delta.lo, &slack, w, 0)) && !(rc = log_end(&delta.hi, &fhi, w, 1)) &&
        !(rc = dyad_add(&delta.hi, &delta.hi, &slack, w, 1)) && !(rc = ivl_copy(lse, top)) && (!exp2 || !(rc = ivl_mul(lse, lse, &ln2, w))) &&
        !(rc = log_end(&lg.lo, &zhat.lo, w, 0)) && !(rc = log_end(&lg.hi, &zhat.hi, w, 1)) && !(rc = ivl_add(lse, lse, &lg, w)))
        rc = ivl_add(lse, lse, &delta, w);
    if (!rc) *lse_ok = 1;
    if (!values_ok) {
        ivl_free(&delta);
        goto out;
    }
    /* the output: phi_i = exp(gamma_i + eta'_i - Delta)(1 + eps'_i) rho_i, centred on the split with the smallest spread */
    if (!rc && !(rc = dyad_add(&rho_hi, &one, OC(SA_REL), w, 1)) && !(rc = dyad_add(&d1, &one, OC(SB_REL), w, 1)) && !(rc = dyad_mul(&rho_hi, &rho_hi, &d1, w, 1)) &&
        !(rc = dyad_add(&d1, &one, OC(SK_REL), w, 1)) && !(rc = dyad_mul(&rho_hi, &rho_hi, &d1, w, 1)) && !(rc = sub_dir(&rho_lo, &one, OC(SA_REL), w, 0)) &&
        !(rc = sub_dir(&d1, &one, OC(SB_REL), w, 0)) && !(rc = dyad_mul(&rho_lo, &rho_lo, &d1, w, 0)) && !(rc = sub_dir(&d1, &one, OC(SK_REL), w, 0)))
        rc = dyad_mul(&rho_lo, &rho_lo, &d1, w, 0);
    k = 0;
    for (size_t i = 0; !rc && i < np; i++) {
        /* eta'_i: the shift's rounding for lse_i - crude */
        if ((rc = ivl_sub(&d, &lses[i], &crude, w)) || (rc = lean_error(&d1, &d.lo, &d.hi, &g->lean[ATT_P_SPLIT_SHIFT], &unb)) || unb ||
            (rc = dyad_add(&spread[i], &gam[i], &d1, w, 1)))
            break;
        /* s_i's least value, exp of the shift's least, rounded: where it is positive s_i is not 0 (Masked keys: 0 times an
           infinite O_i is NaN) */
        int sunb = 0;
        if ((rc = dyad_copy(&d.hi, &d.lo)) || (rc = lean_ivl(&d, &g->lean[ATT_P_SPLIT_SHIFT], w, &sunb))) break;
        if (!sunb && !(rc = exp_clamped(&d.lo, &d.lo, w, 0)) && !(rc = dyad_copy(&d.hi, &d.lo)) && !(rc = lean_ivl(&d, &g->lean[ATT_P_SPLIT_EXP], w, &sunb)) &&
            !sunb && dyad_sign(&d.lo) > 0)
            szero[i] = 0;
        if (rc) break;
        /* phi_out = exp(s_i - Delta.lo)(1 + rel_e); phi_hi = phi_out rho_hi; phi_lo = exp(-s_i - Delta.hi)(1 - rel_e) rho_lo */
        if ((rc = sub_dir(&d1, &spread[i], &delta.lo, w, 1)) || (rc = exp_end(&d1, &d1, w, 1)) || (rc = dyad_add(&d2, &one, OC(SE_REL), w, 1)) ||
            (rc = dyad_mul(&phi_out[i], &d1, &d2, w, 1)) || (rc = dyad_mul(&phi_hi[i], &phi_out[i], &rho_hi, w, 1)) || (rc = dyad_neg(&d1, &spread[i])) ||
            (rc = sub_dir(&d1, &d1, &delta.hi, w, 0)) || (rc = exp_end(&d1, &d1, w, 0)) || (rc = sub_dir(&d2, &one, OC(SE_REL), w, 0)) ||
            (rc = dyad_mul(&d1, &d1, &d2, w, 0)) || (rc = dyad_mul(&phi_lo[i], &d1, &rho_lo, w, 0)))
            break;
        if (dyad_cmp(&spread[i], &spread[k]) < 0) k = i;
    }
    ivl_free(&delta);
    if (rc || unb) goto out;
    /* phi_c at or above split k's midpoint; radii; the other splits' excess by their shares */
    if ((rc = dyad_add(&phi_c, &phi_hi[k], &phi_lo[k], w, 1)) || (rc = dyad_shift(&phi_c, -1))) goto out;
    if ((rc = dyad_set_si(&rmore, 0, 0)) || (rc = dyad_set_si(&pmore, 0, 0)) || (rc = dyad_set_si(&big_phi, 0, 0)) || (rc = dyad_copy(&weighed, &phi_out[k])))
        goto out;
    for (size_t i = 0; !rc && i < np; i++) {
        if ((rc = dyad_set_si(&radius[i], 0, 0)) || (rc = sub_dir(&d1, &phi_hi[i], &phi_c, w, 1)) || (rc = sub_dir(&d2, &phi_c, &phi_lo[i], w, 1))) break;
        if (dyad_cmp(&d1, &radius[i]) > 0 && (rc = dyad_copy(&radius[i], &d1))) break;
        if (dyad_cmp(&d2, &radius[i]) > 0 && (rc = dyad_copy(&radius[i], &d2))) break;
        if (dyad_cmp(&phi_out[i], &big_phi) > 0 && (rc = dyad_copy(&big_phi, &phi_out[i]))) break;
        /* phi_out_k + sum_i W_i (phi_out_i - phi_out_k)+ (Each split's share) */
        if (dyad_cmp(&phi_out[i], &phi_out[k]) > 0 && !(rc = sub_dir(&d1, &phi_out[i], &phi_out[k], w, 1)) && !(rc = dyad_mul(&d1, &share[i], &d1, w, 1)))
            rc = dyad_add(&weighed, &weighed, &d1, w, 1);
    }
    /* sum_i w_i phi'_i at most the smaller of the two */
    if (!rc && dyad_cmp(&weighed, &big_phi) < 0) rc = dyad_copy(&big_phi, &weighed);
    for (size_t i = 0; !rc && i < np; i++) {
        if (dyad_cmp(&radius[i], &radius[k]) > 0 && !(rc = sub_dir(&d1, &radius[i], &radius[k], w, 1)) && !(rc = dyad_mul(&d1, &share[i], &d1, w, 1)))
            rc = dyad_add(&rmore, &rmore, &d1, w, 1);
        if (!rc && dyad_cmp(&phi_hi[i], &phi_hi[k]) > 0 && !(rc = sub_dir(&d1, &phi_hi[i], &phi_hi[k], w, 1)) && !(rc = dyad_mul(&d1, &share[i], &d1, w, 1)))
            rc = dyad_add(&pmore, &pmore, &d1, w, 1);
    }
    /* 1 / Z_i at most, and the reciprocal's */
    if (rc || (rc = dyad_div(&inv_low, &one, &zlo, w, 1)) || (rc = dyad_add(&d1, &one, OC(SB_REL), w, 1)) || (rc = dyad_mul(&recip, &inv_low, &d1, w, 1)) ||
        (rc = dyad_add(&recip, &recip, OC(SB_ABS), w, 1)))
        goto out;
    int recip_on = g->methods & ATT_M_RECIPROCAL, out_unb = recip_on && lean_overflows(&g->lean[ATT_P_RECIPROCAL], &inv_low);
    const dyad *gd = &ca->terms[3 * nt], *kd = &ca->terms[3 * nt + 1], *limd = &ca->terms[3 * nt + 2];
    dyad mm, e2, hn, v, omax, oabs, second, q, ht, dot, r1, vscale, vplus, vmax, vb;
    dyad *cs2[] = {&mm, &e2, &hn, &v, &omax, &oabs, &second, &q, &ht, &dot, &r1, &vscale, &vplus, &vmax, &vb};
    for (size_t z = 0; z < sizeof cs2 / sizeof *cs2; z++) dyad_init(cs2[z]);
    /* V's second bound, (1 + rel_a)(c_W vmax (1 + gamma_V) + const_V growth / g_min) / (1 - E_Z) + abs_a / floor_z (The output): vscale =
       c_W (1 + gamma_V), vplus = const_V growth / g_min, with c_W = 1 + rel_W + most abs_W growth / g_min (weights) and Lemma T's
       G and H with mma */
    const att_acc *cv = &g->acc[ATT_A_VALUES];
    int mma = g->methods & ATT_M_MMA_VALUES, vbound = 0;
    if (!rc && floored && (mma || (cv->kind == ATT_ACC_TERMS && on <= cv->nmax && cv->has[on]))) {
        const dyad *gv = mma ? OC(MMA_G) : &cv->terms[3 * on], *kv = mma ? OC(MMA_H) : &cv->terms[3 * on + 1];
        if (!(rc = dyad_add(&vscale, &one, gv, w, 1)) && (g->methods & ATT_M_WEIGHTS) && !(rc = dyad_mul(&d1, OC(ABS_W), OC(GROWTH), w, 1)) &&
            !(rc = dyad_times(&d1, &d1, (int64_t)most, w)) && !(rc = dyad_div(&d1, &d1, OC(G_MIN), w, 1)) && !(rc = dyad_add(&d1, &d1, OC(REL_W), w, 1)) &&
            !(rc = dyad_add(&d1, &one, &d1, w, 1)))
            rc = dyad_mul(&vscale, &vscale, &d1, w, 1);
        if (!rc && !(rc = dyad_mul(&vplus, kv, OC(GROWTH), w, 1)) && !(rc = dyad_div(&vplus, &vplus, OC(G_MIN), w, 1)))
            vbound = 1;
    }
    /* the splits' own bounds' largest, where each has one (own_ok) */
    int owned = 1;
    dyad own_max, own_v;
    dyad_init(&own_max);
    dyad_init(&own_v);
    for (size_t i = 0; owned && i < np; i++) owned = own_ok[i];
    if (!rc && owned && !(rc = dyad_copy(&own_max, &own[0])))
        for (size_t i = 1; !rc && i < np; i++)
            if (dyad_cmp(&own[i], &own_max) > 0) rc = dyad_copy(&own_max, &own[i]);
    for (size_t c = 0; !rc && !out_unb && c < Ev; c++) {
        /* a split's own value sum, unscaled, and its rounding to F */
        if (owned && (rc = dyad_mul(&own_v, &own_max, &largest[c], w, 1))) break;
        if ((rc = pass_error(&mm, &e2, &out_unb, g, ATT_A_VALUES, g->online_S, &largest[c], owned ? &own_v : NULL)) || out_unb) break;
        if ((rc = dyad_add(&d1, &mm, &e2, w, 1))) break;
        if (lean_overflows(&g->lean[ATT_P_VALUES_OUT], &d1)) {
            out_unb = 1;
            break;
        }
        /* |N_i| after values.output, |N_i| / Z_i (V) and |fl(N_i)| times the reciprocal's largest value */
        if ((rc = dyad_add(&d2, &one, OC(SA_REL), w, 1)) || (rc = dyad_mul(&hn, &d1, &d2, w, 1)) || (rc = dyad_add(&hn, &hn, OC(SA_ABS), w, 1)) ||
            (rc = dyad_mul(&v, &hn, &inv_low, w, 1)) || (rc = dyad_mul(&d1, &hn, &recip, w, 1)))
            break;
        /* V at most the second bound from the column's largest |v|; then fl(N_i) fl(1 / Z_i) at most V (1 + rel_b) + |fl(N_i)| abs_b */
        if (vbound && !(rc = ends_magnitude(&vmax, &vlo[c], &vhi[c])) && !(rc = dyad_mul(&vb, &vscale, &vmax, w, 1)) && !(rc = dyad_add(&vb, &vb, &vplus, w, 1)) &&
            !(rc = dyad_mul(&vb, &vb, &d2, w, 1)) && !(rc = sub_dir(&d3, &one, &ez, w, 0)) && !(rc = dyad_div(&vb, &vb, &d3, w, 1)) &&
            !(rc = dyad_div(&d3, OC(SA_ABS), &floor_z, w, 1)) && !(rc = dyad_add(&vb, &vb, &d3, w, 1)) && dyad_cmp(&vb, &v) < 0 && !(rc = dyad_copy(&v, &vb)) &&
            !(rc = dyad_add(&d3, &one, OC(SB_REL), w, 1)) && !(rc = dyad_mul(&vb, &v, &d3, w, 1)) && !(rc = dyad_mul(&d3, &hn, OC(SB_ABS), w, 1)) &&
            !(rc = dyad_add(&vb, &vb, &d3, w, 1)) && dyad_cmp(&vb, &d1) < 0)
            rc = dyad_copy(&d1, &vb);
        if (rc) break;
        if (lean_overflows(&g->lean[ATT_P_NORMALIZE], recip_on ? &d1 : &v)) {
            out_unb = 1;
            break;
        }
        if ((rc = dyad_add(&d3, &one, OC(SK_REL), w, 1))) break;
        if (recip_on) {
            /* o_max = |fl(N_i) fl(1 / Z_i)| (1 + rel_k) + abs_k; o_abs = (abs_a recip + held_n abs_b)(1 + rel_k) + abs_k */
            if ((rc = dyad_mul(&omax, &d1, &d3, w, 1)) || (rc = dyad_add(&omax, &omax, OC(SK_ABS), w, 1)) || (rc = dyad_mul(&oabs, OC(SA_ABS), &recip, w, 1)) ||
                (rc = dyad_mul(&d2, &hn, OC(SB_ABS), w, 1)) || (rc = dyad_add(&oabs, &oabs, &d2, w, 1)) || (rc = dyad_mul(&oabs, &oabs, &d3, w, 1)) ||
                (rc = dyad_add(&oabs, &oabs, OC(SK_ABS), w, 1)))
                break;
        } else if ((rc = dyad_mul(&omax, &v, &d3, w, 1)) || (rc = dyad_add(&omax, &omax, OC(SK_ABS), w, 1)) ||
                   (rc = dyad_mul(&oabs, OC(SA_ABS), &inv_low, w, 1)) || (rc = dyad_mul(&oabs, &oabs, &d3, w, 1)) ||
                   (rc = dyad_add(&oabs, &oabs, OC(SK_ABS), w, 1))) {
            break;
        }
        /* the absolute parts; sum |s_i O_i| at most; the merge's products and their sum */
        if ((rc = dyad_mul(&second, &big_phi, &oabs, w, 1)) || (rc = dyad_times(&d1, OC(SE_ABS), P, w)) || (rc = dyad_mul(&d1, &d1, &omax, w, 1)) ||
            (rc = dyad_add(&second, &second, &d1, w, 1)) || (rc = dyad_mul(&q, &phi_hi[k], &sizes[c], w, 1)) || (rc = dyad_div(&q, &q, &zhat.lo, w, 1)) ||
            (rc = dyad_mul(&d1, &v, &pmore, w, 1)) || (rc = dyad_add(&q, &q, &d1, w, 1)) || (rc = dyad_add(&q, &q, &second, w, 1)) ||
            (rc = dyad_add(&d1, &one, OC(SP_REL), w, 1)) || (rc = dyad_mul(&ht, &q, &d1, w, 1)) || (rc = dyad_times(&r1, OC(SP_ABS), P, w)) ||
            (rc = dyad_add(&ht, &ht, &r1, w, 1)) || (rc = dyad_mul(&dot, OC(SP_REL), &q, w, 1)) || (rc = dyad_add(&dot, &dot, &r1, w, 1)) ||
            (rc = dyad_mul(&d1, gd, &ht, w, 1)) || (rc = dyad_add(&dot, &dot, &d1, w, 1)) || (rc = dyad_add(&dot, &dot, kd, w, 1)) ||
            (rc = dyad_add(&d1, &ht, &dot, w, 1)))
            break;
        if (lean_overflows(&g->lean[ATT_P_SPLIT_PRODUCT], &q) || dyad_cmp(&d1, limd) > 0) {
            out_unb = 1;
            break;
        }
        /* (phi_c x widened by r_k size) / Z^, widened by v radius_more + second + dot */
        ivl pc;
        ivl_init(&pc);
        if (!(rc = ivl_point(&pc, &phi_c)) && !(rc = ivl_mul(&out[c], &pc, &out[c], w)) && !(rc = dyad_mul(&d1, &radius[k], &sizes[c], w, 1)) &&
            !(rc = widen_by(&out[c], &d1, w)) && !(rc = ivl_div(&out[c], &out[c], &zhat, w)) && !(rc = dyad_mul(&d1, &v, &rmore, w, 1)) &&
            !(rc = dyad_add(&d1, &d1, &second, w, 1)) && !(rc = dyad_add(&d1, &d1, &dot, w, 1)))
            rc = widen_by(&out[c], &d1, w);
        ivl_free(&pc);
    }
    for (size_t z = 0; z < sizeof cs2 / sizeof *cs2; z++) dyad_free(cs2[z]);
    dyad_free(&own_max);
    dyad_free(&own_v);
    if (!rc && !out_unb) *row_ok = 1;
out:
    if (ready)
        for (size_t z = 0; z < sizeof arrays / sizeof *arrays; z++)
            for (size_t i = 0; i < np; i++) dyad_free(&arrays[z][i]);
    for (size_t z = 0; z < sizeof arrays / sizeof *arrays; z++) free(arrays[z]);
    ivls_free(lses, lses ? np : 0);
    for (size_t z = 0; z < sizeof ivs / sizeof *ivs; z++) ivl_free(ivs[z]);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* |sigma|'s ends at p bits: num / den, or 1 / sqrt(E) when den is 0 */
int abs_sigma(dyad *lo, dyad *hi, const att_call *a, int64_t p) {
    dyad x, y;
    dyad_init(&x);
    dyad_init(&y);
    int rc;
    if (a->den.n) {
        if (!(rc = big_copy(&x.m, &a->num)) && !(rc = big_copy(&y.m, &a->den))) {
            x.m.neg = y.m.neg = 0;
            x.e = y.e = 0;
            if (!(rc = dyad_div(lo, &x, &y, p, 0))) rc = dyad_div(hi, &x, &y, p, 1);
        }
    } else if (!(rc = dyad_set_si(&x, a->E, 0)) && !(rc = dyad_set_si(&y, 1, 0)) && !(rc = dyad_sqrt(lo, &x, p + 4, 1)) &&
               !(rc = dyad_div(lo, &y, lo, p, 0)) && !(rc = dyad_sqrt(hi, &x, p + 4, 0))) {
        rc = dyad_div(hi, &y, hi, p, 1);
    }
    dyad_free(&x);
    dyad_free(&y);
    return rc;
}

/* exp(y) rounded to the side for an end y of an interval of a computed shift of key j's logit, from Lemma D's tables
   (attention_algorithms.md, The exponentials from Lemma D's tables): exp(y) = exp(delta) exp(c) exp(y - delta - c), with D key j's
   score distance from the top key's at 2**u, delta = -|sigma| D 2**u its exact logit shift, exp(delta) within [pw, pw + h] 2**-F
   (Lemma D), c a row's exact constant (NULL: 0) with exp(c) at most ec's end on the side (NULL: 1), and the rest by Lemma W3.
   sig: |sigma|'s end on the side, at p + 64 bits or more. *done 0, r unchanged, where the rest is not below 2**-4 or
   exp(delta) is below 2**-48 (there Lemma X keeps the relative precision). */
int table_exp(dyad *r, int *done, const dyad *y, w128 D, const att_call *a, const dyad *sig, const dyad *c, const dyad *ec, int64_t p, int up) {
    *done = 0;
    uint64_t pw[3] = {0, 0, (uint64_t)1 << (ATT_F - 128)};
    int z = 0, ok = 0, rc = att_digits_mul(pw, &z, D, a->tables);
    if (rc || pw[2] < ((uint64_t)1 << 15)) return rc;
    if (up) { /* the table's upper end, pw + h */
        uint64_t h = z ? 3 * (uint64_t)z - 1 : 0;
        for (int i = 0; i < 3 && h; i++) {
            pw[i] += h;
            h = pw[i] < h;
        }
    }
    int64_t P = p + 64;
    dyad dist, rest, e, w;
    dyad *ds[] = {&dist, &rest, &e, &w};
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_init(ds[k]);
    if (!(rc = big_reserve(&dist.m, 2))) {
        dist.m.w[0] = w128_lo(D);
        dist.m.w[1] = w128_hi(D);
        dist.m.n = 2;
        dist.m.neg = 0;
        big_trim(&dist.m);
        dist.e = dist.m.n ? a->u : 0;
    }
    /* the rest y - delta - c = y + |sigma| D 2**u - c, to the side */
    if (!rc && !(rc = dyad_mul(&rest, sig, &dist, P, up)) && !(rc = dyad_add(&rest, y, &rest, P, up)) && (!c || !(rc = sub_dir(&rest, &rest, c, P, up))) &&
        !(rc = exp_near_one(&e, &ok, &rest, P, up)) && ok && !(rc = dyad_words(&w, pw, 3, -ATT_F)) && (!ec || !(rc = dyad_mul(&w, &w, ec, P, up))) &&
        !(rc = dyad_mul(r, &w, &e, p, up)))
        *done = 1;
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_free(ds[k]);
    return rc;
}

/* The top key's score at 2**u among the n kept keys' (the largest, the least with sigma < 0), and the distance of key j's from it */
w128 table_top(const w128 *scores, size_t n, int sign) {
    w128 top = scores[0], bias = w128_make(0, (uint64_t)1 << 63);
    for (size_t j = 1; j < n; j++)
        if (sign > 0 ? w128_less(w128_add(top, bias), w128_add(scores[j], bias)) : w128_less(w128_add(scores[j], bias), w128_add(top, bias))) top = scores[j];
    return top;
}

w128 table_distance(w128 top, w128 score, int sign) { return sign > 0 ? w128_sub(top, score) : w128_sub(score, top); }

/* A row's constant c for the given logsumexp lse: sigma s_top 2**u - lse (an exact dyad near the top key's shift), and exp(c)'s ends */
int table_row_constant(dyad *c, ivl *ec, w128 top, const att_call *a, const dyad *sl, const dyad *lse, int64_t p) {
    dyad st;
    dyad_init(&st);
    w128 mag = w128_negative(top) ? w128_sub(w128_zero(), top) : top;
    int rc = big_reserve(&st.m, 2);
    if (!rc) {
        st.m.w[0] = w128_lo(mag);
        st.m.w[1] = w128_hi(mag);
        st.m.n = 2;
        st.m.neg = w128_negative(top) != (a->sign < 0);
        big_trim(&st.m);
        st.e = st.m.n ? a->u : 0;
    }
    if (!rc && !(rc = dyad_mul(&st, &st, sl, p, 0)) && !(rc = sub_dir(c, &st, lse, p, 0)) && !(rc = exp_end(&ec->lo, c, p + 64, 0)))
        rc = exp_end(&ec->hi, c, p + 64, 1);
    dyad_free(&st);
    return rc;
}

/* The step-by-step program's steps 1 to 7 for one row whose n kept keys att_row_logits has given: every key's score, the
   kept keys' logits L (with a float mask, its values in B) and, unless *soft_unb, their weights Wt, the maximum top and the
   row sum total; *masked_nan where a masked key's logit may overflow (the row may be NaN, Masked keys), *soft_unb where the softmax has
   none, and for the padded rows' bound (*pad_unb where it has none) the kept keys' logits before their rounding, Y. V[j] is
   1 where kept key j may vanish (attention_algorithms.md, Keys that may vanish): a rounding of its mask value, logit or
   shift may overflow toward -inf, its interval there holds the finite outcomes and its weight's the hull with 0; 2 where
   it surely vanishes (one of those roundings has no finite result), its weight and R exactly [0, 0]. P is
   split_kv's count of splits. With no kept key, or logits_only, only the scores and logits. mnan (NULL: none): per key, 1
   at a masked key whose logit may overflow (each masked key checked, not only up to the first). */
int att_prog_softmax(ivl *L, ivl *B, ivl *Y, ivl *Wt, ivl *R, unsigned char *V, ivl *top, ivl *total, int *masked_nan_out, int *soft_unb_out,
                     int *pad_unb_out, att_row *w, const att_call *a, const att_prog *g, size_t n, size_t qr, size_t kr0, size_t moff, size_t mstep,
                     int64_t P, int logits_only, unsigned char *mnan) {
    const operand *q = a->q, *k = a->k;
    size_t S = a->S, *keys = w->keys;
    int64_t p = g->w;
    int online = g->methods & ATT_M_ONLINE, padded = g->methods & ATT_M_PADDED, rc = 0;
    ivl sc;
    ivl_init(&sc);
    dyad d1, mass, e1;
    dyad_init(&d1);
    dyad_init(&mass);
    dyad_init(&e1);
    product_sums ps, dev_ps;
    product_sums_init(&ps);
    product_sums_init(&dev_ps);
    /* every key's score; the masked keys' logits are computed too, before the -inf, only to find an overflow */
    int masked_nan = 0, soft_unb = g->online_unbounded, pad_unb = !padded || g->online_unbounded;
    /* the weights from Lemma D's tables where the call has them and the logits are sigma times exact scores */
    int tabled = a->tables && !a->mask_float && !a->uncertain && !g->flush && a->sign;
    w128 *scores = w->scores;
    if (n) memset(V, 0, n);
    size_t kj = 0;
    for (size_t j = 0; !rc && j < S; j++) {
        int is_kept = kj < n && keys[kj] == j, unb = 0;
        if (!is_kept && ((masked_nan && !mnan) || !w->scored[j])) continue; /* a masked key's score that is not finite: att_masked_nan's */
        if (is_kept && soft_unb && pad_unb) {
            kj++;
            continue;
        }
        if ((rc = product_element(&ps, q, qr, k, kr0 + j, PRODUCT_MASS, g->flush, g->emin, NULL))) break;
        if (a->rq || a->rk) { /* the radii's deviation, in the mass and the moved part (proofs/matmul.md, Results as inputs) */
            int none = 0;
            if ((rc = product_deviation(&ps, &none, &dev_ps, q, a->rq, qr, k, a->rk, kr0 + j, 0))) break;
        }
        if (!is_kept) {
            if (!(rc = prog_score(&sc, &unb, g, &ps, a->E)) && !unb) rc = prog_logit(top, &unb, NULL, g, &sc, NULL); /* top as scratch */
            if (unb) masked_nan = 1;
            if (unb && mnan) mnan[j] = 1;
            continue;
        }
        size_t m = kj++;
        if (tabled) { /* the score at the tables' unit 2**u, exactly */
            bigint sh;
            big_init(&sh);
            if (!(rc = big_shr(&sh, &ps.s, a->u - ps.e, 0))) {
                w128 v = w128_make(sh.n > 0 ? sh.w[0] : 0, sh.n > 1 ? sh.w[1] : 0);
                scores[m] = sh.neg ? w128_sub(w128_zero(), v) : v;
            }
            big_free(&sh);
            if (rc) break;
        }
        if (a->mask_float) {
            int bunb = 0, finite = 1;
            if (a->mrad && (rc = radius_f64(&e1, &finite, &g->f64, a->mrad + 8 * (moff + j * mstep)))) break;
            if ((rc = prog_bias(&B[m], &bunb, &V[m], g, &w->bias[m], a->mrad ? &e1 : NULL))) break;
            if (bunb) soft_unb = pad_unb = 1;
        }
        if (!soft_unb) {
            if (!(rc = prog_score(&sc, &unb, g, &ps, a->E)) && !unb) rc = prog_logit(&L[m], &unb, &V[m], g, &sc, a->mask_float ? &B[m] : NULL);
            if (unb) soft_unb = 1;
        }
        if (!rc && !pad_unb) {
            /* the any-order bound of the score, rounded, times the factor plus the mask's value: not rounded */
            int fl = 0;
            unb = 0;
            if (!(rc = sum_bounds_any_order(&Y[m].lo, &Y[m].hi, &fl, &g->scores, &ps.s, &ps.mass, &ps.moved, ps.e)) && fl) pad_unb = 1;
            if (!rc && !pad_unb && !(rc = lean_ivl(&Y[m], &g->lean[ATT_P_SCORES_OUT], p, &unb)) && unb) pad_unb = 1;
            if (!rc && !pad_unb) {
                ivl f;
                f.lo = *OC(FACTOR_LO);
                f.hi = *OC(FACTOR_HI);
                if (!(rc = ivl_mul(&Y[m], &Y[m], &f, p)) && a->mask_float) rc = ivl_add(&Y[m], &Y[m], &B[m], p);
            }
        }
    }
    if (rc || !n || logits_only) goto out;
    /* the softmax: the maximum (its lower end over the keys that cannot vanish; none: no bound), the weights and the row sum */
    size_t k0 = 0;
    while (k0 < n && V[k0]) k0++;
    if (k0 == n) soft_unb = 1;
    if (!soft_unb) {
        ivl ln2;
        ln2.lo = g->ln2_lo;
        ln2.hi = g->ln2_hi;
        rc = ivl_copy(top, &L[k0]);
        for (size_t j = 0; !rc && j < n; j++) {
            if (dyad_cmp(&L[j].hi, &top->hi) > 0) rc = dyad_copy(&top->hi, &L[j].hi);
            if (!rc && !V[j] && dyad_cmp(&L[j].lo, &top->lo) > 0) rc = dyad_copy(&top->lo, &L[j].lo);
        }
        dyad lsum, hsum, msum;
        dyad_init(&lsum);
        dyad_init(&hsum);
        dyad_init(&msum);
        if (!rc && !(rc = dyad_set_si(&lsum, 0, 0)) && !(rc = dyad_set_si(&hsum, 0, 0)) && !(rc = dyad_set_si(&msum, 0, 0)) &&
            !(rc = dyad_set_si(&total->lo, 0, 0)))
            rc = dyad_set_si(&total->hi, 0, 0);
        /* with tables: the top key's score (the largest, the least with sigma < 0) and |sigma|'s ends */
        w128 topv = w128_zero();
        dyad sl, sh;
        dyad_init(&sl);
        dyad_init(&sh);
        if (!rc && tabled && n) {
            topv = table_top(scores, n, a->sign);
            rc = abs_sigma(&sl, &sh, a, p + 128);
        }
        for (size_t j = 0; !rc && !soft_unb && j < n; j++) {
            ivl d;
            ivl_init(&d);
            int unb = 0;
            rc = ivl_sub(&d, &L[j], top, p);
            if (!online) {
                if (!rc && V[j] != 2) rc = lean_vanish(&d, &unb, &V[j], &g->lean[ATT_P_SHIFT], p);
            } else if (!rc && !(rc = ends_magnitude(&d1, &d.lo, &d.hi)) && !(rc = dyad_mul(&d1, OC(REL_SHIFT), &d1, p, 1)) &&
                       !(rc = dyad_add(&d1, &d1, OC(ABS_SHIFT), p, 1))) {
                rc = widen_by(&d, &d1, p);
            }
            if (!rc && !unb && V[j] == 2) {
                /* a key that surely vanishes: its weight exactly 0, with its later rescaling factors and in R too */
                if (!(rc = dyad_set_si(&Wt[j].lo, 0, 0)) && !(rc = dyad_set_si(&Wt[j].hi, 0, 0)) && R) rc = ivl_copy(&R[j], &Wt[j]);
                ivl_free(&d);
                continue;
            }
            if (!rc && !unb && (g->methods & ATT_M_EXP2)) rc = ivl_mul(&d, &d, &ln2, p);
            if (!rc && !unb && exp_overflows(&d.hi)) unb = 1;
            if (!rc && !unb) {
                int dl = 0, dh = 0;
                w128 D = tabled ? table_distance(topv, scores[j], a->sign) : w128_zero();
                if (tabled && !(rc = table_exp(&Wt[j].lo, &dl, &d.lo, D, a, &sl, NULL, NULL, p, 0)))
                    rc = table_exp(&Wt[j].hi, &dh, &d.hi, D, a, &sh, NULL, NULL, p, 1);
                if (!rc && !dl) rc = exp_clamped(&Wt[j].lo, &d.lo, p, 0);
                if (!rc && !dh) rc = exp_clamped(&Wt[j].hi, &d.hi, p, 1);
            }
            if (!rc && !unb && !online) {
                /* a key that may vanish: the hull with its weight 0 */
                if (!(rc = lean_ivl(&Wt[j], &g->lean[ATT_P_EXP], p, &unb)) && !unb && !(V[j] && dyad_sign(&Wt[j].lo) > 0 && (rc = dyad_set_si(&Wt[j].lo, 0, 0))) &&
                    !(rc = ivl_add(total, total, &Wt[j], p)) && !(rc = ends_magnitude(&d1, &Wt[j].lo, &Wt[j].hi)))
                    rc = dyad_add(&msum, &msum, &d1, p, 1);
            } else if (!rc && !unb) {
                /* the weight times its later rescaling factors: [y.lo down - own, y.hi up + own]; R keeps [y.lo down, y.hi up];
                   for a key that may vanish, the hulls with 0 */
                if (lean_overflows(&g->lean[ATT_P_EXP], &Wt[j].hi)) {
                    unb = 1;
                } else if (!(rc = dyad_mul(&Wt[j].lo, &Wt[j].lo, OC(DOWN), p, 0)) && !(rc = dyad_mul(&Wt[j].hi, &Wt[j].hi, OC(UP), p, 1)) &&
                           !(V[j] && (rc = dyad_set_si(&Wt[j].lo, 0, 0))) && !(R && (rc = ivl_copy(&R[j], &Wt[j]))) &&
                           !(rc = sub_dir(&Wt[j].lo, &Wt[j].lo, OC(OWN), p, 0)) && !(rc = dyad_add(&Wt[j].hi, &Wt[j].hi, OC(OWN), p, 1)) &&
                           !(rc = dyad_add(&lsum, &lsum, &Wt[j].lo, p + 64, 0)) && !(rc = dyad_add(&hsum, &hsum, &Wt[j].hi, p + 64, 1)) &&
                           !(rc = ends_magnitude(&d1, &Wt[j].lo, &Wt[j].hi))) {
                    rc = dyad_add(&msum, &msum, &d1, p + 64, 1);
                }
            }
            if (unb) soft_unb = 1;
            ivl_free(&d);
        }
        dyad_free(&sl);
        dyad_free(&sh);
        if (!rc && !soft_unb) {
            int unb = 0;
            if (!online) {
                rc = acc_apply(total, &g->acc[ATT_A_SUM], live_keys(V, n), &msum, p, &unb);
            } else if (!(rc = dyad_rnd(&total->lo, &lsum, p, 0)) && !(rc = dyad_rnd(&total->hi, &hsum, p, 1)) && !(rc = dyad_rnd(&mass, &msum, p, 1))) {
                rc = online_bound(total, &unb, g, ATT_A_SUM, &mass, OC(GROWTH), P);
            }
            if (unb) soft_unb = 1;
        }
        dyad_free(&lsum);
        dyad_free(&hsum);
        dyad_free(&msum);
    }
out:
    *masked_nan_out = masked_nan;
    *soft_unb_out = soft_unb;
    *pad_unb_out = pad_unb;
    ivl_free(&sc);
    dyad_free(&d1);
    dyad_free(&mass);
    dyad_free(&e1);
    product_sums_free(&ps);
    product_sums_free(&dev_ps);
    return rc;
}

int att_program_row(att_row *w, const att_call *a, const att_prog *g, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0,
                    const unsigned char *out_refs, const unsigned char *lse_ref, unsigned char *tol, unsigned char *lows, unsigned char *highs,
                    unsigned char *flags, unsigned char *lse_tol, unsigned char *lse_low, unsigned char *lse_high, unsigned char *lse_flag,
                    unsigned char *status) {
    const operand *v = a->v;
    size_t S = a->S, Ev = a->Ev;
    int64_t p = g->w;
    int online = g->methods & ATT_M_ONLINE;
    *status = 1;
    int rc = att_row_reserve(w, S, Ev);
    if (rc) return rc;
    int kept_count;
    if ((rc = att_row_logits(w, a, qr, kr0, i, moff, mstep, NULL, &kept_count))) return rc;
    size_t n = kept_count > 0 ? (size_t)kept_count : 0, *keys = w->keys;
    ivl *L = ivls_new(n), *Wt = ivls_new(n), *B = ivls_new(n), *Y = ivls_new(n), *out = ivls_new(Ev), *pad = ivls_new(Ev);
    ivl *R = (g->methods & ATT_M_SPLIT) ? ivls_new(n) : NULL; /* split_kv: the weights' relative parts (att_prog_softmax) */
    unsigned char *V = malloc(n ? n : 1);                    /* the kept keys that may vanish (att_prog_softmax) */
    /* the weights' ends and magnitudes as two-word integers at 2**uW; per column six sums of ATT_QW words: the ends times
       V's integers, the magnitudes times |v| and times V's radii, and the sums of |v| and of the radii */
    w128 *li = malloc((n ? n : 1) * sizeof *li), *hi_i = malloc((n ? n : 1) * sizeof *hi_i), *sz = malloc((n ? n : 1) * sizeof *sz);
    int64_t *lowest = malloc((Ev ? Ev : 1) * sizeof *lowest), *highest = malloc((Ev ? Ev : 1) * sizeof *highest);
    uint64_t *acc = calloc((Ev ? Ev : 1) * 6 * ATT_QW, sizeof *acc);
    dyad *sizes = NULL, *larg = NULL; /* split_kv: per column, its terms' magnitudes' bound and its largest term */
    dyad *own = NULL; /* split_kv: each split's own bound (split_own), owned of them, and where each has one */
    unsigned char *own_ok = NULL, *szero = NULL; /* ... and where each split's merge factor may be 0 (split_merge) */
    size_t owned = 0;
    /* with V's radii, per column: the least v - r and the largest v + r over the kept keys (at 2**ve); and whether an
       element of the column has no bound, or a kept key's value is not finite */
    w128 *rlo = a->vv ? malloc((Ev ? Ev : 1) * sizeof *rlo) : NULL, *rhi = a->vv ? malloc((Ev ? Ev : 1) * sizeof *rhi) : NULL;
    unsigned char *cunb = calloc(Ev ? Ev : 1, 1);
    unsigned char *cnan = calloc(Ev ? Ev : 1, 1); /* ... and whether it may be NaN (Masked keys) */
    /* per column: the value sums' exact lo, hi and mass (before V's radii), the kept keys' least and largest value, and with
       V's radii the deviation sum size r */
    dyad *cd = malloc(6 * (Ev ? Ev : 1) * sizeof *cd), *clo = cd, *chi = cd ? cd + Ev : NULL, *cmass = cd ? cd + 2 * Ev : NULL,
         *vlo = cd ? cd + 3 * Ev : NULL, *vhi = cd ? cd + 4 * Ev : NULL, *dev = cd ? cd + 5 * Ev : NULL;
    if (cd)
        for (size_t z = 0; z < 6 * Ev; z++) dyad_init(&cd[z]);
    if ((g->methods & ATT_M_SPLIT) && (sizes = malloc((Ev ? Ev : 1) * sizeof *sizes)) && (larg = malloc((Ev ? Ev : 1) * sizeof *larg)))
        for (size_t c = 0; c < Ev; c++) {
            dyad_init(&sizes[c]);
            dyad_init(&larg[c]);
        }
    ivl top, total, lse, inv;
    ivl *row_ivls[] = {&top, &total, &lse, &inv};
    for (size_t z = 0; z < sizeof row_ivls / sizeof *row_ivls; z++) ivl_init(row_ivls[z]);
    dyad d1, d2, mass, e1;
    dyad *ds[] = {&d1, &d2, &mass, &e1};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int row_ok = 0, lse_ok = 0, missing = a->uncertain ? FLAG_ANY : 0, nan_row = 0; /* nan_row: the row may be NaN (Masked keys) */
    if (!L || !Wt || !B || !Y || !V || !out || !pad || !li || !hi_i || !sz || !lowest || !highest || !acc ||
        ((g->methods & ATT_M_SPLIT) && (!sizes || !larg || !R)) || !cunb || !cnan || !cd || (a->vv && (!rlo || !rhi))) {
        rc = EXACTREF_NOMEM;
        goto done;
    }
    if (kept_count < 0 || g->unbounded) goto final; /* a NaN row, or a factor without a bound: no bound */
    if (a->uncertain && row_unbounded(a, &g->f64, w, qr, kr0, moff, mstep, &nan_row)) goto final;
    for (size_t m = 0; a->vunb && m < n; m++) /* a kept key's value with no bound leaves its column none */
        for (size_t c = 0; c < Ev; c++) cunb[c] |= a->vunb[(vr0 + keys[m]) * Ev + c];
    att_masked_nan(&nan_row, cnan, w, a, vr0);
    int masked_nan = 0, soft_unb = 0, pad_unb = 0;
    /* split_kv: the kept keys' splits */
    int64_t P = 1;
    if (g->methods & ATT_M_SPLIT)
        for (size_t j = 1; j < n; j++) P += keys[j] / (size_t)g->split != keys[j - 1] / (size_t)g->split;
    rc = att_prog_softmax(L, B, Y, Wt, R, V, &top, &total, &masked_nan, &soft_unb, &pad_unb, w, a, g, n, qr, kr0, moff, mstep, P, 0, NULL);
    nan_row |= masked_nan;
    if (rc) goto final;
    if (!n) { /* every weight is exactly 0: each output is 0 */
        for (size_t c = 0; !rc && c < Ev; c++)
            if (!(rc = dyad_set_si(&out[c].lo, 0, 0))) rc = dyad_set_si(&out[c].hi, 0, 0);
        row_ok = 1;
        goto final;
    }
    /* split_kv: each split's own bound, from the weights before methods' weights rounds them */
    if (sizes && !soft_unb) {
        size_t np = (size_t)P;
        if (!(own = malloc(np * sizeof *own)) || !(own_ok = calloc(np, 1)) || !(szero = malloc(np))) {
            rc = EXACTREF_NOMEM;
            goto final;
        }
        for (size_t z = 0; z < np; z++) dyad_init(&own[z]);
        owned = np;
        if ((rc = split_own(own, own_ok, g, a, keys, n, L, R, V, np))) goto final;
    }
    /* the value sums: the weights (rounded with methods' weights) as words, times V's integers */
    int row_unb = soft_unb, products = !soft_unb;
    int64_t uW = 0;
    if (products && (g->methods & ATT_M_WEIGHTS)) {
        int unb = 0;
        if (online && (rc = dyad_mul(&e1, OC(ABS_W), OC(GROWTH), p, 1))) goto final;
        for (size_t j = 0; !rc && !unb && j < n; j++) {
            if (V[j] == 2) continue; /* a key that surely vanishes holds fl(0) = 0 */
            if (!online) {
                rc = lean_ivl(&Wt[j], &g->lean[ATT_P_WEIGHTS], p, &unb);
            } else if (!(rc = dyad_abs(&d1, &Wt[j].lo)) && !(rc = dyad_mul(&d1, OC(REL_W), &d1, p, 1)) &&
                       !(rc = sub_dir(&Wt[j].lo, &Wt[j].lo, &d1, p, 0)) && !(rc = sub_dir(&Wt[j].lo, &Wt[j].lo, &e1, p, 0)) &&
                       !(rc = dyad_abs(&d1, &Wt[j].hi)) && !(rc = dyad_mul(&d1, OC(REL_W), &d1, p, 1)) &&
                       !(rc = dyad_add(&Wt[j].hi, &Wt[j].hi, &d1, p, 1))) {
                rc = dyad_add(&Wt[j].hi, &Wt[j].hi, &e1, p, 1);
            }
        }
        if (unb) row_unb = products = 0;
    }
    if (rc) goto final;
    /* a kept key whose held weight (online: times its later rescaling factors) may be exactly 0, its interval's lower end at
       most 0 (every kept key where the weights have no interval), times a value entry that may not be finite: NaN (Masked keys) */
    for (size_t j = 0; j < n; j++)
        if (value_special(a, vr0, keys[j]) && (!products || dyad_sign(&Wt[j].lo) <= 0)) value_nan(cnan, a, vr0, keys[j]);
    if (products) {
        const dyad *big = NULL;
        for (size_t j = 0; !rc && j < n; j++) {
            if ((rc = ends_magnitude(&d1, &Wt[j].lo, &Wt[j].hi))) break;
            if (dyad_sign(&d1) > 0 && (!big || dyad_cmp(&d1, &d2) > 0)) {
                rc = dyad_copy(&d2, &d1);
                big = &d2;
            }
        }
        uW = big ? top_bit(big) + 1 - 126 : 0;
        for (size_t j = 0; !rc && j < n; j++) {
            if (!(rc = to_w128(&li[j], &Wt[j].lo, uW, 0))) rc = to_w128(&hi_i[j], &Wt[j].hi, uW, 1);
            w128 a1 = w128_hi(li[j]) >> 63 ? w128_sub(w128_zero(), li[j]) : li[j], a2 = w128_hi(hi_i[j]) >> 63 ? w128_sub(w128_zero(), hi_i[j]) : hi_i[j];
            sz[j] = w128_less(a1, a2) ? a2 : a1;
        }
    }
    if (rc) goto final;
    /* with flush, V's integers below limit are values below 2**emin */
    uint64_t limit = g->flush && g->emin > v->exponent ? (g->emin - v->exponent < 64 ? (uint64_t)1 << (g->emin - v->exponent) : UINT64_MAX) : 0;
    if (products || !pad_unb) {
        /* the integer products, exact but for the weights' units; exact sums of the decoded values instead for V's slots past
           one word (values past its range, or not finite), and where the units may matter */
        int exact = v->flagged_count || v->slots_hi;
        if (!exact) {
            for (size_t c = 0; c < Ev; c++) {
                lowest[c] = INT64_MAX;
                highest[c] = INT64_MIN;
            }
            for (size_t j = 0; j < n; j++) {
                const uint64_t *vs = v->slots + (vr0 + keys[j]) * Ev;
                for (size_t c = 0; c < Ev; c++) {
                    int64_t x0 = (int64_t)vs[c];
                    if (x0 < lowest[c]) lowest[c] = x0;
                    if (x0 > highest[c]) highest[c] = x0;
                    if (!products || !x0) continue;
                    uint64_t *col = acc + c * 6 * ATT_QW, mag = x0 < 0 ? (uint64_t)0 - (uint64_t)x0 : (uint64_t)x0;
                    if (mag < limit) {
                        /* a value a flush may read as zero: its product lies between 0 and w v */
                        w128 below = w128_hi(li[j]) >> 63 ? li[j] : w128_zero(), above = w128_hi(hi_i[j]) >> 63 ? w128_zero() : hi_i[j];
                        acc_s128(col, ATT_QW, x0 > 0 ? below : above, x0);
                        acc_s128(col + ATT_QW, ATT_QW, x0 > 0 ? above : below, x0);
                    } else {
                        acc_s128(col, ATT_QW, x0 >= 0 ? li[j] : hi_i[j], x0);
                        acc_s128(col + ATT_QW, ATT_QW, x0 >= 0 ? hi_i[j] : li[j], x0);
                    }
                    acc_wide(col + 2 * ATT_QW, ATT_QW, mag, sz[j]);
                    acc_word(col + 4 * ATT_QW, ATT_QW, 1, mag);
                }
            }
        }
        /* V's radii: the stored value lies within r of v, so v - r and v + r bound the kept keys' values (at 2**ve), and the
           deviation sum size r moves each value sum */
        for (size_t j = 0; !rc && a->vv && j < n; j++)
            for (size_t c = 0; c < Ev; c++) {
                size_t at = (vr0 + keys[j]) * Ev + c;
                w128 vv = read128(a->vv + 16 * at), r = read128(a->vr + 16 * at), lo = w128_sub(vv, r), hi = w128_add(vv, r);
                if (!j || s128_lt(lo, rlo[c])) rlo[c] = lo;
                if (!j || s128_lt(rhi[c], hi)) rhi[c] = hi;
                if (products && !exact) {
                    uint64_t *col = acc + c * 6 * ATT_QW;
                    acc_u128(col + 3 * ATT_QW, ATT_QW, sz[j], r);
                    acc_wide(col + 5 * ATT_QW, ATT_QW, 1, r);
                }
            }
        if (!exact) {
            int64_t ev = v->exponent, pout = (int64_t)a->out.man_bits + 1;
            for (size_t c = 0; !rc && c < Ev; c++) {
                uint64_t *col = acc + c * 6 * ATT_QW;
                if ((rc = dyad_set_si(&vlo[c], lowest[c], ev)) || (rc = dyad_set_si(&vhi[c], highest[c], ev)) || !products) continue;
                if ((rc = dyad_signed_words(&clo[c], col, ATT_QW, uW + ev)) || (rc = dyad_signed_words(&chi[c], col + ATT_QW, ATT_QW, uW + ev)) ||
                    (rc = dyad_words(&cmass[c], col + 2 * ATT_QW, ATT_QW, uW + ev)) || (a->vv && (rc = dyad_words(&dev[c], col + 3 * ATT_QW, ATT_QW, uW + a->ve))))
                    break;
                /* the units may matter where the ends' and sizes' units, 2**uW (2 sum |v| + sum r), exceed 2**-(p + 40) of the
                   sums' larger end (p the output's precision): then the exact sums */
                if ((rc = dyad_words(&d1, col + 4 * ATT_QW, ATT_QW, uW + ev + 1)) ||
                    (a->vv && ((rc = dyad_words(&d2, col + 5 * ATT_QW, ATT_QW, uW + a->ve)) || (rc = dyad_add_exact(&d1, &d1, &d2)))) ||
                    (rc = ends_magnitude(&d2, &clo[c], &chi[c])))
                    break;
                if ((rc = dyad_shift(&d1, pout + 40))) break;
                if (dyad_sign(&d1) && dyad_cmp(&d1, &d2) > 0) exact = 1;
            }
        }
        if (!rc && exact) {
            rc = general_values(clo, chi, cmass, vlo, vhi, cunb, a, keys, n, vr0, Wt, products, g->flush, g->emin);
            /* with V's radii, the deviation sum size r, exactly */
            for (size_t c = 0; !rc && products && a->vv && c < Ev; c++) rc = dyad_set_si(&dev[c], 0, 0);
            int64_t f = tiny_floor(Wt, n);
            for (size_t j = 0; !rc && products && a->vv && j < n; j++) {
                ivl wj;
                ivl_init(&wj);
                if (!(rc = tiny_widened(&wj, &Wt[j], f))) rc = ends_magnitude(&d1, &wj.lo, &wj.hi);
                ivl_free(&wj);
                if (rc) break;
                for (size_t c = 0; !rc && c < Ev; c++)
                    if (!(rc = dyad_s128(&d2, read128(a->vr + 16 * ((vr0 + keys[j]) * Ev + c)), a->ve)) && !(rc = dyad_mul(&d2, &d1, &d2, INT64_C(1) << 20, 1)))
                        rc = dyad_add_exact(&dev[c], &dev[c], &d2);
            }
        }
        for (size_t c = 0; !rc && a->vv && c < Ev; c++)
            if (!(rc = dyad_s128(&vlo[c], rlo[c], a->ve))) rc = dyad_s128(&vhi[c], rhi[c], a->ve);
    }
    if (rc) goto final;
    if (products) {
        int unb = 0;
        for (size_t c = 0; !rc && !unb && c < Ev; c++) {
            /* with V's radii, the deviation sum size r moves each end and adds to the mass, exactly */
            if ((rc = a->vv ? dyad_copy(&e1, &dev[c]) : dyad_set_si(&e1, 0, 0)) || (rc = dyad_neg(&d2, &e1)) ||
                (rc = dyad_add_exact(&d1, &clo[c], &d2)) || (rc = dyad_rnd(&out[c].lo, &d1, p, 0)) || (rc = dyad_add_exact(&d1, &chi[c], &e1)) ||
                (rc = dyad_rnd(&out[c].hi, &d1, p, 1)) || (rc = dyad_add_exact(&e1, &cmass[c], &e1)) || (rc = dyad_rnd(&mass, &e1, p, 1)))
                break;
            if (!online) {
                rc = acc_apply(&out[c], &g->acc[ATT_A_VALUES], live_keys(V, n), &mass, p, &unb);
            } else {
                /* each term before its rescaling factors is at most growth vmax (and its weights rounding), vmax the largest
                   |v| (|v| + r with V's radii) */
                if ((rc = ends_magnitude(&d2, &vlo[c], &vhi[c])) || (rc = dyad_mul(&d2, OC(GROWTH), &d2, p, 1))) break;
                if ((g->methods & ATT_M_WEIGHTS) && (!(rc = dyad_mul(&d1, OC(GROW_W), &d2, p, 1)))) rc = dyad_add(&d2, &d2, &d1, p, 1);
                if (!rc) rc = online_bound(&out[c], &unb, g, ATT_A_VALUES, &mass, &d2, P);
                if (!rc && !unb && sizes) {
                    /* split_kv: the bound on the terms' magnitudes, and the largest term */
                    ivl x;
                    ivl_init(&x);
                    if (!(rc = dyad_rnd(&x.lo, &e1, p, 0)) && !(rc = dyad_copy(&x.hi, &mass)) && !(rc = online_bound(&x, &unb, g, ATT_A_VALUES, &mass, &d2, P)) &&
                        !(rc = dyad_copy(&sizes[c], &x.hi)))
                        rc = dyad_copy(&larg[c], &d2);
                    ivl_free(&x);
                }
            }
        }
        if (unb) row_unb = 1;
    }
    if (rc) goto final;
    if (sizes) { /* split_kv: the merge gives the outputs and logsumexp */
        if (!soft_unb) rc = split_merge(out, &row_ok, &lse, &lse_ok, szero, g, a, keys, n, L, V, &top, &total, sizes, larg, vlo, vhi, own, own_ok, P, !row_unb);
        /* a split whose factor s_i may be 0 times an infinite O_i: its kept keys' values that may not be finite (Masked keys) */
        for (size_t j = 0, i = 0; !rc && !soft_unb && j < n; j++) {
            i += j && keys[j] / (size_t)g->split != keys[j - 1] / (size_t)g->split;
            if (szero[i] && value_special(a, vr0, keys[j])) value_nan(cnan, a, vr0, keys[j]);
        }
        goto final;
    }
    /* the output roundings and the normalization */
    if (!row_unb) {
        int unb = 0;
        if (dyad_sign(&total.lo) <= 0) unb = 1;
        if (!unb && (g->methods & ATT_M_RECIPROCAL)) {
            if (!(rc = dyad_set_si(&d1, 1, 0)) && !(rc = dyad_div(&inv.lo, &d1, &total.hi, p, 0)) && !(rc = dyad_div(&inv.hi, &d1, &total.lo, p, 1)))
                rc = lean_ivl(&inv, &g->lean[ATT_P_RECIPROCAL], p, &unb);
        }
        for (size_t c = 0; !rc && !unb && c < Ev; c++) {
            if ((rc = lean_ivl(&out[c], &g->lean[ATT_P_VALUES_OUT], p, &unb)) || unb) break;
            rc = (g->methods & ATT_M_RECIPROCAL) ? ivl_mul(&out[c], &out[c], &inv, p) : ivl_div(&out[c], &out[c], &total, p);
            if (!rc) rc = lean_ivl(&out[c], &g->lean[ATT_P_NORMALIZE], p, &unb);
        }
        row_ok = !rc && !unb;
    }
    if (rc) goto final;
    /* the fully padded rows' bound, intersected with the row's */
    if (!pad_unb) {
        int none = 1;
        rc = padded_row(pad, &none, g, a, Y, V, n, vlo, vhi);
        for (size_t c = 0; !rc && !none && c < Ev; c++) {
            if (!row_ok) {
                rc = ivl_copy(&out[c], &pad[c]);
            } else {
                if (dyad_cmp(&pad[c].lo, &out[c].lo) > 0) rc = dyad_copy(&out[c].lo, &pad[c].lo);
                if (!rc && dyad_cmp(&pad[c].hi, &out[c].hi) < 0) rc = dyad_copy(&out[c].hi, &pad[c].hi);
            }
        }
        if (!rc && !none) row_ok = 1;
    }
    if (rc) goto final;
    /* logsumexp: the maximum (in natural units) plus the log of the row sum */
    if (!soft_unb) {
        int unb = 0;
        rc = ivl_copy(&lse, &top);
        if (!rc && (g->methods & ATT_M_EXP2)) {
            ivl ln2;
            ln2.lo = g->ln2_lo;
            ln2.hi = g->ln2_hi;
            if (!(rc = ivl_mul(&lse, &lse, &ln2, p))) rc = lean_ivl(&lse, &g->lean[ATT_P_EXP2_LSE], p, &unb);
        }
        ivl lg;
        ivl_init(&lg);
        if (!rc && !unb && dyad_sign(&total.lo) <= 0) unb = 1;
        if (!rc && !unb && !(rc = log_end(&lg.lo, &total.lo, p, 0)) && !(rc = log_end(&lg.hi, &total.hi, p, 1)))
            rc = lean_ivl(&lg, &g->lean[ATT_P_LOG], p, &unb);
        if (!rc && !unb && (g->methods & ATT_M_FAST_LOG)) rc = widen_by(&lg, OC(FLOOR), p);
        if (!rc && !unb && !(rc = ivl_add(&lse, &lse, &lg, p))) rc = lean_ivl(&lse, &g->lean[ATT_P_ADD], p, &unb);
        ivl_free(&lg);
        lse_ok = !rc && !unb;
    }
final:
    if (rc == EXACTREF_OVERFLOW) { /* an exponent beyond the dyadics' range: the row has no bound */
        rc = 0;
        row_ok = lse_ok = 0;
    }
    /* Lemma 3 per element, and for logsumexp */
    if (!rc) {
        bigint tb, lb, hb, ref;
        big_init(&tb);
        big_init(&lb);
        big_init(&hb);
        big_init(&ref);
        for (size_t c = 0; !rc && c < Ev; c++) {
            int fl = 0;
            int ok = row_ok && !cunb[c]; /* where an input has no bound, the output may be any value or flag */
            if (!(rc = bytes_load(&ref, out_refs + c * a->out_size, a->out_size, 0)) &&
                !(rc = final_bounds(&tb, &lb, &hb, &fl, &a->out, &g->f64, a->rnd, a->ovf, ok ? &out[c].lo : NULL, ok ? &out[c].hi : NULL, &ref, p,
                                    (ok ? 0 : missing) | (nan_row || cnan[c] ? FLAG_NAN : 0)))) {
                bytes_store(tol + 8 * c, 8, &tb);
                bytes_store(lows + c * a->out_size, a->out_size, &lb);
                bytes_store(highs + c * a->out_size, a->out_size, &hb);
                flags[c] = (unsigned char)fl;
            }
        }
        if (!rc) {
            int fl = 0;
            if (!(rc = bytes_load(&ref, lse_ref, a->lse_size, 0)) &&
                !(rc = final_bounds(&tb, &lb, &hb, &fl, &a->lse, &g->f64, a->lrnd, a->lovf, lse_ok ? &lse.lo : NULL, lse_ok ? &lse.hi : NULL, &ref, p,
                                    (lse_ok ? 0 : missing) | (nan_row ? FLAG_NAN : 0)))) {
                bytes_store(lse_tol, 8, &tb);
                bytes_store(lse_low, a->lse_size, &lb);
                bytes_store(lse_high, a->lse_size, &hb);
                *lse_flag = (unsigned char)fl;
                *status = 0;
            }
        }
        big_free(&tb);
        big_free(&lb);
        big_free(&hb);
        big_free(&ref);
    }
done:
    ivls_free(L, n);
    ivls_free(Wt, n);
    ivls_free(R, n);
    free(V);
    ivls_free(B, n);
    ivls_free(Y, n);
    ivls_free(out, Ev);
    ivls_free(pad, Ev);
    free(li);
    free(hi_i);
    free(lowest);
    free(highest);
    free(sz);
    free(acc);
    if (sizes && larg)
        for (size_t c = 0; c < Ev; c++) {
            dyad_free(&sizes[c]);
            dyad_free(&larg[c]);
        }
    free(sizes);
    free(larg);
    for (size_t z = 0; z < owned; z++) dyad_free(&own[z]);
    free(own);
    free(own_ok);
    free(szero);
    free(rlo);
    free(rhi);
    free(cunb);
    free(cnan);
    if (cd)
        for (size_t z = 0; z < 6 * Ev; z++) dyad_free(&cd[z]);
    free(cd);
    for (size_t z = 0; z < sizeof row_ivls / sizeof *row_ivls; z++) ivl_free(row_ivls[z]);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* ---------------------------------------------------------------- the constants from the settings
   (proofs/scaled_dot_product_attention.md and attention_algorithms.md): Lemma 1's units at each boundary,
   Lemma 2's and T's terms per term count, the closed form's constants and sums per count, online softmax's
   constants and the factor the scores are multiplied by, at the bounds' precision w (upper bounds rounded up,
   lower ones down). */

void att_settings_init(att_settings *s) {
    memset(s, 0, sizeof *s);
    for (int i = 0; i < ATT_SC_COUNT; i++) dyad_init(&s->a[i].slack);
    big_init(&s->num);
    big_init(&s->den);
    dyad_init(&s->factor_lo);
    dyad_init(&s->factor_hi);
    dyad_init(&s->scale_lo);
    dyad_init(&s->scale_hi);
}

void att_settings_free(att_settings *s) {
    for (int i = 0; i < ATT_SC_COUNT; i++) dyad_free(&s->a[i].slack);
    big_free(&s->num);
    big_free(&s->den);
    dyad_free(&s->factor_lo);
    dyad_free(&s->factor_hi);
    dyad_free(&s->scale_lo);
    dyad_free(&s->scale_hi);
}

static int is_nearest(int rnd) { return rnd == ROUND_NEAREST_EVEN || rnd == ROUND_NEAREST_AWAY; }

/* An accumulator's units (foundations.md, The normal standard, Units): r = 2**-p, a = 2**(emin - p) and a2 = 0 (with flush both 2**emin), and
   theta, its largest value plus half its ulp; EXACTREF_NEAREST, *bad its rounding, where it does not round to
   nearest (_rounding.nearest_accumulator) */
int acc_units(dyad *r, dyad *a, dyad *a2, dyad *theta, int *bad, const floatfmt *f, int rnd, int flush) {
    if (!is_nearest(rnd)) {
        *bad = rnd;
        return EXACTREF_NEAREST;
    }
    int64_t p = f->man_bits + 1;
    int rc = 0;
    if (r) rc = dyad_set_si(r, 1, -p);
    if (!rc && a) rc = dyad_set_si(a, 1, flush ? f->emin : f->emin - p);
    if (!rc && a2) rc = flush ? dyad_set_si(a2, 1, f->emin) : dyad_set_si(a2, 0, 0);
    if (!rc && theta) rc = floatfmt_threshold(theta, NULL, f);
    return rc;
}

/* Lemma 2's (or with mma Lemma T's) terms for n terms at an accumulation:
   (gamma, constant, largest finite value), *bounded 0 where they do not exist */
int acc_terms(dyad *gamma, dyad *constant, dyad *largest, int *bounded, const att_acc_set *a, int64_t n, int64_t w, int flush) {
    if (a->kind == ATT_SA_MMA) {
        /* mma_terms with the mma's count, or n; float32's largest value, or the reduction format's if smaller */
        floatfmt f32;
        int rc = floatfmt_set(&f32, 8, 23, SPECIAL_IEEE, NULL, 1);
        dyad theta, other;
        dyad_init(&theta);
        dyad_init(&other);
        if (!rc) rc = floatfmt_threshold(&theta, largest, &f32);
        if (!rc && a->has_reduction && !(rc = floatfmt_threshold(&theta, &other, &a->reduction)) && dyad_cmp(&other, largest) < 0) rc = dyad_copy(largest, &other);
        if (!rc)
            rc = mma_terms(gamma, constant, a->products, a->fraction_bits, a->exponent_floor, a->split, a->has_reduction ? &a->reduction : NULL, &a->slack,
                           a->count >= 0 ? a->count : n, 0, -1, w);
        dyad_free(&theta);
        dyad_free(&other);
        *bounded = !rc;
        return rc;
    }
    if (n < 0) return EXACTREF_INTERNAL;
    int64_t depth, roundings;
    any_order_counts(&depth, &roundings, n, flush, 0);
    return growth_terms(gamma, constant, largest, bounded, &a->f, a->rnd, depth, roundings, w, flush);
}

/* The factor the scores are multiplied by: the scale held in the compute format (scale_factor's rounding of the given
   scale, 0 without terms, else of 1/sqrt(E)), or with exp2 the exact sigma times log2(e) held within exp2_scale's
   ulps; s->bounded 0 where a rounding may overflow. */
int att_settings_prepare(att_settings *s) {
    int64_t w = s->w;
    ivl x, l2e, ln2;
    ivl_init(&x);
    ivl_init(&l2e);
    ivl_init(&ln2);
    round_bound lean;
    lean_init(&lean);
    dyad one, root;
    dyad_init(&one);
    dyad_init(&root);
    int rc = dyad_set_si(&one, 1, 0), unbounded = 0;
    /* sigma exactly where it is dyadic, else enclosed */
    int exact_given = s->den.n != 0;
    if (!rc && exact_given) {
        dyad a, b;
        dyad_init(&a);
        dyad_init(&b);
        if (!(rc = big_copy(&a.m, &s->num)) && !(rc = big_copy(&b.m, &s->den))) {
            if (big_bits(&s->den) - 1 == big_trailing(&s->den)) { /* a power of 2 */
                if (!(rc = dyad_div(&x.lo, &a, &b, INT64_C(1) << 20, 0))) rc = dyad_copy(&x.hi, &x.lo);
            } else if (!(rc = dyad_div(&x.lo, &a, &b, w, 0))) {
                rc = dyad_div(&x.hi, &a, &b, w, 1);
            }
        }
        dyad_free(&a);
        dyad_free(&b);
    }
    /* the scale factor: rounded from the given scale, 0 without terms, else from 1/sqrt(E) (1/root exactly where E is a square) */
    if (!rc && (exact_given || s->E > 0)) {
        if (!exact_given) {
            uint64_t r = 0;
            while ((r + 1) * (r + 1) <= (uint64_t)s->E) r++;
            if (r * r == (uint64_t)s->E) {
                if (!(r & (r - 1))) {
                    if (!(rc = dyad_set_si(&x.lo, 1, -(int64_t)word_bits(r) + 1))) rc = dyad_copy(&x.hi, &x.lo);
                } else if (!(rc = dyad_ratio(&x.lo, 1, (int64_t)r, w, 0))) {
                    rc = dyad_ratio(&x.hi, 1, (int64_t)r, w, 1);
                }
            } else if (!(rc = dyad_set_si(&root, s->E, 0)) && !(rc = dyad_sqrt(&ln2.hi, &root, w + 8, 1)) && !(rc = dyad_sqrt(&ln2.lo, &root, w + 8, 0)) &&
                       !(rc = dyad_div(&x.lo, &one, &ln2.hi, w, 0))) {
                rc = dyad_div(&x.hi, &one, &ln2.lo, w, 1);
            }
        }
        if (!rc && !(rc = lean_of(&lean, &s->r[ATT_SR_SCALE], s->flush))) rc = lean_ends(&x.lo, &x.hi, &unbounded, &lean, w);
    } else if (!rc) {
        if (!(rc = dyad_set_si(&x.lo, 0, 0))) rc = dyad_set_si(&x.hi, 0, 0);
    }
    /* the scale held in the compute format (the backward's scale_gradients and scale_ds multiply by it) */
    s->scale_bounded = !rc && !unbounded;
    if (!rc && s->scale_bounded && !(rc = dyad_copy(&s->scale_lo, &x.lo))) rc = dyad_copy(&s->scale_hi, &x.hi);
    if (!rc && !unbounded && s->exp2) {
        /* sigma exactly (the given scale, else 1/sqrt(E) enclosed) times log2(e), held within exp2_scale's ulps */
        if (!exact_given) {
            if (!s->E) {
                rc = dyad_set_si(&x.lo, 0, 0);
                if (!rc) rc = dyad_set_si(&x.hi, 0, 0);
            } else if (!(rc = dyad_set_si(&root, s->E, 0)) && !(rc = dyad_sqrt(&ln2.hi, &root, w + 8, 1)) && !(rc = dyad_sqrt(&ln2.lo, &root, w + 8, 0)) &&
                       !(rc = dyad_div(&x.lo, &one, &ln2.hi, w, 0))) {
                rc = dyad_div(&x.hi, &one, &ln2.lo, w, 1);
            }
        } else {
            dyad a, b;
            dyad_init(&a);
            dyad_init(&b);
            if (!(rc = big_copy(&a.m, &s->num)) && !(rc = big_copy(&b.m, &s->den))) {
                if (big_bits(&s->den) - 1 == big_trailing(&s->den)) {
                    if (!(rc = dyad_div(&x.lo, &a, &b, INT64_C(1) << 20, 0))) rc = dyad_copy(&x.hi, &x.lo);
                } else if (!(rc = dyad_div(&x.lo, &a, &b, w, 0))) {
                    rc = dyad_div(&x.hi, &a, &b, w, 1);
                }
            }
            dyad_free(&a);
            dyad_free(&b);
        }
        if (!rc && !(rc = ln2_bounds(&ln2.lo, &ln2.hi, w)) && !(rc = dyad_div(&l2e.lo, &one, &ln2.hi, w, 0)) && !(rc = dyad_div(&l2e.hi, &one, &ln2.lo, w, 1)) &&
            !(rc = ivl_mul(&x, &x, &l2e, w)) && !(rc = lean_of(&lean, &s->r[ATT_SR_EXP2_SCALE], s->flush)))
            rc = lean_ends(&x.lo, &x.hi, &unbounded, &lean, w);
    }
    s->bounded = !rc && !unbounded;
    if (!rc && s->bounded && !(rc = dyad_copy(&s->factor_lo, &x.lo))) rc = dyad_copy(&s->factor_hi, &x.hi);
    ivl_free(&x);
    ivl_free(&l2e);
    ivl_free(&ln2);
    lean_free(&lean);
    dyad_free(&one);
    dyad_free(&root);
    return rc;
}

/* Online softmax's constants for one row of S keys (attention_algorithms.md, online_softmax) */
typedef struct {
    int64_t n, nb;
    dyad rel_shift, abs_shift, up, down, own, growth, g_min, eta, k, rescale, h_factor;
    int values;              /* the value sums by tensor-core steps: mma's G, H, largest, k and rescale */
    dyad mma[5];
} att_online;

#define ONLINE_DYADS(o) \
    {&(o)->rel_shift, &(o)->abs_shift, &(o)->up, &(o)->down, &(o)->own, &(o)->growth, &(o)->g_min, &(o)->eta, &(o)->k, &(o)->rescale, &(o)->h_factor, \
     &(o)->mma[0], &(o)->mma[1], &(o)->mma[2], &(o)->mma[3], &(o)->mma[4]}

static void online_init(att_online *o) {
    memset(o, 0, sizeof *o);
    dyad *all[] = ONLINE_DYADS(o);
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) dyad_init(all[z]);
}

static void online_free(att_online *o) {
    dyad *all[] = ONLINE_DYADS(o);
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) dyad_free(all[z]);
}

/* whether two boundaries' settings are equal (a dataclass's ==) */
static int same_setting(const round_to *a, const round_to *b) {
    return a->active == b->active && a->rnd == b->rnd && a->flush == b->flush && a->ulps_num == b->ulps_num && a->ulps_den == b->ulps_den &&
           a->f.exp_bits == b->f.exp_bits && a->f.man_bits == b->f.man_bits && a->f.bias == b->f.bias && a->f.special == b->f.special;
}

/* *bounded 0 where the row sum's terms do not exist */
static int online_of(att_online *o, int *bounded, const att_settings *s, int64_t S) {
    int64_t w = s->w;
    int flush = s->flush;
    const round_to *shift = &s->r[ATT_SR_SHIFT], *exp = &s->r[ATT_SR_EXP];
    const round_to *factor = s->r[ATT_SR_RESCALE].active ? &s->r[ATT_SR_RESCALE] : exp; /* the rescaling factors' exp */
    const att_acc_set *total = &s->a[ATT_SC_SUM];
    *bounded = 0;
    o->nb = s->block_size ? (S + s->block_size - 1) / s->block_size : S;
    int64_t block = s->block_size && s->block_size < S ? s->block_size : S, nb = o->nb, later = nb > 1 ? nb - 1 : 0;
    o->n = block + 2 * nb > S + nb + 1 ? block + 2 * nb : S + nb + 1;
    dyad one, rel_exp, h, one_plus, t1, t2, t3, abs_shift, gamma, constant, largest;
    dyad *ds[] = {&one, &rel_exp, &h, &one_plus, &t1, &t2, &t3, &abs_shift, &gamma, &constant, &largest};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int rc = dyad_set_si(&one, 1, 0);
    if (!rc && !(rc = error_units(&o->rel_shift, &abs_shift, &shift->f, shift->rnd, 0, 0, flush)) && !(rc = dyad_set_si(&t1, o->nb, 0)))
        rc = dyad_mul(&o->abs_shift, &t1, &abs_shift, w, 1);
    if (!rc) rc = error_units(&rel_exp, &h, &exp->f, exp->rnd, exp->ulps_num, exp->ulps_den, flush || exp->flush);
    if (!rc) rc = dyad_add(&one_plus, &one, &rel_exp, w, 1);
    if (rc) goto done;
    if (factor == exp || same_setting(factor, exp)) { /* every exp call alike */
        if (!(rc = dyad_power(&o->up, &one_plus, nb, w, 1)) && !(rc = sub_dir(&t1, &one, &rel_exp, w, 0)) && !(rc = dyad_power(&o->down, &t1, nb, w, 0)) &&
            !(rc = dyad_power(&t1, &one_plus, later, w, 1)) && !(rc = dyad_mul(&o->own, &h, &t1, w, 1)) && !(rc = dyad_add(&t1, &one_plus, &h, w, 1)) &&
            !(rc = dyad_power(&o->growth, &t1, nb, w, 1)) && !(rc = dyad_add(&t1, &rel_exp, &h, w, 1)) && !(rc = sub_dir(&t1, &one, &t1, w, 0)) &&
            !(rc = dyad_power(&o->g_min, &t1, nb, w, 0)))
            rc = dyad_copy(&o->h_factor, &h);
    } else { /* the weight's own exp, then nb - 1 rescaling factors with their own units */
        dyad rel_factor, factor_up;
        dyad_init(&rel_factor);
        dyad_init(&factor_up);
        if (!(rc = error_units(&rel_factor, &o->h_factor, &factor->f, factor->rnd, factor->ulps_num, factor->ulps_den, flush || factor->flush)) &&
            !(rc = dyad_add(&t1, &one, &rel_factor, w, 1)) && !(rc = dyad_power(&factor_up, &t1, later, w, 1)) && !(rc = dyad_mul(&o->up, &one_plus, &factor_up, w, 1)) &&
            !(rc = sub_dir(&t1, &one, &rel_exp, w, 0)) && !(rc = sub_dir(&t2, &one, &rel_factor, w, 0)) && !(rc = dyad_power(&t2, &t2, later, w, 0)) &&
            !(rc = dyad_mul(&o->down, &t1, &t2, w, 0)) && !(rc = dyad_mul(&o->own, &h, &factor_up, w, 1)) && !(rc = dyad_add(&t1, &one_plus, &h, w, 1)) &&
            !(rc = dyad_add(&t2, &one, &rel_factor, w, 1)) && !(rc = dyad_add(&t2, &t2, &o->h_factor, w, 1)) && !(rc = dyad_power(&t2, &t2, later, w, 1)) &&
            !(rc = dyad_mul(&o->growth, &t1, &t2, w, 1)) && !(rc = dyad_add(&t1, &rel_exp, &h, w, 1)) && !(rc = sub_dir(&t1, &one, &t1, w, 0)) &&
            !(rc = dyad_add(&t2, &rel_factor, &o->h_factor, w, 1)) && !(rc = sub_dir(&t2, &one, &t2, w, 0))) {
            if (dyad_sign(&t2) > 0) {
                if (!(rc = dyad_power(&t2, &t2, later, w, 0))) rc = dyad_mul(&o->g_min, &t1, &t2, w, 0);
            } else {
                rc = dyad_set_si(&o->g_min, 0, 0);
            }
        }
        dyad_free(&rel_factor);
        dyad_free(&factor_up);
    }
    if (rc) goto done;
    /* the row sum's terms at n + 1, eta = (n + 1) times its absolute unit, K and the rescalings' part */
    int has;
    if ((rc = acc_terms(&gamma, &constant, &largest, &has, total, o->n + 1, w, flush)) || !has) goto done;
    if ((rc = error_units(&t1, &t2, &total->f, total->rnd, 0, 0, flush)) || (rc = dyad_set_si(&t1, o->n + 1, 0)) || (rc = dyad_mul(&o->eta, &t1, &t2, w, 1)) ||
        (rc = dyad_add(&t1, &one, &gamma, w, 1)) || (rc = dyad_mul(&o->k, &o->growth, &t1, w, 1)) || (rc = dyad_set_si(&t1, later, 0)) ||
        (rc = dyad_mul(&t1, &t1, &o->h_factor, w, 1)) || (rc = dyad_mul(&o->rescale, &t1, &o->k, w, 1)))
        goto done;
    /* tensor-core value sums: a term passes its block's steps and every later one's, ceil(S / N) + nb - 1 in all, and two float32
       operations per later block */
    const att_acc_set *values = &s->a[ATT_SC_VALUES];
    if (values->kind == ATT_SA_MMA) {
        floatfmt f32;
        if ((rc = floatfmt_set(&f32, 8, 23, SPECIAL_IEEE, NULL, 1)) || (rc = floatfmt_threshold(&t1, &o->mma[2], &f32))) goto done;
        if (values->has_reduction && !(rc = floatfmt_threshold(&t1, &t2, &values->reduction)) && dyad_cmp(&t2, &o->mma[2]) < 0) rc = dyad_copy(&o->mma[2], &t2);
        int64_t steps = (S + values->products - 1) / values->products + nb - 1;
        if (!rc)
            rc = mma_terms(&o->mma[0], &o->mma[1], values->products, values->fraction_bits, values->exponent_floor, values->split,
                           values->has_reduction ? &values->reduction : NULL, &values->slack, S, 2 * nb, steps, w);
        if (!rc && !(rc = dyad_add(&t1, &one, &o->mma[0], w, 1)) && !(rc = dyad_mul(&o->mma[3], &o->growth, &t1, w, 1)) && !(rc = dyad_set_si(&t1, later, 0)) &&
            !(rc = dyad_mul(&t1, &t1, &o->h_factor, w, 1)))
            rc = dyad_mul(&o->mma[4], &t1, &o->mma[3], w, 1);
        o->values = 1;
    }
    if (!rc) *bounded = 1;
done:
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* The closed form's constants (The closed form): *applicable 0 where it does not
   cover the call (tensor-core sums, splits, a boundary without a rounding setting, an exact accumulation, unbounded
   online constants, rounded weights past their format); *bad the rounding of an accumulator that does not round to
   nearest with EXACTREF_NEAREST. t's flush, emin, f64 and scores' bounds are set here too (att_tol_prepare). */
int att_tol_build(att_tol *t, int *applicable, int *bad, const att_settings *s, int normal) {
    *applicable = 0;
    if (s->has_mma || s->split || !s->bounded) return 0;
    int64_t w = s->w;
    int backward = s->backward, flush = s->flush, online = s->online && !backward, rc = 0, weights = s->has_weights && !online && !backward;
    t->normal = normal && !online;
    t->backward = backward;
    t->flush = flush;
    t->emin = s->emin;
    t->p = 128;
    t->w = w;
    t->methods = (online ? ATT_T_ONLINE : 0) | (s->exp2 ? ATT_T_EXP2 : 0) | (weights ? ATT_T_WEIGHTS : 0) | (s->reciprocal ? ATT_T_RECIPROCAL : 0);
    /* the rounding and function boundaries, each with a rounding setting (Lemma 1's units) */
    struct {
        int slot, path, want;
    } bs[] = {{ATT_R_SCORES, ATT_SR_SCORES_OUT, 1},
              {ATT_R_LOGITS, ATT_SR_LOGITS, 1},
              {ATT_R_SHIFT, ATT_SR_SHIFT, 1},
              {ATT_R_EXP, ATT_SR_EXP, 1},
              {ATT_R_BIAS, ATT_SR_BIAS, s->has_bias},
              {ATT_R_WEIGHTS, ATT_SR_WEIGHTS, weights},
              {ATT_R_EXP2_BIAS, ATT_SR_EXP2_BIAS, s->exp2},
              {ATT_R_EXP2_LSE, ATT_SR_EXP2_LSE, s->exp2},
              {ATT_R_VALUES, ATT_SR_VALUES_OUT, !backward},
              {ATT_R_LOG, ATT_SR_LOG, !backward},
              {ATT_R_RECIPROCAL, ATT_SR_RECIPROCAL, s->reciprocal},
              {ATT_R_PROBS, ATT_SR_PROBS, backward}};
    for (size_t z = 0; z < sizeof bs / sizeof *bs; z++) {
        if (!bs[z].want) continue;
        if (!s->r[bs[z].path].active) return 0;
        if ((rc = rnd_of(&t->rnd[bs[z].slot], &s->r[bs[z].path], flush))) return rc;
    }
    if (s->a[ATT_SC_SCORES].kind == ATT_SA_NONE || s->a[ATT_SC_SUM].kind == ATT_SA_NONE || (!backward && s->a[ATT_SC_VALUES].kind == ATT_SA_NONE)) return 0;
    if (s->has_weights && !backward && !s->r[ATT_SR_WEIGHTS].active) return 0;
    for (int z = 0; z < ATT_CONSTS; z++) t->has[z] = 0;
#define SET(name) (t->has[ATT_C_##name] = 1, &t->c[ATT_C_##name])
    dyad r_e, a_e, d1, d2, one;
    dyad *ds[] = {&r_e, &a_e, &d1, &d2, &one};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    att_online o;
    online_init(&o);
    rc = dyad_set_si(&one, 1, 0);
    /* the largest finite values the boundaries may reach, and the thresholds below which they round to finite values */
    struct {
        int slot, path, want;
    } ms[] = {{ATT_C_MAX_SCORES, ATT_SR_SCORES_OUT, 1},
              {ATT_C_MAX_LOGITS, ATT_SR_LOGITS, 1},
              {ATT_C_MAX_SHIFT, ATT_SR_SHIFT, 1},
              {ATT_C_MAX_VALUES, ATT_SR_VALUES_OUT, !backward},
              {ATT_C_MAX_EXP2_LOGSUMEXP, ATT_SR_EXP2_LSE, s->exp2},
              {ATT_C_MAX_RECIPROCAL, ATT_SR_RECIPROCAL, s->reciprocal},
              {ATT_C_MAX_EXP, ATT_SR_EXP, backward},
              {ATT_C_MAX_PROBS, ATT_SR_PROBS, backward}};
    for (size_t z = 0; !rc && z < sizeof ms / sizeof *ms; z++)
        if (ms[z].want && !(rc = units_of(&d1, &d2, &t->c[ms[z].slot], &s->r[ms[z].path], flush))) t->has[ms[z].slot] = 1;
    struct {
        int slot, path, want;
    } ths[] = {{ATT_C_THETA_LOGITS, ATT_SR_LOGITS, 1},
               {ATT_C_THETA_SHIFT, ATT_SR_SHIFT, 1},
               {ATT_C_THETA_BIAS, ATT_SR_BIAS, s->has_bias},
               {ATT_C_THETA_EXP2_BIAS, ATT_SR_EXP2_BIAS, s->exp2}};
    for (size_t z = 0; !rc && z < sizeof ths / sizeof *ths; z++)
        if (ths[z].want && !(rc = acc_units(NULL, NULL, NULL, &t->c[ths[z].slot], bad, &s->r[ths[z].path].f, s->r[ths[z].path].rnd, flush)))
            t->has[ths[z].slot] = 1;
    /* the factor, its magnitude, the logits' units for the masked keys' check, and wmax = 1 + r_e + a_e */
    if (!rc && !(rc = dyad_copy(SET(FACTOR_LO), &s->factor_lo)) && !(rc = dyad_copy(SET(FACTOR_HI), &s->factor_hi)))
        rc = ends_magnitude(SET(FAC), &s->factor_lo, &s->factor_hi);
    if (!rc) rc = units_of(SET(R_L), SET(A_L), NULL, &s->r[ATT_SR_LOGITS], flush);
    if (!rc && !(rc = units_of(&r_e, &a_e, NULL, &s->r[ATT_SR_EXP], flush)) && !(rc = dyad_add(&d1, &one, &r_e, w, 1))) rc = dyad_add(SET(WMAX), &d1, &a_e, w, 1);
    if (!rc && !(rc = dyad_set_si(SET(X_Z), 0, 0)) && !(rc = dyad_set_si(SET(XN1), 0, 0))) rc = dyad_set_si(SET(XN0), 0, 0);
    int64_t online_n = 0;
    if (!rc && online) {
        int bounded;
        if ((rc = online_of(&o, &bounded, s, s->S)) || !bounded) goto done;
        online_n = o.n + 1;
        dyad S_d;
        dyad_init(&S_d);
        if (!(rc = dyad_copy(SET(R_H), &o.rel_shift)) && !(rc = dyad_copy(SET(A_H), &o.abs_shift)) && !(rc = dyad_copy(SET(UP), &o.up)) &&
            !(rc = dyad_copy(SET(OWN), &o.own)) && !(rc = dyad_copy(SET(WMAX), &o.growth)) && !(rc = dyad_set_si(&S_d, s->S, 0)) &&
            /* x_z = rescale (S growth + eta) k, xn0 = rescale eta k, xn1 = rescale S growth k */
            !(rc = dyad_mul(&d1, &S_d, &o.growth, w, 1)) && !(rc = dyad_add(&d2, &d1, &o.eta, w, 1)) && !(rc = dyad_mul(&d2, &d2, &o.k, w, 1)) &&
            !(rc = dyad_mul(SET(X_Z), &o.rescale, &d2, w, 1)) && !(rc = dyad_mul(&d2, &o.eta, &o.k, w, 1)) && !(rc = dyad_mul(SET(XN0), &o.rescale, &d2, w, 1)) &&
            !(rc = dyad_mul(&d2, &d1, &o.k, w, 1)))
            rc = dyad_mul(SET(XN1), &o.rescale, &d2, w, 1);
        dyad_free(&S_d);
    }
    if (!rc) rc = dyad_copy(SET(WMAX_V), &t->c[ATT_C_WMAX]);
    if (!rc && s->has_weights && !backward) {
        /* weights rounded before the value product (online too: a_w grows with the rescaling): wmax_v = wmax (1 + r_w) + a_w at most
           their format's largest value */
        dyad a_w, max_w;
        dyad_init(&a_w);
        dyad_init(&max_w);
        if (!(rc = units_of(SET(R_W), &a_w, &max_w, &s->r[ATT_SR_WEIGHTS], flush))) rc = online ? dyad_mul(SET(A_W), &a_w, &o.growth, w, 1) : dyad_copy(SET(A_W), &a_w);
        if (!rc && !(rc = dyad_add(&d1, &one, &t->c[ATT_C_R_W], w, 1)) && !(rc = dyad_mul(&d1, &t->c[ATT_C_WMAX], &d1, w, 1)))
            rc = dyad_add(&t->c[ATT_C_WMAX_V], &d1, &t->c[ATT_C_A_W], w, 1);
        int past = !rc && dyad_cmp(&t->c[ATT_C_WMAX_V], &max_w) > 0;
        if (!rc && !past && online && !(rc = dyad_add(&d1, &t->c[ATT_C_R_W], &a_w, w, 1)) && !(rc = dyad_add(&d1, &one, &d1, w, 1)))
            rc = dyad_mul(&t->c[ATT_C_XN1], &t->c[ATT_C_XN1], &d1, w, 1);
        dyad_free(&a_w);
        dyad_free(&max_w);
        if (rc || past) goto done;
    }
    /* the row sum's and the value sums' units */
    const att_acc_set *z = &s->a[ATT_SC_SUM], *v = &s->a[ATT_SC_VALUES];
    if (!rc) rc = acc_units(SET(R_Z), SET(A_Z), SET(A2_Z), SET(THETA_Z), bad, &z->f, z->rnd, flush);
    if (!rc && !backward) rc = acc_units(SET(R_V), SET(A_V), SET(A2_V), SET(THETA_V), bad, &v->f, v->rnd, flush);
    const floatfmt *ef = &s->r[ATT_SR_EXP].f;
    if (!rc && !(z->f.man_bits >= ef->man_bits && z->f.emin <= ef->emin && z->f.emax >= ef->emax)) rc = dyad_set_si(SET(NARROW_Z), 1, 0);
    /* fast_log: the log within c ulps of 1, c 2**(1 - p) */
    if (!rc && s->fast_log && s->r[ATT_SR_LOG].active && !(rc = dyad_ratio(SET(LOG_FLOOR), s->r[ATT_SR_LOG].ulps_num, s->r[ATT_SR_LOG].ulps_den, w, 1)))
        rc = dyad_shift(&t->c[ATT_C_LOG_FLOOR], -s->r[ATT_SR_LOG].f.man_bits);
    /* the online sums at the online count, else the sums per count n = 1 .. S */
    if (!rc && online) {
        int hz, hv;
        dyad gz, kz, lz, gv, kv, lv;
        dyad *g6[] = {&gz, &kz, &lz, &gv, &kv, &lv};
        for (int i = 0; i < 6; i++) dyad_init(g6[i]);
        if (!(rc = acc_terms(&gz, &kz, &lz, &hz, z, online_n, w, flush)) && !(rc = acc_terms(&gv, &kv, &lv, &hv, v, online_n, w, flush)) && hz && hv) {
            int slots[6] = {ATT_C_ON_G_Z, ATT_C_ON_K_Z, ATT_C_ON_LIM_Z, ATT_C_ON_G_V, ATT_C_ON_K_V, ATT_C_ON_LIM_V};
            for (int i = 0; !rc && i < 6; i++) {
                rc = dyad_copy(&t->c[slots[i]], g6[i]);
                t->has[slots[i]] = 1;
            }
        }
        for (int i = 0; i < 6; i++) dyad_free(g6[i]);
    }
    t->nmax = (size_t)s->S;
    if (!rc && (!(t->per_n = malloc((t->nmax + 1) * ATT_PER_N * sizeof *t->per_n)) || !(t->per_n_has = calloc((t->nmax + 1) * ATT_PER_N, 1)))) {
        free(t->per_n);
        t->per_n = NULL;
        rc = EXACTREF_NOMEM;
    }
    if (t->per_n)
        for (size_t i = 0; i < (t->nmax + 1) * ATT_PER_N; i++) dyad_init(&t->per_n[i]);
    for (int64_t n = 1; !rc && !online && n <= s->S; n++) {
        dyad *pn = t->per_n + n * ATT_PER_N;
        unsigned char *ph = t->per_n_has + n * ATT_PER_N;
        int64_t n1 = n > 1 ? n - 1 : 0;
        if (t->normal) {
            if (!(rc = growth_up(&pn[ATT_N_GZ], &t->c[ATT_C_R_Z], n1, w))) ph[ATT_N_GZ] = 1;
            if (!rc && !backward && !(rc = growth_up(&pn[ATT_N_GV], &t->c[ATT_C_R_V], n1, w))) ph[ATT_N_GV] = 1;
            continue;
        }
        int has;
        dyad lim;
        dyad_init(&lim);
        if (!(rc = acc_terms(&pn[ATT_N_G_Z], &pn[ATT_N_K_Z], &lim, &has, z, n, w, flush)) && has) ph[ATT_N_G_Z] = ph[ATT_N_K_Z] = 1;
        if (!rc && !backward && !(rc = acc_terms(&pn[ATT_N_G_V], &pn[ATT_N_K_V], &lim, &has, v, n, w, flush)) && has) ph[ATT_N_G_V] = ph[ATT_N_K_V] = 1;
        dyad_free(&lim);
        if (!rc && !(rc = growth_up(&pn[ATT_N_GZ1], &t->c[ATT_C_R_Z], n1, w)) && !(rc = growth_up(&pn[ATT_N_GZN], &t->c[ATT_C_R_Z], n, w)))
            ph[ATT_N_GZ1] = ph[ATT_N_GZN] = 1;
        if (!rc && !backward && !(rc = growth_up(&pn[ATT_N_GV1], &t->c[ATT_C_R_V], n1, w)) && !(rc = growth_up(&pn[ATT_N_GVN], &t->c[ATT_C_R_V], n, w)))
            ph[ATT_N_GV1] = ph[ATT_N_GVN] = 1;
    }
    /* Lemma 1 at normalize and at logsumexp's add */
    if (!rc && !backward && !(rc = lean_of(&t->normalize, &s->r[ATT_SR_NORMALIZE], flush))) rc = lean_of(&t->add, &s->r[ATT_SR_ADD], flush);
    if (!rc) *applicable = 1;
#undef SET
done:
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    online_free(&o);
    return rc;
}

/* An accumulation as the program takes it: none, or the terms for the counts first .. last */
int acc_of(att_acc *c, const att_acc_set *a, int64_t first, int64_t last, int64_t w, int flush) {
    c->kind = a->kind == ATT_SA_NONE ? ATT_ACC_NONE : ATT_ACC_TERMS;
    if (c->kind == ATT_ACC_NONE) return 0;
    c->nmax = (size_t)last;
    if (!(c->terms = malloc(3 * (c->nmax + 1) * sizeof *c->terms)) || !(c->has = calloc(c->nmax + 1, 1))) {
        free(c->terms);
        c->terms = NULL;
        return EXACTREF_NOMEM;
    }
    for (size_t z = 0; z < 3 * (c->nmax + 1); z++) dyad_init(&c->terms[z]);
    int rc = 0;
    for (int64_t n = first; !rc && n <= last; n++) {
        int has;
        if (!(rc = acc_terms(&c->terms[3 * n], &c->terms[3 * n + 1], &c->terms[3 * n + 2], &has, a, n, w, flush))) c->has[n] = (unsigned char)has;
    }
    return rc;
}

/* The step-by-step program's constants, for every call: where the factor has no bound no row has one, and where online
   softmax's constants have none only rows without a kept key have one; *bad as att_tol_build's. */
int att_prog_build(att_prog *g, int *applicable, int *bad, const att_settings *s, int sign) {
    *applicable = 0;
    int64_t w = s->w, S = s->S;
    int flush = s->flush, rc = 0;
    g->w = w;
    g->sign = sign;
    g->flush = flush;
    g->emin = s->emin;
    g->split = s->split;
    g->online_n = g->online_S = 0;
    g->unbounded = !s->bounded;
    g->online_unbounded = 0;
    if (g->unbounded) {
        *applicable = 1;
        return 0;
    }
    int methods = s->has_methods ? ATT_M_METHODS : 0;
    att_online o;
    online_init(&o);
    dyad d1, d2, one;
    dyad_init(&d1);
    dyad_init(&d2);
    dyad_init(&one);
    for (int z = 0; z < ATT_O_CONSTS; z++) g->has[z] = 0;
#define SET(name) (g->has[ATT_O_##name] = 1, &g->o[ATT_O_##name])
    rc = dyad_set_si(&one, 1, 0);
    if (!rc && s->has_methods) {
        if (s->online && !s->backward) {
            /* with split_kv, one split's constants: they do not depend on the number of splits */
            int64_t S1 = s->split && s->split < S ? s->split : S;
            int bounded;
            if ((rc = online_of(&o, &bounded, s, S1))) goto done;
            g->online_unbounded = !bounded;
            methods |= ATT_M_ONLINE | (s->split ? ATT_M_SPLIT : 0);
            g->online_n = o.n;
            g->online_S = S1;
        }
        methods |= (s->exp2 ? ATT_M_EXP2 : 0) | (s->has_weights ? ATT_M_WEIGHTS : 0) | (s->reciprocal ? ATT_M_RECIPROCAL : 0) | (s->fast_log ? ATT_M_FAST_LOG : 0);
    }
    if (!rc && !(rc = dyad_copy(SET(FACTOR_LO), &s->factor_lo))) rc = dyad_copy(SET(FACTOR_HI), &s->factor_hi);
    if (!rc && (methods & ATT_M_ONLINE) && !g->online_unbounded) {
        struct {
            int slot;
            const dyad *x;
        } os[] = {{ATT_O_REL_SHIFT, &o.rel_shift}, {ATT_O_ABS_SHIFT, &o.abs_shift}, {ATT_O_UP, &o.up}, {ATT_O_DOWN, &o.down}, {ATT_O_OWN, &o.own},
                  {ATT_O_GROWTH, &o.growth}, {ATT_O_G_MIN, &o.g_min}, {ATT_O_ETA, &o.eta}, {ATT_O_K, &o.k}, {ATT_O_RESCALE, &o.rescale}};
        for (size_t z = 0; !rc && z < sizeof os / sizeof *os; z++) {
            rc = dyad_copy(&g->o[os[z].slot], os[z].x);
            g->has[os[z].slot] = 1;
        }
        if (!rc && o.values) {
            methods |= ATT_M_MMA_VALUES;
            int slots[5] = {ATT_O_MMA_G, ATT_O_MMA_H, ATT_O_MMA_TOP, ATT_O_MMA_K, ATT_O_MMA_RESCALE};
            for (int z = 0; !rc && z < 5; z++) {
                rc = dyad_copy(&g->o[slots[z]], &o.mma[z]);
                g->has[slots[z]] = 1;
            }
        }
        if (!rc && s->has_weights) {
            const round_to *ws = &s->r[ATT_SR_WEIGHTS];
            if (!(rc = error_units(SET(REL_W), SET(ABS_W), &ws->f, ws->rnd, 0, 0, flush)) && !(rc = error_units(&d1, &d2, &s->weights, ROUND_NEAREST_EVEN, 0, 0, flush)))
                rc = dyad_add(SET(GROW_W), &d1, &d2, w, 1);
        }
    }
    /* fast_log: the log within c ulps of 1, c 2**(1 - p) */
    if (!rc && s->fast_log && s->r[ATT_SR_LOG].active && !(rc = dyad_ratio(SET(FLOOR), s->r[ATT_SR_LOG].ulps_num, s->r[ATT_SR_LOG].ulps_den, w, 1)))
        rc = dyad_shift(&g->o[ATT_O_FLOOR], -s->r[ATT_SR_LOG].f.man_bits);
    if (!rc && s->split) { /* the units of the merge's roundings: split_kv.exp, values.output, reciprocal, normalize, split_kv.product */
        int paths[5] = {ATT_SR_SPLIT_EXP, ATT_SR_VALUES_OUT, ATT_SR_RECIPROCAL, ATT_SR_NORMALIZE, ATT_SR_SPLIT_PRODUCT};
        int slots[5] = {ATT_O_SE_REL, ATT_O_SA_REL, ATT_O_SB_REL, ATT_O_SK_REL, ATT_O_SP_REL};
        for (int z = 0; !rc && z < 5; z++) {
            const round_to *r = &s->r[paths[z]];
            g->has[slots[z]] = g->has[slots[z] + 1] = 1;
            if (!r->active) {
                if (!(rc = dyad_set_si(&g->o[slots[z]], 0, 0))) rc = dyad_set_si(&g->o[slots[z] + 1], 0, 0);
            } else {
                rc = error_units(&g->o[slots[z]], &g->o[slots[z] + 1], &r->f, r->rnd, r->ulps_num, r->ulps_den, flush || r->flush);
            }
        }
    }
    /* the fully padded rows' bound: the default model, and online softmax without rounded weights, reciprocal or splits */
    if (!rc && !s->backward && (!s->has_methods || (s->online && !s->has_weights && !s->reciprocal && !s->split))) {
        const att_acc_set *z = &s->a[ATT_SC_SUM], *v = &s->a[ATT_SC_VALUES];
        const floatfmt *ef = &s->r[ATT_SR_EXP].f;
        methods |= ATT_M_PADDED;
        if (!(rc = acc_units(NULL, NULL, NULL, SET(THETA_LOGITS), bad, &s->r[ATT_SR_LOGITS].f, s->r[ATT_SR_LOGITS].rnd, flush)) &&
            !(rc = units_of(SET(EXP_R), SET(EXP_A), NULL, &s->r[ATT_SR_EXP], flush)) &&
            !(rc = acc_units(SET(Z_R), SET(Z_A), SET(Z_A2), SET(Z_THETA), bad, &z->f, z->rnd, flush)))
            rc = acc_units(SET(V_R), SET(V_A), SET(V_A2), SET(V_THETA), bad, &v->f, v->rnd, flush);
        if (!(z->f.man_bits >= ef->man_bits && z->f.emin <= ef->emin && z->f.emax >= ef->emax)) methods |= ATT_M_NARROW;
    }
    /* the accumulations: the scores at E terms, the sums at every count (the online count with online softmax), the merge's
       at the counts of splits */
    int64_t lo = (methods & ATT_M_ONLINE) ? o.n + 1 : 1, hi = (methods & ATT_M_ONLINE) ? o.n + 1 : S, splits = s->split ? (S + s->split - 1) / s->split : 0;
    if (!rc) rc = acc_of(&g->acc[ATT_A_SCORES], &s->a[ATT_SC_SCORES], s->E, s->E, w, flush);
    if (!rc) rc = acc_of(&g->acc[ATT_A_SUM], &s->a[ATT_SC_SUM], lo, hi, w, flush);
    if (!rc && !s->backward) rc = acc_of(&g->acc[ATT_A_VALUES], &s->a[ATT_SC_VALUES], lo, hi, w, flush);
    if (!rc && s->split && !(rc = acc_of(&g->acc[ATT_A_SPLIT_SUM], &s->a[ATT_SC_SPLIT_SUM], 1, splits, w, flush)))
        rc = acc_of(&g->acc[ATT_A_SPLIT_ACC], &s->a[ATT_SC_SPLIT_ACC], 1, splits, w, flush);
    /* Lemma 1 at each of its boundaries */
    int leans[ATT_P_LEANS] = {ATT_SR_SCORES_OUT, ATT_SR_LOGITS, ATT_SR_SHIFT, ATT_SR_EXP, ATT_SR_WEIGHTS, ATT_SR_VALUES_OUT, ATT_SR_RECIPROCAL,
                              ATT_SR_NORMALIZE, ATT_SR_LOG, ATT_SR_ADD, ATT_SR_EXP2_LSE, ATT_SR_BIAS, ATT_SR_EXP2_BIAS, ATT_SR_SPLIT_SHIFT,
                              ATT_SR_SPLIT_EXP, ATT_SR_SPLIT_LOG, ATT_SR_SPLIT_ADD, ATT_SR_SPLIT_PRODUCT};
    for (int z = 0; !rc && z < ATT_P_LEANS; z++) rc = lean_of(&g->lean[z], &s->r[leans[z]], flush);
    g->methods = methods;
    if (!rc) *applicable = 1;
#undef SET
done:
    online_free(&o);
    dyad_free(&d1);
    dyad_free(&d2);
    dyad_free(&one);
    return rc;
}
