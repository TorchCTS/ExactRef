/* The constants of error bounds (internal): Lemma 1's units of one rounding's error and its bound at a boundary,
   and Lemma 2's growth factor and constant for an accumulation of a method's depth and rounding count
   (proofs/foundations.md). */
#include "exactref.h"

/* r = num / den * 2**k rounded up to 64 bits, for num > 0 and den > 0 without a common factor:
   as the reduced fraction's numerator over its denominator. */
static int scaled_up(dyad *r, int64_t num, int64_t den, int64_t k) {
    while (!(num & 1)) {
        num >>= 1;
        k++;
    }
    while (!(den & 1)) {
        den >>= 1;
        k--;
    }
    dyad a, b;
    dyad_init(&a);
    dyad_init(&b);
    int rc = big_set_si(&a.m, num);
    if (!rc) rc = big_set_si(&b.m, den);
    if (!rc) rc = k >= 0 ? big_shl(&a.m, &a.m, k) : big_shl(&b.m, &b.m, -k);
    if (!rc) rc = dyad_div(r, &a, &b, 64, 1);
    dyad_free(&a);
    dyad_free(&b);
    return rc;
}

/* Upper bounds rel and abs with one rounding's error at most rel |y| + abs for a result y
   (Lemma 1): ulps units in the last place, ulps = ulps_num / ulps_den (half an ulp to nearest and
   one otherwise when ulps_den is 0), since ulps ulp(y) <= 2 ulps (2**-p |y| + 2**(emin - p)). With
   flush, a result below 2**emin may become zero, so abs is 2**emin. */
int error_units(dyad *rel, dyad *abs, const floatfmt *f, int rnd, int64_t ulps_num, int64_t ulps_den, int flush) {
    if (!ulps_den) {
        ulps_num = 1;
        ulps_den = rnd == ROUND_NEAREST_EVEN || rnd == ROUND_NEAREST_AWAY ? 2 : 1;
    }
    int64_t p = f->man_bits + 1;
    int rc = scaled_up(rel, ulps_num, ulps_den, 1 - p);
    if (rc) return rc;
    if (flush) return dyad_set_si(abs, 1, f->emin);
    rc = scaled_up(abs, ulps_num, ulps_den, 1);
    return rc ? rc : dyad_shift(abs, f->emin - p);
}

/* Lemma 2's depth D and rounding count R for n terms summed in any order (foundations.md, Lemma 2): any tree of
   additions or fused multiply-adds, a term perhaps rounded on its own first. A tree of n leaves has height at most
   n - 1, so each term passes at most n roundings: D = n. Only a rounding that involves an exact term can make an
   absolute (underflow) error, and each term passes one, so R = n; with flush any rounding can flush its result, the
   n - 1 merges and at most n single-term roundings, R = 2n - 1. With exact_terms (TERMS_HELD or TERMS_EXACT) no term is
   rounded on its own (foundations.md, Lemma U): D = n - 1. Each term a value of the format (TERMS_HELD), R = 0, or
   n - 1 with flush; with exact terms the format need not hold (TERMS_EXACT), any of the n - 1 merges may take one, so
   R = n - 1. */
void any_order_counts(int64_t *depth, int64_t *roundings, int64_t n, int flush, int exact_terms) {
    int64_t edges = n > 0 ? n - 1 : 0;
    *depth = n <= 0 ? 0 : exact_terms ? edges : n;
    *roundings = n <= 0 ? 0 : exact_terms ? (flush || exact_terms == TERMS_EXACT ? edges : 0) : (flush ? 2 * n - 1 : n);
}

/* Lemma 2's terms for an accumulation of depth D and R roundings in f, rounded up to w bits:
   gamma(D) = D u / (1 - D u) for the relative unit u, the constant R eta (1 + gamma(D)) for the
   absolute unit eta, and f's largest finite value (normalized, as a value object holds it);
   *bounded = 0 when D u >= 1. */
int growth_terms(dyad *gamma, dyad *constant, dyad *largest, int *bounded, const floatfmt *f, int rnd, int64_t depth,
                 int64_t roundings, int64_t w, int flush) {
    dyad rel, abs, t, one;
    dyad *all[] = {&rel, &abs, &t, &one};
    for (int i = 0; i < 4; i++) dyad_init(all[i]);
    int rc = error_units(&rel, &abs, f, rnd, 0, 0, flush);
    if (!rc) rc = dyad_set_si(&one, 1, 0);
    if (!rc) rc = dyad_set_si(&t, depth, 0);
    if (!rc) rc = dyad_mul(&t, &t, &rel, w, 1); /* D u */
    *bounded = !rc && dyad_cmp(&t, &one) < 0;
    if (rc || !*bounded) goto done;
    dyad down = t; /* a read-only view of -D u */
    down.m.neg = 1;
    if ((rc = dyad_add(gamma, &one, &down, w, 0)) || (rc = dyad_div(gamma, &t, gamma, w, 1))) goto done;
    if ((rc = dyad_add(&t, &one, gamma, w, 1))) goto done; /* 1 + gamma */
    if ((rc = dyad_set_si(constant, roundings, 0)) || (rc = dyad_mul(constant, constant, &abs, w, 1)) || (rc = dyad_mul(constant, constant, &t, w, 1)))
        goto done;
    bigint bits;
    big_init(&bits);
    int kind, neg;
    if (!(rc = floatfmt_special(&bits, f, ENCODING_MAX_FINITE, 0)) && !(rc = floatfmt_decode(&kind, &neg, &largest->m, &largest->e, f, &bits))) {
        int64_t trailing = big_trailing(&largest->m);
        if (!(rc = big_shr(&largest->m, &largest->m, trailing, 0))) rc = dyad_shift(largest, trailing);
    }
    big_free(&bits);
done:
    for (int i = 0; i < 4; i++) dyad_free(all[i]);
    return rc;
}

/* g = (1 + r)**d rounded up to w bits (Lemma 2's growth) */
int growth_up(dyad *g, const dyad *r, int64_t d, int64_t w) {
    dyad one;
    dyad_init(&one);
    int rc = dyad_set_si(&one, 1, 0);
    if (!rc && !(rc = dyad_add(&one, &one, r, w, 1))) rc = dyad_power(g, &one, d, w, 1);
    dyad_free(&one);
    return rc;
}

/* ---------------------------------------------------------------- Lemma 1 at a boundary: the setting's units, how far
   the rounding moves a value, and the ends it can give */

/* the ulps Lemma 1 takes at a boundary: the setting's, where its default 1/2 means one ulp for a directed rounding */
static void lemma1_ulps(int64_t *num, int64_t *den, const round_to *r) {
    *num = r->ulps_num;
    *den = r->ulps_den;
    if (*num == 1 && *den == 2 && r->rnd != ROUND_NEAREST_EVEN && r->rnd != ROUND_NEAREST_AWAY) *num = *den = 1;
}

/* (r, a) of one rounding at a boundary, and its format's largest finite value */
int units_of(dyad *rel, dyad *abs, dyad *largest, const round_to *r, int flush) {
    int half = r->ulps_num == 1 && r->ulps_den == 2;
    int rc = error_units(rel, abs, &r->f, r->rnd, half ? 0 : r->ulps_num, half ? 0 : r->ulps_den, flush || r->flush);
    if (!rc && largest) {
        dyad theta;
        dyad_init(&theta);
        rc = floatfmt_threshold(&theta, largest, &r->f);
        dyad_free(&theta);
    }
    return rc;
}

/* Lemma 1's units at one of the closed form's boundaries and its flushing case */
int rnd_of(round_units *t, const round_to *r, int flush) {
    int64_t num, den;
    lemma1_ulps(&num, &den, r);
    t->active = 1;
    t->emin = r->f.emin;
    t->mantissa = r->f.man_bits;
    t->flush = flush || r->flush;
    return dyad_ratio(&t->ulps, num, den, 64, 1);
}

/* Lemma 1's bound at a boundary of the closed form, for a value of magnitude at most high: c ulps
   of high's top bit, plus 2**emin where the boundary flushes (exp_ulps) unless every value is above
   it (at least low > 2**emin; low NULL: not known). */
int boundary_error(dyad *err, const round_units *b, const dyad *high, const dyad *low) {
    if (dyad_sign(high) <= 0) return dyad_set_si(err, 0, 0);
    int64_t top = top_bit(high), at = top > b->emin ? top : b->emin;
    int rc = dyad_copy(err, &b->ulps);
    if (!rc) rc = dyad_shift(err, at - b->mantissa);
    if (!rc && b->flush) {
        dyad tiny;
        dyad_init(&tiny);
        if (!(rc = dyad_set_si(&tiny, 1, b->emin)) && !(low && dyad_cmp(low, &tiny) > 0)) rc = dyad_add(err, err, &tiny, 1 << 20, 1);
        dyad_free(&tiny);
    }
    return rc;
}

void lean_init(round_bound *l) {
    dyad_init(&l->ulps);
    dyad_init(&l->largest);
    dyad_init(&l->threshold);
}

void lean_free(round_bound *l) {
    dyad_free(&l->ulps);
    dyad_free(&l->largest);
    dyad_free(&l->threshold);
}

/* Lemma 1 at a boundary (foundations.md, Lemma 1), flush the option's */
int lean_of(round_bound *l, const round_to *r, int flush) {
    l->active = r->active;
    l->flush = r->active && (flush || r->flush);
    l->has_threshold = 0;
    l->emin = l->mantissa = 0;
    if (!r->active) return 0;
    int64_t num, den;
    lemma1_ulps(&num, &den, r);
    l->emin = r->f.emin;
    l->mantissa = r->f.man_bits;
    l->has_threshold = num == 1 && den == 2; /* the threshold for one correct rounding to nearest */
    int rc = dyad_ratio(&l->ulps, num, den, 64, 1);
    return rc ? rc : floatfmt_threshold(&l->threshold, &l->largest, &r->f);
}

/* x rounded down, or up with up, to the format's grid at its top bit (a unit of at least 2**(emin - mantissa)) */
int grid_end(dyad *x, int64_t emin, int64_t mantissa, int up) {
    if (!x->m.n) return 0;
    int64_t top = top_bit(x), q = (top > emin ? top : emin) - mantissa;
    if (q <= x->e) return 0;
    int rc = big_shr(&x->m, &x->m, q - x->e, up);
    x->e = x->m.n ? q : 0;
    return rc;
}

/* With flush, a point of [lo, hi] below 2**emin in magnitude may become zero: err gains 2**emin,
   rounded up to 64 bits. */
static int flush_error(dyad *err, const dyad *lo, const dyad *hi, int64_t emin) {
    int below;
    if (dyad_sign(lo) <= 0 && dyad_sign(hi) >= 0) {
        below = 1;
    } else {
        const dyad *small = dyad_sign(lo) > 0 ? lo : hi; /* the end nearest zero */
        below = top_bit(small) < emin;
    }
    if (!below) return 0;
    dyad x;
    dyad_init(&x);
    int rc = dyad_set_si(&x, 1, emin);
    if (!rc) rc = dyad_add(err, err, &x, 64, 1);
    dyad_free(&x);
    return rc;
}

/* The ends of every value one rounding of a point of [lo, hi] can give (Lemma 1),
   clipped to the largest finite value and snapped inward to the format's grid; *unbounded where a
   point may round past the largest finite value. */
int lean_ends(dyad *lo, dyad *hi, int *unbounded, const round_bound *b, int64_t w) {
    *unbounded = 0;
    if (!b->active) return 0;
    dyad mag, err, a, c;
    dyad_init(&mag);
    dyad_init(&err);
    dyad_init(&a);
    dyad_init(&c);
    int rc = dyad_neg(&a, lo);
    if (!rc) {
        if (dyad_sign(&a) < 0) a.m.neg = 0;
        rc = dyad_copy(&mag, dyad_cmp(&a, hi) < 0 ? hi : &a);
    }
    if (!rc && dyad_sign(&mag) > 0) {
        if (b->has_threshold ? dyad_cmp(&mag, &b->threshold) >= 0 : dyad_cmp(&mag, &b->largest) > 0) {
            *unbounded = 1;
        } else {
            int64_t top = top_bit(&mag), at = top > b->emin ? top : b->emin;
            if (!(rc = dyad_copy(&err, &b->ulps))) rc = dyad_shift(&err, at - b->mantissa);
            if (!rc && b->flush) rc = flush_error(&err, lo, hi, b->emin);
            if (!rc) rc = sub_dir(&a, lo, &err, w, 0);
            if (!rc) rc = dyad_add(&c, hi, &err, w, 1);
            /* clipped to [-largest, largest] */
            dyad least;
            dyad_init(&least);
            if (!rc) rc = dyad_neg(&least, &b->largest);
            if (!rc && dyad_cmp(&a, &least) < 0) rc = dyad_copy(&a, &least);
            if (!rc && dyad_cmp(&c, &b->largest) > 0) rc = dyad_copy(&c, &b->largest);
            dyad_free(&least);
            /* snapped inward to the grid, unless that empties it */
            dyad sa, sc;
            dyad_init(&sa);
            dyad_init(&sc);
            if (!rc && !(rc = dyad_copy(&sa, &a)) && !(rc = dyad_copy(&sc, &c)) && !(rc = grid_end(&sa, b->emin, b->mantissa, 1)))
                rc = grid_end(&sc, b->emin, b->mantissa, 0);
            if (!rc) {
                int ok = dyad_cmp(&sa, &sc) <= 0;
                if (!(rc = dyad_copy(lo, ok ? &sa : &a))) rc = dyad_copy(hi, ok ? &sc : &c);
            }
            dyad_free(&sa);
            dyad_free(&sc);
        }
    }
    dyad_free(&mag);
    dyad_free(&err);
    dyad_free(&a);
    dyad_free(&c);
    return rc;
}

/* Lemma 1 at a boundary (lean_ends) on an interval; *unb set where a point may overflow */
int lean_ivl(ivl *x, const round_bound *b, int64_t w, int *unb) {
    int u = 0, rc = lean_ends(&x->lo, &x->hi, &u, b, w);
    if (u) *unb = 1;
    return rc;
}

/* whether a magnitude may round past the boundary's largest finite value */
int lean_overflows(const round_bound *b, const dyad *mag) {
    return b->active && (b->has_threshold ? dyad_cmp(mag, &b->threshold) >= 0 : dyad_cmp(mag, &b->largest) > 0);
}

/* Lemma 1's bound on how far the rounding moves a point of [lo, hi] */
int lean_error(dyad *err, const dyad *lo, const dyad *hi, const round_bound *b, int *unb) {
    dyad mag;
    dyad_init(&mag);
    int rc = ends_magnitude(&mag, lo, hi);
    if (!rc && (!b->active || !dyad_sign(&mag))) {
        rc = dyad_set_si(err, 0, 0);
    } else if (!rc && lean_overflows(b, &mag)) {
        *unb = 1;
    } else if (!rc) {
        int64_t top = top_bit(&mag), at = top > b->emin ? top : b->emin;
        if (!(rc = dyad_copy(err, &b->ulps))) rc = dyad_shift(err, at - b->mantissa);
        if (!rc && b->flush) rc = flush_error(err, lo, hi, b->emin);
    }
    dyad_free(&mag);
    return rc;
}
