/* Certified exact values in ExactArray's packed storage (internal; exact.py describes the layout),
   and deciding an output's rounding from an enclosure of its exact value (proofs/foundations.md,
   Deciding a rounding by a sign). An entry is a midpoint and a radius: a 128-bit significand and
   the word 2 * e + sign, with a flag byte holding the kind and the radius code. */
#include "exactref.h"

#define KIND_FINITE 0
#define KIND_NAN 1
#define KIND_INFINITE 2
#define KIND_NEAR_ZERO 3
#define CODE_OUT_OF_RANGE 1 /* the radius code marking an infinite or near-zero entry out of range */
#define ENTRY_OFFSET 63

static void pack(entry *x, int kind, int code, int neg, const bigint *m, int64_t e) {
    uint64_t lo = m->n > 0 ? m->w[0] : 0, hi = m->n > 1 ? m->w[1] : 0, word = (uint64_t)(2 * e + (neg != 0));
    x->flag = (unsigned char)(kind << 6 | code);
    for (int k = 0; k < 8; k++) {
        x->mid[k] = (unsigned char)(lo >> (8 * k));
        x->mid[8 + k] = (unsigned char)(hi >> (8 * k));
        x->mid[16 + k] = (unsigned char)(word >> (8 * k));
    }
}

/* An entry's kind, radius code, sign, significand and exponent. */
static int unpack(int *kind, int *code, int *neg, bigint *m, int64_t *e, const entry *x) {
    uint64_t lo = 0, hi = 0, word = 0;
    for (int k = 0; k < 8; k++) {
        lo |= (uint64_t)x->mid[k] << (8 * k);
        hi |= (uint64_t)x->mid[8 + k] << (8 * k);
        word |= (uint64_t)x->mid[16 + k] << (8 * k);
    }
    *kind = x->flag >> 6;
    *code = x->flag & 63;
    *neg = (int)(word & 1);
    *e = (word_signed(word) - *neg) / 2;
    if (big_reserve(m, 2)) return EXACTREF_NOMEM;
    m->w[0] = lo;
    m->w[1] = hi;
    m->n = 2;
    m->neg = 0;
    big_trim(m);
    return 0;
}

static void pack_zero(entry *x, int kind, int code, int neg) {
    bigint zero;
    big_init(&zero);
    pack(x, kind, code, neg, &zero, 0);
}

/* The out-of-range entry of a nonzero value with 2**(top - 1) <= |x| < 2**top; 0 when it is in
   range. */
static int range_mark(entry *x, int64_t top, int neg) {
    if (top > EXACT_LIMIT) {
        pack_zero(x, KIND_INFINITE, CODE_OUT_OF_RANGE, neg);
        return 1;
    }
    if (top <= -EXACT_LIMIT) {
        pack_zero(x, KIND_NEAR_ZERO, CODE_OUT_OF_RANGE, neg);
        return 1;
    }
    return 0;
}

/* r * 2**re = a * 2**e (a >= 0) with at most 128 significant bits: to nearest with ties away from
   zero, or upward when up; no trailing zero bits, so one encoding per value. *exact says whether
   nothing was dropped. */
static int round128(bigint *r, int64_t *re, int *exact, const bigint *a, int64_t e, int up) {
    int64_t shift = big_bits(a) - ENTRY_BITS;
    int rc;
    *exact = 1;
    if (shift <= 0) {
        shift = 0;
        rc = big_copy(r, a);
    } else {
        int half = big_bit(a, shift - 1), sticky = big_low_nonzero(a, shift - 1);
        *exact = !half && !sticky;
        rc = big_shr(r, a, shift, 0);
        if (!rc && (up ? !*exact : half)) {
            bigint one;
            big_init(&one);
            if (!(rc = big_set_word(&one, 1, 0))) rc = big_add(r, r, &one);
            big_free(&one);
            if (!rc && big_bits(r) > ENTRY_BITS) {
                /* carried into a new bit: r is a power of two */
                rc = big_shr(r, r, 1, 0);
                shift++;
            }
        }
    }
    r->neg = 0;
    if (!rc && r->n) {
        int64_t trailing = big_trailing(r);
        rc = big_shr(r, r, trailing, 0);
        shift += trailing;
    }
    *re = e + shift;
    return rc;
}

/* The entry of an exactly known value (exact.py's exact_entry and scaled_entry): NaN, an infinity,
   or the finite (-1)**neg * a * 2**e for a >= 0. A value longer than 128 bits is rounded to
   nearest, with a radius of half a unit. */
int entry_value(entry *x, int kind, int neg, const bigint *a, int64_t e) {
    if (kind == VALUE_NAN || kind == VALUE_INFINITE) {
        pack_zero(x, kind == VALUE_NAN ? KIND_NAN : KIND_INFINITE, 0, neg);
        return 0;
    }
    if (!a->n) {
        pack_zero(x, KIND_FINITE, 0, 0);
        return 0;
    }
    if (range_mark(x, e + big_bits(a), neg)) return 0;
    bigint m;
    big_init(&m);
    int exact;
    int64_t me;
    int rc = round128(&m, &me, &exact, a, e, 0);
    if (!rc) pack(x, KIND_FINITE, exact ? 0 : ENTRY_OFFSET - 1, neg, &m, me);
    big_free(&m);
    return rc;
}

/* The entry of (-1)**neg * num / den for num >= 0 and den > 0 (exact.py's exact_entry of a
   Fraction): rounded once to nearest when it is not dyadic. */
int entry_ratio(entry *x, int neg, const bigint *num, const bigint *den) {
    if (!num->n) {
        pack_zero(x, KIND_FINITE, 0, 0);
        return 0;
    }
    if (big_trailing(den) == big_bits(den) - 1) return entry_value(x, VALUE_FINITE, neg, num, 1 - big_bits(den));
    bigint a, b, q, m;
    big_init(&a);
    big_init(&b);
    big_init(&q);
    big_init(&m);
    /* top with 2**top <= num / den < 2**(top + 1) */
    int64_t top = big_bits(num) - big_bits(den);
    int rc = top >= 0 ? big_shl(&b, den, top) : big_shl(&a, num, -top);
    if (!rc && big_cmp_mag(top >= 0 ? num : &a, top >= 0 ? &b : den) < 0) top--;
    if (!rc && !range_mark(x, top + 1, neg)) {
        /* q = round(num / den / 2**e) with 2**127 <= num / den / 2**e < 2**128: the quotient at
           one more bit, then halved upward (no ties: the value is not dyadic) */
        int64_t e = top - (ENTRY_BITS - 1), me;
        int exact;
        rc = 1 - e >= 0 ? big_shl(&a, num, 1 - e) : big_copy(&a, num);
        if (!rc) rc = 1 - e >= 0 ? big_copy(&b, den) : big_shl(&b, den, e - 1);
        if (!rc) rc = big_divmod(&q, &exact, &a, &b);
        if (!rc) rc = big_shr(&q, &q, 1, 1);
        if (!rc) rc = round128(&m, &me, &exact, &q, e, 0); /* only a carry to 2**128 is renormalized */
        if (!rc) pack(x, KIND_FINITE, ENTRY_OFFSET - 1, neg, &m, me);
    }
    big_free(&a);
    big_free(&b);
    big_free(&q);
    big_free(&m);
    return rc;
}

/* The entry of a value known to lie in [lo, hi] (exact.py's interval_entry); *wide when the
   interval is too wide for an entry. One that contains zero gives a near-zero entry, and one
   wholly beyond the stored range an out-of-range entry; one that crosses the range's edge is too
   wide. The ends are compared by their exponents before any alignment. */
int entry_interval(entry *x, int *wide, const dyad *lo, const dyad *hi) {
    *wide = 0;
    const bigint *ml = &lo->m, *mh = &hi->m;
    if (!dyad_cmp(lo, hi)) { /* a point */
        bigint a = big_magnitude(ml);
        return entry_value(x, VALUE_FINITE, ml->neg, &a, lo->e);
    }
    bigint m, a, b, t;
    big_init(&m);
    big_init(&a);
    big_init(&b);
    big_init(&t);
    int rc = 0, exact;
    int64_t me;
    if ((ml->neg || !ml->n) && !mh->neg) {
        /* contains zero: the radius is the larger magnitude, rounded up */
        dyad left = *lo;
        left.m.neg = 0;
        const dyad *large = dyad_cmp(&left, hi) >= 0 ? lo : hi;
        bigint r = big_magnitude(&large->m);
        rc = round128(&m, &me, &exact, &r, large->e, 1);
        if (!rc && me + big_bits(&m) > EXACT_LIMIT) {
            *wide = 1;
        } else if (!rc) {
            if (me + big_bits(&m) <= -EXACT_LIMIT) {
                rc = big_set_word(&m, 1, 0); /* a larger radius is still an enclosure */
                me = -EXACT_LIMIT;
            }
            if (!rc) pack(x, KIND_NEAR_ZERO, 0, 0, &m, me);
        }
        goto done;
    }
    int neg = mh->neg;
    int64_t tl = lo->e + big_bits(ml), th = hi->e + big_bits(mh);
    int64_t small = neg ? th : tl, large = neg ? tl : th; /* tops of the smaller and larger magnitudes */
    if (range_mark(x, small > EXACT_LIMIT ? small : large <= -EXACT_LIMIT ? large : 0, neg)) goto done;
    if (small <= -EXACT_LIMIT || large > EXACT_LIMIT || large - small > 1) {
        *wide = 1; /* crosses the range's edge, or wider than half its smaller end */
        goto done;
    }
    /* the center (a + b) / 2 in units of 2**e0, exact since e0 is one below both exponents */
    int64_t e0 = (lo->e < hi->e ? lo->e : hi->e) - 1;
    if ((rc = big_shl(&a, ml, lo->e - e0)) || (rc = big_shl(&b, mh, hi->e - e0)) || (rc = big_add(&t, &a, &b)) ||
        (rc = big_shr(&t, &t, 1, 0)))
        goto done;
    t.neg = 0;
    if ((rc = round128(&m, &me, &exact, &t, e0, 0))) goto done;
    /* the radius max(b - mid, mid - a), in units of 2**e0 */
    if ((rc = big_shl(&t, &m, me - e0))) goto done;
    t.neg = neg && t.n;
    a.neg = !a.neg && a.n;
    if ((rc = big_add(&a, &t, &a))) goto done; /* mid - a */
    t.neg = !t.neg && t.n;
    if ((rc = big_add(&b, &b, &t))) goto done; /* b - mid */
    const bigint *radius = !a.neg && big_cmp_mag(&a, &b) >= 0 ? &a : b.neg ? &a : &b;
    if (!radius->n) {
        pack(x, KIND_FINITE, 0, neg, &m, me);
        goto done;
    }
    /* the smallest code k with 2**(k - 63) units of the midpoint's last place >= the radius */
    int64_t unit = me + big_bits(&m) - ENTRY_BITS, s = big_bits(radius) + e0 - unit;
    int64_t exponent = big_trailing(radius) == big_bits(radius) - 1 ? s - 1 : s;
    int64_t code = exponent + ENTRY_OFFSET > 1 ? exponent + ENTRY_OFFSET : 1;
    if (code > ENTRY_OFFSET) {
        *wide = 1;
    } else {
        pack(x, KIND_FINITE, (int)code, neg, &m, me);
    }
done:
    big_free(&m);
    big_free(&a);
    big_free(&b);
    big_free(&t);
    return rc;
}

/* The value of an entry (exact.py's Enclosure): the midpoint's kind (a VALUE_ code), sign and
   m * 2**e, the radius rm * 2**re, and whether it is out of range. */
int entry_read(int *kind, int *neg, bigint *m, int64_t *e, bigint *rm, int64_t *re, int *out_of_range, const entry *x) {
    int stored, code, rc = unpack(&stored, &code, neg, m, e, x);
    *kind = stored == KIND_NAN ? VALUE_NAN : stored == KIND_INFINITE ? VALUE_INFINITE : VALUE_FINITE;
    *out_of_range = (stored == KIND_INFINITE || stored == KIND_NEAR_ZERO) && code == CODE_OUT_OF_RANGE;
    rm->n = 0;
    *re = 0;
    if (rc || stored == KIND_NAN || stored == KIND_INFINITE || *out_of_range) {
        m->n = 0;
        *e = 0;
        return rc;
    }
    if (stored == KIND_NEAR_ZERO) {
        /* the midpoint is +0 and the stored value the radius */
        rc = big_copy(rm, m);
        *re = *e;
        m->n = 0;
        *e = 0;
        *neg = 0;
        return rc;
    }
    if (code) {
        *re = *e + big_bits(m) - ENTRY_BITS + code - ENTRY_OFFSET;
        rc = big_set_word(rm, 1, 0);
    }
    return rc;
}

/* lo and hi bounding an entry's value; *none for NaN and infinities. */
int entry_bounds(dyad *lo, dyad *hi, int *none, const entry *x) {
    int kind, code, neg;
    int64_t e;
    bigint m;
    big_init(&m);
    int rc = unpack(&kind, &code, &neg, &m, &e, x);
    *none = kind == KIND_NAN || kind == KIND_INFINITE;
    if (rc || *none) goto done;
    if (kind == KIND_NEAR_ZERO && code == CODE_OUT_OF_RANGE) {
        if (!(rc = dyad_set_si(lo, -1, -EXACT_LIMIT))) rc = dyad_set_si(hi, 1, -EXACT_LIMIT);
        goto done;
    }
    if (kind == KIND_NEAR_ZERO) {
        if (!(rc = dyad_from_big(lo, &m, e)) && !(rc = dyad_from_big(hi, &m, e))) lo->m.neg = lo->m.n != 0;
        goto done;
    }
    m.neg = neg && m.n;
    if (!code) {
        if (!(rc = dyad_from_big(lo, &m, e))) rc = dyad_from_big(hi, &m, e);
        goto done;
    }
    /* the radius is 2**(code - 63) units of 2**(top - 128): the midpoint at that unit, plus and
       minus one, or at its own exponent, plus and minus the radius */
    int64_t k = e + big_bits(&m) - ENTRY_BITS + code - ENTRY_OFFSET;
    bigint r;
    big_init(&r);
    int64_t at = k >= e ? e : k; /* the ends' exponent */
    if (k >= e) {
        if (!(rc = big_set_word(&r, 1, 0))) rc = big_shl(&r, &r, k - e);
    } else {
        if (!(rc = big_set_word(&r, 1, 0))) rc = big_shl(&m, &m, e - k);
    }
    if (!rc) rc = big_add(&hi->m, &m, &r);
    r.neg = 1;
    if (!rc) rc = big_add(&lo->m, &m, &r);
    lo->e = lo->m.n ? at : 0;
    hi->e = hi->m.n ? at : 0;
    big_free(&r);
done:
    big_free(&m);
    return rc;
}

/* The encoding and entry of a value in [lo, hi]; *none while the
   interval must narrow: its ends encode differently, it contains zero before the last attempt
   (at the last, the entry is near zero), or it is too wide for an entry. */
int decide_rounding(bigint *bits, entry *x, int *none, const floatfmt *f, int rnd, int ovf, const dyad *lo, const dyad *hi, int last) {
    bigint other;
    big_init(&other);
    int rc = encode_dyad(bits, f, lo, rnd, ovf);
    if (!rc) rc = encode_dyad(&other, f, hi, rnd, ovf);
    *none = 1;
    if (!rc && !big_cmp_mag(bits, &other) && (last || dyad_sign(lo) * dyad_sign(hi) > 0)) rc = entry_interval(x, none, lo, hi);
    big_free(&other);
    return rc;
}

/* Whether the dyadic b is 0 or a multiple of ulp(b) / 2 in f: every point where rounding to f
   changes is such a grid point. */
static int on_grid(const floatfmt *f, const dyad *b) {
    if (!b->m.n) return 1;
    int64_t top = b->e + big_bits(&b->m) - 1;
    return b->e + big_trailing(&b->m) >= (top > f->emin ? top : f->emin) - (f->man_bits + 1);
}

/* r = b + side * eps exactly for a grid point b, with eps = ulp(b) / 8 (2**(emin - p - 2) for 0). */
static int probe(dyad *r, const floatfmt *f, const dyad *b, int side) {
    int64_t top = b->m.n ? b->e + big_bits(&b->m) - 1 : f->emin;
    int64_t k = (top > f->emin ? top : f->emin) - (f->man_bits + 1) - 2;
    int rc = b->e >= k ? big_shl(&r->m, &b->m, b->e - k) : big_shr(&r->m, &b->m, k - b->e, 0); /* exact: b is a multiple of 4 eps */
    bigint one;
    big_init(&one);
    if (!rc && !(rc = big_set_si(&one, side))) rc = big_add(&r->m, &r->m, &one);
    big_free(&one);
    r->e = r->m.n ? k : 0;
    return rc;
}

/* Whether rounding switches at the candidate b: b is a grid point in [lo, hi], every value in
   [lo, b) rounds to low and every value in (b, hi] to high (a side with no values needs no check). */
static int switches_at(int *yes, const floatfmt *f, int rnd, int ovf, const dyad *lo, const dyad *hi, const dyad *b,
                       const bigint *low, const bigint *high) {
    int below = dyad_cmp(lo, b), above = dyad_cmp(hi, b), rc = 0;
    *yes = below <= 0 && above >= 0 && on_grid(f, b);
    dyad p;
    bigint bits;
    dyad_init(&p);
    big_init(&bits);
    if (*yes && below && !(rc = probe(&p, f, b, -1)) && !(rc = encode_dyad(&bits, f, &p, rnd, ovf))) *yes = !big_cmp_mag(&bits, low);
    if (*yes && above && !(rc = probe(&p, f, b, 1)) && !(rc = encode_dyad(&bits, f, &p, rnd, ovf))) *yes = !big_cmp_mag(&bits, high);
    dyad_free(&p);
    big_free(&bits);
    return rc;
}

/* A grid point b in [lo, hi] where rounding switches from lo's encoding to hi's; *none when
   there is none. The candidates are 0, the values lo and hi round to
   and their midpoint, or the overflow thresholds where an end rounds past the largest value. */
int switch_point(dyad *b, int *none, const floatfmt *f, int rnd, int ovf, const dyad *lo, const dyad *hi) {
    bigint low, high;
    dyad cand[4];
    big_init(&low);
    big_init(&high);
    for (int i = 0; i < 4; i++) dyad_init(&cand[i]);
    int n = 1, finite = 1, rc = encode_dyad(&low, f, lo, rnd, ovf);
    if (!rc) rc = encode_dyad(&high, f, hi, rnd, ovf);
    /* cand[0] is 0; then the ends' values in f */
    for (int side = 0; !rc && side < 2; side++) {
        int kind, neg;
        int64_t e;
        bigint *m = &cand[n].m;
        if ((rc = floatfmt_decode(&kind, &neg, m, &e, f, side ? &high : &low))) break;
        if (kind != VALUE_FINITE) {
            finite = 0;
            continue;
        }
        m->neg = neg && m->n;
        cand[n++].e = e;
    }
    if (!rc && finite) {
        /* the midpoint of the two ends' values */
        int64_t e0 = cand[1].e < cand[2].e ? cand[1].e : cand[2].e;
        dyad *mid = &cand[3];
        if (!(rc = big_shl(&mid->m, &cand[1].m, cand[1].e - e0)) && !(rc = big_shl(&low, &cand[2].m, cand[2].e - e0)))
            rc = big_add(&mid->m, &mid->m, &low);
        mid->e = mid->m.n ? e0 - 1 : 0;
        n = 4;
        if (!rc) rc = encode_dyad(&low, f, lo, rnd, ovf); /* low was a scratch value */
    } else if (!rc) {
        /* rounding to nearest switches to an infinity (or NaN) at the overflow threshold: the
           largest finite value plus half its ulp */
        int kind, neg;
        dyad *t = &cand[n];
        if (!(rc = floatfmt_special(&high, f, ENCODING_MAX_FINITE, 0)) && !(rc = floatfmt_decode(&kind, &neg, &t->m, &t->e, f, &high))) {
            /* normalized, as a value object holds it */
            int64_t trailing = big_trailing(&t->m);
            if (!(rc = big_shr(&t->m, &t->m, trailing, 0))) rc = dyad_shift(t, trailing);
            dyad half;
            dyad_init(&half);
            if (!(rc = dyad_set_si(&half, 1, f->emax - f->man_bits - 1))) rc = dyad_add(t, t, &half, 256, 1);
            dyad_free(&half);
        }
        if (!rc && !(rc = dyad_copy(&cand[n + 1], t))) cand[n + 1].m.neg = 1;
        n += 2;
        if (!rc) rc = encode_dyad(&high, f, hi, rnd, ovf); /* high was a scratch value */
    }
    *none = 1;
    for (int i = 0; !rc && i < n && *none; i++) {
        int yes;
        if (!(rc = switches_at(&yes, f, rnd, ovf, lo, hi, &cand[i], &low, &high)) && yes) {
            *none = 0;
            rc = dyad_copy(b, &cand[i]);
        }
    }
    big_free(&low);
    big_free(&high);
    for (int i = 0; i < 4; i++) dyad_free(&cand[i]);
    return rc;
}

/* The encoding and entry of a value c + o with c in [clo, chi] (exact when a point) and o in
   [olo, ohi] (foundations.md, Deciding a rounding by a sign); *none while undecided. The outward-rounded sum, at p
   bits, is tried first. If that is undecided, c is an exact grid point and o has a sign, the
   rounding is the one just beyond c on o's side. */
int offset_rounding(bigint *bits, entry *x, int *none, const floatfmt *f, int rnd, int ovf, const dyad *clo, const dyad *chi,
                    const dyad *olo, const dyad *ohi, int64_t p, int last) {
    dyad lo, hi, q;
    bigint other;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&q);
    big_init(&other);
    int rc = dyad_add(&lo, clo, olo, p, 0);
    if (!rc) rc = dyad_add(&hi, chi, ohi, p, 1);
    if (!rc) rc = decide_rounding(bits, x, none, f, rnd, ovf, &lo, &hi, last);
    int point = !dyad_cmp(clo, chi);
    int side = dyad_sign(olo) > 0 ? 1 : dyad_sign(ohi) < 0 ? -1 : 0;
    if (rc || !*none || !point || !side || !on_grid(f, clo)) goto done;
    if ((rc = encode_dyad(bits, f, side > 0 ? &hi : &lo, rnd, ovf)) || (rc = probe(&q, f, clo, side)) ||
        (rc = encode_dyad(&other, f, &q, rnd, ovf)) || big_cmp_mag(bits, &other))
        goto done;
    if (!clo->m.n) {
        rc = entry_interval(x, none, olo, ohi);
    } else {
        rc = side > 0 ? entry_interval(x, none, clo, &hi) : entry_interval(x, none, &lo, clo);
    }
done:
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&q);
    big_free(&other);
    return rc;
}
