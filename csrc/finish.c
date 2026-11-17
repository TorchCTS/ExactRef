/* The final rounding of outputs (internal): Lemma 3's bounds on what a correct implementation
   outputs for one real value (proofs/foundations.md): the tolerance against the reference, the
   range of finite outputs and the flags of non-finite ones. */
#include "exactref.h"

/* The finite outputs found so far (a range of dyadics) and the flags of the non-finite ones. */
typedef struct {
    const floatfmt *out;
    int rnd, ovf, any, flags;
    dyad lo, hi;
} outputs;

static void outputs_init(outputs *o, const floatfmt *out, int rnd, int ovf) {
    o->out = out;
    o->rnd = rnd;
    o->ovf = ovf;
    o->any = o->flags = 0;
    dyad_init(&o->lo);
    dyad_init(&o->hi);
}

static void outputs_free(outputs *o) {
    dyad_free(&o->lo);
    dyad_free(&o->hi);
}

static int extend(outputs *o, const dyad *lo, const dyad *hi) {
    int rc = 0;
    if (!o->any || dyad_cmp(lo, &o->lo) < 0) rc = dyad_copy(&o->lo, lo);
    if (!rc && (!o->any || dyad_cmp(hi, &o->hi) > 0)) rc = dyad_copy(&o->hi, hi);
    o->any = 1;
    return rc;
}

/* A non-finite value reaching the output: what the output format encodes for it. */
static int add_value(outputs *o, int kind, int neg) {
    bigint bits, zero;
    dyad point;
    big_init(&bits);
    big_init(&zero);
    dyad_init(&point);
    int rc = floatfmt_encode(&bits, o->out, kind, neg, &zero, 0, o->rnd, o->ovf);
    int k, n;
    if (!rc) rc = floatfmt_decode(&k, &n, &point.m, &point.e, o->out, &bits);
    if (!rc && k == VALUE_NAN) {
        o->flags |= FLAG_NAN;
    } else if (!rc && k == VALUE_INFINITE) {
        o->flags |= n ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    } else if (!rc) {
        point.m.neg = n && point.m.n;
        rc = extend(o, &point, &point);
    }
    big_free(&bits);
    dyad_free(&point);
    return rc;
}

static int add_flags(outputs *o, int flags) {
    int rc = 0;
    if (flags & FLAG_PLUS_INF) rc = add_value(o, VALUE_INFINITE, 0);
    if (!rc && flags & FLAG_MINUS_INF) rc = add_value(o, VALUE_INFINITE, 1);
    if (!rc && flags & FLAG_NAN) rc = add_value(o, VALUE_NAN, 0);
    return rc;
}

/* The value of an encoding as a dyadic; *finite = 0 when it is not finite. */
static int decode_dyad(dyad *x, int *finite, const floatfmt *f, const bigint *bits) {
    int kind, neg;
    int rc = floatfmt_decode(&kind, &neg, &x->m, &x->e, f, bits);
    x->m.neg = neg && x->m.n;
    *finite = kind == VALUE_FINITE;
    return rc;
}

/* The range's ends as encodings, outward to values of the output format; NaN for an empty range. */
static int ends(bigint *low, bigint *high, const outputs *o) {
    if (!o->any) {
        int rc = floatfmt_special(low, o->out, ENCODING_NAN, 0);
        return rc ? rc : big_copy(high, low);
    }
    int rc = encode_dyad(low, o->out, &o->lo, ROUND_TOWARD_NEGATIVE, OVERFLOW_IEEE);
    return rc ? rc : encode_dyad(high, o->out, &o->hi, ROUND_TOWARD_POSITIVE, OVERFLOW_IEEE);
}

/* r = x - y exactly. */
static int difference(dyad *r, const dyad *x, const dyad *y) {
    bigint t, ny = y->m;
    big_init(&t);
    ny.neg = !ny.neg && ny.n; /* a read-only view of -y */
    int rc;
    if (x->e >= y->e) {
        if (!(rc = big_shl(&t, &x->m, x->e - y->e))) rc = big_add(&r->m, &t, &ny);
        r->e = y->e;
    } else {
        if (!(rc = big_shl(&t, &ny, y->e - x->e))) rc = big_add(&r->m, &x->m, &t);
        r->e = x->e;
    }
    if (!r->m.n) r->e = 0;
    big_free(&t);
    return rc;
}

/* The float64 encoding (f64), rounded up, of the larger exact distance from ref to lo and hi. */
static int distance(bigint *tol, const floatfmt *f64, const dyad *lo, const dyad *hi, const dyad *ref) {
    dyad upper, lower;
    dyad_init(&upper);
    dyad_init(&lower);
    int rc = difference(&upper, hi, ref);
    if (!rc) rc = difference(&lower, ref, lo);
    if (!rc) {
        const dyad *worst = dyad_cmp(&upper, &lower) >= 0 ? &upper : &lower;
        if (dyad_sign(worst) <= 0) {
            tol->n = 0;
            tol->neg = 0;
        } else {
            rc = encode_dyad(tol, f64, worst, ROUND_TOWARD_POSITIVE, OVERFLOW_IEEE);
        }
    }
    dyad_free(&upper);
    dyad_free(&lower);
    return rc;
}

/* Lemma 1's bound on one rounding to f of a value in the binade of the nonzero magnitude. */
static int rounding_error(dyad *error, const floatfmt *f, const dyad *magnitude, int rnd) {
    int64_t top = magnitude->e + big_bits(&magnitude->m) - 1, ulp = (top > f->emin ? top : f->emin) - f->man_bits;
    int nearest = rnd == ROUND_NEAREST_EVEN || rnd == ROUND_NEAREST_AWAY;
    return dyad_set_si(error, 1, nearest ? ulp - 1 : ulp);
}

/* lo and hi widened by the rounding error at the larger magnitude of their ends, and that error
   (zero when both ends are). */
static int widened(dyad *lo, dyad *hi, dyad *error, dyad *magnitude, const floatfmt *f, int rnd, int64_t w) {
    dyad a = *lo, b = *hi; /* read-only views of the magnitudes */
    a.m.neg = b.m.neg = 0;
    int rc = dyad_copy(magnitude, dyad_cmp(&a, &b) >= 0 ? &a : &b);
    magnitude->m.neg = 0;
    if (rc || !magnitude->m.n) {
        dyad_set_si(error, 0, 0);
        return rc;
    }
    if ((rc = rounding_error(error, f, magnitude, rnd))) return rc;
    dyad down = *error; /* a read-only view of -error */
    down.m.neg = 1;
    if (!(rc = dyad_add(lo, lo, &down, w, 0))) rc = dyad_add(hi, hi, error, w, 1);
    return rc;
}

/* final_bounds for a finite reference and an x that cannot round past the largest finite value,
   with nothing else reaching the output; *done = 0 in every other case. */
static int finite_bounds(int *done, bigint *tol, bigint *low, bigint *high, const floatfmt *out, const floatfmt *f64, int rnd,
                         const dyad *x_lo, const dyad *x_hi, const dyad *ref, int64_t w, const dyad *largest) {
    dyad lo, hi, error, magnitude, lo_s, hi_s;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&error);
    dyad_init(&magnitude);
    dyad_init(&lo_s);
    dyad_init(&hi_s);
    *done = 0;
    int rc = dyad_copy(&lo, x_lo), finite_lo, finite_hi;
    if (!rc) rc = dyad_copy(&hi, x_hi);
    if (!rc) rc = widened(&lo, &hi, &error, &magnitude, out, rnd, w);
    if (rc || (magnitude.m.n && ((rc = dyad_add(&magnitude, &magnitude, &error, w, 1)) || dyad_cmp(&magnitude, largest) > 0))) goto done;
    if ((rc = encode_dyad(low, out, &lo, ROUND_TOWARD_POSITIVE, OVERFLOW_IEEE)) || (rc = encode_dyad(high, out, &hi, ROUND_TOWARD_NEGATIVE, OVERFLOW_IEEE)) ||
        (rc = decode_dyad(&lo_s, &finite_lo, out, low)) || (rc = decode_dyad(&hi_s, &finite_hi, out, high)))
        goto done;
    if (!finite_lo || !finite_hi || dyad_cmp(&lo_s, &hi_s) > 0) goto done;
    *done = 1;
    rc = distance(tol, f64, &lo_s, &hi_s, ref);
done:
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&error);
    dyad_free(&magnitude);
    dyad_free(&lo_s);
    dyad_free(&hi_s);
    return rc;
}

/* x rounded to a value of f in the direction rnd; *finite = 0 when that is not finite. */
static int snapped(dyad *r, int *finite, const floatfmt *f, const dyad *x, int rnd) {
    *finite = 1;
    if (!x->m.n) return dyad_set_si(r, 0, 0);
    bigint bits;
    big_init(&bits);
    int rc = encode_dyad(&bits, f, x, rnd, OVERFLOW_IEEE);
    if (!rc) rc = decode_dyad(r, finite, f, &bits);
    big_free(&bits);
    return rc;
}

/* Every value one rounding to nearest in f of a point of [lo, hi] can give (Lemma 1): where a
   point reaches f's overflow threshold, *flags holds that side (NaN in a format without infinity)
   and the interval is first held within f's largest finite value, the outputs that did not
   overflow; then the ends move out by the rounding's error at their larger magnitude (rounded
   outward to w bits), are clipped to f's finite range and snapped inward to f's values. */
int rounding_interval(dyad *lo, dyad *hi, int *flags, const floatfmt *f, int64_t w) {
    dyad theta, largest, least, error, magnitude, a, b;
    dyad *all[] = {&theta, &largest, &least, &error, &magnitude, &a, &b};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    *flags = 0;
    int rc = floatfmt_threshold(&theta, &largest, f);
    if (!rc && !(rc = dyad_copy(&least, &largest))) {
        least.m.neg = least.m.n != 0;
        dyad nlo = *lo; /* a read-only view of -lo */
        nlo.m.neg = !nlo.m.neg && nlo.m.n;
        if (dyad_cmp(hi, &theta) >= 0) *flags |= FLAG_PLUS_INF;
        if (dyad_cmp(&nlo, &theta) >= 0) *flags |= FLAG_MINUS_INF;
        if (*flags && f->special != SPECIAL_IEEE) *flags = FLAG_NAN;
    }
    /* held within the finite range */
    if (!rc && dyad_cmp(lo, &least) < 0) rc = dyad_copy(lo, &least);
    if (!rc && dyad_cmp(lo, &largest) > 0) rc = dyad_copy(lo, &largest);
    if (!rc && dyad_cmp(hi, &largest) > 0) rc = dyad_copy(hi, &largest);
    if (!rc && dyad_cmp(hi, &least) < 0) rc = dyad_copy(hi, &least);
    if (!rc) rc = widened(lo, hi, &error, &magnitude, f, ROUND_NEAREST_EVEN, w);
    if (!rc && magnitude.m.n) {
        if (dyad_cmp(lo, &least) < 0) rc = dyad_copy(lo, &least);
        if (!rc && dyad_cmp(hi, &largest) > 0) rc = dyad_copy(hi, &largest);
        /* every rounded value is a value of f: snap inward */
        int fa, fb;
        if (!rc && !(rc = snapped(&a, &fa, f, lo, ROUND_TOWARD_POSITIVE)) && !(rc = snapped(&b, &fb, f, hi, ROUND_TOWARD_NEGATIVE)) && fa && fb &&
            dyad_cmp(&a, &b) <= 0 && !(rc = dyad_copy(lo, &a)))
            rc = dyad_copy(hi, &b);
    }
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
    return rc;
}

/* Lemma 3 for one real output value whose value before its final rounding to out lies in
   [x_lo, x_hi] (both NULL: no bound): the tolerance (f64's encoding), the encodings low and
   high of the lowest and highest finite output (both NaN when none can occur), and the flags of
   the non-finite outputs. x is widened by the final rounding's error (Lemma 1) and each end is
   snapped inward to a value of out, no further than the largest finite value; the tolerance is
   the larger exact distance from the reference to the range's ends, rounded up. Where x rounds
   past the largest finite value, out's encoding there (with the overflow mode) is an infinity,
   NaN or the largest finite value, and sets its flag or joins the range. possible holds the
   flags of non-finite values that may reach the final rounding; FLAG_ANY means no bound. A
   reference that is not finite gives a NaN tolerance and its own flag; no bound gives an
   infinite tolerance, every finite output and every flag. */
int final_bounds(bigint *tol, bigint *low, bigint *high, int *flags, const floatfmt *out, const floatfmt *f64, int rnd, int ovf,
                 const dyad *x_lo, const dyad *x_hi, const bigint *reference, int64_t w, int possible) {
    dyad ref, largest, least, lo, hi, error, magnitude, lo_s, hi_s, t;
    dyad *all[] = {&ref, &largest, &least, &lo, &hi, &error, &magnitude, &lo_s, &hi_s, &t};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    outputs o;
    outputs_init(&o, out, rnd, ovf);
    int finite_ref, finite, rc = decode_dyad(&ref, &finite_ref, out, reference);
    if (!rc && (rc = floatfmt_special(tol, out, ENCODING_MAX_FINITE, 0)) == 0) rc = decode_dyad(&largest, &finite, out, tol);
    if (!rc && !(rc = dyad_copy(&least, &largest))) least.m.neg = 1;
    *flags = 0;
    if (rc) goto done;
    if (x_lo && finite_ref && !possible) {
        int found;
        if ((rc = finite_bounds(&found, tol, low, high, out, f64, rnd, x_lo, x_hi, &ref, w, &largest)) || found) goto done;
    }
    if ((rc = add_flags(&o, possible))) goto done;
    if (!finite_ref) {
        int kind, neg;
        if ((rc = floatfmt_decode(&kind, &neg, &t.m, &t.e, out, reference)) || (rc = add_value(&o, kind, neg))) goto done;
    }
    if (possible & FLAG_ANY) {
        x_lo = x_hi = NULL;
        if ((rc = extend(&o, &least, &largest)) || (rc = add_flags(&o, FLAG_ALL))) goto done;
    }
    if (!x_lo) {
        if (finite_ref && ((rc = extend(&o, &least, &largest)) || (rc = add_flags(&o, FLAG_ALL)))) goto done;
        rc = floatfmt_special(tol, f64, finite_ref ? ENCODING_INFINITY : ENCODING_NAN, 0);
        goto finish;
    }
    /* an end that rounds past the largest finite value, and whether every value of x does */
    int past = 0;
    for (int side = 0; side < 2; side++) {
        const dyad *end = side ? x_hi : x_lo;
        int beyond = side ? end->m.neg : !end->m.neg, kind, neg;
        if (!end->m.n) continue;
        if ((rc = encode_dyad(&t.m, out, end, rnd, ovf)) || (rc = floatfmt_decode(&kind, &neg, &lo_s.m, &lo_s.e, out, &t.m))) goto done;
        if (kind != VALUE_FINITE) {
            if ((rc = add_value(&o, kind, neg))) goto done;
            past = past || beyond;
        }
    }
    if ((rc = dyad_copy(&lo, x_lo)) || (rc = dyad_copy(&hi, x_hi)) || (rc = widened(&lo, &hi, &error, &magnitude, out, rnd, w))) goto done;
    int finite_lo, finite_hi;
    if ((rc = snapped(&lo_s, &finite_lo, out, &lo, ROUND_TOWARD_POSITIVE)) || (!finite_lo && (rc = dyad_copy(&lo_s, &lo))) ||
        (rc = snapped(&hi_s, &finite_hi, out, &hi, ROUND_TOWARD_NEGATIVE)) || (!finite_hi && (rc = dyad_copy(&hi_s, &hi))))
        goto done;
    if (dyad_cmp(&lo_s, &hi_s) > 0 && ((rc = dyad_copy(&lo_s, &lo)) || (rc = dyad_copy(&hi_s, &hi)))) goto done;
    if (!past) {
        /* the snapped ends, clipped to the finite values */
        const dyad *a = dyad_cmp(&lo_s, &least) >= 0 ? &lo_s : &least, *b = dyad_cmp(&hi_s, &largest) <= 0 ? &hi_s : &largest;
        a = dyad_cmp(a, &largest) <= 0 ? a : &largest;
        b = dyad_cmp(b, &least) >= 0 ? b : &least;
        if ((rc = extend(&o, a, b))) goto done;
    }
    if (!finite_ref) {
        rc = floatfmt_special(tol, f64, ENCODING_NAN, 0);
    } else if (!o.any) {
        tol->n = 0;
        tol->neg = 0;
    } else {
        rc = distance(tol, f64, &o.lo, &o.hi, &ref);
    }
finish:
    if (!rc) rc = ends(low, high, &o);
    *flags = o.flags;
done:
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
    outputs_free(&o);
    return rc;
}

/* The order of the integers a and b (-1, 0, 1). */
static int int_cmp(const bigint *a, const bigint *b) {
    dyad x, y;
    x.m = *a; /* read-only views */
    y.m = *b;
    x.e = y.e = 0;
    return dyad_cmp(&x, &y);
}

/* The bounds of an integer output (itemsize bytes, is_signed) whose exact value lies in [lo, hi]
   and whose reference is the encoding ref: low and high, the encodings of the lowest and highest
   output, and tol, the float64 encoding (f64) of the larger distance from the reference's value to
   them, rounded up. Exact values become outputs as intfmt_encode makes them in the overflow mode
   ovf: saturated; wrapped, where a range that crosses a wrap boundary or spans a whole period
   covers every value; or with INT_OVERFLOW_ERROR raising, so values outside the format give no
   output (the reference's own value is inside). */
int int_bounds(bigint *tol, bigint *low, bigint *high, const floatfmt *f64, int64_t itemsize, int is_signed, int ovf, const bigint *lo,
               const bigint *hi, const bigint *ref) {
    bigint min, max, a, b, t, u;
    dyad bottom, top, center;
    bigint *bigs[] = {&min, &max, &a, &b, &t, &u};
    for (size_t i = 0; i < sizeof bigs / sizeof *bigs; i++) big_init(bigs[i]);
    dyad_init(&bottom);
    dyad_init(&top);
    dyad_init(&center);
    int fits, rc = intfmt_range(&min, &max, itemsize, is_signed);
    if (!rc && ovf == INT_OVERFLOW_WRAP) {
        /* hi - lo > max - min: the range spans a whole period */
        bigint nlo = *lo, nmin = min; /* read-only views of -lo and -min */
        nlo.neg = !nlo.neg && nlo.n;
        nmin.neg = !nmin.neg && nmin.n;
        int whole = 0;
        if (!(rc = big_add(&t, hi, &nlo)) && !(rc = big_add(&u, &max, &nmin))) whole = int_cmp(&t, &u) > 0;
        if (!rc && !whole && !(rc = intfmt_encode(&t, &fits, lo, itemsize, is_signed, INT_OVERFLOW_WRAP)) &&
            !(rc = intfmt_decode(&a, &t, itemsize, is_signed)) && !(rc = intfmt_encode(&t, &fits, hi, itemsize, is_signed, INT_OVERFLOW_WRAP)) &&
            !(rc = intfmt_decode(&b, &t, itemsize, is_signed)))
            whole = int_cmp(&a, &b) > 0; /* the range crosses a wrap boundary */
        if (!rc && whole && !(rc = big_copy(&a, &min))) rc = big_copy(&b, &max);
    } else if (!rc) {
        /* saturated, or with an error the values inside the format */
        const bigint *x = int_cmp(lo, &min) < 0 ? &min : int_cmp(lo, &max) > 0 ? &max : lo;
        const bigint *y = int_cmp(hi, &min) < 0 ? &min : int_cmp(hi, &max) > 0 ? &max : hi;
        if (!(rc = big_copy(&a, x))) rc = big_copy(&b, y);
    }
    if (!rc && !(rc = intfmt_encode(low, &fits, &a, itemsize, is_signed, INT_OVERFLOW_WRAP)))
        rc = intfmt_encode(high, &fits, &b, itemsize, is_signed, INT_OVERFLOW_WRAP);
    if (!rc && !(rc = intfmt_decode(&center.m, ref, itemsize, is_signed)) && !(rc = big_copy(&bottom.m, &a)) && !(rc = big_copy(&top.m, &b))) {
        bottom.e = top.e = center.e = 0;
        rc = distance(tol, f64, &bottom, &top, &center);
    }
    for (size_t i = 0; i < sizeof bigs / sizeof *bigs; i++) big_free(bigs[i]);
    dyad_free(&bottom);
    dyad_free(&top);
    dyad_free(&center);
    return rc;
}

/* final_bounds for one real output value that every implementation computes as one of n values
   before the final rounding: kinds[i] and negs[i] give
   each value's kind and sign, a finite one nums[i] / dens[i] in magnitude; the reference is
   finite. The tolerance is the exact worst case over the finite outputs. */
int candidate_bounds(bigint *tol, bigint *low, bigint *high, int *flags, const floatfmt *out, const floatfmt *f64, int rnd, int ovf,
                     size_t n, const int *kinds, const int *negs, const bigint *nums, const bigint *dens, const bigint *reference) {
    dyad ref, point, gap, worst;
    bigint bits;
    dyad_init(&ref);
    dyad_init(&point);
    dyad_init(&gap);
    dyad_init(&worst);
    big_init(&bits);
    outputs o;
    outputs_init(&o, out, rnd, ovf);
    int finite, any = 0, rc = decode_dyad(&ref, &finite, out, reference);
    for (size_t i = 0; !rc && i < n; i++) {
        bigint zero;
        big_init(&zero);
        rc = kinds[i] == VALUE_FINITE ? floatfmt_encode_ratio(&bits, out, negs[i], &nums[i], &dens[i], rnd, ovf)
                                      : floatfmt_encode(&bits, out, kinds[i], negs[i], &zero, 0, rnd, ovf);
        if (rc || (rc = decode_dyad(&point, &finite, out, &bits))) break;
        if (!finite) {
            int kind, neg;
            if (!(rc = floatfmt_decode(&kind, &neg, &gap.m, &gap.e, out, &bits))) rc = add_value(&o, kind, neg);
            continue;
        }
        if ((rc = extend(&o, &point, &point)) || (rc = difference(&gap, &point, &ref))) break;
        gap.m.neg = 0;
        if (!any || dyad_cmp(&gap, &worst) > 0) rc = dyad_copy(&worst, &gap);
        any = 1;
    }
    if (!rc && (!any || !worst.m.n)) {
        tol->n = 0;
        tol->neg = 0;
    } else if (!rc) {
        rc = encode_dyad(tol, f64, &worst, ROUND_TOWARD_POSITIVE, OVERFLOW_IEEE);
    }
    if (!rc) rc = ends(low, high, &o);
    *flags = o.flags;
    dyad_free(&ref);
    dyad_free(&point);
    dyad_free(&gap);
    dyad_free(&worst);
    big_free(&bits);
    outputs_free(&o);
    return rc;
}

/* The order of values with infinities: -1, 0 or 1. */
static int value_cmp(const value *a, const value *b) {
    int ra = a->kind == VALUE_INFINITE ? (a->neg ? -1 : 1) : 0, rb = b->kind == VALUE_INFINITE ? (b->neg ? -1 : 1) : 0;
    if (ra != rb || ra) return ra < rb ? -1 : ra > rb;
    return dyad_cmp(&a->v, &b->v);
}

static int value_copy(value *r, const value *x) {
    r->kind = x->kind;
    r->neg = x->neg;
    return dyad_copy(&r->v, &x->v);
}

/* any_of's bounds of one part around the center's value c, from n results' bounds of it: their
   tolerances (float64 values), range ends (a NaN end: an empty range) and flags (ops/any_of.py's
   _widest). The range [lo, hi] holds every result's range (*empty when none has one), the flags
   are every result's, and the tolerance (f64's encoding) is the largest distance from c to an
   end of those ranges, rounded up: NaN when c is not finite, infinite when a result has no
   bound. */
int widest_part(bigint *tol, value *lo, value *hi, int *empty, int *flags, const floatfmt *f64, const value *c, size_t n,
                const value *tols, const value *los, const value *his, const int *part_flags) {
    int rc = 0, unbounded = 0;
    *empty = 1;
    *flags = 0;
    dyad gap, worst;
    dyad_init(&gap);
    dyad_init(&worst);
    int infinite = 0; /* an end is infinite: the distance is too */
    for (size_t i = 0; !rc && i < n; i++) {
        *flags |= part_flags[i];
        unbounded = unbounded || (tols[i].kind == VALUE_INFINITE && !tols[i].neg);
        if (los[i].kind == VALUE_NAN) continue;
        if (*empty || value_cmp(&los[i], lo) < 0) rc = value_copy(lo, &los[i]);
        if (!rc && (*empty || value_cmp(&his[i], hi) > 0)) rc = value_copy(hi, &his[i]);
        *empty = 0;
        for (int side = 0; !rc && side < 2 && c->kind == VALUE_FINITE; side++) {
            const value *end = side ? &his[i] : &los[i];
            if (end->kind != VALUE_FINITE) {
                infinite = 1;
                continue;
            }
            if ((rc = difference(&gap, &end->v, &c->v))) break;
            gap.m.neg = 0;
            if (dyad_cmp(&gap, &worst) > 0) rc = dyad_copy(&worst, &gap);
        }
    }
    if (rc) goto done;
    if (c->kind != VALUE_FINITE) {
        rc = floatfmt_special(tol, f64, ENCODING_NAN, 0);
    } else if (unbounded || infinite) {
        rc = floatfmt_special(tol, f64, ENCODING_INFINITY, 0);
    } else if (!worst.m.n) {
        tol->n = 0;
        tol->neg = 0;
    } else {
        rc = encode_dyad(tol, f64, &worst, ROUND_TOWARD_POSITIVE, OVERFLOW_IEEE);
    }
done:
    dyad_free(&gap);
    dyad_free(&worst);
    return rc;
}
