/* Emulated arithmetic of a float format (internal): the roundings of normal accumulation and the
   named algorithms (proofs/foundations.md, Lemma E), each exactly as IEEE 754 defines it for
   the format, in integer arithmetic on words. A value is a kind, a sign, and for a finite value
   sig * 2**e with sig below 2**p. Operations round to nearest, ties to even or away; an overflow
   gives an infinity, or NaN in a format without one; with flush, a result below 2**emin becomes
   a zero of its sign. For formats of at most 62 bits of precision (every accepted float format,
   at most 64 bits wide). */
#include "exactref.h"
#include "words.h"

/* The format's facts the arithmetic needs. */
int femu_set(femu *x, const floatfmt *f, int rnd, int flush) {
    x->p = f->man_bits + 1;
    x->emin = f->emin;
    x->rnd = rnd;
    x->flush = flush;
    x->has_infinity = f->special == SPECIAL_IEEE;
    /* the largest finite value, max_sig * 2**max_e with max_sig of p bits */
    bigint bits, sig;
    big_init(&bits);
    big_init(&sig);
    int kind, neg;
    int rc = floatfmt_special(&bits, f, ENCODING_MAX_FINITE, 0);
    if (!rc) rc = floatfmt_decode(&kind, &neg, &sig, &x->max_e, f, &bits);
    x->max_sig = sig.n ? sig.w[0] : 0;
    big_free(&bits);
    big_free(&sig);
    return rc;
}

static void zero(fvalue *r, int neg) {
    r->kind = VALUE_FINITE;
    r->neg = neg;
    r->sig = 0;
    r->e = 0;
}

static void special(fvalue *r, int kind, int neg) {
    r->kind = kind;
    r->neg = kind == VALUE_NAN ? 0 : neg;
    r->sig = 0;
    r->e = 0;
}

/* The overflow result: an infinity, or NaN in a format without one. */
static void overflowed(fvalue *r, const femu *x, int neg) { special(r, x->has_infinity ? VALUE_INFINITE : VALUE_NAN, neg); }

/* r = (-1)**neg (hi 2**64 + lo + a fraction in (0, 1) where sticky) * 2**e, rounded. */
void femu_round(fvalue *r, const femu *x, int neg, uint64_t hi, uint64_t lo, int64_t e, int sticky) {
    if (!hi && !lo && !sticky) {
        zero(r, neg);
        return;
    }
    int64_t bits = hi ? 64 + word_bits(hi) : word_bits(lo), top = e + bits;
    /* the unit of the result: p bits below its top, no finer than the subnormal spacing */
    int64_t q = top - x->p > x->emin - x->p + 1 ? top - x->p : x->emin - x->p + 1, shift = q - e;
    uint64_t sig;
    if (shift <= 0) {
        sig = lo << -shift; /* exact: the value has at most p bits, so hi is 0 */
    } else {
        /* sig = m >> shift, rounded by the bits shifted out (and the sticky fraction below them) */
        uint64_t rest_hi, rest_lo;
        if (shift >= 128) {
            sig = 0;
            rest_hi = hi;
            rest_lo = lo;
        } else if (shift >= 64) {
            sig = hi >> (shift - 64);
            rest_hi = shift > 64 ? hi & (((uint64_t)1 << (shift - 64)) - 1) : 0;
            rest_lo = lo;
        } else {
            sig = lo >> shift | hi << (64 - shift); /* hi << (64 - shift) holds the bits above: at most p in all */
            rest_hi = 0;
            rest_lo = lo & (((uint64_t)1 << shift) - 1);
        }
        /* the rest against half a unit, 2**(shift - 1) */
        int cmp;
        if (shift > 128) {
            cmp = -1;
        } else {
            uint64_t h_hi = shift - 1 >= 64 ? (uint64_t)1 << (shift - 1 - 64) : 0, h_lo = shift - 1 < 64 ? (uint64_t)1 << (shift - 1) : 0;
            cmp = rest_hi != h_hi ? (rest_hi > h_hi ? 1 : -1) : rest_lo != h_lo ? (rest_lo > h_lo ? 1 : -1) : sticky ? 1 : 0;
        }
        int up = cmp > 0 || (cmp == 0 && (x->rnd == ROUND_NEAREST_AWAY || (sig & 1)));
        if (up && ++sig >> x->p) {
            sig >>= 1;
            q++;
        }
    }
    /* overflow: past the largest finite value; a normal result has p bits, as that value has */
    if (q > x->max_e || (q == x->max_e && sig > x->max_sig)) {
        overflowed(r, x, neg);
        return;
    }
    r->kind = VALUE_FINITE;
    r->neg = neg;
    r->sig = sig;
    r->e = q;
    /* flush: a result below 2**emin becomes a zero of its sign */
    if (x->flush && sig && q + word_bits(sig) <= x->emin) r->sig = 0;
    if (!r->sig) r->e = 0;
}

/* r = a b, rounded (IEEE's special cases: NaN from a NaN or an infinity times zero). */
void femu_mul(fvalue *r, const femu *x, const fvalue *a, const fvalue *b) {
    int neg = a->neg != b->neg;
    if (a->kind == VALUE_NAN || b->kind == VALUE_NAN) {
        special(r, VALUE_NAN, 0);
        return;
    }
    if (a->kind == VALUE_INFINITE || b->kind == VALUE_INFINITE) {
        if ((a->kind == VALUE_FINITE && !a->sig) || (b->kind == VALUE_FINITE && !b->sig))
            special(r, VALUE_NAN, 0);
        else
            special(r, VALUE_INFINITE, neg);
        return;
    }
    uint64_t hi, lo = word_mul(&hi, a->sig, b->sig);
    if (!hi && !lo) {
        zero(r, neg);
        return;
    }
    femu_round(r, x, neg, hi, lo, a->e + b->e, 0);
}

/* a with its significand moved up to p bits (a value's sig may have fewer) */
static fvalue widened(const fvalue *a, const femu *x) {
    fvalue r = *a;
    int64_t s = x->p - word_bits(r.sig);
    if (s > 0) {
        r.sig <<= s;
        r.e -= s;
    }
    return r;
}

/* r = a + b, rounded. An exact zero sum of opposite signs is +0 (rounding to nearest); zeros of
   one sign keep it. */
void femu_add(fvalue *r, const femu *x, const fvalue *a, const fvalue *b) {
    if (a->kind == VALUE_NAN || b->kind == VALUE_NAN || (a->kind == VALUE_INFINITE && b->kind == VALUE_INFINITE && a->neg != b->neg)) {
        special(r, VALUE_NAN, 0);
        return;
    }
    if (a->kind == VALUE_INFINITE || b->kind == VALUE_INFINITE) {
        const fvalue *inf = a->kind == VALUE_INFINITE ? a : b;
        special(r, VALUE_INFINITE, inf->neg);
        return;
    }
    if (!a->sig || !b->sig) {
        if (!a->sig && !b->sig) {
            zero(r, a->neg && b->neg);
            return;
        }
        *r = a->sig ? *a : *b; /* a value of the format: rounding keeps it */
        return;
    }
    /* both at p bits, so that big's unit is p bits below its top (the far case below needs it) */
    fvalue wa = widened(a, x), wb = widened(b, x);
    a = &wa;
    b = &wb;
    /* big: the operand with the higher unit */
    const fvalue *big = a->e >= b->e ? a : b, *small = a->e >= b->e ? b : a;
    int64_t gap = big->e - small->e;
    uint64_t hi, lo;
    int neg, sticky = 0;
    int64_t e;
    if (gap <= x->p + 3) {
        /* exact in two words: big's significand moved up by the gap, below 2**(2p + 3) <= 2**127 */
        uint64_t bh = !gap ? 0 : gap < 64 ? big->sig >> (64 - gap) : big->sig << (gap - 64), bl = gap < 64 ? big->sig << gap : 0, sl = small->sig;
        e = small->e;
        if (big->neg == small->neg) {
            lo = bl + sl;
            hi = bh + (lo < bl);
            neg = big->neg;
        } else {
            /* the larger magnitude minus the smaller */
            int big_larger = bh || bl >= sl;
            if (big_larger) {
                lo = bl - sl;
                hi = bh - (bl < sl);
                neg = big->neg;
            } else {
                lo = sl - bl;
                hi = 0;
                neg = small->neg;
            }
            if (!hi && !lo) {
                zero(r, 0); /* an exact zero of opposite signs: +0 to nearest */
                return;
            }
        }
    } else {
        /* small is below an eighth of big's unit (gap > p + 3, small < 2**(p + small's e)): big with three guard bits
           (p + 3 <= 65 bits, in two words) and a sticky bit */
        e = big->e - 3;
        lo = big->sig << 3;
        hi = big->sig >> 61;
        neg = big->neg;
        if (big->neg != small->neg) {
            hi -= !lo;
            lo -= 1;
        }
        sticky = 1;
    }
    femu_round(r, x, neg, hi, lo, e, sticky);
}

/* r = the value a slot of an operand holds (a slot's integer at the operand's exponent, or a
   flagged value's encoding), read in the emulated format's terms: with flush, an input below
   2**emin is a zero of its sign. Integers wider than the format stay exact here: the callers
   round them when they enter the arithmetic. */
void femu_read(fvalue *r, const femu *x, int kind, int neg, uint64_t sig, int64_t e) {
    if (kind != VALUE_FINITE) {
        special(r, kind, neg);
        return;
    }
    r->kind = VALUE_FINITE;
    r->neg = neg;
    r->sig = sig;
    r->e = sig ? e : 0;
    if (x->flush && sig && e + word_bits(sig) <= x->emin) {
        r->sig = 0;
        r->e = 0;
    }
}

static int finite_value(const fvalue *v) { return v->kind == VALUE_FINITE; }

/* Lemma O's overflow-safe sum of the n finite values v (index order), in place: a value that
   cannot be added without overflowing starts a new partial sum, pass after pass; a pass that
   merges nothing gives what its last addition gave. */
void femu_slot_sum(fvalue *r, const femu *x, fvalue *v, size_t n) {
    fvalue s, overflow;
    zero(&overflow, 0);
    while (n > 1) {
        size_t out = 1;
        for (size_t i = 1; i < n; i++) {
            femu_add(&s, x, &v[out - 1], &v[i]);
            if (finite_value(&s)) {
                v[out - 1] = s;
            } else {
                v[out++] = v[i];
                overflow = s;
            }
        }
        if (out == n) {
            *r = overflow;
            return;
        }
        n = out;
    }
    if (n)
        *r = v[0];
    else
        zero(r, 0);
}

/* w = (hi 2**64 + lo) 2**s in four words, for s >= 0 and a result below 2**256. */
static void place4(uint64_t w[4], uint64_t hi, uint64_t lo, int64_t s) {
    uint64_t v[2] = {lo, hi};
    int words = (int)(s / 64), bits = (int)(s % 64);
    w[0] = w[1] = w[2] = w[3] = 0;
    for (int k = 0; k < 2 && k + words < 4; k++) {
        w[k + words] |= v[k] << bits;
        if (bits && k + words + 1 < 4) w[k + words + 1] |= v[k] >> (64 - bits);
    }
}

/* r = a + (-1)**neg (hi 2**64 + lo) 2**e rounded once, for a finite nonzero value a with a one-word
   significand (a value of the format, or an exact value femu_add_exact takes) and a nonzero exact term of
   at most 128 bits (an exact product: a fused multiply-add; Lemma E). Of the two operands, big has the
   higher top. A small operand wholly below 2**G, G = min(big's unit, big's top - p - 3), becomes (its
   sign) 2**(G - 1): big is a multiple of 2**G and every value and midpoint the sum may round to a multiple
   of 2**(G + 1), so the rounding is the same for every small in (0, 2**G). The exact sum then spans fewer
   than max(big's width, p + 3) + the other's width <= 65 + 128 bits, in four words. */
static void fused_add(fvalue *r, const femu *x, const fvalue *a, int neg, uint64_t hi, uint64_t lo, int64_t e) {
    struct {
        int neg;
        uint64_t hi, lo;
        int64_t e, top;
    } o[2] = {{a->neg, 0, a->sig, a->e, a->e + word_bits(a->sig)}, {neg, hi, lo, e, e + (hi ? 64 + word_bits(hi) : word_bits(lo))}};
    int b = o[1].top > o[0].top, s = !b;
    int64_t g = o[b].e < o[b].top - x->p - 3 ? o[b].e : o[b].top - x->p - 3;
    if (o[s].top <= g) {
        o[s].hi = 0;
        o[s].lo = 1;
        o[s].e = g - 1;
    }
    int64_t base = o[0].e < o[1].e ? o[0].e : o[1].e;
    if (o[b].top - base <= 127) {
        /* both below 2**127 at the unit 2**base, so their sum fits two words */
        uint64_t v[2][2];
        for (int k = 0; k < 2; k++) {
            int64_t sh = o[k].e - base; /* below 127 */
            v[k][1] = sh >= 64 ? o[k].lo << (sh - 64) : sh ? o[k].hi << sh | o[k].lo >> (64 - sh) : o[k].hi;
            v[k][0] = sh >= 64 ? 0 : o[k].lo << sh;
        }
        uint64_t h, l;
        int sign = o[b].neg;
        if (o[0].neg == o[1].neg) {
            l = v[0][0] + v[1][0];
            h = v[0][1] + v[1][1] + (l < v[0][0]);
        } else {
            int first = v[0][1] != v[1][1] ? v[0][1] > v[1][1] : v[0][0] >= v[1][0]; /* |o[0]| >= |o[1]| */
            const uint64_t *big = v[first ? 0 : 1], *small = v[first ? 1 : 0];
            l = big[0] - small[0];
            h = big[1] - small[1] - (big[0] < small[0]);
            if (!h && !l) {
                zero(r, 0); /* an exact zero of opposite signs: +0 to nearest */
                return;
            }
            sign = o[first ? 0 : 1].neg;
        }
        femu_round(r, x, sign, h, l, base, 0);
        return;
    }
    uint64_t wb[4], ws[4], sum[4];
    place4(wb, o[b].hi, o[b].lo, o[b].e - base);
    place4(ws, o[s].hi, o[s].lo, o[s].e - base);
    int sign = o[b].neg;
    if (o[b].neg == o[s].neg) {
        nat_add(sum, wb, ws, 4, 0);
    } else {
        int c = nat_cmp(wb, ws, 4);
        if (!c) {
            zero(r, 0); /* an exact zero of opposite signs: +0 to nearest */
            return;
        }
        if (c > 0) {
            nat_sub(sum, wb, ws, 4, 0);
        } else {
            nat_sub(sum, ws, wb, 4, 0);
            sign = o[s].neg;
        }
    }
    /* the top two words from the highest nonzero one, the rest as sticky */
    int t = sum[3] ? 3 : sum[2] ? 2 : 1;
    femu_round(r, x, sign, sum[t], sum[t - 1], base + 64 * (int64_t)(t - 1), t == 3 ? (sum[1] | sum[0]) != 0 : t == 2 ? sum[0] != 0 : 0);
}

/* r = a + t rounded once, for a value a of the format and a term's exact product (a fused multiply-add). */
static void term_add(fvalue *r, const femu *x, const fvalue *a, const fterm *t) {
    if (a->kind != VALUE_FINITE || t->v.kind != VALUE_FINITE || !(t->hi | t->lo)) {
        fvalue tv = t->v;
        tv.sig = 0;
        femu_add(r, x, a, &tv); /* IEEE's rules where a term is not finite, and the zero rules */
        return;
    }
    if (!a->sig)
        femu_round(r, x, t->v.neg, t->hi, t->lo, t->e, 0); /* 0 + t: t rounded */
    else
        fused_add(r, x, a, t->v.neg, t->hi, t->lo, t->e);
}

/* r = a + b rounded once, for values a and b with one-word significands that need not be values of the format: exact
   terms the model does not round on their own (add's with a unit alpha, gemm's beta c with a unit beta; foundations.md,
   Lemma U), IEEE's rules where one is not finite, a zero sum of opposite signs +0, and 0 + b, b rounded. */
void femu_add_exact(fvalue *r, const femu *x, const fvalue *a, const fvalue *b) {
    if (a->kind != VALUE_FINITE || b->kind != VALUE_FINITE || (!a->sig && !b->sig)) {
        femu_add(r, x, a, b); /* by the kinds and signs alone */
        return;
    }
    if (!a->sig || !b->sig) {
        const fvalue *v = a->sig ? a : b;
        femu_round(r, x, v->neg, 0, v->sig, v->e, 0);
        return;
    }
    fused_add(r, x, a, b->neg, 0, b->sig, b->e);
}

/* r = the nonzero exact sum s rounded once, through formats.c. */
static int round_exact(fvalue *r, const femu *x, const floatfmt *f, const dyad *s) {
    bigint bits, sig, mag = s->m; /* a read-only view of the magnitude */
    mag.neg = 0;
    big_init(&bits);
    big_init(&sig);
    int kind, neg, rc;
    if (!(rc = floatfmt_encode(&bits, f, VALUE_FINITE, s->m.neg, &mag, s->e, x->rnd, OVERFLOW_NAN)) &&
        !(rc = floatfmt_decode(&kind, &neg, &sig, &r->e, f, &bits))) {
        r->kind = kind;
        r->neg = neg;
        r->sig = sig.n ? sig.w[0] : 0;
        if (kind != VALUE_FINITE) r->e = 0;
        if (x->flush && r->kind == VALUE_FINITE && r->sig && r->e + word_bits(r->sig) <= x->emin) r->sig = 0;
        if (r->kind == VALUE_FINITE && !r->sig) r->e = 0;
    }
    big_free(&bits);
    big_free(&sig);
    return rc;
}

/* A named algorithm's output before its final rounding (ops/matmul.py's MatmulAlgorithm.run,
   accumulation.py's methods): method 1 sequential, 2 pairwise, 3 blocked (block terms), 4 exact
   accumulation, over the n terms: exact products (fused multiply-adds), or each rounded to the
   accumulator first (fterm). Sums start from an exact 0, which counts as +0. */
static void run_range(fvalue *r, const femu *x, const fterm *terms, int fma, size_t lo, size_t hi);

static void run_sequential(fvalue *r, const femu *x, const fterm *terms, int fma, size_t lo, size_t hi) {
    zero(r, 0);
    for (size_t q = lo; q < hi; q++) {
        if (fma)
            term_add(r, x, r, &terms[q]);
        else
            femu_add(r, x, r, &terms[q].v);
    }
}

/* pairwise over [lo, hi): the halves' sums added, the first half the larger by one where the count is odd */
static void run_range(fvalue *r, const femu *x, const fterm *terms, int fma, size_t lo, size_t hi) {
    if (hi - lo == 1) {
        run_sequential(r, x, terms, fma, lo, hi); /* the term rounded once */
        return;
    }
    size_t mid = lo + (hi - lo + 1) / 2;
    fvalue left, right;
    run_range(&left, x, terms, fma, lo, mid);
    run_range(&right, x, terms, fma, mid, hi);
    femu_add(r, x, &left, &right);
}

int femu_run(fvalue *r, const femu *x, const floatfmt *f, int method, size_t block, const fterm *terms, int fma, size_t n, fvalue *scratch) {
    if (!n) {
        zero(r, 0);
        return 0;
    }
    if (method == 2) {
        run_range(r, x, terms, fma, 0, n);
        return 0;
    }
    if (method == 3) {
        /* each block sequentially, then the block sums sequentially */
        size_t blocks = 0;
        for (size_t start = 0; start < n; start += block) run_sequential(&scratch[blocks++], x, terms, fma, start, start + block < n ? start + block : n);
        zero(r, 0);
        for (size_t k = 0; k < blocks; k++) femu_add(r, x, r, &scratch[k]);
        return 0;
    }
    if (method == 4) {
        /* the exact sum, rounded once: IEEE's rules where a term is not finite; an exact zero sum is +0 */
        int nan = 0, plus = 0, minus = 0, rc = 0;
        dyad s, t;
        dyad_init(&s);
        dyad_init(&t);
        for (size_t q = 0; !rc && q < n; q++) {
            const fterm *u = &terms[q];
            if (u->v.kind == VALUE_NAN) nan = 1;
            if (u->v.kind == VALUE_INFINITE) *(u->v.neg ? &minus : &plus) = 1;
            if (u->v.kind != VALUE_FINITE || !(u->hi | u->lo)) continue;
            if (!(rc = big_reserve(&t.m, 2))) {
                t.m.w[0] = u->lo;
                t.m.w[1] = u->hi;
                t.m.n = 2;
                big_trim(&t.m);
                t.m.neg = u->v.neg;
                t.e = u->e;
                rc = dyad_add_exact(&s, &s, &t);
            }
        }
        if (!rc && (nan || (plus && minus)))
            special(r, VALUE_NAN, 0);
        else if (!rc && (plus || minus))
            special(r, VALUE_INFINITE, minus);
        else if (!rc && !s.m.n)
            zero(r, 0);
        else if (!rc)
            rc = round_exact(r, x, f, &s);
        dyad_free(&s);
        dyad_free(&t);
        return rc;
    }
    run_sequential(r, x, terms, fma, 0, n);
    return 0;
}
