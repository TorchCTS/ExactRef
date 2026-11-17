/* Dyadic numbers m * 2**e with directed rounding (internal): exactref's one implementation of
   rounding a dyadic result to at most p significant bits toward -inf or +inf. Each operation
   rounds its exact result once: `rnd` keeps an operand of at most p bits unchanged and otherwise
   shifts its integer, `add` never aligns operands far apart exactly, `div` and `sqrt` compute just
   enough quotient or root bits and a sticky bit. Integers here are signed magnitudes of any size, held
   in the integer itself up to BIG_INLINE words and allocated beyond. Built on them: intervals of dyadics rounded
   outward, and two's-complement integers of a fixed number of words. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

#define EXP_LIMIT ((int64_t)1 << 62)

void big_init(bigint *b) {
    b->w = b->small;
    b->n = 0;
    b->cap = BIG_INLINE;
    b->neg = 0;
}

void big_free(bigint *b) {
    if (b->w != b->small) free(b->w);
    big_init(b);
}

int big_reserve(bigint *b, size_t cap) {
    if (cap <= b->cap) return 0;
    uint64_t *w = malloc(cap * sizeof *w);
    if (!w) return EXACTREF_NOMEM;
    if (b->n) memcpy(w, b->w, b->n * sizeof *w);
    if (b->w != b->small) free(b->w);
    b->w = w;
    b->cap = cap;
    return 0;
}

void big_trim(bigint *b) {
    while (b->n && !b->w[b->n - 1]) b->n--;
    if (!b->n) b->neg = 0;
}

/* r takes t's value, and t is left empty (r may hold its own storage, which is freed). */
static void move(bigint *r, bigint *t) {
    if (r == t) return;
    if (r->w != r->small) free(r->w);
    if (t->w == t->small) {
        r->w = r->small;
        r->cap = BIG_INLINE;
        if (t->n) memcpy(r->small, t->small, t->n * sizeof *t->small);
    } else {
        r->w = t->w;
        r->cap = t->cap;
    }
    r->n = t->n;
    r->neg = t->neg;
    big_init(t);
}

static int copy(bigint *r, const bigint *a) {
    if (r == a) return 0;
    if (big_reserve(r, a->n)) return EXACTREF_NOMEM;
    if (a->n <= 8)
        for (size_t i = 0; i < a->n; i++) r->w[i] = a->w[i];
    else
        memcpy(r->w, a->w, a->n * sizeof *a->w);
    r->n = a->n;
    r->neg = a->neg;
    return 0;
}

int64_t big_bits(const bigint *b) { return b->n ? 64 * (int64_t)(b->n - 1) + word_bits(b->w[b->n - 1]) : 0; }

int big_set_word(bigint *b, uint64_t v, int neg) {
    if (big_reserve(b, 1)) return EXACTREF_NOMEM;
    b->w[0] = v;
    b->n = 1;
    b->neg = neg;
    big_trim(b);
    return 0;
}

/* r = a * 2**s, s >= 0 (r may be a). */
static int shl(bigint *r, const bigint *a, int64_t s) {
    if (!a->n) {
        r->n = 0;
        r->neg = 0;
        return 0;
    }
    size_t words = (size_t)(s / 64), n = a->n + words + 1;
    int bits = (int)(s % 64), neg = a->neg;
    if (big_reserve(r, n)) return EXACTREF_NOMEM;
    memmove(r->w + words, a->w, a->n * sizeof *a->w);
    r->w[a->n + words] = bits ? nat_shl(r->w + words, r->w + words, a->n, bits) : 0;
    memset(r->w, 0, words * sizeof *r->w);
    r->n = n;
    r->neg = neg;
    big_trim(r);
    return 0;
}

/* r = floor(a / 2**s), or the ceiling when up, s >= 0 (r may be a). */
static int shr_dir(bigint *r, const bigint *a, int64_t s, int up) {
    if (!a->n) {
        r->n = 0;
        r->neg = 0;
        return 0;
    }
    int neg = a->neg;
    uint64_t sticky = 0;
    size_t words = (size_t)(s / 64);
    int bits = (int)(s % 64);
    if (words >= a->n) {
        sticky = 1; /* a != 0 and every bit is shifted out */
        r->n = 0;
    } else {
        size_t n = a->n - words;
        if (big_reserve(r, n)) return EXACTREF_NOMEM;
        for (size_t i = 0; i < words; i++) sticky |= a->w[i];
        memmove(r->w, a->w + words, n * sizeof *a->w);
        if (bits) sticky |= nat_shr(r->w, r->w, n, bits, 0);
        r->n = n;
    }
    big_trim(r);
    /* the magnitude's quotient goes up by one when the rounding moves away from zero */
    if (sticky && (up ? !neg : neg)) {
        if (big_reserve(r, r->n + 1)) return EXACTREF_NOMEM;
        r->w[r->n] = 0;
        r->n++;
        nat_add_word(r->w, 1, r->n);
        big_trim(r);
    }
    r->neg = r->n ? neg : 0;
    return 0;
}

/* r = a + b (signed; r may be a or b). */
int big_add(bigint *r, const bigint *a, const bigint *b) {
    if (!b->n) return copy(r, a);
    if (!a->n) return copy(r, b);
    const bigint *x = a, *y = b;
    if (a->neg != b->neg) {
        /* subtract the smaller magnitude from the larger */
        int c = a->n != b->n ? (a->n < b->n ? -1 : 1) : nat_cmp(a->w, b->w, a->n);
        if (!c) {
            r->n = 0;
            r->neg = 0;
            return 0;
        }
        if (c < 0) {
            x = b;
            y = a;
        }
        bigint t;
        big_init(&t);
        if (big_reserve(&t, x->n)) return EXACTREF_NOMEM;
        uint64_t borrow = nat_sub(t.w, x->w, y->w, y->n, 0);
        if (x->n > y->n) memcpy(t.w + y->n, x->w + y->n, (x->n - y->n) * sizeof *t.w);
        nat_sub_word(t.w + y->n, borrow, x->n - y->n);
        t.n = x->n;
        t.neg = x->neg;
        big_trim(&t);
        move(r, &t);
        return 0;
    }
    if (x->n < y->n) {
        x = b;
        y = a;
    }
    bigint t;
    big_init(&t);
    if (big_reserve(&t, x->n + 1)) return EXACTREF_NOMEM;
    uint64_t carry = nat_add(t.w, x->w, y->w, y->n, 0);
    if (x->n > y->n) memcpy(t.w + y->n, x->w + y->n, (x->n - y->n) * sizeof *t.w);
    carry = nat_add_word(t.w + y->n, carry, x->n - y->n);
    t.w[x->n] = carry;
    t.n = x->n + 1;
    t.neg = x->neg;
    big_trim(&t);
    move(r, &t);
    return 0;
}

/* r = a * b (r may be a or b). */
static int mul_int(bigint *r, const bigint *a, const bigint *b) {
    if (!a->n || !b->n) {
        r->n = 0;
        r->neg = 0;
        return 0;
    }
    bigint t;
    big_init(&t);
    if (big_reserve(&t, a->n + b->n)) return EXACTREF_NOMEM;
    nat_mul(t.w, a->w, a->n, b->w, b->n);
    t.n = a->n + b->n;
    t.neg = a->neg != b->neg;
    big_trim(&t);
    move(r, &t);
    return 0;
}

/* For magnitudes a >= 0 and b > 0: q = floor(a / b) and whether the remainder is zero. */
static int divmod_mag(bigint *q, int *exact, const bigint *a, const bigint *b) {
    bigint t;
    big_init(&t);
    if (a->n < b->n) {
        *exact = a->n == 0;
        q->n = 0;
        q->neg = 0;
        return 0;
    }
    if (b->n == 1) {
        if (big_reserve(&t, a->n)) return EXACTREF_NOMEM;
        uint64_t rem = nat_div_word(t.w, a->w, a->n, b->w[0]);
        t.n = a->n;
        *exact = rem == 0;
    } else {
        size_t na = a->n + 1, nb = b->n;
        int s = word_clz(b->w[nb - 1]);
        uint64_t an_small[2 * BIG_INLINE + 1], bn_small[BIG_INLINE];
        uint64_t *an = na <= 2 * BIG_INLINE + 1 ? an_small : malloc(na * sizeof *an);
        uint64_t *bn = nb <= BIG_INLINE ? bn_small : malloc(nb * sizeof *bn);
        if (!an || !bn || big_reserve(&t, na - nb + 1)) {
            if (an != an_small) free(an);
            if (bn != bn_small) free(bn);
            big_free(&t);
            return EXACTREF_NOMEM;
        }
        if (s) {
            nat_shl(bn, b->w, nb, s);
            an[a->n] = nat_shl(an, a->w, a->n, s);
        } else {
            memcpy(bn, b->w, nb * sizeof *bn);
            memcpy(an, a->w, a->n * sizeof *an);
            an[a->n] = 0;
        }
        nat_divrem_norm(t.w, an, na, bn, nb);
        t.n = na - nb + 1;
        uint64_t any = 0;
        for (size_t i = 0; i < nb; i++) any |= an[i];
        *exact = any == 0;
        if (an != an_small) free(an);
        if (bn != bn_small) free(bn);
    }
    t.neg = 0;
    big_trim(&t);
    move(q, &t);
    return 0;
}

/* For a magnitude a >= 0: s = floor(sqrt(a)) and whether a is a perfect square. */
static int isqrt_mag(bigint *s, int *exact, const bigint *a) {
    if (!a->n) {
        s->n = 0;
        s->neg = 0;
        *exact = 1;
        return 0;
    }
    int64_t bits = big_bits(a);
    size_t n = (size_t)((bits + 127) / 128);
    int64_t gap = 128 * (int64_t)n - bits, k = gap / 2; /* a * 4**k has 128 n or 128 n - 1 bits */
    bigint t;
    big_init(&t);
    uint64_t buf_small[2 * BIG_INLINE], tmp_small[BIG_INLINE / 2 + 1];
    uint64_t *buf = n <= BIG_INLINE ? buf_small : malloc(2 * n * sizeof *buf);
    uint64_t *tmp = n <= BIG_INLINE ? tmp_small : malloc((n / 2 + 1) * sizeof *tmp);
    if (!buf || !tmp || big_reserve(&t, n) || shl(&t, a, 2 * k)) {
        if (buf != buf_small) free(buf);
        if (tmp != tmp_small) free(tmp);
        big_free(&t);
        return EXACTREF_NOMEM;
    }
    memset(buf, 0, 2 * n * sizeof *buf);
    memcpy(buf, t.w, t.n * sizeof *buf);
    uint64_t rh = nat_sqrtrem(t.w, buf, n, tmp);
    uint64_t any = rh;
    for (size_t i = 0; i < n; i++) any |= buf[i];
    *exact = any == 0;
    t.n = n;
    t.neg = 0;
    big_trim(&t);
    int rc = shr_dir(&t, &t, k, 0);
    if (buf != buf_small) free(buf);
    if (tmp != tmp_small) free(tmp);
    if (rc) {
        big_free(&t);
        return rc;
    }
    move(s, &t);
    return 0;
}

static int exp_add(int64_t *r, int64_t a, int64_t b) {
    if ((b > 0 && a > EXP_LIMIT - b) || (b < 0 && a < -EXP_LIMIT - b)) return EXACTREF_OVERFLOW;
    *r = a + b;
    if (*r > EXP_LIMIT || *r < -EXP_LIMIT) return EXACTREF_OVERFLOW;
    return 0;
}

/* x times 2**s in place: EXACTREF_OVERFLOW past the exponent limit (a zero stays 0 2**0) */
int dyad_shift(dyad *x, int64_t s) { return x->m.n ? exp_add(&x->e, x->e, s) : 0; }

void dyad_init(dyad *x) {
    big_init(&x->m);
    x->e = 0;
}

void dyad_free(dyad *x) { big_free(&x->m); }

/* The sums and products of operands that fit SMALL_WORDS words, in local words: the same exact result and rounding as
   the general path, without its temporaries. */
#define SMALL_WORDS 8

/* r = (-1)**neg s 2**e rounded to at most p bits, for a magnitude s of n words (dyad_rnd's rounding); s is changed. */
static int round_words(dyad *r, uint64_t *s, size_t n, int neg, int64_t e, int64_t p, int up) {
    while (n && !s[n - 1]) n--;
    if (!n) {
        r->m.n = 0;
        r->m.neg = 0;
        r->e = 0;
        return 0;
    }
    int64_t bits = 64 * (int64_t)(n - 1) + word_bits(s[n - 1]);
    if (bits > p) {
        int64_t shift = bits - p;
        int rc = exp_add(&e, e, shift);
        if (rc) return rc;
        size_t words = (size_t)(shift / 64);
        int b = (int)(shift % 64);
        uint64_t sticky = 0;
        for (size_t i = 0; i < words; i++) sticky |= s[i];
        if (b) sticky |= s[words] & (((uint64_t)1 << b) - 1);
        for (size_t i = 0; i + words < n; i++) s[i] = b ? s[i + words] >> b | (i + words + 1 < n ? s[i + words + 1] << (64 - b) : 0) : s[i + words];
        n -= words;
        if (sticky && (up ? !neg : neg)) { /* the magnitude goes up by one when the rounding moves away from zero */
            s[n] = 0;
            nat_add_word(s, 1, n + 1);
            n++;
        }
        while (n && !s[n - 1]) n--;
        if (64 * (int64_t)(n - 1) + word_bits(s[n - 1]) > p) { /* the carry made it 2**p: 2**(p - 1) one place up */
            if ((rc = exp_add(&e, e, 1))) return rc;
            nat_shr(s, s, n, 1, 0);
            while (n && !s[n - 1]) n--;
        }
    }
    if (big_reserve(&r->m, n)) return EXACTREF_NOMEM;
    for (size_t i = 0; i < n; i++) r->m.w[i] = s[i];
    r->m.n = n;
    r->m.neg = neg;
    r->e = e;
    return 0;
}

/* s (w words, zeroed) gets |a| 2**shift, for 0 <= shift and a result below 2**(64 w) */
static void place_words(uint64_t *s, size_t w, const bigint *a, int64_t shift) {
    size_t words = (size_t)(shift / 64);
    int b = (int)(shift % 64);
    for (size_t i = 0; i < a->n && i + words < w; i++) {
        s[i + words] |= a->w[i] << b;
        if (b && i + words + 1 < w) s[i + words + 1] |= a->w[i] >> (64 - b);
    }
}

/* r = x rounded to at most p bits; r may be x. */
int dyad_rnd(dyad *r, const dyad *x, int64_t p, int up) {
    int64_t n = big_bits(&x->m), e = x->e;
    if (n <= p) {
        r->e = x->m.n ? e : 0;
        return copy(&r->m, &x->m);
    }
    if (x->m.n <= SMALL_WORDS) {
        uint64_t s[SMALL_WORDS + 1];
        for (size_t i = 0; i < x->m.n; i++) s[i] = x->m.w[i];
        return round_words(r, s, x->m.n, x->m.neg, e, p, up);
    }
    int64_t shift = n - p;
    int rc = exp_add(&e, e, shift);
    if (rc || (rc = shr_dir(&r->m, &x->m, shift, up))) return rc;
    if (big_bits(&r->m) > p) { /* the carry made it 2**p: 2**(p - 1) one place up */
        if ((rc = exp_add(&e, e, 1))) return rc;
        nat_shr(r->m.w, r->m.w, r->m.n, 1, 0);
        big_trim(&r->m);
    }
    r->e = e;
    return 0;
}

/* r = x + y rounded. */
int dyad_add(dyad *r, const dyad *x, const dyad *y, int64_t p, int up) {
    if (!y->m.n) return dyad_rnd(r, x, p, up);
    if (!x->m.n) return dyad_rnd(r, y, p, up);
    const dyad *a = x, *b = y;
    int64_t ta = a->e + big_bits(&a->m), tb = b->e + big_bits(&b->m);
    if (ta < tb) {
        a = y;
        b = x;
        int64_t t = ta;
        ta = tb;
        tb = t;
    }
    int64_t unit = ta - p - 2 < a->e ? ta - p - 2 : a->e;
    int far = tb <= unit;
    int64_t base = far ? unit - 1 : a->e < b->e ? a->e : b->e;
    if (ta - base < 64 * SMALL_WORDS - 1) {
        /* both aligned at 2**base in w local words (b as sign(b) 2**(unit - 1) when far below): the sum is below
           2**(ta - base + 1) */
        size_t w = (size_t)((ta - base + 64) / 64);
        uint64_t sa[SMALL_WORDS + 1], sb[SMALL_WORDS + 1];
        for (size_t i = 0; i < w; i++) sa[i] = sb[i] = 0;
        place_words(sa, w, &a->m, a->e - base);
        if (far)
            sb[0] = 1;
        else
            place_words(sb, w, &b->m, b->e - base);
        int neg = a->m.neg;
        if (a->m.neg == b->m.neg) {
            nat_add(sa, sa, sb, w, 0);
        } else if (nat_cmp(sa, sb, w) < 0) {
            nat_sub(sa, sb, sa, w, 0);
            neg = b->m.neg;
        } else {
            nat_sub(sa, sa, sb, w, 0);
        }
        return round_words(r, sa, w, neg, base, p, up);
    }
    dyad t;
    dyad_init(&t);
    int rc;
    if (far) {
        /* |b| < 2**unit and every p-bit value near a is a multiple of 2**unit: a + b rounds like
           a + sign(b) 2**(unit - 1) */
        bigint one;
        big_init(&one);
        if (!(rc = shl(&t.m, &a->m, a->e - unit + 1)) && !(rc = big_set_word(&one, 1, b->m.neg)))
            rc = big_add(&t.m, &t.m, &one);
        big_free(&one);
        t.e = unit - 1;
    } else if (a->e >= b->e) {
        if (!(rc = shl(&t.m, &a->m, a->e - b->e))) rc = big_add(&t.m, &t.m, &b->m);
        t.e = b->e;
    } else {
        if (!(rc = shl(&t.m, &b->m, b->e - a->e))) rc = big_add(&t.m, &a->m, &t.m);
        t.e = a->e;
    }
    if (!t.m.n) t.e = 0; /* an exact sum that cancels: 0 2**0 */
    if (!rc) rc = dyad_rnd(r, &t, p, up);
    dyad_free(&t);
    return rc;
}

/* r = x * y rounded. */
int dyad_mul(dyad *r, const dyad *x, const dyad *y, int64_t p, int up) {
    if (!x->m.n || !y->m.n) return dyad_set_si(r, 0, 0);
    if (x->m.n + y->m.n <= SMALL_WORDS) {
        int64_t e;
        int rc = exp_add(&e, x->e, y->e);
        if (rc) return rc;
        uint64_t s[SMALL_WORDS + 1];
        nat_mul(s, x->m.w, x->m.n, y->m.w, y->m.n);
        return round_words(r, s, x->m.n + y->m.n, x->m.neg != y->m.neg, e, p, up);
    }
    dyad t;
    dyad_init(&t);
    int rc = exp_add(&t.e, x->e, y->e);
    if (!rc) rc = mul_int(&t.m, &x->m, &y->m);
    if (!rc) rc = dyad_rnd(r, &t, p, up);
    dyad_free(&t);
    return rc;
}

/* r = x * count, rounded up to w bits. */
int dyad_times(dyad *r, const dyad *x, int64_t count, int64_t w) {
    dyad c;
    dyad_init(&c);
    int rc = dyad_set_si(&c, count, 0);
    if (!rc) rc = dyad_mul(r, x, &c, w, 1);
    dyad_free(&c);
    return rc;
}

/* r = x + y exactly (r may be x or y). */
int dyad_add_exact(dyad *r, const dyad *x, const dyad *y) {
    if (!y->m.n) return dyad_copy(r, x);
    if (!x->m.n) return dyad_copy(r, y);
    const dyad *hi = x->e >= y->e ? x : y, *lo = x->e >= y->e ? y : x;
    int64_t tx = x->e + big_bits(&x->m), ty = y->e + big_bits(&y->m);
    int64_t top = tx > ty ? tx : ty;
    if (top - lo->e < 64 * SMALL_WORDS - 1) {
        size_t w = (size_t)((top - lo->e + 64) / 64);
        uint64_t sx[SMALL_WORDS + 1], sy[SMALL_WORDS + 1];
        for (size_t i = 0; i < w; i++) sx[i] = sy[i] = 0;
        place_words(sx, w, &hi->m, hi->e - lo->e);
        place_words(sy, w, &lo->m, 0);
        int neg = hi->m.neg;
        if (hi->m.neg == lo->m.neg) {
            nat_add(sx, sx, sy, w, 0);
        } else if (nat_cmp(sx, sy, w) < 0) {
            nat_sub(sx, sy, sx, w, 0);
            neg = lo->m.neg;
        } else {
            nat_sub(sx, sx, sy, w, 0);
        }
        return round_words(r, sx, w, neg, lo->e, INT64_MAX, 0);
    }
    bigint t;
    big_init(&t);
    int64_t e = lo->e;
    int rc = shl(&t, &hi->m, hi->e - lo->e);
    if (!rc) rc = big_add(&r->m, &t, &lo->m);
    r->e = r->m.n ? e : 0;
    big_free(&t);
    return rc;
}

/* r = x * y exactly (r may be x or y). */
int dyad_mul_exact(dyad *r, const dyad *x, const dyad *y) {
    if (!x->m.n || !y->m.n) return dyad_set_si(r, 0, 0);
    int64_t e = 0;
    int rc = exp_add(&e, x->e, y->e);
    if (!rc && x->m.n && y->m.n && x->m.n + y->m.n <= SMALL_WORDS) {
        uint64_t s[SMALL_WORDS + 1];
        nat_mul(s, x->m.w, x->m.n, y->m.w, y->m.n);
        return round_words(r, s, x->m.n + y->m.n, x->m.neg != y->m.neg, e, INT64_MAX, 0);
    }
    if (!rc) rc = mul_int(&r->m, &x->m, &y->m);
    r->e = e;
    return rc;
}

/* r = x / y rounded. */
int dyad_div(dyad *r, const dyad *x, const dyad *y, int64_t p, int up) {
    if (!y->m.n) return EXACTREF_ZERO_DIVISION;
    if (!x->m.n) {
        r->m.n = 0;
        r->m.neg = 0;
        r->e = 0;
        return 0;
    }
    int negative = x->m.neg != y->m.neg, exact, rc;
    bigint ax = big_magnitude(&x->m), ay = big_magnitude(&y->m);
    int64_t k = p + 2 + big_bits(&ay) - big_bits(&ax);
    if (k < 0) k = 0;
    size_t na = (size_t)((big_bits(&ax) + k + 63) / 64), nb = ay.n;
    if (big_bits(&ax) + k < 64 * (SMALL_WORDS - 1) && na >= nb) {
        /* |x| 2**k over |y| in local words, the quotient's magnitude up by one when inexact and rounding away from zero */
        uint64_t an[SMALL_WORDS + 1], bn[SMALL_WORDS], q[SMALL_WORDS + 1];
        size_t nq;
        for (size_t i = 0; i < na; i++) an[i] = 0;
        place_words(an, na, &ax, k);
        if (nb == 1) {
            exact = nat_div_word(q, an, na, ay.w[0]) == 0;
            nq = na;
        } else {
            int sh = word_clz(ay.w[nb - 1]);
            if (sh) {
                nat_shl(bn, ay.w, nb, sh);
                an[na] = nat_shl(an, an, na, sh);
            } else {
                for (size_t i = 0; i < nb; i++) bn[i] = ay.w[i];
                an[na] = 0;
            }
            nat_divrem_norm(q, an, na + 1, bn, nb);
            uint64_t any = 0;
            for (size_t i = 0; i < nb; i++) any |= an[i];
            exact = any == 0;
            nq = na + 2 - nb;
        }
        q[nq] = 0;
        if (!exact && (negative ? !up : up)) {
            nat_add_word(q, 1, nq + 1);
            nq++;
        }
        int64_t e;
        if ((rc = exp_add(&e, x->e, -y->e)) || (rc = exp_add(&e, e, -k))) return rc;
        return round_words(r, q, nq, negative, e, p, up);
    }
    dyad t;
    dyad_init(&t);
    bigint scaled;
    big_init(&scaled);
    if (!(rc = shl(&scaled, &ax, k)) && !(rc = divmod_mag(&t.m, &exact, &scaled, &ay))) {
        if (!exact && (negative ? !up : up)) { /* the quotient's magnitude goes up by one */
            bigint one;
            big_init(&one);
            if (!(rc = big_set_word(&one, 1, 0))) rc = big_add(&t.m, &t.m, &one);
            big_free(&one);
        }
        t.m.neg = negative && t.m.n;
        int64_t e;
        if (!rc && !(rc = exp_add(&e, x->e, -y->e)) && !(rc = exp_add(&t.e, e, -k))) rc = dyad_rnd(r, &t, p, up);
    }
    big_free(&scaled);
    dyad_free(&t);
    return rc;
}

/* r = sqrt(x) rounded. */
int dyad_sqrt(dyad *r, const dyad *x, int64_t p, int up) {
    if (x->m.neg) return EXACTREF_DOMAIN;
    if (!x->m.n) {
        r->m.n = 0;
        r->m.neg = 0;
        r->e = 0;
        return 0;
    }
    dyad t;
    dyad_init(&t);
    int64_t e = x->e;
    int rc = copy(&t.m, &x->m), exact;
    if (!rc && (e & 1)) {
        rc = shl(&t.m, &t.m, 1);
        e -= 1;
    }
    int64_t k = rc ? 0 : p + 2 - big_bits(&t.m) / 2;
    if (k < 0) k = 0;
    if (!rc && !(rc = shl(&t.m, &t.m, 2 * k)) && !(rc = isqrt_mag(&t.m, &exact, &t.m))) {
        if (up && !exact) {
            bigint one;
            big_init(&one);
            if (!(rc = big_set_word(&one, 1, 0))) rc = big_add(&t.m, &t.m, &one);
            big_free(&one);
        }
        if (!rc && !(rc = exp_add(&t.e, e, -2 * k))) {
            t.e /= 2; /* e - 2 k is even */
            rc = dyad_rnd(r, &t, p, up);
        }
    }
    dyad_free(&t);
    return rc;
}

/* r = x**n, each product rounded, for x >= 0 and n >= 0. */
int dyad_power(dyad *r, const dyad *x, int64_t n, int64_t p, int up) {
    dyad result, base;
    dyad_init(&result);
    dyad_init(&base);
    int rc = big_set_word(&result.m, 1, 0);
    if (!rc) {
        rc = copy(&base.m, &x->m);
        base.e = x->e;
    }
    while (!rc && n) {
        if (n & 1) rc = dyad_mul(&result, &result, &base, p, up);
        n >>= 1;
        if (!rc && n) rc = dyad_mul(&base, &base, &base, p, up);
    }
    if (!rc) {
        rc = copy(&r->m, &result.m);
        r->e = result.e;
    }
    dyad_free(&result);
    dyad_free(&base);
    return rc;
}

/* r = x * 2**bits rounded to an integer toward -inf or +inf. */
int dyad_to_fixed(bigint *r, const dyad *x, int64_t bits, int up) {
    if (!x->m.n) return big_set_word(r, 0, 0);
    int64_t shift;
    int rc = exp_add(&shift, x->e, bits);
    if (rc) return rc;
    return shift >= 0 ? shl(r, &x->m, shift) : shr_dir(r, &x->m, -shift, up);
}

int big_set_si(bigint *b, int64_t v) { return big_set_word(b, v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v, v < 0); }

int dyad_copy(dyad *r, const dyad *x) {
    r->e = x->m.n ? x->e : 0; /* a zero view made from other words keeps its exponent: it becomes 0 2**0 */
    return copy(&r->m, &x->m);
}

/* r = m 2**e: a zero m gives 0 2**0 */
int dyad_from_big(dyad *r, const bigint *m, int64_t e) {
    r->e = m->n ? e : 0;
    return copy(&r->m, m);
}

/* r = -x */
int dyad_neg(dyad *r, const dyad *x) {
    int rc = dyad_copy(r, x);
    if (!rc && r->m.n) r->m.neg = !r->m.neg;
    return rc;
}

/* r = |x| */
int dyad_abs(dyad *r, const dyad *x) {
    int rc = dyad_copy(r, x);
    r->m.neg = 0;
    return rc;
}

int dyad_set_si(dyad *r, int64_t v, int64_t e) {
    r->e = v ? e : 0;
    return big_set_si(&r->m, v);
}

int dyad_sign(const dyad *x) { return x->m.n ? (x->m.neg ? -1 : 1) : 0; }

/* the sign of a 2**d - b for magnitudes of the same bit length (d >= 0): their words from the top down, a's shifted ones
   formed as they are read */
static int cmp_shifted(const bigint *a, int64_t d, const bigint *b) {
    size_t q = (size_t)(d / 64);
    int r = (int)(d % 64);
    for (size_t i = b->n; i-- > 0;) {
        uint64_t x = i >= q && i - q < a->n ? a->w[i - q] << r : 0;
        if (r && i >= q + 1 && i - q - 1 < a->n) x |= a->w[i - q - 1] >> (64 - r);
        if (x != b->w[i]) return x > b->w[i] ? 1 : -1;
    }
    return 0;
}

/* The sign of x - y. */
int dyad_cmp(const dyad *x, const dyad *y) {
    int sx = dyad_sign(x), sy = dyad_sign(y);
    if (sx != sy) return sx > sy ? 1 : -1;
    if (!sx) return 0;
    int64_t tx = x->e + big_bits(&x->m), ty = y->e + big_bits(&y->m);
    int c;
    if (tx != ty) {
        c = tx > ty ? 1 : -1;
    } else {
        /* the same top bit: the magnitudes aligned at the lower exponent */
        const dyad *a = x->e >= y->e ? x : y, *b = x->e >= y->e ? y : x;
        c = cmp_shifted(&a->m, a->e - b->e, &b->m);
        if (a != x) c = -c;
    }
    return sx > 0 ? c : -c;
}

/* Natural-number helpers for fixed-point work (functions.c): operands >= 0. */
int big_mul(bigint *r, const bigint *a, const bigint *b) { return mul_int(r, a, b); }

int big_shl(bigint *r, const bigint *a, int64_t s) { return shl(r, a, s); }

/* r = floor(a / 2**s), or the ceiling when up. */
int big_shr(bigint *r, const bigint *a, int64_t s, int up) { return shr_dir(r, a, s, up); }

/* r = floor(a / d), or the ceiling when up, for a >= 0 and d > 0. */
int big_div_word(bigint *r, const bigint *a, uint64_t d, int up) {
    if (big_reserve(r, a->n ? a->n : 1)) return EXACTREF_NOMEM;
    uint64_t rem = a->n ? nat_div_word(r->w, a->w, a->n, d) : 0;
    r->n = a->n;
    r->neg = 0;
    big_trim(r);
    if (up && rem) {
        if (big_reserve(r, r->n + 1)) return EXACTREF_NOMEM;
        r->w[r->n] = 0;
        r->n++;
        nat_add_word(r->w, 1, r->n);
        big_trim(r);
    }
    return 0;
}

/* q = floor(a / b), or the ceiling when up, for a >= 0 and b > 0. */
int big_div(bigint *q, const bigint *a, const bigint *b, int up) {
    int exact, rc = divmod_mag(q, &exact, a, b);
    if (!rc && up && !exact) {
        bigint one;
        big_init(&one);
        if (!(rc = big_set_word(&one, 1, 0))) rc = big_add(q, q, &one);
        big_free(&one);
    }
    return rc;
}

/* Bit-level helpers for the format rules (formats.c); they read magnitudes. */
int big_copy(bigint *r, const bigint *a) { return copy(r, a); }

/* q = floor(|a| / |b|) for b != 0, and whether the remainder is zero. */
int big_divmod(bigint *q, int *exact, const bigint *a, const bigint *b) {
    bigint ma = big_magnitude(a), mb = big_magnitude(b);
    return divmod_mag(q, exact, &ma, &mb);
}

/* The sign of |a| - |b|. */
int big_cmp_mag(const bigint *a, const bigint *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    return a->n ? nat_cmp(a->w, b->w, a->n) : 0;
}

/* Bit k of |a|. */
int big_bit(const bigint *a, int64_t k) {
    if (k < 0 || (uint64_t)k / 64 >= a->n) return 0;
    return (int)(a->w[k / 64] >> (k % 64) & 1);
}

/* Whether |a| mod 2**k is not zero. */
int big_low_nonzero(const bigint *a, int64_t k) {
    if (k <= 0) return 0;
    size_t words = (size_t)(k / 64);
    for (size_t i = 0; i < words && i < a->n; i++)
        if (a->w[i]) return 1;
    return words < a->n && k % 64 && a->w[words] << (64 - k % 64) != 0;
}

/* Whether |a| is 2**k - 1 (k >= 1). */
int big_is_ones(const bigint *a, int64_t k) {
    size_t words = (size_t)(k / 64);
    int bits = (int)(k % 64);
    if (a->n != words + (bits != 0)) return 0;
    for (size_t i = 0; i < words; i++)
        if (a->w[i] != ~(uint64_t)0) return 0;
    return !bits || a->w[words] == ((uint64_t)1 << bits) - 1;
}

/* r = |a| mod 2**k, k >= 0 (r may be a). */
int big_low(bigint *r, const bigint *a, int64_t k) {
    size_t words = (size_t)(k / 64);
    int bits = (int)(k % 64);
    size_t n = a->n < words + (bits != 0) ? a->n : words + (bits != 0);
    if (big_reserve(r, n)) return EXACTREF_NOMEM;
    if (r != a && n) memcpy(r->w, a->w, n * sizeof *a->w);
    if (bits && n == words + 1) r->w[words] &= ((uint64_t)1 << bits) - 1;
    r->n = n;
    r->neg = 0;
    big_trim(r);
    return 0;
}

/* r = |r| with bit k set. */
int big_set_bit(bigint *r, int64_t k) {
    size_t word = (size_t)(k / 64);
    if (word >= r->n) {
        if (big_reserve(r, word + 1)) return EXACTREF_NOMEM;
        memset(r->w + r->n, 0, (word + 1 - r->n) * sizeof *r->w);
        r->n = word + 1;
    }
    r->w[word] |= (uint64_t)1 << (k % 64);
    r->neg = 0;
    return 0;
}

/* The number of trailing zero bits of |a| (0 for a = 0). */
int64_t big_trailing(const bigint *a) {
    for (size_t i = 0; i < a->n; i++) {
        uint64_t w = a->w[i];
        if (w) {
            int64_t k = 0;
            while (!(w & 1)) {
                w >>= 1;
                k++;
            }
            return 64 * (int64_t)i + k;
        }
    }
    return 0;
}

/* The exponent of a nonzero dyad's top bit: 2**top <= |x| < 2**(top + 1) */
int64_t top_bit(const dyad *x) { return x->e + big_bits(&x->m) - 1; }

const dyad *dmax(const dyad *x, const dyad *y) { return dyad_cmp(x, y) >= 0 ? x : y; }

const dyad *dmin(const dyad *x, const dyad *y) { return dyad_cmp(x, y) <= 0 ? x : y; }

int sub_dir(dyad *r, const dyad *x, const dyad *y, int64_t p, int up) {
    dyad t;
    dyad_init(&t);
    int rc = dyad_neg(&t, y);
    if (!rc) rc = dyad_add(r, x, &t, p, up);
    dyad_free(&t);
    return rc;
}

/* the magnitude bound of [lo, hi] */
int ends_magnitude(dyad *r, const dyad *lo, const dyad *hi) {
    dyad a;
    dyad_init(&a);
    int rc = dyad_copy(&a, lo);
    a.m.neg = 0;
    if (!rc) rc = dyad_copy(r, dyad_cmp(&a, hi) < 0 ? hi : &a);
    dyad_free(&a);
    return rc;
}

/* num / den rounded to p bits */
int dyad_ratio(dyad *r, int64_t num, int64_t den, int64_t p, int up) {
    dyad a, b;
    dyad_init(&a);
    dyad_init(&b);
    int rc = dyad_set_si(&a, num, 0);
    if (!rc && !(rc = dyad_set_si(&b, den, 0))) rc = dyad_div(r, &a, &b, p, up);
    dyad_free(&a);
    dyad_free(&b);
    return rc;
}

/* ---------------------------------------------------------------- intervals [lo, hi] of dyadics, each end rounded
   outward (foundations.md, Directed dyadic arithmetic) */

void ivl_init(ivl *x) {
    dyad_init(&x->lo);
    dyad_init(&x->hi);
}

void ivl_free(ivl *x) {
    dyad_free(&x->lo);
    dyad_free(&x->hi);
}

int ivl_copy(ivl *r, const ivl *x) {
    int rc = dyad_copy(&r->lo, &x->lo);
    return rc ? rc : dyad_copy(&r->hi, &x->hi);
}

int ivl_point(ivl *r, const dyad *x) {
    int rc = dyad_copy(&r->lo, x);
    return rc ? rc : dyad_copy(&r->hi, x);
}

/* r's ends are written as each is found: each end reads only the operands' ends of its own side (r may be x or y) */
int ivl_add(ivl *r, const ivl *x, const ivl *y, int64_t p) {
    int rc = dyad_add(&r->lo, &x->lo, &y->lo, p, 0);
    return rc ? rc : dyad_add(&r->hi, &x->hi, &y->hi, p, 1);
}

int ivl_sub(ivl *r, const ivl *x, const ivl *y, int64_t p) {
    if (r == y) { /* the lower end reads y's upper end, which the upper end's result would replace */
        ivl t;
        ivl_init(&t);
        int rc = sub_dir(&t.lo, &x->lo, &y->hi, p, 0);
        if (!rc) rc = sub_dir(&t.hi, &x->hi, &y->lo, p, 1);
        if (!rc) rc = ivl_copy(r, &t);
        ivl_free(&t);
        return rc;
    }
    int rc = sub_dir(&r->lo, &x->lo, &y->hi, p, 0);
    return rc ? rc : sub_dir(&r->hi, &x->hi, &y->lo, p, 1);
}

/* r = x y for any signs */
int ivl_mul(ivl *r, const ivl *x, const ivl *y, int64_t p) {
    ivl t;
    ivl_init(&t);
    int rc;
    if (dyad_sign(&x->lo) >= 0 && dyad_sign(&y->lo) >= 0) {
        /* each end reads only the operands' ends of its own side */
        ivl_free(&t);
        if ((rc = dyad_mul(&r->lo, &x->lo, &y->lo, p, 0))) return rc;
        return dyad_mul(&r->hi, &x->hi, &y->hi, p, 1);
    } else {
        const dyad *a[2] = {&x->lo, &x->hi}, *b[2] = {&y->lo, &y->hi};
        dyad c;
        dyad_init(&c);
        rc = 0;
        for (int i = 0; !rc && i < 4; i++) {
            if ((rc = dyad_mul(&c, a[i / 2], b[i % 2], p, 0))) break;
            if (!i || dyad_cmp(&c, &t.lo) < 0) rc = dyad_copy(&t.lo, &c);
            if (!rc && !(rc = dyad_mul(&c, a[i / 2], b[i % 2], p, 1)) && (!i || dyad_cmp(&c, &t.hi) > 0)) rc = dyad_copy(&t.hi, &c);
        }
        dyad_free(&c);
    }
    if (!rc) rc = ivl_copy(r, &t);
    ivl_free(&t);
    return rc;
}

/* r = x / y for y > 0 */
int ivl_div(ivl *r, const ivl *x, const ivl *y, int64_t p) {
    const dyad *dl = dyad_sign(&x->lo) >= 0 ? &y->hi : &y->lo, *dh = dyad_sign(&x->hi) >= 0 ? &y->lo : &y->hi;
    if (r == y) { /* the upper end may read y's lower end */
        ivl t;
        ivl_init(&t);
        int rc = dyad_div(&t.lo, &x->lo, dl, p, 0);
        if (!rc) rc = dyad_div(&t.hi, &x->hi, dh, p, 1);
        if (!rc) rc = ivl_copy(r, &t);
        ivl_free(&t);
        return rc;
    }
    int rc = dyad_div(&r->lo, &x->lo, dl, p, 0); /* x's lower end is read before its replacement */
    return rc ? rc : dyad_div(&r->hi, &x->hi, dh, p, 1);
}

int ivl_max(ivl *r, const ivl *x, const ivl *y) {
    int rc = dyad_copy(&r->lo, dmax(&x->lo, &y->lo));
    return rc ? rc : dyad_copy(&r->hi, dmax(&x->hi, &y->hi));
}

int ivl_abs(ivl *r, const ivl *x) {
    ivl t;
    ivl_init(&t);
    int rc;
    if (dyad_sign(&x->lo) >= 0) {
        rc = ivl_copy(&t, x);
    } else if (dyad_sign(&x->hi) <= 0) {
        rc = dyad_neg(&t.lo, &x->hi);
        if (!rc) rc = dyad_neg(&t.hi, &x->lo);
    } else {
        rc = dyad_set_si(&t.lo, 0, 0);
        if (!rc) rc = dyad_neg(&t.hi, &x->lo);
        if (!rc && dyad_cmp(&x->hi, &t.hi) > 0) rc = dyad_copy(&t.hi, &x->hi);
    }
    if (!rc) rc = ivl_copy(r, &t);
    ivl_free(&t);
    return rc;
}

int widen_by(ivl *x, const dyad *r, int64_t w) {
    int rc = sub_dir(&x->lo, &x->lo, r, w, 0);
    return rc ? rc : dyad_add(&x->hi, &x->hi, r, w, 1);
}

/* ---------------------------------------------------------------- two's-complement integers of n 64-bit words */

/* x = -x over n words */
void mw_neg(uint64_t *x, int n) {
    uint64_t c = 1;
    for (int i = 0; i < n; i++) {
        x[i] = ~x[i] + c;
        c = c && !x[i];
    }
}

int mw_zero(const uint64_t *x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i]) return 0;
    return 1;
}

/* r (n words) = x (m <= n words), sign-extended */
void mw_extend(uint64_t *r, int n, const uint64_t *x, int m) {
    uint64_t fill = mw_negative(x, m) ? UINT64_MAX : 0;
    for (int i = 0; i < n; i++) r[i] = i < m ? x[i] : fill;
}

/* x += y over n words */
void mw_add(uint64_t *x, const uint64_t *y, int n) { nat_add(x, x, y, (size_t)n, 0); }

/* x += |y| over n words */
void mw_add_abs(uint64_t *x, const uint64_t *y, int n, uint64_t *tmp) {
    if (!mw_negative(y, n)) {
        nat_add(x, x, y, (size_t)n, 0);
        return;
    }
    memcpy(tmp, y, (size_t)n * sizeof *tmp);
    mw_neg(tmp, n);
    nat_add(x, x, tmp, (size_t)n, 0);
}

/* r (n words) = a (na words) b (nb words), signed, na + nb <= n; tmp holds na + nb + 1 words */
void mw_mul(uint64_t *r, int n, const uint64_t *a, int na, const uint64_t *b, int nb, uint64_t *tmp) {
    int sa = mw_negative(a, na), sb = mw_negative(b, nb);
    uint64_t *ma = tmp, *mb = tmp + na;
    memcpy(ma, a, (size_t)na * sizeof *ma);
    memcpy(mb, b, (size_t)nb * sizeof *mb);
    if (sa) mw_neg(ma, na);
    if (sb) mw_neg(mb, nb);
    memset(r, 0, (size_t)n * sizeof *r);
    nat_mul(r, ma, (size_t)na, mb, (size_t)nb);
    if (sa != sb) mw_neg(r, n);
}

/* [lo, hi] x into the ends (st: lo, then hi, n words each), x exact: lo x and hi x, swapped for x < 0; tmp holds
   n + na + nx + 1 words */
void mw_add_ends(uint64_t *st, int n, const uint64_t *lo, const uint64_t *hi, int na, const uint64_t *x, int nx, uint64_t *tmp) {
    if (mw_zero(x, nx)) return;
    int neg = mw_negative(x, (size_t)nx);
    uint64_t *t = tmp, *scratch = tmp + n;
    mw_mul(t, n, neg ? hi : lo, na, x, nx, scratch);
    mw_add(st, t, n);
    mw_mul(t, n, neg ? lo : hi, na, x, nx, scratch);
    mw_add(st + n, t, n);
}

/* x = floor(x / 2**s) for s >= 0 (an arithmetic shift), or ceil with up */
void mw_shr(uint64_t *x, int n, int64_t s, int up) {
    if (up) mw_neg(x, n);
    uint64_t fill = mw_negative(x, (size_t)n) ? UINT64_MAX : 0;
    int64_t q = s / 64;
    int b = (int)(s % 64);
    for (int i = 0; i < n; i++) {
        if (s >= 64 * (int64_t)n) {
            x[i] = fill;
            continue;
        }
        uint64_t lo = i + q < n ? x[i + q] : fill, hi = i + q + 1 < n ? x[i + q + 1] : fill;
        x[i] = b ? lo >> b | hi << (64 - b) : lo;
    }
    if (up) mw_neg(x, n);
}

/* x as an integer at 2**u over n words, rounded down or up; EXACTREF_OVERFLOW where it takes more than bits bits, its
   sign's included (bits <= 64 n), and then *need (when given) the bits it takes */
int mw_from_dyad_need(uint64_t *r, int n, const dyad *x, int64_t u, int up, int64_t bits, int64_t *need) {
    bigint b;
    big_init(&b);
    int rc = dyad_to_fixed(&b, x, -u, up);
    if (!rc && (int64_t)big_bits(&b) + 1 > bits) {
        rc = EXACTREF_OVERFLOW;
        if (need) *need = big_bits(&b) + 1;
    }
    if (!rc) {
        for (int i = 0; i < n; i++) r[i] = (size_t)i < b.n ? b.w[i] : 0;
        if (b.neg) mw_neg(r, n);
    }
    big_free(&b);
    return rc;
}

int mw_from_dyad(uint64_t *r, int n, const dyad *x, int64_t u, int up) { return mw_from_dyad_need(r, n, x, u, up, 64 * (int64_t)n, NULL); }

/* a bigint from n words, two's complement */
int mw_to_big(bigint *r, const uint64_t *x, int n, uint64_t *tmp) {
    int neg = mw_negative(x, n);
    memcpy(tmp, x, (size_t)n * sizeof *tmp);
    if (neg) mw_neg(tmp, n);
    int rc = big_reserve(r, (size_t)n);
    if (rc) return rc;
    memcpy(r->w, tmp, (size_t)n * sizeof *tmp);
    r->n = (size_t)n;
    r->neg = 0;
    big_trim(r);
    r->neg = neg && r->n;
    return 0;
}

/* a dyad from n words (two's complement) at 2**e: a zero gives 0 2**0 */
int mw_to_dyad(dyad *r, const uint64_t *x, int n, int64_t e, uint64_t *tmp) {
    int rc = mw_to_big(&r->m, x, n, tmp);
    r->e = r->m.n ? e : 0;
    return rc;
}

int mw_words(int64_t bits) { return (int)((bits + 63) / 64); }
