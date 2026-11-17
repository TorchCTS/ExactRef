/* rational.c: exact fractions on exactref's integers, reduced. Adapted from imath 1.35 (M. J. Fromberger, MIT; see
   LICENSE): the binary gcd (imath.c's mp_int_gcd) and the fractions' reduction and sums (imrat.c's s_rat_reduce and
   s_rat_combine). */
#include <stdlib.h>

#include "exactref.h"
#include "words.h"

/* r = -a */
static int negated(bigint *r, const bigint *a) {
    int rc = big_copy(r, a);
    if (!rc && r->n) r->neg = !a->neg;
    return rc;
}

/* r = gcd(|a|, |b|): positive, 0 when both are 0. The binary algorithm: the common powers of 2 out, then t = u or -v,
   repeatedly made odd and taken as the new u (t > 0) or v (t < 0), and t = u - v, until t is 0. */
int big_gcd(bigint *r, const bigint *a, const bigint *b) {
    if (!a->n || !b->n) {
        int rc = big_copy(r, a->n ? a : b);
        r->neg = 0;
        return rc;
    }
    bigint u, v, t, s;
    big_init(&u);
    big_init(&v);
    big_init(&t);
    big_init(&s);
    int rc = big_copy(&u, a);
    if (!rc) rc = big_copy(&v, b);
    u.neg = v.neg = 0;
    int64_t ku = big_trailing(&u), kv = big_trailing(&v), k = ku < kv ? ku : kv;
    if (!rc) rc = big_shr(&u, &u, k, 0);
    if (!rc) rc = big_shr(&v, &v, k, 0);
    if (!rc) rc = big_bit(&u, 0) ? negated(&t, &v) : big_copy(&t, &u);
    while (!rc) {
        rc = big_shr(&t, &t, big_trailing(&t), 0); /* exact: t's low zeros */
        if (!rc) rc = t.neg ? negated(&v, &t) : big_copy(&u, &t);
        if (!rc && !(rc = negated(&s, &v))) rc = big_add(&t, &u, &s);
        if (!rc && !t.n) break;
    }
    if (!rc) rc = big_shl(r, &u, k);
    big_free(&u);
    big_free(&v);
    big_free(&t);
    big_free(&s);
    return rc;
}

void rat_init(rat *x) {
    big_init(&x->num);
    big_init(&x->den);
    big_set_word(&x->den, 1, 0); /* one word: inline storage */
}

void rat_free(rat *x) {
    big_free(&x->num);
    big_free(&x->den);
}

/* x in lowest terms, its denominator positive (0 is 0 / 1) */
static int rat_reduce(rat *x) {
    if (!x->num.n) return big_set_word(&x->den, 1, 0);
    int neg = x->num.neg != x->den.neg, exact, rc;
    bigint g, q;
    big_init(&g);
    big_init(&q);
    x->num.neg = x->den.neg = 0;
    rc = big_gcd(&g, &x->num, &x->den);
    if (!rc && (g.n != 1 || g.w[0] != 1)) {
        if (!(rc = big_divmod(&q, &exact, &x->num, &g)) && !(rc = big_copy(&x->num, &q)) && !(rc = big_divmod(&q, &exact, &x->den, &g)))
            rc = big_copy(&x->den, &q);
    }
    x->num.neg = x->num.n ? neg : 0;
    big_free(&g);
    big_free(&q);
    return rc;
}

/* x = num / den (den != 0), reduced */
int rat_set(rat *x, const bigint *num, const bigint *den) {
    int rc = big_copy(&x->num, num);
    if (!rc) rc = big_copy(&x->den, den);
    return rc ? rc : rat_reduce(x);
}

int rat_copy(rat *r, const rat *a) {
    int rc = big_copy(&r->num, &a->num);
    return rc ? rc : big_copy(&r->den, &a->den);
}

/* r = a + b, reduced; over the common denominator when a and b share one */
int rat_add(rat *r, const rat *a, const rat *b) {
    bigint n, d, t;
    big_init(&n);
    big_init(&d);
    big_init(&t);
    int rc;
    if (a->den.n == b->den.n && !big_cmp_mag(&a->den, &b->den)) {
        if (!(rc = big_add(&n, &a->num, &b->num))) rc = big_copy(&d, &a->den);
    } else if (!(rc = big_mul(&n, &a->num, &b->den)) && !(rc = big_mul(&t, &b->num, &a->den)) && !(rc = big_add(&n, &n, &t))) {
        rc = big_mul(&d, &a->den, &b->den);
    }
    if (!rc && !(rc = big_copy(&r->num, &n))) rc = big_copy(&r->den, &d);
    big_free(&n);
    big_free(&d);
    big_free(&t);
    return rc ? rc : rat_reduce(r);
}

/* r = a b, reduced */
int rat_mul(rat *r, const rat *a, const rat *b) {
    bigint n, d;
    big_init(&n);
    big_init(&d);
    int rc = big_mul(&n, &a->num, &b->num);
    if (!rc) rc = big_mul(&d, &a->den, &b->den);
    if (!rc && !(rc = big_copy(&r->num, &n))) rc = big_copy(&r->den, &d);
    big_free(&n);
    big_free(&d);
    return rc ? rc : rat_reduce(r);
}

/* r = -a */
int rat_neg(rat *r, const rat *a) {
    int rc = rat_copy(r, a);
    if (!rc && r->num.n) r->num.neg = !r->num.neg;
    return rc;
}

/* r = 1 / a (a != 0) */
int rat_inv(rat *r, const rat *a) {
    bigint n, d;
    big_init(&n);
    big_init(&d);
    int rc = big_copy(&n, &a->den), neg = a->num.neg;
    if (!rc) rc = big_copy(&d, &a->num);
    if (!rc) {
        d.neg = 0;
        n.neg = neg;
        if (!(rc = big_copy(&r->num, &n))) rc = big_copy(&r->den, &d);
    }
    big_free(&n);
    big_free(&d);
    return rc;
}

/* r = a / b (b != 0) */
int rat_div(rat *r, const rat *a, const rat *b) {
    rat t;
    rat_init(&t);
    int rc = rat_inv(&t, b);
    if (!rc) rc = rat_mul(r, a, &t);
    rat_free(&t);
    return rc;
}

/* x = the dyadic m 2**e, reduced */
int rat_set_dyad(rat *x, const bigint *m, int64_t e) {
    int rc = big_copy(&x->num, m);
    if (!rc) rc = big_set_word(&x->den, 1, 0);
    if (!rc && e > 0) rc = big_shl(&x->num, &x->num, e);
    if (!rc && e < 0) rc = big_shl(&x->den, &x->den, -e);
    return rc ? rc : rat_reduce(x);
}

int rat_set_si(rat *x, int64_t v) {
    int rc = big_set_si(&x->num, v);
    return rc ? rc : big_set_word(&x->den, 1, 0);
}

int rat_is_zero(const rat *a) { return a->num.n == 0; }

/* ---------------------------------------------------------------- polynomials over Q (ours) */

void rpoly_init(rpoly *p) {
    p->c = NULL;
    p->len = p->cap = 0;
}

void rpoly_free(rpoly *p) {
    for (size_t k = 0; k < p->cap; k++) rat_free(&p->c[k]);
    free(p->c);
    rpoly_init(p);
}

/* room for cap coefficients; the old ones copied, not moved (a bigint is never copied by value) */
static int rpoly_reserve(rpoly *p, size_t cap) {
    if (cap <= p->cap) return 0;
    if (cap < 2 * p->cap) cap = 2 * p->cap;
    rat *c = malloc(cap * sizeof *c);
    if (!c) return EXACTREF_NOMEM;
    int rc = 0;
    for (size_t k = 0; k < cap; k++) rat_init(&c[k]);
    for (size_t k = 0; !rc && k < p->len; k++) rc = rat_copy(&c[k], &p->c[k]);
    if (rc) {
        for (size_t k = 0; k < cap; k++) rat_free(&c[k]);
        free(c);
        return rc;
    }
    for (size_t k = 0; k < p->cap; k++) rat_free(&p->c[k]);
    free(p->c);
    p->c = c;
    p->cap = cap;
    return 0;
}

/* p's length without its zero top coefficients */
static void rpoly_trim(rpoly *p) {
    while (p->len && rat_is_zero(&p->c[p->len - 1])) p->len--;
}

/* p = 0 with room for len coefficients, each 0 */
static int rpoly_zero(rpoly *p, size_t len) {
    int rc = rpoly_reserve(p, len);
    for (size_t k = 0; !rc && k < len; k++) rc = rat_set_si(&p->c[k], 0);
    p->len = rc ? 0 : len;
    return rc;
}

int rpoly_copy(rpoly *r, const rpoly *a) {
    if (r == a) return 0;
    int rc = rpoly_reserve(r, a->len);
    for (size_t k = 0; !rc && k < a->len; k++) rc = rat_copy(&r->c[k], &a->c[k]);
    r->len = rc ? 0 : a->len;
    return rc;
}

/* p = c x**k */
int rpoly_set_term(rpoly *p, const rat *c, size_t k) {
    int rc = rpoly_zero(p, k + 1);
    if (!rc) rc = rat_copy(&p->c[k], c);
    rpoly_trim(p);
    return rc;
}

/* r = a + b */
int rpoly_add(rpoly *r, const rpoly *a, const rpoly *b) {
    rpoly t;
    rpoly_init(&t);
    size_t len = a->len > b->len ? a->len : b->len;
    int rc = rpoly_zero(&t, len);
    for (size_t k = 0; !rc && k < len; k++) {
        if (k < a->len) rc = rat_copy(&t.c[k], &a->c[k]);
        if (!rc && k < b->len) rc = rat_add(&t.c[k], &t.c[k], &b->c[k]);
    }
    rpoly_trim(&t);
    if (!rc) rc = rpoly_copy(r, &t);
    rpoly_free(&t);
    return rc;
}

/* r = a b */
int rpoly_mul(rpoly *r, const rpoly *a, const rpoly *b) {
    rpoly t;
    rat u;
    rpoly_init(&t);
    rat_init(&u);
    int rc = 0;
    if (a->len && b->len) {
        rc = rpoly_zero(&t, a->len + b->len - 1);
        for (size_t i = 0; !rc && i < a->len; i++)
            for (size_t j = 0; !rc && j < b->len; j++)
                if (!rat_is_zero(&a->c[i]) && !rat_is_zero(&b->c[j]) && !(rc = rat_mul(&u, &a->c[i], &b->c[j]))) rc = rat_add(&t.c[i + j], &t.c[i + j], &u);
        rpoly_trim(&t);
    }
    if (!rc) rc = rpoly_copy(r, &t);
    rpoly_free(&t);
    rat_free(&u);
    return rc;
}

/* ---------------------------------------------------------------- polynomials over Z (ours) */

void ipoly_init(ipoly *p) {
    p->c = NULL;
    p->len = p->cap = 0;
}

void ipoly_free(ipoly *p) {
    for (size_t k = 0; k < p->cap; k++) big_free(&p->c[k]);
    free(p->c);
    ipoly_init(p);
}

/* room for cap coefficients; the old ones copied, not moved (a bigint is never copied by value) */
static int ipoly_reserve(ipoly *p, size_t cap) {
    if (cap <= p->cap) return 0;
    if (cap < 2 * p->cap) cap = 2 * p->cap;
    bigint *c = malloc(cap * sizeof *c);
    if (!c) return EXACTREF_NOMEM;
    int rc = 0;
    for (size_t k = 0; k < cap; k++) big_init(&c[k]);
    for (size_t k = 0; !rc && k < p->len; k++) rc = big_copy(&c[k], &p->c[k]);
    if (rc) {
        for (size_t k = 0; k < cap; k++) big_free(&c[k]);
        free(c);
        return rc;
    }
    for (size_t k = 0; k < p->cap; k++) big_free(&p->c[k]);
    free(p->c);
    p->c = c;
    p->cap = cap;
    return 0;
}

void ipoly_trim(ipoly *p) {
    while (p->len && !p->c[p->len - 1].n) p->len--;
}

/* p = 0 with room for len coefficients, each 0 (p->len = len, untrimmed) */
int ipoly_zero(ipoly *p, size_t len) {
    int rc = ipoly_reserve(p, len);
    for (size_t k = 0; !rc && k < len; k++) rc = big_set_word(&p->c[k], 0, 0);
    p->len = rc ? 0 : len;
    return rc;
}

int ipoly_copy(ipoly *r, const ipoly *a) {
    if (r == a) return 0;
    int rc = ipoly_reserve(r, a->len);
    for (size_t k = 0; !rc && k < a->len; k++) rc = big_copy(&r->c[k], &a->c[k]);
    r->len = rc ? 0 : a->len;
    return rc;
}

/* p = c x**k */
int ipoly_set_term(ipoly *p, const bigint *c, size_t k) {
    int rc = ipoly_zero(p, k + 1);
    if (!rc) rc = big_copy(&p->c[k], c);
    ipoly_trim(p);
    return rc;
}

/* r = a + s b, s = +1 or -1 */
static int ipoly_addsub(ipoly *r, const ipoly *a, const ipoly *b, int s) {
    ipoly t;
    bigint u;
    ipoly_init(&t);
    big_init(&u);
    size_t len = a->len > b->len ? a->len : b->len;
    int rc = ipoly_zero(&t, len);
    for (size_t k = 0; !rc && k < len; k++) {
        if (k < a->len) rc = big_copy(&t.c[k], &a->c[k]);
        if (!rc && k < b->len && b->c[k].n) {
            if (s > 0) {
                rc = big_add(&t.c[k], &t.c[k], &b->c[k]);
            } else if (!(rc = big_copy(&u, &b->c[k]))) {
                u.neg = !u.neg;
                rc = big_add(&t.c[k], &t.c[k], &u);
            }
        }
    }
    ipoly_trim(&t);
    if (!rc) rc = ipoly_copy(r, &t);
    ipoly_free(&t);
    big_free(&u);
    return rc;
}

int ipoly_add(ipoly *r, const ipoly *a, const ipoly *b) { return ipoly_addsub(r, a, b, 1); }
int ipoly_sub(ipoly *r, const ipoly *a, const ipoly *b) { return ipoly_addsub(r, a, b, -1); }

/* r = a b */
int ipoly_mul(ipoly *r, const ipoly *a, const ipoly *b) {
    ipoly t;
    bigint u;
    ipoly_init(&t);
    big_init(&u);
    int rc = 0;
    if (a->len && b->len) {
        rc = ipoly_zero(&t, a->len + b->len - 1);
        for (size_t i = 0; !rc && i < a->len; i++)
            for (size_t j = 0; !rc && j < b->len; j++)
                if (a->c[i].n && b->c[j].n && !(rc = big_mul(&u, &a->c[i], &b->c[j]))) rc = big_add(&t.c[i + j], &t.c[i + j], &u);
        ipoly_trim(&t);
    }
    if (!rc) rc = ipoly_copy(r, &t);
    ipoly_free(&t);
    big_free(&u);
    return rc;
}

/* r = c a */
int ipoly_scale(ipoly *r, const ipoly *a, const bigint *c) {
    int rc = ipoly_copy(r, a);
    for (size_t k = 0; !rc && k < r->len; k++) rc = big_mul(&r->c[k], &r->c[k], c);
    ipoly_trim(r);
    return rc;
}

/* a = q g + r for a monic g of degree >= 1; q or r may be NULL */
int ipoly_divrem_monic(ipoly *q, ipoly *r, const ipoly *a, const ipoly *g) {
    ipoly rem, quo;
    bigint u;
    ipoly_init(&rem);
    ipoly_init(&quo);
    big_init(&u);
    size_t d = g->len - 1;
    int rc = ipoly_copy(&rem, a);
    if (!rc) rc = ipoly_zero(&quo, rem.len > d ? rem.len - d : 0);
    for (size_t top = rem.len; !rc && top-- > d;) {
        if (!rem.c[top].n) continue;
        size_t shift = top - d;
        if ((rc = big_copy(&quo.c[shift], &rem.c[top]))) break;
        for (size_t k = 0; !rc && k < d; k++)
            if (g->c[k].n && !(rc = big_mul(&u, &quo.c[shift], &g->c[k]))) {
                u.neg = u.n ? !u.neg : 0;
                rc = big_add(&rem.c[k + shift], &rem.c[k + shift], &u);
            }
        if (!rc) rc = big_set_word(&rem.c[top], 0, 0);
    }
    ipoly_trim(&rem);
    ipoly_trim(&quo);
    if (!rc && q) rc = ipoly_copy(q, &quo);
    if (!rc && r) rc = ipoly_copy(r, &rem);
    ipoly_free(&rem);
    ipoly_free(&quo);
    big_free(&u);
    return rc;
}

/* r = a' */
static int ipoly_deriv(ipoly *r, const ipoly *a) {
    ipoly t;
    bigint k;
    ipoly_init(&t);
    big_init(&k);
    int rc = ipoly_zero(&t, a->len ? a->len - 1 : 0);
    for (size_t i = 1; !rc && i < a->len; i++)
        if (!(rc = big_set_word(&k, i, 0))) rc = big_mul(&t.c[i - 1], &a->c[i], &k);
    ipoly_trim(&t);
    if (!rc) rc = ipoly_copy(r, &t);
    ipoly_free(&t);
    big_free(&k);
    return rc;
}

/* p divided by the gcd of its coefficients, its leading coefficient positive */
static int ipoly_primitive(ipoly *p) {
    bigint g, q;
    big_init(&g);
    big_init(&q);
    int rc = 0, exact;
    for (size_t k = 0; !rc && k < p->len; k++) rc = big_gcd(&g, &g, &p->c[k]);
    if (!rc && p->len && (g.n != 1 || g.w[0] != 1))
        for (size_t k = 0; !rc && k < p->len; k++)
            if (!(rc = big_divmod(&q, &exact, &p->c[k], &g))) { /* of the magnitudes: the sign kept */
                q.neg = q.n ? p->c[k].neg : 0;
                rc = big_copy(&p->c[k], &q);
            }
    if (!rc && p->len && p->c[p->len - 1].neg)
        for (size_t k = 0; k < p->len; k++) p->c[k].neg = p->c[k].n ? !p->c[k].neg : 0;
    big_free(&g);
    big_free(&q);
    return rc;
}

/* lc(b)**(deg a - deg b + 1) a mod b, for b of degree <= deg a (pseudo-division) */
static int ipoly_prem(ipoly *r, const ipoly *a, const ipoly *b) {
    ipoly rem;
    bigint u, lead;
    ipoly_init(&rem);
    big_init(&u);
    big_init(&lead);
    size_t d = b->len - 1;
    int rc = ipoly_copy(&rem, a);
    if (!rc) rc = big_copy(&lead, &b->c[d]);
    while (!rc && rem.len > d) {
        /* rem = lead rem - rem_top x**shift b */
        size_t top = rem.len - 1, shift = top - d;
        bigint c;
        big_init(&c);
        rc = big_copy(&c, &rem.c[top]);
        for (size_t k = 0; !rc && k < rem.len; k++) rc = big_mul(&rem.c[k], &rem.c[k], &lead);
        for (size_t k = 0; !rc && k <= d; k++)
            if (b->c[k].n && !(rc = big_mul(&u, &c, &b->c[k]))) {
                u.neg = u.n ? !u.neg : 0;
                rc = big_add(&rem.c[k + shift], &rem.c[k + shift], &u);
            }
        big_free(&c);
        ipoly_trim(&rem);
    }
    if (!rc) rc = ipoly_copy(r, &rem);
    ipoly_free(&rem);
    big_free(&u);
    big_free(&lead);
    return rc;
}

/* g = the monic gcd of a and b, monic integer polynomials' factors (so g has integer coefficients): Euclid's
   algorithm on primitive pseudo-remainders */
static int ipoly_gcd(ipoly *g, const ipoly *a, const ipoly *b) {
    ipoly r0, r1, t;
    ipoly_init(&r0);
    ipoly_init(&r1);
    ipoly_init(&t);
    int rc = ipoly_copy(&r0, a);
    if (!rc) rc = ipoly_copy(&r1, b);
    if (!rc && r0.len < r1.len) {
        ipoly x = r0;
        r0 = r1;
        r1 = x;
    }
    while (!rc && r1.len) {
        if ((rc = ipoly_prem(&t, &r0, &r1)) || (rc = ipoly_primitive(&t))) break;
        ipoly x = r0;
        r0 = r1;
        r1 = t;
        t = x;
    }
    if (!rc) rc = ipoly_primitive(&r0);
    if (!rc && r0.len && (r0.c[r0.len - 1].n != 1 || r0.c[r0.len - 1].w[0] != 1)) rc = EXACTREF_INTERNAL; /* not monic */
    if (!rc) rc = ipoly_copy(g, &r0);
    ipoly_free(&r0);
    ipoly_free(&r1);
    ipoly_free(&t);
    return rc;
}

/* ---------------------------------------------------------------- word primes */

uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t hi, lo = word_mul(&hi, a, b), r;
    word_div(&r, hi, lo, p);
    return r;
}

uint64_t mod_pow(uint64_t a, uint64_t e, uint64_t p) {
    uint64_t r = 1 % p;
    for (; e; e >>= 1, a = mod_mul(a, a, p))
        if (e & 1) r = mod_mul(r, a, p);
    return r;
}

/* a mod p, in [0, p) */
uint64_t big_mod(const bigint *a, uint64_t p) {
    uint64_t r = 0;
    for (size_t k = a->n; k-- > 0;) word_div(&r, r, a->w[k], p); /* (r 2**64 + w) mod p, r < p */
    return a->neg && r ? p - r : r;
}

/* whether n < 2**64 is prime: Miller-Rabin with the first 12 primes as bases, exact below 3.3 * 10**24 */
int word_prime(uint64_t n) {
    static const uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2) return 0;
    for (int i = 0; i < 12; i++)
        if (n % bases[i] == 0) return n == bases[i];
    uint64_t d = n - 1;
    int s = 0;
    while (!(d & 1)) {
        d >>= 1;
        s++;
    }
    for (int i = 0; i < 12; i++) {
        uint64_t x = mod_pow(bases[i], d, n);
        if (x == 1 || x == n - 1) continue;
        int composite = 1;
        for (int r = 1; r < s && composite; r++) {
            x = mod_mul(x, x, n);
            if (x == n - 1) composite = 0;
        }
        if (composite) return 0;
    }
    return 1;
}

/* the degree of gcd(a, b) over F_p (a, b low first, n and m coefficients, consumed), -1 for 0 */
static int64_t mod_gcd_degree(uint64_t *a, size_t n, uint64_t *b, size_t m, uint64_t p) {
    while (n && !a[n - 1]) n--;
    while (m && !b[m - 1]) m--;
    while (m) {
        uint64_t inv = mod_pow(b[m - 1], p - 2, p);
        while (n >= m) {
            uint64_t f = mod_mul(a[n - 1], inv, p);
            size_t shift = n - m;
            for (size_t k = 0; k < m; k++) a[k + shift] = (a[k + shift] + p - mod_mul(f, b[k], p)) % p;
            while (n && !a[n - 1]) n--;
        }
        uint64_t *t = a;
        a = b;
        b = t;
        size_t u = n;
        n = m;
        m = u;
    }
    return (int64_t)n - 1;
}

#define SQUAREFREE_P 0x1fffffffffffffffULL /* 2**61 - 1, prime */

/* *yes when the monic integer f of degree >= 1 is square-free modulo p = 2**61 - 1 (deg f < p), which makes it
   square-free over Q: a monic common factor of f and f' over Q has integer coefficients and divides both modulo
   p with its degree kept. *yes 0 decides nothing. */
static int squarefree_mod_p(int *yes, const ipoly *f) {
    size_t n = f->len;
    uint64_t *a = malloc((n + 1) * sizeof *a), *b = malloc((n + 1) * sizeof *b);
    *yes = 0;
    if (!a || !b) {
        free(a);
        free(b);
        return EXACTREF_NOMEM;
    }
    for (size_t k = 0; k < n; k++) a[k] = big_mod(&f->c[k], SQUAREFREE_P);
    for (size_t k = 1; k < n; k++) b[k - 1] = mod_mul(a[k], (uint64_t)k, SQUAREFREE_P);
    if (n > 1) *yes = mod_gcd_degree(a, n, b, n - 1, SQUAREFREE_P) == 0;
    free(a);
    free(b);
    return 0;
}

/* The square-free factors of a monic integer f of degree >= 1 (Yun's algorithm over Z, each factor monic with
   integer coefficients): factors[i] (1 for none) has multiplicity i + 1, i < *count; factors has room for deg f. */
int ipoly_squarefree(ipoly *factors, size_t *count, const ipoly *f) {
    int yes, rc = squarefree_mod_p(&yes, f);
    *count = 0;
    if (rc) return rc;
    if (yes) { /* one factor, f itself */
        *count = 1;
        return ipoly_copy(&factors[0], f);
    }
    ipoly a, b, c, d, t;
    ipoly *all[] = {&a, &b, &c, &d, &t};
    for (int k = 0; k < 5; k++) ipoly_init(all[k]);
    rc = ipoly_deriv(&t, f);
    if (!rc) rc = ipoly_gcd(&a, f, &t);                 /* a0 = gcd(f, f'), monic */
    if (!rc) rc = ipoly_divrem_monic(&b, NULL, f, &a);  /* b1 = f / a0 */
    if (!rc) rc = ipoly_divrem_monic(&c, NULL, &t, &a); /* c1 = f' / a0 */
    while (!rc && b.len > 1) {
        if ((rc = ipoly_deriv(&t, &b)) || (rc = ipoly_sub(&d, &c, &t))) break; /* d = c - b' */
        if (!d.len) { /* the rest of b is one factor */
            if ((rc = ipoly_copy(&factors[*count], &b))) break;
            (*count)++;
            break;
        }
        if ((rc = ipoly_gcd(&a, &b, &d))) break;                              /* a_i = gcd(b, d) */
        if ((rc = ipoly_copy(&factors[*count], &a))) break;
        (*count)++;
        if ((rc = ipoly_divrem_monic(&b, NULL, &b, &a)) || (rc = ipoly_divrem_monic(&c, NULL, &d, &a))) break;
    }
    for (int k = 0; k < 5; k++) ipoly_free(all[k]);
    return rc;
}
