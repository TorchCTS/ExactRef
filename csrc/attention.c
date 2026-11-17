/* The attention forward's reference (proofs/scaled_dot_product_attention.md, The reference output
   and Lemma D): the weights from digit tables, and per row the exact scores, the row and value
   sums, the exact cases and the first enclosures of the outputs and of logsumexp. Outputs it
   leaves undecided are refined by the caller. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

#define ATT_G (ATT_F + ATT_B + 8) /* the tables' working bits (Lemma D) */
#define ATT_GW 4                  /* words holding a value up to 2**ATT_G */

/* r (na + nb words) = a b: nat_mul's product, kept in this unit so that the compiler can inline it at
   the small fixed sizes of its per-element loops */
static void words_mul(uint64_t *r, const uint64_t *a, size_t na, const uint64_t *b, size_t nb) {
    memset(r, 0, (na + nb) * sizeof *r);
    for (size_t i = 0; i < na; i++) {
        uint64_t carry = 0;
        for (size_t j = 0; j < nb; j++) {
            uint64_t hi, lo = word_mul(&hi, a[i], b[j]);
            uint64_t s = r[i + j] + lo;
            hi += s < lo;
            s += carry;
            hi += s < carry;
            r[i + j] = s;
            carry = hi;
        }
        r[i + nb] = carry;
    }
}

/* r (rn words) = floor(a / 2**s), a of n words */
static void words_shr(uint64_t *r, size_t rn, const uint64_t *a, size_t n, int64_t s) {
    size_t q = (size_t)(s / 64);
    int b = (int)(s % 64);
    for (size_t i = 0; i < rn; i++) {
        uint64_t lo = i + q < n ? a[i + q] : 0, hi = i + q + 1 < n ? a[i + q + 1] : 0;
        r[i] = b ? lo >> b | hi << (64 - b) : lo;
    }
}

/* r (n words) += a (m <= n words) */
static void words_add(uint64_t *r, size_t n, const uint64_t *a, size_t m) {
    uint64_t carry = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t x = i < m ? a[i] : 0, s = r[i] + x;
        uint64_t c = s < x;
        s += carry;
        c += s < carry;
        r[i] = s;
        carry = c;
    }
}

/* r (n words) -= a (m <= n words), modulo 2**(64 n) */
static void words_sub(uint64_t *r, size_t n, const uint64_t *a, size_t m) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t x = i < m ? a[i] : 0, d = r[i] - x;
        uint64_t b = r[i] < x;
        b += d < borrow;
        d -= borrow;
        r[i] = d;
        borrow = b;
    }
}

/* a bigint from n words of two's complement (signed) or of magnitude (!signed) */
static int words_big(bigint *r, const uint64_t *a, size_t n, int is_signed) {
    uint64_t t[ATT_ACC];
    int neg = is_signed && a[n - 1] >> 63;
    memcpy(t, a, n * sizeof *t);
    if (neg) {
        uint64_t one = 1;
        for (size_t i = 0; i < n; i++) t[i] = ~t[i];
        words_add(t, n, &one, 1);
    }
    size_t m = n;
    while (m && !t[m - 1]) m--;
    int rc = big_reserve(r, m ? m : 1);
    if (rc) return rc;
    memcpy(r->w, t, m * sizeof *t);
    r->n = m;
    r->neg = m ? neg : 0;
    return 0;
}

void att_tables_init(att_tables *t) {
    t->digits = 0;
    t->unit = 0;
    t->entries = NULL;
}

void att_tables_free(att_tables *t) {
    free(t->entries);
    att_tables_init(t);
}

/* c = |sigma| 2**u rounded up to pc bits: sigma = num / den, or 1/sqrt(E) when den is 0 */
static int scale_up(dyad *c, const bigint *num, const bigint *den, int64_t E, int64_t u, int64_t pc) {
    dyad x, y;
    dyad_init(&x);
    dyad_init(&y);
    int rc;
    if (den->n) {
        rc = big_copy(&x.m, num);
        x.m.neg = 0;
        x.e = x.m.n ? u : 0;
        if (!rc && !(rc = big_copy(&y.m, den))) {
            y.e = 0;
            rc = dyad_div(c, &x, &y, pc, 1);
        }
    } else {
        /* 2**u / sqrt(E) <= 2**u / r_lo with r_lo <= sqrt(E) */
        dyad e;
        dyad_init(&e);
        rc = dyad_set_si(&e, E, 0);
        if (!rc) rc = dyad_sqrt(&y, &e, pc + 2, 0);
        if (!rc) rc = dyad_set_si(&x, 1, u);
        if (!rc) rc = dyad_div(c, &x, &y, pc, 1);
        dyad_free(&e);
    }
    dyad_free(&x);
    dyad_free(&y);
    return rc;
}

/* Lemma D's tables for c = |sigma| 2**u, k digits of ATT_B bits: T_i[d] <= 2**F exp(-c d 2**(B i))
   < T_i[d] + 2, T_i[0] = 2**F, at entries[((i << B) + d) * 3]. */
int att_tables_build(att_tables *t, const bigint *num, const bigint *den, int64_t E, int64_t u, int digits) {
    att_tables_init(t);
    size_t per = (size_t)1 << ATT_B;
    t->entries = calloc((size_t)digits * per * 3, sizeof *t->entries);
    if (!t->entries && digits) return EXACTREF_NOMEM;
    t->digits = digits;
    t->unit = u;
    /* c_hi - c <= 2**-(G + B (k - 1) + 2): c at G + B k + 8 bits beyond its top bit (at most
       bits(num) - bits(den) + 2 + u, or u for 1/sqrt(E) <= 1) */
    int64_t top = u + (den->n ? big_bits(num) - big_bits(den) + 2 : 1);
    int64_t pc = ATT_G + (int64_t)ATT_B * digits + 8 + (top > 0 ? top : 0);
    dyad c, x, lo, hi;
    dyad_init(&c);
    dyad_init(&x);
    dyad_init(&lo);
    dyad_init(&hi);
    bigint n, A;
    big_init(&n);
    big_init(&A);
    int rc = scale_up(&c, num, den, E, u, pc);
    for (int i = 0; !rc && i < digits; i++) {
        uint64_t *T = t->entries + (size_t)i * per * 3;
        /* A = floor(2**G a_lo), a_lo <= exp(-c_hi 2**(B i)) by Lemma X at G + 2 bits */
        if ((rc = dyad_copy(&x, &c)) || (rc = dyad_shift(&x, (int64_t)ATT_B * i))) break;
        x.m.neg = 1;
        if ((rc = exp_bounds(&lo, &hi, &n, &x, ATT_G + 2))) break;
        uint64_t Aw[ATT_GW] = {0};
        if (n.n <= 1 && !(n.n && n.w[0] > ((uint64_t)1 << 40))) {
            int64_t shift = n.n ? (n.neg ? -(int64_t)n.w[0] : (int64_t)n.w[0]) : 0;
            if ((rc = dyad_shift(&lo, shift))) break;
            if (lo.e + big_bits(&lo.m) > -(ATT_G + 2)) {
                if ((rc = dyad_to_fixed(&A, &lo, ATT_G, 0))) break;
                for (size_t w = 0; w < A.n && w < ATT_GW; w++) Aw[w] = A.w[w];
            }
        }
        /* T_i[0] = 2**F; L_d = floor(L_(d - 1) A / 2**G), T_i[d] = floor(L_d / 2**(G - F)) */
        T[2] = (uint64_t)1 << (ATT_F - 128);
        uint64_t L[ATT_GW] = {0}, prod[2 * ATT_GW];
        L[ATT_G / 64] = (uint64_t)1 << (ATT_G % 64);
        for (size_t d = 1; d < per; d++) {
            words_mul(prod, L, ATT_GW, Aw, ATT_GW);
            words_shr(L, ATT_GW, prod, 2 * ATT_GW, ATT_G);
            words_shr(T + d * 3, 3, L, ATT_GW, ATT_G - ATT_F);
        }
    }
    dyad_free(&c);
    dyad_free(&x);
    dyad_free(&lo);
    dyad_free(&hi);
    big_free(&n);
    big_free(&A);
    if (rc) att_tables_free(t);
    return rc;
}

/* signed 128-bit order */
int s128_less(w128 a, w128 b) {
    w128 bias = w128_make(0, (uint64_t)1 << 63);
    return w128_less(w128_add(a, bias), w128_add(b, bias));
}

w128 s128_negate(w128 a) { return w128_sub(w128_zero(), a); }

/* a signed w128 times 2**-s, exactly (a is a multiple of 2**s). A score's shift s = u - (Q's exponent + K's) is below 0
   only where Q or K holds no nonzero value (attention_tables' unit is then 0), so that every score, a, is 0. */
w128 s128_shr(w128 a, int64_t s) {
    if (s <= 0) return a;
    int neg = w128_negative(a);
    w128 m = neg ? s128_negate(a) : a;
    uint64_t lo = w128_lo(m), hi = w128_hi(m);
    if (s >= 64) {
        lo = s >= 128 ? 0 : hi >> (s - 64);
        hi = 0;
    } else {
        lo = lo >> s | hi << (64 - s);
        hi >>= s;
    }
    m = w128_make(lo, hi);
    return neg ? s128_negate(m) : m;
}

/* equal logits: equal s and equal keys */
int keyed_same(const keyed *a, const keyed *b) {
    if (s128_less(a->s, b->s) || s128_less(b->s, a->s)) return 0;
    return !a->key || !dyad_cmp(a->key, b->key);
}

int keyed_order(const void *x, const void *y) {
    const keyed *a = x, *b = y;
    if (s128_less(a->s, b->s)) return -1;
    if (s128_less(b->s, a->s)) return 1;
    if (a->key) {
        int c = dyad_cmp(a->key, b->key);
        if (c) return c;
    }
    return a->j < b->j ? -1 : a->j > b->j;
}

/* p (3 words, a weight at 2**-F) times the entries of d's nonzero digits in T (Lemma D), z counting them */
int att_digits_mul(uint64_t p[3], int *z, w128 d, const att_tables *T) {
    size_t per = (size_t)1 << ATT_B;
    uint64_t dw[2] = {w128_lo(d), w128_hi(d)}, prod[6];
    for (int t = 8; t-- > 0;) {
        uint64_t digit = dw[t / 4] >> (16 * (t % 4)) & 0xffff;
        if (!digit) continue;
        if (t >= T->digits) return EXACTREF_INTERNAL;
        const uint64_t *e = T->entries + (((size_t)t * per) + digit) * 3;
        if (!*z) {
            memcpy(p, e, 3 * sizeof *p);
        } else {
            words_mul(prod, p, 3, e, 3);
            words_shr(p, 3, prod, 6, ATT_F);
        }
        ++*z;
    }
    return 0;
}

static uint64_t word_gcd(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* a mod d for a word d > 0 */
static uint64_t big_mod_word(const bigint *a, uint64_t d) {
    uint64_t r = 0;
    for (size_t i = a->n; i-- > 0;) word_div(&r, r, a->w[i], d);
    return r;
}

/* The encoding and entry of the exact value t / n (t a dyad, n >= 1): reduced, and +0 (-0 rounding toward -inf)
   when zero. */
static int exact_mean(bigint *bits, entry *x, const floatfmt *f, int rnd, int ovf, const dyad *t, uint64_t n) {
    bigint num, den, zero;
    big_init(&num);
    big_init(&den);
    big_init(&zero);
    int neg = t->m.neg, rc = big_copy(&num, &t->m);
    num.neg = 0;
    if (!rc && !num.n) {
        rc = floatfmt_encode(bits, f, VALUE_FINITE, rnd == ROUND_TOWARD_NEGATIVE, &zero, 0, rnd, ovf);
        if (!rc && !(rc = big_set_word(&den, 1, 0))) rc = entry_ratio(x, 0, &zero, &den);
    } else if (!rc) {
        /* num / den = t / n, reduced: n's odd part by a word gcd, then the powers of 2 */
        uint64_t odd = n;
        int64_t twos = 0;
        while (!(odd & 1)) {
            odd >>= 1;
            twos++;
        }
        uint64_t g = word_gcd(big_mod_word(&num, odd), odd);
        if (g > 1) rc = big_div_word(&num, &num, g, 0);
        odd /= g;
        int64_t shift = t->e - twos + big_trailing(&num); /* num 2**shift / odd, num odd after the shift */
        if (!rc) rc = big_shr(&num, &num, big_trailing(&num), 0);
        if (!rc) rc = big_set_word(&den, odd, 0);
        if (!rc) rc = shift >= 0 ? big_shl(&num, &num, shift) : big_shl(&den, &den, -shift);
        if (!rc) rc = floatfmt_encode_ratio(bits, f, neg, &num, &den, rnd, ovf);
        if (!rc) rc = entry_ratio(x, neg, &num, &den);
    }
    big_free(&num);
    big_free(&den);
    big_free(&zero);
    return rc;
}

void att_row_init(att_row *r) {
    r->cap = r->ev = 0;
    r->keys = NULL;
    r->bias = NULL;
    r->capped = NULL;
    r->kinds = NULL;
    r->scored = NULL;
    r->mdist = NULL;
    dyad_init(&r->btop);
    r->scores = NULL;
    r->order = NULL;
    r->acc = NULL;
    r->sums = NULL;
}

void att_row_free(att_row *r) {
    if (r->bias)
        for (size_t j = 0; j < r->cap; j++) dyad_free(&r->bias[j]);
    free(r->bias);
    free(r->capped);
    free(r->kinds);
    free(r->scored);
    free(r->mdist);
    dyad_free(&r->btop);
    free(r->keys);
    free(r->scores);
    free(r->order);
    free(r->acc);
    free(r->sums);
    att_row_init(r);
}

int att_row_reserve(att_row *r, size_t S, size_t Ev) {
    if (r->cap >= S && r->ev >= Ev) return 0;
    att_row_free(r);
    r->scores = malloc(S * sizeof(w128));
    r->order = malloc(S * sizeof(keyed));
    r->acc = malloc(Ev * ATT_ACC * sizeof *r->acc);
    r->sums = malloc(Ev * 2 * sizeof(w128));
    r->keys = malloc((S ? S : 1) * sizeof *r->keys);
    r->bias = malloc((S ? S : 1) * sizeof *r->bias);
    r->capped = malloc(S ? S : 1);
    r->kinds = malloc(S ? S : 1);
    r->scored = malloc(S ? S : 1);
    r->mdist = malloc((S ? S : 1) * sizeof(w128));
    if (r->bias)
        for (size_t j = 0; j < (S ? S : 1); j++) dyad_init(&r->bias[j]);
    r->cap = S ? S : 1; /* what att_row_free frees */
    if (!r->scores || !r->order || !r->acc || !r->sums || !r->keys || !r->bias || !r->capped || !r->kinds || !r->scored || !r->mdist) {
        att_row_free(r);
        return EXACTREF_NOMEM;
    }
    r->cap = S;
    r->ev = Ev;
    return 0;
}

/* The max logit m (sigma s_top U) enclosed in [lo, hi] at p bits. */
int top_logit(dyad *lo, dyad *hi, const att_call *a, w128 top) {
    dyad s, y, r;
    dyad_init(&s);
    dyad_init(&y);
    dyad_init(&r);
    int neg = w128_negative(top);
    w128 m = neg ? s128_negate(top) : top;
    uint64_t parts[2] = {w128_lo(m), w128_hi(m)};
    int rc = words_big(&s.m, parts, 2, 0);
    s.m.neg = s.m.n ? neg : 0;
    s.e = s.m.n ? a->u : 0;
    if (!rc && a->den.n) {
        /* sigma = num / den: m = s num / den */
        dyad sn;
        dyad_init(&sn);
        if (!(rc = big_copy(&sn.m, &a->num))) {
            sn.e = 0;
            rc = dyad_mul_exact(&r, &s, &sn);
        }
        if (!rc && !(rc = big_copy(&y.m, &a->den))) {
            y.e = 0;
            if (!(rc = dyad_div(lo, &r, &y, a->p, 0))) rc = dyad_div(hi, &r, &y, a->p, 1);
        }
        dyad_free(&sn);
    } else if (!rc) {
        /* sigma = 1/sqrt(E): s / sqrt(E) with sqrt(E) in [r_lo, r_hi] */
        dyad e, rlo, rhi;
        dyad_init(&e);
        dyad_init(&rlo);
        dyad_init(&rhi);
        rc = dyad_set_si(&e, a->E, 0);
        if (!rc) rc = dyad_sqrt(&rlo, &e, a->p + 4, 0);
        if (!rc) rc = dyad_sqrt(&rhi, &e, a->p + 4, 1);
        if (!rc) rc = dyad_div(lo, &s, neg ? &rlo : &rhi, a->p, 0);
        if (!rc) rc = dyad_div(hi, &s, neg ? &rhi : &rlo, a->p, 1);
        dyad_free(&e);
        dyad_free(&rlo);
        dyad_free(&rhi);
    }
    dyad_free(&s);
    dyad_free(&y);
    dyad_free(&r);
    return rc;
}

/* Element at of a float mask (encodings in the mask's byte order) as a value. */
int att_mask_read(int *kind, int *neg, uint64_t *sig, int64_t *e, const att_call *a, size_t at) {
    uint64_t bits = 0;
    const unsigned char *x = a->mask + at * a->mask_size;
    for (size_t b = 0; b < a->mask_size; b++) bits |= (uint64_t)x[a->mask_big ? a->mask_size - 1 - b : b] << (8 * b);
    floatfmt_decode_word(kind, neg, sig, e, &a->mask_fmt, bits);
    return 0;
}

/* The row's logits (proofs/scaled_dot_product_attention.md, The exact value): a masked key (above the diagonal with
   is_causal, false in a bool mask, -inf in a float mask at mask[moff + j mstep]) takes no part whatever its score;
   every other key's logit is by IEEE's rules its score's kind (finite unless the query or key rows hold NaN or an
   infinity: then product_element's kind, times sigma's sign, NaN for an infinity times sigma 0) plus its float mask
   value. The keys with a finite logit go to w->keys, with a float mask their values to w->bias and the largest to
   w->btop, and with the mask's tables each one's cap (133 or more below the largest) or distance from it in units of
   2**unit of those tables (w->capped, w->mdist); per key, w->kinds holds its part (ATT_KEY_*) and w->scored is 1
   where its score is finite. *kept is the kept count, or -1 for a NaN row (*kind VALUE_NAN where a logit is NaN, else
   VALUE_INFINITE for a +inf one); w->kinds and w->scored cover every key either way. */
int att_row_logits(att_row *w, const att_call *a, size_t qr, size_t kr0, size_t i, size_t moff, size_t mstep, int *kind, int *kept) {
    size_t n = 0;
    *kept = -1;
    int specials = a->q->special_rows[qr], nan = 0, inf = 0, rc = 0;
    for (size_t j = 0; !specials && j < a->S; j++) specials = a->k->special_rows[kr0 + j];
    product_sums ps;
    if (specials) product_sums_init(&ps);
    for (size_t j = 0; !rc && j < a->S; j++) {
        int bk = VALUE_FINITE, bn = 0, sk = VALUE_FINITE, sn = 0;
        uint64_t sig = 0;
        int64_t e = 0;
        if (a->mask_float) {
            att_mask_read(&bk, &bn, &sig, &e, a, moff + j * mstep);
        } else if (a->causal ? j > i : a->mask && !a->mask[moff + j * mstep]) {
            bk = VALUE_INFINITE;
            bn = 1;
        }
        if (specials) {
            if ((rc = product_element(&ps, a->q, qr, a->k, kr0 + j, PRODUCT_SUM, 0, 0, NULL))) break;
            sk = ps.kind;
            sn = ps.neg;
            if (sk == VALUE_INFINITE && !a->sign) sk = VALUE_NAN;
            if (sk == VALUE_INFINITE && a->sign < 0) sn = !sn;
        }
        w->scored[j] = sk == VALUE_FINITE;
        if (bk == VALUE_INFINITE && bn) {
            w->kinds[j] = ATT_KEY_MASKED;
        } else if (sk == VALUE_NAN || bk == VALUE_NAN || (sk == VALUE_INFINITE && bk == VALUE_INFINITE && sn != bn)) {
            w->kinds[j] = ATT_KEY_SPECIAL;
            nan = 1;
        } else if (sk == VALUE_INFINITE || bk == VALUE_INFINITE) {
            int pos = !(sk == VALUE_INFINITE ? sn : bn); /* +inf makes the row NaN; -inf removes the key */
            w->kinds[j] = pos ? ATT_KEY_SPECIAL : ATT_KEY_REMOVED;
            inf |= pos;
        } else {
            w->kinds[j] = ATT_KEY_KEPT;
            if (a->mask_float) {
                dyad *b = &w->bias[n];
                if (big_set_word(&b->m, sig, bn && sig)) {
                    rc = EXACTREF_NOMEM;
                    break;
                }
                b->e = e;
                if ((!n || dyad_cmp(b, &w->btop) > 0) && (rc = dyad_copy(&w->btop, b))) break;
            }
            w->keys[n++] = j;
        }
    }
    if (specials) product_sums_free(&ps);
    if (rc) return rc;
    if (nan || inf) {
        if (kind) *kind = nan ? VALUE_NAN : VALUE_INFINITE;
        return 0;
    }
    *kept = (int)n;
    if (!a->mask_float || !a->mtables) return 0; /* without the mask's tables, the mask values only (the step-by-step program) */
    dyad d, cap;
    dyad_init(&d);
    dyad_init(&cap);
    rc = dyad_set_si(&cap, 133, 0);
    w128 *dist = w->mdist;
    for (size_t m = 0; !rc && m < n; m++) {
        /* btop - b, exactly, then in units of 2**unit */
        dyad nb;
        dyad_init(&nb);
        if (!(rc = dyad_copy(&nb, &w->bias[m]))) {
            if (nb.m.n) nb.m.neg = !nb.m.neg;
            rc = dyad_add_exact(&d, &w->btop, &nb);
        }
        dyad_free(&nb);
        if (rc) break;
        w->capped[m] = (unsigned char)(dyad_cmp(&d, &cap) >= 0);
        if (w->capped[m]) continue;
        bigint t;
        big_init(&t);
        int64_t shift = d.e - a->mtables->unit;
        rc = shift >= 0 ? big_shl(&t, &d.m, shift) : big_shr(&t, &d.m, -shift, 0);
        if (!rc && t.n > 2) rc = EXACTREF_INTERNAL;
        if (!rc) dist[m] = w128_make(t.n > 0 ? t.w[0] : 0, t.n > 1 ? t.w[1] : 0);
        big_free(&t);
    }
    dyad_free(&d);
    dyad_free(&cap);
    return rc;
}

/* the value kind (finite, a NaN or an infinity of sign neg) into an output's encoding and entry */
static int special_out(unsigned char *enc, size_t size, entry *x, const floatfmt *f, int rnd, int ovf, int kind, int neg) {
    bigint bits, zero;
    big_init(&bits);
    big_init(&zero);
    int rc = floatfmt_encode(&bits, f, kind, neg, &zero, 0, rnd, ovf);
    if (!rc && !(rc = entry_value(x, kind, neg, &zero, 0))) bytes_store(enc, size, &bits);
    big_free(&bits);
    big_free(&zero);
    return rc;
}

int gkeyed_cmp(const gkeyed *a, const gkeyed *b) {
    int c = a->s ? dyad_cmp(a->s, b->s) : 0;
    return c ? c : a->key ? dyad_cmp(a->key, b->key) : 0;
}

int gkeyed_order(const void *x, const void *y) {
    const gkeyed *a = x, *b = y;
    int c = gkeyed_cmp(a, b);
    return c ? c : a->j < b->j ? -1 : a->j > b->j;
}

/* A logit difference sigma X + Y, for exact dyads X (a score difference) and Y (a mask difference, NULL for 0): its exact
   sign in *sign, and with lo its enclosure [lo, hi] at p bits without cancellation between the two terms
   (proofs/scaled_dot_product_attention.md, Logit differences): for sigma = num / den (0 included) the exact
   (num X + den Y) / den rounded outward; for sigma = 1/sqrt(E) (sabs_lo, sabs_hi its ends) X sigma + Y where X and Y do not
   have opposite signs, and else sign(X) (X**2 - E Y**2) / (E (sigma |X| + |Y|)), with an exact numerator and a denominator of
   two terms of one sign. Exactly 0 when the value is. */
int att_affine(int *sign, dyad *lo, dyad *hi, const att_call *a, const dyad *X, const dyad *Y, const dyad *sabs_lo, const dyad *sabs_hi, int64_t p) {
    dyad n, t;
    dyad_init(&n);
    dyad_init(&t);
    int rc = 0, sx = dyad_sign(X), sy = Y ? dyad_sign(Y) : 0, s;
    if (a->den.n) {
        /* N = num X + den Y, exactly; the value N / den, den > 0 */
        dyad dd;
        dyad_init(&dd);
        if (!(rc = big_copy(&t.m, &a->num)) && !(rc = dyad_mul_exact(&n, &t, X)) && !(rc = big_copy(&dd.m, &a->den)) && Y &&
            !(rc = dyad_mul_exact(&t, &dd, Y)))
            rc = dyad_add_exact(&n, &n, &t);
        s = dyad_sign(&n);
        if (!rc && lo && !(rc = dyad_div(lo, &n, &dd, p, 0))) rc = dyad_div(hi, &n, &dd, p, 1);
        dyad_free(&dd);
    } else if (!sx || !sy || sx == sy) {
        /* no cancellation: X sigma + Y, the product at p + 2 bits, its end of sigma by X's sign */
        s = sx ? sx : sy;
        for (int up = 0; !rc && lo && up < 2; up++) {
            dyad *r = up ? hi : lo;
            if (!sx) rc = Y ? dyad_rnd(r, Y, p, up) : dyad_set_si(r, 0, 0);
            else if (!(rc = dyad_mul(&t, (sx > 0) == up ? sabs_hi : sabs_lo, X, p + 2, up))) rc = Y ? dyad_add(r, &t, Y, p, up) : dyad_rnd(r, &t, p, up);
        }
    } else {
        /* X and Y of opposite signs: X sigma + Y = (X + Y sqrt(E)) / sqrt(E) = sign(X) N / (E Q), N = X**2 - E Y**2 exactly and
           Q = sigma |X| + |Y| > 0 enclosed from sigma's ends */
        dyad e;
        dyad_init(&e);
        if (!(rc = dyad_mul_exact(&n, X, X)) && !(rc = dyad_mul_exact(&t, Y, Y)) && !(rc = dyad_set_si(&e, a->E, 0)) && !(rc = dyad_mul_exact(&t, &t, &e)) &&
            !(rc = dyad_neg(&t, &t)))
            rc = dyad_add_exact(&n, &n, &t);
        s = sx * dyad_sign(&n);
        if (!rc && lo) {
            /* |value| = |N| / (E Q): its lower end over Q's upper end, its upper end over Q's lower end */
            dyad ax, ay, qlo, qhi;
            dyad_init(&ax);
            dyad_init(&ay);
            dyad_init(&qlo);
            dyad_init(&qhi);
            if (!(rc = dyad_abs(&ax, X)) && !(rc = dyad_abs(&ay, Y)) && !(rc = dyad_abs(&n, &n)) && !(rc = dyad_mul(&t, sabs_lo, &ax, p + 4, 0)) &&
                !(rc = dyad_add(&qlo, &t, &ay, p + 4, 0)) && !(rc = dyad_mul_exact(&qlo, &qlo, &e)) && !(rc = dyad_mul(&t, sabs_hi, &ax, p + 4, 1)) &&
                !(rc = dyad_add(&qhi, &t, &ay, p + 4, 1)) && !(rc = dyad_mul_exact(&qhi, &qhi, &e))) {
                if (s >= 0) {
                    if (!(rc = dyad_div(lo, &n, &qhi, p, 0))) rc = dyad_div(hi, &n, &qlo, p, 1);
                } else if (!(rc = dyad_div(&t, &n, &qlo, p, 1)) && !(rc = dyad_neg(lo, &t)) && !(rc = dyad_div(&t, &n, &qhi, p, 0))) {
                    rc = dyad_neg(hi, &t);
                }
            }
            dyad_free(&ax);
            dyad_free(&ay);
            dyad_free(&qlo);
            dyad_free(&qhi);
        }
        dyad_free(&e);
    }
    if (sign) *sign = s;
    dyad_free(&n);
    dyad_free(&t);
    return rc;
}

/* The argument of kept key j measured from kept key k: x = l_k - l_j = sigma (s_k - s_j) + b_k - b_j (sd the exact scores,
   bias the float mask's values or NULL), so e**-x = e**(l_j - l_k); in [lo, hi] at p bits by att_affine, exactly 0 for
   j = k, and at least 0 when k has the largest logit */
int att_argument(dyad *lo, dyad *hi, const att_call *a, const dyad *sd, const dyad *bias, size_t k, size_t j, const dyad *sabs_lo, const dyad *sabs_hi,
                 int64_t p) {
    dyad X, Y;
    dyad_init(&X);
    dyad_init(&Y);
    int rc = dyad_neg(&X, &sd[j]);
    if (!rc) rc = dyad_add_exact(&X, &sd[k], &X);
    if (!rc && bias && !(rc = dyad_neg(&Y, &bias[j]))) rc = dyad_add_exact(&Y, &bias[k], &Y);
    if (!rc) rc = att_affine(NULL, lo, hi, a, &X, bias ? &Y : NULL, sabs_lo, sabs_hi, p);
    dyad_free(&X);
    dyad_free(&Y);
    return rc;
}

/* the exact sign of l_i - l_j (att_affine) */
int att_logit_cmp(int *sign, const att_call *a, const dyad *sd, const dyad *bias, size_t i, size_t j) {
    dyad X, Y;
    dyad_init(&X);
    dyad_init(&Y);
    int rc = dyad_neg(&X, &sd[j]);
    if (!rc) rc = dyad_add_exact(&X, &sd[i], &X);
    if (!rc && bias && !(rc = dyad_neg(&Y, &bias[j]))) rc = dyad_add_exact(&Y, &bias[i], &Y);
    if (!rc) rc = att_affine(sign, NULL, NULL, a, &X, bias ? &Y : NULL, NULL, NULL, 0);
    dyad_free(&X);
    dyad_free(&Y);
    return rc;
}

/* The kept key of the row's largest logit, compared exactly (att_logit_cmp), the first of equal ones; the extreme score's
   key top without a float mask. att_argument measures the weights from it, so one weight is exactly 1 and every
   argument at least 0. */
int att_reference_key(size_t *k, const att_call *a, const dyad *sd, const dyad *bias, size_t n, size_t top) {
    *k = top;
    if (!bias) return 0;
    *k = 0;
    int rc = 0;
    for (size_t j = 1; !rc && j < n; j++) {
        int s;
        if (!(rc = att_logit_cmp(&s, a, sd, bias, j, *k)) && s > 0) *k = j;
    }
    return rc;
}

/* The output's sign test (proofs/scaled_dot_product_attention.md, The sign test): for an output o left undecided in
   [lo, hi], a grid point b where its rounding switches, o - b = e**-(x_a) t' / Z' with t' = sum_g C_g e**(x_a - x_g),
   C_g = S_g - b |g|, a the group of the largest logit among those with C_g != 0; decided by the sign of t' (case 2) or its
   enclosure. */
static int output_sign_test(bigint *bits, entry *x, int *none, const att_row *w, const att_call *a, size_t n, const gkeyed *order, const dyad *sd,
                            const dyad *vals, const dyad *alo, const dyad *ahi, const dyad *sabs_lo, const dyad *sabs_hi, const dyad *Zlo,
                            const dyad *Zhi, const dyad *lo, const dyad *hi, int64_t p, int last) {
    dyad b, sg, cg, xl, xh, el, eh, tlo, thi, t1, olo, ohi, *C = malloc((n + 1) * sizeof *C);
    size_t *rep = malloc((n + 1) * sizeof *rep), G = 0;
    dyad *ds[] = {&b, &sg, &cg, &xl, &xh, &el, &eh, &tlo, &thi, &t1, &olo, &ohi};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int rc = !C || !rep ? EXACTREF_NOMEM : 0, nob = 1;
    if (!rc) rc = switch_point(&b, &nob, &a->out, a->rnd, a->ovf, lo, hi);
    /* the groups' C_g and representatives */
    for (size_t at = 0; !rc && !nob && at < n;) {
        size_t start = at;
        rc = dyad_set_si(&sg, 0, 0);
        do {
            if (!rc) rc = dyad_add_exact(&sg, &sg, &vals[order[at].j]);
            at++;
        } while (!rc && at < n && !gkeyed_cmp(&order[at - 1], &order[at]));
        dyad_init(&C[G]);
        if (!rc && !(rc = dyad_set_si(&t1, (int64_t)(at - start), 0)) && !(rc = dyad_mul_exact(&t1, &b, &t1)) && !(rc = dyad_neg(&t1, &t1)))
            rc = dyad_add_exact(&C[G], &sg, &t1);
        rep[G++] = order[start].j;
    }
    /* a: the group of the largest logit among those with C_g != 0, compared exactly, so every x_a - x_g <= 0 */
    size_t ga = G;
    const dyad *bias = a->mask_float ? w->bias : NULL;
    for (size_t g = 0; !rc && !nob && g < G; g++) {
        int s = 1;
        if (dyad_sign(&C[g]) && (ga == G || (!(rc = att_logit_cmp(&s, a, sd, bias, rep[g], rep[ga])) && s > 0))) ga = g;
    }
    if (!rc && !nob && ga < G) {
        /* t' = C_a + sum over the other groups of C_g e**(x_a - x_g), x_a - x_g = l_g - l_a by att_affine */
        size_t ja = rep[ga];
        rc = dyad_copy(&tlo, &C[ga]);
        if (!rc) rc = dyad_copy(&thi, &C[ga]);
        for (size_t g = 0; !rc && g < G; g++) {
            if (g == ga || !dyad_sign(&C[g])) continue;
            size_t jg = rep[g];
            if ((rc = att_argument(&xl, &xh, a, sd, bias, jg, ja, sabs_lo, sabs_hi, p))) break;
            if ((rc = exp_clamped(&el, &xl, p, 0)) || (rc = exp_clamped(&eh, &xh, p, 1))) break;
            int pos = dyad_sign(&C[g]) > 0;
            if ((rc = dyad_mul(&t1, pos ? &el : &eh, &C[g], p + 64, 0)) || (rc = dyad_add(&tlo, &tlo, &t1, p + 64, 0))) break;
            if ((rc = dyad_mul(&t1, pos ? &eh : &el, &C[g], p + 64, 1)) || (rc = dyad_add(&thi, &thi, &t1, p + 64, 1))) break;
        }
        int side = dyad_sign(&tlo) > 0 ? 1 : dyad_sign(&thi) < 0 ? -1 : 0;
        dyad big40;
        dyad_init(&big40);
        if (!rc) rc = dyad_set_si(&big40, 1, 40);
        if (!rc && side && dyad_cmp(&alo[ja], &big40) < 0) { /* from 2**40, where Lemma X would be clamped, the tiny rule */
            /* e**-(x_a) by Lemma X; o - b = e**-(x_a) t' / Z', each end over the end of Z' that makes it least (largest) */
            if (!(rc = dyad_neg(&t1, &ahi[ja])) && !(rc = exp_clamped(&el, &t1, p, 0)) && !(rc = dyad_neg(&t1, &alo[ja]))) rc = exp_clamped(&eh, &t1, p, 1);
            if (!rc && !(rc = dyad_mul(&t1, &tlo, side > 0 ? &el : &eh, p + 64, 0))) rc = dyad_div(&olo, &t1, side > 0 ? Zhi : Zlo, p, 0);
            if (!rc && !(rc = dyad_mul(&t1, &thi, side > 0 ? &eh : &el, p + 64, 1))) rc = dyad_div(&ohi, &t1, side > 0 ? Zlo : Zhi, p, 1);
        } else if (!rc && side) {
            /* past 2**40: |o - b| is below 2**-(LIMIT + 1); the tiny-offset rule takes that point with t''s sign */
            if (!(rc = dyad_set_si(&olo, side, -(EXACT_LIMIT + 1)))) rc = dyad_copy(&ohi, &olo);
        }
        dyad_free(&big40);
        if (!rc && side) rc = offset_rounding(bits, x, none, &a->out, a->rnd, a->ovf, &b, &b, &olo, &ohi, p, last);
    }
    for (size_t g = 0; C && g < G; g++) dyad_free(&C[g]);
    free(C);
    free(rep);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* logsumexp's sign test (proofs/scaled_dot_product_attention.md, logsumexp): for log(Z) left undecided in [lo, hi], a
   grid point b and each group's s_g - b in [Y_lo, Y_hi]. log(Z) - b lies in [max Y_lo, max Y_hi + log n], whose
   positive lower end or negative upper end gives the offset at once; otherwise, once every Y_hi is at most 1 (a larger
   one waits for the next round), T = sum_g n_g e**(s_g - b) - 1 with the -1 joined to the group at b (else the largest
   term's), and log(Z) - b = log1p(T) */
static int lse_sign_test(bigint *bits, entry *x, int *none, const att_row *w, const att_call *a, size_t n, const gkeyed *order, const dyad *sd,
                         const dyad *sabs_lo, const dyad *sabs_hi, const dyad *lo, const dyad *hi, int64_t p, int last) {
    dyad b, yl, yh, t1, t2, tlo, thi, olo, ohi, *Y = malloc((2 * n + 1) * sizeof *Y);
    size_t *rep = malloc((n + 1) * sizeof *rep), *cnt = malloc((n + 1) * sizeof *cnt), G = 0;
    dyad *ds[] = {&b, &yl, &yh, &t1, &t2, &tlo, &thi, &olo, &ohi};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int rc = !Y || !rep || !cnt ? EXACTREF_NOMEM : 0, nob = 1;
    if (!rc) rc = switch_point(&b, &nob, &a->lse, a->lrnd, a->lovf, lo, hi);
    for (size_t at = 0; !rc && !nob && at < n;) {
        size_t start = at;
        do at++;
        while (at < n && !gkeyed_cmp(&order[at - 1], &order[at]));
        rep[G] = order[start].j;
        cnt[G++] = at - start;
    }
    /* each group's s_g - b in [Y[2g], Y[2g + 1]], and the group exactly at b */
    size_t gb = G;
    for (size_t g = 0; !rc && !nob && g < G; g++) {
        size_t j = rep[g];
        dyad_init(&Y[2 * g]);
        dyad_init(&Y[2 * g + 1]);
        /* s_g - b = sigma s + (b_g - b) by att_affine, exactly 0 for the group at b */
        int at_b;
        if ((rc = dyad_neg(&t2, &b))) break;
        if (a->mask_float && (rc = dyad_add_exact(&t2, &w->bias[j], &t2))) break;
        if ((rc = att_affine(&at_b, &Y[2 * g], &Y[2 * g + 1], a, &sd[j], &t2, sabs_lo, sabs_hi, p))) break;
        if (!at_b) gb = g;
    }
    size_t gj = gb;
    if (gj == G) /* else the largest term's group */
        for (size_t g = 0; g < G; g++)
            if (gj == G || dyad_cmp(&Y[2 * g + 1], &Y[2 * gj + 1]) > 0) gj = g;
    if (!rc && !nob && G) {
        int side = 0;
        /* log(Z) >= s_g for every group and Z <= n e**(max s_g): the largest lower end, and the largest upper end plus
           log n rounded up */
        size_t glo = 0, ghi = 0;
        for (size_t g = 1; g < G; g++) {
            if (dyad_cmp(&Y[2 * g], &Y[2 * glo]) > 0) glo = g;
            if (dyad_cmp(&Y[2 * g + 1], &Y[2 * ghi + 1]) > 0) ghi = g;
        }
        if (!(rc = dyad_set_si(&t1, (int64_t)n, 0)) && !(rc = log_end(&t2, &t1, p, 1))) rc = dyad_add(&t2, &Y[2 * ghi + 1], &t2, p, 1);
        /* the T path where it is safe: every upper end at most 1, and the largest at least -log n, so that the exact sum
           1 + T = sum n_g e**(s_g - b) stays within a few binades of 1; the arguments' ends otherwise */
        int safe = 0;
        if (!rc && !(rc = dyad_set_si(&t1, 1, 0))) safe = dyad_sign(&t2) >= 0 && dyad_cmp(&Y[2 * ghi + 1], &t1) <= 0;
        if (!rc && !safe && (dyad_sign(&Y[2 * glo]) > 0 || dyad_sign(&t2) < 0)) {
            side = dyad_sign(&Y[2 * glo]) > 0 ? 1 : -1;
            if (!(rc = dyad_copy(&olo, &Y[2 * glo]))) rc = dyad_copy(&ohi, &t2);
        } else if (!rc && safe) {
            /* every s_g - b at most 1: each term at most n_g e */
            int tiny = gb < G && cnt[gb] == 1;
            dyad m40;
            dyad_init(&m40);
            rc = dyad_set_si(&m40, -1, 40);
            if (!rc) rc = dyad_set_si(&tlo, 0, 0);
            if (!rc) rc = dyad_set_si(&thi, 0, 0);
            for (size_t g = 0; !rc && g < G; g++) {
                if (g != gj) {
                    if (dyad_cmp(&Y[2 * g + 1], &m40) > 0) tiny = 0;
                    if ((rc = exp_clamped(&yl, &Y[2 * g], p, 0)) || (rc = exp_clamped(&yh, &Y[2 * g + 1], p, 1)) || (rc = dyad_set_si(&t1, (int64_t)cnt[g], 0)) ||
                        (rc = dyad_mul_exact(&yl, &yl, &t1)) || (rc = dyad_mul_exact(&yh, &yh, &t1)))
                        break;
                } else if (g == gb) { /* n_g - 1, exactly */
                    if ((rc = dyad_set_si(&yl, (int64_t)cnt[g] - 1, 0))) break;
                    rc = dyad_copy(&yh, &yl);
                } else { /* n_g expm1(y) + (n_g - 1), expm1 by Lemma X at p + max(0, -top(y)) + 8 bits less 1 */
                    dyad one;
                    dyad_init(&one);
                    rc = dyad_set_si(&one, -1, 0);
                    for (int end = 0; !rc && end < 2; end++) {
                        const dyad *y = &Y[2 * g + end];
                        int64_t ty = dyad_sign(y) ? y->e + big_bits(&y->m) : 0, pe = p + (ty < 0 ? -ty : 0) + 8;
                        dyad *r = end ? &yh : &yl;
                        if (!(rc = exp_clamped(r, y, pe, end)) && !(rc = dyad_add_exact(r, r, &one)) && !(rc = dyad_set_si(&t1, (int64_t)cnt[g], 0)) &&
                            !(rc = dyad_mul_exact(r, r, &t1)) && !(rc = dyad_set_si(&t1, (int64_t)cnt[g] - 1, 0)))
                            rc = dyad_add_exact(r, r, &t1);
                    }
                    dyad_free(&one);
                    if (rc) break;
                }
                if ((rc = dyad_add(&tlo, &tlo, &yl, p + 64, 0))) break;
                rc = dyad_add(&thi, &thi, &yh, p + 64, 1);
            }
            dyad_free(&m40);
            if (!rc && tiny) {
                /* only terms past -2**40 left: T > 0 and below 2**-(LIMIT + 1); the tiny-offset rule */
                side = 1;
                if (!(rc = dyad_set_si(&olo, 1, -(EXACT_LIMIT + 1)))) rc = dyad_copy(&ohi, &olo);
            } else if (!rc && (dyad_sign(&tlo) > 0 || dyad_sign(&thi) < 0)) {
                side = dyad_sign(&tlo) > 0 ? 1 : -1;
                if (tlo.e + big_bits(&tlo.m) <= -p && thi.e + big_bits(&thi.m) <= -p) {
                    /* |T| < 2**-p: T - 2 T**2 <= log1p(T) <= T (for |T| <= 1/2), both ends at p bits */
                    if (!(rc = dyad_mul(&t1, &tlo, &tlo, p, 1)) && !(rc = dyad_shift(&t1, 1))) { /* 2 T**2, rounded up */
                        if (!(rc = dyad_neg(&t1, &t1)) && !(rc = dyad_add(&olo, &tlo, &t1, p, 0))) rc = dyad_rnd(&ohi, &thi, p, 1);
                    }
                } else {
                    /* each end by its own size: below 2**-p, T - 2 T**2 rounded down (the lower end) or T rounded up (the
                       upper); else log1p(T) = log(1 + T) by Lemma Y at that end of 1 + T (an exact sum of at most about
                       2p + 64 bits, as T is at least 2**-p there) */
                    int small_lo = tlo.e + big_bits(&tlo.m) <= -p, small_hi = thi.e + big_bits(&thi.m) <= -p;
                    dyad one;
                    dyad_init(&one);
                    if (!(rc = dyad_set_si(&one, 1, 0))) {
                        if (small_lo) {
                            if (!(rc = dyad_mul(&t1, &tlo, &tlo, p, 1)) && !(rc = dyad_shift(&t1, 1))) { /* 2 T**2, rounded up */
                                if (!(rc = dyad_neg(&t1, &t1))) rc = dyad_add(&olo, &tlo, &t1, p, 0);
                            }
                        } else if (!(rc = dyad_add_exact(&t1, &tlo, &one))) {
                            if (dyad_sign(&t1) > 0) rc = log_end(&olo, &t1, p, 0);
                            else side = 0;
                        }
                        if (!rc && side) {
                            if (small_hi) rc = dyad_rnd(&ohi, &thi, p, 1);
                            else if (!(rc = dyad_add_exact(&t2, &thi, &one))) rc = log_end(&ohi, &t2, p, 1);
                        }
                    }
                    dyad_free(&one);
                }
            }
        }
        /* else some s_g - b may pass 1 with the sign still open: the next round */
        if (!rc && side) rc = offset_rounding(bits, x, none, &a->lse, a->lrnd, a->lovf, &b, &b, &olo, &ohi, p, last);
    }
    for (size_t g = 0; Y && g < 2 * G; g++) dyad_free(&Y[g]);
    free(Y);
    free(rep);
    free(cnt);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* The enclosures without tables at precision p: in the first round (round 0) every output's and logsumexp's, the
   special columns and rows with no kept key decided; in a refinement round only those still undecided (status 1),
   with the sign tests for what the enclosures leave (proofs/scaled_dot_product_attention.md, Other outputs). */
static int reference_general(att_row *w, const att_call *a, size_t qr, size_t kr0, size_t vr0, size_t n, int64_t p, int round, unsigned char *enc,
                             entry *entries, unsigned char *status, unsigned char *lse_enc, entry *lse_entry, unsigned char *lse_status) {
    const operand *v = a->v;
    size_t S = a->S, Ev = a->Ev, *keys = w->keys;
    int rc = 0, last = round && p >= a->cap;
    if (!round) *lse_status = 1;
    /* the special columns: a kept key's value that is not finite (the keys the row does not keep take no part) */
    for (size_t c = 0; !rc && !round && c < Ev; c++) {
        status[c] = 1;
        if (!v->flagged_count) continue;
        int nan = 0, pos = 0, negs = 0;
        for (size_t m = 0; m < n; m++) {
            int kind, neg;
            uint64_t sig;
            int64_t e;
            element_decode_slot(&kind, &neg, &sig, &e, v, (vr0 + keys[m]) * Ev + c);
            if (kind == VALUE_FINITE) continue;
            if (kind == VALUE_NAN) nan = 1;
            else if (neg) negs = 1;
            else pos = 1;
        }
        if (nan || pos || negs) {
            int kind = nan || (pos && negs) ? VALUE_NAN : VALUE_INFINITE;
            rc = special_out(enc + c * a->out_size, a->out_size, &entries[c], &a->out, a->rnd, a->ovf, kind, kind == VALUE_INFINITE && negs);
            status[c] = 0;
        }
    }
    if (rc) return rc;
    if (!n && !round) {
        /* no kept key: every weight is 0, so each output is an exact zero (-0 rounding toward -inf over keys), and
           logsumexp is log(0) = -inf */
        for (size_t c = 0; !rc && c < Ev; c++)
            if (status[c] && !(rc = special_out(enc + c * a->out_size, a->out_size, &entries[c], &a->out, a->rnd, a->ovf, VALUE_FINITE,
                                                a->rnd == ROUND_TOWARD_NEGATIVE && S > 0)))
                status[c] = 0;
        if (!rc && !(rc = special_out(lse_enc, a->lse_size, lse_entry, &a->lse, a->lrnd, a->lovf, VALUE_INFINITE, 1))) *lse_status = 0;
        return rc;
    }
    /* per kept key: the score, the rational logit's order, the value, the weight's ends and its argument's */
    dyad *sd = malloc(7 * n * sizeof *sd), *lk = sd ? sd + n : NULL, *vals = sd ? sd + 2 * n : NULL, *wlo = sd ? sd + 3 * n : NULL, *whi = sd ? sd + 4 * n : NULL;
    dyad *alo = sd ? sd + 5 * n : NULL, *ahi = sd ? sd + 6 * n : NULL;
    gkeyed *order = malloc(n * sizeof *order);
    if (!sd || !order) {
        free(sd);
        free(order);
        return EXACTREF_NOMEM;
    }
    for (size_t z = 0; z < 7 * n; z++) dyad_init(&sd[z]);
    dyad sabs_lo, sabs_hi, d1, d2, d3, Zlo, Zhi, Nlo, Nhi, lo, hi, gs, g0, t1, t2;
    dyad *ds[] = {&sabs_lo, &sabs_hi, &d1, &d2, &d3, &Zlo, &Zhi, &Nlo, &Nhi, &lo, &hi, &gs, &g0, &t1, &t2};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    bigint bits;
    big_init(&bits);
    product_sums ps;
    product_sums_init(&ps);
    /* the kept keys' exact scores */
    for (size_t j = 0; !rc && j < n; j++)
        if (!(rc = product_element(&ps, a->q, qr, a->k, kr0 + keys[j], PRODUCT_SUM, 0, 0, NULL)) ) rc = dyad_from_big(&sd[j], &ps.s, ps.e);
    /* groups of equal logits: equal scores (all keys with sigma 0); with a float mask equal (score, mask value) pairs for
       sigma = 1/sqrt(E), equal sigma s + b as rationals (num s + den b) for a rational sigma, equal b for sigma 0 */
    int rational = a->mask_float && a->den.n && a->sign;
    if (!rc && rational) {
        dyad nn, dd;
        dyad_init(&nn);
        dyad_init(&dd);
        if (!(rc = big_copy(&nn.m, &a->num))) rc = big_copy(&dd.m, &a->den);
        for (size_t j = 0; !rc && j < n; j++)
            if (!(rc = dyad_mul_exact(&t1, &sd[j], &nn)) && !(rc = dyad_mul_exact(&t2, &w->bias[j], &dd))) rc = dyad_add_exact(&lk[j], &t1, &t2);
        dyad_free(&nn);
        dyad_free(&dd);
    }
    for (size_t j = 0; j < n; j++) {
        order[j].s = a->sign && !rational ? &sd[j] : NULL;
        order[j].key = !a->mask_float ? NULL : rational ? &lk[j] : &w->bias[j];
        order[j].j = j;
    }
    qsort(order, n, sizeof *order, gkeyed_order);
    /* |sigma|'s ends, the largest logit's score s_top, and the weights' ends (Lemma X) and the row sum's */
    if (!rc && a->den.n) {
        dyad x, y;
        dyad_init(&x);
        dyad_init(&y);
        if (!(rc = big_copy(&x.m, &a->num)) && !(rc = big_copy(&y.m, &a->den))) {
            x.m.neg = 0;
            if (!(rc = dyad_div(&sabs_lo, &x, &y, p, 0))) rc = dyad_div(&sabs_hi, &x, &y, p, 1);
        }
        dyad_free(&x);
        dyad_free(&y);
    } else if (!rc) {
        dyad x, r;
        dyad_init(&x);
        dyad_init(&r);
        if (!(rc = dyad_set_si(&x, a->E, 0)) && !(rc = dyad_sqrt(&r, &x, p + 4, 1)) && !(rc = dyad_set_si(&x, 1, 0)) && !(rc = dyad_div(&sabs_lo, &x, &r, p, 0)) &&
            !(rc = dyad_set_si(&x, a->E, 0)) && !(rc = dyad_sqrt(&r, &x, p + 4, 0)) && !(rc = dyad_set_si(&x, 1, 0)))
            rc = dyad_div(&sabs_hi, &x, &r, p, 1);
        dyad_free(&x);
        dyad_free(&r);
    }
    size_t top = 0, ref = 0;
    for (size_t j = 1; j < n; j++)
        if (a->sign > 0 ? dyad_cmp(&sd[j], &sd[top]) > 0 : dyad_cmp(&sd[j], &sd[top]) < 0) top = j;
    const dyad *bias = a->mask_float ? w->bias : NULL;
    if (!rc) rc = att_reference_key(&ref, a, sd, bias, n, top);
    if (!rc && !(rc = dyad_set_si(&Zlo, 0, 0))) rc = dyad_set_si(&Zhi, 0, 0);
    for (size_t j = 0; !rc && j < n; j++) {
        /* the argument from the largest logit's key, whose weight is exactly 1 (so Z' >= 1) */
        if ((rc = att_argument(&alo[j], &ahi[j], a, sd, bias, ref, j, &sabs_lo, &sabs_hi, p))) break;
        if ((rc = dyad_neg(&d2, &alo[j])) || (rc = dyad_neg(&d3, &ahi[j])) || (rc = exp_clamped(&wlo[j], &d3, p, 0)) || (rc = exp_clamped(&whi[j], &d2, p, 1)) ||
            (rc = dyad_add(&Zlo, &Zlo, &wlo[j], p, 0)))
            break;
        rc = dyad_add(&Zhi, &Zhi, &whi[j], p, 1);
    }
    /* per column: the exact case (every group's mean equal), else N's enclosure over Z's */
    for (size_t c = 0; !rc && c < Ev; c++) {
        if (!status[c]) continue;
        for (size_t j = 0; !rc && j < n; j++) {
            int kind, neg;
            uint64_t sig;
            int64_t e;
            element_decode_slot(&kind, &neg, &sig, &e, v, (vr0 + keys[j]) * Ev + c);
            if (!(rc = big_set_word(&vals[j].m, sig, neg && sig))) vals[j].e = sig ? e : 0;
        }
        int exact = 1;
        size_t s0 = 0;
        for (size_t at = 0; !rc && exact && at < n;) {
            size_t start = at;
            rc = dyad_set_si(&gs, 0, 0);
            do {
                if (!rc) rc = dyad_add_exact(&gs, &gs, &vals[order[at].j]);
                at++;
            } while (!rc && at < n && !gkeyed_cmp(&order[at - 1], &order[at]));
            size_t size = at - start;
            if (rc) break;
            if (!s0) {
                rc = dyad_copy(&g0, &gs);
                s0 = size;
            } else if (!(rc = dyad_set_si(&t1, (int64_t)size, 0)) && !(rc = dyad_mul_exact(&t1, &g0, &t1)) && !(rc = dyad_set_si(&t2, (int64_t)s0, 0)) &&
                       !(rc = dyad_mul_exact(&t2, &gs, &t2))) {
                exact = !dyad_cmp(&t1, &t2); /* g0 size == gs s0 */
            }
        }
        if (rc) break;
        if (exact) {
            if (!(rc = exact_mean(&bits, &entries[c], &a->out, a->rnd, a->ovf, &g0, s0))) {
                bytes_store(enc + c * a->out_size, a->out_size, &bits);
                status[c] = 0;
            }
            continue;
        }
        if ((rc = dyad_set_si(&Nlo, 0, 0)) || (rc = dyad_set_si(&Nhi, 0, 0))) break;
        for (size_t j = 0; !rc && j < n; j++) {
            if (!vals[j].m.n) continue;
            int pos = !vals[j].m.neg;
            if (!(rc = dyad_mul(&d1, pos ? &wlo[j] : &whi[j], &vals[j], p + 64, 0)) && !(rc = dyad_add(&Nlo, &Nlo, &d1, p + 64, 0)) &&
                !(rc = dyad_mul(&d1, pos ? &whi[j] : &wlo[j], &vals[j], p + 64, 1)))
                rc = dyad_add(&Nhi, &Nhi, &d1, p + 64, 1);
        }
        int none = 1;
        if (!rc && !(rc = dyad_div(&lo, &Nlo, Nlo.m.neg ? &Zlo : &Zhi, p, 0)) && !(rc = dyad_div(&hi, &Nhi, Nhi.m.neg ? &Zhi : &Zlo, p, 1)))
            rc = decide_rounding(&bits, &entries[c], &none, &a->out, a->rnd, a->ovf, &lo, &hi, last);
        if (!rc && none && round)
            rc = output_sign_test(&bits, &entries[c], &none, w, a, n, order, sd, vals, alo, ahi, &sabs_lo, &sabs_hi, &Zlo, &Zhi, &lo, &hi, p, last);
        if (!rc) {
            status[c] = (unsigned char)none;
            if (!none) bytes_store(enc + c * a->out_size, a->out_size, &bits);
        }
    }
    /* logsumexp = m' + log(Z'), m' = sigma s_ref + b_ref the reference key's logit; one kept key is left to the caller in
       the first round (its exact cases), and refined as its logit s / sqrt(E) + b otherwise */
    if (!rc && *lse_status && n == 1 && round) {
        rc = att_affine(NULL, &lo, &hi, a, &sd[0], a->mask_float ? &w->bias[0] : NULL, &sabs_lo, &sabs_hi, p);
        int none = 1;
        if (!rc) rc = decide_rounding(&bits, lse_entry, &none, &a->lse, a->lrnd, a->lovf, &lo, &hi, last);
        if (!rc && none) rc = lse_sign_test(&bits, lse_entry, &none, w, a, n, order, sd, &sabs_lo, &sabs_hi, &lo, &hi, p, last);
        if (!rc && !none) {
            bytes_store(lse_enc, a->lse_size, &bits);
            *lse_status = 0;
        }
    }
    if (!rc && *lse_status && n > 1) {
        /* m' = sigma s_ref + b_ref by att_affine */
        rc = att_affine(NULL, &lo, &hi, a, &sd[ref], a->mask_float ? &w->bias[ref] : NULL, &sabs_lo, &sabs_hi, p);
        if (!rc && !(rc = log_end(&d1, &Zlo, p, 0)) && !(rc = log_end(&d2, &Zhi, p, 1)) && !(rc = dyad_add(&lo, &lo, &d1, p, 0)))
            rc = dyad_add(&hi, &hi, &d2, p, 1);
        int none = 1;
        if (!rc) rc = decide_rounding(&bits, lse_entry, &none, &a->lse, a->lrnd, a->lovf, &lo, &hi, last);
        if (!rc && none && round) rc = lse_sign_test(&bits, lse_entry, &none, w, a, n, order, sd, &sabs_lo, &sabs_hi, &lo, &hi, p, last);
        if (!rc && !none) {
            bytes_store(lse_enc, a->lse_size, &bits);
            *lse_status = 0;
        }
    }
    for (size_t z = 0; z < 7 * n; z++) dyad_free(&sd[z]);
    free(sd);
    free(order);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    big_free(&bits);
    product_sums_free(&ps);
    return rc;
}

/* the first round: the row's logits, then Lemma D's tables (or, without them, reference_general at the first
   precision); *kept is the number of kept keys, -1 for a NaN row */
static int reference_first(att_row *w, const att_call *a, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, unsigned char *enc,
                           entry *entries, unsigned char *status, unsigned char *lse_enc, entry *lse_entry, unsigned char *lse_status, int *kept_keys) {
    const operand *q = a->q, *k = a->k, *v = a->v;
    size_t S = a->S, Ev = a->Ev, E = q->cols;
    int rc = att_row_reserve(w, S, Ev), kind = VALUE_NAN;
    if (rc) return rc;
    int kept;
    rc = att_row_logits(w, a, qr, kr0, i, moff, mstep, &kind, &kept);
    *kept_keys = kept;
    if (rc) return rc;
    if (kept < 0) { /* a NaN row: every output NaN, logsumexp NaN or +inf */
        for (size_t c = 0; !rc && c < Ev; c++)
            if (!(rc = special_out(enc + c * a->out_size, a->out_size, &entries[c], &a->out, a->rnd, a->ovf, VALUE_NAN, 0))) status[c] = 0;
        if (!rc && !(rc = special_out(lse_enc, a->lse_size, lse_entry, &a->lse, a->lrnd, a->lovf, kind, 0))) *lse_status = 0;
        return rc;
    }
    size_t n = (size_t)kept, *keys = w->keys;
    if (!a->tables || q->flagged_count || q->slots_hi || k->flagged_count || k->slots_hi || v->flagged_count || v->slots_hi)
        return reference_general(w, a, qr, kr0, vr0, n, a->p, 0, enc, entries, status, lse_enc, lse_entry, lse_status);
    w128 *scores = w->scores, *sums = w->sums;
    if (!n) {
        /* no kept key: every weight is 0, so each output is an exact zero (-0 rounding toward -inf),
           and logsumexp is log(0) = -inf */
        bigint bits, zero;
        big_init(&bits);
        big_init(&zero);
        int neg = a->rnd == ROUND_TOWARD_NEGATIVE && S > 0;
        for (size_t c = 0; !rc && c < Ev; c++) {
            if (!(rc = floatfmt_encode(&bits, &a->out, VALUE_FINITE, neg, &zero, 0, a->rnd, a->ovf)) && !(rc = entry_value(&entries[c], VALUE_FINITE, neg, &zero, 0))) {
                bytes_store(enc + c * a->out_size, a->out_size, &bits);
                status[c] = 0;
            }
        }
        if (!rc && !(rc = floatfmt_encode(&bits, &a->lse, VALUE_INFINITE, 1, &zero, 0, a->lrnd, a->lovf)) &&
            !(rc = entry_value(lse_entry, VALUE_INFINITE, 1, &zero, 0))) {
            bytes_store(lse_enc, a->lse_size, &bits);
            *lse_status = 0;
        }
        big_free(&bits);
        big_free(&zero);
        return rc;
    }
    /* scores at the unit 2**u: exact dot products of the main-range integers, shifted */
    int64_t shift = a->u - (q->exponent + k->exponent);
    const uint64_t *qs = q->slots + qr * E;
    for (size_t j = 0; j < n; j++) {
        const uint64_t *ks = k->slots + (kr0 + keys[j]) * E;
        w2 dot = {0}, p;
        for (size_t e = 0; e < E; e++) {
            w2_product(&p, qs[e], ks[e]);
            w2_add(&dot, &p);
        }
        scores[j] = s128_shr(w2_get(&dot), shift);
    }
    /* the extreme score: the largest logit's */
    w128 top = scores[0];
    for (size_t j = 1; j < n; j++)
        if (a->sign > 0 ? s128_less(top, scores[j]) : s128_less(scores[j], top)) top = scores[j];
    /* weights (Lemma D), and the row's sums */
    uint64_t Z[ATT_ZW] = {0}, pmax[3] = {0};
    memset(w->acc, 0, Ev * ATT_ACC * sizeof *w->acc);
    for (size_t c = 0; c < 2 * Ev; c++) sums[c] = w128_zero();
    int zmax = 0, any_capped = 0;
    for (size_t j = 0; j < n; j++) {
        uint64_t p[3] = {0, 0, (uint64_t)1 << (ATT_F - 128)};
        int z = 0;
        if (a->mask_float && w->capped[j]) {
            /* 133 or more below the row's largest mask value: 0 <= 2**F w < 1 */
            p[2] = 0;
            any_capped = 1;
        } else {
            if (a->sign && (rc = att_digits_mul(p, &z, a->sign > 0 ? w128_sub(top, scores[j]) : w128_sub(scores[j], top), a->tables))) return rc;
            if (a->mask_float && (rc = att_digits_mul(p, &z, ((w128 *)w->mdist)[j], a->mtables))) return rc;
        }
        if (z > zmax) zmax = z;
        if (p[2] > pmax[2] || (p[2] == pmax[2] && (p[1] > pmax[1] || (p[1] == pmax[1] && p[0] > pmax[0])))) memcpy(pmax, p, sizeof pmax);
        words_add(Z, ATT_ZW, p, 3);
        const uint64_t *vs = v->slots + (vr0 + keys[j]) * Ev;
        for (size_t c = 0; c < Ev; c++) {
            uint64_t x = vs[c], mag = x >> 63 ? ~x + 1 : x, t[4];
            if (!mag) continue;
            words_mul(t, p, 3, &mag, 1);
            if (x >> 63) {
                words_sub(w->acc + c * ATT_ACC, ATT_ACC, t, 4);
                sums[2 * c + 1] = w128_sub(sums[2 * c + 1], w128_make(mag, 0));
            } else {
                words_add(w->acc + c * ATT_ACC, ATT_ACC, t, 4);
                sums[2 * c] = w128_add(sums[2 * c], w128_make(mag, 0));
            }
        }
    }
    uint64_t h = zmax ? 3 * (uint64_t)zmax - 1 : (uint64_t)any_capped;
    /* the tables only where the row sum's width, n h units, is at most 2**-20 of the largest weight's lower end, as in
       the closed form (the largest score and the largest mask value on keys far apart can leave every weight a few
       units); else the weights measured from the largest logit's key */
    uint64_t top20[3];
    words_shr(top20, 3, pmax, 3, 20);
    if (!top20[1] && !top20[2] && top20[0] < (uint64_t)n * h)
        return reference_general(w, a, qr, kr0, vr0, n, a->p, 0, enc, entries, status, lse_enc, lse_entry, lse_status);
    /* groups of equal logits, for the exact cases: equal scores (all keys with sigma 0); with a float
       mask, equal (score, mask value) pairs for sigma = 1/sqrt(E), equal sigma s + b as rationals
       (num s 2**u + den b) for a rational sigma, equal b for sigma = 0 */
    keyed *order = w->order;
    dyad *lk = NULL;
    int rational = a->mask_float && a->den.n && a->sign;
    if (rational) {
        if (!(lk = malloc(n * sizeof *lk))) return EXACTREF_NOMEM;
        for (size_t j = 0; j < n; j++) dyad_init(&lk[j]);
        dyad t1, t2, nn, dd;
        dyad_init(&t1);
        dyad_init(&t2);
        dyad_init(&nn);
        dyad_init(&dd);
        rc = big_copy(&nn.m, &a->num);
        if (!rc) rc = big_copy(&dd.m, &a->den);
        for (size_t j = 0; !rc && j < n; j++) {
            int neg = w128_negative(scores[j]);
            w128 mag = neg ? w128_sub(w128_zero(), scores[j]) : scores[j];
            uint64_t mw[2] = {w128_lo(mag), w128_hi(mag)};
            if (!(rc = words_big(&t1.m, mw, 2, 0))) {
                t1.m.neg = t1.m.n ? neg : 0;
                t1.e = t1.m.n ? a->u : 0;
                if (!(rc = dyad_mul_exact(&t1, &t1, &nn)) && !(rc = dyad_mul_exact(&t2, &w->bias[j], &dd))) rc = dyad_add_exact(&lk[j], &t1, &t2);
            }
        }
        dyad_free(&t1);
        dyad_free(&t2);
        dyad_free(&nn);
        dyad_free(&dd);
    }
    for (size_t j = 0; j < n; j++) {
        order[j].s = a->sign && !rational ? scores[j] : w128_zero();
        order[j].j = j;
        order[j].key = !a->mask_float ? NULL : rational ? &lk[j] : &w->bias[j];
    }
    if (rc) {
        if (lk)
            for (size_t j = 0; j < n; j++) dyad_free(&lk[j]);
        free(lk);
        return rc;
    }
    qsort(order, n, sizeof *order, keyed_order);
    dyad lo, hi, nd, zd;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&nd);
    dyad_init(&zd);
    bigint bits, zlo, zhi;
    big_init(&bits);
    big_init(&zlo);
    big_init(&zhi);
    rc = words_big(&zlo, Z, ATT_ZW, 0);
    uint64_t Zh[ATT_ZW], nh = (uint64_t)n * h;
    memcpy(Zh, Z, sizeof Zh);
    words_add(Zh, ATT_ZW, &nh, 1);
    if (!rc) rc = words_big(&zhi, Zh, ATT_ZW, 0);
    size_t itemsize = a->out_size;
    for (size_t c = 0; !rc && c < Ev; c++) {
        const uint64_t *vs0 = v->slots + vr0 * Ev + c;
        /* the exact case: every group's mean equal (sum_g |h| == sum_h |g|) */
        w128 g0 = w128_zero();
        size_t s0 = 0, at = 0;
        int exact = 1;
        while (exact && at < n) {
            w128 gs = w128_zero();
            size_t start = at;
            do {
                gs = w128_add(gs, w128_make(vs0[keys[order[at].j] * Ev], word_signed(vs0[keys[order[at].j] * Ev]) < 0 ? UINT64_MAX : 0));
                at++;
            } while (at < n && keyed_same(&order[at - 1], &order[at]));
            size_t size = at - start;
            if (!s0) {
                g0 = gs;
                s0 = size;
            } else {
                /* g0 size == gs s0, both products below 2**100 */
                int n0 = w128_negative(g0), n1 = w128_negative(gs);
                w128 a0 = n0 ? s128_negate(g0) : g0, a1 = n1 ? s128_negate(gs) : gs;
                uint64_t r0[3], r1[3], x0[2] = {w128_lo(a0), w128_hi(a0)}, x1[2] = {w128_lo(a1), w128_hi(a1)}, m0 = size, m1 = s0;
                words_mul(r0, x0, 2, &m0, 1);
                words_mul(r1, x1, 2, &m1, 1);
                int z0 = !(r0[0] | r0[1] | r0[2]), z1 = !(r1[0] | r1[1] | r1[2]);
                exact = (z0 && z1) || (n0 == n1 && !memcmp(r0, r1, sizeof r0));
            }
        }
        entry *x = &entries[c];
        if (exact) {
            w128 total = w128_add(sums[2 * c], sums[2 * c + 1]);
            int tneg = w128_negative(total);
            w128 tmag = tneg ? s128_negate(total) : total;
            uint64_t tw[2] = {w128_lo(tmag), w128_hi(tmag)};
            if (!(rc = words_big(&nd.m, tw, 2, 0))) {
                nd.m.neg = nd.m.n ? tneg : 0;
                nd.e = nd.m.n ? v->exponent : 0;
                rc = exact_mean(&bits, x, &a->out, a->rnd, a->ovf, &nd, n);
            }
            if (!rc) {
                bytes_store(enc + c * itemsize, itemsize, &bits);
                status[c] = 0;
            }
            continue;
        }
        /* N 2**F in [Np + h Vneg, Np + h Vpos]; o = N / Z, each end over the end of Z that makes
           it smallest (largest) */
        uint64_t lo_w[ATT_ACC], hi_w[ATT_ACC], t[4];
        memcpy(lo_w, w->acc + c * ATT_ACC, sizeof lo_w);
        memcpy(hi_w, lo_w, sizeof hi_w);
        w128 vp = sums[2 * c], vn = s128_negate(sums[2 * c + 1]);
        uint64_t hw = h, vpw[2] = {w128_lo(vp), w128_hi(vp)}, vnw[2] = {w128_lo(vn), w128_hi(vn)};
        words_mul(t, vpw, 2, &hw, 1);
        words_add(hi_w, ATT_ACC, t, 3);
        words_mul(t, vnw, 2, &hw, 1);
        words_sub(lo_w, ATT_ACC, t, 3);
        int none = 1;
        if (!(rc = words_big(&nd.m, lo_w, ATT_ACC, 1))) {
            nd.e = nd.m.n ? v->exponent : 0;
            if (!(rc = big_copy(&zd.m, nd.m.neg ? &zlo : &zhi))) {
                zd.e = 0;
                rc = dyad_div(&lo, &nd, &zd, a->p, 0);
            }
        }
        if (!rc && !(rc = words_big(&nd.m, hi_w, ATT_ACC, 1))) {
            nd.e = nd.m.n ? v->exponent : 0;
            if (!(rc = big_copy(&zd.m, nd.m.neg ? &zhi : &zlo))) {
                zd.e = 0;
                rc = dyad_div(&hi, &nd, &zd, a->p, 1);
            }
        }
        if (!rc) rc = decide_rounding(&bits, x, &none, &a->out, a->rnd, a->ovf, &lo, &hi, 0);
        if (!rc) {
            status[c] = (unsigned char)none;
            if (!none) bytes_store(enc + c * itemsize, itemsize, &bits);
        }
    }
    /* logsumexp = m + log(Z e**-m), Z e**-m in [Zp, Zp + n h] 2**-F; one kept key is left to the
       caller (its exact cases) */
    *lse_status = 1;
    if (!rc && n > 1 && zlo.n) {
        dyad mlo, mhi, llo, lhi, t;
        dyad_init(&mlo);
        dyad_init(&mhi);
        dyad_init(&llo);
        dyad_init(&lhi);
        dyad_init(&t);
        bigint fe;
        big_init(&fe);
        rc = big_set_si(&fe, -ATT_F);
        if (!rc) rc = a->sign ? top_logit(&mlo, &mhi, a, top) : dyad_set_si(&mlo, 0, 0);
        if (!rc && !a->sign) rc = dyad_set_si(&mhi, 0, 0);
        /* with a float mask the weights are relative to m' = sigma s_top U + b_top */
        if (!rc && a->mask_float && !(rc = dyad_add(&mlo, &mlo, &w->btop, a->p, 0))) rc = dyad_add(&mhi, &mhi, &w->btop, a->p, 1);
        if (!rc) rc = log_bounds(&llo, &t, &zlo, &fe, a->p);
        if (!rc) rc = log_bounds(&t, &lhi, &zhi, &fe, a->p);
        if (!rc) rc = dyad_add(&lo, &mlo, &llo, a->p, 0);
        if (!rc) rc = dyad_add(&hi, &mhi, &lhi, a->p, 1);
        int none = 1;
        if (!rc) rc = decide_rounding(&bits, lse_entry, &none, &a->lse, a->lrnd, a->lovf, &lo, &hi, 0);
        if (!rc && !none) {
            bytes_store(lse_enc, a->lse_size, &bits);
            *lse_status = 0;
        }
        dyad_free(&mlo);
        dyad_free(&mhi);
        dyad_free(&llo);
        dyad_free(&lhi);
        dyad_free(&t);
        big_free(&fe);
    }
    if (lk)
        for (size_t j = 0; j < n; j++) dyad_free(&lk[j]);
    free(lk);
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&nd);
    dyad_free(&zd);
    big_free(&bits);
    big_free(&zlo);
    big_free(&zhi);
    return rc;
}

/* logsumexp of a row with one kept key: its logit sigma s + b, exact when sigma is rational (0 included) or the score
   is 0, rounded once (an exact 0 as +0); otherwise left to refinement (*lse_status stays 1) */
static int lse_one_key(const att_row *w, const att_call *a, size_t qr, size_t kr0, unsigned char *lse_enc, entry *lse_entry, unsigned char *lse_status) {
    product_sums ps;
    product_sums_init(&ps);
    dyad s, x, t;
    dyad_init(&s);
    dyad_init(&x);
    dyad_init(&t);
    bigint bits, num, den;
    big_init(&bits);
    big_init(&num);
    big_init(&den);
    int rc = product_element(&ps, a->q, qr, a->k, kr0 + w->keys[0], PRODUCT_SUM, 0, 0, NULL), exact = 0;
    if (!rc) rc = dyad_from_big(&s, &ps.s, ps.e);
    if (!rc && a->den.n) {
        /* x = num s + den b, the value x / den */
        dyad nn, dd;
        dyad_init(&nn);
        dyad_init(&dd);
        if (!(rc = big_copy(&nn.m, &a->num)) && !(rc = big_copy(&dd.m, &a->den)) && !(rc = dyad_mul_exact(&x, &nn, &s)) && a->mask_float &&
            !(rc = dyad_mul_exact(&t, &dd, &w->bias[0])))
            rc = dyad_add_exact(&x, &x, &t);
        if (!rc && !(rc = big_copy(&num, &x.m))) rc = big_copy(&den, &a->den);
        num.neg = 0;
        if (!rc && x.m.n) rc = x.e >= 0 ? big_shl(&num, &num, x.e) : big_shl(&den, &den, -x.e);
        dyad_free(&nn);
        dyad_free(&dd);
        exact = 1;
    } else if (!rc && !dyad_sign(&s)) {
        /* sigma = 1/sqrt(E) and the score 0: the value b */
        if (a->mask_float) rc = dyad_copy(&x, &w->bias[0]);
        if (!rc && !(rc = big_copy(&num, &x.m)) && !(rc = big_set_word(&den, 1, 0))) {
            num.neg = 0;
            if (x.m.n) rc = x.e >= 0 ? big_shl(&num, &num, x.e) : big_shl(&den, &den, -x.e);
        }
        exact = 1;
    }
    if (!rc && exact) {
        int neg = x.m.n && x.m.neg;
        if (!num.n) {
            if (!(rc = floatfmt_encode(&bits, &a->lse, VALUE_FINITE, 0, &num, 0, a->lrnd, a->lovf))) rc = entry_value(lse_entry, VALUE_FINITE, 0, &num, 0);
        } else if (!(rc = floatfmt_encode_ratio(&bits, &a->lse, neg, &num, &den, a->lrnd, a->lovf))) {
            rc = entry_ratio(lse_entry, neg, &num, &den);
        }
        if (!rc) {
            bytes_store(lse_enc, a->lse_size, &bits);
            *lse_status = 0;
        }
    }
    product_sums_free(&ps);
    dyad_free(&s);
    dyad_free(&x);
    dyad_free(&t);
    big_free(&bits);
    big_free(&num);
    big_free(&den);
    return rc;
}

/* One row's outputs and logsumexp (proofs/scaled_dot_product_attention.md, The reference output and logsumexp): query
   row qr (query index i) against key rows kr0.. and value rows vr0..; the Ev outputs' encodings (itemsize bytes each),
   entries and status (0 decided, 1 refine), and logsumexp's encoding, entry and status. The first round, then
   refinement of what it leaves undecided, from twice the first precision, doubling to the cap; a part still undecided
   there keeps status 1. */
int att_reference_row(att_row *w, const att_call *a, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, unsigned char *enc, entry *entries,
                      unsigned char *status, unsigned char *lse_enc, entry *lse_entry, unsigned char *lse_status) {
    int kept, rc = reference_first(w, a, qr, i, moff, mstep, kr0, vr0, enc, entries, status, lse_enc, lse_entry, lse_status, &kept);
    if (rc || kept < 0) return rc;
    size_t n = (size_t)kept;
    if (n == 1 && *lse_status) rc = lse_one_key(w, a, qr, kr0, lse_enc, lse_entry, lse_status);
    for (int64_t p = a->p; !rc && p < a->cap;) {
        int open = *lse_status;
        for (size_t c = 0; !open && c < a->Ev; c++) open = status[c];
        if (!open) break;
        p = 2 * p < a->cap ? 2 * p : a->cap;
        rc = reference_general(w, a, qr, kr0, vr0, n, p, 1, enc, entries, status, lse_enc, lse_entry, lse_status);
    }
    return rc;
}
