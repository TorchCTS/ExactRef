/* Enclosures of ln 2, exp, log, pi, cos and sin (internal), each proven in proofs/foundations.md:
   ln 2 by the series 2 atanh(1/3) in fixed point (Lemma Q); exp by reduction modulo ln 2, a Taylor
   series of positive terms in fixed point and squarings (Lemma X); log by
   2 atanh((y - 1) / (y + 1)) in fixed point and a multiple of ln 2 (Lemma Y); pi by Machin's
   formula in fixed point (Lemma M); cos and sin by reduction modulo pi / 2 and their alternating
   Taylor series in fixed point (Lemma Z). A lower end is computed with floors and roundings toward
   -inf throughout, an upper end with ceilings and roundings toward +inf, so the intervals contain
   the exact values; exp's and log's are at most 2**(3 - p) wide relative to their value (2**(2 - p)
   for ln 2), pi's at most 2**(4 - p) wide, cos's 2**(3 - p) and sin's 2**(4 - p) min(1, |x|). */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

/* A precision at which bigfloat.c's operations on these operands are exact. */
#define EXACT ((int64_t)1 << 40)

/* ln 2 lies in [LN2_LO, LN2_HI] / 2**4160: Lemma Q's sums L and U + 1 at W = 4160, computed once with exact
   integers; the tests check them against MPFR. */
#define LN2_BITS 4160
static const uint64_t LN2_LO[66] = {
    0x85db6ab03a49ba7d, 0x175eb4afc8daadd8, 0xf07afff3a892374e, 0x8f6826250dea891e,
    0xcecb72f19c38339d, 0x5f6f7cebac9f45ae, 0x6c472096e76115c0, 0x972cd18bfbbd9d42,
    0x0ab111bbbd67c724, 0x473826fda0c238b9, 0x61c1696dd24aaebd, 0x156e0c292413d5e3,
    0x95184460dc4e7487, 0xd7622658901e646a, 0xef2f0ce2d7373958, 0x2ac5b61cc4e9207c,
    0x57339ba2beba7d05, 0x0060e49908391a0c, 0x621363196af50302, 0x05c128d53d0bd2f9,
    0x36e02b20cee886b9, 0x0bbb16faf3d949f2, 0x422183edc9942109, 0x5e9222b88c66d3c5,
    0x61affd446b1ca3cf, 0x268a5c1f9538b982, 0x8d6f5177fbcf0755, 0xa17293d1228a4ef9,
    0x44a02554731cdc8e, 0x96d4e6d330af889b, 0x5570b6c68f969834, 0x7598a1951ae273ee,
    0x4d162db3b365853d, 0x5f50b5185064c18b, 0x078f735d1b2db31b, 0xae313cdb6c606cb1,
    0x955d5179b1e17b9d, 0x0c480a5417350d2c, 0x074db6015cfe7aa3, 0x6a9c7f8a5e148e82,
    0x25669b333564a337, 0x4c1a1e0bd1d6095d, 0xcccc4e659393514c, 0xc943e732b479cd33,
    0x17460775db8990e5, 0x7d2e23de1400b396, 0xee569d6dfc1efa15, 0x610d30f88fe551a2,
    0x07f4ca11fb5bfb90, 0xda2d97c50f3fd5c6, 0x655fa1872f20e3a2, 0xf5dfa6bd38303248,
    0x72ce87b19d6548ca, 0x256fa0ec7657f74b, 0xb9ea9bc3b136603b, 0x1acbda11317c387e,
    0x3e96ca16224ae8c5, 0x27573b291169b825, 0xed2eae35c1382144, 0x559552fb4afa1b10,
    0xe7b876206debac98, 0x8a0d175b8baafa2b, 0x40f343267298b62d, 0xc9e3b39803f2f6af,
    0xb17217f7d1cf79ab, 0x0000000000000000,
};
static const uint64_t LN2_HI[66] = {
    0x85db6ab03a49bf9f, 0x175eb4afc8daadd8, 0xf07afff3a892374e, 0x8f6826250dea891e,
    0xcecb72f19c38339d, 0x5f6f7cebac9f45ae, 0x6c472096e76115c0, 0x972cd18bfbbd9d42,
    0x0ab111bbbd67c724, 0x473826fda0c238b9, 0x61c1696dd24aaebd, 0x156e0c292413d5e3,
    0x95184460dc4e7487, 0xd7622658901e646a, 0xef2f0ce2d7373958, 0x2ac5b61cc4e9207c,
    0x57339ba2beba7d05, 0x0060e49908391a0c, 0x621363196af50302, 0x05c128d53d0bd2f9,
    0x36e02b20cee886b9, 0x0bbb16faf3d949f2, 0x422183edc9942109, 0x5e9222b88c66d3c5,
    0x61affd446b1ca3cf, 0x268a5c1f9538b982, 0x8d6f5177fbcf0755, 0xa17293d1228a4ef9,
    0x44a02554731cdc8e, 0x96d4e6d330af889b, 0x5570b6c68f969834, 0x7598a1951ae273ee,
    0x4d162db3b365853d, 0x5f50b5185064c18b, 0x078f735d1b2db31b, 0xae313cdb6c606cb1,
    0x955d5179b1e17b9d, 0x0c480a5417350d2c, 0x074db6015cfe7aa3, 0x6a9c7f8a5e148e82,
    0x25669b333564a337, 0x4c1a1e0bd1d6095d, 0xcccc4e659393514c, 0xc943e732b479cd33,
    0x17460775db8990e5, 0x7d2e23de1400b396, 0xee569d6dfc1efa15, 0x610d30f88fe551a2,
    0x07f4ca11fb5bfb90, 0xda2d97c50f3fd5c6, 0x655fa1872f20e3a2, 0xf5dfa6bd38303248,
    0x72ce87b19d6548ca, 0x256fa0ec7657f74b, 0xb9ea9bc3b136603b, 0x1acbda11317c387e,
    0x3e96ca16224ae8c5, 0x27573b291169b825, 0xed2eae35c1382144, 0x559552fb4afa1b10,
    0xe7b876206debac98, 0x8a0d175b8baafa2b, 0x40f343267298b62d, 0xc9e3b39803f2f6af,
    0xb17217f7d1cf79ab, 0x0000000000000000,
};

/* Lemma Q: ln 2 in [lo, hi], at most p bits each. */
int ln2_bounds(dyad *lo, dyad *hi, int64_t p) {
    dyad l, h;
    dyad_init(&l);
    dyad_init(&h);
    size_t n = sizeof LN2_LO / sizeof *LN2_LO;
    int rc = 0;
    if (p <= LN2_BITS - 64) {
        if (big_reserve(&l.m, n) || big_reserve(&h.m, n)) {
            rc = EXACTREF_NOMEM;
        } else {
            memcpy(l.m.w, LN2_LO, sizeof LN2_LO);
            memcpy(h.m.w, LN2_HI, sizeof LN2_HI);
            l.m.n = h.m.n = n;
            l.e = h.e = -LN2_BITS;
        }
    } else {
        /* the series at W = p + 8 + bits(p + 72) */
        int64_t W0 = p + 8, W = W0 + word_bits((uint64_t)(W0 + 64));
        int64_t odd = (W * 10000 + 15848) / 15849, N = odd / 2 + 1; /* 3**(2N + 1) > 2**W */
        size_t m = (size_t)((W + 1) / 64 + 2);
        uint64_t *a = calloc(5 * m, sizeof *a);
        if (!a) return EXACTREF_NOMEM;
        uint64_t *b = a + m, *t = a + 2 * m, *sum_lo = a + 3 * m, *sum_hi = a + 4 * m;
        a[(W + 1) / 64] = (uint64_t)1 << ((W + 1) % 64);
        memcpy(b, a, m * sizeof *b);
        /* a = floor(2**(W + 1) / 3), b its ceiling */
        if (nat_div_word(a, a, m, 3)) {
            memcpy(b, a, m * sizeof *b);
            nat_add_word(b, 1, m);
        } else {
            memcpy(b, a, m * sizeof *b);
        }
        for (int64_t j = 0; j < N; j++) {
            uint64_t d = (uint64_t)(2 * j + 1);
            nat_div_word(t, a, m, d); /* floor(a_j / (2j + 1)) */
            nat_add(sum_lo, sum_lo, t, m, 0);
            memcpy(t, b, m * sizeof *t); /* ceil(b_j / (2j + 1)) */
            nat_add_word(t, d - 1, m);
            nat_div_word(t, t, m, d);
            nat_add(sum_hi, sum_hi, t, m, 0);
            nat_div_word(a, a, m, 9);
            nat_add_word(b, 8, m);
            nat_div_word(b, b, m, 9);
        }
        nat_add_word(sum_hi, 1, m); /* the rest is below 3**-(2N + 1) < 2**-W */
        if (big_reserve(&l.m, m) || big_reserve(&h.m, m)) {
            rc = EXACTREF_NOMEM;
        } else {
            memcpy(l.m.w, sum_lo, m * sizeof *sum_lo);
            memcpy(h.m.w, sum_hi, m * sizeof *sum_hi);
            l.m.n = h.m.n = m;
            l.e = h.e = -W;
        }
        free(a);
    }
    if (!rc) {
        big_trim(&l.m);
        big_trim(&h.m);
        rc = dyad_rnd(lo, &l, p, 0);
        if (!rc) rc = dyad_rnd(hi, &h, p, 1);
    }
    dyad_free(&l);
    dyad_free(&h);
    return rc;
}

static int64_t isqrt64(int64_t v) {
    int64_t r = 0;
    while ((r + 1) * (r + 1) <= v) r++;
    return r;
}

/* Lemma X, steps 4 and 5: E * 2**-F bounds exp(t) from below (!up) or above (up), for a dyadic
   |t| < 1, with s = t / 2**K and N Taylor terms whose rest is at most 2**-(F + 1). */
static int exp_fixed(bigint *E, const dyad *t, int64_t K, int64_t N, int64_t F, int up) {
    int negative = dyad_sign(t) < 0;
    int side = negative ? !up : up; /* the side of exp(|s|) wanted: 1 / exp(|s|) reverses it */
    dyad a;
    dyad_init(&a);
    bigint A, term, sum, one;
    big_init(&A);
    big_init(&term);
    big_init(&sum);
    big_init(&one);
    int rc = dyad_copy(&a, t);
    if (!rc) {
        a.m.neg = 0;
        if (!(rc = dyad_shift(&a, -K)))           /* |s| = |t| / 2**K */
            rc = dyad_to_fixed(&A, &a, F, side); /* A: |s| 2**F, rounded to the side */
    }
    if (!rc && !(rc = big_set_word(&one, 1, 0)) && !(rc = big_shl(&term, &one, F))) rc = big_shl(&sum, &one, F);
    for (int64_t j = 1; !rc && j < N; j++) {
        /* term_j = term_(j-1) A / (j 2**F), rounded to the side */
        if ((rc = big_mul(&term, &term, &A)) || (rc = big_shr(&term, &term, F, side)) || (rc = big_div_word(&term, &term, (uint64_t)j, side))) break;
        rc = big_add(&sum, &sum, &term);
    }
    if (!rc && side) rc = big_add(&sum, &sum, &one); /* the rest: under half a unit */
    if (!rc) {
        if (negative) {
            /* exp(s) = 1 / exp(|s|): 2**(2F) / sum, rounded to the wanted end */
            bigint num;
            big_init(&num);
            if (!(rc = big_shl(&num, &one, 2 * F))) rc = big_div(E, &num, &sum, up);
            big_free(&num);
        } else {
            rc = big_shl(E, &sum, 0);
        }
    }
    /* square K times: exp(s)**(2**K) = exp(t) */
    for (int64_t k = 0; !rc && k < K; k++) {
        if (!(rc = big_mul(E, E, E))) rc = big_shr(E, E, F, up);
    }
    dyad_free(&a);
    big_free(&A);
    big_free(&term);
    big_free(&sum);
    big_free(&one);
    return rc;
}

/* Lemma X: exp(x) in [lo * 2**n, hi * 2**n], lo and hi at most p bits. */
int exp_bounds(dyad *lo, dyad *hi, bigint *n, const dyad *x, int64_t p) {
    int sign = dyad_sign(x);
    int64_t top = x->e + big_bits(&x->m);
    n->n = 0;
    n->neg = 0;
    if (!sign) {
        int rc = dyad_set_si(lo, 1, 0);
        return rc ? rc : dyad_set_si(hi, 1, 0);
    }
    if (top <= -(p + 4)) {
        /* step 1: |x| < 2**-(p + 4), so exp(x) lies in [1, 1 + 2**-(p + 2)] for x > 0 and in
           [1 - 2**-(p + 3), 1] for x < 0 */
        dyad one, end;
        dyad_init(&one);
        dyad_init(&end);
        int rc = dyad_set_si(&one, 1, 0);
        if (!rc) rc = dyad_set_si(&end, sign, sign > 0 ? -(p + 2) : -(p + 3));
        if (!rc) rc = dyad_add(&end, &one, &end, EXACT, 0);
        if (!rc) rc = sign > 0 ? dyad_copy(lo, &one) : dyad_rnd(lo, &end, p, 0);
        if (!rc) rc = sign > 0 ? dyad_rnd(hi, &end, p, 1) : dyad_copy(hi, &one);
        dyad_free(&one);
        dyad_free(&end);
        return rc;
    }
    int64_t T = top > 0 ? top : 0, K = isqrt64(p) + 1, F;
    if (K < 2) K = 2;
    F = p + K + 26;
    /* N: K N + sum_{j=2..N} floor(log2 j) >= F + 2, so the rest 2**(1 - K N) / N! <= 2**-(F + 1) */
    int64_t N = 1, total = K;
    while (total < F + 2) {
        N++;
        total += K + word_bits((uint64_t)N) - 1;
    }
    dyad l2lo, l2hi, q, half, tlo, thi, nd, prod, end;
    dyad_init(&l2lo);
    dyad_init(&l2hi);
    dyad_init(&q);
    dyad_init(&half);
    dyad_init(&tlo);
    dyad_init(&thi);
    dyad_init(&nd);
    dyad_init(&prod);
    dyad_init(&end);
    /* step 2: ln 2 to W = F + T + 8 bits, and n near x / ln 2 */
    int rc = ln2_bounds(&l2lo, &l2hi, F + T + 8);
    if (!rc) rc = dyad_div(&q, x, &l2lo, T + 16, 0);
    if (!rc) rc = dyad_set_si(&half, 1, -1);
    if (!rc) rc = dyad_add(&q, &q, &half, EXACT, 0);
    if (!rc) rc = dyad_to_fixed(n, &q, 0, 0);
    /* step 3: t = x - n ln 2, exactly at each end of ln 2's enclosure */
    if (!rc && !(rc = big_shl(&nd.m, n, 0))) {
        nd.e = 0;
        int positive = !n->neg;
        /* the larger multiple of ln 2 gives the smaller t */
        if (!(rc = dyad_mul(&prod, &nd, positive ? &l2hi : &l2lo, EXACT, 0))) {
            prod.m.neg = prod.m.n ? !prod.m.neg : 0;
            rc = dyad_add(&tlo, x, &prod, EXACT, 0);
        }
        if (!rc && !(rc = dyad_mul(&prod, &nd, positive ? &l2lo : &l2hi, EXACT, 0))) {
            prod.m.neg = prod.m.n ? !prod.m.neg : 0;
            rc = dyad_add(&thi, x, &prod, EXACT, 0);
        }
    }
    /* |t| < 1: its top bit at most 0 */
    if (!rc && ((dyad_sign(&tlo) && tlo.e + big_bits(&tlo.m) > 0) || (dyad_sign(&thi) && thi.e + big_bits(&thi.m) > 0))) rc = EXACTREF_INTERNAL;
    /* steps 4 to 6 at each end */
    if (!rc && !(rc = exp_fixed(&end.m, &tlo, K, N, F, 0))) {
        end.e = end.m.n ? -F : 0;
        rc = dyad_rnd(lo, &end, p, 0);
    }
    if (!rc && !(rc = exp_fixed(&end.m, &thi, K, N, F, 1))) {
        end.e = end.m.n ? -F : 0;
        rc = dyad_rnd(hi, &end, p, 1);
    }
    dyad_free(&l2lo);
    dyad_free(&l2hi);
    dyad_free(&q);
    dyad_free(&half);
    dyad_free(&tlo);
    dyad_free(&thi);
    dyad_free(&nd);
    dyad_free(&prod);
    dyad_free(&end);
    return rc;
}

/* Lemma M, one series: with a_j and b_j the floor and ceiling of 2**W / k**(2j + 1), sums over m words of N_k
   terms (d below 10000 log2 k): lp and ln what L_k adds and subtracts (floors of a_j / (2j + 1) for even j, ceilings
   of b_j / (2j + 1) for odd j), up and un U_k's (the ceilings and floors exchanged) */
static void atan_sums(uint64_t *lp, uint64_t *ln, uint64_t *up, uint64_t *un, uint64_t *a, uint64_t *b, uint64_t *t, size_t m,
                      int64_t W, uint64_t k, int64_t d) {
    int64_t c = ((W + 2) * 10000 + d - 1) / d, N = c / 2 + 1; /* k**(2N + 1) > 2**(W + 2) */
    memset(a, 0, m * sizeof *a);
    a[W / 64] = (uint64_t)1 << (W % 64);
    int inexact = nat_div_word(a, a, m, k) != 0;
    memcpy(b, a, m * sizeof *b);
    if (inexact) nat_add_word(b, 1, m);
    for (int64_t j = 0; j < N; j++) {
        uint64_t q = (uint64_t)(2 * j + 1);
        nat_div_word(t, a, m, q); /* floor(a_j / (2j + 1)) */
        nat_add(j % 2 ? un : lp, j % 2 ? un : lp, t, m, 0);
        if (nat_div_word(t, b, m, q)) nat_add_word(t, 1, m); /* ceil(b_j / (2j + 1)) */
        nat_add(j % 2 ? ln : up, j % 2 ? ln : up, t, m, 0);
        nat_div_word(a, a, m, k * k);
        if (nat_div_word(b, b, m, k * k)) nat_add_word(b, 1, m);
    }
}

/* Lemma M: pi in [lo, hi], at most p bits each, from 16 atan(1/5) - 4 atan(1/239) at W = p + 16 + bits(p). */
int pi_bounds(dyad *lo, dyad *hi, int64_t p) {
    int64_t W = p + 16 + word_bits((uint64_t)p);
    size_t m = (size_t)((W + 4) / 64 + 2);
    uint64_t *buf = calloc(13 * m, sizeof *buf);
    if (!buf) return EXACTREF_NOMEM;
    uint64_t *s5 = buf, *s239 = buf + 4 * m, *a = buf + 8 * m, *b = buf + 9 * m, *t = buf + 10 * m, *x = buf + 11 * m, *y = buf + 12 * m;
    /* s[0 .. 3]: L_k's added and subtracted terms, then U_k's */
    atan_sums(s5, s5 + m, s5 + 2 * m, s5 + 3 * m, a, b, t, m, W, 5, 23219);
    atan_sums(s239, s239 + m, s239 + 2 * m, s239 + 3 * m, a, b, t, m, W, 239, 79009);
    dyad end;
    dyad_init(&end);
    int rc = 0;
    for (int side = 0; !rc && side < 2; side++) {
        /* lower: 16 (L_5 - 1) - 4 (U_239 + 1); upper: 16 (U_5 + 1) - 4 (L_239 - 1) */
        const uint64_t *k5 = s5 + (side ? 2 : 0) * m, *k239 = s239 + (side ? 0 : 2) * m;
        nat_mul_word(x, k5, m, 16, 0); /* the terms added */
        nat_addmul_word(x, k239 + m, m, 4);
        nat_mul_word(y, k5 + m, m, 16, 0); /* and subtracted */
        nat_addmul_word(y, k239, m, 4);
        if (side)
            nat_add_word(x, 20, m);
        else
            nat_add_word(y, 20, m);
        nat_sub(x, x, y, m, 0);
        if (!(rc = big_reserve(&end.m, m))) {
            memcpy(end.m.w, x, m * sizeof *x);
            end.m.n = m;
            end.m.neg = 0;
            end.e = end.m.n ? -W : 0;
            big_trim(&end.m);
            rc = dyad_rnd(side ? hi : lo, &end, p, side);
        }
    }
    dyad_free(&end);
    free(buf);
    return rc;
}

/* sum += v, or sum -= v (negative); v >= 0 */
static int add_signed(bigint *sum, bigint *v, int negative) {
    if (negative && v->n) v->neg = 1;
    int rc = big_add(sum, sum, v);
    v->neg = 0;
    return rc;
}

/* Lemma Z, step 4: 2**G cos(t0) in [c[0], c[1]] and 2**G sin(|t0|) in [s[0], s[1]], for a dyadic |t0| < 1, by the
   Taylor series in fixed point: l_k and h_k bound 2**G |t0|**k / k! from below and above, through the first h_K <= 1,
   and each sum moves one unit outward for the rest. */
static int cos_sin_fixed(bigint *c, bigint *s, const dyad *t0, int64_t G) {
    dyad a;
    bigint Alo, Ahi, l, h, one;
    dyad_init(&a);
    big_init(&Alo);
    big_init(&Ahi);
    big_init(&l);
    big_init(&h);
    big_init(&one);
    int rc = dyad_abs(&a, t0);
    if (!rc) rc = dyad_to_fixed(&Alo, &a, G, 0);
    if (!rc) rc = dyad_to_fixed(&Ahi, &a, G, 1);
    if (!rc && !(rc = big_set_word(&one, 1, 0)) && !(rc = big_shl(&l, &one, G)) && !(rc = big_shl(&h, &one, G)) && !(rc = big_copy(&c[0], &l)))
        rc = big_copy(&c[1], &l);
    if (!rc) {
        s[0].n = s[1].n = 0;
        s[0].neg = s[1].neg = 0;
    }
    for (int64_t k = 1; !rc; k++) {
        if (k > G + 2) { /* the lemma ends the terms by k = G */
            rc = EXACTREF_INTERNAL;
            break;
        }
        if ((rc = big_mul(&l, &l, &Alo)) || (rc = big_shr(&l, &l, G, 0)) || (rc = big_div_word(&l, &l, (uint64_t)k, 0))) break;
        if ((rc = big_mul(&h, &h, &Ahi)) || (rc = big_shr(&h, &h, G, 1)) || (rc = big_div_word(&h, &h, (uint64_t)k, 1))) break;
        bigint *sum = k % 2 ? s : c;
        int minus = k % 4 >= 2; /* the signs + - - + of k = 0, 1, 2, 3 modulo 4 */
        if ((rc = add_signed(&sum[0], minus ? &h : &l, minus)) || (rc = add_signed(&sum[1], minus ? &l : &h, minus))) break;
        if (big_bits(&h) <= 1) break; /* h_k <= 1: every term left out is below one unit */
    }
    if (!rc && !(rc = add_signed(&c[0], &one, 1)) && !(rc = add_signed(&c[1], &one, 0)) && !(rc = add_signed(&s[0], &one, 1)))
        rc = add_signed(&s[1], &one, 0);
    dyad_free(&a);
    big_free(&Alo);
    big_free(&Ahi);
    big_free(&l);
    big_free(&h);
    big_free(&one);
    return rc;
}

/* Lemma Z: cos x in [c[0], c[1]] and sin x in [s[0], s[1]], at most p bits each. */
int cos_sin_bounds(dyad *c, dyad *s, const dyad *x, int64_t p) {
    int sign = dyad_sign(x), rc = 0;
    if (!sign) {
        if ((rc = dyad_set_si(&c[0], 1, 0)) || (rc = dyad_set_si(&c[1], 1, 0)) || (rc = dyad_set_si(&s[0], 0, 0))) return rc;
        return dyad_set_si(&s[1], 0, 0);
    }
    int64_t top = x->e + big_bits(&x->m);
    if (top <= -(p + 4)) {
        /* step 1: cos x in [1 - 2**-(2p + 9), 1], sin x between x and x (1 - 2**-(2p + 8)) */
        dyad one, end;
        dyad_init(&one);
        dyad_init(&end);
        rc = dyad_set_si(&one, 1, 0);
        if (!rc) rc = dyad_set_si(&end, -1, -(2 * p + 9));
        if (!rc) rc = dyad_add_exact(&end, &one, &end);
        if (!rc) rc = dyad_rnd(&c[0], &end, p, 0);
        if (!rc) rc = dyad_copy(&c[1], &one);
        if (!rc && !(rc = dyad_neg(&end, x)) && !(rc = dyad_shift(&end, -(2 * p + 8)))) {
            rc = dyad_add_exact(&end, x, &end);
        }
        if (!rc) rc = dyad_rnd(&s[0], sign > 0 ? &end : x, p, 0);
        if (!rc) rc = dyad_rnd(&s[1], sign > 0 ? x : &end, p, 1);
        dyad_free(&one);
        dyad_free(&end);
        return rc;
    }
    int64_t T = top > 0 ? top : 0, G = p + 16 + word_bits((uint64_t)p) + (top < 0 ? -top : 0);
    dyad pilo, pihi, q, half, tlo, thi, nd, prod, slack, end[4];
    bigint n, fc[2], fs[2];
    dyad_init(&pilo);
    dyad_init(&pihi);
    dyad_init(&q);
    dyad_init(&half);
    dyad_init(&tlo);
    dyad_init(&thi);
    dyad_init(&nd);
    dyad_init(&prod);
    dyad_init(&slack);
    for (int k = 0; k < 4; k++) dyad_init(&end[k]);
    big_init(&n);
    for (int k = 0; k < 2; k++) {
        big_init(&fc[k]);
        big_init(&fs[k]);
    }
    if (top >= 0) {
        /* step 2: pi to W = G + T + 8 bits, and n near 2x / pi */
        rc = pi_bounds(&pilo, &pihi, G + T + 8);
        if (!rc && !(rc = dyad_copy(&q, x)) && !(rc = dyad_shift(&q, 1))) {
            rc = dyad_div(&q, &q, &pilo, T + 16, 0);
        }
        if (!rc) rc = dyad_set_si(&half, 1, -1);
        if (!rc) rc = dyad_add(&q, &q, &half, EXACT, 0);
        if (!rc) rc = dyad_to_fixed(&n, &q, 0, 0);
    }
    /* step 3: t = x - n pi / 2, exactly at each end of pi's enclosure (t = x for n = 0) */
    if (!rc) rc = dyad_copy(&tlo, x);
    if (!rc) rc = dyad_copy(&thi, x);
    if (!rc && n.n && !(rc = big_copy(&nd.m, &n))) {
        nd.e = -1;
        int positive = !n.neg;
        /* the larger multiple of pi gives the smaller t */
        if (!(rc = dyad_mul(&prod, &nd, positive ? &pihi : &pilo, EXACT, 0)) && !(rc = dyad_neg(&prod, &prod))) rc = dyad_add_exact(&tlo, x, &prod);
        if (!rc && !(rc = dyad_mul(&prod, &nd, positive ? &pilo : &pihi, EXACT, 0)) && !(rc = dyad_neg(&prod, &prod)))
            rc = dyad_add_exact(&thi, x, &prod);
    }
    /* |t| < 1: its top bit at most 0 */
    if (!rc && ((dyad_sign(&tlo) && tlo.e + big_bits(&tlo.m) > 0) || (dyad_sign(&thi) && thi.e + big_bits(&thi.m) > 0))) rc = EXACTREF_INTERNAL;
    /* step 4 at t0 = t_lo: end[0 .. 1] cos t0, end[2 .. 3] sin t0 */
    if (!rc) rc = cos_sin_fixed(fc, fs, &tlo, G);
    for (int k = 0; !rc && k < 2; k++) {
        if (!(rc = dyad_from_big(&end[k], &fc[k], -G))) rc = dyad_from_big(&end[2 + k], &fs[k], -G);
    }
    if (!rc && dyad_sign(&tlo) < 0) { /* sin t0 = -sin |t0| */
        if (!(rc = dyad_neg(&prod, &end[2])) && !(rc = dyad_neg(&end[2], &end[3]))) rc = dyad_copy(&end[3], &prod);
    }
    /* step 5: the quadrant n mod 4 maps (cos t, sin t) to (cos x, sin x) */
    int r = n.n ? (int)(n.w[0] & 3) : 0;
    if (n.neg) r = (4 - r) & 3;
    int from_c = r % 2 ? 2 : 0, from_s = r % 2 ? 0 : 2; /* cos x from sin t for odd r, sin x from cos t */
    int neg_c = r == 1 || r == 2, neg_s = r >= 2;
    /* step 6: each enclosure moved t_hi - t_lo outward, then rounded outward to p bits */
    if (!rc && !(rc = dyad_neg(&slack, &tlo))) rc = dyad_add_exact(&slack, &thi, &slack);
    for (int k = 0; !rc && k < 2; k++) {
        dyad *out = k ? s : c;
        int from = k ? from_s : from_c, negate = k ? neg_s : neg_c;
        for (int side = 0; !rc && side < 2; side++) {
            /* a negated enclosure exchanges its ends */
            if (negate)
                rc = dyad_neg(&prod, &end[from + 1 - side]);
            else
                rc = dyad_copy(&prod, &end[from + side]);
            if (!rc && side) rc = dyad_add_exact(&prod, &prod, &slack);
            if (!rc && !side && !(rc = dyad_neg(&q, &slack))) rc = dyad_add_exact(&prod, &prod, &q);
            if (!rc) rc = dyad_rnd(&out[side], &prod, p, side);
        }
    }
    dyad_free(&pilo);
    dyad_free(&pihi);
    dyad_free(&q);
    dyad_free(&half);
    dyad_free(&tlo);
    dyad_free(&thi);
    dyad_free(&nd);
    dyad_free(&prod);
    dyad_free(&slack);
    for (int k = 0; k < 4; k++) dyad_free(&end[k]);
    big_free(&n);
    for (int k = 0; k < 2; k++) {
        big_free(&fc[k]);
        big_free(&fs[k]);
    }
    return rc;
}

/* Lemma Z over an interval: cos and sin over [lo, hi] in [c[0], c[1]] and [s[0], s[1]], at most p bits each: the hull
   of their enclosures at lo and hi, and +-1 where a multiple of pi / 2 inside makes one an extreme. */
int cos_sin_range(dyad *c, dyad *s, const dyad *lo, const dyad *hi, int64_t p) {
    dyad t, c2[2], s2[2], pl, ph, u;
    bigint jlo, jhi;
    dyad_init(&t);
    dyad_init(&pl);
    dyad_init(&ph);
    dyad_init(&u);
    for (int k = 0; k < 2; k++) {
        dyad_init(&c2[k]);
        dyad_init(&s2[k]);
    }
    big_init(&jlo);
    big_init(&jhi);
    int rc = dyad_neg(&t, lo), full = 0;
    if (!rc && !(rc = dyad_add_exact(&t, hi, &t)) && !(rc = dyad_set_si(&u, 7, 0))) full = dyad_cmp(&t, &u) >= 0; /* past 2 pi */
    if (!rc && full) {
        for (int k = 0; !rc && k < 2; k++)
            if (!(rc = dyad_set_si(&c[k], k ? 1 : -1, 0))) rc = dyad_set_si(&s[k], k ? 1 : -1, 0);
    } else if (!rc && !(rc = cos_sin_bounds(c, s, lo, p)) && dyad_cmp(lo, hi) != 0 && !(rc = cos_sin_bounds(c2, s2, hi, p))) {
        if (dyad_cmp(&c2[0], &c[0]) < 0) rc = dyad_copy(&c[0], &c2[0]);
        if (!rc && dyad_cmp(&c2[1], &c[1]) > 0) rc = dyad_copy(&c[1], &c2[1]);
        if (!rc && dyad_cmp(&s2[0], &s[0]) < 0) rc = dyad_copy(&s[0], &s2[0]);
        if (!rc && dyad_cmp(&s2[1], &s[1]) > 0) rc = dyad_copy(&s[1], &s2[1]);
        /* the integers j in [2 lo / pi, 2 hi / pi], those bounds rounded outward at p + 8 bits past their integer part: j pi / 2
           for the critical points inside */
        int64_t tl = dyad_sign(lo) ? lo->e + big_bits(&lo->m) : 0, th = dyad_sign(hi) ? hi->e + big_bits(&hi->m) : 0;
        int64_t T = tl > th ? tl : th, q = p + 8 + (T > 0 ? T : 0);
        if (!rc) rc = pi_bounds(&pl, &ph, q);
        if (!rc && !(rc = dyad_copy(&t, lo)) && !(rc = dyad_shift(&t, 1))) {
            if (!(rc = dyad_div(&u, &t, dyad_sign(lo) >= 0 ? &ph : &pl, q, 0))) rc = dyad_to_fixed(&jlo, &u, 0, 1);
        }
        if (!rc && !(rc = dyad_copy(&t, hi)) && !(rc = dyad_shift(&t, 1))) {
            if (!(rc = dyad_div(&u, &t, dyad_sign(hi) >= 0 ? &pl : &ph, q, 1))) rc = dyad_to_fixed(&jhi, &u, 0, 0);
        }
        /* jhi - jlo < 5: hi - lo < 7 */
        if (!rc && jlo.n && !(rc = big_copy(&t.m, &jlo))) {
            t.m.neg = !t.m.neg;
            rc = big_add(&t.m, &jhi, &t.m);
        } else if (!rc) {
            rc = big_copy(&t.m, &jhi);
        }
        int64_t count = !rc && t.m.n ? (t.m.neg ? -(int64_t)t.m.w[0] : (int64_t)t.m.w[0]) + 1 : 1;
        if (!rc && t.m.n > 1) rc = EXACTREF_INTERNAL;
        int r = jlo.n ? (int)(jlo.w[0] & 3) : 0;
        if (jlo.neg) r = (4 - r) & 3;
        for (int64_t k = 0; !rc && k < count; k++, r = (r + 1) & 3) {
            /* cos is 1 at j = 0, -1 at j = 2 (mod 4); sin is 1 at j = 1, -1 at j = 3 */
            dyad *f = r % 2 ? s : c;
            rc = dyad_set_si(&f[r < 2 ? 1 : 0], r < 2 ? 1 : -1, 0);
        }
    }
    dyad_free(&t);
    dyad_free(&pl);
    dyad_free(&ph);
    dyad_free(&u);
    for (int k = 0; k < 2; k++) {
        dyad_free(&c2[k]);
        dyad_free(&s2[k]);
    }
    big_free(&jlo);
    big_free(&jhi);
    return rc;
}

/* Lemma Y, step 3: a bound of atanh(zeta) for a dyadic |zeta| < 1/4, from below (!up) or above
   (up): the series of a = |zeta|, terms a**(2j + 1) / (2j + 1) > 0, in fixed point at
   G = w - top(a) + 8 fractional bits, N terms and the rest under a unit; atanh is odd. */
static int atanh_fixed(dyad *out, const dyad *zeta, int64_t w, int up) {
    if (!dyad_sign(zeta)) return dyad_set_si(out, 0, 0);
    int negative = dyad_sign(zeta) < 0, side = negative ? !up : up;
    int64_t G = w - (zeta->e + big_bits(&zeta->m)) + 8, N = (G + 3) / 4 + 1; /* 16**-N <= 2**-(G + 4) */
    dyad a;
    dyad_init(&a);
    bigint A, A2, P, term, sum, one;
    big_init(&A);
    big_init(&A2);
    big_init(&P);
    big_init(&term);
    big_init(&sum);
    big_init(&one);
    int rc = dyad_copy(&a, zeta);
    if (!rc) {
        a.m.neg = 0;
        rc = dyad_to_fixed(&A, &a, G, side); /* A: a 2**G */
    }
    if (!rc && !(rc = big_mul(&A2, &A, &A))) rc = big_shr(&A2, &A2, G, side); /* A2: a**2 2**G */
    if (!rc && !(rc = big_shl(&P, &A, 0))) rc = big_shl(&sum, &A, 0);
    for (int64_t j = 1; !rc && j < N; j++) {
        /* P_j = P_(j-1) A2 / 2**G, the j-th term P_j / (2j + 1), rounded to the side */
        if ((rc = big_mul(&P, &P, &A2)) || (rc = big_shr(&P, &P, G, side)) || (rc = big_div_word(&term, &P, (uint64_t)(2 * j + 1), side))) break;
        rc = big_add(&sum, &sum, &term);
    }
    if (!rc && side && !(rc = big_set_word(&one, 1, 0))) rc = big_add(&sum, &sum, &one); /* the rest */
    if (!rc && !(rc = big_shl(&out->m, &sum, 0))) {
        out->m.neg = negative && out->m.n;
        out->e = out->m.n ? -G : 0;
    }
    dyad_free(&a);
    big_free(&A);
    big_free(&A2);
    big_free(&P);
    big_free(&term);
    big_free(&sum);
    big_free(&one);
    return rc;
}

/* Lemma Y: log(m * 2**e) in [lo, hi] for an integer m > 0 and any integer e, lo and hi at most
   p bits. */
int log_bounds(dyad *lo, dyad *hi, const bigint *m, const bigint *e, int64_t p) {
    if (!m->n || m->neg) return EXACTREF_LOG_DOMAIN;
    int64_t b = big_bits(m), w = p + 16;
    dyad y, one, d, yp1, zlo, zhi, llo, lhi, l2lo, l2hi, kd, prod;
    dyad_init(&y);
    dyad_init(&one);
    dyad_init(&d);
    dyad_init(&yp1);
    dyad_init(&zlo);
    dyad_init(&zhi);
    dyad_init(&llo);
    dyad_init(&lhi);
    dyad_init(&l2lo);
    dyad_init(&l2hi);
    dyad_init(&kd);
    dyad_init(&prod);
    bigint k, shift;
    big_init(&k);
    big_init(&shift);
    /* step 1: x = y * 2**k with y in [3/4, 3/2); y >= 3/2 when m's two top bits are set */
    int high = b >= 2 && ((m->w[(b - 2) / 64] >> ((b - 2) % 64)) & 1);
    int rc = big_shl(&y.m, m, 0);
    y.e = y.m.n ? (high ? -b : -(b - 1)) : 0;
    if (!rc && !(rc = big_set_si(&shift, high ? b : b - 1))) rc = big_add(&k, e, &shift);
    if (!rc) rc = dyad_set_si(&one, 1, 0);
    /* step 2: d = y - 1 and y + 1, exactly; z = d / (y + 1) rounded down and up */
    if (!rc) {
        dyad neg_one;
        dyad_init(&neg_one);
        rc = dyad_set_si(&neg_one, -1, 0);
        if (!rc) rc = dyad_add(&d, &y, &neg_one, EXACT, 0);
        if (!rc) rc = dyad_add(&yp1, &y, &one, EXACT, 0);
        dyad_free(&neg_one);
    }
    if (!rc) {
        if (!dyad_sign(&d)) {
            rc = dyad_set_si(&llo, 0, 0);
            if (!rc) rc = dyad_set_si(&lhi, 0, 0);
        } else if (!(rc = dyad_div(&zlo, &d, &yp1, w, 0)) && !(rc = dyad_div(&zhi, &d, &yp1, w, 1)) &&
                   !(rc = atanh_fixed(&llo, &zlo, w, 0)) && !(rc = atanh_fixed(&lhi, &zhi, w, 1)) && !(rc = dyad_shift(&llo, 1)) &&
                   !(rc = dyad_shift(&lhi, 1))) { /* step 3: log y = 2 atanh(z), atanh increasing */
        }
    }
    /* step 4: + k ln 2 */
    if (!rc && k.n) {
        int64_t kb = big_bits(&k), w2 = w + kb;
        if (!(rc = ln2_bounds(&l2lo, &l2hi, w2 + 8)) && !(rc = big_shl(&kd.m, &k, 0))) {
            kd.e = 0;
            /* k ln 2 in [k * l2lo, k * l2hi] for k > 0, the ends swapped for k < 0 */
            if (!(rc = dyad_mul(&prod, &kd, k.neg ? &l2hi : &l2lo, EXACT, 0))) rc = dyad_add(&llo, &llo, &prod, w2, 0);
            if (!rc && !(rc = dyad_mul(&prod, &kd, k.neg ? &l2lo : &l2hi, EXACT, 1))) rc = dyad_add(&lhi, &lhi, &prod, w2, 1);
        }
    }
    /* step 5 */
    if (!rc) rc = dyad_rnd(lo, &llo, p, 0);
    if (!rc) rc = dyad_rnd(hi, &lhi, p, 1);
    dyad_free(&y);
    dyad_free(&one);
    dyad_free(&d);
    dyad_free(&yp1);
    dyad_free(&zlo);
    dyad_free(&zhi);
    dyad_free(&llo);
    dyad_free(&lhi);
    dyad_free(&l2lo);
    dyad_free(&l2hi);
    dyad_free(&kd);
    dyad_free(&prod);
    big_free(&k);
    big_free(&shift);
    return rc;
}

/* Lemma W: exp(R) - 1 bounded from above (up) or below as m 2**-e (2**63 <= m < 2**64), for
   R = r 2**-s with 2**63 <= r < 2**64 and s >= 68 (0 < R < 2**-4), on words: the series
   S = sum R**k / (k + 1)! in fixed point (2**62 = 1), each term rounded to the side, stopped at a
   term of at most one unit (up, then one unit added for the rest) or of 0 (down); then r S. */
void exp_small(uint64_t *m, int64_t *e, uint64_t r, int64_t s, int up) {
    int64_t shift = s - 64;
    uint64_t rho; /* R 2**64 rounded to the side */
    if (shift >= 64)
        rho = up ? 1 : 0;
    else
        rho = (r >> shift) + (up && (r & (((uint64_t)1 << shift) - 1)) != 0);
    uint64_t t = (uint64_t)1 << 62, sum = t;
    for (uint64_t k = 2; up ? t > 1 : t > 0; k++) {
        uint64_t lo, hi;
        lo = word_mul(&hi, t, rho);
        hi += up && lo != 0;
        t = hi / k + (up && hi % k != 0);
        sum += t;
    }
    if (up) sum += 1;
    uint64_t lo, hi;
    lo = word_mul(&hi, r, sum); /* r S < 2**127, at least 2**125 */
    int lz = word_clz(hi);
    uint64_t kept = lz ? hi << lz | lo >> (64 - lz) : hi, rest = lo << lz;
    int64_t at = s - 2 + lz; /* exp(R) - 1 ~ kept 2**-at */
    if (up && rest) {
        kept++;
        if (!kept) {
            kept = (uint64_t)1 << 63;
            at--;
        }
    }
    *m = kept;
    *e = at;
}

/* Lemma W at three words: for R = r 2**-s with 2**191 <= r < 2**192 (three words) and s >= 196 (0 < R < 2**-4), m (three
   words, its top bit set) and e with exp(R) - 1 <= m 2**-e <= (exp(R) - 1)(1 + 2**-183) when up, and
   (exp(R) - 1)(1 - 2**-183) <= m 2**-e <= exp(R) - 1 otherwise. */
void exp_small3(uint64_t *m, int64_t *e, const uint64_t *r, int64_t s, int up) {
    int64_t shift = s - 192;
    uint64_t rho[3] = {0, 0, 0}, t[3] = {0, 0, (uint64_t)1 << 62}, sum[4] = {0, 0, (uint64_t)1 << 62, 0}, prod[6], wide[7];
    if (shift >= 192) {
        rho[0] = up ? 1 : 0;
    } else {
        /* rho = R 2**192 rounded to the side */
        size_t words = (size_t)(shift / 64);
        int b = (int)(shift % 64);
        uint64_t lost = 0;
        for (size_t i = 0; i < words; i++) lost |= r[i];
        if (b) lost |= r[words] & (((uint64_t)1 << b) - 1);
        for (size_t i = 0; i + words < 3; i++) rho[i] = b ? r[i + words] >> b | (i + words + 1 < 3 ? r[i + words + 1] << (64 - b) : 0) : r[i + words];
        if (up && lost) nat_add_word(rho, 1, 3);
    }
    for (uint64_t k = 2; up ? (t[2] | t[1]) || t[0] > 1 : (t[2] | t[1] | t[0]) != 0; k++) {
        /* t = ceil(ceil(t rho / 2**192) / k), or floors */
        nat_mul(prod, t, 3, rho, 3);
        t[0] = prod[3];
        t[1] = prod[4];
        t[2] = prod[5];
        if (up && (prod[0] | prod[1] | prod[2])) nat_add_word(t, 1, 3);
        if (nat_div_word(t, t, 3, k) && up) nat_add_word(t, 1, 3);
        sum[3] += nat_add(sum, sum, t, 3, 0);
    }
    if (up) nat_add_word(sum, 1, 4);
    nat_mul(prod, r, 3, sum, 3); /* r S < 2**383, at least 2**381 */
    int lz = word_clz(prod[5]);
    for (int i = 0; i < 6; i++) wide[i] = prod[i];
    wide[6] = 0;
    if (lz) nat_shl(wide, wide, 6, lz);
    *e = s - 2 + lz; /* exp(R) - 1 ~ (the top three words) 2**-e */
    m[0] = wide[3];
    m[1] = wide[4];
    m[2] = wide[5];
    if (up && (wide[0] | wide[1] | wide[2]) && nat_add_word(m, 1, 3)) { /* carried out of the top: 2**192 */
        m[2] = (uint64_t)1 << 63;
        *e -= 1;
    }
}

/* exp(x) rounded to the side, for |x| < 2**-4: 1 + Lemma W at three words (for x < 0, 1 / exp(|x|)), rounded to p bits;
   *ok 0, and r unchanged, when |x| (rounded to 192 bits) is not below 2**-4. */
int exp_near_one(dyad *r, int *ok, const dyad *x, int64_t p, int up) {
    *ok = 0;
    if (!x->m.n) {
        *ok = 1;
        return dyad_set_si(r, 1, 0);
    }
    if (x->e + big_bits(&x->m) > -4) return 0;
    int negative = x->m.neg, side = negative ? !up : up; /* the side of exp(|x|) - 1 wanted */
    dyad a, one, sum;
    dyad_init(&a);
    dyad_init(&one);
    dyad_init(&sum);
    int rc = dyad_abs(&a, x);
    if (!rc) rc = dyad_rnd(&a, &a, 192, side); /* |x| to the side, at most 192 bits */
    if (!rc && a.e + big_bits(&a.m) <= -4) { /* not rounded up to 2**-4 */
        *ok = 1;
        int64_t b = big_bits(&a.m);
        uint64_t rw[4] = {0, 0, 0, 0}, mw[3];
        /* rw = |x|'s integer moved up to 192 bits */
        size_t words = (size_t)((192 - b) / 64);
        int sh = (int)((192 - b) % 64);
        for (size_t i = 0; i < a.m.n; i++) rw[i + words] = a.m.w[i];
        if (sh) nat_shl(rw, rw, 3, sh);
        int64_t s = 192 - b - a.e, e;
        exp_small3(mw, &e, rw, s, side);
        if (!(rc = big_reserve(&sum.m, 3))) {
            for (int i = 0; i < 3; i++) sum.m.w[i] = mw[i];
            sum.m.n = 3;
            sum.m.neg = 0;
            big_trim(&sum.m);
            sum.e = sum.m.n ? -e : 0;
            if (!(rc = dyad_set_si(&one, 1, 0)) && !(rc = dyad_add_exact(&sum, &sum, &one)))
                rc = negative ? dyad_div(r, &one, &sum, p, up) : dyad_rnd(r, &sum, p, up);
        }
    }
    dyad_free(&a);
    dyad_free(&one);
    dyad_free(&sum);
    return rc;
}

/* exp(x) rounded to the side (Lemma X), as one dyad; EXACTREF_OVERFLOW where its exponent passes every dyad's, at once
   for |x| >= 2**63 (its exponent n then has |n| > 2**62) */
int exp_end(dyad *r, const dyad *x, int64_t p, int up) {
    if (x->m.n && x->e + big_bits(&x->m) - 1 >= 63) return EXACTREF_OVERFLOW;
    dyad lo, hi;
    bigint n;
    dyad_init(&lo);
    dyad_init(&hi);
    big_init(&n);
    int rc = exp_bounds(&lo, &hi, &n, x, p);
    if (!rc && (n.n > 1 || (n.n && n.w[0] >> 62))) rc = EXACTREF_OVERFLOW; /* 2**n past every exponent */
    if (!rc && !(rc = dyad_copy(r, up ? &hi : &lo)) && n.n) rc = dyad_shift(r, n.neg ? -(int64_t)n.w[0] : (int64_t)n.w[0]);
    dyad_free(&lo);
    dyad_free(&hi);
    big_free(&n);
    return rc;
}

/* exp(x) rounded to the side; below -2**40, 0 down and exp(-2**40) up */
int exp_clamped(dyad *r, const dyad *x, int64_t p, int up) {
    if (dyad_sign(x) < 0 && x->e + big_bits(&x->m) - 1 >= 40) {
        if (!up) return dyad_set_si(r, 0, 0);
        dyad f;
        dyad_init(&f);
        int rc = dyad_set_si(&f, -1, 40);
        if (!rc) rc = exp_end(r, &f, p, 1);
        dyad_free(&f);
        return rc;
    }
    return exp_end(r, x, p, up);
}

/* log of a positive dyad rounded to the side (Lemma Y) */
int log_end(dyad *r, const dyad *x, int64_t p, int up) {
    dyad lo, hi;
    bigint e;
    dyad_init(&lo);
    dyad_init(&hi);
    big_init(&e);
    int rc = big_set_si(&e, x->e);
    if (!rc) rc = log_bounds(&lo, &hi, &x->m, &e, p);
    if (!rc) rc = dyad_copy(r, up ? &hi : &lo);
    dyad_free(&lo);
    dyad_free(&hi);
    big_free(&e);
    return rc;
}
