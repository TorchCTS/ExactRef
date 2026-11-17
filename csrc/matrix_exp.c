/* matrix_exp's kernels (proofs/matrix_exp.md): its exact parts (Rational parts), the reference (The reference output)
   and the tolerance (The tolerance). The exact parts: which entries of exp(A) are rational, and their values, for a
   matrix of dyadics. With P = 2**k B an integer matrix, the work for each eigenvalue other than 0 is
   over the integers modulo a monic factor of P's characteristic polynomial; the eigenvalue 0's values are rational
   constants. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"

/* ---------------------------------------------------------------- Z[mu]/(g) */

typedef struct {
    const ipoly *g; /* monic, of degree >= 1 */
    ipoly mu;       /* z modulo g */
} field;

static int field_init(field *F, const ipoly *g) {
    F->g = g;
    ipoly_init(&F->mu);
    bigint one;
    big_init(&one);
    int rc = big_set_word(&one, 1, 0);
    if (!rc) rc = ipoly_set_term(&F->mu, &one, 1);
    if (!rc) rc = ipoly_divrem_monic(NULL, &F->mu, &F->mu, g);
    big_free(&one);
    return rc;
}

static void field_free(field *F) { ipoly_free(&F->mu); }

/* r = a b mod g */
static int field_mul(ipoly *r, const field *F, const ipoly *a, const ipoly *b) {
    int rc = ipoly_mul(r, a, b);
    return rc ? rc : ipoly_divrem_monic(NULL, r, r, F->g);
}

/* polynomials in z over the field: arrays of n + 1 ipolys */
static ipoly *zpoly_new(size_t n) {
    ipoly *p = malloc((n + 1) * sizeof *p);
    if (p)
        for (size_t k = 0; k <= n; k++) ipoly_init(&p[k]);
    return p;
}

static void zpoly_free(ipoly *p, size_t n) {
    if (!p) return;
    for (size_t k = 0; k <= n; k++) ipoly_free(&p[k]);
    free(p);
}

/* Synthetic division of p (n >= 1 coefficients) by (z - mu): quotient q (n - 1 coefficients) and remainder *rem */
static int divide_by_root(ipoly *q, ipoly *rem, const ipoly *p, size_t n, const field *F) {
    ipoly carry, t;
    ipoly_init(&carry);
    ipoly_init(&t);
    int rc = 0;
    for (size_t k = n - 1; !rc && k >= 1; k--)
        if (!(rc = field_mul(&t, F, &F->mu, &carry)) && !(rc = ipoly_add(&carry, &p[k], &t))) rc = ipoly_copy(&q[k - 1], &carry);
    if (!rc && !(rc = field_mul(&t, F, &F->mu, &carry))) rc = ipoly_add(rem, &p[0], &t);
    ipoly_free(&carry);
    ipoly_free(&t);
    return rc;
}

/* n! as an integer */
static int factorial(bigint *r, size_t n) {
    bigint w;
    big_init(&w);
    int rc = big_set_word(r, 1, 0);
    for (size_t i = 2; !rc && i <= n; i++)
        if (!(rc = big_set_word(&w, i, 0))) rc = big_mul(r, r, &w);
    big_free(&w);
    return rc;
}

/* For a root mu of g of multiplicity m in chi (n + 1 coefficients): f = q T times the unit
   w**m 2**(k (m - 1)) (m - 1)!, w = q(mu), with chi = (z - mu)**m q and T the Taylor expansion of
   e**(v 2**-k) / q(mu + v) to order m in powers of (z - mu); f gets n coefficients, each modulo g. A unit keeps
   every entry's vanishing, and these make every quantity an integer (matrix_exp.md, Rational parts). */
static int coefficient_polynomial(ipoly *f, const ipoly *chi, size_t n, size_t m, int64_t k, const field *F) {
    size_t qlen = n + 1;
    ipoly *q = zpoly_new(qlen), *nq = zpoly_new(qlen), *quot = zpoly_new(qlen), *taylor = zpoly_new(m), *J = zpoly_new(m), *wpow = zpoly_new(m),
          *sigma = zpoly_new(m), *T = zpoly_new(m), *power = zpoly_new(m + 1), *npower = zpoly_new(m + 1);
    bigint *e = malloc((m + 1) * sizeof *e), fm, fb, one;
    ipoly rem, t, total;
    ipoly_init(&rem);
    ipoly_init(&t);
    ipoly_init(&total);
    big_init(&fm);
    big_init(&fb);
    big_init(&one);
    for (size_t b = 0; e && b <= m; b++) big_init(&e[b]);
    int rc = !q || !nq || !quot || !taylor || !J || !wpow || !sigma || !T || !power || !npower || !e ? EXACTREF_NOMEM : 0;
    if (!rc) rc = big_set_word(&one, 1, 0);
    for (size_t s = 0; !rc && s < qlen && s < chi->len; s++) rc = ipoly_set_term(&q[s], &chi->c[s], 0);
    /* q = chi / (z - mu)**m */
    for (size_t r = 0; !rc && r < m; r++) {
        if ((rc = divide_by_root(nq, &rem, q, qlen, F))) break;
        if (rem.len) rc = EXACTREF_INTERNAL; /* the multiplicity does not match */
        ipoly *x = q;
        q = nq;
        nq = x;
        qlen--;
    }
    /* q's Taylor coefficients at mu, by repeated division by (z - mu) */
    size_t rlen = qlen;
    for (size_t s = 0; !rc && s < qlen; s++) rc = ipoly_copy(&nq[s], &q[s]);
    for (size_t j = 0; !rc && j < m; j++) {
        if (!rlen) continue;
        if (rlen == 1) {
            rc = ipoly_copy(&taylor[j], &nq[0]);
            rlen = 0;
            continue;
        }
        if ((rc = divide_by_root(quot, &taylor[j], nq, rlen, F))) break;
        rlen--;
        for (size_t i = 0; !rc && i < rlen; i++) rc = ipoly_copy(&nq[i], &quot[i]);
    }
    /* w**x, and J_0 = 1, J_j = -sum over 1 <= i <= j of t_i w**(i - 1) J_(j - i): w**(j + 1) times the series of
       1 / q(mu + v) */
    if (!rc) rc = ipoly_set_term(&wpow[0], &one, 0);
    for (size_t x = 1; !rc && x < m; x++) rc = field_mul(&wpow[x], F, &wpow[x - 1], &taylor[0]);
    if (!rc) rc = ipoly_set_term(&J[0], &one, 0);
    for (size_t j = 1; !rc && j < m; j++) {
        ipoly_free(&total);
        ipoly_init(&total);
        for (size_t i = 1; !rc && i <= j; i++)
            if (!(rc = field_mul(&t, F, &taylor[i], &J[j - i])) && !(rc = field_mul(&t, F, &t, &wpow[i - 1]))) rc = ipoly_add(&total, &total, &t);
        if (!rc) rc = ipoly_sub(&J[j], &J[j], &total);
    }
    /* e_b = 2**(k (m - 1 - b)) (m - 1)! / b!, the exponential's coefficients times 2**(k (m - 1)) (m - 1)! */
    if (!rc) rc = factorial(&fm, m - 1);
    for (size_t b = 0; !rc && b < m; b++) {
        int exact;
        if (!(rc = factorial(&fb, b)) && !(rc = big_divmod(&e[b], &exact, &fm, &fb))) rc = big_shl(&e[b], &e[b], k * (int64_t)(m - 1 - b));
    }
    /* sigma_j = sum over a <= j of w**(m - a - 1) J_a e_(j - a) */
    for (size_t j = 0; !rc && j < m; j++)
        for (size_t a = 0; !rc && a <= j; a++)
            if (!(rc = field_mul(&t, F, &wpow[m - a - 1], &J[a])) && !(rc = ipoly_scale(&t, &t, &e[j - a]))) rc = ipoly_add(&sigma[j], &sigma[j], &t);
    /* T(z) = sum over j < m of sigma_j (z - mu)**j, power = (z - mu)**j */
    size_t plen = 1;
    if (!rc) rc = ipoly_set_term(&power[0], &one, 0);
    for (size_t j = 0; !rc && j < m; j++) {
        for (size_t i = 0; !rc && i < plen; i++)
            if (!(rc = field_mul(&t, F, &sigma[j], &power[i]))) rc = ipoly_add(&T[i], &T[i], &t);
        for (size_t i = 0; !rc && i <= plen; i++) { /* power (z - mu): shifted, less mu times power */
            ipoly_free(&npower[i]);
            ipoly_init(&npower[i]);
            if (i >= 1) rc = ipoly_copy(&npower[i], &power[i - 1]);
            if (!rc && i < plen && !(rc = field_mul(&t, F, &F->mu, &power[i]))) rc = ipoly_sub(&npower[i], &npower[i], &t);
        }
        ipoly *x = power;
        power = npower;
        npower = x;
        plen++;
    }
    /* f = q T */
    for (size_t i = 0; !rc && i < qlen; i++)
        for (size_t j = 0; !rc && j < m; j++)
            if (q[i].len && T[j].len && !(rc = field_mul(&t, F, &q[i], &T[j]))) rc = ipoly_add(&f[i + j], &f[i + j], &t);
    zpoly_free(q, n + 1);
    zpoly_free(nq, n + 1);
    zpoly_free(quot, n + 1);
    zpoly_free(taylor, m);
    zpoly_free(J, m);
    zpoly_free(wpow, m);
    zpoly_free(sigma, m);
    zpoly_free(T, m);
    zpoly_free(power, m + 1);
    zpoly_free(npower, m + 1);
    for (size_t b = 0; e && b <= m; b++) big_free(&e[b]);
    free(e);
    ipoly_free(&rem);
    ipoly_free(&t);
    ipoly_free(&total);
    big_free(&fm);
    big_free(&fb);
    big_free(&one);
    return rc;
}

/* For the root 0 of multiplicity m in chi: the rational f = q T, with chi = z**m q and T the Taylor expansion of
   e**(v 2**-k) / q(v) to order m */
static int zero_polynomial(rpoly *f, const ipoly *chi, size_t m, int64_t k) {
    rpoly q, T, term;
    rat *inv = malloc((m + 1) * sizeof *inv), c, t, total;
    bigint one, fb;
    rpoly_init(&q);
    rpoly_init(&T);
    rpoly_init(&term);
    rat_init(&c);
    rat_init(&t);
    rat_init(&total);
    big_init(&one);
    big_init(&fb);
    for (size_t j = 0; inv && j <= m; j++) rat_init(&inv[j]);
    int rc = !inv ? EXACTREF_NOMEM : big_set_word(&one, 1, 0);
    /* q = chi / z**m, its coefficient of z**j the Taylor coefficient t_j */
    for (size_t s = m; !rc && s < chi->len; s++)
        if (!(rc = rat_set_dyad(&c, &chi->c[s], 0)) && !(rc = rpoly_set_term(&term, &c, s - m))) rc = rpoly_add(&q, &q, &term);
    /* inv: the series of 1 / q(v) (q(0) != 0: 0 has multiplicity exactly m) */
    if (!rc && q.len) rc = rat_inv(&inv[0], &q.c[0]);
    for (size_t j = 1; !rc && j < m; j++) {
        if ((rc = rat_set_si(&total, 0))) break;
        for (size_t i = 1; !rc && i <= j && i < q.len; i++)
            if (!(rc = rat_mul(&t, &q.c[i], &inv[j - i]))) rc = rat_add(&total, &total, &t);
        if (!rc && !(rc = rat_mul(&t, &inv[0], &total))) rc = rat_neg(&inv[j], &t);
    }
    /* T = sum over j < m of tau_j z**j, tau_j = sum over a <= j of inv_a 2**(-k (j - a)) / (j - a)! */
    for (size_t j = 0; !rc && j < m; j++) {
        if ((rc = rat_set_si(&total, 0))) break;
        for (size_t a = 0; !rc && a <= j; a++) {
            if ((rc = factorial(&fb, j - a)) || (rc = big_shl(&fb, &fb, k * (int64_t)(j - a))) || (rc = rat_set(&c, &one, &fb))) break;
            if (!(rc = rat_mul(&t, &inv[a], &c))) rc = rat_add(&total, &total, &t);
        }
        if (!rc && !(rc = rpoly_set_term(&term, &total, j))) rc = rpoly_add(&T, &T, &term);
    }
    if (!rc) rc = rpoly_mul(f, &q, &T);
    for (size_t j = 0; inv && j <= m; j++) rat_free(&inv[j]);
    free(inv);
    rpoly_free(&q);
    rpoly_free(&T);
    rpoly_free(&term);
    rat_free(&c);
    rat_free(&t);
    rat_free(&total);
    big_free(&one);
    big_free(&fb);
    return rc;
}

/* ---------------------------------------------------------------- the characteristic polynomial */

/* chi, the characteristic polynomial modulo p of the M x M matrix H (P modulo p, overwritten), M + 1
   coefficients: Hessenberg's form by similarity, then the recurrence on its columns; pk has room for
   (M + 1)**2 words. */
static void charpoly_mod(uint64_t *chi, uint64_t *H, size_t M, uint64_t p, uint64_t *pk) {
#define AT(i, j) H[(i) * M + (j)]
    for (size_t c = 1; c + 1 < M; c++) {
        size_t piv = c;
        while (piv < M && !AT(piv, c - 1)) piv++;
        if (piv == M) continue;
        if (piv != c) {
            for (size_t j = 0; j < M; j++) {
                uint64_t x = AT(piv, j);
                AT(piv, j) = AT(c, j);
                AT(c, j) = x;
            }
            for (size_t i = 0; i < M; i++) {
                uint64_t x = AT(i, piv);
                AT(i, piv) = AT(i, c);
                AT(i, c) = x;
            }
        }
        uint64_t inv = mod_pow(AT(c, c - 1), p - 2, p);
        for (size_t r = c + 1; r < M; r++) {
            if (!AT(r, c - 1)) continue;
            uint64_t u = mod_mul(AT(r, c - 1), inv, p);
            for (size_t j = 0; j < M; j++) AT(r, j) = (AT(r, j) + p - mod_mul(u, AT(c, j), p)) % p; /* row r -= u row c */
            for (size_t i = 0; i < M; i++) AT(i, c) = (AT(i, c) + mod_mul(u, AT(i, r), p)) % p;     /* column c += u column r */
        }
    }
    /* p_k (k + 1 coefficients at pk + k (M + 1)) = (z - h_kk) p_(k-1) - sum over i < k of h_(k-i, k) (h_(k, k-1) ..
       h_(k-i+1, k-i)) p_(k-i-1) */
    size_t W = M + 1;
    memset(pk, 0, W * W * sizeof *pk);
    pk[0] = 1 % p;
    for (size_t k = 1; k <= M; k++) {
        uint64_t *cur = pk + k * W, *prev = pk + (k - 1) * W, h = AT(k - 1, k - 1);
        for (size_t s = 0; s < k; s++) {
            cur[s + 1] = (cur[s + 1] + prev[s]) % p;
            cur[s] = (cur[s] + p - mod_mul(h, prev[s], p)) % p;
        }
        uint64_t prod = 1 % p;
        for (size_t i = 1; i < k; i++) {
            prod = mod_mul(prod, AT(k - i, k - i - 1), p);
            if (!prod) break;
            uint64_t f = mod_mul(prod, AT(k - i - 1, k - 1), p);
            const uint64_t *older = pk + (k - i - 1) * W;
            for (size_t s = 0; s <= k - i - 1; s++) cur[s] = (cur[s] + p - mod_mul(f, older[s], p)) % p;
        }
    }
    memcpy(chi, pk + M * W, W * sizeof *chi);
#undef AT
}

/* chi, the characteristic polynomial of the integer matrix P (M x M): its residues modulo word primes put
   together by the Chinese remainder theorem. The coefficient of z**(M - j) is a signed sum of the C(M, j)
   principal minors of order j, each at most R**j by Hadamard's inequality (R < 2**r the largest row's Euclidean
   length), so below 2**(M + M r) in magnitude; primes whose product passes 2**(M + M r + 2) determine it. */
static int charpoly_int(ipoly *chi, const bigint *P, size_t M) {
    size_t W = M + 1;
    int64_t r = 0;
    bigint sq, u, mod, x, t;
    big_init(&sq);
    big_init(&u);
    big_init(&mod);
    big_init(&x);
    big_init(&t);
    uint64_t *H = malloc((M * M + 1) * sizeof *H), *res = malloc(W * sizeof *res), *pk = malloc((W * W + 1) * sizeof *pk);
    int rc = !H || !res || !pk ? EXACTREF_NOMEM : 0;
    for (size_t i = 0; !rc && i < M; i++) {
        if ((rc = big_set_word(&sq, 0, 0))) break;
        for (size_t j = 0; !rc && j < M; j++)
            if (P[i * M + j].n && !(rc = big_mul(&u, &P[i * M + j], &P[i * M + j]))) rc = big_add(&sq, &sq, &u);
        int64_t half = (big_bits(&sq) + 1) / 2; /* sqrt(sq) < 2**half */
        if (half > r) r = half;
    }
    int64_t need = (int64_t)M + (int64_t)M * r + 2;
    if (!rc) rc = ipoly_zero(chi, W);
    if (!rc) rc = big_set_word(&mod, 1, 0);
    uint64_t p = ((uint64_t)1 << 62) + 1;
    while (!rc && big_bits(&mod) <= need) {
        do p -= 2;
        while (!word_prime(p));
        for (size_t a = 0; a < M * M; a++) H[a] = big_mod(&P[a], p);
        charpoly_mod(res, H, M, p, pk);
        /* Garner: x += mod ((res - x) / mod mod p) */
        uint64_t minv = mod_pow(big_mod(&mod, p), p - 2, p);
        for (size_t s = 0; !rc && s < W; s++) {
            uint64_t d = (res[s] + p - big_mod(&chi->c[s], p)) % p, c = mod_mul(d, minv, p);
            if (c && !(rc = big_set_word(&u, c, 0)) && !(rc = big_mul(&u, &u, &mod))) rc = big_add(&chi->c[s], &chi->c[s], &u);
        }
        if (!rc && !(rc = big_set_word(&u, p, 0))) rc = big_mul(&mod, &mod, &u);
    }
    /* the symmetric range: x > mod / 2 stands for x - mod */
    for (size_t s = 0; !rc && s < W; s++) {
        if ((rc = big_shl(&t, &chi->c[s], 1))) break;
        if (big_cmp_mag(&t, &mod) > 0 && !(rc = big_copy(&x, &mod))) {
            x.neg = 1;
            rc = big_add(&chi->c[s], &chi->c[s], &x);
        }
    }
    if (!rc) ipoly_trim(chi);
    free(H);
    free(res);
    free(pk);
    big_free(&sq);
    big_free(&u);
    big_free(&mod);
    big_free(&x);
    big_free(&t);
    return rc;
}

/* ---------------------------------------------------------------- one real matrix */

typedef struct {
    size_t m;
    int64_t k;   /* B = P 2**-k, P integer */
    bigint *pw;  /* P**d for d < m: m x m integers each */
    size_t nparts;
    ipoly *g;    /* per nonzero part its factor */
    ipoly **f;   /* and its coefficient polynomial (scaled), m coefficients modulo g */
    int zero;    /* 0 is an eigenvalue: */
    rpoly f0;    /* its coefficient polynomial, rational */
} rtest;

static void rtest_free(rtest *t) {
    if (t->pw)
        for (size_t a = 0; a < t->m * t->m * t->m; a++) big_free(&t->pw[a]);
    free(t->pw);
    for (size_t p = 0; p < t->nparts; p++) {
        ipoly_free(&t->g[p]);
        zpoly_free(t->f[p], t->m);
    }
    free(t->g);
    free(t->f);
    rpoly_free(&t->f0);
    memset(t, 0, sizeof *t);
}

/* the test for B (m x m dyads, row-major) */
static int rtest_init(rtest *t, const dyad *B, size_t m) {
    memset(t, 0, sizeof *t);
    rpoly_init(&t->f0);
    t->m = m;
    size_t mm = m * m;
    int rc = 0;
    int64_t k = 0;
    for (size_t a = 0; a < mm; a++)
        if (B[a].m.n && -B[a].e > k) k = -B[a].e;
    t->k = k;
    t->pw = malloc((m * mm + 1) * sizeof *t->pw);
    bigint *P = malloc((mm + 1) * sizeof *P), prod;
    ipoly chi, *factors = malloc((m + 1) * sizeof *factors);
    ipoly_init(&chi);
    big_init(&prod);
    if (!t->pw || !P || !factors) rc = EXACTREF_NOMEM;
    for (size_t a = 0; t->pw && a < m * mm; a++) big_init(&t->pw[a]);
    for (size_t a = 0; P && a < mm; a++) big_init(&P[a]);
    for (size_t a = 0; factors && a <= m; a++) ipoly_init(&factors[a]);
    /* P = B 2**k and its powers P**0 = I, ..., P**(m - 1) */
    for (size_t a = 0; !rc && a < mm; a++) rc = B[a].m.n ? big_shl(&P[a], &B[a].m, B[a].e + k) : big_set_word(&P[a], 0, 0);
    for (size_t a = 0; !rc && a < mm; a++) rc = big_set_word(&t->pw[a], a / m == a % m, 0);
    for (size_t d = 1; !rc && d < m; d++) {
        bigint *prev = t->pw + (d - 1) * mm, *cur = t->pw + d * mm;
        for (size_t i = 0; !rc && i < m; i++)
            for (size_t j = 0; !rc && j < m; j++) {
                if ((rc = big_set_word(&cur[i * m + j], 0, 0))) break;
                for (size_t l = 0; !rc && l < m; l++)
                    if (prev[i * m + l].n && P[l * m + j].n && !(rc = big_mul(&prod, &prev[i * m + l], &P[l * m + j])))
                        rc = big_add(&cur[i * m + j], &cur[i * m + j], &prod);
            }
    }
    /* chi of P, its square-free factors, the factor z apart */
    if (!rc) rc = charpoly_int(&chi, P, m);
    size_t count = 0;
    if (!rc) rc = ipoly_squarefree(factors, &count, &chi);
    t->g = calloc(count + 1, sizeof *t->g);
    t->f = calloc(count + 1, sizeof *t->f);
    size_t *mult = calloc(count + 1, sizeof *mult);
    if (!rc && (!t->g || !t->f || !mult)) rc = EXACTREF_NOMEM;
    for (size_t a = 0; !rc && a <= count; a++) ipoly_init(&t->g[a]);
    for (size_t i = 0; !rc && i < count; i++) {
        ipoly *g = &factors[i];
        if (g->len <= 1) continue; /* no factor of that multiplicity */
        if (!g->c[0].n) {          /* z divides g: the eigenvalue 0, of multiplicity i + 1 */
            t->zero = 1;
            rc = zero_polynomial(&t->f0, &chi, i + 1, k);
            for (size_t s = 1; !rc && s < g->len; s++) rc = big_copy(&g->c[s - 1], &g->c[s]); /* g / z */
            g->len--;
            if (g->len <= 1) continue;
        }
        if (!rc) rc = ipoly_copy(&t->g[t->nparts], g);
        mult[t->nparts++] = i + 1;
    }
    for (size_t p = 0; !rc && p < t->nparts; p++) {
        field F;
        if ((rc = field_init(&F, &t->g[p]))) break;
        if (!(t->f[p] = zpoly_new(m))) rc = EXACTREF_NOMEM;
        if (!rc) rc = coefficient_polynomial(t->f[p], &chi, m, mult[p], k, &F);
        field_free(&F);
    }
    free(mult);
    if (P)
        for (size_t a = 0; a < mm; a++) big_free(&P[a]);
    free(P);
    if (factors)
        for (size_t a = 0; a <= m; a++) ipoly_free(&factors[a]);
    free(factors);
    ipoly_free(&chi);
    big_free(&prod);
    return rc;
}

/* whether exp(B)_ij is rational, and then its value: every nonzero eigenvalue's coefficient vanishes, and the
   value is the eigenvalue 0's */
static int rtest_entry(int *rational, rat *value, const rtest *t, size_t i, size_t j) {
    size_t m = t->m, mm = m * m;
    ipoly sum, term;
    rat c, u;
    ipoly_init(&sum);
    ipoly_init(&term);
    rat_init(&c);
    rat_init(&u);
    int rc = rat_set_si(value, 0);
    *rational = 1;
    for (size_t p = 0; !rc && p < t->nparts && *rational; p++) {
        ipoly_free(&sum);
        ipoly_init(&sum);
        for (size_t d = 0; !rc && d < m; d++) {
            const bigint *x = &t->pw[d * mm + i * m + j];
            if (t->f[p][d].len && x->n && !(rc = ipoly_scale(&term, &t->f[p][d], x))) rc = ipoly_add(&sum, &sum, &term);
        }
        if (!rc && sum.len) *rational = 0;
    }
    for (size_t d = 0; !rc && *rational && t->zero && d < t->f0.len; d++) {
        const bigint *x = &t->pw[d * mm + i * m + j];
        if (x->n && !rat_is_zero(&t->f0.c[d]) && !(rc = rat_set_dyad(&u, x, 0)) && !(rc = rat_mul(&c, &t->f0.c[d], &u))) rc = rat_add(value, value, &c);
    }
    ipoly_free(&sum);
    ipoly_free(&term);
    rat_free(&c);
    rat_free(&u);
    return rc;
}

/* ---------------------------------------------------------------- the test for a matrix's wanted parts */

/* For the wanted (i, j, part) triples of A (parts x n x n dyads, a plane per part, row-major), whether each is
   rational and then its value. Every walk from i to j stays among the vertices S on such walks, so
   exp(A)_ij = exp(A[S][S])_ij; a complex X + iY is tested through the real [[X, -Y], [Y, X]]. */
int mexp_rational(int *rational, rat *values, size_t n, int parts, const dyad *A, const size_t *wanted, size_t count) {
    size_t words = (n + 63) / 64, nn = n * n;
    uint64_t *reach = calloc(n * words + 1, sizeof *reach), *S = calloc(count * words + 1, sizeof *S);
    size_t *group = malloc((count + 1) * sizeof *group), *index = malloc((n + 1) * sizeof *index);
    int rc = !reach || !S || !group || !index ? EXACTREF_NOMEM : 0;
    for (size_t i = 0; !rc && i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (int p = 0; p < parts; p++)
                if (A[(size_t)p * nn + i * n + j].m.n) reach[i * words + j / 64] |= (uint64_t)1 << (j % 64);
    for (size_t k = 0; !rc && k < n; k++)
        for (size_t i = 0; i < n; i++)
            if (reach[i * words + k / 64] >> (k % 64) & 1)
                for (size_t w = 0; w < words; w++) reach[i * words + w] |= reach[k * words + w];
    /* each wanted entry's vertex set, and the first entry with the same set */
    for (size_t a = 0; !rc && a < count; a++) {
        size_t i = wanted[3 * a], j = wanted[3 * a + 1];
        uint64_t *s = S + a * words;
        s[i / 64] |= (uint64_t)1 << (i % 64);
        s[j / 64] |= (uint64_t)1 << (j % 64);
        for (size_t v = 0; v < n; v++)
            if ((reach[i * words + v / 64] >> (v % 64) & 1) && (reach[v * words + j / 64] >> (j % 64) & 1)) s[v / 64] |= (uint64_t)1 << (v % 64);
        group[a] = a;
        for (size_t b = 0; b < a; b++)
            if (group[b] == b && !memcmp(S + b * words, s, words * sizeof *s)) {
                group[a] = b;
                break;
            }
    }
    for (size_t a = 0; !rc && a < count; a++) {
        if (group[a] != a) continue;
        const uint64_t *s = S + a * words;
        size_t m = 0;
        for (size_t v = 0; v < n; v++) index[v] = s[v / 64] >> (v % 64) & 1 ? m++ : (size_t)-1;
        size_t M = parts * m;
        dyad *B = malloc((M * M + 1) * sizeof *B);
        if (!B) {
            rc = EXACTREF_NOMEM;
            break;
        }
        for (size_t x = 0; x < M * M; x++) dyad_init(&B[x]);
        for (size_t i = 0; !rc && i < n; i++)
            for (size_t j = 0; !rc && j < n; j++) {
                if (index[i] == (size_t)-1 || index[j] == (size_t)-1) continue;
                size_t r = index[i], c = index[j];
                const dyad *X = &A[i * n + j];
                rc = dyad_copy(&B[r * M + c], X);
                if (!rc && parts == 2) {
                    const dyad *Y = &A[nn + i * n + j];
                    if (!(rc = dyad_copy(&B[(r + m) * M + c + m], X)) && !(rc = dyad_copy(&B[(r + m) * M + c], Y))) rc = dyad_neg(&B[r * M + c + m], Y);
                }
            }
        rtest t;
        if (!rc) rc = rtest_init(&t, B, M);
        for (size_t b = a; !rc && b < count; b++)
            if (group[b] == a) rc = rtest_entry(&rational[b], &values[b], &t, index[wanted[3 * b]] + (size_t)wanted[3 * b + 2] * m, index[wanted[3 * b + 1]]);
        rtest_free(&t);
        for (size_t x = 0; x < M * M; x++) dyad_free(&B[x]);
        free(B);
    }
    free(reach);
    free(S);
    free(group);
    free(index);
    return rc;
}

/* ---------------------------------------------------------------- the reference (matrix_exp.md, The reference output) */

enum { MREF_START, MREF_EXP, MREF_DECIDE, MREF_RATIONAL, MREF_DONE };

static size_t mref_wanted(const mref *x) { return (x->r1 - x->r0) * (x->c1 - x->c0); }

/* the wanted part (i, j, part)'s slot */
static size_t mref_slot(const mref *x, size_t m, size_t i, size_t j, int part) {
    return ((m * (x->r1 - x->r0) + (i - x->r0)) * (x->c1 - x->c0) + (j - x->c0)) * (size_t)x->parts + (size_t)part;
}

static const dyad *mref_at(const mref *x, size_t m, int part, size_t i, size_t j) {
    return &x->A[((m * (size_t)x->parts + (size_t)part) * x->n + i) * x->n + j];
}

/* a part's encoding and entry: a special value or an exact zero (+0, or -0 rounding toward -inf) */
static int mref_kind(mref *x, size_t slot, int kind, int neg) {
    bigint bits, zero;
    big_init(&bits);
    big_init(&zero);
    int rc = floatfmt_encode(&bits, &x->out, kind, neg, &zero, 0, x->rnd, x->ovf);
    if (!rc) rc = entry_value(&x->entries[slot], kind, neg, &zero, 0);
    if (!rc) bytes_store(x->enc + slot * x->size, x->size, &bits);
    x->status[slot] = 0;
    big_free(&bits);
    big_free(&zero);
    return rc;
}

/* matrix m's 2**scale as a rational into r */
static int mref_scale_rat(rat *r, const mref *x, size_t m) {
    int64_t k = x->scale ? x->scale[m] : 0;
    bigint one, pw;
    big_init(&one);
    big_init(&pw);
    int rc = big_set_word(&one, 1, 0);
    if (!rc) rc = big_shl(&pw, &one, k >= 0 ? k : -k);
    if (!rc) rc = k >= 0 ? rat_set(r, &pw, &one) : rat_set(r, &one, &pw);
    big_free(&one);
    big_free(&pw);
    return rc;
}

static size_t mref_matrix_of(const mref *x, size_t slot) { return slot / (mref_wanted(x) * (size_t)x->parts); }

/* an exactly known rational part (of the computed matrix: times its 2**scale), rounded once; an exact zero is +0, or -0
   rounding toward -inf */
static int mref_rational(mref *x, size_t slot, const rat *q0) {
    bigint bits, a;
    rat scaled, f;
    big_init(&bits);
    big_init(&a);
    rat_init(&scaled);
    rat_init(&f);
    const rat *q = q0;
    int rc = 0;
    if (x->scale && x->scale[mref_matrix_of(x, slot)] && q0->num.n) {
        if (!(rc = mref_scale_rat(&f, x, mref_matrix_of(x, slot)))) rc = rat_mul(&scaled, q0, &f);
        q = &scaled;
    }
    if (!rc) rc = big_copy(&a, &q->num);
    int neg = q->num.neg;
    a.neg = 0;
    if (!rc && !q->num.n) rc = floatfmt_encode(&bits, &x->out, VALUE_FINITE, x->rnd == ROUND_TOWARD_NEGATIVE, &a, 0, x->rnd, x->ovf);
    else if (!rc) rc = floatfmt_encode_ratio(&bits, &x->out, neg, &a, &q->den, x->rnd, x->ovf);
    if (!rc) rc = entry_ratio(&x->entries[slot], neg, &a, &q->den);
    if (!rc) bytes_store(x->enc + slot * x->size, x->size, &bits);
    x->status[slot] = 0;
    big_free(&bits);
    big_free(&a);
    rat_free(&scaled);
    rat_free(&f);
    return rc;
}

/* a part known to lie in [lo, hi] (of the computed matrix: the ends times its 2**scale, exactly): decided when both ends
   round alike (and, before the last attempt, it excludes 0) */
static int mref_interval(mref *x, size_t slot, const dyad *lo0, const dyad *hi0, int last) {
    bigint bits;
    dyad slo, shi;
    big_init(&bits);
    dyad_init(&slo);
    dyad_init(&shi);
    const dyad *lo = lo0, *hi = hi0;
    int64_t k = x->scale ? x->scale[mref_matrix_of(x, slot)] : 0;
    int none = 1, rc = 0;
    if (k) {
        if (!(rc = dyad_copy(&slo, lo0)) && !(rc = dyad_copy(&shi, hi0)) && !(rc = dyad_shift(&slo, k))) rc = dyad_shift(&shi, k);
        lo = &slo;
        hi = &shi;
    }
    if (!rc) rc = decide_rounding(&bits, &x->entries[slot], &none, &x->out, x->rnd, x->ovf, lo, hi, last);
    if (!rc && !none) {
        bytes_store(x->enc + slot * x->size, x->size, &bits);
        x->status[slot] = 0;
    }
    big_free(&bits);
    dyad_free(&slo);
    dyad_free(&shi);
    return rc;
}

void mref_free(mref *x) {
    if (x->A)
        for (size_t a = 0; a < x->count * (size_t)x->parts * x->n * x->n; a++) dyad_free(&x->A[a]);
    free(x->A);
    free(x->finite);
    free(x->enc);
    free(x->status);
    free(x->entries);
    free(x->reach);
    free(x->scale);
    mexp_free(&x->ex);
    imat_free(&x->M);
    big_free(&x->out_nan[0]);
    big_free(&x->out_nan[1]);
    memset(x, 0, sizeof *x);
}

/* Room for count matrices n x n of `parts` parts (x->A, set by the caller), the wanted rectangle, the output part format
   and modes, and the precision loop's first and last precisions. Every wanted part starts undecided (status 1). */
int mref_init(mref *x, size_t count, size_t n, int parts, size_t r0, size_t r1, size_t c0, size_t c1, const floatfmt *out, size_t size, int rnd,
              int ovf, int64_t start, int64_t cap) {
    memset(x, 0, sizeof *x);
    imat_clear(&x->M);
    mexp_clear(&x->ex);
    x->count = count;
    x->n = n;
    x->parts = parts;
    x->r0 = r0;
    x->r1 = r1;
    x->c0 = c0;
    x->c1 = c1;
    big_init(&x->out_nan[0]);
    big_init(&x->out_nan[1]);
    if (floatfmt_copy(&x->out, x->out_nan, out)) return EXACTREF_NOMEM;
    x->rnd = rnd;
    x->ovf = ovf;
    x->size = size;
    x->start = start;
    x->cap = cap;
    x->whole = n <= MEXP_SMALL;
    size_t total = count * (size_t)parts * n * n, slots = count * mref_wanted(x) * (size_t)parts;
    x->A = malloc((total + 1) * sizeof *x->A);
    x->finite = calloc(count + 1, 1);
    x->enc = calloc(slots * x->size + 1, 1);
    x->status = malloc(slots + 1);
    x->entries = calloc(slots + 1, sizeof *x->entries);
    x->scale = calloc(count + 1, sizeof *x->scale);
    if (!x->A || !x->finite || !x->enc || !x->status || !x->entries || !x->scale) return EXACTREF_NOMEM;
    for (size_t a = 0; a < total; a++) dyad_init(&x->A[a]);
    memset(x->status, 1, slots);
    return 0;
}

/* whether matrix m's every part (or its imaginary parts) is 0 off the diagonal (or everywhere) */
static int mref_diagonal(const mref *x, size_t m) {
    for (int p = 0; p < x->parts; p++)
        for (size_t i = 0; i < x->n; i++)
            for (size_t j = 0; j < x->n; j++)
                if (i != j && mref_at(x, m, p, i, j)->m.n) return 0;
    return 1;
}

static int mref_imaginary_zero(const mref *x, size_t m) {
    if (x->parts < 2) return 0;
    for (size_t i = 0; i < x->n; i++)
        for (size_t j = 0; j < x->n; j++)
            if (mref_at(x, m, 1, i, j)->m.n) return 0;
    return 1;
}

/* Matrix m past the kernel's norm range: a diagonal matrix's pending entries are exp of its diagonal, e**a or
   e**a (cos b + i sin b) (any exponent), e**a sin 0 an exact zero; the others have no reference (status 2) */
static int mref_past(mref *x, size_t m) {
    int diagonal = mref_diagonal(x, m), P = x->P, rc = 0;
    for (size_t i = x->r0; !rc && i < x->r1; i++)
        for (size_t j = x->c0; !rc && j < x->c1; j++)
            for (int p = 0; !rc && p < x->parts; p++) {
                size_t slot = mref_slot(x, m, i, j, p);
                if (!x->status[slot]) continue;
                if (!diagonal || i != j) {
                    x->status[slot] = 2;
                    continue;
                }
                if (p == 1 && !mref_at(x, m, 1, i, i)->m.n) {
                    rat zero;
                    rat_init(&zero);
                    rc = mref_rational(x, slot, &zero);
                    rat_free(&zero);
                    continue;
                }
                for (int64_t prec = x->start; !rc && x->status[slot]; prec = 2 * prec < x->cap ? 2 * prec : x->cap) {
                    dyad lo, hi, c[2], sn[2], end[2];
                    bigint shift;
                    dyad_init(&lo);
                    dyad_init(&hi);
                    for (int k = 0; k < 2; k++) {
                        dyad_init(&c[k]);
                        dyad_init(&sn[k]);
                        dyad_init(&end[k]);
                    }
                    big_init(&shift);
                    rc = exp_bounds(&lo, &hi, &shift, mref_at(x, m, 0, i, i), prec);
                    if (!rc && P == 2 && !(rc = cos_sin_bounds(c, sn, mref_at(x, m, 1, i, i), prec))) {
                        /* e**a times cos b or sin b: [lo, hi] [f], lo > 0, rounded outward */
                        const dyad *f = p ? sn : c;
                        if (!(rc = dyad_mul(&end[0], dyad_sign(&f[0]) >= 0 ? &lo : &hi, &f[0], prec, 0)))
                            rc = dyad_mul(&end[1], dyad_sign(&f[1]) >= 0 ? &hi : &lo, &f[1], prec, 1);
                        if (!rc && !(rc = dyad_copy(&lo, &end[0]))) rc = dyad_copy(&hi, &end[1]);
                    }
                    if (!rc) { /* the scale 2**shift, clamped to 2**+-2**61: beyond every format's and entry's range */
                        int64_t b = big_bits(&shift), e = b > 61 ? (shift.neg ? -((int64_t)1 << 61) : (int64_t)1 << 61) : (shift.neg ? -(int64_t)shift.w[0] : (b ? (int64_t)shift.w[0] : 0));
                        if (!(rc = dyad_shift(&lo, e)) && !(rc = dyad_shift(&hi, e))) rc = mref_interval(x, slot, &lo, &hi, prec >= x->cap);
                    }
                    if (!rc && x->status[slot] && prec >= x->cap) x->status[slot] = 1; /* needs more than the cap */
                    dyad_free(&lo);
                    dyad_free(&hi);
                    for (int k = 0; k < 2; k++) {
                        dyad_free(&c[k]);
                        dyad_free(&sn[k]);
                        dyad_free(&end[k]);
                    }
                    big_free(&shift);
                    if (prec >= x->cap) break;
                }
            }
    return rc;
}

/* Matrix m's start: a non-finite entry makes it NaN throughout; with every imaginary part 0 the imaginary parts are exact
   zeros; entries no walk reaches are exact zeros. *more when the precision loop must run. x->reach is set. */
static int mref_start(mref *x, size_t m, int finite, int *more) {
    size_t n = x->n, words = (n + 63) / 64;
    int P = mref_imaginary_zero(x, m) ? 1 : x->parts, rc = 0;
    *more = 0;
    for (size_t i = x->r0; !rc && i < x->r1; i++)
        for (size_t j = x->c0; !rc && j < x->c1; j++)
            for (int p = 0; !rc && p < x->parts; p++) {
                size_t slot = mref_slot(x, m, i, j, p);
                if (!finite) {
                    rc = mref_kind(x, slot, VALUE_NAN, 0);
                } else if (p >= P) {
                    rat zero;
                    rat_init(&zero);
                    rc = mref_rational(x, slot, &zero);
                    rat_free(&zero);
                }
            }
    if (rc || !finite) return rc;
    free(x->reach);
    x->reach = calloc(n * words + 1, sizeof *x->reach);
    if (!x->reach) return EXACTREF_NOMEM;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (int p = 0; p < P; p++)
                if (mref_at(x, m, p, i, j)->m.n) x->reach[i * words + j / 64] |= (uint64_t)1 << (j % 64);
    for (size_t k = 0; k < n; k++)
        for (size_t i = 0; i < n; i++)
            if (x->reach[i * words + k / 64] >> (k % 64) & 1)
                for (size_t w = 0; w < words; w++) x->reach[i * words + w] |= x->reach[k * words + w];
    rat zero;
    rat_init(&zero);
    for (size_t i = x->r0; !rc && i < x->r1; i++)
        for (size_t j = x->c0; !rc && j < x->c1; j++)
            if (i != j && !(x->reach[i * words + j / 64] >> (j % 64) & 1))
                for (int p = 0; !rc && p < P; p++) rc = mref_rational(x, mref_slot(x, m, i, j, p), &zero);
    rat_free(&zero);
    x->P = P;
    for (size_t s = 0; !rc && s < mref_wanted(x) * (size_t)x->parts; s++)
        if (x->status[m * mref_wanted(x) * (size_t)x->parts + s]) *more = 1;
    return rc;
}

/* matrix m at precision p: its exponential prepared at the limbs for p plus the extra bits; *bounded 0 where the kernel
   does not run it: past its norm bound's range (the one range test, mexp_init's; its pending entries then by mref_past),
   else past the products' 128-bit sums (status 3) */
static int mref_exp_init(mref *x, size_t m, int *bounded) {
    size_t n = x->n;
    int L = (int)((x->p + x->extra + MAT_LIMB - 1) / MAT_LIMB), rc;
    if (L > MAT_LIMBS_MAX) L = MAT_LIMBS_MAX;
    imat_free(&x->M);
    mexp_free(&x->ex);
    if ((rc = imat_init(&x->M, n, n, L, x->P))) return rc;
    for (size_t i = 0; !rc && i < n; i++)
        for (size_t j = 0; !rc && j < n; j++)
            for (int p = 0; !rc && p < x->P; p++) rc = imat_set(&x->M, p, i, j, mref_at(x, m, p, i, j), NULL);
    if (!rc) rc = mexp_init(&x->ex, &x->M, bounded);
    imat_free(&x->M); /* the exponential holds its copy */
    if (!rc && !*bounded) {
        if (x->ex.past) return mref_past(x, m);
        for (size_t i = x->r0; i < x->r1; i++)
            for (size_t j = x->c0; j < x->c1; j++)
                for (int p = 0; p < x->parts; p++)
                    if (x->status[mref_slot(x, m, i, j, p)]) x->status[mref_slot(x, m, i, j, p)] = 3;
    }
    return rc;
}

/* the least t with every row sum of |part| of matrix m over rows r0 .. r1 - 1 and columns c0 .. c1 - 1 below 2**t
   (INT64_MIN for no part nonzero) */
static int mref_row_top(int64_t *t, const mref *x, size_t m, size_t r0, size_t r1, size_t c0, size_t c1) {
    int rc = 0;
    *t = INT64_MIN;
    for (size_t i = r0; !rc && i < r1; i++) {
        dyad sum, a;
        dyad_init(&sum);
        dyad_init(&a);
        for (size_t j = c0; !rc && j < c1; j++)
            for (int p = 0; !rc && p < x->parts; p++)
                if (!(rc = dyad_abs(&a, mref_at(x, m, p, i, j)))) rc = dyad_add_exact(&sum, &sum, &a);
        int64_t b = sum.m.n ? sum.e + big_bits(&sum.m) : INT64_MIN;
        if (b > *t) *t = b;
        dyad_free(&sum);
        dyad_free(&a);
    }
    return rc;
}

/* Matrix m as the block [[A, E], [0, A]] (n = 2N, its wanted parts the top-right block L(A, E), linear in E): E
   scaled by 2**-k, exactly, and x->scale[m] = k, so that E's largest row sum of |part| lies in the binade of
   max(A's, 1); where A's reaches 2**57, E is only scaled down, below 2**57, so that the block's row sums stay below
   A's plus 2**57 (matrix_exp_frechet.md, The reference output) */
int mref_balance(mref *x, size_t m) {
    size_t N = x->n / 2;
    int64_t ta, te;
    int rc = mref_row_top(&ta, x, m, 0, N, 0, N);
    if (!rc) rc = mref_row_top(&te, x, m, 0, N, N, x->n);
    if (rc || te == INT64_MIN) return rc; /* E = 0 */
    int64_t k = ta > 57 ? (te > 57 ? te - 57 : 0) : te - (ta > 1 ? ta : 1);
    for (int p = 0; !rc && p < x->parts; p++)
        for (size_t i = 0; !rc && i < N; i++)
            for (size_t j = N; !rc && j < x->n; j++) rc = dyad_shift(&x->A[((m * (size_t)x->parts + (size_t)p) * x->n + i) * x->n + j], -k);
    if (!rc) x->scale[m] = k;
    return rc;
}

/* the extra bits: 4 bitlen(n) + 32 plus the norm's integer bits, at most 61 of them, as the kernel runs no matrix whose
   norm reaches 2**60 (matrix_exp.md, Ziv refinement) */
static int64_t mref_extra(const mref *x, size_t m) {
    int64_t top = 0;
    mref_row_top(&top, x, m, 0, x->n, 0, x->n);
    if (top < 0) top = 0;
    if (top > 61) top = 61;
    int bits = 0;
    for (size_t v = x->n; v; v >>= 1) bits++;
    return 4 * bits + 32 + top;
}

/* rows i0 .. i1 of the wanted entries from the exponential's result, at precision x->p */
static int mref_decide(mref *x, size_t m, size_t i0, size_t i1) {
    const imat *E = mexp_result(&x->ex);
    int last = x->p >= x->cap, rc = 0;
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    for (size_t i = i0; !rc && i < i1; i++)
        for (size_t j = x->c0; !rc && j < x->c1; j++)
            for (int p = 0; !rc && p < x->P; p++) {
                size_t slot = mref_slot(x, m, i, j, p);
                if (!x->status[slot]) continue;
                if (!(rc = imat_ends(&lo, &hi, E, p, i, j))) rc = mref_interval(x, slot, &lo, &hi, last);
            }
    dyad_free(&lo);
    dyad_free(&hi);
    return rc;
}

/* the rationality test for matrix m's undecided parts: a rational one is rounded once from its exact value */
static int mref_rational_parts(mref *x, size_t m) {
    size_t n = x->n, nn = n * n, count = 0, cap = mref_wanted(x) * (size_t)x->P;
    size_t *wanted = malloc((3 * cap + 1) * sizeof *wanted), *slots = malloc((cap + 1) * sizeof *slots);
    int *rational = calloc(cap + 1, sizeof *rational);
    rat *values = malloc((cap + 1) * sizeof *values);
    int rc = !wanted || !slots || !rational || !values ? EXACTREF_NOMEM : 0;
    for (size_t i = x->r0; !rc && i < x->r1; i++)
        for (size_t j = x->c0; j < x->c1; j++)
            for (int p = 0; p < x->P; p++) {
                size_t slot = mref_slot(x, m, i, j, p);
                if (!x->status[slot]) continue;
                wanted[3 * count] = i;
                wanted[3 * count + 1] = j;
                wanted[3 * count + 2] = (size_t)p;
                slots[count++] = slot;
            }
    for (size_t a = 0; values && a < count; a++) rat_init(&values[a]);
    if (!rc && count) rc = mexp_rational(rational, values, n, x->P, &x->A[m * (size_t)x->parts * nn], wanted, count);
    for (size_t a = 0; !rc && a < count; a++)
        if (rational[a]) rc = mref_rational(x, slots[a], &values[a]);
    for (size_t a = 0; values && a < count; a++) rat_free(&values[a]);
    free(wanted);
    free(slots);
    free(rational);
    free(values);
    return rc;
}

static int mref_pending(const mref *x, size_t m) {
    size_t w = mref_wanted(x) * (size_t)x->parts;
    for (size_t s = 0; s < w; s++)
        if (x->status[m * w + s] == 1) return 1;
    return 0;
}

static uint64_t sat_mul(uint64_t a, uint64_t b) { return a && b > UINT64_MAX / a ? UINT64_MAX : a * b; }

/* *now when matrix m's rationality test is due (matrix_exp.md, The reference output): at the latest one attempt
   before the cap (the last attempt decides what is left), or once its cost is at most the next precision step's, both
   in units of about 0.5 to 0.8 ns (measured): a step (n**3 per product, plus n**2 for the decisions) L**2 for L-limb
   entries, the next step four times this one; the test 64 M**4 ceil(B / 64) for the largest vertex set of an undecided
   part (M its vertices times the parts, B the bits A's entries span; M**4 ceil(B / 64) measured at 50 to 130 ns) */
static int mref_test_now(int *now, const mref *x, size_t m) {
    *now = 2 * x->p >= x->cap; /* at the latest before the last attempt, which decides what is left */
    if (*now) return 0;
    size_t n = x->n, nn = n * n, words = (n + 63) / 64, most = 0;
    uint64_t *col = malloc((words + 1) * sizeof *col);
    if (!col) return EXACTREF_NOMEM;
    int64_t top = INT64_MIN, low = INT64_MAX;
    for (int p = 0; p < x->parts; p++)
        for (size_t at = 0; at < nn; at++) {
            const dyad *a = &x->A[(m * (size_t)x->parts + (size_t)p) * nn + at];
            if (!a->m.n) continue;
            int64_t t = a->e + big_bits(&a->m);
            if (t > top) top = t;
            if (a->e < low) low = a->e;
        }
    for (size_t j = x->c0; j < x->c1; j++) {
        int any = 0;
        for (size_t i = x->r0; !any && i < x->r1; i++)
            for (int p = 0; p < x->P; p++) any = any || x->status[mref_slot(x, m, i, j, p)] == 1;
        if (!any) continue;
        /* column j of the closure: the vertices with a walk to j */
        for (size_t w = 0; w < words; w++) col[w] = 0;
        for (size_t v = 0; v < n; v++)
            if (x->reach[v * words + j / 64] >> (j % 64) & 1) col[v / 64] |= (uint64_t)1 << (v % 64);
        for (size_t i = x->r0; i < x->r1; i++) {
            int undecided = 0;
            for (int p = 0; p < x->P; p++) undecided = undecided || x->status[mref_slot(x, m, i, j, p)] == 1;
            if (!undecided) continue;
            size_t k = 0; /* the vertex set {i, j} and every v on a walk from i to j */
            for (size_t w = 0; w < words; w++) {
                uint64_t b = x->reach[i * words + w] & col[w];
                if (w == i / 64) b |= (uint64_t)1 << (i % 64);
                if (w == j / 64) b |= (uint64_t)1 << (j % 64);
                for (; b; b &= b - 1) k++;
            }
            if (k > most) most = k;
        }
    }
    free(col);
    uint64_t M = (uint64_t)most * (uint64_t)x->P, B = top > low ? (uint64_t)(top - low + 63) / 64 : 1, L = (uint64_t)x->ex.L * (uint64_t)x->ex.parts;
    uint64_t test = sat_mul(sat_mul(sat_mul(sat_mul(64, M), M), sat_mul(M, M)), B);
    uint64_t prod = sat_mul(mexp_products(&x->ex), sat_mul(sat_mul(n, n), n)), work = prod + (uint64_t)nn < prod ? UINT64_MAX : prod + (uint64_t)nn;
    uint64_t step = sat_mul(sat_mul(4, work), sat_mul(L, L)); /* the products and the n**2 decisions: nonzero without products */
    *now = test <= step;
    return 0;
}

/* one matrix's whole reference on this thread */
static int mref_matrix(mref *x, size_t m) {
    int more, rc = mref_start(x, m, x->finite[m], &more), bounded;
    if (rc || !more) return rc;
    x->extra = mref_extra(x, m);
    x->tested = 0;
    for (x->p = x->start; !rc; x->p = 2 * x->p < x->cap ? 2 * x->p : x->cap) {
        if ((rc = mref_exp_init(x, m, &bounded))) break;
        if (!bounded) break; /* past the kernel's range: decided by mref_exp_init */
        mexp_run_all(&x->ex);
        if ((rc = mref_decide(x, m, x->r0, x->r1)) || !mref_pending(x, m)) break;
        int now = 0;
        if (!x->tested && !(rc = mref_test_now(&now, x, m)) && now) { /* once, when due */
            x->tested = 1;
            if ((rc = mref_rational_parts(x, m)) || !mref_pending(x, m)) break;
        }
        if (rc || x->p >= x->cap) break;
    }
    return rc;
}

/* The tiles of the next phase (0 when done). Small matrices: one phase, tiles of MEXP_GROUP whole references. Larger
   ones: per matrix its exponential's phases at each precision, then its decisions in row tiles, then (once, when due)
   its rationality test in one tile. */
int mref_next(mref *x, size_t *tiles) {
    *tiles = 0;
    if (x->whole) {
        if (!x->started) {
            x->started = 1;
            *tiles = (x->count + MEXP_GROUP - 1) / MEXP_GROUP;
        }
        return 0;
    }
    int rc = 0, bounded, more;
    for (;;) {
        switch (x->stage) {
        case MREF_START:
            if (x->mi >= x->count) return 0;
            if ((rc = mref_start(x, x->mi, x->finite[x->mi], &more))) return rc;
            if (!more) {
                x->mi++;
                continue;
            }
            x->extra = mref_extra(x, x->mi);
            x->tested = 0;
            x->p = x->start;
            x->stage = MREF_EXP;
            x->k = 0;
            if ((rc = mref_exp_init(x, x->mi, &bounded))) return rc;
            if (!bounded) {
                x->stage = MREF_START;
                x->mi++;
                continue;
            }
            if ((*tiles = mexp_tiles(&x->ex, 0))) return 0;
            continue; /* no phases (a diagonal matrix): straight to the decisions */
        case MREF_EXP:
            mexp_done(&x->ex, x->k++);
            if ((*tiles = mexp_tiles(&x->ex, x->k))) return 0;
            x->stage = MREF_DECIDE;
            *tiles = (x->r1 - x->r0 + MEXP_TILE_ROWS - 1) / MEXP_TILE_ROWS;
            if (*tiles) return 0;
            continue;
        case MREF_DECIDE: /* the decisions at precision p are in */
        case MREF_RATIONAL: /* or the rationality test's */
            if (!mref_pending(x, x->mi)) {
                x->stage = MREF_START;
                x->mi++;
                continue;
            }
            if (!x->tested) { /* once, when due */
                int now;
                if ((rc = mref_test_now(&now, x, x->mi))) return rc;
                if (now) {
                    x->tested = 1;
                    x->stage = MREF_RATIONAL;
                    *tiles = 1;
                    return 0;
                }
            }
            if (x->p >= x->cap) {
                x->stage = MREF_START;
                x->mi++;
                continue;
            }
            x->p = 2 * x->p < x->cap ? 2 * x->p : x->cap;
            x->stage = MREF_EXP;
            x->k = 0;
            if ((rc = mref_exp_init(x, x->mi, &bounded))) return rc;
            if (!bounded) { /* past the kernel's range: decided by mref_exp_init */
                x->stage = MREF_START;
                x->mi++;
                continue;
            }
            if ((*tiles = mexp_tiles(&x->ex, 0))) return 0;
            continue;
        }
    }
}

/* runs tile t of the current phase */
int mref_tile(mref *x, size_t t) {
    if (x->whole) {
        int rc = 0;
        for (size_t m = t * MEXP_GROUP; !rc && m < x->count && m < (t + 1) * MEXP_GROUP; m++) {
            mref one = *x; /* this matrix's own working state; the outputs and inputs are shared */
            imat_clear(&one.M);
            mexp_clear(&one.ex);
            one.reach = NULL;
            rc = mref_matrix(&one, m);
            imat_free(&one.M);
            mexp_free(&one.ex);
            free(one.reach);
        }
        return rc;
    }
    switch (x->stage) {
    case MREF_EXP: mexp_run(&x->ex, x->k, t); return 0;
    case MREF_DECIDE: {
        size_t i0 = x->r0 + t * MEXP_TILE_ROWS, i1 = i0 + MEXP_TILE_ROWS < x->r1 ? i0 + MEXP_TILE_ROWS : x->r1;
        return mref_decide(x, x->mi, i0, i1);
    }
    case MREF_RATIONAL: return mref_rational_parts(x, x->mi);
    }
    return 0;
}

/* ---------------------------------------------------------------- the tolerance (matrix_exp.md, The tolerance) */

/* x rounded to 64 bits upward (up) or downward */
static int round64(dyad *r, const dyad *x, int up) { return dyad_rnd(r, x, 64, up); }

/* max(|lo|, |hi|) of a part's interval */
static int part_magnitude(dyad *r, const dyad *lo, const dyad *hi) {
    dyad a, b;
    dyad_init(&a);
    dyad_init(&b);
    int rc = dyad_abs(&a, lo);
    if (!rc) rc = dyad_abs(&b, hi);
    if (!rc) rc = dyad_copy(r, dyad_cmp(&a, &b) >= 0 ? &a : &b);
    dyad_free(&a);
    dyad_free(&b);
    return rc;
}

/* entry (p, i, j) of M = the interval [lo, hi]: midpoint (lo + hi) / 2 and radius (hi - lo) / 2, exactly */
static int imat_set_interval(imat *M, int p, size_t i, size_t j, const dyad *lo, const dyad *hi) {
    dyad mid, rad;
    dyad_init(&mid);
    dyad_init(&rad);
    int rc = dyad_add_exact(&mid, lo, hi);
    if (!rc) rc = dyad_shift(&mid, -1);
    if (!rc && !(rc = dyad_neg(&rad, lo)) && !(rc = dyad_add_exact(&rad, &rad, hi))) rc = dyad_shift(&rad, -1);
    if (!rc) rc = imat_set(M, p, i, j, &mid, &rad);
    dyad_free(&mid);
    dyad_free(&rad);
    return rc;
}

/* |entry| as an upper bound: |midpoint| + radius, rounded up to 64 bits */
static int imat_upper(dyad *r, const imat *M, int p, size_t i, size_t j) {
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    int rc = imat_ends_round(&lo, &hi, M, p, i, j, 64); /* the larger end's magnitude rounded up to 64 bits, as |midpoint| + radius */
    if (!rc) rc = part_magnitude(r, &lo, &hi);
    dyad_free(&lo);
    dyad_free(&hi);
    return rc;
}

enum { JOB_PREPARE, JOB_EXP, JOB_COLS, JOB_PIECE, JOB_MERGE, JOB_SCALARS, JOB_S, JOB_FINISH };

static int mtol_add_job(mtol *x, int kind, int a, size_t b) {
    if (x->njobs == x->capjobs) {
        size_t cap = x->capjobs ? 2 * x->capjobs : 64;
        mtol_job *jobs = realloc(x->jobs, cap * sizeof *jobs);
        if (!jobs) return EXACTREF_NOMEM;
        x->jobs = jobs;
        x->capjobs = cap;
    }
    x->jobs[x->njobs].kind = kind;
    x->jobs[x->njobs].a = a;
    x->jobs[x->njobs].b = b;
    x->njobs++;
    return 0;
}

static void mtol_work_free(mtol *x) {
    for (int k = 0; k < MTOL_EXPS; k++) mexp_free(&x->ex[k]);
    imat *ms[] = {&x->box, &x->center, &x->A0, &x->Win, &x->Bin, &x->Hin, &x->G, &x->Gn, &x->GB, &x->H, &x->Hn, &x->HA, &x->Lq, &x->bq, &x->Q, &x->Sin, &x->Cols, &x->S};
    for (size_t k = 0; k < sizeof ms / sizeof *ms; k++) imat_free(ms[k]);
    imat_lines_free(&x->rows);
    imat_lines_free(&x->colsB);
    imat_lines_free(&x->colsA);
    imat_lines_free(&x->colsW);
    imat_lines_free(&x->srows);
    imat_lines_free(&x->scols);
    if (x->stats)
        for (size_t a = 0; a < x->nstats; a++) dyad_free(&x->stats[a]);
    free(x->stats);
    free(x->jobs);
    dyad *ds[] = {&x->norm, &x->r, &x->spread, &x->rho_h, &x->half_h, &x->rest};
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_free(ds[k]);
    x->stats = NULL;
    x->jobs = NULL;
    x->njobs = x->capjobs = x->job = 0;
    for (int k = 0; k < MTOL_EXPS; k++) mexp_clear(&x->ex[k]);
    for (size_t k = 0; k < sizeof ms / sizeof *ms; k++) imat_clear(ms[k]);
    imat_lines_clear(&x->rows);
    imat_lines_clear(&x->colsB);
    imat_lines_clear(&x->colsA);
    imat_lines_clear(&x->colsW);
    imat_lines_clear(&x->srows);
    imat_lines_clear(&x->scols);
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_init(ds[k]);
}

void mtol_free(mtol *x) {
    mtol_work_free(x);
    if (x->A)
        for (size_t a = 0; a < x->count * (size_t)x->parts * x->n * x->n; a++) dyad_free(&x->A[a]);
    free(x->A);
    if (x->R)
        for (size_t a = 0; a < x->count * (size_t)x->parts * x->n * x->n; a++) dyad_free(&x->R[a]);
    free(x->R);
    free(x->finite);
    free(x->bounded);
    free(x->refs);
    free(x->tol);
    free(x->low);
    free(x->high);
    free(x->flags);
    lean_free(&x->held);
    dyad_free(&x->allowance);
    for (int k = 0; k < 2; k++) {
        big_free(&x->F_nan[k]);
        big_free(&x->out_nan[k]);
        big_free(&x->f64_nan[k]);
    }
    memset(x, 0, sizeof *x);
}

/* Room for count matrices n x n (x->A set by the caller, x->R too when radii is set), the wanted rectangle, the compute
   format F (and whether it may flush), the allowance c, the output part format, rounding and overflow, float64, and
   the reference's encodings of the wanted parts (refs, size bytes each). */
int mtol_init(mtol *x, size_t count, size_t n, int parts, int radii, size_t r0, size_t r1, size_t c0, size_t c1, const floatfmt *F, int flush,
              const bigint *allow_num, const bigint *allow_den, const floatfmt *out, size_t size, int rnd, int ovf, const floatfmt *f64,
              const unsigned char *refs) {
    memset(x, 0, sizeof *x);
    x->count = count;
    x->n = n;
    x->parts = parts;
    x->r0 = r0;
    x->r1 = r1;
    x->c0 = c0;
    x->c1 = c1;
    for (int k = 0; k < 2; k++) {
        big_init(&x->F_nan[k]);
        big_init(&x->out_nan[k]);
        big_init(&x->f64_nan[k]);
    }
    if (floatfmt_copy(&x->F, x->F_nan, F) || floatfmt_copy(&x->out, x->out_nan, out) || floatfmt_copy(&x->f64, x->f64_nan, f64)) return EXACTREF_NOMEM;
    x->flush = flush;
    x->size = size;
    x->rnd = rnd;
    x->ovf = ovf;
    x->whole = n <= MEXP_SMALL;
    lean_init(&x->held);
    dyad_init(&x->allowance);
    dyad *ds[] = {&x->norm, &x->r, &x->spread, &x->rho_h, &x->half_h, &x->rest};
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_init(ds[k]);
    for (int k = 0; k < MTOL_EXPS; k++) mexp_clear(&x->ex[k]);
    imat *ms[] = {&x->box, &x->center, &x->A0, &x->Win, &x->Bin, &x->Hin, &x->G, &x->Gn, &x->GB, &x->H, &x->Hn, &x->HA, &x->Lq, &x->bq, &x->Q, &x->Sin, &x->Cols, &x->S};
    for (size_t k = 0; k < sizeof ms / sizeof *ms; k++) imat_clear(ms[k]);
    imat_lines_clear(&x->rows);
    imat_lines_clear(&x->colsB);
    imat_lines_clear(&x->colsA);
    imat_lines_clear(&x->colsW);
    imat_lines_clear(&x->srows);
    imat_lines_clear(&x->scols);
    size_t total = count * (size_t)parts * n * n, slots = count * (r1 - r0) * (c1 - c0) * (size_t)parts;
    x->A = malloc((total + 1) * sizeof *x->A);
    x->R = radii ? malloc((total + 1) * sizeof *x->R) : NULL;
    x->finite = calloc(count + 1, 1);
    x->bounded = calloc(count + 1, 1);
    x->refs = malloc(slots * size + 1);
    x->tol = calloc(slots * 8 + 1, 1);
    x->low = calloc(slots * size + 1, 1);
    x->high = calloc(slots * size + 1, 1);
    x->flags = calloc(slots + 1, 1);
    if (!x->A || (radii && !x->R) || !x->finite || !x->bounded || !x->refs || !x->tol || !x->low || !x->high || !x->flags) return EXACTREF_NOMEM;
    for (size_t a = 0; a < total; a++) dyad_init(&x->A[a]);
    for (size_t a = 0; radii && a < total; a++) dyad_init(&x->R[a]);
    memcpy(x->refs, refs, slots * size);
    /* the factor c (n + 16) / 2**p, rounded up to 64 bits once (c = allow_num / allow_den) */
    dyad num, den;
    dyad_init(&num);
    dyad_init(&den);
    bigint k;
    big_init(&k);
    int rc = big_set_word(&k, (uint64_t)n + 16, 0);
    if (!rc && !(rc = big_mul(&num.m, allow_num, &k)) && !(rc = big_copy(&den.m, allow_den))) {
        num.e = num.m.n ? -((int64_t)F->man_bits + 1) : 0;
        den.e = 0;
        rc = dyad_div(&x->allowance, &num, &den, 64, 1);
    }
    dyad_free(&num);
    dyad_free(&den);
    big_free(&k);
    /* Lemma 1 at F: one rounding to nearest, half an ulp, flushing with the option */
    round_to rs;
    memset(&rs, 0, sizeof rs);
    rs.active = 1;
    rs.rnd = ROUND_NEAREST_EVEN;
    rs.f = *F;
    rs.ulps_num = 1;
    rs.ulps_den = 2;
    if (!rc) rc = lean_of(&x->held, &rs, flush);
    x->w = 2 * ((int64_t)F->man_bits + 1 > (int64_t)out->man_bits + 1 ? (int64_t)F->man_bits + 1 : (int64_t)out->man_bits + 1) + 64;
    return rc;
}

static size_t mtol_slot(const mtol *x, size_t m, size_t i, size_t j, int part) {
    return ((m * (x->r1 - x->r0) + (i - x->r0)) * (x->c1 - x->c0) + (j - x->c0)) * (size_t)x->parts + (size_t)part;
}

/* entry part (p, i, j) of matrix m as [lo, hi]: its value widened by its radius (rounded up to 64 bits, the ends outward
   at w), then with flushing a part that may be subnormal in F widened to include 0; t is scratch */
static int mtol_point(dyad *lo, dyad *hi, dyad *t, const mtol *x, size_t m, int p, size_t i, size_t j) {
    size_t at = ((m * (size_t)x->parts + (size_t)p) * x->n + i) * x->n + j;
    int rc = dyad_copy(lo, &x->A[at]);
    if (!rc) rc = dyad_copy(hi, &x->A[at]);
    if (!rc && x->R && x->R[at].m.n && !(rc = round64(t, &x->R[at], 1)) && !(rc = dyad_add(hi, hi, t, x->w, 1)) && !(rc = dyad_neg(t, t)))
        rc = dyad_add(lo, lo, t, x->w, 0);
    if (!rc && x->flush) {
        if (dyad_sign(lo) > 0 && lo->e + big_bits(&lo->m) <= x->F.emin) rc = dyad_set_si(lo, 0, 0);
        else if (dyad_sign(hi) < 0 && hi->e + big_bits(&hi->m) <= x->F.emin) rc = dyad_set_si(hi, 0, 0);
    }
    return rc;
}

/* Lemma 3 for one wanted part: held = [lo, hi] in F, or NULL for no bound */
static int mtol_put(mtol *x, size_t slot, const dyad *lo, const dyad *hi, int possible) {
    bigint tol, low, high, ref;
    big_init(&tol);
    big_init(&low);
    big_init(&high);
    big_init(&ref);
    int flags, rc = bytes_load(&ref, x->refs + slot * x->size, x->size, 0);
    if (!rc) rc = final_bounds(&tol, &low, &high, &flags, &x->out, &x->f64, x->rnd, x->ovf, lo, hi, &ref, 128, possible);
    if (!rc) {
        bytes_store(x->tol + slot * 8, 8, &tol);
        bytes_store(x->low + slot * x->size, x->size, &low);
        bytes_store(x->high + slot * x->size, x->size, &high);
        x->flags[slot] = (unsigned char)flags;
    }
    big_free(&tol);
    big_free(&low);
    big_free(&high);
    big_free(&ref);
    return rc;
}

/* matrix m's start: no bound for non-finite radii (every flag) or entries; else the points, the norm, r, the box, the
   center and the ball's W and B, and the jobs that follow */
static int mtol_prepare(mtol *x, size_t m) {
    size_t n = x->n;
    int P = x->parts, rc = 0;
    int64_t w = x->w;
    if (!x->bounded[m] || !x->finite[m]) {
        for (size_t i = x->r0; !rc && i < x->r1; i++)
            for (size_t j = x->c0; !rc && j < x->c1; j++)
                for (int p = 0; !rc && p < P; p++) rc = mtol_put(x, mtol_slot(x, m, i, j, p), NULL, NULL, x->bounded[m] ? 0 : FLAG_ANY);
        x->skip = 1;
        return rc;
    }
    x->skip = 0;
    dyad t, u, one, plo, phi;
    dyad_init(&t);
    dyad_init(&u);
    dyad_init(&one);
    dyad_init(&plo);
    dyad_init(&phi);
    /* the norm: the largest row sum of the entries' moduli (a complex entry's by its parts' magnitudes), each sum
       rounded up to 64 bits */
    if (!rc) rc = dyad_set_si(&x->norm, 0, 0);
    for (size_t i = 0; !rc && i < n; i++) {
        dyad row;
        dyad_init(&row);
        for (size_t j = 0; !rc && j < n; j++) {
            dyad size, sq;
            dyad_init(&size);
            dyad_init(&sq);
            for (int p = 0; !rc && p < P; p++)
                if (!(rc = mtol_point(&plo, &phi, &u, x, m, p, i, j)) && !(rc = part_magnitude(&t, &plo, &phi)))
                    rc = P == 1 ? dyad_copy(&size, &t) : !(rc = dyad_mul_exact(&t, &t, &t)) ? dyad_add_exact(&sq, &sq, &t) : rc;
            if (!rc && P == 2) rc = dyad_sqrt(&size, &sq, 64, 1);
            if (!rc) rc = dyad_add(&row, &row, &size, 64, 1);
            dyad_free(&size);
            dyad_free(&sq);
        }
        if (!rc && dyad_cmp(&row, &x->norm) > 0) rc = dyad_copy(&x->norm, &row);
        dyad_free(&row);
    }
    int64_t norm_bits = x->norm.m.n ? x->norm.e + big_bits(&x->norm.m) : 0;
    if (norm_bits < 0) norm_bits = 0;
    /* r = c (n + 16) 2**-p max(||A||, 1): the factor (x->allowance) times it, rounded up to 64 bits */
    if (!rc && !(rc = dyad_set_si(&one, 1, 0))) rc = dyad_mul(&x->r, &x->allowance, dyad_cmp(&x->norm, &one) >= 0 ? &x->norm : &one, 64, 1);
    /* the box: each part widened by r at w; its exponential at w plus the norm's bits */
    int Lbox = (int)((w + norm_bits + MAT_LIMB - 1) / MAT_LIMB), bounded = 0;
    if (Lbox > MAT_LIMBS_MAX) Lbox = MAT_LIMBS_MAX;
    if (!rc) rc = imat_init(&x->box, n, n, Lbox, P);
    if (!rc) rc = dyad_neg(&u, &x->r);
    for (int p = 0; !rc && p < P; p++)
        for (size_t i = 0; !rc && i < n; i++)
            for (size_t j = 0; !rc && j < n; j++) {
                dyad blo, bhi;
                dyad_init(&blo);
                dyad_init(&bhi);
                if (!(rc = mtol_point(&plo, &phi, &t, x, m, p, i, j)) && !(rc = dyad_add(&blo, &plo, &u, w, 0)) && !(rc = dyad_add(&bhi, &phi, &x->r, w, 1)))
                    rc = imat_set_interval(&x->box, p, i, j, &blo, &bhi);
                dyad_free(&blo);
                dyad_free(&bhi);
            }
    if (!rc) rc = mexp_init(&x->ex[0], &x->box, &bounded);
    imat_free(&x->box); /* the exponential holds its copy */
    x->has_box = bounded;
    /* the ball, for ||A||_inf <= 256: m pieces, its precision, the centers (exact midpoints), the spread */
    x->has_ball = 0;
    dyad limit;
    dyad_init(&limit);
    if (!rc) rc = dyad_set_si(&limit, MEXP_MAX_PIECES, 0);
    if (!rc && dyad_cmp(&x->norm, &limit) <= 0) {
        x->has_ball = 1;
        /* ceil(norm) */
        int64_t cn;
        {
            bigint q;
            big_init(&q);
            rc = x->norm.e >= 0 ? big_shl(&q, &x->norm.m, x->norm.e) : big_shr(&q, &x->norm.m, -x->norm.e, 1);
            cn = !rc && q.n ? (int64_t)q.w[0] : 0;
            big_free(&q);
        }
        /* m = max(1, min(ceil(4 norm), 256)) */
        {
            dyad four;
            dyad_init(&four);
            bigint q;
            big_init(&q);
            if (!rc && !(rc = dyad_copy(&four, &x->norm)) && !(rc = dyad_shift(&four, 2))) {
                rc = four.e >= 0 ? big_shl(&q, &four.m, four.e) : big_shr(&q, &four.m, -four.e, 1);
            }
            int64_t c4 = !rc && q.n ? (int64_t)q.w[0] : 0;
            x->pieces = (int)(c4 < 1 ? 1 : c4 > MEXP_MAX_PIECES ? MEXP_MAX_PIECES : c4);
            dyad_free(&four);
            big_free(&q);
        }
        int64_t prec = 64 + (145 * cn + 99) / 100;
        int Lball = (int)((prec + MAT_LIMB - 1) / MAT_LIMB), Lc = Lbox;
        x->Lball = Lball;
        if (!rc) rc = imat_init(&x->center, n, n, Lc, P);
        if (!rc) rc = imat_init(&x->A0, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->Win, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->Bin, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->Hin, n, n, Lball, P);
        if (!rc) rc = dyad_set_si(&x->spread, 0, 0);
        dyad mlo, mhi, zero, dm, dm2;
        dyad_init(&mlo);
        dyad_init(&mhi);
        dyad_init(&zero);
        dyad_init(&dm);
        dyad_init(&dm2);
        if (!rc && !(rc = dyad_set_si(&dm, x->pieces, 0))) rc = dyad_set_si(&dm2, 2 * (int64_t)x->pieces, 0);
        /* h / 2 = 1 / (2m), and the rest's bound r >= (s**2 / 2)(1 + 2 s) >= e**s - 1 - s, s = ||A0||_inf h / 2 <= 1/2,
           all rounded up */
        {
            dyad sv, factor;
            dyad_init(&sv);
            dyad_init(&factor);
            if (!rc) rc = dyad_div(&x->half_h, &one, &dm2, 64, 1); /* one holds 1 */
            if (!rc && !(rc = dyad_mul(&sv, &x->norm, &x->half_h, 64, 1)) && !(rc = dyad_copy(&factor, &sv)) && !(rc = dyad_shift(&factor, 1))) { /* 2 s */
                if (!(rc = dyad_add(&factor, &factor, &one, 64, 1)) && !(rc = dyad_mul(&x->rest, &sv, &sv, 64, 1)) && !(rc = dyad_shift(&x->rest, -1))) /* s**2 / 2 */
                    rc = dyad_mul(&x->rest, &x->rest, &factor, 64, 1);
            }
            dyad_free(&sv);
            dyad_free(&factor);
        }
        for (size_t i = 0; !rc && i < n; i++) {
            dyad total;
            dyad_init(&total);
            for (size_t j = 0; !rc && j < n; j++)
                for (int p = 0; !rc && p < P; p++) {
                    dyad c, half;
                    dyad_init(&c);
                    dyad_init(&half);
                    /* the midpoint and half-width, exactly */
                    if (!(rc = mtol_point(&plo, &phi, &t, x, m, p, i, j)) && !(rc = dyad_add_exact(&c, &plo, &phi)) && !(rc = dyad_neg(&half, &plo)) &&
                        !(rc = dyad_add_exact(&half, &half, &phi)) && !(rc = dyad_shift(&c, -1)) && !(rc = dyad_shift(&half, -1))) {
                        rc = dyad_add(&total, &total, &half, 64, 1);
                    }
                    if (!rc) rc = imat_set(&x->center, p, i, j, &c, NULL);
                    if (!rc) rc = imat_set(&x->A0, p, i, j, &c, NULL);
                    /* W's entry c / m, B's the hull of 0 and c / m, and H_0's c / (2m), outward at the ball's precision */
                    if (!rc && !(rc = dyad_div(&mlo, &c, &dm, prec, 0)) && !(rc = dyad_div(&mhi, &c, &dm, prec, 1))) rc = imat_set_interval(&x->Win, p, i, j, &mlo, &mhi);
                    if (!rc) rc = imat_set_interval(&x->Bin, p, i, j, dyad_sign(&mlo) < 0 ? &mlo : &zero, dyad_sign(&mhi) > 0 ? &mhi : &zero);
                    if (!rc && !(rc = dyad_shift(&mlo, -1)) && !(rc = dyad_shift(&mhi, -1))) rc = imat_set_interval(&x->Hin, p, i, j, &mlo, &mhi);
                    dyad_free(&c);
                    dyad_free(&half);
                }
            if (!rc && dyad_cmp(&total, &x->spread) > 0) rc = dyad_copy(&x->spread, &total);
            dyad_free(&total);
        }
        dyad_free(&mlo);
        dyad_free(&mhi);
        dyad_free(&zero);
        dyad_free(&dm);
        dyad_free(&dm2);
        int b1, b2, b3, b4;
        if (!rc) rc = mexp_init(&x->ex[1], &x->center, &b1);
        if (!rc) rc = mexp_init(&x->ex[2], &x->Win, &b2);
        if (!rc) rc = mexp_init(&x->ex[3], &x->Bin, &b3);
        if (!rc) rc = mexp_init(&x->ex[4], &x->Hin, &b4);
        imat_free(&x->center); /* the exponentials hold their copies */
        imat_free(&x->Win);
        imat_free(&x->Bin);
        imat_free(&x->Hin);
        if (!rc && (!b1 || !b2 || !b3 || !b4)) x->has_ball = 0;
        /* the pieces' statistics: per piece its rows' sums and its columns' maxima; the tiles' partial maxima, by each
           construction, and their largest row sums of |H_k| */
        size_t tiles = (n + MEXP_TILE_ROWS - 1) / MEXP_TILE_ROWS;
        x->nstats = (size_t)x->pieces * (2 * n) + 2 * tiles * n + tiles;
        x->tiles = tiles;
        x->stats = malloc((x->nstats + 1) * sizeof *x->stats);
        if (!rc && !x->stats) rc = EXACTREF_NOMEM;
        for (size_t a = 0; x->stats && a < x->nstats; a++) dyad_init(&x->stats[a]);
        if (!rc) rc = imat_init(&x->G, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->Gn, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->GB, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->H, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->Hn, n, n, Lball, P);
        if (!rc) rc = imat_init(&x->HA, n, n, Lball, P);
        if (!rc) rc = imat_lines_init(&x->rows, n, n, Lball, P);
        if (!rc) rc = imat_lines_init(&x->colsB, n, n, Lball, P);
        if (!rc) rc = imat_lines_init(&x->colsA, n, n, Lball, P);
        if (!rc) rc = imat_lines_init(&x->colsW, n, n, Lball, P);
    }
    dyad_free(&limit);
    /* the jobs: the exponentials' phases, then the ball's pieces, scalars and product, then the parts */
    for (int k = 0; !rc && k < MTOL_EXPS; k++) {
        if (k == 0 && !x->has_box) continue;
        if (k > 0 && !x->has_ball) continue;
        for (size_t ph = 0; !rc && ph < x->ex[k].nops; ph++) rc = mtol_add_job(x, JOB_EXP, k, ph);
    }
    if (!rc && x->has_ball) {
        rc = mtol_add_job(x, JOB_COLS, 0, 0);
        for (int k = 0; !rc && k < x->pieces; k++)
            if (!(rc = mtol_add_job(x, JOB_PIECE, k, 0))) rc = mtol_add_job(x, JOB_MERGE, k, 0);
        if (!rc) rc = mtol_add_job(x, JOB_SCALARS, 0, 0);
        if (!rc) rc = mtol_add_job(x, JOB_S, 0, 0);
    }
    if (!rc) rc = mtol_add_job(x, JOB_FINISH, 0, 0);
    dyad_free(&t);
    dyad_free(&u);
    dyad_free(&one);
    dyad_free(&plo);
    dyad_free(&phi);
    return rc;
}

#define ROWSUM(x, k, i) (&(x)->stats[(size_t)(k) * 2 * (x)->n + (i)])
#define COLMAX(x, k, j) (&(x)->stats[(size_t)(k) * 2 * (x)->n + (x)->n + (j)])
#define PARTIAL(x, t, j) (&(x)->stats[(size_t)(x)->pieces * 2 * (x)->n + (t) * (x)->n + (j)])

#define PARTIAL_H(x, t, j) (&(x)->stats[(size_t)(x)->pieces * 2 * (x)->n + ((x)->tiles + (t)) * (x)->n + (j)])
#define HSUM(x, t) (&(x)->stats[(size_t)(x)->pieces * 2 * (x)->n + 2 * (x)->tiles * (x)->n + (t)])

/* e >= |M_ij|, for a complex entry its modulus: sqrt(re**2 + im**2) of the parts' bounds, or |midpoint| + the disk radius
   when smaller, rounded up */
static int entry_upper(dyad *e, const imat *M, int P, size_t n, size_t i, size_t j) {
    dyad v, sq;
    dyad_init(&v);
    dyad_init(&sq);
    int rc = 0;
    for (int p = 0; !rc && p < P; p++)
        if (!(rc = imat_upper(&v, M, p, i, j))) rc = P == 1 ? dyad_copy(e, &v) : !(rc = dyad_mul_exact(&v, &v, &v)) ? dyad_add_exact(&sq, &sq, &v) : rc;
    if (!rc && P == 2) rc = dyad_sqrt(e, &sq, 64, 1);
    if (!rc && P == 2) {
        dyad lo, ms, d;
        dyad_init(&lo);
        dyad_init(&ms);
        dyad_init(&d);
        for (int p = 0; !rc && p < 2; p++) /* the midpoint's parts, squared */
            if (!(rc = imat_mid(&lo, M, p, i, j)) && !(rc = dyad_mul_exact(&lo, &lo, &lo))) rc = dyad_add_exact(&ms, &ms, &lo);
        if (!rc && !(rc = dyad_sqrt(&ms, &ms, 64, 1)) && !(rc = ub_to_dyad(&d, M->rd[i * n + j])) && !(rc = dyad_add(&ms, &ms, &d, 64, 1)) &&
            dyad_cmp(&ms, e) < 0)
            rc = dyad_copy(e, &ms);
        dyad_free(&lo);
        dyad_free(&ms);
        dyad_free(&d);
    }
    dyad_free(&v);
    dyad_free(&sq);
    return rc;
}

/* piece k's rows i0 .. i1 (matrix_exp.md, The ball, step 3), each statistic the smaller of two constructions' bounds:
   - the box: G_k B (G_0 = I: B itself) bounds every entry of exp(t A0) on the piece; G_(k+1) = G_k W (G_1 = W);
   - the centered: with H = H_k, containing exp((k + 1/2) h A0) (H_0 from its exponential), and HA = H A0, every entry is
     at most |H| + (h/2) |HA| plus (|H| |R|)'s, which adds rest times |H|'s row sum to a row's sum and rest times its
     largest row sum to a column's maximum (at the merge); H_(k+1) = H W.
   The rows' sums, and the tile's column maxima by each and its largest row sum of |H|. */
static int mtol_piece_rows(mtol *x, int k, size_t t, size_t i0, size_t i1) {
    size_t n = x->n;
    int P = x->parts, rc = 0;
    const imat *GBm = mexp_result(&x->ex[3]), *H = k > 0 ? &x->H : mexp_result(&x->ex[4]);
    if (k > 0) {
        imat_lines_prepare(&x->rows, &x->G, 0, i0, i1);
        imat_product_rows(&x->GB, &x->rows, &x->colsB, i0, i1);
        GBm = &x->GB;
        if (k + 1 < x->pieces) imat_product_rows(&x->Gn, &x->rows, &x->colsW, i0, i1);
    }
    imat_lines_prepare(&x->rows, H, 0, i0, i1);
    imat_product_rows(&x->HA, &x->rows, &x->colsA, i0, i1);
    if (k + 1 < x->pieces) imat_product_rows(&x->Hn, &x->rows, &x->colsW, i0, i1);
    dyad eb, hv, av, e, sumb, hsum;
    dyad *ds[] = {&eb, &hv, &av, &e, &sumb, &hsum};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    for (size_t j = 0; j < n; j++)
        if (!rc && !(rc = dyad_set_si(PARTIAL(x, t, j), 0, 0))) rc = dyad_set_si(PARTIAL_H(x, t, j), 0, 0);
    if (!rc) rc = dyad_set_si(HSUM(x, t), 0, 0);
    for (size_t i = i0; !rc && i < i1; i++) {
        dyad *sum = ROWSUM(x, k, i);
        if ((rc = dyad_set_si(sum, 0, 0)) || (rc = dyad_set_si(&sumb, 0, 0)) || (rc = dyad_set_si(&hsum, 0, 0))) break;
        for (size_t j = 0; !rc && j < n; j++) {
            if ((rc = entry_upper(&eb, GBm, P, n, i, j)) || (rc = entry_upper(&hv, H, P, n, i, j)) || (rc = entry_upper(&av, &x->HA, P, n, i, j)) ||
                (rc = dyad_mul(&e, &x->half_h, &av, 64, 1)) || (rc = dyad_add_exact(&e, &e, &hv)) || (rc = dyad_add_exact(&sumb, &sumb, &eb)) ||
                (rc = dyad_add_exact(sum, sum, &e)) || (rc = dyad_add_exact(&hsum, &hsum, &hv)))
                break;
            if (dyad_cmp(&eb, PARTIAL(x, t, j)) > 0) rc = dyad_copy(PARTIAL(x, t, j), &eb);
            if (!rc && dyad_cmp(&e, PARTIAL_H(x, t, j)) > 0) rc = dyad_copy(PARTIAL_H(x, t, j), &e);
        }
        if (!rc && !(rc = round64(&hsum, &hsum, 1)) && !(rc = dyad_mul(&e, &x->rest, &hsum, 64, 1)) && !(rc = dyad_add_exact(sum, sum, &e)) &&
            !(rc = round64(sum, sum, 1)) && !(rc = round64(&sumb, &sumb, 1)) && dyad_cmp(&sumb, sum) < 0)
            rc = dyad_copy(sum, &sumb);
        if (!rc && dyad_cmp(&hsum, HSUM(x, t)) > 0) rc = dyad_copy(HSUM(x, t), &hsum);
    }
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* the ball's scalars (matrix_exp.md, The ball, steps 4 and 5): rho h, g, c0, kappa, ell, b; then Q = (ell) b and the
   inputs of S = (Q reversed) (the columns' maxima) */
static int mtol_scalars(mtol *x) {
    size_t n = x->n, mp = (size_t)x->pieces;
    int rc = 0;
    dyad t, u, one, den, c0, *g = malloc((mp + 1) * sizeof *g), *kappa = malloc((mp + 1) * sizeof *kappa), *ell = malloc((mp + 1) * sizeof *ell);
    dyad_init(&t);
    dyad_init(&u);
    dyad_init(&one);
    dyad_init(&den);
    dyad_init(&c0);
    if (!g || !kappa || !ell) rc = EXACTREF_NOMEM;
    for (size_t k = 0; !rc && k < mp; k++) {
        dyad_init(&g[k]);
        dyad_init(&kappa[k]);
        dyad_init(&ell[k]);
    }
    /* rho h = (r + spread) / m, rounded up */
    if (!rc && !(rc = dyad_add(&t, &x->r, &x->spread, 64, 1)) && !(rc = dyad_set_si(&u, (int64_t)mp, 0))) rc = dyad_div(&x->rho_h, &t, &u, 64, 1);
    for (size_t k = 0; !rc && k < mp; k++) {
        if ((rc = dyad_set_si(&g[k], 0, 0))) break;
        for (size_t i = 0; !rc && i < n; i++)
            if (dyad_cmp(ROWSUM(x, k, i), &g[k]) > 0) rc = dyad_copy(&g[k], ROWSUM(x, k, i));
    }
    if (!rc && !(rc = dyad_mul(&c0, &x->rho_h, &g[0], 64, 1)) && !(rc = dyad_set_si(&one, 1, 0)) && dyad_cmp(&c0, &one) >= 0) {
        x->has_ball = 0; /* c0 >= 1: no ball */
        goto done;
    }
    /* den = 1 - c0, rounded down */
    if (!rc && !(rc = dyad_neg(&t, &c0))) rc = dyad_add(&den, &one, &t, 64, 0);
    for (size_t k = 1; !rc && k < mp; k++)
        if (!(rc = dyad_add(&t, &g[k], &g[k - 1], 64, 1)) && !(rc = dyad_mul(&t, &x->rho_h, &t, 64, 1))) rc = dyad_div(&kappa[k], &t, &den, 64, 1);
    if (!rc) rc = dyad_set_si(&ell[0], 1, 0);
    for (size_t q = 1; !rc && q < mp; q++) {
        dyad sum;
        dyad_init(&sum);
        for (size_t k = 1; !rc && k <= q; k++)
            if (!(rc = dyad_mul(&t, &kappa[k], &ell[q - k], 64, 1))) rc = dyad_add_exact(&sum, &sum, &t);
        if (!rc) rc = round64(&ell[q], &sum, 1);
        dyad_free(&sum);
    }
    /* Q = (ell[q - j]) (b), b[k][i] = rows[k][i] / den rounded up: an m x m by m x n product of nonnegative bounds */
    int L = 2;
    if (!rc) rc = imat_init(&x->Lq, mp, mp, L, 1);
    if (!rc) rc = imat_init(&x->bq, mp, n, L, 1);
    if (!rc) rc = imat_init(&x->Q, mp, n, L, 1);
    for (size_t q = 0; !rc && q < mp; q++)
        for (size_t j = 0; !rc && j <= q; j++) rc = imat_set(&x->Lq, 0, q, j, &ell[q - j], NULL);
    for (size_t k = 0; !rc && k < mp; k++)
        for (size_t i = 0; !rc && i < n; i++)
            if (!(rc = dyad_div(&t, ROWSUM(x, k, i), &den, 64, 1))) rc = imat_set(&x->bq, 0, k, i, &t, NULL);
    imat_lines A, B;
    imat_lines_clear(&A);
    imat_lines_clear(&B);
    if (!rc) rc = imat_lines_init(&A, mp, mp, L, 1);
    if (!rc) rc = imat_lines_init(&B, n, mp, L, 1);
    if (!rc) {
        imat_lines_prepare(&A, &x->Lq, 0, 0, mp);
        imat_lines_prepare(&B, &x->bq, 1, 0, n);
        imat_product_rows(&x->Q, &A, &B, 0, mp);
    }
    imat_lines_free(&A);
    imat_lines_free(&B);
    /* S's factors: Sin[i][k] = Q[m - 1 - k][i] (upper bounds), Cols[k][j] = the columns' maxima */
    if (!rc) rc = imat_init(&x->Sin, n, mp, L, 1);
    if (!rc) rc = imat_init(&x->Cols, mp, n, L, 1);
    if (!rc) rc = imat_init(&x->S, n, n, L, 1);
    for (size_t i = 0; !rc && i < n; i++)
        for (size_t k = 0; !rc && k < mp; k++)
            if (!(rc = imat_upper(&t, &x->Q, 0, mp - 1 - k, i))) rc = imat_set(&x->Sin, 0, i, k, &t, NULL);
    for (size_t k = 0; !rc && k < mp; k++)
        for (size_t j = 0; !rc && j < n; j++) rc = imat_set(&x->Cols, 0, k, j, COLMAX(x, k, j), NULL);
    if (!rc) rc = imat_lines_init(&x->srows, n, mp, L, 1);
    if (!rc) rc = imat_lines_init(&x->scols, n, mp, L, 1);
    if (!rc) imat_lines_prepare(&x->scols, &x->Cols, 1, 0, n);
done:
    for (size_t k = 0; g && kappa && ell && k < mp; k++) {
        dyad_free(&g[k]);
        dyad_free(&kappa[k]);
        dyad_free(&ell[k]);
    }
    free(g);
    free(kappa);
    free(ell);
    dyad_free(&t);
    dyad_free(&u);
    dyad_free(&one);
    dyad_free(&den);
    dyad_free(&c0);
    return rc;
}

/* rows i0 .. i1 of the wanted parts: the box's and the ball's enclosures intersected, held in F (Lemma 1), Lemma 3 */
static int mtol_finish(mtol *x, size_t m, size_t i0, size_t i1) {
    int P = x->parts, rc = 0;
    int64_t w = x->w;
    const imat *Ebox = x->has_box ? mexp_result(&x->ex[0]) : NULL, *Ec = x->has_ball ? mexp_result(&x->ex[1]) : NULL;
    dyad lo, hi, blo, bhi, d, nd;
    dyad *ds[] = {&lo, &hi, &blo, &bhi, &d, &nd};
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_init(ds[k]);
    for (size_t i = i0; !rc && i < i1; i++)
        for (size_t j = x->c0; !rc && j < x->c1; j++) {
            /* D_ij = rho h S_ij, rounded up */
            if (Ec && !(rc = imat_upper(&d, &x->S, 0, i, j))) rc = dyad_mul(&d, &x->rho_h, &d, 64, 1);
            if (!rc && Ec) rc = dyad_neg(&nd, &d);
            for (int p = 0; !rc && p < P; p++) {
                size_t slot = mtol_slot(x, m, i, j, p);
                int have = 0;
                /* the ends rounded outward at w, as Lemma 3 rounds them next */
                if (Ebox && !(rc = imat_ends_round(&lo, &hi, Ebox, p, i, j, w))) have = 1;
                if (!rc && Ec && !(rc = imat_ends_round(&blo, &bhi, Ec, p, i, j, w)) && !(rc = dyad_add(&blo, &blo, &nd, w, 0)) &&
                    !(rc = dyad_add(&bhi, &bhi, &d, w, 1))) {
                    if (!have) {
                        if (!(rc = dyad_copy(&lo, &blo))) rc = dyad_copy(&hi, &bhi);
                        have = 1;
                    } else {
                        if (dyad_cmp(&blo, &lo) > 0) rc = dyad_copy(&lo, &blo);
                        if (!rc && dyad_cmp(&bhi, &hi) < 0) rc = dyad_copy(&hi, &bhi);
                    }
                }
                if (rc) break;
                int unbounded = !have;
                if (have) rc = lean_ends(&lo, &hi, &unbounded, &x->held, w);
                if (!rc) rc = mtol_put(x, slot, unbounded ? NULL : &lo, unbounded ? NULL : &hi, 0);
            }
        }
    for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_free(ds[k]);
    return rc;
}

static size_t mtol_job_tiles(const mtol *x, const mtol_job *jb) {
    switch (jb->kind) {
    case JOB_EXP: return mexp_tiles(&x->ex[jb->a], jb->b);
    case JOB_COLS:
    case JOB_PIECE:
    case JOB_S: return x->tiles;
    case JOB_FINISH: return (x->r1 - x->r0 + MEXP_TILE_ROWS - 1) / MEXP_TILE_ROWS;
    default: return 1;
    }
}

static int mtol_run_job(mtol *x, size_t m, const mtol_job *jb, size_t t) {
    size_t i0 = t * MEXP_TILE_ROWS, i1 = i0 + MEXP_TILE_ROWS < x->n ? i0 + MEXP_TILE_ROWS : x->n;
    switch (jb->kind) {
    case JOB_EXP: mexp_run(&x->ex[jb->a], jb->b, t); return 0;
    case JOB_COLS:
        imat_lines_prepare(&x->colsB, mexp_result(&x->ex[3]), 1, i0, i1);
        imat_lines_prepare(&x->colsA, &x->A0, 1, i0, i1);
        imat_lines_prepare(&x->colsW, mexp_result(&x->ex[2]), 1, i0, i1);
        return 0;
    case JOB_PIECE: return mtol_piece_rows(x, jb->a, t, i0, i1);
    case JOB_MERGE: { /* the columns' maxima of piece a, the smaller of the box's and the centered one's (each of its tiles'
                         maxima plus rest times |H_a|'s largest row sum); and G = G_(a+1), H = H_(a+1) */
        int rc = 0;
        dyad extra, ch;
        dyad_init(&extra);
        dyad_init(&ch);
        rc = dyad_set_si(&extra, 0, 0);
        for (size_t tt = 0; !rc && tt < x->tiles; tt++)
            if (dyad_cmp(HSUM(x, tt), &extra) > 0) rc = dyad_copy(&extra, HSUM(x, tt));
        if (!rc) rc = dyad_mul(&extra, &x->rest, &extra, 64, 1);
        for (size_t j = 0; !rc && j < x->n; j++) {
            dyad *c = COLMAX(x, jb->a, j);
            if ((rc = dyad_set_si(c, 0, 0)) || (rc = dyad_set_si(&ch, 0, 0))) break;
            for (size_t tt = 0; !rc && tt < x->tiles; tt++) {
                if (dyad_cmp(PARTIAL(x, tt, j), c) > 0) rc = dyad_copy(c, PARTIAL(x, tt, j));
                if (!rc && dyad_cmp(PARTIAL_H(x, tt, j), &ch) > 0) rc = dyad_copy(&ch, PARTIAL_H(x, tt, j));
            }
            if (!rc && !(rc = dyad_add(&ch, &ch, &extra, 64, 1)) && dyad_cmp(&ch, c) < 0) rc = dyad_copy(c, &ch);
        }
        dyad_free(&extra);
        dyad_free(&ch);
        if (!rc && jb->a + 1 < x->pieces) {
            if (jb->a == 0) { /* G_1 = W */
                const imat *W = mexp_result(&x->ex[2]);
                size_t count = (size_t)x->parts * x->n * x->n;
                memcpy(x->G.d, W->d, count * W->L * sizeof *W->d);
                memcpy(x->G.e, W->e, count * sizeof *W->e);
                memcpy(x->G.r, W->r, count * sizeof *W->r);
                if (W->rd) memcpy(x->G.rd, W->rd, x->n * x->n * sizeof *W->rd);
            } else {
                imat s = x->G;
                x->G = x->Gn;
                x->Gn = s;
            }
            imat s = x->H;
            x->H = x->Hn;
            x->Hn = s;
        }
        return rc;
    }
    case JOB_SCALARS: return mtol_scalars(x);
    case JOB_S:
        if (!x->has_ball) return 0;
        imat_lines_prepare(&x->srows, &x->Sin, 0, i0, i1);
        imat_product_rows(&x->S, &x->srows, &x->scols, i0, i1);
        return 0;
    case JOB_FINISH: {
        size_t r0 = x->r0 + t * MEXP_TILE_ROWS, r1 = r0 + MEXP_TILE_ROWS < x->r1 ? r0 + MEXP_TILE_ROWS : x->r1;
        return mtol_finish(x, m, r0, r1);
    }
    }
    return 0;
}

static void mtol_job_done(mtol *x, const mtol_job *jb) {
    if (jb->kind == JOB_EXP) mexp_done(&x->ex[jb->a], jb->b);
}

/* one matrix's whole tolerance on this thread */
static int mtol_matrix(mtol *x, size_t m) {
    int rc = mtol_prepare(x, m);
    for (size_t k = 0; !rc && !x->skip && k < x->njobs; k++) {
        size_t tiles = mtol_job_tiles(x, &x->jobs[k]);
        for (size_t t = 0; !rc && t < tiles; t++) rc = mtol_run_job(x, m, &x->jobs[k], t);
        mtol_job_done(x, &x->jobs[k]);
    }
    mtol_work_free(x);
    return rc;
}

/* The tiles of the next phase (0 when done): small matrices one phase of whole tolerances per MEXP_GROUP matrices;
   larger ones per matrix its preparation, then its jobs */
int mtol_next(mtol *x, size_t *tiles) {
    *tiles = 0;
    if (x->whole) {
        if (!x->started) {
            x->started = 1;
            *tiles = (x->count + MEXP_GROUP - 1) / MEXP_GROUP;
        }
        return 0;
    }
    int rc = 0;
    for (;;) {
        if (x->started) {
            if (x->job > 0) mtol_job_done(x, &x->jobs[x->job - 1]);
            if (!x->skip && x->job < x->njobs) {
                *tiles = mtol_job_tiles(x, &x->jobs[x->job]);
                x->job++;
                if (*tiles) return 0;
                continue;
            }
            mtol_work_free(x);
            x->mi++;
        }
        if (x->mi >= x->count) return 0;
        x->started = 1;
        x->job = 0;
        if ((rc = mtol_prepare(x, x->mi))) return rc;
    }
}

/* runs tile t of the current phase */
int mtol_tile(mtol *x, size_t t) {
    if (!x->whole) return mtol_run_job(x, x->mi, &x->jobs[x->job - 1], t);
    int rc = 0;
    for (size_t m = t * MEXP_GROUP; !rc && m < x->count && m < (t + 1) * MEXP_GROUP; m++) {
        mtol one = *x; /* this matrix's own working state; the inputs and outputs are shared */
        one.stats = NULL;
        one.jobs = NULL;
        one.njobs = one.capjobs = one.job = 0;
        for (int k = 0; k < MTOL_EXPS; k++) mexp_clear(&one.ex[k]);
        imat *ms[] = {&one.box, &one.center, &one.A0, &one.Win, &one.Bin, &one.Hin, &one.G, &one.Gn, &one.GB, &one.H, &one.Hn, &one.HA, &one.Lq, &one.bq, &one.Q, &one.Sin, &one.Cols, &one.S};
        for (size_t k = 0; k < sizeof ms / sizeof *ms; k++) imat_clear(ms[k]);
        imat_lines_clear(&one.rows);
        imat_lines_clear(&one.colsB);
        imat_lines_clear(&one.colsA);
        imat_lines_clear(&one.colsW);
        imat_lines_clear(&one.srows);
        imat_lines_clear(&one.scols);
        dyad *ds[] = {&one.norm, &one.r, &one.spread, &one.rho_h, &one.half_h, &one.rest};
        for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_init(ds[k]);
        rc = mtol_matrix(&one, m);
        for (size_t k = 0; k < sizeof ds / sizeof *ds; k++) dyad_free(ds[k]);
    }
    return rc;
}
