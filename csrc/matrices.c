/* Interval matrices, midpoint-radius, and their exponential (proofs/matrix_exp.md, The enclosure).

   A midpoint is MAT_LIMBS_MAX or fewer signed limbs of MAT_LIMB bits, all of one sign, limb 0 least
   significant, times 2**e; a radius is an upper bound (ub). A product puts each row of its left
   factor and each column of its right one on one exponent (imat_lines), sums the limbs' products
   exactly in 128-bit words, and truncates the sum toward zero to L limbs; every part dropped goes
   into a radius, and radii are rounded up. The exponential (mexp) is scaling and squaring with a
   Taylor sum by Paterson and Stockmeyer's scheme, its remainder added only where a walk connects
   the entries, run as phases of row tiles on the caller's threads. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

#define LB MAT_LIMB
#define LMASK (((uint64_t)1 << LB) - 1)
#define LMAX MAT_LIMBS_MAX

/* ---------------------------------------------------------------- upper bounds */

/* m * 2**e rounded up to 32 bits */
ub ub_make(uint64_t m, int64_t e) {
    ub r = {0, 0};
    if (!m) return r;
    int b = word_bits(m);
    if (b <= 32) {
        r.m = m << (32 - b);
        r.e = e - (32 - b);
        return r;
    }
    int s = b - 32;
    uint64_t q = m >> s;
    if ((q << s) != m) q++;
    if (q >> 32) {
        q >>= 1;
        s++;
    }
    r.m = q;
    r.e = e + s;
    return r;
}

/* v * 2**e (v a nonnegative 128-bit word) rounded up to 32 bits */
static ub ub_make_wide(w128 v, int64_t e) {
    uint64_t hi = w128_hi(v), lo = w128_lo(v);
    if (!hi) return ub_make(lo, e);
    /* v / 2**32 rounded up fits 96 bits; its top 64 bits, rounded up again, then ub_make */
    uint64_t q = (hi << 32) | (lo >> 32), dropped = lo & 0xffffffffu;
    if (hi >> 32) { /* v / 2**64 rounded up */
        q = hi + (lo != 0);
        return q ? ub_make(q, e + 64) : ub_make((uint64_t)1 << 63, e + 65);
    }
    q += dropped != 0;
    return q ? ub_make(q, e + 32) : ub_make((uint64_t)1 << 63, e + 33);
}

/* ceil(m / 2**s), s >= 0 */
static uint64_t ceil_shr(uint64_t m, int64_t s) {
    if (!m) return 0;
    if (s >= 64) return 1;
    uint64_t q = m >> s;
    return q + ((q << s) != m);
}

ub ub_add(ub a, ub b) {
    if (!a.m) return b;
    if (!b.m) return a;
    if (a.e < b.e) {
        ub t = a;
        a = b;
        b = t;
    }
    int64_t d = a.e - b.e;
    uint64_t s = (a.m << 30) + (d <= 30 ? b.m << (30 - d) : ceil_shr(b.m, d - 30));
    return ub_make(s, a.e - 30);
}

ub ub_mul(ub a, ub b) {
    ub z = {0, 0};
    return a.m && b.m ? ub_make(a.m * b.m, a.e + b.e) : z;
}

/* a / k rounded up, k >= 1 */
ub ub_div_word(ub a, uint64_t k) {
    if (!a.m) return a;
    uint64_t x = a.m << 30, q = x / k;
    return ub_make(q + (q * k != x), a.e - 30);
}

ub ub_shift(ub a, int64_t s) {
    if (a.m) a.e += s;
    return a;
}

int ub_le(ub a, ub b) {
    if (!a.m) return 1;
    if (!b.m) return 0;
    if (a.e != b.e) return a.e < b.e; /* both mantissas lie in [2**31, 2**32) */
    return a.m <= b.m;
}

/* 2**t > a: the least such t (a nonzero) */
static int64_t ub_top(ub a) { return a.e + 32; }

int ub_to_dyad(dyad *r, ub a) { return dyad_set_si(r, (int64_t)a.m, a.e); }

/* |x| rounded up */
int ub_of_dyad(ub *r, const dyad *x) {
    int64_t b = big_bits(&x->m);
    r->m = 0;
    r->e = 0;
    if (!b) return 0;
    if (b <= 63) {
        *r = ub_make(x->m.w[0], x->e);
        return 0;
    }
    bigint t;
    big_init(&t);
    int rc = big_copy(&t, &x->m);
    if (!rc) {
        t.neg = 0;
        rc = big_shr(&t, &t, b - 63, 1);
    }
    if (!rc) *r = ub_make(t.n ? t.w[0] : 0, x->e + b - 63);
    big_free(&t);
    return rc;
}

/* sqrt(a**2 + b**2) rounded up */
ub ub_hypot(ub a, ub b) {
    ub s = ub_add(ub_mul(a, a), ub_mul(b, b)), z = {0, 0};
    if (!s.m) return z;
    /* s = m 2**e, m < 2**32: m 2**30 (e - 30 even, so one more bit when e is odd) below 2**63, its integer root rounded up */
    int64_t e = s.e - 30;
    uint64_t m = s.m << 30;
    if (e & 1) {
        m <<= 1;
        e -= 1;
    }
    uint64_t lo = 0, hi = (uint64_t)1 << 32;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (mid * mid < m) lo = mid + 1;
        else hi = mid;
    }
    return ub_make(lo, e / 2);
}

/* ---------------------------------------------------------------- midpoints of L limbs */

static int64_t limb_abs(int64_t x) { return x < 0 ? -x : x; }

/* the bits of x (0 for zero) */
static int64_t fx_bits(const int64_t *x, int L) {
    for (int h = L - 1; h >= 0; h--)
        if (x[h]) return (int64_t)LB * h + word_bits((uint64_t)limb_abs(x[h]));
    return 0;
}

/* y (ny limbs) = x (nx limbs) * 2**-s: for s > 0 toward zero, *lost set where bits are dropped; for
   s <= 0 exactly, the result fitting ny limbs */
static void fx_shift(int64_t *y, int ny, const int64_t *x, int nx, int64_t s, int *lost) {
    int neg = 0;
    int64_t a[2 * LMAX + 4];
    for (int k = 0; k < nx; k++) {
        if (x[k] < 0) neg = 1;
        a[k] = limb_abs(x[k]);
    }
    *lost = 0;
    if (s >= 0) {
        int64_t q = s / LB, t = s % LB;
        for (int64_t k = 0; k < q && k < nx; k++)
            if (a[k]) *lost = 1;
        if (q < nx && t && (a[q] & (((int64_t)1 << t) - 1))) *lost = 1;
        for (int k = 0; k < ny; k++) {
            int64_t i = k + q, v = i < nx ? a[i] >> t : 0;
            if (t && i + 1 < nx) v |= (int64_t)(((uint64_t)a[i + 1] << (LB - t)) & LMASK);
            y[k] = v;
        }
    } else {
        int64_t u = -s, q = u / LB, t = u % LB;
        for (int k = ny - 1; k >= 0; k--) {
            int64_t i = k - q, v = i >= 0 && i < nx ? (int64_t)(((uint64_t)a[i] << t) & LMASK) : 0;
            if (t && i - 1 >= 0 && i - 1 < nx) v |= a[i - 1] >> (LB - t);
            y[k] = v;
        }
    }
    if (neg)
        for (int k = 0; k < ny; k++) y[k] = -y[k];
}

/* a 128-bit word shifted right by LB bits, its sign kept */
static w128 sar_limb(w128 t) {
    uint64_t lo = w128_lo(t), hi = w128_hi(t);
    return w128_make((lo >> LB) | (hi << (64 - LB)), w128_negative(t) ? ~(~hi >> LB) : hi >> LB);
}

static int all_ones(w128 t) { return w128_lo(t) == ~(uint64_t)0 && w128_hi(t) == ~(uint64_t)0; }

/* The integer sum of acc[d] * 2**(LB d), d < D, times 2**e, truncated toward zero to L limbs: y and
   its exponent *ey; *lost is an upper bound of the part dropped (0 when none). */
static void fx_normalize(int64_t *y, int64_t *ey, ub *lost, const w128 *acc, int D, int64_t e, int L) {
    int64_t dig[2 * LMAX + 2 * MAT_BANDS + 16];
    int nd = 0;
    w128 carry = w128_zero();
    for (int d = 0; d < D; d++) {
        w128 t = w128_add(acc[d], carry);
        dig[nd++] = (int64_t)(w128_lo(t) & LMASK);
        carry = sar_limb(t);
    }
    while ((w128_lo(carry) || w128_hi(carry)) && !all_ones(carry)) {
        dig[nd++] = (int64_t)(w128_lo(carry) & LMASK);
        carry = sar_limb(carry);
    }
    int neg = all_ones(carry);
    if (neg) { /* the magnitude: 2**(LB nd) less the digits' value */
        int64_t c = 1;
        for (int d = 0; d < nd; d++) {
            int64_t v = ((int64_t)LMASK - dig[d]) + c;
            dig[d] = v & (int64_t)LMASK;
            c = v >> LB;
        }
        if (c) dig[nd++] = c; /* the digits were all 0: the magnitude is 2**(LB nd) itself */
    }
    int h = nd - 1;
    while (h >= 0 && !dig[h]) h--;
    lost->m = 0;
    lost->e = 0;
    *ey = e;
    if (h < 0) {
        for (int k = 0; k < L; k++) y[k] = 0;
        return;
    }
    int64_t s = (int64_t)LB * h + word_bits((uint64_t)dig[h]) - (int64_t)LB * L;
    if (s <= 0) {
        for (int k = 0; k < L; k++) y[k] = k < nd ? dig[k] : 0;
    } else {
        int64_t q = s / LB, t = s % LB;
        int any = 0;
        for (int64_t k = 0; k < q; k++)
            if (dig[k]) any = 1;
        if (t && (dig[q] & (((int64_t)1 << t) - 1))) any = 1;
        for (int k = 0; k < L; k++) {
            int64_t i = k + q, v = i < nd ? dig[i] >> t : 0;
            if (t && i + 1 < nd) v |= (int64_t)(((uint64_t)dig[i + 1] << (LB - t)) & LMASK);
            y[k] = v;
        }
        *ey = e + s;
        if (any) *lost = ub_make(1, e + s);
    }
    if (neg)
        for (int k = 0; k < L; k++) y[k] = -y[k];
}

/* |x| * 2**e rounded up */
static ub fx_mag(const int64_t *x, int L, int64_t e) {
    int64_t b = fx_bits(x, L);
    ub z = {0, 0};
    if (!b) return z;
    int64_t y[LMAX], s = b > 63 ? b - 63 : 0;
    int lost;
    fx_shift(y, L, x, L, s, &lost);
    uint64_t v = 0;
    for (int k = 0; k < L && k * LB < 64; k++) v |= (uint64_t)limb_abs(y[k]) << (k * LB);
    return ub_make(v + (uint64_t)lost, e + s);
}

/* x (a dyad) truncated toward zero to L limbs: y, *ey, and *lost where bits are dropped */
static int fx_from_dyad(int64_t *y, int64_t *ey, int *lost, const dyad *x, int L) {
    int64_t b = big_bits(&x->m), s = b - (int64_t)LB * L;
    for (int k = 0; k < L; k++) y[k] = 0;
    *lost = 0;
    *ey = x->e;
    if (!b) return 0;
    bigint t;
    big_init(&t);
    int rc = big_copy(&t, &x->m);
    if (!rc) {
        t.neg = 0;
        if (s > 0) {
            *lost = big_low_nonzero(&t, s);
            rc = big_shr(&t, &t, s, 0);
            *ey = x->e + s;
        }
    }
    for (int k = 0; !rc && k < L; k++) {
        int64_t bit = (int64_t)LB * k, word = bit / 64, off = bit % 64;
        uint64_t v = (size_t)word < t.n ? t.w[word] >> off : 0;
        if (off > 64 - LB && (size_t)word + 1 < t.n) v |= t.w[word + 1] << (64 - off);
        y[k] = (int64_t)(v & LMASK);
        if (x->m.neg) y[k] = -y[k];
    }
    big_free(&t);
    return rc;
}

/* x * 2**e exactly */
static int fx_to_dyad(dyad *r, const int64_t *x, int L, int64_t e) {
    bigint t;
    big_init(&t);
    int rc = big_set_si(&r->m, 0), neg = 0;
    for (int k = L - 1; !rc && k >= 0; k--) {
        if (x[k] < 0) neg = 1;
        if (!(rc = big_shl(&r->m, &r->m, LB)) && !(rc = big_set_word(&t, (uint64_t)limb_abs(x[k]), 0))) rc = big_add(&r->m, &r->m, &t);
    }
    if (!rc && neg && r->m.n) r->m.neg = 1;
    r->e = r->m.n ? e : 0;
    big_free(&t);
    return rc;
}

/* z += c x for interval scalars: z (L limbs at *ez, radius *rz), c (at ec, radius rc), x (at ex,
   radius rx) (matrix_exp.md, Sums and scalars) */
static void fx_axpy(int64_t *z, int64_t *ez, ub *rz, ub *trunc, const int64_t *c, int64_t ec, ub rc, const int64_t *x, int64_t ex, ub rx, int L) {
    w128 acc[2 * LMAX + 2];
    acc[0] = w128_zero();
    for (int d = 1; d < 2 * L - 1; d++) acc[d] = w128_zero();
    for (int a = 0; a < L; a++)
        for (int b = 0; b < L; b++) acc[a + b] = w128_add(acc[a + b], w128_product((uint64_t)c[a], (uint64_t)x[b]));
    int64_t t[LMAX], et;
    ub lost;
    fx_normalize(t, &et, &lost, acc, 2 * L - 1, ec + ex, L);
    ub rt = ub_add(ub_add(ub_mul(fx_mag(c, L, ec), rx), ub_mul(rc, fx_mag(x, L, ex))), ub_mul(rc, rx));
    *trunc = lost; /* the parts dropped, apart from the radii's propagation */
    int64_t bz = fx_bits(z, L), bt = fx_bits(t, L);
    if (!bt) {
        *rz = ub_add(ub_add(*rz, rt), lost);
        return;
    }
    if (!bz) {
        memcpy(z, t, (size_t)L * sizeof *z);
        *ez = et;
        *rz = ub_add(ub_add(*rz, rt), lost);
        return;
    }
    /* z + t on one exponent with a guard limb, then truncated to L limbs */
    int64_t top = *ez + bz > et + bt ? *ez + bz : et + bt, E = top + 1 - (int64_t)LB * (L + 1);
    int64_t zs[LMAX + 1], ts[LMAX + 1];
    int lz, lt;
    fx_shift(zs, L + 1, z, L, E - *ez, &lz);
    fx_shift(ts, L + 1, t, L, E - et, &lt);
    for (int d = 0; d < L + 1; d++) acc[d] = w128_add(w128_product((uint64_t)zs[d], 1), w128_product((uint64_t)ts[d], 1));
    ub lost2;
    fx_normalize(z, ez, &lost2, acc, L + 1, E, L);
    ub tr = ub_add(lost, lost2);
    if (lz) tr = ub_add(tr, ub_make(1, E));
    if (lt) tr = ub_add(tr, ub_make(1, E));
    *trunc = tr;
    *rz = ub_add(ub_add(*rz, rt), tr);
}

/* ---------------------------------------------------------------- interval matrices */

static size_t at_of(const imat *M, int part, size_t i, size_t j) { return ((size_t)part * M->rows + i) * M->cols + j; }

void imat_clear(imat *M) {
    M->d = NULL;
    M->e = NULL;
    M->r = NULL;
    M->rd = NULL;
    M->rows = M->cols = 0;
}

int imat_init(imat *M, size_t rows, size_t cols, int L, int parts) {
    size_t count = (size_t)parts * rows * cols;
    M->rows = rows;
    M->cols = cols;
    M->L = L;
    M->parts = parts;
    M->d = calloc(count * L + 1, sizeof *M->d);
    M->e = calloc(count + 1, sizeof *M->e);
    M->r = calloc(count + 1, sizeof *M->r);
    M->rd = parts == 2 ? calloc(rows * cols + 1, sizeof *M->rd) : NULL;
    if (!M->d || !M->e || !M->r || (parts == 2 && !M->rd)) {
        imat_free(M);
        return EXACTREF_NOMEM;
    }
    return 0;
}

void imat_free(imat *M) {
    free(M->d);
    free(M->e);
    free(M->r);
    free(M->rd);
    imat_clear(M);
}

/* entry (part, i, j) = [mid +- rad]: the midpoint truncated toward zero to L limbs, the part dropped
   and rad (NULL: 0) in the radius */
int imat_set(imat *M, int part, size_t i, size_t j, const dyad *mid, const dyad *rad) {
    size_t at = at_of(M, part, i, j);
    int lost;
    int rc = fx_from_dyad(M->d + at * M->L, &M->e[at], &lost, mid, M->L);
    ub r = {0, 0};
    if (!rc && rad) rc = ub_of_dyad(&r, rad);
    if (!rc && lost) r = ub_add(r, ub_make(1, M->e[at]));
    M->r[at] = r;
    if (M->parts == 2) M->rd[i * M->cols + j] = ub_hypot(M->r[at_of(M, 0, i, j)], M->r[at_of(M, 1, i, j)]); /* the disk around both parts */
    return rc;
}

/* the midpoint of entry (part, i, j), exactly */
int imat_mid(dyad *mid, const imat *M, int part, size_t i, size_t j) {
    size_t at = at_of(M, part, i, j);
    return fx_to_dyad(mid, M->d + at * M->L, M->L, M->e[at]);
}

/* the ends of entry (part, i, j), exactly (p 0) or rounded outward at p bits: a radius far above the midpoint's last
   bit, as in a box exponential of a large norm, then costs p bits, not the distance between them */
int imat_ends_round(dyad *lo, dyad *hi, const imat *M, int part, size_t i, size_t j, int64_t p) {
    size_t at = at_of(M, part, i, j);
    dyad mid, rad;
    dyad_init(&mid);
    dyad_init(&rad);
    int rc = imat_mid(&mid, M, part, i, j);
    ub r = M->r[at];
    if (M->parts == 2 && ub_le(M->rd[i * M->cols + j], r)) r = M->rd[i * M->cols + j]; /* a part within the disk too */
    if (!rc) rc = ub_to_dyad(&rad, r);
    if (!rc) rc = p ? dyad_add(hi, &mid, &rad, p, 1) : dyad_add_exact(hi, &mid, &rad);
    if (!rc && !(rc = dyad_neg(&rad, &rad))) rc = p ? dyad_add(lo, &mid, &rad, p, 0) : dyad_add_exact(lo, &mid, &rad);
    dyad_free(&mid);
    dyad_free(&rad);
    return rc;
}

int imat_ends(dyad *lo, dyad *hi, const imat *M, int part, size_t i, size_t j) { return imat_ends_round(lo, hi, M, part, i, j, 0); }

/* whether entry (i, j) is exactly 0 in every part */
static int imat_zero_at(const imat *M, size_t i, size_t j) {
    for (int p = 0; p < M->parts; p++) {
        size_t at = at_of(M, p, i, j);
        if (M->r[at].m || fx_bits(M->d + at * M->L, M->L)) return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- products */

void imat_lines_clear(imat_lines *F) { memset(F, 0, sizeof *F); }

void imat_lines_free(imat_lines *F) {
    void *all[] = {F->x, F->q, F->rq, F->qm, F->rd, F->bv, F->br, F->lost, F->E, F->U, F->UR, F->banded, F->bvmax, F->brmax};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) free(all[k]);
    imat_lines_clear(F);
}

/* room for `lines` lines of `len` entries in `parts` parts, each L + 1 limbs (a banded line's) */
int imat_lines_init(imat_lines *F, size_t lines, size_t len, int L, int parts) {
    imat_lines_clear(F);
    F->lines = lines;
    F->len = len;
    F->L = L;
    F->S = L + 1;
    F->parts = parts;
    /* |x| and the radii at Q = 52 bits: products of two below 2**104, at most 2**21 of them summed in 128-bit words
       (imat_product_fits), so an entry far below the largest of its band is overstated by 2**-52 of that largest */
    F->Q = 52;
    size_t n = lines * len * parts, e = lines * len;
    F->x = malloc((n * F->S + 1) * sizeof *F->x);
    F->q = malloc((n + 1) * sizeof *F->q);
    F->rq = malloc((n + 1) * sizeof *F->rq);
    F->qm = parts == 2 ? malloc((e + 1) * sizeof *F->qm) : NULL;
    F->rd = parts == 2 ? malloc((e + 1) * sizeof *F->rd) : NULL;
    F->bv = malloc(e + 1);
    F->br = malloc(e + 1);
    F->lost = malloc(n + 1);
    F->E = malloc((lines + 1) * sizeof *F->E);
    F->U = malloc((lines + 1) * sizeof *F->U);
    F->UR = malloc((lines + 1) * sizeof *F->UR);
    F->banded = malloc(lines + 1);
    F->bvmax = malloc(lines + 1);
    F->brmax = malloc(lines + 1);
    if (!F->x || !F->q || !F->rq || (parts == 2 && (!F->qm || !F->rd)) || !F->bv || !F->br || !F->lost || !F->E || !F->U || !F->UR || !F->banded ||
        !F->bvmax || !F->brmax) {
        imat_lines_free(F);
        return EXACTREF_NOMEM;
    }
    return 0;
}

/* sqrt(a**2 + b**2) for nonnegative integers below 2**53, rounded up */
static uint64_t isqrt_up(w128 v) {
    /* the integer square root of v < 2**106, by bisection on 54-bit candidates */
    uint64_t lo = 0, hi = (uint64_t)1 << 54;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2, h, l = word_mul(&h, mid, mid);
        if (w128_less(w128_make(l, h), v)) lo = mid + 1;
        else hi = mid;
    }
    return lo; /* the least r with r**2 >= v */
}

static uint64_t hypot_up(uint64_t a, uint64_t b) {
    uint64_t h1, l1 = word_mul(&h1, a, a), h2, l2 = word_mul(&h2, b, b);
    return isqrt_up(w128_add(w128_make(l1, h1), w128_make(l2, h2)));
}

/* Lines l0 .. l1 of M (its rows, or its columns with by_cols). A line whose entries all lie within LB bits of its
   largest is put on one exponent, LB L bits below that largest, its entries truncated toward zero, the parts dropped in
   their radius; a wider line is banded: an entry in band b (its top LB b to LB (b + 1) bits below the line's) keeps
   L + 1 limbs at the line's exponent less LB (L + 1 + b) bits, so at least LB L bits of its own. |x| and the radii are
   rounded up to Q bits within each entry's band (matrix_exp.md, Products). A complex entry has one band for both parts,
   its modulus |x| and its disk radius too. */
void imat_lines_prepare(imat_lines *F, const imat *M, int by_cols, size_t l0, size_t l1) {
    size_t len = F->len;
    int L = F->L, S = F->S, P = F->parts, Q = F->Q;
    for (size_t l = l0; l < l1; l++) {
        int64_t top = INT64_MIN, low = INT64_MAX, rtop = INT64_MIN, rlow = INT64_MAX;
        for (size_t k = 0; k < len; k++) {
            int64_t et = INT64_MIN; /* the entry's top over its parts */
            for (int p = 0; p < P; p++) {
                size_t at = by_cols ? at_of(M, p, k, l) : at_of(M, p, l, k);
                int64_t b = fx_bits(M->d + at * L, L);
                if (b && M->e[at] + b > et) et = M->e[at] + b;
            }
            if (et != INT64_MIN) {
                if (et > top) top = et;
                if (et < low) low = et;
            }
        }
        if (top == INT64_MIN) top = low = 0;
        int banded = top - low > LB;
        int64_t E = banded ? top - (int64_t)LB * S : top - (int64_t)LB * L;
        int bvmax = 0, brmax = 0;
        for (size_t k = 0; k < len; k++) {
            size_t ent = l * len + k;
            int64_t et = INT64_MIN;
            for (int p = 0; p < P; p++) {
                size_t at = by_cols ? at_of(M, p, k, l) : at_of(M, p, l, k);
                int64_t b = fx_bits(M->d + at * L, L);
                if (b && M->e[at] + b > et) et = M->e[at] + b;
            }
            int64_t band = banded && et != INT64_MIN ? (top - et) / LB : 0;
            if (band > MAT_BANDS - 1) band = MAT_BANDS - 1; /* past the last band: truncated there, into the radius */
            F->bv[ent] = (unsigned char)band;
            if (band > bvmax) bvmax = (int)band;
            int64_t Eb = E - (int64_t)LB * band, Ub = top - (int64_t)LB * band - Q;
            uint64_t qp[2] = {0, 0};
            ub rmax = {0, 0};
            for (int p = 0; p < P; p++) {
                size_t at = by_cols ? at_of(M, p, k, l) : at_of(M, p, l, k), slot = ent * P + p;
                int64_t *y = F->x + slot * S, z[1];
                int lost, under;
                fx_shift(y, S, M->d + at * L, L, Eb - M->e[at], &lost);
                fx_shift(z, 1, y, S, Ub - Eb, &under); /* |y| / 2**(Ub - Eb), below 2**Q */
                qp[p] = (uint64_t)limb_abs(z[0]) + (uint64_t)under;
                F->q[slot] = qp[p];
                F->lost[slot] = (unsigned char)lost;
                ub r = M->r[at];
                if (lost) r = ub_add(r, ub_make(1, Eb));
                if (!ub_le(r, rmax)) rmax = r;
            }
            if (P == 2) {
                F->qm[ent] = hypot_up(qp[0], qp[1]); /* the modulus, below 2**(Q + 1) */
                ub d = M->rd[(by_cols ? k * M->cols + l : l * M->cols + k)];
                if (F->lost[ent * P] || F->lost[ent * P + 1]) d = ub_add(d, ub_make(2, Eb)); /* both parts' truncations */
                if (!ub_le(d, rmax)) rmax = d;
            }
            if (rmax.m) {
                if (ub_top(rmax) > rtop) rtop = ub_top(rmax);
                if (ub_top(rmax) < rlow) rlow = ub_top(rmax);
            }
        }
        if (rtop == INT64_MIN) rtop = rlow = 0;
        int rbanded = rtop - rlow > LB;
        F->banded[l] = (unsigned char)(banded || rbanded);
        F->E[l] = E;
        F->U[l] = top - Q;
        F->UR[l] = rtop - Q;
        /* the radii: Q bits within the band of each entry's largest radius */
        for (size_t k = 0; k < len; k++) {
            size_t ent = l * len + k;
            int64_t Eb = E - (int64_t)LB * F->bv[ent];
            ub rs[3];
            ub rmax = {0, 0};
            for (int p = 0; p < P; p++) {
                size_t at = by_cols ? at_of(M, p, k, l) : at_of(M, p, l, k);
                rs[p] = M->r[at];
                if (F->lost[ent * P + p]) rs[p] = ub_add(rs[p], ub_make(1, Eb));
                if (!ub_le(rs[p], rmax)) rmax = rs[p];
            }
            if (P == 2) {
                rs[2] = M->rd[(by_cols ? k * M->cols + l : l * M->cols + k)];
                if (F->lost[ent * P] || F->lost[ent * P + 1]) rs[2] = ub_add(rs[2], ub_make(2, Eb));
                if (!ub_le(rs[2], rmax)) rmax = rs[2];
            }
            int64_t band = rbanded && rmax.m ? (rtop - ub_top(rmax)) / LB : 0;
            if (band > MAT_BANDS - 1) band = MAT_BANDS - 1;
            F->br[ent] = (unsigned char)band;
            if (band > brmax) brmax = (int)band;
            int64_t URb = rtop - (int64_t)LB * band - Q;
            for (int p = 0; p < P + (P == 2); p++) {
                ub r = rs[p];
                uint64_t v = !r.m ? 0 : r.e >= URb ? r.m << (r.e - URb) : ceil_shr(r.m, URb - r.e);
                if (p < P) F->rq[ent * P + p] = v;
                else F->rd[ent] = v;
            }
        }
        F->bvmax[l] = (unsigned char)bvmax;
        F->brmax[l] = (unsigned char)brmax;
    }
}

/* the exact sums of the limbs' products of a line pair, entries S limbs apart, the first L of them used: real, or both
   parts of a complex one */
#define DOT_REAL(LL)                                                                                                   \
    static void dot_real_##LL(w128 *acc, const int64_t *x, const int64_t *y, size_t n, int S) {                        \
        w2 s[2 * LL - 1] = {0}, t;                                                                                     \
        for (size_t k = 0; k < n; k++, x += S, y += S)                                                                 \
            for (int a = 0; a < LL; a++)                                                                               \
                for (int b = 0; b < LL; b++) {                                                                         \
                    w2_product(&t, (uint64_t)x[a], (uint64_t)y[b]);                                                    \
                    w2_add(&s[a + b], &t);                                                                             \
                }                                                                                                      \
        for (int d = 0; d < 2 * LL - 1; d++) acc[d] = w2_get(&s[d]);                                                   \
    }
DOT_REAL(1)
DOT_REAL(2)
DOT_REAL(3)
DOT_REAL(4)
DOT_REAL(5)
DOT_REAL(6)
DOT_REAL(7)
DOT_REAL(8)

static void dot_real(w128 *acc, const int64_t *x, const int64_t *y, size_t n, int L, int S) {
    switch (L) {
    case 1: dot_real_1(acc, x, y, n, S); return;
    case 2: dot_real_2(acc, x, y, n, S); return;
    case 3: dot_real_3(acc, x, y, n, S); return;
    case 4: dot_real_4(acc, x, y, n, S); return;
    case 5: dot_real_5(acc, x, y, n, S); return;
    case 6: dot_real_6(acc, x, y, n, S); return;
    case 7: dot_real_7(acc, x, y, n, S); return;
    case 8: dot_real_8(acc, x, y, n, S); return;
    default: break;
    }
    w2 s[2 * LMAX - 1], t;
    memset(s, 0, (size_t)(2 * L - 1) * sizeof *s);
    for (size_t k = 0; k < n; k++, x += S, y += S)
        for (int a = 0; a < L; a++)
            for (int b = 0; b < L; b++) {
                w2_product(&t, (uint64_t)x[a], (uint64_t)y[b]);
                w2_add(&s[a + b], &t);
            }
    for (int d = 0; d < 2 * L - 1; d++) acc[d] = w2_get(&s[d]);
}

/* complex: re = sum xr yr - xi yi, im = sum xr yi + xi yr; x and y hold (re, im) per entry */
static void dot_complex(w128 *re, w128 *im, const int64_t *x, const int64_t *y, size_t n, int L, int S) {
    w2 sr[2 * LMAX - 1], si[2 * LMAX - 1], t;
    memset(sr, 0, (size_t)(2 * L - 1) * sizeof *sr);
    memset(si, 0, (size_t)(2 * L - 1) * sizeof *si);
    for (size_t k = 0; k < n; k++, x += 2 * S, y += 2 * S) {
        const int64_t *xr = x, *xi = x + S, *yr = y, *yi = y + S;
        for (int a = 0; a < L; a++)
            for (int b = 0; b < L; b++) {
                w2_product(&t, (uint64_t)xr[a], (uint64_t)yr[b]);
                w2_add(&sr[a + b], &t);
                w2_product(&t, (uint64_t)xi[a], (uint64_t)yi[b]);
                w2_sub(&sr[a + b], &t);
                w2_product(&t, (uint64_t)xr[a], (uint64_t)yi[b]);
                w2_add(&si[a + b], &t);
                w2_product(&t, (uint64_t)xi[a], (uint64_t)yr[b]);
                w2_add(&si[a + b], &t);
            }
    }
    for (int d = 0; d < 2 * L - 1; d++) {
        re[d] = w2_get(&sr[d]);
        im[d] = w2_get(&si[d]);
    }
}

/* banded lines: every entry's S limbs, its product placed (bands_x - bx) + (bands_y - by) digits up, so terms of every
   pair of bands meet in one array of digits from the exponent E_x + E_y - LB (bands_x + bands_y) */
static void dot_banded(w128 *re, w128 *im, const int64_t *x, const int64_t *y, const unsigned char *bx, const unsigned char *by, int mx, int my,
                       size_t n, int S, int P) {
    int D = 2 * S - 1 + mx + my;
    w2 sr[2 * LMAX + 2 * MAT_BANDS + 4], si[2 * LMAX + 2 * MAT_BANDS + 4], t;
    memset(sr, 0, (size_t)D * sizeof *sr);
    memset(si, 0, (size_t)D * sizeof *si);
    for (size_t k = 0; k < n; k++) {
        const int64_t *xr = x + k * P * S, *yr = y + k * P * S;
        int o = (mx - bx[k]) + (my - by[k]);
        if (P == 1) {
            for (int a = 0; a < S; a++)
                for (int b = 0; b < S; b++) {
                    w2_product(&t, (uint64_t)xr[a], (uint64_t)yr[b]);
                    w2_add(&sr[o + a + b], &t);
                }
        } else {
            const int64_t *xi = xr + S, *yi = yr + S;
            for (int a = 0; a < S; a++)
                for (int b = 0; b < S; b++) {
                    w2_product(&t, (uint64_t)xr[a], (uint64_t)yr[b]);
                    w2_add(&sr[o + a + b], &t);
                    w2_product(&t, (uint64_t)xi[a], (uint64_t)yi[b]);
                    w2_sub(&sr[o + a + b], &t);
                    w2_product(&t, (uint64_t)xr[a], (uint64_t)yi[b]);
                    w2_add(&si[o + a + b], &t);
                    w2_product(&t, (uint64_t)xi[a], (uint64_t)yr[b]);
                    w2_add(&si[o + a + b], &t);
                }
        }
    }
    for (int d = 0; d < D; d++) {
        re[d] = w2_get(&sr[d]);
        im[d] = w2_get(&si[d]);
    }
}

/* whether a product's exact sums fit their 128-bit words: each limb product is below 2**106, and at most parts (L + 1) n of
   them meet in one word (matrix_exp.md, Products) */
int imat_product_fits(size_t n, int L, int parts) { return (uint64_t)parts * (uint64_t)(L + 1) * n <= ((uint64_t)1 << 20); }

/* one radius sum: sum over k of a_k b_k, a_k at 2**(Ua - LB ba_k) and b_k at 2**(Ub - LB bb_k), as an upper bound */
static ub radius_sum(const uint64_t *a, size_t sa, const unsigned char *ba, int ma, int64_t Ua, const uint64_t *b, size_t sb, const unsigned char *bb,
                     int mb, int64_t Ub, size_t n, int banded) {
    w2 t;
    if (!banded) {
        w2 s = {0};
        for (size_t k = 0; k < n; k++) {
            w2_product(&t, a[k * sa], b[k * sb]);
            w2_add(&s, &t);
        }
        return ub_make_wide(w2_get(&s), Ua + Ub);
    }
    w2 s[2 * MAT_BANDS];
    int D = ma + mb + 1;
    memset(s, 0, (size_t)D * sizeof *s);
    for (size_t k = 0; k < n; k++) {
        int o = (ma - ba[k]) + (mb - bb[k]);
        w2_product(&t, a[k * sa], b[k * sb]);
        w2_add(&s[o], &t);
    }
    ub r = {0, 0};
    for (int d = 0; d < D; d++) {
        w128 v = w2_get(&s[d]);
        if (w128_lo(v) || w128_hi(v)) r = ub_add(r, ub_make_wide(v, Ua + Ub - (int64_t)LB * (ma + mb - d)));
    }
    return r;
}

/* Rows i0 .. i1 of C = X Y, A holding X's rows and B Y's columns prepared (matrix_exp.md, Products). A complex entry's
   parts get their own radii (each part's |X| RY + RX |Y| + RX RY over its two pairings) and the entry a disk radius
   (|X| RY + RX |Y| + RX RY of moduli and disk radii); each part's radius is then at most the disk's. */
void imat_product_rows(imat *C, const imat_lines *A, const imat_lines *B, size_t i0, size_t i1) {
    size_t n = A->len;
    int L = C->L, S = A->S, P = C->parts;
    w128 acc[2][2 * LMAX + 2 * MAT_BANDS + 4];
    for (size_t i = i0; i < i1; i++) {
        const int64_t *x = A->x + i * n * P * S;
        const uint64_t *xq = A->q + i * n * P, *xr = A->rq + i * n * P;
        const unsigned char *bvx = A->bv + i * n, *brx = A->br + i * n;
        for (size_t j = 0; j < C->cols; j++) {
            const int64_t *y = B->x + j * n * P * S;
            const uint64_t *yq = B->q + j * n * P, *yr = B->rq + j * n * P;
            const unsigned char *bvy = B->bv + j * n, *bry = B->br + j * n;
            int banded = A->banded[i] || B->banded[j], mx = A->bvmax[i], my = B->bvmax[j], rx = A->brmax[i], ry = B->brmax[j];
            int D = 2 * L - 1;
            int64_t base = A->E[i] + B->E[j];
            if (!banded) {
                if (P == 1)
                    dot_real(acc[0], x, y, n, L, S);
                else
                    dot_complex(acc[0], acc[1], x, y, n, L, S);
            } else {
                dot_banded(acc[0], acc[1], x, y, bvx, bvy, mx, my, n, S, P);
                D = 2 * S - 1 + mx + my;
                base -= (int64_t)LB * (mx + my);
            }
            ub dr = {0, 0};
            if (P == 2) { /* the disk radius: moduli and disk radii */
                const uint64_t *xm = A->qm + i * n, *xd = A->rd + i * n, *ym = B->qm + j * n, *yd = B->rd + j * n;
                dr = ub_add(radius_sum(xm, 1, bvx, mx, A->U[i], yd, 1, bry, ry, B->UR[j], n, banded),
                            radius_sum(xd, 1, brx, rx, A->UR[i], ym, 1, bvy, my, B->U[j], n, banded));
                dr = ub_add(dr, radius_sum(xd, 1, brx, rx, A->UR[i], yd, 1, bry, ry, B->UR[j], n, banded));
            }
            ub lostp[2] = {{0, 0}, {0, 0}};
            for (int p = 0; p < P; p++) {
                /* |X| RY, RX |Y| and RX RY for this part: real, or the part's two pairings */
                ub r = {0, 0};
                for (int a = 0; a < P; a++) {
                    int b = p == 0 ? a : 1 - a; /* re: (re, re), (im, im); im: (re, im), (im, re) */
                    r = ub_add(r, radius_sum(xq + a, P, bvx, mx, A->U[i], yr + b, P, bry, ry, B->UR[j], n, banded));
                    r = ub_add(r, radius_sum(xr + a, P, brx, rx, A->UR[i], yq + b, P, bvy, my, B->U[j], n, banded));
                    r = ub_add(r, radius_sum(xr + a, P, brx, rx, A->UR[i], yr + b, P, bry, ry, B->UR[j], n, banded));
                }
                size_t at = at_of(C, p, i, j);
                fx_normalize(C->d + at * L, &C->e[at], &lostp[p], acc[p], D, base, L);
                C->r[at] = ub_add(r, lostp[p]);
            }
            if (P == 2) {
                size_t at = i * C->cols + j;
                C->rd[at] = ub_add(ub_add(dr, lostp[0]), lostp[1]);
                for (int p = 0; p < 2; p++) { /* each part within the disk */
                    size_t pat = at_of(C, p, i, j);
                    if (ub_le(C->rd[at], C->r[pat])) C->r[pat] = C->rd[at];
                }
                ub h = ub_hypot(C->r[at_of(C, 0, i, j)], C->r[at_of(C, 1, i, j)]);
                if (ub_le(h, C->rd[at])) C->rd[at] = h;
            }
        }
    }
}

/* ---------------------------------------------------------------- the exponential */

enum { OP_COLS, OP_POWER, OP_TERMS, OP_HORNER, OP_TAIL, OP_SQUARE };

/* the bound on every ||X||_inf, X in M: row sums of |midpoint| + radius over the parts */
static ub imat_norm(const imat *M) {
    ub norm = {0, 0};
    for (size_t i = 0; i < M->rows; i++) {
        ub row = {0, 0};
        for (size_t j = 0; j < M->cols; j++) {
            if (M->parts == 2) { /* |midpoint| + disk radius, or the parts' sums when smaller */
                size_t a0 = at_of(M, 0, i, j), a1 = at_of(M, 1, i, j);
                ub m0 = fx_mag(M->d + a0 * M->L, M->L, M->e[a0]), m1 = fx_mag(M->d + a1 * M->L, M->L, M->e[a1]);
                ub disk = ub_add(ub_hypot(m0, m1), M->rd[i * M->cols + j]), rect = ub_add(ub_add(m0, M->r[a0]), ub_add(m1, M->r[a1]));
                row = ub_add(row, ub_le(disk, rect) ? disk : rect);
                continue;
            }
            size_t at = at_of(M, 0, i, j);
            row = ub_add(row, ub_add(fx_mag(M->d + at * M->L, M->L, M->e[at]), M->r[at]));
        }
        if (!ub_le(row, norm)) norm = row;
    }
    return norm;
}

void mexp_clear(mexp *x) { memset(x, 0, sizeof *x); }

void mexp_free(mexp *x) {
    for (int k = 0; k < 2; k++) imat_free(&x->m[k]);
    if (x->pw)
        for (int k = 0; k <= x->q; k++) imat_free(&x->pw[k]);
    free(x->pw);
    imat_lines_free(&x->rows);
    imat_lines_free(&x->cols);
    free(x->c);
    free(x->ec);
    free(x->rc);
    free(x->reach);
    free(x->ops);
    mexp_clear(x);
}

/* 1/k!, k = 0 .. N, as interval scalars at L limbs (matrix_exp.md, The algorithm, step 3) */
static int factorials(mexp *x) {
    int L = x->L, N = x->N;
    x->c = calloc((size_t)(N + 1) * L, sizeof *x->c);
    x->ec = calloc((size_t)N + 1, sizeof *x->ec);
    x->rc = calloc((size_t)N + 1, sizeof *x->rc);
    if (!x->c || !x->ec || !x->rc) return EXACTREF_NOMEM;
    x->c[L - 1] = (int64_t)1 << (LB - 1);
    x->ec[0] = -((int64_t)LB * L - 1);
    for (int k = 1; k <= N; k++) {
        const int64_t *prev = x->c + (size_t)(k - 1) * L;
        /* (prev * 2**LB) / k toward zero over L + 1 limbs, then truncated to L limbs */
        w128 acc[LMAX + 1];
        uint64_t rem = 0;
        for (int d = L; d >= 0; d--) { /* (rem 2**LB + limb) / k, rem < k */
            uint64_t limb = (uint64_t)(d >= 1 ? prev[d - 1] : 0);
            acc[d] = w128_make(word_div(&rem, rem >> (64 - LB), (rem << LB) | limb, (uint64_t)k), 0);
        }
        ub lost;
        fx_normalize(x->c + (size_t)k * L, &x->ec[k], &lost, acc, L + 1, x->ec[k - 1] - LB, L);
        x->rc[k] = ub_add(ub_add(ub_div_word(x->rc[k - 1], (uint64_t)k), lost), rem ? ub_make(1, x->ec[k - 1] - LB) : (ub){0, 0});
    }
    return 0;
}

/* the reachability closure of A's pattern by bit sets (Warshall) */
static int closure(mexp *x, const imat *A) {
    size_t n = x->n, words = (n + 63) / 64;
    x->words = words;
    x->reach = calloc(n * words + 1, sizeof *x->reach);
    if (!x->reach) return EXACTREF_NOMEM;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            if (!imat_zero_at(A, i, j)) x->reach[i * words + j / 64] |= (uint64_t)1 << (j % 64);
    for (size_t k = 0; k < n; k++)
        for (size_t i = 0; i < n; i++)
            if (x->reach[i * words + k / 64] >> (k % 64) & 1)
                for (size_t w = 0; w < words; w++) x->reach[i * words + w] |= x->reach[k * words + w];
    return 0;
}

static int add_op(mexp *x, int kind, int arg) {
    if (x->nops == x->cap) {
        size_t cap = x->cap ? 2 * x->cap : 32;
        mexp_op *ops = realloc(x->ops, cap * sizeof *ops);
        if (!ops) return EXACTREF_NOMEM;
        x->ops = ops;
        x->cap = cap;
    }
    x->ops[x->nops].kind = kind;
    x->ops[x->nops].arg = arg;
    x->nops++;
    return 0;
}

/* entry (part, i, j) of M = the interval [lo, hi] */
static int imat_set_ends(imat *M, int part, size_t i, size_t j, const dyad *lo, const dyad *hi) {
    dyad mid, rad;
    dyad_init(&mid);
    dyad_init(&rad);
    int rc = dyad_add_exact(&mid, lo, hi);
    if (!rc && !(rc = dyad_neg(&rad, lo)) && !(rc = dyad_add_exact(&rad, hi, &rad)) && !(rc = dyad_shift(&mid, -1)) && !(rc = dyad_shift(&rad, -1)))
        rc = imat_set(M, part, i, j, &mid, &rad);
    dyad_free(&mid);
    dyad_free(&rad);
    return rc;
}

/* [e] [f] for e[0] > 0, rounded outward to p bits */
static int scale_ends(dyad *out, const dyad *e, const dyad *f, int64_t p) {
    int rc = dyad_mul(&out[0], dyad_sign(&f[0]) >= 0 ? &e[0] : &e[1], &f[0], p, 0);
    return rc ? rc : dyad_mul(&out[1], dyad_sign(&f[1]) >= 0 ? &e[1] : &e[0], &f[1], p, 1);
}

/* exp of a diagonal A into x->m[0]: each diagonal entry's scalar exp, e**[a] = [exp(a_lo), exp(a_hi)] and for a complex
   entry e**[a] (cos [b] + i sin [b]) (matrix_exp.md, Diagonal matrices) */
static int diagonal_exp(mexp *x, const imat *A) {
    int L = x->L, P = x->parts, rc = imat_init(&x->m[0], x->n, x->n, L, P);
    int64_t p = (int64_t)LB * L + 8;
    dyad lo, hi, e[2], c[2], s[2], out[2];
    dyad_init(&lo);
    dyad_init(&hi);
    for (int k = 0; k < 2; k++) {
        dyad_init(&e[k]);
        dyad_init(&c[k]);
        dyad_init(&s[k]);
        dyad_init(&out[k]);
    }
    for (size_t i = 0; !rc && i < x->n; i++) {
        if ((rc = imat_ends(&lo, &hi, A, 0, i, i)) || (rc = exp_end(&e[0], &lo, p, 0)) || (rc = exp_end(&e[1], &hi, p, 1))) break;
        if (P == 1) {
            rc = imat_set_ends(&x->m[0], 0, i, i, &e[0], &e[1]);
            continue;
        }
        /* cos and sin over [b] */
        if ((rc = imat_ends(&lo, &hi, A, 1, i, i)) || (rc = cos_sin_range(c, s, &lo, &hi, p))) break;
        if ((rc = scale_ends(out, e, c, p)) || (rc = imat_set_ends(&x->m[0], 0, i, i, &out[0], &out[1])) || (rc = scale_ends(out, e, s, p))) break;
        rc = imat_set_ends(&x->m[0], 1, i, i, &out[0], &out[1]);
    }
    dyad_free(&lo);
    dyad_free(&hi);
    for (int k = 0; k < 2; k++) {
        dyad_free(&e[k]);
        dyad_free(&c[k]);
        dyad_free(&s[k]);
        dyad_free(&out[k]);
    }
    return rc;
}

/* whether every entry of A off the diagonal is exactly 0 */
static int imat_diagonal(const imat *A) {
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = 0; j < A->cols; j++)
            if (i != j && !imat_zero_at(A, i, j)) return 0;
    return 1;
}

/* Prepares exp(A) for A (n x n, L limbs, real or complex; A is copied): the norm, s, N and q, the
   remainder, the closure and the phases; a diagonal A's result directly, with no phases. *bounded is
   0 where the norm bound reaches 2**60, beyond the exponents the kernel holds (x->past set), or where
   the products' sums would not fit their words (matrix_exp.md, The algorithm). */
int mexp_init(mexp *x, const imat *A, int *bounded) {
    mexp_clear(x);
    int L = A->L, P = A->parts, rc;
    size_t n = A->rows;
    x->n = n;
    x->L = L;
    x->parts = P;
    ub norm = imat_norm(A);
    int diagonal = imat_diagonal(A);
    x->past = norm.m && ub_top(norm) > 60;
    *bounded = x->bounded = !x->past && (diagonal || imat_product_fits(n, L, P));
    if (!x->bounded) return 0;
    if (diagonal) return diagonal_exp(x, A);
    int64_t ce = norm.m ? ub_top(norm) : INT64_MIN; /* every ||X|| < 2**ce */
    ub target = ub_make(1, -(int64_t)LB * L);
    int best = -1, rmin = 0;
    /* With radii past half the precision relative to the norm (a box, not rounding errors), ||B|| <= 2**-MEXP_BOX_SCALE:
       an alternating Taylor sum's interval radius exceeds the true spread by at most e**(2 ||B||) (about 1.002), where
       with few squarings it could reach e**(2 ||A||); rounding errors alone take the fewest products */
    for (size_t at = 0; at < (size_t)P * n * n && !rmin && ce != INT64_MIN; at++)
        if (A->r[at].m && ub_top(A->r[at]) > ce - (int64_t)LB * L / 2) rmin = MEXP_BOX_SCALE;
    for (int r = rmin; r <= 40 + rmin; r++) {
        int64_t s = ce == INT64_MIN || ce + r < 0 ? 0 : ce + r;
        ub b = ub_shift(norm, -s), term = b, tail = ub_shift(b, 1); /* tail(N) = 2 b**(N+1) / (N+1)! */
        int N = 0;
        while (!ub_le(tail, target) && N < 100000) {
            N++;
            term = ub_div_word(ub_mul(term, b), (uint64_t)N + 1);
            tail = ub_shift(term, 1);
        }
        int q = 1;
        while ((q + 1) * (q + 1) <= N + 1) q++;
        if (q * q < N + 1) q++;
        int cost = (int)s + (q - 1) + ((N + 1 + q - 1) / q - 1);
        if (best < 0 || cost < best) {
            best = cost;
            x->s = s;
            x->N = N;
            x->q = q;
            x->tail = tail;
        }
        if (ce == INT64_MIN) break;
    }
    int q = x->q, N = x->N, blocks = (N + 1 + q - 1) / q;
    if ((rc = factorials(x)) || (rc = closure(x, A))) return rc;
    x->pw = calloc((size_t)q + 1, sizeof *x->pw);
    if (!x->pw) return EXACTREF_NOMEM;
    for (int k = 1; k <= q; k++)
        if ((rc = imat_init(&x->pw[k], n, n, L, P))) return rc;
    for (int k = 0; k < 2; k++)
        if ((rc = imat_init(&x->m[k], n, n, L, P))) return rc;
    if ((rc = imat_lines_init(&x->rows, n, n, L, P)) || (rc = imat_lines_init(&x->cols, n, n, L, P))) return rc;
    /* B = A 2**-s */
    size_t count = (size_t)P * n * n;
    memcpy(x->pw[1].d, A->d, count * L * sizeof *A->d);
    for (size_t at = 0; at < count; at++) {
        x->pw[1].e[at] = A->e[at] - x->s;
        x->pw[1].r[at] = ub_shift(A->r[at], -x->s);
    }
    for (size_t at = 0; P == 2 && at < n * n; at++) x->pw[1].rd[at] = ub_shift(A->rd[at], -x->s);
    /* the phases: B's columns, the powers, the Horner blocks, the remainder, the squarings */
    x->S = 0;
    if (q > 1 && (rc = add_op(x, OP_COLS, 1))) return rc;
    for (int k = 2; k <= q; k++)
        if ((rc = add_op(x, OP_POWER, k))) return rc;
    if (blocks > 1 && (rc = add_op(x, OP_COLS, q))) return rc;
    for (int jb = blocks - 1; jb >= 0; jb--)
        if ((rc = add_op(x, jb == blocks - 1 ? OP_TERMS : OP_HORNER, jb))) return rc;
    if ((rc = add_op(x, OP_TAIL, 0))) return rc;
    for (int64_t k = 0; k < x->s; k++)
        if ((rc = add_op(x, OP_COLS, -1)) || (rc = add_op(x, OP_SQUARE, 0))) return rc;
    return 0;
}

/* the tiles of phase k: lines or rows in groups of MEXP_TILE_ROWS */
size_t mexp_tiles(const mexp *x, size_t k) { return k < x->nops ? (x->n + MEXP_TILE_ROWS - 1) / MEXP_TILE_ROWS : 0; }

/* S += sum over i < q, jb q + i <= N of c_(jb q + i) B**i, rows i0 .. i1 (B**0 = I) */
static void terms_rows(mexp *x, imat *S, int jb, size_t i0, size_t i1) {
    int L = x->L, P = x->parts, q = x->q;
    size_t n = x->n;
    int64_t one[LMAX] = {0};
    one[L - 1] = (int64_t)1 << (LB - 1);
    int64_t eone = -((int64_t)LB * L - 1);
    ub zero = {0, 0};
    for (int i = 0; i < q && jb * q + i <= x->N; i++) {
        int k = jb * q + i;
        const int64_t *c = x->c + (size_t)k * L;
        for (size_t r = i0; r < i1; r++)
            for (size_t col = 0; col < n; col++) {
                ub tr[2] = {{0, 0}, {0, 0}};
                const imat *B = &x->pw[i];
                for (int p = 0; p < P; p++) {
                    size_t at = at_of(S, p, r, col);
                    if (i == 0) {
                        if (r == col && p == 0) fx_axpy(S->d + at * L, &S->e[at], &S->r[at], &tr[0], c, x->ec[k], x->rc[k], one, eone, zero, L);
                    } else {
                        fx_axpy(S->d + at * L, &S->e[at], &S->r[at], &tr[p], c, x->ec[k], x->rc[k], B->d + at * L, B->e[at], B->r[at], L);
                    }
                }
                if (P < 2 || (i == 0 && r != col)) continue;
                /* the disk: |c| RX + rc |x| + rc RX of the moduli and disk radii, and the parts dropped; then each part
                   within the disk and the disk within the parts' */
                size_t e = r * n + col, a0 = at_of(S, 0, r, col), a1 = at_of(S, 1, r, col);
                ub cm = fx_mag(c, L, x->ec[k]), xm = {(uint64_t)1 << 31, -31}, xd = zero; /* the identity term: |x| = 1, no radius */
                if (i > 0) {
                    size_t b0 = at_of(B, 0, r, col), b1 = at_of(B, 1, r, col);
                    xm = ub_hypot(fx_mag(B->d + b0 * L, L, B->e[b0]), fx_mag(B->d + b1 * L, L, B->e[b1]));
                    xd = B->rd[e];
                }
                ub inc = ub_add(ub_add(ub_mul(cm, xd), ub_mul(x->rc[k], xm)), ub_mul(x->rc[k], xd));
                S->rd[e] = ub_add(ub_add(S->rd[e], inc), ub_add(tr[0], tr[1]));
                if (ub_le(S->rd[e], S->r[a0])) S->r[a0] = S->rd[e];
                if (ub_le(S->rd[e], S->r[a1])) S->r[a1] = S->rd[e];
                ub h = ub_hypot(S->r[a0], S->r[a1]);
                if (ub_le(h, S->rd[e])) S->rd[e] = h;
            }
    }
}

static void clear_rows(imat *M, size_t i0, size_t i1) {
    for (int p = 0; p < M->parts; p++)
        for (size_t i = i0; i < i1; i++) {
            size_t at = at_of(M, p, i, 0);
            memset(M->d + at * M->L, 0, M->cols * M->L * sizeof *M->d);
            memset(M->e + at, 0, M->cols * sizeof *M->e);
            memset(M->r + at, 0, M->cols * sizeof *M->r);
            if (M->rd && p == 0) memset(M->rd + i * M->cols, 0, M->cols * sizeof *M->rd);
        }
}

/* runs tile t of phase k; phases run in order, a phase's tiles in any order or at once */
void mexp_run(mexp *x, size_t k, size_t t) {
    const mexp_op *op = &x->ops[k];
    size_t i0 = t * MEXP_TILE_ROWS, i1 = i0 + MEXP_TILE_ROWS < x->n ? i0 + MEXP_TILE_ROWS : x->n;
    imat *S = &x->m[x->S], *U = &x->m[1 - x->S];
    switch (op->kind) {
    case OP_COLS: /* a right factor's columns: B**arg, or (arg -1) the running square S */
        imat_lines_prepare(&x->cols, op->arg < 0 ? S : &x->pw[op->arg], 1, i0, i1);
        break;
    case OP_POWER: /* B**arg = B**(arg - 1) B */
        imat_lines_prepare(&x->rows, &x->pw[op->arg - 1], 0, i0, i1);
        imat_product_rows(&x->pw[op->arg], &x->rows, &x->cols, i0, i1);
        break;
    case OP_TERMS: /* S = the last block's terms */
        clear_rows(S, i0, i1);
        terms_rows(x, S, op->arg, i0, i1);
        break;
    case OP_HORNER: /* S = S B**q + the block's terms, through U */
        imat_lines_prepare(&x->rows, S, 0, i0, i1);
        imat_product_rows(U, &x->rows, &x->cols, i0, i1);
        terms_rows(x, U, op->arg, i0, i1);
        break;
    case OP_TAIL: /* the remainder where a walk connects the entries */
        for (size_t i = i0; i < i1; i++)
            for (size_t j = 0; j < x->n; j++)
                if (x->reach[i * x->words + j / 64] >> (j % 64) & 1) {
                    for (int p = 0; p < x->parts; p++) {
                        size_t at = at_of(S, p, i, j);
                        S->r[at] = ub_add(S->r[at], x->tail);
                    }
                    if (S->rd) S->rd[i * S->cols + j] = ub_add(S->rd[i * S->cols + j], x->tail); /* |R_ij| <= tail */
                }
        break;
    case OP_SQUARE: /* U = S S */
        imat_lines_prepare(&x->rows, S, 0, i0, i1);
        imat_product_rows(U, &x->rows, &x->cols, i0, i1);
        break;
    }
}

/* the end of phase k, after all its tiles: a Horner step's or a square's result becomes S */
void mexp_done(mexp *x, size_t k) {
    if (k < x->nops && (x->ops[k].kind == OP_HORNER || x->ops[k].kind == OP_SQUARE)) x->S = 1 - x->S;
}

/* runs every phase of x on this thread */
void mexp_run_all(mexp *x) {
    for (size_t k = 0; k < x->nops; k++) {
        size_t tiles = mexp_tiles(x, k);
        for (size_t t = 0; t < tiles; t++) mexp_run(x, k, t);
        mexp_done(x, k);
    }
}

/* the result once every phase has run */
const imat *mexp_result(const mexp *x) { return &x->m[x->S]; }

/* its matrix products (the powers, the Horner steps and the squarings), each n**3 products of L-limb entries */
size_t mexp_products(const mexp *x) {
    size_t k = 0;
    for (size_t i = 0; i < x->nops; i++) k += x->ops[i].kind == OP_POWER || x->ops[i].kind == OP_HORNER || x->ops[i].kind == OP_SQUARE;
    return k;
}

/* ---------------------------------------------------------------- batches */

void mexp_batch_free(mexp_batch *b) {
    if (b->in)
        for (size_t m = 0; m < b->count; m++) imat_free(&b->in[m]);
    if (b->out)
        for (size_t m = 0; m < b->count; m++) imat_free(&b->out[m]);
    free(b->in);
    free(b->out);
    free(b->ok);
    mexp_free(&b->cur);
    memset(b, 0, sizeof *b);
}

/* room for `count` matrices n x n of L limbs and `parts` parts, set by imat_set on b->in[m] */
int mexp_batch_init(mexp_batch *b, size_t count, size_t n, int L, int parts) {
    memset(b, 0, sizeof *b);
    b->count = count;
    b->n = n;
    b->L = L;
    b->parts = parts;
    b->whole = n <= MEXP_SMALL;
    b->in = calloc(count + 1, sizeof *b->in);
    b->out = calloc(count + 1, sizeof *b->out);
    b->ok = calloc(count + 1, 1);
    if (!b->in || !b->out || !b->ok) return EXACTREF_NOMEM;
    for (size_t m = 0; m < count; m++) {
        imat_clear(&b->out[m]);
        int rc = imat_init(&b->in[m], n, n, L, parts);
        if (rc) return rc;
    }
    return 0;
}

/* matrix m's result from a finished exponential */
static int keep_result(mexp_batch *b, size_t m, const mexp *x) {
    const imat *R = mexp_result(x);
    int rc = imat_init(&b->out[m], b->n, b->n, b->L, b->parts);
    if (rc) return rc;
    size_t count = (size_t)b->parts * b->n * b->n;
    memcpy(b->out[m].d, R->d, count * b->L * sizeof *R->d);
    memcpy(b->out[m].e, R->e, count * sizeof *R->e);
    memcpy(b->out[m].r, R->r, count * sizeof *R->r);
    if (R->rd) memcpy(b->out[m].rd, R->rd, b->n * b->n * sizeof *R->rd);
    b->ok[m] = 1;
    return 0;
}

/* The tiles of the next phase (0 when every exponential is done). Small matrices: one phase, each
   tile MEXP_GROUP whole exponentials. Larger ones: each matrix's phases in turn, in row tiles. */
int mexp_batch_next(mexp_batch *b, size_t *tiles) {
    *tiles = 0;
    if (b->whole) {
        if (!b->started) {
            b->started = 1;
            *tiles = (b->count + MEXP_GROUP - 1) / MEXP_GROUP;
        }
        return 0;
    }
    int rc = 0;
    if (b->started) mexp_done(&b->cur, b->k++);
    for (;;) {
        if (b->started && b->k < b->cur.nops) {
            *tiles = mexp_tiles(&b->cur, b->k);
            return 0;
        }
        if (b->started) { /* the current matrix is done */
            if (b->cur.bounded && (rc = keep_result(b, b->mi, &b->cur))) return rc;
            mexp_free(&b->cur);
            imat_free(&b->in[b->mi]);
            b->mi++;
        }
        if (b->mi >= b->count) return 0;
        int bounded;
        b->started = 1;
        b->k = 0;
        if (b->in[b->mi].rows != b->n) { /* a non-finite entry: no result */
            mexp_clear(&b->cur);
            continue;
        }
        if ((rc = mexp_init(&b->cur, &b->in[b->mi], &bounded))) return rc;
    }
}

/* runs tile t of the current phase */
int mexp_batch_tile(mexp_batch *b, size_t t) {
    if (!b->whole) {
        mexp_run(&b->cur, b->k, t);
        return 0;
    }
    int rc = 0;
    for (size_t m = t * MEXP_GROUP; !rc && m < b->count && m < (t + 1) * MEXP_GROUP; m++) {
        mexp x;
        int bounded;
        if (b->in[m].rows != b->n) continue; /* a non-finite entry: no result */
        rc = mexp_init(&x, &b->in[m], &bounded);
        if (!rc && bounded) {
            mexp_run_all(&x);
            rc = keep_result(b, m, &x);
        }
        mexp_free(&x);
    }
    return rc;
}
