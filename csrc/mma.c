/* Sums by tensor-core steps (internal; proofs/foundations.md, Lemma T): the constants G and H of
   |s' - s| <= G M + H, rounded up as ops/_mma.py's mma_terms rounds them, and per element the interval
   of every finite output's value before its final rounding with the flags of the products' stage,
   computed exactly from the kernel's sums. */
#include "exactref.h"

/* The most steps on a product's path for K products (the method's depth): ceil(K / N), and one more per piece where
   pieces may end mid-step (split not a multiple of N). */
int64_t mma_steps(int64_t N, int64_t split, int64_t K) {
    int64_t full = (K + N - 1) / N, pieces = (K + split - 1) / split;
    return full + (split % N ? (pieces - 1 > 0 ? pieces - 1 : 0) : 0);
}

/* H for gemm's alpha P: Lemma T's absolute part times max(1, |alpha|) (proofs/gemm.md), rounded up to w bits */
int mma_alpha(dyad *H, const dyad *alpha, int64_t w) {
    dyad one, a;
    dyad_init(&one);
    a = *alpha; /* a read-only view of |alpha| */
    a.m.neg = 0;
    int rc = dyad_set_si(&one, 1, 0);
    if (!rc && dyad_cmp(&a, &one) > 0) rc = dyad_mul(H, H, &a, w, 1);
    dyad_free(&one);
    return rc;
}

/* G and H for a sum of K products (proofs/foundations.md, Lemma T, and its gemm paragraph): steps
   of N products with f fraction bits and exponent floor e_f, pieces on multiples of split, the
   optional reduction format (NULL for none), the slack sigma, extra float32 roundings on every
   path, and the steps on a product's path (steps >= 0) or the method's count (steps < 0):

       G = (1 + rho)**(K + r) (1 + tau)**r (1 + 2**-24)**extra (1 + u_R) - 1,
       H = (G + 1) (r_all h + s eta_R + extra 2**-150),

   with rho = 2**-f (1 + 2**-23), tau = 2**-23 and h = (N + 1) rho (sigma + 2**e_f + 2**-126) +
   2**-149. Both are 0 without steps or extra roundings. */
int mma_terms(dyad *G, dyad *H, int64_t N, int64_t f, int64_t e_f, int64_t split, const floatfmt *reduction, const dyad *sigma, int64_t K,
              int64_t extra, int64_t steps, int64_t w) {
    int64_t full = (K + N - 1) / N, pieces = (K + split - 1) / split;
    int64_t r = steps >= 0 ? steps : mma_steps(N, split, K);
    int64_t r_all = steps >= 0 ? r : full + (pieces - 1 > 0 ? pieces - 1 : 0);
    int rc = dyad_set_si(G, 0, 0);
    if (!rc) rc = dyad_set_si(H, 0, 0);
    if (rc || (!r && !extra)) return rc;
    dyad rho, tau, t, u, h, growth;
    dyad *all[] = {&rho, &tau, &t, &u, &h, &growth};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    /* rho = 2**-f + 2**(-f - 23), exactly */
    if (!(rc = dyad_set_si(&rho, 1, -f)) && !(rc = dyad_set_si(&t, 1, -f - 23))) rc = dyad_add_exact(&rho, &rho, &t);
    if (!rc) rc = dyad_set_si(&tau, 1, -23);
    /* h = (N + 1) rho (sigma + 2**e_f + 2**-126) + 2**-149, exactly */
    if (!rc && !(rc = dyad_set_si(&t, 1, e_f)) && !(rc = dyad_add_exact(&h, sigma, &t)) && !(rc = dyad_set_si(&t, 1, -126)) &&
        !(rc = dyad_add_exact(&h, &h, &t)) && !(rc = dyad_mul_exact(&h, &h, &rho)) && !(rc = dyad_set_si(&t, N + 1, 0)) &&
        !(rc = dyad_mul_exact(&h, &h, &t)) && !(rc = dyad_set_si(&t, 1, -149)))
        rc = dyad_add_exact(&h, &h, &t);
    /* growth = (1 + rho)**(K + r) (1 + tau)**r (1 + 2**-24)**extra (1 + u_R), rounded up */
    if (!rc) rc = growth_up(&growth, &rho, K + r, w);
    if (!rc && !(rc = growth_up(&t, &tau, r, w))) rc = dyad_mul(&growth, &growth, &t, w, 1);
    if (!rc && !(rc = dyad_set_si(&u, 1, -24)) && !(rc = growth_up(&t, &u, extra, w))) rc = dyad_mul(&growth, &growth, &t, w, 1);
    if (!rc && reduction && !(rc = dyad_set_si(&u, 1, -(reduction->man_bits + 1))) && !(rc = growth_up(&t, &u, 1, w)))
        rc = dyad_mul(&growth, &growth, &t, w, 1);
    if (!rc) {
        dyad one;
        dyad_init(&one);
        if (!(rc = dyad_set_si(&one, -1, 0))) rc = dyad_add(G, &growth, &one, w, 1);
        dyad_free(&one);
    }
    /* H = growth (r_all h + pieces eta_R + extra 2**-150), the sum exact and then rounded up */
    if (!rc && !(rc = dyad_set_si(&t, r_all, 0)) && !(rc = dyad_mul_exact(&u, &h, &t))) {
        if (reduction && !(rc = dyad_set_si(&t, pieces, reduction->emin - reduction->man_bits - 1))) rc = dyad_add_exact(&u, &u, &t);
        if (!rc && !(rc = dyad_set_si(&t, extra, -150))) rc = dyad_add_exact(&u, &u, &t);
        if (!rc && !(rc = dyad_rnd(&u, &u, w, 1))) rc = dyad_mul(H, &growth, &u, w, 1);
    }
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
    return rc;
}

/* The flags of a stage whose values lie within err of partial sums of terms whose positive ones
   total pos and negative ones neg: an infinity of a side where that total plus err reaches one of
   the thresholds, NaN where both can, each NaN in a format without infinity. */
int mma_overflow(int *flags, const dyad *pos, const dyad *neg, const dyad *err, const dyad *thresholds, int count, int has_infinity) {
    dyad t;
    dyad_init(&t);
    int rc = 0;
    *flags = 0;
    for (int side = 0; !rc && side < 2; side++) {
        if ((rc = dyad_add_exact(&t, side ? neg : pos, err))) break;
        for (int k = 0; k < count; k++)
            if (dyad_cmp(&t, &thresholds[k]) >= 0) *flags |= side ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    }
    if (*flags == (FLAG_PLUS_INF | FLAG_MINUS_INF)) *flags |= FLAG_NAN;
    if (*flags && !has_infinity) *flags = FLAG_NAN;
    dyad_free(&t);
    return rc;
}

/* err = G (mass 2**e) + H, exactly. */
int mma_error(dyad *err, const dyad *G, const dyad *H, const bigint *mass, int64_t e) {
    dyad v;
    dyad_view(&v, mass, e);
    int rc = dyad_mul_exact(err, G, &v);
    return rc ? rc : dyad_add_exact(err, err, H);
}

/* [lo, hi] = [s_lo 2**e - err, s_hi 2**e + err], clamped to float32's finite range (top its largest
   finite value). */
int mma_interval(dyad *lo, dyad *hi, const bigint *s_lo, const bigint *s_hi, int64_t e, const dyad *err, const dyad *top) {
    dyad v, n = *err, least = *top; /* read-only views of -err and -top */
    n.m.neg = n.m.n != 0;
    least.m.neg = least.m.n != 0;
    dyad_view(&v, s_lo, e);
    int rc = dyad_add_exact(lo, &v, &n);
    if (!rc) {
        dyad_view(&v, s_hi, e);
        rc = dyad_add_exact(hi, &v, err);
    }
    if (rc) return rc;
    if (dyad_cmp(lo, &least) < 0) rc = dyad_copy(lo, &least);
    if (!rc && dyad_cmp(lo, top) > 0) rc = dyad_copy(lo, top);
    if (!rc && dyad_cmp(hi, top) > 0) rc = dyad_copy(hi, top);
    if (!rc && dyad_cmp(hi, &least) < 0) rc = dyad_copy(hi, &least);
    return rc;
}

/* An operand's extent for Lemma T's slack (ops/_mma.py): over its n parts of format f, *sub where a
   finite one may be a nonzero value below 2**emin_f (always with radii: an earlier result may
   hold one), and the largest |value| among the finite ones, with radii (float64, n of them) the
   largest |value| plus its finite radius. */
int mma_extent(int *sub, dyad *largest, const partfmt *f, const unsigned char *data, int big, size_t n, const partfmt *f64,
               const unsigned char *radii, int radii_big) {
    value v, r;
    dyad tiny, t;
    dyad_init(&v.v);
    dyad_init(&r.v);
    dyad_init(&tiny);
    dyad_init(&t);
    *sub = radii != NULL;
    int rc = dyad_set_si(largest, 0, 0);
    if (!rc) rc = dyad_set_si(&tiny, 1, f->f.emin);
    for (size_t i = 0; !rc && i < n; i++) {
        if ((rc = part_read(&v, f, data, i, big))) break;
        if (v.kind != VALUE_FINITE) continue;
        dyad mag = v.v; /* a read-only view of |value| */
        mag.m.neg = 0;
        const dyad *size = &mag;
        if (radii) {
            if ((rc = part_read(&r, f64, radii, i, radii_big))) break;
            if (r.kind != VALUE_FINITE) continue;
            if ((rc = dyad_add_exact(&t, &mag, &r.v))) break;
            size = &t;
        } else if (mag.m.n && dyad_cmp(&mag, &tiny) < 0) {
            *sub = 1;
        }
        if (dyad_cmp(size, largest) > 0) rc = dyad_copy(largest, size);
    }
    dyad_free(&v.v);
    dyad_free(&r.v);
    dyad_free(&tiny);
    dyad_free(&t);
    return rc;
}

/* Lemma T's sigma for products of two factors whose format has smallest normal 2**emin, each
   given by its extent: tiny max_y where x may be subnormal, plus tiny max_x where y may be, plus
   tiny**2 where both may (tiny = 2**emin), exactly. */
int mma_slack(dyad *sigma, int64_t emin, int sub_x, const dyad *max_x, int sub_y, const dyad *max_y) {
    dyad tiny, t;
    dyad_init(&tiny);
    dyad_init(&t);
    int rc = dyad_set_si(sigma, 0, 0);
    if (!rc) rc = dyad_set_si(&tiny, 1, emin);
    if (!rc && sub_x && !(rc = dyad_mul_exact(&t, &tiny, max_y))) rc = dyad_add_exact(sigma, sigma, &t);
    if (!rc && sub_y && !(rc = dyad_mul_exact(&t, &tiny, max_x))) rc = dyad_add_exact(sigma, sigma, &t);
    if (!rc && sub_x && sub_y && !(rc = dyad_mul_exact(&t, &tiny, &tiny))) rc = dyad_add_exact(sigma, sigma, &t);
    dyad_free(&tiny);
    dyad_free(&t);
    return rc;
}
