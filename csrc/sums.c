/* Bounds on sums accumulated in a float format (internal; proofs/foundations.md, The normal
   standard): Lemma S's normal bound with Lemma O's overflow check, intersected with Lemma A's
   bound where that check passes (Lemma O (e)), and Lemma A's bound for any order that does not
   overflow, with the sides on which the final value may overflow. The constants are rounded up
   once; each element's error is then computed exactly from its exact quantities. */
#include "exactref.h"

static void init_all(dyad **all, size_t n) {
    for (size_t i = 0; i < n; i++) dyad_init(all[i]);
}

static void free_all(dyad **all, size_t n) {
    for (size_t i = 0; i < n; i++) dyad_free(all[i]);
}

#define BOUND_DYADS(b) {&(b)->half_r, &(b)->c_s, &(b)->g, &(b)->any_r, &(b)->any_g, &(b)->any_c, &(b)->side_g, &(b)->unit_r, &(b)->unit_a, &(b)->gamma, &(b)->constant, &(b)->cap_r, &(b)->cap_g, &(b)->cap_c, &(b)->theta}

void sum_bounds_free(sum_bounds *b) {
    dyad *all[] = BOUND_DYADS(b);
    free_all(all, sizeof all / sizeof *all);
}

/* The constants for sums of n terms in the format f (rounding to nearest, flush to zero or not),
   over a tree of the given depth for Lemma S, rounded up to w bits. With exact_terms no term is rounded on its own
   (proofs/foundations.md, Lemma U): every term a value of f (TERMS_HELD), whose additions have the absolute part a2, or
   exact values f need not hold (TERMS_EXACT), whose additions have a. */
int sum_bounds_init(sum_bounds *b, const floatfmt *f, int flush, int64_t n, int64_t depth, int exact_terms, int64_t w) {
    dyad *all[] = BOUND_DYADS(b);
    init_all(all, sizeof all / sizeof *all);
    dyad r, a, a2, G, t, u, largest;
    dyad *tmp[] = {&r, &a, &a2, &G, &t, &u, &largest};
    init_all(tmp, sizeof tmp / sizeof *tmp);
    b->n = n;
    b->emin = f->emin;
    b->flush = flush;
    b->exact_terms = exact_terms;
    b->has_infinity = f->special == SPECIAL_IEEE;
    int64_t p = f->man_bits + 1, leaves = exact_terms ? 0 : n, edges = n > 0 ? n - 1 : 0;
    /* Units: r = 2**-p; a = 2**(emin - p), or 2**emin with flush; a2 = 0, or 2**emin with flush */
    int rc = dyad_set_si(&r, 1, -p);
    if (!rc) rc = dyad_set_si(&a, 1, flush ? f->emin : f->emin - p);
    if (!rc) rc = dyad_set_si(&a2, flush ? 1 : 0, f->emin);
    /* theta: the largest finite value plus half its ulp */
    if (!rc) rc = floatfmt_threshold(&b->theta, NULL, f);
    /* leaves a + edges a2 (edges a with TERMS_EXACT), the constant part of Lemma S and B_any */
    if (!rc && !(rc = dyad_times(&t, &a, leaves, w)) && !(rc = dyad_times(&u, exact_terms == TERMS_EXACT ? &a : &a2, edges, w)))
        rc = dyad_add(&u, &t, &u, w, 1);
    /* Lemma S: G (r N / 2 + leaves a + edges a2 + moved) */
    if (!rc && !(rc = growth_up(&G, &r, depth, w)) && !(rc = dyad_copy(&b->g, &G))) {
        dyad half = r; /* a read-only view of r / 2 */
        if (!(rc = dyad_shift(&half, -1)) && !(rc = dyad_mul(&b->half_r, &G, &half, w, 1))) rc = dyad_mul(&b->c_s, &G, &u, w, 1);
    }
    /* B_any (Lemma O (c)): Growth(n - 1) (n r mass + leaves a + edges a2 + moved) */
    if (!rc && !(rc = growth_up(&G, &r, edges, w)) && !(rc = dyad_copy(&b->any_g, &G)) && !(rc = dyad_times(&t, &r, n, w)) &&
        !(rc = dyad_mul(&b->any_r, &G, &t, w, 1)))
        rc = dyad_mul(&b->any_c, &G, &u, w, 1);
    /* a side's reach (Lemma O (g)): Growth(n - 1) ((1 + r) side + leaves a + edges a2) = side side_g + any_c */
    if (!rc && !(rc = dyad_set_si(&t, 1, 0)) && !(rc = dyad_add(&t, &t, &r, w, 1))) rc = dyad_mul(&b->side_g, &b->any_g, &t, w, 1);
    if (!rc && !(rc = dyad_copy(&b->unit_r, &r))) rc = dyad_copy(&b->unit_a, &a);
    /* Lemma A, Lemma 2's form: gamma mass + constant + moved, where Lemma 2 applies */
    b->bounded = 0;
    if (!rc && n) {
        int64_t d, roundings;
        any_order_counts(&d, &roundings, n, flush, exact_terms);
        rc = growth_terms(&b->gamma, &b->constant, &largest, &b->bounded, f, ROUND_NEAREST_EVEN, d, roundings, w, flush);
    }
    /* Lemma A, capped: Growth(n) (r mass + n a + edges (r theta + a2) + moved) */
    if (!rc && !(rc = growth_up(&G, &r, n, w)) && !(rc = dyad_copy(&b->cap_g, &G)) && !(rc = dyad_mul(&b->cap_r, &G, &r, w, 1)) &&
        !(rc = dyad_mul(&t, &r, &b->theta, w, 1)) && !(rc = dyad_add(&t, &t, &a2, w, 1)) && !(rc = dyad_times(&t, &t, edges, w)) &&
        !(rc = dyad_times(&u, &a, n, w)) && !(rc = dyad_add(&t, &u, &t, w, 1)))
        rc = dyad_mul(&b->cap_c, &G, &t, w, 1);
    free_all(tmp, sizeof tmp / sizeof *tmp);
    if (rc) sum_bounds_free(b);
    return rc;
}

/* r += x 2**e * c exactly. */
static int add_scaled(dyad *r, const bigint *x, int64_t e, const dyad *c, dyad *t) {
    dyad v;
    dyad_view(&v, x, e);
    int rc = dyad_mul_exact(t, &v, c);
    return rc ? rc : dyad_add_exact(r, r, t);
}

/* Lemma A's error for an element's mass and moved magnitudes (times 2**e), exactly from the
   constants: the smaller of Lemma 2's form, where it applies, and the capped form. */
static int lemma_a(dyad *err, const sum_bounds *b, const bigint *mass, const bigint *moved, int64_t e) {
    dyad other, t;
    dyad_init(&other);
    dyad_init(&t);
    int rc = dyad_copy(err, &b->cap_c);
    if (!rc) rc = add_scaled(err, mass, e, &b->cap_r, &t);
    if (!rc) rc = add_scaled(err, moved, e, &b->cap_g, &t);
    if (!rc && b->bounded) {
        dyad one;
        dyad_init(&one);
        if (!(rc = dyad_copy(&other, &b->constant)) && !(rc = add_scaled(&other, mass, e, &b->gamma, &t)) && !(rc = dyad_set_si(&one, 1, 0)))
            rc = add_scaled(&other, moved, e, &one, &t);
        dyad_free(&one);
        if (!rc && dyad_cmp(&other, err) < 0) rc = dyad_copy(err, &other);
    }
    dyad_free(&other);
    dyad_free(&t);
    return rc;
}

/* The flags of the sides on which a value within B_any of s may reach theta (Lemma A): none
   where |s| + B_any < theta, else the side of s's sign, and the other where B_any - |s| reaches
   theta too; NaN for each in a format without infinity. */
static int sides(int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *moved, int64_t e) {
    dyad any, t, size;
    dyad_init(&any);
    dyad_init(&t);
    dyad_init(&size);
    *flags = 0;
    int rc = dyad_copy(&any, &b->any_c);
    if (!rc) rc = add_scaled(&any, mass, e, &b->any_r, &t);
    if (!rc) rc = add_scaled(&any, moved, e, &b->any_g, &t);
    if (!rc) {
        dyad v;
        dyad_view(&v, s, e); /* |s| 2**e */
        v.m.neg = 0;
        rc = dyad_add_exact(&size, &any, &v);
        if (!rc && dyad_cmp(&size, &b->theta) >= 0) {
            *flags = s->neg ? FLAG_MINUS_INF : FLAG_PLUS_INF;
            v.m.neg = v.m.n != 0; /* -|s| */
            if (!(rc = dyad_add_exact(&size, &any, &v)) && dyad_cmp(&size, &b->theta) >= 0) *flags |= FLAG_PLUS_INF | FLAG_MINUS_INF;
            if (!b->has_infinity) *flags = FLAG_NAN;
        }
    }
    dyad_free(&any);
    dyad_free(&t);
    dyad_free(&size);
    return rc;
}

/* lo, hi = s 2**e -+ err. */
static int around(dyad *lo, dyad *hi, const bigint *s, int64_t e, const dyad *err) {
    dyad v, n = *err;
    dyad_view(&v, s, e);
    n.m.neg = n.m.n != 0;
    int rc = dyad_add_exact(lo, &v, &n);
    return rc ? rc : dyad_add_exact(hi, &v, err);
}

static int trivial(const sum_bounds *b, const bigint *mass, const bigint *moved) { return !b->n || (!mass->n && !moved->n); }

/* Lemma S for an element's exact sum s, mass, N >= 2 (mass + sum of |node|) and moved
   magnitudes, all times 2**e: [lo, hi] where Lemma O's check passes (*checked), intersected with
   Lemma A's bound (Lemma O (e)); otherwise Lemma A's interval, which holds for every output that
   did not overflow, and *flags the sides on which the last pass's value may overflow (terms that
   may overflow are the caller's). Terms all exactly 0 give s itself. */
int sum_bounds_normal(dyad *lo, dyad *hi, int *checked, int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *N,
                      const bigint *moved, int64_t e) {
    *checked = 1;
    *flags = 0;
    dyad zero;
    dyad_init(&zero);
    if (trivial(b, mass, moved)) {
        int rc = around(lo, hi, s, e, &zero);
        dyad_free(&zero);
        return rc;
    }
    dyad err, other, t;
    bigint nodes;
    dyad_init(&err);
    dyad_init(&other);
    dyad_init(&t);
    big_init(&nodes);
    /* with exact terms, N less 2 mass: no term is rounded on its own */
    int rc = big_copy(&nodes, N);
    if (!rc && b->exact_terms) {
        bigint twice;
        big_init(&twice);
        if (!(rc = big_shl(&twice, mass, 1))) {
            twice.neg = twice.n != 0;
            rc = big_add(&nodes, &nodes, &twice);
            if (!rc && nodes.neg) nodes.n = 0, nodes.neg = 0;
        }
        big_free(&twice);
    }
    if (!rc && !(rc = dyad_copy(&err, &b->c_s)) && !(rc = add_scaled(&err, &nodes, e, &b->half_r, &t))) rc = add_scaled(&err, moved, e, &b->g, &t);
    if (!rc) {
        /* Lemma O's check: mass + B < theta */
        dyad v;
        dyad_view(&v, mass, e);
        if (!(rc = dyad_add_exact(&t, &err, &v))) *checked = dyad_cmp(&t, &b->theta) < 0;
    }
    if (!rc) rc = lemma_a(&other, b, mass, moved, e);
    if (!rc && (!*checked || dyad_cmp(&other, &err) < 0)) rc = dyad_copy(&err, &other);
    if (!rc) rc = around(lo, hi, s, e, &err);
    if (!rc && !*checked) rc = sides(flags, b, s, mass, moved, e);
    dyad_free(&err);
    dyad_free(&other);
    dyad_free(&t);
    dyad_free(&zero);
    big_free(&nodes);
    return rc;
}

/* Lemma A for an element: [lo, hi] for every order that does not overflow, and *flags the sides
   on which the final value may overflow. */
int sum_bounds_any_order(dyad *lo, dyad *hi, int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *moved, int64_t e) {
    *flags = 0;
    dyad err;
    dyad_init(&err);
    int rc = trivial(b, mass, moved) ? 0 : lemma_a(&err, b, mass, moved, e);
    if (!rc) rc = around(lo, hi, s, e, &err);
    if (!rc && !trivial(b, mass, moved)) rc = sides(flags, b, s, mass, moved, e);
    dyad_free(&err);
    return rc;
}

/* Lemma S over a fixed tree, a named algorithm's (sum_bounds_init with its depth), which is itself
   the implementation, so it has no slots: [lo, hi] for every finite output. Where the largest
   magnitude among the terms and nodes (largest; NULL for the mass, which bounds it) plus B may
   reach theta, *flags holds +inf where the positive terms' total plus B can reach it, -inf for the
   negative terms', and NaN where both can (NaN for each in a format without infinity). The
   deviation, times 2**e, is the inputs' radii's (Results as inputs), counted in mass, N and moved
   already (matmul.md, Algorithms). */
int sum_bounds_tree(dyad *lo, dyad *hi, int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *N, const bigint *moved,
                    const bigint *deviation, const bigint *largest, int64_t e) {
    *flags = 0;
    dyad zero;
    dyad_init(&zero);
    if (trivial(b, mass, moved)) {
        int rc = around(lo, hi, s, e, &zero);
        dyad_free(&zero);
        return rc;
    }
    dyad err, t, v; /* v only views other integers: it owns no words */
    bigint total;
    dyad_init(&err);
    dyad_init(&t);
    dyad_init(&v);
    big_init(&total);
    int rc = dyad_copy(&err, &b->c_s);
    if (!rc) rc = add_scaled(&err, N, e, &b->half_r, &t);
    if (!rc) rc = add_scaled(&err, moved, e, &b->g, &t);
    if (!rc) rc = around(lo, hi, s, e, &err);
    if (!rc) {
        dyad_view(&v, largest ? largest : mass, e); /* the largest node or term magnitude, at most the mass */
        if (!(rc = dyad_add_exact(&t, &v, &err)) && dyad_cmp(&t, &b->theta) >= 0) {
            /* each side's total: (mass - deviation +- s) / 2 + deviation */
            bigint nd = *deviation; /* a read-only view of -deviation */
            nd.neg = !nd.neg && nd.n;
            for (int side = 0; !rc && side < 2; side++) {
                bigint ss = *s; /* a read-only view of s or -s */
                if (side) ss.neg = !ss.neg && ss.n;
                if (!(rc = big_add(&total, mass, &nd)) && !(rc = big_add(&total, &total, &ss)) && !(rc = big_shr(&total, &total, 1, 0)) &&
                    !(rc = big_add(&total, &total, deviation))) {
                    dyad_view(&v, &total, e);
                    if (!(rc = dyad_add_exact(&t, &v, &err)) && dyad_cmp(&t, &b->theta) >= 0) *flags |= side ? FLAG_MINUS_INF : FLAG_PLUS_INF;
                    v.m.n = 0; /* the view no longer refers to total's words */
                }
            }
            if (*flags == (FLAG_PLUS_INF | FLAG_MINUS_INF)) *flags |= FLAG_NAN;
            if (*flags && !b->has_infinity) *flags = FLAG_NAN;
        }
    }
    dyad_free(&err);
    dyad_free(&t);
    big_free(&total);
    dyad_free(&zero);
    return rc;
}
