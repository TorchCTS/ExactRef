/* attention_gref.c: the backward's reference (proofs/scaled_dot_product_attention_backward.md, The reference output).
   Per output row the exact analysis (kept keys, special values, equal-logit groups, the dS = 0 test) and the first
   enclosure of P and dS from Lemma D's weights; per element of the gradients its special value, rational part, count
   of irrational terms and enclosure, decided at the end or left to the refinement. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

/* ---------------------------------------------------------------- special values
   An element's non-finite value as IEEE arithmetic on exact values gives it: 0 for none, else 8 | VALUE_* | neg << 2 */
#define SPEC(kind, neg) ((unsigned char)(8 | (kind) | ((neg) ? 4 : 0)))
#define SPEC_NAN SPEC(VALUE_NAN, 0)

static int spec_kind(unsigned char s) { return s & 3; }
static int spec_neg(unsigned char s) { return s >> 2 & 1; }

/* a + b */
static unsigned char spec_add(unsigned char a, unsigned char b) {
    if (!a) return b;
    if (!b) return a;
    if (spec_kind(a) == VALUE_NAN || spec_kind(b) == VALUE_NAN || spec_neg(a) != spec_neg(b)) return SPEC_NAN;
    return a;
}

/* x y where x is not finite and y is any value (its kind, sign, and whether it is zero) */
static unsigned char spec_mul(unsigned char x, int kind, int neg, int zero) {
    if (spec_kind(x) == VALUE_NAN || kind == VALUE_NAN || (kind == VALUE_FINITE && zero)) return SPEC_NAN;
    return SPEC(VALUE_INFINITE, spec_neg(x) != neg);
}

/* a factor slot's value: kind, sign, zero */
static void factor_kind(int *kind, int *neg, int *zero, const operand *x, size_t at) {
    uint64_t sig;
    int64_t e;
    slot_value(kind, neg, &sig, &e, x, at);
    *zero = *kind == VALUE_FINITE && !sig;
}

/* ---------------------------------------------------------------- the call */

void att_gref_call_init(att_gref_call *c) {
    memset(c, 0, sizeof *c);
    big_init(&c->a.num);
    big_init(&c->a.den);
    for (int z = 0; z < 2; z++) big_init(&c->out_nan[z]);
    rat_init(&c->sigma);
    dyad_init(&c->slo);
    dyad_init(&c->shi);
    dyad_init(&c->sabs_lo);
    dyad_init(&c->sabs_hi);
}

static void rats_free(rat **x, size_t n) {
    for (size_t z = 0; x && z < n; z++)
        if (x[z]) {
            rat_free(x[z]);
            free(x[z]);
        }
    free(x);
}

void att_gref_call_free(att_gref_call *c) {
    for (int z = 0; z < 3; z++) free(c->x[z]);
    size_t qn = c->qslots * (size_t)c->a.E;
    free(c->qsum);
    free(c->qcount);
    free(c->qspec);
    free(c->qflags);
    free(c->qterms);
    rats_free(c->qrat, qn);
    big_free(&c->a.num);
    big_free(&c->a.den);
    for (int z = 0; z < 2; z++) big_free(&c->out_nan[z]);
    rat_free(&c->sigma);
    dyad_free(&c->slo);
    dyad_free(&c->shi);
    dyad_free(&c->sabs_lo);
    dyad_free(&c->sabs_hi);
    memset(c, 0, sizeof *c);
}

/* The exact factors (the output gradient's, Q's and K's finite values) as integers at their finest set bit; dP's unit;
   dS's ends' unit, ATT_REF_Q bits below 2**td > 2 Ev max|g| max|v| >= |dS|; each gradient's words from its ends'
   bits, its factor's and its terms' counts (nkv rows for grad_key and grad_value, nq for grad_query); the scale.
   c->a, the operands and c->p are set. */
int att_gref_call_prepare(att_gref_call *c, int64_t nq, int64_t nkv) {
    const operand *xs[3] = {c->gs, c->qs, c->ks};
    int64_t xbits[3];
    int rc = 0;
    for (int z = 0; !rc && z < 3; z++) {
        const operand *x = xs[z];
        int64_t top = 0, low = 0;
        int any;
        operand_extent(&top, &low, &any, x);
        c->ux[z] = any ? low : 0;
        xbits[z] = (any ? top - low : 0) + 2;
        c->wx[z] = mw_words(xbits[z]);
        size_t count = x->rows * x->cols;
        if (!(c->x[z] = calloc((count ? count : 1) * (size_t)c->wx[z], sizeof *c->x[z]))) {
            rc = EXACTREF_NOMEM;
            break;
        }
        dyad v;
        dyad_init(&v);
        for (size_t at = 0; !rc && at < count; at++) {
            int kind;
            if (!(rc = operand_dyad(&v, &kind, x, at)) && kind == VALUE_FINITE) rc = mw_from_dyad(c->x[z] + at * (size_t)c->wx[z], c->wx[z], &v, c->ux[z], 0);
        }
        dyad_free(&v);
    }
    int64_t tg = 0, lg = 0, tv = 0, lv = 0;
    int ag, av;
    operand_extent(&tg, &lg, &ag, c->gd);
    operand_extent(&tv, &lv, &av, c->vd);
    c->edp = ag && av ? lg + lv : 0; /* every finite product's lowest set bit is at or above lg + lv */
    int64_t td = (ag && av ? tg + tv : 0) + word_bits((uint64_t)(c->a.Ev ? c->a.Ev : 1)) + 1;
    c->ud = td - ATT_REF_Q;
    /* the sums: three-word ends (below 2**190), the factor's bits, twice the terms' bits, and a sign */
    int64_t counts[3] = {nkv, nkv, nq};
    c->fast = 1;
    for (int z = 0; z < 3; z++) {
        int64_t cbits = counts[z] > 1 ? word_bits((uint64_t)(counts[z] - 1)) : 0;
        c->ws[z] = mw_words(ATT_REF_Q + 1 + xbits[z] + cbits + 2);
        if (c->ws[z] < 3 + c->wx[z] + 2) c->ws[z] = 3 + c->wx[z] + 2;
        c->fast = c->fast && c->wx[z] == 1 && ATT_REF_Q + 1 + xbits[z] + cbits + 2 <= 256;
    }
    if (c->fast) c->ws[0] = c->ws[1] = c->ws[2] = 4;
    /* the scale: a fraction, or 1/sqrt(E) between 1 / sqrt(E)'s upper and lower ends; |sigma| enclosed */
    if (!rc && c->a.den.n) {
        dyad x, y;
        dyad_init(&x);
        dyad_init(&y);
        if (!(rc = rat_set(&c->sigma, &c->a.num, &c->a.den)) && !(rc = big_copy(&x.m, &c->a.num)) && !(rc = big_copy(&y.m, &c->a.den))) {
            x.m.neg = 0;
            if (!(rc = dyad_div(&c->sabs_lo, &x, &y, c->p, 0))) rc = dyad_div(&c->sabs_hi, &x, &y, c->p, 1);
        }
        dyad_free(&x);
        dyad_free(&y);
    } else if (!rc) {
        dyad e, r, one;
        dyad_init(&e);
        dyad_init(&r);
        dyad_init(&one);
        if (!(rc = dyad_set_si(&e, c->a.E, 0)) && !(rc = dyad_set_si(&one, 1, 0)) && !(rc = dyad_sqrt(&r, &e, c->p + 8, 1)) &&
            !(rc = dyad_div(&c->slo, &one, &r, c->p, 0)) && !(rc = dyad_sqrt(&r, &e, c->p + 8, 0)))
            rc = dyad_div(&c->shi, &one, &r, c->p, 1);
        if (!rc && !(rc = dyad_copy(&c->sabs_lo, &c->slo))) rc = dyad_copy(&c->sabs_hi, &c->shi);
        dyad_free(&e);
        dyad_free(&r);
        dyad_free(&one);
    }
    return rc;
}

/* slots for count query rows broadcast over several key and value blocks: each its E grad_query elements' state */
int att_gref_query_slots(att_gref_call *c, size_t count) {
    size_t E = (size_t)c->a.E, n = count * E;
    c->qslots = count;
    c->qsum = calloc((n ? n : 1) * 2 * (size_t)c->ws[2], sizeof *c->qsum);
    c->qcount = calloc(n ? n : 1, sizeof *c->qcount);
    c->qspec = calloc(n ? n : 1, 1);
    c->qflags = calloc(count ? count : 1, 1);
    c->qterms = calloc(count ? count : 1, sizeof *c->qterms);
    c->qrat = calloc(n ? n : 1, sizeof *c->qrat);
    return c->qsum && c->qcount && c->qspec && c->qflags && c->qterms && c->qrat ? 0 : EXACTREF_NOMEM;
}

/* ---------------------------------------------------------------- the group */

void att_gref_group_init(att_gref_group *g) {
    memset(g, 0, sizeof *g);
    big_init(&g->lp);
    big_init(&g->ld);
}

static void dsnum_free(att_gref_group *g, size_t r) {
    if (!g->dsnum || !g->dsnum[r]) return;
    for (size_t j = 0; j < g->S; j++) big_free(&g->dsnum[r][j]);
    free(g->dsnum[r]);
    g->dsnum[r] = NULL;
}

void att_gref_group_free(att_gref_group *g) {
    for (size_t r = 0; g->dsnum && r < g->cap; r++) dsnum_free(g, r);
    for (size_t r = 0; g->mp && r < g->cap; r++) {
        big_free(&g->mp[r]);
        big_free(&g->md[r]);
    }
    void *all[] = {g->rinfo, g->flags, g->n, g->kept, g->dspec, g->plo, g->phi, g->dlo, g->dhi, g->dsnum, g->mp, g->md};
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) free(all[z]);
    big_free(&g->lp);
    big_free(&g->ld);
    memset(g, 0, sizeof *g);
}

/* A key block's grad_key elements (key) or a value block's grad_value elements, at zero terms */
int att_gref_block_start(att_gref_block *b, const att_gref_call *c, int key) {
    size_t S = c->a.S, E = (size_t)c->a.E;
    memset(b, 0, sizeof *b);
    b->cols = key ? E : c->a.Ev;
    b->ws = c->ws[key ? 1 : 0];
    size_t n = S && b->cols ? S * b->cols : 1;
    b->elements = n;
    if (!(b->sum = calloc(n * 2 * (size_t)b->ws, sizeof *b->sum)) || !(b->count = calloc(n, sizeof *b->count)) || !(b->spec = calloc(n, 1)) ||
        !(b->rat = calloc(n, sizeof *b->rat)))
        return EXACTREF_NOMEM;
    return 0;
}

void att_gref_block_free(att_gref_block *b) {
    free(b->sum);
    free(b->count);
    free(b->spec);
    rats_free(b->rat, b->elements);
    memset(b, 0, sizeof *b);
}

/* A group for the unit of key rows kr0.. and value rows vr0.., adding to blocks kb and vb, its tiles up to cap rows */
int att_gref_group_start(att_gref_group *g, const att_gref_call *c, size_t kr0, size_t vr0, att_gref_block *kb, att_gref_block *vb, size_t cap) {
    size_t S = c->a.S, S1 = S ? S : 1;
    g->kr0 = kr0;
    g->vr0 = vr0;
    g->kb = kb;
    g->vb = vb;
    g->cap = cap ? cap : 1;
    g->S = S;
    g->E = (size_t)c->a.E;
    g->Ev = c->a.Ev;
    size_t cells = g->cap * S1;
    if (!(g->rinfo = calloc(g->cap * ATT_GR_INFO, sizeof *g->rinfo)) || !(g->flags = calloc(g->cap, 1)) || !(g->n = calloc(g->cap, sizeof *g->n)) ||
        !(g->kept = calloc(cells, 1)) || !(g->dspec = calloc(cells, 1)) || !(g->plo = calloc(cells * 3, sizeof *g->plo)) ||
        !(g->phi = calloc(cells * 3, sizeof *g->phi)) || !(g->dlo = calloc(cells * 3, sizeof *g->dlo)) || !(g->dhi = calloc(cells * 3, sizeof *g->dhi)) ||
        !(g->dsnum = calloc(g->cap, sizeof *g->dsnum)) || !(g->mp = malloc(g->cap * sizeof *g->mp)) || !(g->md = malloc(g->cap * sizeof *g->md)))
        return EXACTREF_NOMEM;
    for (size_t r = 0; r < g->cap; r++) {
        big_init(&g->mp[r]);
        big_init(&g->md[r]);
    }
    return 0;
}

/* ---------------------------------------------------------------- the rows */

/* x = p (three words) as a dyad */
static int dyad_words3(dyad *x, const uint64_t p[3]) {
    int n = p[2] ? 3 : p[1] ? 2 : p[0] ? 1 : 0, rc = big_reserve(&x->m, 3);
    if (rc) return rc;
    for (int z = 0; z < n; z++) x->m.w[z] = p[z];
    x->m.n = (size_t)n;
    x->m.neg = 0;
    x->e = 0;
    return 0;
}

/* r = the integer of x at 2**u in three words, rounded down (up), x clamped to [-2**(u + 190), 2**(u + 190)] */
static int ends3(uint64_t r[3], const dyad *x, int64_t u, int up) {
    dyad lim;
    dyad_init(&lim);
    int rc = dyad_set_si(&lim, 1, u + ATT_REF_Q);
    const dyad *y = x;
    dyad neg;
    dyad_init(&neg);
    if (!rc && dyad_cmp(x, &lim) > 0) y = &lim;
    if (!rc && !(rc = dyad_neg(&neg, &lim)) && dyad_cmp(x, &neg) < 0) y = &neg;
    if (!rc) rc = mw_from_dyad(r, 3, y, u, up);
    dyad_free(&lim);
    dyad_free(&neg);
    return rc;
}

/* The tile's rows r0 to r1 (their info in g->rinfo): each row's kept keys, flags, non-finite dS, the dS = 0 test, a
   uniform row's dS numerators, and the first enclosure's P and dS ends (The first enclosure). */
int att_gref_tile_rows(att_gref_group *g, const att_gref_call *c, size_t r0, size_t r1) {
    const att_call *a = &c->a;
    const operand *q = a->q, *k = a->k;
    size_t S = g->S, E = g->E;
    int64_t p = c->p, Q = ATT_REF_Q;
    att_row w;
    att_row_init(&w);
    product_sums ps;
    product_sums_init(&ps);
    int rc = att_row_reserve(&w, S, g->Ev);
    /* the weights from Lemma D's tables when the scores fit one-word slots, else from Lemma X */
    int tabled = a->tables && !q->flagged_count && !q->slots_hi && !k->flagged_count && !k->slots_hi;
    unsigned char *dps = malloc(S ? S : 1), *brk = malloc(S ? S : 1);
    size_t *perm = malloc((S ? S : 1) * sizeof *perm);
    dyad *dp = malloc((S ? S : 1) * sizeof *dp), *lk = malloc((S ? S : 1) * sizeof *lk), *sd = malloc((S ? S : 1) * sizeof *sd);
    uint64_t *pw = malloc((S ? S : 1) * 3 * sizeof *pw);
    int64_t *hw = malloc((S ? S : 1) * sizeof *hw);
    dyad wp, wph, t1, t2, t3, xl, xh, lo, hi, num, den;
    dyad *ds[] = {&wp, &wph, &t1, &t2, &t3, &xl, &xh, &lo, &hi, &num, &den};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    bigint gsum, gn, s1, n1, tb;
    bigint *bs[] = {&gsum, &gn, &s1, &n1, &tb};
    for (size_t z = 0; z < sizeof bs / sizeof *bs; z++) big_init(bs[z]);
    dyad *bl = NULL, *bh = NULL;
    if (!rc && (!dps || !brk || !perm || !dp || !lk || !sd || !pw || !hw || !(bl = malloc((S ? S : 1) * sizeof *bl)) || !(bh = malloc((S ? S : 1) * sizeof *bh))))
        rc = EXACTREF_NOMEM;
    for (size_t j = 0; dp && lk && sd && j < (S ? S : 1); j++) {
        dyad_init(&dp[j]);
        dyad_init(&lk[j]);
        dyad_init(&sd[j]);
    }
    for (size_t j = 0; bl && bh && j < (S ? S : 1); j++) {
        dyad_init(&bl[j]);
        dyad_init(&bh[j]);
    }
    for (size_t r = r0; !rc && r < r1; r++) {
        const int64_t *ri = g->rinfo + ATT_GR_INFO * r;
        size_t qr = (size_t)ri[0], i = (size_t)ri[1], moff = (size_t)ri[2], mstep = (size_t)ri[3], gr0 = (size_t)ri[4];
        unsigned char *kept = g->kept + r * S, *dspec = g->dspec + r * S;
        uint64_t *pl = g->plo + r * S * 3, *ph = g->phi + r * S * 3, *dl = g->dlo + r * S * 3, *dh = g->dhi + r * S * 3;
        memset(kept, 0, S);
        memset(dspec, 0, S);
        memset(pl, 0, S * 3 * sizeof *pl);
        memset(ph, 0, S * 3 * sizeof *ph);
        memset(dl, 0, S * 3 * sizeof *dl);
        memset(dh, 0, S * 3 * sizeof *dh);
        dsnum_free(g, r);
        g->flags[r] = 0;
        g->n[r] = 0;
        int kc;
        if ((rc = att_row_logits(&w, a, qr, g->kr0, i, moff, mstep, NULL, &kc))) break;
        if (kc < 0) {
            /* every gradient it touches is NaN: through the keys whose logit is not -inf (in kept), as the others take no part */
            g->flags[r] = ATT_RF_NAN;
            for (size_t j = 0; j < S; j++) kept[j] = w.kinds[j] == ATT_KEY_KEPT || w.kinds[j] == ATT_KEY_SPECIAL;
            continue;
        }
        size_t n = (size_t)kc, *keys = w.keys;
        g->n[r] = n;
        for (size_t m = 0; m < n; m++) kept[keys[m]] = 1;
        /* dP at the kept keys (the others take no part): exact, or its IEEE value where a term is not finite (then D is not
           finite: P_j > 0 there, so -D is NaN or the infinity opposite D, and dS_j = (dP_j - D) P_j) */
        int any = 0;
        for (size_t j = 0; !rc && j < S; j++) {
            dps[j] = 0;
            if (!kept[j]) continue;
            if ((rc = product_element(&ps, c->gd, gr0, c->vd, g->vr0 + j, PRODUCT_SUM, 0, 0, NULL))) break;
            if (ps.kind != VALUE_FINITE) {
                dps[j] = SPEC(ps.kind, ps.neg);
                any = 1;
            } else {
                /* at 2**edp, exactly */
                if (ps.e >= c->edp) {
                    rc = big_shl(&dp[j].m, &ps.s, ps.e - c->edp);
                } else if (ps.s.n && big_trailing(&ps.s) < c->edp - ps.e) {
                    rc = EXACTREF_INTERNAL;
                } else {
                    rc = big_shr(&dp[j].m, &ps.s, c->edp - ps.e, 0);
                }
                dp[j].e = dp[j].m.n ? c->edp : 0;
            }
        }
        if (rc) break;
        if (any) {
            unsigned char d = 0;
            for (size_t j = 0; j < S; j++)
                if (dps[j]) d = spec_add(d, dps[j]);
            unsigned char md = spec_kind(d) == VALUE_NAN ? d : SPEC(VALUE_INFINITE, !spec_neg(d));
            for (size_t j = 0; j < S; j++) dspec[j] = kept[j] ? spec_add(dps[j], md) : 0;
            g->flags[r] |= ATT_RF_DS_SPECIAL;
        }
        if (!n) continue; /* every P is 0, so every dS */
        /* the logits' order (groups of equal logits, as the forward's exact cases: equal scores, with a float mask equal
           (score, mask value) pairs for sigma = 1/sqrt(E), sigma s + b as rationals for a rational sigma, b for sigma 0)
           and the weights' bounds p_j <= 2**F w_j <= p_j + h_j relative to the row's reference logit: from Lemma D's
           tables with one-word scores, else from the exact scores and Lemma X (The first enclosure) */
        int rational = a->mask_float && a->den.n && a->sign;
        dyad nn, dd;
        dyad_init(&nn);
        dyad_init(&dd);
        if (rational && !(rc = big_copy(&nn.m, &a->num))) rc = big_copy(&dd.m, &a->den);
        if (!rc && (rc = dyad_set_si(&wp, 0, 0))) break;
        int64_t H = 0;
        if (tabled) {
            /* the scores at 2**u (exact dot products of the main-range integers) and the extreme one */
            w128 *scores = w.scores, top;
            int64_t shift = a->u - (q->exponent + k->exponent);
            const uint64_t *qs = q->slots + qr * E;
            for (size_t m = 0; m < n; m++) {
                const uint64_t *kss = k->slots + (g->kr0 + keys[m]) * E;
                w2 dot = {0}, t;
                for (size_t e = 0; e < E; e++) {
                    w2_product(&t, qs[e], kss[e]);
                    w2_add(&dot, &t);
                }
                scores[m] = s128_shr(w2_get(&dot), shift);
            }
            top = scores[0];
            for (size_t m = 1; m < n; m++)
                if (a->sign > 0 ? s128_less(top, scores[m]) : s128_less(scores[m], top)) top = scores[m];
            for (size_t m = 0; !rc && rational && m < n; m++) {
                int neg = w128_negative(scores[m]);
                w128 mag = neg ? s128_negate(scores[m]) : scores[m];
                if ((rc = big_reserve(&t1.m, 2))) break;
                t1.m.w[0] = w128_lo(mag);
                t1.m.w[1] = w128_hi(mag);
                t1.m.n = t1.m.w[1] ? 2 : t1.m.w[0] ? 1 : 0;
                t1.m.neg = t1.m.n ? neg : 0;
                t1.e = t1.m.n ? a->u : 0;
                if (!(rc = dyad_mul_exact(&t1, &t1, &nn)) && !(rc = dyad_mul_exact(&t2, &w.bias[m], &dd))) rc = dyad_add_exact(&lk[m], &t1, &t2);
            }
            keyed *order = w.order;
            for (size_t m = 0; m < n; m++) {
                order[m].s = a->sign && !rational ? scores[m] : w128_zero();
                order[m].j = m;
                order[m].key = !a->mask_float ? NULL : rational ? &lk[m] : &w.bias[m];
            }
            qsort(order, n, sizeof *order, keyed_order);
            for (size_t m = 0; m < n; m++) {
                perm[m] = order[m].j;
                brk[m] = m && !keyed_same(&order[m - 1], &order[m]);
            }
            for (size_t m = 0; !rc && m < n; m++) {
                uint64_t *pm = pw + 3 * m;
                pm[0] = pm[1] = 0;
                pm[2] = (uint64_t)1 << (ATT_F - 128);
                int z = 0;
                if (a->mask_float && w.capped[m]) {
                    pm[2] = 0; /* 133 or more below the row's largest mask value: 0 <= 2**F w < 1 */
                    hw[m] = 1;
                } else {
                    if (a->sign && (rc = att_digits_mul(pm, &z, a->sign > 0 ? w128_sub(top, scores[m]) : w128_sub(scores[m], top), a->tables))) break;
                    if (a->mask_float && (rc = att_digits_mul(pm, &z, ((w128 *)w.mdist)[m], a->mtables))) break;
                    hw[m] = z ? 3 * (int64_t)z - 1 : 0;
                }
            }
        } else {
            /* the exact scores, and each weight exp(-(|sigma| D_j + b_top - b_j)) by Lemma X at p bits, D_j = |s_top - s_j| */
            for (size_t m = 0; !rc && m < n; m++)
                if (!(rc = product_element(&ps, q, qr, k, g->kr0 + keys[m], PRODUCT_SUM, 0, 0, NULL)) ) rc = dyad_from_big(&sd[m], &ps.s, ps.e);
            for (size_t m = 0; !rc && rational && m < n; m++)
                if (!(rc = dyad_mul_exact(&t1, &sd[m], &nn)) && !(rc = dyad_mul_exact(&t2, &w.bias[m], &dd))) rc = dyad_add_exact(&lk[m], &t1, &t2);
            if (rc) break;
            gkeyed *order = malloc(n * sizeof *order);
            if (!order) {
                rc = EXACTREF_NOMEM;
                break;
            }
            for (size_t m = 0; m < n; m++) {
                order[m].s = a->sign && !rational ? &sd[m] : NULL;
                order[m].key = !a->mask_float ? NULL : rational ? &lk[m] : &w.bias[m];
                order[m].j = m;
            }
            qsort(order, n, sizeof *order, gkeyed_order);
            for (size_t m = 0; m < n; m++) {
                perm[m] = order[m].j;
                brk[m] = m && gkeyed_cmp(&order[m - 1], &order[m]) != 0;
            }
            free(order);
            size_t top = 0;
            for (size_t m = 1; m < n; m++)
                if (a->sign > 0 ? dyad_cmp(&sd[m], &sd[top]) > 0 : dyad_cmp(&sd[m], &sd[top]) < 0) top = m;
            if (!a->mask_float) {
                rc = dyad_set_si(&t3, 0, 0);
            } else {
                rc = dyad_copy(&t3, &w.btop);
            }
            for (size_t m = 0; !rc && m < n; m++) {
                /* the argument |sigma| D_j + b_top - b_j in [lo, hi] */
                if ((rc = dyad_neg(&t1, &sd[top])) || (rc = dyad_add_exact(&t1, &sd[m], &t1)) || (rc = dyad_abs(&t1, &t1)) ||
                    (rc = dyad_mul(&lo, &c->sabs_lo, &t1, p, 0)) || (rc = dyad_mul(&hi, &c->sabs_hi, &t1, p, 1)))
                    break;
                if (a->mask_float && ((rc = dyad_neg(&t1, &w.bias[m])) || (rc = dyad_add_exact(&t1, &t3, &t1)) || (rc = dyad_add(&lo, &lo, &t1, p, 0)) ||
                                      (rc = dyad_add(&hi, &hi, &t1, p, 1))))
                    break;
                /* 2**F exp(-hi) rounded down is p_j; 2**F exp(-lo) rounded up, minus p_j, is h_j */
                if ((rc = dyad_neg(&lo, &lo)) || (rc = dyad_neg(&hi, &hi)) || (rc = exp_clamped(&t1, &hi, p, 0)) || (rc = exp_clamped(&t2, &lo, p, 1)))
                    break;
                if ((rc = dyad_shift(&t1, ATT_F)) || (rc = dyad_shift(&t2, ATT_F))) break;
                uint64_t lw[4], up[4]; /* at most 2**191: unsigned in three words, two's complement in four */
                if ((rc = mw_from_dyad(lw, 4, &t1, 0, 0)) || (rc = mw_from_dyad(up, 4, &t2, 0, 1))) break;
                memcpy(pw + 3 * m, lw, 3 * sizeof *lw);
                uint64_t borrow = 0;
                for (int z = 0; z < 3; z++) {
                    uint64_t x = up[z] - pw[3 * m + z] - borrow;
                    borrow = up[z] < pw[3 * m + z] || (up[z] == pw[3 * m + z] && borrow);
                    up[z] = x;
                }
                if (up[1] || up[2] || up[0] > (uint64_t)INT64_MAX) {
                    rc = EXACTREF_INTERNAL; /* the ends at p bits are far closer */
                    break;
                }
                hw[m] = (int64_t)up[0];
            }
        }
        dyad_free(&nn);
        dyad_free(&dd);
        if (rc) break;
        size_t groups = 1;
        for (size_t m = 1; m < n; m++) groups += brk[m];
        int special = g->flags[r] & ATT_RF_DS_SPECIAL;
        if (groups == 1) {
            /* equal weights: P = 1/n; with D finite, dS_j = (n dP_j - sum dP) / n**2 */
            g->flags[r] |= ATT_RF_EQUAL;
            if (!special) {
                g->flags[r] |= ATT_RF_UNIFORM;
                if (!(g->dsnum[r] = malloc(S * sizeof **g->dsnum))) {
                    rc = EXACTREF_NOMEM;
                    break;
                }
                for (size_t j = 0; j < S; j++) big_init(&g->dsnum[r][j]);
                if (!(rc = big_set_si(&gsum, 0)) && !(rc = big_set_word(&gn, n, 0)))
                    for (size_t m = 0; !rc && m < n; m++) rc = big_add(&gsum, &gsum, &dp[keys[m]].m);
                for (size_t m = 0; !rc && m < n; m++) {
                    size_t j = keys[m];
                    if (!(rc = big_mul(&tb, &dp[j].m, &gn)) && !(rc = big_copy(&s1, &gsum))) {
                        s1.neg = s1.n ? !s1.neg : 0;
                        rc = big_add(&g->dsnum[r][j], &tb, &s1);
                    }
                }
            }
        } else if (!special) {
            /* dS_j = 0 exactly when every group has one mean dP m and dP_j = m (Lindemann-Weierstrass) */
            int means = 1;
            size_t at = 0, first = 1;
            while (!rc && means && at < n) {
                size_t start = at;
                if ((rc = big_set_si(&gsum, 0))) break;
                do {
                    rc = big_add(&gsum, &gsum, &dp[keys[perm[at]]].m);
                    at++;
                } while (!rc && at < n && !brk[at]);
                if (rc) break;
                if (first) {
                    if (!(rc = big_copy(&s1, &gsum))) rc = big_set_word(&n1, at - start, 0);
                    first = 0;
                } else if (!(rc = big_set_word(&gn, at - start, 0)) && !(rc = big_mul(&tb, &gsum, &n1)) && !(rc = big_mul(&gsum, &s1, &gn))) {
                    means = tb.neg == gsum.neg && !big_cmp_mag(&tb, &gsum); /* s_g n_1 == s_1 n_g */
                }
            }
            if (rc) break;
            for (size_t m = 0; !rc && m < n; m++) {
                size_t j = keys[m];
                int unknown = 1;
                if (means && !(rc = big_mul(&tb, &dp[j].m, &n1))) unknown = !(tb.neg == s1.neg && !big_cmp_mag(&tb, &s1)); /* dP_j n_1 != s_1 */
                if (unknown) kept[j] |= 2;
            }
            if (rc) break;
        }
        for (size_t m = 0; !rc && m < n; m++) {
            H += hw[m];
            if (!(rc = dyad_words3(&t1, pw + 3 * m))) rc = dyad_add_exact(&wp, &wp, &t1);
        }
        if (rc) break;
        if (!(rc = dyad_set_si(&t1, H, 0))) rc = dyad_add_exact(&wph, &wp, &t1); /* Wp + H */
        if (rc) break;
        int none = !wp.m.n; /* every kept key capped: P in [0, 1], dS within its bound */
        /* P_j in [p_j / (Wp + H), (p_j + h_j) / Wp], at 2**-Q */
        for (size_t m = 0; !rc && m < n; m++) {
            size_t j = keys[m];
            if (none) {
                if (!(rc = dyad_set_si(&lo, 0, 0))) rc = dyad_set_si(&hi, 1, 0);
            } else if (!(rc = dyad_words3(&t1, pw + 3 * m)) && !(rc = dyad_div(&lo, &t1, &wph, p, 0)) && !(rc = dyad_set_si(&t2, hw[m], 0)) &&
                       !(rc = dyad_add_exact(&t1, &t1, &t2)) && !(rc = dyad_div(&hi, &t1, &wp, p, 1))) {
                if (!(rc = dyad_set_si(&t3, 1, 0)) && dyad_cmp(&hi, &t3) > 0) rc = dyad_copy(&hi, &t3);
            }
            if (!rc && !(rc = ends3(pl + 3 * j, &lo, -Q, 0))) rc = ends3(ph + 3 * j, &hi, -Q, 1);
        }
        if (rc) break;
        if (special) continue; /* dS is not finite: its terms are decided by the IEEE rules */
        /* dS_j = u x / Omega**2 with u = 2**F w_j, x = 2**F X_j, X_j = dP_j A_j - B_j (The first enclosure): b_k and c_k
           the least and largest of p_k dP_k and (p_k + h_k) dP_k, and the row's sums Bl, Bh */
        dyad sbl, sbh;
        dyad_init(&sbl);
        dyad_init(&sbh);
        if (!(rc = dyad_set_si(&sbl, 0, 0))) rc = dyad_set_si(&sbh, 0, 0);
        for (size_t m = 0; !rc && m < n; m++) {
            size_t j = keys[m];
            /* p_k <= p_k + h_k: the least product is p_k dP_k when dP_k >= 0 */
            int up = dyad_sign(&dp[j]) >= 0;
            if (!(rc = dyad_words3(&t1, pw + 3 * m)) && !(rc = dyad_mul_exact(up ? &bl[m] : &bh[m], &t1, &dp[j])) && !(rc = dyad_set_si(&t2, hw[m], 0)) &&
                !(rc = dyad_add_exact(&t1, &t1, &t2)) && !(rc = dyad_mul_exact(up ? &bh[m] : &bl[m], &t1, &dp[j])) && !(rc = dyad_add_exact(&sbl, &sbl, &bl[m])))
                rc = dyad_add_exact(&sbh, &sbh, &bh[m]);
        }
        dyad lim;
        dyad_init(&lim);
        if (!rc) rc = dyad_set_si(&lim, 1, c->ud + Q);
        for (size_t m = 0; !rc && m < n; m++) {
            size_t j = keys[m];
            if (!(kept[j] & 2) && !(g->flags[r] & ATT_RF_UNIFORM)) {
                if (!(rc = dyad_set_si(&lo, 0, 0))) rc = dyad_set_si(&hi, 0, 0); /* dS_j = 0 exactly */
            } else if (none) {
                if (!(rc = dyad_neg(&lo, &lim))) rc = dyad_copy(&hi, &lim);
            } else {
                /* 2**F A_j in [Wp - p_j, Wp + H - p_j - h_j]; 2**F B_j in [Bl - b_j, Bh - c_j] */
                dyad alo, ahi, blo, bhi;
                dyad *as[] = {&alo, &ahi, &blo, &bhi};
                for (size_t z = 0; z < 4; z++) dyad_init(as[z]);
                if (!(rc = dyad_words3(&t1, pw + 3 * m)) && !(rc = dyad_neg(&t1, &t1)) && !(rc = dyad_add_exact(&alo, &wp, &t1)) &&
                    !(rc = dyad_set_si(&t2, -hw[m], 0)) && !(rc = dyad_add_exact(&ahi, &wph, &t1)) && !(rc = dyad_add_exact(&ahi, &ahi, &t2)) &&
                    !(rc = dyad_neg(&t3, &bl[m])) && !(rc = dyad_add_exact(&blo, &sbl, &t3)) && !(rc = dyad_neg(&t3, &bh[m])) &&
                    !(rc = dyad_add_exact(&bhi, &sbh, &t3)) && !(rc = dyad_mul_exact(&t1, &dp[j], &alo)) && !(rc = dyad_mul_exact(&t2, &dp[j], &ahi))) {
                    /* x_l = min(dP_j A) - (Bh - c_j), x_h = max(dP_j A) - (Bl - b_j) */
                    const dyad *mn = dyad_cmp(&t1, &t2) <= 0 ? &t1 : &t2, *mx = mn == &t1 ? &t2 : &t1;
                    if (!(rc = dyad_neg(&t3, &bhi)) && !(rc = dyad_add_exact(&xl, mn, &t3)) && !(rc = dyad_neg(&t3, &blo))) rc = dyad_add_exact(&xh, mx, &t3);
                }
                for (size_t z = 0; z < 4; z++) dyad_free(as[z]);
                /* dS_j between p_j x_l / (Wp + H)**2 (x_l >= 0, else (p_j + h_j) x_l / Wp**2) and (p_j + h_j) x_h / Wp**2
                   (x_h >= 0, else p_j x_h / (Wp + H)**2) */
                if (!rc && !(rc = dyad_words3(&t1, pw + 3 * m)) && !(rc = dyad_set_si(&t2, hw[m], 0)) && !(rc = dyad_add_exact(&t2, &t1, &t2))) {
                    const dyad *ul = dyad_sign(&xl) >= 0 ? &t1 : &t2, *uh = dyad_sign(&xh) >= 0 ? &t2 : &t1;
                    const dyad *zl = dyad_sign(&xl) >= 0 ? &wph : &wp, *zh = dyad_sign(&xh) >= 0 ? &wp : &wph;
                    if (!(rc = dyad_mul_exact(&num, ul, &xl)) && !(rc = dyad_mul_exact(&den, zl, zl)) && !(rc = dyad_div(&lo, &num, &den, p, 0)) &&
                        !(rc = dyad_mul_exact(&num, uh, &xh)) && !(rc = dyad_mul_exact(&den, zh, zh)))
                        rc = dyad_div(&hi, &num, &den, p, 1);
                }
            }
            if (!rc && !(rc = ends3(dl + 3 * j, &lo, c->ud, 0))) rc = ends3(dh + 3 * j, &hi, c->ud, 1);
        }
        dyad_free(&lim);
        dyad_free(&sbl);
        dyad_free(&sbh);
    }
    for (size_t j = 0; dp && lk && sd && j < (S ? S : 1); j++) {
        dyad_free(&dp[j]);
        dyad_free(&lk[j]);
        dyad_free(&sd[j]);
    }
    free(sd);
    free(perm);
    free(brk);
    for (size_t j = 0; bl && bh && j < (S ? S : 1); j++) {
        dyad_free(&bl[j]);
        dyad_free(&bh[j]);
    }
    free(bl);
    free(bh);
    free(dps);
    free(dp);
    free(lk);
    free(pw);
    free(hw);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    for (size_t z = 0; z < sizeof bs / sizeof *bs; z++) big_free(bs[z]);
    product_sums_free(&ps);
    att_row_free(&w);
    return rc;
}

/* ---------------------------------------------------------------- the sums */

/* The tile's common denominators: lp of the equal rows' n, ld of the uniform rows' n**2, and each row's multiplier (lp / n,
   ld / n**2), so their rational terms add over one denominator per element (The reference output). */
int att_gref_tile_prep(att_gref_group *g, const att_gref_call *c) {
    (void)c;
    bigint t, gc, q;
    big_init(&t);
    big_init(&gc);
    big_init(&q);
    int rc = big_set_word(&g->lp, 1, 0), exact;
    if (!rc) rc = big_set_word(&g->ld, 1, 0);
    for (int pass = 0; !rc && pass < 2; pass++) {
        bigint *L = pass ? &g->ld : &g->lp;
        int need = pass ? ATT_RF_UNIFORM : ATT_RF_EQUAL;
        for (size_t r = 0; !rc && r < g->T; r++) {
            if (!(g->flags[r] & need) || (g->flags[r] & ATT_RF_NAN)) continue;
            uint64_t n = g->n[r];
            if (!(rc = big_set_word(&t, n, 0)) && pass) rc = big_mul(&t, &t, &t);
            /* L = lcm(L, t) = L t / gcd(L, t) */
            if (!rc && !(rc = big_gcd(&gc, L, &t)) && !(rc = big_divmod(&q, &exact, &t, &gc))) rc = big_mul(L, L, &q);
        }
        for (size_t r = 0; !rc && r < g->T; r++) {
            bigint *m = pass ? &g->md[r] : &g->mp[r];
            if (!(g->flags[r] & need) || (g->flags[r] & ATT_RF_NAN)) continue;
            if (!(rc = big_set_word(&t, g->n[r], 0)) && pass) rc = big_mul(&t, &t, &t);
            if (!rc) rc = big_divmod(m, &exact, L, &t);
        }
    }
    big_free(&t);
    big_free(&gc);
    big_free(&q);
    return rc;
}

static inline void add_ends4(uint64_t *st, const uint64_t *lo, const uint64_t *hi, int64_t x) {
    if (!x) return;
    w4 a1, a2, t;
    w4_load3(&a1, x < 0 ? hi : lo);
    w4_load3(&a2, x < 0 ? lo : hi);
    w4_smul(&t, &a1, x);
    w4_add((w4 *)st, &t);
    w4_smul(&t, &a2, x);
    w4_add((w4 *)(st + 4), &t);
}

/* x = num 2**e / den added to *r (allocated on first use) */
static int rat_add_scaled(rat **r, const bigint *num, int64_t e, const bigint *den) {
    if (!num->n) return 0;
    if (!*r) {
        if (!(*r = malloc(sizeof **r))) return EXACTREF_NOMEM;
        rat_init(*r);
    }
    rat x;
    rat_init(&x);
    bigint n, d;
    big_init(&n);
    big_init(&d);
    int rc = big_shl(&n, num, e > 0 ? e : 0);
    if (!rc) rc = big_shl(&d, den, e < 0 ? -e : 0);
    if (!rc) rc = rat_set(&x, &n, &d);
    if (!rc) rc = rat_add(*r, *r, &x);
    rat_free(&x);
    big_free(&n);
    big_free(&d);
    return rc;
}

/* the factor at element at of z's operand as an integer (its words) */
static int factor_big(bigint *r, const att_gref_call *c, int z, size_t at, uint64_t *tmp) { return mw_to_big(r, c->x[z] + at * (size_t)c->wx[z], c->wx[z], tmp); }

/* The tile's terms into the elements of keys j0 to j1: grad_value's (P times the output gradient's row) and grad_key's
   (dS times the query row), over the tile's rows: special values, ends, irrational terms and rational parts. */
int att_gref_tile_keys(att_gref_group *g, const att_gref_call *c, size_t j0, size_t j1) {
    size_t S = g->S, E = g->E, Ev = g->Ev, T = g->T;
    int wsv = c->ws[0], wsk = c->ws[1], big = wsv > wsk ? wsv : wsk, rc = 0;
    uint64_t *tmp = malloc((size_t)(4 * big + 16) * sizeof *tmp);
    bigint num, t, x;
    big_init(&num);
    big_init(&t);
    big_init(&x);
    if (!tmp) rc = EXACTREF_NOMEM;
    for (size_t j = j0; !rc && j < j1; j++) {
        for (size_t col = 0; !rc && col < Ev; col++) {
            size_t el = j * Ev + col;
            uint64_t *st = g->vb->sum + el * 2 * (size_t)wsv;
            int has = 0;
            if ((rc = big_set_si(&num, 0))) break;
            for (size_t r = 0; !rc && r < T; r++) {
                unsigned char f = g->flags[r], kept = g->kept[r * S + j];
                if (!(kept & 1)) continue; /* a key the row does not keep takes no part */
                if (f & ATT_RF_NAN) {
                    g->vb->spec[el] = SPEC_NAN; /* a NaN row's P at a key whose logit is not -inf */
                    continue;
                }
                size_t gr0 = (size_t)g->rinfo[ATT_GR_INFO * r + 4], xa = gr0 * Ev + col;
                if (c->gs->special_rows[gr0]) {
                    int kind, neg, zero;
                    factor_kind(&kind, &neg, &zero, c->gs, xa);
                    if (kind != VALUE_FINITE) {
                        g->vb->spec[el] = spec_add(g->vb->spec[el], SPEC(kind, neg)); /* P x with P > 0 */
                        continue;
                    }
                }
                const uint64_t *pl = g->plo + (r * S + j) * 3, *ph = g->phi + (r * S + j) * 3, *xv = c->x[0] + xa * (size_t)c->wx[0];
                if (c->fast)
                    add_ends4(st, pl, ph, word_signed(*xv));
                else
                    mw_add_ends(st, wsv, pl, ph, 3, xv, c->wx[0], tmp);
                if (!(f & ATT_RF_EQUAL)) {
                    g->vb->count[el] += !mw_zero(xv, c->wx[0]); /* P irrational */
                } else if (!(rc = factor_big(&x, c, 0, xa, tmp)) && !(rc = big_mul(&t, &g->mp[r], &x))) {
                    rc = big_add(&num, &num, &t); /* P = 1/n = mp / lp */
                    has = 1;
                }
            }
            if (!rc && has) rc = rat_add_scaled(&g->vb->rat[el], &num, c->ux[0], &g->lp);
        }
        for (size_t e = 0; !rc && e < E; e++) {
            size_t el = j * E + e;
            uint64_t *st = g->kb->sum + el * 2 * (size_t)wsk;
            int has = 0;
            if ((rc = big_set_si(&num, 0))) break;
            for (size_t r = 0; !rc && r < T; r++) {
                unsigned char f = g->flags[r], kept = g->kept[r * S + j];
                if (!(kept & 1)) continue; /* a key the row does not keep takes no part */
                if (f & ATT_RF_NAN) {
                    g->kb->spec[el] = SPEC_NAN; /* a NaN row's dS at a key whose logit is not -inf */
                    continue;
                }
                size_t qr = (size_t)g->rinfo[ATT_GR_INFO * r], xa = qr * E + e;
                if (f & ATT_RF_DS_SPECIAL) {
                    int kind, neg, zero; /* the query row is finite: its row keeps a key */
                    factor_kind(&kind, &neg, &zero, c->qs, xa);
                    g->kb->spec[el] = spec_add(g->kb->spec[el], spec_mul(g->dspec[r * S + j], kind, neg, zero));
                    continue;
                }
                const uint64_t *dl = g->dlo + (r * S + j) * 3, *dh = g->dhi + (r * S + j) * 3, *xv = c->x[1] + xa * (size_t)c->wx[1];
                if (c->fast)
                    add_ends4(st, dl, dh, word_signed(*xv));
                else
                    mw_add_ends(st, wsk, dl, dh, 3, xv, c->wx[1], tmp);
                if (kept & 2) {
                    g->kb->count[el] += !mw_zero(xv, c->wx[1]); /* dS irrational */
                } else if ((f & ATT_RF_UNIFORM) && g->dsnum[r][j].n && !(rc = factor_big(&x, c, 1, xa, tmp)) && !(rc = big_mul(&t, &g->md[r], &x)) &&
                           !(rc = big_mul(&t, &t, &g->dsnum[r][j]))) {
                    rc = big_add(&num, &num, &t); /* dS = dsnum / n**2 = dsnum md / ld */
                    has = 1;
                }
            }
            if (!rc && has) rc = rat_add_scaled(&g->kb->rat[el], &num, c->edp + c->ux[1], &g->ld);
        }
    }
    free(tmp);
    big_free(&num);
    big_free(&t);
    big_free(&x);
    return rc;
}

/* ---------------------------------------------------------------- the decision */

/* the encoding and entry of an IEEE value (kind, sign), and of a finite zero */
static int put_kind(unsigned char *enc, entry *x, const att_gref_call *c, int kind, int neg) {
    bigint bits, zero;
    big_init(&bits);
    big_init(&zero);
    int rc = floatfmt_encode(&bits, &c->out, kind, neg, &zero, 0, c->rnd, c->ovf);
    if (!rc) rc = entry_value(x, kind, neg, &zero, 0);
    if (!rc) bytes_store(enc, c->out_size, &bits);
    big_free(&bits);
    big_free(&zero);
    return rc;
}

/* The element's encoding, entry and status (0 decided, 1 left to the refinement), as the reference's rules decide it:
   a NaN row or a non-finite value (times the scale's sign when scaled); with scale 0 an exact zero; no irrational term
   (and not a nonzero rational times 1/sqrt(E)): its rational part times the scale; else its first enclosure, from the
   ends st (n words each at 2**unit) times the scale. An exact zero is -0 rounding toward -inf when its block has terms. */
static int decide(unsigned char *enc, entry *x, unsigned char *status, const att_gref_call *c, int scaled, int nan, unsigned char spec, int64_t terms,
                  int64_t count, const rat *q, const uint64_t *st, int n, int64_t unit, uint64_t *tmp) {
    const att_call *a = &c->a;
    *status = 0;
    int zneg = c->neg_zero && terms > 0;
    if (nan) return put_kind(enc, x, c, VALUE_NAN, 0);
    if (spec) {
        int kind = spec_kind(spec), neg = spec_neg(spec);
        if (scaled && kind == VALUE_INFINITE && !a->sign) kind = VALUE_NAN;
        if (scaled && kind == VALUE_INFINITE && a->sign < 0) neg = !neg;
        return put_kind(enc, x, c, kind, kind == VALUE_NAN ? 0 : neg);
    }
    if (scaled && !a->sign) return put_kind(enc, x, c, VALUE_FINITE, zneg);
    int irrational = scaled && !a->den.n && q && q->num.n;
    int rc = 0;
    if (count || irrational) {
        dyad lo, hi, t1, t2;
        dyad *ds[] = {&lo, &hi, &t1, &t2};
        for (size_t z = 0; z < 4; z++) dyad_init(ds[z]);
        if (!(rc = mw_to_dyad(&lo, st, n, unit, tmp)) && !(rc = mw_to_dyad(&hi, st + n, n, unit, tmp))) {
            if (scaled && a->den.n) {
                /* times num / den, den > 0 */
                dyad nn, dd;
                dyad_init(&nn);
                dyad_init(&dd);
                if (!(rc = big_copy(&nn.m, &c->sigma.num)) && !(rc = big_copy(&dd.m, &c->sigma.den)) && !(rc = dyad_mul_exact(&t1, &lo, &nn)) &&
                    !(rc = dyad_mul_exact(&t2, &hi, &nn))) {
                    const dyad *mn = dyad_cmp(&t1, &t2) <= 0 ? &t1 : &t2, *mx = mn == &t1 ? &t2 : &t1;
                    if (!(rc = dyad_div(&lo, mn, &dd, c->p, 0))) rc = dyad_div(&hi, mx, &dd, c->p, 1);
                }
                dyad_free(&nn);
                dyad_free(&dd);
            } else if (scaled) {
                /* times 1/sqrt(E) in [slo, shi], positive */
                if (!(rc = dyad_mul(&t1, &lo, dyad_sign(&lo) >= 0 ? &c->slo : &c->shi, c->p, 0)) &&
                    !(rc = dyad_mul(&t2, &hi, dyad_sign(&hi) >= 0 ? &c->shi : &c->slo, c->p, 1)) && !(rc = dyad_copy(&lo, &t1)))
                    rc = dyad_copy(&hi, &t2);
            }
        }
        if (!rc) {
            bigint bits;
            big_init(&bits);
            int none = 1;
            rc = decide_rounding(&bits, x, &none, &c->out, c->rnd, c->ovf, &lo, &hi, 0);
            if (!rc && !none) bytes_store(enc, c->out_size, &bits);
            *status = (unsigned char)none;
            big_free(&bits);
        }
        for (size_t z = 0; z < 4; z++) dyad_free(ds[z]);
        return rc;
    }
    if (!q || !q->num.n) return put_kind(enc, x, c, VALUE_FINITE, zneg);
    rat v;
    rat_init(&v);
    rc = scaled ? rat_mul(&v, q, &c->sigma) : rat_copy(&v, q);
    if (!rc && !v.num.n) {
        rat_free(&v);
        return put_kind(enc, x, c, VALUE_FINITE, zneg);
    }
    bigint bits;
    big_init(&bits);
    int neg = v.num.neg;
    v.num.neg = 0;
    if (!rc) rc = floatfmt_encode_ratio(&bits, &c->out, neg, &v.num, &v.den, c->rnd, c->ovf);
    if (!rc) rc = entry_ratio(x, neg, &v.num, &v.den);
    if (!rc) bytes_store(enc, c->out_size, &bits);
    big_free(&bits);
    rat_free(&v);
    return rc;
}

/* grad_query of the tile's rows r0 to r1: dS times the key block's rows over the keys, in the call's slot from copy to
   copy for a query row broadcast over several key and value blocks, decided at its last copy into the outputs (E per
   row from enc, entries and status at row r - r0). */
int att_gref_tile_queries(const att_gref_group *g, att_gref_call *c, size_t r0, size_t r1, unsigned char *enc, entry *entries, unsigned char *status) {
    size_t S = g->S, E = g->E;
    int n = c->ws[2], wx = c->wx[2], rc = 0;
    uint64_t *own = calloc((E ? E : 1) * 2 * (size_t)n, sizeof *own), *tmp = malloc((size_t)(4 * n + 16) * sizeof *tmp);
    int64_t *ocount = calloc(E ? E : 1, sizeof *ocount);
    unsigned char *ospec = calloc(E ? E : 1, 1);
    rat **orat = calloc(E ? E : 1, sizeof *orat);
    bigint num, x;
    big_init(&num);
    big_init(&x);
    if (!own || !tmp || !ocount || !ospec || !orat) rc = EXACTREF_NOMEM;
    for (size_t r = r0; !rc && r < r1; r++) {
        const int64_t *ri = g->rinfo + ATT_GR_INFO * r;
        int64_t slot = ri[5], copy = ri[6], copies = ri[7];
        unsigned char f = g->flags[r], oflags = 0, *flags = &oflags;
        uint64_t *sums = own;
        int64_t *count = ocount, oterms = 0, *terms = &oterms;
        unsigned char *spec = ospec;
        rat **rats = orat;
        if (slot >= 0) {
            sums = c->qsum + (size_t)slot * E * 2 * (size_t)n;
            count = c->qcount + (size_t)slot * E;
            spec = c->qspec + (size_t)slot * E;
            rats = c->qrat + (size_t)slot * E;
            flags = c->qflags + slot;
            terms = c->qterms + slot;
        }
        if (slot < 0 || copy == 0) {
            memset(sums, 0, E * 2 * (size_t)n * sizeof *sums);
            memset(count, 0, E * sizeof *count);
            memset(spec, 0, E);
            for (size_t e = 0; e < E; e++)
                if (rats[e]) {
                    rat_free(rats[e]);
                    free(rats[e]);
                    rats[e] = NULL;
                }
            *flags = 0;
            *terms = 0;
        }
        *terms += (int64_t)S;
        const unsigned char *kept = g->kept + r * S, *dspec = g->dspec + r * S;
        if (f & ATT_RF_NAN) {
            *flags = 1;
        } else if (f & ATT_RF_DS_SPECIAL) {
            for (size_t e = 0; S && e < E; e++) {
                unsigned char sum = 0;
                for (size_t j = 0; j < S; j++) {
                    if (!(kept[j] & 1)) continue; /* a key the row does not keep takes no part; a kept key's row is finite */
                    int kind, neg, zero;
                    factor_kind(&kind, &neg, &zero, c->ks, (g->kr0 + j) * E + e);
                    sum = spec_add(sum, spec_mul(dspec[j], kind, neg, zero));
                }
                spec[e] = spec_add(spec[e], sum);
            }
        } else {
            for (size_t e = 0; !rc && e < E; e++) {
                uint64_t *st = sums + e * 2 * (size_t)n;
                int has = 0;
                if ((rc = big_set_si(&num, 0))) break;
                for (size_t j = 0; !rc && j < S; j++) {
                    if (!(kept[j] & 1)) continue;
                    size_t xa = (g->kr0 + j) * E + e;
                    const uint64_t *dl = g->dlo + (r * S + j) * 3, *dh = g->dhi + (r * S + j) * 3, *xv = c->x[2] + xa * (size_t)wx;
                    if (c->fast)
                        add_ends4(st, dl, dh, word_signed(*xv));
                    else
                        mw_add_ends(st, n, dl, dh, 3, xv, wx, tmp);
                    if (kept[j] & 2) {
                        count[e] += !mw_zero(xv, wx);
                    } else if ((f & ATT_RF_UNIFORM) && g->dsnum[r][j].n && !(rc = factor_big(&x, c, 2, xa, tmp)) && !(rc = big_mul(&x, &x, &g->dsnum[r][j]))) {
                        rc = big_add(&num, &num, &x);
                        has = 1;
                    }
                }
                if (!rc && has) {
                    bigint n2;
                    big_init(&n2);
                    if (!(rc = big_set_word(&n2, g->n[r], 0)) && !(rc = big_mul(&n2, &n2, &n2))) rc = rat_add_scaled(&rats[e], &num, c->edp + c->ux[2], &n2);
                    big_free(&n2);
                }
            }
        }
        if (rc || (slot >= 0 && copy < copies - 1)) continue; /* decided at its last copy */
        for (size_t e = 0; !rc && e < E; e++) {
            size_t o = (r - r0) * E + e;
            rc = decide(enc + o * c->out_size, &entries[o], &status[o], c, 1, *flags, spec[e], *terms, count[e], rats[e], sums + e * 2 * (size_t)n, n,
                        c->ud + c->ux[2], tmp);
        }
    }
    for (size_t e = 0; orat && e < E; e++)
        if (orat[e]) {
            rat_free(orat[e]);
            free(orat[e]);
        }
    free(orat);
    free(own);
    free(tmp);
    free(ocount);
    free(ospec);
    big_free(&num);
    big_free(&x);
    return rc;
}

/* After a tile's phases: its rows join the block's terms */
void att_gref_tile_done(att_gref_group *g) {
    g->kb->terms += (int64_t)g->T;
    g->vb->terms += (int64_t)g->T;
}

/* A block's elements of keys j0 to j1, after the last unit that adds to it, decided into the outputs (cols per key from j0):
   grad_key's (key) scaled, grad_value's not */
int att_gref_block_finish(const att_gref_block *b, const att_gref_call *c, int key, size_t j0, size_t j1, unsigned char *enc, entry *entries,
                          unsigned char *status) {
    int n = b->ws, rc = 0;
    uint64_t *tmp = malloc((size_t)(4 * n + 16) * sizeof *tmp);
    if (!tmp) return EXACTREF_NOMEM;
    int64_t unit = key ? c->ud + c->ux[1] : -ATT_REF_Q + c->ux[0];
    for (size_t el = j0 * b->cols; !rc && el < j1 * b->cols; el++) {
        size_t o = el - j0 * b->cols;
        rc = decide(enc + o * c->out_size, &entries[o], &status[o], c, key, 0, b->spec[el], b->terms, b->count[el], b->rat[el], b->sum + el * 2 * (size_t)n,
                    n, unit, tmp);
    }
    free(tmp);
    return rc;
}
