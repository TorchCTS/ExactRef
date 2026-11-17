/* attention_grad_refine.c: the backward's refinement (proofs/scaled_dot_product_attention_backward.md, Refinement, The
   cancellation-free dS, Far keys, grad_query of one row). The elements the reference kernel leaves undecided are
   enclosed again at doubling precision. Each round streams the rows of the pending elements in tiles on the threads:
   a tile makes a row's exact quantities and its P and dS in every form as intervals of dyadics, takes its terms of each
   of its pending elements' sums (with exact factors), and frees it; the terms are added to the elements' sums in row
   order. After the first round the grad_query pair test; in each round the cancellation-free dS and the far keys'
   split, with the kept keys' gradient exact on rational.c. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

/* ---------------------------------------------------------------- rows */

/* row r's quantities freed, its descriptor kept */
static void grow_free(att_grow *r) {
    if (!r->ready) return;
    size_t n = r->n;
    for (size_t m = 0; m < n; m++) {
        dyad_free(&r->sd[m]);
        dyad_free(&r->dp[m]);
        dyad_free(&r->bias[m]);
        dyad_free(&r->lk[m]);
        ivl_free(&r->x[m]);
        ivl_free(&r->P[m]);
        ivl_free(&r->dS[m]);
        ivl_free(&r->dS2[m]);
        ivl_free(&r->dPr[m]);
        ivl_free(&r->dSr[m]);
        ivl_free(&r->om[m]);
    }
    ivl_free(&r->xF);
    free(r->keys);
    free(r->pos);
    free(r->sd);
    free(r->x);
    free(r->order);
    free(r->far);
    size_t d[7] = {r->qr, r->i, r->moff, r->mstep, r->gr0, r->kr0, r->vr0}; /* its descriptor stays, for the next round */
    memset(r, 0, sizeof *r);
    r->qr = d[0], r->i = d[1], r->moff = d[2], r->mstep = d[3], r->gr0 = d[4], r->kr0 = d[5], r->vr0 = d[6];
}

/* row r's exact quantities (w the tile's scratch for its logits): its kept keys (none for a NaN row), per kept key the
   score, dP and mask value, the logits' groups (the forward's exact cases' order) and the extreme score's key */
static int grow_ready(att_grow *r, const att_gref_refine *f, att_row *w) {
    if (r->ready) return 0;
    const att_gref_call *c = f->c;
    const att_call *a = &c->a;
    size_t S = a->S;
    int rc = att_row_reserve(w, S, a->Ev);
    if (rc) return rc;
    int kept;
    if ((rc = att_row_logits(w, a, r->qr, r->kr0, r->i, r->moff, r->mstep, NULL, &kept))) return rc;
    r->ready = 1;
    ivl_init(&r->xF);
    if (kept < 0) {
        r->nan = 1;
        return 0;
    }
    size_t n = (size_t)kept;
    r->keys = malloc((n + 1) * sizeof *r->keys);
    r->pos = calloc(S + 1, sizeof *r->pos);
    r->sd = malloc((4 * n + 1) * sizeof *r->sd);
    r->order = malloc((n + 1) * sizeof *r->order);
    r->far = calloc(n + 1, 1);
    r->x = malloc((7 * n + 1) * sizeof *r->x);
    if (!r->keys || !r->pos || !r->sd || !r->order || !r->far || !r->x) return EXACTREF_NOMEM;
    r->n = n;
    r->dp = r->sd + n;
    r->bias = r->sd + 2 * n;
    r->lk = r->sd + 3 * n;
    r->P = r->x + n;
    r->dS = r->x + 2 * n;
    r->dS2 = r->x + 3 * n;
    r->dPr = r->x + 4 * n;
    r->dSr = r->x + 5 * n;
    r->om = r->x + 6 * n;
    for (size_t m = 0; m < 4 * n; m++) dyad_init(&r->sd[m]);
    for (size_t m = 0; m < 7 * n; m++) ivl_init(&r->x[m]);
    product_sums ps;
    product_sums_init(&ps);
    int rational = a->mask_float && a->den.n && a->sign;
    dyad nn, dd, t;
    dyad_init(&nn);
    dyad_init(&dd);
    dyad_init(&t);
    if (rational && !(rc = big_copy(&nn.m, &a->num))) rc = big_copy(&dd.m, &a->den);
    for (size_t m = 0; !rc && m < n; m++) {
        size_t j = w->keys[m];
        r->keys[m] = j;
        r->pos[j] = m + 1;
        if (a->mask_float && (rc = dyad_copy(&r->bias[m], &w->bias[m]))) break;
        if ((rc = product_element(&ps, a->q, r->qr, a->k, r->kr0 + j, PRODUCT_SUM, 0, 0, NULL)) || (rc = dyad_from_big(&r->sd[m], &ps.s, ps.e))) break;
        if ((rc = product_element(&ps, c->gd, r->gr0, c->vd, r->vr0 + j, PRODUCT_SUM, 0, 0, NULL))) break;
        if (ps.kind == VALUE_FINITE) { /* a non-finite dP enters as 0: the row's dS terms are then decided by the IEEE rules, never refined */
            if ((rc = dyad_from_big(&r->dp[m], &ps.s, ps.e))) break;
        }
        if (rational && !(rc = dyad_mul_exact(&r->lk[m], &r->sd[m], &nn)) && !(rc = dyad_mul_exact(&t, &r->bias[m], &dd)))
            rc = dyad_add_exact(&r->lk[m], &r->lk[m], &t);
    }
    for (size_t m = 0; m < n; m++) {
        r->order[m].s = a->sign && !rational ? &r->sd[m] : NULL;
        r->order[m].key = !a->mask_float ? NULL : rational ? &r->lk[m] : &r->bias[m];
        r->order[m].j = m;
    }
    r->top = 0;
    for (size_t m = 1; m < n; m++)
        if (a->sign > 0 ? dyad_cmp(&r->sd[m], &r->sd[r->top]) > 0 : dyad_cmp(&r->sd[m], &r->sd[r->top]) < 0) r->top = m;
    product_sums_free(&ps);
    dyad_free(&nn);
    dyad_free(&dd);
    dyad_free(&t);
    return rc;
}

/* the kept keys sorted into groups of equal logits, once: row_stats and the pair test read them */
static void grow_order(att_grow *r) {
    if (r->sorted) return;
    if (!r->nan && r->n) qsort(r->order, r->n, sizeof *r->order, gkeyed_order);
    r->sorted = 1;
}

/* x_m - x_l = l_l - l_m by att_argument (exact differences of scores and mask values) */
static int arg_diff(ivl *y, const att_grow *r, const att_gref_refine *f, size_t m, size_t l, int64_t p) {
    return att_argument(&y->lo, &y->hi, &f->c->a, r->sd, f->c->a.mask_float ? r->bias : NULL, l, m, &f->sabs_lo, &f->sabs_hi, p);
}

/* The far terms' relative factors keep exponents above -2**FAR_BITS * 1.45 (Far keys). */
#define FAR_BITS 56

/* e**y rounded to the side by Lemma X, for y below 2**40 (the caller's check); y at or below -2**FAR_BITS gives 0 down and
   2**-(2**FAR_BITS) > e**y up */
static int exp_far(dyad *r, const dyad *y, int64_t p, int up) {
    if (dyad_sign(y) < 0 && top_bit(y) >= FAR_BITS) return up ? dyad_set_si(r, 1, -((int64_t)1 << FAR_BITS)) : dyad_set_si(r, 0, 0);
    return exp_end(r, y, p, up);
}

static int ivl_exp_far(ivl *r, const ivl *y, int64_t p) {
    int rc = exp_far(&r->lo, &y->lo, p, 0);
    return rc ? rc : exp_far(&r->hi, &y->hi, p, 1);
}

/* key m's score distance from the reference key's, in units of 2**u for Lemma D's tables: *ok 0 where it is negative (a
   key whose score passes the reference's, its mask lower) or no 128-bit integer there */
static int score_distance(w128 *D, int *ok, const att_grow *r, const att_call *a, size_t m, dyad *t) {
    *ok = 0;
    const dyad *hi = a->sign > 0 ? &r->sd[r->ref] : &r->sd[m], *lo = a->sign > 0 ? &r->sd[m] : &r->sd[r->ref];
    int rc = dyad_neg(t, lo);
    if (rc || (rc = dyad_add_exact(t, hi, t)) || dyad_sign(t) < 0) return rc;
    if (t->m.n) {
        int64_t shift = t->e - a->u;
        if (shift < 0 ? (int64_t)big_trailing(&t->m) < -shift : big_bits(&t->m) + shift > 128) return 0;
        if ((rc = shift < 0 ? big_shr(&t->m, &t->m, -shift, 0) : big_shl(&t->m, &t->m, shift))) return rc;
    }
    *D = w128_make(t->m.n > 0 ? t->m.w[0] : 0, t->m.n > 1 ? t->m.w[1] : 0);
    *ok = 1;
    return 0;
}

/* e**-(|sigma| D 2**u) by Lemma D's tables: [p, p + h] 2**-F; *done 0 where it is below 2**-48 (Lemma X keeps the relative
   precision there) */
static int table_weight(ivl *w, int *done, w128 D, const att_tables *T) {
    uint64_t pw[3] = {0, 0, (uint64_t)1 << (ATT_F - 128)};
    int z = 0, rc = att_digits_mul(pw, &z, D, T);
    *done = 0;
    if (rc || pw[2] < ((uint64_t)1 << 15) || (rc = dyad_words(&w->lo, pw, 3, -ATT_F))) return rc;
    uint64_t h = z ? 3 * (uint64_t)z - 1 : 0;
    for (int i = 0; i < 3 && h; i++) {
        pw[i] += h;
        h = pw[i] < h;
    }
    if (!(rc = dyad_words(&w->hi, pw, 3, -ATT_F))) *done = 1;
    return rc;
}

/* row r's intervals at precision p (Refinement), every operation's ends rounded outward at p + 64 bits: every key's
   argument and weight (with tabled, in the first round, the weight from Lemma D's tables where they apply: a key with the
   reference key's mask value, so that x_m = |sigma| D 2**u, and a weight from 2**-48; else Lemma X), Z and D; the far keys' split
   (Far keys); and at the keys at need[0 .. nneed), or at every key with need NULL, P, dS, the cancellation-free dS2, and
   Delta P and Delta dS. The far keys' omega_m = e**(x_F - x_m) are enclosed by Lemma X where every key or a kept one is
   needed, else by [0, 1] (x_F <= x_m; F's is 1): they then reach only the far keys' second-order terms. */
static int grow_round(att_grow *r, const att_gref_refine *f, int64_t p, int tabled, const size_t *need, size_t nneed) {
    r->p = p;
    r->has_far = 0;
    size_t n = r->n;
    if (r->nan || !n) return 0;
    const att_call *a = &f->c->a;
    int64_t q = p + 64;
    tabled = tabled && a->tables && a->sign && !a->q->flagged_count && !a->q->slots_hi && !a->k->flagged_count && !a->k->slots_hi;
    if (!need) nneed = n;
    ivl Z, D, t, u, ZK, WD, WK, Om, phi, DK, dD, eF, OD;
    ivl *iv[] = {&Z, &D, &t, &u, &ZK, &WD, &WK, &Om, &phi, &DK, &dD, &eF, &OD};
    for (size_t z = 0; z < sizeof iv / sizeof *iv; z++) ivl_init(iv[z]);
    dyad cap, dist;
    dyad_init(&cap);
    dyad_init(&dist);
    /* w the weights; for more than two needed keys, SW and SD every key's other keys' sums of w and of w dP (prefix plus
       suffix), else each needed key's by its own pass */
    int sums = !need || nneed > 2;
    ivl *w = malloc((n + 1) * sizeof *w), *SW = sums ? malloc((2 * n + 1) * sizeof *SW) : NULL, *SD = SW ? SW + n : NULL;
    int rc = !w || (sums && !SW) ? EXACTREF_NOMEM : 0;
    for (size_t m = 0; w && m < n; m++) ivl_init(&w[m]);
    for (size_t m = 0; SW && m < 2 * n; m++) ivl_init(&SW[m]);
    if (!rc && !(rc = dyad_set_si(&cap, -1, 40))) rc = exp_clamped(&cap, &cap, p, 1);
    /* the arguments x_m measured from the largest logit's key (x_ref = 0, so Z >= 1), the weights e**-(x_m), Z and sum w dP */
    const dyad *bias = a->mask_float ? r->bias : NULL;
    if (!rc) rc = att_reference_key(&r->ref, a, r->sd, bias, n, r->top);
    ivl *zero[] = {&Z, &WD, &ZK, &WK, &Om, &OD};
    for (size_t z = 0; !rc && z < sizeof zero / sizeof *zero; z++)
        if (!(rc = dyad_set_si(&zero[z]->lo, 0, 0))) rc = dyad_set_si(&zero[z]->hi, 0, 0);
    dyad pd;
    dyad_init(&pd);
    size_t F = n;
    if (!rc) rc = dyad_set_si(&pd, p, 0);
    for (size_t m = 0; !rc && m < n; m++) {
        if ((rc = att_argument(&r->x[m].lo, &r->x[m].hi, a, r->sd, bias, r->ref, m, &f->sabs_lo, &f->sabs_hi, p)) || (rc = dyad_neg(&t.lo, &r->x[m].hi)) ||
            (rc = dyad_neg(&t.hi, &r->x[m].lo)))
            break;
        /* far: its argument past p; F the far key of the largest logit, compared exactly, so every x_F - x_m of a far key is at
           most 0 */
        int sgn = 1;
        r->far[m] = dyad_cmp(&r->x[m].lo, &pd) > 0;
        if (r->far[m] && (F == n || (!(rc = att_logit_cmp(&sgn, a, r->sd, bias, m, F)) && sgn > 0))) F = m;
        if (rc) break;
        int done = 0, ok = 0;
        w128 Dm;
        if (tabled && (!bias || !dyad_cmp(&r->bias[m], &r->bias[r->ref])) && !(rc = score_distance(&Dm, &ok, r, a, m, &dist)) && ok)
            rc = table_weight(&w[m], &done, Dm, a->tables);
        if (!rc && !done && !(rc = exp_clamped(&w[m].lo, &t.lo, p, 0)))
            rc = dyad_sign(&t.hi) < 0 && top_bit(&t.hi) >= 40 ? dyad_copy(&w[m].hi, &cap) : exp_clamped(&w[m].hi, &t.hi, p, 1);
        if (!rc && !(rc = ivl_add(&Z, &Z, &w[m], q)) && !(rc = ivl_point(&u, &r->dp[m])) && !(rc = ivl_mul(&u, &w[m], &u, q))) rc = ivl_add(&WD, &WD, &u, q);
        if (!rc && !r->far[m] && !(rc = ivl_add(&ZK, &ZK, &w[m], q))) rc = ivl_add(&WK, &WK, &u, q);
    }
    dyad_free(&pd);
    /* D = sum P dP */
    if (!rc) rc = ivl_div(&D, &WD, &Z, q);
    if (!rc && SW) {
        /* the other keys' sums for every key: prefix, plus suffix */
        for (int pass = 0; !rc && pass < 2; pass++) {
            ivl *out = pass ? SD : SW;
            if ((rc = dyad_set_si(&t.lo, 0, 0)) || (rc = dyad_set_si(&t.hi, 0, 0))) break;
            for (size_t m = 0; !rc && m < n; m++) {
                if ((rc = ivl_copy(&out[m], &t))) break;
                if (!pass) rc = ivl_add(&t, &t, &w[m], q);
                else if (!(rc = ivl_point(&u, &r->dp[m])) && !(rc = ivl_mul(&u, &w[m], &u, q))) rc = ivl_add(&t, &t, &u, q);
            }
            if (!rc && !(rc = dyad_set_si(&t.lo, 0, 0))) rc = dyad_set_si(&t.hi, 0, 0);
            for (size_t m = n; !rc && m-- > 0;) {
                if ((rc = ivl_add(&out[m], &out[m], &t, q))) break;
                if (!pass) rc = ivl_add(&t, &t, &w[m], q);
                else if (!(rc = ivl_point(&u, &r->dp[m])) && !(rc = ivl_mul(&u, &w[m], &u, q))) rc = ivl_add(&t, &t, &u, q);
            }
        }
    }
    /* at the needed keys: P = w / Z, dS = P (dP - D), dS2 = P (dP SW - SD) / Z */
    for (size_t k = 0; !rc && k < nneed; k++) {
        size_t m = need ? need[k] : k;
        ivl sw, sd;
        ivl_init(&sw);
        ivl_init(&sd);
        if (SW) {
            if (!(rc = ivl_copy(&sw, &SW[m]))) rc = ivl_copy(&sd, &SD[m]);
        } else if (!(rc = dyad_set_si(&sw.lo, 0, 0)) && !(rc = dyad_set_si(&sw.hi, 0, 0)) && !(rc = ivl_copy(&sd, &sw))) {
            for (size_t l = 0; !rc && l < n; l++)
                if (l != m && !(rc = ivl_add(&sw, &sw, &w[l], q)) && !(rc = ivl_point(&u, &r->dp[l])) && !(rc = ivl_mul(&u, &w[l], &u, q))) rc = ivl_add(&sd, &sd, &u, q);
        }
        if (!rc && !(rc = ivl_div(&r->P[m], &w[m], &Z, q)) && !(rc = ivl_point(&u, &r->dp[m])) && !(rc = ivl_sub(&t, &u, &D, q)))
            rc = ivl_mul(&r->dS[m], &r->P[m], &t, q);
        if (!rc && !(rc = ivl_mul(&t, &u, &sw, q)) && !(rc = ivl_sub(&t, &t, &sd, q)) && !(rc = ivl_div(&t, &t, &Z, q)))
            rc = ivl_mul(&r->dS2[m], &r->P[m], &t, q);
        ivl_free(&sw);
        ivl_free(&sd);
    }
    if (!rc) {
        int exact = !need;
        for (size_t k = 0; !exact && k < nneed; k++) exact = !r->far[need[k]];
        if (!rc && F < n) {
            r->has_far = 1;
            r->F = F;
            rc = ivl_copy(&r->xF, &r->x[F]);
            /* omega (exactly 1 for F), Omega and sum_far omega dP; [0, 1] dP is dP's hull with 0 */
            for (size_t m = 0; !rc && m < n; m++) {
                if (!r->far[m]) continue;
                if (m == F) {
                    if (!(rc = dyad_set_si(&r->om[m].lo, 1, 0))) rc = dyad_set_si(&r->om[m].hi, 1, 0);
                } else if (!exact) {
                    if (!(rc = dyad_set_si(&r->om[m].lo, 0, 0))) rc = dyad_set_si(&r->om[m].hi, 1, 0);
                } else if (!(rc = arg_diff(&t, r, f, F, m, p))) {
                    rc = ivl_exp_far(&r->om[m], &t, p);
                }
                if (rc || (rc = ivl_add(&Om, &Om, &r->om[m], q)) || (rc = ivl_point(&u, &r->dp[m]))) break;
                if (exact || m == F) rc = ivl_mul(&u, &r->om[m], &u, q);
                else if (dyad_sign(&u.lo) > 0) rc = dyad_set_si(&u.lo, 0, 0);
                else rc = dyad_set_si(&u.hi, 0, 0);
                if (!rc) rc = ivl_add(&OD, &OD, &u, q);
            }
            /* phi' = Omega / Z, D^K = sum_kept w dP / Z_K, Delta D' = -phi' D^K + sum_far omega dP / Z */
            if (!rc && !(rc = ivl_div(&phi, &Om, &Z, q)) && !(rc = ivl_div(&DK, &WK, &ZK, q)) && !(rc = ivl_mul(&t, &phi, &DK, q)) && !(rc = ivl_div(&u, &OD, &Z, q)))
                rc = ivl_sub(&dD, &u, &t, q);
            /* Delta P' = -P^K phi' (kept, relative to e**-(x_F)), 1 / Z (far, relative to e**-(x_m)); Delta dS' = Delta P' (dP - D^K)
               - P' Delta D', with P' = w / Z for a kept key and e**-(x_F) / Z for a far one (its one more factor e**-(x_F)) */
            if (!rc && !(rc = dyad_neg(&t.lo, &r->xF.hi)) && !(rc = dyad_neg(&t.hi, &r->xF.lo))) rc = ivl_exp_far(&eF, &t, p);
            if (!rc && !(rc = dyad_set_si(&D.lo, 1, 0)) && !(rc = dyad_set_si(&D.hi, 1, 0))) rc = ivl_div(&D, &D, &Z, q); /* 1 / Z (in D) */
            for (size_t k = 0; !rc && k < nneed; k++) {
                size_t m = need ? need[k] : k;
                if (r->far[m]) {
                    rc = ivl_copy(&r->dPr[m], &D);
                } else if (!(rc = ivl_div(&t, &w[m], &ZK, q)) && !(rc = ivl_mul(&t, &t, &phi, q)) && !(rc = dyad_neg(&r->dPr[m].lo, &t.hi))) {
                    rc = dyad_neg(&r->dPr[m].hi, &t.lo);
                }
                if (rc) break;
                if ((rc = ivl_point(&u, &r->dp[m])) || (rc = ivl_sub(&u, &u, &DK, q)) || (rc = ivl_mul(&r->dSr[m], &r->dPr[m], &u, q))) break;
                if (r->far[m]) {
                    if ((rc = ivl_mul(&t, &D, &eF, q))) break;
                } else if ((rc = ivl_copy(&t, &r->P[m]))) {
                    break;
                }
                if (!(rc = ivl_mul(&t, &t, &dD, q))) rc = ivl_sub(&r->dSr[m], &r->dSr[m], &t, q);
            }
        }
    }
    for (size_t m = 0; w && m < n; m++) ivl_free(&w[m]);
    for (size_t m = 0; SW && m < 2 * n; m++) ivl_free(&SW[m]);
    free(w);
    free(SW);
    for (size_t z = 0; z < sizeof iv / sizeof *iv; z++) ivl_free(iv[z]);
    dyad_free(&cap);
    dyad_free(&dist);
    return rc;
}

/* ---------------------------------------------------------------- elements */

/* an element's exact factor for row r and kept key position m: g (grad_value), q (grad_key) or k (grad_query) */
static int factor(dyad *x, const att_gref_refine *f, const att_gelem *el, const att_grow *r, size_t m) {
    const att_gref_call *c = f->c;
    size_t E = (size_t)c->a.E, Ev = c->a.Ev, at;
    int z = el->kind;
    if (z == 0)
        at = r->gr0 * Ev + el->col;
    else if (z == 1)
        at = r->qr * E + el->col;
    else
        at = (r->kr0 + r->keys[m]) * E + el->col;
    size_t n = (size_t)c->wx[z];
    uint64_t local[16], *tmp = n <= 16 ? local : malloc(n * sizeof *tmp); /* the factor's words: a float64 spans up to 33 */
    if (!tmp) return EXACTREF_NOMEM;
    int rc = mw_to_dyad(x, c->x[z] + at * n, c->wx[z], c->ux[z], tmp);
    if (tmp != local) free(tmp);
    return rc;
}

/* row r's term of the element's sum: v (P, dS or dS2 by which 0, 1, 2) times the factor, over the element's key (every
   kept key for grad_query) */
static int row_sum(ivl *s, const att_gref_refine *f, const att_gelem *el, const att_grow *r, int which, int64_t p) {
    int64_t q = p + 64;
    ivl t;
    ivl_init(&t);
    dyad x;
    dyad_init(&x);
    int rc = dyad_set_si(&s->lo, 0, 0);
    if (!rc) rc = dyad_set_si(&s->hi, 0, 0);
    size_t m0 = 0, m1 = r->nan ? 0 : r->n;
    if (!rc && m1 && el->kind != 2) { /* one key */
        size_t pos = r->pos[el->j];
        m0 = pos ? pos - 1 : 0;
        m1 = pos ? pos : 0;
    }
    for (size_t m = m0; !rc && m < m1; m++) {
        const ivl *v = which == 0 ? &r->P[m] : which == 1 ? &r->dS[m] : &r->dS2[m];
        if ((rc = factor(&x, f, el, r, m))) break;
        if (!dyad_sign(&x)) continue;
        if (!(rc = ivl_point(&t, &x)) && !(rc = ivl_mul(&t, v, &t, q))) rc = ivl_add(s, s, &t, q);
    }
    ivl_free(&t);
    dyad_free(&x);
    return rc;
}

/* Row r's term of the element's far keys' Delta' (Far keys): each of its terms, Delta P' or Delta dS' (by which 2 or 3)
   times its factor, is relative to e**-(y) with y = x_F for a kept key and the key's own argument for a far one. The
   row's scale is the least y of its terms with a nonzero factor (x_F when one is a kept key's or F's, else the far key of
   the largest logit), its other terms times e**(y_row - y) (omega, or e**(x_s - x_m) from exact differences): *u at the
   scale *y, which is sigma X + Y exactly (the score and mask differences from the reference key), *has 0 when no term has
   a nonzero factor or *u is exactly 0. */
static int row_far(int *has, ivl *y, dyad *X, dyad *Y, ivl *u, const att_gref_refine *f, const att_gelem *el, const att_grow *r, int which,
                   int64_t p) {
    int64_t q = p + 64;
    ivl t, fr;
    ivl_init(&t);
    ivl_init(&fr);
    dyad x;
    dyad_init(&x);
    *has = 0;
    size_t n = r->n, sk = n;
    const dyad *bias = f->c->a.mask_float ? r->bias : NULL;
    int rc = dyad_set_si(&u->lo, 0, 0);
    if (!rc) rc = dyad_set_si(&u->hi, 0, 0);
    const ivl *v = which == 2 ? r->dPr : r->dSr;
    if (rc || r->nan || !n || !r->has_far) goto done;
    if (el->kind != 2) { /* one key: its own scale */
        size_t pos = r->pos[el->j];
        if (!pos) goto done;
        size_t m = pos - 1;
        if ((rc = factor(&x, f, el, r, m)) || !dyad_sign(&x)) goto done;
        sk = r->far[m] ? m : r->F;
        if (!(rc = ivl_point(&t, &x))) rc = ivl_mul(u, &v[m], &t, q);
        goto done;
    }
    /* grad_query: every key at x_F (a far key times omega), and the far key of the largest logit with a nonzero factor,
       compared exactly as F, so that every x_least - x_m below is at most 0 */
    size_t least = n;
    int atF = 0, sgn = 1;
    for (size_t m = 0; !rc && m < n; m++) {
        if ((rc = factor(&x, f, el, r, m))) break;
        if (!dyad_sign(&x)) continue;
        if (!r->far[m] || m == r->F) atF = 1;
        else if (least == n || (!(rc = att_logit_cmp(&sgn, &f->c->a, r->sd, bias, m, least)) && sgn > 0)) least = m;
        if (rc) break;
        if ((rc = ivl_point(&t, &x)) || (rc = ivl_mul(&t, &v[m], &t, q))) break;
        if (r->far[m] && (rc = ivl_mul(&t, &t, &r->om[m], q))) break;
        rc = ivl_add(u, u, &t, q);
    }
    if (rc) goto done;
    if (atF) {
        sk = r->F;
    } else if (least < n) {
        /* no kept key's or F's term: the far keys' terms relative to e**-(x_least) */
        sk = least;
        if ((rc = dyad_set_si(&u->lo, 0, 0)) || (rc = dyad_set_si(&u->hi, 0, 0))) goto done;
        for (size_t m = 0; !rc && m < n; m++) {
            if (!r->far[m] || (rc = factor(&x, f, el, r, m)) || !dyad_sign(&x)) continue;
            if ((rc = ivl_point(&t, &x)) || (rc = ivl_mul(&t, &v[m], &t, q))) break;
            if (m != least && ((rc = arg_diff(&fr, r, f, least, m, p)) || (rc = ivl_exp_far(&fr, &fr, p)) || (rc = ivl_mul(&t, &t, &fr, q)))) break;
            rc = ivl_add(u, u, &t, q);
        }
    }
done:
    if (!rc && sk < n && (dyad_sign(&u->lo) || dyad_sign(&u->hi))) { /* a term that is exactly 0 sets no scale */
        *has = 1;
        if (!(rc = ivl_copy(y, &r->x[sk])) && !(rc = dyad_neg(X, &r->sd[sk])) && !(rc = dyad_add_exact(X, &r->sd[r->ref], X)))
            rc = !bias ? dyad_set_si(Y, 0, 0) : (rc = dyad_neg(Y, &r->bias[sk])) ? rc : dyad_add_exact(Y, &r->bias[r->ref], Y);
    }
    ivl_free(&t);
    ivl_free(&fr);
    dyad_free(&x);
    return rc;
}

/* s times sigma for grad_key and grad_query: exactly num / den's when rational (num exact, den outward), else 1/sqrt(E)'s
   ends at p bits */
static int scale_sigma(ivl *s, const att_gref_refine *f, const att_gelem *el, const dyad *slo, const dyad *shi, int64_t p) {
    const att_call *a = &f->c->a;
    if (el->kind == 0) return 0;
    ivl t;
    ivl_init(&t);
    int rc;
    if (a->den.n) {
        dyad nn, dd;
        dyad_init(&nn);
        dyad_init(&dd);
        if (!(rc = big_copy(&nn.m, &a->num)) && !(rc = big_copy(&dd.m, &a->den)) && !(rc = ivl_point(&t, &nn)) && !(rc = ivl_mul(&t, s, &t, p + 64)) &&
            !(rc = ivl_point(s, &dd)))
            rc = ivl_div(s, &t, s, p);
        dyad_free(&nn);
        dyad_free(&dd);
    } else if (!(rc = dyad_copy(&t.lo, slo)) && !(rc = dyad_copy(&t.hi, shi))) {
        rc = ivl_mul(s, s, &t, p);
    }
    ivl_free(&t);
    return rc;
}

static int put_bits(att_gref_refine *f, size_t e, const bigint *bits) {
    bytes_store(f->enc + e * f->c->out_size, f->c->out_size, bits);
    f->status[e] = 0;
    return 0;
}

/* an element known exactly: the rational q (sigma's factor already in), an exact zero as +0, or -0 rounding toward -inf */
static int put_rational(att_gref_refine *f, size_t e, rat *q) {
    const att_gref_call *c = f->c;
    bigint bits, zero;
    big_init(&bits);
    big_init(&zero);
    int rc, neg = q->num.neg;
    if (!q->num.n) {
        if (!(rc = floatfmt_encode(&bits, &c->out, VALUE_FINITE, c->neg_zero, &zero, 0, c->rnd, c->ovf))) rc = entry_value(&f->entries[e], VALUE_FINITE, c->neg_zero, &zero, 0);
    } else {
        q->num.neg = 0;
        if (!(rc = floatfmt_encode_ratio(&bits, &c->out, neg, &q->num, &q->den, c->rnd, c->ovf))) rc = entry_ratio(&f->entries[e], neg, &q->num, &q->den);
        q->num.neg = neg;
    }
    if (!rc) put_bits(f, e, &bits);
    big_free(&bits);
    big_free(&zero);
    return rc;
}

/* ---------------------------------------------------------------- exact parts */

/* the end of the run of equal logits that starts at position start of row r's sorted order */
static size_t group_end(const att_grow *r, size_t start) {
    size_t end = start + 1;
    while (end < r->n && !gkeyed_cmp(&r->order[end - 1], &r->order[end])) end++;
    return end;
}

/* the kept keys' row: the keys not far (every kept key without kept_only) */
static int in_kept(const att_grow *r, size_t m, int kept_only) { return !kept_only || !r->far[m]; }

/* row r's groups of equal logits over its kept keys (far ones removed with kept_only), for the exact terms of grad_value
   and grad_key (Which terms are rational): their count, the kept keys, the sum of dP, the first group's size and dP sum,
   and whether every group's mean dP is one value */
typedef struct {
    size_t groups, nk, n0;
    dyad sum, g0;
    int same;
} rstats;

static void rstats_init(rstats *st) {
    memset(st, 0, sizeof *st);
    dyad_init(&st->sum);
    dyad_init(&st->g0);
}

static void rstats_free(rstats *st) {
    dyad_free(&st->sum);
    dyad_free(&st->g0);
}

static int row_stats(rstats *st, const att_grow *r, int kept_only) {
    st->groups = st->nk = st->n0 = 0;
    st->same = 1;
    dyad gs, t1, t2;
    dyad_init(&gs);
    dyad_init(&t1);
    dyad_init(&t2);
    int rc = dyad_set_si(&st->sum, 0, 0);
    for (size_t start = 0, end; !rc && !r->nan && start < r->n; start = end) {
        end = group_end(r, start);
        size_t size = 0;
        rc = dyad_set_si(&gs, 0, 0);
        for (size_t k = start; !rc && k < end; k++)
            if (in_kept(r, r->order[k].j, kept_only)) {
                size++;
                rc = dyad_add_exact(&gs, &gs, &r->dp[r->order[k].j]);
            }
        if (!size || rc) continue;
        st->groups++;
        st->nk += size;
        rc = dyad_add_exact(&st->sum, &st->sum, &gs);
        if (!st->n0) {
            if (!rc) rc = dyad_copy(&st->g0, &gs);
            st->n0 = size;
        } else if (!rc && !(rc = dyad_set_si(&t1, (int64_t)size, 0)) && !(rc = dyad_mul_exact(&t1, &st->g0, &t1)) && !(rc = dyad_set_si(&t2, (int64_t)st->n0, 0)) &&
                   !(rc = dyad_mul_exact(&t2, &gs, &t2))) {
            st->same = st->same && !dyad_cmp(&t1, &t2);
        }
    }
    dyad_free(&gs);
    dyad_free(&t1);
    dyad_free(&t2);
    return rc;
}

/* Whether the row's dS terms of the element (grad_key: its key's; grad_query: every key's) are exactly 0 (Which terms are
   rational): every group of equal logits over all of the row's keys has one mean dP m, so D = m, and the key's dP is m
   (for grad_query every key's dP is one value). st: row_stats over every key. */
static int row_ds_zero(int *zero, const att_grow *r, const att_gelem *el, const rstats *st) {
    *zero = 0;
    if (r->nan || !r->n || el->kind == 0) return 0;
    if (el->kind == 2) {
        for (size_t m = 1; m < r->n; m++)
            if (dyad_cmp(&r->dp[m], &r->dp[0])) return 0;
        *zero = 1;
        return 0;
    }
    size_t pos = r->pos[el->j];
    if (!pos || !st->same) return 0;
    dyad t;
    dyad_init(&t);
    int rc = dyad_set_si(&t, (int64_t)st->n0, 0);
    if (!rc && !(rc = dyad_mul_exact(&t, &t, &r->dp[pos - 1]))) *zero = !dyad_cmp(&t, &st->g0);
    dyad_free(&t);
    return rc;
}

/* row r's exact term of a grad_value or grad_key element over its kept keys (st, with kept_only as there): *known 0
   when it is irrational. grad_value: P_j g with P_j = 1/n for equal logits; grad_key: dS_j q, dS_j = (n dP_j - sum dP) / n**2
   for equal logits, 0 when every group's mean dP is one value m and dP_j = m (Which terms are rational). */
static int row_term(rat *term, int *known, const att_gref_refine *f, const att_gelem *el, const att_grow *r, const rstats *st, int kept_only) {
    *known = 1;
    int rc = rat_set_si(term, 0);
    if (rc || r->nan || !r->n) return rc;
    size_t pos = r->pos[el->j];
    if (!pos || !in_kept(r, pos - 1, kept_only)) return 0;
    size_t m = pos - 1;
    dyad x, t1, t2;
    dyad *ds[] = {&x, &t1, &t2};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    rat v;
    rat_init(&v);
    if ((rc = factor(&x, f, el, r, m)) || !dyad_sign(&x)) goto done;
    if (el->kind == 0) {
        if (st->groups != 1) {
            *known = 0;
        } else if (!(rc = rat_set_dyad(term, &x.m, x.e)) && !(rc = rat_set_si(&v, (int64_t)st->nk))) { /* g / n */
            rc = rat_div(term, term, &v);
        }
    } else if (st->groups == 1) { /* (n dP_j - sum dP) q / n**2 */
        if (!(rc = dyad_set_si(&t1, (int64_t)st->nk, 0)) && !(rc = dyad_mul_exact(&t1, &t1, &r->dp[m])) && !(rc = dyad_neg(&t2, &st->sum)) &&
            !(rc = dyad_add_exact(&t1, &t1, &t2)) && !(rc = dyad_mul_exact(&t1, &t1, &x)) && !(rc = rat_set_dyad(term, &t1.m, t1.e)) &&
            !(rc = rat_set_si(&v, (int64_t)(st->nk * st->nk))))
            rc = rat_div(term, term, &v);
    } else if (st->same) { /* dS_j = 0 when dP_j n0 = g0 */
        if (!(rc = dyad_set_si(&t1, (int64_t)st->n0, 0)) && !(rc = dyad_mul_exact(&t1, &t1, &r->dp[m]))) *known = !dyad_cmp(&t1, &st->g0);
    } else {
        *known = 0;
    }
done:
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    rat_free(&v);
    return rc;
}

/* a group's sums for the pair test */
typedef struct {
    size_t n, head;          /* members, and one member's position (its logit) */
    dyad d, c, a;            /* sums of dP, of the key column's entries, of their products */
} pgroup;

/* The pair keys: a pair (g, h)'s logit sum in comparable parts, (s_g + s_h, key_g + key_h), each 0 where the order has
   none. Where each part of the row's groups spans at most 64 PK_WORDS - 2 bits, it is held as a two's-complement integer
   of PK_WORDS words at one exponent (on): a pair's sums are then exact integer sums, compared word by word; else dyads. */
#define PK_WORDS 4

typedef struct {
    int on;
    uint64_t (*w)[2][PK_WORDS]; /* per group its two parts */
} pkeys;

typedef struct {
    uint64_t code;            /* the pair (g, h) as g << 32 | h */
    dyad ts, tb;              /* its sums as dyads, */
    uint64_t k[2][PK_WORDS];  /* or as integers */
} pnode;

static void pkeys_free(pkeys *pk) {
    free(pk->w);
    pk->w = NULL;
    pk->on = 0;
}

/* the integer keys of the row's G groups, where they fit (pk->on) */
static int pkeys_init(pkeys *pk, const att_grow *r, const pgroup *g, size_t G) {
    pk->on = 0;
    pk->w = NULL;
    int64_t lo[2] = {0, 0}, hi[2] = {0, 0};
    int any[2] = {0, 0};
    for (size_t x = 0; x < G; x++)
        for (int z = 0; z < 2; z++) {
            const dyad *v = z ? r->order[g[x].head].key : r->order[g[x].head].s;
            if (!v || !v->m.n) continue;
            int64_t t = v->e + big_bits(&v->m);
            if (!any[z] || v->e < lo[z]) lo[z] = v->e;
            if (!any[z] || t > hi[z]) hi[z] = t;
            any[z] = 1;
        }
    for (int z = 0; z < 2; z++)
        if (any[z] && hi[z] - lo[z] > 64 * PK_WORDS - 2) return 0;
    pk->w = calloc(G + 1, sizeof *pk->w);
    if (!pk->w) return EXACTREF_NOMEM;
    for (size_t x = 0; x < G; x++)
        for (int z = 0; z < 2; z++) {
            const dyad *v = z ? r->order[g[x].head].key : r->order[g[x].head].s;
            uint64_t *w = pk->w[x][z];
            if (!v || !v->m.n) continue;
            int64_t shift = v->e - lo[z];
            size_t q = (size_t)(shift / 64);
            int bshift = (int)(shift % 64);
            for (size_t i = 0; i < v->m.n && i + q < PK_WORDS; i++) {
                w[i + q] |= v->m.w[i] << bshift;
                if (bshift && i + q + 1 < PK_WORDS) w[i + q + 1] |= v->m.w[i] >> (64 - bshift);
            }
            if (v->m.neg) { /* two's complement */
                uint64_t carry = 1;
                for (int i = 0; i < PK_WORDS; i++) {
                    w[i] = ~w[i] + carry;
                    carry = carry && !w[i];
                }
            }
        }
    pk->on = 1;
    return 0;
}

/* a pair (g, h)'s logit sum in comparable parts, each 0 where the order has none */
static int pair_key(pnode *x, const att_grow *r, const pgroup *G, const pkeys *pk) {
    size_t g = (size_t)(x->code >> 32), h = (size_t)(x->code & 0xffffffffu);
    if (pk->on) {
        for (int z = 0; z < 2; z++) {
            uint64_t carry = 0;
            for (int i = 0; i < PK_WORDS; i++) {
                uint64_t s = pk->w[g][z][i] + carry, t = s + pk->w[h][z][i];
                carry = (s < carry) + (t < s);
                x->k[z][i] = t;
            }
        }
        return 0;
    }
    const gkeyed *a = &r->order[G[g].head], *b = &r->order[G[h].head];
    int rc = 0;
    if (a->s) rc = dyad_add_exact(&x->ts, a->s, b->s);
    else rc = dyad_set_si(&x->ts, 0, 0);
    if (!rc && a->key) rc = dyad_add_exact(&x->tb, a->key, b->key);
    else if (!rc) rc = dyad_set_si(&x->tb, 0, 0);
    return rc;
}

static int pk_cmp(const uint64_t *x, const uint64_t *y) {
    if (x[PK_WORDS - 1] != y[PK_WORDS - 1]) return (int64_t)x[PK_WORDS - 1] < (int64_t)y[PK_WORDS - 1] ? -1 : 1;
    for (int i = PK_WORDS - 2; i >= 0; i--)
        if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}

static int pair_cmp(const pnode *x, const pnode *y, const pkeys *pk) {
    int c = pk->on ? pk_cmp(x->k[0], y->k[0]) : dyad_cmp(&x->ts, &y->ts);
    return c ? c : pk->on ? pk_cmp(x->k[1], y->k[1]) : dyad_cmp(&x->tb, &y->tb);
}

/* restores the heap order (least logit sum first) below place i of n: h holds indices into the nodes, which never move
   (a dyad is never copied by value) */
static void pair_sift(size_t *h, const pnode *nodes, size_t n, size_t i, const pkeys *pk) {
    for (;;) {
        size_t l = 2 * i + 1, m = i;
        if (l < n && pair_cmp(&nodes[h[l]], &nodes[h[m]], pk) < 0) m = l;
        if (l + 1 < n && pair_cmp(&nodes[h[l + 1]], &nodes[h[m]], pk) < 0) m = l + 1;
        if (m == i) return;
        size_t t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

/* the pair test's shortcuts (grad_query of one row, decided exactly): every pair's own ratio one value, and the image of
   F = n_1**2 (N A - D C) - k_1 N**2 modulo the prime P = 2**61 - 1, its exponents modulo M = 2**60 - 1 (both odd, so 2 is
   invertible modulo each); Y1 and Y2 are 37**2 and 43**2 (37 and 43 generate the units modulo P, so Y**M = 1) */
#define PT_P ((((uint64_t)1) << 61) - 1)
#define PT_M ((((uint64_t)1) << 60) - 1)
#define PT_Y1 ((uint64_t)1369)
#define PT_Y2 ((uint64_t)1849)

/* x modulo q = 2**b - 1, b 60 or 61 */
static uint64_t pt_red(uint64_t x, int b, uint64_t q) {
    x = (x & q) + (x >> b);
    x = (x & q) + (x >> b);
    return x >= q ? x - q : x;
}

/* a b modulo P, for a, b < P: 2**64 is 8 modulo P */
static uint64_t pt_mul(uint64_t a, uint64_t b) {
    uint64_t hi, lo = word_mul(&hi, a, b);
    return pt_red((lo & PT_P) + (lo >> 61) + (hi << 3), 61, PT_P);
}

/* T[64 i + j] = y**(j 2**(6 i)) for i < 10, j < 64: y**e for e < 2**60 as ten table entries' product */
static void pt_table(uint64_t *T, uint64_t y) {
    for (int i = 0; i < 10; i++) {
        T[64 * i] = 1;
        for (int j = 1; j < 64; j++) T[64 * i + j] = pt_mul(T[64 * i + j - 1], y);
        y = pt_mul(T[64 * i + 63], y);
    }
}

static uint64_t pt_pow(const uint64_t *T, uint64_t e) {
    uint64_t z = T[e & 63];
    for (int i = 1; i < 10; i++) z = pt_mul(z, T[64 * i + ((e >> (6 * i)) & 63)]);
    return z;
}

/* a bigint modulo q = 2**b - 1 (2**64 is 2**(64 - b) modulo q) */
static uint64_t pt_big(const bigint *x, int b, uint64_t q) {
    uint64_t z = 0;
    for (size_t i = x->n; i-- > 0;) z = pt_red(pt_red(z << (64 - b), b, q) + pt_red(x->w[i], b, q), b, q);
    return x->neg && z ? q - z : z;
}

/* a dyadic m 2**e modulo P (2**61 is 1 modulo P) */
static uint64_t pt_coef(const dyad *x) {
    int64_t k = x->e % 61;
    return pt_mul(pt_big(&x->m, 61, PT_P), (uint64_t)1 << (k < 0 ? k + 61 : k));
}

/* a dyadic exponent modulo M (2**60 is 1 modulo M: times 2**k is a rotation of 60 bits) */
static uint64_t pt_exp(const dyad *x) {
    int64_t k = x->e % 60;
    uint64_t v = pt_big(&x->m, 60, PT_M);
    if (k < 0) k += 60;
    return k ? ((v << k) & PT_M) | (v >> (60 - k)) : v;
}

/* every pair's own ratio A_gh / (n_g n_h) the value 2 W_1: every W_g = (n_g a_g - d_g c_g) / n_g**2 one value and the
   means d_g / n_g, or c_g / n_g, one value; *k1 = n_1 a_1 - d_1 c_1 */
static int pair_uniform(int *same, dyad *k1, const pgroup *G, size_t n) {
    dyad k, t, u;
    dyad_init(&k);
    dyad_init(&t);
    dyad_init(&u);
    int w = 1, dm = 1, cm = 1, rc = 0;
    for (size_t g = 0; !rc && w && (dm || cm) && g < n; g++) {
        if ((rc = dyad_set_si(&t, (int64_t)G[g].n, 0)) || (rc = dyad_mul_exact(&k, &t, &G[g].a)) || (rc = dyad_mul_exact(&t, &G[g].d, &G[g].c)) ||
            (rc = dyad_neg(&t, &t)) || (rc = dyad_add_exact(&k, &k, &t)))
            break;
        if (!g) {
            rc = dyad_copy(k1, &k);
            continue;
        }
        if ((rc = dyad_set_si(&t, (int64_t)(G[0].n * G[0].n), 0)) || (rc = dyad_mul_exact(&t, &t, &k)) || (rc = dyad_set_si(&u, (int64_t)(G[g].n * G[g].n), 0)) ||
            (rc = dyad_mul_exact(&u, &u, k1)))
            break;
        w = !dyad_cmp(&t, &u);
        if (dm && !(rc = dyad_set_si(&t, (int64_t)G[0].n, 0)) && !(rc = dyad_mul_exact(&t, &t, &G[g].d)) && !(rc = dyad_set_si(&u, (int64_t)G[g].n, 0)) &&
            !(rc = dyad_mul_exact(&u, &u, &G[0].d)))
            dm = !dyad_cmp(&t, &u);
        if (!rc && cm && !(rc = dyad_set_si(&t, (int64_t)G[0].n, 0)) && !(rc = dyad_mul_exact(&t, &t, &G[g].c)) && !(rc = dyad_set_si(&u, (int64_t)G[g].n, 0)) &&
            !(rc = dyad_mul_exact(&u, &u, &G[0].c)))
            cm = !dyad_cmp(&t, &u);
    }
    *same = !rc && w && (dm || cm);
    dyad_free(&k);
    dyad_free(&t);
    dyad_free(&u);
    return rc;
}

/* F's image modulo P: X**s for a group's logit parts (u, v) maps to Y1**psi(u) Y2**psi(v), a coefficient to itself */
static uint64_t pair_image(const att_grow *r, const pgroup *G, size_t n) {
    uint64_t N = 0, D = 0, C = 0, A = 0, T1[640], T2[640];
    pt_table(T1, PT_Y1);
    pt_table(T2, PT_Y2);
    for (size_t g = 0; g < n; g++) {
        const gkeyed *o = &r->order[G[g].head];
        uint64_t x = 1;
        if (o->s) x = pt_pow(T1, pt_exp(o->s));
        if (o->key) x = pt_mul(x, pt_pow(T2, pt_exp(o->key)));
        N = pt_red(N + pt_mul(pt_red((uint64_t)G[g].n, 61, PT_P), x), 61, PT_P);
        D = pt_red(D + pt_mul(pt_coef(&G[g].d), x), 61, PT_P);
        C = pt_red(C + pt_mul(pt_coef(&G[g].c), x), 61, PT_P);
        A = pt_red(A + pt_mul(pt_coef(&G[g].a), x), 61, PT_P);
    }
    uint64_t n1 = pt_red((uint64_t)G[0].n, 61, PT_P);
    uint64_t k1 = pt_red(pt_mul(n1, pt_coef(&G[0].a)) + PT_P - pt_mul(pt_coef(&G[0].d), pt_coef(&G[0].c)), 61, PT_P);
    uint64_t lhs = pt_mul(pt_mul(n1, n1), pt_red(pt_mul(N, A) + PT_P - pt_mul(D, C), 61, PT_P));
    return pt_red(lhs + PT_P - pt_mul(k1, pt_mul(N, N)), 61, PT_P);
}

/* A_ab = n_b a_a - d_a c_b - d_b c_a + n_a a_b, the pair (a, b)'s own C over its N = n_a n_b (before the factor 2 of a < b) */
static int pair_own(dyad *A, dyad *t, const pgroup *g, size_t a, size_t b) {
    int rc;
    if ((rc = dyad_set_si(t, (int64_t)g[b].n, 0)) || (rc = dyad_mul_exact(A, t, &g[a].a)) || (rc = dyad_mul_exact(t, &g[a].d, &g[b].c)) ||
        (rc = dyad_neg(t, t)) || (rc = dyad_add_exact(A, A, t)) || (rc = dyad_mul_exact(t, &g[b].d, &g[a].c)) || (rc = dyad_neg(t, t)) ||
        (rc = dyad_add_exact(A, A, t)) || (rc = dyad_set_si(t, (int64_t)g[a].n, 0)) || (rc = dyad_mul_exact(t, t, &g[b].a)))
        return rc;
    return dyad_add_exact(A, A, t);
}

/* the merge for the columns active marks (their tests left them), pg[c] column c's groups: every ratio C_t / N_t one
   value over the runs of equal pair keys, x = C_0 / (2 N_0), else known 0. The pairs come once for every column (their
   order is the row's logits'). */
static int pair_merge_cols(rat **xq, int **known, int *active, size_t cols, const att_grow *r, pgroup *const *pg, size_t G, const pkeys *pk) {
    size_t c0 = 0, live = 0;
    for (size_t c = 0; c < cols; c++)
        if (active[c]) live++;
    while (c0 < cols && !active[c0]) c0++;
    if (!live) return 0;
    dyad A, t, t2, *C = malloc((2 * cols + 1) * sizeof *C), *Cc = C ? C + cols : NULL;
    dyad_init(&A);
    dyad_init(&t);
    dyad_init(&t2);
    pnode *nodes = malloc(G * sizeof *nodes), run;
    size_t *heap = nodes ? malloc(G * sizeof *heap) : NULL;
    dyad_init(&run.ts);
    dyad_init(&run.tb);
    int rc = !C || !nodes || !heap ? EXACTREF_NOMEM : 0;
    for (size_t c = 0; C && c < 2 * cols; c++) dyad_init(&C[c]);
    for (size_t a = 0; nodes && a < G; a++) {
        dyad_init(&nodes[a].ts);
        dyad_init(&nodes[a].tb);
    }
    if (!rc) {
        /* the pairs (g, h), g <= h, in the order of their logit sums without holding them all: each g's pairs with
           h = g, g + 1, ... come in that order (the groups are sorted), so a heap of G merges them; per run of equal pair
           keys C_t (2 A_gh for g < h, A_gg) and N_t (2 n_g n_h, n_g**2), every ratio equal, the first unequal one ending a
           column */
        const pgroup *g = pg[c0];
        size_t hn = G, runs = 0;
        for (size_t a = 0; !rc && a < G; a++) {
            heap[a] = a;
            nodes[a].code = (uint64_t)a << 32 | a;
            rc = pair_key(&nodes[a], r, g, pk);
        }
        for (size_t i = hn / 2; !rc && i-- > 0;) pair_sift(heap, nodes, hn, i, pk);
        int64_t N0 = 0, N = 0;
        while (!rc && live && hn) {
            pnode *top = &nodes[heap[0]];
            if (!runs || pair_cmp(top, &run, pk)) { /* a new run: the last one's ratios first */
                for (size_t c = 0; !rc && c < cols; c++) {
                    if (!active[c]) continue;
                    if (runs == 1) {
                        rc = dyad_copy(&Cc[c], &C[c]);
                    } else if (runs > 1 && !(rc = dyad_set_si(&t, N0, 0)) && !(rc = dyad_mul_exact(&t, &t, &C[c])) && !(rc = dyad_set_si(&t2, N, 0)) &&
                               !(rc = dyad_mul_exact(&t2, &t2, &Cc[c])) && dyad_cmp(&t, &t2)) {
                        active[c] = 0;
                        *known[c] = 0;
                        live--;
                    }
                    if (!rc) rc = dyad_set_si(&C[c], 0, 0);
                }
                if (runs == 1) N0 = N;
                if (rc || !live || (pk->on ? 0 : (rc = dyad_copy(&run.ts, &top->ts)) || (rc = dyad_copy(&run.tb, &top->tb)))) break;
                if (pk->on) memcpy(run.k, top->k, sizeof run.k);
                N = 0;
                runs++;
            }
            size_t a = (size_t)(top->code >> 32), b = (size_t)(top->code & 0xffffffffu);
            for (size_t c = 0; !rc && c < cols; c++) {
                if (!active[c] || (rc = pair_own(&A, &t, pg[c], a, b))) continue;
                if (a != b && (rc = dyad_shift(&A, 1))) continue;
                rc = dyad_add_exact(&C[c], &C[c], &A);
            }
            if (rc) break;
            N += a != b ? 2 * (int64_t)(g[a].n * g[b].n) : (int64_t)(g[a].n * g[a].n);
            if (b + 1 < G) { /* g's next pair */
                top->code = (uint64_t)a << 32 | (b + 1);
                rc = pair_key(top, r, g, pk);
            } else { /* g's pairs done */
                heap[0] = heap[--hn];
            }
            if (!rc) pair_sift(heap, nodes, hn, 0, pk);
        }
        /* the last run, and x = C_0 / (2 N_0) */
        if (!rc && runs == 1) N0 = N;
        rat v;
        rat_init(&v);
        for (size_t c = 0; !rc && c < cols; c++) {
            if (!active[c]) continue;
            if (runs == 1) {
                rc = dyad_copy(&Cc[c], &C[c]);
            } else if (runs > 1 && !(rc = dyad_set_si(&t, N0, 0)) && !(rc = dyad_mul_exact(&t, &t, &C[c])) && !(rc = dyad_set_si(&t2, N, 0)) &&
                       !(rc = dyad_mul_exact(&t2, &t2, &Cc[c])) && dyad_cmp(&t, &t2)) {
                *known[c] = 0;
                continue;
            }
            if (!rc && !(rc = rat_set_dyad(xq[c], &Cc[c].m, Cc[c].e)) && !(rc = rat_set_si(&v, 2 * N0))) rc = rat_div(xq[c], xq[c], &v);
        }
        rat_free(&v);
    }
    for (size_t a = 0; nodes && a < G; a++) {
        dyad_free(&nodes[a].ts);
        dyad_free(&nodes[a].tb);
    }
    free(nodes);
    free(heap);
    dyad_free(&run.ts);
    dyad_free(&run.tb);
    for (size_t c = 0; C && c < 2 * cols; c++) dyad_free(&C[c]);
    free(C);
    dyad_free(&A);
    dyad_free(&t);
    dyad_free(&t2);
    return rc;
}

/* the witness check (grad_query of one row, decided exactly): x is transcendental when one run of equal pair keys has a
   ratio other than run 0's, (0, 0) alone. A run is found whole by walking in from both ends of the sorted groups (their
   logits distinct and their order translation-invariant, so for each i at most one j has s_i + s_j = t). It is tried for
   up to PT_WITNESSES pairs whose own ratio A_ab / (n_a n_b) is not run 0's, as a run whose pairs all keep run 0's ratio
   keeps it: (0, h), (h, h) and (h, G - 1) for h = 1, 2, ..., else the pair (a, b) with d_a / n_a != d_0 / n_0 and
   c_b / n_b != c_0 / n_0, which the uniform test's failure leaves */
#define PT_WITNESSES 4

/* *odd: the pair (a, b)'s own ratio is not run 0's, C0 / n_0**2 */
static int pair_odd(int *odd, dyad *A, dyad *t, dyad *u, const pgroup *g, size_t a, size_t b, const dyad *C0) {
    int rc = pair_own(A, t, g, a, b);
    if (!rc && !(rc = dyad_set_si(t, (int64_t)(g[0].n * g[0].n), 0)) && !(rc = dyad_mul_exact(t, t, A)) && !(rc = dyad_set_si(u, (int64_t)(g[a].n * g[b].n), 0)) &&
        !(rc = dyad_mul_exact(u, u, C0)))
        *odd = dyad_cmp(t, u) != 0;
    return rc;
}

/* *differs: the run of the pair (a, b)'s key has a ratio other than run 0's */
static int pair_run(int *differs, const att_grow *r, const pgroup *g, size_t G, size_t a, size_t b, const dyad *C0, const pkeys *pk) {
    pnode w, x;
    dyad C, A, t, u;
    dyad *ds[] = {&w.ts, &w.tb, &x.ts, &x.tb, &C, &A, &t, &u};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    w.code = (uint64_t)a << 32 | b;
    int rc = pair_key(&w, r, g, pk);
    int64_t N = 0;
    size_t i = 0, j = G - 1;
    while (!rc && i <= j) {
        x.code = (uint64_t)i << 32 | j;
        if ((rc = pair_key(&x, r, g, pk))) break;
        int c = pair_cmp(&x, &w, pk);
        if (c < 0) {
            i++;
        } else if (c > 0) {
            if (!j--) break;
        } else {
            if ((rc = pair_own(&A, &t, g, i, j))) break;
            if (i != j) {
                if ((rc = dyad_shift(&A, 1))) break;
                N += 2 * (int64_t)(g[i].n * g[j].n);
            } else {
                N += (int64_t)(g[i].n * g[i].n);
            }
            if ((rc = dyad_add_exact(&C, &C, &A))) break;
            i++;
            if (!j--) break;
        }
    }
    if (!rc && !(rc = dyad_set_si(&t, (int64_t)(g[0].n * g[0].n), 0)) && !(rc = dyad_mul_exact(&t, &t, &C)) && !(rc = dyad_set_si(&u, N, 0)) &&
        !(rc = dyad_mul_exact(&u, &u, C0)))
        *differs = dyad_cmp(&t, &u) != 0;
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* *known 0 when a run proves x transcendental */
static int pair_witness(int *known, const att_grow *r, const pgroup *g, size_t G, const pkeys *pk) {
    *known = 1;
    if (G < 2) return 0;
    dyad C0, A, t, u;
    dyad *ds[] = {&C0, &A, &t, &u};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int rc = pair_own(&C0, &t, g, 0, 0), tried = 0;
    for (size_t h = 1; !rc && *known && tried < PT_WITNESSES && h < G; h++) {
        size_t cand[3][2] = {{0, h}, {h, h}, {h, G - 1}};
        for (int k = 0; !rc && *known && tried < PT_WITNESSES && k < (h == G - 1 ? 2 : 3); k++) {
            int odd = 0, differs = 0;
            if ((rc = pair_odd(&odd, &A, &t, &u, g, cand[k][0], cand[k][1], &C0)) || !odd) continue;
            tried++;
            if (!(rc = pair_run(&differs, r, g, G, cand[k][0], cand[k][1], &C0, pk)) && differs) *known = 0;
        }
    }
    if (!rc && *known && !tried) {
        size_t a = G, b = G;
        for (size_t h = 1; !rc && h < G && (a == G || b == G); h++) {
            if (a == G && !(rc = dyad_set_si(&t, (int64_t)g[0].n, 0)) && !(rc = dyad_mul_exact(&t, &t, &g[h].d)) && !(rc = dyad_set_si(&u, (int64_t)g[h].n, 0)) &&
                !(rc = dyad_mul_exact(&u, &u, &g[0].d)) && dyad_cmp(&t, &u))
                a = h;
            if (!rc && b == G && !(rc = dyad_set_si(&t, (int64_t)g[0].n, 0)) && !(rc = dyad_mul_exact(&t, &t, &g[h].c)) && !(rc = dyad_set_si(&u, (int64_t)g[h].n, 0)) &&
                !(rc = dyad_mul_exact(&u, &u, &g[0].c)) && dyad_cmp(&t, &u))
                b = h;
        }
        if (!rc && a < G && b < G && a != b) {
            int odd = 0, differs = 0;
            size_t lo = a < b ? a : b, hi = a < b ? b : a;
            if (!(rc = pair_odd(&odd, &A, &t, &u, g, lo, hi, &C0)) && odd && !(rc = pair_run(&differs, r, g, G, lo, hi, &C0, pk)) && differs) *known = 0;
        }
    }
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* grad_query of one row decided exactly (grad_query of one row, decided exactly), for several of the row's columns: each
   sum_j dS_j k_je over the kept keys (far ones removed with kept_only), before sigma, as the rational xq[c] when every
   ratio C_t / N_t is one value (x = C_t / (2 N_t)), else known[c] 0. The groups once; per column a constant key column
   (x = 0), every pair's own ratio one value (x = k_1 / n_1**2), F's image not 0 or a witness run (a ratio unequal); one
   merge for the columns those leave. */
static int pair_test_cols(rat **xq, int **known, size_t cols, const att_gref_refine *f, const att_gelem *const *els, const att_grow *r, int kept_only) {
    int rc = 0;
    for (size_t c = 0; !rc && c < cols; c++) {
        *known[c] = 1;
        rc = rat_set_si(xq[c], 0);
    }
    if (rc || r->nan || !r->n) return rc;
    size_t G = 0, *head = malloc((r->n + 1) * sizeof *head), *size = malloc((r->n + 1) * sizeof *size);
    dyad *d = malloc((r->n + 1) * sizeof *d);
    pgroup **pg = calloc(cols + 1, sizeof *pg);
    int *active = calloc(cols + 1, sizeof *active), *constant = calloc(cols + 1, sizeof *constant);
    pkeys pk = {0, NULL};
    int keyed = 0;
    dyad x, t, k1;
    dyad_init(&x);
    dyad_init(&t);
    dyad_init(&k1);
    if (!head || !size || !d || !pg || !active || !constant) rc = EXACTREF_NOMEM;
    /* the groups over the kept keys: their heads, sizes and sums of dP */
    for (size_t start = 0, end; !rc && start < r->n; start = end) {
        end = group_end(r, start);
        size_t n = 0;
        dyad_init(&d[G]);
        for (size_t k = start; !rc && k < end; k++) {
            size_t m = r->order[k].j;
            if (!in_kept(r, m, kept_only)) continue;
            if (!n++) head[G] = k;
            rc = dyad_add_exact(&d[G], &d[G], &r->dp[m]);
        }
        if (n) size[G++] = n;
        else dyad_free(&d[G]);
    }
    /* per column its groups (d a read-only view of the shared sums), its key column's sums c and a = sum dP c */
    for (size_t c = 0; !rc && c < cols; c++) {
        if (!(pg[c] = malloc((G + 1) * sizeof *pg[c]))) {
            rc = EXACTREF_NOMEM;
            break;
        }
        int any = 0;
        constant[c] = 1;
        dyad first;
        dyad_init(&first);
        for (size_t g = 0; g < G; g++) {
            pgroup *q = &pg[c][g];
            q->n = size[g];
            q->head = head[g];
            q->d = d[g];
            dyad_init(&q->c);
            dyad_init(&q->a);
        }
        for (size_t g = 0, k = 0; !rc && g < G; g++)
            for (k = head[g]; !rc && k < r->n && !gkeyed_cmp(&r->order[head[g]], &r->order[k]); k++) {
                size_t m = r->order[k].j;
                if (!in_kept(r, m, kept_only)) continue;
                if (!(rc = factor(&x, f, els[c], r, m)) && !(rc = dyad_add_exact(&pg[c][g].c, &pg[c][g].c, &x)) && !(rc = dyad_mul_exact(&t, &r->dp[m], &x)))
                    rc = dyad_add_exact(&pg[c][g].a, &pg[c][g].a, &t);
                if (!rc && !any) rc = dyad_copy(&first, &x);
                constant[c] = constant[c] && (!any || !dyad_cmp(&x, &first));
                any = 1;
            }
        dyad_free(&first);
    }
    /* per column its tests, the keys made when a witness or the merge needs them */
    for (size_t c = 0; !rc && G && c < cols; c++) {
        if (constant[c]) continue; /* every A_gh is 0: x = 0 */
        int same = 0;
        if ((rc = pair_uniform(&same, &k1, pg[c], G))) break;
        if (same) {
            rat v;
            rat_init(&v);
            if (!(rc = rat_set_dyad(xq[c], &k1.m, k1.e)) && !(rc = rat_set_si(&v, (int64_t)(size[0] * size[0])))) rc = rat_div(xq[c], xq[c], &v);
            rat_free(&v);
        } else if (pair_image(r, pg[c], G)) {
            *known[c] = 0;
        } else {
            if (!keyed && (rc = pkeys_init(&pk, r, pg[c], G))) break;
            keyed = 1;
            if (!(rc = pair_witness(known[c], r, pg[c], G, &pk)) && *known[c]) active[c] = 1;
        }
    }
    if (!rc && G) rc = pair_merge_cols(xq, known, active, cols, r, pg, G, &pk);
    pkeys_free(&pk);
    for (size_t c = 0; pg && c < cols; c++) {
        for (size_t g = 0; pg[c] && g < G; g++) {
            dyad_free(&pg[c][g].c);
            dyad_free(&pg[c][g].a);
        }
        free(pg[c]);
    }
    for (size_t g = 0; d && g < G; g++) dyad_free(&d[g]);
    free(head);
    free(size);
    free(d);
    free(pg);
    free(active);
    free(constant);
    dyad_free(&x);
    dyad_free(&t);
    dyad_free(&k1);
    return rc;
}

/* ---------------------------------------------------------------- the rounds */

/* |sigma|'s ends at p bits into lo and hi: num/den's, or 1/sqrt(E)'s without a given scale */
static int sigma_ends(dyad *lo, dyad *hi, const att_call *a, int64_t p) {
    int rc;
    if (a->den.n) {
        dyad x, y;
        dyad_init(&x);
        dyad_init(&y);
        if (!(rc = big_copy(&x.m, &a->num)) && !(rc = big_copy(&y.m, &a->den))) {
            x.m.neg = 0;
            if (!(rc = dyad_div(lo, &x, &y, p, 0))) rc = dyad_div(hi, &x, &y, p, 1);
        }
        dyad_free(&x);
        dyad_free(&y);
        return rc;
    }
    dyad e, r, one;
    dyad_init(&e);
    dyad_init(&r);
    dyad_init(&one);
    if (!(rc = dyad_set_si(&e, a->E, 0)) && !(rc = dyad_set_si(&one, 1, 0)) && !(rc = dyad_sqrt(&r, &e, p + 8, 1)) && !(rc = dyad_div(lo, &one, &r, p, 0)) &&
        !(rc = dyad_sqrt(&r, &e, p + 8, 0)))
        rc = dyad_div(hi, &one, &r, p, 1);
    dyad_free(&e);
    dyad_free(&r);
    dyad_free(&one);
    return rc;
}

/* |sigma|'s ends at p bits, and 1/sqrt(E)'s; and |sigma|'s at p + 128 for the far groups' factors (far_take) */
static int sigma_at(att_gref_refine *f, int64_t p) {
    int rc = sigma_ends(&f->sabs_lo, &f->sabs_hi, &f->c->a, p);
    if (!rc && !f->c->a.den.n && !(rc = dyad_copy(&f->slo, &f->sabs_lo))) rc = dyad_copy(&f->shi, &f->sabs_hi);
    return rc ? rc : sigma_ends(&f->sfar_lo, &f->sfar_hi, &f->c->a, p + 128);
}

/* a bigint exponent as int64, clamped to [-lim - 1, lim] */
static int64_t shift_of(const bigint *n, int64_t lim) {
    if (!n->n) return 0;
    if (n->n > 1 || n->w[0] >= (uint64_t)lim) return n->neg ? -lim - 1 : lim;
    return n->neg ? -(int64_t)n->w[0] : (int64_t)n->w[0];
}

/* the far keys' split for element e: G^K exact (gk, sigma's factor in) and Delta = e**-(x_m) Delta' (s; x_m = sigma X + Y
   its group's scale, xm0 the round's enclosure of it), decided by Deciding a rounding by a sign with c = G^K. The offset at q = p + 64 bits (sigma's factor, e**-(x_m)
   and their products), x_m itself from G's exact X and Y with sigma's ends at q plus x_m's bits, so that its width
   leaves e**-(x_m) within 2**-q of itself however large x_m: the round's own x_m (at p) would keep the offset too wide
   to decide at p = 128. */
static int far_decide(att_gref_refine *f, size_t e, const rat *gk, ivl *s, const ivl *xm0, const dyad *X, const dyad *Y, int64_t p, int last) {
    const att_gelem *el = &f->elems[e];
    int64_t q = p + 64;
    ivl ctr, off, xm;
    dyad elo, ehi, t, u, slo, shi;
    bigint bits, n1, n2;
    ivl_init(&ctr);
    ivl_init(&off);
    ivl_init(&xm);
    dyad *ds[] = {&elo, &ehi, &t, &u, &slo, &shi};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    big_init(&bits);
    big_init(&n1);
    big_init(&n2);
    /* x_m's bits from the round's enclosure, then x_m = sigma X + Y at q plus those */
    int64_t xb = 0;
    for (int z = 0; z < 2; z++) {
        const dyad *d = z ? &xm0->hi : &xm0->lo;
        int64_t tz = dyad_sign(d) ? d->e + big_bits(&d->m) : 0;
        if (tz > xb) xb = tz;
    }
    int64_t qx = q + (xb < ((int64_t)1 << 40) ? xb : (int64_t)1 << 40);
    int rc = sigma_ends(&slo, &shi, &f->c->a, qx), side = 0;
    if (!rc) rc = att_affine(NULL, &xm.lo, &xm.hi, &f->c->a, X, f->c->a.mask_float ? Y : NULL, &slo, &shi, qx);
    if (!rc) rc = scale_sigma(s, f, el, &slo, &shi, q);
    if (!rc) side = dyad_sign(&s->lo) > 0 ? 1 : dyad_sign(&s->hi) < 0 ? -1 : 0;
    /* e**-(x_m) by Lemma X, its scale 2**n of any size */
    if (!rc && side && !(rc = dyad_neg(&t, &xm.hi)) && !(rc = exp_bounds(&elo, &u, &n1, &t, q)) && !(rc = dyad_neg(&t, &xm.lo)))
        rc = exp_bounds(&u, &ehi, &n2, &t, q);
    if (!rc && side) {
        int64_t lim = (int64_t)1 << 61, s1 = shift_of(&n1, lim), s2 = shift_of(&n2, lim);
        if (s1 >= -lim && s2 >= -lim && s1 < lim && s2 < lim) {
            if (!(rc = dyad_shift(&elo, s1)) && !(rc = dyad_shift(&ehi, s2)) && !(rc = dyad_mul(&off.lo, &s->lo, side > 0 ? &elo : &ehi, q, 0))) rc = dyad_mul(&off.hi, &s->hi, side > 0 ? &ehi : &elo, q, 1);
        } else {
            /* tiny: with Delta''s ends below 2**(2**60), |Delta| < 2**-(LIMIT + 1); the tiny-offset rule */
            int64_t top = 0;
            for (int z = 0; z < 2; z++) {
                const dyad *d = z ? &s->hi : &s->lo;
                int64_t tz = dyad_sign(d) ? d->e + big_bits(&d->m) : 0;
                if (tz > top) top = tz;
            }
            if (s2 >= -lim || top >= ((int64_t)1 << 60)) side = 0; /* both of e**-(x_m)'s ends below 2**(1 - 2**61) */
            else if (!(rc = dyad_set_si(&off.lo, side, -(EXACT_LIMIT + 1)))) rc = dyad_copy(&off.hi, &off.lo);
        }
    }
    /* c = G^K: a point when dyadic, else rounded outward at p */
    if (!rc && side) {
        int dyadic = big_trailing(&gk->den) == big_bits(&gk->den) - 1;
        if (dyadic) {
            if (!(rc = dyad_from_big(&ctr.lo, &gk->num, 1 - big_bits(&gk->den)))) rc = dyad_copy(&ctr.hi, &ctr.lo);
        } else {
            dyad nn, dd;
            dyad_init(&nn);
            dyad_init(&dd);
            if (!(rc = big_copy(&nn.m, &gk->num)) && !(rc = big_copy(&dd.m, &gk->den)) && !(rc = dyad_div(&ctr.lo, &nn, &dd, p, 0))) rc = dyad_div(&ctr.hi, &nn, &dd, p, 1);
            dyad_free(&nn);
            dyad_free(&dd);
        }
        int none = 1;
        if (!rc) rc = offset_rounding(&bits, &f->entries[e], &none, &f->c->out, f->c->rnd, f->c->ovf, &ctr.lo, &ctr.hi, &off.lo, &off.hi, p, last);
        if (!rc && !none) put_bits(f, e, &bits);
    }
    ivl_free(&ctr);
    ivl_free(&off);
    ivl_free(&xm);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    big_free(&bits);
    big_free(&n1);
    big_free(&n2);
    return rc;
}

/* Each round streams the active rows (those of a pending element's row list) in waves of at most WAVE_ROWS rows and
   WAVE_TERMS terms (at least one row: a wave's memory bounded however many elements a row reaches), each wave in tiles
   of up to TILE_ROWS rows on the threads (one row each where a wave is small, so that its rows run at once); a tile takes each of its rows' terms of the row's pending elements (gterm), and
   after the wave they are added to the elements' sums in row order, so that the sums are the same however the tiles
   ran. Elements sharing a row list (a span) and a key share the far terms' scale (a group). A wave of fewer than
   SPLIT_ROWS rows holding at least SPLIT_TERMS terms (rows that reach many elements, as with many padded keys) runs
   split, so that its few rows still fill the threads: its rows made (a tile each), its terms taken in tiles of
   consecutive terms of one row, its rows' grad_query pair tests (a tile each), then its terms added to the sums in tiles
   of consecutive groups, each group's terms in row order: the same terms and sums as unsplit. A split wave holds its
   rows at once, about row_key_bytes(p) per key each, so its rows are also kept within SPLIT_BYTES (at least one). */
#define TILE_ROWS 4
#define WAVE_ROWS 128
#define WAVE_TERMS ((size_t)1 << 18)
#define SPLIT_ROWS 32
#define SPLIT_TERMS ((size_t)1 << 14)
#define SPLIT_TILES 64 /* a split wave's term tiles and fold tiles: about this many each */
#define ROW_KEY_BYTES 3072 /* a split wave's row's bytes per key up to BIG_INLINE words a dyad */
/* two rows of 65536 keys at ROW_KEY_BYTES: one-row waves past about 65536 keys at the first rounds (p <= 512), and from
   fewer keys at higher precision, row_key_bytes(p) growing (about 45600 at p = 4096, 22300 at the cap) */
#define SPLIT_BYTES ((size_t)3 << 27)

/* a split wave's row's bytes per key at precision p: ROW_KEY_BYTES and, past BIG_INLINE words, the words of three p-bit
   dyads a key (measured from p = 128 to the cap, 16384: about 2700 to 6800 bytes) */
static size_t row_key_bytes(int64_t p) {
    int64_t words = (p + 63) / 64;
    return ROW_KEY_BYTES + (words > BIG_INLINE ? 3 * 8 * (size_t)(words - BIG_INLINE) : 0);
}

typedef struct {             /* a row's terms of one of its pending elements */
    size_t e;
    ivl plain, ds, u, y;     /* the plain and cancellation-free sums' terms; the far keys' term u at the row's scale y */
    dyad X, Y;               /* y = sigma X + Y exactly */
    rat term, *term0;        /* its exact terms over the kept keys and, for the first round's pair test (grad_query), over
                                every key: held only there */
    int has_u, has_far, known, known0, need, need0;
} gterm;

typedef struct {             /* an element's sums in the round */
    ivl plain, ds, far;      /* far: the far keys' Delta' at its group's scale */
    rat exact, exact0;
    int found, any_far, known, known0;
} gsum;

typedef struct {             /* elements sharing the far terms' scale: one key's columns over one span, or a grad_query element */
    size_t e0, e1;
    ivl xm;
    dyad X, Y;               /* x_m = sigma X + Y exactly */
    int found;
    int pending, kept, grid; /* an element pending; a far group's (Far keys' sums): a row keeps its key, not far; its grid set */
    int64_t gf, gtop;        /* ... its grid 2**gf, set from a factor below 2**gtop */
} ggroup;

/* The far groups (proofs/scaled_dot_product_attention_backward.md, Far keys): a grad_value or grad_key group whose key is far
   in every row of its span that keeps it. A pass over the rows first (the scales) finds whether a row keeps the key not
   far, and the least scale x_m over the rows where it is far with a nonzero term; then each such row's factor, Delta P'
   or Delta dS' times e**(x_m - y) (y the key's argument in the row, so the factor is at most the interval), and each
   column's sums of the factor's ends times the exact input as integers on a grid 2**gf: gf lies p + 64 + FAR_GUARD bits
   below the group's largest factor, and moves up (the sums shifted, rounded outward) when a factor passes it by more than
   FAR_SLACK bits. */
#define FAR_GUARD 8
#define FAR_SLACK 48

enum { SC_KEPT = 1, SC_CAND = 2 };

typedef struct {             /* the scales: a row's entry of one grad_value or grad_key group */
    size_t g;
    ivl y;                   /* the key's argument */
    dyad X, Y;               /* y = sigma X + Y exactly */
    unsigned char flags;     /* SC_KEPT: the row keeps the key, not far; SC_CAND: far, with a nonzero term */
} scent;

typedef struct {             /* a row's factor of one far group, relative to e**-(x_m) */
    size_t g;
    ivl v;
    int nonzero;
} fent;

struct att_gref_work {
    int64_t p, cap;
    int first, open, out, done; /* the first round; a round under way; a wave's tiles handed out; finished */
    size_t nspans, *span_g0, *span_g1; /* per span its groups */
    size_t *row_at, *row_span;  /* per row its spans: row_span[row_at[r] .. row_at[r + 1]) */
    size_t ngroups, *group_of;  /* per element its group */
    ggroup *groups;
    gsum *sums;
    unsigned char *mark;        /* per row: in the round */
    size_t nactive, *active;    /* the round's rows, ascending */
    size_t wave0, wave1;        /* the wave's rows of active */
    size_t tile_rows;           /* rows per tile in the wave */
    size_t *term_at;            /* per row of the wave its terms' first index */
    gterm *terms;               /* the wave's terms, in an array kept across waves (tcap places) */
    size_t nterms, tcap;
    int phase;                  /* a split wave's step: 1 rows made, 2 terms, 3 pair tests, 4 sums (0: not split) */
    size_t *term_e;             /* a split wave's terms' elements, placed by the term tiles */
    unsigned char *live;        /* per term of a split wave: initialized and not yet freed */
    size_t nparts, *part_at, *part_row; /* its term tiles: terms part_at[c] .. part_at[c + 1]) of row part_row[c] */
    size_t nfolds, *fold_g;     /* its fold tiles: groups fold_g[c] .. fold_g[c + 1]) */
    size_t nst;                 /* its rows' statistics over their kept keys and over every key (has_st bits 1 and 2) */
    rstats *st, *st0;
    unsigned char *has_st;      /* bit 4: the row has grad_query terms */
    int stage;                  /* the round's step: 1 the scales' rows, 2 their merge, 0 the terms */
    int farfold;                /* a wave's far fold tiles handed out */
    unsigned char *gfar;        /* per group: a far group in the round */
    size_t nfar, *farg;         /* the far groups with a nonzero term, ascending */
    size_t *ent_at;             /* per row of the wave its entries' first index: the scales' (sc) or the far factors' (fe) */
    scent *sc;
    fent *fe;
    size_t nsc, nfe;
    size_t nrange, *range;      /* the merge's and the far fold's tiles: groups range[c] .. range[c + 1]) (of every group; of farg) */
    uint64_t *acc;              /* the far groups' pending elements' sums: lo, then hi, wacc[kind] words each, at acc_at[e] */
    size_t *acc_at;
    int kf, wacc[2];            /* a factor's words on its grid; the sums' words */
    size_t waves, split_waves;  /* the run's waves, and those that ran split (att_gref_refine_waves, for tests) */
};

static void term_init(gterm *t) {
    ivl_init(&t->plain);
    ivl_init(&t->ds);
    ivl_init(&t->u);
    ivl_init(&t->y);
    dyad_init(&t->X);
    dyad_init(&t->Y);
    rat_init(&t->term);
}

static void term_free(gterm *t) {
    ivl_free(&t->plain);
    ivl_free(&t->ds);
    ivl_free(&t->u);
    ivl_free(&t->y);
    dyad_free(&t->X);
    dyad_free(&t->Y);
    rat_free(&t->term);
    if (t->term0) rat_free(t->term0);
    free(t->term0);
    t->term0 = NULL;
}

static void terms_free(att_gref_work *w) {
    for (size_t i = 0; w->terms && i < w->nterms; i++)
        if (!w->live || w->live[i]) term_free(&w->terms[i]);
    for (size_t k = 0; k < w->nst; k++) {
        rstats_free(&w->st[k]);
        rstats_free(&w->st0[k]);
    }
    free(w->term_at);
    free(w->term_e);
    free(w->live);
    free(w->part_at);
    free(w->part_row);
    free(w->fold_g);
    free(w->st);
    free(w->st0);
    free(w->has_st);
    w->term_at = w->term_e = w->part_at = w->part_row = w->fold_g = NULL;
    w->live = w->has_st = NULL;
    w->st = w->st0 = NULL;
    w->nterms = w->nparts = w->nfolds = w->nst = 0;
    w->phase = 0;
}

/* a wave's entries freed: the scales' and the far factors' */
static void entries_free(att_gref_work *w) {
    for (size_t i = 0; w->sc && i < w->nsc; i++) {
        ivl_free(&w->sc[i].y);
        dyad_free(&w->sc[i].X);
        dyad_free(&w->sc[i].Y);
    }
    for (size_t i = 0; w->fe && i < w->nfe; i++) ivl_free(&w->fe[i].v);
    free(w->sc);
    free(w->fe);
    free(w->ent_at);
    w->sc = NULL;
    w->fe = NULL;
    w->ent_at = NULL;
    w->nsc = w->nfe = 0;
}

/* the round's far sums freed */
static void far_free(att_gref_work *w) {
    free(w->acc);
    free(w->acc_at);
    w->acc = NULL;
    w->acc_at = NULL;
    w->nfar = 0;
}

static void work_free(att_gref_work *w, size_t count) {
    if (!w) return;
    terms_free(w);
    entries_free(w);
    far_free(w);
    free(w->gfar);
    free(w->farg);
    free(w->range);
    free(w->terms);
    for (size_t g = 0; w->groups && g < w->ngroups; g++) {
        ivl_free(&w->groups[g].xm);
        dyad_free(&w->groups[g].X);
        dyad_free(&w->groups[g].Y);
    }
    for (size_t e = 0; w->sums && e < count; e++) {
        gsum *x = &w->sums[e];
        ivl_free(&x->plain);
        ivl_free(&x->ds);
        ivl_free(&x->far);
        rat_free(&x->exact);
        rat_free(&x->exact0);
    }
    free(w->span_g0);
    free(w->span_g1);
    free(w->row_at);
    free(w->row_span);
    free(w->group_of);
    free(w->groups);
    free(w->sums);
    free(w->mark);
    free(w->active);
    free(w);
}

static int any_pending(const att_gref_refine *f) {
    for (size_t e = 0; e < f->count; e++)
        if (f->status[e]) return 1;
    return 0;
}

/* The spans (elements in a row with the same row list), their groups and each row's spans; the first round at p0 */
int att_gref_refine_start(att_gref_refine *f, int64_t p0, int64_t cap) {
    att_gref_work *w = calloc(1, sizeof *w);
    if (!w) return EXACTREF_NOMEM;
    f->work = w;
    w->cap = cap;
    w->p = p0 < cap ? p0 : cap;
    w->first = 1;
    size_t ne = f->count, nr = f->nrows;
    w->span_g0 = malloc((ne + 1) * sizeof *w->span_g0);
    w->span_g1 = malloc((ne + 1) * sizeof *w->span_g1);
    w->group_of = malloc((ne + 1) * sizeof *w->group_of);
    w->groups = calloc(ne + 1, sizeof *w->groups);
    w->sums = calloc(ne + 1, sizeof *w->sums);
    w->row_at = calloc(nr + 2, sizeof *w->row_at);
    w->mark = calloc(nr + 1, 1);
    w->active = malloc((nr + 1) * sizeof *w->active);
    w->gfar = calloc(ne + 1, 1);
    w->farg = malloc((ne + 1) * sizeof *w->farg);
    w->range = malloc((SPLIT_TILES + 2) * sizeof *w->range);
    if (!w->span_g0 || !w->span_g1 || !w->group_of || !w->groups || !w->sums || !w->row_at || !w->mark || !w->active || !w->gfar || !w->farg || !w->range)
        return EXACTREF_NOMEM;
    for (size_t e = 0; e < ne; e++) {
        gsum *x = &w->sums[e];
        ivl_init(&x->plain);
        ivl_init(&x->ds);
        ivl_init(&x->far);
        rat_init(&x->exact);
        rat_init(&x->exact0);
    }
    /* groups: consecutive grad_value or grad_key elements of one key and row list, each grad_query element alone; spans:
       consecutive groups with one row list */
    for (size_t e = 0; e < ne; e++) {
        const att_gelem *el = &f->elems[e], *prev = e ? &f->elems[e - 1] : NULL;
        int same_list = prev && prev->first == el->first && prev->count == el->count;
        if (!(same_list && el->kind != 2 && prev->kind == el->kind && prev->j == el->j)) {
            ggroup *g = &w->groups[w->ngroups++];
            g->e0 = e;
            ivl_init(&g->xm);
            dyad_init(&g->X);
            dyad_init(&g->Y);
            if (!same_list) {
                w->span_g0[w->nspans] = w->ngroups - 1;
                w->nspans++;
            }
            w->span_g1[w->nspans - 1] = w->ngroups;
        }
        w->groups[w->ngroups - 1].e1 = e + 1;
        w->group_of[e] = w->ngroups - 1;
    }
    /* each row's spans */
    size_t total = 0;
    for (size_t s = 0; s < w->nspans; s++) {
        const att_gelem *el = &f->elems[w->groups[w->span_g0[s]].e0];
        for (size_t k = 0; k < el->count; k++) w->row_at[f->rowlist[el->first + k] + 1]++;
        total += el->count;
    }
    for (size_t r = 0; r < nr; r++) w->row_at[r + 1] += w->row_at[r];
    w->row_span = malloc((total + 1) * sizeof *w->row_span);
    size_t *fill = calloc(nr + 1, sizeof *fill);
    if (!w->row_span || !fill) {
        free(fill);
        return EXACTREF_NOMEM;
    }
    for (size_t s = 0; s < w->nspans; s++) {
        const att_gelem *el = &f->elems[w->groups[w->span_g0[s]].e0];
        for (size_t k = 0; k < el->count; k++) {
            size_t r = f->rowlist[el->first + k];
            w->row_span[w->row_at[r] + fill[r]++] = s;
        }
    }
    free(fill);
    return 0;
}

/* a round's start: sigma at p, the pending elements' sums cleared, the groups' scales, the active rows */
static int round_open(att_gref_refine *f) {
    att_gref_work *w = f->work;
    int rc = sigma_at(f, w->p);
    for (size_t e = 0; !rc && e < f->count; e++) {
        if (!f->status[e]) continue;
        gsum *x = &w->sums[e];
        if ((rc = dyad_set_si(&x->plain.lo, 0, 0)) || (rc = dyad_set_si(&x->plain.hi, 0, 0)) || (rc = ivl_copy(&x->ds, &x->plain)) ||
            (rc = ivl_copy(&x->far, &x->plain)) || (rc = rat_set_si(&x->exact, 0)) || (rc = rat_set_si(&x->exact0, 0)))
            break;
        x->found = x->any_far = 0;
        x->known = x->known0 = 1;
    }
    /* the groups' pending elements; the far groups found again by the scales, which run where a grad_value or grad_key
       group is pending, in ranges of about equal group counts */
    far_free(w);
    memset(w->gfar, 0, w->ngroups);
    int scales = 0;
    for (size_t g = 0; g < w->ngroups; g++) {
        ggroup *G = &w->groups[g];
        G->found = G->kept = G->grid = G->pending = 0;
        for (size_t e = G->e0; !G->pending && e < G->e1; e++) G->pending = f->status[e];
        scales = scales || (G->pending && f->elems[G->e0].kind != 2);
    }
    w->stage = scales;
    w->nrange = w->ngroups < SPLIT_TILES ? w->ngroups : SPLIT_TILES;
    for (size_t c = 0; c <= w->nrange; c++) w->range[c] = w->nrange ? c * w->ngroups / w->nrange : 0;
    memset(w->mark, 0, f->nrows);
    for (size_t s = 0; s < w->nspans; s++) {
        int pending = 0;
        for (size_t g = w->span_g0[s]; !pending && g < w->span_g1[s]; g++)
            for (size_t e = w->groups[g].e0; !pending && e < w->groups[g].e1; e++) pending = f->status[e];
        if (!pending) continue;
        const att_gelem *el = &f->elems[w->groups[w->span_g0[s]].e0];
        for (size_t k = 0; k < el->count; k++) w->mark[f->rowlist[el->first + k]] = 1;
    }
    w->nactive = 0;
    for (size_t r = 0; r < f->nrows; r++)
        if (w->mark[r]) w->active[w->nactive++] = r;
    w->wave0 = 0;
    w->open = 1;
    return rc;
}

/* row r's terms in the round: one per pending element of its spans outside the far groups */
static size_t row_terms(const att_gref_refine *f, size_t r) {
    const att_gref_work *w = f->work;
    size_t n = 0;
    for (size_t i = w->row_at[r]; i < w->row_at[r + 1]; i++)
        for (size_t g = w->span_g0[w->row_span[i]]; g < w->span_g1[w->row_span[i]]; g++)
            for (size_t e = w->groups[g].e0; w->groups[g].pending && !w->gfar[g] && e < w->groups[g].e1; e++) n += f->status[e] != 0;
    return n;
}

/* row r's entries: its spans' pending grad_value and grad_key groups (the scales), or with far its far groups with a
   nonzero term */
static size_t row_entries(const att_gref_refine *f, size_t r, int far, size_t *gs) {
    const att_gref_work *w = f->work;
    size_t n = 0;
    for (size_t i = w->row_at[r]; i < w->row_at[r + 1]; i++)
        for (size_t g = w->span_g0[w->row_span[i]]; g < w->span_g1[w->row_span[i]]; g++) {
            const ggroup *G = &w->groups[g];
            if (far ? w->gfar[g] && G->found : G->pending && f->elems[G->e0].kind != 2) {
                if (gs) gs[n] = g;
                n++;
            }
        }
    return n;
}

/* a split wave's tiles: its rows' terms cut into parts of about nterms / SPLIT_TILES consecutive terms, its groups into
   at most SPLIT_TILES fold ranges of about as many of the wave's terms, over every row; its rows' statistics, empty;
   its first step (the rows made, a tile each) */
static int split_plan(att_gref_work *w, size_t rows, size_t *tiles) {
    size_t n = w->nterms, size = (n + SPLIT_TILES - 1) / SPLIT_TILES, parts = 0;
    for (size_t k = 0; k < rows; k++) {
        size_t m = w->term_at[k + 1] - w->term_at[k];
        parts += (m + size - 1) / size;
    }
    w->part_at = malloc((parts + 1) * sizeof *w->part_at);
    w->part_row = malloc((parts + 1) * sizeof *w->part_row);
    w->fold_g = malloc((SPLIT_TILES + 2) * sizeof *w->fold_g);
    w->st = malloc((rows + 1) * sizeof *w->st);
    w->st0 = malloc((rows + 1) * sizeof *w->st0);
    w->has_st = calloc(rows + 1, 1);
    if (!w->part_at || !w->part_row || !w->fold_g || !w->st || !w->st0 || !w->has_st) return EXACTREF_NOMEM;
    for (size_t k = 0; k < rows; k++, w->nst++) {
        rstats_init(&w->st[k]);
        rstats_init(&w->st0[k]);
    }
    for (size_t k = 0; k < rows; k++)
        for (size_t a = w->term_at[k]; a < w->term_at[k + 1]; a += size) {
            w->part_at[w->nparts] = a;
            w->part_row[w->nparts++] = k;
        }
    w->part_at[w->nparts] = n;
    /* the wave's terms per group, then ranges of groups closing where the running count passes each multiple of size */
    size_t *count = calloc(w->ngroups + 1, sizeof *count), folds = 0, seen = 0, next = size;
    if (!count) return EXACTREF_NOMEM;
    for (size_t t = 0; t < n; t++) count[w->group_of[w->term_e[t]]]++;
    w->fold_g[folds++] = 0;
    for (size_t g = 0; g + 1 < w->ngroups && folds < SPLIT_TILES; g++) {
        seen += count[g];
        if (size && seen >= next) {
            w->fold_g[folds++] = g + 1;
            while (next <= seen) next += size;
        }
    }
    free(count);
    w->fold_g[folds] = w->ngroups;
    w->nfolds = folds;
    w->phase = 1;
    *tiles = rows;
    return 0;
}

/* the wave's far factors' places: per row, its far groups with a nonzero term, ascending */
static int far_entries(att_gref_refine *f, size_t rows, size_t held) {
    att_gref_work *w = f->work;
    w->ent_at = malloc((rows + 1) * sizeof *w->ent_at);
    w->fe = malloc((held + 1) * sizeof *w->fe);
    size_t *gs = malloc((w->ngroups + 1) * sizeof *gs);
    int rc = !w->ent_at || !w->fe || !gs ? EXACTREF_NOMEM : 0;
    for (size_t k = 0, n = 0; !rc && k < rows; k++) {
        w->ent_at[k] = n;
        size_t m = row_entries(f, w->active[w->wave0 + k], 1, gs);
        for (size_t i = 0; i < m; i++, n++) {
            w->fe[n].g = gs[i];
            w->fe[n].nonzero = 0;
            ivl_init(&w->fe[n].v);
        }
        w->nfe = n;
        w->ent_at[k + 1] = n;
    }
    free(gs);
    return rc;
}

/* the wave's rows and their terms' places: per row, its spans, their groups, their pending elements */
static int wave_open(att_gref_refine *f, size_t *tiles) {
    att_gref_work *w = f->work;
    size_t held = 0, fheld = 0, key_bytes = row_key_bytes(w->p);
    w->wave1 = w->wave0;
    while (w->wave1 < w->nactive && w->wave1 - w->wave0 < WAVE_ROWS) {
        size_t t = row_terms(f, w->active[w->wave1]), u = w->nfar ? row_entries(f, w->active[w->wave1], 1, NULL) : 0;
        if (w->wave1 > w->wave0 && (held + t > WAVE_TERMS || fheld + u > WAVE_TERMS)) break;
        /* rows that would run split, past their bytes */
        size_t rows_after = w->wave1 - w->wave0 + 1;
        if (w->wave1 > w->wave0 && held + t >= SPLIT_TERMS && rows_after < SPLIT_ROWS && rows_after * f->c->a.S * key_bytes > SPLIT_BYTES) break;
        held += t;
        fheld += u;
        w->wave1++;
    }
    size_t rows = w->wave1 - w->wave0, n = 0;
    int split = rows < SPLIT_ROWS && held >= SPLIT_TERMS, rc;
    w->waves++;
    w->split_waves += (size_t)split;
    if (fheld && (rc = far_entries(f, rows, fheld))) return rc;
    w->term_at = malloc((rows + 1) * sizeof *w->term_at);
    if (!w->term_at) return EXACTREF_NOMEM;
    for (int pass = 0; pass < 2; pass++) {
        n = 0;
        for (size_t k = 0; k < rows; k++) {
            size_t r = w->active[w->wave0 + k];
            if (pass) w->term_at[k] = n;
            for (size_t i = w->row_at[r]; i < w->row_at[r + 1]; i++) {
                size_t s = w->row_span[i];
                for (size_t g = w->span_g0[s]; g < w->span_g1[s]; g++)
                    for (size_t e = w->groups[g].e0; !w->gfar[g] && e < w->groups[g].e1; e++) {
                        if (!f->status[e]) continue;
                        if (pass && split) w->term_e[n] = e;
                        else if (pass) w->terms[n].e = e;
                        n++;
                    }
            }
        }
        if (!pass) {
            /* split, the terms are initialized by the term tiles, on the threads */
            if (split && (!(w->live = calloc(n + 1, 1)) || !(w->term_e = malloc((n + 1) * sizeof *w->term_e)))) return EXACTREF_NOMEM;
            if (n + 1 > w->tcap) {
                free(w->terms);
                if (!(w->terms = calloc(n + 1, sizeof *w->terms))) {
                    w->tcap = 0;
                    return EXACTREF_NOMEM;
                }
                w->tcap = n + 1;
            }
            w->nterms = n;
            for (size_t i = 0; !split && i < n; i++) term_init(&w->terms[i]);
        }
    }
    w->term_at[rows] = n;
    if (split) return split_plan(w, rows, tiles);
    /* one row a tile where rows have far factors (their work is the rows') */
    w->tile_rows = fheld || rows / 32 < 1 ? 1 : rows / 32 > TILE_ROWS ? TILE_ROWS : rows / 32;
    *tiles = (rows + w->tile_rows - 1) / w->tile_rows;
    return 0;
}

static size_t term_elem(const att_gref_work *w, size_t i) { return w->term_e ? w->term_e[i] : w->terms[i].e; }

/* row k's far factors in the wave */
static size_t row_far_count(const att_gref_work *w, size_t k) { return w->fe ? w->ent_at[k + 1] - w->ent_at[k] : 0; }

/* row k of the wave made and enclosed at p at the keys its terms and far factors read: one per grad_value or grad_key
   element or far group (its key, where the row keeps it), every key for grad_query; need holds the row's term and far
   factor count, *seen is the scratch it replaces */
static int row_prepare(att_gref_refine *f, att_grow *r, size_t k, att_row *scratch, size_t *need, unsigned char **seen) {
    att_gref_work *w = f->work;
    int rc = grow_ready(r, f, scratch), all = 0;
    if (rc) return rc;
    size_t nneed = 0, nt = w->term_at[k + 1] - w->term_at[k], nf = row_far_count(w, k);
    free(*seen);
    if (!(*seen = calloc(r->n + 1, 1))) return EXACTREF_NOMEM;
    for (size_t i = 0; !all && i < nt + nf; i++) {
        const att_gelem *el = &f->elems[i < nt ? term_elem(w, w->term_at[k] + i) : w->groups[w->fe[w->ent_at[k] + i - nt].g].e0];
        size_t pos = el->kind == 2 || r->nan ? 0 : r->pos[el->j];
        all = el->kind == 2;
        if (pos && !(*seen)[pos - 1]) {
            (*seen)[pos - 1] = 1;
            need[nneed++] = pos - 1;
        }
    }
    return grow_round(r, f, w->p, w->first, all ? NULL : need, nneed);
}

/* term x of row r at p: its plain and cancellation-free terms, its far term and its exact term where needed; st and st0
   are the row's statistics over its kept keys and over every key, made here when first needed (*made bits 1 and 2)
   except in a split wave, whose row tiles make them beforehand so that its term tiles only read the row */
static int term_take(gterm *x, att_gref_refine *f, att_grow *r, rstats *st, rstats *st0, unsigned *made) {
    const att_gref_work *w = f->work;
    int64_t p = w->p;
    const att_gelem *el = &f->elems[x->e];
    int which = el->kind == 0 ? 0 : 1, rc;
    if ((rc = row_sum(&x->plain, f, el, r, which, p))) return rc;
    if ((rc = el->kind == 0 ? ivl_copy(&x->ds, &x->plain) : row_sum(&x->ds, f, el, r, 2, p))) return rc;
    if ((rc = row_far(&x->has_u, &x->y, &x->X, &x->Y, &x->u, f, el, r, el->kind == 0 ? 2 : 3, p))) return rc;
    x->has_far = r->has_far;
    if (x->has_u && el->kind != 0 && dyad_sign(&x->u.lo) <= 0 && dyad_sign(&x->u.hi) >= 0) {
        /* a dS term whose enclosure holds 0: exactly 0 where the exact analysis proves it, so it sets no scale */
        int zero = 0;
        if (el->kind == 1 && !(*made & 2)) {
            if (w->phase) return EXACTREF_INTERNAL;
            grow_order(r);
            if ((rc = row_stats(st0, r, 0))) return rc;
            *made |= 2;
        }
        if ((rc = row_ds_zero(&zero, r, el, st0))) return rc;
        if (zero) {
            x->has_u = 0;
            if ((rc = dyad_set_si(&x->u.lo, 0, 0)) || (rc = dyad_set_si(&x->u.hi, 0, 0))) return rc;
        }
    }
    /* the exact terms: the first round's pair test over every key; over the kept keys where a far split may need them (a
       grad_query element of one row only where that row has far keys); grad_query's by row_pair_tests */
    x->need0 = w->first && el->kind == 2;
    if (x->need0 && !x->term0) {
        if (!(x->term0 = malloc(sizeof *x->term0))) return EXACTREF_NOMEM;
        rat_init(x->term0);
    }
    x->need = el->kind != 2 || r->has_far || el->count > 1;
    if (x->need && el->kind != 2) {
        size_t pos = r->nan ? 0 : r->pos[el->j]; /* row_term's term is 0 where the row does not keep the key */
        if (pos && !r->far[pos - 1] && !(*made & 1)) {
            if (w->phase) return EXACTREF_INTERNAL;
            grow_order(r);
            if ((rc = row_stats(st, r, 1))) return rc;
            *made |= 1;
        }
        rc = row_term(&x->term, &x->known, f, el, r, st, 1);
    }
    return rc;
}

/* ---------------------------------------------------------------- far groups */

/* element el's exact input in row r, its words at 2**ux: g (grad_value) or q (grad_key), as factor reads it */
static const uint64_t *input_words(const att_gref_refine *f, const att_gelem *el, const att_grow *r) {
    const att_gref_call *c = f->c;
    size_t at = el->kind == 0 ? r->gr0 * c->a.Ev + el->col : r->qr * (size_t)c->a.E + el->col;
    return c->x[el->kind] + at * (size_t)c->wx[el->kind];
}

/* whether a pending element of group G has a nonzero input in row r */
static int group_inputs(const att_gref_refine *f, const ggroup *G, const att_grow *r) {
    for (size_t e = G->e0; e < G->e1; e++)
        if (f->status[e] && !mw_zero(input_words(f, &f->elems[e], r), f->c->wx[f->elems[e].kind])) return 1;
    return 0;
}

typedef struct {             /* a row's dP: whether every one is equal, and their sum */
    int ready, all;
    dyad sum;
} dpstat;

/* Whether row r's dS at element el's key (grad_key) is exactly 0 (row_ds_zero): every dP equal makes it 0, and n dP_j
   unequal to the sum of dP rules it out (were every group's mean dP one value m, the sum would be n m and dP_j m), so the
   groups of equal logits (st0, *made bit 2) are formed only where neither decides */
static int ds_zero(int *zero, att_grow *r, const att_gelem *el, dpstat *d, rstats *st0, unsigned *made) {
    *zero = 0;
    size_t pos = r->nan || !r->n ? 0 : r->pos[el->j];
    if (!pos) return 0;
    int rc = 0;
    if (!d->ready) {
        d->ready = d->all = 1;
        for (size_t l = 1; d->all && l < r->n; l++) d->all = !dyad_cmp(&r->dp[l], &r->dp[0]);
        rc = dyad_set_si(&d->sum, 0, 0);
        for (size_t l = 0; !rc && !d->all && l < r->n; l++) rc = dyad_add_exact(&d->sum, &d->sum, &r->dp[l]);
    }
    if (rc || d->all) {
        *zero = !rc;
        return rc;
    }
    dyad t;
    dyad_init(&t);
    if (!(rc = dyad_set_si(&t, (int64_t)r->n, 0)) && !(rc = dyad_mul_exact(&t, &t, &r->dp[pos - 1])) && !dyad_cmp(&t, &d->sum)) {
        if (!(*made & 2)) { /* in the row's own tile (the scales', or a wave's row tile): no other tile reads its order */
            grow_order(r);
            if (!(rc = row_stats(st0, r, 0))) *made |= 2;
        }
        if (!rc) rc = row_ds_zero(zero, r, el, st0);
    }
    dyad_free(&t);
    return rc;
}

/* y - x_m for a scale y = sigma X + Y and group G's x_m, from the exact differences of the parts (att_affine): its exact
   sign, and with t its enclosure at p bits, as wide as 2**-p of the difference, however large y and x_m are */
static int scale_diff_at(int *sign, ivl *t, const att_gref_refine *f, const dyad *X, const dyad *Y, const ggroup *G, const dyad *slo, const dyad *shi,
                         int64_t p) {
    dyad dX, dY;
    dyad_init(&dX);
    dyad_init(&dY);
    int rc = dyad_neg(&dX, &G->X);
    if (!rc) rc = dyad_add_exact(&dX, X, &dX);
    if (!rc) rc = dyad_neg(&dY, &G->Y);
    if (!rc) rc = dyad_add_exact(&dY, Y, &dY);
    if (!rc) rc = att_affine(sign, t ? &t->lo : NULL, t ? &t->hi : NULL, &f->c->a, &dX, f->c->a.mask_float ? &dY : NULL, slo, shi, p);
    dyad_free(&dX);
    dyad_free(&dY);
    return rc;
}

static int scale_diff_xy(int *sign, ivl *t, const att_gref_refine *f, const dyad *X, const dyad *Y, const ggroup *G, int64_t p) {
    return scale_diff_at(sign, t, f, X, Y, G, &f->sabs_lo, &f->sabs_hi, p);
}

/* key m's scale in row r: X = s_ref - s_m and Y = b_ref - b_m (0 without a float mask), so x_m = sigma X + Y */
static int key_scale(dyad *X, dyad *Y, const att_gref_refine *f, const att_grow *r, size_t m) {
    int rc = dyad_neg(X, &r->sd[m]);
    if (!rc) rc = dyad_add_exact(X, &r->sd[r->ref], X);
    if (!rc) rc = !f->c->a.mask_float ? dyad_set_si(Y, 0, 0) : (rc = dyad_neg(Y, &r->bias[m])) ? rc : dyad_add_exact(Y, &r->bias[r->ref], Y);
    return rc;
}

/* The scales' wave from active row wave0: at most WAVE_ROWS rows and WAVE_TERMS entries, one per pending grad_value or
   grad_key group of a row's spans, a row a tile */
static int scales_open(att_gref_refine *f, size_t *tiles) {
    att_gref_work *w = f->work;
    size_t held = 0;
    w->wave1 = w->wave0;
    while (w->wave1 < w->nactive && w->wave1 - w->wave0 < WAVE_ROWS) {
        size_t t = row_entries(f, w->active[w->wave1], 0, NULL);
        if (w->wave1 > w->wave0 && held + t > WAVE_TERMS) break;
        held += t;
        w->wave1++;
    }
    size_t rows = w->wave1 - w->wave0, *gs = malloc((w->ngroups + 1) * sizeof *gs);
    w->ent_at = malloc((rows + 1) * sizeof *w->ent_at);
    w->sc = malloc((held + 1) * sizeof *w->sc);
    int rc = !gs || !w->ent_at || !w->sc ? EXACTREF_NOMEM : 0;
    for (size_t k = 0, n = 0; !rc && k < rows; k++) {
        w->ent_at[k] = n;
        size_t m = row_entries(f, w->active[w->wave0 + k], 0, gs);
        for (size_t i = 0; i < m; i++, n++) {
            scent *x = &w->sc[n];
            x->g = gs[i];
            x->flags = 0;
            ivl_init(&x->y);
            dyad_init(&x->X);
            dyad_init(&x->Y);
        }
        w->nsc = n;
        w->ent_at[k + 1] = n;
    }
    free(gs);
    *tiles = rows;
    return rc;
}

/* the scales' row tile k: the row's exact quantities and, per entry, whether it keeps the key not far, or far with a
   nonzero term (a pending element's input nonzero, and for grad_key a dS not exactly 0) and then the key's scale; the row
   freed */
static int scales_tile(att_gref_refine *f, size_t k) {
    att_gref_work *w = f->work;
    att_grow *r = &f->rows[w->active[w->wave0 + k]];
    if (w->ent_at[k] == w->ent_at[k + 1]) return 0;
    const att_call *a = &f->c->a;
    att_row scratch;
    att_row_init(&scratch);
    rstats st0;
    rstats_init(&st0);
    dpstat d;
    d.ready = d.all = 0;
    dyad_init(&d.sum);
    dyad pd;
    dyad_init(&pd);
    unsigned made = 0;
    int rc = grow_ready(r, f, &scratch);
    const dyad *bias = a->mask_float ? r->bias : NULL; /* the row's, once made */
    if (!rc) rc = dyad_set_si(&pd, w->p, 0);
    if (!rc && !r->nan && r->n) rc = att_reference_key(&r->ref, a, r->sd, bias, r->n, r->top);
    size_t *at = rc || r->nan || !r->n ? NULL : calloc(r->n + 1, sizeof *at); /* per kept key: 1 + an entry holding its argument, or 0 */
    if (!rc && !r->nan && r->n && !at) rc = EXACTREF_NOMEM;
    for (size_t i = w->ent_at[k]; !rc && !r->nan && r->n && i < w->ent_at[k + 1]; i++) {
        scent *x = &w->sc[i];
        const ggroup *G = &w->groups[x->g];
        const att_gelem *el = &f->elems[G->e0];
        size_t pos = r->pos[el->j];
        if (!pos) continue;
        size_t m = pos - 1;
        int zero = 0;
        /* the key's argument, once a row (a key's grad_value and grad_key groups both read it) */
        if ((rc = at[m] ? ivl_copy(&x->y, &w->sc[at[m] - 1].y) : att_argument(&x->y.lo, &x->y.hi, a, r->sd, bias, r->ref, m, &f->sabs_lo, &f->sabs_hi, w->p))) break;
        at[m] = i + 1;
        if (dyad_cmp(&x->y.lo, &pd) <= 0) { /* kept, as grow_round splits it */
            x->flags = SC_KEPT;
            continue;
        }
        if (!group_inputs(f, G, r) || (el->kind == 1 && ((rc = ds_zero(&zero, r, el, &d, &st0, &made)) || zero))) continue;
        if (!(rc = key_scale(&x->X, &x->Y, f, r, m))) x->flags = SC_CAND;
    }
    free(at);
    grow_free(r);
    rstats_free(&st0);
    dyad_free(&d.sum);
    dyad_free(&pd);
    att_row_free(&scratch);
    return rc;
}

/* the first of entries a .. b - 1 (one row's, ascending groups) whose group is g or later */
static size_t sc_first(const att_gref_work *w, size_t a, size_t b, size_t g) {
    while (a < b) {
        size_t m = a + (b - a) / 2;
        if ((w->sc ? w->sc[m].g : w->fe[m].g) < g) a = m + 1;
        else b = m;
    }
    return a;
}

/* the scales' merge tile c: groups range[c] .. range[c + 1]), their rows' entries in row order: kept where a row keeps
   the key not far, and the least scale over the rows where it is far with a nonzero term (the earliest of equal ones),
   compared by the scales' enclosures, and exactly where they meet */
static int scales_merge_tile(att_gref_refine *f, size_t c) {
    att_gref_work *w = f->work;
    size_t g0 = w->range[c], g1 = w->range[c + 1];
    int rc = 0;
    for (size_t k = 0; !rc && k < w->wave1 - w->wave0; k++) {
        size_t a = sc_first(w, w->ent_at[k], w->ent_at[k + 1], g0), b = sc_first(w, a, w->ent_at[k + 1], g1);
        for (size_t i = a; !rc && i < b; i++) {
            scent *x = &w->sc[i];
            ggroup *G = &w->groups[x->g];
            if (x->flags & SC_KEPT) G->kept = 1;
            if (!(x->flags & SC_CAND)) continue;
            int take = !G->found, sgn = 0;
            if (!take && dyad_cmp(&x->y.hi, &G->xm.lo) < 0) take = 1;
            else if (!take && dyad_cmp(&x->y.lo, &G->xm.hi) <= 0 && !(rc = scale_diff_xy(&sgn, NULL, f, &x->X, &x->Y, G, w->p))) take = sgn < 0;
            if (!rc && take && !(rc = ivl_copy(&G->xm, &x->y)) && !(rc = dyad_copy(&G->X, &x->X))) rc = dyad_copy(&G->Y, &x->Y);
            G->found = G->found || take;
        }
    }
    return rc;
}

/* After the scales: the far groups (no row keeps the key not far), each far group's pending elements' sums cleared at
   their words, and the far fold's tiles: ranges of the far groups with a nonzero term of about equal element counts */
static int scales_done(att_gref_refine *f) {
    att_gref_work *w = f->work;
    const att_gref_call *c = f->c;
    size_t rmax = 1, count[2] = {0, 0}, total = 0;
    w->nfar = 0;
    for (size_t g = 0; g < w->ngroups; g++) {
        ggroup *G = &w->groups[g];
        const att_gelem *el = &f->elems[G->e0];
        if (!G->pending || el->kind == 2) continue;
        if (G->kept) { /* some row keeps the key not far: its scale is found by the folds */
            G->found = 0;
            continue;
        }
        w->gfar[g] = 1;
        if (!G->found) continue; /* every term exactly 0 */
        w->farg[w->nfar++] = g;
        if (el->count > rmax) rmax = el->count;
        for (size_t e = G->e0; e < G->e1; e++) count[el->kind] += f->status[e] != 0;
    }
    if (!w->nfar) return 0;
    int64_t bits = 64 - word_clz((uint64_t)rmax) + w->p + 64 + FAR_GUARD + FAR_SLACK + 2;
    w->kf = mw_words(w->p + 64 + FAR_GUARD + FAR_SLACK + 2);
    for (int z = 0; z < 2; z++) w->wacc[z] = mw_words(bits + 64 * (int64_t)c->wx[z]);
    w->acc = calloc(2 * (count[0] * (size_t)w->wacc[0] + count[1] * (size_t)w->wacc[1]) + 1, sizeof *w->acc);
    w->acc_at = malloc((f->count + 1) * sizeof *w->acc_at);
    if (!w->acc || !w->acc_at) return EXACTREF_NOMEM;
    size_t at = 0;
    for (size_t i = 0; i < w->nfar; i++) {
        const ggroup *G = &w->groups[w->farg[i]];
        for (size_t e = G->e0; e < G->e1; e++) {
            if (!f->status[e]) continue;
            w->acc_at[e] = at;
            at += 2 * (size_t)w->wacc[f->elems[e].kind];
            total++;
        }
    }
    /* the fold's ranges: closing where the running count of elements passes each multiple of total / SPLIT_TILES */
    size_t size = (total + SPLIT_TILES - 1) / SPLIT_TILES, seen = 0, next = size;
    w->nrange = 0;
    w->range[w->nrange++] = 0;
    for (size_t i = 0; i + 1 < w->nfar && w->nrange < SPLIT_TILES; i++) {
        const ggroup *G = &w->groups[w->farg[i]];
        for (size_t e = G->e0; e < G->e1; e++) seen += f->status[e] != 0;
        if (seen >= next) {
            w->range[w->nrange++] = i + 1;
            while (next <= seen) next += size;
        }
    }
    w->range[w->nrange] = w->nfar;
    return 0;
}

/* Row k's far factors (the row made at p): per far group with a nonzero term, where the row keeps its key (far, as the
   scales found) and has a nonzero term there (as the scales test it), Delta P' (grad_value) or Delta dS' (grad_key) at the
   key times e**(x_m - y), y >= x_m the key's scale here; else none. A key's groups with one least scale (its grad_value
   and grad_key columns, as a rule) share e**(x_m - y). st0 and *made as in term_take. */
static int far_take(att_gref_refine *f, att_grow *r, size_t k, rstats *st0, unsigned *made) {
    att_gref_work *w = f->work;
    int64_t p = w->p, q = p + 64;
    size_t i0 = w->ent_at[k], nf = w->ent_at[k + 1] - i0;
    ivl t, *frs = malloc((nf + 1) * sizeof *frs);
    ivl_init(&t);
    size_t *at = calloc(r->n + 1, sizeof *at); /* per kept key: 1 + the entry whose e**(x_m - y) frs holds, or 0 */
    int *sgns = malloc((nf + 1) * sizeof *sgns);
    for (size_t i = 0; frs && i < nf; i++) ivl_init(&frs[i]);
    dyad X, Y;
    dyad_init(&X);
    dyad_init(&Y);
    dpstat d;
    d.ready = d.all = 0;
    dyad_init(&d.sum);
    int rc = !frs || !at || !sgns ? EXACTREF_NOMEM : 0;
    for (size_t i = i0; !rc && i < w->ent_at[k + 1]; i++) {
        fent *x = &w->fe[i];
        const ggroup *G = &w->groups[x->g];
        const att_gelem *el = &f->elems[G->e0];
        size_t pos = r->nan || !r->n ? 0 : r->pos[el->j];
        x->nonzero = 0;
        if (!pos || !group_inputs(f, G, r)) continue;
        size_t m = pos - 1;
        if (!r->far[m] || !r->has_far) {
            rc = EXACTREF_INTERNAL;
            break;
        }
        const ivl *v = el->kind == 0 ? &r->dPr[m] : &r->dSr[m];
        int zero = 0, sgn = 0;
        /* a dS exactly 0 holds 0, so the test is needed only there */
        if (el->kind == 1 && dyad_sign(&v->lo) <= 0 && dyad_sign(&v->hi) >= 0 && ((rc = ds_zero(&zero, r, el, &d, st0, made)) || zero)) continue;
        size_t li = i - i0, prev = at[m];
        const ggroup *H = prev ? &w->groups[w->fe[i0 + prev - 1].g] : NULL;
        if (H && !dyad_cmp(&H->X, &G->X) && !dyad_cmp(&H->Y, &G->Y)) { /* the same least scale: the same factor */
            sgn = sgns[prev - 1];
            li = prev - 1;
        } else {
            /* y - x_m at q + 64 = p + 128 bits (sigma's ends there) and e**(x_m - y) at q, so that a far value's factor is not
               wider than its sum's own rounding however far y is from x_m (scale_diff_xy's at p would be |y - x_m| 2**-p
               wide) */
            if ((rc = key_scale(&X, &Y, f, r, m)) || (rc = scale_diff_at(&sgn, &t, f, &X, &Y, G, &f->sfar_lo, &f->sfar_hi, q + 64))) break;
            if (sgn < 0) { /* the least scale is the scales' */
                rc = EXACTREF_INTERNAL;
                break;
            }
            if (sgn && !(rc = dyad_neg(&frs[li].lo, &t.hi)) && !(rc = dyad_neg(&frs[li].hi, &t.lo)) && !(rc = ivl_copy(&t, &frs[li])))
                rc = ivl_exp_far(&frs[li], &t, q);
            sgns[li] = sgn;
            at[m] = li + 1;
        }
        if (!rc) rc = sgn ? ivl_mul(&x->v, v, &frs[li], q) : ivl_copy(&x->v, v);
        x->nonzero = !rc && (dyad_sign(&x->v.lo) || dyad_sign(&x->v.hi));
    }
    for (size_t i = 0; frs && i < nf; i++) ivl_free(&frs[i]);
    free(frs);
    free(at);
    free(sgns);
    ivl_free(&t);
    dyad_free(&X);
    dyad_free(&Y);
    dyad_free(&d.sum);
    return rc;
}

/* |v| < 2**t for the factor's larger end */
static int64_t factor_top(const ivl *v) {
    int64_t t = INT64_MIN;
    for (int z = 0; z < 2; z++) {
        const dyad *e = z ? &v->hi : &v->lo;
        if (dyad_sign(e) && top_bit(e) + 1 > t) t = top_bit(e) + 1;
    }
    return t;
}

/* the far fold's tile c: far groups farg[range[c]] .. farg[range[c + 1] - 1], each row's factor in the wave's row order
   put on the group's grid (its ends rounded outward) and its ends times each pending element's input added to the
   element's sums; the grid set by the first factor, and moved up, its sums shifted with outward rounding, by a factor
   more than FAR_SLACK bits above the one that set it */
static int far_fold_tile(att_gref_refine *f, size_t c) {
    att_gref_work *w = f->work;
    const att_gref_call *c0 = f->c;
    int64_t p = w->p;
    size_t g0 = w->farg[w->range[c]], g1 = w->range[c + 1] < w->nfar ? w->farg[w->range[c + 1]] : w->ngroups;
    int kf = w->kf, wa = w->wacc[0] > w->wacc[1] ? w->wacc[0] : w->wacc[1], wx = c0->wx[0] > c0->wx[1] ? c0->wx[0] : c0->wx[1];
    uint64_t *lo = malloc((size_t)(4 * kf + wa + kf + wx + 1) * sizeof *lo), *hi = lo ? lo + kf : NULL, *ml = lo ? lo + 2 * kf : NULL, *mh = lo ? lo + 3 * kf : NULL,
             *tmp = lo ? lo + 4 * kf : NULL;
    int rc = lo ? 0 : EXACTREF_NOMEM;
    for (size_t k = 0; !rc && k < w->wave1 - w->wave0; k++) {
        const att_grow *r = &f->rows[w->active[w->wave0 + k]];
        size_t a = sc_first(w, w->ent_at[k], w->ent_at[k + 1], g0), b = sc_first(w, a, w->ent_at[k + 1], g1);
        for (size_t i = a; !rc && i < b; i++) {
            const fent *x = &w->fe[i];
            if (!x->nonzero) continue;
            ggroup *G = &w->groups[x->g];
            int64_t t = factor_top(&x->v);
            if (!G->grid) {
                G->grid = 1;
                G->gtop = t;
                G->gf = t - (p + 64 + FAR_GUARD);
            } else if (t > G->gtop + FAR_SLACK) {
                int64_t nf = t - (p + 64 + FAR_GUARD);
                for (size_t e = G->e0; e < G->e1; e++) {
                    if (!f->status[e]) continue;
                    int n = w->wacc[f->elems[e].kind];
                    mw_shr(w->acc + w->acc_at[e], n, nf - G->gf, 0);
                    mw_shr(w->acc + w->acc_at[e] + n, n, nf - G->gf, 1);
                }
                G->gf = nf;
                G->gtop = t;
            }
            if ((rc = mw_from_dyad(lo, kf, &x->v.lo, G->gf, 0)) || (rc = mw_from_dyad(hi, kf, &x->v.hi, G->gf, 1))) break;
            int z = f->elems[G->e0].kind, one = c0->wx[z] == 1, sl = mw_negative(lo, (size_t)kf), sh = mw_negative(hi, (size_t)kf);
            if (one) { /* one-word inputs: the ends' magnitudes once, then a word product per column and end */
                memcpy(ml, lo, (size_t)kf * sizeof *ml);
                memcpy(mh, hi, (size_t)kf * sizeof *mh);
                if (sl) mw_neg(ml, kf);
                if (sh) mw_neg(mh, kf);
            }
            for (size_t e = G->e0; e < G->e1; e++) {
                if (!f->status[e]) continue;
                const att_gelem *el = &f->elems[e];
                const uint64_t *xv = input_words(f, el, r);
                if (mw_zero(xv, c0->wx[z])) continue;
                w->sums[e].found = 1;
                uint64_t *acc = w->acc + w->acc_at[e];
                int n = w->wacc[z];
                if (!one) {
                    mw_add_ends(acc, n, lo, hi, kf, xv, c0->wx[z], tmp);
                    continue;
                }
                /* [lo, hi] x: lo x and hi x for x > 0, hi x and lo x for x < 0 */
                int64_t xs = word_signed(xv[0]);
                uint64_t ax = xs < 0 ? (uint64_t)(-(xs + 1)) + 1 : (uint64_t)xs;
                mw_addmul1(acc, n, xs > 0 ? ml : mh, kf, ax, xs > 0 ? sl : !sh);
                mw_addmul1(acc + n, n, xs > 0 ? mh : ml, kf, ax, xs > 0 ? sh : !sl);
            }
        }
    }
    free(lo);
    return rc;
}

/* a far group's element's Delta': its sums at 2**(gf + ux) */
static int far_sums(ivl *s, const att_gref_refine *f, size_t e) {
    const att_gref_work *w = f->work;
    const ggroup *G = &w->groups[w->group_of[e]];
    int z = f->elems[e].kind, n = w->wacc[z];
    uint64_t *tmp = malloc((size_t)n * sizeof *tmp);
    int rc = tmp ? 0 : EXACTREF_NOMEM;
    for (int end = 0; !rc && end < 2; end++) {
        dyad *d = end ? &s->hi : &s->lo;
        rc = mw_to_dyad(d, w->acc + w->acc_at[e] + (size_t)end * (size_t)n, n, G->gf + f->c->ux[z], tmp);
    }
    free(tmp);
    return rc;
}

/* row r's grad_query terms among terms a .. b - 1: the pair test for all of its columns at once, over every key (the
   first round) and over the kept keys (the same where the row has no far key); the cols arrays hold b - a entries */
static int row_pair_tests(att_gref_refine *f, att_grow *r, size_t a, size_t b, rat **cols_term, int **cols_known, const att_gelem **cols_el) {
    att_gref_work *w = f->work;
    int rc = 0;
    for (int kept = 0; !rc && kept < 2; kept++) {
        size_t cols = 0;
        for (size_t i = a; i < b; i++) {
            gterm *x = &w->terms[i];
            if (f->elems[x->e].kind != 2 || !(kept ? x->need : x->need0)) continue;
            if (kept && x->need0 && !r->has_far) {
                x->known = x->known0;
                if ((rc = rat_copy(&x->term, x->term0))) break;
                continue;
            }
            cols_term[cols] = kept ? &x->term : x->term0;
            cols_known[cols] = kept ? &x->known : &x->known0;
            cols_el[cols++] = &f->elems[x->e];
        }
        if (!rc && cols) {
            grow_order(r);
            rc = pair_test_cols(cols_term, cols_known, cols, f, cols_el, r, kept);
        }
    }
    return rc;
}

/* a split wave's row tile k: the row made and enclosed at p, and the statistics its terms may read */
static int split_rows_tile(att_gref_refine *f, size_t k) {
    att_gref_work *w = f->work;
    att_grow *r = &f->rows[w->active[w->wave0 + k]];
    if (w->term_at[k] == w->term_at[k + 1] && !row_far_count(w, k)) return 0;
    att_row scratch;
    att_row_init(&scratch);
    size_t *need = malloc((w->term_at[k + 1] - w->term_at[k] + row_far_count(w, k) + 1) * sizeof *need);
    unsigned char *seen = NULL;
    int rc = need ? row_prepare(f, r, k, &scratch, need, &seen) : EXACTREF_NOMEM, kept = 0, dk = 0, gq = 0;
    /* st where a grad_value or grad_key term's key is kept and not far, st0 where a grad_key term may hold 0 (term_take) */
    for (size_t i = w->term_at[k]; !rc && !(kept && dk && gq) && i < w->term_at[k + 1]; i++) {
        const att_gelem *el = &f->elems[w->term_e[i]];
        size_t pos = el->kind == 2 || r->nan ? 0 : r->pos[el->j];
        kept = kept || (pos && !r->far[pos - 1]);
        dk = dk || el->kind == 1;
        gq = gq || el->kind == 2;
    }
    if (gq) w->has_st[k] |= 4;
    if (!rc && (kept || dk)) grow_order(r);
    if (!rc && kept && !(rc = row_stats(&w->st[k], r, 1))) w->has_st[k] |= 1;
    if (!rc && dk && !(rc = row_stats(&w->st0[k], r, 0))) w->has_st[k] |= 2;
    if (!rc && row_far_count(w, k)) {
        unsigned made = w->has_st[k];
        if (!(rc = far_take(f, r, k, &w->st0[k], &made))) w->has_st[k] |= (unsigned char)(made & 2);
    }
    free(need);
    free(seen);
    att_row_free(&scratch);
    return rc;
}

/* a split wave's term tile c: its terms (consecutive terms of one row) placed and taken */
static int split_terms_tile(att_gref_refine *f, size_t c) {
    att_gref_work *w = f->work;
    size_t k = w->part_row[c];
    att_grow *r = &f->rows[w->active[w->wave0 + k]];
    unsigned made = w->has_st[k];
    int rc = 0;
    for (size_t i = w->part_at[c]; !rc && i < w->part_at[c + 1]; i++) {
        gterm *x = &w->terms[i];
        term_init(x);
        x->e = w->term_e[i];
        w->live[i] = 1;
        rc = term_take(x, f, r, &w->st[k], &w->st0[k], &made);
    }
    return rc;
}

/* a split wave's pair-test tile k: the row's grad_query pair tests (where it has grad_query terms), then the row freed */
static int split_pairs_tile(att_gref_refine *f, size_t k) {
    att_gref_work *w = f->work;
    att_grow *r = &f->rows[w->active[w->wave0 + k]];
    size_t a = w->term_at[k], b = w->term_at[k + 1];
    if (!(w->has_st[k] & 4)) {
        grow_free(r);
        return 0;
    }
    rat **cols_term = malloc((b - a + 1) * sizeof *cols_term);
    int **cols_known = malloc((b - a + 1) * sizeof *cols_known);
    const att_gelem **cols_el = malloc((b - a + 1) * sizeof *cols_el);
    int rc = !cols_term || !cols_known || !cols_el ? EXACTREF_NOMEM : row_pair_tests(f, r, a, b, cols_term, cols_known, cols_el);
    free(cols_term);
    free(cols_known);
    free((void *)cols_el); /* an array of pointers to const elements, itself not const */
    grow_free(r);
    return rc;
}

static int split_fold_tile(att_gref_refine *f, size_t c);

/* tile t of the round's step: the scales' rows or their merge, a wave's far fold, or the wave's step: unsplit, each of its
   rows made, enclosed at p in every form, its terms and far factors taken, and freed */
int att_gref_refine_tile(att_gref_refine *f, size_t t) {
    att_gref_work *w = f->work;
    if (w->stage == 1) return scales_tile(f, t);
    if (w->stage == 2) return scales_merge_tile(f, t);
    if (w->farfold) return far_fold_tile(f, t);
    switch (w->phase) {
    case 1: return split_rows_tile(f, t);
    case 2: return split_terms_tile(f, t);
    case 3: return split_pairs_tile(f, t);
    case 4: return split_fold_tile(f, t);
    default: break;
    }
    att_row scratch;
    att_row_init(&scratch);
    rstats st, st0; /* the row's statistics over its kept keys and over every key */
    rstats_init(&st);
    rstats_init(&st0);
    size_t k0 = t * w->tile_rows, k1 = k0 + w->tile_rows < w->wave1 - w->wave0 ? k0 + w->tile_rows : w->wave1 - w->wave0, most = 0;
    for (size_t k = k0; k < k1; k++)
        if (w->term_at[k + 1] - w->term_at[k] + row_far_count(w, k) > most) most = w->term_at[k + 1] - w->term_at[k] + row_far_count(w, k);
    rat **cols_term = malloc((most + 1) * sizeof *cols_term);
    int **cols_known = malloc((most + 1) * sizeof *cols_known);
    const att_gelem **cols_el = malloc((most + 1) * sizeof *cols_el);
    size_t *need = malloc((most + 1) * sizeof *need);
    unsigned char *seen = NULL;
    int rc = !cols_term || !cols_known || !cols_el || !need ? EXACTREF_NOMEM : 0;
    for (size_t k = k0; !rc && k < k1; k++) {
        att_grow *r = &f->rows[w->active[w->wave0 + k]];
        unsigned made = 0;
        if (w->term_at[k] == w->term_at[k + 1] && !row_far_count(w, k)) continue;
        if ((rc = row_prepare(f, r, k, &scratch, need, &seen))) {
            grow_free(r);
            break;
        }
        for (size_t i = w->term_at[k]; !rc && i < w->term_at[k + 1]; i++) rc = term_take(&w->terms[i], f, r, &st, &st0, &made);
        if (!rc && row_far_count(w, k)) rc = far_take(f, r, k, &st0, &made);
        if (!rc) rc = row_pair_tests(f, r, w->term_at[k], w->term_at[k + 1], cols_term, cols_known, cols_el);
        grow_free(r);
    }
    free(cols_term);
    free(cols_known);
    free((void *)cols_el); /* an array of pointers to const elements, itself not const */
    free(need);
    free(seen);
    rstats_free(&st);
    rstats_free(&st0);
    att_row_free(&scratch);
    return rc;
}

/* s's ends of magnitude below 2**-(2**61) moved outward, to 0 or +-2**-(2**61): repeated rescaling keeps the exponents in
   range */
static int far_floor(ivl *s) {
    int64_t lim = (int64_t)1 << 61;
    int rc = 0;
    for (int z = 0; !rc && z < 2; z++) {
        dyad *d = z ? &s->hi : &s->lo;
        if (!dyad_sign(d) || d->e + big_bits(&d->m) >= -lim) continue;
        int outward = z ? dyad_sign(d) > 0 : dyad_sign(d) < 0; /* zero is inward for this end */
        rc = outward ? dyad_set_si(d, z ? 1 : -1, -lim) : dyad_set_si(d, 0, 0);
    }
    return rc;
}

/* y - x_m for a term's row scale y and its group's x_m (scale_diff_xy) */
static int scale_diff(int *sign, ivl *t, const att_gref_refine *f, const gterm *x, const ggroup *G, int64_t p) {
    return scale_diff_xy(sign, t, f, &x->X, &x->Y, G, p);
}

/* the wave's terms into the sums, in row order; a group's far terms at the least scale so far, the earlier sums
   rescaled when a row's is below it (every factor at most 1) */
/* terms i .. i1 - 1, one row's terms of one group, added to their elements' sums, the far ones at the group's scale; t
   and fr are scratch */
static int fold_block(att_gref_refine *f, size_t i, size_t i1, ivl *t, ivl *fr) {
    att_gref_work *w = f->work;
    int64_t p = w->p, q = p + 64;
    ggroup *G = &w->groups[w->group_of[w->terms[i].e]];
    const gterm *ys = NULL; /* the row's scale (one for the group's terms) */
    int rc = 0;
    for (size_t k = i; k < i1; k++) {
        gterm *x = &w->terms[k];
        gsum *s = &w->sums[x->e];
        if ((rc = ivl_add(&s->plain, &s->plain, &x->plain, q)) || (rc = ivl_add(&s->ds, &s->ds, &x->ds, q))) break;
        s->any_far |= x->has_far;
        if (x->need && s->known) {
            if (!x->known) s->known = 0;
            else if ((rc = rat_add(&s->exact, &s->exact, &x->term))) break;
        }
        if (x->need0 && s->known0) {
            if (!x->known0) s->known0 = 0;
            else if ((rc = rat_add(&s->exact0, &s->exact0, x->term0))) break;
        }
        if (x->has_u && !ys) ys = x;
    }
    if (!rc && ys) {
        int rescale = 0, scaled = 0, sgn = 0, take = !G->found;
        if (G->found && !(rc = scale_diff(&sgn, t, f, ys, G, p))) {
            if (sgn < 0) { /* a lower scale: the sums so far times e**(y - x_m) */
                if (!(rc = ivl_exp_far(fr, t, p))) rescale = take = 1;
            } else if (sgn > 0 && !(rc = dyad_neg(&fr->lo, &t->hi)) && !(rc = dyad_neg(&fr->hi, &t->lo)) && !(rc = ivl_copy(t, fr))) {
                if (!(rc = ivl_exp_far(fr, t, p))) scaled = 1; /* this row's terms times e**(x_m - y) */
            }
        }
        if (!rc && take && !(rc = ivl_copy(&G->xm, &ys->y)) && !(rc = dyad_copy(&G->X, &ys->X))) rc = dyad_copy(&G->Y, &ys->Y);
        G->found = 1;
        for (size_t e = G->e0; !rc && rescale && e < G->e1; e++)
            if (f->status[e] && !(rc = ivl_mul(&w->sums[e].far, &w->sums[e].far, fr, q))) rc = far_floor(&w->sums[e].far);
        for (size_t k = i; !rc && k < i1; k++) {
            gterm *x = &w->terms[k];
            if (!x->has_u) continue;
            gsum *s = &w->sums[x->e];
            s->found = 1;
            if (scaled && (rc = ivl_mul(&x->u, &x->u, fr, q))) break;
            rc = ivl_add(&s->far, &s->far, &x->u, q);
        }
    }
    return rc;
}

static int wave_fold(att_gref_refine *f) {
    att_gref_work *w = f->work;
    ivl t, fr;
    ivl_init(&t);
    ivl_init(&fr);
    int rc = 0;
    for (size_t i = 0; !rc && i < w->nterms;) {
        size_t g = w->group_of[w->terms[i].e], i1 = i;
        while (i1 < w->nterms && w->group_of[w->terms[i1].e] == g && (i1 == i || w->terms[i1].e > w->terms[i1 - 1].e)) i1++;
        rc = fold_block(f, i, i1, &t, &fr);
        i = i1;
    }
    ivl_free(&t);
    ivl_free(&fr);
    terms_free(w);
    return rc;
}

/* the first of terms a .. b - 1 (one row's, in ascending group order) whose group is g or later */
static size_t first_of_group(const att_gref_work *w, size_t a, size_t b, size_t g) {
    while (a < b) {
        size_t m = a + (b - a) / 2;
        if (w->group_of[w->terms[m].e] < g) a = m + 1;
        else b = m;
    }
    return a;
}

/* a split wave's fold tile c: the terms of groups fold_g[c] .. fold_g[c + 1]) added to their sums row by row in the
   wave's order, as wave_fold adds them, then freed */
static int split_fold_tile(att_gref_refine *f, size_t c) {
    att_gref_work *w = f->work;
    size_t g0 = w->fold_g[c], g1 = w->fold_g[c + 1];
    ivl t, fr;
    ivl_init(&t);
    ivl_init(&fr);
    int rc = 0;
    for (size_t k = 0; !rc && k < w->wave1 - w->wave0; k++) {
        size_t a = first_of_group(w, w->term_at[k], w->term_at[k + 1], g0), b = first_of_group(w, a, w->term_at[k + 1], g1);
        for (size_t i = a; !rc && i < b;) {
            size_t g = w->group_of[w->terms[i].e], i1 = i + 1;
            while (i1 < b && w->group_of[w->terms[i1].e] == g) i1++;
            rc = fold_block(f, i, i1, &t, &fr);
            i = i1;
        }
        for (size_t i = a; !rc && i < b; i++) {
            term_free(&w->terms[i]);
            w->live[i] = 0;
        }
    }
    ivl_free(&t);
    ivl_free(&fr);
    return rc;
}

/* times sigma: rational when sigma is, or when the sum is 0 (grad_key, grad_query) */
static int exact_sigma(rat *val, int *known, const att_gref_refine *f, const att_gelem *el) {
    const att_call *a = &f->c->a;
    if (!*known || !el->kind) return 0;
    if (!a->den.n) {
        if (!rat_is_zero(val)) *known = 0;
        return 0;
    }
    rat s;
    rat_init(&s);
    int rc = rat_set(&s, &a->num, &a->den);
    if (!rc) rc = rat_mul(val, val, &s);
    rat_free(&s);
    return rc;
}

/* the round's decisions, in order: the plain sums, the first round's pair test, the cancellation-free sums, the far
   keys' split (a far group's element by its far sums alone: G^K = 0, its key never kept and not far) */
static int round_decide(att_gref_refine *f) {
    att_gref_work *w = f->work;
    int64_t p = w->p;
    int last = p >= w->cap, rc = 0;
    ivl s;
    ivl_init(&s);
    bigint bits;
    big_init(&bits);
    for (int form = 0; !rc && form < 2; form++) {
        for (size_t e = 0; !rc && e < f->count; e++) {
            if (!f->status[e] || w->gfar[w->group_of[e]]) continue;
            const att_gelem *el = &f->elems[e];
            int none = 1;
            if (!(rc = ivl_copy(&s, form ? &w->sums[e].ds : &w->sums[e].plain)) && !(rc = scale_sigma(&s, f, el, &f->slo, &f->shi, p)))
                rc = decide_rounding(&bits, &f->entries[e], &none, &f->c->out, f->c->rnd, f->c->ovf, &s.lo, &s.hi, last);
            if (!rc && !none) put_bits(f, e, &bits);
        }
        for (size_t e = 0; !rc && !form && w->first && e < f->count; e++) { /* the pair test */
            if (!f->status[e] || f->elems[e].kind != 2) continue;
            int known = w->sums[e].known0;
            if (!(rc = exact_sigma(&w->sums[e].exact0, &known, f, &f->elems[e])) && known) rc = put_rational(f, e, &w->sums[e].exact0);
        }
    }
    for (size_t e = 0; !rc && e < f->count; e++) {
        gsum *x = &w->sums[e];
        int far = w->gfar[w->group_of[e]];
        if (!f->status[e] || (!far && !x->any_far)) continue;
        const ggroup *G = &w->groups[w->group_of[e]];
        int known = far || x->known;
        if ((rc = exact_sigma(&x->exact, &known, f, &f->elems[e])) || !known) continue;
        if (!x->found) { /* every far term exactly 0: the value is G^K */
            rc = put_rational(f, e, &x->exact);
            continue;
        }
        if (!(rc = far ? far_sums(&s, f, e) : ivl_copy(&s, &x->far))) rc = far_decide(f, e, &x->exact, &s, &G->xm, &G->X, &G->Y, p, last);
    }
    ivl_free(&s);
    big_free(&bits);
    return rc;
}

/* The tiles of the next wave (0 when the refinement is done): rounds from p0 bits, doubling to the cap; what is still
   undecided at the cap keeps status 1 */
int att_gref_refine_next(att_gref_refine *f, size_t *tiles) {
    att_gref_work *w = f->work;
    *tiles = 0;
    int rc = 0;
    for (;;) {
        if (w->out) { /* the wave's tiles have run */
            w->out = 0;
            if (w->stage == 1) { /* the scales' rows: their merge */
                w->stage = 2;
                *tiles = w->nrange;
                w->out = 1;
                return 0;
            }
            if (w->stage == 2) {
                entries_free(w);
                w->stage = 1;
            } else {
                if (w->phase && w->phase < 4) { /* a split wave's next step */
                    w->phase++;
                    *tiles = w->phase == 2 ? w->nparts : w->phase == 3 ? w->wave1 - w->wave0 : w->nfolds;
                    w->out = 1;
                    return 0;
                }
                if (!w->farfold && w->nfe) { /* its far factors into the far groups' sums */
                    w->farfold = 1;
                    *tiles = w->nrange;
                    w->out = 1;
                    return 0;
                }
                w->farfold = 0;
                entries_free(w);
                if (w->phase) terms_free(w); /* its fold tiles added and freed the terms */
                else if ((rc = wave_fold(f))) return rc;
            }
            w->wave0 = w->wave1;
        }
        if (!w->open) {
            if (w->done || !any_pending(f)) return 0;
            if ((rc = round_open(f))) return rc;
        }
        if (w->stage == 1) { /* the scales, in waves, before the terms */
            if (w->wave0 < w->nactive) {
                if ((rc = scales_open(f, tiles))) return rc;
                w->out = 1;
                return 0;
            }
            if ((rc = scales_done(f))) return rc;
            w->stage = 0;
            w->wave0 = 0;
        }
        if (w->wave0 < w->nactive) {
            if ((rc = wave_open(f, tiles))) return rc;
            w->out = 1;
            return 0;
        }
        if ((rc = round_decide(f))) return rc;
        w->open = 0;
        w->first = 0;
        if (w->p >= w->cap) {
            w->done = 1;
            return 0;
        }
        w->p = 2 * w->p < w->cap ? 2 * w->p : w->cap;
    }
}

void att_gref_refine_init(att_gref_refine *f) {
    memset(f, 0, sizeof *f);
    dyad *ds[] = {&f->sabs_lo, &f->sabs_hi, &f->slo, &f->shi, &f->sfar_lo, &f->sfar_hi};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
}

/* the run's waves so far, and those that ran split (a binding for tests: which path a case takes) */
void att_gref_refine_waves(const att_gref_refine *f, size_t *waves, size_t *split) {
    *waves = f->work ? f->work->waves : 0;
    *split = f->work ? f->work->split_waves : 0;
}

void att_gref_refine_free(att_gref_refine *f) {
    for (size_t r = 0; f->rows && r < f->nrows; r++) grow_free(&f->rows[r]);
    work_free(f->work, f->count);
    free(f->rows);
    free(f->elems);
    free(f->rowlist);
    dyad *ds[] = {&f->sabs_lo, &f->sabs_hi, &f->slo, &f->shi, &f->sfar_lo, &f->sfar_hi};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    memset(f, 0, sizeof *f);
}
