/* The attention backward's tolerance (proofs/scaled_dot_product_attention_backward.md, The tolerance): per row the
   probabilities P' (the closed form where its validity conditions hold, else the step-by-step program), dP', D' and dS',
   intervals of dyadics each narrowed to the values of the format its last rounding gives (Values of a format), the
   products in them exact; then the three gradient sums over the rows and keys, and per element the bound through its
   rounding, the scale and Lemma 3. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

#define EXACT_BITS (INT64_C(1) << 20) /* a precision at which the dyadics' sums and products here are exact */

void att_gtol_init(att_gtol *g) {
    memset(g, 0, sizeof *g);
    att_tol_init(&g->t);
    att_prog_init(&g->g);
    for (int z = 0; z < ATT_GL_COUNT; z++) lean_init(&g->lean[z]);
    dyad_init(&g->scale_lo);
    dyad_init(&g->scale_hi);
    for (int z = 0; z < 3; z++) att_carry_init(&g->carry[z]);
}

static void acc_free(att_acc *c) {
    if (c->terms)
        for (size_t z = 0; z < 3 * (c->nmax + 1); z++) dyad_free(&c->terms[z]);
    free(c->terms);
    free(c->has);
    memset(c, 0, sizeof *c);
}

void att_gtol_free(att_gtol *g) {
    att_tol_free(&g->t);
    att_prog_free(&g->g);
    for (int z = 0; z < ATT_GL_COUNT; z++) lean_free(&g->lean[z]);
    for (int z = 0; z < ATT_GA_COUNT; z++) acc_free(&g->acc[z]);
    if (g->normal) sum_bounds_free(&g->dp);
    dyad_free(&g->scale_lo);
    dyad_free(&g->scale_hi);
    for (int z = 0; z < 3; z++) att_carry_free(&g->carry[z]);
}

/* The backward's constants from its settings (attention_tol.c's att_settings with backward set): the closed form's
   (gr->closed where it covers the call), the program's steps 1 to 7, Lemma 1 at the backward's own boundaries, its
   accumulations' bounds for every count of terms they take (Ev for dP, up to S for D, the gradients' counts nq and nkv,
   and the heads a key and value block serves for grouped_heads' sums),
   and under the normal standard Lemma S's constants for dP and the gradient sums; *bad as att_tol_build's. a holds the
   call's scale and E (att_tol_prepare). Without closed, every row takes the program. */
int att_gtol_build(att_gtol *gr, int *bad, const att_settings *s, const att_call *a, int normal, int closed, int sign, int64_t Ev, int64_t nq,
                   int64_t nkv, int64_t heads) {
    int64_t w = s->w, S = s->S;
    int flush = s->flush, rc = 0, applicable = 0;
    gr->normal = normal;
    gr->sums_normal = normal && !s->query_block; /* backward_blocks: the blocks' sums in any order under both standards */
    gr->query_block = s->query_block;
    gr->key_block = s->key_block;
    gr->w = w;
    gr->flush = flush;
    gr->emin = s->emin;
    gr->S = S;
    gr->Ev = Ev;
    gr->nq = nq;
    gr->nkv = nkv;
    gr->methods = (s->exp2 ? ATT_GM_EXP2 : 0) | (s->reciprocal ? ATT_GM_RECIPROCAL : 0) | (s->has_weights ? ATT_GM_WEIGHTS : 0) |
                  (s->rounded_p ? ATT_GM_ROUNDED_P : 0) | (s->scale_ds ? ATT_GM_SCALE_DS : 0) | (s->grouped ? ATT_GM_GROUPED : 0) |
                  (s->query_block ? ATT_GM_BLOCKED : 0) | (s->has_mma ? ATT_GM_MMA : 0);
    if (closed && (rc = att_tol_build(&gr->t, &applicable, bad, s, normal))) return rc;
    gr->closed = closed && applicable;
    /* Lemma D's tables hold the weights to 2**-ATT_F absolutely: where the probabilities' format has values below
       2**-(ATT_F - 16), the closed form takes each weight from Lemma X instead, to its relative precision */
    gr->tables = s->r[ATT_SR_PROBS].active && s->r[ATT_SR_PROBS].f.emin - (int64_t)s->r[ATT_SR_PROBS].f.man_bits >= -(ATT_F - 16);
    if (gr->closed && (rc = att_tol_prepare(&gr->t, a, &s->a[ATT_SC_SCORES].f, w))) return rc;
    if ((rc = att_prog_build(&gr->g, &applicable, bad, s, sign))) return rc;
    if (applicable && !gr->g.unbounded && (rc = att_prog_prepare(&gr->g, &s->a[ATT_SC_SCORES].f, s->E))) return rc;
    int leans[ATT_GL_COUNT] = {ATT_SR_PROBS,   ATT_SR_RECIPROCAL, ATT_SR_DP_OUT,  ATT_SR_DIFF,    ATT_SR_DS,      ATT_SR_SCALE_GRADS, ATT_SR_SCALE_DS,
                               ATT_SR_WEIGHTS, ATT_SR_EXP2_LSE,   ATT_SR_SHIFT,   ATT_SR_EXP,     ATT_SR_OUT_Q,   ATT_SR_OUT_K,       ATT_SR_OUT_V,
                               ATT_SR_GROUP_K, ATT_SR_GROUP_V};
    for (int z = 0; !rc && z < ATT_GL_COUNT; z++) rc = lean_of(&gr->lean[z], &s->r[leans[z]], flush);
    struct {
        int slot, setting;
        int64_t last;
    } accs[] = {{ATT_GA_DP, ATT_SC_DP, Ev},       {ATT_GA_ROW_DOT, ATT_SC_ROW_DOT, S > Ev ? S : Ev}, {ATT_GA_Q, ATT_SC_Q, nq}, {ATT_GA_K, ATT_SC_K, nkv},
                {ATT_GA_V, ATT_SC_V, nkv},        {ATT_GA_GROUP_K, ATT_SC_GROUP_K, heads}, {ATT_GA_GROUP_V, ATT_SC_GROUP_V, heads}};
    for (size_t z = 0; !rc && z < sizeof accs / sizeof *accs; z++) {
        const att_acc_set *x = &s->a[accs[z].setting];
        int64_t last = accs[z].last;
        if (x->kind != ATT_SA_NONE) rc = acc_of(&gr->acc[accs[z].slot], x, last ? 1 : 0, last, w, flush);
    }
    gr->row_dot_f = s->a[ATT_SC_ROW_DOT].f;
    gr->q_f = s->a[ATT_SC_Q].f;
    gr->k_f = s->a[ATT_SC_K].f;
    gr->v_f = s->a[ATT_SC_V].f;
    gr->row_dot_active = s->a[ATT_SC_ROW_DOT].kind != ATT_SA_NONE;
    if (!rc && normal) {
        /* Lemma S's constants (sums.c), the accumulators rounding to nearest (foundations.md, The normal standard, Units) */
        const att_acc_set *dp = &s->a[ATT_SC_DP], *g3[3] = {&s->a[ATT_SC_Q], &s->a[ATT_SC_K], &s->a[ATT_SC_V]};
        if (!(rc = acc_units(NULL, NULL, NULL, NULL, bad, &dp->f, dp->rnd, flush))) rc = sum_bounds_init(&gr->dp, &dp->f, flush, Ev, Ev > 0 ? Ev - 1 : 0, 0, w);
        for (int z = 0; !rc && z < 3; z++) rc = acc_units(NULL, NULL, NULL, NULL, bad, &g3[z]->f, g3[z]->rnd, flush);
        const att_acc_set *rd = &s->a[ATT_SC_ROW_DOT];
        if (!rc) rc = acc_units(NULL, NULL, NULL, NULL, bad, &rd->f, rd->rnd, flush);
        gr->row_dot_rnd = rd->rnd;
    }
    gr->scale_bounded = s->scale_bounded;
    if (!rc && s->scale_bounded && !(rc = dyad_copy(&gr->scale_lo, &s->scale_lo))) rc = dyad_copy(&gr->scale_hi, &s->scale_hi);
    if (!rc && s->query_block) {
        /* backward_blocks: Lemma C's constants per gradient, each block's sums carried on its accumulator in any order, for a
           block's terms + 2 */
        int sr[3] = {ATT_SR_CARRY_Q, ATT_SR_CARRY_K, ATT_SR_CARRY_V}, sc[3] = {ATT_SC_CARRY_Q, ATT_SC_CARRY_K, ATT_SC_CARRY_V};
        int64_t last[3] = {s->key_block + 2, s->query_block + 2, s->query_block + 2};
        for (int z = 0; !rc && z < 3; z++)
            rc = att_carry_build(&gr->carry[z], &s->r[sr[z]], &s->a[sc[z]], last[z], s->scale_bounded, &s->scale_lo, &s->scale_hi, flush, w);
    }
    return rc;
}

void att_gtol_out_init(att_gtol_out *o) {
    memset(o, 0, sizeof *o);
    att_soft_init(&o->sf);
}

void att_gtol_out_free(att_gtol_out *o) {
    ivl *arrays[] = {o->P, o->DP, o->DS};
    for (int z = 0; z < 3; z++) {
        if (!arrays[z]) continue;
        for (size_t j = 0; j < o->cap; j++) ivl_free(&arrays[z][j]);
        free(arrays[z]);
    }
    ivl *row[] = {o->L, o->B, o->Y, o->Wt};
    for (int z = 0; z < 4; z++) {
        if (!row[z]) continue;
        for (size_t j = 0; j < o->cap; j++) ivl_free(&row[z][j]);
        free(row[z]);
    }
    free(o->V);
    att_soft_free(&o->sf);
    memset(o, 0, sizeof *o);
}

/* o's per-key arrays for rows of S keys */
static int out_reserve(att_gtol_out *o, size_t S) {
    if (o->cap >= S && o->P && o->V) return 0;
    size_t cap = S ? S : 1;
    ivl **arrays[] = {&o->P, &o->DP, &o->DS, &o->L, &o->B, &o->Y, &o->Wt};
    for (int z = 0; z < 7; z++) {
        ivl *x = *arrays[z];
        if (x) {
            for (size_t j = 0; j < o->cap; j++) ivl_free(&x[j]);
            free(x);
        }
        if (!(*arrays[z] = malloc(cap * sizeof **arrays[z]))) {
            for (int y = 0; y <= z; y++) *arrays[y] = NULL;
            o->cap = 0;
            return EXACTREF_NOMEM;
        }
    }
    for (int z = 0; z < 7; z++)
        for (size_t j = 0; j < cap; j++) ivl_init(&(*arrays[z])[j]);
    o->cap = cap;
    free(o->V);
    /* V, then the marks and the masked keys' overflow, per key */
    if (!(o->V = malloc(3 * cap))) return EXACTREF_NOMEM;
    o->mark = o->V + cap;
    o->mnan = o->mark + cap;
    return 0;
}

static int set_zero(ivl *x) {
    int rc = dyad_set_si(&x->lo, 0, 0);
    return rc ? rc : dyad_set_si(&x->hi, 0, 0);
}

/* x narrowed to the values of the format a boundary rounds to (Values of a format), where that leaves it nonempty */
static int narrow(ivl *x, int64_t emin, int64_t mantissa) {
    dyad a, b;
    dyad_init(&a);
    dyad_init(&b);
    int rc = dyad_copy(&a, &x->lo);
    if (!rc) rc = dyad_copy(&b, &x->hi);
    if (!rc) rc = grid_end(&a, emin, mantissa, 1);
    if (!rc) rc = grid_end(&b, emin, mantissa, 0);
    if (!rc && dyad_cmp(&a, &b) <= 0 && !(rc = dyad_copy(&x->lo, &a))) rc = dyad_copy(&x->hi, &b);
    dyad_free(&a);
    dyad_free(&b);
    return rc;
}

/* the exact interval product */
static int mul_exact(ivl *r, const ivl *x, const ivl *y) { return ivl_mul(r, x, y, EXACT_BITS); }

/* The closed form's P' for each kept key (scaled_dot_product_attention_backward.md, The probability step in closed form):
   from the row's quantities in sf, Y_hi = (P exp(R) + H) / (D (1 - E_Z)) and Y_lo = P exp(-R) / ((1 + tau)(1 + E_Z)) - H /
   (D (1 - E_Z)) (without the first part for a key that may vanish), with reciprocal widened by wh d_r, then the rounding
   to the probabilities' format below its largest value; *ok 0 where that fails. */
static int closed_probabilities(ivl *P, int *ok, const att_soft *sf, const att_tol *t, const size_t *keys, size_t n) {
    int64_t p = t->p;
    *ok = 0;
    dyad high, low, one, d1, d2, lo, hi, err;
    dyad *ds[] = {&high, &low, &one, &d1, &d2, &lo, &hi, &err};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    /* high = D.lo (1 - E_Z) down, low = (1 + tau.hi)(1 + E_Z) up */
    int rc = dyad_set_si(&one, 1, 0), fail = 0;
    if (!rc && !(rc = sub_dir(&d1, &one, &sf->EZ, p, 0)) && !(rc = dyad_mul(&high, &sf->D.lo, &d1, p, 0)) && !(rc = dyad_add(&d1, &one, &sf->tau.hi, p, 1)) &&
        !(rc = dyad_add(&d2, &one, &sf->EZ, p, 1)))
        rc = dyad_mul(&low, &d1, &d2, p, 1);
    if (!rc && dyad_sign(&high) <= 0) fail = 1;
    int reciprocal = t->methods & ATT_T_RECIPROCAL;
    for (size_t z = 0; !rc && !fail && z < n; z++) {
        /* d1 = H.hi / high */
        if ((rc = dyad_div(&d1, &sf->Hs[z].hi, &high, p, 1))) break;
        if (sf->vanish[z]) {
            rc = dyad_neg(&lo, &d1);
        } else if (!(rc = dyad_div(&d2, &sf->pe_lo[z], &low, p, 0))) {
            rc = sub_dir(&lo, &d2, &d1, p, 0);
        }
        if (!rc && !(rc = dyad_add(&hi, &sf->pe_hi[z], &sf->Hs[z].hi, p, 1))) rc = dyad_div(&hi, &hi, &high, p, 1);
        if (!rc && reciprocal && !(rc = dyad_mul(&d1, &sf->wh[z], &sf->dr, p, 1)) && !(rc = sub_dir(&lo, &lo, &d1, p, 0))) rc = dyad_add(&hi, &hi, &d1, p, 1);
        /* the rounding to the probabilities' format, below its largest value */
        if (!rc && !(rc = boundary_error(&err, &t->rnd[ATT_R_PROBS], &hi, &lo)) && !(rc = dyad_add(&d1, &hi, &err, p, 1))) {
            if (dyad_cmp(&d1, &t->c[ATT_C_MAX_PROBS]) > 0) {
                fail = 1;
                break;
            }
            if (!(rc = dyad_copy(&P[keys[z]].hi, &d1)) && !(rc = sub_dir(&d2, &lo, &err, p, 0)))
                rc = dyad_sign(&d2) < 0 ? dyad_set_si(&P[keys[z]].lo, 0, 0) : dyad_copy(&P[keys[z]].lo, &d2);
        }
    }
    if (!rc && !fail) *ok = 1;
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* The closed form's P' from a given logsumexp lse (attention_algorithms.md, The backward of fused kernels, Closed form
   for P): per kept key, from its logit x and bound L (sf, steps 1 to 3), the shift l - lse within R' = L + k + its
   rounding, k = |lse| + rho when a flush may read lse as zero (flushed), else rho, the stored logsumexp's radius; with
   exp2 in base-2 units, lse times log2(e) within ulps; then exp's ends and its rounding, below the largest values; *ok 0
   where a check fails. */
static int closed_given(ivl *P, int *ok, const att_soft *sf, const att_tol *t, const size_t *keys, size_t n, const dyad *lse, const dyad *rho, int flushed,
                        const att_call *a, const w128 *scores) {
    int64_t p = t->p;
    int exp2 = t->methods & ATT_T_EXP2, rc = 0, fail = 0;
    /* exp's ends from Lemma D's tables (table_exp) where the logits are sigma times exact scores */
    int tabled = a->tables && !a->mask_float && !a->uncertain && !t->flush && a->sign && n;
    *ok = 0;
    ivl xg, d, R, ec;
    ivl_init(&xg);
    ivl_init(&d);
    ivl_init(&R);
    ivl_init(&ec);
    dyad k, mag, shift, low, err, hi, lo, d1, sl, sh, c;
    dyad *ds[] = {&k, &mag, &shift, &low, &err, &hi, &lo, &d1, &sl, &sh, &c};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    w128 topv = tabled ? table_top(scores, n, a->sign) : w128_zero();
    if (tabled && !(rc = abs_sigma(&sl, &sh, a, p + 128))) rc = table_row_constant(&c, &ec, topv, a, &sl, lse, p + 64);
    /* k, and lse (times log2(e) with exp2) */
    if (!(rc = dyad_abs(&mag, lse)) && !(rc = dyad_add(&mag, &mag, rho, p, 1))) rc = flushed ? dyad_copy(&k, &mag) : dyad_copy(&k, rho);
    if (!rc) rc = ivl_point(&xg, lse);
    if (!rc && exp2) {
        ivl l2e;
        l2e.lo = t->log2e_lo; /* read-only views */
        l2e.hi = t->log2e_hi;
        /* held = (|lse| + rho) log2(e), within exp2_logsumexp's ulps */
        if (!(rc = dyad_mul(&d1, &mag, &t->log2e_hi, p, 1)) && !(rc = boundary_error(&err, &t->rnd[ATT_R_EXP2_LSE], &d1, &d1)) &&
            !(rc = dyad_add(&hi, &d1, &err, p, 1))) {
            if (dyad_cmp(&hi, &t->c[ATT_C_MAX_EXP2_LOGSUMEXP]) > 0) fail = 1;
            else if (!(rc = dyad_mul(&k, &k, &t->log2e_hi, p, 1)) && !(rc = dyad_add(&k, &k, &err, p, 1))) rc = ivl_mul(&xg, &xg, &l2e, p);
        }
    }
    for (size_t z = 0; !rc && !fail && z < n; z++) {
        /* the shift's bound: |x - lse| + L + k, and its rounding */
        if ((rc = ivl_sub(&d, &sf->xs[z], &xg, p)) || (rc = ivl_abs(&R, &d)) || (rc = dyad_add(&shift, &R.hi, &sf->Ls[z].hi, p, 1)) ||
            (rc = dyad_add(&shift, &shift, &k, p, 1)) || (rc = sub_dir(&low, &R.lo, &sf->Ls[z].hi, p, 0)) || (rc = sub_dir(&low, &low, &k, p, 0)) ||
            (rc = boundary_error(&err, &t->rnd[ATT_R_SHIFT], &shift, &low)) || (rc = dyad_add(&d1, &shift, &err, p, 1)))
            break;
        if (dyad_cmp(&d1, &t->c[ATT_C_MAX_SHIFT]) > 0) {
            fail = 1;
            break;
        }
        /* R' = L + k + that rounding; exp's ends (natural units) */
        if ((rc = dyad_add(&R.hi, &sf->Ls[z].hi, &k, p, 1)) || (rc = dyad_add(&R.hi, &R.hi, &err, p, 1))) break;
        if (exp2) {
            ivl ln2;
            ln2.lo = t->ln2_lo;
            ln2.hi = t->ln2_hi;
            if ((rc = ivl_mul(&d, &d, &ln2, p)) || (rc = dyad_mul(&R.hi, &R.hi, &t->ln2_hi, p, 1))) break;
        }
        if ((rc = dyad_add(&d1, &d.hi, &R.hi, p, 1))) break;
        if (exp_overflows(&d1)) {
            fail = 1;
            break;
        }
        int dh = 0, dl = 0;
        w128 D = tabled ? table_distance(topv, scores[z], a->sign) : w128_zero();
        if ((tabled && (rc = table_exp(&hi, &dh, &d1, D, a, &sh, &c, &ec.hi, p, 1))) || (!dh && (rc = exp_clamped(&hi, &d1, p, 1))) ||
            (rc = sub_dir(&d1, &d.lo, &R.hi, p, 0)) || (tabled && (rc = table_exp(&lo, &dl, &d1, D, a, &sl, &c, &ec.lo, p, 0))) ||
            (!dl && (rc = exp_clamped(&lo, &d1, p, 0))) || (rc = boundary_error(&err, &t->rnd[ATT_R_EXP], &hi, sf->vanish[z] ? NULL : &lo)) ||
            (rc = dyad_add(&hi, &hi, &err, p, 1)))
            break;
        if (dyad_cmp(&hi, &t->c[ATT_C_MAX_EXP]) > 0) {
            fail = 1;
            break;
        }
        ivl *pj = &P[keys[z]];
        if ((rc = dyad_copy(&pj->hi, &hi))) break;
        if (sf->vanish[z]) {
            rc = dyad_set_si(&pj->lo, 0, 0);
        } else if (!(rc = sub_dir(&d1, &lo, &err, p, 0))) {
            rc = dyad_sign(&d1) < 0 ? dyad_set_si(&pj->lo, 0, 0) : dyad_copy(&pj->lo, &d1);
        }
    }
    if (!rc && !fail) *ok = 1;
    ivl_free(&xg);
    ivl_free(&d);
    ivl_free(&R);
    ivl_free(&ec);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* The exact node sums of index order and lanes 2 and 4 over terms fed in order at increasing positions, the lanes by
   position (Lemma S with every node exact; terms exactly 0 between them left out, Lemma L): the running sums, and per
   order the sum of |node| (each lane's first term a leaf, not a node). */
typedef struct {
    dyad s, p4[4], n1, n2, n4, mass, rad;
    int64_t k;
    unsigned has4; /* the lanes of four with terms */
} node_sums;

static void nodes_init(node_sums *x) {
    dyad *all[] = {&x->s, &x->p4[0], &x->p4[1], &x->p4[2], &x->p4[3], &x->n1, &x->n2, &x->n4, &x->mass, &x->rad};
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) dyad_init(all[z]);
    x->k = 0;
    x->has4 = 0;
}

static void nodes_free(node_sums *x) {
    dyad *all[] = {&x->s, &x->p4[0], &x->p4[1], &x->p4[2], &x->p4[3], &x->n1, &x->n2, &x->n4, &x->mass, &x->rad};
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) dyad_free(all[z]);
}

/* the next term t, at position pos, with its distance r from its reference */
static int nodes_add(node_sums *x, const dyad *t, const dyad *r, size_t pos) {
    dyad m, p2;
    dyad_init(&m);
    dyad_init(&p2);
    int64_t k = x->k++;
    int lane = (int)(pos & 3), rc = dyad_add_exact(&x->s, &x->s, t);
    int node4 = x->has4 >> lane & 1, node2 = x->has4 >> (lane ^ 2) & 1 || node4; /* not the lane's first term */
    x->has4 |= 1u << lane;
    if (!rc && !(rc = dyad_abs(&m, t))) rc = dyad_add_exact(&x->mass, &x->mass, &m);
    if (!rc) rc = dyad_add_exact(&x->rad, &x->rad, r);
    if (!rc && k >= 1 && !(rc = dyad_abs(&m, &x->s))) rc = dyad_add_exact(&x->n1, &x->n1, &m);
    if (!rc) rc = dyad_add_exact(&x->p4[lane], &x->p4[lane], t);
    if (!rc && node4 && !(rc = dyad_abs(&m, &x->p4[lane]))) rc = dyad_add_exact(&x->n4, &x->n4, &m);
    if (!rc && node2 && !(rc = dyad_add_exact(&p2, &x->p4[lane & 1], &x->p4[(lane & 1) + 2])) && !(rc = dyad_abs(&m, &p2)))
        rc = dyad_add_exact(&x->n2, &x->n2, &m);
    dyad_free(&m);
    dyad_free(&p2);
    return rc;
}

/* N = 2 (mass + the largest node sum of index order and lanes 2 and 4, with their combinations) */
static int nodes_N(dyad *N, const node_sums *x) {
    dyad c, o2, o4, t2[2];
    dyad_init(&c);
    dyad_init(&o2);
    dyad_init(&o4);
    dyad_init(&t2[0]);
    dyad_init(&t2[1]);
    unsigned has2 = (x->has4 | x->has4 >> 2) & 3;
    int rc = dyad_add_exact(&t2[0], &x->p4[0], &x->p4[2]);
    if (!rc) rc = dyad_add_exact(&t2[1], &x->p4[1], &x->p4[3]);
    if (!rc && !(rc = lane_nodes_dyad(&c, t2, has2, 2))) rc = dyad_add_exact(&o2, &x->n2, &c);
    if (!rc && !(rc = lane_nodes_dyad(&c, x->p4, x->has4, 4))) rc = dyad_add_exact(&o4, &x->n4, &c);
    if (!rc) rc = dyad_copy(N, dmax(dmax(&x->n1, &o2), &o4));
    if (!rc) rc = dyad_add_exact(N, N, &x->mass);
    if (!rc) rc = dyad_shift(N, 1);
    dyad_free(&c);
    dyad_free(&o2);
    dyad_free(&o4);
    dyad_free(&t2[0]);
    dyad_free(&t2[1]);
    return rc;
}

/* x as an integer times 2**e, e at most x's exponent where x is not 0 */
static int at_exponent(bigint *r, const dyad *x, int64_t e) { return x->m.n ? big_shl(r, &x->m, x->e - e) : big_set_word(r, 0, 0); }

/* Lemma S for sums whose quantities are dyadics (sum_bounds_normal): [lo, hi] and *unb where no finite bound exists */
static int normal_ends(ivl *x, int *unb, const sum_bounds *b, const dyad *s, const dyad *mass, const dyad *N, const dyad *moved) {
    const dyad *all[] = {s, mass, N, moved};
    int64_t e = INT64_MAX;
    for (int z = 0; z < 4; z++)
        if (all[z]->m.n && all[z]->e < e) e = all[z]->e;
    if (e == INT64_MAX) e = 0;
    bigint v[4];
    for (int z = 0; z < 4; z++) big_init(&v[z]);
    int rc = 0, checked = 1, flags = 0;
    for (int z = 0; !rc && z < 4; z++) rc = at_exponent(&v[z], all[z], e);
    if (!rc) rc = sum_bounds_normal(&x->lo, &x->hi, &checked, &flags, b, &v[0], &v[1], &v[2], &v[3], e);
    if (!rc && !checked && flags) *unb = 1;
    for (int z = 0; z < 4; z++) big_free(&v[z]);
    return rc;
}

/* One row of the backward: P', dP', D' and dS' over the S keys of its key and value block (The tolerance, steps 1 to
   11), into o. The row's query row qr and index i, its mask (moff, mstep), its key and value block's first rows kr0 and
   vr0, its output-gradient row gr0. A NaN row (o->nan) has no terms; o->p_unb where P has no bound (every gradient the
   row touches has none), o->ds_unb where dS has none (the row's grad_query and its block's grad_key), o->ds_skip where
   dS is not finite (no dS terms; those gradients' references are not finite); o->p_nan where P' may be NaN and o->d_nan
   where D' may be NaN through the keys the row does not keep, whose terms an implementation may compute
   (scaled_dot_product_attention_backward.md, Masked keys, cases 1 and 2), and per key o->mark where only that pair's P'
   (with the given logsumexp, case 1) or dS' (case 2) may be NaN, o->q_nan where one is. With the forward's given output
   and logsumexp (c->od, c->ld), P' is exp(l - logsumexp) and D the given output's dot product with the output gradient's
   row (attention_algorithms.md, The backward of fused kernels). */
int att_gtol_row(att_gtol_out *o, att_row *w, const att_gtol_call *c, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, size_t gr0) {
    const att_call *a = &c->a;
    const att_gtol *gr = c->gr;
    const operand *gd = c->gd, *vd = c->vd;
    size_t S = a->S, Ev = (size_t)gr->Ev;
    int64_t wp = gr->w;
    o->nan = o->p_unb = o->ds_unb = o->ds_skip = o->closed = o->p_nan = o->d_nan = o->q_nan = 0;
    /* an output gradient element with no bound as an earlier result: its grad_value block has none */
    o->g_unb = c->rg && c->rg->special_rows[gr0];
    int rc = att_row_reserve(w, S, Ev), given = c->ld != NULL;
    if (!rc) rc = out_reserve(o, S);
    if (!rc) memset(o->mark, 0, S);
    for (size_t j = 0; !rc && j < S; j++)
        if (!(rc = set_zero(&o->P[j]))) rc = set_zero(&o->DS[j]);
    if (rc) return rc;
    int kept_count;
    if ((rc = att_row_logits(w, a, qr, kr0, i, moff, mstep, NULL, &kept_count))) return rc;
    /* a mask element, a query or key element that reaches a key the row does not mask, or the given logsumexp, with no
       bound as an earlier result: P' has none, even in a NaN row; one that reaches only masked keys: P' may be NaN
       (Results as inputs) */
    dyad rho, lse;
    dyad_init(&rho);
    dyad_init(&lse);
    int finite = 1, rnan = 0, snan = 0;
    if (c->lrad) rc = radius_f64(&rho, &finite, &c->f64, c->lrad + 8 * gr0);
    if (!rc && ((a->uncertain && row_unbounded(a, &c->f64, w, qr, kr0, moff, mstep, &rnan)) || !finite)) o->p_unb = 1;
    if (rc || o->p_unb || kept_count < 0) {
        dyad_free(&rho);
        dyad_free(&lse);
        if (!rc && !o->p_unb) o->nan = 1; /* every gradient it touches is NaN */
        return rc;
    }
    /* a masked key whose logit may be NaN (a score that is not finite, or a key element with no bound): P' NaN at every key
       through Z' and D', or with the given logsumexp at that key only; the query row with no bound reaches every key */
    att_masked_nan(&snan, NULL, w, a, vr0);
    if (!given || (rnan && a->rq && a->rq->special_rows[qr])) {
        o->p_nan = rnan || snan;
    } else {
        for (size_t j = 0; (rnan || snan) && j < S; j++)
            if (w->kinds[j] == ATT_KEY_MASKED && (!w->scored[j] || (a->rk && a->rk->special_rows[kr0 + j]))) o->mark[j] |= ATT_KEY_P_NAN;
    }
    size_t n = (size_t)kept_count, *keys = w->keys;
    ivl *P = o->P, *DP = o->DP, *DS = o->DS;
    ivl x, y, top, total;
    ivl *iv[] = {&x, &y, &top, &total};
    for (size_t z = 0; z < sizeof iv / sizeof *iv; z++) ivl_init(iv[z]);
    dyad d1, d2;
    dyad_init(&d1);
    dyad_init(&d2);
    product_sums ps, dev_ps;
    product_sums_init(&ps);
    product_sums_init(&dev_ps);
    node_sums ns;
    nodes_init(&ns);
    sum_bounds rb;
    int rb_set = 0, unb = 0;
    /* steps 1 to 8, P': the closed form, else the program */
    int closed = 0;
    if (gr->g.unbounded) {
        o->p_unb = 1;
        goto done;
    }
    int flushed = 0;
    if (given) {
        /* the given logsumexp: not finite, no bound; with flush, one below 2**emin may be read as zero */
        int kind;
        if ((rc = operand_dyad(&lse, &kind, c->ld, gr0))) goto done;
        if (kind != VALUE_FINITE) {
            o->p_unb = 1;
            goto done;
        }
        flushed = gr->flush && lse.m.n && top_bit(&lse) < gr->emin;
        if (!c->lrad && (rc = dyad_set_si(&rho, 0, 0))) goto done; /* rho: its radius as an earlier result */
    }
    if (gr->closed && n) {
        att_soft_free(&o->sf);
        att_soft_init(&o->sf);
        int ok = 0;
        att_call x_call = *a; /* a read-only view, without the tables where they are too coarse */
        if (!gr->tables) x_call.tables = x_call.mtables = NULL;
        if (given) {
            memset(o->mnan, 0, S);
            if (!(rc = att_soft_row(&o->sf, &ok, w, &x_call, &gr->t, n, qr, kr0, moff, mstep, 0, 1, o->mnan)) && ok)
                rc = closed_given(P, &closed, &o->sf, &gr->t, keys, n, &lse, &rho, flushed, &x_call, w->scores);
        } else if (!(rc = att_soft_row(&o->sf, &ok, w, &x_call, &gr->t, n, qr, kr0, moff, mstep, 1, 0, NULL)) && ok) {
            rc = closed_probabilities(P, &closed, &o->sf, &gr->t, keys, n);
        }
        /* condition 4 fails: P' may be NaN (Masked keys), with the given logsumexp at the keys that fail it */
        if (!rc && closed && o->sf.masked_nan && !given) o->p_nan = 1;
        for (size_t j = 0; !rc && closed && given && j < S; j++)
            if (o->mnan[j]) o->mark[j] |= ATT_KEY_P_NAN;
        if (rc == EXACTREF_OVERFLOW) rc = 0, closed = 0; /* an exponent past the dyadics' range: the program */
        if (rc) goto done;
        for (size_t z = 0; closed && !rc && z < n; z++) rc = narrow(&P[keys[z]], gr->lean[ATT_GL_PROBS].emin, gr->lean[ATT_GL_PROBS].mantissa);
        if (!closed)
            for (size_t z = 0; !rc && z < n; z++) rc = set_zero(&P[keys[z]]);
        if (rc) goto done;
    }
    o->closed = closed;
    if (!closed) {
        int masked_nan = 0, soft_unb = 0, pad_unb = 0;
        if (given) memset(o->mnan, 0, S);
        if ((rc = att_prog_softmax(o->L, o->B, o->Y, o->Wt, NULL, o->V, &top, &total, &masked_nan, &soft_unb, &pad_unb, w, a, &gr->g, n, qr, kr0, moff,
                                   mstep, 1, given, given ? o->mnan : NULL)))
            goto done;
        /* a masked key's logit may overflow: P' may be NaN (Masked keys), with the given logsumexp at that key */
        if (masked_nan && !given) o->p_nan = 1;
        for (size_t j = 0; masked_nan && given && j < S; j++)
            if (o->mnan[j]) o->mark[j] |= ATT_KEY_P_NAN;
        if (n && soft_unb) {
            o->p_unb = 1;
            goto done;
        }
        if (n && given) {
            /* P' = exp(round(l - the given logsumexp)) within ulps (with exp2 in base-2 units, the logsumexp times log2(e)
               within ulps), the logsumexp read with flush as the hull of 0 and itself */
            if ((rc = ivl_point(&y, &lse))) goto done;
            if (flushed && (rc = dyad_sign(&lse) < 0 ? dyad_set_si(&y.hi, 0, 0) : dyad_set_si(&y.lo, 0, 0))) goto done;
            if (rho.m.n && (rc = widen_by(&y, &rho, wp))) goto done; /* within its radius as an earlier result */
            if (gr->methods & ATT_GM_EXP2) {
                ivl l2e;
                l2e.lo = gr->g.log2e_lo; /* read-only views */
                l2e.hi = gr->g.log2e_hi;
                if ((rc = ivl_mul(&y, &y, &l2e, wp)) || (rc = lean_ivl(&y, &gr->g.lean[ATT_P_EXP2_LSE], wp, &unb))) goto done;
            }
            /* exp's ends from Lemma D's tables (table_exp) where att_prog_softmax kept the exact scores */
            int tabled = a->tables && !a->mask_float && !a->uncertain && !gr->g.flush && a->sign;
            w128 topv = tabled ? table_top(w->scores, n, a->sign) : w128_zero();
            dyad sl, sh, c;
            ivl ec;
            dyad_init(&sl);
            dyad_init(&sh);
            dyad_init(&c);
            ivl_init(&ec);
            if (tabled && !(rc = abs_sigma(&sl, &sh, a, wp + 128))) rc = table_row_constant(&c, &ec, topv, a, &sl, &lse, wp + 64);
            for (size_t z = 0; !rc && !unb && z < n; z++) {
                /* a key whose logit or shift may overflow toward -inf may vanish (attention_algorithms.md, Keys that may vanish):
                   its P' the hull with 0, or exactly 0 where it surely vanishes */
                ivl *pj = &P[keys[z]];
                if ((rc = ivl_sub(pj, &o->L[z], &y, wp)) || (o->V[z] != 2 && (rc = lean_vanish(pj, &unb, &o->V[z], &gr->g.lean[ATT_P_SHIFT], wp))) || unb) break;
                if (o->V[z] == 2) {
                    if ((rc = set_zero(pj))) break;
                    continue;
                }
                if (gr->methods & ATT_GM_EXP2) {
                    ivl ln2;
                    ln2.lo = gr->g.ln2_lo;
                    ln2.hi = gr->g.ln2_hi;
                    if ((rc = ivl_mul(pj, pj, &ln2, wp))) break;
                }
                if (exp_overflows(&pj->hi)) {
                    unb = 1;
                    break;
                }
                int dl = 0, dh = 0;
                w128 D = tabled ? table_distance(topv, ((const w128 *)w->scores)[z], a->sign) : w128_zero();
                if (tabled && !(rc = table_exp(&pj->lo, &dl, &pj->lo, D, a, &sl, &c, &ec.lo, wp, 0)))
                    rc = table_exp(&pj->hi, &dh, &pj->hi, D, a, &sh, &c, &ec.hi, wp, 1);
                if (!rc && !dl) rc = exp_clamped(&pj->lo, &pj->lo, wp, 0);
                if (!rc && !dh) rc = exp_clamped(&pj->hi, &pj->hi, wp, 1);
                if (!rc) rc = lean_ivl(pj, &gr->g.lean[ATT_P_EXP], wp, &unb);
                if (!rc && o->V[z] && dyad_sign(&pj->lo) > 0) rc = dyad_set_si(&pj->lo, 0, 0);
            }
            dyad_free(&sl);
            dyad_free(&sh);
            dyad_free(&c);
            ivl_free(&ec);
            if (rc) goto done;
            if (unb) {
                o->p_unb = 1;
                goto done;
            }
        } else if (n) {
            /* P' = w' / Z' rounded (with reciprocal, w' times 1 / Z' rounded, rounded) */
            if (dyad_sign(&total.lo) <= 0 && dyad_sign(&total.hi) >= 0) {
                o->p_unb = 1;
                goto done;
            }
            const ivl *by = &total;
            if (gr->methods & ATT_GM_RECIPROCAL) {
                if ((rc = dyad_set_si(&y.lo, 1, 0)) || (rc = dyad_set_si(&y.hi, 1, 0)) || (rc = ivl_div(&y, &y, &total, wp)) ||
                    (rc = lean_ivl(&y, &gr->lean[ATT_GL_RECIPROCAL], wp, &unb)))
                    goto done;
                if (unb) {
                    o->p_unb = 1;
                    goto done;
                }
            }
            for (size_t z = 0; !rc && !unb && z < n; z++) {
                ivl *pj = &P[keys[z]];
                rc = (gr->methods & ATT_GM_RECIPROCAL) ? ivl_mul(pj, &o->Wt[z], &y, wp) : ivl_div(pj, &o->Wt[z], by, wp);
                if (!rc) rc = lean_ivl(pj, &gr->lean[ATT_GL_PROBS], wp, &unb);
            }
            if (rc) goto done;
            if (unb) {
                o->p_unb = 1;
                goto done;
            }
        }
    }
    /* step 9, dP': at the kept keys its sum's bound, then its rounding, where dP is not finite at one, so is D, and dS has
       no terms; at the other keys only whether it may be non-finite, which makes D' possibly NaN (Masked keys, case 2) */
    int what = gr->normal ? PRODUCT_NODES : PRODUCT_MASS;
    /* an element of the row's output gradient or given output, or a value element at a kept key, with no bound as an
       earlier result: dP, so D and dS', have none; a value element at another key: D' may be NaN */
    int dsu = (c->rg && c->rg->special_rows[gr0]) || (given && c->ro && c->ro->special_rows[gr0]);
    for (size_t j = 0; !dsu && c->rv && j < S; j++)
        if (c->rv->special_rows[vr0 + j]) {
            if (w->kinds[j] == ATT_KEY_KEPT) dsu = 1;
            else o->d_nan = 1;
        }
    if (dsu) {
        o->ds_unb = 1;
        goto done;
    }
    for (size_t j = 0; !rc && !o->ds_skip && j < S; j++)
        if ((gd->special_rows[gr0] || vd->special_rows[vr0 + j]) && !(rc = product_element(&ps, gd, gr0, vd, vr0 + j, PRODUCT_SUM, gr->flush, gr->emin, NULL)) &&
            ps.kind != VALUE_FINITE) {
            if (w->kinds[j] == ATT_KEY_KEPT) o->ds_skip = 1;
            else o->d_nan = 1;
        }
    if (rc || o->ds_skip) goto done;
    for (size_t j = 0; !rc && j < S; j++) {
        int is_kept = w->kinds[j] == ATT_KEY_KEPT;
        if (!is_kept && o->d_nan) continue; /* D' may be NaN already: the other keys' dP' only decided that */
        if ((rc = product_element(&ps, gd, gr0, vd, vr0 + j, what, gr->flush, gr->emin, NULL))) break;
        /* the stored output gradient's and values' deviation (proofs/matmul.md, Results as inputs) */
        if ((c->rg || c->rv) && (rc = product_deviation(&ps, &unb, &dev_ps, gd, c->rg, gr0, vd, c->rv, vr0 + j, 0))) break;
        if (gr->normal) {
            dyad sd, md, Nd, vd2; /* views */
            dyad_view(&sd, &ps.s, ps.e);
            dyad_view(&md, &ps.mass, ps.e);
            dyad_view(&Nd, &ps.N, ps.e);
            dyad_view(&vd2, &ps.moved, ps.e);
            rc = normal_ends(&DP[j], &unb, &gr->dp, &sd, &md, &Nd, &vd2);
        } else {
            /* [s - moved, s + moved], then Lemma 2 at Ev terms */
            dyad sd, vd2;
            dyad_view(&sd, &ps.s, ps.e);
            dyad_view(&vd2, &ps.moved, ps.e);
            if (!(rc = dyad_neg(&d1, &vd2)) && !(rc = dyad_add_exact(&DP[j].lo, &sd, &d1)) && !(rc = dyad_add_exact(&DP[j].hi, &sd, &vd2))) {
                dyad md;
                dyad_view(&md, &ps.mass, ps.e);
                rc = acc_apply(&DP[j], &gr->acc[ATT_GA_DP], Ev, &md, wp, &unb);
            }
        }
        if (!rc && !unb) rc = lean_ivl(&DP[j], &gr->lean[ATT_GL_DP], wp, &unb);
        if (unb && !is_kept) {
            o->d_nan = 1;
            unb = 0;
        }
        if (unb) break;
    }
    if (rc || o->ds_skip) goto done;
    if (unb) {
        o->ds_unb = 1;
        goto done;
    }
    /* step 10 with the given output: D' = sum_c g_c out_c, exact products, Lemma 2 at Ev terms under both standards (any order
       covers normal accumulation) */
    ivl *D = &top;
    if (given) {
        if ((rc = product_element(&ps, gd, gr0, c->od, gr0, PRODUCT_MASS, gr->flush, gr->emin, NULL))) goto done;
        if (ps.kind != VALUE_FINITE) {
            o->ds_unb = 1; /* a non-finite given output makes D unbounded */
            goto done;
        }
        if ((c->rg || c->ro) && (rc = product_deviation(&ps, &unb, &dev_ps, gd, c->rg, gr0, c->od, c->ro, gr0, 0))) goto done;
        dyad sd, vd2, md; /* views */
        dyad_view(&sd, &ps.s, ps.e);
        dyad_view(&vd2, &ps.moved, ps.e);
        dyad_view(&md, &ps.mass, ps.e);
        if ((rc = dyad_neg(&d1, &vd2)) || (rc = dyad_add_exact(&D->lo, &sd, &d1)) || (rc = dyad_add_exact(&D->hi, &sd, &vd2)) ||
            (rc = acc_apply(D, &gr->acc[ATT_GA_ROW_DOT], Ev, &md, wp, &unb)))
            goto done;
        if (!unb && gr->row_dot_active && (rc = narrow(D, gr->row_dot_f.emin, (int64_t)gr->row_dot_f.man_bits))) goto done;
        if (unb) {
            o->ds_unb = 1;
            goto done;
        }
    }
    /* step 10, D' over the kept keys: the products' exact ends; the normal standard's nodes exact, its lanes by key index
       (the masked keys' exact zeros left out, Lemma L), each product's distance from its midpoint a leaf's error; any
       order, Lemma 2; then narrowed to D's accumulator's values */
    if (n && !given) {
        if (!(rc = set_zero(&total))) rc = dyad_set_si(&d2, 0, 0); /* total: the any-order ends; d2: the mass */
        size_t live = 0; /* the products that can be nonzero (P' is [0, 0] at a key that surely vanishes) */
        for (size_t z = 0; !rc && z < n; z++) {
            size_t j = keys[z];
            if ((rc = mul_exact(&x, &P[j], &DP[j]))) break;
            live += dyad_sign(&x.lo) || dyad_sign(&x.hi);
            if (gr->normal) {
                /* twice the midpoint and twice the radius */
                if (!(rc = dyad_add_exact(&d1, &x.lo, &x.hi)) && !(rc = dyad_neg(&y.lo, &x.lo)) && !(rc = dyad_add_exact(&y.hi, &x.hi, &y.lo)))
                    rc = nodes_add(&ns, &d1, &y.hi, j);
            } else if (!(rc = ivl_add(&total, &total, &x, EXACT_BITS)) && !(rc = ends_magnitude(&d1, &x.lo, &x.hi))) {
                rc = dyad_add_exact(&d2, &d2, &d1);
            }
        }
        if (rc) goto done;
        if (gr->normal) {
            /* Lemma S at the doubled units: mass + R, N + 2R, moved R */
            dyad N, m2, N2;
            dyad_init(&N);
            dyad_init(&m2);
            dyad_init(&N2);
            if (!(rc = sum_bounds_init(&rb, &gr->row_dot_f, gr->flush, (int64_t)n, (int64_t)n - 1, 0, wp)))
                rb_set = 1;
            if (!rc && !(rc = nodes_N(&N, &ns)) && !(rc = dyad_add_exact(&m2, &ns.mass, &ns.rad)) && !(rc = dyad_add_exact(&N2, &N, &ns.rad)) &&
                !(rc = dyad_add_exact(&N2, &N2, &ns.rad)) && !(rc = dyad_shift(&ns.s, -1)) && !(rc = dyad_shift(&m2, -1)) && !(rc = dyad_shift(&N2, -1)) &&
                !(rc = dyad_shift(&ns.rad, -1))) { /* the halves of the doubled quantities */
                rc = normal_ends(D, &unb, &rb, &ns.s, &m2, &N2, &ns.rad);
            }
            dyad_free(&N);
            dyad_free(&m2);
            dyad_free(&N2);
        } else if (!(rc = ivl_copy(D, &total))) {
            if (live) rc = acc_apply(D, &gr->acc[ATT_GA_ROW_DOT], live, &d2, wp, &unb); /* no term that can be nonzero: exactly 0 */
        }
        if (!rc && !unb && gr->row_dot_active) rc = narrow(D, gr->row_dot_f.emin, (int64_t)gr->row_dot_f.man_bits);
        if (rc) goto done;
        if (unb) {
            o->ds_unb = 1;
            goto done;
        }
    }
    /* the keys the row does not keep, their dP' bounded: where dP'_j - D' may overflow, P' = 0 times it makes that pair's dS'
       NaN (Masked keys, case 2), so the row's grad_query and that key's grad_key; without kept keys or a given output, D' is
       0 */
    if (!given && !n && (rc = set_zero(D))) goto done;
    for (size_t j = 0; !rc && !o->d_nan && j < S; j++) {
        if (w->kinds[j] == ATT_KEY_KEPT) continue;
        int u = 0;
        if (!(rc = dyad_neg(&d1, &D->hi)) && !(rc = dyad_add_exact(&x.lo, &DP[j].lo, &d1)) && !(rc = dyad_neg(&d1, &D->lo)) &&
            !(rc = dyad_add_exact(&x.hi, &DP[j].hi, &d1)))
            rc = lean_ivl(&x, &gr->lean[ATT_GL_DIFF], wp, &u);
        if (u) o->mark[j] |= ATT_KEY_DS_NAN;
    }
    if (rc) goto done;
    if (n) {
        /* step 11: dS' = round(round(dP' - D') P') per kept key (with rounded_p, P' rounded to the weights' format) */
        for (size_t z = 0; !rc && !unb && z < n; z++) {
            size_t j = keys[z];
            if ((rc = dyad_neg(&d1, &D->hi)) || (rc = dyad_add_exact(&x.lo, &DP[j].lo, &d1)) || (rc = dyad_neg(&d1, &D->lo)) ||
                (rc = dyad_add_exact(&x.hi, &DP[j].hi, &d1)) || (rc = lean_ivl(&x, &gr->lean[ATT_GL_DIFF], wp, &unb)) || unb)
                break;
            const ivl *pj = &P[j];
            if (gr->methods & ATT_GM_ROUNDED_P) {
                if ((rc = ivl_copy(&y, &P[j])) || (rc = lean_ivl(&y, &gr->lean[ATT_GL_WEIGHTS], wp, &unb)) || unb) break;
                pj = &y;
            }
            if (!(rc = mul_exact(&DS[j], &x, pj))) rc = lean_ivl(&DS[j], &gr->lean[ATT_GL_DS], wp, &unb);
        }
        if (rc) goto done;
        if (unb) {
            for (size_t z = 0; !rc && z < n; z++) rc = set_zero(&DS[keys[z]]);
            o->ds_unb = 1;
            goto done;
        }
    }
    /* scale_ds: dS times the scale, rounded once */
    if ((gr->methods & ATT_GM_SCALE_DS) && n) {
        if (!gr->scale_bounded) {
            o->ds_unb = 1;
        } else {
            ivl sig;
            sig.lo = gr->scale_lo; /* read-only views */
            sig.hi = gr->scale_hi;
            for (size_t z = 0; !rc && !unb && z < n; z++)
                if (!(rc = mul_exact(&x, &DS[keys[z]], &sig)) && !(rc = lean_ivl(&x, &gr->lean[ATT_GL_SCALE_DS], wp, &unb))) rc = ivl_copy(&DS[keys[z]], &x);
            if (unb) o->ds_unb = 1;
        }
    }
done:
    for (size_t j = 0; j < S; j++) o->q_nan |= o->mark[j] != 0;
    /* weights: P' rounded to the weights' format wherever it has a bound, also where dS' has none or is not finite (P'
       still reaches grad_value); dS' where it has one */
    if (!rc && !o->p_unb && (gr->methods & ATT_GM_WEIGHTS) && n) {
        int ds = !o->ds_unb && !o->ds_skip;
        unb = 0;
        for (size_t z = 0; !rc && !unb && z < n; z++)
            if (!(rc = lean_ivl(&P[keys[z]], &gr->lean[ATT_GL_WEIGHTS], wp, &unb)) && !unb && ds) rc = lean_ivl(&DS[keys[z]], &gr->lean[ATT_GL_WEIGHTS], wp, &unb);
        if (unb) o->p_unb = 1;
    }
    for (size_t z = 0; z < sizeof iv / sizeof *iv; z++) ivl_free(iv[z]);
    dyad_free(&d1);
    dyad_free(&d2);
    dyad_free(&lse);
    dyad_free(&rho);
    product_sums_free(&ps);
    product_sums_free(&dev_ps);
    nodes_free(&ns);
    if (rb_set) sum_bounds_free(&rb);
    return rc;
}

/* ---------------------------------------------------------------- the gradient sums
   (The tolerance, step 12; kernels.md, Lemma J): per element the terms w x, w in [lo, hi] (P' or dS', integers at their
   format's least value 2**up or 2**ud) and x exact (the output gradient's, Q's or K's value, an integer at its operand's
   finest set bit 2**ux), in the model's order. Normal: twice the midpoints' products with every node exact (index
   order, lanes 2 and 4), the radius sum R; any order: the ends' products' sums, the magnitudes and the count of terms
   that can be nonzero. Integers are two's complement over the call's words. */

/* the magnitude of the larger end of [lo, hi] (n words each) added to a dyad sum at 2**e */
static int add_size(dyad *sum, const uint64_t *lo, const uint64_t *hi, int n, int64_t e, uint64_t *tmp) {
    dyad a, b;
    dyad_init(&a);
    dyad_init(&b);
    int rc = mw_to_dyad(&a, lo, n, e, tmp);
    if (!rc) rc = mw_to_dyad(&b, hi, n, e, tmp);
    a.m.neg = b.m.neg = 0;
    if (!rc) rc = dyad_add_exact(sum, sum, dmax(&a, &b));
    dyad_free(&a);
    dyad_free(&b);
    return rc;
}

void att_gtol_call_init(att_gtol_call *c) {
    memset(c, 0, sizeof *c);
    big_init(&c->a.num);
    big_init(&c->a.den);
    for (int z = 0; z < 2; z++) {
        big_init(&c->out_nan[z]);
        big_init(&c->f64_nan[z]);
    }
}

void att_gtol_call_free(att_gtol_call *c) {
    free(c->qstates);
    free(c->qcounts);
    free(c->qunb);
    free(c->qnan);
    for (int z = 0; z < 3; z++) {
        free(c->x[z]);
        free(c->xr[z]);
        free(c->xsub[z]);
    }
    big_free(&c->a.num);
    big_free(&c->a.den);
    for (int z = 0; z < 2; z++) {
        big_free(&c->out_nan[z]);
        big_free(&c->f64_nan[z]);
    }
    memset(c, 0, sizeof *c);
}

/* An operand's largest magnitude's top bit (2**t above every |x|) and its finest set bit, over its finite nonzero values */
void operand_extent(int64_t *top, int64_t *low, int *any, const operand *x) {
    *any = 0;
    for (size_t r = 0; r < x->rows; r++) {
        if (!x->top[r] && !x->low[r] && !x->max_sig[r]) continue;
        if (!*any || x->top[r] > *top) *top = x->top[r];
        if (!*any || x->low[r] < *low) *low = x->low[r];
        *any = 1;
    }
}

/* The call's exact factors (the output gradient's, Q's and K's values) and their radii as earlier results as integers at
   their finest set bit, and their words; the units and words of P' and dS' ends, and of each gradient's sums (kernels.md, Lemma J's
   bounds: P' below 4, dS' below 32 Ev max|g| max|v| (times |scale| with scale_ds), the factors' |x| + r, the gradients'
   term counts nq and nkv), and the most bits an end may take for those sums (plim, dlim). c->gr, c->a, the operands, their
   radii and c->Ev are set. */
int att_gtol_call_prepare(att_gtol_call *c) {
    const att_gtol *gr = c->gr;
    const operand *xs[3] = {c->gs, c->qs, c->ks}; /* grad_value's factor, grad_key's, grad_query's */
    const operand *rs[3] = {c->rg, c->a.rq, c->a.rk}; /* ... and their radii's (each read as its factor is) */
    int64_t emin = gr->flush ? gr->emin : INT64_MIN, xbits[3];
    int rc = 0;
    for (int z = 0; !rc && z < 3; z++) {
        const operand *x = xs[z], *rx = rs[z];
        int64_t top = 0, low = 0, rtop = 0, rlow = 0;
        int any, rany = 0;
        operand_extent(&top, &low, &any, x);
        if (rx) operand_extent(&rtop, &rlow, &rany, rx);
        if (rany) {
            /* |x| + r below 2**(max(top, rtop) + 1), at the finer of the two units */
            top = any ? (top > rtop ? top : rtop) + 1 : rtop;
            low = any && low < rlow ? low : rlow;
            any = 1;
        }
        c->ux[z] = any ? low : 0;
        xbits[z] = (any ? top - low : 0) + 2;
        c->wx[z] = mw_words(xbits[z]);
        size_t count = x->rows * x->cols;
        if (!(c->x[z] = calloc((count ? count : 1) * (size_t)c->wx[z], sizeof *c->x[z])) || !(c->xsub[z] = calloc(count ? count : 1, 1)) ||
            (rany && !(c->xr[z] = calloc((count ? count : 1) * (size_t)c->wx[z], sizeof *c->xr[z])))) {
            rc = EXACTREF_NOMEM;
            break;
        }
        dyad v;
        dyad_init(&v);
        for (size_t at = 0; !rc && at < count; at++) {
            int kind;
            /* a radius with no bound leaves its rows without one (att_gtol_row's flags): 0 here */
            if (c->xr[z] && !(rc = operand_dyad(&v, &kind, rx, at)) && kind == VALUE_FINITE)
                rc = mw_from_dyad(c->xr[z] + at * (size_t)c->wx[z], c->wx[z], &v, c->ux[z], 1);
            if (rc || (rc = operand_dyad(&v, &kind, x, at))) break;
            if (kind != VALUE_FINITE) continue; /* a non-finite factor: its elements' references are not finite */
            rc = mw_from_dyad(c->x[z] + at * (size_t)c->wx[z], c->wx[z], &v, c->ux[z], 0);
            /* with flush, a value below 2**emin in magnitude may be read as zero */
            if (!rc && v.m.n && gr->flush && big_bits(&v.m) + v.e - 1 < emin) c->xsub[z][at] = 1;
        }
        dyad_free(&v);
    }
    /* P' and dS': their last rounding's format's least value */
    const round_bound *pl = &gr->lean[(gr->methods & ATT_GM_WEIGHTS) ? ATT_GL_WEIGHTS : ATT_GL_PROBS];
    const round_bound *dl = &gr->lean[(gr->methods & ATT_GM_WEIGHTS) ? ATT_GL_WEIGHTS : (gr->methods & ATT_GM_SCALE_DS) ? ATT_GL_SCALE_DS : ATT_GL_DS];
    c->up = pl->active ? pl->emin - pl->mantissa : 0;
    c->ud = dl->active ? dl->emin - dl->mantissa : 0;
    int64_t pbits = 2 - c->up + 2, dbits = 0;
    if (pbits < c->pmin) pbits = c->pmin;
    c->wp = mw_words(pbits);
    {
        int64_t tg = 0, lg = 0, tv = 0, lv = 0;
        int ag, av;
        operand_extent(&tg, &lg, &ag, c->gd);
        operand_extent(&tv, &lv, &av, c->vd);
        int64_t td = (ag && av ? tg + tv : 0) + word_bits((uint64_t)(c->a.Ev ? c->a.Ev : 1)) + 5;
        if ((gr->methods & ATT_GM_SCALE_DS) && gr->scale_bounded) {
            dyad m;
            dyad_init(&m);
            if (!(rc = ends_magnitude(&m, &gr->scale_lo, &gr->scale_hi)) && m.m.n) td += top_bit(&m) + 2;
            dyad_free(&m);
        }
        dbits = td - c->ud + 2;
        if (dbits < c->dmin) dbits = c->dmin;
        c->wd = mw_words(dbits);
    }
    /* each gradient's words (kernels.md, Lemma J): its ends' bits and its exact factors', 2 ceil(log2 n) + 6, and a sign bit */
    int64_t ends[3] = {pbits, dbits, dbits}, counts[3] = {gr->nkv, gr->nkv, gr->nq};
    int na[3] = {c->wp, c->wd, c->wd};
    int64_t room[3]; /* the most bits an end may take for each gradient's sums to fit four words */
    c->fast = c->wp <= 3 && c->wd <= 3;
    for (int z = 0; z < 3; z++) {
        int64_t cbits = counts[z] > 1 ? word_bits((uint64_t)(counts[z] - 1)) : 0;
        c->ws[z] = mw_words(ends[z] + xbits[z] + 2 * cbits + 7);
        c->fast = c->fast && c->wx[z] == 1 && c->ws[z] <= 4;
        room[z] = 256 - (xbits[z] + 2 * cbits + 7);
    }
    if (c->fast) {
        /* the fast path's layout: three-word ends, four-word sums, each end at most the bits that keep every sum it joins
           in four words (P' grad_value's, dS' grad_key's and grad_query's) */
        c->plim = room[0];
        c->dlim = room[1] < room[2] ? room[1] : room[2];
        if (c->plim > 192) c->plim = 192;
        if (c->dlim > 192) c->dlim = 192;
        c->wp = c->wd = 3;
        c->ws[0] = c->ws[1] = c->ws[2] = 4;
    } else {
        for (int z = 0; z < 3; z++)
            if (c->ws[z] < na[z] + c->wx[z] + 2) c->ws[z] = na[z] + c->wx[z] + 2; /* the general product of an end's midpoint and a factor */
        c->plim = 64 * (int64_t)c->wp;
        c->dlim = 64 * (int64_t)c->wd;
    }
    return rc;
}

enum { SUM_S, SUM_P4, SUM_N1 = SUM_P4 + 4, SUM_N2, SUM_N4, SUM_MASS, SUM_RAD, SUM_VALUES }; /* a normal sum's values */
enum { ANY_LO, ANY_HI, ANY_MASS, ANY_VALUES };                                       /* an any-order sum's */

static int sum_values(const att_gtol *gr) { return gr->sums_normal ? SUM_VALUES : ANY_VALUES; }

/* slots for count query rows broadcast over several key and value blocks: each its E grad_query sums between copies */
int att_gtol_query_slots(att_gtol_call *c, size_t count) {
    size_t E = (size_t)c->a.E, values = (size_t)sum_values(c->gr) * (size_t)c->ws[2];
    free(c->qstates);
    free(c->qcounts);
    free(c->qunb);
    free(c->qnan);
    c->qstates = calloc((count && E ? count * E : 1) * values, sizeof *c->qstates);
    c->qcounts = calloc(count && E ? count * E : 1, sizeof *c->qcounts);
    c->qunb = calloc(count ? count : 1, 1);
    c->qnan = calloc(count && E ? count * E : 1, 1);
    return c->qstates && c->qcounts && c->qunb && c->qnan ? 0 : EXACTREF_NOMEM;
}

void att_gtol_group_init(att_gtol_group *g) { memset(g, 0, sizeof *g); }

void att_gtol_group_free(att_gtol_group *g) {
    void *all[] = {g->rinfo, g->flags, g->plo, g->phi, g->dlo, g->dhi, g->need, g->kinds};
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) free(all[z]);
    memset(g, 0, sizeof *g);
}

/* A key block's grad_key sums (key, K rows r0..) or a value block's grad_value sums at zero terms */
int att_gtol_block_start(att_gtol_block *b, const att_gtol_call *c, int key, size_t r0) {
    size_t S = c->a.S, E = (size_t)c->a.E, nv = (size_t)sum_values(c->gr);
    memset(b, 0, sizeof *b);
    b->r0 = r0;
    b->cols = key ? E : c->a.Ev;
    b->ws = c->ws[key ? 1 : 0];
    size_t n = S && b->cols ? S * b->cols : 1;
    b->elements = n;
    if (!(b->sum = calloc(n * nv * (size_t)b->ws, sizeof *b->sum)) || !(b->count = calloc(n, sizeof *b->count)) || !(b->started = calloc(S ? S : 1, 1)) ||
        !(b->nanel = calloc(n, 1)))
        return EXACTREF_NOMEM;
    if (c->gr->methods & ATT_GM_GROUPED) {
        /* per element its sum over the heads' rounded sums (lo, hi, mass) */
        if (!(b->group = malloc(3 * n * sizeof *b->group)) || !(b->gunb = calloc(n, 1))) return EXACTREF_NOMEM;
        for (size_t z = 0; z < 3 * n; z++) dyad_init(&b->group[z]);
    }
    if (c->gr->query_block) {
        /* backward_blocks: per element its carried value (lo, hi) and status; a key block's per key sum of |dS'| over the
           block of query rows and per column of Q its sum of |q| (Lemma C's factors for grad_key) */
        if (!(b->carried = malloc(2 * n * sizeof *b->carried)) || !(b->cstatus = calloc(n, 1)) || (key && !(b->dsum = malloc((S + E ? S + E : 1) * sizeof *b->dsum))))
            return EXACTREF_NOMEM;
        for (size_t z = 0; z < 2 * n; z++) dyad_init(&b->carried[z]);
        b->dcount = key ? S + E : 0;
        for (size_t z = 0; z < b->dcount; z++) dyad_init(&b->dsum[z]);
    }
    return 0;
}

void att_gtol_block_free(att_gtol_block *b) {
    free(b->sum);
    free(b->count);
    free(b->started);
    free(b->nanel);
    free(b->gunb);
    free(b->cstatus);
    if (b->group)
        for (size_t z = 0; z < 3 * b->elements; z++) dyad_free(&b->group[z]);
    free(b->group);
    if (b->carried)
        for (size_t z = 0; z < 2 * b->elements; z++) dyad_free(&b->carried[z]);
    free(b->carried);
    for (size_t z = 0; b->dsum && z < b->dcount; z++) dyad_free(&b->dsum[z]);
    free(b->dsum);
    memset(b, 0, sizeof *b);
}

/* A group for the unit of key rows kr0.. and value rows vr0.., adding to blocks kb and vb, its tiles up to cap rows */
int att_gtol_group_start(att_gtol_group *g, const att_gtol_call *c, size_t kr0, size_t vr0, att_gtol_block *kb, att_gtol_block *vb, size_t cap) {
    size_t S = c->a.S;
    g->kr0 = kr0;
    g->vr0 = vr0;
    g->kb = kb;
    g->vb = vb;
    g->cap = cap ? cap : 1;
    size_t cells = g->cap * (S ? S : 1);
    if (!(g->rinfo = calloc(g->cap * ATT_GR_INFO, sizeof *g->rinfo)) || !(g->flags = calloc(g->cap, 1)) || !(g->need = calloc(2 * g->cap, sizeof *g->need)) ||
        !(g->plo = calloc(cells * (size_t)c->wp, sizeof *g->plo)) || !(g->phi = calloc(cells * (size_t)c->wp, sizeof *g->phi)) ||
        !(g->dlo = calloc(cells * (size_t)c->wd, sizeof *g->dlo)) || !(g->dhi = calloc(cells * (size_t)c->wd, sizeof *g->dhi)) || !(g->kinds = calloc(cells, 1)))
        return EXACTREF_NOMEM;
    return 0;
}

/* Rows r0 to r1 of the group's tile (their info in g->rinfo, T of them): P' and dS' as integers at their units, and the
   rows' flags; *closed the rows whose P' the closed form gave. A row whose ends take more bits than the call's sums are
   sized for (plim, dlim) records the bits they need in g->need (the call is then made again with them, kernels.md, Lemma J). */
int att_gtol_tile_rows(size_t *closed, att_gtol_group *g, const att_gtol_call *c, size_t r0, size_t r1) {
    size_t S = c->a.S;
    int wp = c->wp, wd = c->wd, rc = 0;
    att_row w;
    att_row_init(&w);
    att_gtol_out o;
    att_gtol_out_init(&o);
    *closed = 0;
    for (size_t r = r0; !rc && r < r1; r++) {
        const int64_t *ri = g->rinfo + ATT_GR_INFO * r;
        if ((rc = att_gtol_row(&o, &w, c, (size_t)ri[0], (size_t)ri[1], (size_t)ri[2], (size_t)ri[3], g->kr0, g->vr0, (size_t)ri[4])))
            break;
        *closed += (size_t)o.closed;
        g->flags[r] = (unsigned char)((o.nan ? ATT_GF_NAN : 0) | (o.p_unb ? ATT_GF_P_UNB : 0) | (o.ds_unb ? ATT_GF_DS_UNB : 0) | (o.ds_skip ? ATT_GF_DS_SKIP : 0) |
                                      (o.g_unb ? ATT_GF_G_UNB : 0) | (o.p_nan ? ATT_GF_P_NAN : 0) | (o.d_nan ? ATT_GF_D_NAN : 0) |
                                      (o.q_nan ? ATT_GF_Q_NAN : 0));
        /* the keys' parts and the pairs' marks, for the NaN of the terms of the keys the row does not keep */
        for (size_t j = 0; j < S; j++) g->kinds[r * S + j] = (unsigned char)(w.kinds[j] | o.mark[j]);
        int pzero = o.nan || o.p_unb, dzero = pzero || o.ds_unb || o.ds_skip; /* those terms are zero */
        int64_t *need = g->need + 2 * r;
        need[0] = need[1] = 0;
        for (size_t j = 0; !rc && j < S; j++) {
            size_t at = r * S + j;
            uint64_t *pl = g->plo + at * (size_t)wp, *ph = g->phi + at * (size_t)wp, *dl = g->dlo + at * (size_t)wd, *dh = g->dhi + at * (size_t)wd;
            int64_t bits = 0;
            memset(pl, 0, (size_t)wp * sizeof *pl);
            memset(ph, 0, (size_t)wp * sizeof *ph);
            for (int end = 0; !rc && !pzero && end < 2; end++)
                if ((rc = mw_from_dyad_need(end ? ph : pl, wp, end ? &o.P[j].hi : &o.P[j].lo, c->up, end, c->plim, &bits)) == EXACTREF_OVERFLOW) {
                    rc = 0;
                    if (bits > need[0]) need[0] = bits;
                }
            memset(dl, 0, (size_t)wd * sizeof *dl);
            memset(dh, 0, (size_t)wd * sizeof *dh);
            for (int end = 0; !rc && !dzero && end < 2; end++)
                if ((rc = mw_from_dyad_need(end ? dh : dl, wd, end ? &o.DS[j].hi : &o.DS[j].lo, c->ud, end, c->dlim, &bits)) == EXACTREF_OVERFLOW) {
                    rc = 0;
                    if (bits > need[1]) need[1] = bits;
                }
        }
    }
    att_gtol_out_free(&o);
    att_row_free(&w);
    return rc;
}

/* The term [lo, hi] x into a sum (st, its values of n words; *count its terms that can be nonzero) as its index-k term:
   normal, twice the midpoints' product with every node exact and the radius; any order, the ends' products, the
   magnitude and the count. lo and hi have na words, x nx words (sub: a flush may read it as zero), xr x's radius as an
   earlier result (nx words, unsigned) or NULL: the stored factor within it of x moves the term by max(|lo|, |hi|) xr. */
static void add_term(uint64_t *st, int64_t *count, int n, int normal, const uint64_t *lo, const uint64_t *hi, int na, const uint64_t *x, const uint64_t *xr,
                     int nx, int sub, int64_t k, uint64_t *tmp) {
    uint64_t *t = tmp, *u = t + n, *v = u + n, *scratch = v + n; /* scratch: 2 n + 2 words */
    int rzero = !xr || mw_zero(xr, nx), xzero = mw_zero(x, nx) && rzero, azero = mw_zero(lo, na) && mw_zero(hi, na);
    if (!normal) {
        if (xzero || azero) return;
        ++*count;
        /* for x > 0, [lo x, hi x]; x < 0, [hi x, lo x]; with a flush, the hull with 0 */
        int neg = mw_negative(x, nx);
        const uint64_t *a1 = neg ? hi : lo, *a2 = neg ? lo : hi;
        uint64_t *z = scratch + n + 2; /* a zero end, for the hull */
        memset(z, 0, (size_t)na * sizeof *z);
        if (sub) {
            /* the low end takes min(lo, 0) (x > 0) or max(hi, 0) (x < 0): whichever end is past 0 on its side */
            if (!neg ? !mw_negative(lo, na) : mw_negative(hi, na)) a1 = z;
            if (!neg ? mw_negative(hi, na) : !mw_negative(lo, na)) a2 = z;
        }
        mw_mul(t, n, a1, na, x, nx, scratch);
        mw_add(st + ANY_LO * n, t, n);
        mw_mul(t, n, a2, na, x, nx, scratch);
        mw_add(st + ANY_HI * n, t, n);
        /* max(|lo|, |hi|) |x| */
        memcpy(u, lo, (size_t)na * sizeof *u);
        if (mw_negative(u, na)) mw_neg(u, na);
        memcpy(v, hi, (size_t)na * sizeof *v);
        if (mw_negative(v, na)) mw_neg(v, na);
        const uint64_t *big = nat_cmp(u, v, (size_t)na) >= 0 ? u : v;
        mw_mul(t, n, big, na, x, nx, scratch);
        if (mw_negative(t, n)) mw_neg(t, n);
        mw_add(st + ANY_MASS * n, t, n);
        if (!rzero) {
            /* the ends widen by max(|lo|, |hi|) r, and the magnitude takes it */
            mw_mul(t, n, big, na, xr, nx, scratch);
            mw_add(st + ANY_HI * n, t, n);
            mw_add(st + ANY_MASS * n, t, n);
            mw_neg(t, n);
            mw_add(st + ANY_LO * n, t, n);
        }
        return;
    }
    if (!xzero && !azero) {
        /* a = lo + hi, ra = hi - lo; b = 2x, rb = 0, or with a flush b = x, rb = |x|; rb takes 2 xr */
        uint64_t *a = u, *ra = v, *bx = scratch + n + 2, *rb = bx + nx + 1;
        mw_extend(a, n, lo, na);
        mw_extend(t, n, hi, na);
        mw_add(a, t, n);
        mw_extend(ra, n, lo, na);
        mw_neg(ra, n);
        mw_add(ra, t, n);
        mw_extend(bx, nx + 1, x, nx);
        memset(rb, 0, (size_t)(nx + 1) * sizeof *rb);
        if (sub) {
            memcpy(rb, bx, (size_t)(nx + 1) * sizeof *rb);
            if (mw_negative(rb, nx + 1)) mw_neg(rb, nx + 1);
        } else {
            mw_add(bx, bx, nx + 1);
        }
        if (!rzero) {
            uint64_t *r2 = rb + 2 * (nx + 1); /* past mb */
            memcpy(r2, xr, (size_t)nx * sizeof *r2);
            r2[nx] = 0;
            mw_add(rb, r2, nx + 1);
            mw_add(rb, r2, nx + 1);
        }
        /* the term a b, and its distance |a| rb + ra |b| + ra rb (a and ra of na + 1 words) */
        int na2 = na + 1;
        mw_mul(t, n, a, na2, bx, nx + 1, scratch);
        uint64_t *d = st + SUM_RAD * n;
        if (!mw_zero(ra, n) || !mw_zero(rb, nx + 1)) {
            uint64_t *mb = rb + nx + 1;
            memcpy(mb, bx, (size_t)(nx + 1) * sizeof *mb);
            if (mw_negative(mb, nx + 1)) mw_neg(mb, nx + 1);
            if (mw_negative(a, n)) mw_neg(a, n);        /* |a| */
            mw_mul(u, n, a, na2, rb, nx + 1, scratch); /* |a| rb, over a's own words (u) */
            mw_add(d, u, n);
            mw_add(mb, rb, nx + 1);                     /* |b| + rb */
            mw_mul(u, n, ra, na2, mb, nx + 1, scratch); /* ra (|b| + rb) */
            mw_add(d, u, n);
        }
        mw_add(st + SUM_S * n, t, n);
        mw_add_abs(st + SUM_MASS * n, t, n, scratch);
        mw_add(st + (SUM_P4 + (k & 3)) * n, t, n);
    } else {
        memset(t, 0, (size_t)n * sizeof *t);
    }
    /* the nodes: index order's from its second term, each lane's from its second */
    if (k >= 1) mw_add_abs(st + SUM_N1 * n, st + SUM_S * n, n, scratch);
    if (k >= 4) mw_add_abs(st + SUM_N4 * n, st + (SUM_P4 + (k & 3)) * n, n, scratch);
    if (k >= 2) {
        memcpy(u, st + (SUM_P4 + (k & 1)) * n, (size_t)n * sizeof *u);
        mw_add(u, st + (SUM_P4 + (k & 1) + 2) * n, n);
        mw_add_abs(st + SUM_N2 * n, u, n, scratch);
    }
}

/* The fast path's term step (add_term at three-word ends, a one-word factor and four-word sums), the values loaded once. */

static void add_term4(uint64_t *st, int64_t *count, int normal, const uint64_t *plo, const uint64_t *phi, int64_t x, uint64_t r, int sub, int64_t k) {
    w4 *v = (w4 *)st, lo, hi, t; /* v[i]: value i of the sums, in place */
    w4_load3(&lo, plo);
    w4_load3(&hi, phi);
    int azero = w4_is_zero(&lo) && w4_is_zero(&hi);
    if (!normal) {
        if ((!x && !r) || azero) return;
        ++*count;
        int neg = x < 0;
        w4 *a1 = neg ? &hi : &lo, *a2 = neg ? &lo : &hi, z = {0};
        if (sub) {
            if (!neg ? !w4_negative(&lo) : w4_negative(&hi)) a1 = &z;
            if (!neg ? w4_negative(&hi) : !w4_negative(&lo)) a2 = &z;
        }
        w4_smul(&t, a1, x);
        w4_add(&v[ANY_LO], &t);
        w4_smul(&t, a2, x);
        w4_add(&v[ANY_HI], &t);
        w4 ml = {0}, mh = {0};
        w4_add_abs(&ml, &lo);
        w4_add_abs(&mh, &hi);
        w4 *big = w4_less(&ml, &mh) ? &mh : &ml;
        w4_mul(&t, big, x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x);
        w4_add(&v[ANY_MASS], &t);
        if (r) {
            /* the ends widen by max(|lo|, |hi|) r, and the magnitude takes it */
            w4_mul(&t, big, r);
            w4_sub(&v[ANY_LO], &t);
            w4_add(&v[ANY_HI], &t);
            w4_add(&v[ANY_MASS], &t);
        }
        return;
    }
    w4 *s = &v[SUM_S], *lane = &v[SUM_P4 + (k & 3)];
    if ((x || r) && !azero) {
        /* a = lo + hi, ra = hi - lo; b = 2x, rb = 0, or with a flush b = x, rb = |x|; rb takes 2r */
        w4 a = {0}, ra = {0};
        w4_add(&a, &lo);
        w4_add(&a, &hi);
        w4_add(&ra, &hi);
        w4_sub(&ra, &lo);
        uint64_t mx = x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x, mb = sub ? mx : 2 * mx, rb = (sub ? mx : 0) + 2 * r;
        int64_t b = sub ? x : 2 * x;
        w4_smul(&t, &a, b);
        if (!w4_is_zero(&ra) || rb) {
            w4 d, e, am = {0};
            w4_mul(&d, &ra, mb + rb);
            if (rb) {
                w4_add_abs(&am, &a);
                w4_mul(&e, &am, rb);
                w4_add(&d, &e);
            }
            w4_add(&v[SUM_RAD], &d);
        }
        w4_add(s, &t);
        w4_add_abs(&v[SUM_MASS], &t);
        w4_add(lane, &t);
    }
    if (k >= 1) w4_add_abs(&v[SUM_N1], s);
    if (k >= 4) w4_add_abs(&v[SUM_N4], lane);
    if (k >= 2) {
        w4 p2 = {0};
        w4_add(&p2, &v[SUM_P4 + (k & 1)]);
        w4_add(&p2, &v[SUM_P4 + (k & 1) + 2]);
        w4_add_abs(&v[SUM_N2], &p2);
    }
}

/* k = k0 .. S - 1 zero terms after the last nonzero one of a normal sum (four-word values), at once: each adds the nodes
   |s|, its lane of two's and of four's */
static void zero_tail4(uint64_t *st, int64_t k0, int64_t S) {
    if (k0 >= S) return;
    w4 *v = (w4 *)st, s = {0}, t;
    w4_add_abs(&s, &v[SUM_S]);
    w4_mul(&t, &s, (uint64_t)(S - (k0 > 1 ? k0 : 1)));
    w4_add(&v[SUM_N1], &t);
    for (int l = 0; l < 4; l++) {
        /* the terms of lane l (k = l mod 4) in k0 .. S - 1 at k >= 4, and of lane l mod 2 at k >= 2 */
        int64_t from = k0 > 4 ? k0 : 4, first = from + ((l - from % 4) + 4) % 4, n = first < S ? (S - 1 - first) / 4 + 1 : 0;
        if (n) {
            w4 m = {0};
            w4_add_abs(&m, &v[SUM_P4 + l]);
            w4_mul(&t, &m, (uint64_t)n);
            w4_add(&v[SUM_N4], &t);
        }
    }
    for (int l = 0; l < 2; l++) {
        int64_t from = k0 > 2 ? k0 : 2, first = from + ((l - from % 2) + 2) % 2, n = first < S ? (S - 1 - first) / 2 + 1 : 0;
        if (n) {
            w4 p2 = {0}, m = {0};
            w4_add(&p2, &v[SUM_P4 + l]);
            w4_add(&p2, &v[SUM_P4 + l + 2]);
            w4_add_abs(&m, &p2);
            w4_mul(&t, &m, (uint64_t)n);
            w4_add(&v[SUM_N2], &t);
        }
    }
}

/* scratch words for add_term and the finish at n words per value, nx per factor */
static size_t term_scratch(int n, int nx) { return (size_t)(12 * n + 5 * nx + 16); }

/* The tile's terms into the sums of keys j0 to j1: grad_value's (P' times the output gradient's row) and grad_key's (dS'
   times the query row), over the tile's rows in order. */
int att_gtol_tile_keys(att_gtol_group *g, const att_gtol_call *c, size_t j0, size_t j1) {
    size_t S = c->a.S, E = (size_t)c->a.E, Ev = c->a.Ev, T = g->T;
    int normal = c->gr->sums_normal, nv = sum_values(c->gr), wp = c->wp, wd = c->wd;
    int wsv = c->ws[0], wsk = c->ws[1], wxg = c->wx[0], wxq = c->wx[1];
    size_t sz = term_scratch(wsv > wsk ? wsv : wsk, wxg > wxq ? wxg : wxq);
    uint64_t *tmp = malloc(sz * sizeof *tmp);
    if (!tmp) return EXACTREF_NOMEM;
    int rc = 0;
    for (size_t j = j0; !rc && j < j1; j++) {
        for (size_t r = 0; !rc && r < T; r++) {
            const int64_t *ri = g->rinfo + ATT_GR_INFO * r;
            int64_t kv = g->vb->rows + (int64_t)r, kk = g->kb->rows + (int64_t)r;
            size_t at = r * S + j;
            const uint64_t *pl = g->plo + at * (size_t)wp, *ph = g->phi + at * (size_t)wp, *dl = g->dlo + at * (size_t)wd, *dh = g->dhi + at * (size_t)wd;
            /* a sum whose terms so far are all zero is zero: its leading zero terms change nothing */
            int pz = mw_zero(pl, wp) && mw_zero(ph, wp), dz = mw_zero(dl, wd) && mw_zero(dh, wd);
            g->vb->started[j] |= !pz;
            g->kb->started[j] |= !dz;
            if (g->vb->started[j] && (normal || !pz))
                for (size_t col = 0; col < Ev; col++) {
                    size_t xa = (size_t)ri[4] * Ev + col, el = j * Ev + col;
                    uint64_t *st = g->vb->sum + el * (size_t)nv * (size_t)wsv;
                    if (c->fast)
                        add_term4(st, g->vb->count + el, normal, pl, ph, word_signed(c->x[0][xa]), c->xr[0] ? c->xr[0][xa] : 0, c->xsub[0][xa], kv);
                    else
                        add_term(st, g->vb->count + el, wsv, normal, pl, ph, wp, c->x[0] + xa * (size_t)wxg, c->xr[0] ? c->xr[0] + xa * (size_t)wxg : NULL, wxg,
                                 c->xsub[0][xa], kv, tmp);
                }
            if (g->kb->dsum && !dz && (rc = add_size(&g->kb->dsum[j], dl, dh, wd, c->ud, tmp))) break;
            if (g->kb->started[j] && (normal || !dz))
                for (size_t e = 0; e < E; e++) {
                    size_t xa = (size_t)ri[0] * E + e, el = j * E + e;
                    uint64_t *st = g->kb->sum + el * (size_t)nv * (size_t)wsk;
                    if (c->fast)
                        add_term4(st, g->kb->count + el, normal, dl, dh, word_signed(c->x[1][xa]), c->xr[1] ? c->xr[1][xa] : 0, c->xsub[1][xa], kk);
                    else
                        add_term(st, g->kb->count + el, wsk, normal, dl, dh, wd, c->x[1] + xa * (size_t)wxq, c->xr[1] ? c->xr[1] + xa * (size_t)wxq : NULL, wxq,
                                 c->xsub[1][xa], kk, tmp);
                }
        }
    }
    free(tmp);
    return rc;
}

/* One element's value before Lemma 3 from its sums (n words per value, at 2**e: twice the midpoints' scale under the
   normal standard) over terms terms (normal) or count that can be nonzero (any order): the gradient's accumulation (sb:
   Lemma S's constants for terms terms), its output rounding and the scale (grad_query and grad_key unless scale_ds); *unb
   where it has no bound (set already: no bound). */
static int sum_interval(ivl *x, int *unb, const att_gtol_call *c, int which, const uint64_t *st, int64_t count, int n, int64_t e, const sum_bounds *sb,
                        uint64_t *tmp) {
    const att_gtol *gr = c->gr;
    int64_t wp = gr->w;
    int rc = 0, lean_out = which == 0 ? ATT_GL_OUT_V : which == 1 ? ATT_GL_OUT_K : ATT_GL_OUT_Q, acc = which == 0 ? ATT_GA_V : which == 1 ? ATT_GA_K : ATT_GA_Q;
    bigint b[4];
    for (int z = 0; z < 4; z++) big_init(&b[z]);
    if (!*unb && gr->sums_normal) {
        /* N = 2 (mass + the largest node sum), then Lemma S with mass + R, N + 2R, moved R */
        uint64_t *N = tmp, *t2 = N + n, *o = t2 + 2 * n, *sc = o + n;
        memcpy(t2, st + SUM_P4 * n, (size_t)n * sizeof *t2);
        mw_add(t2, st + (SUM_P4 + 2) * n, n);
        memcpy(t2 + n, st + (SUM_P4 + 1) * n, (size_t)n * sizeof *t2);
        mw_add(t2 + n, st + (SUM_P4 + 3) * n, n);
        int64_t terms = count;
        lane_nodes_words(o, t2, terms < 2 ? (1u << terms) - 1 : 3u, 2, (size_t)n, sc);
        mw_add(o, st + SUM_N2 * n, n);
        memcpy(N, st + SUM_N1 * n, (size_t)n * sizeof *N);
        if (nat_cmp(o, N, (size_t)n) > 0) memcpy(N, o, (size_t)n * sizeof *N);
        lane_nodes_words(o, st + SUM_P4 * n, terms < 4 ? (1u << terms) - 1 : 15u, 4, (size_t)n, sc);
        mw_add(o, st + SUM_N4 * n, n);
        if (nat_cmp(o, N, (size_t)n) > 0) memcpy(N, o, (size_t)n * sizeof *N);
        mw_add(N, st + SUM_MASS * n, n);
        mw_add(N, N, n); /* 2 (mass + nodes) */
        mw_add(N, st + SUM_RAD * n, n);
        mw_add(N, st + SUM_RAD * n, n); /* + 2R */
        memcpy(o, st + SUM_MASS * n, (size_t)n * sizeof *o);
        mw_add(o, st + SUM_RAD * n, n); /* mass + R */
        int checked = 1, flags = 0;
        if (!(rc = mw_to_big(&b[0], st + SUM_S * n, n, sc)) && !(rc = mw_to_big(&b[1], o, n, sc)) && !(rc = mw_to_big(&b[2], N, n, sc)) &&
            !(rc = mw_to_big(&b[3], st + SUM_RAD * n, n, sc)))
            rc = sum_bounds_normal(&x->lo, &x->hi, &checked, &flags, sb, &b[0], &b[1], &b[2], &b[3], e);
        if (!rc && !checked && flags) *unb = 1;
    } else if (!*unb) {
        uint64_t *sc = tmp;
        dyad m;
        dyad_init(&m);
        if (!(rc = mw_to_dyad(&x->lo, st + ANY_LO * n, n, e, sc)) && !(rc = mw_to_dyad(&x->hi, st + ANY_HI * n, n, e, sc)) &&
            !(rc = mw_to_dyad(&m, st + ANY_MASS * n, n, e, sc))) {
            if (count) rc = acc_apply(x, &gr->acc[acc], (size_t)count, &m, wp, unb); /* no term that can be nonzero: exactly 0 */
        }
        dyad_free(&m);
    }
    if (!rc && !*unb) rc = lean_ivl(x, &gr->lean[lean_out], wp, unb);
    if (!rc && !*unb && which != 0 && !(gr->methods & ATT_GM_SCALE_DS)) {
        /* times the scale held in the compute format, rounded */
        if (!gr->scale_bounded) {
            *unb = 1;
        } else {
            ivl sig;
            sig.lo = gr->scale_lo; /* read-only views */
            sig.hi = gr->scale_hi;
            if (!(rc = ivl_mul(x, x, &sig, wp))) rc = lean_ivl(x, &gr->lean[ATT_GL_SCALE], wp, unb);
        }
    }
    for (int z = 0; z < 4; z++) big_free(&b[z]);
    return rc;
}

/* Lemma 3 for an element's value x (none where unb) against its reference ref, into its four parts; nan where the terms
   of the keys a row does not keep may make it NaN (Masked keys) */
static int final_element(unsigned char *tol, unsigned char *low, unsigned char *high, unsigned char *flag, const att_gtol_call *c, const ivl *x, int unb,
                         int nan, const unsigned char *ref) {
    bigint tb, lb, hb, rb;
    big_init(&tb);
    big_init(&lb);
    big_init(&hb);
    big_init(&rb);
    int fl = 0, rc = bytes_load(&rb, ref, c->out_size, 0);
    /* no bound with an input that is an earlier result: any value may reach the final rounding, NaN and the infinities too
       (a stored input within its radius may be finite where the reference's is not), as in the forward */
    int any = (unb && (c->a.uncertain || c->rv || c->rg || c->ro || c->lrad) ? FLAG_ANY : 0) | (nan ? FLAG_NAN : 0);
    if (!rc && !(rc = final_bounds(&tb, &lb, &hb, &fl, &c->out, &c->f64, c->rnd, c->ovf, unb ? NULL : &x->lo, unb ? NULL : &x->hi, &rb, c->gr->w, any))) {
        bytes_store(tol, 8, &tb);
        bytes_store(low, c->out_size, &lb);
        bytes_store(high, c->out_size, &hb);
        *flag = (unsigned char)fl;
    }
    big_free(&tb);
    big_free(&lb);
    big_free(&hb);
    big_free(&rb);
    return rc;
}

/* One element's bound from its sums: sum_interval, then Lemma 3 against ref (nan as final_element's) */
static int finish_element(unsigned char *tol, unsigned char *low, unsigned char *high, unsigned char *flag, const att_gtol_call *c, int which,
                          const uint64_t *st, int64_t count, int n, int64_t e, int unb, int nan, const sum_bounds *sb, const unsigned char *ref, uint64_t *tmp) {
    ivl x;
    ivl_init(&x);
    int rc = sum_interval(&x, &unb, c, which, st, count, n, e, sb, tmp);
    if (!rc) rc = final_element(tol, low, high, flag, c, &x, unb, nan, ref);
    ivl_free(&x);
    return rc;
}

/* Lemma C (foundations.md; attention_algorithms.md, backward_blocks): one block's exact sums [lo, hi] (times 2**e as
   words, n of them per value), its mass and n terms that can be nonzero, and with a scale (grad_query and grad_key) its
   factors' magnitudes F, added to an element's value carried in the output format: c (lo, hi), status 0 for none yet, 1
   for a value, 2 for no bound. */
void att_carry_init(att_carry *k) {
    memset(k, 0, sizeof *k);
    lean_init(&k->round);
    dyad *ds[] = {&k->abs, &k->scale_lo, &k->scale_hi, &k->alpha, &k->alpha_one};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
}

void att_carry_free(att_carry *k) {
    lean_free(&k->round);
    acc_free(&k->acc);
    dyad *ds[] = {&k->abs, &k->scale_lo, &k->scale_hi, &k->alpha, &k->alpha_one};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
}

/* Lemma C's constants for a sum carried in the format of carry, each block summed on the accumulator of acc (any order,
   up to last terms): Lemma 1 at the carry's rounding, Lemma 2's terms, the accumulator's absolute unit, whether the carry
   format's values are the accumulator's, and the scale (with scaled sums) with |scale| and max(1, |scale|) */
int att_carry_build(att_carry *k, const round_to *carry, const att_acc_set *acc, int64_t last, int scale_bounded, const dyad *scale_lo,
                    const dyad *scale_hi, int flush, int64_t w) {
    k->w = w;
    int rc = lean_of(&k->round, carry, flush);
    if (!rc && acc->kind != ATT_SA_NONE) rc = acc_of(&k->acc, acc, last ? 1 : 0, last, w, flush);
    k->scale_bounded = scale_bounded;
    if (!rc && scale_bounded && !(rc = dyad_copy(&k->scale_lo, scale_lo)) && !(rc = dyad_copy(&k->scale_hi, scale_hi)) &&
        !(rc = ends_magnitude(&k->alpha, scale_lo, scale_hi)) && !(rc = dyad_set_si(&k->alpha_one, 1, 0)) && dyad_cmp(&k->alpha, &k->alpha_one) > 0)
        rc = dyad_copy(&k->alpha_one, &k->alpha);
    const floatfmt *f = &acc->f, *cf = &carry->f;
    dyad rel, t1, l1, t2, l2;
    dyad *tmp5[] = {&rel, &t1, &l1, &t2, &l2};
    for (int y = 0; y < 5; y++) dyad_init(tmp5[y]);
    if (!rc && !(rc = error_units(&rel, &k->abs, f, acc->rnd, 0, 0, flush)) && !(rc = floatfmt_threshold(&t1, &l1, f)) && !(rc = floatfmt_threshold(&t2, &l2, cf)))
        k->contained = cf->man_bits <= f->man_bits && cf->emin - cf->man_bits >= f->emin - f->man_bits && dyad_cmp(&l2, &l1) <= 0;
    for (int y = 0; y < 5; y++) dyad_free(tmp5[y]);
    return rc;
}

/* One block into a carried value c (lo, hi; *status 0 none yet, 1 a value, 2 no bound) by Lemma C: the previous value
   rounded to the carry format, plus the block's exact sums [lo, hi] (scaled: times the scale, with the factors' magnitude
   sum F), within gamma (mass + |previous|) + constant, Lemma 2's terms at count + 2 */
int att_carry_add(dyad c[2], unsigned char *status, const att_carry *k, int scaled, const dyad *lo, const dyad *hi, const dyad *mass_in, int64_t count,
                  const dyad *F) {
    const att_acc *ac = &k->acc;
    int64_t w = k->w;
    int rc = 0, unb = 0;
    if (*status == 2 || (!count && k->contained)) return 0; /* no bound already, or an empty block that leaves the value */
    ivl prev, blk;
    ivl_init(&prev);
    ivl_init(&blk);
    dyad mass, t1, t2, constant;
    dyad *ds[] = {&mass, &t1, &t2, &constant};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    /* the previous value, rounded to the carry format */
    if (*status == 1) {
        if (!(rc = dyad_copy(&prev.lo, &c[0])) && !(rc = dyad_copy(&prev.hi, &c[1]))) rc = lean_ends(&prev.lo, &prev.hi, &unb, &k->round, w);
    } else if (!(rc = dyad_set_si(&prev.lo, 0, 0))) {
        rc = dyad_set_si(&prev.hi, 0, 0);
    }
    if (!rc && !unb && !(rc = dyad_copy(&blk.lo, lo)) && !(rc = dyad_copy(&blk.hi, hi)) && !(rc = dyad_copy(&mass, mass_in)))
        if ((size_t)(count + 2) > ac->nmax || !ac->has[count + 2]) unb = 1;
    const dyad *gamma = !unb && ac->terms ? &ac->terms[3 * (count + 2)] : NULL, *limit = gamma ? gamma + 2 : NULL;
    if (!rc && !unb && !(rc = dyad_copy(&constant, gamma + 1)) && scaled) {
        /* the scaled block: times the scale, the mass times alpha, and the factors' part F abs (1 + gamma), the constant times
           max(1, alpha); a scaled factor past the limit has no bound */
        if (!k->scale_bounded) {
            unb = 1;
        } else {
            ivl sig;
            sig.lo = k->scale_lo; /* read-only views */
            sig.hi = k->scale_hi;
            if (!(rc = ivl_mul(&blk, &blk, &sig, EXACT_BITS)) && !(rc = dyad_mul(&mass, &mass, &k->alpha, w, 1)) && !(rc = dyad_mul(&t1, F, &k->alpha, w, 1))) {
                if (dyad_cmp(&t1, limit) > 0) {
                    unb = 1;
                } else if (!(rc = dyad_set_si(&t2, 1, 0)) && !(rc = dyad_add(&t2, &t2, gamma, w, 1)) && !(rc = dyad_mul(&t1, F, &k->abs, w, 1)) &&
                           !(rc = dyad_mul(&t1, &t1, &t2, w, 1)) && !(rc = dyad_mul(&constant, &constant, &k->alpha_one, w, 1))) {
                    rc = dyad_add(&constant, &constant, &t1, w, 1);
                }
            }
        }
    }
    /* the mass with the carried value's magnitude; the error gamma mass + constant, below the limit */
    if (!rc && !unb && !(rc = ends_magnitude(&t1, &prev.lo, &prev.hi)) && !(rc = dyad_add(&mass, &mass, &t1, w, 1)) && !(rc = dyad_mul(&t1, gamma, &mass, w, 1)) &&
        !(rc = dyad_add(&t1, &t1, &constant, w, 1)) && !(rc = dyad_add(&t2, &mass, &t1, w, 1))) {
        if (dyad_cmp(&t2, limit) > 0) {
            unb = 1;
        } else if (!(rc = dyad_add_exact(&c[0], &prev.lo, &blk.lo)) && !(rc = sub_dir(&c[0], &c[0], &t1, w, 0)) && !(rc = dyad_add_exact(&c[1], &prev.hi, &blk.hi))) {
            rc = dyad_add(&c[1], &c[1], &t1, w, 1);
        }
    }
    if (!rc) *status = unb ? 2 : 1;
    ivl_free(&prev);
    ivl_free(&blk);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return rc;
}

/* A block's sums (st: any order's lo, hi and mass of n words at 2**e) of gradient which (2 grad_query, 1 grad_key, 0
   grad_value) carried into c by Lemma C (att_carry_add) */
static int carry_add(dyad *c, unsigned char *status, const att_gtol_call *call, int which, const uint64_t *st, int n, int64_t e, int64_t count,
                     const dyad *F, uint64_t *tmp) {
    const att_carry *k = &call->gr->carry[which == 2 ? 0 : which == 1 ? 1 : 2];
    if (*status == 2 || (!count && k->contained)) return 0;
    dyad lo, hi, mass;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&mass);
    int rc = mw_to_dyad(&lo, st + ANY_LO * n, n, e, tmp);
    if (!rc) rc = mw_to_dyad(&hi, st + ANY_HI * n, n, e, tmp);
    if (!rc) rc = mw_to_dyad(&mass, st + ANY_MASS * n, n, e, tmp);
    if (!rc) rc = att_carry_add(c, status, k, which != 0, &lo, &hi, &mass, count, F);
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&mass);
    return rc;
}

/* backward_blocks: the block's grad_key and grad_value sums of keys j0 to j1 added to their carried values (Lemma C, the
   block's factors for grad_key its keys' sums of |dS'| and its columns' sums of |q|), then cleared for the next block */
int att_gtol_group_carry(att_gtol_group *g, const att_gtol_call *c, size_t j0, size_t j1) {
    size_t E = (size_t)c->a.E, Ev = c->a.Ev, S = c->a.S;
    att_gtol_block *kb = g->kb, *vb = g->vb;
    int rc = 0, wk = c->ws[1], wv = c->ws[0];
    int64_t ek = c->ud + c->ux[1], ev = c->up + c->ux[0];
    uint64_t *tmp = malloc(term_scratch(wk > wv ? wk : wv, 2) * sizeof *tmp);
    if (!tmp) return EXACTREF_NOMEM;
    dyad F;
    dyad_init(&F);
    for (size_t j = j0; !rc && j < j1; j++) {
        for (size_t e = 0; !rc && e < E; e++) {
            size_t el = j * E + e;
            uint64_t *st = kb->sum + el * ANY_VALUES * (size_t)wk;
            if (!(rc = dyad_add_exact(&F, &kb->dsum[j], &kb->dsum[S + e])))
                rc = carry_add(&kb->carried[2 * el], &kb->cstatus[el], c, 1, st, wk, ek, kb->count[el], &F, tmp);
            memset(st, 0, ANY_VALUES * (size_t)wk * sizeof *st);
            kb->count[el] = 0;
        }
        for (size_t col = 0; !rc && col < Ev; col++) {
            size_t el = j * Ev + col;
            uint64_t *st = vb->sum + el * ANY_VALUES * (size_t)wv;
            rc = carry_add(&vb->carried[2 * el], &vb->cstatus[el], c, 0, st, wv, ev, vb->count[el], NULL, tmp);
            memset(st, 0, ANY_VALUES * (size_t)wv * sizeof *st);
            vb->count[el] = 0;
        }
        if (!rc) rc = dyad_set_si(&kb->dsum[j], 0, 0);
        kb->started[j] = vb->started[j] = 0;
    }
    dyad_free(&F);
    free(tmp);
    return rc;
}

/* backward_blocks: after a block's carry, its columns' sums of |q| cleared */
int att_gtol_group_carried(att_gtol_group *g) {
    att_gtol_block *kb = g->kb;
    size_t S = kb->dcount - kb->cols;
    int rc = 0;
    for (size_t e = 0; !rc && e < kb->cols; e++) rc = dyad_set_si(&kb->dsum[S + e], 0, 0);
    return rc;
}

/* grouped_heads (attention_algorithms.md): the head's grad_key and grad_value sums of keys j0 to j1, each through its
   accumulation, output rounding and scale, rounded to the output format (the group rounding) and added to its element's
   sum over the heads; the head's sums then cleared for the next head's rows. */
int att_gtol_group_fold(att_gtol_group *g, const att_gtol_call *c, size_t j0, size_t j1) {
    size_t E = (size_t)c->a.E, Ev = c->a.Ev;
    const att_gtol *gr = c->gr;
    int nv = sum_values(gr), rc = 0, wk = c->ws[1], wv = c->ws[0];
    int64_t ek = c->ud + c->ux[1], ev = c->up + c->ux[0];
    uint64_t *tmp = malloc(term_scratch(wk > wv ? wk : wv, 2) * sizeof *tmp);
    if (!tmp) return EXACTREF_NOMEM;
    ivl x;
    ivl_init(&x);
    dyad m;
    dyad_init(&m);
    for (size_t j = j0; !rc && j < j1; j++) {
        for (int which = 1; !rc && which >= 0; which--) {
            att_gtol_block *b = which ? g->kb : g->vb;
            size_t cols = which ? E : Ev;
            int n = which ? wk : wv;
            for (size_t col = 0; !rc && col < cols; col++) {
                size_t el = j * cols + col;
                uint64_t *st = b->sum + el * (size_t)nv * (size_t)n;
                int64_t *count = b->count + el;
                int unb = b->unb || b->gunb[el];
                if (!unb && !(rc = sum_interval(&x, &unb, c, which, st, *count, n, which ? ek : ev, NULL, tmp)) && !unb)
                    rc = lean_ivl(&x, &gr->lean[which ? ATT_GL_GROUP_K : ATT_GL_GROUP_V], gr->w, &unb);
                if (!rc && unb) b->gunb[el] = 1;
                if (!rc && !unb && !(rc = dyad_add_exact(&b->group[3 * el], &b->group[3 * el], &x.lo)) &&
                    !(rc = dyad_add_exact(&b->group[3 * el + 1], &b->group[3 * el + 1], &x.hi)) && !(rc = ends_magnitude(&m, &x.lo, &x.hi)))
                    rc = dyad_add_exact(&b->group[3 * el + 2], &b->group[3 * el + 2], &m);
                memset(st, 0, (size_t)nv * (size_t)n * sizeof *st);
                *count = 0;
            }
        }
        g->kb->started[j] = g->vb->started[j] = 0;
    }
    ivl_free(&x);
    dyad_free(&m);
    free(tmp);
    return rc;
}

/* grouped_heads: one more head folded into the blocks' sums, which start again for the next head's rows */
void att_gtol_group_folded(att_gtol_group *g) {
    g->kb->parts++;
    g->vb->parts++;
    g->kb->rows = g->vb->rows = 0;
}

/* grad_query of the tile's rows r0 to r1: dS' times the key block's rows, over the keys in order, finished into the
   outputs (E elements per row, from tol, low, high and flags at row r - r0). */
int att_gtol_tile_queries(const att_gtol_group *g, att_gtol_call *c, size_t r0, size_t r1, unsigned char *tol, unsigned char *low,
                          unsigned char *high, unsigned char *flags) {
    size_t S = c->a.S, E = (size_t)c->a.E;
    int normal = c->gr->sums_normal, nv = sum_values(c->gr), n = c->ws[2], wd = c->wd, wx = c->wx[2], rc = 0;
    uint64_t *st = malloc((size_t)nv * (size_t)n * sizeof *st), *tmp = malloc(term_scratch(n, wx) * sizeof *tmp);
    sum_bounds sb;
    int sb_set = 0;
    /* backward_blocks: each column's carried value and status, the block's sum of |dS'|, a factor sum and the value */
    dyad *cval = c->gr->key_block ? malloc(2 * (E ? E : 1) * sizeof *cval) : NULL, dsz, F;
    unsigned char *cstat = c->gr->key_block ? malloc(E ? E : 1) : NULL, *qn = malloc(E ? E : 1); /* qn: per column, may be NaN */
    ivl xv;
    ivl_init(&xv);
    dyad_init(&dsz);
    dyad_init(&F);
    for (size_t z = 0; cval && z < 2 * E; z++) dyad_init(&cval[z]);
    if (!st || !tmp || !qn || (c->gr->key_block && (!cval || !cstat))) rc = EXACTREF_NOMEM;
    if (!rc && normal && !(rc = sum_bounds_init(&sb, &c->gr->q_f, c->gr->flush, (int64_t)S, S ? (int64_t)S - 1 : 0, 0, c->gr->w))) sb_set = 1;
    sum_bounds csb; /* Lemma S's constants over a broadcast row's copies' terms */
    int64_t copies_sb = 0;
    int64_t e = c->ud + c->ux[2] - (normal ? 2 : 0);
    for (size_t r = r0; !rc && r < r1; r++) {
        const int64_t *ri = g->rinfo + ATT_GR_INFO * r;
        int unb = (g->flags[r] & (ATT_GF_P_UNB | ATT_GF_DS_UNB)) != 0;
        /* the NaN the terms of the keys the row does not keep may give (Masked keys): every column where P' or D' may be
           NaN, or one pair's P' or dS' (with a NaN row, or D not finite, the reference is NaN), and at a key removed by a
           -inf score (0 times its entries) the columns of its entries that are not finite */
        memset(qn, (g->flags[r] & (ATT_GF_NAN | ATT_GF_P_NAN | ATT_GF_D_NAN | ATT_GF_DS_SKIP | ATT_GF_Q_NAN)) != 0, E);
        for (size_t j = 0; j < S; j++)
            for (size_t col = 0; (g->kinds[r * S + j] & ATT_KEY_KIND) == ATT_KEY_REMOVED && c->ks->special_rows[g->kr0 + j] && col < E; col++) {
                int kind, neg;
                uint64_t sig;
                int64_t ex;
                element_decode_slot(&kind, &neg, &sig, &ex, c->ks, (g->kr0 + j) * E + col);
                if (kind != VALUE_FINITE) qn[col] = 1;
            }
        /* a query row broadcast over several key and value blocks: its sums in the call's slot from copy to copy, its terms
           after the earlier copies' (index copy S + key), finished at its last copy over S copies terms */
        int64_t slot = ri[5], copy = ri[6], copies = ri[7], base = copy * (int64_t)S;
        if (slot >= 0) {
            if (copy == 0) {
                c->qunb[slot] = 0;
                memset(c->qnan + (size_t)slot * E, 0, E);
            }
            c->qunb[slot] |= (unsigned char)unb;
            unb = c->qunb[slot];
            for (size_t col = 0; col < E; col++) qn[col] = c->qnan[(size_t)slot * E + col] |= qn[col];
        }
        /* the row's first and last keys with a dS' not zero: before them the sums are zero, after them zero terms */
        size_t first = S, last = 0;
        for (size_t j = 0; !unb && j < S; j++) {
            size_t at = r * S + j;
            if (mw_zero(g->dlo + at * (size_t)wd, wd) && mw_zero(g->dhi + at * (size_t)wd, wd)) continue;
            if (first == S) first = j;
            last = j;
        }
        if (c->gr->key_block) {
            /* backward_blocks: per block of keys its sums carried in the output format (Lemma C), its factors the block's
               sum of |dS'| and each column's sum of |k| */
            size_t B = (size_t)c->gr->key_block;
            for (size_t z = 0; z < E; z++) cstat[z] = 0;
            for (size_t b0 = 0; !rc && !unb && b0 < S; b0 += B) {
                size_t b1 = b0 + B < S ? b0 + B : S;
                if (!(rc = dyad_set_si(&dsz, 0, 0)))
                    for (size_t j = b0; !rc && j < b1; j++) {
                        size_t at = r * S + j;
                        rc = add_size(&dsz, g->dlo + at * (size_t)wd, g->dhi + at * (size_t)wd, wd, c->ud, tmp);
                    }
                for (size_t col = 0; !rc && col < E; col++) {
                    memset(st, 0, (size_t)nv * (size_t)n * sizeof *st);
                    int64_t count = 0;
                    if (!(rc = dyad_copy(&F, &dsz)))
                        for (size_t j = b0; !rc && j < b1; j++) {
                            size_t at = r * S + j, xa = (g->kr0 + j) * E + col;
                            const uint64_t *xk = c->x[2] + xa * (size_t)wx, *rk = c->xr[2] ? c->xr[2] + xa * (size_t)wx : NULL;
                            if (c->fast)
                                add_term4(st, &count, 0, g->dlo + at * (size_t)wd, g->dhi + at * (size_t)wd, word_signed(*xk), rk ? *rk : 0, c->xsub[2][xa], (int64_t)j);
                            else
                                add_term(st, &count, n, 0, g->dlo + at * (size_t)wd, g->dhi + at * (size_t)wd, wd, xk, rk, wx, c->xsub[2][xa], (int64_t)j, tmp);
                            /* |k| + r */
                            if (!(rc = add_size(&F, xk, xk, wx, c->ux[2], tmp)) && rk) rc = add_size(&F, rk, rk, wx, c->ux[2], tmp);
                        }
                    if (!rc) rc = carry_add(&cval[2 * col], &cstat[col], c, 2, st, n, e, count, &F, tmp);
                }
            }
            for (size_t col = 0; !rc && col < E; col++) {
                size_t o = (r - r0) * E + col, el = (size_t)ri[0] * E + col;
                int none = unb || cstat[col] == 2;
                if (!none && cstat[col] == 0 && !(rc = dyad_set_si(&xv.lo, 0, 0))) rc = dyad_set_si(&xv.hi, 0, 0);
                if (!rc && !none && cstat[col] == 1 && !(rc = dyad_copy(&xv.lo, &cval[2 * col]))) rc = dyad_copy(&xv.hi, &cval[2 * col + 1]);
                if (!rc) rc = final_element(tol + 8 * o, low + o * c->out_size, high + o * c->out_size, flags + o, c, &xv, none, qn[col], c->ref[2] + el * c->out_size);
            }
            continue;
        }
        /* after an earlier copy the sums may be nonzero: every key's term counts from the row's first */
        if (slot >= 0 && copy > 0 && first < S) first = 0;
        for (size_t col = 0; !rc && col < E; col++) {
            uint64_t *sc = st;
            int64_t own = 0, *count = &own;
            if (slot >= 0) {
                sc = c->qstates + ((size_t)slot * E + col) * (size_t)nv * (size_t)n;
                count = c->qcounts + (size_t)slot * E + col;
                if (copy == 0) {
                    memset(sc, 0, (size_t)nv * (size_t)n * sizeof *sc);
                    *count = 0;
                }
            } else {
                memset(sc, 0, (size_t)nv * (size_t)n * sizeof *sc);
            }
            for (size_t j = first; !unb && j <= last && j < S; j++) {
                size_t at = r * S + j, xa = (g->kr0 + j) * E + col;
                if (c->fast)
                    add_term4(sc, count, normal, g->dlo + at * (size_t)wd, g->dhi + at * (size_t)wd, word_signed(c->x[2][xa]), c->xr[2] ? c->xr[2][xa] : 0,
                              c->xsub[2][xa], base + (int64_t)j);
                else
                    add_term(sc, count, n, normal, g->dlo + at * (size_t)wd, g->dhi + at * (size_t)wd, wd, c->x[2] + xa * (size_t)wx,
                             c->xr[2] ? c->xr[2] + xa * (size_t)wx : NULL, wx, c->xsub[2][xa], base + (int64_t)j, tmp);
            }
            if (normal && !unb && (first < S || (slot >= 0 && copy > 0))) {
                /* the zero terms after the row's last nonzero one (every term of a row with none, after an earlier copy) */
                int64_t k0 = base + (first < S ? (int64_t)last + 1 : 0), kend = base + (int64_t)S;
                if (c->fast) {
                    zero_tail4(sc, k0, kend);
                } else {
                    uint64_t zero[2] = {0, 0};
                    for (int64_t k = k0; k < kend; k++) add_term(sc, count, n, normal, zero, zero, 1, zero, NULL, 1, 0, k, tmp);
                }
            }
            if (slot >= 0 && copy < copies - 1) continue; /* finished at its last copy */
            if (slot >= 0 && normal && copies > 1 && !copies_sb) {
                if ((rc = sum_bounds_init(&csb, &c->gr->q_f, c->gr->flush, (int64_t)S * copies, (int64_t)S * copies - 1, 0, c->gr->w))) break;
                copies_sb = copies;
            } else if (slot >= 0 && normal && copies_sb && copies_sb != copies) {
                sum_bounds_free(&csb);
                copies_sb = 0;
                if ((rc = sum_bounds_init(&csb, &c->gr->q_f, c->gr->flush, (int64_t)S * copies, (int64_t)S * copies - 1, 0, c->gr->w))) break;
                copies_sb = copies;
            }
            const sum_bounds *use = slot >= 0 && normal ? &csb : sb_set ? &sb : NULL;
            size_t o = (r - r0) * E + col, el = (size_t)ri[0] * E + col;
            rc = finish_element(tol + 8 * o, low + o * c->out_size, high + o * c->out_size, flags + o, c, 2, sc, normal ? (int64_t)S * (slot >= 0 ? copies : 1) : *count,
                                n, e, unb, qn[col], use, c->ref[2] + el * c->out_size, tmp);
        }
        continue;
    }
    if (sb_set) sum_bounds_free(&sb);
    if (copies_sb) sum_bounds_free(&csb);
    free(st);
    free(tmp);
    for (size_t z = 0; cval && z < 2 * E; z++) dyad_free(&cval[z]);
    free(cval);
    free(cstat);
    free(qn);
    ivl_free(&xv);
    dyad_free(&dsz);
    dyad_free(&F);
    return rc;
}

/* After a tile's phases: its rows join the group's count, their flags its blocks' (the NaN of Masked keys: a row whose
   P' may be NaN reaches every element of both blocks, one whose D' may be NaN every grad_key element, a pair whose dS'
   may be NaN its key's grad_key and one whose P' may be NaN also its grad_value, a P' that may be 0 (at a key the row
   does not keep, or P''s lower end at most 0) times a non-finite entry of the output gradient's row that key's
   grad_value column, and 0 times a non-finite entry of the query row at a key removed by a -inf score its grad_key
   column); with backward_blocks their query rows' |q| join the columns' sums */
int att_gtol_tile_done(att_gtol_group *g, const att_gtol_call *c) {
    int rc = 0;
    size_t E = (size_t)c->a.E, S = c->a.S, Ev = c->a.Ev;
    att_gtol_block *kb = g->kb, *vb = g->vb;
    for (size_t r = 0; r < g->T; r++) {
        unsigned char f = g->flags[r];
        if (f & ATT_GF_P_UNB) vb->unb = kb->unb = 1;
        if (f & ATT_GF_DS_UNB) kb->unb = 1;
        if (f & ATT_GF_G_UNB) vb->unb = 1;
        if (f & (ATT_GF_NAN | ATT_GF_P_NAN)) vb->nan = kb->nan = 1;
        if (f & (ATT_GF_D_NAN | ATT_GF_DS_SKIP)) kb->nan = 1;
        size_t qr = (size_t)g->rinfo[ATT_GR_INFO * r], gr0 = (size_t)g->rinfo[ATT_GR_INFO * r + 4];
        int gsp = c->gs->special_rows[gr0], qsp = c->qs->special_rows[qr], wp = c->wp;
        if ((f & (ATT_GF_NAN | ATT_GF_P_NAN | ATT_GF_P_UNB)) || (!gsp && !qsp && !(f & ATT_GF_Q_NAN))) continue;
        for (size_t j = 0; j < S; j++) {
            unsigned char mark = g->kinds[r * S + j], kind = mark & ATT_KEY_KIND;
            int neg;
            uint64_t sig;
            int64_t ex;
            if (mark & (ATT_KEY_DS_NAN | ATT_KEY_P_NAN)) memset(kb->nanel + j * E, 1, E);
            if (mark & ATT_KEY_P_NAN) memset(vb->nanel + j * Ev, 1, Ev);
            /* case 4 where the pair's P' lower end is at most 0: a key the row does not keep (P' = 0), or a kept one */
            const uint64_t *pl = g->plo + (r * S + j) * (size_t)wp;
            int pzero = mw_zero(pl, wp) || mw_negative(pl, wp);
            for (size_t col = 0; pzero && gsp && col < Ev; col++) {
                int k;
                element_decode_slot(&k, &neg, &sig, &ex, c->gs, gr0 * Ev + col);
                if (k != VALUE_FINITE) vb->nanel[j * Ev + col] = 1;
            }
            for (size_t col = 0; kind == ATT_KEY_REMOVED && qsp && col < E; col++) {
                int k;
                element_decode_slot(&k, &neg, &sig, &ex, c->qs, qr * E + col);
                if (k != VALUE_FINITE) kb->nanel[j * E + col] = 1;
            }
        }
    }
    if (kb->dsum) {
        uint64_t *tmp = malloc((size_t)(c->wx[1] + 2) * sizeof *tmp);
        if (!tmp) return EXACTREF_NOMEM;
        for (size_t r = 0; !rc && r < g->T; r++) {
            size_t row = (size_t)g->rinfo[ATT_GR_INFO * r] * E * (size_t)c->wx[1];
            const uint64_t *x = c->x[1] + row, *xr = c->xr[1] ? c->xr[1] + row : NULL;
            /* |q| + r */
            for (size_t e = 0; !rc && e < E; e++) {
                size_t at = e * (size_t)c->wx[1];
                if (!(rc = add_size(&kb->dsum[S + e], x + at, x + at, c->wx[1], c->ux[1], tmp)) && xr)
                    rc = add_size(&kb->dsum[S + e], xr + at, xr + at, c->wx[1], c->ux[1], tmp);
            }
        }
        free(tmp);
    }
    kb->rows += (int64_t)g->T;
    vb->rows += (int64_t)g->T;
    return rc;
}

/* A block's grad_key (key) or grad_value elements of keys j0 to j1, after the last unit that adds to it, finished into the
   outputs (cols per key from j0) */
int att_gtol_block_finish(const att_gtol_block *b, const att_gtol_call *c, int key, size_t j0, size_t j1, unsigned char *tol, unsigned char *low,
                          unsigned char *high, unsigned char *flags) {
    const att_gtol *gr = c->gr;
    int normal = gr->sums_normal, nv = sum_values(gr), rc = 0, n = b->ws;
    uint64_t *tmp = malloc(term_scratch(n, 2) * sizeof *tmp);
    sum_bounds sb;
    int set = 0;
    if (!tmp) rc = EXACTREF_NOMEM;
    if (!rc && normal && !(rc = sum_bounds_init(&sb, key ? &gr->k_f : &gr->v_f, gr->flush, b->rows, b->rows > 0 ? b->rows - 1 : 0, 0, gr->w))) set = 1;
    int64_t e = key ? c->ud + c->ux[1] - (normal ? 2 : 0) : c->up + c->ux[0] - (normal ? 2 : 0);
    const unsigned char *ref = c->ref[key ? 1 : 0] + b->r0 * b->cols * c->out_size;
    ivl x;
    ivl_init(&x);
    for (size_t el = j0 * b->cols; !rc && el < j1 * b->cols; el++) {
        size_t o = el - j0 * b->cols;
        unsigned char *t = tol + 8 * o, *lo = low + o * c->out_size, *hi = high + o * c->out_size, *f = flags + o;
        if (gr->methods & ATT_GM_GROUPED) {
            /* the sum of the heads' rounded sums, in any order (Lemma 2 over the heads; empty: exactly 0), then Lemma 3 */
            int unb = b->unb || b->gunb[el];
            if (!unb && !(rc = dyad_copy(&x.lo, &b->group[3 * el])) && !(rc = dyad_copy(&x.hi, &b->group[3 * el + 1])) && b->parts)
                rc = acc_apply(&x, &gr->acc[key ? ATT_GA_GROUP_K : ATT_GA_GROUP_V], (size_t)b->parts, &b->group[3 * el + 2], gr->w, &unb);
            if (!rc) rc = final_element(t, lo, hi, f, c, &x, unb, b->nan || b->nanel[el], ref + el * c->out_size);
        } else if (b->carried) {
            /* backward_blocks: the carried value before its last rounding (0 with no block), then Lemma 3 */
            int unb = b->unb || b->cstatus[el] == 2;
            if (!unb && b->cstatus[el] == 1 && !(rc = dyad_copy(&x.lo, &b->carried[2 * el]))) rc = dyad_copy(&x.hi, &b->carried[2 * el + 1]);
            if (!unb && b->cstatus[el] == 0 && !(rc = dyad_set_si(&x.lo, 0, 0))) rc = dyad_set_si(&x.hi, 0, 0);
            if (!rc) rc = final_element(t, lo, hi, f, c, &x, unb, b->nan || b->nanel[el], ref + el * c->out_size);
        } else {
            rc = finish_element(t, lo, hi, f, c, key, b->sum + el * (size_t)nv * (size_t)n, normal ? b->rows : b->count[el], n, e, b->unb,
                                b->nan || b->nanel[el], set ? &sb : NULL, ref + el * c->out_size, tmp);
        }
    }
    ivl_free(&x);
    if (set) sum_bounds_free(&sb);
    free(tmp);
    return rc;
}
