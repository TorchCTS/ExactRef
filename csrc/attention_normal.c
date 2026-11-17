/* The attention normal answers: the forward's (proofs/scaled_dot_product_attention.md, The normal answer) and, in the
   section below, the backward's. The forward's, per row of a tile: every step in the compute
   format by the emulated arithmetic of emulate.c (to nearest, ties to even, with the call's flush),
   every sum in index order with Lemma O's slots where it overflows, the logits rounded once from
   their exact value, exp and log correctly rounded (enclosures refined until both ends round
   alike), and the outputs and logsumexp encoded in their formats. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

/* the scale held in the compute format f: a rational sigma = num / den (den > 0) rounded once to nearest, ties to even;
   with den 0, 1/sqrt(E) correctly rounded so, its enclosure refined until both ends round alike (E not a square, so
   1/sqrt(E) is irrational and never a tie); the value as its kind, sign and sig * 2**e (an overflow is an infinity, or
   NaN in a format without one, as every compute rounding) */
int normal_scale(int *kind, int *neg, bigint *sig, int64_t *e, const floatfmt *f, const bigint *num, const bigint *den, int64_t E) {
    bigint bits, other, a;
    big_init(&bits);
    big_init(&other);
    big_init(&a);
    int rc = 0;
    if (den->n) {
        if (!(rc = big_copy(&a, num))) {
            a.neg = 0;
            rc = a.n ? floatfmt_encode_ratio(&bits, f, num->neg, &a, den, ROUND_NEAREST_EVEN, OVERFLOW_NAN)
                     : floatfmt_encode(&bits, f, VALUE_FINITE, 0, &a, 0, ROUND_NEAREST_EVEN, OVERFLOW_NAN);
        }
    } else {
        dyad x, r, one, lo, hi;
        dyad *ds[] = {&x, &r, &one, &lo, &hi};
        for (size_t z = 0; z < 5; z++) dyad_init(ds[z]);
        rc = dyad_set_si(&x, E, 0);
        if (!rc) rc = dyad_set_si(&one, 1, 0);
        for (int64_t p = 2 * (f->man_bits + 1) + 16; !rc; p *= 2) {
            if ((rc = dyad_sqrt(&r, &x, p + 8, 1)) || (rc = dyad_div(&lo, &one, &r, p, 0)) || (rc = dyad_sqrt(&r, &x, p + 8, 0)) ||
                (rc = dyad_div(&hi, &one, &r, p, 1)))
                break;
            if ((rc = floatfmt_encode(&bits, f, VALUE_FINITE, 0, &lo.m, lo.e, ROUND_NEAREST_EVEN, OVERFLOW_NAN)) ||
                (rc = floatfmt_encode(&other, f, VALUE_FINITE, 0, &hi.m, hi.e, ROUND_NEAREST_EVEN, OVERFLOW_NAN)))
                break;
            if (!big_cmp_mag(&bits, &other)) break;
        }
        for (size_t z = 0; z < 5; z++) dyad_free(ds[z]);
    }
    if (!rc) rc = floatfmt_decode(kind, neg, sig, e, f, &bits);
    big_free(&bits);
    big_free(&other);
    big_free(&a);
    return rc;
}

/* The value v, exactly a dyad, rounded once into the compute format (IEEE overflow), then read
   with the flush; an exact zero is +0. */
static int round_once(fvalue *r, const femu *x, const floatfmt *f, const dyad *v) {
    if (!v->m.n) {
        femu_read(r, x, VALUE_FINITE, 0, 0, 0);
        return 0;
    }
    bigint bits, sig, mag;
    big_init(&bits);
    big_init(&sig);
    big_init(&mag);
    int kind, neg;
    int64_t e;
    int rc = big_copy(&mag, &v->m);
    mag.neg = 0;
    if (!rc) rc = floatfmt_encode(&bits, f, VALUE_FINITE, v->m.neg, &mag, v->e, ROUND_NEAREST_EVEN, OVERFLOW_NAN);
    if (!rc) rc = floatfmt_decode(&kind, &neg, &sig, &e, f, &bits);
    if (!rc) femu_read(r, x, kind, neg, sig.n ? sig.w[0] : 0, e);
    big_free(&bits);
    big_free(&sig);
    big_free(&mag);
    return rc;
}

/* exp (log with is_log) of the dyad v, correctly rounded into the compute format: the enclosure
   at p bits, p doubled until both ends round alike. An
   argument below -(|emin| + p + 2) gives +0: exp is below half the least subnormal there. */
static int rounded_function(fvalue *r, const femu *x, const floatfmt *f, const dyad *v, int is_log) {
    dyad lo, hi, cut;
    bigint n, e, bl, bh;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&cut);
    big_init(&n);
    big_init(&e);
    big_init(&bl);
    big_init(&bh);
    int rc = 0;
    if (!is_log) {
        rc = dyad_set_si(&cut, -(int64_t)(-x->emin + x->p + 2), 0);
        if (!rc && dyad_cmp(v, &cut) < 0) {
            femu_read(r, x, VALUE_FINITE, 0, 0, 0);
            goto done;
        }
    } else {
        rc = big_set_si(&e, v->e);
    }
    for (int64_t p = 2 * x->p + 16; !rc; p *= 2) {
        rc = is_log ? log_bounds(&lo, &hi, &v->m, &e, p) : exp_bounds(&lo, &hi, &n, v, p);
        if (rc) break;
        int64_t shift = 0;
        if (!is_log) {
            if (n.n > 1) {
                rc = EXACTREF_OVERFLOW;
                break;
            }
            shift = n.n ? (n.neg ? -(int64_t)n.w[0] : (int64_t)n.w[0]) : 0;
        }
        /* both ends' encodings */
        bigint ml, mh;
        big_init(&ml);
        big_init(&mh);
        if (!(rc = big_copy(&ml, &lo.m)) && !(rc = big_copy(&mh, &hi.m))) {
            int nl = ml.neg, nh = mh.neg;
            ml.neg = mh.neg = 0;
            if (!(rc = floatfmt_encode(&bl, f, VALUE_FINITE, nl, &ml, lo.e + shift, ROUND_NEAREST_EVEN, OVERFLOW_NAN)))
                rc = floatfmt_encode(&bh, f, VALUE_FINITE, nh, &mh, hi.e + shift, ROUND_NEAREST_EVEN, OVERFLOW_NAN);
        }
        big_free(&ml);
        big_free(&mh);
        if (!rc && big_cmp_mag(&bl, &bh) == 0) { /* encodings: nonnegative integers */
            int kind, neg;
            int64_t ee;
            if (!(rc = floatfmt_decode(&kind, &neg, &n, &ee, f, &bl))) femu_read(r, x, kind, neg, n.n ? n.w[0] : 0, ee);
            break;
        }
    }
done:
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&cut);
    big_free(&n);
    big_free(&e);
    big_free(&bl);
    big_free(&bh);
    return rc;
}

/* a / b for a value a and a finite positive b of the compute format, correctly rounded (IEEE's
   special cases for a), then read with the flush. */
static int divide(fvalue *r, const femu *x, const floatfmt *f, const fvalue *a, const fvalue *b) {
    if (a->kind != VALUE_FINITE || !a->sig) {
        *r = *a; /* NaN, an infinity over a positive value, or a zero of a's sign */
        return 0;
    }
    bigint na, nb, bits, sig;
    big_init(&na);
    big_init(&nb);
    big_init(&bits);
    big_init(&sig);
    int64_t d = a->e - b->e;
    int kind, neg;
    int64_t e;
    int rc = big_set_word(&na, a->sig, 0);
    if (!rc) rc = big_set_word(&nb, b->sig, 0);
    if (!rc) rc = d >= 0 ? big_shl(&na, &na, d) : big_shl(&nb, &nb, -d);
    if (!rc) rc = floatfmt_encode_ratio(&bits, f, a->neg != b->neg, &na, &nb, ROUND_NEAREST_EVEN, OVERFLOW_NAN);
    if (!rc) rc = floatfmt_decode(&kind, &neg, &sig, &e, f, &bits);
    if (!rc) femu_read(r, x, kind, neg, sig.n ? sig.w[0] : 0, e);
    big_free(&na);
    big_free(&nb);
    big_free(&bits);
    big_free(&sig);
    return rc;
}

/* a value of the format as a dyad */
static int fvalue_dyad(dyad *r, const fvalue *v) {
    int rc = big_set_word(&r->m, v->sig, v->neg && v->sig);
    r->e = v->e;
    return rc;
}

/* whether a > b for finite values */
static int greater(const fvalue *a, const fvalue *b) {
    dyad x, y;
    dyad_init(&x);
    dyad_init(&y);
    int g = !fvalue_dyad(&x, a) && !fvalue_dyad(&y, b) && dyad_cmp(&x, &y) > 0;
    dyad_free(&x);
    dyad_free(&y);
    return g;
}

/* the value encoded in fmt with the call's rounding and overflow, into out */
static int encode_into(unsigned char *out, size_t size, const floatfmt *fmt, const fvalue *v, int rnd, int ovf) {
    bigint bits, sig;
    big_init(&bits);
    big_init(&sig);
    int rc = big_set_word(&sig, v->sig, 0);
    if (!rc) rc = floatfmt_encode(&bits, fmt, v->kind, v->neg, &sig, v->e, rnd, ovf);
    if (!rc) bytes_store(out, size, &bits);
    big_free(&bits);
    big_free(&sig);
    return rc;
}

/* sum of the n values in index order, each addition rounded; where that overflows with every
   value finite, Lemma O's slots (the values are overwritten) */
static void index_sum(fvalue *r, const femu *x, fvalue *v, size_t n) {
    if (!n) {
        femu_read(r, x, VALUE_FINITE, 0, 0, 0);
        return;
    }
    fvalue acc = v[0];
    int all_finite = v[0].kind == VALUE_FINITE;
    for (size_t j = 1; j < n; j++) {
        femu_add(&acc, x, &acc, &v[j]);
        all_finite &= v[j].kind == VALUE_FINITE;
    }
    if (acc.kind != VALUE_FINITE && all_finite)
        femu_slot_sum(r, x, v, n);
    else
        *r = acc;
}

/* One row's softmax in the compute format: every key's logit (its score's normal sum times the scale plus the mask
   value, rounded once; IEEE's special values), the largest finite one, and each weight exp(logit - top) correctly
   rounded (0 for -inf, 1 for a zero shift), and the row sum in index order. *kind: 0 a row, 1 NaN (a NaN logit), 2 +inf
   (a +inf logit), 3 no finite logit (every weight 0). logits, weights and terms hold S values, scratch E. */
int att_normal_softmax(int *kind, fvalue *top, fvalue *total, fvalue *logits, fvalue *weights, fvalue *terms, fvalue *scratch, const att_call *a,
                       const att_normal *nm, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0) {
    const femu *x = &nm->x;
    const floatfmt *f = &nm->f;
    size_t S = a->S;
    fvalue s, bias, d;
    dyad exact, t1, t2;
    dyad_init(&exact);
    dyad_init(&t1);
    dyad_init(&t2);
    int rc = 0;
    *kind = 0;
    /* every key's logit: its score (normal), times the scale plus the mask value, rounded once */
    size_t m = 0, top_at = S;
    for (size_t j = 0; !rc && j < S; j++) {
        if ((rc = product_normal(&s, scratch, x, a->q, qr, a->k, kr0 + j))) break;
        int masked = 0;
        femu_read(&bias, x, VALUE_FINITE, 0, 0, 0);
        if (a->causal) {
            masked = j > i;
        } else if (a->mask && !a->mask_float) {
            masked = !a->mask[moff + j * mstep];
        } else if (a->mask) {
            int mk, mneg;
            uint64_t msig;
            int64_t me;
            att_mask_read(&mk, &mneg, &msig, &me, a, moff + j * mstep);
            femu_read(&bias, x, mk, mneg, msig, me);
        }
        if (masked) femu_read(&bias, x, VALUE_INFINITE, 1, 0, 0);
        if (s.kind != VALUE_FINITE || nm->sigma.kind != VALUE_FINITE) {
            femu_mul(&d, x, &s, &nm->sigma); /* an infinite or NaN score or scale: IEEE's special values */
            femu_add(&logits[j], x, &d, &bias);
        } else if (bias.kind != VALUE_FINITE) {
            logits[j] = bias; /* a finite score times the scale, exactly, plus an infinity or NaN */
        } else if (!(rc = fvalue_dyad(&t1, &s)) && !(rc = fvalue_dyad(&t2, &nm->sigma)) && !(rc = dyad_mul(&exact, &t1, &t2, 1 << 20, 0)) &&
                   !(rc = fvalue_dyad(&t1, &bias)) && !(rc = dyad_add(&exact, &exact, &t1, 1 << 20, 0))) {
            rc = round_once(&logits[j], x, f, &exact);
        }
        if (rc) break;
        if (logits[j].kind == VALUE_NAN) *kind = 1;
        if (logits[j].kind == VALUE_INFINITE && !logits[j].neg && *kind != 1) *kind = 2;
        if (logits[j].kind == VALUE_FINITE) {
            if (top_at == S || greater(&logits[j], top)) {
                *top = logits[j];
                top_at = j;
            }
            m++;
        }
    }
    if (!rc && !*kind && !m) *kind = 3;
    /* the weights: exp of each shift, correctly rounded (0 for -inf, 1 for a zero shift) */
    for (size_t j = 0; !rc && !(*kind == 1 || *kind == 2) && j < S; j++) {
        if (*kind == 3) {
            femu_read(&weights[j], x, VALUE_FINITE, 0, 0, 0);
            continue;
        }
        fvalue neg_top = *top;
        neg_top.neg = !top->neg;
        femu_add(&d, x, &logits[j], &neg_top);
        if (d.kind == VALUE_INFINITE && d.neg)
            femu_read(&weights[j], x, VALUE_FINITE, 0, 0, 0);
        else if (d.kind == VALUE_FINITE && !d.sig)
            femu_read(&weights[j], x, VALUE_FINITE, 0, 1, 0);
        else if (!(rc = fvalue_dyad(&t1, &d)))
            rc = rounded_function(&weights[j], x, f, &t1, 0);
    }
    if (!rc && !*kind) {
        memcpy(terms, weights, S * sizeof *terms);
        index_sum(total, x, terms, S);
    }
    dyad_free(&exact);
    dyad_free(&t1);
    dyad_free(&t2);
    return rc;
}

int att_normal_row(const att_call *a, const att_normal *nm, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, unsigned char *enc,
                   unsigned char *lse_enc) {
    const femu *x = &nm->x;
    const floatfmt *f = &nm->f;
    size_t S = a->S, Ev = a->Ev;
    fvalue *logits = malloc((S ? S : 1) * sizeof *logits), *weights = malloc((S ? S : 1) * sizeof *weights), *terms = malloc((S ? S : 1) * sizeof *terms);
    fvalue *scratch = malloc((a->E ? (size_t)a->E : 1) * sizeof *scratch), nan, top, total, lse, d;
    dyad t1;
    dyad_init(&t1);
    int rc = 0, kind = 0; /* 0 a row, 1 NaN, 2 +inf, 3 no finite logit */
    if (!logits || !weights || !terms || !scratch) {
        rc = EXACTREF_NOMEM;
        goto done;
    }
    femu_read(&nan, x, VALUE_NAN, 0, 0, 0);
    if ((rc = att_normal_softmax(&kind, &top, &total, logits, weights, terms, scratch, a, nm, qr, i, moff, mstep, kr0))) goto done;
    if (kind == 1 || kind == 2) { /* a NaN or +inf logit makes the row NaN; logsumexp NaN or +inf */
        for (size_t c = 0; !rc && c < Ev; c++) rc = encode_into(enc + c * a->out_size, a->out_size, &a->out, &nan, a->rnd, a->ovf);
        femu_read(&lse, x, kind == 1 ? VALUE_NAN : VALUE_INFINITE, 0, 0, 0);
        if (!rc) rc = encode_into(lse_enc, a->lse_size, &a->lse, &lse, a->lrnd, a->lovf);
        goto done;
    }
    /* each column's sum of the weights times the values in index order, then over the row sum */
    for (size_t c = 0; !rc && c < Ev; c++) {
        fvalue r, v;
        for (size_t j = 0; j < S; j++) {
            int vk, vn;
            uint64_t vs;
            int64_t ve;
            element_decode_slot(&vk, &vn, &vs, &ve, a->v, (vr0 + j) * Ev + c); /* any slot: wide, flagged, non-finite, -0 */
            femu_read(&v, x, vk, vn, vs, ve);
            femu_mul(&terms[j], x, &weights[j], &v);
        }
        index_sum(&r, x, terms, S);
        if (kind != 3) rc = divide(&r, x, f, &r, &total);
        if (!rc) rc = encode_into(enc + c * a->out_size, a->out_size, &a->out, &r, a->rnd, a->ovf);
    }
    if (rc) goto done;
    /* logsumexp: the maximum plus the correctly rounded log of the row sum; -inf with no finite logit */
    if (kind == 3) {
        femu_read(&lse, x, VALUE_INFINITE, 1, 0, 0);
    } else if (!(rc = fvalue_dyad(&t1, &total)) && !(rc = rounded_function(&d, x, f, &t1, 1))) {
        femu_add(&lse, x, &top, &d);
    }
    if (!rc) rc = encode_into(lse_enc, a->lse_size, &a->lse, &lse, a->lrnd, a->lovf);
done:
    free(logits);
    free(weights);
    free(terms);
    free(scratch);
    dyad_free(&t1);
    return rc;
}

/* ---------------------------------------------------------------- the backward's normal answer
   (proofs/scaled_dot_product_attention_backward.md, The normal answer): per row P' = w' / Z' rounded (0 for every key
   of a row without a finite logit, NaN for a NaN row), dP' over the Ev products in index order, D' over the keys'
   products, dS' = fl(fl(dP' - D') P'); each gradient's sum over its terms in the model's order (rows in tile order,
   keys in order, copies in order), each product rounded and added, with Lemma O's slots where index order overflows
   with every term finite (att_gnorm_slots, from the element's terms again), grad_query and grad_key times the scale
   held in the compute format, then encoded. */

void att_gnorm_call_init(att_gnorm_call *c) {
    memset(c, 0, sizeof *c);
    big_init(&c->a.num);
    big_init(&c->a.den);
    for (int z = 0; z < 2; z++) {
        big_init(&c->out_nan[z]);
        big_init(&c->compute_nan[z]);
    }
}

void att_gnorm_call_free(att_gnorm_call *c) {
    free(c->qacc);
    free(c->qall);
    free(c->qn);
    big_free(&c->a.num);
    big_free(&c->a.den);
    for (int z = 0; z < 2; z++) {
        big_free(&c->out_nan[z]);
        big_free(&c->compute_nan[z]);
    }
    memset(c, 0, sizeof *c);
}

int att_gnorm_query_slots(att_gnorm_call *c, size_t count) {
    size_t n = count * (size_t)c->a.E;
    c->qslots = count;
    c->qacc = malloc((n ? n : 1) * sizeof *c->qacc);
    c->qall = malloc(n ? n : 1);
    c->qn = calloc(n ? n : 1, sizeof *c->qn);
    return c->qacc && c->qall && c->qn ? 0 : EXACTREF_NOMEM;
}

/* A block's sums at zero terms: S keys of cols elements */
int att_gnorm_block_start(att_gnorm_block *b, const att_gnorm_call *c, size_t cols) {
    size_t n = c->a.S && cols ? c->a.S * cols : 1;
    memset(b, 0, sizeof *b);
    b->cols = cols;
    if (!(b->acc = malloc(n * sizeof *b->acc)) || !(b->all = malloc(n)) || !(b->n = calloc(n, sizeof *b->n))) return EXACTREF_NOMEM;
    return 0;
}

void att_gnorm_block_free(att_gnorm_block *b) {
    free(b->acc);
    free(b->all);
    free(b->n);
    memset(b, 0, sizeof *b);
}

void att_gnorm_group_init(att_gnorm_group *g) { memset(g, 0, sizeof *g); }

void att_gnorm_group_free(att_gnorm_group *g) {
    free(g->rinfo);
    free(g->P);
    free(g->DS);
    memset(g, 0, sizeof *g);
}

/* A group for the unit of key rows kr0.. and value rows vr0.., adding to blocks kb and vb, its tiles up to cap rows */
int att_gnorm_group_start(att_gnorm_group *g, const att_gnorm_call *c, size_t kr0, size_t vr0, att_gnorm_block *kb, att_gnorm_block *vb, size_t cap) {
    size_t S = c->a.S;
    g->kr0 = kr0;
    g->vr0 = vr0;
    g->kb = kb;
    g->vb = vb;
    g->cap = cap ? cap : 1;
    g->S = S;
    g->E = (size_t)c->a.E;
    g->Ev = c->a.Ev;
    size_t cells = g->cap * (S ? S : 1);
    if (!(g->rinfo = calloc(g->cap * ATT_GR_INFO, sizeof *g->rinfo)) || !(g->P = malloc(cells * sizeof *g->P)) || !(g->DS = malloc(cells * sizeof *g->DS)))
        return EXACTREF_NOMEM;
    return 0;
}

/* the value of an operand's slot, read in the compute format (with the call's flush) */
static void read_slot(fvalue *r, const femu *x, const operand *op, size_t at) {
    int kind, neg;
    uint64_t sig;
    int64_t e;
    element_decode_slot(&kind, &neg, &sig, &e, op, at); /* any slot: wide, flagged, non-finite, -0 */
    femu_read(r, x, kind, neg, sig, e);
}

/* a term into a running sum in index order: the first term starts it */
static void acc_add(fvalue *acc, unsigned char *all, int64_t *n, const femu *x, const fvalue *t) {
    if (!*n) {
        *acc = *t;
        *all = t->kind == VALUE_FINITE;
    } else {
        femu_add(acc, x, acc, t);
        *all &= t->kind == VALUE_FINITE;
    }
    ++*n;
}

/* One row's P' and dS' over its S keys (the row's info ri, against key rows kr0.. and value rows vr0..). buf holds
   4 S + 2 Ev + E values. */
static int gnorm_row(fvalue *P, fvalue *DS, const att_gnorm_call *c, const int64_t *ri, size_t kr0, size_t vr0, fvalue *buf) {
    const femu *x = &c->nm.x;
    const floatfmt *f = &c->nm.f;
    size_t S = c->a.S, Ev = c->a.Ev, n = S > Ev ? S : Ev;
    fvalue *logits = buf, *weights = logits + S, *dp = weights + S, *terms = dp + S, *grad = terms + n, *scratch = grad + Ev, top, total, D, t;
    int kind, rc = att_normal_softmax(&kind, &top, &total, logits, weights, terms, scratch, &c->a, &c->nm, (size_t)ri[0], (size_t)ri[1], (size_t)ri[2],
                                      (size_t)ri[3], kr0);
    for (size_t j = 0; !rc && j < S; j++) {
        if (kind == 1 || kind == 2)
            femu_read(&P[j], x, VALUE_NAN, 0, 0, 0);
        else if (kind == 3)
            femu_read(&P[j], x, VALUE_FINITE, 0, 0, 0);
        else
            rc = divide(&P[j], x, f, &weights[j], &total);
    }
    if (rc) return rc;
    size_t gr0 = (size_t)ri[4];
    for (size_t e = 0; e < Ev; e++) read_slot(&grad[e], x, c->gd, gr0 * Ev + e);
    for (size_t j = 0; j < S; j++) {
        for (size_t e = 0; e < Ev; e++) {
            fvalue v;
            read_slot(&v, x, c->vd, (vr0 + j) * Ev + e);
            femu_mul(&terms[e], x, &grad[e], &v);
        }
        index_sum(&dp[j], x, terms, Ev);
    }
    for (size_t j = 0; j < S; j++) femu_mul(&terms[j], x, &P[j], &dp[j]);
    index_sum(&D, x, terms, S);
    D.neg = !D.neg; /* dP - D as dP + (-D) */
    for (size_t j = 0; j < S; j++) {
        femu_add(&t, x, &dp[j], &D);
        femu_mul(&DS[j], x, &t, &P[j]);
    }
    return 0;
}

static size_t gnorm_buf(const att_gnorm_call *c) {
    size_t S = c->a.S, Ev = c->a.Ev;
    return 4 * S + 2 * Ev + (size_t)c->a.E + 4;
}

/* Rows r0 to r1 of the group's tile: their P' and dS' */
int att_gnorm_tile_rows(att_gnorm_group *g, const att_gnorm_call *c, size_t r0, size_t r1) {
    size_t S = g->S;
    fvalue *buf = malloc(gnorm_buf(c) * sizeof *buf);
    int rc = buf ? 0 : EXACTREF_NOMEM;
    for (size_t r = r0; !rc && r < r1; r++) rc = gnorm_row(g->P + r * S, g->DS + r * S, c, g->rinfo + ATT_GR_INFO * r, g->kr0, g->vr0, buf);
    free(buf);
    return rc;
}

/* The tile's terms into the sums of keys j0 to j1: grad_value's P' g and grad_key's dS' q, over the tile's rows in order */
int att_gnorm_tile_keys(att_gnorm_group *g, const att_gnorm_call *c, size_t j0, size_t j1) {
    const femu *x = &c->nm.x;
    size_t S = g->S, E = g->E, Ev = g->Ev, T = g->T;
    fvalue y, t;
    for (size_t j = j0; j < j1; j++) {
        for (size_t col = 0; col < Ev; col++) {
            size_t el = j * Ev + col;
            for (size_t r = 0; r < T; r++) {
                read_slot(&y, x, c->gd, (size_t)g->rinfo[ATT_GR_INFO * r + 4] * Ev + col);
                femu_mul(&t, x, &g->P[r * S + j], &y);
                acc_add(&g->vb->acc[el], &g->vb->all[el], &g->vb->n[el], x, &t);
            }
        }
        for (size_t e = 0; e < E; e++) {
            size_t el = j * E + e;
            for (size_t r = 0; r < T; r++) {
                read_slot(&y, x, c->a.q, (size_t)g->rinfo[ATT_GR_INFO * r] * E + e);
                femu_mul(&t, x, &g->DS[r * S + j], &y);
                acc_add(&g->kb->acc[el], &g->kb->all[el], &g->kb->n[el], x, &t);
            }
        }
    }
    return 0;
}

/* An element's sum (n terms, every one finite: all), times the scale when scaled, encoded; *status 1 when index order
   overflowed with every term finite (Lemma O's slots: att_gnorm_slots) */
static int gnorm_finish(unsigned char *enc, unsigned char *status, const att_gnorm_call *c, const fvalue *acc, int all, int64_t n, int scaled) {
    const femu *x = &c->nm.x;
    fvalue r;
    *status = 0;
    if (!n) {
        femu_read(&r, x, VALUE_FINITE, 0, 0, 0); /* an empty sum: +0 */
    } else if (acc->kind != VALUE_FINITE && all) {
        *status = 1;
        return 0;
    } else {
        r = *acc;
    }
    if (scaled) femu_mul(&r, x, &r, &c->nm.sigma);
    return encode_into(enc, c->out_size, &c->out, &r, c->rnd, c->ovf);
}

/* grad_query of the tile's rows r0 to r1: dS' times the key block's rows over the keys, in the call's slot from copy to
   copy for a query row broadcast over several key and value blocks, finished at its last copy into the outputs (E per
   row from enc and status at row r - r0) */
int att_gnorm_tile_queries(const att_gnorm_group *g, att_gnorm_call *c, size_t r0, size_t r1, unsigned char *enc, unsigned char *status) {
    const femu *x = &c->nm.x;
    size_t S = g->S, E = g->E;
    fvalue *oacc = malloc((E ? E : 1) * sizeof *oacc), y, t;
    unsigned char *oall = malloc(E ? E : 1);
    int64_t *on = calloc(E ? E : 1, sizeof *on);
    int rc = oacc && oall && on ? 0 : EXACTREF_NOMEM;
    for (size_t r = r0; !rc && r < r1; r++) {
        const int64_t *ri = g->rinfo + ATT_GR_INFO * r;
        int64_t slot = ri[5], copy = ri[6], copies = ri[7];
        fvalue *acc = oacc;
        unsigned char *all = oall;
        int64_t *n = on;
        if (slot >= 0) {
            acc = c->qacc + (size_t)slot * E;
            all = c->qall + (size_t)slot * E;
            n = c->qn + (size_t)slot * E;
        }
        if (slot < 0 || copy == 0) memset(n, 0, E * sizeof *n);
        for (size_t e = 0; e < E; e++)
            for (size_t j = 0; j < S; j++) {
                read_slot(&y, x, c->a.k, (g->kr0 + j) * E + e);
                femu_mul(&t, x, &g->DS[r * S + j], &y);
                acc_add(&acc[e], &all[e], &n[e], x, &t);
            }
        if (slot >= 0 && copy < copies - 1) continue; /* finished at its last copy */
        for (size_t e = 0; !rc && e < E; e++) {
            size_t o = (r - r0) * E + e;
            rc = gnorm_finish(enc + o * c->out_size, &status[o], c, &acc[e], all[e], n[e], 1);
        }
    }
    free(oacc);
    free(oall);
    free(on);
    return rc;
}

/* A block's elements of keys j0 to j1, after the last unit that adds to it, into the outputs (cols per key from j0):
   grad_key's scaled, grad_value's not */
int att_gnorm_block_finish(const att_gnorm_block *b, const att_gnorm_call *c, int scaled, size_t j0, size_t j1, unsigned char *enc, unsigned char *status) {
    int rc = 0;
    for (size_t el = j0 * b->cols; !rc && el < j1 * b->cols; el++) {
        size_t o = el - j0 * b->cols;
        rc = gnorm_finish(enc + o * c->out_size, &status[o], c, &b->acc[el], b->all[el], b->n[el], scaled);
    }
    return rc;
}

/* An element whose index order overflowed with every term finite: its terms again, from its rows (each ATT_GR_INFO
   values then its key and value rows: count of them, in the model's order), added in Lemma O's slots, times the scale
   for grad_key and grad_query, encoded. which: 0 grad_value, 1 grad_key, 2 grad_query; index: the element within its
   block (key j's column, or the query row's column). */
int att_gnorm_slots(unsigned char *enc, const att_gnorm_call *c, int which, size_t index, const int64_t *rows, size_t count) {
    const femu *x = &c->nm.x;
    size_t S = c->a.S, E = (size_t)c->a.E, Ev = c->a.Ev, cols = which ? E : Ev, j = index / (cols ? cols : 1), col = index % (cols ? cols : 1);
    size_t cap = which == 2 ? count * S : count;
    fvalue *P = malloc((S ? S : 1) * sizeof *P), *DS = malloc((S ? S : 1) * sizeof *DS), *buf = malloc(gnorm_buf(c) * sizeof *buf);
    fvalue *terms = malloc((cap ? cap : 1) * sizeof *terms), y, r;
    int rc = P && DS && buf && terms ? 0 : EXACTREF_NOMEM;
    size_t n = 0;
    for (size_t z = 0; !rc && z < count; z++) {
        const int64_t *ri = rows + z * (ATT_GR_INFO + 2);
        size_t kr0 = (size_t)ri[ATT_GR_INFO], vr0 = (size_t)ri[ATT_GR_INFO + 1];
        if ((rc = gnorm_row(P, DS, c, ri, kr0, vr0, buf))) break;
        if (which == 0) {
            read_slot(&y, x, c->gd, (size_t)ri[4] * Ev + col);
            femu_mul(&terms[n++], x, &P[j], &y);
        } else if (which == 1) {
            read_slot(&y, x, c->a.q, (size_t)ri[0] * E + col);
            femu_mul(&terms[n++], x, &DS[j], &y);
        } else {
            for (size_t k = 0; k < S; k++) {
                read_slot(&y, x, c->a.k, (kr0 + k) * E + index);
                femu_mul(&terms[n++], x, &DS[k], &y);
            }
        }
    }
    if (!rc) {
        femu_slot_sum(&r, x, terms, n);
        if (which) femu_mul(&r, x, &r, &c->nm.sigma);
        rc = encode_into(enc, c->out_size, &c->out, &r, c->rnd, c->ovf);
    }
    free(P);
    free(DS);
    free(buf);
    free(terms);
    return rc;
}
