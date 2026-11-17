/* The product kernel (internal): the exact quantities of each output element of a real product
   A B, from operands the read kernel made: A by rows, and B transposed, so that row j of b holds
   column j of B. Per element, the exact sum; for the bounds, the magnitude sum, the node sums of
   index order and of lanes 2 and 4 with every node exact (proofs/foundations.md, The normal
   bound), and with flush to zero the magnitudes of the terms with a factor a flush may read as
   zero. An element whose row of A and column of B hold only main-range integers takes 128-bit
   words (proofs/kernels.md, Lemma P); any other takes multi-word integers at its own exponent,
   or is a sum with a non-finite term, whose IEEE value it gets. */
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

void product_sums_init(product_sums *r) {
    r->e = 0;
    r->kind = VALUE_FINITE;
    r->neg = 0;
    r->scratch = NULL;
    r->scratch_size = 0;
    big_init(&r->s);
    big_init(&r->mass);
    big_init(&r->N);
    big_init(&r->moved);
    big_init(&r->dev);
}

void product_sums_free(product_sums *r) {
    free(r->scratch);
    r->scratch = NULL;
    r->scratch_size = 0;
    big_free(&r->s);
    big_free(&r->mass);
    big_free(&r->N);
    big_free(&r->moved);
    big_free(&r->dev);
}

/* x = the two's-complement word pair w as a signed integer (signed) or a magnitude. */
static int big_from_w128(bigint *x, w128 w, int is_signed) {
    int neg = is_signed && w128_negative(w);
    if (neg) w = w128_abs(w);
    if (big_reserve(x, 2)) return EXACTREF_NOMEM;
    x->w[0] = w128_lo(w);
    x->w[1] = w128_hi(w);
    x->n = 2;
    x->neg = 0;
    big_trim(x);
    x->neg = neg && x->n;
    return 0;
}

/* The main range, in bits per operand, of products of a given number of terms: the largest W
   with 2 W + 2 ceil(log2 terms) + 1 <= 128, so that every accumulator fits 128 bits (Lemma P),
   and at most 62, the read kernel's one-word slots; wide (float64 operands, whose 53-bit values
   would mostly fall below that range): the same with 256 bits and two-word slots, at most 126. */
int64_t product_main_bits(size_t terms, int wide) {
    int64_t c = terms > 1 ? word_bits((uint64_t)terms - 1) : 0, w = ((wide ? 255 : 127) - 2 * c) / 2, most = wide ? 126 : 62;
    return w < most ? w : most;
}

/* The magnitude below which an operand's integers are below 2**emin, for flush to zero: 0 when no
   integer at the operand's exponent is (2**emin at or below one unit). */
static uint64_t tiny_limit(const operand *x, int64_t emin) {
    int64_t d = emin - x->exponent;
    return d <= 0 ? 0 : d >= 63 ? UINT64_MAX : (uint64_t)1 << d;
}

static uint64_t slot_magnitude(uint64_t w) { return w >> 63 ? ~w + 1 : w; }

/* Whether the coefficient rule leaves out the term that slot at of a coefficient operand makes. */
static int slot_absent(const operand *x, size_t at) { return x->absent && (int)(x->absent[at / 64] >> (at % 64) & 1); }

/* The value of slot at: its kind, sign, and for a finite value sig * 2**e, with e at or above
   the operand's exponent or its row's lowest set bit (a flagged value's significand odd). */
void slot_value(int *kind, int *neg, uint64_t *sig, int64_t *e, const operand *x, size_t at) {
    uint64_t w = x->slots[at];
    if (x->flagged[at / 64] >> (at % 64) & 1) {
        element_decode(kind, neg, sig, e, &x->pf, w);
        for (; *sig && !(*sig & 1); ++*e) *sig >>= 1;
        return;
    }
    *kind = VALUE_FINITE;
    if (x->slots_hi) {
        /* a two-word integer: its magnitude is an odd significand of at most 64 bits shifted up, so it fits a word once its trailing zeros go */
        uint64_t hi = x->slots_hi[at], lo = w;
        *neg = (hi | lo) ? (int)(hi >> 63) : (int)(x->negzero[at / 64] >> (at % 64) & 1);
        if (hi >> 63) {
            lo = ~lo + 1;
            hi = ~hi + (lo == 0);
        }
        int64_t t = 0;
        if (!lo && hi) {
            lo = hi;
            hi = 0;
            t = 64;
        }
        while (lo && !(lo & 1)) {
            lo = lo >> 1 | hi << 63;
            hi >>= 1;
            t++;
        }
        *sig = lo;
        *e = x->exponent + t;
        return;
    }
    *neg = w ? (int)(w >> 63) : (int)(x->negzero[at / 64] >> (at % 64) & 1);
    *sig = slot_magnitude(w);
    *e = x->exponent;
}

/* Lemma L: the largest node sum of combining L = 2 or 4 lane totals t, the lanes with terms marked in has (bit l), every
   node exact. The orders are in order, and for L = 4 pairwise ((T_0 + T_1) + (T_2 + T_3)) and folded ((T_0 + T_2) +
   (T_1 + T_3)): lane_order lists each order's lanes; in order they join one chain x, in pairwise and folded the first two
   join x and the last two y, which then joins x. Where both sides hold a lane with terms, a join adds the magnitude of
   its sum to the order's nodes; an empty lane joins none. In 128-bit and 256-bit words (w128_lane_nodes, w256_lane_nodes), in
   two's-complement integers of n words (lane_nodes_words) and in dyadics (lane_nodes_dyad). */
static const int lane_order[3][4] = {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}};

#define LANE_NODES(W)                                                                                                  \
    static void W##_lane_join(W *nodes, W *x, int *held, W y) {                                                        \
        if (!*held) {                                                                                                  \
            *x = y;                                                                                                    \
            *held = 1;                                                                                                 \
            return;                                                                                                    \
        }                                                                                                              \
        *x = W##_add(*x, y);                                                                                           \
        *nodes = W##_add(*nodes, W##_abs(*x));                                                                         \
    }                                                                                                                  \
    static W W##_lane_nodes(const W *t, unsigned has, int L) {                                                         \
        W best = W##_zero();                                                                                           \
        for (int o = 0; o < (L == 4 ? 3 : 1); o++) {                                                                   \
            W x = W##_zero(), y = W##_zero(), nodes = W##_zero();                                                      \
            int hx = 0, hy = 0;                                                                                        \
            for (int i = 0; i < L; i++) {                                                                              \
                int l = lane_order[o][i];                                                                              \
                if (!(has >> l & 1)) continue;                                                                         \
                if (o && 2 * i >= L)                                                                                   \
                    W##_lane_join(&nodes, &y, &hy, t[l]);                                                              \
                else                                                                                                   \
                    W##_lane_join(&nodes, &x, &hx, t[l]);                                                              \
            }                                                                                                          \
            if (hy) W##_lane_join(&nodes, &x, &hx, y);                                                                 \
            if (W##_less(best, nodes)) best = nodes;                                                                   \
        }                                                                                                              \
        return best;                                                                                                   \
    }
LANE_NODES(w128)
LANE_NODES(w256)

static void words_lane_join(uint64_t *nodes, uint64_t *x, int *held, const uint64_t *y, size_t n) {
    if (!*held) {
        memcpy(x, y, n * sizeof *x);
        *held = 1;
        return;
    }
    nat_add(x, x, y, n, 0);
    if (mw_negative(x, n))
        nat_sub(nodes, nodes, x, n, 0);
    else
        nat_add(nodes, nodes, x, n, 0);
}

/* r: n words; lane l at t + l n; scratch: 3 n words */
void lane_nodes_words(uint64_t *r, const uint64_t *t, unsigned has, int L, size_t n, uint64_t *scratch) {
    uint64_t *x = scratch, *y = scratch + n, *nodes = scratch + 2 * n;
    memset(r, 0, n * sizeof *r);
    for (int o = 0; o < (L == 4 ? 3 : 1); o++) {
        int hx = 0, hy = 0;
        memset(nodes, 0, n * sizeof *nodes);
        for (int i = 0; i < L; i++) {
            int l = lane_order[o][i];
            if (!(has >> l & 1)) continue;
            if (o && 2 * i >= L)
                words_lane_join(nodes, y, &hy, t + (size_t)l * n, n);
            else
                words_lane_join(nodes, x, &hx, t + (size_t)l * n, n);
        }
        if (hy) words_lane_join(nodes, x, &hx, y, n);
        if (nat_cmp(nodes, r, n) > 0) memcpy(r, nodes, n * sizeof *r);
    }
}

static int dyad_lane_join(dyad *nodes, dyad *x, int *held, const dyad *y, dyad *m) {
    if (!*held) {
        *held = 1;
        return dyad_copy(x, y);
    }
    int rc = dyad_add_exact(x, x, y);
    if (!rc && !(rc = dyad_abs(m, x))) rc = dyad_add_exact(nodes, nodes, m);
    return rc;
}

int lane_nodes_dyad(dyad *r, const dyad *t, unsigned has, int L) {
    dyad x, y, m, nodes, best;
    dyad *ds[] = {&x, &y, &m, &nodes, &best};
    for (size_t z = 0; z < 5; z++) dyad_init(ds[z]);
    int rc = dyad_set_si(&best, 0, 0);
    for (int o = 0; !rc && o < (L == 4 ? 3 : 1); o++) {
        int hx = 0, hy = 0;
        rc = dyad_set_si(&nodes, 0, 0);
        for (int i = 0; !rc && i < L; i++) {
            int l = lane_order[o][i];
            if (has >> l & 1) rc = o && 2 * i >= L ? dyad_lane_join(&nodes, &y, &hy, &t[l], &m) : dyad_lane_join(&nodes, &x, &hx, &t[l], &m);
        }
        if (!rc && hy) rc = dyad_lane_join(&nodes, &x, &hx, &y, &m);
        if (!rc && dyad_cmp(&nodes, &best) > 0) rc = dyad_copy(&best, &nodes);
    }
    if (!rc) rc = dyad_copy(r, &best);
    for (size_t z = 0; z < 5; z++) dyad_free(ds[z]);
    return rc;
}

/* Named summation orders (the matmul algorithms'): every internal node is the exact difference
   of two running sums, P_hi - P_lo, with P_q the sum of the first q terms. */

void product_order_free(product_order *o) {
    free(o->lo);
    free(o->hi);
    o->lo = o->hi = NULL;
    o->count = 0;
}

static int order_alloc(product_order *o, size_t k) {
    o->count = 0;
    o->lo = malloc((k + 1) * sizeof *o->lo);
    o->hi = malloc((k + 1) * sizeof *o->hi);
    if (o->lo && o->hi) return 0;
    product_order_free(o);
    return EXACTREF_NOMEM;
}

static void order_add(product_order *o, size_t lo, size_t hi) {
    o->lo[o->count] = lo;
    o->hi[o->count] = hi;
    o->count++;
}

/* Index order (sequential summation): the nodes are the running sums of 2 or more terms. */
int product_order_sequential(product_order *o, size_t k) {
    if (order_alloc(o, k)) return EXACTREF_NOMEM;
    for (size_t q = 2; q <= k; q++) order_add(o, 0, q);
    return 0;
}

/* Blocked summation: each block of b consecutive terms a chain from its first term, then the
   block totals added in order. */
int product_order_blocked(product_order *o, size_t k, size_t b) {
    if (order_alloc(o, k)) return EXACTREF_NOMEM;
    for (size_t start = 0; start < k; start += b) {
        size_t end = start + b < k ? start + b : k;
        for (size_t q = start + 2; q <= end; q++) order_add(o, start, q);
        if (start) order_add(o, 0, end);
    }
    return 0;
}

/* Pairwise summation: the balanced tree, the first ceil(n / 2) terms and then the rest. */
static void pairwise_split(product_order *o, size_t lo, size_t hi) {
    if (hi - lo < 2) return;
    order_add(o, lo, hi);
    size_t mid = lo + (hi - lo + 1) / 2;
    pairwise_split(o, lo, mid);
    pairwise_split(o, mid, hi);
}

int product_order_pairwise(product_order *o, size_t k) {
    if (order_alloc(o, k)) return EXACTREF_NOMEM;
    pairwise_split(o, 0, k);
    return 0;
}

/* The depth of a named order's tree over k terms, which Lemma S takes (the most additions on a
   term's path): sequential k - 1, blocked (blocks of b) min(b, k) - 1 + ceil(k / b) - 1, pairwise
   ceil(log2 k); 0 for at most one term. */
int64_t product_order_depth(int kind, size_t k, size_t b) {
    if (k <= 1) return 0;
    if (kind == 2) {
        int64_t d = 0;
        while (((size_t)1 << d) < k) d++;
        return d;
    }
    if (kind == 3) return (int64_t)(b < k ? b : k) - 1 + (int64_t)((k + b - 1) / b) - 1;
    return (int64_t)k - 1;
}

static int scratch_reserve(product_sums *r, size_t bytes) {
    if (bytes <= r->scratch_size) return 0;
    void *x = realloc(r->scratch, bytes);
    if (!x) return EXACTREF_NOMEM;
    r->scratch = x;
    r->scratch_size = bytes;
    return 0;
}

/* The value of an operand's slot at: a dyad where it is finite (*kind VALUE_FINITE), else only its kind. */
int operand_dyad(dyad *r, int *kind, const operand *x, size_t at) {
    int neg;
    uint64_t sig;
    int64_t e;
    slot_value(kind, &neg, &sig, &e, x, at);
    if (*kind != VALUE_FINITE) return 0;
    r->e = sig ? e : 0;
    return big_set_word(&r->m, sig, neg && sig);
}

/* The fast path: terms a[q] b[q] of main-range integers. */
static w128 fast_sum(const uint64_t *a, const uint64_t *b, size_t k) {
    w2 s = {0}, t;
    for (size_t q = 0; q < k; q++) {
        w2_product(&t, a[q], b[q]);
        w2_add(&s, &t);
    }
    return w2_get(&s);
}

/* The exact sum and the magnitude sum of terms a[q] b[q], unrolled by four. */
static void fast_mass(w128 *s, w128 *mass, const uint64_t *a, const uint64_t *b, size_t k) {
    w2 sum = {0}, m = {0};
    size_t q = 0;
    for (; q + 4 <= k; q += 4) {
        w2 t[4], tm = {0};
        for (int u = 0; u < 4; u++) w2_product(&t[u], a[q + u], b[q + u]);
        for (int u = 0; u < 4; u++) w2_add_abs(&tm, &t[u]);
        w2_add(&m, &tm);
        w2_add(&t[0], &t[1]);
        w2_add(&t[2], &t[3]);
        w2_add(&t[0], &t[2]);
        w2_add(&sum, &t[0]);
    }
    for (; q < k; q++) {
        w2 t;
        w2_product(&t, a[q], b[q]);
        w2_add(&sum, &t);
        w2_add_abs(&m, &t);
    }
    *s = w2_get(&sum);
    *mass = w2_get(&m);
}

/* The magnitudes of the terms with |a| < la or |b| < lb, which a flush may read as zero. */
static w128 fast_moved(const uint64_t *a, const uint64_t *b, size_t k, uint64_t la, uint64_t lb) {
    w2 mv = {0}, t;
    for (size_t q = 0; q < k; q++)
        if (slot_magnitude(a[q]) < la || slot_magnitude(b[q]) < lb) {
            w2_product(&t, a[q], b[q]);
            w2_add_abs(&mv, &t);
        }
    return w2_get(&mv);
}

/* One term into index order (running sum s, node sum n1), its lane of two (p2, n2) and of four
   (p4, n4). */
static inline void node_step(w2 *t, w2 *s, w2 *n1, w2 *p2, w2 *n2, w2 *p4, w2 *n4) {
    w2_add(s, t);
    w2_add_abs(n1, s);
    w2_add(p2, t);
    w2_add_abs(n2, p2);
    w2_add(p4, t);
    w2_add_abs(n4, p4);
}

/* The exact sum s, the magnitude sum and N: twice the mass plus the largest node sum of index
   order and lanes 2 and 4 (every node exact), unrolled by four with the lanes in registers (words.h's w2). */
void product_fast_nodes(w128 *s, w128 *mass, w128 *N, const uint64_t *a, const uint64_t *b, size_t k) {
    w2 sw = {0}, mw = {0}, n1w = {0}, n2w = {0}, n4w = {0}, l2w[2] = {0}, l4w[4] = {0};
    size_t q = 0;
    for (; q + 4 <= k; q += 4) {
        w2 t[4], tm = {0};
        for (int u = 0; u < 4; u++) w2_product(&t[u], a[q + u], b[q + u]);
        for (int u = 0; u < 4; u++) w2_add_abs(&tm, &t[u]);
        w2_add(&mw, &tm);
        node_step(&t[0], &sw, &n1w, &l2w[0], &n2w, &l4w[0], &n4w);
        node_step(&t[1], &sw, &n1w, &l2w[1], &n2w, &l4w[1], &n4w);
        node_step(&t[2], &sw, &n1w, &l2w[0], &n2w, &l4w[2], &n4w);
        node_step(&t[3], &sw, &n1w, &l2w[1], &n2w, &l4w[3], &n4w);
    }
    for (; q < k; q++) {
        w2 t;
        w2_product(&t, a[q], b[q]);
        w2_add_abs(&mw, &t);
        node_step(&t, &sw, &n1w, &l2w[q & 1], &n2w, &l4w[q & 3], &n4w);
    }
    w128 sum = w2_get(&sw), m = w2_get(&mw), n1 = w2_get(&n1w), n2 = w2_get(&n2w), n4 = w2_get(&n4w);
    w128 l2[2] = {w2_get(&l2w[0]), w2_get(&l2w[1])}, l4[4] = {w2_get(&l4w[0]), w2_get(&l4w[1]), w2_get(&l4w[2]), w2_get(&l4w[3])};
    /* each lane's first running sum is its first term, a leaf: those were counted above */
    for (q = 0; q < k && q < 4; q++) {
        w128 u = w128_abs(w128_product(a[q], b[q]));
        if (q < 1) n1 = w128_sub(n1, u);
        if (q < 2) n2 = w128_sub(n2, u);
        n4 = w128_sub(n4, u);
    }
    n2 = w128_add(n2, w128_lane_nodes(l2, k < 2 ? (1u << k) - 1 : 3u, 2));
    n4 = w128_add(n4, w128_lane_nodes(l4, k < 4 ? (1u << k) - 1 : 15u, 4));
    w128 best = w128_less(n1, n2) ? n2 : n1;
    best = w128_less(best, n4) ? n4 : best;
    *s = sum;
    *mass = m;
    *N = w128_twice(w128_add(m, best));
}

/* product_fast_nodes for the terms a[q] b[pos[q]] at the increasing positions pos[q]: index order over the terms, and the
   lanes by position, an empty lane joining no node (the exact zeros between the terms left out, Lemma L) */
void product_fast_nodes_at(w128 *s, w128 *mass, w128 *N, const uint64_t *a, const uint64_t *b, const size_t *pos, size_t k) {
    w2 sw = {0}, mw = {0}, n1w = {0}, n2w = {0}, n4w = {0}, l2w[2] = {0}, l4w[4] = {0};
    unsigned has2 = 0, has4 = 0;
    for (size_t q = 0; q < k; q++) {
        size_t p = pos[q];
        w2 t;
        w2_product(&t, a[q], b[p]);
        w2_add_abs(&mw, &t);
        w2_add(&sw, &t);
        if (q) w2_add_abs(&n1w, &sw);
        w2_add(&l2w[p & 1], &t);
        if (has2 >> (p & 1) & 1) w2_add_abs(&n2w, &l2w[p & 1]);
        w2_add(&l4w[p & 3], &t);
        if (has4 >> (p & 3) & 1) w2_add_abs(&n4w, &l4w[p & 3]);
        has2 |= 1u << (p & 1);
        has4 |= 1u << (p & 3);
    }
    w128 l2[2] = {w2_get(&l2w[0]), w2_get(&l2w[1])}, l4[4] = {w2_get(&l4w[0]), w2_get(&l4w[1]), w2_get(&l4w[2]), w2_get(&l4w[3])};
    w128 m = w2_get(&mw), n1 = w2_get(&n1w), n2 = w128_add(w2_get(&n2w), w128_lane_nodes(l2, has2, 2)),
         n4 = w128_add(w2_get(&n4w), w128_lane_nodes(l4, has4, 4));
    w128 best = w128_less(n1, n2) ? n2 : n1;
    best = w128_less(best, n4) ? n4 : best;
    *s = w2_get(&sw);
    *mass = m;
    *N = w128_twice(w128_add(m, best));
}

/* The fast path with the wide main range: two-word slots, 256-bit words. */

static void slot256_product(w4 *r, const operand *a, size_t i, const operand *b, size_t j) {
    w4_product(r, a->slots[i], a->slots_hi[i], b->slots[j], b->slots_hi[j]);
}

static int big_from_w256(bigint *x, w256 v, int is_signed) {
    int neg = is_signed && w256_negative(v);
    if (neg) v = w256_abs(v);
    if (big_reserve(x, 4)) return EXACTREF_NOMEM;
    for (int k = 0; k < 4; k++) x->w[k] = v.w[k];
    x->n = 4;
    x->neg = 0;
    big_trim(x);
    x->neg = neg && x->n;
    return 0;
}

/* product_element's quantities on the wide main range (two-word slots), in 256-bit words: the exact sum, the mass,
   with nodes N, and with flush the moved magnitudes (factors below la or lb in magnitude, two-word limits). */
static void fast256(w256 *s, w256 *mass, w256 *moved, w256 *N, const operand *a, size_t i0, const operand *b, size_t j0, size_t k, int nodes,
                    int flush, uint64_t la_lo, uint64_t la_hi, uint64_t lb_lo, uint64_t lb_hi) {
    w4 sw = {0}, mw = {0}, mvw = {0}, n1w = {0}, n2w = {0}, n4w = {0}, l2w[2] = {0}, l4w[4] = {0}, t;
    for (size_t q = 0; q < k; q++) {
        slot256_product(&t, a, i0 + q, b, j0 + q);
        w4_add_abs(&mw, &t);
        w4_add(&sw, &t);
        if (flush) {
            /* the factors' magnitudes against the limits, as two-word integers */
            uint64_t xs[2][2] = {{a->slots[i0 + q], a->slots_hi[i0 + q]}, {b->slots[j0 + q], b->slots_hi[j0 + q]}};
            uint64_t limits[2][2] = {{la_lo, la_hi}, {lb_lo, lb_hi}};
            int tiny = 0;
            for (int side = 0; side < 2; side++) {
                uint64_t lo = xs[side][0], hi = xs[side][1];
                if (hi >> 63) {
                    lo = ~lo + 1;
                    hi = ~hi + (lo == 0);
                }
                tiny |= hi != limits[side][1] ? hi < limits[side][1] : lo < limits[side][0];
            }
            if (tiny) w4_add_abs(&mvw, &t);
        }
        if (!nodes) continue;
        w4_add_abs(&n1w, &sw);
        w4_add(&l2w[q & 1], &t);
        w4_add_abs(&n2w, &l2w[q & 1]);
        w4_add(&l4w[q & 3], &t);
        w4_add_abs(&n4w, &l4w[q & 3]);
    }
    w256 m = w4_get(&mw);
    *s = w4_get(&sw);
    *mass = m;
    *moved = w4_get(&mvw);
    if (!nodes) return;
    for (size_t q = 0; q < k && q < 4; q++) {
        w4 u = {0};
        slot256_product(&t, a, i0 + q, b, j0 + q);
        w4_add_abs(&u, &t);
        if (q < 1) w4_sub(&n1w, &u);
        if (q < 2) w4_sub(&n2w, &u);
        w4_sub(&n4w, &u);
    }
    w256 n1 = w4_get(&n1w), n2 = w4_get(&n2w), n4 = w4_get(&n4w);
    w256 l2[2] = {w4_get(&l2w[0]), w4_get(&l2w[1])}, l4[4] = {w4_get(&l4w[0]), w4_get(&l4w[1]), w4_get(&l4w[2]), w4_get(&l4w[3])};
    n2 = w256_add(n2, w256_lane_nodes(l2, k < 2 ? (1u << k) - 1 : 3u, 2));
    n4 = w256_add(n4, w256_lane_nodes(l4, k < 4 ? (1u << k) - 1 : 15u, 4));
    w256 best = w256_less(n1, n2) ? n2 : n1;
    best = w256_less(best, n4) ? n4 : best;
    *N = w256_twice(w256_add(m, best));
}

/* tiny_limit for two-word integers: 2**(emin - E) as (lo, hi), all ones where every integer is below. */
static void tiny_limit256(uint64_t *lo, uint64_t *hi, const operand *x, int64_t emin) {
    int64_t d = emin - x->exponent;
    *lo = *hi = 0;
    if (d <= 0) return;
    if (d >= 127) {
        *lo = *hi = UINT64_MAX;
        return;
    }
    if (d < 64)
        *lo = (uint64_t)1 << d;
    else
        *hi = (uint64_t)1 << (d - 64);
}

/* A named order on the fast path: the exact sum, the mass and N = 2 (mass + its node sum), from
   the running sums kept in prefix (k + 1 words). */
static void fast_order(w128 *s, w128 *mass, w128 *N, w2 *prefix, const uint64_t *a, const uint64_t *b, size_t k, const product_order *o) {
    w2 m = {0}, nodes = {0}, t;
    w2_set(&prefix[0], w128_zero());
    for (size_t q = 0; q < k; q++) {
        w2_product(&t, a[q], b[q]);
        w2_add_abs(&m, &t);
        w2_set(&prefix[q + 1], w2_get(&prefix[q]));
        w2_add(&prefix[q + 1], &t);
    }
    for (size_t n = 0; n < o->count; n++) {
        w2_set(&t, w128_zero());
        w2_add(&t, &prefix[o->hi[n]]);
        w2_sub(&t, &prefix[o->lo[n]]);
        w2_add_abs(&nodes, &t);
    }
    *s = w2_get(&prefix[k]);
    *mass = w2_get(&m);
    *N = w128_twice(w128_add(w2_get(&m), w2_get(&nodes)));
}

/* Multi-word two's-complement integers of W words, for the wide path. */

/* x = sign * (lo + 2**64 hi) * 2**shift. */
static void mw_term(uint64_t *x, size_t W, uint64_t lo, uint64_t hi, int neg, int64_t shift) {
    for (size_t i = 0; i < W; i++) x[i] = 0;
    size_t at = (size_t)(shift / 64);
    int b = (int)(shift % 64);
    uint64_t w[3] = {lo << b, b ? hi << b | lo >> (64 - b) : hi, b ? hi >> (64 - b) : 0};
    for (size_t i = 0; i < 3 && at + i < W; i++) x[at + i] = w[i];
    if (neg) {
        for (size_t i = 0; i < W; i++) x[i] = ~x[i];
        nat_add_word(x, 1, W);
    }
}

/* acc += |x|. */
static void mw_abs_add(uint64_t *acc, const uint64_t *x, size_t W) {
    if (mw_negative(x, W))
        nat_sub(acc, acc, x, W, 0);
    else
        nat_add(acc, acc, x, W, 0);
}

/* r = x as a signed integer. */
static int mw_big(bigint *r, const uint64_t *x, size_t W) {
    if (big_reserve(r, W)) return EXACTREF_NOMEM;
    int neg = mw_negative(x, W);
    for (size_t i = 0; i < W; i++) r->w[i] = neg ? ~x[i] : x[i];
    r->n = W;
    r->neg = 0;
    if (neg) nat_add_word(r->w, 1, W);
    big_trim(r);
    r->neg = neg && r->n;
    return 0;
}

/* The wide path: element (i, j) at the exponent of its finest term, in multi-word integers. */
static int wide_terms(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, size_t k, int what, int flush, int64_t emin,
                      const product_order *order) {
    /* every term is an integer at 2**f: a flagged value's lowest bit, or a slot's integer at its operand's exponent, is at or above it */
    int64_t f = (a->low[i] < a->exponent ? a->low[i] : a->exponent) + (b->low[j] < b->exponent ? b->low[j] : b->exponent);
    int64_t top = a->top[i] + b->top[j];
    /* products below 2**(top - f) as integers at 2**f; sums of k of them, and node sums of k of those, with a sign */
    int64_t bits = (top > f ? top - f : 0) + 2 * (int64_t)word_bits((uint64_t)k) + 3;
    size_t W = (size_t)(bits + 63) / 64;
    enum { S, MASS, MOVED, N1, N2, N4, L2, L4 = L2 + 2, T = L4 + 4, X, COUNT = X + 3 };
    uint64_t *buf = calloc(COUNT * W, sizeof *buf);
    if (!buf) return EXACTREF_NOMEM;
    uint64_t *x[COUNT];
    for (int n = 0; n < COUNT; n++) x[n] = buf + (size_t)n * W;
    uint64_t *prefix = NULL; /* a named order's running sums, (k + 1) W words */
    if (order && what >= PRODUCT_NODES && !(prefix = calloc((k + 1) * W, sizeof *prefix))) {
        free(buf);
        return EXACTREF_NOMEM;
    }
    for (size_t q = 0; q < k; q++) {
        int ka, kb, na, nb;
        uint64_t sa, sb, hi;
        int64_t ea, eb;
        slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
        slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
        uint64_t lo = word_mul(&hi, sa, sb);
        mw_term(x[T], W, lo, hi, na != nb, lo | hi ? ea + eb - f : 0);
        mw_abs_add(x[MASS], x[T], W);
        if (flush && ((sa && ea + word_bits(sa) <= emin) || (sb && eb + word_bits(sb) <= emin))) mw_abs_add(x[MOVED], x[T], W);
        nat_add(x[S], x[S], x[T], W, 0);
        if (prefix) nat_add(prefix + (q + 1) * W, prefix + q * W, x[T], W, 0);
        if (what < PRODUCT_NODES || prefix) continue;
        if (q >= 1) mw_abs_add(x[N1], x[S], W);
        nat_add(x[L2 + (q & 1)], x[L2 + (q & 1)], x[T], W, 0);
        if (q >= 2) mw_abs_add(x[N2], x[L2 + (q & 1)], W);
        nat_add(x[L4 + (q & 3)], x[L4 + (q & 3)], x[T], W, 0);
        if (q >= 4) mw_abs_add(x[N4], x[L4 + (q & 3)], W);
    }
    r->e = f;
    int rc = mw_big(&r->s, x[S], W);
    if (!rc && what >= PRODUCT_MASS && !(rc = mw_big(&r->mass, x[MASS], W))) rc = mw_big(&r->moved, x[MOVED], W);
    if (!rc && prefix) {
        /* N = 2 (mass + the order's node sum) */
        for (size_t n = 0; n < order->count; n++) {
            nat_sub(x[T], prefix + order->hi[n] * W, prefix + order->lo[n] * W, W, 0);
            mw_abs_add(x[N1], x[T], W);
        }
        nat_add(x[N1], x[N1], x[MASS], W, 0);
        nat_add(x[N1], x[N1], x[N1], W, 0);
        rc = mw_big(&r->N, x[N1], W);
    } else if (!rc && what >= PRODUCT_NODES) {
        /* each lane order's node sum with its combination's; N = 2 (mass + the largest) */
        const uint64_t *best = x[N1];
        for (int L = 2; L <= 4; L += 2) {
            uint64_t *n = x[L == 2 ? N2 : N4];
            lane_nodes_words(x[T], x[L == 2 ? L2 : L4], k < (size_t)L ? (1u << k) - 1 : (1u << L) - 1, L, W, x[X]);
            nat_add(n, n, x[T], W, 0);
            if (nat_cmp(n, best, W) > 0) best = n;
        }
        bigint b;
        big_init(&b);
        if (!(rc = mw_big(&b, best, W)) && !(rc = big_add(&b, &b, &r->mass))) rc = big_shl(&r->N, &b, 1);
        big_free(&b);
    }
    free(buf);
    free(prefix);
    return rc;
}

/* The per-block path's running sums (index order's, then each lane of two and of four) */
enum { RUN_S, RUN_L2, RUN_L4 = RUN_L2 + 2, RUNS = RUN_L4 + 4 };

/* x = v 2**shift as a W-word two's complement integer, for a signed 128-bit v. */
static void mw_from_w128(uint64_t *x, size_t W, w128 v, int64_t shift) {
    int neg = w128_negative(v);
    w128 m = w128_abs(v);
    mw_term(x, W, w128_lo(m), w128_hi(m), neg, shift);
}

/* For a W-word two's complement x and sh >= 0: *main = floor(x / 2**sh), which fits 128 bits, and
   rem = x - main 2**sh, with 0 <= rem < 2**sh. */
static void mw_split(w128 *main, uint64_t *rem, const uint64_t *x, size_t W, int64_t sh) {
    size_t at = (size_t)(sh / 64);
    int bit = (int)(sh % 64);
    uint64_t fill = mw_negative(x, W) ? UINT64_MAX : 0, w[3];
    for (size_t n = 0; n < 3; n++) w[n] = at + n < W ? x[at + n] : fill;
    uint64_t lo = bit ? w[0] >> bit | w[1] << (64 - bit) : w[0], hi = bit ? w[1] >> bit | w[2] << (64 - bit) : w[1];
    *main = w128_make(lo, hi);
    for (size_t n = 0; n < W; n++) rem[n] = n < at ? x[n] : n == at && bit ? x[n] & (((uint64_t)1 << bit) - 1) : 0;
}

/* node += rem (pos - neg): a remainder's share of the nodes since it was set. */
static void mw_count(uint64_t *node, const uint64_t *rem, size_t W, int64_t pos, int64_t neg) {
    if (pos > neg)
        nat_addmul_word(node, rem, W, (uint64_t)(pos - neg));
    else if (neg > pos)
        nat_submul_word(node, rem, W, (uint64_t)(neg - pos));
}

/* The term block size for element (i, j)'s per-block path (proofs/kernels.md): the read kernel's blocks of 8,
   or 0, the wide path throughout, where more than 60% of them hold a flagged slot. Measured on
   GPT-2's operands with flagged values injected: with clean
   stretches on the fast loops, blocks of 8 beat 16 and 32 at every share, and the wide path
   overtakes them at about 60% of the tolerance's blocks and 80% of the reference's. */
static size_t block_size_for(const operand *a, size_t i, const operand *b, size_t j) {
    size_t blocks = a->blocks, flagged = 0;
    const unsigned char *x = a->summary + i * blocks, *y = b->summary + j * blocks;
    for (size_t n = 0; n < blocks; n++) flagged += (x[n] | y[n]) != 0;
    return 5 * flagged > 3 * blocks ? 0 : READ_BLOCK;
}

/* A clean stretch of terms q0 .. q1 - 1 (q0 a multiple of 4) on the main grid: the magnitude sum,
   each running sum's main part M (index order, lanes of two and four by q's position), the
   orders' sums of |M| (nm) and each running sum's count of negative M (neg). */
static void run_nodes(w128 *mass, w128 *M, w128 *nm, int64_t *neg, const uint64_t *a, const uint64_t *b, size_t q0, size_t q1) {
    w2 m, s, l2[2], l4[4], n[3];
    w2_set(&m, *mass);
    w2_set(&s, M[RUN_S]);
    for (int l = 0; l < 2; l++) w2_set(&l2[l], M[RUN_L2 + l]);
    for (int l = 0; l < 4; l++) w2_set(&l4[l], M[RUN_L4 + l]);
    for (int l = 0; l < 3; l++) w2_set(&n[l], nm[l]);
    int64_t c[RUNS] = {0};
    size_t q = q0;
    for (; q + 4 <= q1; q += 4) {
        w2 t[4], tm = {0};
        for (int u = 0; u < 4; u++) w2_product(&t[u], a[q + u], b[q + u]);
        for (int u = 0; u < 4; u++) w2_add_abs(&tm, &t[u]);
        w2_add(&m, &tm);
        for (int u = 0; u < 4; u++) {
            node_step(&t[u], &s, &n[0], &l2[u & 1], &n[1], &l4[u], &n[2]);
            c[RUN_S] += w2_negative(&s);
            c[RUN_L2 + (u & 1)] += w2_negative(&l2[u & 1]);
            c[RUN_L4 + u] += w2_negative(&l4[u]);
        }
    }
    for (; q < q1; q++) {
        w2 t;
        w2_product(&t, a[q], b[q]);
        w2_add_abs(&m, &t);
        node_step(&t, &s, &n[0], &l2[q & 1], &n[1], &l4[q & 3], &n[2]);
        c[RUN_S] += w2_negative(&s);
        c[RUN_L2 + (q & 1)] += w2_negative(&l2[q & 1]);
        c[RUN_L4 + (q & 3)] += w2_negative(&l4[q & 3]);
    }
    *mass = w2_get(&m);
    M[RUN_S] = w2_get(&s);
    for (int l = 0; l < 2; l++) M[RUN_L2 + l] = w2_get(&l2[l]);
    for (int l = 0; l < 4; l++) M[RUN_L4 + l] = w2_get(&l4[l]);
    for (int l = 0; l < 3; l++) nm[l] = w2_get(&n[l]);
    for (int r = 0; r < RUNS; r++) neg[r] += c[r];
}

/* The number of terms of q0 .. q1 - 1 that feed running sum n (index order all, lane l of L those
   with q mod L = l). */
static int64_t run_count(int n, size_t q0, size_t q1) {
    if (n == RUN_S) return (int64_t)(q1 - q0);
    size_t L = n < RUN_L4 ? 2 : 4, l = (size_t)(n < RUN_L4 ? n - RUN_L2 : n - RUN_L4);
    /* terms q with q0 <= q < q1 and q mod L = l */
    size_t first = q0 + (l + L - q0 % L) % L;
    return first < q1 ? (int64_t)((q1 - first + L - 1) / L) : 0;
}

/* The per-block path for element (i, j) (proofs/kernels.md; one-word slots, index order and the lanes): blocks of
   `block` terms with no flagged slot on 128-bit words at the main grid 2**E, the others in
   multi-word integers at the element's finest exponent f, as the wide path takes them. Each
   running sum is held as a main part M on the main grid and a remainder 0 <= rem < u = 2**(E - f);
   in a clean block its node |M u + rem| is |M| u + rem where M >= 0 and |M| u - rem where M < 0,
   so the loop adds |M| to a 128-bit node sum and counts each sign, and rem times the difference
   of the counts joins the node sum exactly whenever rem changes (proofs/kernels.md, Lemma P). */
static int mixed_terms(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, size_t k, int what, int flush, int64_t emin,
                       size_t block) {
    int64_t f = (a->low[i] < a->exponent ? a->low[i] : a->exponent) + (b->low[j] < b->exponent ? b->low[j] : b->exponent);
    int64_t E = a->exponent + b->exponent, sh = E - f, top = a->top[i] + b->top[j];
    /* as the wide path's words, over the main grid's unit too (a remainder is below it), and a main part (128 bits) times it */
    int64_t high = top > E ? top : E, bits = (high > f ? high - f : 0) + 2 * (int64_t)word_bits((uint64_t)k) + 4;
    if (bits < sh + 130) bits = sh + 130;
    size_t W = (size_t)(bits + 63) / 64;
    /* per running sum: its value while in a flagged block, and its remainder; the node sums' and
       sums' multi-word parts; a term */
    enum { V = 0, REM = V + RUNS, NX = REM + RUNS, SX = NX + 3, MASSX, MOVEDX, T, X, COUNT = X + 3 };
    if (scratch_reserve(r, COUNT * W * sizeof(uint64_t))) return EXACTREF_NOMEM;
    uint64_t *buf = r->scratch;
    memset(buf, 0, COUNT * W * sizeof *buf);
    uint64_t *x[COUNT];
    for (int n = 0; n < COUNT; n++) x[n] = buf + (size_t)n * W;
    w128 z = w128_zero(), M[RUNS], nm[3] = {z, z, z}, mass = z, moved = z;
    int64_t pos[RUNS] = {0}, neg[RUNS] = {0};
    for (int n = 0; n < RUNS; n++) M[n] = z;
    static const int order_of[RUNS] = {0, 1, 1, 2, 2, 2, 2};
    int nodes = what >= PRODUCT_NODES, runs = nodes ? RUNS : 1;
    uint64_t la = flush ? tiny_limit(a, emin) : 0, lb = flush ? tiny_limit(b, emin) : 0;
    const uint64_t *xa = a->slots + i * k, *xb = b->slots + j * k;
    const unsigned char *sa_sum = a->summary + i * a->blocks, *sb_sum = b->summary + j * b->blocks;
    for (size_t q0 = 0; q0 < k;) {
        /* the clean stretch from q0: whole blocks with no flagged slot, on the fast loops in one call */
        size_t q1 = q0;
        for (;;) {
            size_t e = q1 + block < k ? q1 + block : k;
            int clean = q1 < k;
            for (size_t n = q1 / READ_BLOCK; clean && n < (e + READ_BLOCK - 1) / READ_BLOCK; n++)
                if (sa_sum[n] || sb_sum[n]) clean = 0;
            if (!clean) break;
            q1 = e;
        }
        if (q1 > q0) {
            if (nodes) {
                int64_t before[RUNS];
                for (int n = 0; n < RUNS; n++) before[n] = neg[n];
                run_nodes(&mass, M, nm, neg, xa, xb, q0, q1);
                for (int n = 0; n < RUNS; n++) pos[n] += run_count(n, q0, q1) - (neg[n] - before[n]);
            } else if (what >= PRODUCT_MASS) {
                w128 s_, m;
                fast_mass(&s_, &m, xa + q0, xb + q0, q1 - q0);
                M[RUN_S] = w128_add(M[RUN_S], s_);
                mass = w128_add(mass, m);
            } else {
                M[RUN_S] = w128_add(M[RUN_S], fast_sum(xa + q0, xb + q0, q1 - q0));
            }
            if (flush && what >= PRODUCT_MASS) moved = w128_add(moved, fast_moved(xa + q0, xb + q0, q1 - q0, la, lb));
            q0 = q1;
            continue;
        }
        /* a flagged block: each running sum in full, its remainder's nodes settled */
        q1 = q0 + block < k ? q0 + block : k;
        for (int n = 0; n < runs; n++) {
            mw_count(x[NX + order_of[n]], x[REM + n], W, pos[n], neg[n]);
            pos[n] = neg[n] = 0;
            mw_from_w128(x[V + n], W, M[n], sh);
            nat_add(x[V + n], x[V + n], x[REM + n], W, 0);
        }
        for (size_t q = q0; q < q1; q++) {
            int ka, kb, na, nb;
            uint64_t sa, sb, hi;
            int64_t ea, eb;
            slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
            slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
            uint64_t lo = word_mul(&hi, sa, sb);
            mw_term(x[T], W, lo, hi, na != nb, lo | hi ? ea + eb - f : 0);
            if (what >= PRODUCT_MASS) mw_abs_add(x[MASSX], x[T], W);
            if (flush && ((sa && ea + word_bits(sa) <= emin) || (sb && eb + word_bits(sb) <= emin))) mw_abs_add(x[MOVEDX], x[T], W);
            int at[3] = {RUN_S, RUN_L2 + (int)(q & 1), RUN_L4 + (int)(q & 3)};
            for (int o = 0; o < (nodes ? 3 : 1); o++) {
                nat_add(x[V + at[o]], x[V + at[o]], x[T], W, 0);
                if (nodes) mw_abs_add(x[NX + o], x[V + at[o]], W);
            }
        }
        for (int n = 0; n < runs; n++) mw_split(&M[n], x[REM + n], x[V + n], W, sh);
        q0 = q1;
    }
    /* the remainders' last nodes; then each running sum, node sum and sum in full at f */
    for (int n = 0; n < runs; n++) {
        mw_count(x[NX + order_of[n]], x[REM + n], W, pos[n], neg[n]);
        mw_from_w128(x[V + n], W, M[n], sh);
        nat_add(x[V + n], x[V + n], x[REM + n], W, 0);
    }
    mw_from_w128(x[T], W, mass, sh);
    nat_add(x[MASSX], x[MASSX], x[T], W, 0);
    mw_from_w128(x[T], W, moved, sh);
    nat_add(x[MOVEDX], x[MOVEDX], x[T], W, 0);
    r->e = f;
    int rc = mw_big(&r->s, x[V + RUN_S], W);
    if (!rc && what >= PRODUCT_MASS && !(rc = mw_big(&r->mass, x[MASSX], W))) rc = mw_big(&r->moved, x[MOVEDX], W);
    if (!rc && nodes) {
        /* each order's node sum in full; each lane's first running sum is its first term, a leaf: taken back */
        for (int o = 0; o < 3; o++) {
            mw_from_w128(x[T], W, nm[o], sh);
            nat_add(x[NX + o], x[NX + o], x[T], W, 0);
        }
        for (size_t q = 0; q < k && q < 4; q++) {
            int ka, kb, na, nb;
            uint64_t sa, sb, hi;
            int64_t ea, eb;
            slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
            slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
            uint64_t lo = word_mul(&hi, sa, sb);
            mw_term(x[T], W, lo, hi, 0, lo | hi ? ea + eb - f : 0);
            if (q < 1) nat_sub(x[NX], x[NX], x[T], W, 0);
            if (q < 2) nat_sub(x[NX + 1], x[NX + 1], x[T], W, 0);
            nat_sub(x[NX + 2], x[NX + 2], x[T], W, 0);
        }
        /* each lane order's node sum with its combination's; N = 2 (mass + the largest) */
        const uint64_t *best = x[NX];
        for (int L = 2; L <= 4; L += 2) {
            uint64_t *n = x[NX + L / 2];
            lane_nodes_words(x[T], x[V + (L == 2 ? RUN_L2 : RUN_L4)], k < (size_t)L ? (1u << k) - 1 : (1u << L) - 1, L, W, x[X]);
            nat_add(n, n, x[T], W, 0);
            if (nat_cmp(n, best, W) > 0) best = n;
        }
        bigint b;
        big_init(&b);
        if (!(rc = mw_big(&b, best, W)) && !(rc = big_add(&b, &b, &r->mass))) rc = big_shl(&r->N, &b, 1);
        big_free(&b);
    }
    return rc;
}

/* The IEEE value of element (i, j)'s sum, which has a non-finite term: NaN where a term is NaN (a
   NaN factor, or an infinity times zero) or infinities of both signs meet, else the infinity. */
static void special_sum(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, size_t k) {
    int nan = 0, plus = 0, minus = 0;
    for (size_t q = 0; q < k && !nan; q++) {
        int ka, kb, na, nb;
        uint64_t sa, sb;
        int64_t ea, eb;
        if (slot_absent(b, j * b->cols + q)) continue;
        slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
        slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
        if (ka == VALUE_FINITE && kb == VALUE_FINITE) continue;
        if (ka == VALUE_NAN || kb == VALUE_NAN || (ka == VALUE_FINITE && !sa) || (kb == VALUE_FINITE && !sb))
            nan = 1;
        else if (na != nb)
            minus = 1;
        else
            plus = 1;
    }
    /* finite where only absent terms were not */
    r->kind = nan || (plus && minus) ? VALUE_NAN : plus || minus ? VALUE_INFINITE : VALUE_FINITE;
    r->neg = r->kind == VALUE_INFINITE && minus;
}

/* An element whose sum has a non-finite term (product_element gave it that kind): its finite terms' sum and mass, a
   non-finite slot reading as 0, on its flagged row's paths (a row with a non-finite value is flagged); r's kind kept. */
int product_finite_terms(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, int flush, int64_t emin) {
    size_t k = a->cols, block = !a->slots_hi && !b->slots_hi ? block_size_for(a, i, b, j) : 0;
    return block ? mixed_terms(r, a, i, b, j, k, PRODUCT_MASS, flush, emin, block) : wide_terms(r, a, i, b, j, k, PRODUCT_MASS, flush, emin, NULL);
}

/* Element (i, j)'s quantities: what is PRODUCT_SUM, PRODUCT_MASS or PRODUCT_NODES, the nodes those
   of index order and lanes 2 and 4, or of order when given (then without flush); with flush, the
   moved magnitudes are those of the terms with a factor below 2**emin. A sum with a non-finite
   term gets only its IEEE kind and sign. */
int product_element(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, int what, int flush, int64_t emin,
                    const product_order *order) {
    size_t k = a->cols;
    r->kind = VALUE_FINITE;
    r->neg = 0;
    if (a->special_rows[i] || b->special_rows[j]) {
        special_sum(r, a, i, b, j, k);
        if (r->kind != VALUE_FINITE) return 0;
        /* only absent terms were not finite: they add nothing (a non-finite slot reads as 0) */
    }
    if (a->flagged_rows[i] || b->flagged_rows[j]) {
        /* a few flagged blocks take the per-block path; named orders, two-word slots and rows flagged throughout the wide path */
        size_t block = !(order && what >= PRODUCT_NODES) && !a->slots_hi && !b->slots_hi ? block_size_for(a, i, b, j) : 0;
        return block ? mixed_terms(r, a, i, b, j, k, what, flush, emin, block) : wide_terms(r, a, i, b, j, k, what, flush, emin, order);
    }
    const uint64_t *x = a->slots + i * k, *y = b->slots + j * k;
    r->e = a->exponent + b->exponent;
    if (a->slots_hi && b->slots_hi && !(order && what >= PRODUCT_NODES)) {
        /* the wide main range: 256-bit words (named orders take the wide path below) */
        w256 s, mass, moved, N = w256_zero();
        uint64_t la_lo = 0, la_hi = 0, lb_lo = 0, lb_hi = 0;
        if (flush) {
            tiny_limit256(&la_lo, &la_hi, a, emin);
            tiny_limit256(&lb_lo, &lb_hi, b, emin);
        }
        fast256(&s, &mass, &moved, &N, a, i * k, b, j * k, k, what >= PRODUCT_NODES, flush && what >= PRODUCT_MASS, la_lo, la_hi, lb_lo, lb_hi);
        int rc = big_from_w256(&r->s, s, 1);
        if (!rc && what >= PRODUCT_MASS && !(rc = big_from_w256(&r->mass, mass, 0)) && !(rc = big_from_w256(&r->moved, moved, 0)))
            rc = big_from_w256(&r->N, N, 0);
        return rc;
    }
    if (a->slots_hi || b->slots_hi) return wide_terms(r, a, i, b, j, k, what, flush, emin, order);
    if (what == PRODUCT_SUM) return big_from_w128(&r->s, fast_sum(x, y, k), 1);
    w128 s, mass, moved = w128_zero(), N = w128_zero();
    if (order && what >= PRODUCT_NODES) {
        if (scratch_reserve(r, (k + 1) * sizeof(w2))) return EXACTREF_NOMEM;
        fast_order(&s, &mass, &N, (w2 *)r->scratch, x, y, k, order);
    } else if (what >= PRODUCT_NODES) {
        product_fast_nodes(&s, &mass, &N, x, y, k);
    } else {
        fast_mass(&s, &mass, x, y, k);
    }
    if (flush && !(order && what >= PRODUCT_NODES)) moved = fast_moved(x, y, k, tiny_limit(a, emin), tiny_limit(b, emin));
    int rc = big_from_w128(&r->s, s, 1);
    if (!rc) rc = big_from_w128(&r->mass, mass, 0);
    if (!rc) rc = big_from_w128(&r->moved, moved, 0);
    if (!rc) rc = big_from_w128(&r->N, N, 0);
    return rc;
}

/* The sign of element (i, j)'s exact zero when it is not a reduction (an elementwise product's
   element): the sign IEEE 754 6.3 gives its expression. Each term's sign is the exclusive or of
   its factors'; terms that are all zeros of one sign keep it, and otherwise (zeros of both signs,
   or nonzero terms that cancel) the sum's sign rule applies: -0 when rounding toward -inf, else
   +0. */
int product_zero_sign(const operand *a, size_t i, const operand *b, size_t j, int rnd) {
    int rule = rnd == ROUND_TOWARD_NEGATIVE, first = -1;
    for (size_t q = 0; q < a->cols; q++) {
        int ka, kb, na, nb;
        uint64_t sa, sb;
        int64_t ea, eb;
        if (slot_absent(b, j * b->cols + q)) continue;
        slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
        slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
        if (sa && sb) return rule; /* nonzero terms that cancel */
        int neg = na != nb;
        if (first < 0)
            first = neg;
        else if (neg != first)
            return rule;
    }
    return first < 0 ? rule : first;
}

/* slot q of row `row` of x as a dyad v, and its radius in rx (NULL where exact) as r: *finite 0 where either is not
   finite */
static int slot_with_radius(int *finite, dyad *v, dyad *r, const operand *x, const operand *rx, size_t row, size_t q) {
    int kind, n, rc;
    uint64_t sig;
    int64_t e;
    slot_value(&kind, &n, &sig, &e, x, row * x->cols + q);
    *finite = kind == VALUE_FINITE;
    if (!*finite) return 0;
    if ((rc = big_set_word(&v->m, sig, 0))) return rc;
    v->m.neg = n && sig;
    v->e = sig ? e : 0;
    if ((rc = dyad_set_si(r, 0, 0)) || !rx) return rc;
    slot_value(&kind, &n, &sig, &e, rx, row * rx->cols + q);
    if (kind != VALUE_FINITE) {
        *finite = 0;
        return 0;
    }
    if ((rc = big_set_word(&r->m, sig, 0))) return rc;
    r->e = sig ? e : 0;
    return 0;
}

/* Term q of element (i, j), each factor anywhere within its radius (rx, ry 0 where exact): *finite where both factors
   are finite, then the factors x and y and the term's least and largest values lo, hi (its corners' products). */
static int term_corners(int *finite, dyad *x, dyad *rx, dyad *y, dyad *ry, dyad *lo, dyad *hi, const operand *a, const operand *ra, size_t i,
                        const operand *b, const operand *rb, size_t j, size_t q) {
    int fa, fb, rc;
    *finite = 0;
    if ((rc = slot_with_radius(&fa, x, rx, a, ra, i, q)) || (rc = slot_with_radius(&fb, y, ry, b, rb, j, q)) || !fa || !fb) return rc;
    dyad xs[2], ys[2], c, t;
    for (int k = 0; k < 2; k++) {
        dyad_init(&xs[k]);
        dyad_init(&ys[k]);
    }
    dyad_init(&c);
    dyad_init(&t);
    if (!(rc = dyad_neg(&t, rx)) && !(rc = dyad_add_exact(&xs[0], x, &t)) && !(rc = dyad_add_exact(&xs[1], x, rx)) && !(rc = dyad_neg(&t, ry)) &&
        !(rc = dyad_add_exact(&ys[0], y, &t)))
        rc = dyad_add_exact(&ys[1], y, ry);
    for (int k = 0; !rc && k < 4; k++) {
        if ((rc = dyad_mul_exact(&c, &xs[k / 2], &ys[k % 2]))) break;
        if (!k) {
            if (!(rc = dyad_copy(lo, &c))) rc = dyad_copy(hi, &c);
        } else if (dyad_cmp(&c, lo) < 0) {
            rc = dyad_copy(lo, &c);
        } else if (dyad_cmp(&c, hi) > 0) {
            rc = dyad_copy(hi, &c);
        }
    }
    for (int k = 0; k < 2; k++) {
        dyad_free(&xs[k]);
        dyad_free(&ys[k]);
    }
    dyad_free(&c);
    dyad_free(&t);
    *finite = !rc;
    return rc;
}

/* The sides on which alpha's application may take a term past theta (gemm.md, Overflow), its corners lo, hi and factors
   x within rx, y within ry: the scaled term, |sigma| (|t| (1 + r) + a); a scaled factor, |sigma| (|x| + rx), itself (an
   infinity times the other factor: NaN where that may be 0); and a scaled factor's product, (|sigma| (|x| + rx) (1 + r) +
   a) (|y| + ry). Each on the sides the term's corners take (the caller flips them for a negative sigma). With two channels
   in the part (sc->two), a factor's fold is the sum of its two channel products, its sign not the term's: its fold
   bound F at the factor's element (EX, EY: its largest part plus radius), F itself and F times the other element, give
   both infinities and NaN. */
static int scaled_term_flags(int *flags, const dyad *lo, const dyad *hi, const dyad *x, const dyad *rx, const dyad *y, const dyad *ry,
                             const dyad *EX, const dyad *EY, const term_scale *sc, const dyad *theta) {
    dyad X, Y, u, v, one_r;
    dyad *all[] = {&X, &Y, &u, &v, &one_r};
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_init(all[n]);
    int rc = dyad_abs(&X, x), sides = (dyad_sign(hi) > 0 ? FLAG_PLUS_INF : 0) | (dyad_sign(lo) < 0 ? FLAG_MINUS_INF : 0);
    if (!rc && !(rc = dyad_add_exact(&X, &X, rx)) && !(rc = dyad_abs(&Y, y)) && !(rc = dyad_add_exact(&Y, &Y, ry)) && !(rc = dyad_set_si(&one_r, 1, 0)))
        rc = dyad_add_exact(&one_r, &one_r, &sc->r);
    /* the scaled term, per side */
    for (int side = 0; !rc && side < 2; side++) {
        if (!(sides & (side ? FLAG_MINUS_INF : FLAG_PLUS_INF))) continue;
        if (!(rc = dyad_abs(&u, side ? lo : hi)) && !(rc = dyad_mul(&u, &u, &one_r, 64, 1)) && !(rc = dyad_add(&u, &u, &sc->a, 64, 1)) &&
            !(rc = dyad_mul(&u, &u, &sc->sigma, 64, 1)) && dyad_cmp(&u, theta) >= 0)
            *flags |= side ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    }
    /* a two-channel fold of either factor, and its product with the other factor */
    for (int k = 0; !rc && sc->two && k < 2; k++) {
        if ((rc = dyad_mul(&u, k ? EY : EX, &sc->fold_mul, 64, 1)) || (rc = dyad_add(&u, &u, &sc->fold_add, 64, 1))) break;
        if (dyad_cmp(&u, theta) >= 0 || (!(rc = dyad_mul(&u, &u, k ? EX : EY, 64, 1)) && dyad_cmp(&u, theta) >= 0)) *flags |= FLAG_ALL;
    }
    /* a scaled factor, and its product with the other factor */
    for (int k = 0; !rc && !sc->two && k < 2; k++) {
        const dyad *F = k ? &Y : &X, *G = k ? &X : &Y, *g = k ? x : y, *rg = k ? rx : ry;
        if ((rc = dyad_mul(&u, F, &sc->sigma, 64, 1))) break;
        if (dyad_cmp(&u, theta) >= 0) {
            *flags |= sides;
            if (!(rc = dyad_abs(&v, g)) && dyad_cmp(&v, rg) <= 0) *flags |= FLAG_NAN; /* the other factor may be 0 */
            continue;
        }
        if (!(rc = dyad_mul(&u, &u, &one_r, 64, 1)) && !(rc = dyad_add(&u, &u, &sc->a, 64, 1)) && !(rc = dyad_mul(&u, &u, G, 64, 1)) &&
            dyad_cmp(&u, theta) >= 0)
            *flags |= sides;
    }
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_free(all[n]);
    return rc;
}

/* Element (i, j)'s finite terms, each factor anywhere within its radius (ra, rb, NULL where exact; a complex output
   part's terms are its operands' slots q times q, operand_parts): *flags the sides on which one may round past theta on
   its own (Lemma O (a)), and with scale (gemm's alpha on the term's path) those its application adds; and where plus is
   not NULL, the sums of their positive and of their negative parts, counts how many terms have each, *mass that of
   their largest magnitudes and *moved that over the terms with a factor that may be below 2**emin (0 without flush),
   each rounded up to 64 bits. */
int product_terms(int *flags, dyad *plus, dyad *minus, dyad *mass, dyad *moved, size_t counts[2], const operand *a, const operand *ra, size_t i,
                  const operand *b, const operand *rb, size_t j, const dyad *theta, int flush, int64_t emin, const term_scale *scale) {
    dyad x, rx, y, ry, c[2], lo, hi, t, tiny, E[2];
    dyad *all[] = {&x, &rx, &y, &ry, &c[0], &c[1], &lo, &hi, &t, &tiny, &E[0], &E[1]};
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_init(all[n]);
    *flags = 0;
    int rc = plus ? dyad_set_si(plus, 0, 0) : 0;
    if (!rc && plus && !(rc = dyad_set_si(minus, 0, 0)) && !(rc = dyad_set_si(mass, 0, 0))) rc = dyad_set_si(moved, 0, 0);
    if (!rc) rc = dyad_set_si(&tiny, 1, emin);
    if (plus) counts[0] = counts[1] = 0;
    for (size_t q = 0; !rc && q < a->cols; q++) {
        int finite;
        if (slot_absent(b, j * b->cols + q)) continue;
        if ((rc = term_corners(&finite, &x, &rx, &y, &ry, &lo, &hi, a, ra, i, b, rb, j, q)) || !finite) continue;
        if (dyad_cmp(&hi, theta) >= 0) *flags |= FLAG_PLUS_INF;
        if (!(rc = dyad_neg(&t, &lo)) && dyad_cmp(&t, theta) >= 0) *flags |= FLAG_MINUS_INF;
        /* with two channels, each factor's element: its parts' largest magnitude plus radius (slots q - q % stride ..) */
        size_t stride = b->parts > 1 ? (size_t)b->parts : 1;
        for (int side = 0; !rc && scale && scale->two && side < 2; side++) {
            const operand *f = side ? b : a, *rf = side ? rb : ra;
            size_t row = side ? j : i;
            int ok;
            if ((rc = dyad_set_si(&E[side], 0, 0))) break;
            for (size_t u = q - q % stride; !rc && u < q - q % stride + stride; u++) {
                if ((rc = slot_with_radius(&ok, &c[0], &c[1], f, rf, row, u)) || !ok) continue;
                c[0].m.neg = 0;
                if (!(rc = dyad_add_exact(&c[0], &c[0], &c[1])) && dyad_cmp(&c[0], &E[side]) > 0) rc = dyad_copy(&E[side], &c[0]);
            }
        }
        if (!rc && scale) rc = scaled_term_flags(flags, &lo, &hi, &x, &rx, &y, &ry, &E[0], &E[1], scale, theta);
        if (rc || !plus) continue;
        /* the largest magnitude, into mass, and into moved where a factor may be below 2**emin: |x| - rx < 2**emin */
        if (!(rc = dyad_abs(&t, &lo)) && dyad_cmp(&t, &hi) < 0) rc = dyad_copy(&t, &hi);
        if (!rc && !(rc = dyad_abs(&t, &t))) rc = dyad_add(mass, mass, &t, 64, 1);
        if (!rc && flush) {
            int below = 0;
            for (int k = 0; !rc && !below && k < 2; k++) {
                const dyad *v = k ? &y : &x, *r = k ? &ry : &rx;
                if (!(rc = dyad_abs(&c[0], v)) && !(rc = dyad_neg(&c[1], r)) && !(rc = dyad_add_exact(&c[0], &c[0], &c[1]))) below = dyad_cmp(&c[0], &tiny) < 0;
            }
            if (!rc && below) rc = dyad_add(moved, moved, &t, 64, 1);
        }
        if (!rc && dyad_sign(&hi) > 0) {
            counts[0]++;
            rc = dyad_add(plus, plus, &hi, 64, 1);
        }
        if (!rc && dyad_sign(&lo) < 0 && !(rc = dyad_neg(&t, &lo))) {
            counts[1]++;
            rc = dyad_add(minus, minus, &t, 64, 1);
        }
    }
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_free(all[n]);
    return rc;
}

/* Element (i, j)'s terms one by one, for a named order's tree (Lemma O (g)): pos[q] and neg[q] finite term q's largest
   positive and negative values (0 where it has none), and leaf[q] where term q may be infinite: a factor not finite, or
   a finite term that may round past theta on its own. pos and neg hold a->cols dyads, initialized. */
int product_term_parts(dyad *pos, dyad *neg, unsigned char *leaf, const operand *a, const operand *ra, size_t i, const operand *b, const operand *rb,
                       size_t j, const dyad *theta) {
    dyad x, rx, y, ry, lo, hi, t;
    dyad *all[] = {&x, &rx, &y, &ry, &lo, &hi, &t};
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_init(all[n]);
    int rc = 0;
    for (size_t q = 0; !rc && q < a->cols; q++) {
        int finite;
        leaf[q] = 0;
        if (!(rc = dyad_set_si(&pos[q], 0, 0))) rc = dyad_set_si(&neg[q], 0, 0);
        if (rc || slot_absent(b, j * b->cols + q)) continue;
        if ((rc = term_corners(&finite, &x, &rx, &y, &ry, &lo, &hi, a, ra, i, b, rb, j, q))) break;
        if (!finite) {
            leaf[q] = 1;
            continue;
        }
        if (dyad_sign(&hi) > 0 && (rc = dyad_copy(&pos[q], &hi))) break;
        if (dyad_sign(&lo) < 0 && (rc = dyad_neg(&neg[q], &lo))) break;
        leaf[q] = dyad_cmp(&pos[q], theta) >= 0 || dyad_cmp(&neg[q], theta) >= 0;
    }
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_free(all[n]);
    return rc;
}

/* Whether a term of element (i, j) has one factor infinite and the other finite where it may be read as 0: within its
   radius of 0 (|y| <= ry), or with flush below 2**emin (|y| - ry < 2**emin: a subnormal read as 0); an implementation
   that read 0 computes inf * 0 = NaN (proofs/foundations.md, Results as inputs). ra, rb NULL where exact. */
int product_zero_meets(int *meets, const operand *a, const operand *ra, size_t i, const operand *b, const operand *rb, size_t j, int flush,
                       int64_t emin) {
    *meets = 0;
    if (!ra && !rb && !flush) return 0;
    dyad v, r, tiny;
    dyad_init(&v);
    dyad_init(&r);
    dyad_init(&tiny);
    int rc = flush ? dyad_set_si(&tiny, 1, emin) : 0;
    for (size_t q = 0; !rc && !*meets && q < a->cols; q++) {
        if (slot_absent(b, j * b->cols + q)) continue;
        for (int side = 0; !rc && !*meets && side < 2; side++) {
            /* the finite factor x (with its radius rx) against the other factor y */
            const operand *x = side ? b : a, *rx = side ? rb : ra, *y = side ? a : b;
            size_t xrow = side ? j : i, yrow = side ? i : j;
            int kind, neg, finite;
            uint64_t sig;
            int64_t e;
            if (!rx && !flush) continue;
            slot_value(&kind, &neg, &sig, &e, y, yrow * y->cols + q);
            if (kind != VALUE_INFINITE || (rc = slot_with_radius(&finite, &v, &r, x, rx, xrow, q)) || !finite) continue;
            v.m.neg = 0;
            *meets = dyad_cmp(&v, &r) <= 0;
            if (!*meets && flush && !(rc = dyad_neg(&r, &r)) && !(rc = dyad_add_exact(&v, &v, &r))) *meets = dyad_cmp(&v, &tiny) < 0;
        }
    }
    dyad_free(&v);
    dyad_free(&r);
    dyad_free(&tiny);
    return rc;
}

/* The reach of a coefficient product's uncertain inputs (B exact, A's radii ra and marks per slot,
   either NULL) into element (i, j) of a part: an input part reaches the element through each term
   the coefficient rule does not leave out. *unbounded where a finite one whose radius is not finite
   does (a non-finite part's stored values are those its marks give);
   *flags the flags its marks give: through a nonzero coefficient an infinity with the sign of the
   coefficient's part in the term, NaN as NaN, through a coefficient that is 0 (both parts, for
   complex values) any infinity as NaN, ANY as ANY; infinities of both signs give NaN. */
void product_reach(int *unbounded, int *flags, const operand *a, const operand *ra, const unsigned char *marks, size_t i, const operand *b, size_t j) {
    size_t stride = b->parts > 1 ? (size_t)b->parts : 1;
    int f = 0;
    *unbounded = 0;
    for (size_t t = 0; t < a->cols; t++) {
        size_t at = i * a->cols + t, bt = j * b->cols + t;
        if (slot_absent(b, bt)) continue;
        int kind, neg;
        uint64_t sig;
        int64_t e;
        if (ra) {
            slot_value(&kind, &neg, &sig, &e, ra, at);
            if (kind != VALUE_FINITE) {
                slot_value(&kind, &neg, &sig, &e, a, at);
                if (kind == VALUE_FINITE) *unbounded = 1;
            }
        }
        int mark = marks ? marks[at] : 0;
        if (!mark) continue;
        /* the coefficient: its parts are this row's slots t - t % stride .. + stride - 1 */
        int zero = 1;
        for (size_t u = t - t % stride; u < t - t % stride + stride; u++) {
            slot_value(&kind, &neg, &sig, &e, b, j * b->cols + u);
            if (kind != VALUE_FINITE || sig) zero = 0;
        }
        if (zero) {
            f |= (mark & FLAG_ALL ? FLAG_NAN : 0) | (mark & FLAG_ANY);
            continue;
        }
        slot_value(&kind, &neg, &sig, &e, b, bt);
        if (mark & FLAG_PLUS_INF) f |= neg ? FLAG_MINUS_INF : FLAG_PLUS_INF;
        if (mark & FLAG_MINUS_INF) f |= neg ? FLAG_PLUS_INF : FLAG_MINUS_INF;
        f |= mark & (FLAG_NAN | FLAG_ANY);
    }
    *flags = f & FLAG_PLUS_INF && f & FLAG_MINUS_INF ? f | FLAG_NAN : f;
}

/* x rounded to nearest in f with ties by rnd (ROUND_NEAREST_EVEN or ROUND_NEAREST_AWAY), a value: where it overflows, an
   infinity, or NaN in a format without one (the compute format's rule, proofs/foundations.md, Lemma 3). */
static int round_value(value *r, const floatfmt *f, const value *x, int rnd) {
    if (x->kind != VALUE_FINITE) {
        r->kind = x->kind;
        r->neg = x->neg;
        return dyad_set_si(&r->v, 0, 0);
    }
    bigint bits, mag = x->v.m; /* a read-only view of the magnitude */
    mag.neg = 0;
    big_init(&bits);
    int rc = floatfmt_encode(&bits, f, VALUE_FINITE, x->v.m.neg, &mag, x->v.e, rnd, OVERFLOW_NAN);
    if (!rc) rc = floatfmt_decode(&r->kind, &r->neg, &r->v.m, &r->v.e, f, &bits);
    if (!rc) r->v.m.neg = r->neg && r->v.m.n;
    if (!rc && r->kind == VALUE_FINITE && !r->v.m.n) r->neg = 0; /* a zero sum of candidates: its sign does not matter */
    big_free(&bits);
    return rc;
}

/* r = x + y: exact for finite values, IEEE's rule otherwise. */
static int add_values(value *r, const value *x, const value *y) {
    if (x->kind == VALUE_FINITE && y->kind == VALUE_FINITE) {
        r->kind = VALUE_FINITE;
        int rc = dyad_add_exact(&r->v, &x->v, &y->v);
        r->neg = r->v.m.neg;
        return rc;
    }
    if (x->kind == VALUE_NAN || y->kind == VALUE_NAN || (x->kind == VALUE_INFINITE && y->kind == VALUE_INFINITE && x->neg != y->neg)) {
        r->kind = VALUE_NAN;
        r->neg = 0;
    } else {
        const value *inf = x->kind == VALUE_INFINITE ? x : y;
        r->kind = VALUE_INFINITE;
        r->neg = inf->neg;
    }
    return dyad_set_si(&r->v, 0, 0);
}

/* Whether the tie rule can change x's rounding to nearest in f: x halfway between two of f's values (its odd significand
   one bit longer than f's precision, or below 2**emin at half the least unit), or near f's overflow threshold */
static int tie_possible(const floatfmt *f, const value *x) {
    if (x->kind != VALUE_FINITE || !x->v.m.n) return 0;
    int64_t p = f->man_bits + 1, trailing = big_trailing(&x->v.m), top = x->v.e + big_bits(&x->v.m) - 1; /* 2**top <= |x| */
    if (top >= f->emax) return 1;
    if (top >= f->emin) return big_bits(&x->v.m) - trailing == p + 1;
    return x->v.e + trailing == f->emin - p;
}

/* r[0..*n) = x rounded to nearest in f with ties to even, and with ties away where that can differ */
static int round_ties(value r[2], int *n, const floatfmt *f, const value *x) {
    int rc = round_value(&r[0], f, x, ROUND_NEAREST_EVEN);
    *n = 1;
    if (!rc && tie_possible(f, x) && !(rc = round_value(&r[1], f, x, ROUND_NEAREST_AWAY))) *n = 2;
    return rc;
}

/* out[*count] = x unless out holds it already; out holds room for PRODUCT_FEW_MAX values */
static int add_candidate(size_t *count, value *out, const value *x) {
    for (size_t n = 0; n < *count; n++)
        if (out[n].kind == x->kind && out[n].neg == x->neg && (x->kind != VALUE_FINITE || !dyad_cmp(&out[n].v, &x->v))) return 0;
    if (*count >= PRODUCT_FEW_MAX) return EXACTREF_INTERNAL;
    out[*count].kind = x->kind;
    out[*count].neg = x->neg;
    int rc = dyad_copy(&out[*count].v, &x->v);
    if (!rc) ++*count;
    return rc;
}

/* The few-term rule (proofs/foundations.md, Lemma F) for element (i, j) of a product of one-row,
   one-column matrices: where its terms are all finite and at most two are nonzero, the distinct
   values every implementation computes before the final rounding, one that performs only some of
   the model's roundings included, fl rounding to nearest in the compute format f with ties to even
   or away (each rounding either). One term t gives fl(t) and t; two give fl(r1 + r2), fl(t1 + r2),
   fl(r1 + t2), fl(t1 + t2) and the same sums unrounded, rn = fl(tn), under either standard; with
   exact_terms no term is rounded on its own, so two give fl(t1 + t2) and t1 + t2 (add.md). The
   reference, the exact sum's rounding, is among them. *count = 0 where the rule does not apply (the
   general bound then applies); out holds room for PRODUCT_FEW_MAX values. */
int product_few_terms(size_t *count, value *out, const operand *a, size_t i, const operand *b, size_t j, const floatfmt *f, int exact_terms) {
    *count = 0;
    value t[2], r[2][2], u, v[2]; /* r[n][y]: term n rounded with the tie rules that can differ (nr[n] of them) */
    int nr[2] = {0, 0}, nv;
    for (int n = 0; n < 2; n++) {
        dyad_init(&t[n].v);
        dyad_init(&r[n][0].v);
        dyad_init(&r[n][1].v);
        dyad_init(&v[n].v);
    }
    dyad_init(&u.v);
    int rc = 0, terms = 0;
    for (size_t q = 0; !rc && q < a->cols; q++) {
        int ka, kb, na, nb;
        uint64_t sa, sb, hi;
        int64_t ea, eb;
        if (slot_absent(b, j * b->cols + q)) continue;
        slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
        slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
        if (ka != VALUE_FINITE || kb != VALUE_FINITE) {
            terms = 3; /* a term that is not finite: the general rule */
            break;
        }
        if (!sa || !sb) continue;
        if (terms == 2) {
            terms = 3;
            break;
        }
        uint64_t lo = word_mul(&hi, sa, sb);
        if (!(rc = big_reserve(&t[terms].v.m, 2))) {
            t[terms].v.m.w[0] = lo;
            t[terms].v.m.w[1] = hi;
            t[terms].v.m.n = 2;
            t[terms].v.m.neg = 0;
            big_trim(&t[terms].v.m);
            t[terms].v.m.neg = na != nb;
            t[terms].v.e = t[terms].v.m.n ? ea + eb : 0;
            t[terms].kind = VALUE_FINITE;
            t[terms].neg = na != nb;
            terms++;
        }
    }
    if (!rc && terms <= 2) {
        if (!terms) {
            out[0].kind = VALUE_FINITE;
            out[0].neg = 0;
            rc = dyad_set_si(&out[0].v, 0, 0);
            *count = 1;
        } else if (terms == 1) {
            /* fl(t) with each tie rule, and t */
            if (!(rc = round_ties(v, &nv, f, &t[0])))
                for (int x = 0; !rc && x < nv; x++) rc = add_candidate(count, out, &v[x]);
            if (!rc) rc = add_candidate(count, out, &t[0]);
        } else {
            /* each term rounded with each tie rule, or unrounded (index nr[n]; with exact_terms only unrounded); each sum of
               the two rounded with each tie rule, and unrounded */
            for (int n = 0; !rc && !exact_terms && n < 2; n++) rc = round_ties(r[n], &nr[n], f, &t[n]);
            for (int y = 0; !rc && y <= nr[0]; y++)
                for (int z = 0; !rc && z <= nr[1]; z++) {
                    if ((rc = add_values(&u, y < nr[0] ? &r[0][y] : &t[0], z < nr[1] ? &r[1][z] : &t[1])) || (rc = round_ties(v, &nv, f, &u))) break;
                    for (int x = 0; !rc && x < nv; x++) rc = add_candidate(count, out, &v[x]);
                    if (!rc) rc = add_candidate(count, out, &u);
                }
        }
    }
    for (int n = 0; n < 2; n++) {
        dyad_free(&t[n].v);
        dyad_free(&r[n][0].v);
        dyad_free(&r[n][1].v);
        dyad_free(&v[n].v);
    }
    dyad_free(&u.v);
    return rc;
}

/* x = x 2**(from - to) for to <= from, exactly. */
static int realign(bigint *x, int64_t from, int64_t to) { return big_shl(x, x, from - to); }

/* Adds to element (i, j)'s quantities r the deviation of its inputs' radii (proofs/matmul.md,
   Results as inputs): over its terms, |a| rb + ra |b| + ra rb, exactly, from the radii's own
   operands ra and rb (NULL where an operand is exact; read as the operands are). mass and moved
   gain it, N twice it, and r->dev holds it, all at the finer exponent. *unbounded where a radius
   it reads has no bound (inf or NaN). t is scratch. */
int product_deviation(product_sums *r, int *unbounded, product_sums *t, const operand *a, const operand *ra, size_t i, const operand *b,
                      const operand *rb, size_t j, int reached) {
    r->dev.n = 0;
    r->dev.neg = 0;
    /* an infinite radius leaves no bound; with reached, the caller found none that reaches the sum (product_reach) */
    *unbounded = !reached && ((ra && ra->special_rows[i]) || (rb && rb->special_rows[j]));
    if (*unbounded || (!ra && !rb)) return 0;
    const operand *pairs[3][2] = {{a, rb}, {ra, b}, {ra, rb}};
    dyad d, v;
    dyad_init(&d);
    int rc = 0;
    for (int k = 0; !rc && k < 3; k++) {
        if (!pairs[k][0] || !pairs[k][1]) continue;
        if ((rc = product_element(t, pairs[k][0], i, pairs[k][1], j, PRODUCT_MASS, 0, 0, NULL))) break;
        /* a pair with a non-finite term (an element's own infinity times a radius): its finite terms' */
        if (t->kind != VALUE_FINITE && (rc = product_finite_terms(t, pairs[k][0], i, pairs[k][1], j, 0, 0))) break;
        dyad_view(&v, &t->mass, t->e);
        rc = dyad_add_exact(&d, &d, &v);
    }
    if (!rc && d.m.n) {
        /* everything at the finer exponent */
        int64_t f = d.e < r->e ? d.e : r->e;
        bigint *all[] = {&r->s, &r->mass, &r->N, &r->moved};
        for (int k = 0; !rc && k < 4; k++) rc = realign(all[k], r->e, f);
        if (!rc && !(rc = big_copy(&r->dev, &d.m)) && !(rc = realign(&r->dev, d.e, f))) {
            r->e = f;
            if (!(rc = big_add(&r->mass, &r->mass, &r->dev)) && !(rc = big_add(&r->moved, &r->moved, &r->dev)) && !(rc = big_add(&r->N, &r->N, &r->dev)))
                rc = big_add(&r->N, &r->N, &r->dev);
        }
    }
    dyad_free(&d);
    return rc;
}

/* The normal answer of element (i, j) before its final rounding (proofs/foundations.md, Lemma E):
   each term's factors read in the emulated format x (with flush, an input below 2**emin as a zero
   of its sign), each product rounded to it, and the products added in index order, each sum
   rounded; where that overflows while every rounded product is finite, Lemma O's slots. Terms
   the coefficient rule leaves out take no part. terms has room for the element's k terms. */
int product_normal(fvalue *r, fvalue *terms, const femu *x, const operand *a, size_t i, const operand *b, size_t j) {
    size_t k = a->cols, n = 0;
    int all_finite = 1;
    fvalue acc, fa, fb;
    for (size_t q = 0; q < k; q++) {
        if (slot_absent(b, j * b->cols + q)) continue;
        int ka, kb, na, nb;
        uint64_t sa, sb;
        int64_t ea, eb;
        slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
        slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
        femu_read(&fa, x, ka, na, sa, ea);
        femu_read(&fb, x, kb, nb, sb, eb);
        femu_mul(&terms[n], x, &fa, &fb);
        all_finite &= terms[n].kind == VALUE_FINITE;
        if (!n)
            acc = terms[0];
        else
            femu_add(&acc, x, &acc, &terms[n]);
        n++;
    }
    if (!n) {
        r->kind = VALUE_FINITE;
        r->neg = 0;
        r->sig = 0;
        r->e = 0;
        return 0;
    }
    if (acc.kind != VALUE_FINITE && all_finite) {
        femu_slot_sum(r, x, terms, n);
        return 0;
    }
    *r = acc;
    return 0;
}

/* Slot at of x read as a value (its kind, sign and sig 2**e), for the bindings. */
void element_decode_slot(int *kind, int *neg, uint64_t *sig, int64_t *e, const operand *x, size_t at) { slot_value(kind, neg, sig, e, x, at); }

/* The product bindings' kernels: module.c's py_product_reference, py_product_int_bounds and py_product_bounds parse
   their arguments and call these over a tile of output elements. */

/* The order a tile visits its elements in: column by column when it covers whole rows of one
   batch, so that each column of B is read once for all the tile's rows; row by row otherwise.
   Step s of count visits element at(s) of the tile. */
typedef struct {
    size_t rows, n;
    int by_columns;
} tile_order;

/* gemm's one-sided reach for a batch (gemm.md, Overflow): every value of one sign an implementation computes is at most
   g (S + c), S the effective magnitudes of that sign's leaves (|sigma| times a channel's terms, beta c's channel
   products) */
typedef struct {
    dyad g, c;
    int ready;
} gemm_reach;

static void tile_order_set(tile_order *x, const addressing *where, size_t o0, size_t count) {
    size_t mn = where->m * where->n;
    x->n = where->n;
    x->by_columns = count && where->n && o0 % where->n == 0 && count % where->n == 0 && o0 / mn == (o0 + count - 1) / mn && count > where->n;
    x->rows = x->by_columns ? count / where->n : 1;
}

static size_t tile_at(const tile_order *x, size_t step) { return x->by_columns ? step % x->rows * x->n + step / x->rows : step; }

/* The exact zero's sign of element (i, j): the elementwise rule, or the sum's (-0 when rounding
   toward -inf and there are terms). */
static int zero_sign(const operand *a, size_t i, const operand *b, size_t j, int rnd, int elementwise) {
    return elementwise ? product_zero_sign(a, i, b, j, rnd) : a->cols > 0 && rnd == ROUND_TOWARD_NEGATIVE;
}

/* Element o's rows (i, j) of a's and b's matrices (addressing). */
static inline void element_rows(size_t *i, size_t *j, const addressing *x, size_t o) {
    size_t mn = x->m * x->n, t = o / mn, r = o % mn;
    *i = (x->bases[0] ? (size_t)x->bases[0][t] : t % x->counts[0] * x->m) + r / x->n;
    *j = (x->bases[1] ? (size_t)x->bases[1][t] : t % x->counts[1] * x->n) + r % x->n;
}

static inline int epilogue_has_c(const epilogue *x) { return x->ch.n_c[0] || (x->ch.parts == 2 && x->ch.n_c[1]); }

/* Part y of C's element o: its kind and sign, and a finite value in v. */
static inline int epilogue_c(value *v, const epilogue *x, size_t o, int y) {
    v->kind = VALUE_FINITE;
    v->neg = 0;
    v->v.m.n = 0;
    v->v.m.neg = 0;
    v->v.e = 0;
    return x->c && epilogue_has_c(x) ? part_read(v, &x->cf, x->c, o * (size_t)x->ch.parts + (size_t)y, x->c_big) : 0;
}

void epilogue_free(epilogue *x) {
    gemm_fold_free(&x->fold);
    dyad_free(&x->alpha);
    dyad_free(&x->beta);
    if (x->on) gemm_channels_free(&x->ch);
    for (int k = 0; k < 4; k++) big_free(&x->nan[k]);
}

static int epilogue_has_p(const epilogue *x) { return x->ch.n_p[0] || (x->ch.parts == 2 && x->ch.n_p[1]); }

/* Part y of C's element o: its radius (finite, or *unbounded). */
static int epilogue_radius(value *r, int *unbounded, const epilogue *x, size_t o, int y) {
    *unbounded = 0;
    if (!x->c_radius || !epilogue_has_c(x)) return 0;
    int rc = part_read(r, &x->f64, x->c_radius, o * (size_t)x->ch.parts + (size_t)y, 0);
    *unbounded = !rc && r->kind != VALUE_FINITE;
    return rc;
}

/* The flags the marks of C's element o give this call's part through its c channels: an infinity
   with the channel's sign, NaN and ANY as they are, NaN for both infinities. */
static int epilogue_marks(const epilogue *x, size_t o) {
    if (!x->c_marks) return 0;
    int flags = 0;
    for (int k = 0; k < x->ch.n_c[x->part]; k++) {
        const gemm_channel *t = &x->ch.c[x->part][k];
        int mark = x->c_marks[o * (size_t)x->ch.parts + (size_t)t->q], negative = t->s.m.neg != 0;
        flags |= mark & (FLAG_NAN | FLAG_ANY);
        if (mark & FLAG_PLUS_INF) flags |= negative ? FLAG_MINUS_INF : FLAG_PLUS_INF;
        if (mark & FLAG_MINUS_INF) flags |= negative ? FLAG_PLUS_INF : FLAG_MINUS_INF;
    }
    return flags & FLAG_PLUS_INF && flags & FLAG_MINUS_INF ? flags | FLAG_NAN : flags;
}

/* The source parts this call's part reads: of P (need[q]) and of C (need_c[y]). */
void epilogue_needs(int need[2], int need_c[2], const epilogue *x) {
    need[0] = need[1] = need_c[0] = need_c[1] = 0;
    for (int k = 0; k < x->ch.n_p[x->part]; k++) need[x->ch.p[x->part][k].q] = 1;
    for (int k = 0; k < x->ch.n_c[x->part]; k++) need_c[x->ch.c[x->part][k].q] = 1;
}

/* The largest magnitude among rows r0 .. r0 + count - 1 of x, into W. */
static int rows_largest(dyad *W, const operand *x, size_t r0, size_t count) {
    dyad v;
    dyad_init(&v);
    int rc = 0;
    for (size_t r = r0; !rc && r < r0 + count && r < x->rows; r++) {
        if (!x->max_sig[r]) continue;
        if ((rc = big_set_word(&v.m, x->max_sig[r], 0))) break;
        v.e = x->max_exp[r];
        if (dyad_cmp(&v, W) > 0) rc = dyad_copy(W, &v);
    }
    dyad_free(&v);
    return rc;
}

/* d = this call's part of alpha P + beta c exactly, from P's parts (NULL where not read) and C's
   finite parts; t is scratch. */
static int epilogue_value(dyad *d, dyad *t, const epilogue *x, const product_sums *const P[2], const value c[2]) {
    int rc = dyad_set_si(d, 0, 0);
    for (int k = 0; !rc && k < x->ch.n_p[x->part]; k++) {
        const gemm_channel *s = &x->ch.p[x->part][k];
        if (!P[s->q]) continue;
        dyad q;
        dyad_view(&q, &P[s->q]->s, P[s->q]->e);
        if (!(rc = dyad_mul_exact(t, &q, &s->s))) rc = dyad_add_exact(d, d, t);
    }
    for (int k = 0; !rc && k < x->ch.n_c[x->part]; k++) {
        const gemm_channel *s = &x->ch.c[x->part][k];
        if (!(rc = dyad_mul_exact(t, &c[s->q].v, &s->s))) rc = dyad_add_exact(d, d, t);
    }
    return rc;
}

/* The reference outputs of elements o0 .. o0 + count - 1 of a product, with gemm's epilogue alpha P + beta c: each exact
   value encoded in pf's format by rnd and ovf into enc (pf's item size each), and as an entry its flag into fl and its
   midpoint into mid (ENTRY_WIDTH each). EXACTREF_INT_OVERFLOW (an integer output outside its format with ovf "error")
   leaves that element's position in *bad and its exact value in bad_value. */
int product_reference_run(unsigned char *enc, unsigned char *fl, unsigned char *mid, const operand *a, const operand *const bs[2],
                          const addressing *where, const epilogue *epi, const partfmt *pf, int rnd, int ovf, int elementwise, size_t o0, size_t count,
                          size_t *bad, bigint *bad_value) {
    const floatfmt f = pf->f;
    size_t itemsize = part_size(pf);
    int rc = 0;
    product_sums r[2];
    bigint bits, zero;
    dyad d, t;
    value c[2];
    entry x;
    for (int q = 0; q < 2; q++) {
        product_sums_init(&r[q]);
        dyad_init(&c[q].v);
    }
    big_init(&bits);
    big_init(&zero);
    dyad_init(&d);
    dyad_init(&t);
    /* with gemm's epilogue, A and B are read only with alpha's terms, and each part only where its channels read it */
    int need[2] = {1, 0}, need_c[2] = {0, 0};
    if (epi->on) epilogue_needs(need, need_c, epi);
    int has_p = !epi->on || (epilogue_has_p(epi) && a->cols > 0), has_terms = has_p && a->cols > 0, has_c = epi->on && epilogue_has_c(epi);
    tile_order visit;
    tile_order_set(&visit, where, o0, count);
    for (size_t step = 0; !rc && step < count; step++) {
        size_t i, j, at = tile_at(&visit, step), o = o0 + at;
        element_rows(&i, &j, where, o);
        int kind = VALUE_FINITE, neg = 0;
        const product_sums *P[2] = {NULL, NULL};
        for (int q = 0; !rc && has_p && q < 2; q++)
            if (need[q] && !(rc = product_element(&r[q], a, i, bs[q], j, PRODUCT_SUM, 0, 0, NULL))) P[q] = &r[q];
        if (rc) break;
        if (epi->on) {
            int c_kind[2] = {VALUE_FINITE, VALUE_FINITE}, c_neg[2] = {0, 0};
            for (int y = 0; !rc && y < 2; y++)
                if (need_c[y] && !(rc = epilogue_c(&c[y], epi, o, y))) {
                    c_kind[y] = c[y].kind;
                    c_neg[y] = c[y].neg;
                }
            if (rc) break;
            gemm_special(&kind, &neg, &epi->ch, epi->part, P, c_kind, c_neg);
            if (kind == VALUE_FINITE) rc = epilogue_value(&d, &t, epi, P, c);
        } else {
            kind = r[0].kind;
            neg = r[0].neg;
            if (kind == VALUE_FINITE) {
                rc = dyad_from_big(&d, &r[0].s, r[0].e);
            }
        }
        if (rc) break;
        if (pf->is_int) {
            /* an exact integer sum, encoded with the overflow mode: wrapped, saturated, or an error */
            int fits;
            if (d.e >= 0)
                rc = big_shl(&d.m, &d.m, d.e);
            else
                rc = big_shr(&d.m, &d.m, -d.e, 0);
            if (!rc && !(rc = intfmt_encode(&bits, &fits, &d.m, pf->itemsize, pf->is_signed, ovf)) && !fits) {
                *bad = o;
                if (!(rc = big_copy(bad_value, &d.m))) rc = EXACTREF_INT_OVERFLOW;
            }
            if (!rc) {
                neg = d.m.neg;
                d.m.neg = 0;
                rc = entry_value(&x, VALUE_FINITE, neg, &d.m, 0);
            }
        } else if (kind != VALUE_FINITE) {
            rc = floatfmt_encode(&bits, &f, kind, neg, &zero, 0, rnd, ovf);
            if (!rc) rc = entry_value(&x, kind, neg, &zero, 0);
        } else {
            neg = d.m.n ? d.m.neg
                 : epi->on ? (has_terms || has_c) && rnd == ROUND_TOWARD_NEGATIVE
                          : zero_sign(a, i, bs[0], j, rnd, elementwise);
            d.m.neg = 0;
            rc = floatfmt_encode(&bits, &f, VALUE_FINITE, neg, &d.m, d.e, rnd, ovf);
            if (!rc) rc = entry_value(&x, VALUE_FINITE, neg, &d.m, d.e);
        }
        if (rc) break;
        bytes_store(enc + at * itemsize, itemsize, &bits);
        fl[at] = x.flag;
        memcpy(mid + at * ENTRY_WIDTH, x.mid, ENTRY_WIDTH);
    }
    for (int q = 0; q < 2; q++) {
        product_sums_free(&r[q]);
        dyad_free(&c[q].v);
    }
    big_free(&bits);
    big_free(&zero);
    dyad_free(&d);
    dyad_free(&t);
    return rc;
}

void mma_args_free(mma_args *x) {
    dyad *all[] = {&x->G, &x->H, &x->top, &x->thresholds[0], &x->thresholds[1], &x->G2, &x->H2};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
    big_free(&x->nan[0]);
    big_free(&x->nan[1]);
}

/* Lemma T for one element from its exact sum and mass: the interval and the products' stage's flags. */
static int mma_box(dyad *lo, dyad *hi, int *flags, const mma_args *x, const product_sums *r) {
    dyad err;
    bigint pos, neg, ns;
    dyad_init(&err);
    big_init(&pos);
    big_init(&neg);
    big_init(&ns);
    /* the inputs' deviation d (Results as inputs) widens the sum to s -+ d and every total by d */
    bigint lo_s, hi_s, nd, m0;
    big_init(&lo_s);
    big_init(&hi_s);
    big_init(&m0);
    nd = r->dev; /* a read-only view of -d */
    nd.neg = !nd.neg && nd.n;
    int rc = mma_error(&err, &x->G, &x->H, &r->mass, r->e);
    if (!rc && !(rc = big_add(&lo_s, &r->s, &nd)) && !(rc = big_add(&hi_s, &r->s, &r->dev)))
        rc = mma_interval(lo, hi, &lo_s, &hi_s, r->e, &err, &x->top);
    /* the positive terms' total (mass - d + s) / 2 + d and the negative ones' (mass - d - s) / 2 + d */
    if (!rc && !(rc = big_add(&m0, &r->mass, &nd)) && !(rc = big_add(&pos, &m0, &r->s)) && !(rc = big_shr(&pos, &pos, 1, 0)) &&
        !(rc = big_add(&pos, &pos, &r->dev)) && !(rc = big_copy(&ns, &r->s))) {
        ns.neg = !ns.neg && ns.n;
        if (!(rc = big_add(&neg, &m0, &ns)) && !(rc = big_shr(&neg, &neg, 1, 0)) && !(rc = big_add(&neg, &neg, &r->dev)))
        {
            dyad dp, dn; /* views at the sums' exponent */
            dyad_view(&dp, &pos, r->e);
            dyad_view(&dn, &neg, r->e);
            rc = mma_overflow(flags, &dp, &dn, &err, x->thresholds, x->count, x->has_infinity);
        }
    }
    big_free(&lo_s);
    big_free(&hi_s);
    big_free(&m0);
    dyad_free(&err);
    big_free(&pos);
    big_free(&neg);
    big_free(&ns);
    return rc;
}

/* Lemma T's gemm form for one element (proofs/foundations.md, Lemma T, gemm; ops/gemm.py's
   GemmMMA): d = alpha s + beta c within dev = |alpha| d_P + |beta| r_c of every finite output
   before its rounding, widened by G' M' + H' and clamped to float32's range; the epilogue's flags
   in output units, and the products' stage's in its own units, through alpha's sign. */
static int gemm_mma_box(dyad *lo, dyad *hi, int *flags, const mma_args *x, const epilogue *epi, const product_sums *r, int has_p, const value *c,
                        const value *c_radius) {
    dyad vs, vm, dev, vc, cm, M0, d, t, err, pos, neg, perr;
    dyad *all[] = {&vs, &vm, &dev, &vc, &cm, &M0, &d, &t, &err, &pos, &neg, &perr};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    dyad abs_alpha = epi->alpha, abs_beta = epi->beta; /* read-only views of the magnitudes */
    abs_alpha.m.neg = abs_beta.m.neg = 0;
    int rc = 0;
    *flags = 0;
    if (has_p) {
        dyad q;
        dyad_view(&q, &r->dev, r->e);
        if (!(rc = dyad_copy(&d, &q))) {
            dyad_view(&q, &r->mass, r->e);
            if (!(rc = dyad_copy(&M0, &q))) {
                dyad nd = d;
                nd.m.neg = nd.m.n != 0;
                rc = dyad_add_exact(&M0, &M0, &nd); /* the mass without the deviation */
            }
        }
        dyad_view(&q, &r->s, r->e);
        if (!rc && !(rc = dyad_mul_exact(&vs, &q, &epi->alpha)) && !(rc = dyad_mul_exact(&vm, &M0, &abs_alpha))) rc = dyad_mul_exact(&dev, &d, &abs_alpha);
    }
    if (!rc && epi->beta.m.n) {
        if (!(rc = dyad_mul_exact(&vc, &c->v, &epi->beta)) && epi->c_radius && !(rc = dyad_mul_exact(&t, &c_radius->v, &abs_beta))) rc = dyad_add_exact(&dev, &dev, &t);
    }
    /* M' = |alpha| M + |beta c| + dev; B' = G' M' + H' */
    if (!rc) {
        dyad mag = vc;
        mag.m.neg = 0;
        if (!(rc = dyad_add_exact(&cm, &vm, &mag)) && !(rc = dyad_add_exact(&cm, &cm, &dev)) && !(rc = dyad_mul_exact(&err, &x->G2, &cm)))
            rc = dyad_add_exact(&err, &err, &x->H2);
    }
    /* [vs + vc - dev - B', vs + vc + dev + B'], clamped */
    if (!rc && !(rc = dyad_add_exact(&t, &vs, &vc))) {
        dyad nd = dev, ne = err, least = x->top;
        nd.m.neg = nd.m.n != 0;
        ne.m.neg = ne.m.n != 0;
        least.m.neg = least.m.n != 0;
        if (!(rc = dyad_add_exact(lo, &t, &nd)) && !(rc = dyad_add_exact(lo, lo, &ne)) && !(rc = dyad_add_exact(hi, &t, &dev)))
            rc = dyad_add_exact(hi, hi, &err);
        if (!rc && dyad_cmp(lo, &least) < 0) rc = dyad_copy(lo, &least);
        if (!rc && dyad_cmp(lo, &x->top) > 0) rc = dyad_copy(lo, &x->top);
        if (!rc && dyad_cmp(hi, &x->top) > 0) rc = dyad_copy(hi, &x->top);
        if (!rc && dyad_cmp(hi, &least) < 0) rc = dyad_copy(hi, &least);
    }
    /* the epilogue's stage: (vm +- vs) / 2 + max(+-vc, 0) + dev, at float32's threshold */
    if (!rc) {
        dyad nvs = vs;
        nvs.m.neg = !nvs.m.neg && nvs.m.n;
        for (int side = 0; !rc && side < 2; side++) {
            dyad *total = side ? &neg : &pos;
            if ((rc = dyad_add_exact(total, &vm, side ? &nvs : &vs)) || (rc = dyad_shift(total, -1))) break; /* halved exactly */
            int c_side = vc.m.n && (vc.m.neg != 0) == side;
            if (c_side) {
                dyad mag = vc;
                mag.m.neg = 0;
                if ((rc = dyad_add_exact(total, total, &mag))) break;
            }
            rc = dyad_add_exact(total, total, &dev);
        }
        if (!rc) rc = mma_overflow(flags, &pos, &neg, &err, x->thresholds, 1, 1);
    }
    /* the products' stage, before alpha: (M + s) / 2 + d and (M - s) / 2 + d within G (M + d) + H */
    if (!rc && has_p) {
        dyad q, ns;
        dyad_view(&q, &r->s, r->e);
        ns = q;
        ns.m.neg = !ns.m.neg && ns.m.n;
        int stage;
        if (!(rc = dyad_add_exact(&pos, &M0, &q)) && !(rc = dyad_add_exact(&neg, &M0, &ns)) && !(rc = dyad_shift(&pos, -1)) && !(rc = dyad_shift(&neg, -1))) {
            if (!(rc = dyad_add_exact(&pos, &pos, &d)) && !(rc = dyad_add_exact(&neg, &neg, &d)) && !(rc = dyad_add_exact(&t, &M0, &d)) &&
                !(rc = dyad_mul_exact(&perr, &x->G, &t)) && !(rc = dyad_add_exact(&perr, &perr, &x->H)) &&
                !(rc = mma_overflow(&stage, &pos, &neg, &perr, x->thresholds, x->count, x->has_infinity))) {
                if (epi->alpha.m.neg)
                    stage = (stage & FLAG_NAN) | (stage & FLAG_MINUS_INF ? FLAG_PLUS_INF : 0) | (stage & FLAG_PLUS_INF ? FLAG_MINUS_INF : 0);
                *flags |= stage;
            }
        }
    }
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_free(all[i]);
    return rc;
}

static int flip_infinities(int f) { return (f & ~(FLAG_PLUS_INF | FLAG_MINUS_INF)) | (f & FLAG_PLUS_INF ? FLAG_MINUS_INF : 0) | (f & FLAG_MINUS_INF ? FLAG_PLUS_INF : 0); }

/* The largest |x| + r over row i of x (r its radii, NULL where exact) */
static int row_reach(dyad *X, const operand *x, const operand *rx, size_t i) {
    dyad y;
    dyad_init(&y);
    int rc = dyad_set_si(X, 0, 0);
    if (!rc) rc = rows_largest(X, x, i, 1);
    if (!rc && rx && !(rc = dyad_set_si(&y, 0, 0)) && !(rc = rows_largest(&y, rx, i, 1))) rc = dyad_add_exact(X, X, &y);
    dyad_free(&y);
    return rc;
}

/* Whether sb's side of count terms reaches theta: side + B (B_any) and side side_g + any_c both (Lemma O (g)); one term
   only by overflowing on its own (its rounding of a value below theta is finite), which the events count */
static int side_reaches(int *reaches, const dyad *side, size_t count, const dyad *B, const sum_bounds *sb, dyad *t) {
    *reaches = 0;
    if (count < 2) return 0;
    int rc = dyad_add(t, side, B, 64, 1);
    *reaches = !rc && dyad_cmp(t, &sb->theta) >= 0;
    if (!rc && *reaches && !(rc = dyad_mul(t, side, &sb->side_g, 64, 1)) && !(rc = dyad_add(t, t, &sb->any_c, 64, 1))) *reaches = dyad_cmp(t, &sb->theta) >= 0;
    return rc;
}

/* A named order's sides (Lemma O (g)): for each leaf that may be infinite, the subtrees that join its path, ranges of
   terms whose positive (negative) values reach theta together by sb's side_g and any_c; a range with one nonzero term
   of that sign only by its overflowing on its own, which the leaves count. More than ORDER_LEAVES such leaves take the
   whole row's sides. */
#define ORDER_LEAVES 64
static int order_sides(int *sides, int kind, size_t block, size_t k, const dyad *pos, const dyad *neg, const unsigned char *leaf, const sum_bounds *sb) {
    size_t count = 0;
    for (size_t q = 0; q < k; q++) count += leaf[q] != 0;
    dyad *pre = malloc((2 * k + 2) * sizeof *pre), t, u;
    size_t *cnt = malloc((2 * k + 2) * sizeof *cnt);
    if (!pre || !cnt) {
        free(pre);
        free(cnt);
        return EXACTREF_NOMEM;
    }
    cnt[0] = cnt[k + 1] = 0;
    for (size_t q = 0; q < k; q++) {
        cnt[q + 1] = cnt[q] + (dyad_sign(&pos[q]) > 0);
        cnt[k + 2 + q] = cnt[k + 1 + q] + (dyad_sign(&neg[q]) > 0);
    }
    dyad_init(&t);
    dyad_init(&u);
    for (size_t q = 0; q < 2 * k + 2; q++) dyad_init(&pre[q]);
    int rc = dyad_set_si(&pre[0], 0, 0);
    if (!rc) rc = dyad_set_si(&pre[k + 1], 0, 0);
    for (size_t q = 0; !rc && q < k; q++)
        if (!(rc = dyad_add_exact(&pre[q + 1], &pre[q], &pos[q]))) rc = dyad_add_exact(&pre[k + 2 + q], &pre[k + 1 + q], &neg[q]);
    /* range [lo, hi)'s sides into *sides */
#define RANGE_SIDES(lo, hi)                                                                                                            \
    for (int z = 0; !rc && z < 2; z++) {                                                                            \
        const dyad *P0 = &pre[z * (k + 1) + (lo)], *P1 = &pre[z * (k + 1) + (hi)];                                                     \
        if (cnt[z * (k + 1) + (hi)] - cnt[z * (k + 1) + (lo)] < 2) continue;                                                     \
        dyad n0 = *P0; /* a read-only view of -P0 */                                                                                   \
        n0.m.neg = !n0.m.neg && n0.m.n;                                                                                                \
        if (!(rc = dyad_add_exact(&t, P1, &n0)) && !(rc = dyad_mul(&u, &t, &sb->side_g, 64, 1)) && !(rc = dyad_add(&u, &u, &sb->any_c, 64, 1)) && \
            dyad_cmp(&u, &sb->theta) >= 0)                                                                                             \
            *sides |= z ? FLAG_MINUS_INF : FLAG_PLUS_INF;                                                                              \
    }
    if (count > ORDER_LEAVES) {
        RANGE_SIDES((size_t)0, k)
    }
    for (size_t m = 0; !rc && count <= ORDER_LEAVES && m < k; m++) {
        if (!leaf[m]) continue;
        if (kind == 1) { /* sequential: the prefix before m; later terms join one by one */
            RANGE_SIDES((size_t)0, m)
        } else if (kind == 2) { /* pairwise: at each split the half without m */
            size_t lo = 0, hi = k;
            while (!rc && hi - lo > 1) {
                size_t mid = lo + (hi - lo + 1) / 2;
                if (m < mid) {
                    RANGE_SIDES(mid, hi)
                    hi = mid;
                } else {
                    RANGE_SIDES(lo, mid)
                    lo = mid;
                }
            }
        } else { /* blocked: within m's block the prefix before m, then the blocks before it as one chain, each later block */
            size_t s = m - m % block, e = s + block < k ? s + block : k;
            RANGE_SIDES(s, m)
            RANGE_SIDES((size_t)0, s)
            for (size_t s2 = e; !rc && s2 < k; s2 += block) {
                size_t e2 = s2 + block < k ? s2 + block : k;
                RANGE_SIDES(s2, e2)
            }
        }
    }
#undef RANGE_SIDES
    for (size_t q = 0; q < 2 * k + 2; q++) dyad_free(&pre[q]);
    free(pre);
    free(cnt);
    dyad_free(&t);
    dyad_free(&u);
    return rc;
}

/* Element (i, j)'s finite terms' mass with the radii's deviation (product_finite_terms, product_deviation): at least each
   finite term and each side of them; *unbounded where an infinite radius leaves no bound. finite_terms_below's and the
   tensor-core paths' screens (mma_event_sides). */
static int finite_mass(dyad *mass, int *unbounded, const operand *a, const operand *ra, size_t i, const operand *b, const operand *rb, size_t j,
                       int flush, int64_t emin) {
    product_sums fin, tmp;
    product_sums_init(&fin);
    product_sums_init(&tmp);
    *unbounded = 0;
    int rc = product_finite_terms(&fin, a, i, b, j, flush, emin);
    if (!rc && (ra || rb)) rc = product_deviation(&fin, unbounded, &tmp, a, ra, i, b, rb, j, 0);
    if (!rc && !*unbounded) rc = dyad_from_big(mass, &fin.mass, fin.e); /* the mass counts the deviation */
    product_sums_free(&fin);
    product_sums_free(&tmp);
    return rc;
}

/* Element (i, j) with an infinity, without gemm's epilogue: *below where its finite terms' mass (finite_mass), through
   Lemma O (g)'s reach, stays below theta, so that no finite term and no side of them reaches it (each is at most that
   mass); an infinite radius leaves *below 0. */
static int finite_terms_below(int *below, const operand *a, const operand *ra, size_t i, const operand *b, const operand *rb, size_t j,
                              const sum_bounds *sb) {
    dyad t, v;
    dyad_init(&t);
    dyad_init(&v);
    int unbounded = 0, rc = finite_mass(&v, &unbounded, a, ra, i, b, rb, j, sb->flush, sb->emin);
    *below = 0;
    if (!rc && !unbounded && !(rc = dyad_mul(&t, &v, &sb->side_g, 64, 1)) && !(rc = dyad_add(&t, &t, &sb->any_c, 64, 1)))
        *below = dyad_cmp(&t, &sb->theta) < 0;
    dyad_free(&t);
    dyad_free(&v);
    return rc;
}

#define MAY_TERM 1 /* terms_may: a term may round past theta on its own */
#define MAY_SIDE 2 /* terms_may: the terms of one sign may reach theta together */

/* Whether a term of a product q with factors at most X and Y (|x| + r) may round past theta on its own (*may |= MAY_TERM:
   X Y; under gemm's alpha |sigma| X, |sigma| Y and |sigma| (1 + r)**2 X Y + (|sigma| + 1) a (1 + X + Y), which bounds each
   scaled form, and with two channels in a part the fold's F(X), F(Y), F(X) Y and F(Y) X; gemm.md, Overflow), and whether
   k of them may reach theta together (*may |= MAY_SIDE: k X Y side_g + any_c, Lemma O (g)); epi's part's channels of q, or
   every channel with all_parts */
static int terms_may(int *may, const dyad *X, const dyad *Y, int64_t k, const epilogue *epi, int q, int all_parts, const sum_bounds *sb) {
    dyad T, t, u, v;
    dyad *all[] = {&T, &t, &u, &v};
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_init(all[n]);
    int rc = dyad_mul(&T, X, Y, 64, 1), term = 0;
    *may = 0;
    if (!rc) term = dyad_cmp(&T, &sb->theta) >= 0;
    if (!rc && !(rc = dyad_set_si(&t, k, 0)) && !(rc = dyad_mul(&t, &t, &T, 64, 1)) && !(rc = dyad_mul(&u, &t, &sb->side_g, 64, 1)) &&
        !(rc = dyad_add(&u, &u, &sb->any_c, 64, 1)) && dyad_cmp(&u, &sb->theta) >= 0)
        *may |= MAY_SIDE;
    for (int part = 0; !rc && !term && epi->on && part < (all_parts ? epi->ch.parts : 1); part++) {
        int pp = all_parts ? part : epi->part;
        /* a two-channel fold into either operand */
        for (int f = 0; !rc && !term && epi->ch.n_p[pp] == 2 && f < 2; f++)
            if (!(rc = gemm_fold_bound(&u, &epi->fold, pp, f ? Y : X, 64)) && !(term = dyad_cmp(&u, &sb->theta) >= 0) &&
                !(rc = dyad_mul(&u, &u, f ? X : Y, 64, 1)))
                term = dyad_cmp(&u, &sb->theta) >= 0;
        for (int n = 0; !rc && !term && n < epi->ch.n_p[pp]; n++) {
            const gemm_channel *x = &epi->ch.p[pp][n];
            dyad sg = x->s; /* a read-only view of |sigma| */
            sg.m.neg = 0;
            if (!all_parts && x->q != q) continue;
            for (int f = 0; !rc && !term && f < 2; f++)
                if (!(rc = dyad_mul(&u, f ? Y : X, &sg, 64, 1))) term = dyad_cmp(&u, &sb->theta) >= 0;
            /* |sigma| (1 + r)**2 T + (|sigma| + 1) a (1 + X + Y) */
            if (!rc && !term && !(rc = dyad_set_si(&t, 1, 0)) && !(rc = dyad_add(&t, &t, &sb->unit_r, 64, 1)) && !(rc = dyad_mul(&t, &t, &t, 64, 1)) &&
                !(rc = dyad_mul(&u, &T, &t, 64, 1)) && !(rc = dyad_mul(&u, &u, &sg, 64, 1)) && !(rc = dyad_add(&t, X, Y, 64, 1)) &&
                !(rc = dyad_set_si(&v, 1, 0)) && !(rc = dyad_add(&t, &t, &v, 64, 1)) && !(rc = dyad_add(&v, &v, &sg, 64, 1)) &&
                !(rc = dyad_mul(&t, &t, &v, 64, 1)) && !(rc = dyad_mul(&t, &t, &sb->unit_a, 64, 1)) && !(rc = dyad_add(&u, &u, &t, 64, 1)))
                term = dyad_cmp(&u, &sb->theta) >= 0;
        }
    }
    if (term) *may |= MAY_TERM;
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_free(all[n]);
    return rc;
}

/* The tensor-core paths' sides where an event occurs (proofs/foundations.md, Results as inputs; Lemma T): element (i, j)'s
   finite terms bounded as mma_box and gemm_mma_box bound a finite element's, from the terms' walk (their positive and
   negative totals and their mass, product_terms), so *sides holds the sides on which a step's sum or a reduced piece may
   reach a threshold (through alpha's sign), and with the epilogue those on which alpha's scaled value with beta c may.
   First a quick test of the row's and the column's largest factors, k X Y within G k X Y + H. */
static int mma_event_sides(int *sides, const mma_args *x, const epilogue *epi, const value *c, const value *c_radius, const operand *a,
                           const operand *ra, size_t i, const operand *b, const operand *rb, size_t j, const sum_bounds *sb) {
    dyad X, Y, T, R, t, vc, cr, least, abs_alpha, plus, minus, mass, moved, err, pos, neg;
    dyad *all[] = {&X, &Y, &T, &R, &t, &vc, &cr, &least, &abs_alpha, &plus, &minus, &mass, &moved, &err, &pos, &neg};
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_init(all[n]);
    size_t counts[2];
    int has_p = b && a->cols > 0 && (!epi->on || epilogue_has_p(epi)), may = 0, rc = dyad_copy(&least, &x->thresholds[0]);
    *sides = 0;
    for (int k = 1; !rc && k < x->count; k++)
        if (dyad_cmp(&x->thresholds[k], &least) < 0) rc = dyad_copy(&least, &x->thresholds[k]);
    if (!rc) rc = dyad_abs(&abs_alpha, &epi->alpha);
    /* beta c and its radius, where c is finite (an infinite c is the event itself) */
    if (!rc && !(rc = dyad_set_si(&vc, 0, 0)) && !(rc = dyad_set_si(&cr, 0, 0)) && epi->on && epi->beta.m.n && c && c->kind == VALUE_FINITE) {
        if (!(rc = dyad_mul_exact(&vc, &c->v, &epi->beta)) && epi->c_radius && c_radius && c_radius->kind == VALUE_FINITE &&
            !(rc = dyad_mul_exact(&cr, &c_radius->v, &epi->beta)))
            cr.m.neg = 0;
    }
    /* the quick test: R = k X Y (1 + G) + H at the least threshold, and with the epilogue |alpha| R + |beta c| + its radius
       within G' (|alpha| k X Y + |beta c| + its radius) + H' at float32's */
    if (!rc && !(rc = dyad_set_si(&T, 0, 0)) && has_p && !(rc = row_reach(&X, a, ra, i)) && !(rc = row_reach(&Y, b, rb, j)) &&
        !(rc = dyad_mul(&T, &X, &Y, 64, 1)) && !(rc = dyad_set_si(&t, (int64_t)a->cols, 0)))
        rc = dyad_mul(&T, &T, &t, 64, 1);
    if (!rc && !(rc = dyad_mul(&R, &T, &x->G, 64, 1)) && !(rc = dyad_add(&R, &R, &T, 64, 1)) && !(rc = dyad_add(&R, &R, &x->H, 64, 1)))
        may = has_p && dyad_cmp(&R, &least) >= 0;
    if (!rc && !may && epi->on) {
        dyad m = vc; /* a read-only view of |beta c| */
        m.m.neg = 0;
        if (!(rc = dyad_mul(&t, &T, &abs_alpha, 64, 1)) && !(rc = dyad_add(&t, &t, &m, 64, 1)) && !(rc = dyad_add(&t, &t, &cr, 64, 1)) &&
            !(rc = dyad_mul(&t, &t, &x->G2, 64, 1)) && !(rc = dyad_add(&t, &t, &x->H2, 64, 1)) && !(rc = dyad_mul(&R, &R, &abs_alpha, 64, 1)) &&
            !(rc = dyad_add(&t, &t, &R, 64, 1)) && !(rc = dyad_add(&t, &t, &m, 64, 1)) && !(rc = dyad_add(&t, &t, &cr, 64, 1)))
            may = dyad_cmp(&t, &x->thresholds[0]) >= 0;
    }
    /* the same test with the element's finite terms' mass M in place of k X Y (finite_mass): where both stages stay below,
       no side is reached and the terms' walk is skipped */
    if (!rc && may && has_p) {
        int unbounded = 0;
        if (!(rc = finite_mass(&mass, &unbounded, a, ra, i, b, rb, j, sb->flush, sb->emin)) && !unbounded &&
            !(rc = dyad_mul(&R, &mass, &x->G, 64, 1)) && !(rc = dyad_add(&R, &R, &mass, 64, 1)) && !(rc = dyad_add(&R, &R, &x->H, 64, 1))) {
            may = dyad_cmp(&R, &least) >= 0;
            if (!may && epi->on) {
                dyad m = vc; /* a read-only view of |beta c| */
                m.m.neg = 0;
                if (!(rc = dyad_mul(&t, &mass, &abs_alpha, 64, 1)) && !(rc = dyad_add(&t, &t, &m, 64, 1)) && !(rc = dyad_add(&t, &t, &cr, 64, 1)) &&
                    !(rc = dyad_mul(&t, &t, &x->G2, 64, 1)) && !(rc = dyad_add(&t, &t, &x->H2, 64, 1)) && !(rc = dyad_mul(&R, &R, &abs_alpha, 64, 1)) &&
                    !(rc = dyad_add(&t, &t, &R, 64, 1)) && !(rc = dyad_add(&t, &t, &m, 64, 1)) && !(rc = dyad_add(&t, &t, &cr, 64, 1)))
                    may = dyad_cmp(&t, &x->thresholds[0]) >= 0;
            }
        }
    }
    if (rc || !may) goto done;
    /* the terms' totals and mass (0 without products) */
    if (!(rc = dyad_set_si(&plus, 0, 0)) && !(rc = dyad_set_si(&minus, 0, 0)) && !(rc = dyad_set_si(&mass, 0, 0)) && has_p) {
        int ignored;
        rc = product_terms(&ignored, &plus, &minus, &mass, &moved, counts, a, ra, i, b, rb, j, &sb->theta, sb->flush, sb->emin, NULL);
    }
    /* the products' stage within G mass + H, every threshold mma_box uses, through alpha's sign */
    if (!rc && has_p && !(rc = mma_error(&err, &x->G, &x->H, &mass.m, mass.e))) {
        int stage;
        if (!(rc = mma_overflow(&stage, &plus, &minus, &err, x->thresholds, x->count, x->has_infinity))) {
            if (epi->on && epi->alpha.m.neg) stage = flip_infinities(stage);
            *sides |= stage;
        }
    }
    /* the epilogue's stage at float32's threshold: |alpha| times the products of each sign through alpha's, beta c on its
       side and its radius on both, within G' (|alpha| mass + |beta c| + its radius) + H' */
    if (!rc && epi->on) {
        dyad m = vc;
        m.m.neg = 0;
        int flip = epi->alpha.m.neg != 0, out;
        for (int side = 0; !rc && side < 2; side++) {
            dyad *total = side ? &neg : &pos;
            if ((rc = dyad_mul(total, (side != flip) ? &minus : &plus, &abs_alpha, 64, 1)) || (rc = dyad_add(total, total, &cr, 64, 1))) break;
            if (vc.m.n && (vc.m.neg != 0) == side) rc = dyad_add(total, total, &m, 64, 1);
        }
        if (!rc && !(rc = dyad_mul(&t, &mass, &abs_alpha, 64, 1)) && !(rc = dyad_add(&t, &t, &m, 64, 1)) && !(rc = dyad_add(&t, &t, &cr, 64, 1)) &&
            !(rc = dyad_mul(&err, &t, &x->G2, 64, 1)) && !(rc = dyad_add(&err, &err, &x->H2, 64, 1)) &&
            !(rc = mma_overflow(&out, &pos, &neg, &err, x->thresholds, 1, 1)))
            *sides |= out;
    }
done:
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_free(all[n]);
    return rc;
}

/* The infinities an implementation's sum can meet, under either standard (proofs/foundations.md, Results as inputs and
   Lemma O (g); gemm.md, Overflow). Its events: an infinity a coefficient passes on (passed, with C's marks through
   beta), the exact value's own (an infinite input, or C's), and with terms_on a term rounded past theta on its own (for
   gemm at each place alpha's application may take, and beta c's channel products), each also a flag. NaN where events of
   both signs occur, or an event and the finite terms of the other sign reaching theta together (side_reaches; a named
   order's own subtrees, order_sides; gemm's reach gr), or an infinite factor (a scaled one too) may meet a factor that
   may be 0 (on every path). In a compute format without infinities any event is NaN. batch_may: terms_may's answers for
   the batch's largest factors; where a term may not round past theta on its own (MAY_TERM), and per element where the
   row's and column's do not either, the terms are walked only for an event's other side (MAY_SIDE). mma: the tensor-core
   paths' arguments (NULL elsewhere), whose other side is Lemma T's (mma_event_sides), and where a reduction format
   without infinities may take an event, NaN. */
static int infinities_meet(int *possible, int passed, const product_sums *const P[2], const epilogue *epi, const value *const c[2],
                           const value *const c_radius[2], const operand *a, const operand *ra, size_t i, const operand *const bs[2],
                           const operand *const rbs[2], size_t j, const sum_bounds *sb, int terms_on, int batch_may, int order_kind, size_t block,
                           const gemm_reach *gr, const mma_args *mma) {
    int events = passed & (FLAG_PLUS_INF | FLAG_MINUS_INF), terms = 0, sides = 0, rc = 0, walked = 0;
    int exact_inf = 0; /* the exact value's own infinity: every implementation's value is it or NaN */
    size_t counts[2] = {0, 0}, leaves[2] = {0, 0}; /* the terms of each sign, and gemm's leaves of each effective sign */
    if (epi->on) {
        int kind, neg, c_kind[2], c_neg[2];
        for (int q = 0; q < 2; q++) {
            c_kind[q] = c[q] ? c[q]->kind : VALUE_FINITE;
            c_neg[q] = c[q] ? c[q]->neg : 0;
        }
        gemm_special(&kind, &neg, &epi->ch, epi->part, P, c_kind, c_neg);
        if (kind == VALUE_INFINITE) exact_inf = neg ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    } else if (P[0] && P[0]->kind == VALUE_INFINITE) {
        exact_inf = P[0]->neg ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    }
    events |= exact_inf;
    dyad X, Y, t, u, B, plus, minus, mass, moved, S[2];
    dyad *all[] = {&X, &Y, &t, &u, &B, &plus, &minus, &mass, &moved, &S[0], &S[1]};
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_init(all[n]);
    int64_t top_theta = sb->theta.e + (int64_t)big_bits(&sb->theta.m);
    /* an infinite factor times a value that may be read as 0 (its radius reaches 0, or with flush a subnormal): inf * 0 =
       NaN in the implementation that read 0 (foundations.md, Results as inputs) */
    if (exact_inf) {
        int meets = 0;
        if (!epi->on) {
            if (bs[0]) rc = product_zero_meets(&meets, a, ra, i, bs[0], rbs[0], j, sb->flush, sb->emin);
        } else {
            for (int k = 0; !rc && !meets && k < epi->ch.n_p[epi->part]; k++) {
                const gemm_channel *x = &epi->ch.p[epi->part][k];
                if (P[x->q] && bs[x->q]) rc = product_zero_meets(&meets, a, ra, i, bs[x->q], rbs[x->q], j, sb->flush, sb->emin);
            }
        }
        if (rc) goto done;
        if (meets) *possible |= FLAG_NAN;
    }
    /* beta c's channel products rounded past theta on their own: |tau| (|c| + its radius) on c tau's side (both where c's
       radius spans 0); a quick test of the exponents first */
    for (int k = 0; !rc && terms_on && epi->on && k < epi->ch.n_c[epi->part]; k++) {
        const gemm_channel *x = &epi->ch.c[epi->part][k];
        const value *cv = c[x->q], *rv = c_radius[x->q];
        if (!cv || cv->kind != VALUE_FINITE || !cv->v.m.n) continue;
        int radius = rv && rv->kind == VALUE_FINITE && rv->v.m.n, spans = 0, neg = cv->v.m.neg != x->s.m.neg;
        int64_t top = cv->v.e + (int64_t)big_bits(&cv->v.m), top_r = radius ? rv->v.e + (int64_t)big_bits(&rv->v.m) : top;
        if ((top_r > top ? top_r : top) + 1 + x->s.e + (int64_t)big_bits(&x->s.m) < top_theta - 1) continue;
        if ((rc = dyad_abs(&u, &cv->v))) break;
        if (radius) {
            spans = dyad_cmp(&rv->v, &u) >= 0;
            if ((rc = dyad_add_exact(&u, &u, &rv->v))) break;
        }
        if ((rc = dyad_abs(&t, &x->s)) || (rc = dyad_mul_exact(&u, &u, &t))) break;
        if (dyad_cmp(&u, &sb->theta) >= 0) terms |= spans ? FLAG_PLUS_INF | FLAG_MINUS_INF : neg ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    }
    /* the terms' walk where the row's and the column's largest factors may take a term past theta, or the terms of one sign
       there with an event */
    int may = 0, event = events || terms;
    if (!rc && terms_on && (batch_may & MAY_TERM || (event && batch_may & MAY_SIDE)) && (!epi->on || epilogue_has_p(epi))) {
        for (int q = 0; !rc && q < (epi->on ? 2 : 1); q++) {
            int m = 0, below;
            if (P[q] && bs[q] && !(rc = row_reach(&X, a, ra, i)) && !(rc = row_reach(&Y, bs[q], rbs[q], j)))
                rc = terms_may(&m, &X, &Y, (int64_t)a->cols, epi, q, 0, sb);
            /* where the rows' and columns' largest factors leave the walk, an element with an infinity whose finite terms
               cannot reach theta has none to make (finite_terms_below) */
            if (!rc && (m & batch_may) && !epi->on && P[q]->kind != VALUE_FINITE && !(rc = finite_terms_below(&below, a, ra, i, bs[q], rbs[q], j, sb)) &&
                below)
                m = 0;
            may |= m & batch_may;
        }
    }
    if (!rc && (may & MAY_TERM || (event && may & MAY_SIDE))) {
        walked = 1;
        if (!epi->on) {
            if (!(rc = product_terms(&terms, &plus, &minus, &mass, &moved, counts, a, ra, i, bs[0], rbs[0], j, &sb->theta, sb->flush, sb->emin, NULL)) &&
                !(rc = dyad_mul(&B, &mass, &sb->any_r, 64, 1)) && !(rc = dyad_mul(&t, &moved, &sb->any_g, 64, 1)) && !(rc = dyad_add(&B, &B, &t, 64, 1)))
                rc = dyad_add(&B, &B, &sb->any_c, 64, 1);
        } else {
            /* per channel: its scaled terms' events, its unscaled tree's sides (alpha applied above them), and its terms into
               the effective sides S */
            for (int k = 0; !rc && k < epi->ch.n_p[epi->part]; k++) {
                const gemm_channel *x = &epi->ch.p[epi->part][k];
                term_scale sc;
                int f, reaches, flip = x->s.m.neg != 0;
                if (!P[x->q]) continue;
                sc.sigma = x->s; /* read-only views */
                sc.sigma.m.neg = 0;
                sc.r = sb->unit_r;
                sc.a = sb->unit_a;
                sc.two = epi->ch.n_p[epi->part] == 2;
                sc.fold_mul = epi->fold.mul[epi->part];
                sc.fold_add = epi->fold.add[epi->part];
                if ((rc = product_terms(&f, &plus, &minus, &mass, &moved, counts, a, ra, i, bs[x->q], rbs[x->q], j, &sb->theta, sb->flush, sb->emin, &sc)))
                    break;
                terms |= flip ? flip_infinities(f) : f;
                if ((rc = dyad_mul(&B, &mass, &sb->any_r, 64, 1)) || (rc = dyad_mul(&t, &moved, &sb->any_g, 64, 1)) || (rc = dyad_add(&B, &B, &t, 64, 1)) ||
                    (rc = dyad_add(&B, &B, &sb->any_c, 64, 1)))
                    break;
                for (int z = 0; !rc && z < 2; z++) {
                    if ((rc = side_reaches(&reaches, z ? &minus : &plus, counts[z], &B, sb, &t))) break;
                    if (reaches) sides |= (z != flip) ? FLAG_MINUS_INF : FLAG_PLUS_INF;
                    leaves[z != flip] += counts[z];
                    if (!(rc = dyad_mul(&t, z ? &minus : &plus, &sc.sigma, 64, 1))) rc = dyad_add(&S[z != flip], &S[z != flip], &t, 64, 1);
                }
            }
        }
    }
    if (rc) goto done;
    events |= terms & (FLAG_PLUS_INF | FLAG_MINUS_INF);
    if (!sb->has_infinity) { /* every overflow is NaN */
        if (events || terms) *possible |= FLAG_NAN;
        goto done;
    }
    *possible |= exact_inf ? terms & FLAG_NAN : terms;
    if (!events) goto done;
    int pl = events & FLAG_PLUS_INF, mi = events & FLAG_MINUS_INF;
    if (pl && mi) {
        *possible |= FLAG_NAN;
        goto done;
    }
    /* one sign: the other side's reach */
    if (mma) {
        if (!mma->has_infinity) *possible |= FLAG_NAN;
        else rc = mma_event_sides(&sides, mma, epi, c[0], c_radius[0], a, ra, i, bs[0], rbs[0], j, sb);
    } else if (terms_on && !epi->on && walked) {
        if (order_kind >= 1 && order_kind <= 3 && !passed) {
            /* a named order's own subtrees */
            size_t k = a->cols;
            dyad *parts = malloc((2 * k + 1) * sizeof *parts);
            unsigned char *leaf = malloc(k + 1);
            rc = parts && leaf ? 0 : EXACTREF_NOMEM;
            for (size_t q = 0; parts && q < 2 * k + 1; q++) dyad_init(&parts[q]);
            if (!rc && !(rc = product_term_parts(parts, parts + k, leaf, a, ra, i, bs[0], rbs[0], j, &sb->theta)))
                rc = order_sides(&sides, order_kind, block > 0 ? block : 1, k, parts, parts + k, leaf, sb);
            for (size_t q = 0; parts && q < 2 * k + 1; q++) dyad_free(&parts[q]);
            free(parts);
            free(leaf);
        } else {
            int reaches;
            if (!(rc = side_reaches(&reaches, &plus, counts[0], &B, sb, &t)) && reaches) sides |= FLAG_PLUS_INF;
            if (!rc && !(rc = side_reaches(&reaches, &minus, counts[1], &B, sb, &t)) && reaches) sides |= FLAG_MINUS_INF;
        }
    } else if (terms_on && epi->on && gr && gr->ready) {
        /* gemm: the channels' sides (each at most k X Y |sigma| where not walked), beta c's values, then every node's
           reach g (S + c) */
        for (int k = 0; !rc && !walked && epilogue_has_p(epi) && k < epi->ch.n_p[epi->part]; k++) {
            const gemm_channel *x = &epi->ch.p[epi->part][k];
            dyad sg = x->s;
            sg.m.neg = 0;
            if (!P[x->q] || !bs[x->q]) continue;
            if ((rc = row_reach(&X, a, ra, i)) || (rc = row_reach(&Y, bs[x->q], rbs[x->q], j)) || (rc = dyad_mul(&u, &X, &Y, 64, 1)) ||
                (rc = dyad_set_si(&t, (int64_t)a->cols, 0)) || (rc = dyad_mul(&t, &t, &u, 64, 1)) || (rc = dyad_mul(&t, &t, &sg, 64, 1)))
                break;
            for (int z = 0; !rc && z < 2; z++) {
                leaves[z] += a->cols;
                rc = dyad_add(&S[z], &S[z], &t, 64, 1);
            }
        }
        for (int k = 0; !rc && k < epi->ch.n_c[epi->part]; k++) {
            const gemm_channel *x = &epi->ch.c[epi->part][k];
            const value *cv = c[x->q], *rv = c_radius[x->q];
            if (!cv || cv->kind != VALUE_FINITE) continue;
            int radius = rv && rv->kind == VALUE_FINITE && rv->v.m.n, spans = 0, neg = cv->v.m.neg != x->s.m.neg;
            if ((rc = dyad_abs(&u, &cv->v))) break;
            if (radius) {
                spans = dyad_cmp(&rv->v, &u) >= 0;
                if ((rc = dyad_add_exact(&u, &u, &rv->v))) break;
            }
            if ((rc = dyad_abs(&t, &x->s)) || (rc = dyad_mul_exact(&u, &u, &t))) break;
            for (int z = 0; !rc && z < 2; z++)
                if (spans || neg == z) {
                    leaves[z]++;
                    rc = dyad_add(&S[z], &S[z], &u, 64, 1);
                }
        }
        /* a side of one leaf reaches theta only by that leaf's own overflow, an event */
        for (int z = 0; !rc && z < 2; z++)
            if (leaves[z] >= 2 && !(rc = dyad_add(&t, &S[z], &gr->c, 64, 1)) && !(rc = dyad_mul(&t, &t, &gr->g, 64, 1)) && dyad_cmp(&t, &sb->theta) >= 0)
                sides |= z ? FLAG_MINUS_INF : FLAG_PLUS_INF;
    }
    if (!rc && ((pl && sides & FLAG_MINUS_INF) || (mi && sides & FLAG_PLUS_INF) || sides & FLAG_NAN)) *possible |= FLAG_NAN;
done:
    for (size_t n = 0; n < sizeof all / sizeof *all; n++) dyad_free(all[n]);
    return rc;
}

/* One element's interval and flags under matmul's bound (sums.c): normal or any order, or Lemma S
   over a named order's tree; or gemm's (gemm.c) with the epilogue. *bounded = 0 where its exact
   value is not finite. */
static int element_box(dyad *lo, dyad *hi, int *bounded, int *possible, const product_sums *const P[2], const sum_bounds *sb, const gemm_bounds *g,
                       const gemm_bounds *g_any, const epilogue *epi, const value *const c[2], const value *const c_radius[2], const operand *a,
                       int normal, int normal_flags, int tree, const mma_args *mma, int exact,
                       const floatfmt *compute, int64_t w) {
    int checked, sides = 0, rc = 0;
    const product_sums *r = P[0];
    *bounded = 0;
    if (epi->on && mma && mma->on) {
        int kind, neg, has_p = epilogue_has_p(epi) && a->cols > 0, c_kind[2] = {c[0]->kind, VALUE_FINITE}, c_neg[2] = {c[0]->neg, 0};
        const product_sums *Pp[2] = {has_p ? P[0] : NULL, NULL};
        gemm_special(&kind, &neg, &epi->ch, 0, Pp, c_kind, c_neg);
        if (kind != VALUE_FINITE) return 0;
        *bounded = 1;
        rc = gemm_mma_box(lo, hi, &sides, mma, epi, P[0], has_p, c[0], c_radius[0]);
        *possible |= sides;
        return rc;
    }
    if (epi->on) {
        int kind, neg, c_kind[2], c_neg[2];
        const product_sums *p[2];
        const dyad *cv[2], *rcv[2];
        for (int q = 0; q < 2; q++) {
            p[q] = g->has_p ? P[q] : NULL;
            c_kind[q] = c[q] ? c[q]->kind : VALUE_FINITE;
            c_neg[q] = c[q] ? c[q]->neg : 0;
            cv[q] = g->has_c && c[q] ? &c[q]->v : NULL;
            rcv[q] = g->has_c && epi->c_radius && c_radius[q] ? &c_radius[q]->v : NULL;
        }
        gemm_special(&kind, &neg, &epi->ch, epi->part, p, c_kind, c_neg);
        if (kind != VALUE_FINITE) return 0;
        *bounded = 1;
        if ((rc = gemm_box(lo, hi, &checked, &sides, g, normal_flags, epi->part, p, cv, rcv))) return rc;
        if (normal) {
            /* the normal form, else the any-order one; where the normal check passes, the narrower */
            dyad alo, ahi, wa, wn;
            dyad_init(&alo);
            dyad_init(&ahi);
            dyad_init(&wa);
            dyad_init(&wn);
            int any_checked, any_sides;
            rc = gemm_box(&alo, &ahi, &any_checked, &any_sides, g_any, 0, epi->part, p, cv, rcv);
            if (!rc && checked) {
                dyad n = *lo, na = alo; /* read-only views of -lo */
                n.m.neg = !n.m.neg && n.m.n;
                na.m.neg = !na.m.neg && na.m.n;
                if (!(rc = dyad_add_exact(&wn, hi, &n)) && !(rc = dyad_add_exact(&wa, &ahi, &na)) && dyad_cmp(&wa, &wn) < 0 &&
                    !(rc = dyad_copy(lo, &alo)))
                    rc = dyad_copy(hi, &ahi);
                sides = 0;
            } else if (!rc) {
                sides = any_checked ? 0 : sides | any_sides;
                if (!(rc = dyad_copy(lo, &alo))) rc = dyad_copy(hi, &ahi);
            }
            dyad_free(&alo);
            dyad_free(&ahi);
            dyad_free(&wa);
            dyad_free(&wn);
        }
        *possible |= sides;
        return rc;
    }
    if (r->kind != VALUE_FINITE) return 0;
    *bounded = 1;
    if (exact) {
        /* exact accumulation: the sum s -+ d rounded once to the accumulator format */
        bigint t, nd = r->dev; /* a read-only view of -d */
        big_init(&t);
        nd.neg = !nd.neg && nd.n;
        if (!(rc = big_add(&t, &r->s, &nd)) && !(rc = dyad_from_big(lo, &t, r->e)) && !(rc = big_add(&t, &r->s, &r->dev)) && !(rc = dyad_from_big(hi, &t, r->e))) {
            rc = rounding_interval(lo, hi, &sides, compute, w);
        }
        big_free(&t);
    } else if (mma && mma->on) {
        rc = mma_box(lo, hi, &sides, mma, r);
    } else if (tree) {
        rc = sum_bounds_tree(lo, hi, &sides, sb, &r->s, &r->mass, &r->N, &r->moved, &r->dev, NULL, r->e);
    } else if (normal) {
        rc = sum_bounds_normal(lo, hi, &checked, &sides, sb, &r->s, &r->mass, &r->N, &r->moved, r->e); /* a term's own overflow: infinities_meet */
    } else {
        rc = sum_bounds_any_order(lo, hi, &sides, sb, &r->s, &r->mass, &r->moved, r->e);
    }
    *possible |= sides;
    return rc;
}

/* Whether row r of a radii operand is all zero radii: its inputs exact. */
static int radii_zero(const operand *x, size_t r) { return !x->max_sig[r] && !x->special_rows[r]; }

/* The few-term rule's bounds for one element (product_few_terms), where it applies (*applied): the
   reference finite, at most two nonzero finite terms; final_bounds' result over the candidates. passed holds
   the flags the inputs' marks pass on (inputs of radius 0: each stored value is its midpoint or a non-finite
   value its marks allow, which makes the value before the final rounding non-finite): their values join the
   candidates, with NaN where a passed infinity can meet a candidate's infinity of the other sign. */
static int few_bounds(int *applied, bigint *tol, bigint *low, bigint *high, int *found, const operand *a, size_t i, const operand *b, size_t j,
                      const floatfmt *compute, int exact_terms, const floatfmt *out, const floatfmt *f64, int rnd, int ovf, const bigint *ref,
                      int passed) {
    *applied = 0;
    int kind, neg, rc;
    dyad r;
    dyad_init(&r);
    rc = floatfmt_decode(&kind, &neg, &r.m, &r.e, out, ref);
    dyad_free(&r);
    if (rc || kind != VALUE_FINITE) return rc;
    value vals[PRODUCT_FEW_MAX];
    int kinds[PRODUCT_FEW_MAX + 3], negs[PRODUCT_FEW_MAX + 3]; /* the candidates, then one slot per passed flag (flag_kinds) */
    bigint nums[PRODUCT_FEW_MAX + 3], dens[PRODUCT_FEW_MAX + 3];
    for (int n = 0; n < PRODUCT_FEW_MAX; n++) dyad_init(&vals[n].v);
    for (int n = 0; n < PRODUCT_FEW_MAX + 3; n++) {
        big_init(&nums[n]);
        big_init(&dens[n]);
    }
    size_t count;
    rc = product_few_terms(&count, vals, a, i, b, j, compute, exact_terms);
    for (size_t n = 0; !rc && n < count; n++) {
        /* each candidate as (kind, sign, numerator, denominator) */
        kinds[n] = vals[n].kind;
        negs[n] = vals[n].neg;
        bigint mag = vals[n].v.m; /* a read-only view */
        mag.neg = 0;
        int64_t e = vals[n].v.e;
        if (!(rc = big_shl(&nums[n], &mag, e > 0 ? e : 0)) && !(rc = big_set_word(&dens[n], 1, 0))) rc = big_shl(&dens[n], &dens[n], e < 0 ? -e : 0);
        if (!rc && kinds[n] != VALUE_FINITE && !(rc = big_set_word(&nums[n], 0, 0))) rc = big_set_word(&dens[n], 1, 0);
    }
    if (!rc && count && passed) {
        int plus = 0, minus = 0;
        for (size_t n = 0; n < count; n++)
            if (kinds[n] == VALUE_INFINITE) *(negs[n] ? &minus : &plus) = 1;
        if ((passed & FLAG_PLUS_INF && minus) || (passed & FLAG_MINUS_INF && plus)) passed |= FLAG_NAN;
        static const int flag_kinds[3][3] = {{FLAG_PLUS_INF, VALUE_INFINITE, 0}, {FLAG_MINUS_INF, VALUE_INFINITE, 1}, {FLAG_NAN, VALUE_NAN, 0}};
        for (int y = 0; !rc && y < 3; y++) {
            if (!(passed & flag_kinds[y][0])) continue;
            kinds[count] = flag_kinds[y][1];
            negs[count] = flag_kinds[y][2];
            if (!(rc = big_set_word(&nums[count], 0, 0)) && !(rc = big_set_word(&dens[count], 1, 0))) count++;
        }
    }
    if (!rc && count) {
        *applied = 1;
        rc = candidate_bounds(tol, low, high, found, out, f64, rnd, ovf, count, kinds, negs, nums, dens, ref);
    }
    for (int n = 0; n < PRODUCT_FEW_MAX; n++) dyad_free(&vals[n].v);
    for (int n = 0; n < PRODUCT_FEW_MAX + 3; n++) {
        big_free(&nums[n]);
        big_free(&dens[n]);
    }
    return rc;
}

/* The normal answer of element (i, j) before its final rounding with exact terms the compute format need not hold
   (TERMS_EXACT: add's with a unit alpha, add.md): no term rounded on its own, so the first two terms' exact sum rounded
   once, each later one added exactly to the sum and rounded (femu_add_exact), one term itself; factors read as
   product_normal reads them. Each term must have a one-word significand, as a term with a factor of 1 has
   (EXACTREF_INTERNAL otherwise). */
static int exact_terms_normal(fvalue *r, const femu *x, const operand *a, size_t i, const operand *b, size_t j) {
    int any = 0;
    for (size_t q = 0; q < a->cols; q++) {
        if (slot_absent(b, j * b->cols + q)) continue;
        int ka, kb, na, nb;
        uint64_t sa, sb, hi;
        int64_t ea, eb;
        fvalue fa, fb, t;
        slot_value(&ka, &na, &sa, &ea, a, i * a->cols + q);
        slot_value(&kb, &nb, &sb, &eb, b, j * b->cols + q);
        femu_read(&fa, x, ka, na, sa, ea);
        femu_read(&fb, x, kb, nb, sb, eb);
        if (fa.kind == VALUE_FINITE && fb.kind == VALUE_FINITE) {
            /* the exact product */
            t.kind = VALUE_FINITE;
            t.neg = fa.neg != fb.neg;
            t.sig = word_mul(&hi, fa.sig, fb.sig);
            t.e = t.sig ? fa.e + fb.e : 0;
            if (hi) return EXACTREF_INTERNAL;
        } else {
            femu_mul(&t, x, &fa, &fb); /* IEEE's rules for its kind and sign */
        }
        if (any)
            femu_add_exact(r, x, r, &t);
        else
            *r = t;
        any = 1;
    }
    if (!any) {
        r->kind = VALUE_FINITE;
        r->neg = 0;
        r->sig = 0;
        r->e = 0;
    }
    return 0;
}

/* The normal answer's outputs (proofs/foundations.md, Lemma E; gemm.md, The normal answer): each element's sum in the
   compute format, rounded to nearest (flush as the option sets), no term rounded on its own with exact_terms
   TERMS_EXACT (exact_terms_normal), with gemm's epilogue fl(fl(alpha P) + fl(beta c)) per part, beta c exact where its
   channels are (gemm_c_terms), then encoded in f (rnd, ovf) at enc, for output elements o0 .. o0 + count - 1. */
int product_normal_run(unsigned char *enc, const operand *a, const operand *const bs[2], const addressing *where, const epilogue *epi,
                       const floatfmt *compute, int flush, int exact_terms, const floatfmt *f, int rnd, int ovf, size_t o0, size_t count) {
    size_t itemsize = (size_t)(1 + f->exp_bits + f->man_bits) / 8;
    int rc = 0;
    femu x;
    fvalue v, P[2], cv[2], scaled[2], coefficients[2][2], *terms = malloc((a->cols + 1) * sizeof *terms);
    bigint bits, sig;
    value c;
    big_init(&bits);
    big_init(&sig);
    dyad_init(&c.v);
    rc = terms ? femu_set(&x, compute, ROUND_NEAREST_EVEN, flush) : EXACTREF_NOMEM;
    int need[2] = {1, 0}, need_c[2] = {0, 0}, counts[2] = {0, 0};
    if (epi->on) {
        epilogue_needs(need, need_c, epi);
        counts[0] = epi->ch.n_p[epi->part];
        counts[1] = epi->ch.n_c[epi->part];
    }
    int has_p = !epi->on || (counts[0] && a->cols > 0);
    /* this part's channels' coefficients as values: their significands fit a word (the caller's check), exact, never flushed */
    for (int side = 0; side < 2; side++)
        for (int k = 0; k < counts[side]; k++) {
            const dyad *s = &(side ? epi->ch.c : epi->ch.p)[epi->part][k].s;
            coefficients[side][k].kind = VALUE_FINITE;
            coefficients[side][k].neg = s->m.neg != 0;
            coefficients[side][k].sig = s->m.w[0];
            coefficients[side][k].e = s->e;
        }
    for (size_t at = 0; !rc && at < count; at++) {
        size_t i, j, o = o0 + at;
        element_rows(&i, &j, where, o);
        for (int q = 0; !rc && has_p && q < 2; q++)
            if (need[q])
                rc = !epi->on && exact_terms == TERMS_EXACT ? exact_terms_normal(&P[q], &x, a, i, bs[q], j)
                                                             : product_normal(&P[q], terms, &x, a, i, bs[q], j);
        if (rc) break;
        if (!epi->on) {
            v = P[0];
        } else {
            /* fl(fl(alpha P) + fl(beta c)) per part from its channels: each product rounded, then with two
               channels their sum (proofs/gemm.md, The normal answer) */
            for (int y = 0; !rc && y < 2; y++)
                if (need_c[y] && !(rc = epilogue_c(&c, epi, o, y))) femu_read(&cv[y], &x, c.kind, c.neg, c.v.m.n ? c.v.m.w[0] : 0, c.v.e);
            if (rc) break;
            int any[2] = {0, 0}, c_exact = epi->c_terms != TERMS_ROUNDED;
            for (int side = 0; side < 2; side++) {
                if (!side && !has_p) continue;
                for (int k = 0; k < counts[side]; k++) {
                    const gemm_channel *ch = &(side ? epi->ch.c : epi->ch.p)[epi->part][k];
                    fvalue t;
                    if (side && c_exact) {
                        /* a unit channel: +-c exactly, which the compute format need not hold */
                        t = cv[ch->q];
                        t.neg = t.kind != VALUE_NAN && t.neg != coefficients[side][k].neg;
                    } else {
                        femu_mul(&t, &x, &coefficients[side][k], side ? &cv[ch->q] : &P[ch->q]);
                    }
                    if (any[side])
                        femu_add(&scaled[side], &x, &scaled[side], &t);
                    else
                        scaled[side] = t;
                    any[side] = 1;
                }
            }
            if (any[0] && any[1]) {
                if (c_exact)
                    femu_add_exact(&v, &x, &scaled[0], &scaled[1]);
                else
                    femu_add(&v, &x, &scaled[0], &scaled[1]);
            } else if (any[0] || any[1]) {
                v = scaled[any[0] ? 0 : 1];
            } else {
                v.kind = VALUE_FINITE;
                v.neg = 0;
                v.sig = 0;
                v.e = 0;
            }
        }
        if ((rc = big_set_word(&sig, v.sig, 0)) || (rc = floatfmt_encode(&bits, f, v.kind, v.neg, &sig, v.e, rnd, ovf))) break;
        bytes_store(enc + at * itemsize, itemsize, &bits);
    }
    free(terms);
    big_free(&bits);
    big_free(&sig);
    dyad_free(&c.v);
    return rc;
}

/* A named algorithm's outputs (MatmulAlgorithm.run): method 1 sequential, 2 pairwise, 3 blocked (block keys a block), 4
   exact accumulation, 5 the 16-slot algorithm (its products rounded), in the accumulator format acc with its rounding
   acc_rnd; products exact (fma) or rounded first; then each rounded to f (rnd, ovf) at enc, for output elements o0 ..
   o0 + count - 1. */
int product_algorithm_run(unsigned char *enc, const operand *a, const operand *b, const addressing *where, const floatfmt *acc, int acc_rnd,
                          int method, size_t block, int fma, const floatfmt *f, int rnd, int ovf, size_t o0, size_t count) {
    size_t itemsize = (size_t)(1 + f->exp_bits + f->man_bits) / 8, k = a->cols;
    int rc = 0;
    femu x;
    fvalue v, fa, fb, *scratch = malloc((k + 1) * sizeof *scratch);
    fterm *terms = malloc((k + 1) * sizeof *terms);
    bigint bits, sig;
    slots16 sl;
    int slots_ready = 0;
    big_init(&bits);
    big_init(&sig);
    rc = terms && scratch ? femu_set(&x, acc, acc_rnd, 0) : EXACTREF_NOMEM;
    if (!rc && method == 5 && !(rc = slots16_init(&sl, k))) slots_ready = 1;
    for (size_t at = 0; !rc && at < count; at++) {
        size_t i, j;
        element_rows(&i, &j, where, o0 + at);
        if (method == 5) {
            /* the 16-slot algorithm: the finite products by their slots, then those that are not finite by IEEE rules */
            int kind, neg;
            if ((rc = slots16_element(&sl, &kind, &neg, &x, a, i, b, j))) break;
            slots16_run(&v, &sl, &x);
            if (kind != VALUE_FINITE && v.kind != VALUE_NAN) {
                v.kind = kind == VALUE_NAN || (v.kind == VALUE_INFINITE && v.neg != neg) ? VALUE_NAN : VALUE_INFINITE;
                v.neg = v.kind == VALUE_INFINITE && neg;
                v.sig = 0;
                v.e = 0;
            }
            if ((rc = big_set_word(&sig, v.sig, 0)) || (rc = floatfmt_encode(&bits, f, v.kind, v.neg, &sig, v.e, rnd, ovf))) break;
            bytes_store(enc + at * itemsize, itemsize, &bits);
            continue;
        }
        for (size_t q = 0; q < k; q++) {
            /* the exact product, its kind and sign by IEEE's rules, and with rounded products its rounding */
            int ka, kb, na, nb;
            uint64_t sa, sb;
            int64_t ea, eb;
            fterm *t = &terms[q];
            element_decode_slot(&ka, &na, &sa, &ea, a, i * k + q);
            element_decode_slot(&kb, &nb, &sb, &eb, b, j * k + q);
            femu_read(&fa, &x, ka, na, sa, ea);
            femu_read(&fb, &x, kb, nb, sb, eb);
            if (fma && fa.kind == VALUE_FINITE && fb.kind == VALUE_FINITE) {
                t->v.kind = VALUE_FINITE;
                t->v.sig = 0;
                t->v.e = 0;
            } else {
                femu_mul(&t->v, &x, &fa, &fb);
            }
            if (t->v.kind == VALUE_FINITE) {
                t->v.neg = fa.neg != fb.neg;
                t->lo = word_mul(&t->hi, fa.sig, fb.sig);
                t->e = fa.e + fb.e;
            }
        }
        if ((rc = femu_run(&v, &x, acc, method, block, terms, fma, k, scratch)) ||
            (rc = big_set_word(&sig, v.sig, 0)) || (rc = floatfmt_encode(&bits, f, v.kind, v.neg, &sig, v.e, rnd, ovf)))
            break;
        bytes_store(enc + at * itemsize, itemsize, &bits);
    }
    if (slots_ready) slots16_free(&sl);
    free(scratch);
    free(terms);
    big_free(&bits);
    big_free(&sig);
    return rc;
}

/* x = the integer of the dyadic v (an integer, or its floor). */
static int dyad_integer(bigint *x, const dyad *v) { return v->e >= 0 ? big_shl(x, &v->m, v->e) : big_shr(x, &v->m, -v->e, 0); }

/* Element at's tolerance (8 bytes), its ends (itemsize bytes each) and its flags in a bounds call's outputs */
static void bounds_store(unsigned char *t, unsigned char *lo_out, unsigned char *hi_out, unsigned char *fl, size_t at, size_t itemsize,
                         const bigint *tol, const bigint *low, const bigint *high, int found) {
    bytes_store(t + 8 * at, 8, tol);
    bytes_store(lo_out + at * itemsize, itemsize, low);
    bytes_store(hi_out + at * itemsize, itemsize, high);
    fl[at] = (unsigned char)found;
}

/* An integer product's bounds (finish.c's int_bounds): each element's exact
   values lie within the deviation of the operands' radii (radii as product_bounds takes them) of
   its exact sum, and become outputs by the overflow mode. For elements o0 .. o0 + count - 1, the
   tolerances into t (8 bytes each), the bounds into lo_out and hi_out (pf's item size each) and
   flags 0 into fl. */
int product_int_bounds_run(unsigned char *t, unsigned char *lo_out, unsigned char *hi_out, unsigned char *fl, const operand *a, const operand *b,
                           const operand *ra, const operand *rb, const addressing *where, const epilogue *epi, const partfmt *pf, const floatfmt *f64,
                           int ovf, const unsigned char *refs, size_t o0, size_t count) {
    size_t itemsize = part_size(pf);
    int rc = 0;
    product_sums r, scratch;
    bigint lo, hi, ref, tol, low, high;
    bigint *bigs[] = {&lo, &hi, &ref, &tol, &low, &high};
    dyad x, y, tc, u;
    value c, cr;
    product_sums_init(&r);
    product_sums_init(&scratch);
    for (size_t k = 0; k < sizeof bigs / sizeof *bigs; k++) big_init(bigs[k]);
    dyad_init(&x);
    dyad_init(&y);
    dyad_init(&tc);
    dyad_init(&u);
    dyad_init(&c.v);
    dyad_init(&cr.v);
    /* with gemm's epilogue, A and B are read only with alpha's terms */
    int has_p = !epi->on || (epilogue_has_p(epi) && a->cols > 0), has_c = epi->on && epilogue_has_c(epi);
    for (size_t at = 0; !rc && at < count; at++) {
        size_t i, j, o = o0 + at;
        int unbounded = 0;
        element_rows(&i, &j, where, o);
        r.s.n = r.dev.n = 0;
        r.e = 0;
        if (has_p && (rc = product_element(&r, a, i, b, j, PRODUCT_MASS, 0, 0, NULL))) break;
        r.dev.n = 0;
        if (has_p && (ra || rb) && (rc = product_deviation(&r, &unbounded, &scratch, a, ra, i, b, rb, j, 0))) break;
        /* the exact integers s and d (rounded down: the sums are integers) */
        bigint *xs[2] = {&r.s, &r.dev};
        for (int k = 0; !rc && k < 2; k++) rc = r.e >= 0 ? big_shl(xs[k], xs[k], r.e) : big_shr(xs[k], xs[k], -r.e, 0);
        if (!rc && epi->on) {
            /* gemm: alpha s + beta c, within |alpha| d + |beta| r of the values stored inputs within their radii give,
               r C's radius rounded down */
            int c_unbounded = 0;
            dyad abs_alpha = epi->alpha, abs_beta = epi->beta; /* read-only views of the magnitudes */
            abs_alpha.m.neg = abs_beta.m.neg = 0;
            dyad sv, dv; /* read-only views of s and d */
            sv.m = r.s;
            dv.m = r.dev;
            sv.e = dv.e = 0;
            if (!(rc = dyad_mul_exact(&x, &sv, &epi->alpha))) rc = dyad_mul_exact(&y, &dv, &abs_alpha);
            if (!rc && has_c && !(rc = epilogue_c(&c, epi, o, 0)) && !(rc = dyad_mul_exact(&tc, &c.v, &epi->beta)) && !(rc = dyad_add_exact(&x, &x, &tc)) &&
                !(rc = epilogue_radius(&cr, &c_unbounded, epi, o, 0)) && epi->c_radius && !c_unbounded && !(rc = dyad_integer(&u.m, &cr.v))) {
                u.e = 0;
                if (!(rc = dyad_mul_exact(&tc, &u, &abs_beta))) rc = dyad_add_exact(&y, &y, &tc);
            }
            unbounded |= c_unbounded;
            if (!rc && !(rc = dyad_integer(&r.s, &x))) rc = dyad_integer(&r.dev, &y);
        }
        if (!rc && unbounded) rc = big_set_bit(&r.dev, 130); /* past every integer format's range */
        bigint nd = r.dev; /* a read-only view of -d */
        nd.neg = !nd.neg && nd.n;
        if (!rc && !(rc = big_add(&lo, &r.s, &nd)) && !(rc = big_add(&hi, &r.s, &r.dev)) && !(rc = bytes_load(&ref, refs + at * itemsize, itemsize, 0)))
            rc = int_bounds(&tol, &low, &high, f64, pf->itemsize, pf->is_signed, ovf, &lo, &hi, &ref);
        if (rc) break;
        bounds_store(t, lo_out, hi_out, fl, at, itemsize, &tol, &low, &high, 0);
    }
    product_sums_free(&r);
    product_sums_free(&scratch);
    for (size_t k = 0; k < sizeof bigs / sizeof *bigs; k++) big_free(bigs[k]);
    dyad_free(&x);
    dyad_free(&y);
    dyad_free(&tc);
    dyad_free(&u);
    dyad_free(&c.v);
    dyad_free(&cr.v);
    return rc;
}

/* Per side of the batch holding output element o, its first: the largest magnitude of its matrix's entries plus their
   largest radius (rows_largest) */
static int batch_largest(dyad x[2], const operand *a, const operand *ra, const operand *b, const operand *rb, const addressing *where, size_t o) {
    size_t i0, j0;
    element_rows(&i0, &j0, where, o);
    const operand *xs[2] = {a, b}, *rs[2] = {ra, rb};
    size_t starts[2] = {i0, j0}, counts[2] = {where->m, where->n};
    dyad y;
    dyad_init(&y);
    int rc = 0;
    for (int side = 0; !rc && side < 2; side++) {
        if (!(rc = dyad_set_si(&x[side], 0, 0)) && !(rc = rows_largest(&x[side], xs[side], starts[side], counts[side])) && rs[side] &&
            !(rc = dyad_set_si(&y, 0, 0)) && !(rc = rows_largest(&y, rs[side], starts[side], counts[side])))
            rc = dyad_add_exact(&x[side], &x[side], &y);
    }
    dyad_free(&y);
    return rc;
}

/* Lemma G's constants for a batch of gemm without mma, after freeing the last batch's (*ready): W, every factor's largest
   entry plus its largest radius where alpha's terms count, then g, and with the normal standard the any-order g_any;
   with gemm's accumulation alone (sb_acc), the batch's gate *may, every factor at most W, and its reach gr (gemm.md,
   Overflow): every value of one sign at most (1 + r)**(n + 2) (S + n M (a (3 + W) + a2)), n the part's leaves,
   M = 2 max(1, |sigma|). *ready once g (and g_any) hold the batch's. */
static int gemm_batch(gemm_bounds *g, gemm_bounds *g_any, int *ready, gemm_reach *gr, int *may, const operand *a, const operand *ra,
                      const operand *b, const operand *rb, const addressing *where, size_t batch, epilogue *epi, const floatfmt *compute, int flush,
                      int normal, int has_p, const sum_bounds *sb_acc, int64_t w) {
    if (*ready) {
        gemm_bounds_free(g);
        if (normal) gemm_bounds_free(g_any);
        *ready = 0;
    }
    size_t k = a->cols;
    dyad W, x[2], t, u;
    dyad *all[] = {&W, &x[0], &x[1], &t, &u};
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) dyad_init(all[z]);
    int rc = has_p ? batch_largest(x, a, ra, b, rb, where, batch * where->m * where->n) : 0;
    if (!rc && has_p) rc = dyad_copy(&W, dmax(&x[0], &x[1]));
    int made = !rc && !(rc = gemm_bounds_init(g, compute, flush, normal, has_p ? (int64_t)k : 0, &epi->ch, &epi->fold, epi->c_terms, &W, w));
    if (!rc && sb_acc) {
        int64_t leaves = 0, sigma_top = 0;
        for (int part = 0; part < epi->ch.parts; part++) {
            int64_t l = (has_p ? (int64_t)k * epi->ch.n_p[part] : 0) + epi->ch.n_c[part];
            if (l > leaves) leaves = l;
            for (int n = 0; n < epi->ch.n_p[part]; n++) {
                int64_t top = epi->ch.p[part][n].s.e + (int64_t)big_bits(&epi->ch.p[part][n].s.m);
                if (top > sigma_top) sigma_top = top;
            }
        }
        rc = terms_may(may, &W, &W, (int64_t)k, epi, 0, 1, sb_acc);
        if (!rc && !(rc = dyad_set_si(&t, 1, 0)) && !(rc = dyad_add(&t, &t, &sb_acc->unit_r, w, 1))) rc = dyad_power(&gr->g, &t, leaves + 2, w, 1);
        /* n M (a (3 + W) + a2), M = 2**(max(sigma_top, 0) + 1) (|sigma| below 2**sigma_top) */
        if (!rc && !(rc = dyad_set_si(&t, 3, 0)) && !(rc = dyad_add(&t, &t, &W, w, 1)) && !(rc = dyad_mul(&t, &t, &sb_acc->unit_a, w, 1)) &&
            !(rc = dyad_set_si(&u, flush ? 1 : 0, compute->emin)) && !(rc = dyad_add(&t, &t, &u, w, 1)) &&
            !(rc = dyad_set_si(&u, 1, (sigma_top > 0 ? sigma_top : 0) + 1)) && !(rc = dyad_mul(&t, &t, &u, w, 1)) && !(rc = dyad_set_si(&u, leaves, 0)))
            rc = dyad_mul(&gr->c, &t, &u, w, 1);
        gr->ready = !rc;
    }
    if (!rc && normal) rc = gemm_bounds_init(g_any, compute, flush, 0, has_p ? (int64_t)k : 0, &epi->ch, &epi->fold, epi->c_terms, &W, w);
    if (rc && made) gemm_bounds_free(g);
    *ready = !rc;
    for (size_t z = 0; z < sizeof all / sizeof *all; z++) dyad_free(all[z]);
    return rc;
}

/* The 16-slot algorithm's state in a bounds call (kind 5): its slots and accumulator, Lemma S's bounds per tree depth
   (made as an element needs one), and an element's own tree's sums */
typedef struct {
    slots16 sl;
    femu acc;
    sum_bounds *trees;
    unsigned char *tree_ready;
    size_t tree_count;
    int slots_ready;
    bigint ts, tmass, tN, tlargest, tzero;
} slots_bounds;

static void slots_bounds_init(slots_bounds *s) {
    s->trees = NULL;
    s->tree_ready = NULL;
    s->tree_count = 0;
    s->slots_ready = 0;
    bigint *all[] = {&s->ts, &s->tmass, &s->tN, &s->tlargest, &s->tzero};
    for (size_t q = 0; q < sizeof all / sizeof *all; q++) big_init(all[q]);
}

/* the state for k terms, the accumulator rounding by rnd */
static int slots_bounds_prepare(slots_bounds *s, const floatfmt *compute, int rnd, size_t k) {
    s->tree_count = k + SLOTS16 + 1;
    s->trees = malloc(s->tree_count * sizeof *s->trees);
    s->tree_ready = calloc(s->tree_count, 1);
    int rc = s->trees && s->tree_ready ? femu_set(&s->acc, compute, rnd, 0) : EXACTREF_NOMEM;
    if (!rc && !(rc = slots16_init(&s->sl, k))) s->slots_ready = 1;
    return rc;
}

static void slots_bounds_free(slots_bounds *s) {
    for (size_t q = 0; s->tree_ready && q < s->tree_count; q++)
        if (s->tree_ready[q]) sum_bounds_free(&s->trees[q]);
    free(s->trees);
    free(s->tree_ready);
    if (s->slots_ready) slots16_free(&s->sl);
    bigint *all[] = {&s->ts, &s->tmass, &s->tN, &s->tlargest, &s->tzero};
    for (size_t q = 0; q < sizeof all / sizeof *all; q++) big_free(all[q]);
}

/* The 16-slot algorithm's bound on element (i, j): none where a factor, a rounded product or a radius is not finite, and
   there the infinities the sum can meet (any tree's sides: its products' tree, in *possible); Lemma S over the element's
   own tree, or with uncertain inputs over any tree of its terms (ops/matmul.py's MatmulSlots16._bounds), the ends in lo
   and hi and *bounded set. r and scratch are the call's product sums. */
static int slots_bound(dyad *lo, dyad *hi, int *bounded, int *possible, slots_bounds *s, product_sums *r, product_sums *scratch, const operand *a,
                       const operand *ra, size_t i, const operand *const bs[2], const operand *const rbs[2], size_t j, const epilogue *epi,
                       const value *const c[2], const value *const c_radius[2], const sum_bounds *sb, int batch_may, const floatfmt *compute,
                       int64_t w) {
    const operand *b = bs[0], *rb = rbs[0];
    size_t k = a->cols;
    int kind, neg, sides = 0, rc;
    *bounded = 0;
    if ((rc = slots16_element(&s->sl, &kind, &neg, &s->acc, a, i, b, j))) return rc;
    if ((ra && ra->special_rows[i]) || (rb && rb->special_rows[j])) return 0;
    if (kind != VALUE_FINITE) {
        if ((rc = product_element(r, a, i, b, j, PRODUCT_MASS, 0, 0, NULL))) return rc;
        const product_sums *P[2] = {r, NULL};
        return infinities_meet(possible, 0, P, epi, c, c_radius, a, ra, i, bs, rbs, j, sb, 1, batch_may, 5, 1, NULL, NULL);
    }
    int64_t e, d;
    const bigint *s_, *mass_, *N_, *moved_, *dev_, *largest_;
    if ((ra && !radii_zero(ra, i)) || (rb && !radii_zero(rb, j))) {
        int ignored;
        if ((rc = product_element(r, a, i, b, j, PRODUCT_MASS, 0, 0, NULL))) return rc;
        r->dev.n = 0;
        if ((rc = product_deviation(r, &ignored, scratch, a, ra, i, b, rb, j, 1))) return rc;
        /* mass and moved hold d: N = 2 (mass + d + (k - 1) (mass - d)) */
        bigint nd = r->dev; /* a read-only view of -d */
        nd.neg = !nd.neg && nd.n;
        if ((rc = big_add(&s->tN, &r->mass, &nd)) || (rc = big_set_word(&s->tzero, k ? (uint64_t)k - 1 : 0, 0)) ||
            (rc = big_mul(&s->tlargest, &s->tN, &s->tzero)) || (rc = big_add(&s->tN, &s->tlargest, &r->mass)) || (rc = big_shl(&s->tN, &s->tN, 1)))
            return rc;
        s->tzero.n = 0;
        s_ = &r->s;
        mass_ = largest_ = &r->mass;
        N_ = &s->tN;
        moved_ = dev_ = &r->dev;
        e = r->e;
        d = k ? (int64_t)k - 1 : 0;
    } else {
        if ((rc = slots16_tree(&s->ts, &s->tmass, &s->tN, &s->tlargest, &e, &d, &s->sl, a, i, b, j))) return rc;
        s_ = &s->ts;
        mass_ = &s->tmass;
        N_ = &s->tN;
        largest_ = &s->tlargest;
        moved_ = dev_ = &s->tzero;
    }
    size_t at_d = (size_t)d < s->tree_count ? (size_t)d : s->tree_count - 1;
    if (!s->tree_ready[at_d]) {
        if ((rc = sum_bounds_init(&s->trees[at_d], compute, 0, (int64_t)k, d, 0, w))) return rc;
        s->tree_ready[at_d] = 1;
    }
    if ((rc = sum_bounds_tree(lo, hi, &sides, &s->trees[at_d], s_, mass_, N_, moved_, dev_, largest_, e))) return rc;
    *bounded = 1;
    *possible |= sides;
    return 0;
}

/* The bounds of elements o0 .. o0 + count - 1 of a product: each element's interval under matmul's bound, gemm's with
   the epilogue, Lemma T's with mma, or a named order's (order_kind, block), and the infinities its sum can meet, made
   final against its reference in refs; the tolerances into t (8 bytes each), the bounds into lo_out and hi_out (f's item
   size each) and the flags into fl. */
int product_bounds_run(unsigned char *t, unsigned char *lo_out, unsigned char *hi_out, unsigned char *fl, const operand *a, const operand *const bs[2],
                       const operand *ra, const operand *const rbs[2], const addressing *where, epilogue *epi, const mma_args *mma, const floatfmt *compute,
                       const floatfmt *f, const floatfmt *f64, int flush, int normal, int exact_terms, int rnd, int ovf, int few, int order_kind,
                       int64_t block, int64_t w, const unsigned char *refs, const unsigned char *row_marks, const unsigned char *col_marks, int reach,
                       const unsigned char *reach_marks, size_t o0, size_t count) {
    const operand *b = bs[0];
    /* with a complex alpha, the any-order bound only, with the call's own standard's flags (proofs/gemm.md, Normal with a
       complex alpha) */
    int normal_flags = normal;
    if (epi->on && !epi->ch.alpha_real) normal = 0;
    size_t itemsize = (size_t)(1 + f->exp_bits + f->man_bits) / 8;
    int rc = 0;
    sum_bounds sb;
    gemm_bounds g, g_any;
    product_sums r[2], scratch;
    dyad lo, hi;
    value c[2], c_radius[2];
    bigint ref, tol, low, high;
    for (int q = 0; q < 2; q++) {
        product_sums_init(&r[q]);
        dyad_init(&c[q].v);
        dyad_init(&c_radius[q].v);
    }
    product_sums_init(&scratch);
    dyad_init(&lo);
    dyad_init(&hi);
    big_init(&ref);
    big_init(&tol);
    big_init(&low);
    big_init(&high);
    size_t k = a->cols, batch = (size_t)-1, mn = where->m * where->n;
    int ready = 0, gemm_ready = 0; /* each init frees what it made when it fails */
    int has_p = !epi->on || (epilogue_has_p(epi) && k > 0), need[2] = {1, 0}, need_c[2] = {0, 0};
    if (epi->on) epilogue_needs(need, need_c, epi);
    const product_sums *P[2];
    const value *cp[2] = {&c[0], &c[1]}, *crp[2] = {&c_radius[0], &c_radius[1]};
    product_order order = {0, NULL, NULL};
    if (order_kind == 1) rc = product_order_sequential(&order, k);
    if (order_kind == 2) rc = product_order_pairwise(&order, k);
    if (order_kind == 3) rc = product_order_blocked(&order, k, block > 0 ? (size_t)block : 1);
    /* exact accumulation (kind 4) takes no nodes */
    int64_t tree_depth = order_kind >= 1 && order_kind <= 3 ? product_order_depth(order_kind, k, block > 0 ? (size_t)block : 1)
                         : order_kind ? 0 : k ? (int64_t)k - 1 : 0;
    if (!rc && !epi->on && !(rc = sum_bounds_init(&sb, compute, flush, (int64_t)k, tree_depth, exact_terms, w))) ready = 1;
    /* gemm's accumulation alone, for the infinities its terms may reach (infinities_meet) */
    sum_bounds sb_acc;
    int acc_ready = 0;
    if (!rc && epi->on && !(rc = sum_bounds_init(&sb_acc, compute, flush, (int64_t)k, tree_depth, exact_terms, w))) acc_ready = 1;
    if (!rc && epi->on) rc = gemm_fold_set(&epi->fold, &epi->ch, compute, flush);
    /* per batch, whether a term or a side may reach theta (infinities_meet), and gemm's one-sided reach */
    int batch_may = MAY_TERM | MAY_SIDE;
    size_t may_batch = (size_t)-1;
    gemm_reach gr;
    dyad_init(&gr.g);
    dyad_init(&gr.c);
    gr.ready = 0;
    /* the 16-slot algorithm (kind 5, block its accumulator's rounding) */
    slots_bounds slots;
    slots_bounds_init(&slots);
    if (!rc && order_kind == 5) rc = slots_bounds_prepare(&slots, compute, (int)block, k);
    tile_order visit;
    tile_order_set(&visit, where, o0, count);
    for (size_t step = 0; !rc && step < count; step++) {
        size_t i, j, at = tile_at(&visit, step), o = o0 + at;
        element_rows(&i, &j, where, o);
        if (epi->on && !mma->on && o / mn != batch) {
            batch = o / mn;
            if ((rc = gemm_batch(&g, &g_any, &gemm_ready, &gr, &batch_may, a, ra, b, rbs[0], where, batch, epi, compute, flush, normal, has_p,
                                 acc_ready ? &sb_acc : NULL, w)))
                break;
        }
        if (!epi->on && ready && mn > 1 && o / mn != may_batch) {
            /* the batch's gate: its rows' largest factors (a batch of one element leaves it to the element's own) */
            may_batch = o / mn;
            dyad x[2];
            dyad_init(&x[0]);
            dyad_init(&x[1]);
            if (!(rc = batch_largest(x, a, ra, b, rbs[0], where, may_batch * mn))) rc = terms_may(&batch_may, &x[0], &x[1], (int64_t)k, epi, 0, 1, &sb);
            dyad_free(&x[0]);
            dyad_free(&x[1]);
            if (rc) break;
        }
        int marks = has_p && (need[0] || need[1]) ? (row_marks ? row_marks[i] : 0) | (col_marks ? col_marks[j] : 0) : 0, possible = 0, bounded;
        if (marks) possible = FLAG_ALL | (marks & FLAG_ANY); /* a data factor may be anything */
        int found, applied = 0;
        if (few && (!ra || radii_zero(ra, i)) && (!rbs[0] || radii_zero(rbs[0], j))) {
            /* the few-term rule first (a one-row, one-column product without flush), where the inputs' radii are 0, with
               the flags their marks pass on as the general rule takes them: all three through a data factor, the
               coefficient part's sign through a coefficient (product_reach); none can be ANY, as a radius of 0 holds a
               finite midpoint (Bounds.marks), but one would leave the element to the general rule */
            int passed = possible;
            if (reach) {
                int ignored, reached;
                product_reach(&ignored, &reached, a, ra, reach_marks, i, b, j);
                passed |= reached;
            }
            if (!(passed & FLAG_ANY)) {
                if ((rc = bytes_load(&ref, refs + at * itemsize, itemsize, 0)) ||
                    (rc = few_bounds(&applied, &tol, &low, &high, &found, a, i, b, j, compute, exact_terms, f, f64, rnd, ovf, &ref, passed)))
                    break;
                if (applied) {
                    bounds_store(t, lo_out, hi_out, fl, at, itemsize, &tol, &low, &high, found);
                    continue;
                }
            }
        }
        if (order_kind == 5) {
            if ((rc = slots_bound(&lo, &hi, &bounded, &possible, &slots, &r[0], &scratch, a, ra, i, bs, rbs, j, epi, cp, crp, &sb, batch_may, compute, w)) ||
                (rc = bytes_load(&ref, refs + at * itemsize, itemsize, 0)) ||
                (rc = final_bounds(&tol, &low, &high, &found, f, f64, rnd, ovf, bounded ? &lo : NULL, bounded ? &hi : NULL, &ref, w, possible)))
                break;
            bounds_store(t, lo_out, hi_out, fl, at, itemsize, &tol, &low, &high, found);
            continue;
        }
        int what = (normal || (order_kind && order_kind != 4)) && !mma->on ? PRODUCT_NODES : PRODUCT_MASS, unbounded = 0, passed_inf = 0;
        for (int q = 0; !rc && q < 2; q++) {
            P[q] = NULL;
            if (!has_p || !need[q]) continue;
            int q_unbounded = 0;
            if ((rc = product_element(&r[q], a, i, bs[q], j, what, flush, compute->emin, order_kind && order_kind != 4 ? &order : NULL))) break;
            r[q].dev.n = 0;
            if (reach) {
                int reached;
                product_reach(&q_unbounded, &reached, a, ra, reach_marks, i, bs[q], j);
                possible |= reached;
                passed_inf |= reached & (FLAG_PLUS_INF | FLAG_MINUS_INF);
            }
            if (!q_unbounded && r[q].kind == VALUE_FINITE && (ra || rbs[q]) &&
                (rc = product_deviation(&r[q], &q_unbounded, &scratch, a, ra, i, bs[q], rbs[q], j, reach)))
                break;
            unbounded |= q_unbounded;
            P[q] = &r[q];
        }
        if (rc) break;
        if (!epi->on) P[0] = &r[0];
        if (epi->on) {
            for (int y = 0; !rc && y < 2; y++) {
                int c_unbounded = 0;
                if (need_c[y] && !(rc = epilogue_c(&c[y], epi, o, y))) rc = epilogue_radius(&c_radius[y], &c_unbounded, epi, o, y);
                unbounded |= c_unbounded && c[y].kind == VALUE_FINITE; /* a non-finite C's stored values are its marks' */
            }
            if (rc) break;
            int c_marks = epilogue_marks(epi, o);
            possible |= c_marks;
            passed_inf |= c_marks & (FLAG_PLUS_INF | FLAG_MINUS_INF); /* C's stored infinities through beta: events */
        }
        bounded = 0;
        int boxed = 1;
        if (order_kind == 4) {
            /* exact accumulation bounds an element whose reference is finite; the infinities rule still applies */
            int kind, neg;
            dyad v;
            dyad_init(&v);
            if (!(rc = bytes_load(&ref, refs + at * itemsize, itemsize, 0))) rc = floatfmt_decode(&kind, &neg, &v.m, &v.e, f, &ref);
            dyad_free(&v);
            if (rc) break;
            boxed = kind == VALUE_FINITE;
        }
        if (!unbounded && boxed && (rc = element_box(&lo, &hi, &bounded, &possible, P, &sb, &g, &g_any, epi, cp, crp, a, normal, normal_flags,
                                                     order_kind >= 1 && order_kind <= 3, mma, order_kind == 4, compute, w)))
            break;
        /* the infinities the sum can meet (either standard; with products exact, mma's and exact accumulation's, only the
           inputs' and the exact value's) */
        if (!unbounded && (rc = infinities_meet(&possible, passed_inf, P, epi, cp, crp, a, ra, i, bs, rbs, j, epi->on ? &sb_acc : &sb,
                                                !mma->on && order_kind != 4, batch_may, order_kind, block > 0 ? (size_t)block : 1, &gr, mma->on ? mma : NULL)))
            break;
        /* an exact value that is not finite is an output itself: the reference, unless it is another expression's (a
           held-scalar algorithm's, proofs/multiply.md, Held scalars) */
        if (!epi->on && r[0].kind != VALUE_FINITE) possible |= r[0].kind == VALUE_NAN ? FLAG_NAN : r[0].neg ? FLAG_MINUS_INF : FLAG_PLUS_INF;
        if (!rc) rc = bytes_load(&ref, refs + at * itemsize, itemsize, 0);
        if (!rc) rc = final_bounds(&tol, &low, &high, &found, f, f64, rnd, ovf, bounded ? &lo : NULL, bounded ? &hi : NULL, &ref, w, possible);
        if (rc) break;
        bounds_store(t, lo_out, hi_out, fl, at, itemsize, &tol, &low, &high, found);
    }
    if (ready) sum_bounds_free(&sb);
    if (acc_ready) sum_bounds_free(&sb_acc);
    dyad_free(&gr.g);
    dyad_free(&gr.c);
    slots_bounds_free(&slots);
    product_order_free(&order);
    if (gemm_ready) {
        gemm_bounds_free(&g);
        if (normal) gemm_bounds_free(&g_any);
    }
    for (int q = 0; q < 2; q++) {
        product_sums_free(&r[q]);
        dyad_free(&c[q].v);
        dyad_free(&c_radius[q].v);
    }
    product_sums_free(&scratch);
    dyad_free(&lo);
    dyad_free(&hi);
    big_free(&ref);
    big_free(&tol);
    big_free(&low);
    big_free(&high);
    return rc;
}
