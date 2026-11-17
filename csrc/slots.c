/* The 16-slot algorithm (internal; proofs/matmul.md, The 16-slot algorithm; ops/matmul.py's
   MatmulSlots16): per output element each product rounded to the accumulator format, the nonzero
   finite ones split by magnitude into 16 slots of a sixteenth of the format's precision each, in
   binades below the largest (anything smaller in the last), each slot added in increasing magnitude
   with positive and negative values alternating, and the slot sums added from the smallest slot up;
   the products that are not finite then join that sum by IEEE rules. */
#include <stdlib.h>
#include "exactref.h"
#include "words.h"

/* A nonzero rounded term's place: its slot, sign, and magnitude normalized as mant 2**(top - 64)
   with mant's top bit set. */
typedef struct {
    size_t q;
    int slot, neg;
    int64_t top;
    uint64_t mant;
} slot_key;

int slots16_init(slots16 *x, size_t k) {
    x->k = k;
    x->n = x->used = 0;
    x->rounded = malloc((k + 1) * sizeof *x->rounded);
    x->order = malloc((k + 1) * sizeof *x->order);
    x->ends = malloc((SLOTS16 + 1) * sizeof *x->ends);
    x->keys = malloc((k + 1) * sizeof(slot_key));
    x->exact = malloc((k + 1) * sizeof *x->exact);
    for (int t = 0; t < 4; t++) big_init(&x->big[t]);
    if (x->exact)
        for (size_t q = 0; q <= k; q++) big_init(&x->exact[q]);
    if (!x->rounded || !x->order || !x->ends || !x->keys || !x->exact) {
        slots16_free(x);
        return EXACTREF_NOMEM;
    }
    return 0;
}

void slots16_free(slots16 *x) {
    if (x->exact)
        for (size_t q = 0; q <= x->k; q++) big_free(&x->exact[q]);
    for (int t = 0; t < 4; t++) big_free(&x->big[t]);
    free(x->rounded);
    free(x->order);
    free(x->ends);
    free(x->keys);
    free(x->exact);
    x->rounded = NULL;
    x->order = x->ends = NULL;
    x->keys = NULL;
    x->exact = NULL;
}

static void normalize(int64_t *top, uint64_t *mant, const fvalue *v) {
    int bits = word_bits(v->sig);
    *top = v->e + bits;
    *mant = v->sig << (64 - bits);
}

/* r = m**16, through t. */
static int power16(bigint *r, bigint *t, uint64_t m) {
    int rc = big_set_word(r, m, 0);
    for (int k = 0; !rc && k < 4; k++)
        if (!(rc = big_mul(t, r, r))) rc = big_copy(r, t);
    return rc;
}

/* floor(log2(X / Y)) for positive integers. */
static int floor_log2_ratio(int64_t *r, const bigint *X, const bigint *Y, bigint *t) {
    int64_t k = big_bits(X) - big_bits(Y);
    int rc = k >= 0 ? big_shl(t, Y, k) : big_shl(t, X, -k);
    if (!rc) *r = (k >= 0 ? big_cmp_mag(X, t) < 0 : big_cmp_mag(t, Y) < 0) ? k - 1 : k;
    return rc;
}

static int key_order(const void *x, const void *y) {
    const slot_key *a = x, *b = y;
    if (a->slot != b->slot) return a->slot > b->slot ? -1 : 1; /* the smallest slot (the highest index) first */
    if (a->neg != b->neg) return a->neg ? 1 : -1;               /* positive values, then negative ones */
    if (a->top != b->top) return a->top < b->top ? -1 : 1;      /* increasing magnitude */
    if (a->mant != b->mant) return a->mant < b->mant ? -1 : 1;
    return a->q < b->q ? -1 : a->q > b->q ? 1 : 0;              /* equal magnitudes in index order */
}

/* Element (i, j)'s rounded products in the accumulator x, the order of the finite nonzero ones (by
   slot, smallest slot first, each slot's positive and negative values in increasing magnitude,
   alternating; slots16_run), and in *kind and *neg the IEEE sum of those that are not finite
   (VALUE_FINITE where there is none; NaN where a NaN or both infinities occur), which joins the
   slots' sum last. */
int slots16_element(slots16 *x, int *kind, int *neg, const femu *acc, const operand *a, size_t i, const operand *b, size_t j) {
    size_t k = a->cols;
    int64_t p = acc->p;
    int nan = 0, plus = 0, minus = 0;
    for (size_t q = 0; q < k; q++) {
        int ka, kb, na, nb;
        uint64_t sa, sb;
        int64_t ea, eb;
        fvalue fa, fb;
        element_decode_slot(&ka, &na, &sa, &ea, a, i * k + q);
        element_decode_slot(&kb, &nb, &sb, &eb, b, j * k + q);
        femu_read(&fa, acc, ka, na, sa, ea);
        femu_read(&fb, acc, kb, nb, sb, eb);
        femu_mul(&x->rounded[q], acc, &fa, &fb);
        const fvalue *v = &x->rounded[q];
        if (v->kind == VALUE_NAN) nan = 1;
        if (v->kind == VALUE_INFINITE) *(v->neg ? &minus : &plus) = 1;
    }
    *kind = nan || (plus && minus) ? VALUE_NAN : plus || minus ? VALUE_INFINITE : VALUE_FINITE;
    *neg = *kind == VALUE_INFINITE && minus;
    x->n = x->used = 0;
    /* the largest finite rounded magnitude */
    int64_t top_l = 0;
    uint64_t mant_l = 0;
    int any = 0;
    for (size_t q = 0; q < k; q++) {
        const fvalue *v = &x->rounded[q];
        if (v->kind != VALUE_FINITE || !v->sig) continue;
        int64_t top;
        uint64_t mant;
        normalize(&top, &mant, v);
        if (!any || top > top_l || (top == top_l && mant > mant_l)) {
            top_l = top;
            mant_l = mant;
        }
        any = 1;
    }
    if (!any) return 0;
    /* Each term's slot: min(15, floor(log2((L / |t|)**16)) // p), with D the binades between the
       normalized magnitudes, 16 D + g for g = floor(log2((mant_L / mant)**16)) in [-16, 15]; g is
       needed only where D leaves two slots open. */
    slot_key *keys = x->keys;
    int have_l = 0, rc = 0;
    for (size_t q = 0; !rc && q < k; q++) {
        const fvalue *v = &x->rounded[q];
        if (v->kind != VALUE_FINITE || !v->sig) continue; /* an exact zero adds nothing; the non-finite join last */
        slot_key *key = &keys[x->n++];
        key->q = q;
        key->neg = v->neg;
        normalize(&key->top, &key->mant, v);
        int64_t d = top_l - key->top, lo = 16 * d - 16, hi = 16 * d + 15;
        if (lo < 0) lo = 0;
        int64_t s_lo = lo / p, s_hi = hi / p;
        if (s_lo > SLOTS16 - 1) s_lo = SLOTS16 - 1;
        if (s_hi > SLOTS16 - 1) s_hi = SLOTS16 - 1;
        if (s_lo == s_hi) {
            key->slot = (int)s_lo;
            continue;
        }
        if (!have_l && !(rc = power16(&x->big[0], &x->big[2], mant_l))) have_l = 1;
        int64_t g;
        if (!rc && !(rc = power16(&x->big[1], &x->big[2], key->mant)) && !(rc = floor_log2_ratio(&g, &x->big[0], &x->big[1], &x->big[3]))) {
            int64_t s = (16 * d + g) / p;
            key->slot = (int)(s > SLOTS16 - 1 ? SLOTS16 - 1 : s);
        }
    }
    if (rc) return rc;
    qsort(keys, x->n, sizeof *keys, key_order);
    /* per slot: positive and negative values alternating, then the rest of the longer list */
    size_t at = 0, filled = 0;
    while (at < x->n) {
        size_t start = at, mid, end;
        while (at < x->n && keys[at].slot == keys[start].slot && !keys[at].neg) at++;
        mid = at;
        while (at < x->n && keys[at].slot == keys[start].slot) at++;
        end = at;
        size_t np = mid - start, nn = end - mid, pairs = np < nn ? np : nn;
        for (size_t t = 0; t < pairs; t++) {
            x->order[filled++] = keys[start + t].q;
            x->order[filled++] = keys[mid + t].q;
        }
        for (size_t t = start + pairs; t < mid; t++) x->order[filled++] = keys[t].q;
        for (size_t t = mid + pairs; t < end; t++) x->order[filled++] = keys[t].q;
        x->ends[x->used++] = filled;
    }
    return 0;
}

/* The sum of the finite products slots16_element ordered: each slot's values added in order, then
   the slot sums from the smallest slot up, each addition rounded in acc; +0 when there is none. */
void slots16_run(fvalue *r, const slots16 *x, const femu *acc) {
    int started = 0;
    size_t from = 0;
    for (size_t s = 0; s < x->used; s++) {
        fvalue part = x->rounded[x->order[from]];
        for (size_t t = from + 1; t < x->ends[s]; t++) femu_add(&part, acc, &part, &x->rounded[x->order[t]]);
        if (started)
            femu_add(r, acc, r, &part);
        else
            *r = part;
        started = 1;
        from = x->ends[s];
    }
    if (!started) {
        r->kind = VALUE_FINITE;
        r->neg = 0;
        r->sig = 0;
        r->e = 0;
    }
}

/* acc += |x|. */
static int add_abs(bigint *acc, const bigint *x) {
    bigint v = *x; /* a read-only view of |x| */
    v.neg = 0;
    return big_add(acc, acc, &v);
}

/* largest = max(largest, |x|). */
static int keep_largest(bigint *largest, const bigint *x) {
    if (big_cmp_mag(x, largest) <= 0) return 0;
    int rc = big_copy(largest, x);
    largest->neg = 0;
    return rc;
}

/* The exact quantities of Lemma S over the tree slots16_element ordered, all times 2**e: the sum s
   and mass of every exact product, N = 2 (mass + the magnitudes of the slots' running sums after
   their first value and of the running total after the first slot), the largest magnitude among
   the products and those sums, and the tree's depth (ops/matmul.py's MatmulSlots16._bounds). */
int slots16_tree(bigint *s, bigint *mass, bigint *N, bigint *largest, int64_t *e, int64_t *depth, slots16 *x, const operand *a, size_t i,
                 const operand *b, size_t j) {
    size_t k = a->cols;
    int rc = 0, any = 0;
    int64_t low = 0;
    /* each exact product sig_a sig_b 2**(e_a + e_b), then all at the lowest exponent */
    int64_t *exps = (int64_t *)(void *)x->keys; /* the keys are done with: k exponents fit their space */
    for (size_t q = 0; !rc && q < k; q++) {
        int ka, kb, na, nb;
        uint64_t sa, sb, hi, lo;
        int64_t ea, eb;
        element_decode_slot(&ka, &na, &sa, &ea, a, i * k + q);
        element_decode_slot(&kb, &nb, &sb, &eb, b, j * k + q);
        lo = word_mul(&hi, sa, sb);
        bigint *t = &x->exact[q];
        if ((rc = big_reserve(t, 2))) break;
        t->w[0] = lo;
        t->w[1] = hi;
        t->n = 2;
        t->neg = 0;
        big_trim(t);
        t->neg = (na != nb) && t->n;
        exps[q] = ea + eb;
        if (t->n && (!any || exps[q] < low)) low = exps[q];
        if (t->n) any = 1;
    }
    *e = low;
    s->n = mass->n = N->n = largest->n = 0;
    s->neg = mass->neg = N->neg = largest->neg = 0;
    for (size_t q = 0; !rc && q < k; q++) {
        bigint *t = &x->exact[q];
        if (!t->n) continue;
        if (!(rc = big_shl(t, t, exps[q] - low)) && !(rc = big_add(s, s, t)) && !(rc = add_abs(mass, t))) rc = keep_largest(largest, t);
    }
    /* the nodes: each slot's running sums after its first value, and the total after the first slot */
    bigint *part = &x->big[0], *total = &x->big[1];
    total->n = 0;
    total->neg = 0;
    *depth = 0;
    size_t from = 0;
    for (size_t pos = 0; !rc && pos < x->used; pos++) {
        part->n = 0;
        part->neg = 0;
        for (size_t t = from; !rc && t < x->ends[pos]; t++) {
            if ((rc = big_add(part, part, &x->exact[x->order[t]]))) break;
            if (t > from && !(rc = add_abs(N, part))) rc = keep_largest(largest, part);
        }
        if (!rc && pos) {
            if (!(rc = big_add(total, total, part)) && !(rc = add_abs(N, total))) rc = keep_largest(largest, total);
        } else if (!rc) {
            rc = big_copy(total, part);
        }
        int64_t size = (int64_t)(x->ends[pos] - from), d = size - 1 + (int64_t)x->used - (pos > 1 ? (int64_t)pos : 1);
        if (d > *depth) *depth = d;
        from = x->ends[pos];
    }
    /* N = 2 (mass + nodes) */
    if (!rc && !(rc = big_add(N, N, mass))) rc = big_shl(N, N, 1);
    return rc;
}
