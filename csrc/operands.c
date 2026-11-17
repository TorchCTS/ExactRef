/* Earlier results as operands (internal): the radius within which an implementation may have
   stored each value, what it may have stored that is not finite, whether an output is one a
   correct implementation can give; and which formats hold every value of others. */
#include <string.h>

#include "exactref.h"
#include "words.h"

/* r = the radius of a part with value x stored within its tolerance r: with emin (a flush to zero
   below 2**emin), a part whose box [x - r, x + r] reaches below 2**emin may be read as zero, so r
   also covers |x| there. */
int part_radius(value *r, const value *x, const int64_t *emin) {
    if (!emin || r->kind != VALUE_FINITE || !r->v.m.n || x->kind != VALUE_FINITE) return 0;
    dyad a = x->v, t, tiny; /* a: a read-only view of |x| */
    a.m.neg = 0;
    dyad_init(&t);
    dyad_init(&tiny);
    int rc = dyad_set_si(&tiny, 1, *emin);
    if (!rc) rc = dyad_add(&t, &tiny, &r->v, INT64_C(1) << 40, 1); /* exact: far more bits than either has */
    if (!rc && dyad_cmp(&a, &t) < 0 && dyad_cmp(&a, &r->v) > 0) rc = dyad_copy(&r->v, &a);
    dyad_free(&t);
    dyad_free(&tiny);
    return rc;
}

static int is_finite(const value *x) { return x->kind == VALUE_FINITE; }

/* A part's marks (result.py's Bounds.marks): its flags, with FLAG_ANY where the reference v
   is not finite but finite values may have been stored (low is finite), or where the tolerance is
   infinite (nothing bounds the stored value). */
int part_marks(int flags, const value *v, const value *low, const value *tol) {
    int unbounded = tol->kind == VALUE_INFINITE && !tol->neg;
    return flags | ((!is_finite(v) && is_finite(low)) || unbounded ? FLAG_ANY : 0);
}

/* Whether a correct implementation can output the part o (Bounds.covers): a finite value within
   [lo, hi] (an empty range when lo is NaN), or a non-finite value its flag allows. */
int part_covers(const value *o, const value *lo, const value *hi, int flags) {
    if (o->kind == VALUE_NAN) return !!(flags & FLAG_NAN);
    if (o->kind == VALUE_INFINITE) return !!(flags & (o->neg ? FLAG_MINUS_INF : FLAG_PLUS_INF));
    return is_finite(lo) && is_finite(hi) && dyad_cmp(&lo->v, &o->v) <= 0 && dyad_cmp(&o->v, &hi->v) <= 0;
}

/* *holds = whether format f represents every value of format g, both integer or both float
   (ops/_operands.py's holds): an integer range holding the other's; a float format with infinities
   where g has them, as much precision, as wide a range at both ends, and as large a largest finite value. */
int format_holds(int *holds, const partfmt *f, const partfmt *g) {
    if (f->is_int) {
        bigint flo, fhi, glo, ghi;
        big_init(&flo);
        big_init(&fhi);
        big_init(&glo);
        big_init(&ghi);
        int rc = intfmt_range(&flo, &fhi, f->itemsize, f->is_signed);
        if (!rc) rc = intfmt_range(&glo, &ghi, g->itemsize, g->is_signed);
        dyad a, b;
        a.e = b.e = 0;
        a.m = flo;
        b.m = glo;
        int low = dyad_cmp(&a, &b) <= 0;
        a.m = fhi;
        b.m = ghi;
        *holds = !rc && low && dyad_cmp(&a, &b) >= 0;
        big_free(&flo);
        big_free(&fhi);
        big_free(&glo);
        big_free(&ghi);
        return rc;
    }
    const floatfmt *a = &f->f, *b = &g->f;
    *holds = !(b->special == SPECIAL_IEEE && a->special != SPECIAL_IEEE) && a->man_bits >= b->man_bits && a->emax >= b->emax &&
             a->emin - a->man_bits <= b->emin - b->man_bits;
    if (!*holds) return 0;
    /* the largest finite values too: with equal widths a format whose all-ones encoding is NaN (fn) reaches less than one
       without infinity (fnuz) */
    dyad ta, la, tb, lb;
    dyad *ds[] = {&ta, &la, &tb, &lb};
    for (int z = 0; z < 4; z++) dyad_init(ds[z]);
    int rc = floatfmt_threshold(&ta, &la, a);
    if (!rc) rc = floatfmt_threshold(&tb, &lb, b);
    *holds = !rc && dyad_cmp(&la, &lb) >= 0;
    for (int z = 0; z < 4; z++) dyad_free(ds[z]);
    return rc;
}

/* The widths of the binary format ops/_operands.py makes for exact scalars no standard format
   holds, values of at most bits significant bits between 2**low and 2**high: float64's or wider,
   a whole number of bytes. */
void binary_holding(int64_t *exp_bits, int64_t *man_bits, int64_t bits, int64_t low, int64_t high) {
    int64_t reach = (low < 0 ? -low : low) > (high < 0 ? -high : high) ? (low < 0 ? -low : low) : (high < 0 ? -high : high);
    int64_t e = 11;
    while (((int64_t)1 << (e - 1)) - 2 < reach + bits) e++;
    int64_t m = bits - 1 > 52 ? bits - 1 : 52;
    m += ((-(1 + e + m)) % 8 + 8) % 8;
    *exp_bits = e;
    *man_bits = m;
}

/* An earlier result's values and radii as integers at one exponent (radius_packed): word arithmetic on each element,
   every format at most 64 bits wide. */

/* The element at data in the given byte order, n <= 8 bytes. */
static uint64_t rpack_word(const unsigned char *data, size_t n, int big_endian) {
    uint64_t w = 0;
    for (size_t k = 0; k < n; k++) w |= (uint64_t)data[big_endian ? n - 1 - k : k] << (8 * k);
    return w;
}

/* Part i at data in p's format as (-1)**neg sig 2**e, sig 0 where it is not finite (kind) */
static void rpack_part(int *kind, int *neg, uint64_t *sig, int64_t *e, const partfmt *p, size_t n, const unsigned char *data, size_t i, int big_endian) {
    uint64_t w = rpack_word(data + i * n, n, big_endian);
    if (!p->is_int) {
        floatfmt_decode_word(kind, neg, sig, e, &p->f, w);
        return;
    }
    *kind = VALUE_FINITE;
    *e = 0;
    *neg = p->is_signed && w >> (8 * n - 1) & 1;
    /* a negative value's magnitude 2**(8 n) - w, in 64-bit arithmetic modulo 2**(8 n) */
    *sig = *neg ? (n == 8 ? (uint64_t)0 - w : ((uint64_t)1 << (8 * n)) - w) : w;
}

static int rpack_trailing(uint64_t x) { return word_bits(x & ((uint64_t)0 - x)) - 1; }

/* The top of a + b for a = sa 2**ea, b = sb 2**eb not both 0: the t with 2**(t - 1) <= a + b < 2**t. With a's top
   ha = ea + bits(sa) the larger, a + b < 2**(ha + 1), and a + b >= 2**ha just when sa + floor(b / 2**ea) >= 2**bits(sa)
   (sa and 2**bits(sa) are integers); floor(b / 2**ea) < 2**bits(sa) is a word, and b < 2**ea adds no carry. */
static int64_t rpack_sum_top(uint64_t sa, int64_t ea, uint64_t sb, int64_t eb) {
    if (!sa) return eb + word_bits(sb);
    if (!sb) return ea + word_bits(sa);
    int64_t ha = ea + word_bits(sa), hb = eb + word_bits(sb);
    if (hb > ha) {
        uint64_t s = sa;
        int64_t e = ea, h = ha;
        sa = sb, ea = eb, ha = hb;
        sb = s, eb = e, hb = h;
    }
    if (hb <= ea) return ha;
    int ba = word_bits(sa);
    uint64_t q = eb >= ea ? sb << (eb - ea) : sb >> (ea - eb); /* shifts below 64: sb < 2**(ha - eb), and hb > ea */
    uint64_t room = ba == 64 ? (uint64_t)0 - sa : ((uint64_t)1 << ba) - sa;
    return q >= room ? ha + 1 : ha;
}

/* Element i's value (-1)**neg sx 2**ex and radius sr 2**er, each 0 where not finite (*unb: the radius bounds nothing),
   the radius widened as part_radius widens it (t: scratch for the rare elements its widening may reach) */
static int rpack_element(int *neg, uint64_t *sx, int64_t *ex, uint64_t *sr, int64_t *er, int *unb, const rpack *x, size_t i, value t[2]) {
    int kx, kr, nr, rc = 0;
    rpack_part(&kr, &nr, sr, er, &x->f64, x->rsize, x->radii, i, x->radii_big);
    rpack_part(&kx, neg, sx, ex, &x->p, x->vsize, x->values, i, x->values_big);
    if (x->has_emin && kr == VALUE_FINITE && *sr && kx == VALUE_FINITE && *sx) {
        /* part_radius: |x| >= 2**(max(emin, top of r) + 1) is at least 2**emin + r and leaves r as it is */
        int64_t ta = *ex + word_bits(*sx), tr = *er + word_bits(*sr), m = x->emin > tr ? x->emin : tr;
        if (ta - 1 <= m) {
            t[0].kind = t[1].kind = VALUE_FINITE;
            t[0].neg = 0;
            t[1].neg = *neg;
            t[0].v.e = *er;
            t[1].v.e = *ex;
            if ((rc = big_set_word(&t[0].v.m, *sr, nr)) || (rc = big_set_word(&t[1].v.m, *sx, *neg)) || (rc = part_radius(&t[0], &t[1], &x->emin))) return rc;
            *sr = t[0].v.m.n ? t[0].v.m.w[0] : 0; /* r, or |x| where widened */
            *er = t[0].v.e;
        }
    }
    *unb = kr != VALUE_FINITE;
    if (kx != VALUE_FINITE) *sx = 0;
    if (*unb) *sr = 0;
    return rc;
}

void rpack_init(rpack *x) {
    memset(x, 0, sizeof *x);
}

/* Pass 1 over elements i0 .. i1 - 1: s->any whether a value or radius is nonzero, s->e0 the least exponent of their odd
   significands, s->top the largest top of |x| + r (INT64_MIN for none) */
int rpack_scan(rpack_part_scan *s, const rpack *x, size_t i0, size_t i1) {
    value t[2];
    dyad_init(&t[0].v);
    dyad_init(&t[1].v);
    int rc = 0, any = 0;
    int64_t e0 = 0, top = INT64_MIN;
    for (size_t i = i0; !rc && i < i1; i++) {
        int neg, unb;
        uint64_t sx, sr;
        int64_t ex, er;
        if ((rc = rpack_element(&neg, &sx, &ex, &sr, &er, &unb, x, i, t))) break;
        if (sx) {
            int64_t e = ex + rpack_trailing(sx);
            if (!any || e < e0) e0 = e;
            any = 1;
        }
        if (sr) {
            int64_t e = er + rpack_trailing(sr);
            if (!any || e < e0) e0 = e;
            any = 1;
        }
        if (sx || sr) {
            int64_t b = rpack_sum_top(sx, ex, sr, er);
            if (b > top) top = b;
        }
    }
    dyad_free(&t[0].v);
    dyad_free(&t[1].v);
    s->any = any;
    s->e0 = e0;
    s->top = top;
    return rc;
}

/* x->e0 from the passes over the parts: the least exponent, coarsened to top - 125 where every |x| + r must stay below
   2**125 units (the same whatever the parts, min and max being exact) */
void rpack_fold(rpack *x, const rpack_part_scan *parts, size_t n) {
    int any = 0;
    int64_t e0 = 0, top = INT64_MIN;
    for (size_t k = 0; k < n; k++) {
        if (!parts[k].any) continue;
        if (!any || parts[k].e0 < e0) e0 = parts[k].e0;
        if (parts[k].top > top) top = parts[k].top;
        any = 1;
    }
    if (any && top - e0 > 125) e0 = top - 125;
    x->e0 = e0;
}

/* s 2**k as a 128-bit word pair (lo, hi), for k >= 0 and s 2**k < 2**128 */
static void rpack_shl(uint64_t w[2], uint64_t s, int64_t k) {
    w[0] = k >= 64 ? 0 : s << k;
    w[1] = k >= 64 ? s << (k - 64) : k ? s >> (64 - k) : 0;
}

/* floor(s / 2**k) and whether bits were cut, k > 0 */
static uint64_t rpack_shr(int *cut, uint64_t s, int64_t k) {
    if (k >= 64) {
        *cut = s != 0;
        return 0;
    }
    *cut = (s & (((uint64_t)1 << k) - 1)) != 0;
    return s >> k;
}

static void rpack_store(unsigned char *out, const uint64_t w[2]) {
    for (int b = 0; b < 8; b++) {
        out[b] = (unsigned char)(w[0] >> (8 * b));
        out[8 + b] = (unsigned char)(w[1] >> (8 * b));
    }
}

/* Pass 2 over elements i0 .. i1 - 1 at x->e0: x' = floor(x 2**-e0), r' = ceil(r 2**-e0) plus one where x' moved, as
   128-bit little-endian integers (two's complement, unsigned), and whether the radius bounds nothing, each at its place
   from x->base */
int rpack_fill(const rpack *x, size_t i0, size_t i1) {
    value t[2];
    dyad_init(&t[0].v);
    dyad_init(&t[1].v);
    int rc = 0;
    for (size_t i = i0; !rc && i < i1; i++) {
        int neg, unb, moved = 0, cut;
        uint64_t sx, sr, w[2];
        int64_t ex, er;
        if ((rc = rpack_element(&neg, &sx, &ex, &sr, &er, &unb, x, i, t))) break;
        if (!sx) {
            w[0] = w[1] = 0;
        } else if (ex >= x->e0) {
            rpack_shl(w, sx, ex - x->e0);
        } else {
            w[0] = rpack_shr(&moved, sx, x->e0 - ex);
            w[1] = 0;
            w[0] += (uint64_t)(neg && moved); /* the floor of a negative value: away from zero; below 2**64 */
        }
        if (neg) { /* -w */
            w[0] = ~w[0] + 1;
            w[1] = ~w[1] + (w[0] == 0);
        }
        rpack_store(x->ov + 16 * (i - x->base), w);
        if (!sr) {
            w[0] = w[1] = 0;
        } else if (er >= x->e0) {
            rpack_shl(w, sr, er - x->e0);
        } else {
            w[0] = rpack_shr(&cut, sr, x->e0 - er);
            w[0] += (uint64_t)cut;
            w[1] = 0;
        }
        if (moved) {
            w[0]++;
            w[1] += w[0] == 0;
        }
        rpack_store(x->orad + 16 * (i - x->base), w);
        x->ou[i - x->base] = (unsigned char)unb;
    }
    dyad_free(&t[0].v);
    dyad_free(&t[1].v);
    return rc;
}
