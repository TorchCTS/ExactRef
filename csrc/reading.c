/* Reading arrays (internal): an array's elements, stored in a format and a byte order, as
   integers at one shared exponent, and the parts of elements as values. */
#include <stdlib.h>

#include "exactref.h"
#include "words.h"

/* b = the n bytes at s as an unsigned integer in the given byte order. */
int bytes_load(bigint *b, const unsigned char *s, size_t n, int big_endian) {
    size_t words = (n + 7) / 8;
    if (big_reserve(b, words)) return EXACTREF_NOMEM;
    for (size_t i = 0; i < words; i++) b->w[i] = 0;
    for (size_t k = 0; k < n; k++) b->w[k / 8] |= (uint64_t)s[big_endian ? n - 1 - k : k] << (8 * (k % 8));
    b->n = words;
    b->neg = 0;
    big_trim(b);
    return 0;
}

/* The n little-endian bytes at s = b, for 0 <= b < 2**(8 n). */
void bytes_store(unsigned char *s, size_t n, const bigint *b) {
    for (size_t k = 0; k < n; k++) s[k] = k / 8 < b->n ? (unsigned char)(b->w[k / 8] >> (8 * (k % 8))) : 0;
}

/* The bytes of one part in p's format. */
size_t part_size(const partfmt *p) { return p->is_int ? (size_t)p->itemsize : (size_t)(1 + p->f.exp_bits + p->f.man_bits) / 8; }

/* x = part i of the parts at data, in p's format and the given byte order. */
int part_read(value *x, const partfmt *p, const unsigned char *data, size_t i, int big_endian) {
    size_t size = part_size(p);
    bigint bits;
    big_init(&bits);
    int rc = bytes_load(&bits, data + i * size, size, big_endian);
    x->v.e = 0;
    if (!rc && p->is_int) {
        x->kind = VALUE_FINITE;
        rc = intfmt_decode(&x->v.m, &bits, p->itemsize, p->is_signed);
        x->neg = x->v.m.neg;
    } else if (!rc) {
        rc = floatfmt_decode(&x->kind, &x->neg, &x->v.m, &x->v.e, &p->f, &bits);
        x->v.m.neg = x->neg && x->v.m.n;
    }
    big_free(&bits);
    return rc;
}

/* Part i of the little-endian parts at data = the encoding of x in p's format: rounded to nearest even with IEEE
   overflow for a float format, wrapped for an integer one (exact where p holds x). */
int part_write(unsigned char *data, size_t i, const partfmt *p, const value *x) {
    bigint bits, a = x->v.m; /* a read-only view of the magnitude */
    big_init(&bits);
    a.neg = 0;
    int fits = 1, rc = p->is_int ? intfmt_encode(&bits, &fits, &x->v.m, p->itemsize, p->is_signed, INT_OVERFLOW_WRAP)
                                 : floatfmt_encode(&bits, &p->f, x->kind, x->neg, &a, x->v.e, ROUND_NEAREST_EVEN, OVERFLOW_IEEE);
    if (!rc) bytes_store(data + i * part_size(p), part_size(p), &bits);
    big_free(&bits);
    return rc;
}

/* The read kernel: an operand's values for the product kernels. */

/* The element at data in the given byte order, n <= 8 bytes. */
static uint64_t word_load(const unsigned char *data, size_t n, int big_endian) {
    uint64_t w = 0;
    for (size_t k = 0; k < n; k++) w |= (uint64_t)data[big_endian ? n - 1 - k : k] << (8 * k);
    return w;
}

void operand_init(operand *x) {
    x->slots = x->slots_hi = x->flagged = x->negzero = x->max_sig = x->absent = NULL;
    x->summary = x->special_rows = x->flagged_rows = NULL;
    x->low = x->top = x->max_exp = NULL;
    x->rows = x->cols = x->blocks = x->flagged_count = 0;
    x->parts = 1;
    x->exponent = x->width = 0;
}

void operand_free(operand *x) {
    free(x->slots);
    free(x->slots_hi);
    free(x->flagged);
    free(x->negzero);
    free(x->summary);
    free(x->special_rows);
    free(x->flagged_rows);
    free(x->low);
    free(x->top);
    free(x->max_sig);
    free(x->max_exp);
    free(x->absent);
    operand_init(x);
}

/* An element's value from its bits (at most 64) in p's format: a float's kind, sign and sig 2**e,
   or an integer's sign and magnitude (e 0). */
void element_decode(int *kind, int *neg, uint64_t *sig, int64_t *e, const partfmt *p, uint64_t bits) {
    if (!p->is_int) {
        floatfmt_decode_word(kind, neg, sig, e, &p->f, bits);
        return;
    }
    int64_t width = 8 * p->itemsize;
    uint64_t v = width < 64 ? bits & (((uint64_t)1 << width) - 1) : bits;
    *kind = VALUE_FINITE;
    *e = 0;
    *neg = p->is_signed && (int)(v >> (width - 1) & 1);
    /* a negative value's magnitude: its two's complement in the width */
    *sig = *neg ? (width < 64 ? ((uint64_t)1 << width) - v : ~v + 1) : v;
}

/* Over an operand's values (read_operand's layout): the largest finite nonzero value's top bit, the least exponent of a
   lowest set bit, whether there is such a value, and whether a value is not finite. A read at main bits b flags a slot
   exactly where a value is not finite or low < top - b. */
void scan_operand(int64_t *top, int64_t *low, int *any, int *nonfinite_out, const partfmt *p, const unsigned char *data, int big_endian, size_t rows,
                  size_t cols, size_t row_step, size_t col_step, size_t inner, size_t outer_step) {
    size_t itemsize = part_size(p);
    int kind, neg, nonfinite = 0;
    int64_t e;
    uint64_t sig;
    *top = *low = 0;
    *any = 0;
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++) {
            element_decode(&kind, &neg, &sig, &e, p, word_load(data + ((i / inner) * outer_step + (i % inner) * row_step + j * col_step) * itemsize, itemsize, big_endian));
            if (kind != VALUE_FINITE) {
                nonfinite = 1;
                continue;
            }
            if (!sig) continue;
            int trailing = 0;
            while (!(sig >> trailing & 1)) trailing++;
            int64_t lo = e + trailing, t = e + word_bits(sig);
            if (!*any || t > *top) *top = t;
            if (!*any || lo < *low) *low = lo;
            *any = 1;
        }
    *nonfinite_out = nonfinite;
}

/* Whether a read at main bits b flags a slot of an operand with scan_operand's statistics: a value not finite, or one
   with a set bit below the main exponent top - b (read_operand_given's slots) */
int scan_flagged(int64_t top, int64_t low, int any, int nonfinite, int64_t main_bits) { return nonfinite || (any && low < top - main_bits); }

/* x = rows x cols values of the float format f (elements of at most 64 bits), element (i, j) at
   data + ((i / inner) outer_step + (i % inner) row_step + j col_step) itemsize in the given byte
   order: rows in groups of inner (an operand's batches), each group outer_step elements on. Every finite value that is a
   multiple of 2**E for the main exponent E is its slot's integer, below 2**main_bits in
   magnitude: E is the largest value's top bit less main_bits (0 when no value is finite and
   nonzero). Any other value, below the main range or not finite, is flagged: its slot holds its
   encoding, a bit of the bitmap marks it, and so does its row's summary byte for its block of
   READ_BLOCK terms; a row holding NaN or an infinity is marked special. main_bits is at most 126;
   above 62 each slot's integer takes two words (slots_hi its high one). */
int read_operand(operand *x, const partfmt *p, const unsigned char *data, int big_endian, size_t rows, size_t cols, size_t row_step,
                 size_t col_step, size_t inner, size_t outer_step, int64_t main_bits) {
    return read_operand_given(x, p, data, big_endian, rows, cols, row_step, col_step, inner, outer_step, main_bits, NULL);
}

/* x allocated for rows x cols (read_operand_given's arrays), its main exponent from the largest finite value's top bit, top,
   where any is set (0 without such a value); operand_fill reads its rows */
int operand_alloc(operand *x, const partfmt *p, size_t rows, size_t cols, int64_t main_bits, int any, int64_t top) {
    size_t n = rows * cols;
    operand_init(x);
    x->rows = rows;
    x->cols = cols;
    x->blocks = (cols + READ_BLOCK - 1) / READ_BLOCK;
    x->pf = *p;
    x->pf.f.nan[0] = x->pf.f.nan[1] = NULL;
    x->f = x->pf.f;
    x->slots = malloc((n ? n : 1) * sizeof *x->slots);
    x->slots_hi = main_bits > 62 ? calloc(n ? n : 1, sizeof *x->slots_hi) : NULL;
    x->flagged = calloc(n / 64 + 1, sizeof *x->flagged);
    x->negzero = calloc(n / 64 + 1, sizeof *x->negzero);
    x->summary = calloc(rows * x->blocks + 1, 1);
    x->special_rows = calloc(rows + 1, 1);
    x->flagged_rows = calloc(rows + 1, 1);
    x->low = calloc(rows + 1, sizeof *x->low);
    x->top = calloc(rows + 1, sizeof *x->top);
    x->max_sig = calloc(rows + 1, sizeof *x->max_sig);
    x->max_exp = calloc(rows + 1, sizeof *x->max_exp);
    if (!x->slots || (main_bits > 62 && !x->slots_hi) || !x->flagged || !x->negzero || !x->summary || !x->special_rows || !x->flagged_rows || !x->low || !x->top || !x->max_sig ||
        !x->max_exp) {
        operand_free(x);
        return EXACTREF_NOMEM;
    }
    x->exponent = any ? top - main_bits : 0;
    return 0;
}

/* The largest finite nonzero value's top bit over rows r0 .. r1 - 1 of an operand's layout: *any 0 where there is none */
void operand_rows_top(int *any, int64_t *top, const partfmt *p, const unsigned char *data, int big_endian, size_t cols, size_t row_step, size_t col_step,
                      size_t inner, size_t outer_step, size_t r0, size_t r1) {
    size_t itemsize = part_size(p);
    int kind, neg;
    int64_t e;
    uint64_t sig;
    *any = 0;
    *top = 0;
    for (size_t i = r0; i < r1; i++)
        for (size_t j = 0; j < cols; j++) {
            element_decode(&kind, &neg, &sig, &e, p, word_load(data + ((i / inner) * outer_step + (i % inner) * row_step + j * col_step) * itemsize, itemsize, big_endian));
            if (kind != VALUE_FINITE || !sig) continue;
            int64_t t = e + word_bits(sig);
            if (!*any || t > *top) *top = t;
            *any = 1;
        }
}

/* Rows r0 .. r1 - 1 of x (operand_alloc) read from data at their places in the whole operand's layout: their slots,
   bitmaps and row figures, *width the largest unflagged slot's bits and *flagged the flagged slots among them. Ranges
   whose first slots r0 cols are multiples of 64 share no bitmap word, so they may be read at once. */
void operand_fill(operand *x, const partfmt *p, const unsigned char *data, int big_endian, size_t row_step, size_t col_step, size_t inner, size_t outer_step,
                  size_t r0, size_t r1, int64_t *width, size_t *flagged) {
    size_t itemsize = part_size(p), cols = x->cols;
    int kind, neg;
    int64_t e;
    uint64_t sig;
    *width = 0;
    *flagged = 0;
    for (size_t i = r0; i < r1; i++)
        for (size_t j = 0, seen = 0; j < cols; j++) {
            size_t at = i * cols + j;
            uint64_t bits = word_load(data + ((i / inner) * outer_step + (i % inner) * row_step + j * col_step) * itemsize, itemsize, big_endian);
            element_decode(&kind, &neg, &sig, &e, p, bits);
            if (kind == VALUE_FINITE && sig) {
                int trailing = 0;
                while (!(sig >> trailing & 1)) trailing++;
                e += trailing;
                sig >>= trailing;
                int64_t t = e + word_bits(sig);
                if (!seen || e < x->low[i]) x->low[i] = e;
                if (!seen || t > x->top[i]) x->top[i] = t;
                /* the largest magnitude: the higher top bit, or at the same top the larger aligned significand */
                int64_t mt = x->max_exp[i] + word_bits(x->max_sig[i]);
                if (!seen || t > mt || (t == mt && (e <= x->max_exp[i] ? sig > x->max_sig[i] << (x->max_exp[i] - e) : sig << (e - x->max_exp[i]) > x->max_sig[i]))) {
                    x->max_sig[i] = sig;
                    x->max_exp[i] = e;
                }
                seen = 1;
            } else if (kind == VALUE_FINITE && neg) {
                x->negzero[at / 64] |= (uint64_t)1 << (at % 64);
            }
            if (kind == VALUE_FINITE && (!sig || e >= x->exponent)) {
                /* below 2**main_bits: e - E + the significand's bits is at most top - E; two words above 62 bits */
                int64_t shift = sig ? e - x->exponent : 0;
                uint64_t lo = shift < 64 ? sig << shift : 0, hi = !shift ? 0 : shift < 64 ? sig >> (64 - shift) : sig << (shift - 64);
                if (neg) {
                    lo = ~lo + 1;
                    hi = ~hi + (lo == 0);
                }
                x->slots[at] = lo;
                if (x->slots_hi) x->slots_hi[at] = hi;
                int64_t w = sig ? word_bits(sig) + shift : 0;
                if (w > *width) *width = w;
                continue;
            }
            x->slots[at] = bits;
            x->flagged[at / 64] |= (uint64_t)1 << (at % 64);
            x->summary[i * x->blocks + j / READ_BLOCK] = 1;
            x->flagged_rows[i] = 1;
            ++*flagged;
            if (kind != VALUE_FINITE) x->special_rows[i] = 1;
        }
}

/* read_operand, with given_top (when not NULL) for its largest finite value's top bit: a part of a larger operand read
   at that operand's main exponent, each slot as in the whole operand's read */
int read_operand_given(operand *x, const partfmt *p, const unsigned char *data, int big_endian, size_t rows, size_t cols, size_t row_step,
                       size_t col_step, size_t inner, size_t outer_step, int64_t main_bits, const int64_t *given_top) {
    int any = given_top != NULL;
    int64_t top = given_top ? *given_top : 0;
    if (!given_top) operand_rows_top(&any, &top, p, data, big_endian, cols, row_step, col_step, inner, outer_step, 0, rows);
    int rc = operand_alloc(x, p, rows, cols, main_bits, any, top);
    if (rc) return rc;
    operand_fill(x, p, data, big_endian, row_step, col_step, inner, outer_step, 0, rows, &x->width, &x->flagged_count);
    return 0;
}

/* Slot at of x negated into slot to of r (r's bitmaps cleared beforehand): an integer by its two's
   complement, a zero by its sign (where the format has a negative zero), a flagged value by its
   encoding's sign bit, NaN unchanged. */
static void negate_slot(operand *r, size_t to, const operand *x, size_t at) {
    uint64_t w = x->slots[at], bit = (uint64_t)1 << (to % 64);
    int flagged = (int)(x->flagged[at / 64] >> (at % 64) & 1), negzero = (int)(x->negzero[at / 64] >> (at % 64) & 1);
    if (flagged) {
        int kind, neg;
        uint64_t sig;
        int64_t e;
        floatfmt_decode_word(&kind, &neg, &sig, &e, &x->f, w);
        r->slots[to] = kind == VALUE_NAN ? w : w ^ (uint64_t)1 << (x->f.exp_bits + x->f.man_bits);
        r->flagged[to / 64] |= bit;
        if (r->slots_hi) r->slots_hi[to] = 0;
    } else {
        uint64_t hi = x->slots_hi ? x->slots_hi[at] : 0;
        r->slots[to] = ~w + 1;
        if (r->slots_hi) r->slots_hi[to] = ~hi + (w == 0);
        if (!w && !hi && !negzero && x->f.special != SPECIAL_FNUZ) r->negzero[to / 64] |= bit;
    }
}

static void copy_slot(operand *r, size_t to, const operand *x, size_t at) {
    uint64_t bit = (uint64_t)1 << (to % 64);
    r->slots[to] = x->slots[at];
    if (r->slots_hi) r->slots_hi[to] = x->slots_hi[at];
    if (x->flagged[at / 64] >> (at % 64) & 1) r->flagged[to / 64] |= bit;
    if (x->negzero[at / 64] >> (at % 64) & 1) r->negzero[to / 64] |= bit;
}

/* Whether slot at of x is a zero or a nonzero value (NaN and infinities are nonzero). */
static int slot_zero(const operand *x, size_t at) {
    if (x->flagged[at / 64] >> (at % 64) & 1) {
        int kind, neg;
        uint64_t sig;
        int64_t e;
        element_decode(&kind, &neg, &sig, &e, &x->pf, x->slots[at]);
        return kind == VALUE_FINITE && !sig;
    }
    return !x->slots[at] && !(x->slots_hi && x->slots_hi[at]);
}

/* A complex operand's parts as the two real operands of its output's parts (B transposed, b holding
   rows 2 j and 2 j + 1, the real and imaginary parts of column j): re's row j is
   (b_re, -b_im) for each term and im's (b_im, b_re), so that with A's parts interleaved each
   output part is a real product of 2 k terms in matmul's order (proofs/matmul.md). With
   coefficients, a zero part of a nonzero coefficient leaves its term out (marked absent). */
int operand_parts(operand *re, operand *im, const operand *b, int coefficients) {
    size_t rows = b->rows / 2, k = b->cols, cols = 2 * k, n = rows * cols;
    operand *both[2] = {re, im};
    operand_init(re);
    operand_init(im);
    for (int p = 0; p < 2; p++) {
        operand *x = both[p];
        x->rows = rows;
        x->cols = cols;
        x->blocks = (cols + READ_BLOCK - 1) / READ_BLOCK;
        x->exponent = b->exponent;
        x->width = b->width;
        x->pf = b->pf;
        x->f = b->f;
        x->flagged_count = 0;
        x->parts = 2;
        x->slots = malloc((n ? n : 1) * sizeof *x->slots);
        x->slots_hi = b->slots_hi ? malloc((n ? n : 1) * sizeof *x->slots_hi) : NULL;
        x->flagged = calloc(n / 64 + 1, sizeof *x->flagged);
        x->negzero = calloc(n / 64 + 1, sizeof *x->negzero);
        x->absent = coefficients ? calloc(n / 64 + 1, sizeof *x->absent) : NULL;
        x->summary = calloc(rows * x->blocks + 1, 1);
        x->special_rows = calloc(rows + 1, 1);
        x->flagged_rows = calloc(rows + 1, 1);
        x->low = calloc(rows + 1, sizeof *x->low);
        x->top = calloc(rows + 1, sizeof *x->top);
        x->max_sig = calloc(rows + 1, sizeof *x->max_sig);
        x->max_exp = calloc(rows + 1, sizeof *x->max_exp);
        if (!x->slots || (b->slots_hi && !x->slots_hi) || !x->flagged || !x->negzero || (coefficients && !x->absent) || !x->summary || !x->special_rows || !x->flagged_rows ||
            !x->low || !x->top || !x->max_sig || !x->max_exp) {
            operand_free(re);
            operand_free(im);
            return EXACTREF_NOMEM;
        }
    }
    for (size_t j = 0; j < rows; j++) {
        size_t r0 = 2 * j, r1 = 2 * j + 1;
        for (int p = 0; p < 2; p++) {
            operand *x = both[p];
            /* the row's facts are its two source rows' */
            x->special_rows[j] = b->special_rows[r0] | b->special_rows[r1];
            x->flagged_rows[j] = b->flagged_rows[r0] | b->flagged_rows[r1];
            int any0 = b->max_sig[r0] != 0, any1 = b->max_sig[r1] != 0;
            x->low[j] = any0 && any1 ? (b->low[r0] < b->low[r1] ? b->low[r0] : b->low[r1]) : any0 ? b->low[r0] : b->low[r1];
            x->top[j] = any0 && any1 ? (b->top[r0] > b->top[r1] ? b->top[r0] : b->top[r1]) : any0 ? b->top[r0] : b->top[r1];
            int64_t t0 = b->max_exp[r0] + word_bits(b->max_sig[r0]), t1 = b->max_exp[r1] + word_bits(b->max_sig[r1]);
            size_t larger = !any1 || (any0 && (t0 > t1 || (t0 == t1 && (b->max_exp[r0] <= b->max_exp[r1]
                                                                          ? b->max_sig[r0] >= b->max_sig[r1] << (b->max_exp[r1] - b->max_exp[r0])
                                                                          : b->max_sig[r0] << (b->max_exp[r0] - b->max_exp[r1]) >= b->max_sig[r1]))))
                                ? r0 : r1;
            x->max_sig[j] = b->max_sig[larger];
            x->max_exp[j] = b->max_exp[larger];
        }
        for (size_t q = 0; q < k; q++) {
            size_t at_re = r0 * k + q, at_im = r1 * k + q, to = j * cols + 2 * q;
            copy_slot(re, to, b, at_re);
            negate_slot(re, to + 1, b, at_im);
            copy_slot(im, to, b, at_im);
            copy_slot(im, to + 1, b, at_re);
            if (coefficients) {
                /* left out: a zero part of a nonzero coefficient */
                int zero_re = slot_zero(b, at_re), zero_im = slot_zero(b, at_im);
                if (zero_re && !zero_im) {
                    re->absent[to / 64] |= (uint64_t)1 << (to % 64);
                    im->absent[(to + 1) / 64] |= (uint64_t)1 << ((to + 1) % 64);
                }
                if (zero_im && !zero_re) {
                    re->absent[(to + 1) / 64] |= (uint64_t)1 << ((to + 1) % 64);
                    im->absent[to / 64] |= (uint64_t)1 << (to % 64);
                }
            }
        }
    }
    for (int p = 0; p < 2; p++) {
        operand *x = both[p];
        for (size_t at = 0; at < n; at++)
            if (x->flagged[at / 64] >> (at % 64) & 1) {
                x->summary[at / cols * x->blocks + at % cols / READ_BLOCK] = 1;
                x->flagged_count++;
            }
    }
    return 0;
}

/* Arrays' elements as Python scalars and as truth values (array.py's Array.scalars, all and any). */

/* The bits of part i of the parts at data, in p's format (at most 64 bits) and the given byte order. */
uint64_t part_bits(const partfmt *p, const unsigned char *data, size_t i, int big_endian) {
    size_t size = part_size(p);
    return word_load(data + i * size, size, big_endian);
}

/* *w = the float64 encoding of the value of bits (at most 64) in f's format, as formats.py's FloatFormat.to_scalar gives
   it: rounded to nearest even with IEEE overflow (floatfmt_encode in f64, float64's format), NaN the quiet NaN with the
   value's sign. held: whether f64 holds every value of f (format_holds); then a finite value sig 2**e is a float64
   value, its fields set without rounding: from 2**-1022 up its at most 53 bits with the top one implicit, below it the
   fraction sig 2**(e + 1074) of 2**-1074, as e is at least f's least exponent, which is at least float64's. */
int part_float64(uint64_t *w, const floatfmt *f, const floatfmt *f64, int held, uint64_t bits) {
    int kind, neg;
    uint64_t sig;
    int64_t e;
    floatfmt_decode_word(&kind, &neg, &sig, &e, f, bits);
    if (held && kind == VALUE_FINITE) {
        uint64_t sign = (uint64_t)(neg != 0) << 63;
        int n = sig ? word_bits(sig) : 0; /* at most 53: f has at most float64's precision */
        int64_t top = e + n - 1;          /* 2**top <= the magnitude < 2**(top + 1) */
        if (!sig) *w = sign;
        else if (top >= -1022) *w = sign | (uint64_t)(top + 1023) << 52 | (sig << (53 - n) & (((uint64_t)1 << 52) - 1));
        else *w = sign | sig << (e + 1074);
        return 0;
    }
    bigint a, out;
    big_init(&a);
    big_init(&out);
    int rc = big_set_word(&a, sig, 0);
    if (!rc) rc = floatfmt_encode(&out, f64, kind, neg, &a, e, ROUND_NEAREST_EVEN, OVERFLOW_IEEE);
    *w = out.n ? out.w[0] : 0;
    big_free(&a);
    big_free(&out);
    return rc;
}

/* Whether part bits (at most 64) of p's format is nonzero by its encoding: an integer's bits; a float's bits but its
   sign bit, as a zero of either sign is zero and NaN and infinities are nonzero; or in an fnuz format all of them, its
   negative-zero encoding being its NaN. */
static int part_nonzero(const partfmt *p, uint64_t bits) {
    if (p->is_int || p->f.special == SPECIAL_FNUZ) return bits != 0;
    return (bits & ~((uint64_t)1 << (p->f.exp_bits + p->f.man_bits))) != 0;
}

/* Whether every element (every) or some element of the n at data is nonzero, each of components parts in p's format
   and the given byte order: a complex element where either part is. */
int elements_nonzero(const partfmt *p, const unsigned char *data, size_t n, size_t components, int big_endian, int every) {
    for (size_t i = 0; i < n; i++) {
        int nonzero = 0;
        for (size_t c = 0; c < components && !nonzero; c++) nonzero = part_nonzero(p, part_bits(p, data, i * components + c, big_endian));
        if (nonzero != every) return nonzero;
    }
    return every;
}
