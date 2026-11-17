/* Declarations shared by exactref's C units (internal). */
#ifndef EXACTREF_H
#define EXACTREF_H

#include <stddef.h>
#include <stdint.h>

#include "words.h"

/* Error codes: 0 is success. */
#define EXACTREF_NOMEM 1
#define EXACTREF_OVERFLOW 2      /* an exponent beyond 2**62 in magnitude */
#define EXACTREF_ZERO_DIVISION 3
#define EXACTREF_DOMAIN 4        /* the square root of a negative number */
#define EXACTREF_LOG_DOMAIN 5    /* the logarithm of a number that is not positive */
#define EXACTREF_INTERNAL 6      /* a check that cannot fail failed */
#define EXACTREF_FORMAT_BITS 7   /* a format with fewer than 2 exponent bits or 1 mantissa bit */
#define EXACTREF_FORMAT_BYTES 8  /* a format whose elements are not whole bytes */
#define EXACTREF_FORMAT_RANGE 9  /* a format whose values reach beyond 2**+-EXACT_LIMIT */
#define EXACTREF_INT_OVERFLOW 10  /* an integer result outside its format, with the overflow mode 'error' */
#define EXACTREF_FORMAT_SIZE 11  /* a format whose elements are wider than 64 bits */
#define EXACTREF_NEAREST 12      /* an accumulator's rounding that is not to nearest (_rounding.nearest_accumulator) */

/* Exact values are stored with magnitudes between 2**-EXACT_LIMIT and 2**EXACT_LIMIT; every
   format lies inside that range. */
#define EXACT_LIMIT ((int64_t)1 << 20)

/* system.c */
int physical_cores(void);
size_t fenv_size(void);
int fenv_save(unsigned char *out);
int fenv_reset(void);
int fenv_restore(const unsigned char *in);

/* natural.c: little-endian arrays of 64-bit words */
uint64_t nat_add(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n, uint64_t carry);
uint64_t nat_sub(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n, uint64_t borrow);
uint64_t nat_add_word(uint64_t *a, uint64_t w, size_t n);
uint64_t nat_sub_word(uint64_t *a, uint64_t w, size_t n);
uint64_t nat_shr(uint64_t *r, const uint64_t *a, size_t n, int shift, uint64_t high);
uint64_t nat_shl(uint64_t *r, const uint64_t *a, size_t n, int shift);
uint64_t nat_mul_word(uint64_t *r, const uint64_t *a, size_t n, uint64_t b, uint64_t l);
uint64_t nat_addmul_word(uint64_t *r, const uint64_t *a, size_t n, uint64_t b);
uint64_t nat_submul_word(uint64_t *r, const uint64_t *a, size_t n, uint64_t b);
void nat_mul(uint64_t *r, const uint64_t *a, size_t na, const uint64_t *b, size_t nb);
uint64_t nat_div_word(uint64_t *q, const uint64_t *a, size_t n, uint64_t b);
int nat_cmp(const uint64_t *a, const uint64_t *b, size_t n);
void nat_divrem_norm(uint64_t *q, uint64_t *a, size_t na, const uint64_t *b, size_t nb);
uint64_t nat_sqrtrem(uint64_t *s, uint64_t *a, size_t n, uint64_t *tmp);

/* bigfloat.c: signed integers of any size, and dyadics m * 2**e */
#define BIG_INLINE 8
typedef struct {
    uint64_t *w; /* magnitude, little-endian: n words with w[n - 1] != 0, or n == 0 for zero */
    size_t n, cap;
    int neg;
    uint64_t small[BIG_INLINE]; /* w's storage up to BIG_INLINE words; never copy a bigint by value */
} bigint;

typedef struct { /* m 2**e; a zero is always 0 2**0, so no zero carries an exponent that arithmetic could move */
    bigint m;
    int64_t e;
} dyad;

void big_init(bigint *b);
void big_free(bigint *b);
int64_t big_bits(const bigint *b);
int big_reserve(bigint *b, size_t cap);
void big_trim(bigint *b);
int big_set_word(bigint *b, uint64_t v, int neg);
int big_set_si(bigint *b, int64_t v);
int big_add(bigint *r, const bigint *a, const bigint *b);
int big_mul(bigint *r, const bigint *a, const bigint *b);
int big_shl(bigint *r, const bigint *a, int64_t s);
int big_shr(bigint *r, const bigint *a, int64_t s, int up);
int big_div_word(bigint *r, const bigint *a, uint64_t d, int up);
int big_div(bigint *q, const bigint *a, const bigint *b, int up);
int big_copy(bigint *r, const bigint *a);
int big_divmod(bigint *q, int *exact, const bigint *a, const bigint *b);
int big_cmp_mag(const bigint *a, const bigint *b);
/* A read-only view of a's magnitude: shares a's words, and is never freed or grown. */
static inline bigint big_magnitude(const bigint *a) {
    bigint v = *a;
    v.neg = 0;
    return v;
}
int big_bit(const bigint *a, int64_t k);
int big_low_nonzero(const bigint *a, int64_t k);
int big_is_ones(const bigint *a, int64_t k);
int big_low(bigint *r, const bigint *a, int64_t k);
int big_set_bit(bigint *r, int64_t k);
int64_t big_trailing(const bigint *a);
void dyad_init(dyad *x);
void dyad_free(dyad *x);
int dyad_rnd(dyad *r, const dyad *x, int64_t p, int up);
int dyad_add(dyad *r, const dyad *x, const dyad *y, int64_t p, int up);
int dyad_mul(dyad *r, const dyad *x, const dyad *y, int64_t p, int up);
int dyad_times(dyad *r, const dyad *x, int64_t count, int64_t w);
int dyad_add_exact(dyad *r, const dyad *x, const dyad *y);
int dyad_mul_exact(dyad *r, const dyad *x, const dyad *y);
int dyad_div(dyad *r, const dyad *x, const dyad *y, int64_t p, int up);
int dyad_sqrt(dyad *r, const dyad *x, int64_t p, int up);
int dyad_power(dyad *r, const dyad *x, int64_t n, int64_t p, int up);
int dyad_to_fixed(bigint *r, const dyad *x, int64_t bits, int up);
int dyad_copy(dyad *r, const dyad *x);
int dyad_from_big(dyad *r, const bigint *m, int64_t e);
/* *v = a read-only view of m 2**e, owning no words (never freed, used while m is); a zero m gives 0 2**0 */
static inline void dyad_view(dyad *v, const bigint *m, int64_t e) {
    v->m = *m;
    v->e = m->n ? e : 0;
}
int dyad_shift(dyad *x, int64_t s);
int dyad_neg(dyad *r, const dyad *x);
int dyad_abs(dyad *r, const dyad *x);
int dyad_set_si(dyad *r, int64_t v, int64_t e);
int dyad_sign(const dyad *x);
int dyad_cmp(const dyad *x, const dyad *y);
int dyad_ratio(dyad *r, int64_t num, int64_t den, int64_t p, int up);
int64_t top_bit(const dyad *x);
const dyad *dmax(const dyad *x, const dyad *y);
const dyad *dmin(const dyad *x, const dyad *y);
int sub_dir(dyad *r, const dyad *x, const dyad *y, int64_t p, int up);
int ends_magnitude(dyad *r, const dyad *lo, const dyad *hi);
/* an interval of dyadics, each end rounded outward */
typedef struct {
    dyad lo, hi;
} ivl;
void ivl_init(ivl *x);
void ivl_free(ivl *x);
int ivl_copy(ivl *r, const ivl *x);
int ivl_point(ivl *r, const dyad *x);
int ivl_add(ivl *r, const ivl *x, const ivl *y, int64_t p);
int ivl_sub(ivl *r, const ivl *x, const ivl *y, int64_t p);
int ivl_mul(ivl *r, const ivl *x, const ivl *y, int64_t p);
int ivl_div(ivl *r, const ivl *x, const ivl *y, int64_t p);
int ivl_max(ivl *r, const ivl *x, const ivl *y);
int ivl_abs(ivl *r, const ivl *x);
int widen_by(ivl *x, const dyad *r, int64_t w);
/* two's-complement integers of n words */
void mw_neg(uint64_t *x, int n);
int mw_zero(const uint64_t *x, int n);
void mw_extend(uint64_t *r, int n, const uint64_t *x, int m);
void mw_add(uint64_t *x, const uint64_t *y, int n);
void mw_add_abs(uint64_t *x, const uint64_t *y, int n, uint64_t *tmp);
void mw_mul(uint64_t *r, int n, const uint64_t *a, int na, const uint64_t *b, int nb, uint64_t *tmp);
void mw_add_ends(uint64_t *st, int n, const uint64_t *lo, const uint64_t *hi, int na, const uint64_t *x, int nx, uint64_t *tmp);
void mw_shr(uint64_t *x, int n, int64_t s, int up);
int mw_from_dyad_need(uint64_t *r, int n, const dyad *x, int64_t u, int up, int64_t bits, int64_t *need);
int mw_from_dyad(uint64_t *r, int n, const dyad *x, int64_t u, int up);
int mw_to_big(bigint *r, const uint64_t *x, int n, uint64_t *tmp);
int mw_to_dyad(dyad *r, const uint64_t *x, int n, int64_t e, uint64_t *tmp);
int mw_words(int64_t bits);

/* matrices.c: interval matrices, midpoint-radius, and their exponential (matrix_exp.md, The enclosure) */
#define MAT_LIMB 53        /* bits per limb of a midpoint */
#define MAT_LIMBS_MAX 160  /* limbs: 8480 bits, past the reference's cap with its extra bits */
#define MEXP_TILE_ROWS 16  /* rows per tile of an exponential's phase */
#define MEXP_SMALL 16      /* matrices up to this size run whole exponentials per tile */
#define MEXP_GROUP 64      /* matrices per tile of those */
#define MEXP_BOX_SCALE 10 /* with radii, the scaled matrix's norm at most 2**-10 (matrix_exp.md, The algorithm) */
#define MAT_BANDS 64       /* a banded line's bands at most (matrix_exp.md, Products) */
typedef struct {           /* an upper bound m * 2**e: m in [2**31, 2**32), or m = 0 for 0 */
    uint64_t m;
    int64_t e;
} ub;

ub ub_make(uint64_t m, int64_t e);
ub ub_add(ub a, ub b);
ub ub_mul(ub a, ub b);
ub ub_div_word(ub a, uint64_t k);
ub ub_shift(ub a, int64_t s);
int ub_le(ub a, ub b);
int ub_to_dyad(dyad *r, ub a);
int ub_of_dyad(ub *r, const dyad *x);
ub ub_hypot(ub a, ub b);

typedef struct {
    size_t rows, cols;
    int L, parts;          /* limbs per midpoint; 1 real, 2 complex (the real, then the imaginary plane) */
    int64_t *d;            /* parts x rows x cols midpoints of L limbs */
    int64_t *e;            /* their exponents (of limb 0's unit) */
    ub *r;                 /* their radii */
    ub *rd;                /* complex: per entry its disk radius, bounding its error's modulus (NULL for real) */
} imat;

typedef struct {           /* a factor's rows (or columns), each on one exponent or banded (matrix_exp.md, Products) */
    size_t lines, len;
    int L, S, parts, Q;    /* S = L + 1 limbs per entry */
    int64_t *x;            /* lines x len x parts entries of S limbs */
    uint64_t *q, *rq;      /* |x| and the radius, Q bits within the entry's bands (exponents U and UR less LB band) */
    uint64_t *qm, *rd;     /* complex: per entry its modulus and disk radius likewise */
    unsigned char *bv, *br; /* per entry its value's and its radius's band */
    unsigned char *lost;   /* whether the entry was truncated */
    int64_t *E, *U, *UR;   /* per line: band 0's exponents */
    unsigned char *banded, *bvmax, *brmax; /* per line: banded, its largest bands */
} imat_lines;

void imat_clear(imat *M);
int imat_init(imat *M, size_t rows, size_t cols, int L, int parts);
void imat_free(imat *M);
int imat_set(imat *M, int part, size_t i, size_t j, const dyad *mid, const dyad *rad);
int imat_ends(dyad *lo, dyad *hi, const imat *M, int part, size_t i, size_t j);
int imat_ends_round(dyad *lo, dyad *hi, const imat *M, int part, size_t i, size_t j, int64_t p);
int imat_mid(dyad *mid, const imat *M, int part, size_t i, size_t j);
void imat_lines_clear(imat_lines *F);
int imat_lines_init(imat_lines *F, size_t lines, size_t len, int L, int parts);
void imat_lines_free(imat_lines *F);
void imat_lines_prepare(imat_lines *F, const imat *M, int by_cols, size_t l0, size_t l1);
int imat_product_fits(size_t n, int L, int parts);
void imat_product_rows(imat *C, const imat_lines *A, const imat_lines *B, size_t i0, size_t i1);

typedef struct {
    int kind, arg;
} mexp_op;

typedef struct {           /* exp of one interval matrix, as phases of row tiles */
    size_t n, words;
    int L, parts, bounded, past, N, q, S; /* bounded: the kernel runs it; past: its norm bound reaches 2**60 */
    int64_t s;
    ub tail;               /* the Taylor remainder's bound */
    imat *pw;              /* B**1 .. B**q */
    imat m[2];             /* the running result S (m[S]) and the other product */
    imat_lines rows, cols;
    int64_t *c, *ec;       /* 1/k!, k = 0 .. N, as interval scalars */
    ub *rc;
    uint64_t *reach;       /* the closure of the pattern, n rows of `words` bits */
    mexp_op *ops;
    size_t nops, cap;
} mexp;

void mexp_clear(mexp *x);
void mexp_free(mexp *x);
int mexp_init(mexp *x, const imat *A, int *bounded);
size_t mexp_tiles(const mexp *x, size_t k);
void mexp_run(mexp *x, size_t k, size_t t);
void mexp_done(mexp *x, size_t k);
void mexp_run_all(mexp *x);
const imat *mexp_result(const mexp *x);
size_t mexp_products(const mexp *x);

typedef struct {           /* exponentials of a batch of matrices */
    size_t count, n;
    int L, parts, whole, started;
    imat *in, *out;        /* per matrix: the input, then the result */
    unsigned char *ok;     /* per matrix: whether it has a result (else no bound) */
    mexp cur;              /* larger matrices: the one in progress (mi), at phase k */
    size_t mi, k;
} mexp_batch;

void mexp_batch_free(mexp_batch *b);
int mexp_batch_init(mexp_batch *b, size_t count, size_t n, int L, int parts);
int mexp_batch_next(mexp_batch *b, size_t *tiles);
int mexp_batch_tile(mexp_batch *b, size_t t);

/* rational.c: exact fractions, reduced, the denominator positive (adapted from imath) */
typedef struct {
    bigint num, den;
} rat;

int big_gcd(bigint *r, const bigint *a, const bigint *b);
void rat_init(rat *x);
void rat_free(rat *x);
int rat_set(rat *x, const bigint *num, const bigint *den);
int rat_copy(rat *r, const rat *a);
int rat_add(rat *r, const rat *a, const rat *b);
int rat_mul(rat *r, const rat *a, const rat *b);
int rat_neg(rat *r, const rat *a);
int rat_inv(rat *r, const rat *a);
int rat_div(rat *r, const rat *a, const rat *b);
int rat_set_dyad(rat *x, const bigint *m, int64_t e);
int rat_set_si(rat *x, int64_t v);
int rat_is_zero(const rat *a);

typedef struct {           /* sum c[k] x**k, k < len: len 0 for 0, else c[len - 1] != 0 */
    rat *c;
    size_t len, cap;
} rpoly;

void rpoly_init(rpoly *p);
void rpoly_free(rpoly *p);
int rpoly_copy(rpoly *r, const rpoly *a);
int rpoly_set_term(rpoly *p, const rat *c, size_t k);
int rpoly_add(rpoly *r, const rpoly *a, const rpoly *b);
int rpoly_mul(rpoly *r, const rpoly *a, const rpoly *b);

typedef struct {           /* sum c[k] x**k with integer coefficients, k < len: len 0 for 0, else c[len - 1] != 0 */
    bigint *c;
    size_t len, cap;
} ipoly;

void ipoly_init(ipoly *p);
void ipoly_free(ipoly *p);
void ipoly_trim(ipoly *p);
int ipoly_zero(ipoly *p, size_t len);
int ipoly_copy(ipoly *r, const ipoly *a);
int ipoly_set_term(ipoly *p, const bigint *c, size_t k);
int ipoly_add(ipoly *r, const ipoly *a, const ipoly *b);
int ipoly_sub(ipoly *r, const ipoly *a, const ipoly *b);
int ipoly_mul(ipoly *r, const ipoly *a, const ipoly *b);
int ipoly_scale(ipoly *r, const ipoly *a, const bigint *c);
int ipoly_divrem_monic(ipoly *q, ipoly *r, const ipoly *a, const ipoly *g);
int ipoly_squarefree(ipoly *factors, size_t *count, const ipoly *f);
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p);
uint64_t mod_pow(uint64_t a, uint64_t e, uint64_t p);
uint64_t big_mod(const bigint *a, uint64_t p);
int word_prime(uint64_t n);

/* functions.c: enclosures of ln 2, exp, log, pi, cos and sin (foundations.md, Lemmas Q, X, W, W3, Y, M and Z) */
int ln2_bounds(dyad *lo, dyad *hi, int64_t p);
int exp_bounds(dyad *lo, dyad *hi, bigint *n, const dyad *x, int64_t p);
int log_bounds(dyad *lo, dyad *hi, const bigint *m, const bigint *e, int64_t p);
int pi_bounds(dyad *lo, dyad *hi, int64_t p);
int cos_sin_bounds(dyad *c, dyad *s, const dyad *x, int64_t p);
int cos_sin_range(dyad *c, dyad *s, const dyad *lo, const dyad *hi, int64_t p);
int exp_end(dyad *r, const dyad *x, int64_t p, int up);
int exp_clamped(dyad *r, const dyad *x, int64_t p, int up);
int log_end(dyad *r, const dyad *x, int64_t p, int up);
void exp_small(uint64_t *m, int64_t *e, uint64_t r, int64_t s, int up);
void exp_small3(uint64_t *m, int64_t *e, const uint64_t *r, int64_t s, int up);
int exp_near_one(dyad *r, int *ok, const dyad *x, int64_t p, int up);

/* formats.c: the rules of formats.py's definitions. Codes follow formats.py's orders: rounding
   modes as formats.Rounding lists them, float overflow modes as FLOAT_OVERFLOW, integer ones as
   INT_OVERFLOW, special-value schemes as _SPECIALS, value kinds as values._KINDS. */
#define ROUND_NEAREST_EVEN 0
#define ROUND_NEAREST_AWAY 1
#define ROUND_TOWARD_ZERO 2
#define ROUND_TOWARD_POSITIVE 3
#define ROUND_TOWARD_NEGATIVE 4

#define OVERFLOW_IEEE 0
#define OVERFLOW_SATURATE 1
#define OVERFLOW_NAN 2

#define INT_OVERFLOW_ERROR 0
#define INT_OVERFLOW_WRAP 1
#define INT_OVERFLOW_SATURATE 2

#define SPECIAL_IEEE 0
#define SPECIAL_FN 1
#define SPECIAL_FNUZ 2

#define VALUE_FINITE 0
#define VALUE_INFINITE 1
#define VALUE_NAN 2

/* floatfmt_special's encodings */
#define ENCODING_ZERO 0
#define ENCODING_INFINITY 1
#define ENCODING_NAN 2
#define ENCODING_MAX_FINITE 3

typedef struct {
    int64_t exp_bits, man_bits, bias, emin, emax, max_exp_field;
    int special;
    const bigint *nan[2]; /* the NaN encodings by sign when the definition gives them, else NULL */
} floatfmt;

int floatfmt_set(floatfmt *f, int64_t exp_bits, int64_t man_bits, int special, const int64_t *bias, int whole_bytes);
int floatfmt_copy(floatfmt *f, bigint own[2], const floatfmt *g);
int floatfmt_special(bigint *bits, const floatfmt *f, int which, int neg);
int floatfmt_decode(int *kind, int *neg, bigint *sig, int64_t *exp, const floatfmt *f, const bigint *bits);
void floatfmt_decode_word(int *kind, int *neg, uint64_t *sig, int64_t *exp, const floatfmt *f, uint64_t bits);
int floatfmt_encode(bigint *bits, const floatfmt *f, int kind, int neg, const bigint *a, int64_t e, int rnd, int ovf);
int encode_dyad(bigint *bits, const floatfmt *f, const dyad *x, int rnd, int ovf);
int floatfmt_encode_ratio(bigint *bits, const floatfmt *f, int neg, const bigint *a, const bigint *b, int rnd, int ovf);
int floatfmt_threshold(dyad *theta, dyad *largest, const floatfmt *f);
int intfmt_range(bigint *lo, bigint *hi, int64_t itemsize, int is_signed);
int intfmt_decode(bigint *v, const bigint *bits, int64_t itemsize, int is_signed);
int intfmt_encode(bigint *bits, int *fits, const bigint *v, int64_t itemsize, int is_signed, int ovf);
int round_integer(bigint *n, const bigint *num, const bigint *den, int rnd);

/* reading.c: arrays' elements as integers, and the parts of arrays' elements as values */
typedef struct {
    int is_int, is_signed; /* an integer format of itemsize bytes, else the float format f */
    int64_t itemsize;
    floatfmt f;
} partfmt;

typedef struct {
    int kind, neg; /* a VALUE_ kind and the sign */
    dyad v;        /* a finite value */
} value;

size_t part_size(const partfmt *p);
int bytes_load(bigint *b, const unsigned char *s, size_t n, int big_endian);
void bytes_store(unsigned char *s, size_t n, const bigint *b);
int part_read(value *x, const partfmt *p, const unsigned char *data, size_t i, int big_endian);
int part_write(unsigned char *data, size_t i, const partfmt *p, const value *x);
/* arrays' elements as Python scalars' encodings and as truth values (array.py) */
uint64_t part_bits(const partfmt *p, const unsigned char *data, size_t i, int big_endian);
int part_float64(uint64_t *w, const floatfmt *f, const floatfmt *f64, int held, uint64_t bits);
int elements_nonzero(const partfmt *p, const unsigned char *data, size_t n, size_t components, int big_endian, int every);

/* The read kernel (reading.c): an operand's values as integers at one main exponent, values
   below the main range and non-finite ones flagged in place. */
#define READ_BLOCK 8 /* terms per summary block: the finest the product kernels use */
typedef struct {
    int64_t exponent, width;         /* the main exponent, and the largest unflagged slot's bits */
    size_t rows, cols, blocks;       /* blocks: READ_BLOCK-term blocks per row */
    uint64_t *slots;                 /* rows x cols: an integer (two's complement), or a flagged value's encoding */
    uint64_t *slots_hi;              /* with a main range above 62 bits: each integer's high word (NULL otherwise) */
    uint64_t *flagged;               /* a bit per slot */
    unsigned char *summary;          /* rows x blocks: 1 where the block holds a flagged slot */
    unsigned char *special_rows;     /* rows: 1 where the row holds NaN or an infinity */
    size_t flagged_count;
    partfmt pf;                      /* the values' format, an integer or a float one (its NaN encodings not kept) */
    floatfmt f;                      /* the float format (pf's) */
    unsigned char *flagged_rows;     /* rows: 1 where the row holds a flagged slot */
    int64_t *low, *top;              /* rows: over the row's finite nonzero values, the exponent of the lowest set bit and
                                        the least t with every magnitude below 2**t (both 0 when there is none) */
    uint64_t *negzero;               /* a bit per slot: a zero with the sign bit */
    uint64_t *max_sig;               /* rows: the largest finite magnitude, max_sig * 2**max_exp (max_sig odd; 0 when none) */
    int64_t *max_exp;
    uint64_t *absent;                /* a bit per slot of a coefficient operand: the coefficient rule leaves its term out (NULL: none) */
    int parts;                       /* 2 for an operand of a complex output part (operand_parts): its terms alternate two pairings of parts */
} operand;

void operand_init(operand *x);
void operand_free(operand *x);
int read_operand(operand *x, const partfmt *p, const unsigned char *data, int big_endian, size_t rows, size_t cols, size_t row_step,
                 size_t col_step, size_t inner, size_t outer_step, int64_t main_bits);
int operand_alloc(operand *x, const partfmt *p, size_t rows, size_t cols, int64_t main_bits, int any, int64_t top);
void operand_rows_top(int *any, int64_t *top, const partfmt *p, const unsigned char *data, int big_endian, size_t cols, size_t row_step, size_t col_step,
                      size_t inner, size_t outer_step, size_t r0, size_t r1);
void operand_fill(operand *x, const partfmt *p, const unsigned char *data, int big_endian, size_t row_step, size_t col_step, size_t inner, size_t outer_step,
                  size_t r0, size_t r1, int64_t *width, size_t *flagged);
int read_operand_given(operand *x, const partfmt *p, const unsigned char *data, int big_endian, size_t rows, size_t cols, size_t row_step,
                       size_t col_step, size_t inner, size_t outer_step, int64_t main_bits, const int64_t *given_top);
int scan_flagged(int64_t top, int64_t low, int any, int nonfinite, int64_t main_bits);
void scan_operand(int64_t *top, int64_t *low, int *any, int *nonfinite, const partfmt *p, const unsigned char *data, int big_endian, size_t rows,
                  size_t cols, size_t row_step, size_t col_step, size_t inner, size_t outer_step);
void element_decode(int *kind, int *neg, uint64_t *sig, int64_t *e, const partfmt *p, uint64_t bits);
int operand_parts(operand *re, operand *im, const operand *b, int coefficients);

/* operands.c: earlier results as operands, and formats holding others */
int part_radius(value *r, const value *x, const int64_t *emin);
int part_marks(int flags, const value *v, const value *low, const value *tol);
int part_covers(const value *o, const value *lo, const value *hi, int flags);
int format_holds(int *holds, const partfmt *f, const partfmt *g);
void binary_holding(int64_t *exp_bits, int64_t *man_bits, int64_t bits, int64_t low, int64_t high);

typedef struct {             /* an earlier result's values and radii packed at one exponent (radius_packed) */
    partfmt p, f64;          /* the values' and the radii's formats (decoded only: no NaN encodings) */
    const unsigned char *values, *radii;
    int values_big, radii_big, has_emin;
    size_t vsize, rsize;     /* their parts' bytes */
    int64_t emin;            /* with has_emin: a flush to zero below 2**emin (part_radius) */
    unsigned char *ov, *orad, *ou; /* per element from base: the value and the radius (16 bytes each), whether the radius bounds nothing */
    size_t base;             /* the first element the outputs hold */
    int64_t e0;              /* the exponent, from rpack_fold */
} rpack;

typedef struct {             /* pass 1 over a range of elements */
    int any;
    int64_t e0, top;
} rpack_part_scan;

void rpack_init(rpack *x);
int rpack_scan(rpack_part_scan *s, const rpack *x, size_t i0, size_t i1);
void rpack_fold(rpack *x, const rpack_part_scan *parts, size_t n);
int rpack_fill(const rpack *x, size_t i0, size_t i1);

/* entries.c: exact values in ExactArray's storage (exact.py), and deciding roundings */
#define ENTRY_WIDTH 24
#define ENTRY_BITS 128 /* the midpoint's significand bits */
typedef struct {
    unsigned char flag;              /* the kind and the radius code */
    unsigned char mid[ENTRY_WIDTH]; /* the midpoint: a 128-bit significand and the word 2 e + sign */
} entry;

int entry_value(entry *x, int kind, int neg, const bigint *a, int64_t e);
int entry_ratio(entry *x, int neg, const bigint *num, const bigint *den);
int entry_interval(entry *x, int *wide, const dyad *lo, const dyad *hi);
int entry_read(int *kind, int *neg, bigint *m, int64_t *e, bigint *rm, int64_t *re, int *out_of_range, const entry *x);
int entry_bounds(dyad *lo, dyad *hi, int *none, const entry *x);
int decide_rounding(bigint *bits, entry *x, int *none, const floatfmt *f, int rnd, int ovf, const dyad *lo, const dyad *hi, int last);
int switch_point(dyad *b, int *none, const floatfmt *f, int rnd, int ovf, const dyad *lo, const dyad *hi);
int offset_rounding(bigint *bits, entry *x, int *none, const floatfmt *f, int rnd, int ovf, const dyad *clo, const dyad *chi,
                    const dyad *olo, const dyad *ohi, int64_t p, int last);

/* errors.c: Lemma 1's error units and its bound at a boundary, and Lemma 2's growth terms */
int error_units(dyad *rel, dyad *abs, const floatfmt *f, int rnd, int64_t ulps_num, int64_t ulps_den, int flush);
int growth_terms(dyad *gamma, dyad *constant, dyad *largest, int *bounded, const floatfmt *f, int rnd, int64_t depth,
                 int64_t roundings, int64_t w, int flush);
int growth_up(dyad *g, const dyad *r, int64_t d, int64_t w);
/* How a sum's terms enter it (any_order_counts, sum_bounds_init, product_bounds_run; gemm's beta c, gemm_c_terms): each
   rounded to the format on its own; each a value of the format, so none is rounded (foundations.md, Lemma U); or each an
   exact value that the model does not round and the format need not hold, so that an addition that takes one has an
   absolute part (add's terms with a unit alpha, add.md; gemm's beta c with a unit beta, gemm.md) */
#define TERMS_ROUNDED 0
#define TERMS_HELD 1
#define TERMS_EXACT 2
void any_order_counts(int64_t *depth, int64_t *roundings, int64_t n, int flush, int exact_terms);
/* Lemma 1 at a boundary: a rounding or function boundary's setting (boundaries.RoundTo; absent: !active), its units,
   and its bound with the format's range */
typedef struct {
    int active, rnd, flush;          /* flush: the setting's own (Lemma 1's flushing case without the option) */
    floatfmt f;
    int64_t ulps_num, ulps_den;
} round_to;

typedef struct {
    dyad ulps;               /* c ulps, at Lemma 1's top bit */
    int64_t emin, mantissa;
    int active, flush;       /* the boundary is set; Lemma 1's flushing case applies (exp_ulps) */
} round_units;

typedef struct {             /* Lemma 1 at a boundary with its range (foundations.md, Lemma 1) */
    int active, has_threshold, flush; /* flush: a result below 2**emin may be zero */
    dyad ulps, largest, threshold;
    int64_t emin, mantissa;
} round_bound;

int units_of(dyad *rel, dyad *abs, dyad *largest, const round_to *r, int flush);
int rnd_of(round_units *t, const round_to *r, int flush);
int boundary_error(dyad *err, const round_units *b, const dyad *high, const dyad *low);
void lean_init(round_bound *l);
void lean_free(round_bound *l);
int lean_of(round_bound *l, const round_to *r, int flush);
int grid_end(dyad *x, int64_t emin, int64_t mantissa, int up);
int lean_ends(dyad *lo, dyad *hi, int *unbounded, const round_bound *b, int64_t w);
int lean_ivl(ivl *x, const round_bound *b, int64_t w, int *unb);
int lean_overflows(const round_bound *b, const dyad *mag);
int lean_error(dyad *err, const dyad *lo, const dyad *hi, const round_bound *b, int *unb);

/* finish.c: the final rounding of outputs (Lemma 3). The flags of an output's non-finite values,
   as Bounds stores them; FLAG_ANY (no bound) only in what may reach a rounding. */
#define FLAG_PLUS_INF 1
#define FLAG_MINUS_INF 2
#define FLAG_NAN 4
#define FLAG_ALL 7
#define FLAG_ANY 8

int final_bounds(bigint *tol, bigint *low, bigint *high, int *flags, const floatfmt *out, const floatfmt *f64, int rnd, int ovf,
                 const dyad *x_lo, const dyad *x_hi, const bigint *reference, int64_t w, int possible);
int candidate_bounds(bigint *tol, bigint *low, bigint *high, int *flags, const floatfmt *out, const floatfmt *f64, int rnd, int ovf,
                     size_t n, const int *kinds, const int *negs, const bigint *nums, const bigint *dens, const bigint *reference);
int rounding_interval(dyad *lo, dyad *hi, int *flags, const floatfmt *f, int64_t w);
int int_bounds(bigint *tol, bigint *low, bigint *high, const floatfmt *f64, int64_t itemsize, int is_signed, int ovf, const bigint *lo,
               const bigint *hi, const bigint *ref);
int widest_part(bigint *tol, value *lo, value *hi, int *empty, int *flags, const floatfmt *f64, const value *c, size_t n,
                const value *tols, const value *los, const value *his, const int *part_flags);

/* sums.c: bounds on sums accumulated in a float format (Lemmas S, O and A) */
typedef struct {
    int64_t n, emin;             /* emin: with flush, a factor below 2**emin reads as zero (the moved terms) */
    int exact_terms, bounded, has_infinity, flush;
    dyad half_r, c_s, g;         /* Lemma S: N half_r + c_s + moved g */
    dyad any_r, any_g, any_c;    /* B_any (Lemma O (c)): mass any_r + moved any_g + any_c */
    dyad side_g;                 /* a side's reach over any tree: its terms' magnitudes side_g + any_c (Lemma O (g)) */
    dyad unit_r, unit_a;         /* r and a */
    dyad gamma, constant;        /* Lemma A, Lemma 2's form (where bounded): gamma mass + constant + moved */
    dyad cap_r, cap_g, cap_c;    /* Lemma A, capped: mass cap_r + moved cap_g + cap_c */
    dyad theta;                  /* the overflow threshold */
} sum_bounds;

int sum_bounds_init(sum_bounds *b, const floatfmt *f, int flush, int64_t n, int64_t depth, int exact_terms, int64_t w);
void sum_bounds_free(sum_bounds *b);
int sum_bounds_normal(dyad *lo, dyad *hi, int *checked, int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *N,
                      const bigint *moved, int64_t e);
int sum_bounds_any_order(dyad *lo, dyad *hi, int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *moved, int64_t e);
int sum_bounds_tree(dyad *lo, dyad *hi, int *flags, const sum_bounds *b, const bigint *s, const bigint *mass, const bigint *N, const bigint *moved,
                    const bigint *deviation, const bigint *largest, int64_t e);

/* product.c: the product kernel's quantities per output element */
#define PRODUCT_SUM 0   /* the exact sum */
#define PRODUCT_MASS 1  /* and the magnitude sum and the moved magnitudes */
#define PRODUCT_NODES 2 /* and N: twice the mass plus the largest node sum of index order and lanes 2 and 4 */
typedef struct {
    int kind, neg;              /* VALUE_FINITE, or the IEEE value of a sum with a non-finite term */
    int64_t e;                  /* the quantities are integers times 2**e */
    bigint s, mass, N, moved;
    bigint dev;                 /* the inputs' radii's deviation (Results as inputs), counted in mass, N and moved */
    void *scratch;              /* the running sums of a named order */
    size_t scratch_size;
} product_sums;

typedef struct {
    size_t count;               /* a named order's internal nodes: node n covers terms lo[n] .. hi[n] - 1 */
    size_t *lo, *hi;
} product_order;

void product_sums_init(product_sums *r);
void product_sums_free(product_sums *r);
int product_order_sequential(product_order *o, size_t k);
int product_order_blocked(product_order *o, size_t k, size_t b);
int product_order_pairwise(product_order *o, size_t k);
int64_t product_order_depth(int kind, size_t k, size_t b);
void product_order_free(product_order *o);
int64_t product_main_bits(size_t terms, int wide);
/* Lemma L's lane combinations: the largest node sum, in two's-complement words and in dyadics */
void lane_nodes_words(uint64_t *r, const uint64_t *t, unsigned has, int L, size_t n, uint64_t *scratch);
int lane_nodes_dyad(dyad *r, const dyad *t, unsigned has, int L);
int operand_dyad(dyad *r, int *kind, const operand *x, size_t at);
void slot_value(int *kind, int *neg, uint64_t *sig, int64_t *e, const operand *x, size_t at);
int product_finite_terms(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, int flush, int64_t emin);
int product_element(product_sums *r, const operand *a, size_t i, const operand *b, size_t j, int what, int flush, int64_t emin,
                    const product_order *order);
int product_zero_sign(const operand *a, size_t i, const operand *b, size_t j, int rnd);
void element_decode_slot(int *kind, int *neg, uint64_t *sig, int64_t *e, const operand *x, size_t at);
int product_deviation(product_sums *r, int *unbounded, product_sums *t, const operand *a, const operand *ra, size_t i, const operand *b,
                      const operand *rb, size_t j, int reached);
void product_reach(int *unbounded, int *flags, const operand *a, const operand *ra, const unsigned char *marks, size_t i, const operand *b, size_t j);
int product_zero_meets(int *meets, const operand *a, const operand *ra, size_t i, const operand *b, const operand *rb, size_t j, int flush,
                       int64_t emin);
typedef struct {             /* alpha's application on a term's path (gemm.md, Overflow): |sigma| > 0, and the format's r and a; where */
    dyad sigma, r, a;        /* the part has two channels (two), alpha folded into an operand element whose parts are at most X */
    dyad fold_mul, fold_add; /* is at most fold_mul X + fold_add (gemm_fold) */
    int two;
} term_scale;
int product_terms(int *flags, dyad *plus, dyad *minus, dyad *mass, dyad *moved, size_t counts[2], const operand *a, const operand *ra, size_t i,
                  const operand *b, const operand *rb, size_t j, const dyad *theta, int flush, int64_t emin, const term_scale *scale);
int product_term_parts(dyad *pos, dyad *neg, unsigned char *leaf, const operand *a, const operand *ra, size_t i, const operand *b, const operand *rb,
                       size_t j, const dyad *theta);
/* product_few_terms' most values: each of the two terms rounded with either tie rule or unrounded (3 ways each), and
   each of those 9 sums rounded with either tie rule or unrounded (3 values each) */
#define PRODUCT_FEW_MAX (3 * 3 * 3)
int product_few_terms(size_t *count, value *out, const operand *a, size_t i, const operand *b, size_t j, const floatfmt *f, int exact_terms);

/* gemm.c: gemm's epilogue (proofs/gemm.md, Lemma G) */
typedef struct {
    dyad s; /* a nonzero coefficient */
    int q;  /* the source part it scales */
} gemm_channel;

typedef struct {
    int parts, alpha_real;   /* 1 for real values, 2 for complex; whether alpha's imaginary part is 0 */
    int n_p[2], n_c[2];      /* per output part, its channels of P and of c */
    gemm_channel p[2][2], c[2][2];
    dyad S[2];               /* per output part, the sum of its P channels' |sigma| */
} gemm_channels;

typedef struct {             /* alpha folded into an operand element whose parts are at most X: part p of the result is at most */
    dyad mul[2], add[2];     /* mul[p] X + add[p] in the compute format (gemm.md, Overflow; 0 where part p has no channel) */
} gemm_fold;

typedef struct {
    const gemm_channels *ch;
    int normal, flush, has_p, has_c, has_infinity, operand_ok;
    int64_t K, kappa, kappa_c, edges, c_factor, c_extra, inverse;
    dyad half_r, constant, g; /* B = X2 half_r + constant + moved g */
    dyad theta, tiny;         /* the overflow threshold, and 2**emin, below which a c part may flush */
} gemm_bounds;

int gemm_channels_set(gemm_channels *x, const dyad *alpha, const dyad *beta, int parts);
void gemm_channels_free(gemm_channels *x);
int gemm_c_terms(const gemm_channels *ch, int unit);
void gemm_fold_init(gemm_fold *x);
int gemm_fold_set(gemm_fold *x, const gemm_channels *ch, const floatfmt *f, int flush);
void gemm_fold_free(gemm_fold *x);
int gemm_fold_bound(dyad *r, const gemm_fold *x, int part, const dyad *X, int64_t w);
int gemm_bounds_init(gemm_bounds *g, const floatfmt *f, int flush, int normal, int64_t K, const gemm_channels *ch, const gemm_fold *fold, int c_terms,
                     const dyad *W, int64_t w);
void gemm_bounds_free(gemm_bounds *g);
int gemm_box(dyad *lo, dyad *hi, int *checked, int *flags, const gemm_bounds *g, int normal_flags, int part, const product_sums *const P[2],
             const dyad *const c[2], const dyad *const rc_c[2]);
void gemm_special(int *kind, int *neg, const gemm_channels *ch, int part, const product_sums *const P[2], const int c_kind[2], const int c_neg[2]);

/* mma.c: sums by tensor-core steps (proofs/foundations.md, Lemma T) */
int64_t mma_steps(int64_t N, int64_t split, int64_t K);
int mma_alpha(dyad *H, const dyad *alpha, int64_t w);
int mma_terms(dyad *G, dyad *H, int64_t N, int64_t f, int64_t e_f, int64_t split, const floatfmt *reduction, const dyad *sigma, int64_t K,
              int64_t extra, int64_t steps, int64_t w);
int mma_error(dyad *err, const dyad *G, const dyad *H, const bigint *mass, int64_t e);
int mma_interval(dyad *lo, dyad *hi, const bigint *s_lo, const bigint *s_hi, int64_t e, const dyad *err, const dyad *top);
int mma_overflow(int *flags, const dyad *pos, const dyad *neg, const dyad *err, const dyad *thresholds, int count, int has_infinity);
int mma_extent(int *sub, dyad *largest, const partfmt *f, const unsigned char *data, int big, size_t n, const partfmt *f64,
               const unsigned char *radii, int radii_big);
int mma_slack(dyad *sigma, int64_t emin, int sub_x, const dyad *max_x, int sub_y, const dyad *max_y);

/* product.c: the product bindings' kernels over a tile of output elements (module.c parses their arguments) */
/* The output elements of a product, in row-major order over its batches: element o of batch
   t = o / (m n) is (i, j) = (o / n mod m, o mod n), and reads row i of a's matrix for batch t and
   row j of b's (B's columns), each operand's matrix given by bases_load. */
typedef struct {
    const int64_t *bases[2];
    size_t counts[2], m, n;
} addressing;

/* gemm's epilogue in a product call: alpha, beta (pairs (m, e)), C's elements (row-major over the
   output, in C's part format and byte order; None when beta is 0) and how beta c enters the sum
   (gemm_c_terms). None for matmul. */
typedef struct {
    int on, c_big, c_terms, part; /* the output part this call computes */
    gemm_channels ch;             /* alpha's and beta's channels per output part */
    dyad alpha, beta;             /* real coefficients (gemm_mma), else alpha's and beta's real parts */
    const unsigned char *c, *c_radius, *c_marks; /* C's radii (float64, widened for flush) and marks per part, or NULL */
    partfmt cf, f64;
    bigint nan[4];
    gemm_fold fold; /* alpha's fold into an operand element in the compute format, set per tolerance call (gemm_fold_set) */
} epilogue;

void epilogue_free(epilogue *x);
void epilogue_needs(int need[2], int need_c[2], const epilogue *x);

/* Lemma T's terms in a product call (matmul_mma): G and H as pairs (m, e) and the reduction format
   (None for none). The products' stage overflows at float32's threshold or the reduction's. */
typedef struct {
    int on, has_infinity, count, epilogue;
    dyad G, H, top, thresholds[2];
    dyad G2, H2; /* gemm_mma: the epilogue's G' and H', H' with max(1, |alpha|) */
    floatfmt f32, reduction;
    bigint nan[2];
} mma_args;

void mma_args_free(mma_args *x);

int product_reference_run(unsigned char *enc, unsigned char *fl, unsigned char *mid, const operand *a, const operand *const bs[2],
                          const addressing *where, const epilogue *epi, const partfmt *pf, int rnd, int ovf, int elementwise, size_t o0, size_t count,
                          size_t *bad, bigint *bad_value);
int product_int_bounds_run(unsigned char *t, unsigned char *lo_out, unsigned char *hi_out, unsigned char *fl, const operand *a, const operand *b,
                           const operand *ra, const operand *rb, const addressing *where, const epilogue *epi, const partfmt *pf, const floatfmt *f64,
                           int ovf, const unsigned char *refs, size_t o0, size_t count);
int product_normal_run(unsigned char *enc, const operand *a, const operand *const bs[2], const addressing *where, const epilogue *epi,
                       const floatfmt *compute, int flush, int exact_terms, const floatfmt *f, int rnd, int ovf, size_t o0, size_t count);
int product_algorithm_run(unsigned char *enc, const operand *a, const operand *b, const addressing *where, const floatfmt *acc, int acc_rnd,
                          int method, size_t block, int fma, const floatfmt *f, int rnd, int ovf, size_t o0, size_t count);
int product_bounds_run(unsigned char *t, unsigned char *lo_out, unsigned char *hi_out, unsigned char *fl, const operand *a, const operand *const bs[2],
                       const operand *ra, const operand *const rbs[2], const addressing *where, epilogue *epi, const mma_args *mma, const floatfmt *compute,
                       const floatfmt *f, const floatfmt *f64, int flush, int normal, int exact_terms, int rnd, int ovf, int few, int order_kind,
                       int64_t block, int64_t w, const unsigned char *refs, const unsigned char *row_marks, const unsigned char *col_marks, int reach,
                       const unsigned char *reach_marks, size_t o0, size_t count);

/* emulate.c: a float format's arithmetic, emulated (Lemma E) */
typedef struct {
    int64_t p, emin, max_e;  /* precision, smallest normal exponent, and the largest value max_sig * 2**max_e */
    uint64_t max_sig;
    int rnd, flush, has_infinity;
} femu;

typedef struct {
    int kind, neg;           /* a VALUE_ kind and the sign */
    uint64_t sig;            /* a finite value sig * 2**e, sig below 2**p (an exact value femu_add_exact takes: any word) */
    int64_t e;
} fvalue;

int femu_set(femu *x, const floatfmt *f, int rnd, int flush);
void femu_round(fvalue *r, const femu *x, int neg, uint64_t hi, uint64_t lo, int64_t e, int sticky);
void femu_mul(fvalue *r, const femu *x, const fvalue *a, const fvalue *b);
void femu_add(fvalue *r, const femu *x, const fvalue *a, const fvalue *b);
void femu_add_exact(fvalue *r, const femu *x, const fvalue *a, const fvalue *b);
void femu_read(fvalue *r, const femu *x, int kind, int neg, uint64_t sig, int64_t e);
void femu_slot_sum(fvalue *r, const femu *x, fvalue *v, size_t n);
typedef struct {      /* a named algorithm's term */
    fvalue v;         /* the product's kind and sign by IEEE's rules; with rounded products, its rounding to the accumulator */
    uint64_t hi, lo;  /* a finite product exactly: (hi 2**64 + lo) 2**e */
    int64_t e;
} fterm;
int femu_run(fvalue *r, const femu *x, const floatfmt *f, int method, size_t block, const fterm *terms, int fma, size_t n, fvalue *scratch);
int product_normal(fvalue *r, fvalue *terms, const femu *x, const operand *a, size_t i, const operand *b, size_t j);

/* slots.c: the 16-slot algorithm (proofs/matmul.md, The 16-slot algorithm) */
#define SLOTS16 16
typedef struct {
    size_t k, n, used;     /* terms; the nonzero rounded ones in order; the nonempty slots */
    fvalue *rounded;       /* k: each product rounded in the accumulator */
    size_t *order, *ends;  /* the nonzero terms in addition order; per nonempty slot, one past its last position */
    void *keys;            /* scratch */
    bigint *exact;         /* k: the exact products, at the element's lowest exponent */
    bigint big[4];         /* scratch */
} slots16;

int slots16_init(slots16 *x, size_t k);
void slots16_free(slots16 *x);
int slots16_element(slots16 *x, int *kind, int *neg, const femu *acc, const operand *a, size_t i, const operand *b, size_t j);
void slots16_run(fvalue *r, const slots16 *x, const femu *acc);
int slots16_tree(bigint *s, bigint *mass, bigint *N, bigint *largest, int64_t *e, int64_t *depth, slots16 *x, const operand *a, size_t i,
                 const operand *b, size_t j);

/* attention.c: the attention forward's reference (proofs/scaled_dot_product_attention.md, Lemma D) */
#define ATT_F 191 /* the weights' fractional bits: 2**ATT_F is a weight of 1 */
#define ATT_B 16  /* bits per table digit */
#define ATT_ZW 4  /* words of a row sum */
#define ATT_ACC 5 /* words of a value sum, two's complement */
typedef struct {
    int digits;
    int64_t unit;      /* the tables are for exp(-c d 2**(B i)), c = |sigma| 2**unit (scores) or 2**unit (a float mask) */
    uint64_t *entries; /* digits x 2**ATT_B entries of 3 words */
} att_tables;

typedef struct {
    const operand *q, *k, *v;  /* Q's and K's rows of E terms, V's rows of Ev values: no flagged slot, one-word slots */
    const operand *vt;         /* V transposed, a head's Ev rows of S terms (the tolerance's value nodes), or NULL */
    const unsigned char *mask; /* a bool mask's bytes (1 keeps a key), or a float mask's (-inf masks a key), or NULL */
    const att_tables *mtables; /* a float mask's tables (Lemma D, Masks) */
    int mask_float, mask_big; /* a float mask's, and its bytes big-endian */
    floatfmt mask_fmt;
    size_t mask_size;
    const att_tables *tables;
    /* inputs that are earlier results (proofs/scaled_dot_product_attention.md, Results as inputs): Q's and K's radii as
       operands (NULL: exact); V's values and radii as 128-bit integers at 2**ve (signed, unsigned), with each element's
       no-bound flag; the float mask's radii as float64, read as the mask is; NULL where exact */
    const operand *rq, *rk;
    const unsigned char *vv, *vr, *vunb, *mrad;
    int64_t ve;
    int uncertain;
    size_t S, Ev, out_size, lse_size;
    int causal, sign;          /* key j kept for query i when j <= i; sigma's sign */
    bigint num, den;           /* sigma = num / den, or 1/sqrt(E) when den is 0 */
    int64_t E, u, p, cap;      /* scores are integers times 2**u; p: the first enclosures' bits; cap: refinement's last */
    floatfmt out, lse;
    int rnd, ovf, lrnd, lovf;
} att_call;

/* a key's part in a row (att_row.kinds): kept (a finite logit), masked (a -inf bias, whatever its score), removed by a
   -inf logit from its score, or a NaN or +inf logit (the row is NaN) */
enum { ATT_KEY_KEPT, ATT_KEY_MASKED, ATT_KEY_REMOVED, ATT_KEY_SPECIAL };
/* the backward's pair byte (att_gtol_group.kinds): the key's part (ATT_KEY_KIND bits) and marks, the pair's dS' may be NaN
   and its P' may be NaN (scaled_dot_product_attention_backward.md, Masked keys, cases 2 and 1) */
enum { ATT_KEY_KIND = 3, ATT_KEY_DS_NAN = 4, ATT_KEY_P_NAN = 8 };

typedef struct {
    size_t cap, ev;
    void *scores, *order, *sums; /* per key and per column scratch */
    size_t *keys;                /* the row's kept keys */
    dyad *bias;                  /* with a float mask, the kept keys' mask values */
    unsigned char *kinds;        /* per key: its part in the row (ATT_KEY_*) */
    unsigned char *scored;       /* per key: its score is finite */
    unsigned char *capped;       /* ... whose value is 133 or more below the row's largest (weight below 2**-191) */
    void *mdist;                 /* ... else its distance from the largest, in units of the mask's tables */
    dyad btop;                   /* the row's largest kept mask value */
    uint64_t *acc;
} att_row;

void att_tables_init(att_tables *t);
void att_tables_free(att_tables *t);
int att_tables_build(att_tables *t, const bigint *num, const bigint *den, int64_t E, int64_t u, int digits);
int att_mask_read(int *kind, int *neg, uint64_t *sig, int64_t *e, const att_call *a, size_t at);
int att_digits_mul(uint64_t p[3], int *z, w128 d, const att_tables *T);
/* the exponentials of computed shifts from Lemma D's tables (attention_tol.c) */
int abs_sigma(dyad *lo, dyad *hi, const att_call *a, int64_t p);
int table_exp(dyad *r, int *done, const dyad *y, w128 D, const att_call *a, const dyad *sig, const dyad *c, const dyad *ec, int64_t p, int up);
w128 table_top(const w128 *scores, size_t n, int sign);
w128 table_distance(w128 top, w128 score, int sign);
int s128_less(w128 a, w128 b);
w128 s128_negate(w128 a);
w128 s128_shr(w128 a, int64_t s);
typedef struct {
    w128 s;
    size_t j;
    const dyad *key; /* with a float mask, the rest of the logit's order (NULL without) */
} keyed;
int keyed_same(const keyed *a, const keyed *b);
int keyed_order(const void *x, const void *y);
typedef struct {
    const dyad *s, *key; /* the score (NULL: every logit's alike), and with a float mask the rest of the logit's order */
    size_t j;
} gkeyed;
int gkeyed_cmp(const gkeyed *a, const gkeyed *b);
int gkeyed_order(const void *x, const void *y);
int top_logit(dyad *lo, dyad *hi, const att_call *a, w128 top);
int att_affine(int *sign, dyad *lo, dyad *hi, const att_call *a, const dyad *X, const dyad *Y, const dyad *sabs_lo, const dyad *sabs_hi, int64_t p);
int att_argument(dyad *lo, dyad *hi, const att_call *a, const dyad *sd, const dyad *bias, size_t k, size_t j, const dyad *sabs_lo, const dyad *sabs_hi,
                 int64_t p);
int att_logit_cmp(int *sign, const att_call *a, const dyad *sd, const dyad *bias, size_t i, size_t j);
int att_reference_key(size_t *k, const att_call *a, const dyad *sd, const dyad *bias, size_t n, size_t top);
void att_row_init(att_row *r);
void att_row_free(att_row *r);
int att_row_reserve(att_row *r, size_t S, size_t Ev);
int att_row_logits(att_row *w, const att_call *a, size_t qr, size_t kr0, size_t i, size_t moff, size_t mstep, int *kind, int *kept);
int att_reference_row(att_row *w, const att_call *a, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, unsigned char *enc, entry *entries,
                      unsigned char *status, unsigned char *lse_enc, entry *lse_entry, unsigned char *lse_status);

/* attention_tol.c: the forward's tolerance by the closed form (proofs/scaled_dot_product_attention.md, The closed form) */
#define ATT_ROUNDINGS 12 /* the closed form's boundaries (the methods' only with methods; probabilities the backward's): */
#define ATT_R_SCORES 0
#define ATT_R_LOGITS 1
#define ATT_R_SHIFT 2
#define ATT_R_EXP 3
#define ATT_R_VALUES 4
#define ATT_R_LOG 5
#define ATT_R_BIAS 6
#define ATT_R_EXP2_BIAS 7
#define ATT_R_EXP2_LSE 8
#define ATT_R_WEIGHTS 9
#define ATT_R_RECIPROCAL 10
#define ATT_R_PROBS 11
/* the call's constants for the closed form, in this order; absent ones have has[] 0 */
enum {
    ATT_C_MAX_SCORES, ATT_C_MAX_LOGITS, ATT_C_MAX_VALUES, ATT_C_THETA_LOGITS, ATT_C_THETA_SHIFT, ATT_C_FAC, ATT_C_R_L, ATT_C_A_L,
    ATT_C_WMAX_V, ATT_C_R_Z, ATT_C_A_Z, ATT_C_A2_Z, ATT_C_THETA_Z, ATT_C_R_V, ATT_C_A_V, ATT_C_A2_V, ATT_C_THETA_V, ATT_C_NARROW_Z,
    ATT_C_FACTOR_LO, ATT_C_FACTOR_HI, ATT_C_MAX_SHIFT, ATT_C_MAX_RECIPROCAL, ATT_C_MAX_EXP2_LOGSUMEXP, ATT_C_THETA_BIAS, ATT_C_THETA_EXP2_BIAS,
    ATT_C_WMAX, ATT_C_R_H, ATT_C_A_H, ATT_C_UP, ATT_C_OWN, ATT_C_X_Z, ATT_C_XN0, ATT_C_XN1, ATT_C_R_W, ATT_C_A_W, ATT_C_LOG_FLOOR,
    ATT_C_ON_G_Z, ATT_C_ON_K_Z, ATT_C_ON_LIM_Z, ATT_C_ON_G_V, ATT_C_ON_K_V, ATT_C_ON_LIM_V, ATT_C_MAX_EXP, ATT_C_MAX_PROBS, ATT_CONSTS
};
/* the closed form's methods (att_tol.methods) */
enum { ATT_T_ONLINE = 1, ATT_T_EXP2 = 2, ATT_T_WEIGHTS = 4, ATT_T_RECIPROCAL = 8 };
/* the row sum's and value sums' constants for n terms, in this order */
enum { ATT_N_GZ, ATT_N_GV, ATT_N_G_Z, ATT_N_K_Z, ATT_N_GZ1, ATT_N_GZN, ATT_N_G_V, ATT_N_K_V, ATT_N_GV1, ATT_N_GVN, ATT_PER_N };

typedef struct {
    int normal;              /* the normal standard, else any order */
    int backward;            /* the backward's probability step, without the forward's value sums and logsumexp */
    int methods;             /* ATT_T_*: the methods the closed form takes (attention_algorithms.md) */
    int flush;               /* flush to zero: inputs and results below 2**emin of the compute format may be zero */
    int64_t emin;
    int64_t p, w;            /* the closed form's precision; the bounds' (Lemma 1 at normalize and add, Lemma 3) */
    round_units rnd[ATT_ROUNDINGS];
    round_bound normalize, add;
    dyad c[ATT_CONSTS];
    int has[ATT_CONSTS];
    size_t nmax;             /* per-n constants for n = 0 .. nmax */
    dyad *per_n;
    unsigned char *per_n_has;
    floatfmt f64;
    sum_bounds scores;       /* the scores' accumulation */
    dyad sig_lo, sig_hi, B_lo, B_hi, ln2_lo, ln2_hi, log2e_lo, log2e_hi; /* sigma: the exact factor, times log2(e) with exp2 */
    dyad abs_lo, abs_hi;     /* |sigma| without log2(e) (the weights without tables) */
} att_tol;

int table_row_constant(dyad *c, ivl *ec, w128 top, const att_call *a, const dyad *sl, const dyad *lse, int64_t p);

/* One row's closed-form quantities, steps 1 to 8 (att_soft_row): per kept key the logit x, L, R, P = w / W and P T + H
   (with rounded weights also the value sums' error weights); the row's maximum m, lam_up, lam_lo, k_up, k_lo, W, D, tau,
   g, G_lo, G_hi, E_Z and with reciprocal d_r; for the backward's probability step per key H, the upper end of P exp(R)
   (pe_hi), the lower end of P exp(-R) (pe_lo) and w'_j's largest value (wh). */
typedef struct {
    size_t n;
    ivl *xs, *Ls, *Rs, *Ps, *PTs, *PVs, *Hs;
    dyad *pe_hi, *pe_lo, *wh;
    unsigned char *vanish;
    ivl m, lam_up, lam_lo, kup, klo, W, D, tau;
    dyad g, Glo, Ghi, EZ, dr;
    int masked_nan;          /* a masked key's logit may be NaN (condition 4 fails): the row may be NaN */
} att_soft;

void att_tol_init(att_tol *t);
void att_tol_free(att_tol *t);
void att_soft_init(att_soft *s);
void att_soft_free(att_soft *s);
int att_soft_row(att_soft *sf, int *ok, att_row *w, const att_call *a, const att_tol *t, size_t n, size_t qr, size_t kr0, size_t moff,
                 size_t mstep, int probs, int logits_only, unsigned char *mnan);
int att_tol_prepare(att_tol *t, const att_call *a, const floatfmt *scores, int64_t w);
int att_tolerance_row(att_row *w, const att_call *a, const att_tol *t, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0,
                      const entry *out_entries,
                      const unsigned char *out_refs, const entry *lse_entry, const unsigned char *lse_ref, unsigned char *tol, unsigned char *lows,
                      unsigned char *highs, unsigned char *flags, unsigned char *lse_tol, unsigned char *lse_low, unsigned char *lse_high,
                      unsigned char *lse_flag, unsigned char *status);
void product_fast_nodes(w128 *s, w128 *mass, w128 *N, const uint64_t *a, const uint64_t *b, size_t k);
void product_fast_nodes_at(w128 *s, w128 *mass, w128 *N, const uint64_t *a, const uint64_t *b, const size_t *pos, size_t k);

/* attention_tol.c, the step-by-step program (proofs/scaled_dot_product_attention.md, The step-by-step program;
   attention_algorithms.md): an implementation rounding at every boundary, as intervals of dyadics */
enum { /* its rounding and function boundaries */
    ATT_P_SCORES_OUT, ATT_P_LOGITS, ATT_P_SHIFT, ATT_P_EXP, ATT_P_WEIGHTS, ATT_P_VALUES_OUT, ATT_P_RECIPROCAL, ATT_P_NORMALIZE, ATT_P_LOG,
    ATT_P_ADD, ATT_P_EXP2_LSE, ATT_P_BIAS, ATT_P_EXP2_BIAS, ATT_P_SPLIT_SHIFT, ATT_P_SPLIT_EXP, ATT_P_SPLIT_LOG, ATT_P_SPLIT_ADD,
    ATT_P_SPLIT_PRODUCT, ATT_P_LEANS
};
enum { ATT_A_SCORES, ATT_A_SUM, ATT_A_VALUES, ATT_A_SPLIT_SUM, ATT_A_SPLIT_ACC, ATT_A_KINDS }; /* its accumulations */
enum { ATT_ACC_NONE, ATT_ACC_TERMS };
enum { /* its constants; absent ones have has[] 0 */
    ATT_O_FACTOR_LO, ATT_O_FACTOR_HI, ATT_O_REL_SHIFT, ATT_O_ABS_SHIFT, ATT_O_UP, ATT_O_DOWN, ATT_O_OWN, ATT_O_GROWTH, ATT_O_G_MIN, ATT_O_ETA,
    ATT_O_K, ATT_O_RESCALE, ATT_O_REL_W, ATT_O_ABS_W, ATT_O_GROW_W, ATT_O_MMA_G, ATT_O_MMA_H, ATT_O_MMA_TOP, ATT_O_MMA_K, ATT_O_MMA_RESCALE,
    ATT_O_FLOOR, ATT_O_THETA_LOGITS, ATT_O_EXP_R, ATT_O_EXP_A, ATT_O_Z_R, ATT_O_Z_A, ATT_O_Z_A2, ATT_O_Z_THETA, ATT_O_V_R, ATT_O_V_A, ATT_O_V_A2,
    ATT_O_V_THETA, ATT_O_SE_REL, ATT_O_SE_ABS, ATT_O_SA_REL, ATT_O_SA_ABS, ATT_O_SB_REL, ATT_O_SB_ABS, ATT_O_SK_REL, ATT_O_SK_ABS, ATT_O_SP_REL,
    ATT_O_SP_ABS, ATT_O_CONSTS /* split_kv: the units of split_kv.exp, values.output, reciprocal, normalize and split_kv.product */
};
enum { ATT_M_ONLINE = 1, ATT_M_EXP2 = 2, ATT_M_WEIGHTS = 4, ATT_M_RECIPROCAL = 8, ATT_M_FAST_LOG = 16, ATT_M_MMA_VALUES = 32, ATT_M_METHODS = 64,
       ATT_M_PADDED = 128, ATT_M_NARROW = 256, ATT_M_SPLIT = 512 };

typedef struct {
    int kind;                /* ATT_ACC_*: no rounding, or Lemma 2's (or T's) terms per n */
    size_t nmax;
    dyad *terms;             /* (nmax + 1) x 3: gamma, constant, largest finite value */
    unsigned char *has;      /* (nmax + 1): the terms exist */
} att_acc;

typedef struct {
    int64_t w;
    int methods;             /* ATT_M_* */
    int sign;                /* sigma's sign (IEEE's rules on the logits) */
    int unbounded;           /* the factor has no bound: no row has one */
    int online_unbounded;    /* online softmax's constants have none: no row with a kept key has one */
    int64_t online_n, online_S, split; /* online softmax's term count n and keys S (one split's with split_kv); keys per split */
    int flush;               /* flush to zero: inputs and results below 2**emin of the compute format may be zero */
    int64_t emin;
    round_bound lean[ATT_P_LEANS];
    att_acc acc[ATT_A_KINDS];
    dyad o[ATT_O_CONSTS];
    int has[ATT_O_CONSTS];
    floatfmt f64;
    sum_bounds scores;       /* the scores' any-order bound (the padded rows') */
    dyad ln2_lo, ln2_hi, log2e_lo, log2e_hi;
} att_prog;

void att_prog_init(att_prog *g);
void att_prog_free(att_prog *g);
int att_prog_prepare(att_prog *g, const floatfmt *scores, int64_t E);
int att_prog_softmax(ivl *L, ivl *B, ivl *Y, ivl *Wt, ivl *R, unsigned char *V, ivl *top, ivl *total, int *masked_nan, int *soft_unb,
                     int *pad_unb, att_row *w, const att_call *a, const att_prog *g, size_t n, size_t qr, size_t kr0, size_t moff, size_t mstep,
                     int64_t P, int logits_only, unsigned char *mnan);

/* attention_tol.c: the forward's constants from its settings */
enum { ATT_SA_NONE, ATT_SA_ANY_ORDER, ATT_SA_MMA };
typedef struct {                     /* an accumulation's setting (boundaries.Accumulation) */
    int kind, rnd;                   /* ATT_SA_*: any order, Lemma 2's counts (any_order_counts), or mma */
    floatfmt f;
    int64_t products, fraction_bits, exponent_floor, split, count; /* mma: its step; count -1 for the caller's */
    int has_reduction;
    floatfmt reduction;
    dyad slack;
} att_acc_set;

enum {                               /* the rounding and function boundaries: the forward's, then the backward's own */
    ATT_SR_SCALE, ATT_SR_EXP2_SCALE, ATT_SR_SCORES_OUT, ATT_SR_LOGITS, ATT_SR_BIAS, ATT_SR_EXP2_BIAS, ATT_SR_SHIFT, ATT_SR_EXP,
    ATT_SR_RESCALE, ATT_SR_WEIGHTS, ATT_SR_VALUES_OUT, ATT_SR_RECIPROCAL, ATT_SR_NORMALIZE, ATT_SR_LOG, ATT_SR_ADD, ATT_SR_EXP2_LSE,
    ATT_SR_SPLIT_SHIFT, ATT_SR_SPLIT_EXP, ATT_SR_SPLIT_LOG, ATT_SR_SPLIT_ADD, ATT_SR_SPLIT_PRODUCT,
    ATT_SR_PROBS, ATT_SR_DP_OUT, ATT_SR_DIFF, ATT_SR_DS, ATT_SR_SCALE_GRADS, ATT_SR_SCALE_DS, ATT_SR_OUT_Q, ATT_SR_OUT_K, ATT_SR_OUT_V,
    ATT_SR_CARRY_Q, ATT_SR_CARRY_K, ATT_SR_CARRY_V, ATT_SR_GROUP_K, ATT_SR_GROUP_V, ATT_SR_COUNT
};
enum { /* the accumulations: the forward's, then the backward's own */
    ATT_SC_SCORES, ATT_SC_SUM, ATT_SC_VALUES, ATT_SC_SPLIT_SUM, ATT_SC_SPLIT_ACC, ATT_SC_DP, ATT_SC_ROW_DOT, ATT_SC_Q, ATT_SC_K, ATT_SC_V,
    ATT_SC_GROUP_K, ATT_SC_GROUP_V, ATT_SC_CARRY_Q, ATT_SC_CARRY_K, ATT_SC_CARRY_V, ATT_SC_COUNT
};

typedef struct {
    round_to r[ATT_SR_COUNT];
    att_acc_set a[ATT_SC_COUNT];
    int flush;                       /* the flush-to-zero option */
    int64_t emin, w;                 /* the compute format's emin (with flush); the bounds' precision */
    int64_t S, E;
    bigint num, den;                 /* sigma = num / den, or 1/sqrt(E) when den is 0 */
    int has_methods, online, exp2, reciprocal, fast_log, has_mma; /* the named algorithm's methods (_problem.Methods) */
    int64_t block_size, split;       /* 0: none */
    int has_weights, has_bias;
    floatfmt weights;                /* the weights' format */
    int backward;                    /* the backward's settings: its paths in the backward's slots, the forward-only ones empty */
    int rounded_p, scale_ds, grouped; /* the backward's methods */
    int64_t query_block, key_block;  /* backward_blocks, 0 without */
    int bounded;                     /* the factor the scores are multiplied by has a bound: */
    dyad factor_lo, factor_hi;       /* ... its interval */
    int scale_bounded;               /* the scale held in the compute format has a bound: */
    dyad scale_lo, scale_hi;         /* ... its interval */
} att_settings;

void att_settings_init(att_settings *s);
void att_settings_free(att_settings *s);
int att_settings_prepare(att_settings *s);
int att_tol_build(att_tol *t, int *applicable, int *bad, const att_settings *s, int normal);
int att_prog_build(att_prog *g, int *applicable, int *bad, const att_settings *s, int sign);

/* attention_tol.c's steps shared with the backward (attention_grad_tol.c): exp(x) - 1's ends, an accumulation's
   bound (att_acc), the program's steps and the constants */
int expm1_end(dyad *r, const dyad *x, int64_t p, int up);
int radius_f64(dyad *r, int *finite, const floatfmt *f64, const unsigned char *x);
int row_unbounded(const att_call *a, const floatfmt *f64, const att_row *w, size_t qr, size_t kr0, size_t moff, size_t mstep, int *nan);
void att_masked_nan(int *row, unsigned char *col, const att_row *w, const att_call *a, size_t vr0);
int acc_apply(ivl *x, const att_acc *c, size_t n, const dyad *mass, int64_t w, int *unb);
int prog_score(ivl *s, int *unb, const att_prog *g, const product_sums *ps, int64_t E);
int lean_vanish(ivl *x, int *unb, unsigned char *vanish, const round_bound *b, int64_t w);
int prog_logit(ivl *x, int *unb, unsigned char *vanish, const att_prog *g, const ivl *score, const ivl *bias);
int prog_bias(ivl *b, int *unb, unsigned char *vanish, const att_prog *g, const dyad *value, const dyad *radius);
int acc_of(att_acc *c, const att_acc_set *a, int64_t first, int64_t last, int64_t w, int flush);
int acc_units(dyad *r, dyad *a, dyad *a2, dyad *theta, int *bad, const floatfmt *f, int rnd, int flush);
int acc_terms(dyad *gamma, dyad *constant, dyad *largest, int *bounded, const att_acc_set *a, int64_t n, int64_t w, int flush);
int dyad_words(dyad *r, const uint64_t *w, size_t n, int64_t e);
int exp_overflows(const dyad *x);

/* attention_grad_tol.c: the backward's tolerance (proofs/scaled_dot_product_attention_backward.md, The tolerance) */
enum { /* Lemma 1 at the backward's boundaries (att_gtol.lean) */
    ATT_GL_PROBS, ATT_GL_RECIPROCAL, ATT_GL_DP, ATT_GL_DIFF, ATT_GL_DS, ATT_GL_SCALE, ATT_GL_SCALE_DS, ATT_GL_WEIGHTS, ATT_GL_EXP2_LSE, ATT_GL_SHIFT,
    ATT_GL_EXP, ATT_GL_OUT_Q, ATT_GL_OUT_K, ATT_GL_OUT_V, ATT_GL_GROUP_K, ATT_GL_GROUP_V, ATT_GL_COUNT
};
enum { ATT_GA_DP, ATT_GA_ROW_DOT, ATT_GA_Q, ATT_GA_K, ATT_GA_V, ATT_GA_GROUP_K, ATT_GA_GROUP_V, ATT_GA_COUNT }; /* its accumulations */
enum { ATT_GM_EXP2 = 1, ATT_GM_RECIPROCAL = 2, ATT_GM_WEIGHTS = 4, ATT_GM_ROUNDED_P = 8, ATT_GM_SCALE_DS = 16, ATT_GM_GROUPED = 32, ATT_GM_BLOCKED = 64,
       ATT_GM_MMA = 128 }; /* its methods (attention_algorithms.md) */

typedef struct {             /* Lemma C's constants for one carried sum (foundations.md, Lemma C) */
    round_bound round;          /* Lemma 1 at the rounding to the carry format */
    att_acc acc;             /* Lemma 2's terms of the any-order method per term count, up to a block's terms + 2 */
    dyad abs;                /* the accumulator's absolute unit */
    int contained;           /* the carry format's values are values of the accumulator's (an empty block leaves the value) */
    int scale_bounded;       /* the scale has a bound: */
    dyad scale_lo, scale_hi, alpha, alpha_one; /* ... its interval, |scale| and max(1, |scale|) */
    int64_t w;
} att_carry;

void att_carry_init(att_carry *k);
void att_carry_free(att_carry *k);
int att_carry_build(att_carry *k, const round_to *carry, const att_acc_set *acc, int64_t last, int scale_bounded, const dyad *scale_lo,
                    const dyad *scale_hi, int flush, int64_t w);
int att_carry_add(dyad c[2], unsigned char *status, const att_carry *k, int scaled, const dyad *lo, const dyad *hi, const dyad *mass, int64_t count,
                  const dyad *F);

typedef struct {
    int normal;              /* the normal standard, else any order */
    int sums_normal;         /* the gradients' sums under the normal standard (not with backward_blocks) */
    int closed;              /* the closed form's constants apply */
    int tables;              /* its weights may come from Lemma D's tables */
    int64_t query_block, key_block; /* backward_blocks, 0 without */
    att_carry carry[3];      /* ... Lemma C's constants per gradient (grad_query, grad_key, grad_value) */
    int methods;             /* ATT_GM_* */
    int flush;               /* flush to zero: inputs and results below 2**emin of the compute format may be zero */
    int64_t emin, w, S, Ev;  /* the compute format's emin; the bounds' precision; keys per row; dP's terms */
    int64_t nq, nkv;         /* the terms of each grad_query sum and of each grad_key and grad_value sum */
    att_tol t;               /* the closed form, for the probabilities */
    att_prog g;              /* the step-by-step program's steps 1 to 7 */
    round_bound lean[ATT_GL_COUNT];
    att_acc acc[ATT_GA_COUNT]; /* Lemma 2's (or T's) terms per count, or one rounding for an exact accumulation */
    sum_bounds dp;           /* the normal standard's Lemma S for dP */
    floatfmt row_dot_f;      /* D's accumulator */
    floatfmt q_f, k_f, v_f;  /* the gradients' accumulators */
    int row_dot_rnd, row_dot_active;
    int scale_bounded;       /* the scale held in the compute format has a bound: */
    dyad scale_lo, scale_hi; /* ... its interval */
} att_gtol;

typedef struct {             /* one row's P', dP' and dS' over its S keys (att_gtol_row) */
    int nan, p_unb, ds_unb, ds_skip, closed; /* closed: P' from the closed form */
    int p_nan, d_nan;        /* P' may be NaN, D' may be NaN (scaled_dot_product_attention_backward.md, Masked keys) */
    int q_nan;               /* a pair's mark (mark) makes grad_query's row NaN */
    int g_unb;               /* an element of its output gradient row has no bound as an earlier result */
    size_t cap;
    ivl *P, *DP, *DS;
    ivl *L, *B, *Y, *Wt;     /* the program's per-key scratch */
    unsigned char *V;        /* ... and which kept keys may vanish */
    unsigned char *mark;     /* per key: its pair's marks (ATT_KEY_DS_NAN, ATT_KEY_P_NAN), in V's block */
    unsigned char *mnan;     /* ... and, with a given logsumexp, whether it is a masked key whose logit may overflow */
    att_soft sf;
} att_gtol_out;

/* a tile row's flags (att_gtol_out's) */
enum { ATT_GF_NAN = 1, ATT_GF_P_UNB = 2, ATT_GF_DS_UNB = 4, ATT_GF_DS_SKIP = 8, ATT_GF_G_UNB = 16, ATT_GF_P_NAN = 32, ATT_GF_D_NAN = 64, ATT_GF_Q_NAN = 128 };

typedef struct {             /* a call of the backward's tolerance: its constants, operands, units and words */
    const att_gtol *gr;
    att_call a;              /* Q's and K's rows for the scores, the tables, the mask, S, Ev, E, the scale */
    const operand *gd, *vd;  /* the output gradient's and V's rows for dP */
    const operand *gs, *qs, *ks; /* the output gradient's, Q's and K's rows for the gradient sums */
    const operand *od, *ld;  /* the forward's given output (rows of Ev, dP's main range) and logsumexp (rows of one), or NULL */
    /* inputs that are earlier results (scaled_dot_product_attention_backward.md, Results as inputs), beside a's Q, K and mask
       radii: V's, the output gradient's and the given output's radii as operands (rows of Ev), the given logsumexp's as
       float64 per row; NULL where exact */
    const operand *rv, *rg, *ro;
    const unsigned char *lrad;
    uint64_t *x[3];          /* the gradient sums' exact factors (grad_value's, grad_key's, grad_query's) at 2**ux, wx words each */
    uint64_t *xr[3];         /* ... their radii as earlier results (unsigned), or NULL where exact */
    unsigned char *xsub[3];  /* ... with flush, those a flush may read as zero */
    int64_t ux[3];
    int wx[3];
    int64_t up, ud;          /* the units of P' and dS' ends (their formats' least values) */
    int64_t pmin, dmin;      /* at least these bits for P' and dS' ends (a call made again where its ends needed them) */
    int wp, wd, ws[3];       /* the words of P' and dS' ends, and of each gradient's sums */
    int64_t plim, dlim;      /* the most bits a P' or dS' end may take, its sign's included: those its sums are sized for */
    int fast;                /* the fast path: three-word ends, one-word factors, four-word sums */
    floatfmt out, f64;       /* the gradients' format, and float64 */
    bigint out_nan[2], f64_nan[2]; /* their NaN encodings, which out and f64 point to */
    size_t out_size;
    int rnd, ovf;
    const unsigned char *ref[3]; /* the references' encodings: grad_value's, grad_key's, grad_query's */
    uint64_t *qstates;       /* query rows broadcast over several key and value blocks: per slot E grad_query sums */
    int64_t *qcounts;        /* ... their counts of terms that can be nonzero */
    unsigned char *qunb;     /* ... and whether some copy has no bound */
    unsigned char *qnan;     /* ... and per element whether some copy may make it NaN (Masked keys) */
} att_gtol_call;

#define ATT_GR_INFO 8        /* a tile row's info: query row, query index, mask offset and step, output-gradient row, its
                                broadcast slot (-1 for none), copy and copies */

typedef struct {             /* a key block's grad_key sums, or a value block's grad_value sums, over the units that share it */
    size_t r0, cols, elements; /* its first row (K's or V's), E or Ev per key, S cols */
    int ws;                  /* its sums' words */
    uint64_t *sum;           /* per element: its sum's values */
    int64_t *count;          /* any order: per element, its terms that can be nonzero */
    unsigned char *started;  /* per key: its sums have had a term not zero */
    int64_t rows;            /* its rows so far: the next term's index */
    int unb;                 /* it has no bound */
    int nan;                 /* a row may make every element NaN (scaled_dot_product_attention_backward.md, Masked keys) */
    unsigned char *nanel;    /* ... and per element whether one may make it NaN */
    dyad *group;             /* grouped_heads: per element its sum over the heads (lo, hi, mass) */
    unsigned char *gunb;     /* ... and whether some head's sum has no bound */
    int64_t parts;           /* ... the heads folded so far */
    dyad *carried;           /* backward_blocks: per element its carried value (lo, hi) */
    unsigned char *cstatus;  /* ... 0 none yet, 1 a value, 2 no bound */
    dyad *dsum;              /* ... a key block's per key sum of |dS'| over the block of query rows, then per column of Q its sum of |q| */
    size_t dcount;           /* ... S + E of them */
} att_gtol_block;

typedef struct {             /* one key and value unit's tile */
    size_t kr0, vr0, cap, T;
    int64_t *rinfo;          /* per tile row: ATT_GR_INFO values */
    unsigned char *flags;    /* per tile row: ATT_GF_* */
    uint64_t *plo, *phi, *dlo, *dhi; /* per tile row and key: P''s and dS''s ends */
    unsigned char *kinds;    /* ... the key's part in the row (ATT_KEY_*) and the pair's marks */
    int64_t *need;           /* per tile row: the bits its P' and dS' ends need where the call's words do not hold them, else 0 */
    att_gtol_block *kb, *vb; /* the blocks it adds to */
} att_gtol_group;

void att_gtol_init(att_gtol *g);
void att_gtol_free(att_gtol *g);
int att_gtol_build(att_gtol *gr, int *bad, const att_settings *s, const att_call *a, int normal, int closed, int sign, int64_t Ev, int64_t nq,
                   int64_t nkv, int64_t heads);
void att_gtol_out_init(att_gtol_out *o);
void att_gtol_out_free(att_gtol_out *o);
void att_gtol_call_init(att_gtol_call *c);
void att_gtol_call_free(att_gtol_call *c);
int att_gtol_row(att_gtol_out *o, att_row *w, const att_gtol_call *c, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, size_t gr0);
int att_gtol_call_prepare(att_gtol_call *c);
void att_gtol_group_init(att_gtol_group *g);
void att_gtol_group_free(att_gtol_group *g);
int att_gtol_block_start(att_gtol_block *b, const att_gtol_call *c, int key, size_t r0);
void att_gtol_block_free(att_gtol_block *b);
int att_gtol_group_start(att_gtol_group *g, const att_gtol_call *c, size_t kr0, size_t vr0, att_gtol_block *kb, att_gtol_block *vb, size_t cap);
int att_gtol_tile_rows(size_t *closed, att_gtol_group *g, const att_gtol_call *c, size_t r0, size_t r1);
int att_gtol_tile_keys(att_gtol_group *g, const att_gtol_call *c, size_t j0, size_t j1);
int att_gtol_query_slots(att_gtol_call *c, size_t count);
int att_gtol_tile_queries(const att_gtol_group *g, att_gtol_call *c, size_t r0, size_t r1, unsigned char *tol, unsigned char *low,
                          unsigned char *high, unsigned char *flags);
int att_gtol_tile_done(att_gtol_group *g, const att_gtol_call *c);
int att_gtol_group_fold(att_gtol_group *g, const att_gtol_call *c, size_t j0, size_t j1);
int att_gtol_group_carry(att_gtol_group *g, const att_gtol_call *c, size_t j0, size_t j1);
int att_gtol_group_carried(att_gtol_group *g);
void att_gtol_group_folded(att_gtol_group *g);
int att_gtol_block_finish(const att_gtol_block *b, const att_gtol_call *c, int key, size_t j0, size_t j1, unsigned char *tol, unsigned char *low,
                          unsigned char *high, unsigned char *flags);

/* attention_grad_tol.c: an operand's extent, shared with attention_gref.c */
void operand_extent(int64_t *top, int64_t *low, int *any, const operand *x);

/* attention_gref.c: the backward's reference (proofs/scaled_dot_product_attention_backward.md, The reference output) */
#define ATT_REF_Q 190        /* P's ends at 2**-ATT_REF_Q; dS's that many bits below the call's bound on |dS| */
enum { ATT_RF_NAN = 1, ATT_RF_DS_SPECIAL = 2, ATT_RF_NO_ENCLOSURE = 4, ATT_RF_EQUAL = 8, ATT_RF_UNIFORM = 16 }; /* a tile row's flags */

typedef struct {             /* a call of the backward's reference: its operands, units and words */
    att_call a;              /* Q's and K's rows for the scores, Lemma D's tables, the mask, S, Ev, E, u, the scale */
    const operand *gd, *vd;  /* the output gradient's and V's rows for dP */
    const operand *gs, *qs, *ks; /* the output gradient's, Q's and K's rows for the gradient sums */
    uint64_t *x[3];          /* their finite values as integers at 2**ux, wx words each (grad_value's, grad_key's, grad_query's factors) */
    int64_t ux[3], ud, edp;  /* ...; dS's ends' unit; dP's (the integers of the exact analysis) */
    int wx[3], ws[3];        /* ...; each gradient's sums' words */
    int fast;                /* three-word ends, one-word factors, four-word sums */
    rat sigma;               /* the scale, when rational */
    dyad slo, shi;           /* ... else 1/sqrt(E) enclosed */
    dyad sabs_lo, sabs_hi;   /* |sigma| enclosed (Lemma X's weights) */
    int64_t p;               /* the first enclosure's working precision */
    floatfmt out;
    bigint out_nan[2];
    size_t out_size;
    int rnd, ovf, neg_zero;  /* neg_zero: an exact zero with terms is -0 (rounding toward -inf) */
    /* query rows broadcast over several key and value blocks: per slot E elements' state between copies */
    uint64_t *qsum;
    int64_t *qcount;
    unsigned char *qspec, *qflags;
    rat **qrat;
    int64_t *qterms;
    size_t qslots;
} att_gref_call;

typedef struct {             /* a key block's grad_key elements, or a value block's grad_value elements, over the units that share it */
    size_t cols, elements;   /* E or Ev per key; S cols */
    int ws;                  /* its sums' words */
    uint64_t *sum;           /* per element: its ends' sums lo, hi */
    int64_t *count;          /* ... its irrational terms */
    unsigned char *spec;     /* ... its non-finite value (0 none, else 8 | VALUE_* | neg << 2) */
    rat **rat;               /* ... its rational part (NULL for 0) */
    int64_t terms;           /* its rows so far */
} att_gref_block;

typedef struct {             /* one key and value unit's tile */
    size_t kr0, vr0, cap, T, S, E, Ev;
    int64_t *rinfo;          /* per tile row: ATT_GR_INFO values */
    unsigned char *flags;    /* per tile row: ATT_RF_* */
    size_t *n;               /* ... its kept count */
    unsigned char *kept;     /* per tile row and key: 1 kept (in a NaN row, its logit is not -inf), 2 its dS may be irrational (not 0) */
    unsigned char *dspec;    /* ... the row's non-finite dS, for ATT_RF_DS_SPECIAL rows */
    uint64_t *plo, *phi, *dlo, *dhi; /* ... P's and dS's ends (three words) */
    bigint **dsnum;          /* per tile row: a uniform row's n dP_j - sum dP per key (at 2**edp), else NULL */
    bigint lp, ld, *mp, *md; /* the tile's equal rows' common denominator (of n) and uniform rows' (of n**2), each row's multiplier */
    att_gref_block *kb, *vb; /* the blocks it adds to */
} att_gref_group;

void att_gref_call_init(att_gref_call *c);
void att_gref_call_free(att_gref_call *c);
int att_gref_call_prepare(att_gref_call *c, int64_t nq, int64_t nkv);
int att_gref_query_slots(att_gref_call *c, size_t count);
void att_gref_group_init(att_gref_group *g);
void att_gref_group_free(att_gref_group *g);
int att_gref_block_start(att_gref_block *b, const att_gref_call *c, int key);
void att_gref_block_free(att_gref_block *b);
int att_gref_group_start(att_gref_group *g, const att_gref_call *c, size_t kr0, size_t vr0, att_gref_block *kb, att_gref_block *vb, size_t cap);
int att_gref_tile_rows(att_gref_group *g, const att_gref_call *c, size_t r0, size_t r1);
int att_gref_tile_prep(att_gref_group *g, const att_gref_call *c);
int att_gref_tile_keys(att_gref_group *g, const att_gref_call *c, size_t j0, size_t j1);
int att_gref_tile_queries(const att_gref_group *g, att_gref_call *c, size_t r0, size_t r1, unsigned char *enc, entry *entries, unsigned char *status);
void att_gref_tile_done(att_gref_group *g);
int att_gref_block_finish(const att_gref_block *b, const att_gref_call *c, int key, size_t j0, size_t j1, unsigned char *enc, entry *entries,
                          unsigned char *status);

int normal_scale(int *kind, int *neg, bigint *sig, int64_t *e, const floatfmt *f, const bigint *num, const bigint *den, int64_t E);

/* attention_grad_refine.c: the backward's refinement (proofs/scaled_dot_product_attention_backward.md, Refinement) */
typedef struct {             /* a row reaching an undecided element: its descriptor, and while a tile works on it its exact
                                quantities and its intervals at p */
    size_t qr, i, moff, mstep, gr0, kr0, vr0;
    int ready, nan, has_far, sorted; /* sorted: order sorted (grow_order) */
    int64_t p;                /* the precision of its intervals */
    size_t n, top, ref, F;    /* kept keys; the extreme score's key; the largest logit's key (the weights' reference); the largest logit's far key */
    size_t *keys, *pos;       /* the kept keys; per key of the block its position + 1 among them (0: not kept) */
    dyad *sd, *dp, *bias, *lk; /* per kept key: score, dP, mask value, a rational sigma's logit order */
    gkeyed *order;            /* the kept keys sorted into groups of equal logits */
    unsigned char *far;       /* per kept key: far (Far keys) */
    ivl *x, *P, *dS, *dS2, *dPr, *dSr, *om, xF; /* per kept key: argument, P, dS, the cancellation-free dS, Delta P and Delta dS
                                                   relative to e**-(x_F) (kept) or to the key's own weight e**-(x_m) (far),
                                                   omega = e**(x_F - x_m) (far); x_F */
} att_grow;

typedef struct {             /* an undecided element */
    int kind;                 /* 0 grad_value (P g), 1 grad_key (sigma dS q), 2 grad_query (sigma dS k) */
    size_t j, col;            /* its key in the block (grad_value, grad_key) and its column */
    size_t first, count;      /* its rows in the row list */
} att_gelem;

typedef struct att_gref_work att_gref_work; /* the rounds' state (attention_grad_refine.c) */

typedef struct {
    const att_gref_call *c;
    att_grow *rows;
    size_t nrows, count;
    att_gelem *elems;
    size_t *rowlist;
    unsigned char *enc, *status; /* per element: its encoding and status (1 undecided) */
    entry *entries;
    dyad sabs_lo, sabs_hi, slo, shi; /* |sigma| and 1/sqrt(E) at the round's precision */
    dyad sfar_lo, sfar_hi;           /* |sigma| at the round's precision plus 128 bits: the far groups' factors */
    att_gref_work *work;
} att_gref_refine;

void att_gref_refine_init(att_gref_refine *f);
void att_gref_refine_free(att_gref_refine *f);
void att_gref_refine_waves(const att_gref_refine *f, size_t *waves, size_t *split);
int att_gref_refine_start(att_gref_refine *f, int64_t p0, int64_t cap);
int att_gref_refine_next(att_gref_refine *f, size_t *tiles);
int att_gref_refine_tile(att_gref_refine *f, size_t t);

/* attention_normal.c: the forward's normal answer (proofs/scaled_dot_product_attention.md, The normal answer), and
   below the backward's */
typedef struct {
    femu x;                  /* the compute format's arithmetic: to nearest, ties to even, with the call's flush */
    floatfmt f;              /* the compute format */
    fvalue sigma;            /* the scale held in it */
} att_normal;

/* the backward's normal answer (proofs/scaled_dot_product_attention_backward.md, The normal answer) */
typedef struct {
    att_call a;              /* Q's and K's rows for the scores, the mask, S, Ev, E */
    att_normal nm;           /* the compute format's arithmetic and the scale held in it */
    const operand *gd, *vd;  /* the output gradient's and V's rows (Ev per row); Q's and K's are a's */
    floatfmt out;
    bigint out_nan[2], compute_nan[2]; /* the NaN encodings out's and nm's format point to */
    size_t out_size;
    int rnd, ovf;
    fvalue *qacc;            /* query rows broadcast over several key and value blocks: per slot E running sums */
    unsigned char *qall;     /* ... whether every term so far is finite */
    int64_t *qn;             /* ... their terms so far */
    size_t qslots;
} att_gnorm_call;

typedef struct {             /* a key block's grad_key sums, or a value block's grad_value sums, over the units that share it */
    size_t cols;             /* E or Ev per key */
    fvalue *acc;             /* per element: its running sum in index order */
    unsigned char *all;      /* ... whether every term so far is finite (else no slots) */
    int64_t *n;              /* ... its terms so far */
} att_gnorm_block;

typedef struct {             /* one key and value unit's tile */
    size_t kr0, vr0, cap, T, S, E, Ev;
    int64_t *rinfo;          /* per tile row: ATT_GR_INFO values */
    fvalue *P, *DS;          /* per tile row and key */
    att_gnorm_block *kb, *vb; /* the blocks it adds to */
} att_gnorm_group;

void att_gnorm_call_init(att_gnorm_call *c);
void att_gnorm_call_free(att_gnorm_call *c);
int att_gnorm_query_slots(att_gnorm_call *c, size_t count);
int att_gnorm_block_start(att_gnorm_block *b, const att_gnorm_call *c, size_t cols);
void att_gnorm_block_free(att_gnorm_block *b);
int att_gnorm_block_finish(const att_gnorm_block *b, const att_gnorm_call *c, int scaled, size_t j0, size_t j1, unsigned char *enc, unsigned char *status);
void att_gnorm_group_init(att_gnorm_group *g);
void att_gnorm_group_free(att_gnorm_group *g);
int att_gnorm_group_start(att_gnorm_group *g, const att_gnorm_call *c, size_t kr0, size_t vr0, att_gnorm_block *kb, att_gnorm_block *vb, size_t cap);
int att_gnorm_tile_rows(att_gnorm_group *g, const att_gnorm_call *c, size_t r0, size_t r1);
int att_gnorm_tile_keys(att_gnorm_group *g, const att_gnorm_call *c, size_t j0, size_t j1);
int att_gnorm_tile_queries(const att_gnorm_group *g, att_gnorm_call *c, size_t r0, size_t r1, unsigned char *enc, unsigned char *status);
int att_gnorm_slots(unsigned char *enc, const att_gnorm_call *c, int which, size_t index, const int64_t *rows, size_t count);

int att_normal_softmax(int *kind, fvalue *top, fvalue *total, fvalue *logits, fvalue *weights, fvalue *terms, fvalue *scratch, const att_call *a,
                       const att_normal *nm, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0);
int att_normal_row(const att_call *a, const att_normal *nm, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0, unsigned char *enc,
                   unsigned char *lse_enc);
int att_program_row(att_row *w, const att_call *a, const att_prog *g, size_t qr, size_t i, size_t moff, size_t mstep, size_t kr0, size_t vr0,
                    const unsigned char *out_refs, const unsigned char *lse_ref, unsigned char *tol, unsigned char *lows, unsigned char *highs,
                    unsigned char *flags, unsigned char *lse_tol, unsigned char *lse_low, unsigned char *lse_high, unsigned char *lse_flag,
                    unsigned char *status);

/* matrix_exp.c: the rationality test (matrix_exp.md, Rational parts) */
int mexp_rational(int *rational, rat *values, size_t n, int parts, const dyad *A, const size_t *wanted, size_t count);

/* matrix_exp.c: the reference */
typedef struct {           /* matrix_exp's reference for a batch of matrices (matrix_exp.md, The reference output) */
    size_t count, n;       /* matrices of n x n entries */
    int parts;             /* 1 real, 2 complex */
    size_t r0, r1, c0, c1; /* the wanted entries: rows r0 .. r1 - 1, columns c0 .. c1 - 1 */
    dyad *A;               /* count x parts x n x n entries, a plane per part */
    unsigned char *finite; /* per matrix: every entry finite */
    floatfmt out;          /* the output part format, its bytes, rounding and overflow mode */
    bigint out_nan[2];     /* out's NaN encodings, which it points to */
    size_t size;
    int rnd, ovf;
    int64_t start, cap;    /* the precision loop's first and last precisions */
    unsigned char *enc;    /* per wanted part: its encoding, */
    unsigned char *status; /* 0 decided, 1 undecided at the cap, 2 past the norm's range, 3 past the products' size */
    entry *entries;        /* and its exact entry */
    int whole, started;    /* small matrices: whole references per tile */
    size_t mi, k;          /* larger ones: the matrix in progress and its exponential's phase */
    int stage, tested, P;  /* its step, whether its rationality test ran, its parts (1 for a real complex matrix) */
    int64_t p, extra;      /* its precision and extra bits */
    uint64_t *reach;       /* its pattern's closure */
    int64_t *scale;        /* per matrix: every wanted part is 2**scale times the computed one (mref_balance) */
    imat M;                /* it at the current limbs */
    mexp ex;               /* and its exponential */
} mref;

void mref_free(mref *x);
int mref_init(mref *x, size_t count, size_t n, int parts, size_t r0, size_t r1, size_t c0, size_t c1, const floatfmt *out, size_t size, int rnd,
              int ovf, int64_t start, int64_t cap);
int mref_balance(mref *x, size_t m);
int mref_next(mref *x, size_t *tiles);
int mref_tile(mref *x, size_t t);

/* matrix_exp.c: the tolerance (matrix_exp.md, The tolerance) */
#define MEXP_MAX_PIECES 256 /* the ball's pieces at most, and no ball past ||A||_inf = 256 */
typedef struct {
    int kind, a;
    size_t b;
} mtol_job;

#define MTOL_EXPS 5
typedef struct {           /* matrix_exp's tolerance for a batch of matrices */
    size_t count, n;       /* matrices of n x n entries */
    int parts;
    size_t r0, r1, c0, c1; /* the wanted entries */
    dyad *A, *R;           /* the stored values and their radii (R NULL: none), count x parts x n x n */
    unsigned char *finite, *bounded; /* per matrix: every value finite; every radius finite */
    floatfmt F, out, f64;  /* the compute format, the output part format, float64 */
    bigint F_nan[2], out_nan[2], f64_nan[2]; /* their NaN encodings, which F, out and f64 point to */
    int flush, rnd, ovf;
    size_t size;           /* the output part's bytes */
    dyad allowance;        /* the factor c (n + 16) 2**-p, rounded up */
    round_bound held;         /* Lemma 1 at F */
    int64_t w;             /* the working precision 2 max(p_F, p_out) + 64 */
    unsigned char *refs;   /* per wanted part: the reference's encoding; then the outputs: */
    unsigned char *tol, *low, *high, *flags;
    int whole, started, skip;
    size_t mi;             /* larger matrices: the one in progress, */
    mtol_job *jobs;      /* its jobs, the next one to run */
    size_t njobs, capjobs, job;
    /* its working state */
    dyad norm, r, spread, rho_h, half_h, rest;
    int has_box, has_ball, pieces, Lball;
    size_t tiles, nstats;
    imat box, center, A0, Win, Bin, Hin, G, Gn, GB, H, Hn, HA, Lq, bq, Q, Sin, Cols, S;
    mexp ex[MTOL_EXPS];    /* exp of the box, the center, and W's, B's and H_0's inputs */
    imat_lines rows, colsB, colsA, colsW, srows, scols;
    dyad *stats;           /* the pieces' rows' sums and columns' maxima, the tiles' partial maxima */
} mtol;

void mtol_free(mtol *x);
int mtol_init(mtol *x, size_t count, size_t n, int parts, int radii, size_t r0, size_t r1, size_t c0, size_t c1, const floatfmt *F, int flush,
              const bigint *allow_num, const bigint *allow_den, const floatfmt *out, size_t size, int rnd, int ovf, const floatfmt *f64,
              const unsigned char *refs);
int mtol_next(mtol *x, size_t *tiles);
int mtol_tile(mtol *x, size_t t);

#endif
