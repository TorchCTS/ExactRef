/* Portable 64-bit word arithmetic (internal): products and quotients of two words and bit
   counts, with the compiler's 128-bit type where it has one, MSVC's intrinsics on x64 and ARM64,
   and plain 64-bit arithmetic elsewhere. EXACTREF_PORTABLE_WORDS forces the plain versions. */
#ifndef EXACTREF_WORDS_H
#define EXACTREF_WORDS_H

#include <stdint.h>

#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER) && !defined(EXACTREF_PORTABLE_WORDS) /* clang-cl has the type, not its runtime */
#define EXACTREF_INT128 1
/* the compiler's 128-bit integers, a GNU extension, marked as one so that -pedantic accepts them */
__extension__ typedef unsigned __int128 exactref_u128;
__extension__ typedef __int128 exactref_i128;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64)) && !defined(EXACTREF_PORTABLE_WORDS)
#define EXACTREF_MSVC_WORDS 1
#include <intrin.h>
#endif

/* a * b = hi * 2**64 + lo; returns lo. */
static inline uint64_t word_mul(uint64_t *hi, uint64_t a, uint64_t b) {
#ifdef EXACTREF_INT128
    exactref_u128 t = (exactref_u128)a * b;
    *hi = (uint64_t)(t >> 64);
    return (uint64_t)t;
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64)
    return _umul128(a, b, hi);
#elif defined(EXACTREF_MSVC_WORDS)
    *hi = __umulh(a, b);
    return a * b;
#else
    uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (mid << 32) | (p00 & 0xffffffffu);
#endif
}

static inline int word_clz(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#elif defined(EXACTREF_MSVC_WORDS)
    unsigned long index;
    return _BitScanReverse64(&index, x) ? 63 - (int)index : 64;
#else
    int n = 0;
    if (!x) return 64;
    while (!(x >> 63)) {
        x <<= 1;
        n++;
    }
    return n;
#endif
}

static inline int word_bits(uint64_t x) { return 64 - word_clz(x); }

/* (hi * 2**64 + lo) / d for hi < d: returns the quotient, *r the remainder. */
static inline uint64_t word_div(uint64_t *r, uint64_t hi, uint64_t lo, uint64_t d) {
#ifdef EXACTREF_INT128
    exactref_u128 n = ((exactref_u128)hi << 64) | lo;
    *r = (uint64_t)(n % d);
    return (uint64_t)(n / d);
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64) && !defined(__clang__)
    return _udiv128(hi, lo, d, r);
#else
    /* Two-digit long division in base 2**32 on the normalized divisor (Hacker's Delight, divlu). */
    const uint64_t b = (uint64_t)1 << 32;
    int s = word_clz(d);
    d <<= s;
    uint64_t un32 = s ? (hi << s) | (lo >> (64 - s)) : hi, un10 = lo << s;
    uint64_t vn1 = d >> 32, vn0 = d & 0xffffffffu, un1 = un10 >> 32, un0 = un10 & 0xffffffffu;
    uint64_t q1 = un32 / vn1, rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) {
        q1--;
        rhat += vn1;
        if (rhat >= b) break;
    }
    uint64_t un21 = un32 * b + un1 - q1 * d;
    uint64_t q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0--;
        rhat += vn1;
        if (rhat >= b) break;
    }
    *r = (un21 * b + un0 - q0 * d) >> s;
    return q1 * b + q0;
#endif
}

/* Two's-complement 128-bit words for the product kernels: every operation wraps modulo 2**128,
   and a word is read as signed (a running sum) or unsigned (a magnitude or node sum) by its user.
   The compiler's unsigned 128-bit type where it has one, else two 64-bit halves. */
#ifdef EXACTREF_INT128
typedef exactref_u128 w128;
static inline w128 w128_add(w128 a, w128 b) { return a + b; }
static inline w128 w128_sub(w128 a, w128 b) { return a - b; }
static inline int w128_negative(w128 a) { return (int)(a >> 127); }
static inline w128 w128_abs(w128 a) {
    w128 m = (w128)0 - (a >> 127);
    return (a ^ m) - m;
}
static inline int w128_less(w128 a, w128 b) { return a < b; }
static inline uint64_t w128_lo(w128 a) { return (uint64_t)a; }
static inline uint64_t w128_hi(w128 a) { return (uint64_t)(a >> 64); }
static inline w128 w128_zero(void) { return 0; }
static inline w128 w128_twice(w128 a) { return a << 1; }
static inline w128 w128_make(uint64_t lo, uint64_t hi) { return (w128)hi << 64 | lo; }
#else
typedef struct {
    uint64_t lo, hi;
} w128;
static inline w128 w128_add(w128 a, w128 b) {
    w128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo);
    return r;
}
static inline w128 w128_sub(w128 a, w128 b) {
    w128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo);
    return r;
}
static inline int w128_negative(w128 a) { return (int)(a.hi >> 63); }
static inline w128 w128_abs(w128 a) {
    if (!w128_negative(a)) return a;
    w128 r;
    r.lo = ~a.lo + 1;
    r.hi = ~a.hi + (r.lo == 0);
    return r;
}
static inline int w128_less(w128 a, w128 b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
static inline uint64_t w128_lo(w128 a) { return a.lo; }
static inline uint64_t w128_hi(w128 a) { return a.hi; }
static inline w128 w128_zero(void) {
    w128 r = {0, 0};
    return r;
}
static inline w128 w128_twice(w128 a) {
    w128 r;
    r.lo = a.lo << 1;
    r.hi = a.hi << 1 | a.lo >> 63;
    return r;
}
static inline w128 w128_make(uint64_t lo, uint64_t hi) {
    w128 r;
    r.lo = lo;
    r.hi = hi;
    return r;
}
#endif

/* whether a two's-complement integer of n words is negative */
static inline int mw_negative(const uint64_t *x, size_t n) { return (int)(x[n - 1] >> 63); }

/* The integer of a two's-complement word, with no implementation-defined conversion. */
static inline int64_t word_signed(uint64_t w) { return w >> 63 ? -(int64_t)(~w) - 1 : (int64_t)w; }

/* The product of two two's-complement words, each below 2**63 in magnitude. */
static inline w128 w128_product(uint64_t x, uint64_t y) {
#ifdef EXACTREF_INT128
    return (w128)((exactref_i128)word_signed(x) * word_signed(y));
#else
    int nx = (int)(x >> 63), ny = (int)(y >> 63);
    uint64_t hi, lo = word_mul(&hi, nx ? ~x + 1 : x, ny ? ~y + 1 : y);
    w128 r = {lo, hi};
    if (nx != ny) r = w128_sub(w128_zero(), r);
    return r;
#endif
}

/* Two's-complement 128-bit values for the kernels' inner loops, changed in place through pointers:
   the compiler's 128-bit type where it has one, else an array of two words, least significant
   first. Compilers keep either in registers, where MSVC keeps a loop-carried w128 struct in memory
   and runs the loop about six times slower. A w2 is set by an initializer ({0}) or these
   operations, never by assignment; every operation wraps modulo 2**128. */
#ifdef EXACTREF_INT128
typedef exactref_u128 w2;
#else
typedef uint64_t w2[2];
#endif

/* r = x y for two's-complement words x, y below 2**63 in magnitude */
static inline void w2_product(w2 *r, uint64_t x, uint64_t y) {
#ifdef EXACTREF_INT128
    *r = (w2)((exactref_i128)word_signed(x) * word_signed(y));
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64)
    __int64 hi;
    (*r)[0] = (uint64_t)_mul128((__int64)x, (__int64)y, &hi);
    (*r)[1] = (uint64_t)hi;
#else
    /* the unsigned product, less y where x is negative and x where y is */
    uint64_t hi, lo = word_mul(&hi, x, y);
    (*r)[0] = lo;
    (*r)[1] = hi - ((y & ((uint64_t)0 - (x >> 63))) + (x & ((uint64_t)0 - (y >> 63))));
#endif
}

/* x += b */
static inline void w2_add(w2 *x, w2 *b) {
#ifdef EXACTREF_INT128
    *x += *b;
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64)
    _addcarry_u64(_addcarry_u64(0, (*x)[0], (*b)[0], &(*x)[0]), (*x)[1], (*b)[1], &(*x)[1]);
#else
    uint64_t lo = (*x)[0] + (*b)[0];
    (*x)[1] += (*b)[1] + (lo < (*b)[0]);
    (*x)[0] = lo;
#endif
}

/* x -= b */
static inline void w2_sub(w2 *x, w2 *b) {
#ifdef EXACTREF_INT128
    *x -= *b;
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64)
    _subborrow_u64(_subborrow_u64(0, (*x)[0], (*b)[0], &(*x)[0]), (*x)[1], (*b)[1], &(*x)[1]);
#else
    uint64_t lo = (*x)[0] - (*b)[0];
    (*x)[1] -= (*b)[1] + ((*x)[0] < (*b)[0]);
    (*x)[0] = lo;
#endif
}

/* x += |b|, b read as signed: b's bits complemented where it is negative, plus its sign bit */
static inline void w2_add_abs(w2 *x, w2 *b) {
#ifdef EXACTREF_INT128
    w2 m = (w2)0 - (*b >> 127);
    *x += (*b ^ m) - m;
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64)
    uint64_t m = (uint64_t)0 - ((*b)[1] >> 63);
    _addcarry_u64(_addcarry_u64((unsigned char)(m & 1), (*x)[0], (*b)[0] ^ m, &(*x)[0]), (*x)[1], (*b)[1] ^ m, &(*x)[1]);
#else
    uint64_t m = (uint64_t)0 - ((*b)[1] >> 63), c = (*b)[0] ^ m, lo = (*x)[0] + c, carry = lo < c, lo2 = lo + (m & 1);
    carry += lo2 < lo;
    (*x)[1] += ((*b)[1] ^ m) + carry;
    (*x)[0] = lo2;
#endif
}

static inline w128 w2_get(w2 *x) {
#ifdef EXACTREF_INT128
    return *x;
#else
    return w128_make((*x)[0], (*x)[1]);
#endif
}

static inline void w2_set(w2 *x, w128 v) {
#ifdef EXACTREF_INT128
    *x = v;
#else
    (*x)[0] = w128_lo(v);
    (*x)[1] = w128_hi(v);
#endif
}

static inline int w2_negative(w2 *x) { return w128_negative(w2_get(x)); }

/* Two's-complement 256-bit words for the product kernel's wide main range (float64 operands):
   four 64-bit words, least significant first, every operation modulo 2**256. */
typedef struct {
    uint64_t w[4];
} w256;

static inline w256 w256_zero(void) {
    w256 r = {{0, 0, 0, 0}};
    return r;
}

static inline w256 w256_add(w256 a, w256 b) {
    w256 r;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s = a.w[i] + b.w[i], k = s < a.w[i];
        r.w[i] = s + carry;
        carry = k | (r.w[i] < s);
    }
    return r;
}

static inline w256 w256_sub(w256 a, w256 b) {
    w256 r;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t d = a.w[i] - b.w[i], k = d > a.w[i];
        r.w[i] = d - borrow;
        borrow = k | (r.w[i] > d);
    }
    return r;
}

static inline int w256_negative(w256 a) { return (int)(a.w[3] >> 63); }

static inline w256 w256_abs(w256 a) { return w256_negative(a) ? w256_sub(w256_zero(), a) : a; }

static inline int w256_less(w256 a, w256 b) {
    for (int i = 3; i >= 0; i--)
        if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
    return 0;
}

static inline w256 w256_twice(w256 a) {
    w256 r;
    for (int i = 3; i > 0; i--) r.w[i] = a.w[i] << 1 | a.w[i - 1] >> 63;
    r.w[0] = a.w[0] << 1;
    return r;
}

/* Two's-complement 256-bit values for the inner loops, changed in place through pointers as a w2
   is: four words, least significant first, never assigned; every operation wraps modulo 2**256. */
typedef uint64_t w4[4];

/* x += (b ^ m) + (m & 1) for m 0 or all ones: x + b, x - b, or with m b's sign mask x + |b| */
static inline void w4_add_masked(w4 *x, w4 *b, uint64_t m) {
#ifdef EXACTREF_INT128
    exactref_u128 xl = (exactref_u128)(*x)[1] << 64 | (*x)[0], bl = (exactref_u128)((*b)[1] ^ m) << 64 | ((*b)[0] ^ m);
    exactref_u128 xh = (exactref_u128)(*x)[3] << 64 | (*x)[2], bh = (exactref_u128)((*b)[3] ^ m) << 64 | ((*b)[2] ^ m);
    exactref_u128 l = xl + bl, l2 = l + (m & 1), h = xh + bh + (l < xl) + (l2 < l);
    (*x)[0] = (uint64_t)l2;
    (*x)[1] = (uint64_t)(l2 >> 64);
    (*x)[2] = (uint64_t)h;
    (*x)[3] = (uint64_t)(h >> 64);
#elif defined(EXACTREF_MSVC_WORDS) && defined(_M_X64)
    unsigned char c = (unsigned char)(m & 1);
    for (int i = 0; i < 4; i++) c = _addcarry_u64(c, (*x)[i], (*b)[i] ^ m, &(*x)[i]);
#else
    uint64_t carry = m & 1;
    for (int i = 0; i < 4; i++) {
        uint64_t y = (*b)[i] ^ m, s = (*x)[i] + y, s2 = s + carry;
        carry = (s < y) + (s2 < s);
        (*x)[i] = s2;
    }
#endif
}

static inline void w4_add(w4 *x, w4 *b) { w4_add_masked(x, b, 0); }

static inline void w4_sub(w4 *x, w4 *b) { w4_add_masked(x, b, ~(uint64_t)0); }

/* x += |b|, b read as signed */
static inline void w4_add_abs(w4 *x, w4 *b) { w4_add_masked(x, b, (uint64_t)0 - ((*b)[3] >> 63)); }

static inline int w4_negative(w4 *x) { return (int)((*x)[3] >> 63); }

static inline int w4_is_zero(w4 *x) { return !((*x)[0] | (*x)[1] | (*x)[2] | (*x)[3]); }

/* a < b, both read as unsigned */
static inline int w4_less(w4 *a, w4 *b) {
    for (int i = 3; i >= 0; i--)
        if ((*a)[i] != (*b)[i]) return (*a)[i] < (*b)[i];
    return 0;
}

/* r = the three-word two's-complement value x, sign-extended */
static inline void w4_load3(w4 *r, const uint64_t *x) {
    (*r)[0] = x[0];
    (*r)[1] = x[1];
    (*r)[2] = x[2];
    (*r)[3] = (uint64_t)0 - (x[2] >> 63);
}

/* r = a b modulo 2**256 for a word b, a read as unsigned */
static inline void w4_mul(w4 *r, w4 *a, uint64_t b) {
    uint64_t h0, l0 = word_mul(&h0, (*a)[0], b), h1, l1 = word_mul(&h1, (*a)[1], b), h2, l2 = word_mul(&h2, (*a)[2], b);
    uint64_t r1 = h0 + l1, c1 = r1 < h0;
    uint64_t s2 = h1 + l2, c2 = s2 < h1;
    uint64_t r2 = s2 + c1;
    c2 += r2 < s2;
    (*r)[0] = l0;
    (*r)[1] = r1;
    (*r)[2] = r2;
    (*r)[3] = h2 + c2 + (*a)[3] * b;
}

/* r = a b modulo 2**256 for a two's-complement a and a signed b: a b read as unsigned, less 2**64 a where b is
   negative */
static inline void w4_smul(w4 *r, w4 *a, int64_t b) {
    uint64_t m = (uint64_t)0 - ((uint64_t)b >> 63);
    w4 d = {0, (*a)[0] & m, (*a)[1] & m, (*a)[2] & m};
    w4_mul(r, a, (uint64_t)b);
    w4_sub(r, &d);
}

static inline w256 w4_get(w4 *x) {
    w256 r = {{(*x)[0], (*x)[1], (*x)[2], (*x)[3]}};
    return r;
}

/* r = a b for two's-complement 128-bit a, b (lo, hi pairs), each below 2**127 in magnitude: the
   unsigned product less 2**128 (b where a is negative, a where b is) */
static inline void w4_product(w4 *r, uint64_t alo, uint64_t ahi, uint64_t blo, uint64_t bhi) {
    uint64_t h00, l00 = word_mul(&h00, alo, blo), h01, l01 = word_mul(&h01, alo, bhi), h10, l10 = word_mul(&h10, ahi, blo), h11, l11 = word_mul(&h11, ahi, bhi);
    uint64_t s1 = h00 + l01, c1 = s1 < h00;
    uint64_t t1 = s1 + l10;
    c1 += t1 < s1;
    uint64_t s2 = h01 + h10, c2 = s2 < h01;
    uint64_t t2 = s2 + l11;
    c2 += t2 < s2;
    uint64_t u2 = t2 + c1;
    c2 += u2 < t2;
    uint64_t ma = (uint64_t)0 - (ahi >> 63), mb = (uint64_t)0 - (bhi >> 63);
    uint64_t dlo = (blo & ma) + (alo & mb), dhi = (bhi & ma) + (ahi & mb) + (dlo < (blo & ma));
    (*r)[0] = l00;
    (*r)[1] = t1;
    (*r)[2] = u2 - dlo;
    (*r)[3] = h11 + c2 - dhi - (u2 < dlo);
}

/* acc (n words, two's complement) plus a x, or minus with sub, for a magnitude a of na < n words and a word x */
static inline void mw_addmul1(uint64_t *acc, int n, const uint64_t *a, int na, uint64_t x, int sub) {
    uint64_t pc = 0, c = 0; /* the product's carry word; the add's carry or the subtraction's borrow */
    for (int i = 0; i < n; i++) {
        uint64_t w = pc;
        if (i < na) {
            uint64_t hi, lo = word_mul(&hi, a[i], x);
            w = lo + pc;
            pc = hi + (w < lo);
        } else {
            pc = 0;
        }
        if (!w && !c) {
            if (i >= na) break;
            continue;
        }
        uint64_t v = acc[i];
        if (!sub) {
            uint64_t t = v + w, k = t < w;
            acc[i] = t + c;
            c = k | (acc[i] < c);
        } else {
            uint64_t t = v - w, k = v < w;
            acc[i] = t - c;
            c = k | (t < c);
        }
    }
}

#endif
