/* Natural numbers as little-endian arrays of 64-bit words (internal): the word-level arithmetic
   under exactref's integers and dyadics. No function allocates; callers pass every buffer.

   Adapted from LibBF (Copyright (c) 2017-2025 Fabrice Bellard, MIT; see LICENSE): the word loops
   of mp_add, mp_sub, mp_add_ui, mp_sub_ui, mp_shr, mp_mul1, mp_add_mul1, mp_sub_mul1,
   mp_mul_basecase, mp_divnorm (base case) and mp_sqrtrem (Zimmermann's Karatsuba square root),
   rewritten for exactref's names and portable word arithmetic (words.h). */
#include <string.h>

#include "exactref.h"
#include "words.h"

/* r = a + b + carry over n words; returns the carry out. Adapted from LibBF (mp_add). */
uint64_t nat_add(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n, uint64_t carry) {
    for (size_t i = 0; i < n; i++) {
        uint64_t v = a[i], s = v + b[i], k = s < v;
        s += carry;
        carry = (s < carry) | k;
        r[i] = s;
    }
    return carry;
}

/* r = a - b - borrow over n words; returns the borrow out. Adapted from LibBF (mp_sub). */
uint64_t nat_sub(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n, uint64_t borrow) {
    for (size_t i = 0; i < n; i++) {
        uint64_t v = a[i], d = v - b[i], k = d > v, e = d - borrow;
        borrow = (e > d) | k;
        r[i] = e;
    }
    return borrow;
}

/* a += w over n words; returns the carry out. Adapted from LibBF (mp_add_ui). */
uint64_t nat_add_word(uint64_t *a, uint64_t w, size_t n) {
    for (size_t i = 0; i < n && w; i++) {
        uint64_t s = a[i] + w;
        w = s < w;
        a[i] = s;
    }
    return w;
}

/* a -= w over n words; returns the borrow out. Adapted from LibBF (mp_sub_ui). */
uint64_t nat_sub_word(uint64_t *a, uint64_t w, size_t n) {
    for (size_t i = 0; i < n && w; i++) {
        uint64_t v = a[i], d = v - w;
        w = d > v;
        a[i] = d;
    }
    return w;
}

/* r = (a + high * 2**(64 n)) >> shift for 1 <= shift <= 63; returns the bits shifted out.
   Adapted from LibBF (mp_shr). */
uint64_t nat_shr(uint64_t *r, const uint64_t *a, size_t n, int shift, uint64_t high) {
    uint64_t l = high;
    for (size_t i = n; i-- > 0;) {
        uint64_t v = a[i];
        r[i] = (v >> shift) | (l << (64 - shift));
        l = v;
    }
    return l & (((uint64_t)1 << shift) - 1);
}

/* r = a << shift (low n words) for 1 <= shift <= 63; returns the word shifted out. */
uint64_t nat_shl(uint64_t *r, const uint64_t *a, size_t n, int shift) {
    uint64_t l = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t v = a[i];
        r[i] = (v << shift) | l;
        l = v >> (64 - shift);
    }
    return l;
}

/* r = a * b + l over n words; returns the high word. Adapted from LibBF (mp_mul1). */
uint64_t nat_mul_word(uint64_t *r, const uint64_t *a, size_t n, uint64_t b, uint64_t l) {
    for (size_t i = 0; i < n; i++) {
        uint64_t hi, lo = word_mul(&hi, a[i], b);
        lo += l;
        hi += lo < l;
        r[i] = lo;
        l = hi;
    }
    return l;
}

/* r += a * b over n words; returns the high word. Adapted from LibBF (mp_add_mul1). */
uint64_t nat_addmul_word(uint64_t *r, const uint64_t *a, size_t n, uint64_t b) {
    uint64_t l = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t hi, lo = word_mul(&hi, a[i], b);
        lo += l;
        hi += lo < l;
        lo += r[i];
        hi += lo < r[i];
        r[i] = lo;
        l = hi;
    }
    return l;
}

/* r -= a * b over n words; returns what to subtract from the next word. Adapted from LibBF
   (mp_sub_mul1). */
uint64_t nat_submul_word(uint64_t *r, const uint64_t *a, size_t n, uint64_t b) {
    uint64_t l = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t hi, lo = word_mul(&hi, a[i], b), x = r[i];
        uint64_t t = x - lo, k1 = t > x, u = t - l, k2 = u > t;
        r[i] = u;
        l = hi + k1 + k2;
    }
    return l;
}

/* r = a * b, r of na + nb words not overlapping a or b, na and nb >= 1. Adapted from LibBF
   (mp_mul_basecase). */
void nat_mul(uint64_t *r, const uint64_t *a, size_t na, const uint64_t *b, size_t nb) {
    r[na] = nat_mul_word(r, a, na, b[0], 0);
    for (size_t i = 1; i < nb; i++) r[i + na] = nat_addmul_word(r + i, a, na, b[i]);
}

/* q = a / b for one word b > 0, q of n words (may be a); returns the remainder. A divisor below
   2**32 takes two one-word divisions per word, which the hardware does directly. */
uint64_t nat_div_word(uint64_t *q, const uint64_t *a, size_t n, uint64_t b) {
    uint64_t r = 0;
    if (b >> 32) {
        for (size_t i = n; i-- > 0;) q[i] = word_div(&r, r, a[i], b);
        return r;
    }
    for (size_t i = n; i-- > 0;) {
        uint64_t v = a[i], hi = (r << 32) | (v >> 32), qh = hi / b;
        r = hi - qh * b;
        uint64_t lo = (r << 32) | (v & 0xffffffffu), ql = lo / b;
        r = lo - ql * b;
        q[i] = (qh << 32) | ql;
    }
    return r;
}

int nat_cmp(const uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = n; i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

/* Base-case division of a (na words) by b (nb >= 2 words, b[nb - 1] >= 2**63), na >= nb: q gets
   na - nb + 1 words (q[na - nb] <= 1), a is left holding the remainder in its low nb words.
   Adapted from LibBF (mp_divnorm, base case). */
void nat_divrem_norm(uint64_t *q, uint64_t *a, size_t na, const uint64_t *b, size_t nb) {
    size_t n = na - nb;
    uint64_t b1 = b[nb - 1], qd = 1;
    for (size_t j = nb; j-- > 0;) {
        if (a[n + j] != b[j]) {
            if (a[n + j] < b[j]) qd = 0;
            break;
        }
    }
    q[n] = qd;
    if (qd) nat_sub(a + n, a + n, b, nb, 0);
    for (size_t i = n; i-- > 0;) {
        uint64_t r;
        if (a[i + nb] >= b1) qd = UINT64_MAX;
        else qd = word_div(&r, a[i + nb], a[i + nb - 1], b1);
        r = nat_submul_word(a + i, b, nb, qd);
        uint64_t v = a[i + nb], d = v - r;
        a[i + nb] = d;
        if (d > v) {
            /* the estimate was too large: add b back until the remainder is not negative */
            for (;;) {
                qd--;
                if (nat_add(a + i, a + i, b, nb, 0) && ++a[i + nb] == 0) break;
            }
        }
        q[i] = qd;
    }
}

/* The 16 -> 8 bit square roots LibBF starts from (sqrt_table, verbatim). */
static const uint16_t sqrt_table[192] = {
    128, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 144, 145, 146, 147, 148, 149,
    150, 150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159, 160, 160, 161, 162, 163, 163, 164, 165, 166, 167, 167, 168,
    169, 170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 178, 179, 180, 181, 181, 182, 183, 183, 184, 185, 185,
    186, 187, 187, 188, 189, 189, 190, 191, 192, 192, 193, 193, 194, 195, 195, 196, 197, 197, 198, 199, 199, 200, 201, 201,
    202, 203, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 209, 210, 211, 211, 212, 212, 213, 214, 214, 215, 215, 216,
    217, 217, 218, 218, 219, 219, 220, 221, 221, 222, 222, 223, 224, 224, 225, 225, 226, 226, 227, 227, 228, 229, 229, 230,
    230, 231, 231, 232, 232, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 240, 240, 241, 241, 242, 242, 243,
    243, 244, 244, 245, 245, 246, 246, 247, 247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255,
};

/* For a >= 2**62: returns s = floor(sqrt(a)) and *r = a - s**2 (0 <= r <= 2 s). Adapted from LibBF
   (mp_sqrtrem1). */
static uint64_t sqrtrem_word(uint64_t *r, uint64_t a) {
    uint64_t s1 = sqrt_table[(a >> 56) - 64], r1 = (a >> 48) - s1 * s1;
    if (r1 > 2 * s1) {
        r1 -= 2 * s1 + 1;
        s1++;
    }
    /* 32 -> 16 bits */
    uint64_t num = (r1 << 8) | ((a >> 40) & 0xff), q = num / (2 * s1), u = num % (2 * s1);
    uint64_t s = (s1 << 8) + q, rr = (u << 8) | ((a >> 32) & 0xff);
    rr -= q * q;
    if (rr >> 63) {
        s--;
        rr += 2 * s + 1;
    }
    /* 64 -> 32 bits */
    s1 = s;
    r1 = rr;
    num = (r1 << 16) | ((a >> 16) & 0xffff);
    q = num / (2 * s1);
    u = num % (2 * s1);
    s = (s1 << 16) + q;
    rr = (u << 16) | (a & 0xffff);
    rr -= q * q;
    if (rr >> 63) {
        s--;
        rr += 2 * s + 1;
    }
    *r = rr;
    return s;
}

/* For a = a[1] 2**64 + a[0] with a[1] >= 2**62: s[0] = floor(sqrt(a)), a[0] gets the low word of
   the remainder; returns its high bit. Adapted from LibBF (mp_sqrtrem2). */
static uint64_t sqrtrem_two(uint64_t *s, uint64_t *a) {
    uint64_t a0 = a[0], r1, s1 = sqrtrem_word(&r1, a[1]);
    /* num = r1 * 2**32 + a0 / 2**32 < 2**65, as num_hi * 2**64 + num_lo */
    uint64_t num_lo = (r1 << 32) | (a0 >> 32), num_hi = r1 >> 32, u;
    uint64_t q = word_div(&u, num_hi, num_lo, 2 * s1);
    uint64_t sq = (s1 << 32) + q;
    /* r = u * 2**32 + (a0 mod 2**32) - q**2, a signed value below 2**65 */
    uint64_t r_lo = (u << 32) | (a0 & 0xffffffffu), r_hi = u >> 32;
    if (q >> 32) {
        r_hi -= 1; /* q = 2**32: subtract 2**64 */
    } else {
        uint64_t qq = q * q;
        r_hi -= r_lo < qq;
        r_lo -= qq;
    }
    if (r_hi >> 63) {
        sq--;
        /* r += 2 sq + 1 */
        uint64_t t_hi = sq >> 63, t_lo = (sq << 1) + 1;
        r_lo += t_lo;
        r_hi += t_hi + (r_lo < t_lo);
    }
    s[0] = sq;
    a[0] = r_lo;
    return r_hi;
}

/* Recursive step: tmp holds n / 2 + 1 words; *rh gets the remainder's high word. Adapted from
   LibBF (mp_sqrtrem_rec). */
static void sqrtrem_rec(uint64_t *s, uint64_t *a, size_t n, uint64_t *tmp, uint64_t *rh) {
    if (n == 1) {
        *rh = sqrtrem_two(s, a);
        return;
    }
    size_t l = n / 2, h = n - l;
    uint64_t qh, ql, rr, c;
    sqrtrem_rec(s + l, a + 2 * l, h, tmp, &qh);
    /* the remainder is in a + 2 l, its high bit in qh */
    if (qh) nat_sub(a + 2 * l, a + 2 * l, s + l, h, 0);
    /* divide by s (normalized) instead of 2 s, and correct q and r */
    if (h == 1) {
        /* nat_divrem_norm needs two divisor words: one-word division of the n words a + l */
        uint64_t rem = 0;
        for (size_t i = n; i-- > 0;) tmp[i] = word_div(&rem, rem, a[l + i], s[l]);
        /* the quotient is l + 1 words: tmp[0..l]; the remainder is one word */
        a[l] = rem;
    } else {
        nat_divrem_norm(tmp, a + l, n, s + l, h);
    }
    qh += tmp[l];
    for (size_t i = 0; i < l; i++) s[i] = tmp[i];
    ql = nat_shr(s, s, l, 1, qh & 1);
    qh >>= 1; /* 0 or 1 */
    rr = ql ? nat_add(a + l, a + l, s + l, h, 0) : 0;
    nat_add_word(s + l, qh, h);
    /* q = qh, s[l - 1 .. 0], r = a[n - 1 .. l]: subtract q**2; for qh = 1, q = 2**(64 l) */
    if (qh) {
        c = qh;
    } else {
        nat_mul(a + n, s, l, s, l);
        c = nat_sub(a, a, a + n, 2 * l, 0);
    }
    rr -= nat_sub_word(a + 2 * l, c, n - 2 * l);
    if (rr >> 63) {
        nat_sub_word(s, 1, n);
        rr += nat_addmul_word(a, s, n, 2);
        rr += nat_add_word(a, 1, n);
    }
    *rh = rr;
}

/* For a of 2 n words (n >= 1) with a[2 n - 1] >= 2**62: s (n words) = floor(sqrt(a)), the
   remainder a - s**2 (0 <= r <= 2 s) left in a's low n words with its high word returned; a's
   other words are used as scratch. tmp holds n / 2 + 1 words. Adapted from LibBF (mp_sqrtrem). */
uint64_t nat_sqrtrem(uint64_t *s, uint64_t *a, size_t n, uint64_t *tmp) {
    uint64_t rh;
    sqrtrem_rec(s, a, n, tmp, &rh);
    return rh;
}
